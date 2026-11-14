# Design

Status: written from a reading of the engine source, tested off the hardware, and run on
one setup on 14 November 2026 (see the table in the README). The gate (two endpoints on
one server) ran on the same setup the same day. Every reference below is to the pinned
commits listed in the README; `path:line` is in the vLLM fork unless it starts with
`flashinfer/`.

## The problem in one paragraph

A weight overlay that replaces a few tensors can be loaded by remapping those tensor
names in `model.safetensors.index.json` to a small side file. That gives a second
checkpoint folder that shares every other file with the base. But a server can hold
only one of the two at a time, and changing between them is a stop and a relaunch.
The goal is one server that runs the base weights and can change the few tensors to
the overlay's, and back, with a single call.

## What the engine does with these tensors

1. **Loading follows the index, per tensor name.** With `--load-format b12x` the loader
   reads each tensor from the file the index names for it
   (`vllm/model_executor/model_loader/b12x_loader.py:238-248`).

2. **`wo_b` is a row-parallel layer holding FP8 values and native 32x32 block scales**
   (`vllm/models/deepseek_v4_1/attention.py:533-542`,
   `vllm/models/deepseek_v4_1/b12x_layers.py:304-343`): a `weight` of type
   `float8_e4m3fn` and a `weight_scale_inv` of type `float8_e8m0fnu`.

3. **The checkpoint bytes are kept as they are.** The layer's weight method is wrapped
   by one whose only job after loading is to check the two types; its own description
   is "Keep checkpoint FP8 tensors unmodified for the fused WO projection"
   (`attention.py:304-324`).

4. **Each rank holds a slice of columns.** The loader cuts a row-parallel weight along
   its input dimension: `local` columns per rank, starting at `rank * local`
   (`vllm/model_executor/layers/linear.py:1867-1883`). When the number of heads was
   padded to divide by the number of ranks, the last rank's tail is padding
   (`vllm/model_executor/parameter.py:20-45`; the attribute is set at
   `attention.py:543-548`). The block scales are cut the same way, in units of 32.

5. **Packing happens once, after loading**
   (`attention.py:1596-1632`, `setup_wo_projection`), with `block_size=(32, 32)`. The
   result is kept in `self._wo_projection_weights`. For 32x32 scales nothing is
   requantized: the packed `values` are `weight.contiguous()`, the Parameter's own
   memory, and two scale tables (`scale_rows`, `scale_mma`) are new memory derived
   from the block scales. For two specific shapes there is also `values_tiled`, a
   second, tiled copy of the values
   (`flashinfer/experimental/b12x/gemm/_shared/wo_mxfp8.py:1447-1544`), which the decode
   kernel reads when its plan asks for it
   (`flashinfer/experimental/b12x/gemm/wo_projection/_preparation.py:503-504`). Whether
   a deployment has it depends on the shape each rank holds, so the switch assumes
   neither: it refreshes every derived tensor the packing function returns and refuses
   if one changed shape. `wo_a` and `wo_b` are packed by two separate calls
   (`wo_mxfp8.py:1547-1585`), so `wo_b` can be handled alone.

6. **The forward pass binds the packed object on every call**
   (`attention.py:1754-1767`). The prepared plan holds scratch space for activations,
   not weights (`wo_mxfp8.py:195-394`). The call sits inside captured CUDA graphs, which
   is why the next point matters.

7. **In place, never by replacement.** The engine's own reload code states the rule:
   "Copy processed values into original tensor storage (preserves cudagraph refs)"
   (`vllm/model_executor/model_loader/reload/layerwise.py:476`). A switch must overwrite
   the bytes of the existing buffers. Assigning a new tensor object would be seen by
   eager execution and missed by replayed graphs.

8. **The prefix cache does not know about weights.** Block hashes are computed from
   the parent hash, the token ids and a few request fields
   (`vllm/v1/core/kv_cache_utils.py:618-666`). After a flip, cached blocks computed with
   the other weights would be reused unless the cache is cleared.

## What does not work, and why

- **LoRA.** The model refuses it: "V4.1 native kernels do not support LoRA adapters"
  (`vllm/models/deepseek_v4_1/nvidia/model.py:370-371`).
- **A choice per request.** The fused kernel takes one set of weights per batch
  (`attention.py:1754-1767`). Two requests decoded together cannot use different sets.
- **The built-in weight reload.** Under the b12x loader it raises "b12x requires an
  active initial-load session" (`b12x_loader.py:276-281`), and it would not rebuild the
  derived tables.

## How the switch gets into the engine without a patch

- **The worker half** is a class given with `--worker-extension-cls`. The engine mixes
  it into its worker class on every rank and logs the method names it gained
  (`vllm/v1/worker/worker_base.py:366-392`). The names must not collide with the
  worker's own; this one adds three (`overlay_status`, `overlay_prepare`,
  `overlay_switch`).
- **The HTTP half** is an endpoint plugin, found through the entry point group
  `vllm.endpoint_plugins` and loaded only when `VLLM_PLUGINS` names it
  (`vllm/plugins/__init__.py:93-121`). The same variable is the allow-list for the
  engine's other plugin groups (`vllm/plugins/__init__.py:36-59`), so it must also name
  whatever the image already loads.
- **The calls it uses** are the engine client's own: `pause_generation`, described there
  as "Pause generation to allow model weight updates", `resume_generation`, `is_paused`
  (`vllm/v1/engine/async_llm.py:913-964`), `reset_prefix_cache` (`:1087`) and
  `collective_rpc` (`:1142`), which runs one method on every rank. While paused, new
  requests are not scheduled; they are flushed on resume (`vllm/v1/engine/core.py:1006`).

The engine also offers pause, resume and "run this method on every rank" as HTTP routes,
but only behind its developer mode, which exposes much more than a switch needs on the
public API port. This project does not enable developer mode.

## The worker half

**Prepare** (once per rank, at the first flip):

1. Read the side file's header. Every tensor must be a `layers.N.attn.wo_b` weight or
   scale of the expected type, and every weight must come with its scale.
2. Find the same tensors in the served checkpoint through its index. If the index
   already points at the side file, stop: the server must be started on the base.
3. For each tensor, work out which columns this rank holds (point 4 above) and check
   rows, types and one-byte elements against the live Parameter.
4. Read both sets from disk. Fingerprint this rank's columns of every tensor, and digest
   the whole of each set. A rank sees only its own columns of the live model, so the
   whole-set digests are what lets the ranks show each other they read the same files;
   the HTTP half compares them and refuses to flip if they differ, with or without
   `force`.
5. **Prove the cut**: fingerprint the same columns of every live tensor. They must all
   match one set. If they match neither, this tool's reading of the layout does not fit
   the engine in front of it; the rank stays unprepared and will not write, whatever it
   is asked.

**Switch** (per rank):

1. For every layer in order, for weight and scale: read the tensor from disk, keep this
   rank's columns, copy them over the live Parameter's memory. Padding columns are
   never read or written.
2. Rebuild the derived tensors with the model's own packing function and copy each over
   the existing one where it is not the Parameter's memory already; check that each
   took the copy.
3. Read every switched tensor back and compare with both sets.

All copies are made on the bytes (through `uint8` views), so nothing depends on which
operations the installed torch offers for the two 8-bit float types. The writes run
inside `torch.inference_mode()`, because tensors made at load time may be inference
tensors.

**Status** reads the first and last switched layer back. A write goes through the
layers in order, so one that stopped halfway leaves those two disagreeing. `?full=1`
reads every tensor, and adds `derived`: one digest per rank of every tensor the model
derived from the switched weights (the packed values, the two scale tables, the tiled
copy where there is one). It can be read on a freshly started server, before anything
was prepared. After a flip there and back it must be what it was, byte for byte.

Which set is active is read from the weights, never remembered.

## The HTTP half

`POST /overlay/switch`:

1. Ask every rank for its status; prepare if any rank is unprepared.
2. If the ranks do not agree on one set, stop, unless `force` was sent.
3. If the wanted set is already active, answer without pausing.
4. Pause in the chosen mode, with the cache cleared.
5. Switch on every rank. If every rank reports the wanted set: clear the prefix cache,
   resume, answer.
6. Otherwise copy the previous set back on every rank. If that works: clear, resume,
   answer with the error. If it does not, or there was no known previous set: leave
   generation paused and say so.

One flip at a time; a second call meanwhile gets 409.

A call that times out does not cancel what the ranks were asked to do: the request stays
queued on the workers and still runs. Since the copy back is queued behind it, the end
state is still the previous set. What the engine does with late answers was not traced.

## The gate: two endpoints on one server

With `OVERLAY_GATE_PREFIX=/free` the base set answers on the usual paths and the overlay
on the same paths under the prefix. `mod/vllm_overlay_switch/gate.py` holds it; it imports
nothing from the engine.

**Where it goes in.** The app is built in a fixed order: the routers, then the endpoint
plugins, then the engine's own middleware (`vllm/entrypoints/launchers/app.py:42-54`,
`vllm/plugins/endpoint_plugins/interface.py:91-105`,
`vllm/entrypoints/serve/middleware/register.py:19-80`). Middleware added later wraps what
was added earlier. So a pure ASGI middleware added inside the plugin's `attach_router` is
the innermost one: it sees a request after CORS, the API key, the request id and whatever
`--middleware` names, and nothing stands between it and the routes. The installer checks
that order in the engine it installs into.

**Which side a request is for.** Under the prefix: the overlay. On the usual paths: the
base, unless the request's `model` is one of the names in `OVERLAY_GATE_MODELS`. Those
names are further served names of the one model (`--served-model-name` takes several;
the model list shows one entry per name, `vllm/entrypoints/launchers/api_server/app_state.py:59-61`,
and answers carry the first, `vllm/entrypoints/openai/models/serving.py:144-147`), so a
client that picks from the list finds both sides under one address. The gate reads the
name from the body it parses anyway; a `model` that is not a string chooses nothing and is
left for the engine to refuse.

**Admission, not the engine's pause.** The kernel takes one set of weights for everything
it decodes together (`vllm/models/deepseek_v4_1/attention.py:1754-1767`), so the two sides
have to take turns. The engine's pause cannot arrange that by itself: what arrives while
it is paused is queued and run after the resume, on whatever is live then
(`vllm/v1/engine/core.py:2091-2124`). The gate therefore decides before the engine sees a
request. It keeps one line. A request is let in when its side is live and nobody for the
other side stands before it; the first in line may flip once nothing is in flight. "In
flight" ends when the app has finished with the request, which for a streamed answer is
the end of the stream or the client going away. Behind that stands the flip's own pause in
`wait` mode, which lets the engine run until it is idle before anything is copied: a
request the client abandoned a moment ago is finished or aborted by then.

**Each side its own cache, and no clearing.** A cached block is known by a hash over its
tokens, its parent's hash and a few extra keys; the request's `cache_salt` is one of them
for the first block (`vllm/v1/core/kv_cache_utils.py:618-629`), and through the parents
for every block after it. The weights are not part of the hash, which is why the manual
flip clears the cache. The gate instead sets the salt of every request it lets in to
`overlay-gate:<side>`, followed by a digest of the client's own salt if it sent one (the
engine accepts at most 128 characters and no `/`, `@` or backslash,
`vllm/entrypoints/generate/base/protocol.py:36-51`). Blocks made on one side can then
only be found by requests of that side, which only run while that side is live. Because
a flip back is byte exact (the `derived` digest), those blocks stay right, and the gate
flips with `pause_generation(clear_cache=False)`
(`vllm/v1/engine/async_llm.py:913-949`). The body is parsed and written again with
exactly one `cache_salt`, so a client cannot slip in a second one.

**What must not pass.** Only requests the gate has marked may run the model. So it lets
through `GET`, `HEAD` and `OPTIONS`, three routes that render and count without running
the model (`/v1/messages/count_tokens` only renders the request and counts: the
`count_tokens` handler of the messages API), and the four that run it and
take a salt; every other route answers 403, and a websocket is closed. A
`/v1/responses` request with `background` set is refused too: the engine answers such a
request at once and runs the model afterwards in a task of its own
(`vllm/entrypoints/openai/responses/serving.py:508-556`), which is after the gate has
counted the request as ended.

**The API key.** The engine's key check looks at how a path begins and guards `/v1`,
`/v2`, `/inference` and `/cohere` (`vllm/entrypoints/serve/middleware/authenticate.py:11,
47-62`). `/free/v1/chat/completions` begins with none of them, so it reaches the gate
unchecked, and the gate then takes the prefix off. Left like that, the prefix would be a
way around the key. The plugin therefore builds the engine's own `AuthenticationMiddleware`
with the keys the engine itself uses (`--api-key`, else `VLLM_API_KEY`, as in
`register.py:33-37`) and the gate runs everything under the prefix through it, on the
inner path. If there is a key and that class cannot be imported, `attach_router` raises
and the server does not start.

**Reasoning effort.** The chat request's schema takes seven words for `reasoning_effort`
(`vllm/entrypoints/openai/chat_completion/protocol.py:257-259`) and hands the value to the
renderer as a chat-template argument (`:606-627`); arguments in `chat_template_kwargs`
override it. The DeepSeek V4.1 renderer then accepts `none` (thinking off), the words of
its own table, and integers from 1 to 100, and raises for anything else
(`vllm/tokenizers/deepseek_v41.py:86-99`; the table `low` 50, `high` 75, `xhigh` 75, `max`
100 is `vllm/tokenizers/deepseek_v41_encoding.py:185-192`). So `minimal` and `medium` pass
the schema and fail the renderer, an integer passes the renderer only inside the kwargs,
and a client that offers the seven words breaks on two of them. `effort.py` translates in
the gate's rewrite step, in both places, and moves a top-level integer into the kwargs.
Which word becomes which is the recipe's choice, not the code's.

**Counting a request out.** A request leaves the count of those in flight in a plain
function call, not an awaited one, in the `finally` of the app call: a request task that
is being cancelled must still be counted out, and an awaited step there could be cut by a
second cancellation, after which the other side would wait for ever.

**When a flip goes wrong.** A flip that fails and puts the previous set back answers 503
to the request that asked for it, and the live side goes on serving. A flip that leaves
generation paused, or that is cut off half way (the task that asked for it was cancelled),
shuts the gate: every request gets 503 until a forced flip by hand has written one set
onto every rank.

**What the acceptance test can read exactly.** The scheduler counts, per request, the
tokens it looked up in the prefix cache and the tokens it found
(`vllm/v1/core/kv_cache_manager.py:329-338`, `vllm/v1/metrics/stats.py:131-142`), and
`/metrics` shows the sums (`vllm/v1/metrics/loggers.py:647-664`). Read around single
requests, with the count of finished requests beside them to notice other traffic, they
say whether a request on one side reused what the other side had cached.

Not traced in the source: whether anything besides the block hash keeps results by prompt
across requests in the speculative-decoding path. The draft layers have weights of their
own that are not switched, and the test above would show a reuse across the sides in the
counters; it would not show a cache the counters do not count.

## Memory and time

For the motivating case the side file holds 26 weights of 5120 x 8192 (42 MB each, as
its header says) with their 160 x 256 block scales. A rank writes its own columns of
each; how many columns that is depends on the number of ranks and is read from the live
model. Nothing is kept between flips. During a flip one file tensor and this rank's cut
of it are held on the CPU side, and the packing function allocates scratch on the
device, which is released at the end. A flip reads every file tensor whole, about
1.1 GB per rank; preparing reads both sets once, about 2.2 GB.

Measured on the tested setup (three ranks): each rank holds 3,072 columns, the eight
groups being padded to nine; two ranks write 409 MB per flip and the third, which holds
2,048 real columns, 273 MB. Preparing took 3.9 s. A flip took 1.4 to 2.0 s end to end;
the pause before the copy was 0.03 s and the rest is the copy and reading every tensor
back. Six flips in a row left the machines' free memory where it was. A full status,
which reads every tensor back and fingerprints the derived ones, took about three
seconds; that is why the fingerprint is not part of a flip.

With the gate on the same setup: an answer took 0.69 s on average when its side was
live and 2.22 s when the server had to flip first (36 answers, 2.02 to 2.66 s). Free
memory, read five times a second on each machine during six flips, dipped at most
0.07 GiB and came back. Requests of both sides arriving together take strict turns, a
flip at every change of side: in two bursts of 12, half for each side, an answer took
0.6 to 26.2 s and 11.2 s on average.

A second launch gave the same server a 1,048,576-token window and a 3 GB KV cache per
machine, which leaves each machine a few GiB free instead of six to eight. The
acceptance test passed there too, with the same times (0.61 s and 2.24 s). The first
two flips dipped free memory by 0.03 GiB, and over four minutes of flips and bursts the
lowest reading on the tightest machine was 1.89 GiB; it had been 1.7 GiB after its first
requests, before any flip. A flip does not need room of its own to speak of.

## Acceptance test

`tools/acceptance.py`. It asks nothing sensitive and needs no reference answers. It
answers two questions separately.

**Does the model read what a flip wrote?** Two models with different weights weigh the
next token differently at some point where both have read exactly the same words.

1. Read the live set (A) several times: a handful of plain prompts at temperature 0 with
   thinking off, recording the tokens and their alternatives. Flip, read the other set
   (B). Flip back, read A again.
2. For each prompt, walk the answers token by token while they all agree. At every
   position up to and including the first place they part, every answer was produced
   from an identical context. At each such position the contenders are the token chosen
   most often and the rival the answers give most weight to; an answer's margin is the
   log probability of the first minus that of the second.
3. A position tells the sets apart when every A margin, before and after, lies on one
   side and every B margin on the other. A prompt does when one of its positions does.
   Looking only at the parting was not enough: the two sets often choose the same word
   while weighing its rival very differently, and that is the steadier signal.

**Does flipping back restore the model?** Exactly: the `derived` digest of every rank
is read when the test begins and again when it ends, and must be equal. The weights
themselves are read back by the switch after every flip. For a switch without the
digest the answers have to do: A after the return must not stand apart from A before.

**Why margins and several passes.** The first version of this test compared choices
over two passes and assumed that a model repeats its own choice. On the first hardware
it ran on, that was false: identical requests moved a margin by one to three logits
(the logits there come out in steps of a quarter), enough to change the chosen token
wherever two tokens were close. With two passes, two readings of one set stand apart by
chance one time in three, and the test reported a failed return that was not one. Now
each set is read six times by default, each request carries its own `cache_salt` so
that no answer reuses another's cached prefix, and the judgement is on whether two
groups of margins overlap at all. Two groups of six from one source do not overlap by
chance one time in 462; twelve against six, one time in about nine thousand, and the
test says how many positions it looked at so that the figure can be read against that.

`--read-only` reads a server without touching any switch and saves the answers: a
reference, for instance of the overlay loaded as a model of its own, to set beside the
flipped state.

## What running it settled, and what it did not

On the tested setup (14 November 2026):

- Captured graphs follow an in-place overwrite: the answers changed with a flip and
  returned with the flip back, and the overlay loaded as a model of its own read like
  the flipped state at every position that told base and flipped apart.
- Nothing else showed up that keeps its own copy of these weights: the same comparison
  would have shown a state that was neither base nor overlay.
- The packing function is deterministic: the digest of the derived tensors after
  flipping there and back equalled the one the startup left, on two separate launches,
  and the startup's own digests were the same on both.
- The image registered no plugins beyond vLLM's own two.
- The gate holds on a real server. Its acceptance test passed: 156 answers each carried
  the mark of the side asked for, and 60 answers asked in turn and in bursts, by
  model name and under the prefix, all lay with their own side at a landmark.
- The cache salt keeps the sides apart in the engine's own counters: a prompt asked
  again on its side reused its prefix, the other side's first request for the same
  prompt under the same client salt reused nothing, and after two flips the first side
  still reused its own.
- Sixty flips into a launch every tensor read back as base on every rank and the
  derived digests were the startup's.
- A client that left in the middle of a streamed answer did not hold the other side up:
  twelve times out of twelve the next request for the other side was answered within
  three seconds, on its own side.
- The effort translation sits in front of the engine's own validation: values the
  engine refuses with 400 were answered 200 at the top level and in
  `chat_template_kwargs`, by either model name and under the prefix.

Still open:

- A long generation on one side while the other is asked: the wait is by design, its
  length was not measured.
- A client that gives up while it waits in the line (it keeps its place).
- Memory with a long cached conversation on each side at once: the tests' prompts are
  short.
- The API-key check under the prefix on a real server, and a server with `--root-path`.
- Worker and queue state after a `collective_rpc` timeout.
- A flip while requests are in flight, and under load from several clients.
- Other tensor-parallel sizes, where the padding falls differently.
