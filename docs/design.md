# Design

How the overlay switch works and why it is built this way. All source references are to
the pinned commits listed under Compatibility in the
[README](../README.md#compatibility). A `path:line` reference is in the vLLM fork unless
it starts with `flashinfer/`.

Terms: a *variant* is one of the two sets of weights (base or overlay). A *rank* is one
tensor-parallel worker; on the test setup each rank runs on its own node.

## Problem

A weight overlay that replaces a few tensors can be loaded by remapping those tensor
names in `model.safetensors.index.json` to a small side file. This gives a second
checkpoint folder that shares every other file with the base. However, a server can hold
only one of the two at a time, and changing between them requires a stop and a relaunch.
The goal is one server that runs the base weights and can switch the affected tensors to
the overlay's, and back, with a single call.

## How the engine handles these tensors

1. **Loading follows the index, per tensor name.** With `--load-format b12x` the loader
   reads each tensor from the file the index names for it
   (`vllm/model_executor/model_loader/b12x_loader.py:238-248`).

2. **`wo_b` is a row-parallel layer holding FP8 values and native 32x32 block scales**
   (`vllm/models/deepseek_v4_1/attention.py:533-542`,
   `vllm/models/deepseek_v4_1/b12x_layers.py:304-343`): a `weight` of type
   `float8_e4m3fn` and a `weight_scale_inv` of type `float8_e8m0fnu`.

3. **The checkpoint bytes are kept unmodified.** The layer's weight method is wrapped by
   one that only checks the two types after loading. Its own description is "Keep
   checkpoint FP8 tensors unmodified for the fused WO projection"
   (`attention.py:304-324`).

4. **Each rank holds a slice of columns.** The loader slices a row-parallel weight along
   its input dimension: `local` columns per rank, starting at `rank * local`
   (`vllm/model_executor/layers/linear.py:1867-1883`). If the number of heads was padded
   to be divisible by the number of ranks, the tail of the last rank is padding
   (`vllm/model_executor/parameter.py:20-45`; the attribute is set at
   `attention.py:543-548`). The block scales are sliced the same way, in units of 32.

5. **Packing happens once, after loading** (`attention.py:1596-1632`,
   `setup_wo_projection`), with `block_size=(32, 32)`. The result is stored in
   `self._wo_projection_weights`. For 32x32 scales nothing is requantized: the packed
   `values` are `weight.contiguous()`, that is, the Parameter's own memory, and two scale
   tables (`scale_rows`, `scale_mma`) are new memory derived from the block scales. For
   two specific shapes there is also `values_tiled`, a second, tiled copy of the values
   (`flashinfer/experimental/b12x/gemm/_shared/wo_mxfp8.py:1447-1544`), which the decode
   kernel reads when its plan requires it
   (`flashinfer/experimental/b12x/gemm/wo_projection/_preparation.py:503-504`). Whether a
   deployment has it depends on the shape each rank holds, so the switch makes no
   assumption: it refreshes every derived tensor the packing function returns and aborts
   if one has changed shape. `wo_a` and `wo_b` are packed by two separate calls
   (`wo_mxfp8.py:1547-1585`), so `wo_b` can be handled on its own.

6. **The forward pass binds the packed object on every call**
   (`attention.py:1754-1767`). The prepared plan holds scratch space for activations, not
   weights (`wo_mxfp8.py:195-394`). The call sits inside captured CUDA graphs, which is
   why the next point matters.

7. **Weights must be updated in place, never replaced.** The engine's own reload code
   states the rule: "Copy processed values into original tensor storage (preserves
   cudagraph refs)" (`vllm/model_executor/model_loader/reload/layerwise.py:476`). A
   switch must overwrite the bytes of the existing buffers. A newly assigned tensor
   object would be seen by eager execution but missed by replayed graphs.

8. **The prefix cache is not aware of weights.** Block hashes are computed from the
   parent hash, the token ids and a few request fields
   (`vllm/v1/core/kv_cache_utils.py:618-666`). After a switch, cached blocks computed
   with the other weights would be reused unless the cache is cleared.

## Alternatives that do not work

- **LoRA.** The model rejects it: "V4.1 native kernels do not support LoRA adapters"
  (`vllm/models/deepseek_v4_1/nvidia/model.py:370-371`).
- **Per-request selection.** The fused kernel takes one set of weights per batch
  (`attention.py:1754-1767`). Two requests decoded together cannot use different
  variants.
- **The built-in weight reload.** Under the b12x loader it raises "b12x requires an
  active initial-load session" (`b12x_loader.py:276-281`), and it would not rebuild the
  derived tables.

## Integration without patching the engine

- **The worker extension** is a class passed with `--worker-extension-cls`. The engine
  mixes it into its worker class on every rank and logs the method names it gained
  (`vllm/v1/worker/worker_base.py:366-392`). The names must not collide with the worker's
  own. This extension adds three: `overlay_status`, `overlay_prepare` and
  `overlay_switch`.
- **The HTTP routes** come from an endpoint plugin, discovered through the entry point
  group `vllm.endpoint_plugins` and loaded only when `VLLM_PLUGINS` names it
  (`vllm/plugins/__init__.py:93-121`). The same variable is the allow-list for the
  engine's other plugin groups (`vllm/plugins/__init__.py:36-59`), so it must also name
  the plugins the image already loads.
- **The engine calls used** are those of the engine client: `pause_generation`,
  described there as "Pause generation to allow model weight updates",
  `resume_generation`, `is_paused` (`vllm/v1/engine/async_llm.py:913-964`),
  `reset_prefix_cache` (`:1087`) and `collective_rpc` (`:1142`), which runs one method on
  every rank. While paused, new requests are not scheduled; they are flushed on resume
  (`vllm/v1/engine/core.py:1006`).

The engine also exposes pause, resume and "run this method on every rank" as HTTP routes,
but only in its developer mode, which exposes much more on the public API port than a
switch needs. This project does not enable developer mode.

## Worker extension

**Prepare** (once per rank, at the first switch):

1. Read the side file's header. Every tensor must be a `layers.N.attn.wo_b` weight or
   scale of the expected type, and every weight must have its scale.
2. Locate the same tensors in the served checkpoint through its index. If the index
   already points at the side file, stop: the server must be started on the base
   checkpoint.
3. For each tensor, determine which columns this rank holds (point 4 above) and check
   the rows, the types and the one-byte element size against the live Parameter.
4. Read both variants from disk. Fingerprint this rank's columns of every tensor, and
   compute a digest of each whole variant. A rank sees only its own columns of the live
   model, so the whole-variant digests are what allows the ranks to confirm that they
   read the same files. The HTTP side compares them and refuses to switch if they
   differ, with or without `force`.
5. **Validate the column mapping**: fingerprint the same columns of every live tensor.
   They must all match one variant. If they match neither, the plugin's model of the
   layout does not fit this engine. The rank then stays unprepared and never writes,
   whatever it is asked.

**Switch** (per rank):

1. For every layer in order, for weight and scale: read the tensor from disk, select
   this rank's columns, and copy them over the live Parameter's memory. Padding columns
   are never read or written.
2. Rebuild the derived tensors with the model's own packing function and copy each over
   the existing one, unless it already is the Parameter's memory. Check that each copy
   took effect.
3. Read every switched tensor back and compare it with both variants.

All copies operate on bytes (through `uint8` views), so nothing depends on which
operations the installed torch supports for the two 8-bit float types. The writes run
inside `torch.inference_mode()`, because tensors created at load time may be inference
tensors.

**Status** reads back the first and the last switched layer. A write processes the
layers in order, so one that stopped halfway leaves those two in disagreement. `?full=1`
reads every tensor and adds `derived`: one digest per rank of every tensor the model
derived from the switched weights (the packed values, the two scale tables, and the
tiled copy where present). It can be read on a freshly started server before anything
has been prepared. After a switch to the overlay and back it must be byte for byte what
it was.

The active variant is always determined from the weights, never from stored state.

## HTTP API

`POST /overlay/switch`:

1. Query the status of every rank; prepare if any rank is unprepared.
2. If the ranks do not agree on one variant, stop, unless `force` was sent.
3. If the requested variant is already active, return without pausing.
4. Pause in the chosen mode, clearing the cache.
5. Switch on every rank. If every rank reports the requested variant: clear the prefix
   cache, resume, return.
6. Otherwise restore the previous variant on every rank. If that succeeds: clear the
   cache, resume, and return the error. If it fails, or no previous variant was known:
   leave generation paused and report it.

Only one switch runs at a time. A second call in the meantime returns 409.

A call that times out does not cancel what the ranks were asked to do: the request stays
queued on the workers and still runs. Because the restore is queued behind it, the final
state is still the previous variant. How the engine handles the late replies was not
traced.

## The gate: two endpoints on one server

With `OVERLAY_GATE_PREFIX=/free`, the base variant is served on the standard paths and
the overlay on the same paths under the prefix. The implementation is in
`mod/vllm_overlay_switch/gate.py` and imports nothing from the engine.

**Where it is installed.** The app is built in a fixed order: the routers, then the
endpoint plugins, then the engine's own middleware
(`vllm/entrypoints/launchers/app.py:42-54`,
`vllm/plugins/endpoint_plugins/interface.py:91-105`,
`vllm/entrypoints/serve/middleware/register.py:19-80`). Middleware added later wraps
middleware added earlier. A pure ASGI middleware added inside the plugin's
`attach_router` is therefore the innermost one: it sees a request after CORS, the API
key check, the request id and any `--middleware`, and nothing sits between it and the
routes. The installer verifies this order in the engine it installs into.

**Variant selection.** Under the prefix: the overlay. On the standard paths: the base,
unless the request's `model` is one of the names in `OVERLAY_GATE_MODELS`. Those names
are additional served names of the one model (`--served-model-name` accepts several; the
model list shows one entry per name,
`vllm/entrypoints/launchers/api_server/app_state.py:59-61`, and responses carry the
first, `vllm/entrypoints/openai/models/serving.py:144-147`), so a client that selects
from the list finds both variants under one address. The gate reads the name from the
body it parses anyway. A `model` that is not a string selects nothing and is left for
the engine to reject.

**Admission control instead of the engine's pause.** The kernel takes one set of weights
for everything it decodes together (`vllm/models/deepseek_v4_1/attention.py:1754-1767`),
so requests for the two variants have to be serialized. The engine's pause cannot do
that on its own: requests that arrive while it is paused are queued and run after the
resume, on whichever variant is active then (`vllm/v1/engine/core.py:2091-2124`). The
gate therefore decides before the engine sees a request. It keeps a single queue. A
request is admitted when its variant is active and no request for the other variant is
ahead of it. The request at the head of the queue may trigger a switch once nothing is
in flight. A request stops being in flight when the app has finished with it, which for
a streamed response is the end of the stream or the client disconnecting. In addition,
the switch itself pauses in `wait` mode, which lets requests the engine is already
running finish before anything is copied. Requests the engine has queued but not yet
started are not waited for: while paused, the scheduler counts only running requests
(`vllm/v1/core/sched/scheduler.py:3708-3718`) and starts no waiting ones (see Open
questions).

**Separate caches, no clearing.** A cached block is identified by a hash over its
tokens, its parent's hash and a few extra keys. The request's `cache_salt` is one of
them for the first block (`vllm/v1/core/kv_cache_utils.py:618-629`), and through the
parents it affects every later block. The weights are not part of the hash, which is why
the manual switch clears the cache. The gate instead sets the salt of every admitted
request to `overlay-gate:<side>`, followed by a digest of the client's own salt if it
sent one (the engine accepts at most 128 characters and no `/`, `@` or backslash,
`vllm/entrypoints/generate/base/protocol.py:36-51`). Blocks created with one variant can
then only be found by requests for that variant, which only run while it is active.
Because switching back is byte-exact (the `derived` digest), those blocks remain valid,
and the gate switches with `pause_generation(clear_cache=False)`
(`vllm/v1/engine/async_llm.py:913-949`). The body is parsed and re-serialized with
exactly one `cache_salt`, so a client cannot inject a second one.

**Allowed routes.** Only requests the gate has salted may run the model. It therefore
passes `GET`, `HEAD` and `OPTIONS`, three routes that render and count without running
the model (`/v1/messages/count_tokens` only renders the request and counts tokens: the
`count_tokens` handler of the messages API), and the four generation routes that accept
a salt. Every other route returns 403, and websockets are closed. A `/v1/responses`
request with `background` set is rejected as well: the engine responds to such a request
immediately and runs the model afterwards in a separate task
(`vllm/entrypoints/openai/responses/serving.py:508-556`), that is, after the gate has
counted the request as finished.

**API key.** The engine's key check matches on the start of the path and protects `/v1`,
`/v2`, `/inference` and `/cohere`
(`vllm/entrypoints/serve/middleware/authenticate.py:11, 47-62`).
`/free/v1/chat/completions` starts with none of them, so it reaches the gate unchecked,
and the gate then strips the prefix. Left as is, the prefix would bypass the key. The
plugin therefore instantiates the engine's own `AuthenticationMiddleware` with the keys
the engine itself uses (`--api-key`, else `VLLM_API_KEY`, as in `register.py:33-37`),
and the gate passes everything under the prefix through it, on the inner path. If a key
is configured and that class cannot be imported, `attach_router` raises and the server
does not start.

**Reasoning effort.** The chat request schema accepts seven words for `reasoning_effort`
(`vllm/entrypoints/openai/chat_completion/protocol.py:257-259`) and passes the value to
the renderer as a chat-template argument (`:606-627`); arguments in
`chat_template_kwargs` override it. The DeepSeek V4.1 renderer then accepts `none`
(thinking off), the words in its own table, and integers from 1 to 100, and raises for
anything else (`vllm/tokenizers/deepseek_v41.py:86-99`; the table `low` 50, `high` 75,
`xhigh` 75, `max` 100 is at `vllm/tokenizers/deepseek_v41_encoding.py:185-192`). So
`minimal` and `medium` pass the schema and fail in the renderer, an integer is accepted
by the renderer only inside the kwargs, and a client that offers all seven words fails
on two of them. `effort.py` translates the value in the gate's rewrite step, in both
places, and moves a top-level integer into the kwargs. The mapping itself is
configuration (the recipe), not code.

**Releasing a request.** A request is removed from the in-flight count by a plain
function call, not an awaited one, in the `finally` block of the app call. A request
task that is being cancelled must still be counted out, and an awaited step there could
be interrupted by a second cancellation, after which the other variant would wait
forever.

**Failed switches.** A switch that fails and restores the previous variant returns 503
to the request that triggered it, and the active variant keeps serving. A switch that
leaves generation paused, or that is interrupted halfway (the task that requested it was
cancelled), closes the gate: every request gets 503 until a manual forced switch has
written one variant to every rank.

**What the acceptance test can measure exactly.** The scheduler counts, per request, the
tokens it looked up in the prefix cache and the tokens it found
(`vllm/v1/core/kv_cache_manager.py:329-338`, `vllm/v1/metrics/stats.py:131-142`), and
`/metrics` exposes the totals (`vllm/v1/metrics/loggers.py:647-664`). Read around
individual requests, together with the count of finished requests to detect other
traffic, they show whether a request for one variant reused what the other had cached.

Not traced in the source: whether anything other than the block hash keeps results by
prompt across requests in the speculative-decoding path. The draft layers have weights
of their own that are not switched. The test above would reveal cross-variant reuse in
the counters, but not a cache that the counters do not cover.

## Memory and timing

For the motivating case, the side file holds 26 weights of 5120 x 8192 (42 MB each
according to its header) with their 160 x 256 block scales. A rank writes its own
columns of each; how many columns that is depends on the number of ranks and is read
from the live model. Nothing is kept between switches. During a switch, one tensor from
the file and this rank's slice of it are held in CPU memory, and the packing function
allocates scratch memory on the device, which is released at the end. A switch reads
every tensor of the variant in full, about 1.1 GB per rank. Preparation reads both
variants once, about 2.2 GB.

Measured on the test setup (three ranks): each rank holds 3,072 columns, the eight
groups being padded to nine. Two ranks write 409 MB per switch, and the third, which
holds 2,048 real columns, 273 MB. Preparation took 3.9 s. A switch took 1.4 to 2.0 s end
to end: the pause before the copy took 0.03 s, and the rest is the copy and the
read-back of every tensor. Six consecutive switches left the nodes' free memory
unchanged. A full status, which reads every tensor back and fingerprints the derived
ones, took about three seconds, which is why the fingerprint is not part of a switch.

With the gate on the same setup: a response took 0.69 s on average when its variant was
active and 2.22 s when the server had to switch first (36 responses, 2.02 to 2.66 s).
Free memory, sampled five times a second on each node during six switches, dropped by at
most 0.07 GiB and recovered. Requests for both variants arriving together are strictly
serialized, with a switch at every change of variant: in two bursts of 12, half for each
variant, a response took 0.6 to 26.2 s, 11.2 s on average.

A second configuration gave the same server a 1,048,576-token context and
`--kv-cache-memory-bytes 3g`, which leaves each node a few GiB free instead of six to
eight. The acceptance test passed there as well, with the same timings (0.61 s and
2.24 s). The first two switches reduced free memory by 0.03 GiB, and over four minutes
of switches and bursts the lowest reading on the tightest node was 1.89 GiB; it had been
1.7 GiB after its first requests, before any switch. A switch needs no significant
memory of its own.

## Acceptance test

`tools/acceptance.py` sends no sensitive prompts and needs no reference outputs. It
answers two questions separately.

**Does the model use the weights a switch wrote?** Two models with different weights
assign different probabilities to the next token at some position where both have seen
exactly the same text.

1. Sample the active variant (A) several times: a handful of neutral prompts at
   temperature 0 with thinking off, recording the tokens and their alternatives. Switch,
   sample the other variant (B). Switch back, sample A again.
2. For each prompt, walk the responses token by token while they all agree. At every
   position up to and including the first divergence, every response was produced from
   an identical context. At each such position the two candidates are the token chosen
   most often and the rival the responses give the most weight. A response's margin is
   the log probability of the first minus that of the second.
3. A position distinguishes the variants when every A margin, before and after, lies on
   one side and every B margin on the other. A prompt does when one of its positions
   does. Looking only at the point of divergence was not enough: the two variants often
   choose the same token while weighting its rival very differently, and that is the
   more stable signal.

**Does switching back restore the model?** Exactly: the `derived` digest of every rank
is read when the test begins and again when it ends, and the two must be equal. The
weights themselves are read back by the worker after every switch. For a plugin version
without the digest, the outputs have to serve: A after the return must not be
distinguishable from A before.

**Why margins and several passes.** The first version of this test compared the chosen
tokens over two passes and assumed that a model repeats its own choice. On the test
hardware that assumption was false: identical requests moved a margin by one to three
logits (the logits there come in steps of a quarter), enough to change the chosen token
wherever two tokens were close. With two passes, two samples of the same variant differ
by chance one time in three, and the test reported a failed restore that had not
happened. The test now samples each variant six times by default, gives each request its
own `cache_salt` so that no response reuses another's cached prefix, and judges whether
two groups of margins overlap at all. Two groups of six from the same source fail to
overlap by chance one time in 462; twelve against six, about one time in nine thousand.
The test reports how many positions it examined so that the result can be read against
those odds.

`--read-only` samples a server without switching and saves the responses. This provides
a reference, for example of the overlay loaded as a model of its own, to compare with
the switched state.

## Validation results

On the test setup:

- Captured CUDA graphs follow an in-place overwrite: the outputs changed with a switch
  and returned with the switch back, and the overlay loaded as a model of its own
  matched the switched state at every position that distinguished base from switched.
- No other component keeps its own copy of these weights: the same comparison would have
  revealed a state that was neither base nor overlay.
- The packing function is deterministic: the digest of the derived tensors after
  switching to the overlay and back equalled the one from startup, on two separate
  launches, and the startup digests were the same on both.
- The image registers no plugins other than vLLM's own two.
- The gate works on a real server. Its acceptance test passed: 156 responses each
  carried the header of the requested variant, and 60 responses requested alternately
  and in bursts, by model name and under the prefix, all matched their own variant at a
  reference position.
- The cache salt separates the variants in the engine's own counters: a prompt repeated
  on its variant reused its prefix, the other variant's first request for the same
  prompt under the same client salt reused nothing, and after two switches the first
  variant still reused its own.
- Sixty switches into a launch, every tensor read back as base on every rank and the
  derived digests were those from startup.
- A client that disconnected in the middle of a streamed response did not block the
  other variant: twelve times out of twelve, the next request for the other variant was
  served within three seconds, by its own variant.
- The effort translation runs before the engine's own validation: values the engine
  rejects with 400 returned 200 at the top level and in `chat_template_kwargs`, by
  either model name and under the prefix.

## Open questions

- A long generation on one variant while the other is requested: the wait is by design,
  but its length was not measured.
- A client that disconnects while queued at the gate (it keeps its place).
- Whether a request can stay queued inside the engine after its HTTP exchange has ended,
  for example when a client disconnects while the request is being submitted. The
  gate's `wait` pause would not drain it, and it would then run on the other variant
  under its own cache salt. Switching with an `abort` pause once nothing is in flight
  would remove the case; that change has not been made or tested.
- Memory with long cached conversations on both variants at once: the test prompts are
  short.
- The API-key check under the prefix on a real server, and a server with `--root-path`.
- Worker and queue state after a `collective_rpc` timeout.
- A switch while requests are in flight, and under load from several clients.
- Other tensor-parallel sizes, where the padding falls differently.
