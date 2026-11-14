# vllm-overlay-switch

Flip a small set of weights in a running vLLM server, in place, without a relaunch.

## Why

A large checkpoint can take minutes to load across several machines. Some weight
overlays change only a handful of tensors. The case that started this: 52 tensors,
about 1.1 GB, out of a checkpoint of roughly 475 GB (the attention output projection
`wo_b` of layers 10 to 35 of DeepSeek V4.1 Flash).

Serving the base model and such an overlay as two separate runs means a stop and a
relaunch every time you change your mind, several minutes each way, and two copies
never fit in memory together. The aim here is one loaded model that normally runs the
base weights and can be flipped to the overlay, and back, with one call. Or, with the
gate switched on, one server that answers with the base weights on its usual paths and
with the overlay on the same paths under a prefix, and flips by itself.

## Status

**Run on one setup: three machines with one GPU each (DGX Spark), tensor parallel 3, the
engine build under Scope, 14 November 2026. The acceptance test passed there.** Treat it as
unproven on any other setup until the acceptance test below has passed on yours.

| Part | How it has been checked |
|---|---|
| The order of a flip, putting things back when a rank fails, the forced way out, who may call | unit tests with a stand-in engine (`tests/test_logic.py`) |
| The HTTP routes | unit tests with FastAPI's test client and a stand-in engine (`tests/test_api.py`) |
| Reading tensors out of safetensors files | unit tests on hand-made files (`tests/test_stfile.py`) |
| The worker half: which columns a rank holds, overwriting in place, reading back, leaving padding alone, refusing what it does not understand | unit tests with real torch tensors (torch 2.13.0, the engine image's version, CPU build), real files, three simulated ranks, and stand-ins for the engine's modules (`tests/test_worker.py`) |
| That the code fits the engine | read against the engine source at the commits listed under Scope; references in [docs/design.md](docs/design.md) |
| The installer `mod/run.sh` | dry run against a folder laid out like the image's package folder; then run inside the image on each of the three machines at every launch. It found no plugins in the image beyond vLLM's own two |
| The acceptance tool itself | run against a stand-in server whose behaviour is known: a pass, each failure verdict, a drift between readings, an older switch without the digest, "cannot tell" (`tests/test_acceptance.py`) |
| A flip on a GPU, under captured CUDA graphs, across machines | run. Preparing took 3.9 s; a flip 1.4 to 2.0 s end to end, of which the pause before the copy is 0.03 s. After every flip each machine read all its tensors back as the set asked for |
| That flipping back restores the model | exact: the digest of everything derived from the switched weights, read on a freshly started server before anything was prepared, was the same on every rank after flipping to the overlay and back. Seen on two separate launches |
| That the model reads what a flip wrote | six independent passes per set: five of eight prompts told the two sets apart (every margin of one set on one side, every margin of the other on the other) |
| That the flipped state is the overlay model | the overlay was also loaded as a model of its own and read with `--read-only`. Wherever base and flipped stood apart, it lay on the flipped side with nearly the same medians, and at none of 34 positions did it stand apart from the flipped state |
| Memory | six flips in a row left free memory where it was |
| The gate (two endpoints on one server): who is let in when, the line, a failed or interrupted flip, what passes and what is refused, the API key under the prefix | unit tests with a stand-in engine and a stand-in server that records what the engine held while each request ran (`tests/test_gate.py`), and through FastAPI as the plugin installs it (`tests/test_api.py`). 31 faults made on purpose in the gate and its acceptance tool were each caught by a test. **Not run on hardware yet** |
| The gate's acceptance tool | run against a stand-in server that can go wrong in each way the tool exists to catch (`tests/test_gate_acceptance.py`). Its rule for reading an answer was set after replaying real readings through it: see "Acceptance test" |
| Not tested | the manual switch while requests are in flight or under load from several clients; the API-key check under the prefix on a real server (the stand-in copies the engine's rule); other tensor-parallel sizes; finding the checkpoint by its served name on hardware (the test setup named the folder with `OVERLAY_SWITCH_BASE_DIR`) |

## How it works

No engine file is edited. Two small pieces are added through extension points the
engine already has:

- **On every machine**, a worker extension (`--worker-extension-cls`) that can overwrite
  the live weight buffers of the switched tensors with one set or the other.
- **On the API server**, an endpoint plugin with two routes: status and switch.

A flip:

1. Pauses generation. Requests that are running finish first; requests that arrive
   meanwhile are queued by the engine.
2. On every machine, reads the chosen set from disk, keeps the columns that machine
   holds, and copies them over the existing buffers. Then it rebuilds the small tables the
   model derived from these weights at startup and copies those over the existing ones
   too. Nothing is replaced, only overwritten, because captured CUDA graphs hold the
   addresses of these buffers.
3. Reads every switched tensor back from the device and compares it with both sets.
4. If the machines do not all report the wanted set, copies the previous set back. If
   that fails as well, generation stays paused and the answer says so: a model whose
   machines hold different weights must not answer anyone.
5. Clears the prefix cache (it does not know about weights) and resumes.

Before its first flip, every machine proves its own reading of the files against the
live model: it must find the live weights to be exactly the base set, tensor by tensor.
If this tool's idea of which columns a machine holds did not fit the engine in front of
it, that proof fails and nothing is ever written.

Every machine also reports a digest of the whole of both sets as it read them from its
own disk. A flip is refused unless all machines report the same two digests, so a side
file that differs on one machine cannot go unnoticed. `force` does not override this.

Nothing is held in memory between flips. The price is that a flip reads the switched
tensors from disk each time (about a gigabyte per machine for the case above).

## Install

The server must be started on the **base** checkpoint. The alternate tensors come from
a **side file**: a safetensors file holding `layers.N.attn.wo_b.weight` (F8_E4M3) and
`layers.N.attn.wo_b.scale` (F8_E8M0) under exactly those names, with the base
checkpoint's shapes. This is the same kind of file an index-remapped overlay folder
points at. It must be readable at the same path inside the container on every machine.

With [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker):

1. Copy this repository's `mod/` folder into your checkout as `mods/overlay-switch`.
2. Merge the four items of [examples/recipe-additions.yaml](examples/recipe-additions.yaml)
   into your recipe: the mod, `VLLM_PLUGINS`, `OVERLAY_SWITCH_FILE`, and
   `--worker-extension-cls vllm_overlay_switch.worker.OverlayWorkerExtension`. A fifth,
   `OVERLAY_GATE_PREFIX`, turns on the two endpoints described below, and a sixth,
   `OVERLAY_GATE_MODELS`, lets the model name choose between them.
3. Launch as usual. The mod prints the exact `VLLM_PLUGINS` value for your image; that
   variable is an allow-list for every vLLM plugin, so it has to name the ones the image
   already loads as well. In the server log, look for
   `Injected ... OverlayWorkerExtension ... for extended collective_rpc calls`.

Without the mods mechanism, `pip install .` inside the image does the same as the mod.

## Use

By default the routes answer only to the machine the server runs on.

```sh
# which set is live (add ?full=1 to read every tensor back, not just the first and last layer)
curl -s http://127.0.0.1:8000/overlay/status

# flip to the overlay, and back
curl -s -X POST http://127.0.0.1:8000/overlay/switch -H 'Content-Type: application/json' -d '{"variant": "overlay"}'
curl -s -X POST http://127.0.0.1:8000/overlay/switch -H 'Content-Type: application/json' -d '{"variant": "base"}'
```

- The first flip also prepares: every machine reads both sets once and proves its cut.
- `"mode": "wait"` (the default) lets running requests finish before the copy;
  `"mode": "abort"` ends them.
- The served model name does not change. Nothing tells a client which set answered,
  except the status route.
- If a flip leaves generation paused, `{"variant": "base", "force": true}` writes that
  set onto every machine whatever is there, and resumes only if they then all agree.
  `force` cannot make a machine write that never proved its cut.
- To let other addresses call the switch, set `OVERLAY_SWITCH_ALLOW` to a comma
  separated list. If `OVERLAY_SWITCH_TOKEN` is set, every call must carry it in the
  `X-Overlay-Token` header. These routes are outside the paths the engine's own API key
  guards.

## Two endpoints on one server

Set `OVERLAY_GATE_PREFIX=/free` in the launch and the one loaded model answers on two
sets of paths:

| Base URL for a client | Answers with |
|---|---|
| `http://HOST:8000/v1` | the base set |
| `http://HOST:8000/free/v1` | the overlay |

A client picks a side by the base URL it is given and needs nothing else. The served
model name is the same on both. Every answer of a request that ran the model carries a
header, `x-overlay-side: base` or `overlay`.

Some clients pick from the server's model list instead of taking a base URL. For those,
give the one model a second served name and tell the gate that this name means the
overlay:

```
vllm serve ... --served-model-name BASE_NAME OVERLAY_NAME
OVERLAY_GATE_MODELS=OVERLAY_NAME
```

`GET /v1/models` then lists both names under the one address, and a request on the usual
paths that names `OVERLAY_NAME` gets the overlay. Under the prefix every request gets the
overlay, whatever it names. The engine puts the first served name into the `model` field
of every answer and into its metrics, so name the base first; the header says which side
answered.

What the gate does:

- Requests for the side that is live are let in as they come and run together, as
  without the gate.
- The first request for the other side waits until the requests in flight have ended.
  Then the server flips (the same flip as above) and that request runs. Requests that
  arrive behind it wait their turn in order, whichever side they are for, so neither
  side can starve the other.
- A request is never handed to the engine while its side is not live. The two sets never
  answer in the same instant: the kernel takes one set of weights for everything it
  decodes together.
- Each side keeps its own cached prefixes. The gate marks every request with a cache
  salt that names its side (a client's own `cache_salt` is kept, as a digest behind the
  side), so nothing computed with one set is reused with the other, and a flip does not
  clear the cache. Coming back to a long conversation on either side does not mean
  reading it again, as long as the cache has room for both.

What it costs: a change of side waits for the requests in flight and then takes one flip
(1.4 to 2.0 s on the setup above). Two clients that take turns, one on each side, pay
that on every request.

What it refuses, so that nothing can run the model behind its back:

- Any route it does not know. Let through are `GET` requests, three routes that only
  count (`/tokenize`, `/detokenize`, `/v1/messages/count_tokens`), and the four that run
  the model and take a cache salt: `/v1/chat/completions`, `/v1/completions`,
  `/v1/responses`, `/v1/messages`. Embeddings, scoring, the realtime websocket and the
  rest answer 403 or are closed.
- A request that asks to be run in the background (`"background": true` on
  `/v1/responses`): the model would run after the gate had let it go.
- A flip by hand. `POST /overlay/switch` then only takes `"force": true`, the way out
  after a flip that failed. If a flip leaves the machines in an unclear state, the gate
  answers 503 to everyone until that forced flip has put one set on all of them.

The server's API key. The engine checks its key by how a path begins (`/v1` and a few
more), and `/free/v1/...` does not begin that way. The gate therefore runs the engine's
own key check again for everything under the prefix, with the same keys. If the server
has a key and the engine's check cannot be found, the server does not start.

`GET /overlay/status` then also reports the gate: its prefix and model names, which side
is live, how many requests are in flight and waiting, how many flips so far, and the
seconds requests have spent waiting at the gate.

## Acceptance test

```sh
python3 tools/acceptance.py http://127.0.0.1:8000 MODEL_NAME --out report.json
```

It reads the live set several times (a few plain prompts at temperature 0), flips, reads
the other set, flips back and reads again. At every position where all answers had read
the same words it compares, by set, the margin between the most chosen token and its
strongest rival. It passes when at least two prompts tell the sets apart and flipping
back restores the model: exactly, by a digest of everything derived from the switched
weights that the status route reports with `?full=1`.

Identical requests do not return identical numbers on every deployment. On the setup
above they moved a margin by one to three logits, which is why the test reads each set
six times, gives every request its own `cache_salt`, and judges whether two groups of
margins overlap at all. The first version of this test compared single choices over
two passes and reported a failed return that was not one. It needs no reference
answers and asks nothing sensitive. Details and the reasoning are at the top of the
file and in [docs/design.md](docs/design.md).

`--read-only` reads a server without touching any switch and saves the answers, for a
reference reading to set beside the flipped state.

For a gated server:

```sh
python3 tools/gate_acceptance.py http://HOST:8000 MODEL_NAME --prefix /free --out report.json
```

With `--overlay-model OVERLAY_NAME` it asks for the overlay by name as well: every other
mixed request for the overlay goes over the usual path under that name, and the cache
check asks once by name for what was cached under the prefix.

It answers four questions, three of them exactly.

- Is every answer marked with the side it was asked on? (the header)
- Do the two sides keep their cached prefixes apart, and does a flip leave them alone?
  One long prompt goes to both endpoints under the same client salt, and the server's
  own counters on `/metrics` are read around each request: the first request on the
  overlay must reuse nothing of what the base side cached, and base must still reuse
  its own after going there and back.
- Did the server flip as often as the side changed? (the gate's own count, with
  `--status`)
- Is a request ever answered by the other side's weights? This one is read from the
  answers. Each endpoint is read six times; a position where the two groups of margins
  stand at least three logits apart becomes a landmark. The prompts that have one are
  then asked on both endpoints in turn, a flip before every request, and in bursts of
  both at once. An answer that lies as far over as the other side's reference answers
  fails the test. One that is past the middle but short of the other group reads like
  neither, and the test says it cannot tell.

That last rule was set by measurement, after a simpler one ("the wrong side of the
middle") turned out to cry wolf. Readings from the setup above in which nothing was
mixed were replayed through the rule: of 738 answers none lay with the other group and
2 lay in the stretch between. With the labels swapped on purpose, 185 of 216 lay with
the other group, 31 in the stretch between, and none passed as its label.

## Limits

- One engine build (see Scope), one model path (DeepSeek V4.1, native kernels), one
  kind of tensor (`attn.wo_b` weight and scale). Anything else in the side file is
  refused.
- One set at a time. A flip changes the model for every request that runs; the kernel
  takes one set of weights per batch. The gate gives each request the set its path asks
  for by taking turns, not by running both at once.
- Behind the gate a request for the other side waits as long as the longest request in
  flight. A client that never finishes reading its answer holds the other side up.
- A client that gives up while its request is still waiting at the gate keeps its place
  in the line. When its turn comes the server flips for it, and the engine then drops
  the request: a flip for nothing.
- The gate needs a single API server process (`--api-server-count 1`, the default without
  data parallelism) and refuses to start otherwise: its line lives in one process.
- The gate has not been tried with `--root-path`, and it holds a request's whole body in
  memory to mark it (up to 64 MB).
- The switched tensors are read from disk at every flip.
- Layers that belong only to a speculative-decoding draft model are not switched.

## Tests

```sh
python3 -m unittest discover -s tests -v
```

`tests/test_worker.py` needs torch and numpy, `tests/test_api.py` needs fastapi and
httpx; each is skipped when its packages are missing. The rest needs only the
standard library. Nothing needs a GPU.

## Scope

Written against one engine build. The installer checks that the pieces it stands on
exist and refuses otherwise; the worker refuses at run time if the live model does not
look as expected.

| Part | Source | Commit |
|---|---|---|
| vLLM fork | github.com/local-inference-lab/vllm, branch `dev/karmic-kraken` | `740e3776b4d645733fe4f8b90809fddea2f82b96` |
| B12X kernels | github.com/local-inference-lab/flashinfer (`flashinfer/experimental/b12x`) | `2f5210dd1566a8dfef754f68691b898b8d870244` |
| Packaging | github.com/eugr/spark-vllm-docker (the B12X image and its mods mechanism) | `a4c1729` |

Model path covered: DeepSeek V4.1 on NVIDIA through `vllm/models/deepseek_v4_1`.
Hardware it was designed for: three NVIDIA DGX Spark machines, tensor parallel 3.

## Credits

- The vLLM project and the local-inference-lab fork that carries the DeepSeek V4.1
  native path.
- FlashInfer and the B12X kernels.
- eugr's spark-vllm-docker, whose mods mechanism this rides on.
- The Keys overlay for DeepSeek-V4.1-Flash-1017 by drowzeys, the overlay that prompted
  the idea. Nothing from that overlay is included here.

Built by Anima.

## Licence

Not chosen yet. Until a licence file is added, all rights are reserved by the owner.
