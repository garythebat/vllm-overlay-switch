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
base weights and can be flipped to the overlay, and back, with one call.

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
| Not tested | a flip while requests are in flight or under load from several clients; other tensor-parallel sizes; finding the checkpoint by its served name on hardware (the test setup named the folder with `OVERLAY_SWITCH_BASE_DIR`) |

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
   `--worker-extension-cls vllm_overlay_switch.worker.OverlayWorkerExtension`.
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

## Limits

- One engine build (see Scope), one model path (DeepSeek V4.1, native kernels), one
  kind of tensor (`attn.wo_b` weight and scale). Anything else in the side file is
  refused.
- One set for everyone. A flip changes the model for all requests; a choice per request
  is not possible with this kernel (one set of weights per batch).
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
