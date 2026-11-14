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

**Design only. Nothing in this repository has run on hardware yet.**

What is established, by reading the engine source at the exact commits listed under
Scope (file and line references are in [docs/design.md](docs/design.md)):

- The overlay tensors are held as plain GPU buffers. For this model's native 32x32
  block scales the kernel library uses the checkpoint bytes as they are, plus two small
  scale tables derived from them. Nothing is baked into compiled kernels.
- `wo_b` is packed separately from `wo_a`, so only `wo_b` has to be touched.
- The engine's own reload code states the rule a switch must follow: copy new bytes
  into the existing tensor storage, so that captured CUDA graphs keep pointing at valid
  memory. Replacing the tensor object instead would leave replayed graphs on the old one.
- The prefix cache is keyed by tokens and a few request fields, not by the weights, so
  it has to be cleared on every flip.
- LoRA cannot do this job for this model (the model refuses LoRA adapters), a choice
  per request is not possible (the fused kernel takes one set of weights per batch),
  and the built-in weight reload refuses to run under the loader this model uses.

What is not established until it runs:

- that captured graphs really follow an in-place overwrite in this engine,
- the exact tensor shapes per rank under tensor parallelism,
- how long a flip takes and how much extra memory it holds.

What was measured before any of this, on stock serving: an index-remapped overlay is
picked up by the loader and changes the model's next-token choices at identical
contexts, repeatably. The same measurement is the acceptance test for the switch (see
[docs/design.md](docs/design.md), "Acceptance test").

## Scope

The patch is pinned. It targets one engine build and refuses to apply to anything else.

| Part | Source | Commit |
|---|---|---|
| vLLM fork | github.com/local-inference-lab/vllm, branch `dev/karmic-kraken` | `740e3776b4d645733fe4f8b90809fddea2f82b96` |
| B12X kernels | github.com/local-inference-lab/flashinfer (`flashinfer/experimental/b12x`) | `2f5210dd1566a8dfef754f68691b898b8d870244` |
| Packaging | github.com/eugr/spark-vllm-docker (the B12X image and its mods mechanism) | `a4c1729` |

Model path covered: DeepSeek V4.1 on NVIDIA through `vllm/models/deepseek_v4_1`.
Hardware it was designed on: three NVIDIA DGX Spark machines, tensor parallel 3.

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
