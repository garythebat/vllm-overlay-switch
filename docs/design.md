# Design

Status: design from source reading. Nothing here has run on hardware yet. Every
reference below is to the pinned commits listed in the README.

## The problem in one paragraph

A weight overlay that replaces a few tensors can be loaded by remapping those tensor
names in `model.safetensors.index.json` to a small side file. That gives a second
checkpoint folder that shares every other file with the base. But a server can hold
only one of the two at a time, and changing between them is a stop and a relaunch.
The goal is one server that holds both versions of the few tensors and switches which
one is active with a single call.

## What the engine does with these tensors

References are `path:line` in the vLLM fork unless they start with `flashinfer/`.

1. **Loading follows the index, per tensor name.** With `--load-format b12x` the loader
   reads each tensor from the file the index names for it and skips other files that
   also contain that name
   (`vllm/model_executor/model_loader/b12x_loader.py:238-248`).

2. **`wo_b` is an FP8 weight with native 32x32 block scales.** After loading, each
   attention module packs its output projection once
   (`vllm/models/deepseek_v4_1/attention.py:1596-1632`, `setup_wo_projection`), with
   `block_size=(32, 32)`, and keeps the result in `self._wo_projection_weights`.

3. **Packing keeps the checkpoint bytes.** For 32x32 scales the kernel library never
   requantizes; the packed `values` are `weight.contiguous()`, and two scale tables
   (`scale_rows`, `scale_mma`) are derived from the block scales. A tiled copy of the
   values exists only for one specific shape
   (`flashinfer/experimental/b12x/gemm/_shared/wo_mxfp8.py:1447-1546`). `wo_a` and `wo_b`
   are packed by two separate calls (`wo_mxfp8.py:1547-1586`), so `wo_b` can be handled
   alone.

4. **The forward pass reads the packed object on every call**
   (`attention.py:1754`) and hands it to a fused kernel. That call sits inside captured
   CUDA graphs, which is why the next point matters.

5. **In place, never by replacement.** The engine's own reload code states the rule:
   "Copy processed values into original tensor storage (preserves cudagraph refs)"
   (`vllm/model_executor/model_loader/reload/layerwise.py:476`). A switch must overwrite
   the bytes of the existing buffers. Assigning a new tensor object would be seen by
   eager execution and missed by replayed graphs.

6. **The prefix cache does not know about weights.** Block hashes are computed from
   the parent hash, the token ids and a few request fields (LoRA name, multimodal
   content, `cache_salt`, prompt embeds); see `vllm/v1/core/kv_cache_utils.py:618-666`.
   After a flip, cached blocks computed with the other weights would be reused unless
   the cache is cleared.

## What does not work, and why

- **LoRA.** The model refuses it: "V4.1 native kernels do not support LoRA adapters"
  (`vllm/models/deepseek_v4_1/nvidia/model.py:370-371`).
- **A choice per request.** The fused kernel takes one set of weights per batch
  (`attention.py:1754-1767`). Two requests decoded together cannot use different sets.
- **The built-in weight reload.** Under the b12x loader it raises "b12x requires an
  active initial-load session" (`b12x_loader.py:276-281`), and it would not rebuild the
  derived scale tables.

## The switch

At startup, on every tensor-parallel rank:

1. Read the alternate tensors from the side file.
2. Cut them down to this rank's shard exactly as the loader did for the base weights.
3. Pack them with the same function the model uses for `wo_b`.
4. Keep the packed alternate set, and a copy of the packed base set, next to the live
   buffers. For the motivating case this is about 26 buffers of a few megabytes per
   rank for each set.

On a flip:

1. Pause generation and let running requests finish.
2. On every rank, copy the chosen set into the live buffers in place: the values, the
   two scale tables and the tiled copy if there is one.
3. Check that every rank reports the same active set. If ranks disagree, copy the
   previous set back; if that fails too, stay paused and say so.
4. Clear the prefix cache.
5. Resume.

The served model name does not change. A status call reports which set is active.

## Control and exposure

The engine has pause, resume, prefix cache reset and "run this method on every rank"
calls, but only behind its developer mode, which exposes much more than a switch needs
on the public API port. This project does not enable developer mode. It adds one narrow
route for the flip and one for status, and by default accepts them from the local
machine only.

## Acceptance test

The test asks nothing sensitive and needs no reference answers. It uses the fact that
two models with different weights choose different next tokens at some point where both
have read exactly the same words, and that each model repeats its own choice.

1. Ask a handful of plain prompts at temperature 0 with thinking off, twice, recording
   the tokens and their log probabilities.
2. Flip. Ask again, twice. Flip back. Ask again, twice.
3. For each prompt, walk the answers token by token while they all agree. At the first
   point of disagreement, both runs on the base set must agree with each other, both
   runs on the alternate set must agree with each other, and the two sets must differ.
4. After flipping back, the choices must be the base set's again.

Two identical requests to the same model do not return identical numbers on this kind
of deployment, and wording can drift after a few tokens. That is why the test compares
choices at identical contexts and each model against its own repeat, not raw numbers.

## Open questions, to be settled by running it

- Do captured graphs follow an in-place overwrite here, as the reload code implies?
- The exact per-rank tensor shapes under tensor parallelism, including padded groups
  (`vllm/model_executor/models/config.py:503-513`).
- Flip latency and the extra memory actually held.
- Whether any other component keeps its own copy of these weights.
