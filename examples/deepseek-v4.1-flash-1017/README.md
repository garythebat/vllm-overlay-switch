<!--
SPDX-FileCopyrightText: 2026 garythebat
SPDX-License-Identifier: Apache-2.0
-->

# Example: DeepSeek V4.1 Flash 1017 on three DGX Spark nodes

The configuration the results in the main README were measured on: DeepSeek V4.1 Flash
1017 on three nodes of the DGX Spark family with tensor parallel 3, serving the base
model and the Keys overlay from one loaded copy. IP addresses below are documentation
addresses.

- [recipe.yaml](recipe.yaml): the recipe used for the tests.
- [start-with-knobs.sh](start-with-knobs.sh): the start script, with three lines changed
  for publication (listed in its header).

## Components

| Component | Source | Version | Licence |
|---|---|---|---|
| Launcher and image | [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) | checkout at `a4c172976a5f9f8669b356a0891bf972cae14fd4`; image `eugr/spark-vllm-b12x:nightly-20261010`, digest `sha256:d999354461ae004c754333fd50a69867816758c864687e41b9a224b748a2c104` | MIT |
| Engine in the image | [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm), branch `dev/karmic-kraken` | `740e3776b4d645733fe4f8b90809fddea2f82b96` | Apache-2.0 |
| Kernels in the image | [local-inference-lab/flashinfer](https://github.com/local-inference-lab/flashinfer) | `2f5210dd1566a8dfef754f68691b898b8d870244` | Apache-2.0 |
| Base model | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | revision `1bae0e1088993b9b82de08f59de5be5a34e42170` (83 files, 48 shards, about 475 GiB) | MIT |
| Chat template | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | `chat_template.jinja` at revision `2cba9e42aa026125f3ed06c6d98c1db82f7ca027` | MIT |
| Overlay side file | [drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated) | `wo_b_l10_35_1017.safetensors` at revision `de412d2249632903dace23399526e00a145aada5` (1,091,589,280 bytes, sha256 `88324ceaa2c0754a2bdb4ef6bb388a088438e352358343e6c2c882009cdde864`) | MIT, behind the author's Responsible Use Agreement |

Notes:

- **Chat template.** At the revision above, the 1017 release had no
  `chat_template.jinja`, so the server uses the one from the first release. The
  `encoding/` folders of the two releases were byte-identical. The 1017 repository has
  since added its own template, which is untested here.
- **Overlay.** The overlay repository is gated: you have to accept the author's
  Responsible Use Agreement on the model page before downloading, and its terms apply to
  your use of the file. Nothing from that repository is included here, and only the one
  file above is used. The server loads the stock checkpoint: no index is rewritten and
  nothing is merged into the shards.

## Memory settings

The recipe is leaner than the one it is derived from, because this model leaves little
free memory on these nodes. Two sources informed the settings:

- Christopher Owen's [spark-ds41f](https://github.com/christopherowen/spark-ds41f)
  (Apache-2.0) publishes memory figures and launch settings for three-node runs of this
  model. The settings here were planned against those figures, and two of them also
  appear in its configuration (as of October 2026): `MAX_JOBS` 2 and
  `--mm-processor-cache-gb 0`.
- [spark-vllm-docker issue 404](https://github.com/eugr/spark-vllm-docker/issues/404)
  describes an engine that hangs after running out of memory during a kernel compile.
  The limits on compile jobs (`B12X_COMPILE_WORKERS`, `MAX_JOBS`, `NINJAFLAGS`,
  `MAKEFLAGS`, `FLASHINFER_NVCC_THREADS`) are there to avoid it.

The remaining choices (text only, a 262,144-token context, 2,048 batched tokens, CUDA
graph capture up to 16) were made for this setup. The resulting free memory is in the
table at the end.

## Hardware

Three nodes of the DGX Spark family (two NVIDIA DGX Spark, one Gigabyte AI TOP ATOM),
128 GB of unified memory each, connected over RoCE. Each node has the model files and
the side file on its own disk at the same paths.

## Setup

1. Set up spark-vllm-docker for three nodes as described in its README, at the commit
   above, with the image above on every node.
2. Download the base model at the revision above into the Hugging Face cache of every
   node. The recipe also reads `chat_template.jinja` from the first release, so that file
   has to be in the cache too.
3. Accept the overlay's agreement and download the side file. Place it on every node so
   that the container sees it as
   `/root/.cache/huggingface/ablit-overlay/wo_b_l10_35_1017.safetensors`, and verify its
   sha256 on each node.
4. Copy this repository's `mod/` folder into the spark-vllm-docker checkout on the head
   node as `mods/overlay-switch`. The launcher copies it into the container on every node
   at each launch.
5. Copy `recipe.yaml` into the checkout's `recipes/` folder as
   `deepseek-v4.1-flash-1017-switch.yaml` and start it, either with the launcher directly
   or with `start-with-knobs.sh`.
6. Check the head node's server log for
   `Injected ... OverlayWorkerExtension ... for extended collective_rpc calls` and
   `Loaded endpoint plugin overlay_switch`.

The server starts on the base weights. The first request for the overlay triggers the
one-time preparation and the first switch.

## Client view

One address, two names in the model list:

```sh
curl -s http://192.0.2.10:8000/v1/models
# deepseek-ai/DeepSeek-V4.1-Flash-1017   the base model
# deepseek-v4.1-flash-1017-free           the overlay

curl -si http://192.0.2.10:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "deepseek-v4.1-flash-1017-free", "messages": [{"role": "user", "content": "Say hello in five words."}], "max_tokens": 20}' \
  | grep -i -e x-overlay-side -e content
# x-overlay-side: overlay
```

`http://192.0.2.10:8000/free/v1` is served by the overlay regardless of the model name.
The `model` field of every response contains the first served name; the header reports
which variant answered.

## Verification

On the head node, or through a tunnel to its port 8000:

```sh
python3 tools/gate_acceptance.py http://127.0.0.1:8000 deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --prefix /free --overlay-model deepseek-v4.1-flash-1017-free --effort-check \
    --status http://127.0.0.1:8000
```

The test passes in both configurations below and takes under four minutes.

## Tested configurations

Free memory is the available memory reported by each of the three nodes.

| | Lean | Large |
|---|---|---|
| KV cache per node (`KV`) | 500m | 3g |
| Context window (`WINDOW`) | 262,144 | 1,048,576 |
| Time from launch to ready | 2 min 16 s | 2 min 43 s |
| Free at ready | 7.1 / 8.8 / 7.9 GiB | 3.4 / 5.2 / 4.2 GiB |
| Free after the first requests | not recorded | 1.7 / 3.4 / 2.4 GiB |
| Lowest during the tests | not recorded | 1.89 / 3.70 / 2.66 GiB |
| Free at the end of the tests | 5.7 / 7.5 / 6.5 GiB | 2.1 / 3.8 / 2.8 GiB |
| Drop during a switch | at most 0.07 GiB (six switches) | 0.03 GiB (the first two) |

The large configuration leaves little headroom on the tightest node. Long cached
conversations on both variants at once were not tested in either configuration.
