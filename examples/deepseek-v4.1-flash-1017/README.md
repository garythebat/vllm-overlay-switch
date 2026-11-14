<!--
SPDX-FileCopyrightText: 2026 garythebat
SPDX-License-Identifier: Apache-2.0
-->

# The run this was built on

Everything needed to repeat what the main README reports: DeepSeek V4.1 Flash 1017 on three
machines of the DGX Spark family, tensor parallel 3, answering as the base model and as the Keys overlay
from one loaded copy. It ran this way on 14 November 2026. Addresses below are
documentation addresses.

- [recipe.yaml](recipe.yaml): the recipe as it ran.
- [start-with-knobs.sh](start-with-knobs.sh): the start script, with three lines changed (it says which).

## The pieces

| Piece | Where it comes from | Exactly which | Licence |
|---|---|---|---|
| Launcher and image | [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) | checkout at `a4c172976a5f9f8669b356a0891bf972cae14fd4`; image `eugr/spark-vllm-b12x:nightly-20261010`, digest `sha256:d999354461ae004c754333fd50a69867816758c864687e41b9a224b748a2c104` | MIT |
| Engine inside the image | [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm), branch `dev/karmic-kraken` | `740e3776b4d645733fe4f8b90809fddea2f82b96` | Apache-2.0 |
| Kernels inside the image | [local-inference-lab/flashinfer](https://github.com/local-inference-lab/flashinfer) | `2f5210dd1566a8dfef754f68691b898b8d870244` | Apache-2.0 |
| Base model | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | revision `1bae0e1088993b9b82de08f59de5be5a34e42170`: 83 files, 48 shards, about 475 GiB | MIT |
| Chat template | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | `chat_template.jinja` at revision `2cba9e42aa026125f3ed06c6d98c1db82f7ca027` | MIT |
| Overlay side file | [drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated) | `wo_b_l10_35_1017.safetensors` at revision `de412d2249632903dace23399526e00a145aada5`: 1,091,589,280 bytes, sha256 `88324ceaa2c0754a2bdb4ef6bb388a088438e352358343e6c2c882009cdde864` | MIT, behind the author's Responsible Use Agreement |

Two notes on these.

The chat template. At the revision above the 1017 release carried no `chat_template.jinja`,
so the server was given the first release's. The two releases' `encoding/` folders were
byte-identical. The 1017 repository has since added a template of its own (its listing
showed one on 14 November 2026). We have not run with that one.

The overlay. Its repository is gated: you accept the author's Responsible Use Agreement on
the model page before a download is allowed, and its terms then apply to what you do with
it. Nothing from that repository is included here. Only the one file above is used. The
checkpoint the server loads stays the stock one: no index is rewritten and nothing is
grafted into the shards.

## Where the memory settings come from

The recipe is leaner than the one it derives from, because these machines have little to
spare with this model loaded. The settings came out of our own preparation for this
cluster in October 2026, and that preparation stood on two pieces of other people's work:

- Christopher Owen's [spark-ds41f](https://github.com/christopherowen/spark-ds41f)
  (Apache-2.0) publishes memory figures and launch settings for three-machine runs of this
  model. We planned against its figures, and two of our settings are also in its
  configuration as we read it on 10 October 2026: `MAX_JOBS` 2 and
  `--mm-processor-cache-gb 0`.
- [Issue 404 of spark-vllm-docker](https://github.com/eugr/spark-vllm-docker/issues/404)
  describes an engine that hangs after running out of memory during a kernel compile.
  The limits on compile jobs (`B12X_COMPILE_WORKERS`, `MAX_JOBS`, `NINJAFLAGS`,
  `MAKEFLAGS`, `FLASHINFER_NVCC_THREADS`) are there because of it.

The other choices (text only, a 262,144 window, 2,048 batched tokens, CUDA graph capture
up to 16) are ours. What they left free is in the table at the end.

## Hardware

Three machines of the DGX Spark family (two NVIDIA DGX Spark and one Gigabyte AI TOP ATOM), each
with 128 GB of unified memory, linked for RoCE, each with the model files and the side file
on its own disk at the same paths.

## Steps

1. Set up spark-vllm-docker for three machines as its own README describes, at the commit
   above, with the image above on each machine.
2. Put the base model, at the revision above, in the Hugging Face cache of every machine.
   We also had the whole first release in the cache; the recipe reads only its
   `chat_template.jinja`.
3. Accept the overlay's agreement and download the side file. Place it on every machine
   where the container will see it as
   `/root/.cache/huggingface/ablit-overlay/wo_b_l10_35_1017.safetensors`, and check its
   sha256 on each.
4. Copy this repository's `mod/` folder into the spark-vllm-docker checkout on the head
   machine as `mods/overlay-switch`. The launcher copies it into the container on every
   machine at each launch.
5. Copy `recipe.yaml` into the checkout's `recipes/` folder as
   `deepseek-v4.1-flash-1017-switch.yaml` and start it, with the launcher directly or with
   `start-with-knobs.sh`.
6. In the head machine's server log, look for
   `Injected ... OverlayWorkerExtension ... for extended collective_rpc calls` and
   `Loaded endpoint plugin overlay_switch`.

The server starts on the base weights. The first request for the overlay prepares and flips.

## What a client sees

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

`http://192.0.2.10:8000/free/v1` answers with the overlay whatever name is sent. The
`model` field of every answer names the first served name; the header says which side
answered.

## Check it yourself

On the head machine, or through a tunnel to its port 8000:

```sh
python3 tools/gate_acceptance.py http://127.0.0.1:8000 deepseek-ai/DeepSeek-V4.1-Flash-1017 \
    --prefix /free --overlay-model deepseek-v4.1-flash-1017-free --effort-check \
    --status http://127.0.0.1:8000
```

It printed PASS on both settings below, in under four minutes each time.

## The two settings we ran

Free memory is what each of the three machines reported as available.

| | Lean | Larger |
|---|---|---|
| KV cache per machine (`KV`) | 500m | 3g |
| Window (`WINDOW`) | 262,144 | 1,048,576 |
| Ready after the launch was accepted | 2 min 16 s | 2 min 43 s |
| Free at ready | 7.1 / 8.8 / 7.9 GiB | 3.4 / 5.2 / 4.2 GiB |
| Free after the first requests | not read on this launch | 1.7 / 3.4 / 2.4 GiB |
| Lowest during the tests | not read on this launch | 1.89 / 3.70 / 2.66 GiB |
| Free at the end of the tests | 5.7 / 7.5 / 6.5 GiB | 2.1 / 3.8 / 2.8 GiB |
| Dip during a flip | at most 0.07 GiB (six flips) | 0.03 GiB (the first two) |

The larger setting leaves little room on the tightest machine. Not measured on either: a
long cached conversation on each side at once.
