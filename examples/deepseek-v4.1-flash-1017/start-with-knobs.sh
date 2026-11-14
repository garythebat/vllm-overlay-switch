#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 garythebat
# SPDX-License-Identifier: Apache-2.0
#
# Start the recipe of this folder through spark-vllm-docker's run-recipe.py, with four settings taken from
# the environment and passed to vLLM after the recipe's own arguments.
#   KV        --kv-cache-memory-bytes   500m | 1g | 2g | 3g        (default 500m: lean)
#   REQUESTS  --max-num-seqs            1..8                        (default 2)
#   IMAGES    --limit-mm-per-prompt     0 = text only, 1 = one image per prompt   (default 0)
#   WINDOW    --max-model-len           32768..1048576              (default 262144)
# Put this file in your spark-vllm-docker checkout, and recipe.yaml in its recipes/ folder under the name
# used on the last line. DRY=1 prints the launcher's dry run (the final vLLM command) and starts nothing.
#
# This is the script the runs of 14 November 2026 were started with, except for three lines changed
# here: the default cache folder, the recipe's path, and the name it prints. It has not been run in this
# form.
set -eu
cd "$(dirname "$0")"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
KV=${KV:-500m}; REQUESTS=${REQUESTS:-2}; IMAGES=${IMAGES:-0}; WINDOW=${WINDOW:-262144}
case "$KV" in 500m|1g|2g|3g) ;; *) echo "KV must be 500m, 1g, 2g or 3g (got $KV)"; exit 2;; esac
[[ "$REQUESTS" =~ ^[1-8]$ ]] || { echo "REQUESTS must be 1-8 (got $REQUESTS)"; exit 2; }
[[ "$IMAGES" =~ ^[01]$ ]] || { echo "IMAGES must be 0 or 1 (got $IMAGES)"; exit 2; }
[[ "$WINDOW" =~ ^[0-9]{5,7}$ ]] && [ "$WINDOW" -ge 32768 ] && [ "$WINDOW" -le 1048576 ] || { echo "WINDOW must be 32768-1048576 (got $WINDOW)"; exit 2; }
echo "start-with-knobs: KV=$KV REQUESTS=$REQUESTS IMAGES=$IMAGES WINDOW=$WINDOW ($(date '+%F %T'))"
exec python3 run-recipe.py --daemon ${DRY:+--dry-run} recipes/deepseek-v4.1-flash-1017-switch.yaml -- \
  --kv-cache-memory-bytes "$KV" --max-num-seqs "$REQUESTS" --max-model-len "$WINDOW" \
  --limit-mm-per-prompt "{\"image\":$IMAGES}"
