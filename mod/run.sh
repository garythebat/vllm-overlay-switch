#!/bin/bash
# Installs the overlay switch into a vLLM container. Runs inside the container, on every node,
# before the server starts (spark-vllm-docker: `--apply-mod <this folder>`, or `mods:` in a recipe).
#
# It edits no engine file. It copies one small package next to vLLM and registers its HTTP routes
# as an endpoint plugin. The server then needs three things in its launch, which this script
# cannot add for you and prints at the end:
#   --worker-extension-cls vllm_overlay_switch.worker.OverlayWorkerExtension
#   VLLM_PLUGINS=overlay_switch,...        (see the note it prints: the variable is an allow-list)
#   OVERLAY_SWITCH_FILE=/path/in/container/to/the/side/file.safetensors
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PKG=vllm_overlay_switch
VERSION=0.1.0
say() { echo "[overlay-switch] $*"; }
die() { echo "[overlay-switch] $*" >&2; exit 1; }

# where vLLM is installed: asked of Python without importing vLLM, unless PYTHON_ROOT says
if [ -z "${PYTHON_ROOT:-}" ]; then
    PYTHON_ROOT="$(python3 -c 'import importlib.util, os; s = importlib.util.find_spec("vllm"); print(os.path.dirname(os.path.dirname(s.origin)) if s and s.origin else "")' 2>/dev/null || true)"
fi
[ -n "$PYTHON_ROOT" ] && [ -d "$PYTHON_ROOT/vllm" ] || die "vLLM package not found (set PYTHON_ROOT to the folder that holds it)"
[ -d "$HERE/$PKG" ] || die "the package folder $PKG is missing next to run.sh"

# Fail closed on an engine this was not written for: every piece the switch stands on must be there.
need() { grep -q -- "$2" "$PYTHON_ROOT/$1" 2>/dev/null || die "not the engine this was written for: '$2' not found in $1"; }
need vllm/plugins/endpoint_plugins/interface.py "def attach_router"
need vllm/v1/worker/worker_base.py "worker_extension_cls"
need vllm/models/deepseek_v4_1/attention.py "def setup_wo_projection"
need vllm/models/deepseek_v4_1/attention.py "_wo_projection_weights"
need vllm/models/deepseek_v4_1/nvidia/b12x_attention.py "class DeepseekV41B12xAttention"
need vllm/model_executor/parameter.py "def copy_tensor_parallel_shard"
# The kernel library is imported as `b12x`, a thin alias package whose code lives inside
# flashinfer. Look in both places, without importing either.
python3 - <<'PY' || die "the b12x kernel library does not offer wo_projection.pack_weights"
import importlib.util, os, sys
roots = []
for name, below in (("flashinfer", ("experimental", "b12x")), ("b12x", ())):
    try:
        spec = importlib.util.find_spec(name)
    except Exception:
        spec = None
    if spec is not None and spec.origin:
        roots.append(os.path.join(os.path.dirname(spec.origin), *below))
for root in roots:
    api = os.path.join(root, "gemm", "wo_projection", "api.py")
    if os.path.exists(api) and "pack_weights" in open(api).read():
        sys.exit(0)
sys.exit(1)
PY

rm -rf "${PYTHON_ROOT:?}/$PKG" "${PYTHON_ROOT:?}/$PKG-"*.dist-info
cp -r "$HERE/$PKG" "$PYTHON_ROOT/$PKG"
rm -rf "${PYTHON_ROOT:?}/$PKG/__pycache__"
INFO="$PYTHON_ROOT/$PKG-$VERSION.dist-info"
mkdir -p "$INFO"
printf 'Metadata-Version: 2.1\nName: vllm-overlay-switch\nVersion: %s\nSummary: Flip a small set of weights in a running vLLM server, in place\n' "$VERSION" > "$INFO/METADATA"
printf '[vllm.endpoint_plugins]\noverlay_switch = vllm_overlay_switch.api:OverlaySwitchEndpointPlugin\n' > "$INFO/entry_points.txt"
printf 'overlay-switch mod\n' > "$INFO/INSTALLER"

python3 - <<'PY' || die "the package was copied but cannot be found as an endpoint plugin"
from importlib.metadata import entry_points
names = {e.name for e in entry_points(group="vllm.endpoint_plugins")}
raise SystemExit(0 if "overlay_switch" in names else 1)
PY

OTHERS=$(python3 - <<'PY'
from importlib.metadata import entry_points
names = []
for group in ("vllm.general_plugins", "vllm.endpoint_plugins", "vllm.io_processor_plugins", "vllm.stat_logger_plugins", "vllm.platform_plugins"):
    try:
        names += [e.name for e in entry_points(group=group)]
    except Exception:
        pass
print(",".join(dict.fromkeys(["overlay_switch", *names])))
PY
)
say "installed $PKG $VERSION into $PYTHON_ROOT"
say "the launch needs: --worker-extension-cls vllm_overlay_switch.worker.OverlayWorkerExtension"
say "the launch needs: OVERLAY_SWITCH_FILE=<the side file, as the container sees it>"
say "the launch needs: VLLM_PLUGINS=$OTHERS"
say "  (VLLM_PLUGINS is an allow-list for every vLLM plugin; the value above keeps what this image already loads)"
if [ -z "${OVERLAY_SWITCH_FILE:-}" ]; then say "note: OVERLAY_SWITCH_FILE is not set in this container yet"; fi
case ",${VLLM_PLUGINS:-}," in *,overlay_switch,*) ;; *) say "note: VLLM_PLUGINS does not name overlay_switch yet: the routes will not load";; esac
