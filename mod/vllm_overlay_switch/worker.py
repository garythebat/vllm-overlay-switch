"""The worker half of the switch.

`OverlayWorkerExtension` is given to the engine with `--worker-extension-cls
vllm_overlay_switch.worker.OverlayWorkerExtension`. The engine mixes it into its worker class
on every rank, which makes the three methods below callable on all ranks at once.

What a flip does on one rank, for every layer named in the side file:

1. read the tensor's bytes from disk (the side file for "overlay", the served checkpoint for
   "base") and keep the columns this rank holds;
2. copy them over the Parameter's existing memory. The packed values the kernel reads are
   that same memory;
3. rebuild what the model derived from them at startup (two scale tables, and for some
   shapes a second, tiled copy of the values that the decode kernel reads) with the function
   the model packed them with, and copy the result over the existing tensors.

Nothing is replaced, only overwritten: captured CUDA graphs and prefetch tables hold the
addresses of these buffers. Nothing is kept in memory between flips either; the price is that
a flip reads the tensors from disk, about a gigabyte per rank for the case this was built for.

Two things are checked, not assumed:

- Before the first flip, every rank reads both sets and compares the live weights with them,
  tensor by tensor. If this tool's idea of which columns a rank holds were wrong for the
  engine in front of it, the live weights would match neither set and it refuses to write.
- After a flip, every tensor is read back from the device and compared again.
- Every rank reports a digest of the whole of both sets as it read them from its own disk. The
  HTTP half refuses to flip unless all ranks report the same two digests.

Which set is active is read from the live weights, never remembered (the one exception: a
rank on whose columns the two sets do not differ reports the set it wrote last).

All copies are done on the bytes (uint8 views), so nothing here depends on which operations
the installed torch offers for the 8-bit float types.

Environment:
    OVERLAY_SWITCH_FILE       the side file (safetensors) with the alternate tensors. Required.
    OVERLAY_SWITCH_BASE_DIR   the served checkpoint folder, if the model was given by name
                              and cannot be resolved from the local hub cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time

from . import stfile

SIDE_ENV = "OVERLAY_SWITCH_FILE"
BASE_ENV = "OVERLAY_SWITCH_BASE_DIR"
VARIANTS = ("base", "overlay")
NAME_RE = re.compile(r"^layers\.(\d+)\.attn\.wo_b\.(weight|scale)$")
KINDS = ("weight", "scale")
PARAM_OF = {"weight": "weight", "scale": "weight_scale_inv"}
HEADER_DTYPE = {"weight": "F8_E4M3", "scale": "F8_E8M0"}
MODEL_DTYPE = {"weight": "torch.float8_e4m3fn", "scale": "torch.float8_e8m0fnu"}


class OverlayWorkerExtension:
    """Mixed into the engine's worker. Only these three names are added to it."""

    def overlay_status(self, full: bool = False) -> dict:
        """Which set is live. Looks at the first and last layer; `full` looks at every tensor."""
        state = getattr(self, "_overlay_switch_state", None)
        out = _answer(self, prepared=state is not None)
        if state is None:
            return out
        try:
            out.update(_report(self, state, full=bool(full)))
        except Exception as e:
            out.update(ok=False, error=_short(e))
        return out

    def overlay_prepare(self) -> dict:
        """Find both sets, check them against the live model, and prove the column cut.

        A rank counts as prepared only once its live weights were found to be exactly one of
        the two sets. Until then it will not write, whatever it is asked.
        """
        t0 = time.monotonic()
        proven = getattr(self, "_overlay_switch_state", None) is not None
        try:
            state = _prepare(self)
            report = _report(self, state, full=True)
        except Exception as e:
            return _answer(self, prepared=proven, ok=False, error=_short(e))
        if report["ok"]:
            # the live weights are exactly one of the two sets as this rank cuts them
            state["last"] = report["active"]
            self._overlay_switch_state = state
            proven = True
        out = _answer(self, prepared=proven)
        out.update(report)
        out["seconds"] = round(time.monotonic() - t0, 3)
        if not proven:
            out["error"] = f"{report['error']}: the column cut is unproven, so nothing will be written"
        return out

    def overlay_switch(self, variant: str) -> dict:
        """Write one set over the live buffers and read every tensor back."""
        t0 = time.monotonic()
        state = getattr(self, "_overlay_switch_state", None)
        if state is None:
            return _answer(self, prepared=False, ok=False, error="not prepared")
        out = _answer(self, prepared=True)
        if variant not in VARIANTS:
            out.update(ok=False, error=f"unknown set {variant!r}")
            return out
        try:
            written = _write(self, state, variant)
            report = _report(self, state, full=True, wrote=variant)
            ok = report["active"] == variant
            if ok:
                state["last"] = variant
            out.update(report)
            out.update(ok=ok, bytes=written, seconds=round(time.monotonic() - t0, 3),
                       error=None if ok else f"after the copy the live weights read as {report['active']!r}")
            return out
        except Exception as e:
            try:
                out.update(_report(self, state, full=True))
            except Exception:
                pass
            out.update(ok=False, error=_short(e))
            return out


def _answer(worker, *, prepared: bool, ok: bool = True, error: str | None = None) -> dict:
    return {"ok": ok, "rank": _rank(worker), "prepared": prepared, "active": None,
            "file": os.path.basename(os.environ.get(SIDE_ENV, "")) or None, "error": error}


def _short(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:400]


def _rank(worker) -> int:
    try:
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())
    except Exception:
        return int(getattr(worker, "rank", -1))


def _attention_by_layer(worker) -> dict:
    from vllm.models.deepseek_v4_1.nvidia.b12x_attention import DeepseekV41B12xAttention

    found = {}
    for module in worker.get_model().modules():
        if isinstance(module, DeepseekV41B12xAttention) and not getattr(module, "is_draft", False):
            layer = int(module.layer_id)
            if layer in found:
                raise RuntimeError(f"two attention modules claim layer {layer}")
            found[layer] = module
    return found


def _base_dir(worker) -> str:
    explicit = os.environ.get(BASE_ENV, "")
    if explicit:
        return explicit
    model_config = getattr(worker, "model_config", None) or worker.vllm_config.model_config
    model = model_config.model
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model, revision=getattr(model_config, "revision", None), local_files_only=True)


def _prepare(worker) -> dict:
    """Find the tensors of both sets, check them against the live model, fingerprint both."""
    side_path = os.environ.get(SIDE_ENV, "")
    if not side_path:
        raise RuntimeError(f"{SIDE_ENV} is not set")
    overlay: dict[tuple[int, str], stfile.Entry] = {}
    for name, entry in stfile.read_header(side_path).items():
        m = NAME_RE.match(name)
        if m is None:
            raise ValueError(f"{name}: this version switches layers.N.attn.wo_b weight and scale only")
        kind = m.group(2)
        if entry.dtype != HEADER_DTYPE[kind]:
            raise ValueError(f"{name}: dtype {entry.dtype}, expected {HEADER_DTYPE[kind]}")
        overlay[(int(m.group(1)), kind)] = entry
    layers = sorted({layer for layer, _ in overlay})
    if not layers:
        raise ValueError("the side file holds no tensors")
    keys = [(layer, kind) for layer in layers for kind in KINDS]
    for layer, kind in keys:
        if (layer, kind) not in overlay:
            raise ValueError(f"layer {layer}: the side file has no wo_b {kind}")

    base_dir = _base_dir(worker)
    with open(os.path.join(base_dir, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    headers: dict[str, dict[str, stfile.Entry]] = {}
    base: dict[tuple[int, str], stfile.Entry] = {}
    side_real = os.path.realpath(side_path)
    for key in keys:
        entry = overlay[key]
        file_name = weight_map.get(entry.name)
        if file_name is None:
            raise ValueError(f"{entry.name} is not in the served checkpoint's index")
        path = os.path.join(base_dir, file_name)
        if os.path.realpath(path) == side_real:
            raise ValueError("the served checkpoint already points these tensors at the side file: "
                             "serve the base checkpoint, the switch brings the overlay")
        if path not in headers:
            headers[path] = stfile.read_header(path)
        twin = headers[path].get(entry.name)
        if twin is None:
            raise ValueError(f"{entry.name} is not in {file_name}")
        if (twin.dtype, twin.shape) != (entry.dtype, entry.shape):
            raise ValueError(f"{entry.name}: the two sets differ in dtype or shape")
        base[key] = twin

    # which columns of each file tensor this rank holds: the engine cuts a row-parallel
    # weight along its input dimension, `local` columns per rank, starting at rank * local;
    # when the padded width does not divide the file's, the last rank's tail is padding
    attention = _attention_by_layer(worker)
    cut: dict[tuple[int, str], tuple[int, int]] = {}
    for layer, kind in keys:
        attn = attention.get(layer)
        if attn is None:
            raise ValueError(f"layer {layer} has no native attention module on this rank")
        if getattr(attn, "_wo_projection_weights", None) is None:
            raise ValueError(f"layer {layer}: the output projection is not packed yet")
        param = getattr(attn.wo_b, PARAM_OF[kind], None)
        entry = overlay[(layer, kind)]
        if param is None or param.dim() != 2 or len(entry.shape) != 2:
            raise ValueError(f"{entry.name}: expected a two-dimensional tensor in the file and in the model")
        if str(param.dtype) != MODEL_DTYPE[kind]:
            raise ValueError(f"{entry.name}: the model holds {param.dtype}, expected {MODEL_DTYPE[kind]}")
        rows, cols = entry.shape
        if entry.nbytes != rows * cols:
            raise ValueError(f"{entry.name}: {entry.nbytes} bytes for shape {entry.shape}")
        if rows != int(param.shape[0]):
            raise ValueError(f"{entry.name}: {rows} rows in the file, {int(param.shape[0])} in the model")
        local = int(param.shape[1])
        start = int(attn.wo_b.tp_rank) * local
        have = max(0, min(local, cols - start))
        if have == 0:
            raise ValueError(f"{entry.name}: this rank holds none of its columns")
        if have < local and not getattr(param, "allow_tp_padding", False):
            raise ValueError(f"{entry.name}: the file has too few columns for this rank and padding is off")
        cut[(layer, kind)] = (start, have)

    # Read both sets once. From each tensor: a fingerprint of this rank's columns (to tell which
    # set is live), and its whole bytes into one digest per set, so that the ranks can show each
    # other they read the same files. A rank sees only its own columns of the live model; without
    # the whole-file digest, a side file that differed on one machine would go unnoticed.
    sources = {"base": base, "overlay": overlay}
    prints, ids = {}, {}
    for v in VARIANTS:
        whole, prints[v] = hashlib.sha256(), {}
        for key in keys:
            entry = sources[v][key]
            raw = stfile.read_bytes(entry)
            whole.update(f"{entry.name}\0{entry.dtype}\0{entry.shape}\0".encode())
            whole.update(raw)
            prints[v][key] = _print(_columns(raw, entry, *cut[key]))
            del raw
        ids[v] = whole.hexdigest()[:16]
    if all(prints["base"][key] == prints["overlay"][key] for key in keys):
        raise ValueError("the two sets do not differ on this rank's columns: is the served checkpoint the base one?")
    ends = sorted({layers[0], layers[-1]})
    return {"layers": layers, "keys": keys, "probe_keys": [(layer, kind) for layer in ends for kind in KINDS],
            "sources": sources, "cut": cut, "prints": prints, "ids": ids, "side": side_path, "last": "base"}


def _columns(raw: bytearray, entry: stfile.Entry, start: int, have: int):
    """This rank's columns out of a file tensor's bytes, as contiguous bytes on the CPU."""
    import torch

    whole = torch.frombuffer(raw, dtype=torch.uint8).view(entry.shape)
    return whole[:, start:start + have].contiguous()


def _cut(entry: stfile.Entry, start: int, have: int):
    """Read a file tensor and keep this rank's columns."""
    return _columns(stfile.read_bytes(entry), entry, start, have)


def _live(attn, kind: str, have: int):
    """The same columns of the live tensor, as a byte view on its device."""
    import torch

    data = getattr(attn.wo_b, PARAM_OF[kind]).data.view(torch.uint8)
    return data if have == int(data.shape[1]) else data[:, :have]


def _print(tensor) -> str:
    return hashlib.sha256(memoryview(tensor.contiguous().cpu().numpy())).hexdigest()


def _report(worker, state: dict, *, full: bool, wrote: str | None = None) -> dict:
    """Compare live tensors with both sets. Probes are the first and last layer: a write goes
    through the layers in order, so one that stopped halfway leaves those two disagreeing."""
    attention = _attention_by_layer(worker)
    keys = state["keys"] if full else state["probe_keys"]
    match = {v: 0 for v in VARIANTS}
    for key in keys:
        seen = _print(_live(attention[key[0]], key[1], state["cut"][key][1]))
        for v in VARIANTS:
            match[v] += state["prints"][v][key] == seen
    whole = [v for v in VARIANTS if match[v] == len(keys)]
    if len(whole) == 1:
        active = whole[0]
    elif whole:  # the two sets do not differ on what was looked at on this rank
        active = wrote or state["last"]
    else:
        active = "mixed"
    ok = active in VARIANTS
    return {"ok": ok, "active": active, "layers": len(state["layers"]), "checked": len(keys), "match": match,
            "sets": dict(state["ids"]),
            "error": None if ok else f"the live weights match neither set (base {match['base']}, "
                                     f"overlay {match['overlay']} of {len(keys)} tensors)"}


def _write(worker, state: dict, variant: str) -> int:
    """Overwrite the live buffers of every layer with one set. Returns the bytes written."""
    import torch

    attention = _attention_by_layer(worker)
    on_gpu = getattr(getattr(worker, "device", None), "type", None) == "cuda"
    written = 0
    # inference mode: the packed tensors were made at load time and may be inference tensors,
    # which can only be written in place from inside it
    with torch.inference_mode():
        for layer in state["layers"]:
            attn = attention[layer]
            for kind in KINDS:
                key = (layer, kind)
                start, have = state["cut"][key]
                source = _cut(state["sources"][variant][key], start, have)
                _live(attn, kind, have).copy_(source)
                written += int(source.numel())
                del source
            _refresh_derived(attn)
        if on_gpu:
            torch.cuda.synchronize(worker.device)
    if on_gpu:
        # the repacking above used a few tens of megabytes of scratch; give it back to the device
        torch.cuda.empty_cache()
    return written


def _refresh_derived(attn) -> None:
    """Rebuild what the model derived from wo_b at startup and copy it over the existing tensors.

    Mirrors `setup_wo_projection` of the model. The packed values are normally the Parameter's
    own memory and need no copy; the scale tables and the tiled copy of the values do.
    """
    import torch
    from b12x.gemm import wo_projection

    groups = attn.n_local_groups
    heads_per_group = attn.n_local_heads // groups
    fresh = wo_projection.pack_weights(
        attn.wo_a.weight.detach(),
        attn.wo_a.weight_scale_inv.detach(),
        attn.wo_b.weight.detach(),
        attn.wo_b.weight_scale_inv.detach(),
        groups=groups,
        group_width=heads_per_group * attn.head_dim,
        rank=attn.o_lora_rank,
        hidden=attn.hidden_size,
        block_size=(32, 32),
    ).wo_b
    live = attn._wo_projection_weights.wo_b
    for field in ("values", "scale_rows", "scale_mma", "values_tiled"):
        old, new = getattr(live, field), getattr(fresh, field)
        if old is None and new is None:
            continue
        if old is None or new is None or old.shape != new.shape or old.dtype != new.dtype:
            raise RuntimeError(f"layer {attn.layer_id}: the packed {field} changed shape, refusing to write")
        if old.data_ptr() == new.data_ptr():
            continue
        old_bytes, new_bytes = old.view(torch.uint8), new.view(torch.uint8)
        old_bytes.copy_(new_bytes)
        if not torch.equal(old_bytes, new_bytes):
            raise RuntimeError(f"layer {attn.layer_id}: the packed {field} did not take the copy")
