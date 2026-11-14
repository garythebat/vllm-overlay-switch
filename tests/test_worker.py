"""The worker half against stand-ins for the engine: real tensors, real files, no GPU.

Real here: this package's code, torch and its two 8-bit float types, the safetensors files.
Stand-ins: the attention module, the loader that leaves each rank its columns of a weight,
and the packing function. The stand-ins keep the memory behaviour of what they replace
(the packed values are the weight's own memory; the scale tables and the tiled copy are
new memory; one scale table is a strided view), because that behaviour is what a flip
relies on. What they cannot show is the engine itself: that is the test on hardware.

Geometry: three ranks. The file tensor is 64 x 160 (five groups of 32 columns); the model
pads to six groups, two per rank, so rank 2 holds one real group and one of padding.

    python3 -m unittest tests.test_worker -v        (skipped when torch is not installed)
"""

import os
import random
import sys
import tempfile
import types
import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "mod"))
import helpers  # noqa: E402

ROWS, GROUP, FILE_GROUPS, LOCAL_GROUPS, RANKS = 64, 32, 5, 2, 3
FILE_COLS, LOCAL = FILE_GROUPS * GROUP, LOCAL_GROUPS * GROUP
MODEL_LAYERS, SWITCHED = range(6), (2, 3, 4)
PADDING = 0xEE  # what the stand-in loader leaves in padding columns; a flip must never touch it
CURRENT = {"rank": 0}
SAVED_MODULES: dict = {}


def cut(whole: bytes, rows: int, cols: int, start: int, width: int) -> bytes:
    """Columns [start, start + width) of a row-major byte matrix, as far as the file has them."""
    return b"".join(whole[r * cols + start: r * cols + min(cols, start + width)] for r in range(rows))


if torch is not None:
    class FakeRowLinear(torch.nn.Module):
        """Stand-in for the engine's row-parallel layer holding FP8 values and block scales."""

        def __init__(self, tp_rank: int):
            super().__init__()
            self.tp_rank = tp_rank
            self.weight = torch.nn.Parameter(
                torch.zeros((ROWS, LOCAL), dtype=torch.uint8).view(torch.float8_e4m3fn), requires_grad=False)
            self.weight_scale_inv = torch.nn.Parameter(
                torch.zeros((ROWS // 32, LOCAL // 32), dtype=torch.uint8).view(torch.float8_e8m0fnu), requires_grad=False)
            for param in (self.weight, self.weight_scale_inv):
                param.allow_tp_padding = True

        def load(self, param, whole: bytes, rows: int, cols: int) -> None:
            """What the engine's loader leaves on this rank: its columns, then padding."""
            local = int(param.shape[1])
            start = self.tp_rank * local
            have = max(0, min(local, cols - start))
            live = param.data.view(torch.uint8)
            live.fill_(PADDING)
            live[:, :have] = torch.frombuffer(bytearray(cut(whole, rows, cols, start, have)), dtype=torch.uint8).view(rows, have)

    class FakeAttention(torch.nn.Module):
        def __init__(self, layer_id: int, tp_rank: int, is_draft: bool = False):
            super().__init__()
            self.layer_id, self.is_draft = layer_id, is_draft
            self.n_local_groups, self.n_local_heads, self.head_dim = LOCAL_GROUPS, 2 * LOCAL_GROUPS, 16
            self.o_lora_rank, self.hidden_size = GROUP, ROWS
            self.wo_a = types.SimpleNamespace(weight=torch.zeros(1), weight_scale_inv=torch.zeros(1))
            self.wo_b = FakeRowLinear(tp_rank)
            self._wo_projection_weights = None

        def pack(self) -> None:
            self._wo_projection_weights = fake_pack_weights(
                self.wo_a.weight, self.wo_a.weight_scale_inv, self.wo_b.weight.detach(), self.wo_b.weight_scale_inv.detach(),
                groups=self.n_local_groups, group_width=(self.n_local_heads // self.n_local_groups) * self.head_dim,
                rank=self.o_lora_rank, hidden=self.hidden_size, block_size=(32, 32))

    def fake_pack_weights(wo_a_weight, wo_a_scale, wo_b_weight, wo_b_scale, *, groups, group_width, rank, hidden, block_size):
        assert block_size == (32, 32) and tuple(wo_b_weight.shape) == (hidden, groups * rank)
        values = wo_b_weight.contiguous()                                   # the weight's own memory
        scale = wo_b_scale.view(torch.uint8)
        scale_rows = scale.repeat_interleave(32, dim=0).unsqueeze(0).contiguous().view(torch.float8_e8m0fnu)
        physical = scale.t().contiguous()                                   # new memory ...
        scale_mma = physical.view(torch.float8_e8m0fnu).t()                 # ... seen through a strided view
        tiled = (values.view(torch.uint8).reshape(hidden // 32, 32, (groups * rank) // 32, 32)
                 .permute(0, 2, 1, 3).contiguous().view(torch.float8_e4m3fn))
        return types.SimpleNamespace(wo_a=None, wo_b=types.SimpleNamespace(
            values=values, scale_rows=scale_rows, scale_mma=scale_mma, values_tiled=tiled))


def setUpModule():
    if torch is None:
        return

    def module(name: str, **attrs):
        SAVED_MODULES[name] = sys.modules.get(name)
        made = types.ModuleType(name)
        made.__dict__.update(attrs)
        sys.modules[name] = made
        return made

    module("vllm")
    module("vllm.distributed")
    module("vllm.distributed.parallel_state", get_tensor_model_parallel_rank=lambda: CURRENT["rank"])
    module("vllm.models")
    module("vllm.models.deepseek_v4_1")
    module("vllm.models.deepseek_v4_1.nvidia")
    module("vllm.models.deepseek_v4_1.nvidia.b12x_attention", DeepseekV41B12xAttention=FakeAttention)
    module("b12x")
    module("b12x.gemm", wo_projection=types.SimpleNamespace(pack_weights=fake_pack_weights))


def tearDownModule():
    for name, old in SAVED_MODULES.items():
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old


@unittest.skipIf(torch is None, "torch is not installed")
class WorkerCase(unittest.TestCase):
    def setUp(self):
        from vllm_overlay_switch import worker as worker_module

        self.w = worker_module
        self.dir = tempfile.TemporaryDirectory()
        rng = random.Random(1017)
        scale_shape = (ROWS // 32, FILE_COLS // 32)
        self.data = {v: {(layer, "weight"): rng.randbytes(ROWS * FILE_COLS) for layer in MODEL_LAYERS}
                     | {(layer, "scale"): rng.randbytes(scale_shape[0] * scale_shape[1]) for layer in MODEL_LAYERS}
                     for v in ("base", "overlay")}
        self.shape = {"weight": (ROWS, FILE_COLS), "scale": scale_shape}
        self.dtype = {"weight": "F8_E4M3", "scale": "F8_E8M0"}
        self.served = os.path.join(self.dir.name, "served")
        os.mkdir(self.served)
        index = {"embed.weight": "model-00001.safetensors"}
        for number, layers in enumerate(((0, 1, 2), (3, 4, 5)), start=1):
            name = f"model-{number:05d}.safetensors"
            helpers.write_safetensors(os.path.join(self.served, name), {
                f"layers.{layer}.attn.wo_b.{kind}": (self.dtype[kind], self.shape[kind], self.data["base"][(layer, kind)])
                for layer in layers for kind in ("weight", "scale")}, metadata={"format": "pt"})
            index.update({f"layers.{layer}.attn.wo_b.{kind}": name for layer in layers for kind in ("weight", "scale")})
        self.index_path = os.path.join(self.served, "model.safetensors.index.json")
        self.write_index(index)
        self.index = index
        self.side = os.path.join(self.dir.name, "side.safetensors")
        self.write_side(SWITCHED)
        self.old_env = {k: os.environ.get(k) for k in (self.w.SIDE_ENV, self.w.BASE_ENV)}
        os.environ[self.w.SIDE_ENV] = self.side
        os.environ.pop(self.w.BASE_ENV, None)

    def tearDown(self):
        for key, value in self.old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.dir.cleanup()

    def write_index(self, index: dict) -> None:
        import json

        with open(self.index_path, "w") as f:
            json.dump({"metadata": {}, "weight_map": index}, f)

    def write_side(self, layers, extra: dict | None = None, source: str = "overlay") -> None:
        tensors = {f"layers.{layer}.attn.wo_b.{kind}": (self.dtype[kind], self.shape[kind], self.data[source][(layer, kind)])
                   for layer in layers for kind in ("weight", "scale")}
        tensors.update(extra or {})
        helpers.write_safetensors(self.side, tensors)

    def worker(self, rank: int):
        """One rank of a model that was loaded from the served (base) checkpoint."""
        test = self

        class FakeWorker(self.w.OverlayWorkerExtension):
            def __init__(self):
                self.rank, self.device = rank, torch.device("cpu")
                self.model_config = types.SimpleNamespace(model=test.served, revision=None)
                self.layers = torch.nn.ModuleList(FakeAttention(layer, rank) for layer in MODEL_LAYERS)
                self.layers.append(FakeAttention(SWITCHED[0], rank, is_draft=True))  # a draft twin: must be ignored
                for attn in self.layers:
                    for kind, name in (("weight", "weight"), ("scale", "weight_scale_inv")):
                        rows, cols = test.shape[kind]
                        attn.wo_b.load(getattr(attn.wo_b, name), test.data["base"][(attn.layer_id, kind)], rows, cols)
                    attn.pack()

            def get_model(self):
                return self.layers

        CURRENT["rank"] = rank
        return FakeWorker()

    @staticmethod
    def raw(tensor) -> bytes:
        return tensor.view(torch.uint8).contiguous().numpy().tobytes()

    def live(self, worker, layer: int, kind: str) -> bytes:
        name = "weight" if kind == "weight" else "weight_scale_inv"
        return self.raw(worker.layers[layer].wo_b.get_parameter(name).data)

    def expected(self, variant: str, layer: int, kind: str, rank: int) -> bytes:
        """What a rank must hold: its columns of the file tensor, then untouched padding."""
        rows, cols = self.shape[kind]
        local = LOCAL if kind == "weight" else LOCAL // 32
        have = max(0, min(local, cols - rank * local))
        got = cut(self.data[variant][(layer, kind)], rows, cols, rank * local, have)
        return b"".join(got[r * have:(r + 1) * have] + bytes([PADDING]) * (local - have) for r in range(rows))

    def derived(self, worker, layer: int) -> dict:
        packed = worker.layers[layer]._wo_projection_weights.wo_b
        return {f: getattr(packed, f) for f in ("values", "scale_rows", "scale_mma", "values_tiled")}


class Prepare(WorkerCase):
    def test_every_rank_proves_its_cut_against_the_live_weights(self):
        for rank in range(RANKS):
            worker = self.worker(rank)
            self.assertEqual(worker.overlay_status()["prepared"], False)
            out = worker.overlay_prepare()
            self.assertEqual((out["ok"], out["prepared"], out["active"], out["rank"]), (True, True, "base", rank), out)
            self.assertEqual((out["layers"], out["checked"], out["match"]), (3, 6, {"base": 6, "overlay": 0}))
            self.assertEqual(out["file"], "side.safetensors")

    def test_every_rank_reports_the_same_digest_of_the_whole_files_and_a_changed_file_shows(self):
        first = [self.worker(rank).overlay_prepare()["sets"] for rank in range(RANKS)]
        self.assertEqual(first[0], first[1])
        self.assertEqual(first[1], first[2])
        self.assertEqual(sorted(first[0]), ["base", "overlay"])
        self.assertNotEqual(first[0]["base"], first[0]["overlay"])
        # one byte differs in the side file, in columns that rank 0 does not hold
        key = (SWITCHED[1], "weight")
        changed = bytearray(self.data["overlay"][key])
        changed[FILE_COLS - 1] ^= 0xFF
        self.data["overlay"][key] = bytes(changed)
        self.write_side(SWITCHED)
        other = self.worker(0).overlay_prepare()
        self.assertEqual((other["ok"], other["active"]), (True, "base"))      # rank 0 alone would see nothing wrong
        self.assertEqual(other["sets"]["base"], first[0]["base"])
        self.assertNotEqual(other["sets"]["overlay"], first[0]["overlay"])

    def test_the_cut_matches_what_the_loader_left_including_the_padded_rank(self):
        worker = self.worker(2)
        worker.overlay_prepare()
        state = worker._overlay_switch_state
        self.assertEqual(state["cut"][(2, "weight")], (2 * LOCAL, FILE_COLS - 2 * LOCAL))   # (128, 32) of 64
        self.assertEqual(state["cut"][(2, "scale")], (4, 1))

    def test_a_rank_whose_weights_are_not_the_served_ones_is_not_prepared_and_will_not_write(self):
        worker = self.worker(1)
        worker.layers[3].wo_b.weight.data.view(torch.uint8).fill_(7)
        out = worker.overlay_prepare()
        self.assertEqual((out["ok"], out["prepared"], out["active"]), (False, False, "mixed"), out)
        self.assertIn("unproven", out["error"])
        before = self.live(worker, 2, "weight")
        self.assertEqual(worker.overlay_switch("overlay")["error"], "not prepared")
        self.assertEqual(self.live(worker, 2, "weight"), before)

    def test_refusals(self):
        cases = {
            "an unknown tensor in the side file": lambda: self.write_side(SWITCHED, {"layers.2.ffn.w1.weight": ("F8_E4M3", (1, 1), b"x")}),
            "a weight without its scale": lambda: helpers.write_safetensors(self.side, {
                "layers.2.attn.wo_b.weight": ("F8_E4M3", self.shape["weight"], self.data["overlay"][(2, "weight")])}),
            "the wrong type in the side file": lambda: helpers.write_safetensors(self.side, {
                "layers.2.attn.wo_b.weight": ("BF16", self.shape["weight"], self.data["overlay"][(2, "weight")]),
                "layers.2.attn.wo_b.scale": ("F8_E8M0", self.shape["scale"], self.data["overlay"][(2, "scale")])}),
            "a layer the model does not have": lambda: self.write_side((2, 9)),
            "a served checkpoint that already points at the side file": lambda: self.point_index_at_side(),
            "two sets that do not differ": lambda: self.write_side(SWITCHED, source="base"),
            "no side file named": lambda: os.environ.pop(self.w.SIDE_ENV),
        }
        for name, arrange in cases.items():
            with self.subTest(name):
                self.write_index(self.index)
                self.write_side(SWITCHED)
                os.environ[self.w.SIDE_ENV] = self.side
                self.data["overlay"].setdefault((9, "weight"), self.data["overlay"][(2, "weight")])
                self.data["overlay"].setdefault((9, "scale"), self.data["overlay"][(2, "scale")])
                arrange()
                worker = self.worker(0)
                out = worker.overlay_prepare()
                self.assertEqual((out["ok"], out["prepared"]), (False, False), out)
                self.assertTrue(out["error"])

    def point_index_at_side(self) -> None:
        link = os.path.join(self.served, "side-link.safetensors")
        os.symlink(self.side, link)
        self.write_index(self.index | {"layers.2.attn.wo_b.weight": "side-link.safetensors"})


class Flip(WorkerCase):
    def test_a_flip_and_back_on_every_rank(self):
        for rank in range(RANKS):
            with self.subTest(rank=rank):
                worker = self.worker(rank)
                worker.overlay_prepare()
                at_start = {(layer, kind): self.live(worker, layer, kind) for layer in MODEL_LAYERS for kind in ("weight", "scale")}
                tables_at_start = {layer: {f: self.raw(t) for f, t in self.derived(worker, layer).items()}
                                   for layer in MODEL_LAYERS}
                objects = {layer: self.derived(worker, layer) for layer in MODEL_LAYERS}
                addresses = {layer: {f: t.data_ptr() for f, t in objects[layer].items()} for layer in MODEL_LAYERS}
                weight_address = {layer: worker.layers[layer].wo_b.weight.data_ptr() for layer in MODEL_LAYERS}

                out = worker.overlay_switch("overlay")
                self.assertEqual((out["ok"], out["active"], out["match"]), (True, "overlay", {"base": 0, "overlay": 6}), out)
                self.assertEqual(worker.overlay_status()["active"], "overlay")
                self.assertEqual(worker.overlay_status(True)["checked"], 6)
                for layer in MODEL_LAYERS:
                    want = "overlay" if layer in SWITCHED else "base"
                    for kind in ("weight", "scale"):
                        self.assertEqual(self.live(worker, layer, kind), self.expected(want, layer, kind, rank), (layer, kind))
                    # the same objects at the same addresses, holding what a fresh packing of the new weights gives
                    now, kept = self.derived(worker, layer), worker.layers[layer]._wo_projection_weights
                    worker.layers[layer].pack()
                    fresh = self.derived(worker, layer)
                    worker.layers[layer]._wo_projection_weights = kept
                    for field, tensor in now.items():
                        self.assertIs(tensor, objects[layer][field])
                        self.assertEqual(tensor.data_ptr(), addresses[layer][field])
                        self.assertTrue(torch.equal(tensor.view(torch.uint8), fresh[field].view(torch.uint8)), (layer, field))
                    self.assertEqual(worker.layers[layer].wo_b.weight.data_ptr(), weight_address[layer])
                    self.assertEqual(now["values"].data_ptr(), weight_address[layer])
                # the draft twin of a switched layer is left alone
                self.assertEqual(self.live(worker, len(MODEL_LAYERS), "weight"), self.expected("base", SWITCHED[0], "weight", rank))

                out = worker.overlay_switch("base")
                self.assertEqual((out["ok"], out["active"], out["match"]), (True, "base", {"base": 6, "overlay": 0}), out)
                for key, was in at_start.items():
                    self.assertEqual(self.live(worker, *key), was, key)
                for layer in MODEL_LAYERS:
                    for field, tensor in self.derived(worker, layer).items():
                        self.assertEqual(self.raw(tensor), tables_at_start[layer][field], (layer, field))

    def test_a_write_that_stopped_halfway_reads_as_mixed(self):
        worker = self.worker(0)
        worker.overlay_prepare()
        worker.overlay_switch("overlay")
        last = worker.layers[SWITCHED[-1]].wo_b
        last.load(last.weight, self.data["base"][(SWITCHED[-1], "weight")], *self.shape["weight"])
        out = worker.overlay_status()
        self.assertEqual((out["ok"], out["active"], out["prepared"]), (False, "mixed", True), out)
        # and a forced write of either set repairs it, because the cut was proven before
        self.assertEqual(worker.overlay_switch("base")["active"], "base")

    def test_only_the_full_look_sees_a_middle_layer(self):
        worker = self.worker(0)
        worker.overlay_prepare()
        middle = worker.layers[SWITCHED[1]].wo_b
        middle.load(middle.weight, self.data["overlay"][(SWITCHED[1], "weight")], *self.shape["weight"])
        self.assertEqual(worker.overlay_status()["active"], "base")
        out = worker.overlay_status(True)
        self.assertEqual((out["ok"], out["active"], out["match"]), (False, "mixed", {"base": 5, "overlay": 1}), out)

    def test_a_table_that_changed_shape_stops_the_write(self):
        worker = self.worker(0)
        worker.overlay_prepare()
        packed = worker.layers[SWITCHED[0]]._wo_projection_weights.wo_b
        packed.scale_rows = packed.scale_rows[:, :1, :]
        out = worker.overlay_switch("overlay")
        self.assertEqual(out["ok"], False)
        self.assertIn("changed shape", out["error"])

    def test_before_prepare_nothing_is_written_and_unknown_sets_are_refused(self):
        worker = self.worker(0)
        self.assertEqual(worker.overlay_switch("overlay")["error"], "not prepared")
        worker.overlay_prepare()
        out = worker.overlay_switch("other")
        self.assertEqual(out["ok"], False)
        self.assertIn("unknown set", out["error"])


class DerivedDigest(WorkerCase):
    """The exact form of "restored": what the model derived from the weights, as one digest per rank."""

    def test_it_can_be_read_before_anything_is_prepared_and_only_by_a_full_look(self):
        worker = self.worker(0)
        self.assertNotIn("derived", worker.overlay_status())
        fresh = worker.overlay_status(True)
        self.assertEqual((fresh["prepared"], fresh["active"]), (False, None))
        self.assertRegex(fresh["derived"], r"^[0-9a-f]{16}$")
        # preparing writes nothing, so the digest is the one the startup left; and neither a prepare
        # nor a flip computes it: a flip holds generation paused and should not take longer for it
        self.assertNotIn("derived", worker.overlay_prepare())
        self.assertNotIn("derived", worker.overlay_status())
        self.assertEqual(worker.overlay_status(True)["derived"], fresh["derived"])
        self.assertNotIn("derived", worker.overlay_switch("overlay"))

    def test_there_and_back_gives_the_digest_of_the_start_on_every_rank(self):
        seen = set()
        for rank in range(RANKS):
            with self.subTest(rank=rank):
                worker = self.worker(rank)
                start = worker.overlay_status(True)["derived"]
                worker.overlay_prepare()
                worker.overlay_switch("overlay")
                over = worker.overlay_status(True)["derived"]
                self.assertNotEqual(over, start)
                worker.overlay_switch("base")
                self.assertEqual(worker.overlay_status(True)["derived"], start)
                worker.overlay_switch("overlay")
                self.assertEqual(worker.overlay_status(True)["derived"], over)
                seen.add(start)
        self.assertEqual(len(seen), RANKS)        # each rank holds other columns: digests are per rank

    def test_a_table_left_behind_by_a_flip_would_show(self):
        worker = self.worker(0)
        worker.overlay_prepare()
        start = worker.overlay_status(True)["derived"]
        for field in ("scale_rows", "scale_mma", "values_tiled"):
            with self.subTest(field=field):
                tensor = getattr(worker.layers[SWITCHED[1]]._wo_projection_weights.wo_b, field)
                flat = tensor.view(torch.uint8)
                index = tuple(0 for _ in flat.shape)
                flat[index] ^= 1                           # one bit of one derived table, weights untouched
                self.assertEqual(worker.overlay_status(True)["active"], "base")
                self.assertNotEqual(worker.overlay_status(True)["derived"], start)
                flat[index] ^= 1
                self.assertEqual(worker.overlay_status(True)["derived"], start)

    def test_a_side_file_that_cannot_be_read_does_not_break_the_status(self):
        worker = self.worker(0)
        os.environ[self.w.SIDE_ENV] = os.path.join(self.dir.name, "missing.safetensors")
        out = worker.overlay_status(True)
        self.assertEqual((out["ok"], out["prepared"]), (True, False))
        self.assertNotIn("derived", out)
        self.assertIn("derived_error", out)


if __name__ == "__main__":
    unittest.main()
