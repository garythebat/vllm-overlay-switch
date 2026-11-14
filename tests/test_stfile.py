"""Reading named tensors out of a safetensors file as bytes. No torch, no GPU.

    python3 -m unittest tests.test_stfile -v
"""

import os
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers  # noqa: E402

stfile = helpers.load("stfile", "overlay_stfile")


class ReadHeader(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "side.safetensors")
        self.a = bytes(range(12))
        self.b = bytes(reversed(range(6)))
        helpers.write_safetensors(self.path, {
            "layers.3.attn.wo_b.weight": ("F8_E4M3", (3, 4), self.a),
            "layers.3.attn.wo_b.scale": ("F8_E8M0", (2, 3), self.b),
        }, metadata={"format": "pt"})

    def tearDown(self):
        self.dir.cleanup()

    def test_names_types_shapes_and_bytes_come_back(self):
        entries = stfile.read_header(self.path)
        self.assertEqual(sorted(entries), ["layers.3.attn.wo_b.scale", "layers.3.attn.wo_b.weight"])
        weight, scale = entries["layers.3.attn.wo_b.weight"], entries["layers.3.attn.wo_b.scale"]
        self.assertEqual((weight.dtype, weight.shape, weight.nbytes), ("F8_E4M3", (3, 4), 12))
        self.assertEqual((scale.dtype, scale.shape, scale.nbytes), ("F8_E8M0", (2, 3), 6))
        self.assertEqual(bytes(stfile.read_bytes(weight)), self.a)
        self.assertEqual(bytes(stfile.read_bytes(scale)), self.b)

    def test_offsets_are_absolute_in_the_file(self):
        weight = stfile.read_header(self.path)["layers.3.attn.wo_b.weight"]
        with open(self.path, "rb") as f:
            f.seek(weight.start)
            self.assertEqual(f.read(weight.nbytes), self.a)

    def test_the_metadata_block_is_not_a_tensor(self):
        self.assertNotIn("__metadata__", stfile.read_header(self.path))

    def test_what_is_read_can_be_written_to(self):
        # the worker hands these bytes to torch.frombuffer, which wants a writable buffer
        got = stfile.read_bytes(stfile.read_header(self.path)["layers.3.attn.wo_b.weight"])
        self.assertIsInstance(got, bytearray)


class Refusals(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "bad.safetensors")

    def tearDown(self):
        self.dir.cleanup()

    def write(self, raw: bytes):
        with open(self.path, "wb") as f:
            f.write(raw)

    def test_a_file_too_short_for_a_header(self):
        self.write(b"\x01\x02")
        with self.assertRaises(ValueError):
            stfile.read_header(self.path)

    def test_an_implausible_header_length(self):
        self.write(struct.pack("<Q", 1 << 40) + b"{}")
        with self.assertRaises(ValueError):
            stfile.read_header(self.path)

    def test_offsets_that_run_past_the_end_of_the_file(self):
        helpers.write_safetensors(self.path, {"t": ("F8_E4M3", (2, 2), b"abcd")})
        with open(self.path, "r+b") as f:
            f.truncate(os.path.getsize(self.path) - 1)
        with self.assertRaises(ValueError):
            stfile.read_header(self.path)

    def test_a_file_that_shrinks_after_its_header_was_read(self):
        helpers.write_safetensors(self.path, {"t": ("F8_E4M3", (2, 2), b"abcd")})
        entry = stfile.read_header(self.path)["t"]
        with open(self.path, "r+b") as f:
            f.truncate(entry.start + 2)
        with self.assertRaises(EOFError):
            stfile.read_bytes(entry)


if __name__ == "__main__":
    unittest.main()
