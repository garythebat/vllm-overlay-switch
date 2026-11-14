"""Shared by the tests: load a module of the package by path, write a safetensors file by hand."""

import importlib.util
import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.join(HERE, "..", "mod", "vllm_overlay_switch")


def load(name: str, as_name: str):
    """Load mod/vllm_overlay_switch/<name>.py on its own, without importing the package."""
    spec = importlib.util.spec_from_file_location(as_name, os.path.join(PACKAGE_DIR, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[as_name] = module
    spec.loader.exec_module(module)
    return module


def write_safetensors(path: str, tensors: dict, metadata: dict | None = None) -> None:
    """tensors: name -> (dtype name, shape, bytes). Laid out as the format asks: an 8-byte
    little-endian header length, the JSON header padded to 8 bytes, then the data."""
    header, blob, offset = {}, bytearray(), 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        blob += data
        offset += len(data)
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(bytes(blob))
