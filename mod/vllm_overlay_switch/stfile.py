"""Read named tensors out of a safetensors file as raw bytes.

Deliberately small and independent of the safetensors package: the two dtypes this project
moves (F8_E4M3 values and F8_E8M0 block scales) are one byte per element, and some releases
of the Python reader do not map the second one. Bytes are bytes; the caller views them as
whatever dtype the destination has.

A tensor is read with plain file reads into a buffer of its own, and the pages are handed
back to the kernel straight away, so a flip on a machine with little free memory does not
fill the page cache with a gigabyte of weights.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass

MAX_HEADER = 100 << 20


@dataclass(frozen=True)
class Entry:
    path: str
    name: str
    dtype: str            # the header's name, e.g. "F8_E4M3"
    shape: tuple[int, ...]
    start: int            # absolute byte offsets in the file
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


def read_header(path: str) -> dict[str, Entry]:
    """Every tensor in the file: name -> Entry."""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: too short to be a safetensors file")
        (n,) = struct.unpack("<Q", raw)
        if not 2 <= n <= MAX_HEADER:
            raise ValueError(f"{path}: implausible header length {n}")
        header = json.loads(f.read(n))
    base = 8 + n
    size = os.path.getsize(path)
    out: dict[str, Entry] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        b, e = info["data_offsets"]
        if not 0 <= b <= e or base + e > size:
            raise ValueError(f"{path}: {name} has offsets outside the file")
        out[name] = Entry(path, name, str(info["dtype"]), tuple(int(x) for x in info["shape"]), base + b, base + e)
    return out


def read_bytes(entry: Entry) -> bytearray:
    """The tensor's bytes, read whole; the file's pages are released afterwards."""
    buf = bytearray(entry.nbytes)
    view = memoryview(buf)
    got = 0
    with open(entry.path, "rb", buffering=0) as f:
        f.seek(entry.start)
        while got < entry.nbytes:
            k = f.readinto(view[got:])
            if not k:
                raise EOFError(f"{entry.path}: {entry.name} ended {entry.nbytes - got} bytes early")
            got += k
        try:
            os.posix_fadvise(f.fileno(), entry.start, entry.nbytes, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
    return buf
