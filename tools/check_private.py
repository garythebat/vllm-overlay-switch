#!/usr/bin/env python3
"""Privacy gate: scan what git would commit for secrets and private details. Exit 1 on any hit.

usage: python3 tools/check_private.py                    the files git would commit, their names, and the repo rules
       python3 tools/check_private.py PATH ...           only these files or folders (contents, names, media rules)
       python3 tools/check_private.py --message FILE     a commit message (also refuses en and em dashes)
       python3 tools/check_private.py --history [RANGE]  every commit in RANGE (default: --all): author, committer,
                                                         time zone, message, and every file the commit added or
                                                         changed (contents and names)
       python3 tools/check_private.py --tag OBJECT       an annotated tag (and any tag it points to): tagger, time
                                                         zone, name and message, by the same rules as a commit
       python3 tools/check_private.py --self-test        prove the detectors on built-in samples (exit 1 if one fails)

Prints the place and the KIND of each hit, never the matched text, so its output is safe to paste anywhere.

What is scanned: every path in `git ls-files --cached --others --exclude-standard`. For a path in the index the
STAGED content is scanned (what the next commit would hold), and the working-tree file too when it differs (what
`git add -A` would stage); an untracked file is scanned as it is on disk. Outside a git checkout, every file under
the root except .git/ and private/. Symlinks are never followed. This file is scanned like any other: its patterns
and test samples are written so that they do not match themselves (the samples are joined at run time).

Two layers of patterns:
  1. generic patterns kept in this file: tokens and keys, private keys, private, CGNAT and public IPv4 addresses,
     IPv6 addresses (IPv4 inside RoCE GIDs and link-local addresses built from a MAC included), user@address,
     e-mail addresses, MAC addresses, home-folder paths. Secrets with a known shape are also caught by gitleaks
     (tools/check-private.sh);
  2. a private deny-list kept OUTSIDE the repository, found through the environment variable PRIVATE_DENYLIST,
     else `git config privacy.denylist` (a path relative to the repository root). One entry per line:
       name = regex                       a named private pattern (or a bare regex)
       allow <path-or-glob>:<line>[ kind] a reviewed false positive of a private pattern at that place
       allow-text <literal>               a reviewed literal, removed from every line before the private patterns
       allow-commit <sha> <kind> [<glob>] a reviewed hit in an existing commit: in its message or identity, or
                                          with a path glob, in a file it added or changed. Printed as a NOTE on
                                          every run until the commit is rewritten
     Host names, user names, domains, places, the owner's own tools and anything else that must never appear here
     belong in that file. Allow lines that match nothing are reported as WARNING, so stale exceptions do not pile up.

Repository rules (default run only): no tracked symlinks, nothing under private/, no deny-lists, *.local.md or
.env files, nothing tracked that .gitignore ignores, and no file over 50 MiB.

Media rules (default run and PATH runs): images, audio, video, SVG and PDF files are read as their containers, and
any metadata that can name a person, a place, a machine or a time of capture is refused: EXIF, XMP, IPTC, GPS,
PNG text chunks (tEXt, zTXt, iTXt; compressed text is unpacked and scanned too) other than 'Software', JPEG
comments, GIF comments, TIFF author and camera tags, ID3 tags, WAV INFO, FLAC and Ogg comments, MP4 and QuickTime
user data and location, Matroska person and place tags, SVG editor metadata, and a PDF /Author or /Creator. Strip
metadata with: exiftool -all= -overwrite_original FILE

History and tag rules (--history, --tag): author, committer and tagger e-mail must be a GitHub no-reply address,
time zones must be +0000, and no message line may be a link trailer ("Key: https://...") unless its key is listed
in `git config --get-all privacy.allowTrailer` (the same rule as tools/hooks/commit-msg).

Allowed on purpose: loopback and documentation addresses (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24,
2001:db8::/32), the placeholders this repository uses (<head-ip> and friends), the needle test's fixed passphrase,
and GitHub no-reply addresses.
"""
import fnmatch
import ipaddress
import os
import re
import struct
import subprocess
import sys
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_BYTES = 50 * 1024 * 1024

GENERIC = {
    "hf-token": r"\bhf_[A-Za-z0-9]{20,}",
    "github-token": r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}",
    "openai-style-key": r"\bsk-[A-Za-z0-9_-]{20,}",
    "xai-key": r"\bxai-[A-Za-z0-9]{20,}",
    "google-api-key": r"AIza[0-9A-Za-z_-]{30,}",
    "aws-key": r"\bAKIA[0-9A-Z]{16}\b",
    "slack-token": r"\bxox[abpors]-[A-Za-z0-9-]{10,}",
    "bot-api-token": r"\b\d{8,10}:[A-Za-z0-9_-]{35}\b",
    "bearer-value": r"Bearer\s+(?![\$<\{\"'])[A-Za-z0-9._~+/=-]{12,}",
    "jwt": r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.",
    "private-key-block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "ssh-public-key": r"\bssh-(?:rsa|ed25519|ecdsa)\s+AAAA",
    "password-assignment": r"(?i)\b(?:password|passwd|secret|api[_-]?key|token)\b\s*[:=]\s*['\"]?"
                           r"(?![\$<\{\s'\"]|$|none\b|null\b|false\b|true\b|unset\b|\*)[^\s'\",)]{6,}",
    # the kind name is split in two so that this line does not match its own pattern
    "ssh" "pass-or-sudo-S": r"\bsshpass\b|\bsudo\s+-S\b",
    "ipv4-private": r"(?<![\d.])(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}(?![\d])",
    "cgnat-ip": r"(?<![\d.])100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}(?![\d])",
    # any other dotted quad that routes on the internet (checked in IS_HIT: loopback, private, CGNAT, link-local,
    # documentation, multicast and reserved addresses and netmasks do not count)
    "ipv4-public": r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w]|\.\d)",
    # IPv6 in any form (checked in IS_HIT: must parse as an address; ::1 and 2001:db8::/32 do not count)
    "ipv6": r"(?<![\w:.])(?:[0-9A-Fa-f]{0,4}:){2,7}(?:\d{1,3}(?:\.\d{1,3}){3}|[0-9A-Fa-f]{0,4})(?![\w:])",
    # an IPv4 address written in hex at the end of a RoCE GID (::ffff:c0a8:xxxx is 192.168.x.x)
    "gid-ipv4-hex": r"(?i)(?<![0-9a-f])ffff:(?:c0a8|0a[0-9a-f]{2}|ac1[0-9a-f]|64[4-7][0-9a-f]):[0-9a-f]{1,4}(?![0-9a-f])",
    # a link-local IPv6 address or GID made from a MAC address (EUI-64: ...xxff:fexx...)
    "eui64": r"(?i)\bfe80:[0-9a-f:]*?[0-9a-f]{1,2}ff:fe[0-9a-f]{2}",
    "user-at-address": r"\b[a-z_][a-z0-9_-]*@(?:\d{1,3}\.){3}\d{1,3}\b",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "mac-address": r"(?<![:\w])(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}(?![:\w])",
    "mac-dash": r"(?<![\w-])(?:[0-9A-Fa-f]{2}-){5}[0-9A-Fa-f]{2}(?![\w-])",
    "home-path": r"/home/(?!user\b)[a-z_][a-z0-9_-]*|/Users/[A-Za-z]|/r[o]ot/(?!\.cache)",
}
ALLOW = [
    r"violet-harbor-7291",                      # the needle's fixed test passphrase
    r"git@github\.com",                         # clone URLs
    r"\b[\w.+-]+@users\.noreply\.github\.com",  # GitHub no-reply author addresses
]
ALLOW_RX = [re.compile(a) for a in ALLOW]
DOC4 = [ipaddress.ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")]
DOC6 = ipaddress.ip_network("2001:db8::/32")


def _ipv4_public(s):
    try:
        a = ipaddress.IPv4Address(s)
    except ValueError:
        return False
    return a.is_global and not a.is_multicast and not a.is_reserved and not any(a in n for n in DOC4)


def _ipv6(s):
    try:
        a = ipaddress.IPv6Address(s)
    except ValueError:
        return False
    if a.is_loopback or a.is_unspecified or a in DOC6:
        return False
    groups = [g for g in re.split(r":+", s) if g]
    # "a::b" or "1::2" is more often code (slices, scopes) than an address: ask for a 3-digit group or 3 groups
    return len(groups) >= 3 or any(len(g) >= 3 for g in groups)


IS_HIT = {"ipv4-public": _ipv4_public, "ipv6": _ipv6}
DASH = re.compile("[\u2013\u2014]")
MEDIA_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".heic", ".heif", ".avif", ".tif", ".tiff", ".wav", ".mp3",
             ".flac", ".ogg", ".opus", ".m4a", ".mp4", ".mov", ".webm", ".mkv", ".svg", ".pdf")
# a last resort for any media file, whatever its container
MEDIA_MARKS = (b"Exif\x00\x00", b"<x:xmpmeta", b"GPSLatitude", b"com.apple.quicktime.location", b"\xa9xyz")
PNG_TEXT_OK = {"Software"}
PRIVATE_FILES = ("denylist*", "*.local.md", ".env*")
TRAILER = re.compile(r"^([A-Za-z-]+): https?://")


def git(*args, check=True):
    p = subprocess.run(["git", "-C", ROOT, *args], capture_output=True, check=False)
    if check and p.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed")
    return p


def in_git():
    try:
        return git("rev-parse", "--is-inside-work-tree", check=False).stdout.strip() == b"true"
    except OSError:
        return False


def denylist_path():
    path = os.environ.get("PRIVATE_DENYLIST")
    if not path and in_git():
        path = git("config", "--get", "privacy.denylist", check=False).stdout.decode().strip()
        if path and not os.path.isabs(path):
            path = os.path.join(ROOT, path)
    return path or None


class Deny:
    """Private patterns and reviewed exceptions from the deny-list."""

    def __init__(self, path):
        self.path, self.rx, self.allow, self.text, self.commits = path, {}, [], [], []
        if not path:
            return
        for i, line in enumerate(open(path, encoding="utf-8"), 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line.startswith("allow-text "):
                self.text.append([line[len("allow-text "):], 0])
                continue
            if line.startswith("allow-commit "):
                f = line[len("allow-commit "):].split()
                if len(f) in (2, 3):
                    self.commits.append([f[0].lower(), f[1], f[2] if len(f) == 3 else None, 0, i])
                    continue
            if line.startswith("allow "):
                spec = line[len("allow "):].strip()
                kind = None
                if " " in spec:
                    spec, kind = spec.split(" ", 1)
                where, _, num = spec.rpartition(":")
                self.allow.append([where, num, kind, 0, i])
                continue
            m = re.match(r"^([a-z0-9-]+) = (.+)$", line)
            name = f"private:{m.group(1)}" if m else f"private:line-{i}"
            self.rx[name] = re.compile(m.group(2) if m else line)

    def strip(self, line):
        for t in self.text:
            if t[0] in line:
                t[1] += 1
                line = line.replace(t[0], "")
        return line

    def allowed(self, rel, n, kind):
        for a in self.allow:
            if fnmatch.fnmatchcase(rel, a[0]) and a[1] == str(n) and a[2] in (None, kind, kind[len("private:"):]):
                a[3] += 1
                return True
        return False

    def commit_allowed(self, full, kind, path=None):
        for c in self.commits:
            if len(c[0]) < 7 or not full.lower().startswith(c[0]) or c[1] not in (kind, kind[len("private:"):]):
                continue
            if (c[2] is None and path is None) or (c[2] and path and fnmatch.fnmatchcase(path, c[2])):
                c[3] += 1
                return True
        return False

    def warnings(self, history=False):
        out = [f"WARNING: deny-list line {a[4]}: an 'allow' line matched nothing" for a in self.allow if not a[3]]
        out += ["WARNING: an 'allow-text' line matched nothing" for t in self.text if not t[1]]
        if history:
            out += [f"WARNING: deny-list line {c[4]}: an 'allow-commit' line matched nothing" for c in self.commits
                    if not c[3]]
        return out


def scan_text(lines, deny, place, extra=None, generic=True):
    """Yield (place, kind, line number) for every pattern hit in these lines."""
    gen = {k: re.compile(v) for k, v in {**(GENERIC if generic else {}), **(extra or {})}.items()}
    for n, line in enumerate(lines, 1):
        clean = line
        for a in ALLOW_RX:
            clean = a.sub("", clean)
        for kind, r in gen.items():
            check = IS_HIT.get(kind)
            if any(check is None or check(m.group()) for m in r.finditer(clean)):
                yield place(n), kind, n
        priv = deny.strip(clean)
        for kind, r in deny.rx.items():
            if r.search(priv):
                yield place(n), kind, n


# ---------------------------------------------------------------- media containers

def _u32be(b, i):
    return struct.unpack(">I", b[i:i + 4])[0] if i + 4 <= len(b) else 0


def _inflate(b, limit=MAX_BYTES):
    try:
        return zlib.decompressobj().decompress(b, limit)
    except zlib.error:
        return None


def png_meta(data):
    """(problems, texts): metadata chunks, and the text of every text chunk (compressed ones unpacked)."""
    probs, texts, i = [], [], 8
    while i + 8 <= len(data):
        n, typ = struct.unpack(">I4s", data[i:i + 8])
        body = data[i + 8:i + 8 + n]
        i += 12 + n
        if typ == b"eXIf":
            probs.append("PNG eXIf (EXIF)")
        elif typ in (b"tEXt", b"zTXt", b"iTXt"):
            key, _, rest = body.partition(b"\0")
            text = None
            if typ == b"tEXt":
                text = rest
            elif typ == b"zTXt":
                text = _inflate(rest[1:])
            elif len(rest) >= 2:
                lang, _, rest2 = rest[2:].partition(b"\0")
                tkey, _, val = rest2.partition(b"\0")
                val = _inflate(val) if rest[0] else val
                text = None if val is None else lang + b" " + tkey + b" " + val
            if text is None:
                probs.append(f"PNG {typ.decode()} (unreadable)")
            else:
                texts.append(key.decode("latin-1") + ": " + text.decode("utf-8", "replace"))
            if key.decode("latin-1") not in PNG_TEXT_OK:
                probs.append(f"PNG {typ.decode()}")
        elif typ == b"IEND":
            break
    return probs, texts


def jpeg_meta(data):
    probs, texts, i = [], [], 2
    while i + 4 <= len(data) and data[i] == 0xFF:
        m = data[i + 1]
        if m == 0xFF:
            i += 1
            continue
        if m in (0x01, 0xD8) or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if m in (0xD9, 0xDA):           # end, or start of the image data: metadata segments come before it
            break
        n = struct.unpack(">H", data[i + 2:i + 4])[0]
        seg = data[i + 4:i + 2 + n]
        if m == 0xE1:
            probs.append("JPEG APP1 (EXIF or XMP)")
        elif m == 0xED:
            probs.append("JPEG APP13 (IPTC)")
        elif m == 0xFE:
            probs.append("JPEG comment")
            texts.append(seg.decode("utf-8", "replace"))
        i += 2 + n
    return probs, texts


def riff_chunks(data, i):
    while i + 8 <= len(data):
        cid, n = data[i:i + 4], struct.unpack("<I", data[i + 4:i + 8])[0]
        yield cid, data[i + 8:i + 8 + n]
        i += 8 + n + (n & 1)


def riff_meta(data):
    probs, texts = [], []
    kind = data[8:12]
    for cid, body in riff_chunks(data, 12):
        if kind == b"WEBP" and cid in (b"EXIF", b"XMP "):
            probs.append(f"WebP {cid.decode().strip()}")
        elif kind == b"WAVE":
            if cid == b"LIST" and body[:4] == b"INFO":
                probs.append("WAV LIST INFO")
                texts.append(body[4:].decode("latin-1"))
            elif cid in (b"id3 ", b"ID3 ", b"bext", b"iXML", b"_PMX"):
                probs.append(f"WAV {cid.decode().strip()}")
    return probs, texts


def gif_meta(data):
    probs, texts = [], []
    if len(data) < 13:
        return probs, texts

    def table(flags):
        return 3 * (2 ** ((flags & 7) + 1)) if flags & 0x80 else 0

    def sub_blocks(i):
        out = b""
        while i < len(data):
            n = data[i]
            i += 1
            if n == 0:
                break
            out += data[i:i + n]
            i += n
        return i, out
    i = 13 + table(data[10])
    while i < len(data):
        b = data[i]
        if b == 0x21 and i + 1 < len(data):
            label = data[i + 1]
            if label == 0xFF and data[i + 3:i + 11] == b"XMP Data":
                probs.append("GIF XMP")
            i, body = sub_blocks(i + 2)
            if label == 0xFE:
                probs.append("GIF comment")
                texts.append(body.decode("latin-1"))
        elif b == 0x2C and i + 10 <= len(data):
            i += 10 + table(data[i + 9]) + 1
            i, _ = sub_blocks(i)
        else:
            break
    return probs, texts


TIFF_TAGS = {0x010E: "ImageDescription", 0x010F: "Make", 0x0110: "Model", 0x0132: "DateTime", 0x013B: "Artist",
             0x013C: "HostComputer", 0x02BC: "XMP", 0x8298: "copyright", 0x83BB: "IPTC", 0x8769: "EXIF",
             0x8825: "GPS", 0x9C9B: "XPTitle", 0x9C9C: "XPComment", 0x9C9D: "XPAuthor"}


def tiff_meta(data):
    probs = []
    e = {b"II": "<", b"MM": ">"}.get(data[:2])
    if not e or len(data) < 8:
        return probs, []
    off, seen = struct.unpack(e + "I", data[4:8])[0], set()
    while off and off not in seen and off + 2 <= len(data):
        seen.add(off)
        n = struct.unpack(e + "H", data[off:off + 2])[0]
        for k in range(n):
            p = off + 2 + 12 * k
            if p + 12 > len(data):
                break
            tag = struct.unpack(e + "H", data[p:p + 2])[0]
            if tag in TIFF_TAGS:
                probs.append(f"TIFF {TIFF_TAGS[tag]}")
        p = off + 2 + 12 * n
        off = struct.unpack(e + "I", data[p:p + 4])[0] if p + 4 <= len(data) else 0
    return probs, []


def mp3_meta(data):
    probs = []
    if data[:3] == b"ID3":
        probs.append("MP3 ID3v2 tag")
    if len(data) >= 128 and data[-128:-125] == b"TAG":
        probs.append("MP3 ID3v1 tag")
    return probs, []


def _comments_after(data, at):
    """The number of user comments in a Vorbis comment header whose vendor length starts at `at`."""
    if at + 4 > len(data):
        return 0
    vlen = struct.unpack("<I", data[at:at + 4])[0]
    p = at + 4 + vlen
    return struct.unpack("<I", data[p:p + 4])[0] if p + 4 <= len(data) else 0


def flac_meta(data):
    probs, i = [], 4
    while i + 4 <= len(data):
        head, n = data[i], int.from_bytes(data[i + 1:i + 4], "big")
        typ = head & 0x7F
        if typ == 4 and _comments_after(data, i + 4):
            probs.append("FLAC comments")
        elif typ == 6:
            probs.append("FLAC picture")
        i += 4 + n
        if head & 0x80:
            break
    return probs, []


def ogg_meta(data):
    probs = []
    for mark in (b"\x03vorbis", b"OpusTags"):
        at = data.find(mark)
        if at >= 0 and _comments_after(data, at + len(mark)):
            probs.append("Ogg comments")
    return probs, []


BMFF_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"udta", b"meta", b"ilst", b"iinf", b"iprp",
                   b"ipco", b"edts", b"dinf", b"moof", b"traf", b"mvex"}
BMFF_FLAG = {b"\xa9xyz": "location", b"loci": "location", b"\xa9ART": "artist", b"\xa9aut": "author",
             b"auth": "author", b"\xa9cmt": "comment", b"\xa9wrt": "writer", b"cprt": "copyright",
             b"\xa9mak": "camera make", b"\xa9mod": "camera model", b"\xa9day": "date"}
XMP_UUID = bytes.fromhex("be7acfcb97a942e89c71999491e3afac")
QT_KEYS = (b"com.apple.quicktime.location", b"com.apple.quicktime.author", b"com.apple.quicktime.make",
           b"com.apple.quicktime.model", b"com.apple.quicktime.artist", b"com.apple.quicktime.comment")


def bmff_meta(data):
    """MP4, QuickTime, HEIC and AVIF: walk the boxes for user data, location, EXIF and XMP items."""
    probs = []

    def walk(i, end, depth):
        while i + 8 <= end and depth < 10:
            size, typ = struct.unpack(">I4s", data[i:i + 8])
            hdr = 8
            if size == 1 and i + 16 <= end:
                size, hdr = struct.unpack(">Q", data[i + 8:i + 16])[0], 16
            elif size == 0:
                size = end - i
            if size < hdr or i + size > end:
                return
            body, stop = i + hdr, i + size
            if typ in BMFF_FLAG:
                probs.append(f"MP4 {BMFF_FLAG[typ]}")
            elif typ == b"uuid" and data[body:body + 16] == XMP_UUID:
                probs.append("MP4 XMP")
            elif typ == b"keys":
                probs.extend("MP4 QuickTime " + k.decode().rsplit(".", 1)[-1] for k in QT_KEYS
                             if k in data[body:stop])
            elif typ == b"infe" and data[body] >= 2:
                idlen = 2 if data[body] == 2 else 4
                item = data[body + 4 + idlen + 2:body + 4 + idlen + 6]
                if item == b"Exif":
                    probs.append("HEIF EXIF item")
                elif item == b"mime" and b"rdf+xml" in data[body:stop]:
                    probs.append("HEIF XMP item")
            if typ in BMFF_CONTAINERS:
                skip = 0
                if typ == b"meta" and data[body + 4:body + 8] != b"hdlr":
                    skip = 4                                      # ISO meta is a full box; QuickTime's is not
                elif typ == b"iinf":
                    skip = 6 if data[body] == 0 else 8            # version, flags and the entry count
                walk(body + skip, stop, depth + 1)
            i = stop
    walk(0, len(data), 0)
    return probs, []


MKV_TAGS = re.compile(rb"\x45\xa3[\x80-\xff](ARTIST|AUTHOR|COMPOSER|COMMENT|COPYRIGHT|DATE_RECORDED|DIRECTOR|"
                      rb"LOCATION|RECORDING_LOCATION|COMPOSITION_LOCATION|PRODUCER|ACTOR|LYRICIST)")


def mkv_meta(data):
    return (["Matroska person or place tag"] if MKV_TAGS.search(data) else []), []


SVG_MARKS = (b"<metadata", b"sodipodi:", b"inkscape:", b"dc:creator", b"<rdf:")


def svg_meta(data):
    return [f"SVG {m.decode().strip('<:')}" for m in SVG_MARKS if m in data], []


def pdf_meta(data):
    probs, texts = [], [data]
    for m in re.finditer(rb"stream\r?\n", data):         # unpack compressed object streams (where /Info can hide)
        head = data[max(0, m.start() - 400):m.start()]
        if b"/FlateDecode" in head and b"/ObjStm" in head[head.rfind(b"obj"):]:
            stop = data.find(b"endstream", m.end())
            raw = _inflate(data[m.end():stop if stop > 0 else len(data)])
            if raw:
                texts.append(raw)
    for t in texts:
        for key in (b"Author", b"Creator"):
            for v in re.finditer(rb"/" + key + rb"\s*(\((?:\\.|[^\\)])*\)|<[0-9A-Fa-f\s]*>)", t):
                inner = re.sub(rb"[\s()<>]|\\376\\377|\\xfe\\xff|^feff", b"", v.group(1), flags=re.I)
                if inner.strip(b"\0"):
                    probs.append(f"PDF /{key.decode()}")
        if b"/Metadata" in t or b"<x:xmpmeta" in t:
            probs.append("PDF XMP")
    return sorted(set(probs)), []


def media_meta(rel, data):
    """(problems, texts) for one media file; texts are decoded metadata strings to scan like file content."""
    ext = os.path.splitext(rel.lower())[1]
    probs, texts = [], []
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        probs, texts = png_meta(data)
    elif data[:2] == b"\xff\xd8":
        probs, texts = jpeg_meta(data)
    elif data[:4] == b"RIFF":
        probs, texts = riff_meta(data)
    elif data[:4] in (b"GIF8",):
        probs, texts = gif_meta(data)
    elif data[:4] in (b"II*\x00", b"MM\x00*"):
        probs, texts = tiff_meta(data)
    elif data[:4] == b"fLaC":
        probs, texts = flac_meta(data)
    elif data[:4] == b"OggS":
        probs, texts = ogg_meta(data)
    elif data[:4] == b"\x1a\x45\xdf\xa3":
        probs, texts = mkv_meta(data)
    elif data[4:8] == b"ftyp" or ext in (".mov", ".mp4", ".m4a", ".heic", ".heif", ".avif"):
        probs, texts = bmff_meta(data)
    elif data[:4] == b"%PDF":
        probs, texts = pdf_meta(data)
        texts = []                      # the PDF's text is scanned as content already
    elif ext == ".svg":
        probs, texts = svg_meta(data)
    if ext == ".mp3" or data[:3] == b"ID3":
        probs += mp3_meta(data)[0]
    if ext not in (".svg", ".pdf") and any(m in data for m in MEDIA_MARKS) and not probs:
        probs.append("EXIF, XMP or location marker")
    return probs, [t if isinstance(t, str) else t.decode("latin-1") for t in texts]


def media_rules(rel, versions, deny):
    """[(place, kind)] for metadata in a media file, and for private text inside its (unpacked) metadata."""
    if not rel.lower().endswith(MEDIA_EXT):
        return []
    hits = []
    for data in versions:
        probs, texts = media_meta(rel, data)
        hits += [(rel, f"media-metadata ({p})") for p in probs]
        for t in texts:
            hits += [(f"{rel} (metadata text)", k) for _, k, _ in scan_text(t.splitlines(), deny, lambda n: n)]
    return hits


# ---------------------------------------------------------------- files and the index

def candidates(paths):
    """Repo-relative paths to scan (never following symlinks)."""
    if paths:
        out = []
        for p in paths:
            p = os.path.abspath(p)
            if os.path.isdir(p) and not os.path.islink(p):
                for root, dirs, fs in os.walk(p):
                    dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
                    out += [os.path.join(root, f) for f in sorted(fs)]
            else:
                out.append(p)
        return [os.path.relpath(p, ROOT) for p in out]
    if in_git():
        z = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").stdout.decode("utf-8", "replace")
        return sorted({p for p in z.split("\0") if p})
    out = []
    for root, dirs, fs in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in (".git", "private", "__pycache__"))
        out += [os.path.relpath(os.path.join(root, f), ROOT) for f in sorted(fs)]
    return out


def staged():
    """{path: (mode, blob sha)} for every index entry."""
    out = {}
    if not in_git():
        return out
    for rec in git("ls-files", "-s", "-z").stdout.decode("utf-8", "replace").split("\0"):
        if rec:
            meta, path = rec.split("\t", 1)
            mode, sha, _ = meta.split(" ")
            out[path] = (mode, sha)
    return out


def blobs(shas):
    """{blob sha: bytes} through one `git cat-file --batch`."""
    shas = sorted(set(shas))
    if not shas:
        return {}
    p = subprocess.run(["git", "-C", ROOT, "cat-file", "--batch"], input="".join(s + "\n" for s in shas).encode(),
                       capture_output=True, check=True)
    out, data, i = {}, p.stdout, 0
    while i < len(data):
        nl = data.index(b"\n", i)
        head = data[i:nl].split()
        i = nl + 1
        if len(head) < 3 or head[1] == b"missing":
            continue
        size = int(head[2])
        out[head[0].decode()] = data[i:i + size]
        i += size + 1
    return out


def work_bytes(rel):
    full = os.path.join(ROOT, rel)
    if os.path.islink(full) or not os.path.isfile(full):
        return None
    with open(full, "rb") as f:
        return f.read()


def repo_rules(rels, index, content):
    hits = []
    if in_git():
        for path, (mode, _) in index.items():
            if mode == "120000":
                hits.append((path, "tracked-symlink"))
        for p in git("ls-files", "-z", "-ci", "--exclude-standard").stdout.decode("utf-8", "replace").split("\0"):
            if p:
                hits.append((p, "tracked-but-ignored"))
    for rel in rels:
        full = os.path.join(ROOT, rel)
        if os.path.islink(full):
            hits.append((rel, "tracked-symlink"))
        if rel == "private" or rel.startswith("private/"):
            hits.append((rel, "private-path"))
        if any(fnmatch.fnmatchcase(os.path.basename(rel), g) for g in PRIVATE_FILES):
            hits.append((rel, "private-file"))
        if any(len(data) > MAX_BYTES for data in content.get(rel, [])):
            hits.append((rel, "over-50MiB"))
    return sorted(set(hits))


def contents(rels, index):
    """{path: [bytes, ...]}: the staged blob (if any), then the working-tree file when it differs."""
    mine = {r: index[r][1] for r in rels if r in index and index[r][0] != "120000"}
    got = blobs(mine.values())
    out = {}
    for rel in rels:
        versions = []
        if rel in mine and mine[rel] in got:
            versions.append(("staged", got[mine[rel]]))
        work = work_bytes(rel)
        if work is not None and all(work != v for _, v in versions):
            versions.append(("", work))
        out[rel] = versions
    return out


def scan_blob(rel, data, deny, where):
    """Hits in one file's content; `where(n)` names the place."""
    if rel.endswith(".pyc"):
        return []
    lines = data.decode("utf-8", errors="replace").splitlines()
    found = []
    for place, kind, n in scan_text(lines, deny, where):
        if kind.startswith("private:") and deny.allowed(rel, n, kind):
            continue
        found.append((place, kind))
    return found


def scan_files(rels, deny, versions):
    hits = 0
    for rel in rels:
        name_place = f"{os.path.dirname(rel) or '.'}/ (name)"
        for _, kind, _ in scan_text([rel], deny, lambda n: name_place):
            print(f"{name_place}: {kind}")
            hits += 1
        seen = set()
        for label, data in versions.get(rel, []):
            tag = f" ({label})" if label else ""
            for place, kind in scan_blob(rel, data, deny, lambda n: f"{rel}:{n}{tag}"):
                key = (place.replace(tag, "") if tag else place, kind)
                if key in seen:
                    continue
                seen.add(key)
                print(f"{place}: {kind}")
                hits += 1
    return hits


def message_lines(path):
    lines = []
    for line in open(path, encoding="utf-8", errors="replace").read().splitlines():
        if line.startswith("# ------------------------ >8 ------------------------"):
            break
        if not line.startswith("#"):
            lines.append(line)
    return lines


# ---------------------------------------------------------------- history and tags

def history_files(revs):
    """[(commit short, path, blob sha)] for every file each commit added or changed (no renames, no deletions)."""
    out = []
    for full in revs:
        p = git("diff-tree", "-r", "-z", "--no-renames", "--root", "--no-commit-id", full, check=False)
        parts = p.stdout.decode("utf-8", "replace").split("\0")
        i = 0
        while i + 1 < len(parts):
            meta, path = parts[i], parts[i + 1]
            i += 2
            f = meta.lstrip(":").split(" ")
            if len(f) < 5 or f[4][:1] == "D" or f[1] == "160000":
                continue
            out.append((full[:7], path, f[3]))
    return out


def allowed_trailers():
    return set(git("config", "--get-all", "privacy.allowTrailer", check=False).stdout.decode().split())


def scan_history(rng, deny):
    hits = 0
    args = ["log", "--format=%H%x00%h%x00%ad%x00%cd%x00%ae%x00%ce%x00%an%n%ae%n%cn%n%ce%n%B%x01", "--date=iso"]
    args += ["--all"] if rng in (None, "--all") else [rng]
    p = git(*args, check=False)
    if p.returncode != 0:
        print("history: git log failed for the range given")
        return 1
    allowed = allowed_trailers()
    revs, shorts = [], {}

    def report(full, short, kind, path=None):
        where = f"commit {short}" + (f" {os.path.dirname(path) or '.'}/" if path else "")
        if deny.commit_allowed(full, kind, path):
            print(f"NOTE: {where}: {kind} (excused by an allow-commit line; rewrite the commit to clear it)")
            return 0
        print(f"{where}: {kind}")
        return 1

    for rec in p.stdout.decode("utf-8", "replace").split("\x01"):
        rec = rec.strip("\n")
        if not rec:
            continue
        full, short, ad, cd, ae, ce, body = rec.split("\x00", 6)
        revs.append(full)
        shorts[full[:7]] = (full, short)
        if any(not e.endswith("@users.noreply.github.com") for e in (ae, ce)):
            hits += report(full, short, "identity")
        if any(not d.rstrip().endswith("+0000") for d in (ad, cd)):
            hits += report(full, short, "timezone")
        if any((m := TRAILER.match(line)) and m.group(1) not in allowed for line in body.splitlines()):
            hits += report(full, short, "link-trailer")
        for k in sorted({k for _, k, _ in scan_text(body.splitlines(), deny, lambda n: n)}):
            hits += report(full, short, k)
    # every file the range added or changed: a name deleted again before the push still leaves in history
    files = history_files(revs)
    got = blobs(sha for _, _, sha in files)
    done = set()
    for key, path, sha in files:
        if (sha, path) in done:
            continue
        done.add((sha, path))
        full, short = shorts[key]
        kinds = {k for _, k, _ in scan_text([path], deny, lambda n: n)}
        kinds |= {k for _, k in scan_blob(path, got.get(sha, b""), deny, lambda n: n)}
        for k in sorted(kinds):
            hits += report(full, short, k, path)
    return hits


def scan_tag(obj, deny):
    """An annotated tag object (and any tag it points to): tag name, tagger, time zone and message."""
    hits, allowed, seen = 0, allowed_trailers(), set()
    while obj not in seen:
        seen.add(obj)
        if git("cat-file", "-t", obj, check=False).stdout.strip() != b"tag":
            break
        text = git("cat-file", "-p", obj).stdout.decode("utf-8", "replace")
        head, _, body = text.partition("\n\n")
        body = body.split("-----BEGIN PGP SIGNATURE-----")[0].split("-----BEGIN SSH SIGNATURE-----")[0]
        where = f"tag {obj[:7]}"
        nxt, tagger = None, False
        for line in head.splitlines():
            key, _, val = line.partition(" ")
            if key == "object":
                nxt = val.strip()
            elif key == "tag":
                for k in sorted({k for _, k, _ in scan_text([val], deny, lambda n: n)}):
                    print(f"{where} (name): {k}")
                    hits += 1
            elif key == "tagger":
                tagger = True
                m = re.match(r"(.*) <([^>]*)> \d+ ([+-]\d{4})$", val)
                if not m:
                    print(f"{where}: tagger line unreadable")
                    hits += 1
                    continue
                if not m.group(2).endswith("@users.noreply.github.com"):
                    print(f"{where}: identity")
                    hits += 1
                if m.group(3) != "+0000":
                    print(f"{where}: timezone (tag with TZ=UTC0; setup-clone.sh defines `git tg`)")
                    hits += 1
                for k in sorted({k for _, k, _ in scan_text([m.group(1)], deny, lambda n: n)}):
                    print(f"{where} (tagger): {k}")
                    hits += 1
        if not tagger:
            print(f"{where}: no tagger line")
            hits += 1
        lines = body.splitlines()
        if any((t := TRAILER.match(line)) and t.group(1) not in allowed for line in lines):
            print(f"{where}: link-trailer")
            hits += 1
        for w, k, _ in scan_text(lines, deny, lambda n: f"{where} message line {n}", {"dash": DASH.pattern}):
            print(f"{w}: {k}")
            hits += 1
        if not nxt:
            break
        obj = nxt
    return hits


# ---------------------------------------------------------------- self-test

def _png(*chunks):
    out = b"\x89PNG\r\n\x1a\n"
    for typ, body in chunks + ((b"IEND", b""),):
        out += struct.pack(">I", len(body)) + typ + body + struct.pack(">I", zlib.crc32(typ + body))
    return out


def _riff(kind, *chunks):
    body = kind + b"".join(c + struct.pack("<I", len(b)) + b + (b"\0" if len(b) & 1 else b"") for c, b in chunks)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _box(typ, body):
    return struct.pack(">I", 8 + len(body)) + typ + body


TIFF_GPS = b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", 1) + struct.pack("<HHII", 0x8825, 4, 1, 0) + b"\0" * 4
# the private samples below are joined at run time, so that this file passes its own generic patterns
PRIVATE_TEXT = b"/ho" b"me/alice/models/checkpoint"


def self_test():
    """Every detector against samples built here: each 'bad' sample must be caught, each 'good' one must pass."""
    deny = Deny(None)
    ihdr = (b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0))
    media = [
        ("bad", "a.png", _png(ihdr, (b"eXIf", TIFF_GPS))),
        ("bad", "b.png", _png(ihdr, (b"zTXt", b"workflow\0\0" + zlib.compress(PRIVATE_TEXT)))),
        ("bad", "c.png", _png(ihdr, (b"iTXt", b"parameters\0\1\0\0\0" + zlib.compress(PRIVATE_TEXT)))),
        ("bad", "d.png", _png(ihdr, (b"tEXt", b"Author\0someone"))),
        ("good", "e.png", _png(ihdr, (b"tEXt", b"Software\0matplotlib version 3.9"))),
        ("bad", "f.webp", _riff(b"WEBP", (b"VP8L", b"\0" * 6), (b"EXIF", TIFF_GPS))),
        ("bad", "g.webp", _riff(b"WEBP", (b"VP8L", b"\0" * 6), (b"XMP ", b"<x:xmpmeta/>"))),
        ("good", "h.webp", _riff(b"WEBP", (b"VP8L", b"\0" * 6))),
        ("bad", "i.jpg", b"\xff\xd8\xff\xe1" + struct.pack(">H", 2 + len(TIFF_GPS) + 6) + b"Exif\0\0" + TIFF_GPS
         + b"\xff\xda\0\x02\xff\xd9"),
        ("bad", "j.jpg", b"\xff\xd8\xff\xfe\x00\x09someone\xff\xda\0\x02\xff\xd9"),
        ("good", "k.jpg", b"\xff\xd8\xff\xdb\x00\x03\x00\xff\xda\0\x02\xff\xd9"),
        ("bad", "l.gif", b"GIF89a\1\0\1\0\0\0\0" + b"\x21\xfe\x07someone\0" + b"\x3b"),
        ("good", "m.gif", b"GIF89a\1\0\1\0\0\0\0" + b"\x2c\0\0\0\0\1\0\1\0\0\2\2\x4c\x01\0" + b"\x3b"),
        ("bad", "n.tif", TIFF_GPS),
        ("bad", "o.mp3", b"ID3\x04\0\0\0\0\0\0" + b"\xff\xfb" + b"\0" * 20),
        ("bad", "p.wav", _riff(b"WAVE", (b"fmt ", b"\0" * 16), (b"LIST", b"INFOIART\x08\0\0\0someone\0"))),
        ("good", "q.wav", _riff(b"WAVE", (b"fmt ", b"\0" * 16), (b"data", b"\0" * 4))),
        ("bad", "r.flac", b"fLaC" + bytes([0x84]) + (16).to_bytes(3, "big") + struct.pack("<I", 4) + b"Lavf"
         + struct.pack("<I", 1) + struct.pack("<I", 0)),
        ("bad", "s.mp4", _box(b"ftyp", b"isom\0\0\0\0") + _box(b"moov", _box(b"udta", _box(b"\xa9xyz", b"+1+1/")))),
        ("good", "t.mp4", _box(b"ftyp", b"isom\0\0\0\0") + _box(b"moov", _box(b"mvhd", b"\0" * 8))),
        ("bad", "u.heic", _box(b"ftyp", b"heic\0\0\0\0") + _box(b"meta", b"\0\0\0\0" + _box(
            b"iinf", b"\0\0\0\0\0\1" + _box(b"infe", b"\2\0\0\0\0\1\0\0Exif\0")))),
        ("bad", "v.svg", b'<svg xmlns="http://www.w3.org/2000/svg"><metadata><dc:creator>someone</dc:creator>'
                         b"</metadata></svg>"),
        ("good", "w.svg", b'<svg xmlns="http://www.w3.org/2000/svg"><title>x</title></svg>'),
        ("bad", "x.pdf", b"%PDF-1.4\n1 0 obj << /Author (someone) >> endobj\n%%EOF"),
        ("bad", "y.pdf", b"%PDF-1.4\n1 0 obj << /Author <FEFF0061> >> endobj\n%%EOF"),
        ("good", "z.pdf", b"%PDF-1.4\n1 0 obj << /Producer (x) >> endobj\n%%EOF"),
    ]
    text = [
        ("bad", "ipv6", "addr fd12:" + "3456:789a:" + "1::17 up"), ("bad", "ipv6", "2a02:1810:4d02:" + "5e00:1:2:3:4"),
        ("bad", "gid-ipv4-hex", "GID 0000:0000:0000:" + "0000:0000:ffff:" + "c0a8:0a11"),
        ("bad", "eui64", "fe80::a1b2:" + "c3ff:fed4:e5f6"), ("bad", "mac-dash", "hw 0A-1B-2C-" + "3D-4E-5F"),
        ("bad", "ipv4-public", "resolved to 8.8." + "4.4."), ("good", "ipv6", "at 12:34:56, x[::2], a::b, std::vector"),
        ("good", "ipv6", "::1 and 2001:db8::7 are fine"), ("good", "ipv4-public", "192.0.2.1 255.255.255.0 0.0.0.0"),
        ("good", "ipv4-public", "127.0.0.1 and 224.0.0.251 and 1.2.3"), ("good", "mac-dash", "2026-10-04-21-21"),
    ]
    fails = []
    for want, name, data in media:
        probs, texts = media_meta(name, data)
        caught = bool(probs) or any(True for t in texts for _ in scan_text(t.splitlines(), deny, lambda n: n))
        if caught != (want == "bad"):
            fails.append(f"media {name}: expected {'a hit' if want == 'bad' else 'no hit'}")
    zt = media_rules("b.png", [media[1][2]], deny)
    if not any(k == "home-path" for _, k in zt):
        fails.append("media b.png: the compressed text was not scanned")
    for want, kind, line in text:
        got = {k for _, k, _ in scan_text([line], deny, lambda n: n)}
        if (kind in got) != (want == "bad"):
            fails.append(f"pattern {kind}: expected {'a hit' if want == 'bad' else 'no hit'} on sample '{line}'")
    for f in fails:
        print(f"self-test FAIL: {f}")
    print(f"check_private --self-test: {len(media) + len(text) + 1 - len(fails)} of {len(media) + len(text) + 1} "
          f"passed")
    return 1 if fails else 0


# ---------------------------------------------------------------- main

def main(argv):
    if argv[:1] == ["--self-test"]:
        return self_test()
    path = denylist_path()
    if path and not os.path.isfile(path):
        print("check_private: the deny-list named by PRIVATE_DENYLIST or privacy.denylist does not exist")
        return 1
    deny = Deny(path)
    note = " (private deny-list loaded)" if path else " (no private deny-list: generic patterns only)"
    if argv[:1] == ["--message"]:
        hits = 0
        for where, kind, _ in scan_text(message_lines(argv[1]), deny, lambda n: f"message line {n}",
                                        {"dash": DASH.pattern}):
            print(f"{where}: {kind}")
            hits += 1
        print(f"check_private --message: {hits} hit(s){note}")
        return 1 if hits else 0
    if argv[:1] == ["--history"]:
        hits = scan_history(argv[1] if len(argv) > 1 else None, deny)
        if len(argv) < 2 or argv[1] == "--all":      # a pushed range need not reach the excused commits
            for w in deny.warnings(history=True):
                if "allow-commit" in w:
                    print(w)
        print(f"check_private --history: {hits} hit(s){note}")
        return 1 if hits else 0
    if argv[:1] == ["--tag"]:
        if len(argv) < 2:
            print("usage: check_private.py --tag OBJECT")
            return 1
        hits = scan_tag(argv[1], deny)
        print(f"check_private --tag: {hits} hit(s){note}")
        return 1 if hits else 0
    rels = candidates(argv)
    index = staged() if not argv else {}
    versions = contents(rels, index) if not argv else {r: [("", b)] for r in rels if (b := work_bytes(r)) is not None}
    hits = scan_files(rels, deny, versions)
    raw = {r: [d for _, d in v] for r, v in versions.items()}
    found = repo_rules(rels, index, raw) if not argv else []
    for rel in rels:
        found += media_rules(rel, raw.get(rel, []), deny)
    for rel, kind in sorted(set(found)):
        print(f"{rel}: {kind}")
        hits += 1
    for w in deny.warnings():
        print(w)
    print(f"check_private: {hits} hit(s) in {len(rels)} file(s){note}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
