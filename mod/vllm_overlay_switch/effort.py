"""Translate `reasoning_effort` into what the served model accepts.

Clients send effort words from a ladder of their own (none, minimal, low, medium, high,
xhigh, ...). A model's renderer may accept fewer, and the engine then answers the others with
HTTP 400: every such client breaks until it carries a workaround of its own. With
OVERLAY_GATE_EFFORT_KEEP set, the gate rewrites the field on /v1/chat/completions before the
engine sees it, for both sides alike:

    OVERLAY_GATE_EFFORT_KEEP=none,low,high,xhigh,max     words passed on as they are (this turns it on)
    OVERLAY_GATE_EFFORT_MAP=minimal=low,medium=high,ultra=max,*=high
                                                          words rewritten; `*` stands for any other value
    OVERLAY_GATE_EFFORT_INT=1-100                         integers are accepted and clamped into this range

A request can carry the field in two places, the top-level `reasoning_effort` and
`chat_template_kwargs.reasoning_effort`, and both are treated alike. An integer that ends up
at the top level is moved into `chat_template_kwargs`: the engine's request schema takes only
words there, and integers only in the kwargs.

Nothing is rewritten silently. Every translation is named in the answer's
`x-reasoning-effort` header, counted for the status route, and logged once per distinct
incoming value. `GET /v1/models` gains, per model, the words and the integer range that are
accepted, so a client can read them instead of hardcoding a ladder.

No engine import here.
"""

from __future__ import annotations

import json
import logging
import os

KEEP_ENV, MAP_ENV, INT_ENV = "OVERLAY_GATE_EFFORT_KEEP", "OVERLAY_GATE_EFFORT_MAP", "OVERLAY_GATE_EFFORT_INT"
FIELD = "reasoning_effort"
CHAT_PATH = "/v1/chat/completions"
MODELS_PATH = "/v1/models"
HEADER = b"x-reasoning-effort"
MOST_SEEN = 64          # distinct translations remembered for the log and the status; clients can send anything

# a child of the engine's own logger, so its lines appear in the server log with the engine's format
log = logging.getLogger("vllm.overlay_switch.effort")


class Effort:
    """What to pass on, what to rewrite, and into what."""

    def __init__(self, keep, mapping=None, other=None, int_range=None):
        self.keep = tuple(dict.fromkeys(keep))
        self.mapping = dict(mapping or {})
        self.other = other
        self.int_range = tuple(int_range) if int_range else None
        self.seen: dict[str, int] = {}
        if not self.keep:
            raise ValueError(f"{KEEP_ENV} names no word to pass on")
        if self.int_range and self.int_range[0] > self.int_range[1]:
            raise ValueError(f"{INT_ENV}: {self.int_range[0]} is above {self.int_range[1]}")
        for source, target in [*self.mapping.items(), *([("*", other)] if other is not None else [])]:
            if not self._accepted(target):
                raise ValueError(f"{MAP_ENV}: {source}={target} leads to a value that is neither kept nor an integer in range")

    def _accepted(self, value) -> bool:
        if type(value) is int:
            return bool(self.int_range) and self.int_range[0] <= value <= self.int_range[1]
        return isinstance(value, str) and value in self.keep

    def translate(self, value):
        """(what to send on, a note for the log or None when nothing changed)."""
        if value is None:
            return None, None
        new = value
        if isinstance(value, float) and not isinstance(value, bool) and value == value and abs(value) != float("inf"):
            new = round(value)                                # 50.0 means 50
        elif isinstance(value, str):
            new = value.strip().lower()
            if new.lstrip("-").isdigit() and self.int_range:
                new = int(new)                                # "50" means 50
        if type(new) is int and self.int_range:
            new = max(self.int_range[0], min(self.int_range[1], new))
        elif isinstance(new, str) and new in self.keep:
            pass
        elif isinstance(new, str) and new in self.mapping:
            new = self.mapping[new]
        elif self.other is not None:
            new = self.other
            return new, f"{_show(value)} -> {new} (unknown)"
        else:
            return value, None                                # nothing configured for it: the engine decides
        return new, (None if new == value and type(new) is type(value) else f"{_show(value)} -> {new}")

    def apply(self, request: dict) -> list[str]:
        """Rewrite the field in a chat request, in both places. Returns the notes of what was changed."""
        notes = []
        kwargs = request.get("chat_template_kwargs")
        if kwargs is not None and not isinstance(kwargs, dict):
            return notes                                      # not a request the engine will take: leave it whole
        in_kwargs = kwargs is not None and kwargs.get(FIELD) is not None
        if in_kwargs:
            kwargs[FIELD], note = self.translate(kwargs[FIELD])
            notes += [note] if note else []
        if request.get(FIELD) is not None:
            came = request[FIELD]
            new, note = self.translate(came)
            if isinstance(new, str):
                request[FIELD] = new
            elif self._accepted(new):
                # an integer: the engine's schema takes only words at the top level, and integers in the kwargs.
                # What the kwargs say wins in the engine anyway, so a value already there stays and this one goes.
                del request[FIELD]
                if in_kwargs:
                    note = f"{_show(came)} dropped (chat_template_kwargs has its own)"
                else:
                    request["chat_template_kwargs"] = {**(kwargs or {}), FIELD: new}
                    note = note or f"{_show(came)} moved to chat_template_kwargs"
            notes += [note] if note else []
        for note in notes:
            if note in self.seen or len(self.seen) < MOST_SEEN:
                self.seen[note] = self.seen.get(note, 0) + 1
                if self.seen[note] == 1:
                    log.info("reasoning_effort coerced: %s (model=%s)", note, _show(request.get("model")))
        return notes

    def describe(self) -> dict:
        return {"keep": list(self.keep), "map": {**self.mapping, **({"*": self.other} if self.other is not None else {})},
                "int": list(self.int_range) if self.int_range else None, "coerced": dict(self.seen)}

    def advertise(self, body: bytes) -> bytes | None:
        """The model list with the accepted values added to every model; None when it is not a model list."""
        try:
            listing = json.loads(body)
        except ValueError:
            return None
        if not isinstance(listing, dict) or not isinstance(listing.get("data"), list):
            return None
        for model in listing["data"]:
            if isinstance(model, dict):
                model["reasoning_efforts"] = list(self.keep)
                model["reasoning_effort_aliases"] = dict(self.mapping)
                if self.int_range:
                    model["reasoning_effort_integer_min"], model["reasoning_effort_integer_max"] = self.int_range
        return json.dumps(listing).encode()


def _show(value) -> str:
    """A value as it came, short and safe for a log line and an HTTP header."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return "".join(c if 32 < ord(c) < 127 else "?" for c in text.replace(" ", "_"))[:40] or "''"


def header_value(notes: list[str]) -> bytes:
    return "; ".join(notes)[:200].encode("ascii", "replace")


def from_env() -> Effort | None:
    """The translation the environment asks for, or None. A mistake in it stops the server: better than guessing."""
    keep = [word.strip().lower() for word in os.environ.get(KEEP_ENV, "").split(",") if word.strip()]
    if not keep:
        return None
    mapping, other = {}, None
    for pair in os.environ.get(MAP_ENV, "").split(","):
        if not pair.strip():
            continue
        source, sep, target = (part.strip().lower() for part in pair.partition("="))
        if not sep or not source or not target:
            raise ValueError(f"{MAP_ENV}: {pair.strip()!r} is not of the form word=word")
        target = int(target) if target.lstrip("-").isdigit() else target
        if source == "*":
            other = target
        else:
            mapping[source] = target
    int_range, raw = None, os.environ.get(INT_ENV, "").strip()
    if raw:
        low, sep, high = raw.partition("-")
        if not sep or not low.strip().isdigit() or not high.strip().isdigit():
            raise ValueError(f"{INT_ENV}: {raw!r} is not of the form 1-100")
        int_range = (int(low), int(high))
    return Effort(keep, mapping, other, int_range)
