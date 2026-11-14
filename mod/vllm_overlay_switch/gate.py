"""One loaded model, two endpoints.

With `OVERLAY_GATE_PREFIX=/free` the server answers on its usual paths with the base set and on
the same paths under `/free` with the overlay: `/v1/chat/completions` and
`/free/v1/chat/completions`. The gate sits in front of every request and flips the weights
when a request arrives for the side that is not live.

A request can also choose the overlay by the model it names. `OVERLAY_GATE_MODELS` lists model
names that mean the overlay on the usual paths; serve them as further names of the one model
(`--served-model-name BASE_NAME OVERLAY_NAME`) and a client that picks from the model list
finds both under one address. Under the prefix every request is for the overlay, whatever it names.

The two sides never run in the same instant: the kernel takes one set of weights for everything
it decodes together. So:

- requests for the live side are let in as they come and run together, as without the gate;
- the first request for the other side waits until the ones in flight have ended, flips, and
  runs; requests that arrive behind it wait their turn in order, whatever their side, so that
  neither side can starve the other;
- a request is never handed to the engine while its side is not live. The engine's own pause
  is not enough for that: what arrives during a pause is run after it, on whatever is live then.

Each side keeps its own cached prefixes. The gate puts a cache salt naming the side into every
request that runs the model, so nothing computed with one set is ever reused with the other, and
a flip does not clear the cache: coming back to a long conversation does not mean reading it
again. That rests on a flip back being exact, which the status route's `derived` digest shows.

Every answer carries `x-overlay-side: base` or `overlay`.

What is let through, and what is not. GET requests pass (under the prefix with the prefix taken
off). Of the other requests only the ones this module knows are let in: the routes that run the
model and accept a cache salt, and a short list that does not run it. Anything else is refused,
so that no route can run the model behind the gate's back. For the same reason a request that
asks to be run in the background is refused: the model would run after the gate had let it go.

The server's API key. The engine checks its key by how a path begins (/v1 and a few more), and a
path under the prefix does not begin that way when the engine looks at it. So the gate is given
the engine's own check (`key_check`) and runs it again for every request under the prefix, on
the path the request is really for. Without it the prefix would be a way around the key.

Since the gate rewrites every chat request anyway, it can also translate `reasoning_effort`
into what the served model accepts (effort.py, off unless configured).

No engine import here: the gate needs an object with the engine client's `collective_rpc`,
`pause_generation`, `resume_generation` and `is_paused`, and is tested without an engine.
"""

from __future__ import annotations

import asyncio
import collections
import functools
import hashlib
import json
import os
import time

from . import effort as efforts
from .logic import VARIANTS, flip

PREFIX_ENV = "OVERLAY_GATE_PREFIX"
MODELS_ENV = "OVERLAY_GATE_MODELS"
MODEL_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages")
PASS_PATHS = ("/tokenize", "/detokenize", "/v1/messages/count_tokens")       # they render and count; none runs the model
SWITCH_PATHS = ("/overlay/switch",)
HEADER = b"x-overlay-side"
MAX_BODY = 64 << 20


class GateError(Exception):
    """A request the gate could not serve; `status` is the HTTP status to answer with."""

    def __init__(self, message: str, status: int = 503):
        super().__init__(message)
        self.status = status


class _Ticket:
    __slots__ = ("side",)

    def __init__(self, side: str):
        self.side = side


class Gate:
    """Who may run now. One per server; every flip goes through it."""

    def __init__(self, flipper):
        self._flip = flipper              # async (variant, force) -> (answer, http status)
        self.live: str | None = None      # the side the weights are known to hold; None until the first request
        self.broken: str | None = None    # set when a flip left the engine paused
        self.in_flight = 0
        self.flips = 0
        self.waited_s = 0.0
        self._queue: collections.deque[_Ticket] = collections.deque()
        self._cond = asyncio.Condition()
        self._wakes: set[asyncio.Future] = set()

    def describe(self) -> dict:
        return {"live": self.live, "in_flight": self.in_flight, "waiting": len(self._queue), "flips": self.flips,
                "waited_s": round(self.waited_s, 3), "broken": self.broken}

    async def enter(self, side: str) -> None:
        """Returns once `side` is live and this request may run. Call `leave()` when it has ended."""
        me, t0 = _Ticket(side), time.monotonic()
        async with self._cond:
            self._queue.append(me)
            try:
                while True:
                    if self.broken:
                        raise GateError(self.broken)
                    ahead = []
                    for ticket in self._queue:
                        if ticket is me:
                            break
                        ahead.append(ticket)
                    if self.live == side and all(t.side == side for t in ahead):
                        break                                   # my side is live and nobody of the other side is before me
                    if not ahead and self.in_flight == 0:
                        await self._change_to(side)             # first in line, other side live, nothing in flight
                        continue
                    await self._cond.wait()
                self.in_flight += 1
            finally:
                self._queue.remove(me)
                self.waited_s += time.monotonic() - t0
                self._cond.notify_all()

    def leave(self) -> None:
        """The request has ended. Not a coroutine, on purpose: a request that is being cancelled must still be
        counted out, and the count must not hang on an await that a second cancellation could cut."""
        self.in_flight -= 1
        wake = asyncio.ensure_future(self._wake())          # those waiting look again; a task of its own, so it outlives us
        self._wakes.add(wake)
        wake.add_done_callback(self._wakes.discard)

    async def _wake(self) -> None:
        async with self._cond:
            self._cond.notify_all()

    async def _change_to(self, side: str, force: bool = False) -> dict:
        """Flip under the gate's lock. Nothing is in flight and nothing can be let in meanwhile."""
        try:
            answer, _ = await self._flip(side, force)
        except Exception as e:                                  # the flip itself is written not to raise; be sure anyway
            answer = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300], "paused": None}
        except BaseException:
            # the request that had asked for this flip was cancelled in the middle of it (a server going down):
            # what the ranks hold is not known here any more, so nobody is let in until a forced flip says
            self.live, self.broken = None, "a flip was interrupted: what the ranks hold is not known; force one side"
            self._cond.notify_all()
            raise
        active = answer.get("active")
        if answer.get("ok") and active == side:
            self.flips += bool(answer.get("changed"))
            self.live, self.broken = side, None
            return answer
        self.live = active if active in VARIANTS else None
        why = str(answer.get("error") or "the flip did not report the side asked for")[:400]
        if answer.get("paused") or self.live is None:
            self.broken = why
        self._cond.notify_all()
        raise GateError(why)

    async def force(self, side: str) -> dict:
        """The way out of a broken state: write one side onto every rank, whatever is there."""
        async with self._cond:
            while self.in_flight:
                await self._cond.wait()
            self.broken = None
            try:
                return await self._change_to(side, force=True)
            finally:
                self._cond.notify_all()


def gate_of(state) -> Gate:
    """The server's one gate, made when first asked for (the engine client is there by then)."""
    gate = getattr(state, "overlay_gate", None)
    if gate is None:
        engine = getattr(state, "overlay_switch_engine", None)
        if engine is None:
            raise GateError("no engine on this server")

        async def flipper(variant: str, force: bool):
            # no cache clear: each side's cached prefixes are kept apart by the salt the gate sets
            return await flip(engine, variant, "wait", force, clear_cache=False)

        gate = state.overlay_gate = Gate(flipper)
    return gate


def install(app, prefix: str, key_check=None, overlay_models=(), effort=None) -> None:
    """Called by the endpoint plugin while the app is built, before the engine adds its own middleware:
    the gate is then the innermost one and sees only requests that passed the engine's own checks.

    `key_check` wraps an ASGI app in the engine's API-key check, or is None on a server without a key.
    `overlay_models` are model names that choose the overlay on the usual paths.
    `effort` translates reasoning_effort on chat requests (effort.Effort), or is None."""
    prefix = "/" + prefix.strip("/")
    if prefix == "/" or any(p == prefix or p.startswith(prefix + "/") for p in MODEL_PATHS + PASS_PATHS + SWITCH_PATHS):
        raise ValueError(f"{PREFIX_ENV}={prefix!r} is not a usable path prefix")
    models = tuple(dict.fromkeys(overlay_models))
    if any(not isinstance(name, str) or not name for name in models):
        raise ValueError(f"{MODELS_ENV} must be model names: {overlay_models!r}")
    app.state.overlay_gate_prefix, app.state.overlay_gate_models, app.state.overlay_gate_effort = prefix, models, effort
    app.add_middleware(OverlayGateMiddleware, prefix=prefix, key_check=key_check, overlay_models=models, effort=effort)


def side_salt(side: str, client) -> str:
    """The cache salt a request runs with: its side, and the client's own salt if it sent one.

    The engine takes at most 128 characters and no '/', '@' or backslash, so the client's salt goes in as
    a digest: whatever the engine would have accepted from the client stays acceptable with the side in front."""
    if client is None:
        return f"overlay-gate:{side}"
    if not isinstance(client, str) or not client:
        raise GateError("Parameter 'cache_salt' must be a non-empty string if provided.", 400)
    return f"overlay-gate:{side}:{hashlib.sha256(client.encode()).hexdigest()[:32]}"


class OverlayGateMiddleware:
    """Pure ASGI middleware. The gate itself is made at the first request, when the engine client is there."""

    def __init__(self, app, prefix: str, key_check=None, overlay_models=(), effort=None):
        self.app, self.prefix, self.overlay_models, self.effort = app, prefix, frozenset(overlay_models), effort
        under = functools.partial(self._serve, True)
        # what comes in under the prefix has not met the engine's key check on the path it is for: it meets it here
        self._under = key_check(under) if key_check else under

    def gate(self, scope) -> Gate:
        return gate_of(scope["app"].state)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})    # nothing may reach the model behind the gate
            return
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope["path"]
        if path == self.prefix or path.startswith(self.prefix + "/"):
            inner = path[len(self.prefix):] or "/"
            return await self._under(dict(scope, path=inner, raw_path=inner.encode()), receive, send)
        return await self._serve(False, scope, receive, send)

    async def _serve(self, under: bool, scope, receive, send):
        """`under`: the request came in under the prefix (and `scope` already names the inner path)."""
        inner = scope["path"]
        if self.effort and scope["method"] == "GET" and inner == efforts.MODELS_PATH:
            return await self.app(scope, receive, _advertising(send, self.effort))
        if scope["method"] in ("GET", "HEAD", "OPTIONS") or inner in PASS_PATHS:
            return await self.app(scope, receive, send)
        if inner in SWITCH_PATHS and not under:
            return await self.app(scope, receive, send)              # the plugin's own route; it asks the gate itself
        if inner not in MODEL_PATHS:
            return await _answer(send, 403, f"{inner} is not served through the overlay gate")
        try:
            body = await _read_body(receive)
            try:
                request = json.loads(body)
            except ValueError:
                request = None
            if not isinstance(request, dict):
                raise GateError("send a JSON object: the gate has to mark the request with its side", 400)
            if request.get("background"):
                raise GateError("a request run in the background is not served through the overlay gate: the model "
                                "would run after the gate had let the request go", 400)
            model = request.get("model")
            side = "overlay" if under or (isinstance(model, str) and model in self.overlay_models) else "base"
            request["cache_salt"] = side_salt(side, request.get("cache_salt"))
            notes = self.effort.apply(request) if self.effort and inner == efforts.CHAT_PATH else []
            body = json.dumps(request).encode()
            gate = self.gate(scope)
            await gate.enter(side)
        except GateError as e:
            return await _answer(send, e.status, str(e))
        try:
            headers = [(k, v) for k, v in scope["headers"] if k.lower() not in (b"content-length", b"transfer-encoding")]
            headers.append((b"content-length", str(len(body)).encode()))
            await self.app(dict(scope, headers=headers), _replay(body, receive), _mark(send, side, notes))
        finally:
            gate.leave()


async def _read_body(receive) -> bytes:
    chunks, size = [], 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            raise GateError("the client went away before its request was complete", 400)
        chunk = message.get("body", b"")
        size += len(chunk)
        if size > MAX_BODY:
            raise GateError("the request is too large for the overlay gate", 413)
        chunks.append(chunk)
        if not message.get("more_body"):
            return b"".join(chunks)


def _replay(body: bytes, receive):
    """Hand the (rewritten) body to the app once, then let it see what the client does (a disconnect)."""
    sent = False

    async def replay():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return replay


def _mark(send, side: str, notes=()):
    """Every answer says its side, and what was done to its reasoning_effort if anything was."""
    extra = [(HEADER, side.encode())] + ([(efforts.HEADER, efforts.header_value(list(notes)))] if notes else [])

    async def marked(message):
        if message["type"] == "http.response.start":
            headers = [(k, v) for k, v in message.get("headers", []) if k.lower() not in (HEADER, efforts.HEADER)]
            message = dict(message, headers=headers + extra)
        await send(message)

    return marked


def _advertising(send, effort):
    """The model list, with the accepted effort values added to every model. Whatever is not a plain model
    list (another status, a body that does not parse, an encoded body) goes out exactly as it came."""
    start, chunks = None, []

    async def held(message):
        nonlocal start
        if message["type"] == "http.response.start":
            start = message
            return
        if message["type"] != "http.response.body" or start is None:
            return await send(message)
        chunks.append(message.get("body", b""))
        if message.get("more_body"):
            return
        body = b"".join(chunks)
        plain = start["status"] == 200 and not any(k.lower() == b"content-encoding" for k, _ in start.get("headers", []))
        new = effort.advertise(body) if plain else None
        if new is not None:
            headers = [(k, v) for k, v in start.get("headers", []) if k.lower() != b"content-length"]
            start, body = dict(start, headers=headers + [(b"content-length", str(len(new)).encode())]), new
        await send(start)
        await send({"type": "http.response.body", "body": body})

    return held


async def _answer(send, status: int, message: str) -> None:
    body = json.dumps({"error": {"message": message, "type": "overlay_gate", "code": status}}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


def prefix_from_env() -> str | None:
    value = os.environ.get(PREFIX_ENV, "").strip()
    return value or None


def models_from_env() -> tuple[str, ...]:
    return tuple(name.strip() for name in os.environ.get(MODELS_ENV, "").split(",") if name.strip())
