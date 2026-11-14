"""The HTTP half of the switch: an endpoint plugin for the vLLM API server.

Two routes, attached through the engine's own `vllm.endpoint_plugins` mechanism, so no
engine file is edited and the developer mode stays off:

    GET  /overlay/status            which set is active on every rank (?full=1 reads every tensor)
    POST /overlay/switch            {"variant": "base" | "overlay", "mode": "wait" | "abort",
                                     "force": false}

A flip is: pause generation (running requests finish in "wait" mode, the prefix cache is
cleared), copy the chosen set into the live buffers on every rank, check that every rank
agrees, resume. If ranks disagree the previous set is copied back; if that fails as well,
generation stays paused and the answer says so, because a model whose ranks hold different
weights must not answer anyone. `"force": true` is the way out of that state: it writes the
chosen set onto every rank whatever is there, and resumes only if every rank then agrees.

With OVERLAY_GATE_PREFIX set (see gate.py) the same server answers on two sets of paths, the
base set on the usual ones and the overlay under the prefix, and flips by itself. The switch
route then only takes a forced flip, the way out of a failed one: which side is live is no
longer anyone's to choose by hand.

Who may call these two routes: the local machine only, unless OVERLAY_SWITCH_ALLOW lists other client
addresses. If OVERLAY_SWITCH_TOKEN is set, every call must also carry it in the
X-Overlay-Token header. These routes are outside the paths the engine's own API key guards.

The paths under the gate's prefix are outside them too, as the engine sees them. The gate is
therefore handed the engine's own key check (`engine_key_check`) and applies it to them.
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import gate as gating
from .logic import PAUSE_MODES, STATUS_TIMEOUT_S, VARIANTS, agree, flip, refusal, same_files


class OverlaySwitchEndpointPlugin:
    """Registered under the entry point group `vllm.endpoint_plugins` as `overlay_switch`."""

    name = "overlay_switch"
    required_tasks = ("generate",)

    def attach_router(self, app: FastAPI) -> None:
        prefix = gating.prefix_from_env()
        if prefix:
            # added here, before the engine adds its own middleware: the gate is then the innermost one
            gating.install(app, prefix, engine_key_check(app))

        @app.get("/overlay/status")
        async def overlay_status(raw_request: Request):
            why = _refuse(raw_request)
            if why:
                return JSONResponse({"ok": False, "error": why}, status_code=403)
            engine = _engine(raw_request)
            if engine is None:
                return JSONResponse({"ok": False, "error": "no engine on this server"}, status_code=503)
            # ?full=1 reads every tensor back instead of the first and last layer
            full = raw_request.query_params.get("full", "") in ("1", "true", "yes")
            try:
                ranks = await engine.collective_rpc("overlay_status", timeout=STATUS_TIMEOUT_S, args=(full,))
                paused = await engine.is_paused()
            except Exception as e:  # the extension is missing, or a rank did not answer
                return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}, status_code=500)
            same, why_not = agree(ranks)
            if same:
                same, why_not = same_files(ranks)
            answer = {"ok": same, "error": why_not or None, "paused": paused,
                      "active": ranks[0].get("active") if same else None, "ranks": ranks}
            gated = getattr(raw_request.app.state, "overlay_gate_prefix", None)
            if gated:
                gate = getattr(raw_request.app.state, "overlay_gate", None)
                answer["gate"] = {"prefix": gated, **(gate.describe() if gate else {"live": None, "in_flight": 0, "waiting": 0,
                                                                                    "flips": 0, "waited_s": 0.0, "broken": None})}
            return JSONResponse(answer)

        @app.post("/overlay/switch")
        async def overlay_switch(raw_request: Request):
            why = _refuse(raw_request)
            if why:
                return JSONResponse({"ok": False, "error": why}, status_code=403)
            engine = _engine(raw_request)
            if engine is None:
                return JSONResponse({"ok": False, "error": "no engine on this server"}, status_code=503)
            try:
                body = await raw_request.json()
            except Exception:
                body = None
            if not isinstance(body, dict):
                return JSONResponse({"ok": False, "error": "send a JSON object"}, status_code=400)
            variant, mode, force = body.get("variant"), body.get("mode", "wait"), body.get("force", False)
            if variant not in VARIANTS:
                return JSONResponse({"ok": False, "error": f"variant must be one of {list(VARIANTS)}"}, status_code=400)
            if mode not in PAUSE_MODES:
                return JSONResponse({"ok": False, "error": f"mode must be one of {list(PAUSE_MODES)}"}, status_code=400)
            if not isinstance(force, bool):
                return JSONResponse({"ok": False, "error": "force must be true or false"}, status_code=400)
            if getattr(raw_request.app.state, "overlay_gate_prefix", None):
                # gated: the path a request comes in on decides the side; by hand only the forced way out remains
                if not force:
                    return JSONResponse({"ok": False, "error": "this server is gated: requests on the usual paths get the base "
                                                               "set, requests under the gate's prefix the overlay. Send "
                                                               "\"force\": true to write one set onto every rank after a "
                                                               "failed flip."}, status_code=409)
                try:
                    answer = await gating.gate_of(raw_request.app.state).force(variant)
                except gating.GateError as e:
                    return JSONResponse({"ok": False, "error": str(e)}, status_code=e.status)
                return JSONResponse(answer)
            lock: asyncio.Lock = raw_request.app.state.overlay_switch_lock
            if lock.locked():
                return JSONResponse({"ok": False, "error": "another flip is under way"}, status_code=409)
            async with lock:
                answer, status = await flip(engine, variant, mode, force)
            return JSONResponse(answer, status_code=status)

    async def init_state(self, engine_client, state, args) -> None:
        state.overlay_switch_engine = engine_client
        state.overlay_switch_lock = asyncio.Lock()


def engine_key_check(app):
    """The engine's own API-key check, for the gate to run on what comes in under its prefix.

    The engine guards paths by how they begin, so a path under the prefix would pass it without a
    key. Returned is a function that wraps an ASGI app in the engine's own middleware, with the
    keys the engine itself uses (--api-key, else VLLM_API_KEY); None on a server without a key.
    If the server has a key and the engine's check cannot be found, this raises and the server
    does not start: a gate that cannot check the key must not open a second set of paths.
    """
    from vllm import envs

    args = getattr(app.state, "args", None)
    tokens = [key for key in (getattr(args, "api_key", None) or [envs.VLLM_API_KEY]) if key]
    if not tokens:
        return None
    from vllm.entrypoints.serve.middleware.authenticate import AuthenticationMiddleware

    return lambda inner: AuthenticationMiddleware(inner, tokens=tokens)


def _engine(raw_request: Request):
    return getattr(raw_request.app.state, "overlay_switch_engine", None)


def _refuse(raw_request: Request) -> str | None:
    client = raw_request.client.host if raw_request.client else None
    return refusal(client, raw_request.headers.get("x-overlay-token"))
