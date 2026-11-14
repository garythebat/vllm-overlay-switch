"""The HTTP half of the switch: an endpoint plugin for the vLLM API server.

Two routes, attached through the engine's own `vllm.endpoint_plugins` mechanism, so no
engine file is edited and the developer mode stays off:

    GET  /overlay/status            which set is active on every rank
    POST /overlay/switch            {"variant": "base" | "overlay", "mode": "wait" | "abort"}

A flip is: pause generation (running requests finish in "wait" mode, the prefix cache is
cleared), copy the chosen set into the live buffers on every rank, check that every rank
agrees, resume. If ranks disagree the previous set is copied back; if that fails as well,
generation stays paused and the answer says so, because a model whose ranks hold different
weights must not answer anyone.

Who may call it: the local machine only, unless OVERLAY_SWITCH_ALLOW lists other client
addresses. If OVERLAY_SWITCH_TOKEN is set, every call must also carry it in the
X-Overlay-Token header. These routes are outside the paths the engine's own API key guards.
"""

from __future__ import annotations

import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .logic import PAUSE_MODES, STATUS_TIMEOUT_S, VARIANTS, agree, flip, refusal


class OverlaySwitchEndpointPlugin:
    """Registered under the entry point group `vllm.endpoint_plugins` as `overlay_switch`."""

    name = "overlay_switch"
    required_tasks = ("generate",)

    def attach_router(self, app: FastAPI) -> None:
        @app.get("/overlay/status")
        async def overlay_status(raw_request: Request):
            why = _refuse(raw_request)
            if why:
                return JSONResponse({"ok": False, "error": why}, status_code=403)
            engine = _engine(raw_request)
            if engine is None:
                return JSONResponse({"ok": False, "error": "no engine on this server"}, status_code=503)
            try:
                ranks = await engine.collective_rpc("overlay_status", timeout=STATUS_TIMEOUT_S)
                paused = await engine.is_paused()
            except Exception as e:  # the extension is missing, or a rank did not answer
                return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}, status_code=500)
            same, why_not = agree(ranks)
            return JSONResponse({"ok": same, "error": why_not or None, "paused": paused,
                                 "active": ranks[0].get("active") if same else None, "ranks": ranks})

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
            variant, mode = body.get("variant"), body.get("mode", "wait")
            if variant not in VARIANTS:
                return JSONResponse({"ok": False, "error": f"variant must be one of {list(VARIANTS)}"}, status_code=400)
            if mode not in PAUSE_MODES:
                return JSONResponse({"ok": False, "error": f"mode must be one of {list(PAUSE_MODES)}"}, status_code=400)
            lock: asyncio.Lock = raw_request.app.state.overlay_switch_lock
            if lock.locked():
                return JSONResponse({"ok": False, "error": "another flip is under way"}, status_code=409)
            async with lock:
                answer, status = await flip(engine, variant, mode)
            return JSONResponse(answer, status_code=status)

    async def init_state(self, engine_client, state, args) -> None:
        state.overlay_switch_engine = engine_client
        state.overlay_switch_lock = asyncio.Lock()


def _engine(raw_request: Request):
    return getattr(raw_request.app.state, "overlay_switch_engine", None)


def _refuse(raw_request: Request) -> str | None:
    client = raw_request.client.host if raw_request.client else None
    return refusal(client, raw_request.headers.get("x-overlay-token"))
