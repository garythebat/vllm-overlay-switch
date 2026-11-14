"""The parts of the switch that need no web framework and no GPU: who may call, do the ranks agree,
and the order of a flip. Kept apart so they can be tested anywhere."""

from __future__ import annotations

import hmac
import os
import time
from typing import Any

VARIANTS = ("base", "overlay")
PAUSE_MODES = ("wait", "abort")
PREPARE_TIMEOUT_S = 600.0
SWITCH_TIMEOUT_S = 120.0
STATUS_TIMEOUT_S = 30.0
LOOPBACK = frozenset({"127.0.0.1", "::1"})


def allowed_clients(env: dict[str, str] | None = None) -> frozenset[str]:
    """Loopback, plus whatever OVERLAY_SWITCH_ALLOW names (comma separated addresses)."""
    env = os.environ if env is None else env
    extra = {a.strip() for a in env.get("OVERLAY_SWITCH_ALLOW", "").split(",") if a.strip()}
    return LOOPBACK | frozenset(extra)


def refusal(client_host: str | None, token_header: str | None, env: dict[str, str] | None = None) -> str | None:
    """Why this caller is refused, or None if it may call. Pure, so it can be tested."""
    env = os.environ if env is None else env
    if client_host is None or client_host not in allowed_clients(env):
        return "this address may not use the overlay switch"
    want = env.get("OVERLAY_SWITCH_TOKEN", "")
    if want and not hmac.compare_digest(want.encode(), (token_header or "").encode()):
        return "missing or wrong X-Overlay-Token"
    return None


def agree(results: list[Any], variant: str | None = None) -> tuple[bool, str]:
    """Do all ranks report ok and the same active set (and, if given, this one)?"""
    if not isinstance(results, list) or not results:
        return False, "no rank answered"
    bad = [r for r in results if not (isinstance(r, dict) and r.get("ok"))]
    if bad:
        first = bad[0] if isinstance(bad[0], dict) else {"error": repr(bad[0])[:200]}
        return False, f"{len(bad)} of {len(results)} ranks failed: {first.get('error', 'not ok')}"
    active = {r.get("active") for r in results}
    if len(active) != 1:
        return False, f"ranks disagree on the active set: {sorted(str(a) for a in active)}"
    if variant is not None and active != {variant}:
        return False, f"ranks report {active.pop()!r}, wanted {variant!r}"
    return True, ""


def all_prepared(results: list[Any]) -> bool:
    """Has every rank found both sets and checked them against its live model?"""
    return isinstance(results, list) and bool(results) and all(isinstance(r, dict) and r.get("prepared") for r in results)


async def flip(engine, variant: str, mode: str, force: bool = False) -> tuple[dict, int]:
    """Pause, copy on every rank, verify, resume. Returns (answer, HTTP status).

    `force` is the way out of an unclear state (ranks that disagree, or weights that match
    neither set, as a flip that failed halfway leaves them): the chosen set is written onto
    every rank whatever is there now. There is then nothing to put back if it fails.
    """
    t0 = time.monotonic()
    timings: dict[str, float] = {}

    async def rpc(method: str, timeout: float, *args):
        return await engine.collective_rpc(method, timeout=timeout, args=args)

    try:
        before = await rpc("overlay_status", STATUS_TIMEOUT_S)
        if not all_prepared(before):
            t = time.monotonic()
            before = await rpc("overlay_prepare", PREPARE_TIMEOUT_S)
            timings["prepare_s"] = round(time.monotonic() - t, 3)
            if not all_prepared(before):
                _, why = agree(before)
                return {"ok": False, "error": f"could not prepare the alternate set: {why}", "ranks": before}, 500
        ok, why = agree(before)
        if not ok and not force:
            return {"ok": False, "ranks": before,
                    "error": f"refusing to flip from an unclear state: {why}. "
                             f"Send \"force\": true to write {variant!r} onto every rank anyway."}, 409
        previous = before[0]["active"] if ok else None
        if previous == variant and not force:
            return {"ok": True, "changed": False, "active": variant, "ranks": before, "timings": timings}, 200
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}, 500

    t = time.monotonic()
    try:
        await engine.pause_generation(mode=mode, clear_cache=True)
    except Exception as e:
        # nothing was copied; make sure the engine is not left half paused
        try:
            await engine.resume_generation()
        except Exception:
            pass
        return {"ok": False, "changed": False, "active": previous,
                "error": f"could not pause generation, nothing was changed: {type(e).__name__}: {e}"[:400]}, 500
    timings["pause_s"] = round(time.monotonic() - t, 3)
    resume = True
    try:
        t = time.monotonic()
        try:
            after = await rpc("overlay_switch", SWITCH_TIMEOUT_S, variant)
            ok, why = agree(after, variant)
        except Exception as e:
            after, ok, why = [], False, f"{type(e).__name__}: {e}"[:300]
        timings["copy_s"] = round(time.monotonic() - t, 3)
        if not ok:
            # put every rank back on what it had; a model split across two sets must not answer
            if previous is None:
                back, back_ok, back_why = [], False, "the state before was unclear, so there is nothing to put back"
            else:
                try:
                    back = await rpc("overlay_switch", SWITCH_TIMEOUT_S, previous)
                    back_ok, back_why = agree(back, previous)
                except Exception as e:
                    back, back_ok, back_why = [], False, f"{type(e).__name__}: {e}"[:300]
            if not back_ok:
                resume = False
                return {"ok": False, "paused": True, "active": None, "ranks": back or after, "timings": timings,
                        "error": f"the flip failed ({why}) and so did putting the previous set back ({back_why}). "
                                 "Generation is left PAUSED: the ranks may hold different weights. "
                                 "Flip again with \"force\": true, or restart the server."}, 500
            await engine.reset_prefix_cache()
            return {"ok": False, "changed": False, "active": previous, "ranks": back, "timings": timings,
                    "error": f"the flip failed ({why}); {previous!r} was put back on every rank"}, 500
        await engine.reset_prefix_cache()
        timings["total_s"] = round(time.monotonic() - t0, 3)
        return {"ok": True, "changed": True, "active": variant, "previous": previous, "ranks": after, "timings": timings}, 200
    finally:
        if resume:
            await engine.resume_generation()
