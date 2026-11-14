"""The gate: one loaded model, two endpoints. Standard library only; no web framework, no engine.

The stand-in engine is the one of test_logic (three ranks that hold "base" or "overlay"). The
stand-in server behind the gate records, for every request it is handed, which set the engine
held when the request started and when it ended, and holds the request open until the test
lets it go. So the tests can say exactly who ran together with whom and on what.

    python3 -m unittest tests.test_gate -v
"""

import asyncio
import hashlib
import json
import os
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "mod"))

from test_logic import FakeEngine  # noqa: E402
from vllm_overlay_switch import effort as efforts  # noqa: E402
from vllm_overlay_switch import gate as gating  # noqa: E402

CHAT, FREE_CHAT = "/v1/chat/completions", "/free/v1/chat/completions"


class Behind:
    """The server behind the gate: holds every request open until told, and writes down what it saw."""

    def __init__(self, engine):
        self.engine, self.seen, self.release, self.running, self.headers = engine, [], {}, 0, []
        self.most_at_once = 0
        self.models = None        # (status, headers, body chunks) to answer GET /v1/models with

    async def __call__(self, scope, receive, send):
        message = await receive()
        body = message.get("body", b"")
        try:
            request = json.loads(body)
        except ValueError:
            request = None
        name = request.get("name") if isinstance(request, dict) else None
        self.headers.append(scope["headers"])
        entry = {"name": name, "path": scope["path"], "method": scope["method"], "body": body, "request": request,
                 "length": dict(scope["headers"]).get(b"content-length"), "held_at_start": list(self.engine.active)}
        self.seen.append(entry)
        if self.models and scope["path"] == "/v1/models":
            status, headers, chunks = self.models
            await send({"type": "http.response.start", "status": status, "headers": list(headers)})
            for n, chunk in enumerate(chunks):
                await send({"type": "http.response.body", "body": chunk, "more_body": n < len(chunks) - 1})
            return
        self.running += 1
        self.most_at_once = max(self.most_at_once, self.running)
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        if name in self.release:
            await self.release[name].wait()                    # "generating": the stream is open
        entry["held_at_end"] = list(self.engine.active)
        self.running -= 1
        await send({"type": "http.response.body", "body": b"{}"})

    def hold(self, *names):
        for name in names:
            self.release[name] = asyncio.Event()


async def call(mw, app_obj, path, body=None, method="POST", raw=None, key=None):
    """One request through the middleware. Returns (status, headers, body)."""
    payload = raw if raw is not None else json.dumps(body if body is not None else {}).encode()
    scope = {"type": "http", "method": method, "path": path, "raw_path": path.encode(), "app": app_obj,
             "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]}
    if key:
        scope["headers"].append((b"authorization", f"Bearer {key}".encode()))
    sent, got = [], [{"type": "http.request", "body": payload, "more_body": False}]

    async def receive():
        if got:
            return got.pop(0)
        await asyncio.sleep(3600)

    async def send(message):
        sent.append(message)

    await mw(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    return start["status"], dict(start.get("headers", [])), b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")


async def settle():
    for _ in range(20):
        await asyncio.sleep(0)


class EngineKey:
    """Stands in for the engine's own API-key middleware, with its rule: a path that begins with
    one of these needs the key, any other passes (vllm/entrypoints/serve/middleware/authenticate.py)."""

    GUARDED = ("/v1", "/v2", "/inference", "/cohere")

    def __init__(self, app, key):
        self.app, self.key = app, key

    def __call__(self, scope, receive, send):
        given = dict(scope["headers"]).get(b"authorization", b"").decode()
        if scope.get("method") != "OPTIONS" and scope["path"].startswith(self.GUARDED) and given != f"Bearer {self.key}":
            return _unauthorized(send)
        return self.app(scope, receive, send)


async def _unauthorized(send):
    await send({"type": "http.response.start", "status": 401, "headers": []})
    await send({"type": "http.response.body", "body": b'{"error": "Unauthorized"}'})


class SlowFlip(FakeEngine):
    """An engine whose copy onto the ranks does not come back until the test says so."""

    def __init__(self):
        super().__init__()
        self.go = asyncio.Event()

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        if method == "overlay_switch":
            self.calls.append(("copy begun", *args))
            await self.go.wait()
        return await super().collective_rpc(method, timeout, args, kwargs)


class GateCase(unittest.TestCase):
    def make(self, engine=None, key=None, models=(), effort=None):
        self.engine = engine or FakeEngine()
        self.behind = Behind(self.engine)
        self.app_obj = types.SimpleNamespace(state=types.SimpleNamespace(overlay_switch_engine=self.engine))
        check = (lambda inner: EngineKey(inner, key)) if key else None
        self.mw = gating.OverlayGateMiddleware(self.behind, prefix="/free", key_check=check, overlay_models=models, effort=effort)
        return self.mw

    def post(self, path, name, **extra):
        return asyncio.ensure_future(call(self.mw, self.app_obj, path, {"name": name, **extra}))

    @property
    def gate(self):
        return self.app_obj.state.overlay_gate

    def flips_asked(self):
        return [c[1] for c in self.engine.calls if c[0] == "overlay_switch"]


class Order(GateCase):
    def test_one_side_runs_together_the_other_waits_and_nobody_jumps_the_line(self):
        async def scenario():
            self.make()
            self.behind.hold("b1", "b2", "f1", "b3")
            b1, b2 = self.post(CHAT, "b1"), self.post(CHAT, "b2")
            await settle()
            self.assertEqual((self.gate.in_flight, self.behind.running), (2, 2))          # the live side runs together
            f1 = self.post(FREE_CHAT, "f1")
            await settle()
            b3 = self.post(CHAT, "b3")                                                    # behind f1: it must wait its turn
            await settle()
            self.assertEqual([e["name"] for e in self.behind.seen], ["b1", "b2"])
            self.assertEqual((self.gate.describe()["waiting"], self.flips_asked()), (2, []))
            self.behind.release["b1"].set()
            await settle()
            self.assertEqual(self.flips_asked(), [])                                      # b2 is still in flight
            self.behind.release["b2"].set()
            await settle()
            self.assertEqual(([e["name"] for e in self.behind.seen], self.flips_asked()), (["b1", "b2", "f1"], ["overlay"]))
            self.assertEqual((self.gate.live, self.gate.in_flight), ("overlay", 1))
            self.behind.release["f1"].set()
            await settle()
            self.assertEqual(([e["name"] for e in self.behind.seen], self.flips_asked()),
                             (["b1", "b2", "f1", "b3"], ["overlay", "base"]))
            self.behind.release["b3"].set()
            results = await asyncio.gather(b1, b2, f1, b3)
            self.assertEqual([r[0] for r in results], [200] * 4)
            self.assertEqual([r[1][b"x-overlay-side"] for r in results], [b"base", b"base", b"overlay", b"base"])
            # nobody ever ran on the other side's weights, from first token to last
            for e in self.behind.seen:
                want = "overlay" if e["name"].startswith("f") else "base"
                self.assertEqual((e["held_at_start"], e["held_at_end"]), ([want] * 3, [want] * 3), e["name"])
            self.assertEqual((self.gate.flips, self.gate.in_flight, self.gate.describe()["waiting"]), (2, 0, 0))
        asyncio.run(scenario())

    def test_many_at_once_from_both_sides_never_mix(self):
        async def scenario():
            self.make()
            names = [f"{'f' if i % 3 == 0 else 'b'}{i}" for i in range(30)]
            tasks = [self.post(FREE_CHAT if n.startswith("f") else CHAT, n) for n in names]
            results = await asyncio.gather(*tasks)
            self.assertEqual({r[0] for r in results}, {200})
            self.assertEqual(len(self.behind.seen), 30)
            for e in self.behind.seen:
                want = "overlay" if e["name"].startswith("f") else "base"
                self.assertEqual((e["held_at_start"], e["held_at_end"]), ([want] * 3, [want] * 3), e["name"])
            self.assertEqual([e["name"] for e in self.behind.seen], names)               # served in the order they came
        asyncio.run(scenario())

    def test_the_first_request_prepares_and_a_base_request_writes_nothing(self):
        async def scenario():
            self.make(FakeEngine(prepared=False))
            status, headers, _ = await call(self.mw, self.app_obj, CHAT, {"name": "b"})
            self.assertEqual((status, headers[b"x-overlay-side"], self.gate.live, self.gate.flips), (200, b"base", "base", 0))
            self.assertEqual(self.engine.calls, [("overlay_status",), ("overlay_prepare",)])
        asyncio.run(scenario())

    def test_a_gated_flip_clears_no_cache(self):
        async def scenario():
            self.make()
            await call(self.mw, self.app_obj, FREE_CHAT, {"name": "f"})
            self.assertIn(("pause", "wait", False), self.engine.calls)
            self.assertNotIn(("reset_prefix_cache",), self.engine.calls)
        asyncio.run(scenario())

    def test_a_request_cut_off_while_it_runs_is_counted_out_at_once_and_the_other_side_gets_its_turn(self):
        async def scenario():
            self.make()
            self.behind.hold("b1")
            running = self.post(CHAT, "b1")
            await settle()
            waiting = self.post(FREE_CHAT, "f1")
            await settle()
            self.assertEqual((self.gate.in_flight, len(self.gate._queue), self.flips_asked()), (1, 1, []))
            running.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await running
            self.assertEqual(self.gate.in_flight, 0)             # counted out without waiting for anything
            status, headers, _ = await waiting
            self.assertEqual((status, headers[b"x-overlay-side"], self.behind.seen[-1]["held_at_start"]), (200, b"overlay", ["overlay"] * 3))
            await settle()                                       # the last wake-up has run by now and is forgotten
            self.assertEqual((self.gate.in_flight, len(self.gate._queue), len(self.gate._wakes)), (0, 0, 0))
        asyncio.run(scenario())

    def test_a_client_that_gives_up_while_waiting_does_not_hold_the_line(self):
        async def scenario():
            self.make()
            self.behind.hold("b1")
            b1 = self.post(CHAT, "b1")
            await settle()
            f1 = self.post(FREE_CHAT, "f1")
            await settle()
            b2 = self.post(CHAT, "b2")
            await settle()
            self.assertEqual(self.gate.describe()["waiting"], 2)
            f1.cancel()
            await settle()
            self.assertEqual(self.gate.describe()["waiting"], 0)                          # b2 no longer has f1 before it
            self.behind.release["b1"].set()
            await asyncio.gather(b1, b2)
            self.assertEqual((self.flips_asked(), [e["name"] for e in self.behind.seen]), ([], ["b1", "b2"]))
        asyncio.run(scenario())


class Failures(GateCase):
    def test_a_failed_flip_answers_503_and_the_live_side_goes_on_serving(self):
        async def scenario():
            self.make(FakeEngine(fail={("overlay_switch", "overlay"): (2,)}))
            self.assertEqual((await call(self.mw, self.app_obj, CHAT, {"name": "b1"}))[0], 200)
            status, _, body = await call(self.mw, self.app_obj, FREE_CHAT, {"name": "f1"})
            self.assertEqual(status, 503)
            self.assertIn("was put back", json.loads(body)["error"]["message"])
            self.assertEqual((self.gate.live, self.gate.broken, self.engine.active), ("base", None, ["base"] * 3))
            self.assertEqual((await call(self.mw, self.app_obj, CHAT, {"name": "b2"}))[0], 200)
            self.assertEqual([e["name"] for e in self.behind.seen], ["b1", "b2"])        # f1 never reached the model
        asyncio.run(scenario())

    def test_a_flip_that_leaves_the_engine_paused_shuts_the_gate_until_it_is_forced(self):
        async def scenario():
            self.make(FakeEngine(fail={("overlay_switch", "overlay"): (2,), ("overlay_switch", "base"): (0,)}))
            await call(self.mw, self.app_obj, CHAT, {"name": "b1"})
            self.assertEqual((await call(self.mw, self.app_obj, FREE_CHAT, {"name": "f1"}))[0], 503)
            self.assertIn("PAUSED", self.gate.broken)
            status, _, body = await call(self.mw, self.app_obj, CHAT, {"name": "b2"})
            self.assertEqual(status, 503)                                                 # nobody runs on a split model
            self.assertIn("PAUSED", json.loads(body)["error"]["message"])
            self.engine.fail = {}
            answer = await self.gate.force("base")
            self.assertEqual((answer["ok"], self.gate.live, self.gate.broken, self.engine.paused), (True, "base", None, False))
            self.assertEqual((await call(self.mw, self.app_obj, CHAT, {"name": "b3"}))[0], 200)
            self.assertEqual([e["name"] for e in self.behind.seen], ["b1", "b3"])
        asyncio.run(scenario())

    def test_a_flip_cut_off_in_the_middle_shuts_the_gate_until_it_is_forced(self):
        async def scenario():
            self.make(SlowFlip())
            self.engine.go.set()
            await call(self.mw, self.app_obj, CHAT, {"name": "b1"})
            self.engine.go.clear()
            asked = self.post(FREE_CHAT, "f1")
            await settle()
            self.assertEqual(self.engine.calls[-1], ("copy begun", "overlay"))     # the copy is under way on the ranks
            asked.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asked
            self.assertEqual((self.gate.live, "interrupted" in self.gate.broken, self.gate.in_flight, len(self.gate._queue)),
                             (None, True, 0, 0))
            for path in (CHAT, FREE_CHAT):
                status, _, body = await call(self.mw, self.app_obj, path, {"name": "x"})
                self.assertEqual((status, "interrupted" in json.loads(body)["error"]["message"]), (503, True), path)
            self.engine.go.set()
            answer = await self.gate.force("base")
            self.assertEqual((answer["ok"], self.gate.live, self.gate.broken, self.engine.paused), (True, "base", None, False))
            self.assertEqual((await call(self.mw, self.app_obj, CHAT, {"name": "b2"}))[0], 200)
            self.assertEqual([e["name"] for e in self.behind.seen], ["b1", "b2"])
        asyncio.run(scenario())

    def test_without_an_engine_the_gate_says_so(self):
        async def scenario():
            self.make()
            del self.app_obj.state.overlay_switch_engine
            status, _, body = await call(self.mw, self.app_obj, CHAT, {"name": "b"})
            self.assertEqual((status, json.loads(body)["error"]["type"]), (503, "overlay_gate"))
        asyncio.run(scenario())


class Key(GateCase):
    """The engine checks its key by how a path begins; under the prefix the gate has to run that check itself."""

    def test_under_the_prefix_nothing_the_engine_guards_is_served_without_the_key(self):
        async def scenario():
            self.make(key="s3cret")
            for path, method in ((FREE_CHAT, "POST"), ("/free/v1/completions", "POST"), ("/free/v1/models", "GET"),
                                 ("/free/v1/embeddings", "POST"), ("/free/v2/rerank", "POST")):
                for given in (None, "wrong"):
                    status, _, body = await call(self.mw, self.app_obj, path, {"name": "f"}, method=method, key=given)
                    self.assertEqual((status, body), (401, b'{"error": "Unauthorized"}'), (path, given))
            self.assertEqual((self.behind.seen, self.engine.calls, hasattr(self.app_obj.state, "overlay_gate")), ([], [], False))
            status, headers, _ = await call(self.mw, self.app_obj, FREE_CHAT, {"name": "f"}, key="s3cret")
            self.assertEqual((status, headers[b"x-overlay-side"], self.behind.seen[-1]["held_at_start"]), (200, b"overlay", ["overlay"] * 3))
            self.assertEqual((await call(self.mw, self.app_obj, "/free/v1/models", method="GET", key="s3cret"))[0], 200)
        asyncio.run(scenario())

    def test_what_the_engine_leaves_open_stays_open_under_the_prefix(self):
        async def scenario():
            self.make(key="s3cret")
            for path, method in (("/free/health", "GET"), ("/free/metrics", "GET"), ("/free/tokenize", "POST")):
                self.assertEqual((await call(self.mw, self.app_obj, path, {"name": "x"}, method=method))[0], 200, path)
            self.assertEqual([e["path"] for e in self.behind.seen], ["/health", "/metrics", "/tokenize"])
        asyncio.run(scenario())

    def test_the_usual_paths_are_left_to_the_engines_own_check(self):
        async def scenario():
            self.make(key="s3cret")
            # here nothing stands in front of the gate, so this passes: on a server the engine's own middleware does
            self.assertEqual((await call(self.mw, self.app_obj, CHAT, {"name": "b"}))[0], 200)
        asyncio.run(scenario())

    def test_a_server_without_a_key_asks_for_none(self):
        async def scenario():
            self.make()
            self.assertEqual((await call(self.mw, self.app_obj, FREE_CHAT, {"name": "f"}))[0], 200)
        asyncio.run(scenario())


class ByName(GateCase):
    """A client that picks from the model list: a name on the list of overlay names chooses the overlay."""

    def test_the_model_a_request_names_chooses_its_side_on_the_usual_paths(self):
        async def scenario():
            self.make(models=("m-free", "m-also-free"))
            for path in gating.MODEL_PATHS:
                for model, side in (("m", "base"), ("m-free", "overlay"), ("m-also-free", "overlay"), (None, "base")):
                    body = {"name": f"{path} {model}"} if model is None else {"name": f"{path} {model}", "model": model}
                    status, headers, _ = await call(self.mw, self.app_obj, path, body)
                    seen = self.behind.seen[-1]
                    self.assertEqual((status, headers[b"x-overlay-side"].decode(), seen["held_at_start"], seen["request"]["cache_salt"],
                                      seen["request"].get("model")), (200, side, [side] * 3, f"overlay-gate:{side}", model), (path, model))
        asyncio.run(scenario())

    def test_under_the_prefix_every_request_is_for_the_overlay_whatever_it_names(self):
        async def scenario():
            self.make(models=("m-free",))
            for model in ("m", "m-free", "nobody"):
                status, headers, _ = await call(self.mw, self.app_obj, FREE_CHAT, {"name": model, "model": model})
                self.assertEqual((status, headers[b"x-overlay-side"], self.behind.seen[-1]["held_at_start"]), (200, b"overlay", ["overlay"] * 3), model)
            self.assertEqual(self.flips_asked(), ["overlay"])
        asyncio.run(scenario())

    def test_a_model_that_is_no_name_at_all_is_left_to_the_engine_on_the_base_side(self):
        async def scenario():
            self.make(models=("m-free",))
            for model in (["m-free"], {"m-free": 1}, 7, True, "M-FREE", " m-free"):
                status, headers, _ = await call(self.mw, self.app_obj, CHAT, {"name": "x", "model": model})
                self.assertEqual((status, headers[b"x-overlay-side"], self.behind.seen[-1]["request"]["model"]), (200, b"base", model), model)
            self.assertEqual(self.flips_asked(), [])
        asyncio.run(scenario())

    def test_without_names_the_model_chooses_nothing(self):
        async def scenario():
            self.make()
            status, headers, _ = await call(self.mw, self.app_obj, CHAT, {"name": "x", "model": "m-free"})
            self.assertEqual((status, headers[b"x-overlay-side"]), (200, b"base"))
        asyncio.run(scenario())

    def test_the_names_are_read_from_the_environment_and_kept_for_the_status(self):
        saved = os.environ.get(gating.MODELS_ENV)
        try:
            os.environ[gating.MODELS_ENV] = " m-free, ,m-also-free,m-free "
            self.assertEqual(gating.models_from_env(), ("m-free", "m-also-free", "m-free"))
            added = []
            app = types.SimpleNamespace(state=types.SimpleNamespace(), add_middleware=lambda *a, **k: added.append(k))
            gating.install(app, "/free", None, gating.models_from_env())
            self.assertEqual((app.state.overlay_gate_models, added[0]["overlay_models"]), (("m-free", "m-also-free"),) * 2)
            os.environ.pop(gating.MODELS_ENV)
            self.assertEqual(gating.models_from_env(), ())
            with self.assertRaises(ValueError):
                gating.install(app, "/free", None, ("ok", 5))
        finally:
            os.environ.pop(gating.MODELS_ENV, None)
            if saved is not None:
                os.environ[gating.MODELS_ENV] = saved


def seven_steps():
    """A model that takes none, low, high, xhigh, max and 1 to 100, behind a client with seven steps."""
    return efforts.Effort(["none", "low", "high", "xhigh", "max"], {"minimal": "low", "medium": "high", "ultra": "max"}, "high", (1, 100))


class Effort(GateCase):
    """The gate rewrites every chat request anyway; it can put reasoning_effort into words the model takes."""

    def test_a_chat_request_is_translated_on_both_sides_and_says_so(self):
        async def scenario():
            self.make(models=("m-free",), effort=seven_steps())
            for path, model, side in ((CHAT, "m", "base"), (CHAT, "m-free", "overlay"), (FREE_CHAT, "m", "overlay")):
                status, headers, _ = await call(self.mw, self.app_obj, path, {"name": "c", "model": model, "reasoning_effort": "medium"})
                sent = self.behind.seen[-1]["request"]
                self.assertEqual((status, headers[b"x-overlay-side"].decode(), headers[b"x-reasoning-effort"], sent["reasoning_effort"]),
                                 (200, side, b"medium -> high", "high"), (path, model))
            status, headers, _ = await call(self.mw, self.app_obj, CHAT, {"name": "c", "reasoning_effort": 0,
                                                                          "chat_template_kwargs": {"reasoning_effort": "ultra"}})
            sent = self.behind.seen[-1]["request"]
            self.assertEqual((headers[b"x-reasoning-effort"], sent["chat_template_kwargs"], "reasoning_effort" in sent),
                             (b"ultra -> max; 0 dropped (chat_template_kwargs has its own)", {"reasoning_effort": "max"}, False))
            self.assertEqual(self.mw.effort.describe()["coerced"]["medium -> high"], 3)
        asyncio.run(scenario())

    def test_what_needs_no_translation_carries_no_such_header(self):
        async def scenario():
            self.make(effort=seven_steps())
            for body in ({"name": "c"}, {"name": "c", "reasoning_effort": "high"}, {"name": "c", "chat_template_kwargs": {"reasoning_effort": 40}}):
                _, headers, _ = await call(self.mw, self.app_obj, CHAT, dict(body))
                sent = {k: v for k, v in self.behind.seen[-1]["request"].items() if k != "cache_salt"}
                self.assertEqual((b"x-reasoning-effort" in headers, sent), (False, body))
        asyncio.run(scenario())

    def test_only_the_chat_route_is_translated(self):
        async def scenario():
            self.make(effort=seven_steps())
            for path in ("/v1/completions", "/v1/responses", "/v1/messages", "/free/v1/completions"):
                _, headers, _ = await call(self.mw, self.app_obj, path, {"name": "c", "reasoning_effort": "medium"})
                self.assertEqual((b"x-reasoning-effort" in headers, self.behind.seen[-1]["request"]["reasoning_effort"]), (False, "medium"), path)
        asyncio.run(scenario())

    def test_without_it_nothing_is_translated(self):
        async def scenario():
            self.make()
            _, headers, _ = await call(self.mw, self.app_obj, CHAT, {"name": "c", "reasoning_effort": "medium"})
            self.assertEqual((b"x-reasoning-effort" in headers, self.behind.seen[-1]["request"]["reasoning_effort"]), (False, "medium"))
        asyncio.run(scenario())

    def test_the_model_list_says_what_is_accepted_on_both_addresses(self):
        async def scenario():
            self.make(effort=seven_steps())
            listing = json.dumps({"object": "list", "data": [{"id": "m"}, {"id": "m-free"}]}).encode()
            self.behind.models = (200, [(b"content-type", b"application/json"), (b"content-length", str(len(listing)).encode())],
                                  [listing[:20], listing[20:]])                              # sent in two pieces
            for path in ("/v1/models", "/free/v1/models"):
                status, headers, body = await call(self.mw, self.app_obj, path, method="GET")
                got = json.loads(body)
                self.assertEqual((status, int(headers[b"content-length"]), [m["id"] for m in got["data"]]), (200, len(body), ["m", "m-free"]), path)
                for model in got["data"]:
                    self.assertEqual((model["reasoning_efforts"], model["reasoning_effort_integer_max"]), (["none", "low", "high", "xhigh", "max"], 100))
        asyncio.run(scenario())

    def test_a_model_list_that_is_not_one_goes_out_as_it_came(self):
        async def scenario():
            self.make(effort=seven_steps())
            for status, extra, body in ((500, [], b'{"error": "down"}'), (200, [], b"not json"), (200, [(b"content-encoding", b"gzip")], b'{"data": []}'),
                                        (200, [], b'{"data": "x"}')):
                headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), *extra]
                self.behind.models = (status, headers, [body])
                got_status, got_headers, got_body = await call(self.mw, self.app_obj, "/v1/models", method="GET")
                self.assertEqual((got_status, got_body, got_headers), (status, body, dict(headers)), body)
            self.make()                                                                     # and without the translation: untouched
            self.behind.models = (200, [(b"content-length", b"26")], [b'{"data": [{"id": "plain"}]}'])
            self.assertEqual((await call(self.mw, self.app_obj, "/v1/models", method="GET"))[2], b'{"data": [{"id": "plain"}]}')
        asyncio.run(scenario())


class WhatPasses(GateCase):
    def test_every_request_that_runs_the_model_is_marked_with_its_side(self):
        async def scenario():
            self.make()
            status, headers, _ = await call(self.mw, self.app_obj, FREE_CHAT, {"name": "f", "cache_salt": "mine", "n": 1})
            self.assertEqual((status, headers[b"x-overlay-side"]), (200, b"overlay"))
            await call(self.mw, self.app_obj, CHAT, {"name": "b"})
            f, b = self.behind.seen
            mine = hashlib.sha256(b"mine").hexdigest()[:32]
            self.assertEqual((f["path"], f["request"]["cache_salt"], f["request"]["n"]), (CHAT, f"overlay-gate:overlay:{mine}", 1))
            self.assertEqual((b["path"], b["request"]["cache_salt"]), (CHAT, "overlay-gate:base"))
            for e in (f, b):
                self.assertEqual(int(e["length"]), len(e["body"]))
        asyncio.run(scenario())

    def test_what_is_refused_and_what_passes_untouched(self):
        async def scenario():
            self.make()
            for raw in (b"not json", b"[1, 2]", b""):
                status, _, body = await call(self.mw, self.app_obj, CHAT, raw=raw)
                self.assertEqual((status, "JSON object" in json.loads(body)["error"]["message"]), (400, True), raw)
            for path in ("/v1/embeddings", "/free/v1/embeddings", "/free/overlay/switch", "/collective_rpc", "/pause", "/free"):
                status, _, body = await call(self.mw, self.app_obj, path, {"name": "x"})
                self.assertEqual((status, "not served through the overlay gate" in json.loads(body)["error"]["message"]), (403, True), path)
            self.assertEqual((self.behind.seen, self.engine.calls), ([], []))             # none of it reached anything
            # what does not run the model passes, under the prefix with the prefix taken off, and asks the gate nothing
            for path, method, inner in (("/v1/models", "GET", "/v1/models"), ("/free/v1/models", "GET", "/v1/models"),
                                        ("/health", "GET", "/health"), ("/free/tokenize", "POST", "/tokenize"),
                                        ("/v1/messages/count_tokens", "POST", "/v1/messages/count_tokens"),
                                        ("/free/v1/messages/count_tokens", "POST", "/v1/messages/count_tokens"),
                                        ("/overlay/switch", "POST", "/overlay/switch"), ("/overlay/status", "GET", "/overlay/status")):
                status, headers, _ = await call(self.mw, self.app_obj, path, {"name": "x"}, method=method)
                self.assertEqual((status, self.behind.seen[-1]["path"], b"x-overlay-side" in headers), (200, inner, False), path)
            self.assertEqual((self.engine.calls, hasattr(self.app_obj.state, "overlay_gate")), ([], False))
        asyncio.run(scenario())

    def test_any_salt_the_engine_would_take_still_fits_with_the_side_in_front(self):
        async def scenario():
            self.make()
            longest = "k" * 128                                  # the engine's own limit; with the side in front it must still pass
            await call(self.mw, self.app_obj, FREE_CHAT, {"name": "f", "cache_salt": longest})
            await call(self.mw, self.app_obj, FREE_CHAT, {"name": "g", "cache_salt": longest[:-1] + "j"})
            await call(self.mw, self.app_obj, CHAT, {"name": "b", "cache_salt": longest})
            salts = [e["request"]["cache_salt"] for e in self.behind.seen]
            self.assertEqual(len(set(salts)), 3)                 # another salt or another side: another cache
            for salt in salts:
                self.assertTrue(len(salt) <= 128 and not set(salt) & set("@/\\\0"), salt)
            for bad in ("", 5, ["x"], {"a": 1}, False):
                status, _, body = await call(self.mw, self.app_obj, CHAT, {"name": "x", "cache_salt": bad})
                self.assertEqual((status, "cache_salt" in json.loads(body)["error"]["message"]), (400, True), bad)
            self.assertEqual(len(self.behind.seen), 3)
        asyncio.run(scenario())

    def test_a_request_that_would_run_after_its_answer_is_refused(self):
        async def scenario():
            self.make()
            for path in ("/v1/responses", "/free/v1/responses"):
                status, _, body = await call(self.mw, self.app_obj, path, {"name": "r", "background": True, "store": True})
                self.assertEqual((status, "background" in json.loads(body)["error"]["message"]), (400, True), path)
            self.assertEqual((self.behind.seen, self.engine.calls), ([], []))
            self.assertEqual((await call(self.mw, self.app_obj, "/v1/responses", {"name": "r", "background": False}))[0], 200)
        asyncio.run(scenario())

    def test_a_body_sent_in_pieces_arrives_whole_with_its_length(self):
        async def scenario():
            self.make()
            pieces = [b'{"name": "c", ', b'"messages": []}']
            scope = {"type": "http", "method": "POST", "path": FREE_CHAT, "raw_path": FREE_CHAT.encode(), "app": self.app_obj,
                     "headers": [(b"content-type", b"application/json"), (b"transfer-encoding", b"chunked")]}
            coming = [{"type": "http.request", "body": pieces[0], "more_body": True},
                      {"type": "http.request", "body": pieces[1], "more_body": False}]
            sent = []

            async def receive():
                return coming.pop(0) if coming else await asyncio.sleep(3600)

            async def send(message):
                sent.append(message)

            await self.mw(scope, receive, send)
            entry = self.behind.seen[0]
            self.assertEqual((entry["request"]["name"], entry["request"]["cache_salt"], int(entry["length"])),
                             ("c", "overlay-gate:overlay", len(entry["body"])))
            self.assertNotIn(b"transfer-encoding", dict(self.behind.headers[0]))
        asyncio.run(scenario())

    def test_a_websocket_is_closed(self):
        async def scenario():
            self.make()
            sent = []

            async def send(message):
                sent.append(message)

            await self.mw({"type": "websocket", "path": "/v1/realtime", "app": self.app_obj}, None, send)
            self.assertEqual((sent, self.behind.seen), ([{"type": "websocket.close", "code": 1008}], []))
        asyncio.run(scenario())

    def test_a_prefix_that_would_swallow_the_real_paths_is_refused(self):
        app = types.SimpleNamespace(state=types.SimpleNamespace(), add_middleware=lambda *a, **k: None)
        for bad in ("/", "", "/v1", "v1/", "/overlay"):
            with self.assertRaises(ValueError, msg=bad):
                gating.install(app, bad)
        gating.install(app, "free/")
        self.assertEqual(app.state.overlay_gate_prefix, "/free")


if __name__ == "__main__":
    unittest.main()
