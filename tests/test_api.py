"""The two routes, with FastAPI's test client and a stand-in engine. No GPU, no vLLM.

The test client presents itself from the address "testclient", which is not the local
machine. That makes the default refusal testable as it stands, and the allow-list the way in.

    python3 -m unittest tests.test_api -v        (skipped when fastapi or httpx is missing)
"""

import asyncio
import hashlib
import os
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "mod"))

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover
    FastAPI = None

from test_logic import FakeEngine  # noqa: E402

ENV = ("OVERLAY_SWITCH_ALLOW", "OVERLAY_SWITCH_TOKEN", "OVERLAY_GATE_PREFIX", "OVERLAY_GATE_MODELS",
       "OVERLAY_GATE_EFFORT_KEEP", "OVERLAY_GATE_EFFORT_MAP", "OVERLAY_GATE_EFFORT_INT")
AUTH_MODULE = "vllm.entrypoints.serve.middleware.authenticate"


class EngineAuth:
    """Stands in for the engine's API-key middleware: same arguments, same rule (a path that begins
    with one of these needs the key; any other passes)."""

    GUARDED = ("/v1", "/v2", "/inference", "/cohere")

    def __init__(self, app, tokens):
        self.app, self.tokens = app, tokens

    def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") == "OPTIONS":
            return self.app(scope, receive, send)
        given = dict(scope["headers"]).get(b"authorization", b"").decode()
        if scope["path"].startswith(self.GUARDED) and given not in [f"Bearer {t}" for t in self.tokens]:
            from fastapi.responses import JSONResponse

            return JSONResponse({"error": "Unauthorized"}, status_code=401)(scope, receive, send)
        return self.app(scope, receive, send)


def stand_in_engine(env_key=None, with_auth=True):
    """The few engine modules the plugin reaches for, as sys.modules entries. Returns their names."""
    names = ["vllm", "vllm.entrypoints", "vllm.entrypoints.serve", "vllm.entrypoints.serve.middleware"]
    for name in names:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["vllm"].envs = types.SimpleNamespace(VLLM_API_KEY=env_key)
    if with_auth:
        sys.modules[AUTH_MODULE] = types.ModuleType(AUTH_MODULE)
        sys.modules[AUTH_MODULE].AuthenticationMiddleware = EngineAuth
        names.append(AUTH_MODULE)
    return names


class BrokenEngine(FakeEngine):
    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        raise AttributeError(f"'Worker' object has no attribute {method!r}")


@unittest.skipIf(FastAPI is None, "fastapi or httpx is not installed")
class Routes(unittest.TestCase):
    def setUp(self):
        from vllm_overlay_switch import api

        self.api = api
        self.saved = {k: os.environ.pop(k, None) for k in ENV}
        os.environ["OVERLAY_SWITCH_ALLOW"] = "testclient"
        self.engine_modules = stand_in_engine()

    def tearDown(self):
        for key, value in self.saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value
        for name in set(self.engine_modules) | {AUTH_MODULE}:
            sys.modules.pop(name, None)

    def client(self, engine=None, attach_engine=True, key=None, servers=1):
        self.engine = engine or FakeEngine()
        self.app = FastAPI()
        # as the engine sets it, before plugins
        self.app.state.args = types.SimpleNamespace(api_key=[key] if key else None, api_server_count=servers)

        @self.app.post("/v1/chat/completions")           # stands in for the engine's own route
        async def chat(body: dict):
            said = {"effort": body.get("reasoning_effort"), "kwargs": body.get("chat_template_kwargs")}
            return {"salt": body.get("cache_salt"), "held": list(self.engine.active), **{k: v for k, v in said.items() if v is not None}}

        @self.app.get("/v1/models")
        async def models():
            return {"object": "list", "data": [{"id": "m"}, {"id": "m-free"}]}

        plugin = self.api.OverlaySwitchEndpointPlugin()
        plugin.attach_router(self.app)
        tokens = [k for k in ([key] if key else [sys.modules["vllm"].envs.VLLM_API_KEY]) if k]
        if tokens:
            self.app.add_middleware(EngineAuth, tokens=tokens)                         # the engine adds its own afterwards: outside
        if attach_engine:
            asyncio.run(plugin.init_state(self.engine, self.app.state, None))
        return TestClient(self.app)

    def test_the_plugin_has_the_shape_the_engine_asks_for(self):
        plugin = self.api.OverlaySwitchEndpointPlugin()
        self.assertEqual((plugin.name, tuple(plugin.required_tasks)), ("overlay_switch", ("generate",)))
        self.assertTrue(callable(plugin.attach_router) and asyncio.iscoroutinefunction(plugin.init_state))

    def test_other_addresses_are_refused_unless_listed(self):
        del os.environ["OVERLAY_SWITCH_ALLOW"]
        client = self.client()
        self.assertEqual(client.get("/overlay/status").status_code, 403)
        self.assertEqual(client.post("/overlay/switch", json={"variant": "overlay"}).status_code, 403)
        self.assertEqual(self.engine.calls, [])

    def test_a_token_when_set_must_come_with_every_call(self):
        os.environ["OVERLAY_SWITCH_TOKEN"] = "s3"
        client = self.client()
        self.assertEqual(client.get("/overlay/status").status_code, 403)
        self.assertEqual(client.get("/overlay/status", headers={"X-Overlay-Token": "wrong"}).status_code, 403)
        self.assertEqual(client.get("/overlay/status", headers={"X-Overlay-Token": "s3"}).status_code, 200)

    def test_status_reports_every_rank_and_passes_the_full_look_on(self):
        client = self.client()
        answer = client.get("/overlay/status")
        self.assertEqual(answer.status_code, 200)
        body = answer.json()
        self.assertEqual((body["ok"], body["active"], body["paused"], len(body["ranks"])), (True, "base", False, 3))
        client.get("/overlay/status?full=1")
        self.assertEqual(self.engine.calls, [("overlay_status", False), ("overlay_status", True)])

    def test_status_says_so_when_ranks_disagree(self):
        client = self.client()
        self.engine.active = ["base", "overlay", "base"]
        body = client.get("/overlay/status").json()
        self.assertEqual((body["ok"], body["active"]), (False, None))
        self.assertIn("disagree", body["error"])

    def test_a_flip_through_the_route(self):
        client = self.client()
        answer = client.post("/overlay/switch", json={"variant": "overlay"})
        self.assertEqual(answer.status_code, 200)
        body = answer.json()
        self.assertEqual((body["ok"], body["changed"], body["active"], body["previous"]), (True, True, "overlay", "base"))
        self.assertIn(("pause", "wait", True), self.engine.calls)
        self.assertEqual(self.engine.calls[-1], ("resume",))
        self.assertEqual(client.post("/overlay/switch", json={"variant": "base", "mode": "abort"}).json()["active"], "base")
        self.assertIn(("pause", "abort", True), self.engine.calls)

    def test_force_reaches_the_flip(self):
        client = self.client()
        self.engine.active = ["base", "overlay", "base"]
        self.assertEqual(client.post("/overlay/switch", json={"variant": "base"}).status_code, 409)
        answer = client.post("/overlay/switch", json={"variant": "base", "force": True})
        self.assertEqual((answer.status_code, answer.json()["active"]), (200, "base"))
        self.assertEqual(self.engine.active, ["base", "base", "base"])

    def test_what_is_not_a_proper_request_is_turned_away_before_the_engine(self):
        client = self.client()
        for body in ({"variant": "other"}, {"variant": "overlay", "mode": "keep"}, {"variant": "overlay", "force": "yes"},
                     {}, ["overlay"]):
            with self.subTest(body=body):
                self.assertEqual(client.post("/overlay/switch", json=body).status_code, 400)
        self.assertEqual(client.post("/overlay/switch", content=b"not json").status_code, 400)
        self.assertEqual(self.engine.calls, [])

    def test_a_second_flip_is_refused_while_one_is_under_way(self):
        client = self.client()
        asyncio.run(self.app.state.overlay_switch_lock.acquire())
        answer = client.post("/overlay/switch", json={"variant": "overlay"})
        self.assertEqual(answer.status_code, 409)
        self.assertEqual(self.engine.calls, [])

    def test_a_server_without_an_engine_says_so(self):
        client = self.client(attach_engine=False)
        self.assertEqual(client.get("/overlay/status").status_code, 503)
        self.assertEqual(client.post("/overlay/switch", json={"variant": "overlay"}).status_code, 503)

    def test_an_engine_without_the_worker_half_is_an_error_not_a_crash(self):
        client = self.client(BrokenEngine())
        answer = client.get("/overlay/status")
        self.assertEqual(answer.status_code, 500)
        self.assertIn("overlay_status", answer.json()["error"])
        answer = client.post("/overlay/switch", json={"variant": "overlay"})
        self.assertEqual((answer.status_code, answer.json()["ok"]), (500, False))

    # ---- the gate, as the plugin installs it

    def test_without_the_prefix_there_is_no_gate(self):
        c = self.client()
        self.assertNotIn("gate", c.get("/overlay/status").json())
        self.assertEqual(c.post("/free/v1/chat/completions", json={}).status_code, 404)
        answer = c.post("/v1/chat/completions", json={})
        self.assertEqual((answer.json()["salt"], "x-overlay-side" in answer.headers), (None, False))

    def test_with_the_prefix_one_server_answers_on_two_sets_of_paths(self):
        os.environ["OVERLAY_GATE_PREFIX"] = "/free"
        c = self.client()
        gate = c.get("/overlay/status").json()["gate"]
        self.assertEqual((gate["prefix"], gate["models"], gate["effort"], gate["live"], gate["flips"]), ("/free", [], None, None, 0))
        free = c.post("/free/v1/chat/completions", json={"messages": []})
        self.assertEqual((free.status_code, free.headers["x-overlay-side"]), (200, "overlay"))
        self.assertEqual(free.json(), {"salt": "overlay-gate:overlay", "held": ["overlay"] * 3})
        usual = c.post("/v1/chat/completions", json={"messages": [], "cache_salt": "k"})
        salt = "overlay-gate:base:" + hashlib.sha256(b"k").hexdigest()[:32]
        self.assertEqual((usual.headers["x-overlay-side"], usual.json()), ("base", {"salt": salt, "held": ["base"] * 3}))
        gate = c.get("/overlay/status").json()["gate"]
        self.assertEqual((gate["live"], gate["flips"], gate["in_flight"], gate["waiting"]), ("base", 2, 0, 0))
        self.assertIn(("pause", "abort", False), self.engine.calls)
        self.assertNotIn(("pause", "wait", False), self.engine.calls)
        self.assertNotIn(("reset_prefix_cache",), self.engine.calls)
        self.assertEqual(c.get("/free/overlay/status").status_code, 200)                 # a GET passes with the prefix taken off
        self.assertEqual(c.post("/free/v1/embeddings", json={}).status_code, 403)

    def test_with_names_a_model_on_the_list_chooses_the_overlay_on_the_usual_path(self):
        os.environ["OVERLAY_GATE_PREFIX"] = "/free"
        os.environ["OVERLAY_GATE_MODELS"] = "m-free"
        c = self.client()
        self.assertEqual(c.get("/overlay/status").json()["gate"]["models"], ["m-free"])
        free = c.post("/v1/chat/completions", json={"model": "m-free", "messages": []})
        self.assertEqual((free.headers["x-overlay-side"], free.json()), ("overlay", {"salt": "overlay-gate:overlay", "held": ["overlay"] * 3}))
        usual = c.post("/v1/chat/completions", json={"model": "m", "messages": []})
        self.assertEqual((usual.headers["x-overlay-side"], usual.json()["held"]), ("base", ["base"] * 3))
        self.assertEqual(c.get("/overlay/status").json()["gate"]["flips"], 2)

    def test_with_the_effort_settings_a_chat_request_is_translated_and_the_list_and_the_status_say_how(self):
        os.environ.update({"OVERLAY_GATE_PREFIX": "/free", "OVERLAY_GATE_MODELS": "m-free", "OVERLAY_GATE_EFFORT_KEEP": "none,low,high,xhigh,max",
                           "OVERLAY_GATE_EFFORT_MAP": "minimal=low,medium=high,ultra=max,*=high", "OVERLAY_GATE_EFFORT_INT": "1-100"})
        c = self.client()
        for path, model in (("/v1/chat/completions", "m"), ("/v1/chat/completions", "m-free"), ("/free/v1/chat/completions", "m")):
            for sent, top, kwargs, note in (("medium", "high", None, "medium -> high"), ("minimal", "low", None, "minimal -> low"),
                                            ("ultra", "max", None, "ultra -> max"), ("banana", "high", None, "banana -> high (unknown)"),
                                            (50, None, 50, "50 moved to chat_template_kwargs"), (0, None, 1, "0 -> 1"), (101, None, 100, "101 -> 100")):
                answer = c.post(path, json={"model": model, "messages": [], "reasoning_effort": sent})
                self.assertEqual((answer.status_code, answer.json().get("effort"), (answer.json().get("kwargs") or {}).get("reasoning_effort"),
                                  answer.headers["x-reasoning-effort"]), (200, top, kwargs, note), (path, model, sent))
                inner = c.post(path, json={"model": model, "messages": [], "chat_template_kwargs": {"reasoning_effort": sent}})
                self.assertEqual(inner.json()["kwargs"]["reasoning_effort"], top if top is not None else kwargs, (path, model, sent))
        plain = c.post("/v1/chat/completions", json={"model": "m", "messages": []})
        self.assertEqual(("x-reasoning-effort" in plain.headers, "effort" in plain.json(), "kwargs" in plain.json()), (False, False, False))
        for path in ("/v1/models", "/free/v1/models"):
            listed = c.get(path).json()["data"]
            self.assertEqual([(m["id"], m["reasoning_efforts"], m["reasoning_effort_integer_max"]) for m in listed],
                             [(name, ["none", "low", "high", "xhigh", "max"], 100) for name in ("m", "m-free")])
        effort = c.get("/overlay/status").json()["gate"]["effort"]
        self.assertEqual((effort["keep"], effort["int"], effort["coerced"]["medium -> high"]), (["none", "low", "high", "xhigh", "max"], [1, 100], 6))

    def test_a_mistake_in_the_effort_settings_stops_the_server(self):
        os.environ.update({"OVERLAY_GATE_PREFIX": "/free", "OVERLAY_GATE_EFFORT_KEEP": "low,high", "OVERLAY_GATE_EFFORT_MAP": "medium=hgih"})
        with self.assertRaises(ValueError):
            self.client()

    def test_under_the_gate_only_a_forced_flip_is_taken_by_hand(self):
        os.environ["OVERLAY_GATE_PREFIX"] = "/free"
        c = self.client()
        refused = c.post("/overlay/switch", json={"variant": "overlay"})
        self.assertEqual(refused.status_code, 409)
        self.assertIn("gated", refused.json()["error"])
        self.assertEqual(self.engine.calls, [])
        forced = c.post("/overlay/switch", json={"variant": "overlay", "force": True})
        self.assertEqual((forced.status_code, forced.json()["active"], self.engine.active), (200, "overlay", ["overlay"] * 3))
        self.assertEqual(c.get("/overlay/status").json()["gate"]["live"], "overlay")
        # and the gate goes on from there: the next request on the usual path flips back by itself
        self.assertEqual(c.post("/v1/chat/completions", json={}).json()["held"], ["base"] * 3)

    # ---- the server's API key, which the engine checks by how a path begins

    def both_ways_in(self, c, key=None):
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        return [c.post(path, json={"messages": []}, headers=headers) for path in ("/v1/chat/completions", "/free/v1/chat/completions")]

    def test_with_a_key_the_paths_under_the_prefix_ask_for_it_too(self):
        os.environ["OVERLAY_GATE_PREFIX"] = "/free"
        c = self.client(key="s3cret")
        for given in (None, "wrong"):
            self.assertEqual([a.status_code for a in self.both_ways_in(c, given)], [401, 401], given)
        self.assertEqual(c.get("/free/v1/models").status_code, 401)
        self.assertEqual(self.engine.calls, [])                                        # nothing was flipped for a stranger
        usual, free = self.both_ways_in(c, "s3cret")
        self.assertEqual((usual.status_code, usual.json()["held"], free.status_code, free.json()["held"]),
                         (200, ["base"] * 3, 200, ["overlay"] * 3))

    def test_the_key_from_the_environment_counts_as_the_engine_counts_it(self):
        os.environ["OVERLAY_GATE_PREFIX"] = "/free"
        sys.modules["vllm"].envs.VLLM_API_KEY = "from-env"
        c = self.client()
        self.assertEqual([a.status_code for a in self.both_ways_in(c)], [401, 401])
        self.assertEqual([a.status_code for a in self.both_ways_in(c, "from-env")], [200, 200])

    def test_a_key_whose_check_cannot_be_found_stops_the_server(self):
        os.environ["OVERLAY_GATE_PREFIX"] = "/free"
        del sys.modules[AUTH_MODULE]
        with self.assertRaises(ImportError):
            self.client(key="s3cret")
        self.client()                                                                  # without a key there is nothing to find

    def test_the_gate_refuses_a_server_of_several_processes(self):
        self.client(servers=4)                                                         # without the gate: nobody's business
        os.environ["OVERLAY_GATE_PREFIX"] = "/free"
        with self.assertRaises(RuntimeError) as refused:
            self.client(servers=4)
        self.assertIn("one API server process, not 4", str(refused.exception))
        self.client(servers=None)                                                      # not said: one

    def test_without_the_prefix_the_key_is_the_engines_business_alone(self):
        del sys.modules[AUTH_MODULE]
        c = self.client(key="s3cret")
        self.assertEqual(c.post("/v1/chat/completions", json={}).status_code, 401)
        self.assertEqual(c.post("/v1/chat/completions", json={}, headers={"Authorization": "Bearer s3cret"}).status_code, 200)


if __name__ == "__main__":
    unittest.main()
