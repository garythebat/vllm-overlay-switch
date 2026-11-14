"""The two routes, with FastAPI's test client and a stand-in engine. No GPU, no vLLM.

The test client presents itself from the address "testclient", which is not the local
machine. That makes the default refusal testable as it stands, and the allow-list the way in.

    python3 -m unittest tests.test_api -v        (skipped when fastapi or httpx is missing)
"""

import asyncio
import os
import sys
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

ENV = ("OVERLAY_SWITCH_ALLOW", "OVERLAY_SWITCH_TOKEN")


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

    def tearDown(self):
        for key, value in self.saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value

    def client(self, engine=None, attach_engine=True):
        self.engine = engine or FakeEngine()
        self.app = FastAPI()
        plugin = self.api.OverlaySwitchEndpointPlugin()
        plugin.attach_router(self.app)
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


if __name__ == "__main__":
    unittest.main()
