"""The acceptance tool judged against a stand-in server whose behaviour is known. Standard library only.

The stand-in answers like a model with two sets: on half of the prompts the two sets choose
a different third token, on the other half they answer alike. It can also be told to
ignore flips, or not to come back, so that the tool's two failure verdicts are exercised.

    python3 -m unittest tests.test_acceptance -v
"""

import contextlib
import http.server
import importlib.util
import io
import json
import os
import sys
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("overlay_acceptance", os.path.join(HERE, "..", "tools", "acceptance.py"))
acceptance = importlib.util.module_from_spec(spec)
sys.modules["overlay_acceptance"] = acceptance
spec.loader.exec_module(acceptance)


class StandIn(http.server.BaseHTTPRequestHandler):
    behaviour = {}

    def log_message(self, *args):
        pass

    def send(self, status: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        b = self.behaviour
        if self.path == "/v1/models":
            return self.send(200, {"data": [{"id": "m"}]})
        if self.path.startswith("/overlay/status"):
            return self.send(200, {"ok": True, "active": b["active"], "paused": False, "ranks": []})
        self.send(404, {})

    def do_POST(self):
        b = self.behaviour
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/overlay/switch":
            if b.get("switch_fails") and b["active"] is not None:
                return self.send(500, {"ok": False, "error": "copy failed on rank 1"})
            changed = b["active"] not in (None, body["variant"])
            b["flips"] += changed
            b["active"] = body["variant"]
            return self.send(200, {"ok": True, "changed": changed, "active": b["active"], "timings": {"copy_s": 0.5}})
        if self.path == "/v1/chat/completions":
            answering = b["active"] or "base"
            if b.get("ignores_flips"):
                answering = "base"
            if b.get("does_not_come_back") and b["flips"] >= 1:
                answering = "overlay"
            prompt = body["messages"][0]["content"]
            parts = acceptance.PROMPTS.index(prompt) % 2 == 0
            words = ["The", " answer", " colors" if parts and answering == "overlay" else " colours", " ends"]
            tokens = [{"token": w, "logprob": -0.1, "top_logprobs": [{"token": w, "logprob": -0.1}]} for w in words]
            return self.send(200, {"choices": [{"message": {"content": "".join(words)}, "logprobs": {"content": tokens}}]})
        self.send(404, {})


class Tool(unittest.TestCase):
    def setUp(self):
        StandIn.behaviour = {"active": None, "flips": 0}
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StandIn)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def run_tool(self, *extra: str) -> tuple[int, str]:
        argv, out = sys.argv, io.StringIO()
        sys.argv = ["acceptance.py", self.url, "m", *extra]
        try:
            with contextlib.redirect_stdout(out):
                try:
                    status = acceptance.main()
                except acceptance.Failed as e:
                    print(f"FAIL: {e}")
                    status = 1
                except acceptance.CouldNotRun as e:
                    print(f"could not run: {e}")
                    status = 3
        finally:
            sys.argv = argv
        return status, out.getvalue()

    def test_a_switch_that_works_passes_and_ends_where_it_began(self):
        status, text = self.run_tool()
        self.assertEqual(status, 0, text)
        self.assertIn("PASS: 4 prompts part clearly", text)
        self.assertEqual((StandIn.behaviour["active"], StandIn.behaviour["flips"]), ("base", 2))

    def test_it_starts_from_whichever_set_is_live(self):
        StandIn.behaviour["active"] = "overlay"
        status, text = self.run_tool()
        self.assertEqual(status, 0, text)
        self.assertIn("A is 'overlay' (live now), B is 'base'", text)
        self.assertEqual(StandIn.behaviour["active"], "overlay")

    def test_a_flip_the_model_does_not_follow_cannot_be_told_unless_parting_is_expected(self):
        StandIn.behaviour["ignores_flips"] = True
        self.assertEqual(self.run_tool()[0], 2)
        StandIn.behaviour.update(active=None, flips=0)
        status, text = self.run_tool("--expect-parting")
        self.assertEqual(status, 1, text)
        self.assertIn("not reading what was written", text)

    def test_a_model_that_does_not_come_back_fails(self):
        StandIn.behaviour["does_not_come_back"] = True
        status, text = self.run_tool()
        self.assertEqual(status, 1, text)
        self.assertIn("does not choose what it chose before", text)

    def test_a_flip_the_switch_reports_as_failed_fails(self):
        StandIn.behaviour["switch_fails"] = True
        status, text = self.run_tool()
        self.assertEqual(status, 1, text)
        self.assertIn("copy failed on rank 1", text)

    def test_a_model_the_server_does_not_serve_is_not_asked(self):
        argv = sys.argv
        sys.argv = ["acceptance.py", self.url, "another"]
        try:
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(acceptance.CouldNotRun):
                acceptance.main()
        finally:
            sys.argv = argv

    def test_the_report_file_holds_what_was_read(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "report.json")
            self.assertEqual(self.run_tool("--out", path)[0], 0)
            with open(path) as f:
                report = json.load(f)
        self.assertEqual((report["A"], report["B"], len(report["flips"]), len(report["verdicts"])), ("base", "overlay", 2, 8))
        self.assertEqual(sorted(report["readings"]), ["A after", "A before", "B"])


class Judge(unittest.TestCase):
    @staticmethod
    def reading(*answers):
        """One prompt; each answer is a list of tokens; returns passes shaped like a reading."""
        return [[{"prompt": acceptance.PROMPTS[0], "tokens": [{"token": t} for t in tokens]}] for tokens in answers]

    def verdict(self, before, other, after):
        return acceptance.judge(0, self.reading(*before), self.reading(*other), self.reading(*after))["verdict"]

    def test_verdicts(self):
        x, y = ["a", "x", "c"], ["a", "y", "c"]
        self.assertEqual(self.verdict([x, x], [y, y], [x, x]), "clear")
        self.assertEqual(self.verdict([x, x], [y, y], [y, y]), "not restored")
        self.assertEqual(self.verdict([x, y], [x, y], [y, x]), "noise")
        self.assertEqual(self.verdict([x, x], [x, x], [x, x]), "same")
        self.assertEqual(self.verdict([x, x], [x, y], [x, x]), "unclear")
        self.assertEqual(self.verdict([x, x], [["a"], ["a"]], [x, x]), "unclear")

    def test_only_the_first_parting_is_judged(self):
        # after the first parting the wording may drift; what follows must not change the verdict
        before, other = ["a", "x", "p", "q"], ["a", "y", "r", "s"]
        drifted = ["a", "x", "p", "z"]
        self.assertEqual(self.verdict([before, drifted], [other, other], [before, before]), "clear")


if __name__ == "__main__":
    unittest.main()
