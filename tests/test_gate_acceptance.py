"""The gate's acceptance tool judged against a stand-in server whose behaviour is known. Standard library only.

The stand-in answers on the usual paths like the base set and under /free like the overlay, with
a wobble on every request. On half of the prompts the two sides weigh the third token six logits
apart; on the other half alike. It keeps a prefix cache by side and salt, counts what it reuses
on /metrics, and counts its flips on /overlay/status. It can be told to go wrong in each of the
ways the tool exists to catch: an answer from the other side's weights lands in the middle of the
other side's answers, a "hybrid" one halfway between the two.

    python3 -m unittest tests.test_gate_acceptance -v
"""

import contextlib
import http.server
import io
import json
import os
import random
import sys
import tempfile
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import acceptance  # noqa: E402
import gate_acceptance  # noqa: E402

METRICS = """# HELP vllm:prefix_cache_queries_total Prefix cache queries, in terms of number of queried tokens.
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{{engine="0",model_name="m"}} {queried:e}
vllm:prefix_cache_queries_created{{engine="0",model_name="m"}} 1.7946795748284824e+09
vllm:prefix_cache_hits_total{{engine="0",model_name="m"}} {hit:e}
vllm:external_prefix_cache_hits_total{{engine="0",model_name="m"}} 77.0
vllm:request_success_total{{engine="0",finished_reason="stop",model_name="m"}} {stopped}
vllm:request_success_total{{engine="0",finished_reason="length",model_name="m"}} {length}
"""


class StandIn(http.server.BaseHTTPRequestHandler):
    b = {}
    lock = threading.Lock()

    def log_message(self, *args):
        pass

    def answer(self, status, body, headers=(), kind="application/json"):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(raw)))
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        b, path = self.b, self.path.removeprefix("/free")
        if path == "/v1/models":
            return self.answer(200, {"data": [{"id": "m"}]})
        if path == "/metrics" and not b.get("no_metrics"):
            with self.lock:
                text = METRICS.format(queried=float(b["queried"]), hit=float(b["hit"]), stopped=float(b["finished"]), length=0.0)
            return self.answer(200, text.encode(), kind="text/plain")
        if path == "/overlay/status":
            with self.lock:
                flips = b["flips"] // 2 if b.get("lazy") else b["flips"]
            return self.answer(200, {"ok": True, "gate": {"prefix": "/free", "live": b["live"], "flips": flips, "broken": None}})
        return self.answer(404, {"detail": "Not Found"})

    def do_POST(self):
        b = self.b
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path not in ("/v1/chat/completions", "/free/v1/chat/completions"):
            return self.answer(404, {"detail": "Not Found"})
        asked = "overlay" if self.path.startswith("/free/") else "base"
        prompt = body["messages"][0]["content"]
        tag = body["cache_salt"].split("-")[1]                 # ref, turn, burst, cache or rest: which part of the test asks
        tokens = len(prompt.split()) + 4
        with self.lock:
            b["count"] += 1
            if b["live"] != asked:
                b["flips"] += 1
                b["live"] = asked
                if b.get("flip_clears"):
                    b["cache"].clear()
            odd_one = tag in ("turn", "burst") and asked == "overlay" and b["count"] % 5 == 0
            leak, hybrid = b.get("leaks") and odd_one, b.get("hybrid") and odd_one
            noise = 0.0 if leak or hybrid else b["rng"].uniform(-0.6, 0.6)
            key = (None if b.get("shared_cache") else asked, body["cache_salt"], prompt)
            reused = 0 if key not in b["cache"] or b.get("no_reuse") else (tokens - 1) // 16 * 16
            b["cache"].add(key)
            b["queried"] += tokens
            b["hit"] += reused
            b["finished"] += 1
            if b.get("others", 0) > 0 and tag == "cache":      # somebody else's request finishes in the same moment
                b["others"] -= 1
                b["finished"] += 1
        weights = "base" if (leak or b.get("no_difference")) else asked
        index = acceptance.PROMPTS.index(prompt) if prompt in acceptance.PROMPTS else 1
        margin = (3.0 if weights == "base" else -3.0) if index % 2 == 0 else 0.2
        margin = 1.0 if hybrid else margin + noise
        contenders = sorted(((" colours", -1.0), (" colors", -1.0 - margin)), key=lambda c: -c[1])
        out = [{"token": w, "logprob": -0.1, "top_logprobs": [{"token": w, "logprob": -0.1}]} for w in ("The", " answer")]
        out.append({"token": contenders[0][0], "logprob": contenders[0][1],
                    "top_logprobs": [{"token": t, "logprob": lp} for t, lp in contenders]})
        self.answer(200, {"choices": [{"message": {"content": "x"}, "logprobs": {"content": out}}], "usage": {"prompt_tokens": tokens}},
                    () if b.get("unmarked") else (("x-overlay-side", "base" if b.get("mismarked") else asked),))


class Tool(unittest.TestCase):
    def setUp(self):
        StandIn.b = {"count": 0, "rng": random.Random(1017), "live": None, "flips": 0, "cache": set(),
                     "queried": 1961386, "hit": 1851392, "finished": 43}
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), StandIn)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def run_tool(self, *extra: str, **behaviour) -> tuple[int, str]:
        StandIn.b.update(behaviour)
        argv, out = sys.argv, io.StringIO()
        sys.argv = ["gate_acceptance.py", self.url, "m", "--passes", "4", "--mixed", "2", "--burst", "1", "--copies", "1",
                    "--counter-wait", "0.3", *extra]
        try:
            with contextlib.redirect_stdout(out):
                try:
                    status = gate_acceptance.main()
                except acceptance.CouldNotRun as e:
                    print(f"could not run: {e}")
                    status = 3
        finally:
            sys.argv = argv
        return status, out.getvalue()

    def test_a_gate_that_keeps_the_sides_apart_passes(self):
        status, text = self.run_tool("--status", self.url)
        self.assertEqual(status, 0, text)
        self.assertIn("PASS: all 88 answers carry their side; the 24 mixed answers that reached a landmark all lie with their "
                      "side, on 4 prompts; each side reused only its own cached prefix, also after two flips.", text)
        self.assertIn("0 read like the other side, 0 like neither", text)
        self.assertIn("the gate counted 16 flips for the 16 requests asked in turn", text)
        self.assertIn("the cache: apart: of 524 tokens: base again reused 512, the overlay's first request 0, "
                      "the overlay again 512, base after two flips 512", text)
        # 2 endpoints x 4 passes x 8 prompts; 4 prompts with a landmark: 2 rounds in turn, 1 burst; 5 for the cache; 1 to rest on base
        self.assertEqual(StandIn.b["count"], 64 + 16 + 8 + 5 + 1)
        self.assertEqual(StandIn.b["live"], "base")

    def test_one_request_in_five_on_the_wrong_weights_fails_and_is_shown(self):
        status, text = self.run_tool(leaks=True)
        self.assertEqual(status, 1, text)
        self.assertIn("ON THE WRONG SIDE: asked on overlay", text)
        self.assertIn("answered by the wrong weights", text)

    def test_an_answer_that_reads_like_neither_side_is_not_passed(self):
        status, text = self.run_tool(hybrid=True)
        self.assertEqual(status, 2, text)
        self.assertIn("between the two: asked on overlay", text)
        self.assertIn("mixed answers read like neither side", text)
        self.assertNotIn("ON THE WRONG SIDE", text)

    def test_answers_without_their_side_or_with_the_wrong_one_fail(self):
        for fault in ("unmarked", "mismarked"):
            with self.subTest(fault=fault):
                self.setUp() if StandIn.b["count"] else None
                status, text = self.run_tool(**{fault: True})
                self.assertEqual(status, 1, text)
                self.assertIn("were not marked with the side they were asked on", text)

    def test_two_sides_that_do_not_differ_cannot_be_told(self):
        status, text = self.run_tool(no_difference=True)
        self.assertEqual(status, 2, text)
        self.assertIn("CANNOT TELL: landmarks on 0 prompts", text)
        self.assertEqual(StandIn.b["count"], 64 + 5 + 1)        # nothing to ask in turn where nothing tells the sides apart

    def test_a_cache_the_two_sides_share_fails(self):
        status, text = self.run_tool(shared_cache=True)
        self.assertEqual(status, 1, text)
        self.assertIn("the first request on the overlay reused 512 tokens that the base side had computed", text)
        self.assertIn("FAIL: the two sides share cached prefixes", text)

    def test_a_flip_that_clears_the_cache_fails(self):
        status, text = self.run_tool(flip_clears=True)
        self.assertEqual(status, 1, text)
        self.assertIn("FAIL: a flip cleared the cache", text)

    def test_fewer_flips_than_changes_of_side_fails(self):
        status, text = self.run_tool("--status", self.url, lazy=True)
        self.assertEqual(status, 1, text)
        self.assertIn("the gate counted only", text)

    def test_other_traffic_during_the_cache_check_is_noticed_and_the_check_repeated(self):
        status, text = self.run_tool(others=2)                    # two foreign requests: the first attempt proves nothing
        self.assertEqual(status, 0, text)
        self.assertEqual(StandIn.b["count"], 64 + 16 + 8 + 10 + 1)
        self.setUp()
        status, text = self.run_tool(others=99)                   # traffic throughout: no clean attempt
        self.assertEqual(status, 2, text)
        self.assertIn("CANNOT TELL: the answers lie with their sides, but the cache could not be checked", text)
        self.assertIn("in 3 attempts", text)

    def test_a_server_without_the_counters_or_without_a_cache(self):
        status, text = self.run_tool(no_metrics=True)
        self.assertEqual(status, 2, text)
        self.assertIn("does not give the prefix cache counters", text)
        self.setUp()
        status, text = self.run_tool("--no-cache-check", no_metrics=True)
        self.assertEqual(status, 0, text)
        self.assertIn("the cache was not looked at", text)
        self.setUp()
        status, text = self.run_tool(no_reuse=True)
        self.assertEqual(status, 0, text)
        self.assertIn("this server reuses no prefixes", text)

    def test_a_server_without_the_gate_is_not_asked_anything(self):
        status, text = self.run_tool("--prefix", "/other")
        self.assertEqual(status, 3, text)
        self.assertIn("is the gate installed under /other?", text)
        self.assertEqual(StandIn.b["count"], 0)

    def test_the_report_holds_every_answer_and_landmark(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gate.json")
            self.assertEqual(self.run_tool("--out", path)[0], 0)
            with open(path) as f:
                report = json.load(f)
        self.assertEqual((len(report["reference"]), len(report["mixed"]), report["prefix"], report["placed"], report["unplaced"]),
                         (64, 24, "/free", 24, 0))
        mark = report["landmarks"][0]["marks"][0]
        self.assertEqual((mark["position"], mark["read"], mark["base_is_high"], len(mark["mixed"]["base"]), len(mark["mixed"]["overlay"])),
                         (2, ["The", " answer"], True, 3, 3))
        self.assertGreaterEqual(mark["gap"], 2.0)
        self.assertEqual((report["cache"]["verdict"], [s["hit"] for s in report["cache"]["attempts"][0]]), ("apart", [0, 512, 0, 512, 512]))


class Landmarks(unittest.TestCase):
    @staticmethod
    def answer(asked, margin, read=("a",)):
        top = sorted((["x", -1.0], ["y", -1.0 - margin]), key=lambda c: -c[1])
        tokens = [{"token": t, "logprob": -0.1, "top": [[t, -0.1]]} for t in read]
        return {"asked": asked, "index": 0, "phase": "turn", "text": "", "tokens": tokens + [{"token": top[0][0], "logprob": top[0][1], "top": top}]}

    def marks(self, base, over, min_gap=2.0):                 # the tests name their gap; the tool's default is wider
        reference = [self.answer("base", m) for m in base] + [self.answer("overlay", m) for m in over]
        return gate_acceptance.find_landmarks(reference, min_gap)

    def laid(self, found, base, over):
        """(read like the other side, read like neither), each as (asked, margin)."""
        mixed = [self.answer("base", m) for m in base] + [self.answer("overlay", m) for m in over]
        strays, between, _, _ = gate_acceptance.judge(mixed, {0: found})
        return [(s["asked"], s["margin"]) for s in strays], [(s["asked"], s["margin"]) for s in between]

    def test_own_side_between_and_the_other_group_whichever_side_is_the_higher(self):
        looked, found = self.marks([6.0, 9.0], [1.0, 2.0])
        self.assertEqual((looked, len(found)), (1, 1))            # where all chose one token and saw no other, nothing to weigh
        mark = found[0]
        self.assertEqual((mark["position"], mark["middle"], mark["gap"], mark["base_is_high"], mark["edge"]),
                         (1, 4.0, 4.0, True, {"base": 6.0, "overlay": 2.0}))
        # base: 4.1 is its own side; 4.0 (the middle itself) and 2.1 are between; 2.0 and 0.5 are where the overlay answers lie
        self.assertEqual(self.laid(found, [6.5, 4.1, 4.0, 2.1, 2.0, 0.5], [1.5, 3.9, 4.0, 5.9, 6.0, 9.5]),
                         ([("base", 2.0), ("base", 0.5), ("overlay", 6.0), ("overlay", 9.5)],
                          [("base", 4.0), ("base", 2.1), ("overlay", 4.0), ("overlay", 5.9)]))
        _, found = self.marks([1.0, 2.0], [6.0, 9.0])
        self.assertEqual((found[0]["middle"], found[0]["base_is_high"], found[0]["edge"]), (4.0, False, {"base": 2.0, "overlay": 6.0}))
        self.assertEqual(self.laid(found, [1.5, 3.9, 4.0, 5.9, 6.0, 9.5], [6.5, 4.1, 4.0, 2.1, 2.0, 0.5]),
                         ([("base", 6.0), ("base", 9.5), ("overlay", 2.0), ("overlay", 0.5)],
                          [("base", 4.0), ("base", 5.9), ("overlay", 4.0), ("overlay", 2.1)]))

    def test_a_gap_that_is_too_small_or_no_gap_at_all_is_no_landmark(self):
        self.assertEqual(self.marks([3.0, 3.5], [1.5, 2.0])[1], [])
        self.assertEqual(len(self.marks([3.0, 3.5], [1.5, 2.0], min_gap=1.0)[1]), 1)
        self.assertEqual(self.marks([3.0, 1.0], [2.0, 2.5], min_gap=0.1)[1], [])

    def test_an_answer_that_had_read_other_words_by_then_is_not_laid_against_the_landmark(self):
        _, found = self.marks([6.0, 7.0], [1.0, 2.0])
        elsewhere = self.answer("base", 1.5, read=("b",))        # it would be far on the wrong side, had it read the same words
        short = {**self.answer("base", 1.5), "tokens": self.answer("base", 1.5)["tokens"][:1]}
        strays, between, placed, unplaced = gate_acceptance.judge([elsewhere, short, self.answer("base", 6.5)], {0: found})
        self.assertEqual((strays, between, placed, unplaced, found[0]["mixed"]), ([], [], 1, 2, {"base": [6.5], "overlay": []}))


if __name__ == "__main__":
    unittest.main()
