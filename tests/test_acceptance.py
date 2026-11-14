"""The acceptance tool judged against a stand-in server whose behaviour is known. Standard library only.

The stand-in answers like a model with two sets and a wobble: every request gets its own small
noise. On half of the prompts the two sets weigh the third token four logits apart; on the other
half they weigh it alike, so only the noise decides. It reports a digest of its "derived tensors"
like the switch does. It can be told to ignore flips, not to come back, to drift, or to be an
older switch without the digest, so that each verdict of the tool is exercised.

    python3 -m unittest tests.test_acceptance -v
"""

import contextlib
import http.server
import importlib.util
import io
import json
import os
import random
import sys
import threading
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("overlay_acceptance", os.path.join(HERE, "..", "tools", "acceptance.py"))
acceptance = importlib.util.module_from_spec(spec)
sys.modules["overlay_acceptance"] = acceptance
spec.loader.exec_module(acceptance)

DIGEST = {"base": "b" * 16, "overlay": "0" * 16}


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

    def answering(self) -> str:
        """Which weights the stand-in model really computes with."""
        b = self.behaviour
        if b.get("ignores_flips"):
            return "base"
        if b.get("does_not_come_back") and b["flips"] >= 1:
            return "overlay"
        return b["active"] or "base"

    def do_GET(self):
        b = self.behaviour
        if self.path == "/v1/models":
            return self.send(200, {"data": [{"id": "m"}]})
        if self.path.startswith("/overlay/status"):
            ranks = []
            if "full=1" in self.path and not b.get("no_digest"):
                # the digest follows the weights the model computes with, except for a model that ignores flips:
                # there the switch wrote, and only the kernels did not follow
                seen = (b["active"] or "base") if b.get("ignores_flips") else self.answering()
                ranks = [{"rank": r, "derived": DIGEST[seen][:-1] + str(r)} for r in range(3)]
            return self.send(200, {"ok": True, "active": b["active"], "paused": False, "ranks": ranks})
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
            b["salts"].append(body.get("cache_salt"))
            index = acceptance.PROMPTS.index(body["messages"][0]["content"])
            margin = (2.0 if self.answering() == "base" else -2.0) if index % 2 == 0 else 0.2
            if b.get("drifts") and index == 1 and b["flips"] >= 2:
                margin += 3.0                                   # the same weights, other numbers: a drift between readings
            margin += b["rng"].uniform(-0.6, 0.6)
            contenders = sorted(((" colours", -1.0), (" colors", -1.0 - margin)), key=lambda c: -c[1])
            tokens = [{"token": w, "logprob": -0.1, "top_logprobs": [{"token": w, "logprob": -0.1}]} for w in ("The", " answer")]
            tokens.append({"token": contenders[0][0], "logprob": contenders[0][1],
                           "top_logprobs": [{"token": t, "logprob": lp} for t, lp in contenders]})
            tokens.append({"token": " ends", "logprob": -0.1, "top_logprobs": [{"token": " ends", "logprob": -0.1}]})
            return self.send(200, {"choices": [{"message": {"content": "".join(t["token"] for t in tokens)},
                                                "logprobs": {"content": tokens}}]})
        self.send(404, {})


class Tool(unittest.TestCase):
    def setUp(self):
        StandIn.behaviour = {"active": None, "flips": 0, "rng": random.Random(1017), "salts": []}
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
        self.assertIn("PASS: 4 prompts tell the two sets apart", text)
        self.assertIn("derived tensors, start against end: equal on every rank (3 ranks)", text)
        self.assertIn("byte for byte what they were", text)
        self.assertEqual((StandIn.behaviour["active"], StandIn.behaviour["flips"]), ("base", 2))

    def test_it_starts_from_whichever_set_is_live(self):
        StandIn.behaviour["active"] = "overlay"
        status, text = self.run_tool()
        self.assertEqual(status, 0, text)
        self.assertIn("A is 'overlay' (live now), B is 'base'", text)
        self.assertEqual(StandIn.behaviour["active"], "overlay")

    def test_every_request_is_its_own_computation_unless_told_otherwise(self):
        self.assertEqual(self.run_tool()[0], 0)
        salts = StandIn.behaviour["salts"]
        self.assertEqual(len(salts), 3 * 6 * len(acceptance.PROMPTS))
        self.assertEqual(len(set(salts)), len(salts))
        self.assertNotIn(None, salts)
        StandIn.behaviour.update(active=None, flips=0, salts=[])
        self.assertEqual(self.run_tool("--no-cache-salt")[0], 0)
        self.assertEqual(set(StandIn.behaviour["salts"]), {None})

    def test_a_flip_the_model_does_not_follow_cannot_be_told_unless_parting_is_expected(self):
        StandIn.behaviour["ignores_flips"] = True
        status, text = self.run_tool()
        self.assertEqual(status, 2, text)
        self.assertIn("CANNOT TELL: 0 prompts", text)
        StandIn.behaviour.update(active=None, flips=0)
        status, text = self.run_tool("--expect-parting")
        self.assertEqual(status, 1, text)
        self.assertIn("not reading what was written", text)

    def test_a_model_that_does_not_come_back_fails_on_the_digest(self):
        StandIn.behaviour["does_not_come_back"] = True
        status, text = self.run_tool()
        self.assertEqual(status, 1, text)
        self.assertIn("DIFFERENT on rank 0, 1, 2", text)
        self.assertIn("are not what they were when the test began", text)

    def test_without_a_digest_a_model_that_does_not_come_back_fails_on_its_answers(self):
        StandIn.behaviour.update(does_not_come_back=True, no_digest=True)
        status, text = self.run_tool()
        self.assertEqual(status, 1, text)
        self.assertIn("does not report a digest", text)
        self.assertIn("A stands apart from A before on 4 prompt(s)", text)
        self.assertIn("one time in 462 per prompt", text)

    def test_without_a_digest_a_switch_that_works_still_passes(self):
        StandIn.behaviour["no_digest"] = True
        status, text = self.run_tool()
        self.assertEqual(status, 0, text)
        self.assertIn("A after the return sits with A before", text)

    def test_a_drift_between_readings_is_named_and_does_not_fail_when_the_digest_is_equal(self):
        StandIn.behaviour["drifts"] = True
        status, text = self.run_tool()
        self.assertEqual(status, 0, text)
        self.assertIn("note: A after the return stands apart from A before on 1 prompt(s)", text)
        self.assertIn("PASS: 4 prompts", text)
        # the same drift on a switch without the digest cannot be told from a failed return, and fails
        StandIn.behaviour.update(active=None, flips=0, no_digest=True)
        status, text = self.run_tool()
        self.assertEqual(status, 1, text)

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
        self.assertEqual(StandIn.behaviour["salts"], [])

    def test_too_few_passes_are_refused_before_anything_is_asked(self):
        status, text = self.run_tool("--passes", "3")
        self.assertEqual(status, 3, text)
        self.assertIn("at least 4 passes", text)
        self.assertEqual((StandIn.behaviour["salts"], StandIn.behaviour["flips"]), ([], 0))

    def test_a_reading_alone_touches_no_switch(self):
        import tempfile

        StandIn.behaviour["switch_fails"] = True            # any call to the switch would show
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "reference.json")
            status, text = self.run_tool("--read-only", "--out", path, "--passes", "4")
            self.assertEqual(status, 0, text)
            with open(path) as f:
                report = json.load(f)
        self.assertEqual((sorted(report["readings"]), len(report["readings"]["only"]), report["passes"]), (["only"], 4, 4))
        self.assertEqual((StandIn.behaviour["flips"], StandIn.behaviour["active"]), (0, None))
        self.assertEqual(len(StandIn.behaviour["salts"]), 4 * len(acceptance.PROMPTS))
        self.assertEqual(self.run_tool("--read-only")[0], 3)

    def test_the_report_file_holds_what_was_read(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "report.json")
            self.assertEqual(self.run_tool("--out", path, "--passes", "4")[0], 0)
            with open(path) as f:
                report = json.load(f)
        self.assertEqual((report["A"], report["B"], report["passes"], len(report["flips"]), len(report["verdicts"])),
                         ("base", "overlay", 4, 2, 8))
        self.assertEqual(sorted(report["readings"]), ["A after", "A before", "B"])
        self.assertEqual(report["derived"]["start"], report["derived"]["end"])
        self.assertNotEqual(report["derived"]["start"], report["derived"]["after flip to B"])
        self.assertEqual(len(report["verdicts"][0]["margins"]["B"]), 4)


class Judge(unittest.TestCase):
    @staticmethod
    def reading(*margins, tail=("z",)):
        """One prompt read once per margin: 'a', then 'x' or 'y' by the sign of the margin, then the tail."""
        passes = []
        for m in margins:
            if m is None:                                   # an answer that ends after the shared token
                passes.append([{"prompt": acceptance.PROMPTS[0], "tokens": [{"token": "a", "logprob": -0.1, "top": [["a", -0.1]]}]}])
                continue
            top = sorted((["x", -1.0], ["y", -1.0 - m]), key=lambda c: -c[1])
            tokens = [{"token": "a", "logprob": -0.1, "top": [["a", -0.1]]},
                      {"token": top[0][0], "logprob": top[0][1], "top": top}]
            tokens += [{"token": t, "logprob": -0.1, "top": [[t, -0.1]]} for t in tail]
            passes.append([{"prompt": acceptance.PROMPTS[0], "tokens": tokens}])
        return passes

    def verdict(self, before, other, after):
        return acceptance.judge(0, self.reading(*before), self.reading(*other), self.reading(*after))

    def test_verdicts(self):
        high, low = [2.0, 2.5, 3.0, 2.2], [-2.0, -1.0, -3.0, -1.5]
        clear = self.verdict(high, low, [2.1, 2.9, 3.5, 1.9])
        self.assertEqual((clear["verdict"], clear["parts"], clear["a_moved"], clear["agree_for"]), ("clear", True, False, 1))
        self.assertEqual(clear["margins"]["B"], sorted(low))
        self.assertEqual(self.verdict(high, low, low)["verdict"], "not restored")
        # A before and A after apart from each other, both far from B: still flagged, and still telling the sets apart
        shifted = self.verdict(high, low, [5.0, 6.0, 7.0, 8.0])
        self.assertEqual((shifted["verdict"], shifted["parts"], shifted["a_moved"]), ("not restored", True, True))
        self.assertEqual(self.verdict([0.2, -0.3, 0.5, 0.1], [0.3, -0.2, 0.4, 0.0], [-0.1, 0.6, 0.2, 0.3])["verdict"], "noise")
        self.assertEqual(self.verdict(high, [3.0, 2.0, 2.6, 2.4], high)["verdict"], "same")
        self.assertEqual(self.verdict(high, [None] * 4, high)["verdict"], "unclear")
        # A before apart from B, but A after reaches into B: the return blurred it
        self.assertEqual(self.verdict(high, low, [2.5, -1.2, 2.8, 2.0])["verdict"], "unclear")

    def test_one_stray_answer_is_enough_to_withhold_clear(self):
        v = self.verdict([2.0, 2.5, 3.0, 2.2], [-2.0, -1.0, 2.1, -1.5], [2.1, 2.9, 3.5, 2.4])
        self.assertEqual((v["verdict"], v["parts"]), ("noise", False))

    def test_only_the_first_parting_is_judged(self):
        # after the first parting the wording may drift; what follows must not change the verdict
        a = self.reading(2.0, 2.5, 3.0, 2.2, tail=("p", "q"))
        drifted = self.reading(2.1, 2.4, 2.9, 2.6, tail=("p", "zzz"))
        b = self.reading(-2.0, -1.0, -3.0, -1.5, tail=("r", "s"))
        self.assertEqual(acceptance.judge(0, a, b, drifted)["verdict"], "clear")

    def test_a_token_outside_the_returned_list_only_narrows_the_margin(self):
        def one(top, chosen):
            return [[{"prompt": acceptance.PROMPTS[0], "tokens": [{"token": chosen, "logprob": dict(top)[chosen], "top": top}]}]]
        # the list of the second answer does not hold "x" at all: it is taken at that list's floor, -4.0
        a = one([["x", -0.1], ["y", -3.0]], "x") + one([["x", -0.2], ["y", -2.0]], "x")
        b = one([["y", -0.1], ["w", -4.0]], "y") + one([["y", -0.3], ["x", -5.0]], "y")
        v = acceptance.judge(0, a, b, a)
        self.assertEqual(v["contenders"], ["x", "y"])
        self.assertEqual(v["margins"]["B"], [-4.7, -3.9])

    @staticmethod
    def two_token_reading(first_margins, second_margins):
        """One prompt: token "a" with rival "b" at the given margin, then "x" or "y" by the sign of the second margin."""
        passes = []
        for m0, m1 in zip(first_margins, second_margins):
            top0 = [["a", -0.1], ["b", -0.1 - m0]]
            top1 = sorted((["x", -1.0], ["y", -1.0 - m1]), key=lambda c: -c[1])
            passes.append([{"prompt": acceptance.PROMPTS[0], "tokens": [
                {"token": "a", "logprob": -0.1, "top": top0},
                {"token": top1[0][0], "logprob": top1[0][1], "top": top1}]}])
        return passes

    def test_a_shared_position_can_tell_the_sets_apart_where_the_parting_cannot(self):
        # everyone chooses "a" first, but the two sets weigh its rival differently; the next token is a coin toss
        toss = [0.3, -0.2, 0.4, -0.1]
        v = acceptance.judge(0, self.two_token_reading([3.0, 3.2, 2.9, 3.4], toss),
                             self.two_token_reading([0.5, 0.8, 0.2, 0.6], toss),
                             self.two_token_reading([3.1, 2.8, 3.3, 3.0], toss))
        self.assertEqual((v["verdict"], v["parts"], v["position"], v["contenders"], v["positions"]), ("clear", True, 0, ["a", "b"], 2))
        self.assertEqual(v["margins"]["B"], [0.2, 0.5, 0.6, 0.8])
        self.assertAlmostEqual(v["gap"], 2.0)
        self.assertIn("at token 0", v["note"])

    def test_identical_answers_can_still_tell_the_sets_apart_by_their_margins(self):
        same = [2.0, 2.1, 2.2, 2.3]                         # "x" every time: the wording never parts
        v = acceptance.judge(0, self.two_token_reading([3.0, 3.2, 2.9, 3.4], same),
                             self.two_token_reading([0.5, 0.8, 0.2, 0.6], same),
                             self.two_token_reading([3.1, 2.8, 3.3, 3.0], same))
        self.assertEqual((v["verdict"], v["agree_for"], v["position"]), ("clear", 2, 0))
        flat = acceptance.judge(0, self.two_token_reading(same, same), self.two_token_reading(same, same),
                                self.two_token_reading(same, same))
        self.assertEqual((flat["verdict"], flat["parts"]), ("same", False))

    def test_the_return_is_judged_where_the_answers_part_and_nowhere_else(self):
        # at token 0 A after stands apart from A before, but nobody parts there: that alone is not "not restored"
        toss = [0.3, -0.2, 0.4, -0.1]
        v = acceptance.judge(0, self.two_token_reading([3.0, 3.2, 2.9, 3.4], toss),
                             self.two_token_reading([3.1, 3.0, 3.3, 2.8], toss),
                             self.two_token_reading([6.0, 6.2, 6.1, 6.4], toss))
        self.assertEqual((v["a_moved"], v["verdict"]), (False, "noise"))

    def test_the_rival_is_the_token_the_answers_give_most_weight_to(self):
        def one(top):
            return [[{"prompt": acceptance.PROMPTS[0], "tokens": [{"token": "a", "logprob": top[0][1], "top": top}]}]]
        a = one([["a", -0.1], ["b", -3.0], ["c", -6.0]]) + one([["a", -0.1], ["c", -5.0], ["b", -3.5]])
        b = one([["a", -0.2], ["b", -2.0], ["c", -7.0]]) + one([["a", -0.2], ["b", -2.2]])
        v = acceptance.judge(0, a, b, a)
        self.assertEqual((v["contenders"], v["agree_for"], v["positions"]), (["a", "b"], 1, 1))

    def test_chance(self):
        self.assertEqual(round(1 / acceptance.chance_apart(6, 6)), 462)
        self.assertEqual(round(1 / acceptance.chance_apart(2, 2)), 3)
        self.assertTrue(acceptance.apart([1, 2], [3, 4]) and acceptance.apart([3, 4], [1, 2]))
        self.assertFalse(acceptance.apart([1, 3], [3, 4]) or acceptance.apart([], [1]))


if __name__ == "__main__":
    unittest.main()
