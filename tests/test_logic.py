"""The order of a flip, who may call, and what happens when ranks fail. No GPU, no engine.

    python3 -m unittest tests.test_logic -v
"""

import asyncio
import importlib.util
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "overlay_logic", os.path.join(HERE, "..", "mod", "vllm_overlay_switch", "logic.py"))
logic = importlib.util.module_from_spec(spec)
sys.modules["overlay_logic"] = logic
spec.loader.exec_module(logic)


class FakeEngine:
    """Three ranks. `fail` maps (method, variant) to the rank numbers that fail that call."""

    def __init__(self, active="base", prepared=True, fail=None, pause_raises=False):
        self.active = [active] * 3
        self.prepared = prepared
        self.fail = fail or {}
        self.pause_raises = pause_raises
        self.calls = []
        self.paused = False

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.calls.append((method, *args))
        if method == "overlay_status":
            return [{"ok": True, "rank": r, "prepared": self.prepared, "active": a} for r, a in enumerate(self.active)]
        if method == "overlay_prepare":
            bad = self.fail.get(("overlay_prepare", None), ())
            self.prepared = not bad
            return [{"ok": r not in bad, "rank": r, "prepared": r not in bad, "active": a, "error": "no file" if r in bad else None}
                    for r, a in enumerate(self.active)]
        if method == "overlay_switch":
            (variant,) = args
            bad = self.fail.get(("overlay_switch", variant), ())
            out = []
            for r in range(3):
                if r in bad:
                    out.append({"ok": False, "rank": r, "active": self.active[r], "error": "copy failed"})
                else:
                    self.active[r] = variant
                    out.append({"ok": True, "rank": r, "active": variant})
            return out
        raise AttributeError(method)

    async def pause_generation(self, *, mode="abort", clear_cache=True):
        self.calls.append(("pause", mode, clear_cache))
        if self.pause_raises:
            raise RuntimeError("engine gone")
        self.paused = True

    async def resume_generation(self):
        self.calls.append(("resume",))
        self.paused = False

    async def is_paused(self):
        return self.paused

    async def reset_prefix_cache(self, *a, **k):
        self.calls.append(("reset_prefix_cache",))
        return True


def run(engine, variant, mode="wait", force=False):
    return asyncio.run(logic.flip(engine, variant, mode, force))


class Flip(unittest.TestCase):
    def test_a_flip_pauses_copies_clears_and_resumes_in_that_order(self):
        e = FakeEngine()
        answer, status = run(e, "overlay")
        self.assertEqual((status, answer["ok"], answer["changed"], answer["active"], answer["previous"]),
                         (200, True, True, "overlay", "base"))
        self.assertEqual(e.calls, [("overlay_status",), ("pause", "wait", True), ("overlay_switch", "overlay"),
                                   ("reset_prefix_cache",), ("resume",)])
        self.assertFalse(e.paused)

    def test_asking_for_the_active_set_changes_nothing_and_never_pauses(self):
        e = FakeEngine(active="overlay")
        answer, status = run(e, "overlay")
        self.assertEqual((status, answer["ok"], answer["changed"]), (200, True, False))
        self.assertEqual(e.calls, [("overlay_status",)])

    def test_the_first_flip_prepares_the_alternate_set(self):
        e = FakeEngine(prepared=False)
        answer, status = run(e, "overlay")
        self.assertEqual(status, 200)
        self.assertEqual(e.calls[:2], [("overlay_status",), ("overlay_prepare",)])
        self.assertIn("prepare_s", answer["timings"])

    def test_a_failed_prepare_stops_before_any_pause(self):
        e = FakeEngine(prepared=False, fail={("overlay_prepare", None): (1,)})
        answer, status = run(e, "overlay")
        self.assertEqual((status, answer["ok"]), (500, False))
        self.assertNotIn(("pause", "wait", True), e.calls)

    def test_one_rank_failing_puts_every_rank_back_and_resumes(self):
        e = FakeEngine(fail={("overlay_switch", "overlay"): (2,)})
        answer, status = run(e, "overlay")
        self.assertEqual((status, answer["ok"], answer["changed"], answer["active"]), (500, False, False, "base"))
        self.assertEqual(e.active, ["base", "base", "base"])
        self.assertEqual(e.calls[-3:], [("overlay_switch", "base"), ("reset_prefix_cache",), ("resume",)])
        self.assertFalse(e.paused)

    def test_if_putting_it_back_fails_too_generation_stays_paused(self):
        e = FakeEngine(fail={("overlay_switch", "overlay"): (2,), ("overlay_switch", "base"): (0,)})
        answer, status = run(e, "overlay")
        self.assertEqual((status, answer["ok"], answer["paused"], answer["active"]), (500, False, True, None))
        self.assertIn("PAUSED", answer["error"])
        self.assertNotIn(("resume",), e.calls)
        self.assertTrue(e.paused)

    def test_a_pause_that_raises_changes_nothing(self):
        e = FakeEngine(pause_raises=True)
        answer, status = run(e, "overlay")
        self.assertEqual((status, answer["ok"], answer["changed"], answer["active"]), (500, False, False, "base"))
        self.assertNotIn(("overlay_switch", "overlay"), e.calls)

    def test_ranks_that_already_disagree_are_not_flipped(self):
        e = FakeEngine()
        e.active = ["base", "overlay", "base"]
        answer, status = run(e, "overlay")
        self.assertEqual((status, answer["ok"]), (409, False))
        self.assertIn("disagree", answer["error"])
        self.assertIn("force", answer["error"])
        self.assertEqual(e.calls, [("overlay_status",)])

    def test_force_writes_one_set_over_ranks_that_disagree(self):
        e = FakeEngine()
        e.active = ["base", "overlay", "base"]
        answer, status = run(e, "base", force=True)
        self.assertEqual((status, answer["ok"], answer["changed"], answer["active"], answer["previous"]),
                         (200, True, True, "base", None))
        self.assertEqual(e.active, ["base", "base", "base"])
        self.assertEqual(e.calls, [("overlay_status",), ("pause", "wait", True), ("overlay_switch", "base"),
                                   ("reset_prefix_cache",), ("resume",)])

    def test_force_rewrites_even_the_set_that_is_active_and_resumes(self):
        # the way out after a flip that left generation paused
        e = FakeEngine(active="base")
        e.paused = True
        answer, status = run(e, "base", force=True)
        self.assertEqual((status, answer["ok"]), (200, True))
        self.assertIn(("overlay_switch", "base"), e.calls)
        self.assertFalse(e.paused)

    def test_a_forced_flip_that_fails_has_nothing_to_put_back_and_stays_paused(self):
        e = FakeEngine(fail={("overlay_switch", "base"): (1,)})
        e.active = ["base", "overlay", "base"]
        answer, status = run(e, "base", force=True)
        self.assertEqual((status, answer["ok"], answer["paused"]), (500, False, True))
        self.assertEqual([c for c in e.calls if c[0] == "overlay_switch"], [("overlay_switch", "base")])
        self.assertNotIn(("resume",), e.calls)

    def test_force_cannot_make_an_unprepared_rank_write(self):
        e = FakeEngine(prepared=False, fail={("overlay_prepare", None): (1,)})
        answer, status = run(e, "overlay", force=True)
        self.assertEqual((status, answer["ok"]), (500, False))
        self.assertNotIn(("overlay_switch", "overlay"), e.calls)

    def test_abort_mode_reaches_the_pause(self):
        e = FakeEngine()
        run(e, "overlay", mode="abort")
        self.assertIn(("pause", "abort", True), e.calls)


class WhoMayCall(unittest.TestCase):
    def test_the_local_machine_may_and_others_may_not(self):
        self.assertIsNone(logic.refusal("127.0.0.1", None, {}))
        self.assertIsNone(logic.refusal("::1", None, {}))
        self.assertIsNotNone(logic.refusal("192.0.2.3", None, {}))
        self.assertIsNotNone(logic.refusal(None, None, {}))

    def test_an_allowed_address_is_let_in(self):
        env = {"OVERLAY_SWITCH_ALLOW": "192.0.2.3, 192.0.2.4"}
        self.assertIsNone(logic.refusal("192.0.2.4", None, env))
        self.assertIsNotNone(logic.refusal("192.0.2.5", None, env))

    def test_a_token_when_set_is_required_even_from_the_local_machine(self):
        env = {"OVERLAY_SWITCH_TOKEN": "s3"}
        self.assertIsNotNone(logic.refusal("127.0.0.1", None, env))
        self.assertIsNotNone(logic.refusal("127.0.0.1", "s4", env))
        self.assertIsNone(logic.refusal("127.0.0.1", "s3", env))


class Agree(unittest.TestCase):
    def test_empty_failed_and_split_answers_do_not_agree(self):
        self.assertFalse(logic.agree([])[0])
        self.assertFalse(logic.agree([{"ok": True, "active": "base"}, {"ok": False, "error": "x"}])[0])
        self.assertFalse(logic.agree([{"ok": True, "active": "base"}, {"ok": True, "active": "overlay"}])[0])
        self.assertFalse(logic.agree([{"ok": True, "active": "base"}], "overlay")[0])
        self.assertTrue(logic.agree([{"ok": True, "active": "base"}, {"ok": True, "active": "base"}], "base")[0])


if __name__ == "__main__":
    unittest.main()
