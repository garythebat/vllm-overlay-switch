"""The translation of reasoning_effort, on its own. Standard library only.

    python3 -m unittest tests.test_effort -v
"""

import json
import logging
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mod"))

from vllm_overlay_switch import effort as eff  # noqa: E402

ENV = (eff.KEEP_ENV, eff.MAP_ENV, eff.INT_ENV)


def ladder():
    """A model that takes none, low, high, xhigh, max and integers 1 to 100, behind a seven-step client."""
    return eff.Effort(["none", "low", "high", "xhigh", "max"], {"minimal": "low", "medium": "high", "ultra": "max"}, "high", (1, 100))


class Translate(unittest.TestCase):
    def test_every_kind_of_value(self):
        e = ladder()
        for value, new, note in (
            (None, None, None),
            ("none", "none", None), ("low", "low", None), ("high", "high", None), ("xhigh", "xhigh", None), ("max", "max", None),
            ("minimal", "low", "minimal -> low"), ("medium", "high", "medium -> high"), ("ultra", "max", "ultra -> max"),
            ("banana", "high", "banana -> high (unknown)"), ("", "high", "'' -> high (unknown)"),
            (50, 50, None), (1, 1, None), (100, 100, None),
            (0, 1, "0 -> 1"), (-7, 1, "-7 -> 1"), (101, 100, "101 -> 100"),
            ("High", "high", "High -> high"), (" medium ", "high", "_medium_ -> high"),
            ("50", 50, "50 -> 50"), ("250", 100, "250 -> 100"), (50.0, 50, "50.0 -> 50"), (62.4, 62, "62.4 -> 62"),
            (True, "high", "true -> high (unknown)"), (["low"], "high", '["low"] -> high (unknown)'),
            (float("nan"), "high", "NaN -> high (unknown)"),
        ):
            with self.subTest(value=value):
                got, said = e.translate(value)
                self.assertEqual((got, type(got), said), (new, type(new), note))

    def test_what_is_not_configured_is_left_for_the_engine(self):
        words_only = eff.Effort(["low", "high"], {"medium": "high"})
        for value in ("banana", 50, True, ["x"]):
            self.assertEqual(words_only.translate(value), (value, None), value)
        self.assertEqual(words_only.translate("medium"), ("high", "medium -> high"))
        self.assertEqual(words_only.translate("50"), ("50", None))

    def test_a_map_may_lead_to_an_integer(self):
        e = eff.Effort(["low", "high"], {"medium": 62}, None, (1, 100))
        self.assertEqual(e.translate("medium"), (62, "medium -> 62"))

    def test_a_translation_that_the_model_would_refuse_is_refused_at_the_start(self):
        for keep, mapping, other, ints in ((["low"], {"medium": "high"}, None, None), (["low"], {}, "high", None),
                                           (["low"], {"medium": 62}, None, None), (["low"], {"medium": 162}, None, (1, 100)),
                                           ([], {}, None, None), (["low"], {}, None, (5, 1))):
            with self.assertRaises(ValueError, msg=(keep, mapping, other, ints)):
                eff.Effort(keep, mapping, other, ints)


class Apply(unittest.TestCase):
    def apply(self, request):
        e = ladder()
        notes = e.apply(request)
        return request, notes

    def test_the_two_places_are_treated_alike(self):
        self.assertEqual(self.apply({"reasoning_effort": "medium"}), ({"reasoning_effort": "high"}, ["medium -> high"]))
        self.assertEqual(self.apply({"chat_template_kwargs": {"reasoning_effort": "medium", "x": 1}}),
                         ({"chat_template_kwargs": {"reasoning_effort": "high", "x": 1}}, ["medium -> high"]))
        self.assertEqual(self.apply({"reasoning_effort": "ultra", "chat_template_kwargs": {"reasoning_effort": 0}}),
                         ({"reasoning_effort": "max", "chat_template_kwargs": {"reasoning_effort": 1}}, ["0 -> 1", "ultra -> max"]))

    def test_an_integer_at_the_top_level_goes_into_the_kwargs(self):
        self.assertEqual(self.apply({"reasoning_effort": 50}),
                         ({"chat_template_kwargs": {"reasoning_effort": 50}}, ["50 moved to chat_template_kwargs"]))
        self.assertEqual(self.apply({"reasoning_effort": 101, "chat_template_kwargs": {"enable_thinking": True}}),
                         ({"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": 100}}, ["101 -> 100"]))
        self.assertEqual(self.apply({"reasoning_effort": "7", "chat_template_kwargs": None}),
                         ({"chat_template_kwargs": {"reasoning_effort": 7}}, ["7 -> 7"]))
        # the kwargs win in the engine; a top-level integer beside them would only be refused by the schema
        self.assertEqual(self.apply({"reasoning_effort": 50, "chat_template_kwargs": {"reasoning_effort": "low"}}),
                         ({"chat_template_kwargs": {"reasoning_effort": "low"}}, ["50 dropped (chat_template_kwargs has its own)"]))

    def test_a_request_without_the_field_is_not_touched(self):
        for request in ({}, {"messages": [], "chat_template_kwargs": {"enable_thinking": False}}, {"reasoning_effort": None},
                        {"chat_template_kwargs": {"reasoning_effort": None}}):
            before = json.dumps(request, sort_keys=True)
            got, notes = self.apply(request)
            self.assertEqual((json.dumps(got, sort_keys=True), notes), (before, []))

    def test_what_passes_as_it_is_is_not_touched_either(self):
        for request in ({"reasoning_effort": "high"}, {"reasoning_effort": "none"}, {"chat_template_kwargs": {"reasoning_effort": 75}}):
            before = json.dumps(request, sort_keys=True)
            got, notes = self.apply(request)
            self.assertEqual((json.dumps(got, sort_keys=True), notes), (before, []))

    def test_kwargs_that_are_no_object_leave_the_request_whole(self):
        request = {"reasoning_effort": "medium", "chat_template_kwargs": "enable_thinking"}
        self.assertEqual(self.apply(dict(request)), (request, []))

    def test_every_translation_is_counted_and_logged_once_per_distinct_value(self):
        e = ladder()
        with self.assertLogs("vllm.overlay_switch.effort", level=logging.INFO) as logs:
            for _ in range(3):
                e.apply({"model": "m", "reasoning_effort": "medium"})
            e.apply({"model": "m-free", "reasoning_effort": "banana"})
        self.assertEqual([r.getMessage() for r in logs.records],
                         ["reasoning_effort coerced: medium -> high (model=m)", "reasoning_effort coerced: banana -> high (unknown) (model=m-free)"])
        self.assertEqual(e.describe()["coerced"], {"medium -> high": 3, "banana -> high (unknown)": 1})

    def test_a_flood_of_made_up_values_does_not_grow_the_memory(self):
        e = ladder()
        logging.getLogger("vllm.overlay_switch.effort").disabled = True
        try:
            for n in range(1000):
                e.apply({"reasoning_effort": f"made-up-{n}"})
        finally:
            logging.getLogger("vllm.overlay_switch.effort").disabled = False
        self.assertEqual(len(e.seen), eff.MOST_SEEN)

    def test_a_value_that_would_break_a_header_or_a_log_line_is_shown_safely(self):
        e = ladder()
        _, notes = self.apply({"reasoning_effort": "x\r\nSet-Cookie: a=bé" + "y" * 90})
        self.assertTrue(notes[0].endswith("-> high (unknown)"))
        value = eff.header_value(notes)
        self.assertTrue(all(32 <= b < 127 for b in value) and len(value) <= 200 and b"\r" not in value and b"\n" not in value, value)
        self.assertEqual(e.seen, {})


class Advertise(unittest.TestCase):
    def test_every_model_in_the_list_says_what_is_accepted(self):
        listing = {"object": "list", "data": [{"id": "m", "root": "m"}, {"id": "m-free"}]}
        got = json.loads(ladder().advertise(json.dumps(listing).encode()))
        for model in got["data"]:
            self.assertEqual((model["reasoning_efforts"], model["reasoning_effort_integer_min"], model["reasoning_effort_integer_max"],
                              model["reasoning_effort_aliases"]),
                             (["none", "low", "high", "xhigh", "max"], 1, 100, {"minimal": "low", "medium": "high", "ultra": "max"}))
        self.assertEqual([m["id"] for m in got["data"]], ["m", "m-free"])
        self.assertNotIn("reasoning_effort_integer_max", json.loads(eff.Effort(["low"]).advertise(b'{"data": [{"id": "m"}]}'))["data"][0])

    def test_what_is_no_model_list_is_left_alone(self):
        for body in (b"not json", b"[1]", b'{"data": "x"}', b'{"error": "no"}', b""):
            self.assertIsNone(ladder().advertise(body), body)


class FromEnv(unittest.TestCase):
    def setUp(self):
        self.saved = {k: os.environ.pop(k, None) for k in ENV}

    def tearDown(self):
        for key, value in self.saved.items():
            os.environ.pop(key, None)
            if value is not None:
                os.environ[key] = value

    def test_off_unless_the_words_to_keep_are_named(self):
        self.assertIsNone(eff.from_env())
        os.environ[eff.MAP_ENV] = "medium=high"
        self.assertIsNone(eff.from_env())

    def test_the_three_settings(self):
        os.environ.update({eff.KEEP_ENV: " none, low ,HIGH,xhigh,max ", eff.MAP_ENV: "minimal=low, medium = high ,ultra=max,*=high,half=50",
                           eff.INT_ENV: "1-100"})
        e = eff.from_env()
        self.assertEqual(e.describe(), {"keep": ["none", "low", "high", "xhigh", "max"], "coerced": {}, "int": [1, 100],
                                        "map": {"minimal": "low", "medium": "high", "ultra": "max", "half": 50, "*": "high"}})

    def test_a_mistake_in_them_stops_the_start(self):
        for bad in ({eff.MAP_ENV: "medium"}, {eff.MAP_ENV: "medium=hgih"}, {eff.MAP_ENV: "=high"}, {eff.INT_ENV: "1..100"},
                    {eff.INT_ENV: "100"}, {eff.INT_ENV: "9-1"}, {eff.MAP_ENV: "medium=62"}):
            for key in ENV:
                os.environ.pop(key, None)
            os.environ[eff.KEEP_ENV] = "low,high"
            os.environ.update(bad)
            with self.assertRaises(ValueError, msg=bad):
                eff.from_env()


if __name__ == "__main__":
    unittest.main()
