#!/usr/bin/env python3
"""Acceptance test for the gate: one server answering on two sets of paths. Standard library only.

The gate lets the base set answer on the usual paths and the overlay under a prefix, flipping
when a request arrives for the other side. Four questions, three of them answered exactly.

Is every answer marked with the side it was asked on?  (exact)
    The gate puts `x-overlay-side` on every answer of a request that ran the model.

Is a request ever answered by the other side's weights?  (by the answers)
    Reference: the usual endpoint is read several times, then the prefixed one, every request
    under a cache salt of its own. For each prompt the reference answers are walked token by
    token while they all agree. At every such position the margin between the most chosen
    token and its strongest rival is taken, by side. A position is a landmark when the two
    sides stand apart there by a clear distance (--min-gap): from then on it tells which
    weights produced an answer. Mixed: the prompts that have a landmark are asked on both
    endpoints in turn, so that the server flips before every single request, and then in
    bursts of both at once. Each mixed answer is laid against every landmark it reaches. On
    its own side of the middle between the two reference groups it lies with its side. As
    far over as the other side's reference answers, it was answered by the other side's
    weights: one such answer fails the test and is shown. In the stretch between (past the
    middle, short of the other group) it reads like neither, and the test then says it
    cannot tell: run it again.

    Where that rule comes from. The same request does not return the same numbers twice on
    every deployment; on the first hardware this ran on, a margin moved by one to four logits
    between identical requests. Readings from that hardware in which nothing was mixed were
    replayed through this rule at the default gap: of 738 answers none lay with the other
    group and 2 lay in the stretch between. Then with the labels swapped on purpose: of 216
    answers 185 lay with the other group, 31 in the stretch between, none on the side of
    their label. So an answer between is far likelier a wrong one than a right one, and is
    not passed.

Do the two sides keep their cached prefixes apart, and does a flip leave them alone?  (exact)
    One long prompt is sent under the same client salt on both endpoints, and the server's own
    counters on /metrics are read around each request: base (nothing reused), base again
    (reused), overlay (nothing reused: the sides do not share), overlay again (reused), base
    once more (reused: two flips later its cache is still there).

Did the server flip as often as the side changed?  (exact, with --status)
    The gate counts its flips. The count is read before and after the requests asked in turn.

    python3 tools/gate_acceptance.py http://HOST:8000 MODEL_NAME --prefix /free --out report.json

Exit status: 0 passed; 1 shown not to work; 2 cannot tell (too few landmarks, or the cache
counters could not be read cleanly); 3 could not run.
"""

import argparse
import concurrent.futures
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acceptance  # noqa: E402

PROMPTS = acceptance.PROMPTS
SIDES = ("base", "overlay")
LONG = " ".join(PROMPTS * 6) + " Answer only the first of these, in one short sentence."
CACHE_STEPS = (("base", "first on base"), ("base", "again on base"), ("overlay", "first on the overlay"),
               ("overlay", "again on the overlay"), ("base", "back on base"))
COUNTERS = {"queried": "vllm:prefix_cache_queries_total", "hit": "vllm:prefix_cache_hits_total",
            "finished": "vllm:request_success_total"}


def endpoint(args, side):
    return args.server + (args.prefix if side == "overlay" else "")


def auth(args):
    return {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}


def read(args, side, tag, k, i):
    started = time.monotonic()
    answer = acceptance.ask(args, PROMPTS[i], f"{args.nonce}-{tag}-{side}-{k}-{i}", base=endpoint(args, side))
    return {**answer, "asked": side, "index": i, "phase": tag, "seconds": round(time.monotonic() - started, 3)}


# ---- landmarks: where the reference tells the two sides apart, and where a later answer lies there

def find_landmarks(reference, min_gap):
    """For one prompt, from the reference answers of both sides: (positions looked at, landmarks)."""
    tokens = [[t["token"] for t in a["tokens"]] for a in reference]
    n = 0
    while all(len(t) > n for t in tokens) and len({t[n] for t in tokens}) == 1:
        n += 1
    parted = all(len(t) > n for t in tokens)            # false when some answer simply ends at n
    runs = [(a["asked"], a) for a in reference]
    looked, found = 0, []
    for position in range(n + 1 if parted else n):
        seen = acceptance.margins_at(runs, position)
        if seen is None:
            continue
        looked += 1
        base, over = seen["margins"].get("base") or [], seen["margins"].get("overlay") or []
        distance = acceptance.gap(base, over)
        if not distance or distance < min_gap:
            continue
        base_is_high = min(base) > max(over)
        middle = (min(base) + max(over)) / 2 if base_is_high else (max(base) + min(over)) / 2
        found.append({"position": position, "read": tokens[0][:position], "contenders": seen["contenders"],
                      "gap": round(distance, 3), "middle": round(middle, 3), "base_is_high": base_is_high,
                      "edge": {"base": min(base) if base_is_high else max(base), "overlay": max(over) if base_is_high else min(over)},
                      "reference": {"base": base, "overlay": over}, "mixed": {"base": [], "overlay": []}})
    return looked, found


def place(answer, mark):
    """Where one answer lies at a landmark: its margin there, or None if it had read other words by then."""
    tokens, position = answer["tokens"], mark["position"]
    if len(tokens) <= position or [t["token"] for t in tokens[:position]] != mark["read"]:
        return None
    return acceptance.margin_of(tokens[position], *mark["contenders"])


def lies(mark, side, margin):
    """Where a margin lies for an answer asked on `side`: "own" (its side of the middle), "other" (as far
    over as the other side's reference answers), or "between" (past the middle, short of the other group)."""
    high = mark["base_is_high"] if side == "base" else not mark["base_is_high"]      # is this side the one with the higher margins?
    other_edge = mark["edge"]["overlay" if side == "base" else "base"]
    if (margin > mark["middle"]) if high else (margin < mark["middle"]):
        return "own"
    if (margin <= other_edge) if high else (margin >= other_edge):
        return "other"
    return "between"


def judge(mixed, marks):
    """Lay every mixed answer against its prompt's landmarks.

    Returns (strays: answered like the other side; between: like neither; answers placed; answers not placed)."""
    strays, between, placed, unplaced = [], [], 0, 0
    for answer in mixed:
        anywhere = False
        for mark in marks[answer["index"]]:
            margin = place(answer, mark)
            if margin is None:
                continue
            anywhere = True
            mark["mixed"][answer["asked"]].append(margin)
            where = lies(mark, answer["asked"], margin)
            if where != "own":
                (strays if where == "other" else between).append(
                    {"prompt": PROMPTS[answer["index"]], "position": mark["position"], "asked": answer["asked"],
                     "phase": answer["phase"], "margin": margin, "middle": mark["middle"], "edge": mark["edge"], "text": answer["text"]})
        placed += anywhere
        unplaced += not anywhere
    return strays, between, placed, unplaced


# ---- the cache: the server's own counters around single requests

def counters(args):
    """The server's prefix cache counters and its count of finished requests, or None when /metrics lacks them."""
    request = urllib.request.Request(args.server + "/metrics", headers=auth(args))
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError):
        return None
    got = {}
    for key, name in COUNTERS.items():
        values = [float(line.rsplit(" ", 1)[1]) for line in text.splitlines() if line.startswith((name + "{", name + " "))]
        if not values:
            return None
        got[key] = sum(values)
    return got


def cache_step(args, side, salt, name):
    """One request for the long prompt, with what the server's counters did around it."""
    before = counters(args)
    body = {"model": args.model, "messages": [{"role": "user", "content": LONG}], "max_tokens": 8, "temperature": 0,
            "cache_salt": salt}
    if args.thinking_kwarg:
        body["chat_template_kwargs"] = {args.thinking_kwarg: False}
    status, answer, headers = acceptance.call(endpoint(args, side) + "/v1/chat/completions", body, auth(args), with_headers=True)
    if status != 200:
        raise acceptance.CouldNotRun(f"the server answered {status} to the long prompt: {str(answer)[:300]}")
    deadline = time.monotonic() + args.counter_wait
    while True:                                           # the counters are published with the engine's next output
        after = counters(args)
        if before is None or after is None or after["finished"] > before["finished"] or time.monotonic() > deadline:
            break
        time.sleep(0.2)
    step = {"step": name, "asked": side, "prompt_tokens": (answer.get("usage") or {}).get("prompt_tokens"),
            "side": next((v for k, v in headers.items() if k.lower() == "x-overlay-side"), None)}
    if before is not None and after is not None:
        step.update({key: round(after[key] - before[key]) for key in COUNTERS})
    return step


def cache_check(args):
    """{"verdict": "apart" | "shared" | "cleared" | "no reuse" | "not measured", "why": ..., "attempts": [...]}"""
    if counters(args) is None:
        return {"verdict": "not measured", "why": "the server's /metrics page does not give the prefix cache counters", "attempts": []}
    attempts = []
    for attempt in range(args.cache_attempts):
        salt = f"{args.nonce}-cache-{attempt}"          # the same client salt on both endpoints: only the gate keeps them apart
        steps = [cache_step(args, side, salt, name) for side, name in CACHE_STEPS]
        attempts.append(steps)
        if any(s.get("finished") != 1 or s.get("queried") != s["prompt_tokens"] for s in steps):
            continue                                      # other requests ran meanwhile: these numbers prove nothing
        first, again, other, other_again, back = (s["hit"] for s in steps)
        if first:
            continue                                      # a salt never used before found something cached: not understood
        if not again:
            return {"verdict": "no reuse", "why": "a plain repeat reused nothing: this server keeps no prefix cache to separate",
                    "attempts": attempts}
        if other:
            return {"verdict": "shared", "attempts": attempts,
                    "why": f"the first request on the overlay reused {other} tokens that the base side had computed"}
        if not other_again:
            continue                                      # the overlay's own repeat reused nothing: not understood
        if not back:
            return {"verdict": "cleared", "attempts": attempts,
                    "why": f"after going to the overlay and back, base reused nothing of the {again} tokens it had cached"}
        return {"verdict": "apart", "attempts": attempts,
                "why": f"of {steps[0]['prompt_tokens']} tokens: base again reused {again}, the overlay's first request 0, "
                       f"the overlay again {other_again}, base after two flips {back}"}
    return {"verdict": "not measured", "attempts": attempts,
            "why": f"in {len(attempts)} attempts the counters never moved by exactly this test's own requests (other traffic?)"}


def main():
    parser = argparse.ArgumentParser(description="Acceptance test for the overlay gate (see the top of this file).")
    parser.add_argument("server", help="base URL of the server, e.g. http://127.0.0.1:8000")
    parser.add_argument("model", help="the served model name")
    parser.add_argument("--prefix", default="/free", help="the path prefix under which the overlay answers (default /free)")
    parser.add_argument("--api-key", help="API key of the server, if it has one")
    parser.add_argument("--passes", type=int, default=6, help="reference readings of each endpoint (default 6)")
    parser.add_argument("--mixed", type=int, default=6, help="rounds asked in turn, a flip before every request (default 6)")
    parser.add_argument("--burst", type=int, default=2, help="bursts of both sides at once (default 2)")
    parser.add_argument("--copies", type=int, default=2, help="how often each prompt is asked per side within one burst (default 2)")
    parser.add_argument("--min-gap", type=float, default=3.0,
                        help="how far apart the reference groups must stand at a landmark, in logits (default 3)")
    parser.add_argument("--need", type=int, default=2, help="prompts that must have a landmark (default 2)")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--thinking-kwarg", default="enable_thinking")
    parser.add_argument("--no-cache-check", action="store_true", help="leave the cache counters out (a server without /metrics)")
    parser.add_argument("--cache-attempts", type=int, default=3)
    parser.add_argument("--counter-wait", type=float, default=5.0, help="seconds to wait for the counters after a request")
    parser.add_argument("--status", help="base URL on which GET /overlay/status answers from here (for the gate's own flip count)")
    parser.add_argument("--token", help="X-Overlay-Token, if the status route asks for one")
    parser.add_argument("--out", help="write everything that was read to this JSON file")
    args = parser.parse_args()
    args.server, args.prefix = args.server.rstrip("/"), "/" + args.prefix.strip("/")
    args.nonce = secrets.token_hex(4)
    if args.passes < acceptance.MIN_PASSES:
        raise acceptance.CouldNotRun(f"at least {acceptance.MIN_PASSES} reference passes")
    for side in SIDES:
        code, models = acceptance.call(endpoint(args, side) + "/v1/models", headers=auth(args), timeout=30)
        served = [m.get("id") for m in models.get("data", [])] if code == 200 else []
        if args.model not in served:
            raise acceptance.CouldNotRun(f"{endpoint(args, side)} answered {code} and serves {served}, not {args.model!r}: "
                                         f"{'is the gate installed under ' + args.prefix + '?' if side == 'overlay' else 'nothing asked'}")

    def gate_counters():
        if not args.status:
            return None
        code, answer = acceptance.call(args.status.rstrip("/") + "/overlay/status", timeout=120,
                                       headers={"X-Overlay-Token": args.token} if args.token else {})
        return answer.get("gate") if code == 200 else None

    reference = []
    for side in SIDES:
        print(f"reference, {side} endpoint: {args.passes} passes of {len(PROMPTS)} prompts", flush=True)
        for k in range(args.passes):
            reference += [read(args, side, "ref", k, i) for i in range(len(PROMPTS))]
    looked, marks = 0, {}
    for i in range(len(PROMPTS)):
        n, marks[i] = find_landmarks([a for a in reference if a["index"] == i], args.min_gap)
        looked += n
    telling = [i for i in marks if marks[i]]
    print(f"{looked} positions looked at; {len(telling)} of {len(PROMPTS)} prompts have a landmark", flush=True)

    mixed, gate_before, gate_after, in_turn = [], None, None, 0
    if telling:
        gate_before = gate_counters()
        print(f"mixed, in turn: {args.mixed} rounds of {len(telling)} prompts, the other endpoint at every request", flush=True)
        for k in range(args.mixed):
            for i in telling:
                mixed += [read(args, side, "turn", k, i) for side in SIDES]
        in_turn = len(mixed)
        gate_after = gate_counters()
        jobs = [(side, i, c) for c in range(args.copies) for i in telling for side in SIDES]
        print(f"mixed, at once: {args.burst} bursts of {len(jobs)} requests", flush=True)
        for k in range(args.burst):
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                mixed += list(pool.map(lambda job: read(args, job[0], "burst", f"{k}.{job[2]}", job[1]), jobs))
    cache = None
    if not args.no_cache_check:
        print("the cache: one long prompt, five requests, the server's counters around each", flush=True)
        cache = cache_check(args)
    read(args, "base", "rest", 0, 0)                      # leave the server on the usual side

    everything = reference + mixed
    cache_steps = [s for attempt in (cache or {}).get("attempts", []) for s in attempt]
    wrong_mark = [a for a in everything + cache_steps if a["side"] != a["asked"]]
    strays, between, placed, unplaced = judge(mixed, marks)
    for i in telling:
        print(f"\n{PROMPTS[i][:70]!r}")
        for mark in marks[i]:
            print(f"  token {mark['position']}, {mark['contenders'][0]!r} over {mark['contenders'][1]!r}: "
                  f"the two sides {mark['gap']} apart, middle {mark['middle']:+.2f}")
            for side in SIDES:
                ref, mix = mark["reference"][side], sorted(mark["mixed"][side])
                print(f"    {side:8} reference {ref[0]:+.2f} to {ref[-1]:+.2f}"
                      + (f"   mixed {mix[0]:+.2f} to {mix[-1]:+.2f} ({len(mix)} answers)" if mix else "   mixed: none reached it"))
    for label, rows in (("ON THE WRONG SIDE", strays), ("between the two", between)):
        for row in rows:
            print(f"{label}: asked on {row['asked']} ({row['phase']}), token {row['position']} of {row['prompt'][:50]!r}: margin "
                  f"{row['margin']:+.2f}; middle {row['middle']:+.2f}, base from {row['edge']['base']:+.2f}, overlay from {row['edge']['overlay']:+.2f}")

    def mean(rows):
        return round(sum(a["seconds"] for a in rows) / max(len(rows), 1), 2)

    turned = [a for a in mixed if a["phase"] == "turn"]
    print(f"\n{len(everything)} answers; {len(mixed)} mixed, of which {placed} reached a landmark and {unplaced} had parted earlier; "
          f"{len(strays)} read like the other side, {len(between)} like neither.")
    print(f"an answer took {mean(reference)} s on average in the reference and {mean(turned)} s when the side changed before it")
    flips = None
    if gate_before is not None and gate_after is not None:
        flips = gate_after.get("flips", 0) - gate_before.get("flips", 0)
        print(f"the gate counted {flips} flips for the {in_turn} requests asked in turn; it now says {gate_after}")
    if cache:
        print(f"the cache: {cache['verdict']}: {cache['why']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"server": args.server, "model": args.model, "prefix": args.prefix, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "passes": args.passes, "min_gap": args.min_gap, "gate": {"before": gate_before, "after": gate_after},
                       "cache": cache, "strays": strays, "between": between, "placed": placed, "unplaced": unplaced,
                       "landmarks": [{"prompt": PROMPTS[i], "marks": marks[i]} for i in marks],
                       "reference": reference, "mixed": mixed}, f, indent=1)
        print("saved", args.out)

    if wrong_mark:
        print(f"FAIL: {len(wrong_mark)} answer(s) were not marked with the side they were asked on "
              f"(e.g. asked {wrong_mark[0]['asked']!r}, marked {wrong_mark[0]['side']!r}). Is the gate installed?")
        return 1
    if strays:
        print(f"FAIL: {len(strays)} mixed answer(s) lie as far over as the other side's reference answers at a landmark: "
              "answered by the wrong weights.")
        return 1
    if cache and cache["verdict"] == "shared":
        print("FAIL: the two sides share cached prefixes: what one set of weights computed was reused under the other.")
        return 1
    if cache and cache["verdict"] == "cleared":
        print("FAIL: a flip cleared the cache. No answer is wrong for that, but each side is meant to keep its own.")
        return 1
    if flips is not None and flips < in_turn:
        print(f"FAIL: {in_turn} requests each changed the side, and the gate counted only {flips} flips.")
        return 1
    if len(telling) < args.need or not placed:
        print(f"CANNOT TELL: landmarks on {len(telling)} prompts (wanted {args.need}), {placed} mixed answers reached one. These "
              "prompts do not separate the two sides clearly enough; this is not proof of anything.")
        return 2
    if between:
        print(f"CANNOT TELL: {len(between)} of {placed} mixed answers read like neither side (past the middle, short of the "
              "other group). A right answer does that rarely, an answer from the wrong weights about one time in seven: "
              "run the test again, and take a second such result as a failure.")
        return 2
    if cache and cache["verdict"] == "not measured":
        print(f"CANNOT TELL: the answers lie with their sides, but the cache could not be checked: {cache['why']}. "
              "With --no-cache-check the test leaves it out and says so.")
        return 2
    said = {None: "the cache was not looked at", "apart": "each side reused only its own cached prefix, also after two flips",
            "no reuse": "this server reuses no prefixes, so there was nothing to keep apart"}[cache and cache["verdict"]]
    print(f"PASS: all {len(everything)} answers carry their side; the {placed} mixed answers that reached a landmark all lie "
          f"with their side, on {len(telling)} prompts; {said}.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except acceptance.Failed as e:
        print(f"FAIL: {e}")
        sys.exit(1)
    except (acceptance.CouldNotRun, urllib.error.URLError, OSError, KeyError, ValueError) as e:
        print(f"could not run: {type(e).__name__}: {e}")
        sys.exit(3)
