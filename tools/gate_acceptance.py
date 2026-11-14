#!/usr/bin/env python3
"""Acceptance test for the gate: one server serving two variants on two sets of paths. Standard library only.

The gate serves the base variant on the standard paths and the overlay under a prefix, and
switches when a request arrives for the other variant. The test runs four checks. Three are
exact, one is statistical.

1. Every response carries the header of the variant it requested.  (exact)
    The gate sets `x-overlay-side` on every response to a request that ran the model.

2. The two variants keep separate prefix caches, and a switch clears neither.  (exact)
    One long prompt is sent with the same client salt to both endpoints, and the server's own
    counters on /metrics are read around each request: base (nothing reused), base again
    (reused), overlay (nothing reused: the variants do not share), overlay again (reused), base
    once more (reused: two switches later its cache is still there).

3. The server switched as often as the variant changed.  (exact, with --status)
    The gate counts its switches. The count is read before and after the alternating requests.

4. No request is served with the other variant's weights.  (statistical, from the outputs)
    Reference: the standard endpoint is sampled several times, then the prefixed one, every
    request with its own cache salt. For each prompt the reference responses are compared token
    by token while they all agree. At every such position the margin between the token chosen
    most often and its strongest rival is recorded per variant. A position becomes a reference
    position (a "landmark" in the output) when the two variants differ there by a clear
    distance (--min-gap): from then on it identifies which weights produced a response. Mixed
    phase: the prompts that have a reference position are sent to both endpoints alternately,
    so that the server switches before every single request, and then in concurrent bursts to
    both. Each mixed response is compared at every reference position it reaches. On its own
    side of the midpoint between the two reference groups, it matches its variant. As far over
    as the other variant's reference responses, it was produced with the other variant's
    weights: one such response fails the test and is printed. Between the two (past the
    midpoint, short of the other group) it matches neither, and the result is inconclusive
    ("cannot tell" in the output): run the test again.

    How that rule was calibrated. The same request does not return the same logits twice on
    every deployment; on the test hardware a margin moved by one to four logits between
    identical requests. Unmixed samples from that hardware were replayed through this rule at
    the default gap: of 738 responses none fell with the other group and 2 fell in between.
    Then with the labels swapped on purpose: of 216 responses 185 fell with the other group, 31
    in between, and none on the side of its label. A response in between is therefore far more
    likely to be wrong than right, and does not pass.

With --overlay-model NAME the overlay is also requested by model name on the standard paths (a
server started with OVERLAY_GATE_MODELS): every second mixed request for the overlay is sent
that way, and the cache check sends one request by name for what was cached under the prefix.
The same four checks then cover both ways of selecting the overlay.

With --effort-check (a server started with OVERLAY_GATE_EFFORT_KEEP) there is a fifth, exact
check: every reasoning_effort value a seven-level client may send is accepted, on every path and
in both places a request can carry it. The words medium, minimal, ultra and an unknown one, and
the integers 50, 0 and 101, must each return 200, and the model list must advertise what the
model accepts.

    python3 tools/gate_acceptance.py http://HOST:8000 MODEL_NAME --prefix /free --out report.json

Exit status: 0 passed; 1 shown not to work; 2 inconclusive (too few reference positions, or the
cache counters could not be read cleanly); 3 the test could not run.
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
EFFORTS = ("medium", "minimal", "ultra", "banana", 50, 0, 101)


def endpoint(args, side):
    return args.server + (args.prefix if side == "overlay" else "")


def auth(args):
    return {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}


def read(args, side, tag, k, i, by_name=False):
    """One answer. `by_name`: ask for the overlay on the usual path, by the model name that chooses it."""
    started = time.monotonic()
    by_name = by_name and side == "overlay"
    answer = acceptance.ask(args, PROMPTS[i], f"{args.nonce}-{tag}-{side}-{k}-{i}", base=args.server if by_name else endpoint(args, side),
                            model=args.overlay_model if by_name else None)
    return {**answer, "asked": side, "via": "name" if by_name else "prefix" if side == "overlay" else "usual", "index": i,
            "phase": tag, "seconds": round(time.monotonic() - started, 3)}


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
                     "phase": answer["phase"], "via": answer.get("via"), "margin": margin, "middle": mark["middle"],
                     "edge": mark["edge"], "text": answer["text"]})
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
    by_name = name.endswith("by name")
    before = counters(args)
    body = {"model": args.overlay_model if by_name else args.model, "messages": [{"role": "user", "content": LONG}],
            "max_tokens": 8, "temperature": 0, "cache_salt": salt}
    if args.thinking_kwarg:
        body["chat_template_kwargs"] = {args.thinking_kwarg: False}
    url = (args.server if by_name else endpoint(args, side)) + "/v1/chat/completions"
    status, answer, headers = acceptance.call(url, body, auth(args), with_headers=True)
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
    plan = list(CACHE_STEPS)
    if args.overlay_model:
        plan.insert(4, ("overlay", "the overlay by name"))    # after the overlay's own repeat, before going back to base
    for attempt in range(args.cache_attempts):
        salt = f"{args.nonce}-cache-{attempt}"          # the same client salt on both endpoints: only the gate keeps them apart
        steps = [cache_step(args, side, salt, name) for side, name in plan]
        attempts.append(steps)
        if any(s.get("finished") != 1 or s.get("queried") != s["prompt_tokens"] for s in steps):
            continue                                      # other requests ran meanwhile: these numbers prove nothing
        named = steps.pop(4)["hit"] if args.overlay_model else None
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
        if named == 0:
            return {"verdict": "name apart", "attempts": attempts,
                    "why": f"asked for by name, the overlay reused nothing of the {other_again} tokens it had cached under the prefix"}
        return {"verdict": "apart", "attempts": attempts,
                "why": f"of {steps[0]['prompt_tokens']} tokens: base again reused {again}, the overlay's first request 0, "
                       f"the overlay again {other_again}, " + (f"the overlay by name {named}, " if named is not None else "")
                       + f"base after two flips {back}"}
    return {"verdict": "not measured", "attempts": attempts,
            "why": f"in {len(attempts)} attempts the counters never moved by exactly this test's own requests (other traffic?)"}


def effort_check(args):
    """Every effort a client may send, on every way in, at the top level and in chat_template_kwargs."""
    ways = [("base", "base", args.server, args.model)]
    if args.overlay_model:
        ways.append((f"overlay by the name {args.overlay_model}", "overlay", args.server, args.overlay_model))
    ways.append((f"overlay under {args.prefix}", "overlay", endpoint(args, "overlay"), args.model))
    rows = []
    for way, side, base, model in ways:
        for value in EFFORTS:
            for place in ("top level", "chat_template_kwargs"):
                body = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4, "temperature": 0}
                body.update({"reasoning_effort": value} if place == "top level" else {"chat_template_kwargs": {"reasoning_effort": value}})
                status, answer, headers = acceptance.call(base + "/v1/chat/completions", body, auth(args), with_headers=True)
                seen = {k.lower(): v for k, v in headers.items()}
                rows.append({"way": way, "asked": side, "side": seen.get("x-overlay-side"), "value": value, "place": place,
                             "status": status, "said": seen.get("x-reasoning-effort"),
                             "error": None if status == 200 else str((answer.get("error") or {}).get("message") or answer)[:200]})
    code, listing = acceptance.call(args.server + "/v1/models", headers=auth(args), timeout=30)
    listed = {m.get("id"): {k: v for k, v in m.items() if k.startswith("reasoning_effort")} for m in listing.get("data", [])} if code == 200 else {}
    return {"rows": rows, "refused": [r for r in rows if r["status"] != 200], "listed": listed}


def main():
    parser = argparse.ArgumentParser(description="Acceptance test for the overlay gate (see the docstring at the top of this file).")
    parser.add_argument("server", help="base URL of the server, e.g. http://127.0.0.1:8000")
    parser.add_argument("model", help="served model name")
    parser.add_argument("--prefix", default="/free", help="path prefix that serves the overlay (default /free)")
    parser.add_argument("--overlay-model", help="model name that selects the overlay on the standard paths; the overlay is then also requested by this name")
    parser.add_argument("--api-key", help="the server's API key, if it has one")
    parser.add_argument("--passes", type=int, default=6, help="reference samples of each endpoint (default 6)")
    parser.add_argument("--mixed", type=int, default=6, help="alternating rounds, with a switch before every request (default 6)")
    parser.add_argument("--burst", type=int, default=2, help="concurrent bursts to both endpoints (default 2)")
    parser.add_argument("--copies", type=int, default=2, help="requests per prompt and variant within one burst (default 2)")
    parser.add_argument("--min-gap", type=float, default=3.0,
                        help="minimum distance between the reference groups at a reference position, in logits (default 3)")
    parser.add_argument("--need", type=int, default=2, help="number of prompts that must have a reference position (default 2)")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--thinking-kwarg", default="enable_thinking")
    parser.add_argument("--effort-check", action="store_true",
                        help="also send every reasoning_effort value of a seven-level client; each must return 200")
    parser.add_argument("--no-cache-check", action="store_true", help="skip the cache check (for a server without /metrics)")
    parser.add_argument("--cache-attempts", type=int, default=3)
    parser.add_argument("--counter-wait", type=float, default=5.0, help="seconds to wait for the counters after a request")
    parser.add_argument("--status", help="base URL where GET /overlay/status is reachable from here (for the gate's own switch count)")
    parser.add_argument("--token", help="X-Overlay-Token, if the status route requires one")
    parser.add_argument("--out", help="write all samples and results to this JSON file")
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
        if args.overlay_model and args.overlay_model not in served:
            raise acceptance.CouldNotRun(f"{endpoint(args, side)} serves {served}, not {args.overlay_model!r}: nothing asked")

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
        for k in range(args.mixed):                       # with --overlay-model every other round asks for the overlay by name
            for i in telling:
                mixed += [read(args, side, "turn", k, i, by_name=bool(args.overlay_model) and k % 2 == 1) for side in SIDES]
        in_turn = len(mixed)
        gate_after = gate_counters()
        jobs = [(side, i, c) for c in range(args.copies) for i in telling for side in SIDES]
        print(f"mixed, at once: {args.burst} bursts of {len(jobs)} requests", flush=True)
        for k in range(args.burst):
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                mixed += list(pool.map(lambda job: read(args, job[0], "burst", f"{k}.{job[2]}", job[1],
                                                        by_name=bool(args.overlay_model) and job[2] % 2 == 1), jobs))
    cache = None
    if not args.no_cache_check:
        print("the cache: one long prompt, five requests, the server's counters around each", flush=True)
        cache = cache_check(args)
    effort = None
    if args.effort_check:
        print(f"reasoning_effort: {len(EFFORTS)} values, at the top level and in chat_template_kwargs, on every way in", flush=True)
        effort = effort_check(args)
    read(args, "base", "rest", 0, 0)                      # leave the server on the usual side

    everything = reference + mixed
    cache_steps = [s for attempt in (cache or {}).get("attempts", []) for s in attempt]
    answered = [r for r in effort["rows"] if r["status"] == 200] if effort else []     # a refusal is reported as one, below
    wrong_mark = [a for a in everything + cache_steps + answered if a["side"] != a["asked"]]
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
            print(f"{label}: asked on {row['asked']} ({row['phase']}, {row['via']}), token {row['position']} of {row['prompt'][:50]!r}: margin "
                  f"{row['margin']:+.2f}; middle {row['middle']:+.2f}, base from {row['edge']['base']:+.2f}, overlay from {row['edge']['overlay']:+.2f}")

    def mean(rows):
        return round(sum(a["seconds"] for a in rows) / max(len(rows), 1), 2)

    turned = [a for a in mixed if a["phase"] == "turn"]
    print(f"\n{len(everything)} answers; {len(mixed)} mixed, of which {placed} reached a landmark and {unplaced} had parted earlier; "
          f"{len(strays)} read like the other side, {len(between)} like neither.")
    if args.overlay_model:
        print(f"of the mixed requests for the overlay, {sum(a['via'] == 'name' for a in mixed)} asked by the name "
              f"{args.overlay_model!r} on the usual path and {sum(a['via'] == 'prefix' for a in mixed)} under {args.prefix}")
    print(f"an answer took {mean(reference)} s on average in the reference and {mean(turned)} s when the side changed before it")
    flips = None
    if gate_before is not None and gate_after is not None:
        flips = gate_after.get("flips", 0) - gate_before.get("flips", 0)
        print(f"the gate counted {flips} flips for the {in_turn} requests asked in turn; it now says {gate_after}")
    if cache:
        print(f"the cache: {cache['verdict']}: {cache['why']}")
    if effort:
        said = {}
        for row in effort["rows"]:
            said.setdefault((repr(row["value"]), row["place"]), set()).add(f"{row['status']} {row['said'] or 'as it is'}")
        for (value, place), answers in said.items():
            print(f"  reasoning_effort {value:10} {place:21} {' | '.join(sorted(answers))}")
        print(f"reasoning_effort: {len(effort['rows']) - len(effort['refused'])} of {len(effort['rows'])} requests answered 200; "
              f"the model list says: {effort['listed'] or 'nothing'}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"server": args.server, "model": args.model, "prefix": args.prefix, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "passes": args.passes, "min_gap": args.min_gap, "overlay_model": args.overlay_model,
                       "gate": {"before": gate_before, "after": gate_after},
                       "cache": cache, "effort": effort, "strays": strays, "between": between, "placed": placed, "unplaced": unplaced,
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
    if cache and cache["verdict"] == "name apart":
        print("FAIL: the overlay asked for by name does not share the cache of the overlay under the prefix: two ways in "
              "that should be one side.")
        return 1
    if flips is not None and flips < in_turn:
        print(f"FAIL: {in_turn} requests each changed the side, and the gate counted only {flips} flips.")
        return 1
    if effort and effort["refused"]:
        first = effort["refused"][0]
        print(f"FAIL: {len(effort['refused'])} reasoning_effort request(s) were not answered 200, e.g. {first['value']!r} "
              f"({first['place']}, {first['way']}): {first['status']} {first['error']}")
        return 1
    if effort and not all(m.get("reasoning_efforts") for m in effort["listed"].values()) or (effort and not effort["listed"]):
        print("FAIL: the model list does not say which reasoning efforts each model accepts.")
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
    efforts_said = f"; all {len(effort['rows'])} reasoning_effort requests were answered" if effort else ""
    print(f"PASS: all {len(everything)} answers carry their side; the {placed} mixed answers that reached a landmark all lie "
          f"with their side, on {len(telling)} prompts; {said}{efforts_said}.")
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
