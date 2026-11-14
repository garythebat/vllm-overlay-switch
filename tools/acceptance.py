#!/usr/bin/env python3
"""Acceptance test for the overlay switch, against a running server. Standard library only.

It asks nothing sensitive and needs no reference answers. Two models with different weights
choose different next tokens at some point where both have read exactly the same words, and
each repeats its own choice. So:

  1. ask a handful of plain prompts at temperature 0, twice, on the set that is live (A);
  2. flip to the other set (B) and ask again, twice;
  3. flip back and ask again, twice.

For each prompt the six answers are walked token by token while they all agree. At the first
place they part, every answer has been produced from an identical context. The flip is shown
to work when, at that place, all four A answers chose one token and both B answers another.
If the two A readings (before and after) disagree with each other, flipping back did not
restore the model.

Two identical requests do not always return identical text on this kind of deployment, and
wording drifts after a few tokens. That is why only the first parting is judged, and each
set against its own repeat.

Run it where the switch accepts calls from (by default: on the machine the server runs on).

    python3 tools/acceptance.py http://127.0.0.1:8000 MODEL_NAME
    python3 tools/acceptance.py http://127.0.0.1:8000 MODEL_NAME --out report.json --need 2

Exit status: 0 the flip is shown and the return restores; 1 it is shown not to work;
2 these prompts cannot tell the two sets apart (not proof of anything); 3 could not run.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

PROMPTS = [
    "Name the three primary colours of light, in one short sentence.",
    "What is the boiling point of water at sea level in Celsius? One sentence.",
    "Write one sentence about a lighthouse at dusk.",
    "Explain in one sentence why the sky is blue.",
    "Describe the smell of rain on dry ground in one sentence.",
    "Give one sentence of advice to someone learning to bake bread.",
    "In one sentence, what does a keel do for a sailing boat?",
    "Write one sentence that begins with the word 'Although'.",
]
OTHER = {"base": "overlay", "overlay": "base"}


class CouldNotRun(Exception):
    """The test could not be carried out (exit status 3)."""


class Failed(Exception):
    """The switch itself reported a failure (exit status 1)."""


def call(url, body=None, headers=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.load(e)
        except Exception:
            return e.code, {"ok": False, "error": f"HTTP {e.code}"}


def ask(args, prompt):
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": args.max_tokens,
            "temperature": 0, "logprobs": True, "top_logprobs": 5}
    if args.thinking_kwarg:
        body["chat_template_kwargs"] = {args.thinking_kwarg: False}
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    status, answer = call(args.server + "/v1/chat/completions", body, headers)
    if status != 200:
        raise CouldNotRun(f"the server answered {status} to a chat request: {str(answer)[:300]}")
    choice = answer["choices"][0]
    tokens = [{"token": t["token"], "logprob": t["logprob"], "top": [[c["token"], c["logprob"]] for c in t.get("top_logprobs") or []]}
              for t in (choice.get("logprobs") or {}).get("content") or []]
    if not tokens:
        raise CouldNotRun("the server returned no per-token log probabilities: nothing to compare")
    return {"prompt": prompt, "text": choice["message"].get("content"), "tokens": tokens}


def reading(args, label):
    print(f"reading {label}: {args.passes} passes of {len(PROMPTS)} prompts")
    return [[ask(args, prompt) for prompt in PROMPTS] for _ in range(args.passes)]


def status(args, full=False):
    code, answer = call(args.switch + "/overlay/status" + ("?full=1" if full else ""), headers=args.switch_headers, timeout=120)
    if code != 200 or not answer.get("ok"):
        raise Failed(f"the switch's status call answered {code}: {str(answer)[:400]}")
    return answer


def flip(args, variant):
    started = time.monotonic()
    code, answer = call(args.switch + "/overlay/switch", {"variant": variant, "mode": "wait"}, args.switch_headers)
    took = round(time.monotonic() - started, 3)
    if code != 200 or not answer.get("ok") or answer.get("active") != variant:
        raise Failed(f"the flip to {variant!r} answered {code}: {str(answer)[:600]}")
    print(f"flip to {variant}: {took} s as seen from here; the server's own timings: {answer.get('timings')}")
    seen = status(args, full=True)
    if seen.get("active") != variant:
        raise Failed(f"after the flip a full status reads {seen.get('active')!r}, not {variant!r}")
    return {"variant": variant, "seconds": took, "timings": answer.get("timings"), "ranks": answer.get("ranks")}


def judge(index, a1, b, a2):
    """Walk the answers of one prompt to their first parting and say what it shows."""
    runs = [("A before", r[index]) for r in a1] + [("B", r[index]) for r in b] + [("A after", r[index]) for r in a2]
    tokens = [[t["token"] for t in r["tokens"]] for _, r in runs]
    n = 0
    while all(len(t) > n for t in tokens) and len({t[n] for t in tokens}) == 1:
        n += 1
    out = {"prompt": PROMPTS[index], "agree_for": n, "shared": "".join(tokens[0][:n])}
    if not all(len(t) > n for t in tokens):
        if len({len(t) for t in tokens}) == 1:
            return {**out, "verdict": "same", "note": "all answers are identical: this prompt does not tell the sets apart"}
        return {**out, "verdict": "unclear", "note": "some answers end where others go on"}
    chosen = [t[n] for t in tokens]
    passes = len(a1)
    before, middle, after = set(chosen[:passes]), set(chosen[passes:2 * passes]), set(chosen[2 * passes:])
    out["chosen"] = {"A before": chosen[:passes], "B": chosen[passes:2 * passes], "A after": chosen[2 * passes:]}
    if len(before) == 1 and len(after) == 1 and len(middle) == 1:
        if before == after and middle != before:
            return {**out, "verdict": "clear", "note": "A chose one token every time, B another: repeatable, by set"}
        if before != after:
            return {**out, "verdict": "not restored", "note": "A after the return does not choose what A chose before"}
    if before | after == middle or (before == middle and after == middle):
        return {**out, "verdict": "noise", "note": "the same choices on both sides"}
    return {**out, "verdict": "unclear", "note": "the parting does not sort by set"}


def main():
    parser = argparse.ArgumentParser(description="Acceptance test for the overlay switch (see the top of this file).")
    parser.add_argument("server", help="base URL of the OpenAI-compatible server, e.g. http://127.0.0.1:8000")
    parser.add_argument("model", help="the served model name")
    parser.add_argument("--switch", help="base URL of the switch routes if not the same as the server")
    parser.add_argument("--api-key", help="API key of the server, if it has one")
    parser.add_argument("--token", help="X-Overlay-Token, if the switch asks for one")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--need", type=int, default=2, help="how many prompts must part clearly (default 2)")
    parser.add_argument("--thinking-kwarg", default="enable_thinking",
                        help="chat template switch to turn thinking off; empty to send none")
    parser.add_argument("--expect-parting", action="store_true",
                        help="the two sets are known to part on these prompts when served separately: "
                             "treat no parting at all as a failure, not as 'cannot tell'")
    parser.add_argument("--out", help="write everything that was read to this JSON file")
    args = parser.parse_args()
    args.server = args.server.rstrip("/")
    args.switch = (args.switch or args.server).rstrip("/")
    args.switch_headers = {"X-Overlay-Token": args.token} if args.token else {}
    if args.passes < 2:
        raise CouldNotRun("at least two passes: each set is judged against its own repeat")

    code, models = call(args.server + "/v1/models", headers={"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}, timeout=30)
    served = [m.get("id") for m in models.get("data", [])] if code == 200 else []
    if args.model not in served:
        raise CouldNotRun(f"{args.server} serves {served}, not {args.model!r}: nothing asked")

    start = status(args)
    a = start.get("active")
    if a not in OTHER:
        # not prepared yet: the first flip prepares. Flipping to "base" on a fresh server changes nothing.
        code, answer = call(args.switch + "/overlay/switch", {"variant": "base", "mode": "wait"}, args.switch_headers)
        if code != 200 or not answer.get("ok"):
            raise Failed(f"could not prepare the switch: {code} {str(answer)[:600]}")
        print(f"prepared: {answer.get('timings')}")
        a = status(args).get("active")
    b = OTHER[a]
    print(f"A is {a!r} (live now), B is {b!r}")

    a1 = reading(args, f"A ({a})")
    flips = [flip(args, b)]
    rb = reading(args, f"B ({b})")
    flips.append(flip(args, a))
    a2 = reading(args, f"A ({a}) again")

    verdicts = [judge(i, a1, rb, a2) for i in range(len(PROMPTS))]
    for v in verdicts:
        print(f"\n{v['prompt'][:70]!r}\n  all answers agree for {v['agree_for']} tokens: {v['shared'][:80]!r}")
        for side, tokens in (v.get("chosen") or {}).items():
            print(f"    {side:9} chose {tokens}")
        print(f"  -> {v['verdict']}: {v['note']}")
    count = {k: sum(1 for v in verdicts if v["verdict"] == k) for k in ("clear", "not restored", "noise", "same", "unclear")}
    print(f"\n{count}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"server": args.server, "model": args.model, "A": a, "B": b, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "flips": flips, "verdicts": verdicts, "readings": {"A before": a1, "B": rb, "A after": a2}}, f, indent=1)
        print("saved", args.out)
    if count["not restored"]:
        print("FAIL: after flipping back, the model does not choose what it chose before.")
        return 1
    if count["clear"] >= args.need:
        print(f"PASS: {count['clear']} prompts part clearly by set, and flipping back restores the first choices.")
        return 0
    if count["clear"] == 0 and args.expect_parting:
        print("FAIL: the status says the sets changed, but no answer parted by set. The model is not reading what was written.")
        return 1
    print(f"CANNOT TELL: {count['clear']} prompts part clearly (wanted {args.need}). An overlay that leaves these "
          "prompts alone looks the same; this is not proof that the flip did nothing.")
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Failed as e:
        print(f"FAIL: {e}")
        sys.exit(1)
    except (CouldNotRun, urllib.error.URLError, OSError, KeyError, ValueError) as e:
        print(f"could not run: {type(e).__name__}: {e}")
        sys.exit(3)
