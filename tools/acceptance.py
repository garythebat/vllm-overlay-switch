#!/usr/bin/env python3
"""Acceptance test for the overlay switch, against a running server. Standard library only.

It asks nothing sensitive and needs no reference answers. It answers two questions.

Does the model read what a flip wrote?
    Two models with different weights weigh the next token differently at some point where
    both have read exactly the same words. The live set (A) is read several times, then the
    other set (B) after a flip, then A again after flipping back. For each prompt the answers
    are walked token by token while they all agree. At every position up to and including
    the first place they part, every answer was produced from an identical context. At each
    such position the contenders are the token chosen most often and its strongest rival, and
    an answer's margin is the log probability of the first minus that of the second. A
    position tells the sets apart when every A margin lies on one side and every B margin on
    the other; a prompt does when one of its positions does.

Does flipping back restore the model?
    Exactly, when the switch offers it: a full status carries one digest per rank of every
    tensor the model derived from the switched weights. It is read when the test begins and
    when it ends, and the two must be equal on every rank. The weights themselves are read
    back by the switch after every flip. Without that digest the answers have to do: A after
    the return must not stand apart from A before.

Why margins and several passes. The same request does not return the same numbers twice on
every deployment: kernels are chosen by timing, prefixes are cached, speculative tokens are
verified in batches. On the first hardware this ran on, identical requests moved a margin by
one to three logits. Two passes cannot tell that from a change of weights; five or more can.
Each request carries its own cache salt, so no answer reuses another's cached prefix and the
passes are independent.

Run it where the switch accepts calls from (by default: on the machine the server runs on).

    python3 tools/acceptance.py http://127.0.0.1:8000 MODEL_NAME
    python3 tools/acceptance.py http://127.0.0.1:8000 MODEL_NAME --passes 8 --out report.json
    python3 tools/acceptance.py http://127.0.0.1:8000 OTHER_MODEL --read-only --out reference.json

Exit status: 0 the flip is shown and the return restores; 1 it is shown not to work;
2 these prompts cannot tell the two sets apart (not proof of anything); 3 could not run.
"""

import argparse
import collections
import json
import math
import secrets
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
SIDES = ("A before", "B", "A after")
MIN_PASSES = 4


class CouldNotRun(Exception):
    """The test could not be carried out (exit status 3)."""


class Failed(Exception):
    """The switch itself reported a failure (exit status 1)."""


def call(url, body=None, headers=None, timeout=900, with_headers=False):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            got = (response.status, json.load(response))
            return (*got, dict(response.headers)) if with_headers else got
    except urllib.error.HTTPError as e:
        try:
            got = (e.code, json.load(e))
        except Exception:
            got = (e.code, {"ok": False, "error": f"HTTP {e.code}"})
        return (*got, dict(e.headers or {})) if with_headers else got


def ask(args, prompt, salt, base=None):
    """One answer with its tokens and their alternatives. `base` overrides the server's URL (another path prefix)."""
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": args.max_tokens,
            "temperature": 0, "logprobs": True, "top_logprobs": args.top}
    if salt:
        body["cache_salt"] = salt          # a fresh computation: nothing cached under another salt is reused
    if args.thinking_kwarg:
        body["chat_template_kwargs"] = {args.thinking_kwarg: False}
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    status, answer, response_headers = call((base or args.server) + "/v1/chat/completions", body, headers, with_headers=True)
    if status != 200:
        raise CouldNotRun(f"the server answered {status} to a chat request: {str(answer)[:300]}")
    choice = answer["choices"][0]
    tokens = [{"token": t["token"], "logprob": t["logprob"], "top": [[c["token"], c["logprob"]] for c in t.get("top_logprobs") or []]}
              for t in (choice.get("logprobs") or {}).get("content") or []]
    if not tokens:
        raise CouldNotRun("the server returned no per-token log probabilities: nothing to compare")
    side = next((v for k, v in response_headers.items() if k.lower() == "x-overlay-side"), None)
    return {"prompt": prompt, "text": choice["message"].get("content"), "tokens": tokens, "side": side}


def reading(args, label, tag):
    print(f"reading {label}: {args.passes} passes of {len(PROMPTS)} prompts", flush=True)
    return [[ask(args, prompt, None if args.no_cache_salt else f"{args.nonce}-{tag}-{k}-{i}")
             for i, prompt in enumerate(PROMPTS)] for k in range(args.passes)]


def status(args, full=False):
    code, answer = call(args.switch + "/overlay/status" + ("?full=1" if full else ""), headers=args.switch_headers, timeout=300)
    if code != 200 or not answer.get("ok"):
        raise Failed(f"the switch's status call answered {code}: {str(answer)[:400]}")
    return answer


def derived(answer):
    """{rank: digest} from a full status, or {} when the switch does not offer it."""
    out = {}
    for r in answer.get("ranks") or []:
        if isinstance(r, dict) and r.get("derived"):
            out[str(r.get("rank"))] = r["derived"]
    return out


def flip(args, variant):
    started = time.monotonic()
    code, answer = call(args.switch + "/overlay/switch", {"variant": variant, "mode": "wait"}, args.switch_headers)
    took = round(time.monotonic() - started, 3)
    if code != 200 or not answer.get("ok") or answer.get("active") != variant:
        raise Failed(f"the flip to {variant!r} answered {code}: {str(answer)[:600]}")
    print(f"flip to {variant}: {took} s as seen from here; the server's own timings: {answer.get('timings')}", flush=True)
    seen = status(args, full=True)
    if seen.get("active") != variant:
        raise Failed(f"after the flip a full status reads {seen.get('active')!r}, not {variant!r}")
    return {"variant": variant, "seconds": took, "timings": answer.get("timings"), "ranks": answer.get("ranks"),
            "derived": derived(seen)}


def apart(x, y):
    """Do two groups of numbers not overlap at all?"""
    return bool(x) and bool(y) and (min(x) > max(y) or min(y) > max(x))


def chance_apart(m, n):
    """How often two groups of m and n draws from ONE source do not overlap at all, by chance."""
    return 2 / math.comb(m + n, n)


def alternatives(entry):
    """{token: log probability} for one position of one answer: what the server returned, the chosen token included."""
    top = dict((token, logprob) for token, logprob in entry["top"])
    top.setdefault(entry["token"], entry["logprob"])
    return top


def margin_of(entry, first, second):
    """One answer's margin between two tokens at one position.

    A token outside the answer's returned list is at most as likely as the list's last entry:
    using that bound can only pull margins together.
    """
    top = alternatives(entry)
    floor = min(top.values())
    return round(top.get(first, floor) - top.get(second, floor), 3)


def margins_at(runs, n):
    """The contenders at position n and each answer's margin between them, by side.

    The first contender is the token chosen most often there; its rival is the other token the
    answers give the most probability to, taken over all of them.
    """
    chosen = collections.Counter(r["tokens"][n]["token"] for _, r in runs).most_common()
    first = chosen[0][0]
    weight = collections.Counter()
    for _, r in runs:
        for token, logprob in alternatives(r["tokens"][n]).items():
            if token != first:
                weight[token] += math.exp(logprob)
    if not weight:
        return None
    second = max(sorted(weight), key=weight.get)
    margins, picks = {}, {}
    for side, r in runs:
        margins.setdefault(side, []).append(margin_of(r["tokens"][n], first, second))
        picks.setdefault(side, []).append(r["tokens"][n]["token"])
    return {"position": n, "contenders": [first, second], "margins": {side: sorted(v) for side, v in margins.items()},
            "chosen": picks}


def gap(x, y):
    """The distance between two groups that do not overlap, else 0."""
    return max(min(x) - max(y), min(y) - max(x), 0) if x and y else 0


def judge(index, a1, b, a2):
    """One prompt: every position at which all answers had read the same words, and what the margins there show."""
    runs = [("A before", r[index]) for r in a1] + [("B", r[index]) for r in b] + [("A after", r[index]) for r in a2]
    tokens = [[t["token"] for t in r["tokens"]] for _, r in runs]
    n = 0
    while all(len(t) > n for t in tokens) and len({t[n] for t in tokens}) == 1:
        n += 1
    parted = all(len(t) > n for t in tokens)            # false when some answer simply ends at n
    out = {"prompt": PROMPTS[index], "agree_for": n, "shared": "".join(tokens[0][:n]), "parts": False, "a_moved": False,
           "positions": 0}
    best, at_parting = None, None
    for position in range(n + 1 if parted else n):
        seen = margins_at(runs, position)
        if seen is None:
            continue
        out["positions"] += 1
        seen["gap"] = gap(seen["margins"]["A before"] + seen["margins"]["A after"], seen["margins"]["B"])
        if position == n:
            at_parting = seen
        if best is None or seen["gap"] > best["gap"]:
            best = seen
    shown = best if best and best["gap"] > 0 else (at_parting or best)
    if shown:
        out.update(shown)
    out["parts"] = bool(best and best["gap"] > 0)
    out["a_moved"] = bool(at_parting and apart(at_parting["margins"]["A before"], at_parting["margins"]["A after"]))
    if out["a_moved"]:
        return {**out, "verdict": "not restored", "note": "where the answers part, A after the return stands apart from A before"}
    if out["parts"]:
        return {**out, "verdict": "clear", "note": f"at token {best['position']} every A margin lies on one side and every B "
                                                    f"margin on the other, {round(best['gap'], 3)} apart"}
    if not parted:
        if len({len(t) for t in tokens}) == 1:
            return {**out, "verdict": "same", "note": "all answers are identical and no margin tells the sets apart"}
        return {**out, "verdict": "unclear", "note": "some answers end where others go on"}
    if apart(at_parting["margins"]["A before"], at_parting["margins"]["B"]):
        return {**out, "verdict": "unclear", "note": "A before stands apart from B, but A after the return does not"}
    return {**out, "verdict": "noise", "note": "the margins of A and B overlap at every shared position"}


def main():
    parser = argparse.ArgumentParser(description="Acceptance test for the overlay switch (see the top of this file).")
    parser.add_argument("server", help="base URL of the OpenAI-compatible server, e.g. http://127.0.0.1:8000")
    parser.add_argument("model", help="the served model name")
    parser.add_argument("--switch", help="base URL of the switch routes if not the same as the server")
    parser.add_argument("--api-key", help="API key of the server, if it has one")
    parser.add_argument("--token", help="X-Overlay-Token, if the switch asks for one")
    parser.add_argument("--passes", type=int, default=6, help=f"readings of each set (default 6, at least {MIN_PASSES})")
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--top", type=int, default=20, help="how many alternatives to ask for per token (default 20)")
    parser.add_argument("--need", type=int, default=2, help="how many prompts must tell the sets apart (default 2)")
    parser.add_argument("--thinking-kwarg", default="enable_thinking",
                        help="chat template switch to turn thinking off; empty to send none")
    parser.add_argument("--no-cache-salt", action="store_true",
                        help="send no cache_salt (for a server that rejects the field); passes then share cached prefixes")
    parser.add_argument("--expect-parting", action="store_true",
                        help="the two sets are known to part on these prompts when served separately: "
                             "treat no parting at all as a failure, not as 'cannot tell'")
    parser.add_argument("--read-only", action="store_true",
                        help="only read the served model and save the answers (needs --out): no status call, no flip. "
                             "For a reference reading, e.g. of the overlay loaded as a model of its own")
    parser.add_argument("--out", help="write everything that was read to this JSON file")
    args = parser.parse_args()
    args.server = args.server.rstrip("/")
    args.switch = (args.switch or args.server).rstrip("/")
    args.switch_headers = {"X-Overlay-Token": args.token} if args.token else {}
    args.nonce = secrets.token_hex(4)
    if args.passes < MIN_PASSES:
        raise CouldNotRun(f"at least {MIN_PASSES} passes: with fewer, two readings of one set stand apart by chance too often "
                          f"(one time in {round(1 / chance_apart(3, 3))} at three passes)")

    code, models = call(args.server + "/v1/models", headers={"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}, timeout=30)
    served = [m.get("id") for m in models.get("data", [])] if code == 200 else []
    if args.model not in served:
        raise CouldNotRun(f"{args.server} serves {served}, not {args.model!r}: nothing asked")

    if args.read_only:
        if not args.out:
            raise CouldNotRun("--read-only needs --out: the reading is all it produces")
        only = reading(args, f"{args.model} as served", "only")
        with open(args.out, "w") as f:
            json.dump({"server": args.server, "model": args.model, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "passes": args.passes, "cache_salt": not args.no_cache_salt, "readings": {"only": only}}, f, indent=1)
        print(f"saved {args.out}: {args.passes} passes of {len(PROMPTS)} prompts, nothing flipped")
        return 0

    start = status(args, full=True)        # on a fresh server this is what the startup derived, before anything was read
    at_start = derived(start)
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

    a1 = reading(args, f"A ({a})", "a1")
    flips = [flip(args, b)]
    rb = reading(args, f"B ({b})", "b")
    flips.append(flip(args, a))
    a2 = reading(args, f"A ({a}) again", "a2")
    at_end = flips[-1]["derived"]

    verdicts = [judge(i, a1, rb, a2) for i in range(len(PROMPTS))]
    for v in verdicts:
        print(f"\n{v['prompt'][:70]!r}\n  all answers agree for {v['agree_for']} tokens: {v['shared'][:80]!r}")
        if "margins" in v:
            print(f"  at token {v['position']}, margin of {v['contenders'][0]!r} over {v['contenders'][1]!r}:")
            for side in SIDES:
                picks = dict(collections.Counter(v["chosen"][side]))
                print(f"    {side:9} {v['margins'][side]}  chose {picks}")
        print(f"  -> {v['verdict']}: {v['note']}")
    count = {k: sum(1 for v in verdicts if v["verdict"] == k) for k in ("clear", "not restored", "noise", "same", "unclear")}
    parts = sum(1 for v in verdicts if v["parts"])
    moved = sum(1 for v in verdicts if v["a_moved"])
    looked = sum(v["positions"] for v in verdicts)
    one_in = math.comb(3 * args.passes, args.passes) / 2
    print(f"\n{count}; told apart on {parts} prompts. {looked} positions were looked at; on one set a position tells "
          f"'apart' by chance one time in {round(one_in)}, so about one run in {max(1, round(one_in / max(looked, 1)))} "
          f"shows a single stray one")

    exact = None
    if at_start and at_end:
        differing = sorted(r for r in set(at_start) | set(at_end) if at_start.get(r) != at_end.get(r))
        same_as_b = sorted(r for r in at_start if flips[0]["derived"].get(r) == at_start[r])
        exact = not differing
        print(f"derived tensors, start against end: {'equal on every rank' if exact else 'DIFFERENT on rank ' + ', '.join(differing)} "
              f"({len(at_start)} ranks)")
        if same_as_b:
            print(f"  note: on rank {', '.join(same_as_b)} the digest did not change with the flip to B")
    else:
        print("derived tensors: this switch does not report a digest; restoration is judged from the answers alone")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"server": args.server, "model": args.model, "A": a, "B": b, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "passes": args.passes, "cache_salt": not args.no_cache_salt,
                       "derived": {"start": at_start, "after flip to B": flips[0]["derived"], "end": at_end},
                       "flips": flips, "verdicts": verdicts, "readings": {"A before": a1, "B": rb, "A after": a2}}, f, indent=1)
        print("saved", args.out)

    by_chance = chance_apart(args.passes, args.passes)
    if exact is False:
        print("FAIL: after flipping back, the tensors derived from the weights are not what they were when the test began.")
        return 1
    if exact is None and moved:
        print(f"FAIL: after flipping back, A stands apart from A before on {moved} prompt(s). On one set this happens by "
              f"chance about one time in {round(1 / by_chance)} per prompt: run it again, and believe it if it repeats.")
        return 1
    if exact and moved:
        print(f"note: A after the return stands apart from A before on {moved} prompt(s), while the weights and everything "
              f"derived from them are byte for byte what they were. That is this deployment's own drift between readings "
              f"(by chance alone: one time in {round(1 / by_chance)} per prompt).")
    if parts >= args.need:
        how = "the derived tensors are byte for byte what they were" if exact else "A after the return sits with A before"
        print(f"PASS: {parts} prompts tell the two sets apart, and after flipping back {how}.")
        return 0
    if parts == 0 and args.expect_parting:
        print("FAIL: the status says the sets changed, but no prompt tells them apart. The model is not reading what was written.")
        return 1
    print(f"CANNOT TELL: {parts} prompts tell the sets apart (wanted {args.need}). An overlay that leaves these "
          "prompts alone looks the same; this is not proof that the flip did nothing. More passes narrow it.")
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
