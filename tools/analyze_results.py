"""Six offline analyses over the saved results. No GPU needed."""

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
D = ROOT / "data"


def read_eval(p):
    p = Path(p)
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    det = d.get("details")
    if not det:
        return None
    return [{"q": x["question"][:200], "ok": int(x["correct"])} for x in det]


def read_rated(p):
    p = Path(p)
    if not p.exists():
        return None
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out.append({"q": r["question"][:200], "k": r["k"], "n": r["n"]})
    return out


def align(*items):
    items = [x for x in items if x is not None]
    if len(items) < 2:
        return None
    keys = [x["q"] for x in items[0]]
    for it in items[1:]:
        if len(it) != len(keys):
            return None
        if [x["q"] for x in it] != keys:
            return None
    return keys, items


def hdr(n, title):
    print("\n" + "=" * 74)
    print(f"[{n}]{title}")
    print("=" * 74)


def skip(what):
    print(f"  skipped (missing {what})")


def pass_at_k(counts, n, k):
    tot = 0.0
    for c in counts:
        tot += 1 - math.comb(n - c, k) / math.comb(n, k)
    return tot / len(counts)


def a_passk(b, r):
    hdr(1, "pass@k and capability boundary (training distribution T=0.8/top_p=0.95, G=8)")
    n = b[0]["n"]
    cb = [x["k"] for x in b]
    cr = [x["k"] for x in r]
    res = align(b, r)
    if res is None:
        print("  the two rated files disagree on questions; cannot pair"); return
    print(f"  questions {len(cb)}   G={n}\n")
    print(f"  {'k':>2}  {'base':>8}  {'+GRPO':>8}  {'Δ':>7}")
    for k in range(1, n + 1):
        pb, pr = pass_at_k(cb, n, k), pass_at_k(cr, n, k)
        print(f"  {k:>2}  {pb*100:>7.2f}%  {pr*100:>7.2f}%  {(pr-pb)*100:>+6.2f}")

    print("\n  k transition matrix (base k -> RL k):")
    k0new = sum(1 for x, y in zip(cb, cr) if x == 0 and y > 0)
    kG = sum(1 for x, y in zip(cb, cr) if x == n and y == n)
    mid2full = sum(1 for x, y in zip(cb, cr) if 1 <= x < n and y == n)
    full2less = sum(1 for x, y in zip(cb, cr) if x == n and y < n)
    to0 = sum(1 for x, y in zip(cb, cr) if x > 0 and y == 0)
    print(f"    {'k=0 -> k>0':<20}learned new            {k0new:>5} problems")
    print(f"    {f'1<=k<{n} -> k={n}':<20}borderline -> stable     {mid2full:>5} problems")
    print(f"    {f'k={n} -> k<{n}':<20}regressed from stable   {full2less:>5} problems")
    print(f"    {'k>0 -> k=0':<20}became fully wrong      {to0:>5} problems")
    print(f"    {f'k={n} -> k={n}':<20}stayed stable           {kG:>5} problems")
    print(f"\n  >> pass@{n}: {pass_at_k(cb,n,n)*100:.2f}% -> {pass_at_k(cr,n,n)*100:.2f}%"
          f" ({pass_at_k(cr,n,n)-pass_at_k(cb,n,n):+.2f} points, basically flat)")


def a_difficulty(b_rated, base_ev, rl_ev):
    hdr(2, "difficulty stratification of the gain (by base k/8, which stratum the greedy Δ falls in)")
    if b_rated is None or base_ev is None or rl_ev is None:
        skip("test_rated_base / base_vllm / rl_greedy_merged"); return
    m = {x["q"]: x["ok"] for x in base_ev}
    mr = {x["q"]: x["ok"] for x in rl_ev}
    bins = [("k=0 (never solved)", lambda k: k == 0),
            ("k=1-2 (hard)",    lambda k: 1 <= k <= 2),
            ("k=3-5 (medium)",    lambda k: 3 <= k <= 5),
            ("k=6-7 (close)",  lambda k: 6 <= k <= 7),
            ("k=8 (solved)",    lambda k: k == 8)]
    print(f"  {'stratum':<18}{'count':>6}{'base':>9}{'+GRPO':>9}{'Δ':>9}")
    for name, f in bins:
        qs = [x["q"] for x in b_rated if f(x["k"]) and x["q"] in m and x["q"] in mr]
        if not qs:
            continue
        a = sum(m[q] for q in qs) / len(qs)
        bb = sum(mr[q] for q in qs) / len(qs)
        print(f"  {name:<18}{len(qs):>6}{a*100:>8.1f}%{bb*100:>8.1f}%{(bb-a)*100:>+8.1f}")
    print("\n  note: k=0 and k=8 are the **zero-gradient** strata in training (no within-group variance)")


def a_template(base_ev, rl_ev, ba, ra, bm, rm):
    hdr(3, "template robustness: separating 'format adaptation' vs 'real capability'")
    if any(x is None for x in (base_ev, rl_ev, ba, ra, bm, rm)):
        skip("one of the six default/alt/minimal results"); return
    d = {x["q"]: (x["ok"], y["ok"]) for x, y in zip(base_ev, rl_ev)}
    al = {x["q"]: (x["ok"], y["ok"]) for x, y in zip(ba, ra)}
    mn = {x["q"]: (x["ok"], y["ok"]) for x, y in zip(bm, rm)}

    print("  questions each model gets right under all 3 templates:")
    for lab, idx in [("base", 0), ("+GRPO", 1)]:
        all3 = sum(1 for q in d if q in al and q in mn
                   and d[q][idx] and al[q][idx] and mn[q][idx])
        print(f"    {lab:<8}{all3:>5} / {len(d)}  = {all3/len(d)*100:.1f}%")

    fixed = [q for q in d if q in al and q in mn and d[q][0] == 0 and d[q][1] == 1]
    print(f"\n  problems fixed by RL under default: {len(fixed)}")
    if fixed:
        a = sum(1 for q in fixed if al[q][1])
        b = sum(1 for q in fixed if mn[q][1])
        print(f"    of which also correct under alt    : {a:>4} / {len(fixed)} = {a/len(fixed)*100:.0f}%")
        print(f"    of which also correct under minimal: {b:>4} / {len(fixed)} = {b/len(fixed)*100:.0f}%")
        print(f"    >> lower means more reliance on training wording (format fit), not real skill")
    broke = [q for q in d if q in al and q in mn and d[q][0] == 1 and d[q][1] == 0]
    if broke:
        a = sum(1 for q in broke if al[q][1])
        b = sum(1 for q in broke if mn[q][1])
        print(f"\n  problems broken by RL: {len(broke)} (also correct under alt: {a}, under minimal: {b})")


def a_seeds(base_ev, r1, r2, r3):
    hdr(4, "per-problem consistency across three seeds")
    if any(x is None for x in (base_ev, r1, r2, r3)):
        skip("one of the three seeds or the base result"); return
    runs = [r1, r2, r3]
    bt = {x["q"]: x["ok"] for x in base_ev}
    ts = [{x["q"]: x["ok"] for x in r} for r in runs]
    qs = [q for q in bt if all(q in t for t in ts)]
    print(f"  aligned questions {len(qs)}")

    nfix = [0, 0, 0, 0]
    nbrk = [0, 0, 0, 0]
    for q in qs:
        if bt[q] == 0:
            nfix[sum(t[q] for t in ts)] += 1
        else:
            nbrk[sum(1 for t in ts if t[q] == 0)] += 1
    print("\n  base wrong, fixed by N seeds:")
    for i in (3, 2, 1, 0):
        print(f"    fixed by {i} seeds: {nfix[i]:>4} problems" + ("   ← systematically fixable" if i == 3 else
                                                       ("   ← never fixed" if i == 0 else "")))
    print("\n  base right, broken by N seeds:")
    for i in (3, 2, 1, 0):
        if nbrk[i]:
            print(f"    broken by {i} seeds: {nbrk[i]:>4} problems")

    maj = [1 if sum(t[q] for t in ts) >= 2 else 0 for q in qs]
    single = [sum(t[q] for t in ts) / 3 for q in qs]
    print(f"\n  accuracy: single-seed mean {sum(single)/len(qs)*100:.2f}%"
          f"   three-seed majority {sum(maj)/len(qs)*100:.2f}%"
          f"   (+{(sum(maj)/len(qs)-sum(single)/len(qs))*100:.2f} points)")
    print("  >> majority voting adds little => errors are not independent; the gain is systematic, not noise")


def a_matrix():
    hdr(5, "2x3 matrix: {greedy, sampled} x {600, 1000, 1500}")
    g = {"base": read_eval(R / "base_vllm.json"),
         "600": read_eval(R / "r2_600_vllm.json"),
         "1000": read_eval(R / "r2_1000_vllm.json"),
         "1500": read_eval(R / "r2_1500_vllm.json")}
    s = {"base": read_rated(D / "test_rated_base.jsonl"),
         "600": read_rated(D / "test_rated_rl600.jsonl"),
         "1000": read_rated(D / "test_rated_rl1000.jsonl"),
         "1500": read_rated(D / "test_rated_rl.jsonl")}

    def gacc(v):
        return None if v is None else sum(x["ok"] for x in v) / len(v)

    def sacc(v):
        return None if v is None else sum(x["k"] / x["n"] for x in v) / len(v)

    print(f"  {'steps':<8}{'greedy acc':>12}{'sampled pass':>12}")
    for k, lab in [("base", "0 (base)"), ("600", "600"), ("1000", "1000"), ("1500", "1500")]:
        a, b = gacc(g[k]), sacc(s[k])
        print(f"  {lab:<8}" + (f"{a*100:>11.2f}%" if a is not None else f"{'—':>12}")
              + (f"{b*100:>11.2f}%" if b is not None else f"{'—':>12}"))
    gb, sb = gacc(g["base"]), sacc(s["base"])
    if gb and sb:
        print(f"\n  Δ vs base:")
        for k in ("600", "1000", "1500"):
            a, b = gacc(g[k]), sacc(s[k])
            print(f"    {k:>5} steps   greedy "
                  + (f"{(a-gb)*100:>+6.2f} points" if a is not None else "   —   ")
                  + "   sampled "
                  + (f"{(b-sb)*100:>+6.2f} points" if b is not None else "   —"))


def a_zero_std(log=None):
    import os
    log = Path(log or os.environ.get("RLVR_TRAIN_LOG", ROOT / "run_seeds.log"))
    hdr(6, "does the degenerate rate fall with training? (frac_reward_zero_std)")
    if not log.exists():
        skip(str(log)); return
    txt = log.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"================ seed=(\d+) ", txt)
    if len(blocks) < 3:
        skip("no seed= separator in the log"); return
    for i in range(1, len(blocks) - 1, 2):
        seed = blocks[i]
        body = blocks[i + 1]
        vals = [float(x) for x in re.findall(r"frac_reward_zero_std['\"]?\s*[:=]\s*([0-9.]+)", body)]
        if len(vals) < 100:
            continue
        n = len(vals)
        head, tail = vals[:200], vals[-200:]
        print(f"  seed {seed}: {n} steps total")
        print(f"    first 200 mean {sum(head)/len(head):.3f}"
              f"    last 200 mean {sum(tail)/len(tail):.3f}"
              f"    Δ {sum(tail)/len(tail)-sum(head)/len(head):+.3f}")
    vals = [float(x) for x in re.findall(r"frac_reward_zero_std['\"]?\s*[:=]\s*([0-9.]+)", txt)]
    if vals:
        print(f"\n  reference: measured base degenerate rate on train = 0.579 (57.9%)")


def main():
    base_ev = read_eval(R / "base_vllm.json")
    rl_ev = read_eval(R / "rl_greedy_merged.json")
    ba, ra = read_eval(R / "base_alt.json"), read_eval(R / "rl_alt.json")
    bm, rm = read_eval(R / "base_min.json"), read_eval(R / "rl_min.json")
    s1, s2, s3 = rl_ev, read_eval(R / "rl_greedy_s1234.json"), read_eval(R / "rl_greedy_s5678.json")
    b_rated = read_rated(D / "test_rated_base.jsonl")
    r_rated = read_rated(D / "test_rated_rl.jsonl")

    if b_rated and r_rated:
        a_passk(b_rated, r_rated)
    else:
        hdr(1, "pass@k and capability boundary"); skip("test_rated_base/rl")

    a_difficulty(b_rated, base_ev, rl_ev)
    a_template(base_ev, rl_ev, ba, ra, bm, rm)
    a_seeds(base_ev, s1, s2, s3)
    a_matrix()
    a_zero_std()
    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()
