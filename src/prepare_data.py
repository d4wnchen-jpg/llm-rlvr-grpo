"""Download and prepare GSM8K / MBPP into the local JSONL format."""

import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

PROXY = "https://gh-proxy.com/"

SOURCES = {
    "gsm8k": {
        "base": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data",
        "files": ["train.jsonl"],
    },
    "code": {
        "base": "https://raw.githubusercontent.com/google-research/google-research/master/mbpp",
        "files": ["mbpp.jsonl", "sanitized-mbpp.json"],
    },
}

GSM8K_INSTRUCTION = ("Please reason step by step, and put your final answer "
                     "within \\boxed{}.")

CODE_SYSTEM_PROMPT = (
    "You are a Python coding assistant. "
    "Write the complete function to solve the problem. "
    "Output only the code, wrapped in a ```python code block."
)


def fetch(name: str, base: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / name
    if dst.exists() and dst.stat().st_size > 1000:
        print(f"  using cached {dst}")
        return dst
    for url in [f"{base}/{name}", f"{PROXY}{base}/{name}"]:
        try:
            print(f"  downloading {url[:72]}...")
            with urllib.request.urlopen(url, timeout=90) as r:
                data = r.read()
            if len(data) > 1000:
                dst.write_bytes(data)
                print(f"  ✓ {len(data)} bytes")
                return dst
        except Exception as e:
            print(f"  failed: {type(e).__name__}")
    raise RuntimeError(f"cannot download {name}, place it manually at {dst}")


def build_gsm8k(raw_dir: Path, out_path: Path, limit=None, seed=42):
    path = fetch("train.jsonl", SOURCES["gsm8k"]["base"], raw_dir)
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]

    random.seed(seed)
    random.shuffle(rows)
    if limit:
        rows = rows[:limit]
        print(f"  pilot mode: {len(rows)} problems")

    n, skipped = 0, 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            ans = r.get("answer", "")
            if "####" not in ans:
                skipped += 1
                continue
            gold = ans.split("####")[-1].strip()
            f.write(json.dumps({
                "prompt": [{"role": "user",
                            "content": f"{r['question'].strip()}\n\n{GSM8K_INSTRUCTION}"}],
                "answer": gold,
                "question": r["question"],
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"  ✓ GSM8K: {n} rows (skipped {skipped})")
    return n


def extract_signature(ref_code: str):
    m = re.search(r"^(\s*def\s+\w+\s*\([^)]*\)\s*(?:->[^:]+)?:)", ref_code or "", re.M)
    return m.group(1).strip() if m else None


def build_gsm8k_test(raw_dir: Path, out_path: Path, limit=None):
    path = fetch("test.jsonl", SOURCES["gsm8k"]["base"], raw_dir)
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    n, skipped = 0, 0
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            ans = r.get("answer", "")
            if "####" not in ans:
                skipped += 1
                continue
            f.write(json.dumps({
                "prompt": [{"role": "user",
                            "content": f"{r['question'].strip()}\n\n{GSM8K_INSTRUCTION}"}],
                "answer": ans.split("####")[-1].strip(),
                "question": r["question"],
            }, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(out_path)
    print(f"  ✓ GSM8K test: {n} rows (skipped {skipped})")
    return n


def build_mbpp(raw_dir: Path, out_path: Path, limit=None, seed=42):
    full_p = fetch("mbpp.jsonl", SOURCES["code"]["base"], raw_dir)
    san_p = fetch("sanitized-mbpp.json", SOURCES["code"]["base"], raw_dir)

    full = [json.loads(l) for l in full_p.open(encoding="utf-8") if l.strip()]
    san_ids = {d["task_id"] for d in json.loads(san_p.read_text(encoding="utf-8"))}

    safe = [d for d in full if d["task_id"] not in san_ids]
    print(f"  MBPP full={len(full)}  excluded sanitized={len(san_ids)}  → safe set={len(safe)}")

    safe.sort(key=lambda d: d["task_id"])
    random.seed(seed)
    random.shuffle(safe)
    if limit:
        safe = safe[:limit]
        print(f"  pilot mode: {len(safe)} problems")

    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in safe:
            tests = list(r.get("test_list") or [])
            setup = (r.get("test_setup_code") or "").strip()
            if setup:
                tests = [setup + "\n" + t for t in tests]
            if not tests:
                continue

            problem = r["text"].strip()
            sig = extract_signature(r.get("code", ""))
            prompt = problem
            if sig:
                prompt += f"\n\nYour function should have this signature:\n{sig}"
            prompt += "\n\nWrite the complete function:"

            f.write(json.dumps({
                "prompt": [{"role": "system", "content": CODE_SYSTEM_PROMPT},
                           {"role": "user", "content": prompt}],
                "test_list": tests,
                "problem": problem,
                "task_id": r["task_id"],
            }, ensure_ascii=False) + "\n")
            n += 1
    print(f"  ✓ MBPP: {n} rows")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gsm8k", choices=["gsm8k", "code"])
    ap.add_argument("--split", default="train", choices=["train", "test"],
                    help="★ test (gsm8k only): write an eval-set copy in eval_grpo order, "
                         "for filter_by_difficulty.py to use as --data. No shuffling")
    ap.add_argument("--out", default=None, help="default: data/<task>_<split>.jsonl")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--limit", type=int, default=None, help="take only N problems (pilot)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.out is None:
        args.out = {("gsm8k", "train"): "data/gsm8k_train.jsonl",
                    ("gsm8k", "test"): "data/gsm8k_test.jsonl",
                    ("code", "train"): "data/mbpp_train.jsonl"}[(args.task, args.split)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)

    print(f"preparing data: task={args.task} split={args.split}")
    if args.task == "gsm8k":
        if args.split == "test":
            build_gsm8k_test(raw_dir, out_path, args.limit)
        else:
            build_gsm8k(raw_dir, out_path, args.limit, args.seed)
    else:
        if args.split == "test":
            raise SystemExit("✗ code task has no test split yet")
        build_mbpp(raw_dir, out_path, args.limit, args.seed)

    with out_path.open(encoding="utf-8") as f:
        first = json.loads(f.readline())
    print("\nExample:")
    print("-" * 60)
    print(first["prompt"][-1]["content"][:350])
    print("-" * 60)
    if "answer" in first:
        print(f"gold: {first['answer']}")
    if "test_list" in first:
        print(f"test cases: {len(first['test_list'])}")
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
