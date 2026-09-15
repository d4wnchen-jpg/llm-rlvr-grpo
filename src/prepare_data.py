# -*- coding: utf-8 -*-
"""数据准备：GSM8K（数学）和 MBPP（代码）。

用法:
    python prepare_data.py --task gsm8k          # 7473 题（推荐先做）
    python prepare_data.py --task code           # 547 题
    python prepare_data.py --task gsm8k --limit 100   # pilot

数据源（GitHub 直下，不依赖 HuggingFace）:
    GSM8K: openai/grade-school-math
    MBPP : google-research/google-research

★ 防污染设计
    GSM8K: 训练用 train(7473)，评测用 test(1319)          ← 天然分离
    MBPP : 训练用 full − sanitized(547)，评测用 EvalPlus(MBPP+)
           （EvalPlus 的 MBPP 基础集就是 sanitized 427 题，脚本自动排除）
"""
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
    """下载数据文件（带缓存 + gh-proxy 兜底）。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / name
    if dst.exists() and dst.stat().st_size > 1000:
        print(f"  使用缓存 {dst}")
        return dst
    for url in [f"{base}/{name}", f"{PROXY}{base}/{name}"]:
        try:
            print(f"  下载 {url[:72]}...")
            with urllib.request.urlopen(url, timeout=90) as r:
                data = r.read()
            if len(data) > 1000:
                dst.write_bytes(data)
                print(f"  ✓ {len(data)} bytes")
                return dst
        except Exception as e:
            print(f"  失败: {type(e).__name__}")
    raise RuntimeError(f"无法下载 {name}，请手动放到 {dst}")


# ============================================================
# GSM8K
# ============================================================
def build_gsm8k(raw_dir: Path, out_path: Path, limit=None, seed=42):
    path = fetch("train.jsonl", SOURCES["gsm8k"]["base"], raw_dir)
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]

    random.seed(seed)
    random.shuffle(rows)
    if limit:
        rows = rows[:limit]
        print(f"  pilot 模式：取 {len(rows)} 题")

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
    print(f"  ✓ GSM8K: {n} 条（跳过 {skipped}）")
    return n


# ============================================================
# MBPP（代码）
# ============================================================
def extract_signature(ref_code: str):
    m = re.search(r"^(\s*def\s+\w+\s*\([^)]*\)\s*(?:->[^:]+)?:)", ref_code or "", re.M)
    return m.group(1).strip() if m else None


def build_gsm8k_test(raw_dir: Path, out_path: Path, limit=None):
    """评测集 GSM8K test 的**同格式**副本，给 filter_by_difficulty.py 当 --data 用。

    与 train 的关键差别：**不打乱顺序** —— eval_grpo.py 的 load_gsm8k_test 按
    test.jsonl 的原始顺序读，两边顺序一致，才能按 index 对齐做配对分析
    （打乱了就对不上号，配对检验全废）。

    ★ 原子写：先写 .tmp 再 os.replace。这个文件马上要被 50 分钟的评分任务读，
      中途崩溃留下半截文件（题数对不上）代价很大；'w' 是 open 时就截断的。
    """
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
    tmp.replace(out_path)       # 原子替换：要么是旧的完整文件，要么是新的
    print(f"  ✓ GSM8K test: {n} 条（跳过 {skipped}）")
    return n


def build_mbpp(raw_dir: Path, out_path: Path, limit=None, seed=42):
    full_p = fetch("mbpp.jsonl", SOURCES["code"]["base"], raw_dir)
    san_p = fetch("sanitized-mbpp.json", SOURCES["code"]["base"], raw_dir)

    full = [json.loads(l) for l in full_p.open(encoding="utf-8") if l.strip()]
    san_ids = {d["task_id"] for d in json.loads(san_p.read_text(encoding="utf-8"))}

    safe = [d for d in full if d["task_id"] not in san_ids]
    print(f"  MBPP full={len(full)}  排除 sanitized={len(san_ids)}  → 安全集={len(safe)}")

    safe.sort(key=lambda d: d["task_id"])
    random.seed(seed)
    random.shuffle(safe)
    if limit:
        safe = safe[:limit]
        print(f"  pilot 模式：取 {len(safe)} 题")

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
    print(f"  ✓ MBPP: {n} 条")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gsm8k", choices=["gsm8k", "code"])
    ap.add_argument("--split", default="train", choices=["train", "test"],
                    help="★ test（仅 gsm8k）：产出与 eval_grpo 顺序一致的评测集副本，"
                         "供 filter_by_difficulty.py 当 --data 用。不打乱、不 shuffle")
    ap.add_argument("--out", default=None, help="默认 data/<task>_<split>.jsonl")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--limit", type=int, default=None, help="只取 N 题（pilot）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.out is None:
        args.out = {("gsm8k", "train"): "data/gsm8k_train.jsonl",
                    ("gsm8k", "test"): "data/gsm8k_test.jsonl",
                    ("code", "train"): "data/mbpp_train.jsonl"}[(args.task, args.split)]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)

    print(f"准备数据：task={args.task} split={args.split}")
    if args.task == "gsm8k":
        if args.split == "test":
            build_gsm8k_test(raw_dir, out_path, args.limit)
        else:
            build_gsm8k(raw_dir, out_path, args.limit, args.seed)
    else:
        if args.split == "test":
            raise SystemExit("✗ code 任务还没有 test 分支")
        build_mbpp(raw_dir, out_path, args.limit, args.seed)

    # 展示一条
    with out_path.open(encoding="utf-8") as f:
        first = json.loads(f.readline())
    print("\n示例:")
    print("-" * 60)
    print(first["prompt"][-1]["content"][:350])
    print("-" * 60)
    if "answer" in first:
        print(f"gold: {first['answer']}")
    if "test_list" in first:
        print(f"测试用例数: {len(first['test_list'])}")
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
