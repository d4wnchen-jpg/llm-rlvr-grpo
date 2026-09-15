# -*- coding: utf-8 -*-
"""代码 RL 的 reward 函数：执行生成代码 + 测试用例，返回得分。

设计要点：
  1. 剥掉 <think> 和 markdown 围栏，只取代码
  2. 在子进程里执行（带超时），避免死循环/崩溃污染训练进程
  3. 支持两种粒度：
     - binary: 全部测试通过 = 1，否则 0
     - partial: 通过的测试比例（信号更稠密，早期更好学）

本文件可直接运行自测：python reward.py
"""
import os
import re
import subprocess
import sys
import tempfile

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE_LANG_RE = re.compile(r"^[a-zA-Z0-9+#]*\s*\n")

DEFAULT_TIMEOUT = 6.0


def extract_code(text: str) -> str:
    """从模型输出里剥出纯代码（去思考块、去 markdown 围栏）。"""
    if not text:
        return ""
    text = THINK_RE.sub("", text)
    if "```" in text:
        parts = text.split("```")
        if len(parts) > 1:
            code = parts[1]
            code = FENCE_LANG_RE.sub("", code, count=1)
            return code.strip()
    return text.strip()


def _run_program(source: str, timeout: float) -> bool:
    """在子进程里跑一段 Python，返回是否正常结束（returncode == 0）。"""
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(source)
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            timeout=timeout,
        )
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def compute_reward(code: str, tests: list, mode: str = "partial",
                   timeout: float = DEFAULT_TIMEOUT) -> float:
    """计算单个样本的 reward。

    Args:
        code:  模型生成的代码（已剥壳或原始输出）
        tests: 测试用例列表（每项是一段可独立执行的 assert / 语句）
        mode:  "binary"（全过=1）或 "partial"（通过比例）
        timeout: 单次执行的超时（秒）

    Returns:
        float in [0, 1]
    """
    if not code or not tests:
        return 0.0

    # --- 先整体跑一次（快路径）---
    full_source = code + "\n\n" + "\n".join(tests)
    if _run_program(full_source, timeout):
        return 1.0

    if mode == "binary":
        return 0.0

    # --- partial：逐个测试跑，统计通过比例 ---
    passed = 0
    for t in tests:
        if _run_program(code + "\n\n" + t, timeout):
            passed += 1
    return passed / len(tests)


def make_reward_fn(mode: str = "partial", timeout: float = DEFAULT_TIMEOUT):
    """生成 TRL GRPOTrainer 用的 reward 函数（代码任务）。

    TRL 会调用 reward_fn(completions, **dataset_columns)，
    所以数据集里需要有 test_list 这一列。
    """

    def reward_fn(completions, test_list=None, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            # TRL 的 completion 可能是 str，也可能是 message 列表
            if isinstance(completion, list):
                text = completion[-1].get("content", "") if completion else ""
            else:
                text = completion
            code = extract_code(text)
            tests = test_list[i] if test_list else []
            rewards.append(compute_reward(code, tests, mode=mode, timeout=timeout))
        return rewards

    return reward_fn


# ============================================================
# GSM8K（数学）reward：正则提取最终答案，与 gold 比对
#   —— 比代码 reward 简单可靠得多（不用执行代码、微秒级）
# ============================================================

BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
HASH_RE = re.compile(r"####\s*([^\n]+)")
NUM_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def normalize_number(s: str) -> str:
    """把答案字符串标准化（去 LaTeX 残留 / 逗号 / 美元符 / 尾部句点，统一数值格式）。"""
    if s is None:
        return ""
    s = str(s).strip()
    # 模型常写成 \(\boxed{\$108}\)，BOXED_RE 抓到的是 "\$108"（带反斜杠），
    # 直接 float() 会失败 → 预测值变成 "\108" ≠ "108" → reward 被误判为 0。
    s = re.sub(r"\\(?:text|mathrm|mathbf|mbox|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\", "").replace(",", "").replace("$", "").replace("%", "").strip()
    s = s.rstrip(".").strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError):
        return s.lower()


def extract_gsm8k_answer(text: str) -> str:
    """从模型输出里提取最终答案（多格式兜底）。

    依次尝试：\\boxed{} → #### X → 最后一个数字
    """
    if not text:
        return ""
    text = THINK_RE.sub("", text).strip()

    # 1) \boxed{X}（最标准）
    m = BOXED_RE.findall(text)
    if m:
        return normalize_number(m[-1])

    # 2) #### X
    m = HASH_RE.findall(text)
    if m:
        return normalize_number(m[-1])

    # 3) 兜底：最后一个数字
    nums = NUM_RE.findall(text)
    if nums:
        return normalize_number(nums[-1])

    return ""


def compute_gsm8k_reward(text: str, gold: str) -> float:
    """GSM8K 的二元 reward：最终答案对 = 1，否则 0。"""
    pred = extract_gsm8k_answer(text)
    gold_n = normalize_number(gold)
    if not pred or not gold_n:
        return 0.0
    return 1.0 if pred == gold_n else 0.0


def make_gsm8k_reward_fn():
    """生成 TRL GRPOTrainer 用的 GSM8K reward 函数。

    数据集里需要有 answer 这一列（gold 答案）。
    """

    def reward_fn(completions, answer=None, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            if isinstance(completion, list):
                text = completion[-1].get("content", "") if completion else ""
            else:
                text = completion
            gold = answer[i] if answer else ""
            rewards.append(compute_gsm8k_reward(text, gold))
        return rewards

    return reward_fn


# ============================================================
# 自测：验证 reward 函数工作正常
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("测试 extract_code")
    print("=" * 60)
    cases = [
        ("```python\ndef f():\n    return 1\n```", "def f():\n    return 1"),
        ("<think>hmm</think>```python\nx = 1\n```", "x = 1"),
        ("def g():\n    return 2", "def g():\n    return 2"),
    ]
    for raw, expected in cases:
        got = extract_code(raw)
        ok = got == expected
        print(f"{'✓' if ok else '✗'} {raw[:35]!r} -> {got!r}")

    print()
    print("=" * 60)
    print("测试 compute_reward")
    print("=" * 60)
    tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"]

    # 1) 全对的代码
    good = "def add(a, b):\n    return a + b"
    print(f"全对代码:      binary={compute_reward(good, tests, 'binary'):.2f}  "
          f"partial={compute_reward(good, tests, 'partial'):.2f}   (期望 1.00 / 1.00)")

    # 2) 部分对（负数出错）
    semi = "def add(a, b):\n    return abs(a) + abs(b)"
    print(f"部分对代码:    binary={compute_reward(semi, tests, 'binary'):.2f}  "
          f"partial={compute_reward(semi, tests, 'partial'):.2f}   (期望 0.00 / 0.66)")

    # 3) 全错
    bad = "def add(a, b):\n    return 42"
    print(f"全错代码:      binary={compute_reward(bad, tests, 'binary'):.2f}  "
          f"partial={compute_reward(bad, tests, 'partial'):.2f}   (期望 0.00 / 0.00)")

    # 4) 死循环（测超时保护）
    loop = "def add(a, b):\n    while True:\n        pass"
    print(f"死循环代码:    binary={compute_reward(loop, tests, 'binary', timeout=2):.2f}  "
          f"(期望 0.00，且不卡住)")

    # 5) 语法错误
    syntax = "def add(a, b)\n    return a + b"
    print(f"语法错误:      binary={compute_reward(syntax, tests, 'binary'):.2f}  (期望 0.00)")

    # 6) 用 TRL 风格调用
    print()
    print("模拟 TRL 调用: ", end="")
    fn = make_reward_fn(mode="partial")
    out = fn(
        completions=[f"```python\n{good}\n```", f"```python\n{bad}\n```"],
        test_list=[tests, tests],
    )
    print(out, " (期望 [1.0, 0.0])")

    print()
    print("=" * 60)
    print("测试 GSM8K reward")
    print("=" * 60)
    gsm_cases = [
        ("Let me compute...\nSo the answer is \\boxed{72}", "72", 1.0),
        ("Reasoning here.\n#### 1,234", "1234", 1.0),
        ("The total is 18 dollars. Therefore \\boxed{$18}", "18", 1.0),
        ("I think the answer is 5.5", "5.5", 1.0),
        ("After thinking, the result is 99", "72", 0.0),
        ("<think>maybe 72</think>\\boxed{13}", "72", 0.0),
        ("", "72", 0.0),
        # ← 实测模型最常见的写法：\(...\) 包裹 + 转义美元符
        ("So the total is \\(\\boxed{\\$108}\\).", "108", 1.0),
        ("\\[ \\boxed{24} \\]", "24", 1.0),
    ]
    for text, gold, expect in gsm_cases:
        got = compute_gsm8k_reward(text, gold)
        mark = "✓" if got == expect else "✗"
        print(f"{mark} gold={gold:6} pred={extract_gsm8k_answer(text):6} "
              f"reward={got:.1f} (期望 {expect:.1f})")

    print()
    print("模拟 TRL 调用（GSM8K）: ", end="")
    gfn = make_gsm8k_reward_fn()
    out2 = gfn(
        completions=["\\boxed{72}", "the answer is 99"],
        answer=["72", "72"],
    )
    print(out2, " (期望 [1.0, 0.0])")

    print()
    print("=" * 60)
    print("✓ reward 函数验证完成（代码 + GSM8K，可安全用于 RL 训练）")
    print("=" * 60)
