"""Reward functions for the code task: run generated code against tests."""

import os
import re
import subprocess
import sys
import tempfile

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE_LANG_RE = re.compile(r"^[a-zA-Z0-9+#]*\s*\n")

DEFAULT_TIMEOUT = 6.0


def extract_code(text: str) -> str:
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
    if not code or not tests:
        return 0.0

    full_source = code + "\n\n" + "\n".join(tests)
    if _run_program(full_source, timeout):
        return 1.0

    if mode == "binary":
        return 0.0

    passed = 0
    for t in tests:
        if _run_program(code + "\n\n" + t, timeout):
            passed += 1
    return passed / len(tests)


def make_reward_fn(mode: str = "partial", timeout: float = DEFAULT_TIMEOUT):

    def reward_fn(completions, test_list=None, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            if isinstance(completion, list):
                text = completion[-1].get("content", "") if completion else ""
            else:
                text = completion
            code = extract_code(text)
            tests = test_list[i] if test_list else []
            rewards.append(compute_reward(code, tests, mode=mode, timeout=timeout))
        return rewards

    return reward_fn


BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
HASH_RE = re.compile(r"####\s*([^\n]+)")
NUM_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


def normalize_number(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\\(?:text|mathrm|mathbf|mbox|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    s = s.replace("\\", "").replace(",", "").replace("$", "").replace("%", "").strip()
    s = s.rstrip(".").strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError):
        return s.lower()


def extract_gsm8k_answer(text: str) -> str:
    if not text:
        return ""
    text = THINK_RE.sub("", text).strip()

    m = BOXED_RE.findall(text)
    if m:
        return normalize_number(m[-1])

    m = HASH_RE.findall(text)
    if m:
        return normalize_number(m[-1])

    nums = NUM_RE.findall(text)
    if nums:
        return normalize_number(nums[-1])

    return ""


def compute_gsm8k_reward(text: str, gold: str) -> float:
    pred = extract_gsm8k_answer(text)
    gold_n = normalize_number(gold)
    if not pred or not gold_n:
        return 0.0
    return 1.0 if pred == gold_n else 0.0


def make_gsm8k_reward_fn():

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


if __name__ == "__main__":
    print("=" * 60)
    print("test extract_code")
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
    print("test compute_reward")
    print("=" * 60)
    tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"]

    good = "def add(a, b):\n    return a + b"
    print(f"{'all correct:':<16}binary={compute_reward(good, tests, 'binary'):.2f}  "
          f"partial={compute_reward(good, tests, 'partial'):.2f}   (expected 1.00 / 1.00)")

    semi = "def add(a, b):\n    return abs(a) + abs(b)"
    print(f"{'partial:':<16}binary={compute_reward(semi, tests, 'binary'):.2f}  "
          f"partial={compute_reward(semi, tests, 'partial'):.2f}   (expected 0.00 / 0.66)")

    bad = "def add(a, b):\n    return 42"
    print(f"{'all wrong:':<16}binary={compute_reward(bad, tests, 'binary'):.2f}  "
          f"partial={compute_reward(bad, tests, 'partial'):.2f}   (expected 0.00 / 0.00)")

    loop = "def add(a, b):\n    while True:\n        pass"
    print(f"{'infinite loop:':<16}binary={compute_reward(loop, tests, 'binary', timeout=2):.2f}  "
          f"(expected 0.00, does not hang)")

    syntax = "def add(a, b)\n    return a + b"
    print(f"{'syntax error:':<16}binary={compute_reward(syntax, tests, 'binary'):.2f}  (expected 0.00)")

    print()
    print("mock TRL call: ", end="")
    fn = make_reward_fn(mode="partial")
    out = fn(
        completions=[f"```python\n{good}\n```", f"```python\n{bad}\n```"],
        test_list=[tests, tests],
    )
    print(out, " (expected [1.0, 0.0])")

    print()
    print("=" * 60)
    print("test GSM8K reward")
    print("=" * 60)
    gsm_cases = [
        ("Let me compute...\nSo the answer is \\boxed{72}", "72", 1.0),
        ("Reasoning here.\n#### 1,234", "1234", 1.0),
        ("The total is 18 dollars. Therefore \\boxed{$18}", "18", 1.0),
        ("I think the answer is 5.5", "5.5", 1.0),
        ("After thinking, the result is 99", "72", 0.0),
        ("<think>maybe 72</think>\\boxed{13}", "72", 0.0),
        ("", "72", 0.0),
        ("So the total is \\(\\boxed{\\$108}\\).", "108", 1.0),
        ("\\[ \\boxed{24} \\]", "24", 1.0),
    ]
    for text, gold, expect in gsm_cases:
        got = compute_gsm8k_reward(text, gold)
        mark = "✓" if got == expect else "✗"
        print(f"{mark} gold={gold:6} pred={extract_gsm8k_answer(text):6} "
              f"reward={got:.1f} (expected {expect:.1f})")

    print()
    print("mock TRL call (GSM8K): ", end="")
    gfn = make_gsm8k_reward_fn()
    out2 = gfn(
        completions=["\\boxed{72}", "the answer is 99"],
        answer=["72", "72"],
    )
    print(out2, " (expected [1.0, 0.0])")

    print()
    print("=" * 60)
    print("✓ reward function self-test passed (code + GSM8K, safe for RL training)")
    print("=" * 60)
