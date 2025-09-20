import ast, math, re, subprocess, sys, tempfile, textwrap

def exact_match(pred: str, gold: str) -> bool:
    def lastnum(s):
        toks = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",",""))
        return toks[-1] if toks else s.strip()
    return lastnum(pred) == lastnum(gold)

def run_python_tests(code: str, tests_src: str) -> bool:
    src = code + "\n" + tests_src + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(src); path=f.name
    try:
        out = subprocess.run([sys.executable, path], capture_output=True, timeout=5)
        return out.returncode == 0
    except Exception:
        return False

def eval_game24(expr: str) -> bool:
    try:
        val = eval(expr, {"__builtins__":{}}, {})
        return abs(val - 24) < 1e-6
    except Exception:
        return False

def subset_unit_tests(tests_src: str, k: int = 3) -> str:
    """
    Heuristically pick up to k test assertions while keeping definitions/imports.
    Falls back to the full test file when structure is not obvious.
    """
    lines = tests_src.splitlines()
    asserts = [ln for ln in lines if "assert" in ln]
    if not asserts:
        return tests_src
    header = [ln for ln in lines if "assert" not in ln]
    return "\n".join(header + asserts[:k]) + "\n"
