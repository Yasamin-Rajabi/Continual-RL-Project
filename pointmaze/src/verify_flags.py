"""Check that every CLI flag one module passes to another actually exists.

A wrong flag name is not caught by any import check and does not surface until
the subprocess is launched -- which, in a ten-task chain, can be an hour in.
This parses the ``add_argument`` calls in each trainer and the flag strings
emitted by the orchestrator and the notebook, and compares them.
"""
from __future__ import annotations

import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent


def declared_flags(path: pathlib.Path) -> set:
    """Collect every option string passed to add_argument in a module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    flags = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "add_argument"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.startswith("--"):
                    flags.add(arg.value)
                    # BooleanOptionalAction also accepts the --no- form.
                    for kw in node.keywords:
                        if (
                            kw.arg == "action"
                            and isinstance(kw.value, ast.Attribute)
                            and kw.value.attr == "BooleanOptionalAction"
                        ):
                            flags.add("--no-" + arg.value[2:])
    return flags


FLAG_RE = re.compile(r'(--[a-zA-Z][a-zA-Z0-9\-]*)')


def emitted_flags(path: pathlib.Path, marker_functions=None) -> set:
    """Collect flag-looking strings emitted from a module's source."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    found = set()

    def scan(node):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                for m in FLAG_RE.findall(sub.value):
                    found.add(m)
            elif isinstance(sub, ast.JoinedStr):
                for value in sub.values:
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        for m in FLAG_RE.findall(value.value):
                            found.add(m)

    if marker_functions:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in marker_functions:
                scan(node)
    else:
        scan(tree)
    return found


def check(name, emitted, declared, ignore=()):  # -> list of problems
    problems = []
    for flag in sorted(emitted):
        if flag in ignore:
            continue
        if flag not in declared:
            problems.append(f"{name}: emits {flag} but the target does not declare it")
    return problems


def main() -> int:
    ours = declared_flags(ROOT / "run_sac_continual.py")
    base = declared_flags(ROOT / "baselines" / "run_baseline.py")
    scratch = declared_flags(ROOT / "scratch_baselines.py")
    bench = declared_flags(ROOT / "run_continual_benchmark.py")

    problems = []
    problems += check(
        "run_continual_benchmark -> run_sac_continual",
        emitted_flags(ROOT / "run_continual_benchmark.py", {"_ours_command"}),
        ours,
    )
    problems += check(
        "run_continual_benchmark -> run_baseline",
        emitted_flags(ROOT / "run_continual_benchmark.py", {"_baseline_command"}),
        base,
    )

    notebook = ROOT / "pointmaze_kaggle.ipynb"
    if notebook.is_file():
        import json

        cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        text = "\n".join(
            "".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code"
        )
        emitted = set(FLAG_RE.findall(text))
        allowed = bench | scratch | ours | base | {
            "--quiet", "--upgrade", "--no-cache-dir", "--version", "--help",
        }
        for flag in sorted(emitted):
            if flag not in allowed:
                problems.append(f"notebook: passes {flag}, which no script declares")

    print(
        f"declared flags -- ours:{len(ours)} baselines:{len(base)} "
        f"scratch:{len(scratch)} benchmark:{len(bench)}"
    )
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
        return 1
    print("flag verification OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
