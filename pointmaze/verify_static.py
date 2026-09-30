"""Static verification that does not need torch, gymnasium or a GPU.

Checks, for every module in the package:

1. it compiles;
2. every ``from <local module> import name`` resolves to a top-level
   definition in that module (catches renames and typos that would otherwise
   only surface at task 7 of a 10-task chain);
3. no local module imports a name from a module that does not define it;
4. every module referenced by an import actually exists.

Run it with ``python verify_static.py``.
"""
from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent


def local_modules():
    mods = {}
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        name = ".".join(rel.with_suffix("").parts)
        if name.endswith(".__init__"):
            name = name[: -len(".__init__")]
        mods[name] = path
    return mods


def top_level_names(tree: ast.Module) -> set:
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            names.add(elt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.Try):
            for sub in node.body + node.orelse + node.finalbody:
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    for alias in sub.names:
                        names.add(alias.asname or alias.name.split(".")[0])
                elif isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        if isinstance(target, ast.Name):
                            names.add(target.id)
            for handler in node.handlers:
                for sub in handler.body:
                    if isinstance(sub, ast.Assign):
                        for target in sub.targets:
                            if isinstance(target, ast.Name):
                                names.add(target.id)
    return names


def main() -> int:
    mods = local_modules()
    trees = {}
    problems = []

    for name, path in mods.items():
        source = path.read_text(encoding="utf-8")
        try:
            trees[name] = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            problems.append(f"SYNTAX {path}: {exc}")

    if problems:
        for p in problems:
            print(p)
        return 1

    exported = {name: top_level_names(tree) for name, tree in trees.items()}

    checked = 0
    for name, tree in trees.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:  # relative import inside the baselines package
                    parts = name.split(".")
                    base = ".".join(parts[: len(parts) - node.level])
                    target = f"{base}.{node.module}" if node.module else base
                else:
                    target = node.module or ""
                if target not in exported:
                    continue
                for alias in node.names:
                    checked += 1
                    if alias.name == "*":
                        continue
                    if alias.name not in exported[target]:
                        problems.append(
                            f"{name}: 'from {target} import {alias.name}' "
                            f"-- {target} does not define {alias.name}"
                        )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in {m.split(".")[0] for m in exported} and alias.name not in exported:
                        if alias.name not in mods:
                            problems.append(f"{name}: imports missing local module {alias.name}")

    print(f"modules: {len(mods)}   cross-module imports checked: {checked}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
        return 1
    print("static verification OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
