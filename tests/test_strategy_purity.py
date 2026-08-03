"""Enforce rule 9 and the architecture's dependency rule mechanically.

"strategies/ imports nothing from backtest/ or execution/" is stated in
CLAUDE.md, and a stated convention decays the moment someone is in a hurry. This
test reads the AST of every module under strategies/ and fails on the import
itself, so the violation is caught at the point it is introduced rather than
discovered later as a mysterious divergence between backtest and live results.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

STRATEGIES_DIR = Path(__file__).resolve().parent.parent / "strategies"

# Importing any of these from strategy code breaks a specific guarantee:
FORBIDDEN = {
    "backtest": "the dependency rule — strategies must not know about the engine",
    "execution": "the dependency rule — strategies must not know about the broker",
    "jobs": "the dependency rule — strategies must not know about scheduling",
    "data": "rule 9 — no I/O inside strategy code",
    "config": "rule 9 — parameters arrive as arguments, not from a file read",
    "yfinance": "rule 9 — no API calls inside strategy code",
    "requests": "rule 9 — no API calls inside strategy code",
    "urllib": "rule 9 — no API calls inside strategy code",
    "random": "rule 9 — no randomness inside strategy code",
    "os": "rule 9 — no I/O or environment access inside strategy code",
    "sys": "rule 9 — no environment access inside strategy code",
    "pathlib": "rule 9 — no filesystem access inside strategy code",
    "datetime": "rule 9 — a pure function must not read the clock",
    "time": "rule 9 — a pure function must not read the clock",
    "sqlite3": "rule 9 — no I/O inside strategy code",
    "logging": "rule 9 — no side effects inside strategy code",
}

STRATEGY_MODULES = sorted(STRATEGIES_DIR.rglob("*.py"))


def test_there_are_strategy_modules_to_check():
    """Guard against this whole file silently passing on an empty glob."""
    assert STRATEGY_MODULES, f"no python modules found under {STRATEGIES_DIR}"


def imported_roots(tree: ast.AST) -> set[str]:
    """Top-level package name of every import in the module."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:            # relative import, stays inside strategies/
                continue
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize("path", STRATEGY_MODULES, ids=lambda p: p.name)
def test_strategy_module_imports_nothing_forbidden(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for root in sorted(imported_roots(tree)):
        assert root not in FORBIDDEN, (
            f"{path.relative_to(STRATEGIES_DIR.parent)} imports {root!r}, which "
            f"violates {FORBIDDEN[root]}"
        )


@pytest.mark.parametrize("path", STRATEGY_MODULES, ids=lambda p: p.name)
def test_strategy_module_has_no_module_level_side_effects(path: Path):
    """Import-time work is state; rule 9 says strategy code carries none.

    Only definitions, imports, docstrings and simple constant assignments are
    allowed at module scope.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    allowed = (
        ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
        ast.ClassDef, ast.Assign, ast.AnnAssign, ast.Expr,
    )
    for node in tree.body:
        assert isinstance(node, allowed), (
            f"{path.name}: module-level {type(node).__name__} is a side effect"
        )
        if isinstance(node, ast.Expr) and not isinstance(node.value, ast.Constant):
            pytest.fail(f"{path.name}: module-level expression is a side effect")
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if value is not None and not isinstance(
                value, (ast.Constant, ast.Tuple, ast.List, ast.Dict, ast.Set)
            ):
                pytest.fail(
                    f"{path.name}: module-level assignment calls code at import time"
                )


@pytest.mark.parametrize("path", STRATEGY_MODULES, ids=lambda p: p.name)
def test_strategy_module_does_not_open_files(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    banned_calls = {"open", "print", "input", "eval", "exec", "compile"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in banned_calls, (
                f"{path.name} calls {node.func.id}() — rule 9 forbids I/O and "
                "side effects in strategy code"
            )
