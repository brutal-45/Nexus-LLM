#!/usr/bin/env python3
"""Import every module in the package and report which ones fail.

This catches breakage that syntax/lint checks miss (missing imports,
circular imports, bad annotations evaluated at runtime).

Usage:
    python scripts/dev/check_imports.py            # report
    python scripts/dev/check_imports.py --strict    # exit 1 on any failure
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SKIP_PATTERNS = ("tests.",)


def discover(package: str) -> list[str]:
    mod = importlib.import_module(package)
    names = [package]
    for info in pkgutil.walk_packages(mod.__path__, prefix=f"{package}."):
        if any(p in info.name for p in SKIP_PATTERNS):
            continue
        names.append(info.name)
    return sorted(names)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default="nexus_llm")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        modules = discover(args.package)
    except Exception:
        traceback.print_exc()
        print(f"FAIL: cannot even import {args.package}")
        return 1

    failures: list[tuple[str, str]] = []
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            tb = traceback.extract_tb(exc.__traceback__)[-1] if exc.__traceback__ else None
            where = f"{Path(tb.filename).name}:{tb.lineno}" if tb else "?"
            failures.append((name, f"{type(exc).__name__}: {exc} [{where}]"))

    total = len(modules)
    print(f"imported {total - len(failures)}/{total} modules of {args.package}")
    for name, err in failures:
        print(f"  FAIL {name}: {err}")
        if args.verbose:
            traceback.print_exc()

    if failures and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
