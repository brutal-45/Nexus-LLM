#!/usr/bin/env python3
"""Print the public API (classes, methods, signatures) of a nexus_llm module.

Usage: python scripts/dev/api_dump.py nexus_llm.tools.tool [more.modules ...]
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def dump(module_name: str) -> None:
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        print(f"### {module_name}: IMPORT FAILED -> {type(exc).__name__}: {exc}")
        return

    print(f"\n{'=' * 70}\n### {module_name}\n{'=' * 70}")
    doc = inspect.getdoc(mod)
    if doc:
        print(f'"""{doc.splitlines()[0]}"""')

    for name, obj in vars(mod).items():
        if name.startswith("_") or not getattr(obj, "__module__", "").startswith(module_name):
            continue
        if inspect.isclass(obj):
            print(f"\nclass {name}{_sig(obj)}")
            for mname, meth in inspect.getmembers(obj, predicate=inspect.isfunction):
                if mname.startswith("_") and mname != "__call__":
                    continue
                if inspect.getattr_static(obj, mname, None) is None:
                    continue
                print(f"    def {mname}{_sig(meth)}")
            for mname, attr in inspect.getmembers(obj):
                if isinstance(attr, property) and not mname.startswith("_"):
                    print(f"    @property {mname}")
        elif inspect.isfunction(obj):
            print(f"\ndef {name}{_sig(obj)}")
        elif not callable(obj):
            print(f"\n{name} = {obj!r}"[:160])


def _sig(obj: object) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("modules", nargs="+")
    args = parser.parse_args()
    for m in args.modules:
        dump(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
