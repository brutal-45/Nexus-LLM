#!/usr/bin/env python3
"""Keep the repo's version strings consistent.

``nexus_llm/__version__.py`` is the source of truth; this rewrites the
``VERSION`` file that other tooling (release automation, the Docker image
label, docs) reads, and fails when the package metadata drifts.

Usage:
    python scripts/dev/sync_version.py           # check + sync VERSION
    python scripts/dev/sync_version.py --check   # check only, non-zero exit
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from nexus_llm.__version__ import __version__  # noqa: E402

VERSION_FILE = ROOT / "VERSION"
CHANGELOG = ROOT / "CHANGELOG.md"


def read_version_file() -> str:
    return VERSION_FILE.read_text(encoding="utf-8").strip() if VERSION_FILE.exists() else ""


def check_changelog() -> list[str]:
    """Return problems found in CHANGELOG.md (missing entry for this version)."""
    problems: list[str] = []
    if CHANGELOG.exists() and f"[{__version__}]" not in CHANGELOG.read_text(encoding="utf-8"):
        problems.append(f"CHANGELOG.md has no section for version {__version__}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="only verify, do not write")
    args = parser.parse_args()

    problems: list[str] = []
    current = read_version_file()
    if current != __version__:
        if args.check:
            problems.append(f"VERSION contains {current!r}, expected {__version__!r}")
        else:
            VERSION_FILE.write_text(f"{__version__}\n", encoding="utf-8")
            print(f"VERSION updated: {current!r} -> {__version__!r}")
    else:
        print(f"VERSION matches package version ({__version__})")

    if not re.match(r"^\d+\.\d+\.\d+", __version__):
        problems.append(f"version {__version__!r} is not PEP 440 style X.Y.Z")

    problems += check_changelog()

    if args.check and not problems:
        problems = []

    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
