"""Check manually entered Phase 2.5 review labels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.manual_review import validate_review_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate manual event review labels.")
    parser.add_argument("review_csv", type=Path)
    args = parser.parse_args()
    try:
        reviewed = validate_review_csv(args.review_csv)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Review CSV valid; {reviewed} labeled rows (confirmed, rejected, or uncertain).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
