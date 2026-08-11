"""CLI for the staged, rollback-protected assignment-template copier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragkit.domain import InvalidDomainValueError
from ragkit.infrastructure.assignment import bootstrap_assignment

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = ROOT / "examples" / "assignment_profiles"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy one reviewed rag-kit assignment template safely."
    )
    parser.add_argument("--template", required=True, help="local-offline or hosted-persistent")
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="validate and print without writes")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only changed managed files after collision preflight",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = bootstrap_assignment(
            args.template,
            args.destination,
            template_root=TEMPLATE_ROOT,
            repository_root=ROOT,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
    except (FileExistsError, InvalidDomainValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
