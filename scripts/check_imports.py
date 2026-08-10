#!/usr/bin/env python3
"""Enforce the inward-only dependency rule for the ragkit core layers."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

INWARD_LAYERS = ("domain", "ports", "application")
ALLOWED_INTERNAL: dict[str, frozenset[str]] = {
    "domain": frozenset({"domain"}),
    "ports": frozenset({"domain", "ports"}),
    "application": frozenset({"domain", "ports", "application"}),
}


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    imported: str
    reason: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.imported}: {self.reason}"


def _module_name(package_root: Path, source: Path) -> tuple[str, str]:
    relative = source.relative_to(package_root).with_suffix("")
    parts = [package_root.name, *relative.parts]
    if parts[-1] == "__init__":
        parts.pop()
        module = ".".join(parts)
        return module, module
    module = ".".join(parts)
    return module, ".".join(parts[:-1])


def _import_names(tree: ast.AST, package: str) -> list[tuple[int, str]]:
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                try:
                    module = importlib.util.resolve_name(f"{'.' * node.level}{module}", package)
                except (ImportError, ValueError):
                    module = f"{'.' * node.level}{module}"
            if "." not in module:
                imports.extend((node.lineno, f"{module}.{alias.name}") for alias in node.names)
            else:
                imports.append((node.lineno, module))
    return imports


def find_violations(package_root: Path) -> list[Violation]:
    """Return dependency violations; absent future layer directories are valid."""
    if not package_root.exists():
        return []

    violations: list[Violation] = []
    for layer in INWARD_LAYERS:
        layer_root = package_root / layer
        if not layer_root.is_dir():
            continue
        for source in sorted(layer_root.rglob("*.py")):
            _, package = _module_name(package_root, source)
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            for line, imported in _import_names(tree, package):
                top = imported.split(".", maxsplit=1)[0]
                if top == package_root.name:
                    parts = imported.split(".")
                    target_layer = parts[1] if len(parts) > 1 else None
                    if target_layer not in ALLOWED_INTERNAL[layer]:
                        violations.append(
                            Violation(
                                source,
                                line,
                                imported,
                                f"{layer} may import only inward layers "
                                f"{sorted(ALLOWED_INTERNAL[layer])}",
                            )
                        )
                elif top not in sys.stdlib_module_names:
                    violations.append(
                        Violation(
                            source,
                            line,
                            imported,
                            f"{layer} must not import third-party/provider packages",
                        )
                    )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("src/ragkit"),
        help="package root to inspect (default: src/ragkit)",
    )
    args = parser.parse_args()

    violations = find_violations(args.root.resolve())
    if violations:
        print("Import-boundary violations:", file=sys.stderr)
        for violation in violations:
            print(violation.render(), file=sys.stderr)
        return 1

    print(f"Import boundaries valid: {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
