#!/usr/bin/env python3
"""Validate release archives without installing or executing their contents."""

from __future__ import annotations

import argparse
import configparser
import email.parser
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

REQUIRED_PROJECT_FILES = {"LICENSE", "README.md", "pyproject.toml", "src/ragkit/py.typed"}


def _safe_parts(name: str) -> tuple[str, ...]:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"archive contains unsafe path: {name}")
    return path.parts


def _one_matching(names: tuple[str, ...], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {suffix} entry, found {len(matches)}")
    return matches[0]


def check_wheel(path: Path, expected_extras: frozenset[str]) -> None:
    with zipfile.ZipFile(path) as archive:
        names = tuple(archive.namelist())
        for name in names:
            _safe_parts(name)
        _one_matching(names, "ragkit/py.typed")
        metadata_name = _one_matching(names, ".dist-info/METADATA")
        entry_points_name = _one_matching(names, ".dist-info/entry_points.txt")
        license_name = _one_matching(names, ".dist-info/licenses/LICENSE")
        metadata_text = archive.read(metadata_name).decode("utf-8")
        entry_points_text = archive.read(entry_points_name).decode("utf-8")
        license_text = archive.read(license_name).decode("utf-8")

    metadata = email.parser.Parser().parsestr(metadata_text)
    provided_extras = frozenset(metadata.get_all("Provides-Extra", []))
    if metadata["Name"] != "rag-kit":
        raise ValueError(f"unexpected distribution name: {metadata['Name']!r}")
    python_specifiers = {item.strip() for item in metadata["Requires-Python"].split(",")}
    if python_specifiers != {">=3.11", "<3.13"}:
        raise ValueError(f"unexpected Python range: {metadata['Requires-Python']!r}")
    if provided_extras != expected_extras:
        expected = sorted(expected_extras)
        actual = sorted(provided_extras)
        raise ValueError(f"wheel extras differ: expected {expected}, got {actual}")
    if metadata["Description-Content-Type"] != "text/markdown" or "# rag-kit" not in metadata_text:
        raise ValueError("wheel metadata does not contain the Markdown README")
    if "MIT License" not in license_text:
        raise ValueError("wheel does not contain the MIT license text")
    entry_points = configparser.ConfigParser()
    entry_points.read_string(entry_points_text)
    scripts = dict(entry_points["console_scripts"])
    expected_scripts = {
        "ragkit": "ragkit.cli.main:main",
        "ragkit-http": "ragkit.delivery.server:main",
    }
    if scripts != expected_scripts:
        raise ValueError(f"console scripts differ: expected {expected_scripts}, got {scripts}")


def check_sdist(path: Path) -> None:
    with tarfile.open(path, mode="r:gz") as archive:
        files: set[str] = set()
        for member in archive.getmembers():
            parts = _safe_parts(member.name)
            if member.isfile() and len(parts) > 1:
                files.add("/".join(parts[1:]))
    missing = REQUIRED_PROJECT_FILES - files
    if missing:
        raise ValueError(f"sdist is missing release files: {sorted(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--extra", action="append", default=[])
    args = parser.parse_args()

    check_wheel(args.wheel, frozenset(args.extra))
    check_sdist(args.sdist)
    print(f"release archives valid: {args.wheel.name}, {args.sdist.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
