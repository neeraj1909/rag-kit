"""Staged, rollback-protected copying for reviewed assignment profiles."""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ragkit.domain import InvalidDomainValueError

from .config import load_config

TEMPLATES = frozenset({"hosted-persistent", "local-offline"})
REQUIRED_FILES = ("ASSIGNMENT.md", "ragkit.toml")


def _copyfile(source: Path, target: Path) -> None:
    shutil.copyfile(source, target)


def _replace(source: Path, target: Path) -> None:
    os.replace(source, target)


@dataclass(frozen=True, slots=True)
class TemplateInspection:
    """Validated template root and deterministic managed-file inventory."""

    root: Path
    files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Machine-readable outcome; timing is measured externally, not serialized."""

    template: str
    destination: Path
    status: str
    files: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "destination": str(self.destination),
            "files": list(self.files),
            "status": self.status,
            "template": self.template,
        }


def inspect_template(template: str, *, template_root: Path) -> TemplateInspection:
    """Require one known, regular, exact-inventory template with a valid profile."""

    if template not in TEMPLATES:
        raise InvalidDomainValueError(
            f"unknown assignment template {template!r}; choose one of {sorted(TEMPLATES)}"
        )
    root = template_root.resolve() / template
    if not root.is_dir() or root.is_symlink():
        raise InvalidDomainValueError(f"assignment template {template!r} is unavailable")
    entries = tuple(sorted(path.name for path in root.iterdir()))
    if entries != REQUIRED_FILES:
        raise InvalidDomainValueError(
            f"assignment template {template!r} must contain exactly {list(REQUIRED_FILES)}"
        )
    for name in REQUIRED_FILES:
        path = root / name
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode) or path.is_symlink() or path.stat().st_nlink != 1:
            raise InvalidDomainValueError(f"assignment template file {name!r} must be regular")
    load_config(root / "ragkit.toml")
    return TemplateInspection(root, REQUIRED_FILES)


def _safe_destination(destination: Path, *, repository_root: Path, template_root: Path) -> Path:
    expanded = destination.expanduser()
    if expanded.is_symlink():
        raise InvalidDomainValueError("assignment destination must not be a symlink")
    resolved = expanded.resolve(strict=False)
    unsafe = {
        Path("/"),
        Path.home().resolve(),
        repository_root.resolve(),
        template_root.resolve(),
    }
    if resolved in unsafe or resolved.is_relative_to(template_root.resolve()):
        raise InvalidDomainValueError(f"unsafe destination: {resolved}")
    for parent in (expanded.absolute(), *expanded.absolute().parents):
        if parent.exists() and parent.is_symlink():
            raise InvalidDomainValueError("assignment destination path must not contain symlinks")
    if resolved.exists() and not resolved.is_dir():
        raise InvalidDomainValueError("assignment destination must be a directory")
    return resolved


def _require_safe_managed_target(target: Path, name: str) -> None:
    if not target.exists() and not target.is_symlink():
        return
    metadata = target.lstat()
    if target.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise InvalidDomainValueError(f"managed destination {name!r} must be a regular file")
    if metadata.st_nlink != 1:
        raise InvalidDomainValueError(f"managed destination {name!r} must not be a hardlink")


def _commit_staged(comparisons: tuple[tuple[Path, Path, bool], ...], staging: Path) -> None:
    changed = tuple(item for item in comparisons if not item[2])
    committed: list[tuple[Path, Path | None]] = []
    try:
        for _, target, _ in changed:
            staged = staging / f"new-{target.name}"
            backup = staging / f"old-{target.name}" if target.exists() else None
            if backup is not None:
                _replace(target, backup)
            try:
                _replace(staged, target)
            except BaseException:
                if backup is not None and backup.exists():
                    _replace(backup, target)
                raise
            committed.append((target, backup))
    except BaseException:
        for target, backup in reversed(committed):
            if target.exists():
                target.unlink()
            if backup is not None and backup.exists():
                _replace(backup, target)
        raise


def bootstrap_assignment(
    template: str,
    destination: Path,
    *,
    template_root: Path,
    repository_root: Path,
    dry_run: bool = False,
    overwrite: bool = False,
) -> BootstrapResult:
    """Copy two managed files with collision preflight and in-process rollback.

    Identical reruns are no-ops. Changed managed files abort every write unless
    ``overwrite`` is explicit. Symlinks and hardlinks are rejected. Staging or
    commit failure restores the complete pre-call managed-file state, while files
    outside the reviewed inventory are never touched. Abrupt process termination
    and filesystem failure are outside this two-file rollback guarantee.
    """

    inspection = inspect_template(template, template_root=template_root)
    target_root = _safe_destination(
        destination,
        repository_root=repository_root,
        template_root=template_root,
    )
    comparisons_list: list[tuple[Path, Path, bool]] = []
    collisions: list[str] = []
    for name in inspection.files:
        source = inspection.root / name
        target = target_root / name
        _require_safe_managed_target(target, name)
        identical = target.is_file() and target.read_bytes() == source.read_bytes()
        if target.exists() and not identical:
            collisions.append(name)
        comparisons_list.append((source, target, identical))
    comparisons = tuple(comparisons_list)
    if collisions and not overwrite:
        raise FileExistsError(
            "assignment files differ and overwrite was not requested: " + ", ".join(collisions)
        )

    if dry_run:
        status = "dry-run"
    elif collisions:
        status = "overwritten"
    elif all(identical for _, _, identical in comparisons):
        status = "unchanged"
    else:
        status = "created"

    if not dry_run and status != "unchanged":
        root_existed = target_root.exists()
        target_root.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(prefix=".ragkit-bootstrap-", dir=target_root) as raw:
                staging = Path(raw)
                for source, target, identical in comparisons:
                    if not identical:
                        _copyfile(source, staging / f"new-{target.name}")
                _commit_staged(comparisons, staging)
        except BaseException:
            if not root_existed:
                target_root.rmdir()
            raise
    return BootstrapResult(template, target_root, status, inspection.files)
