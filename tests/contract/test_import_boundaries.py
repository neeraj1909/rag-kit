from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKER = PROJECT_ROOT / "scripts" / "check_imports.py"


def _run_checker(package_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), "--root", str(package_root)],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.contract
def test_repository_obeys_inward_dependency_rule() -> None:
    completed = _run_checker(PROJECT_ROOT / "src" / "ragkit")

    assert completed.returncode == 0, completed.stderr


@pytest.mark.contract
def test_checker_accepts_absent_future_layers(tmp_path: Path) -> None:
    completed = _run_checker(tmp_path / "ragkit")

    assert completed.returncode == 0, completed.stderr


@pytest.mark.contract
def test_checker_accepts_inward_import_from_package_root(tmp_path: Path) -> None:
    package_root = tmp_path / "ragkit"
    layer_root = package_root / "ports"
    layer_root.mkdir(parents=True)
    (layer_root / "probe.py").write_text("from ragkit import domain\n", encoding="utf-8")

    completed = _run_checker(package_root)

    assert completed.returncode == 0, completed.stderr


@pytest.mark.contract
@pytest.mark.parametrize(
    ("layer", "forbidden_import", "reported_import"),
    [
        ("domain", "from ragkit.adapters import vector_store", "ragkit.adapters"),
        ("ports", "import chromadb", "chromadb"),
        ("application", "from ..adapters import embeddings", "ragkit.adapters"),
    ],
)
def test_checker_rejects_outward_and_provider_imports(
    tmp_path: Path, layer: str, forbidden_import: str, reported_import: str
) -> None:
    package_root = tmp_path / "ragkit"
    layer_root = package_root / layer
    layer_root.mkdir(parents=True)
    (layer_root / "probe.py").write_text(f"{forbidden_import}\n", encoding="utf-8")

    completed = _run_checker(package_root)

    assert completed.returncode == 1
    assert "Import-boundary violations" in completed.stderr
    assert reported_import in completed.stderr
