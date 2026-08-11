from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRILL_EVIDENCE = PROJECT_ROOT / "reports/agent-guidance/cold-agent-drill-v1.md"
AGENT_SURFACE = (
    PROJECT_ROOT / "AGENTS.md",
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "docs" / "agent-map.md",
)


def _agent_surface() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in AGENT_SURFACE)


def _class_bases(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                base.id if isinstance(base, ast.Name) else ast.unparse(base) for base in node.bases
            }
    raise AssertionError(f"{class_name} is absent from {path.relative_to(PROJECT_ROOT)}")


@pytest.mark.contract
@pytest.mark.parametrize(
    ("question", "answer_path"),
    [
        ("Where is the contract?", "src/ragkit/ports/interfaces.py"),
        ("Where is the adapter?", "src/ragkit/adapters/"),
        ("Where is composition?", "src/ragkit/infrastructure/bootstrap.py"),
        ("How do I validate?", "CONTRIBUTING.md"),
        ("What must not be changed?", "ARCHITECTURE.md"),
    ],
)
def test_cold_agent_can_locate_each_grounded_answer(question: str, answer_path: str) -> None:
    surface = _agent_surface()

    assert question in surface
    assert answer_path in surface
    assert (PROJECT_ROOT / answer_path).exists()


@pytest.mark.contract
def test_contract_adapter_and_composition_answers_match_source() -> None:
    interfaces = PROJECT_ROOT / "src/ragkit/ports/interfaces.py"
    retrieval = PROJECT_ROOT / "src/ragkit/adapters/retrieval.py"
    bootstrap = PROJECT_ROOT / "src/ragkit/infrastructure/bootstrap.py"

    assert "ABC" in _class_bases(interfaces, "Retriever")
    assert "Retriever" in _class_bases(retrieval, "DenseRetriever")
    bootstrap_tree = ast.parse(bootstrap.read_text(encoding="utf-8"), filename=str(bootstrap))
    assert any(
        isinstance(node, ast.FunctionDef) and node.name == "bootstrap"
        for node in bootstrap_tree.body
    )


@pytest.mark.contract
def test_validation_and_prohibited_shortcut_answers_are_enforced() -> None:
    agent_map = (PROJECT_ROOT / "docs/agent-map.md").read_text(encoding="utf-8")
    contributing = (PROJECT_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    for command in (
        "uv run ruff format --check .",
        "uv run mypy src tests",
        "uv run python scripts/check_imports.py",
        "timeout 60 uv run pytest -m contract --no-cov",
    ):
        assert command in agent_map
        assert command in contributing
    assert (PROJECT_ROOT / "scripts/check_imports.py").is_file()
    assert (PROJECT_ROOT / "tests/contract/test_import_boundaries.py").is_file()
    assert "Do not weaken" in _agent_surface()


@pytest.mark.contract
def test_path_instructions_are_scoped_and_nonduplicative() -> None:
    instruction_root = PROJECT_ROOT / ".github/instructions"
    instruction_files = sorted(instruction_root.glob("*.instructions.md"))

    assert instruction_files
    bodies: list[str] = []
    for path in instruction_files:
        text = path.read_text(encoding="utf-8")
        match = re.match(r'^---\napplyTo: "([^"\n]+)"\n---\n', text)
        assert match, f"{path.relative_to(PROJECT_ROOT)} needs one applyTo scope"
        assert match.group(1).strip()
        body = text[match.end() :].strip()
        assert body
        assert body not in bodies, f"duplicate instruction body: {path.relative_to(PROJECT_ROOT)}"
        bodies.append(body)


@pytest.mark.contract
def test_cold_agent_evidence_scores_five_source_grounded_answers() -> None:
    evidence = DRILL_EVIDENCE.read_text(encoding="utf-8")
    rows = re.findall(r"^\| (Q[1-5]) \| ([^|]+) \| (PASS) \| ([^|]+) \|$", evidence, re.MULTILINE)

    assert [row[0] for row in rows] == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert {row[2] for row in rows} == {"PASS"}
    assert "No question was awarded a pass" in evidence
    assert "AST-backed symbol-link gate" in evidence
    assert "Final score: 5/5" in evidence

    for _, question, _, answer in rows:
        assert question.strip() in _agent_surface()
        cited_paths = re.findall(r"`([^`]+(?:\.md|\.py|/))`", answer)
        assert cited_paths, f"answer has no repository path: {question}"
        for cited_path in cited_paths:
            assert (PROJECT_ROOT / cited_path).exists(), f"missing cited path: {cited_path}"
