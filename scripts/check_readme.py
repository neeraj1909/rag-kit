"""Validate README structure, local links, and the copied offline quickstart."""

from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
import subprocess
from pathlib import Path

REQUIRED_SECTIONS = (
    "## Purpose and non-goals",
    "## Offline quickstart",
    "## Data flow",
    "## Choose a profile",
    "## Extension map",
    "## Validation",
    "## Deeper documentation",
)
QUICKSTART = re.compile(
    r"<!-- readme-quickstart:start -->\s*```bash\s*(.*?)\s*```\s*"
    r"<!-- readme-quickstart:end -->",
    re.DOTALL,
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
SYMBOL_LINK = re.compile(r"\[`([A-Za-z_][A-Za-z0-9_]*)`\]\(([^)#]+\.py)(?:#[^)]+)?\)")
REQUIRED_IMPLEMENTATIONS = frozenset(
    {
        "BM25Retriever",
        "SQLiteVectorStore",
        "DeclaredFamilyClassifier",
        "DenseRetriever",
        "DeterministicEvaluator",
        "EvidenceChunker",
        "ExtractiveGenerator",
        "FilesystemSourceConnector",
        "HashingEmbedder",
        "HybridRetriever",
        "InMemoryVectorStore",
        "InMemoryTelemetry",
        "LayoutDocumentExtractor",
        "LocalCrossEncoderReranker",
        "MediaDocumentExtractor",
        "NoOpDocumentProjector",
        "NoOpReranker",
        "OcrDocumentExtractor",
        "OpenAIHostedGenerator",
        "StructureAwareChunker",
        "TemplatePromptBuilder",
        "TextDocumentExtractor",
        "TextFamilyClassifier",
        "TorchTextEmbedder",
        "VisionDocumentExtractor",
    }
)


def _commands(block: str) -> tuple[str, ...]:
    commands: list[str] = []
    pending = ""
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pending = f"{pending} {line}".strip()
        if pending.endswith("\\"):
            pending = pending[:-1].rstrip()
            continue
        commands.append(pending)
        pending = ""
    if pending:
        raise ValueError("quickstart ends with an unfinished line continuation")
    return tuple(commands)


def _validate_links(text: str, root: Path) -> None:
    missing: list[str] = []
    for raw_target in MARKDOWN_LINK.findall(text):
        target = raw_target.strip().strip("<>")
        if target.startswith(("https://", "http://", "mailto:", "#")):
            continue
        path_text = target.split("#", maxsplit=1)[0]
        if path_text:
            candidate = (root / path_text).resolve()
            if not candidate.is_relative_to(root) or not candidate.exists():
                missing.append(target)
    if missing:
        raise ValueError(f"README local links do not exist: {', '.join(sorted(set(missing)))}")


def _validate_symbol_links(text: str, root: Path) -> None:
    links = SYMBOL_LINK.findall(text)
    linked_symbols = {symbol for symbol, _ in links}
    omitted = sorted(REQUIRED_IMPLEMENTATIONS - linked_symbols)
    if omitted:
        raise ValueError(f"README extension map omits implementations: {', '.join(omitted)}")

    mismatches: list[str] = []
    definitions_by_path: dict[str, set[str]] = {}
    for symbol, target in links:
        definitions = definitions_by_path.get(target)
        if definitions is None:
            tree = ast.parse((root / target).read_text(encoding="utf-8"), filename=target)
            definitions = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            }
            definitions_by_path[target] = definitions
        if symbol not in definitions:
            mismatches.append(f"{symbol} -> {target}")
    if mismatches:
        raise ValueError(f"README symbol links point to the wrong file: {', '.join(mismatches)}")


def validate(readme: Path, root: Path) -> tuple[str, ...]:
    """Return the quickstart commands after checking the README contract."""

    text = readme.read_text(encoding="utf-8")
    positions = [text.find(section) for section in REQUIRED_SECTIONS]
    if -1 in positions:
        missing = [
            section
            for section, position in zip(REQUIRED_SECTIONS, positions, strict=True)
            if position < 0
        ]
        raise ValueError(f"README is missing required sections: {', '.join(missing)}")
    if positions != sorted(positions):
        raise ValueError("README required sections are out of order")

    match = QUICKSTART.search(text)
    if match is None:
        raise ValueError("README has no executable offline quickstart block")
    commands = _commands(match.group(1))
    if not 1 <= len(commands) <= 3:
        raise ValueError("offline quickstart must contain one to three commands")
    if not commands[0].startswith("uv sync --frozen"):
        raise ValueError("offline quickstart must start with a frozen install")
    if not commands[-1].startswith("uv run ragkit ask "):
        raise ValueError("offline quickstart must end with the cited-answer command")

    data_flow = text[positions[2] : positions[3]]
    if data_flow.count("```text") != 1:
        raise ValueError("README data-flow section must contain exactly one text diagram")
    _validate_links(text, root)
    _validate_symbol_links(text, root)
    return commands


def execute(commands: tuple[str, ...], root: Path) -> dict[str, object]:
    """Run the copied commands and return proof that the final JSON cites evidence."""

    final_stdout = ""
    for command in commands:
        result = subprocess.run(
            shlex.split(command),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        final_stdout = result.stdout
    payload = json.loads(final_stdout)
    if not isinstance(payload.get("answer"), str) or not payload["answer"].strip():
        raise ValueError("quickstart did not return a non-empty answer")
    if "cobalt observatory" not in payload["answer"].casefold():
        raise ValueError("quickstart answer no longer matches the documented fixture")
    citations = payload.get("citations")
    if not isinstance(citations, list) or not citations:
        raise ValueError("quickstart did not return a citation")
    evidence = citations[0].get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("quickstart citation has no source evidence")
    if not evidence[0].get("source_uri") or not evidence[0].get("locator"):
        raise ValueError("quickstart source evidence has no URI or locator")
    locator = evidence[0]["locator"]
    if citations[0].get("rank") != 1:
        raise ValueError("quickstart's first citation is not rank 1")
    if not evidence[0]["source_uri"].endswith("/answer.txt"):
        raise ValueError("quickstart's first citation does not point to answer.txt")
    if locator.get("kind") != "text_span":
        raise ValueError("quickstart's first citation does not have a text_span locator")
    return {
        "answer": payload["answer"],
        "citation_rank": citations[0].get("rank"),
        "source_uri": evidence[0]["source_uri"],
        "locator_kind": locator.get("kind"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    commands = validate(root / "README.md", root)
    proof = execute(commands, root) if args.execute else None
    print(
        json.dumps(
            {"commands": len(commands), "executed": args.execute, "links": "ok", "proof": proof},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
