"""The dependency-light ``ragkit`` command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from time import perf_counter_ns
from typing import TextIO, cast

from ragkit.application import (
    AnsweringRequest,
    AnsweringResult,
    IndexingRequest,
    IndexingResult,
    StageTiming,
)
from ragkit.domain import InvalidDomainValueError, RagkitError, locator_to_dict
from ragkit.infrastructure.bootstrap import OfflineRuntime, bootstrap, inspect_profile
from ragkit.infrastructure.config import OfflineProfile, load_config
from ragkit.ports import (
    ChunkingStrategy,
    EvaluationCase,
    EvaluationExample,
    EvaluationRequest,
    IndexingStrategy,
    PhysicalIndexStrategy,
    VectorDatabase,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragkit")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect-config", help="validate and describe a profile")
    inspect.add_argument("--config", required=True, type=Path)

    index = commands.add_parser("index", help="build and validate the configured index")
    _profile_arguments(index)

    ask = commands.add_parser("ask", help="index the source and answer one query")
    _profile_arguments(ask)
    ask.add_argument("query")

    evaluate = commands.add_parser("evaluate", help="evaluate a JSONL dataset offline")
    _profile_arguments(evaluate)
    evaluate.add_argument("--dataset", required=True, type=Path)
    return parser


def _profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--chunking-strategy",
        choices=tuple(strategy.value for strategy in ChunkingStrategy),
        help="override the profile's indexing-time chunking strategy",
    )
    parser.add_argument(
        "--indexing-strategy",
        choices=tuple(strategy.value for strategy in IndexingStrategy),
        help="override the logical dense, sparse, or hybrid indexing strategy",
    )
    parser.add_argument(
        "--vector-database",
        choices=tuple(database.value for database in VectorDatabase),
        help="override the vector database selected during composition",
    )
    parser.add_argument(
        "--physical-index-strategy",
        choices=tuple(strategy.value for strategy in PhysicalIndexStrategy),
        help="override the provider-compatible physical vector index",
    )


def _source(args: argparse.Namespace, profile: OfflineProfile) -> str:
    source = cast(Path | None, args.source)
    return str(source if source is not None else Path(profile.source))


def _index(runtime: OfflineRuntime, profile: OfflineProfile, source: str) -> IndexingResult:
    limits = profile.limits
    return runtime.pipeline.index(
        IndexingRequest(
            source_uri=source,
            manifest=runtime.manifest_for(source),
            max_assets=limits.max_assets,
            max_bytes_per_asset=limits.max_bytes_per_asset,
            max_documents=limits.max_documents,
            max_parts_per_document=limits.max_parts_per_document,
            max_chunks=limits.max_chunks,
            chunking_policy=runtime.chunking_policy,
            indexing_policy=runtime.indexing_policy,
        )
    )


def _ask(
    runtime: OfflineRuntime, profile: OfflineProfile, source: str, query: str
) -> AnsweringResult:
    limits = profile.limits
    return runtime.pipeline.ask(
        AnsweringRequest(
            query=query,
            expected_manifest=runtime.manifest_for(source),
            retrieval_top_k=limits.top_k,
            rerank_top_k=limits.top_k,
            max_context_chars=limits.max_context_chars,
            temperature=0.0,
            max_output_tokens=limits.max_output_tokens,
        )
    )


def _timings(values: tuple[StageTiming, ...]) -> dict[str, float]:
    return {
        item.operation.split(".", maxsplit=1)[-1]: round(item.duration_ns / 1_000_000, 3)
        for item in values
    }


def _citations(result: AnsweringResult) -> list[dict[str, object]]:
    citations = result.citations
    return [
        {
            "chunk_id": str(citation.chunk_id),
            "document_id": str(citation.document_id),
            "rank": citation.rank,
            "evidence": [
                {
                    "asset_id": evidence.asset.asset_id,
                    "source_uri": evidence.asset.uri,
                    "locator": locator_to_dict(evidence.locator),
                    "extractor_fingerprint": str(evidence.extractor),
                    "confidence": evidence.confidence,
                    "notices": [notice.code for notice in evidence.notices],
                }
                for evidence in citation.provenance
            ],
        }
        for citation in citations
    ]


def _load_dataset(path: Path) -> tuple[EvaluationExample, ...]:
    examples: list[EvaluationExample] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise InvalidDomainValueError(f"cannot load evaluation dataset {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise InvalidDomainValueError(
                f"invalid evaluation JSON on line {line_number}: {error.msg}"
            ) from error
        if not isinstance(value, dict) or set(value) != {
            "example_id",
            "query",
            "expected_answer",
        }:
            raise InvalidDomainValueError(
                "evaluation rows require exactly example_id, query, and expected_answer"
            )
        if not all(
            isinstance(value[key], str) for key in ("example_id", "query", "expected_answer")
        ):
            raise InvalidDomainValueError("evaluation row values must be strings")
        examples.append(
            EvaluationExample(
                cast(str, value["example_id"]),
                cast(str, value["query"]),
                (),
                cast(str, value["expected_answer"]),
            )
        )
    if not examples:
        raise InvalidDomainValueError("evaluation dataset must contain at least one example")
    return tuple(examples)


def _emit(payload: object, stream: TextIO) -> None:
    json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
    stream.write("\n")


def _run(args: argparse.Namespace) -> dict[str, object]:
    profile = load_config(cast(Path, args.config))
    if args.command == "inspect-config":
        return {
            "profile": profile.name,
            "family": profile.family.value,
            "source": profile.source,
            "components": profile.to_dict()["components"],
            "limits": profile.to_dict()["limits"],
            "config_fingerprint": str(profile.fingerprint),
            "capabilities": inspect_profile(profile),
        }

    selected_strategy = cast(str | None, args.chunking_strategy)
    if selected_strategy is not None:
        profile = replace(
            profile,
            settings=replace(
                profile.settings,
                chunking_strategy=ChunkingStrategy(selected_strategy),
            ),
        )

    selected_indexing = cast(str | None, args.indexing_strategy)
    selected_database = cast(str | None, args.vector_database)
    selected_physical = cast(str | None, args.physical_index_strategy)
    if (
        selected_indexing is not None
        or selected_database is not None
        or selected_physical is not None
    ):
        strategy = (
            IndexingStrategy(selected_indexing)
            if selected_indexing is not None
            else IndexingStrategy(profile.components.retriever)
        )
        if strategy is IndexingStrategy.AUTO:
            strategy = IndexingStrategy(profile.components.retriever)
        database = (
            VectorDatabase(selected_database)
            if selected_database is not None
            else VectorDatabase(profile.components.vector_store)
        )
        if strategy is IndexingStrategy.SPARSE:
            if selected_database is not None and database is not VectorDatabase.NONE:
                raise InvalidDomainValueError("sparse indexing requires --vector-database none")
            database = VectorDatabase.NONE
        profile = replace(
            profile,
            components=replace(
                profile.components,
                retriever=strategy.value,
                vector_store=database.value,
            ),
            settings=replace(
                profile.settings,
                indexing_strategy=strategy,
                physical_index_strategy=(
                    PhysicalIndexStrategy(selected_physical)
                    if selected_physical is not None
                    else profile.settings.physical_index_strategy
                ),
            ),
        )

    runtime = bootstrap(profile)
    source = _source(args, profile)
    index_result = _index(runtime, profile, source)
    if args.command == "index":
        persistent = runtime.indexing_policy.vector_database not in {
            VectorDatabase.NONE,
            VectorDatabase.MEMORY,
        }
        return {
            "profile": profile.name,
            "documents": index_result.document_count,
            "chunks": index_result.chunk_count,
            "index_manifest_fingerprint": str(index_result.manifest.fingerprint),
            "config_fingerprint": str(profile.fingerprint),
            "storage": "persistent" if persistent else "process_local",
            "timings_ms": _timings(index_result.timings),
            "diagnostics": [item.code for item in index_result.diagnostics],
            "chunking_strategy": runtime.chunking_policy.strategy.value,
            "indexing_strategy": runtime.indexing_policy.strategy.value,
            "vector_database": runtime.indexing_policy.vector_database.value,
            "physical_index_strategy": runtime.indexing_policy.physical_index.value,
        }

    if args.command == "ask":
        answer_result = _ask(runtime, profile, source, cast(str, args.query))
        if answer_result.generation is None:
            answer = ""
            model_fingerprint = None
        else:
            answer = answer_result.generation.answer
            model_fingerprint = str(answer_result.generation.model)
        timings = _timings(answer_result.timings)
        timings["index_total"] = round(
            sum(item.duration_ns for item in index_result.timings) / 1_000_000,
            3,
        )
        return {
            "answer": answer,
            "citations": _citations(answer_result),
            "model_fingerprint": model_fingerprint,
            "config_fingerprint": str(profile.fingerprint),
            "index_mode": (
                "persistent_upserted_in_process"
                if runtime.indexing_policy.vector_database
                not in {VectorDatabase.NONE, VectorDatabase.MEMORY}
                else "rebuilt_in_process"
            ),
            "timings_ms": timings,
            "diagnostics": [item.code for item in answer_result.diagnostics],
            "chunking_strategy": runtime.chunking_policy.strategy.value,
            "indexing_strategy": runtime.indexing_policy.strategy.value,
            "vector_database": runtime.indexing_policy.vector_database.value,
            "physical_index_strategy": runtime.indexing_policy.physical_index.value,
        }

    examples = _load_dataset(cast(Path, args.dataset))
    cases: list[EvaluationCase] = []
    query_timings_ns = 0
    for example in examples:
        answer_result = _ask(runtime, profile, source, example.query)
        query_timings_ns += sum(item.duration_ns for item in answer_result.timings)
        cases.append(EvaluationCase(example, answer_result.context, answer_result.generation))
    started_ns = perf_counter_ns()
    report = runtime.evaluator.evaluate(
        EvaluationRequest(tuple(cases), runtime.evaluator_fingerprint)
    )
    evaluate_ns = perf_counter_ns() - started_ns
    return {
        "evaluated_cases": len(report.evaluated_case_ids),
        "metrics": {metric.name: metric.value for metric in report.metrics},
        "config_fingerprint": str(profile.fingerprint),
        "index_mode": "rebuilt_in_process",
        "chunking_strategy": runtime.chunking_policy.strategy.value,
        "indexing_strategy": runtime.indexing_policy.strategy.value,
        "vector_database": runtime.indexing_policy.vector_database.value,
        "physical_index_strategy": runtime.indexing_policy.physical_index.value,
        "timings_ms": {
            "index_total": round(
                sum(item.duration_ns for item in index_result.timings) / 1_000_000,
                3,
            ),
            "queries_total": round(query_timings_ns / 1_000_000, 3),
            "evaluate": round(evaluate_ns / 1_000_000, 3),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run a CLI command and emit exactly one JSON value."""

    args = _parser().parse_args(argv)
    try:
        _emit(_run(args), sys.stdout)
    except RagkitError as error:
        _emit({"error": type(error).__name__, "message": str(error)}, sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
