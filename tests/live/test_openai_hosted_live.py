"""Explicitly authorized, bounded OpenAI provider smoke test."""

from __future__ import annotations

import os

import pytest

from ragkit.adapters import OpenAIHostedGenerator
from ragkit.ports import GenerationRequest, Prompt


@pytest.mark.live
def test_openai_hosted_generator_returns_bounded_sanitized_output() -> None:
    if os.environ.get("RAGKIT_RUN_LIVE") != "1":
        pytest.skip("set RAGKIT_RUN_LIVE=1 to authorize the bounded paid smoke")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("set OPENAI_API_KEY for the explicitly authorized live smoke")

    model = os.environ.get("RAGKIT_LIVE_OPENAI_MODEL", "gpt-5-mini")
    generator = OpenAIHostedGenerator(
        model=model,
        api_key=api_key,
        timeout_seconds=30.0,
        max_retries=0,
    )
    result = generator.generate(
        GenerationRequest(
            query="Return one short provider health acknowledgement.",
            context=(),
            prompt=Prompt("Reply with the exact text LIVE_OK and nothing else.", ()),
            temperature=0.0,
            max_output_tokens=8,
        )
    )

    assert result.answer.strip()
    assert len(result.answer) <= 256
    assert result.cited_chunk_ids == ()
    assert "openai_responses" in str(result.model)
    assert api_key not in repr(result)
    if result.usage is not None:
        assert result.usage.input_tokens >= 0
        assert 0 <= result.usage.output_tokens <= 8
