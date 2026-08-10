from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from conftest import ContractCorpus
from ragkit.adapters import MixedImageDocumentExtractor
from ragkit.domain import Document
from ragkit.ports import (
    AcquiredAsset,
    AssetClassification,
    DocumentExtractor,
    DocumentFamily,
    ExtractionRequest,
)

pytestmark = pytest.mark.unit


@dataclass
class _FamilyExtractor(DocumentExtractor):
    family: DocumentFamily
    document: Document

    def extract(self, request: ExtractionRequest) -> tuple[Document, ...]:
        assert all(item.family is self.family for item in request.classifications)
        return (self.document,)


def test_mixed_image_extraction_requires_and_retains_both_evidence_streams(
    contract_corpus: ContractCorpus,
) -> None:
    document = contract_corpus.document
    ocr_document = replace(document, parts=(document.parts[2],))
    vision_document = replace(document, parts=(document.parts[4],))
    asset = document.assets[0]
    acquired = AcquiredAsset(asset, b"fixture")
    classification = AssetClassification(
        asset.asset_id,
        DocumentFamily.VISION,
        1.0,
        contract_corpus.manifest.domain_schema_fingerprint,
    )
    request = ExtractionRequest((acquired,), (classification,), 1)

    result = MixedImageDocumentExtractor(
        _FamilyExtractor(DocumentFamily.OCR, ocr_document),
        _FamilyExtractor(DocumentFamily.VISION, vision_document),
    ).extract(request)[0]

    assert result.parts == ocr_document.parts + vision_document.parts
    assert result.assets == (asset,)
    assert result.metadata["content_mode"] == "mixed_image_text"
