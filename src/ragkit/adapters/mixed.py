"""Explicit composition for raster pages containing both text and visual evidence."""

from __future__ import annotations

from ragkit.domain import Document, DocumentId, SourceId, UnsupportedCapabilityError
from ragkit.ports import (
    AssetClassification,
    DocumentExtractor,
    DocumentFamily,
    ExtractionRequest,
)


class MixedImageDocumentExtractor(DocumentExtractor):
    """Run OCR and vision on the same raster asset and merge provenance-complete parts.

    This adapter is intentionally explicit: it performs both requested capabilities
    and fails if either one is unavailable. It does not silently downgrade a mixed
    page to text-only or description-only evidence.
    """

    def __init__(self, ocr: DocumentExtractor, vision: DocumentExtractor) -> None:
        self._ocr = ocr
        self._vision = vision

    def extract(self, request: ExtractionRequest) -> tuple[Document, ...]:
        if any(item.family is not DocumentFamily.VISION for item in request.classifications):
            raise UnsupportedCapabilityError(
                "mixed image extraction requires the vision family",
                capability="mixed_image_family",
            )
        ocr_classifications = tuple(
            AssetClassification(
                item.asset_id,
                DocumentFamily.OCR,
                item.confidence,
                item.classifier,
            )
            for item in request.classifications
        )
        ocr_documents = self._ocr.extract(
            ExtractionRequest(request.assets, ocr_classifications, request.max_documents)
        )
        vision_documents = self._vision.extract(request)
        documents: list[Document] = []
        for asset, ocr_document, vision_document in zip(
            request.assets, ocr_documents, vision_documents, strict=True
        ):
            source_id = SourceId.from_locator(
                "mixed_image", {"uri": asset.reference.uri or asset.reference.asset_id}
            )
            document_id = DocumentId.from_assets(source_id, (asset.reference.sha256,))
            documents.append(
                Document(
                    document_id,
                    source_id,
                    (asset.reference,),
                    ocr_document.parts + vision_document.parts,
                    {
                        "source_uri": asset.reference.uri,
                        "content_mode": "mixed_image_text",
                        "ocr_part_count": len(ocr_document.parts),
                        "vision_part_count": len(vision_document.parts),
                    },
                )
            )
        return tuple(documents)
