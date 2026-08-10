# Financial and pricing layout extraction

Use the `layout` extra for machine-generated PDF, PPTX, and XLSX evidence.
`LayoutDocumentExtractor` preserves zero-based PDF word boxes, slide/shape or
table boundaries, and workbook sheet/cell/merged-range coordinates. Explicit
`continues` relationships retain deterministic source order.

PDF reading order is heuristic and carries a notice. Image-only/scanned PDF
pages must be routed to OCR, while slide images must be routed to vision; the
layout adapter fails rather than silently dropping either. Workbook formulas
are retained alongside cached displayed values when the file provides them.
Missing cached values are explicit notices, not invented results. Encrypted,
corrupt, oversized, or empty containers fail with typed, actionable errors.
