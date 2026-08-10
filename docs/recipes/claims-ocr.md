# Claims OCR with human review

Use the `ocr` extra for bounded printed scans. Tesseract and the requested
language data are system prerequisites; the Python extra does not install them.

Configure `OcrDocumentExtractor` with a review threshold appropriate to the
claim type. Every word carries a zero-based page, normalized source box, raw
engine confidence, and a `low_ocr_confidence` notice below that threshold.
Route those notices to a reviewer and cite the original image region. Do not
turn confidence into a claim that the word is correct.

Handwriting and form modes are explicitly degraded. They preserve recognized
words, but emit `handwriting_best_effort` or `form_structure_unverified` because
the printed-text baseline does not establish handwriting accuracy, checkbox
state, or field/value relationships. Empty pages, corrupt files, missing
Tesseract/language data, timeouts, and page/pixel limits fail actionably.
