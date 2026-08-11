# Internal knowledge-base text

Family: `text`

## Business use case

Index reviewed policy Markdown, plain text, HTML, source files, and single-part email
so a support agent can answer with exact character-span citations.

## Contract

`TextDocumentExtractor` implements `DocumentExtractor`; `StructureAwareChunker`
implements `Chunker`. Original decoded spans must survive extraction and adjusted
chunk spans must still resolve to the acquired asset.

## Config schema

Start with `configs/offline.toml`: select `family = "text"`, classifier/extractor
`"text"`, and chunker `"structure_aware"`. Bound acquisition and chunking through
`RuntimeLimits`; `chunk_chars` changes chunk identity and the index manifest.

## Registry and bootstrap

The existing factories are `"text"` and `"structure_aware"` in
`infrastructure/bootstrap.py`. A replacement gets a new explicit selection and factory;
do not replace the baseline under its old name.

## Tests

Run textual unit tests, the shared extractor/chunker contracts, and the offline CLI
journey. Add a fixture for each newly supported syntax rather than asserting only that
some text was returned.

## Optional extra

None. The baseline uses the Python standard library and the dependency-free core.

## Limits

Set asset bytes, document/part/chunk counts, and `chunk_chars` in the profile. Multipart
or encoded email and ambiguous decoding are outside the exact-span baseline.

## Determinism

Fixed bytes, decoding rules, and component fingerprints produce stable order, spans,
and chunk IDs.

## Confidence and fallback

Confidence is not applicable to exact decoded text. Unsupported MIME structures do
not fall back to lossy plain text.

## Failure modes

Malformed or ambiguous encoding, unsafe filesystem inputs, excluded attachments,
unsupported MIME transfer encoding, and any limit breach fail with typed errors.
