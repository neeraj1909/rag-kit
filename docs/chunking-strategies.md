# Chunking strategies

Rag-kit selects chunking at indexing time with a typed `ChunkingPolicy`. The resolved
policy is part of the chunker fingerprint and therefore part of the index manifest.
Changing a strategy or any of its bounds requires a compatible re-index; rag-kit does
not mix chunks produced under different policies in one index.

`auto` is an explicit alias resolved before composition: text uses `recursive`, while
OCR, layout, vision, and media use `evidence`. Unsupported family/strategy pairs fail
before indexing. There is no silent fallback.

## Catalog

| Strategy | Boundary | Typical use |
|---|---|---|
| `fixed` | Non-overlapping character windows | Uniform logs or normalized exports |
| `sliding_window` | Overlapping character windows | Narrative context across boundaries |
| `recursive` | Largest available separator, then smaller separators | General text with mixed paragraphs and lines |
| `sentence` | Deterministic sentence boundaries | Short factual prose |
| `paragraph` | Blank-line boundaries | Books, articles, and reports |
| `section` | Heading-to-heading sections | Policies and manuals |
| `hierarchical` | Sections plus bounded child chunks and parent context metadata | Long structured documents |
| `semantic` | Lexical boundary change using the configured threshold | Topic shifts without a model download |
| `proposition` | Deterministic clause/proposition boundaries | Dense factual statements |
| `book` | Chapter/section/paragraph hierarchy | Books and long-form publications |
| `legal` | Article, clause, schedule, and numbered-section boundaries | Contracts and statutes |
| `medical` | Clinical headings and report-section boundaries | Medical reports and clinical notes |
| `code` | Language-neutral declaration/block boundaries | Source code and technical snippets |
| `conversation` | Speaker-turn boundaries | Email threads, chats, and transcripts |
| `table` | Header-aware structured rows; row-preserving plain-text tables | Spreadsheets, pricing tables, and extracted tables |
| `layout_region` | Page/slide/sheet regions | Layout-preserving PDF, slide, and workbook evidence |
| `image_region` | Caption/description per detected image region | Diagrams, charts, and equipment photos |
| `transcript_segment` | Timestamped transcript segments | Calls, interviews, and audio recordings |
| `scene` | Scene/keyframe groups retaining time linkage | Video evidence |
| `evidence` | One atomic chunk per provenance-bearing content part | Conservative multimodal default |

All text-derived strategies preserve exact `TextSpanLocator` offsets. Structured
table chunks preserve header relations; plain-text tables preserve exact source rows.
Table, layout, image, transcript, and scene strategies retain their original cell,
page/box, timestamp, and keyframe provenance and relation metadata. When a non-text
region cannot be truthfully subdivided into character-addressable evidence, it stays
atomic and may exceed `max_chars`; prose strategies group these atomic extracted
parts instead of inventing finer locators.

## Family compatibility

| Family | Supported strategies |
|---|---|
| Text | fixed, sliding window, recursive, sentence, paragraph, section, hierarchical, semantic, proposition, book, legal, medical, code, conversation, table, evidence |
| OCR | fixed, sliding window, recursive, sentence, paragraph, section, hierarchical, semantic, proposition, book, legal, medical, conversation, table, layout region, evidence |
| Layout | fixed, sliding window, recursive, sentence, paragraph, section, hierarchical, semantic, proposition, book, legal, medical, code, conversation, table, layout region, evidence |
| Vision | fixed, sliding window, recursive, sentence, paragraph, semantic, proposition, image region, evidence |
| Media | fixed, sliding window, recursive, sentence, paragraph, semantic, proposition, conversation, transcript segment, scene, evidence |

`auto` is accepted for every family in addition to the entries above.

## Selection

Set `settings.chunking_strategy` in a profile, or pass `--chunking-strategy` to an
indexing CLI command. The remaining policy fields control maximum characters,
overlap, minimum chunk size, lexical semantic threshold, and whether hierarchical
children include parent context metadata. The application passes the resolved policy
through `IndexingRequest` and `ChunkingRequest`; the bound adaptive chunker rejects a
different request policy.

Query-aware chunking and model-specific late chunking are retrieval/embedding
techniques, not pure indexing-time boundary strategies. Rag-kit does not label either
as implemented by this catalog.
