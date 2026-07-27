# Roadmap

The roadmap prioritizes trustworthy retrieval and safe on-premises operation.
Items are ordered by expected user value rather than delivery date.

## Retrieval quality

- Add a versioned evaluation dataset with expected source documents.
- Measure retrieval recall, citation accuracy, and unsupported-answer rate.
- Add optional hybrid search and local reranking.
- Preserve page and heading metadata through ingestion.
- Detect index incompatibility when the embedding model changes.

## Security and operations

- Add authenticated multi-user browser deployment.
- Encrypt local chat and attachment storage.
- Add configurable retention and secure deletion.
- Record administrative actions in an audit log.
- Add rate limiting and background job controls.

## Document lifecycle

- Track document versions and index status.
- Support incremental re-indexing and deletion.
- Add OCR through an optional local processing pipeline.
- Export answers and citations in portable formats.

## Distribution

- Add reproducible desktop builds for supported platforms.
- Sign release artifacts and publish checksums.
- Add a documented container deployment after authentication is available.

Feature proposals should explain the user problem, privacy impact, operational
cost, and how the behavior can be tested.
