"""Small RAG-ready interfaces for future notes and PDF ingestion."""
from dataclasses import dataclass
from typing import Protocol

@dataclass
class DocumentChunk:
    document_id: str
    text: str
    metadata: dict

class DocumentIngestor(Protocol):
    def ingest(self, document_id: str, text: str, metadata: dict | None = None) -> list[DocumentChunk]: ...

class Chunker:
    def split(self, document_id: str, text: str, metadata: dict | None = None, size: int = 800) -> list[DocumentChunk]:
        words = text.split()
        return [DocumentChunk(document_id, " ".join(words[start:start + size]), metadata or {}) for start in range(0, len(words), size)]

class Retriever(Protocol):
    def search(self, query: str, limit: int = 5) -> list[DocumentChunk]: ...

class ContextInjector:
    def build(self, chunks: list[DocumentChunk]) -> str:
        return "\n\n".join(chunk.text for chunk in chunks)
