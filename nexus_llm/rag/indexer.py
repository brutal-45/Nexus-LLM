"""Document indexing for building and maintaining search indices.

Provides document indexing with support for building indices,
incremental updates, persistence, and document tracking.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from nexus_llm.rag.chunker import Chunk, TextChunker
from nexus_llm.rag.embeddings import EmbeddingModel
from nexus_llm.rag.vector_store import FAISSVectorStore, VectorDocument

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    """Tracks a document that has been indexed."""

    doc_id: str
    source: str
    content_hash: str
    num_chunks: int
    indexed_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    chunk_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "content_hash": self.content_hash,
            "num_chunks": self.num_chunks,
            "indexed_at": self.indexed_at,
            "metadata": self.metadata,
            "chunk_ids": self.chunk_ids,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentRecord":
        return cls(
            doc_id=data["doc_id"],
            source=data["source"],
            content_hash=data["content_hash"],
            num_chunks=data["num_chunks"],
            indexed_at=data.get("indexed_at", 0.0),
            metadata=data.get("metadata", {}),
            chunk_ids=data.get("chunk_ids", []),
        )


class DocumentIndexer:
    """Indexes documents by chunking, embedding, and storing in a vector store.

    Manages the full lifecycle of document indexing: chunking text,
    generating embeddings, storing vectors, and tracking indexed
    documents for updates and deletion.
    """

    def __init__(
        self,
        chunker: TextChunker,
        embedding_model: EmbeddingModel,
        vector_store: FAISSVectorStore,
        batch_size: int = 64,
    ):
        """Initialize the document indexer.

        Args:
            chunker: Text chunker for splitting documents.
            embedding_model: Model for generating embeddings.
            vector_store: Vector store for storing embeddings.
            batch_size: Batch size for embedding generation.
        """
        self.chunker = chunker
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.batch_size = batch_size

        self._document_records: Dict[str, DocumentRecord] = {}
        self._chunk_to_doc: Dict[str, str] = {}  # chunk_id -> doc_id

    def _compute_hash(self, content: str) -> str:
        """Compute SHA-256 hash of document content for change detection."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def index_document(
        self,
        content: str,
        source: str = "",
        doc_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> DocumentRecord:
        """Index a single document.

        Args:
            content: The document text content.
            source: Source identifier (e.g., file path, URL).
            doc_id: Optional document ID (auto-generated if not provided).
            metadata: Optional metadata to attach to chunks.

        Returns:
            DocumentRecord for the indexed document.
        """
        import uuid

        if doc_id is None:
            doc_id = str(uuid.uuid4())

        content_hash = self._compute_hash(content)

        # Check if document already indexed with same content
        if doc_id in self._document_records:
            existing = self._document_records[doc_id]
            if existing.content_hash == content_hash:
                logger.info("Document %s unchanged, skipping.", doc_id)
                return existing
            # Content changed, remove old chunks
            self.remove_document(doc_id)

        # Chunk the document
        chunks = self.chunker.chunk(content, metadata=metadata, source_doc_id=doc_id)
        if not chunks:
            logger.warning("No chunks produced for document %s.", doc_id)
            record = DocumentRecord(
                doc_id=doc_id,
                source=source,
                content_hash=content_hash,
                num_chunks=0,
                metadata=metadata or {},
            )
            self._document_records[doc_id] = record
            return record

        # Generate embeddings in batches
        chunk_texts = [chunk.text for chunk in chunks]
        all_embeddings = []

        for i in range(0, len(chunk_texts), self.batch_size):
            batch = chunk_texts[i : i + self.batch_size]
            batch_embeddings = self.embedding_model.embed(batch)
            all_embeddings.append(batch_embeddings)

        if all_embeddings:
            embeddings = np.vstack(all_embeddings)
        else:
            embeddings = np.array([])

        # Create VectorDocuments
        vector_docs = []
        for j, chunk in enumerate(chunks):
            chunk_embedding = embeddings[j] if len(embeddings) > 0 else None
            vdoc = VectorDocument(
                doc_id=chunk.chunk_id,
                text=chunk.text,
                embedding=chunk_embedding,
                metadata={
                    **chunk.metadata,
                    "source": source,
                    "doc_id": doc_id,
                    "chunk_index": j,
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                },
            )
            vector_docs.append(vdoc)
            self._chunk_to_doc[chunk.chunk_id] = doc_id

        # Add to vector store
        self.vector_store.add(vector_docs)

        # Create and store document record
        record = DocumentRecord(
            doc_id=doc_id,
            source=source,
            content_hash=content_hash,
            num_chunks=len(chunks),
            metadata=metadata or {},
            chunk_ids=[chunk.chunk_id for chunk in chunks],
        )
        self._document_records[doc_id] = record

        logger.info(
            "Indexed document %s: %d chunks from source '%s'.",
            doc_id,
            len(chunks),
            source,
        )
        return record

    def index_documents(
        self,
        documents: List[dict],
    ) -> List[DocumentRecord]:
        """Index multiple documents.

        Args:
            documents: List of dicts with keys 'content', 'source',
                      'doc_id' (optional), 'metadata' (optional).

        Returns:
            List of DocumentRecord objects.
        """
        records = []
        for doc in documents:
            record = self.index_document(
                content=doc["content"],
                source=doc.get("source", ""),
                doc_id=doc.get("doc_id"),
                metadata=doc.get("metadata"),
            )
            records.append(record)
        return records

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document and its chunks from the index.

        Args:
            doc_id: The document ID to remove.

        Returns:
            True if the document was found and removed.
        """
        if doc_id not in self._document_records:
            return False

        record = self._document_records[doc_id]
        chunk_ids = record.chunk_ids

        # Delete chunks from vector store
        self.vector_store.delete(chunk_ids)

        # Clean up mappings
        for chunk_id in chunk_ids:
            self._chunk_to_doc.pop(chunk_id, None)

        del self._document_records[doc_id]
        logger.info("Removed document %s (%d chunks).", doc_id, len(chunk_ids))
        return True

    def update_document(
        self,
        doc_id: str,
        content: str,
        source: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> DocumentRecord:
        """Update an existing document by re-indexing it.

        Args:
            doc_id: The document ID to update.
            content: New document content.
            source: New source (keeps existing if None).
            metadata: New metadata (keeps existing if None).

        Returns:
            Updated DocumentRecord.
        """
        old_record = self._document_records.get(doc_id)
        if source is None and old_record:
            source = old_record.source
        if metadata is None and old_record:
            metadata = old_record.metadata

        self.remove_document(doc_id)
        return self.index_document(
            content=content,
            source=source or "",
            doc_id=doc_id,
            metadata=metadata,
        )

    def get_document_record(self, doc_id: str) -> Optional[DocumentRecord]:
        """Get the record for an indexed document."""
        return self._document_records.get(doc_id)

    def list_documents(self) -> List[DocumentRecord]:
        """List all indexed document records."""
        return list(self._document_records.values())

    def save(self, path: str | Path) -> None:
        """Save indexer state to disk.

        Args:
            path: Directory path for saving state.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save vector store
        self.vector_store.save(path / "vector_store")

        # Save document records
        records_path = path / "document_records.json"
        data = {
            doc_id: record.to_dict()
            for doc_id, record in self._document_records.items()
        }
        with open(records_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Save chunk-to-doc mapping
        mapping_path = path / "chunk_mapping.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(self._chunk_to_doc, f, indent=2)

        logger.info("Saved indexer state to %s (%d documents).", path, len(self._document_records))

    def load(self, path: str | Path) -> None:
        """Load indexer state from disk.

        Args:
            path: Directory path containing saved state.
        """
        path = Path(path)

        # Load vector store
        vs_path = path / "vector_store"
        if vs_path.exists():
            self.vector_store = FAISSVectorStore.load(vs_path)

        # Load document records
        records_path = path / "document_records.json"
        if records_path.exists():
            with open(records_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._document_records = {
                doc_id: DocumentRecord.from_dict(record_data)
                for doc_id, record_data in data.items()
            }

        # Load chunk-to-doc mapping
        mapping_path = path / "chunk_mapping.json"
        if mapping_path.exists():
            with open(mapping_path, "r", encoding="utf-8") as f:
                self._chunk_to_doc = json.load(f)

        logger.info("Loaded indexer state from %s (%d documents).", path, len(self._document_records))


class IncrementalIndexer(DocumentIndexer):
    """Extended indexer with incremental indexing support.

    Supports indexing only new or changed documents, tracking
    file modifications via content hashing and timestamps.
    """

    def __init__(
        self,
        chunker: TextChunker,
        embedding_model: EmbeddingModel,
        vector_store: FAISSVectorStore,
        batch_size: int = 64,
    ):
        super().__init__(chunker, embedding_model, vector_store, batch_size)
        self._source_hashes: Dict[str, str] = {}  # source -> content_hash

    def index_file(self, file_path: str | Path, metadata: Optional[dict] = None) -> Optional[DocumentRecord]:
        """Index a file, only re-indexing if content has changed.

        Args:
            file_path: Path to the file to index.
            metadata: Optional metadata.

        Returns:
            DocumentRecord if file was indexed or already current, None on error.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error("File not found: %s", file_path)
            return None

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = file_path.read_text(encoding="latin-1")
            except Exception as e:
                logger.error("Could not read file %s: %s", file_path, e)
                return None

        content_hash = self._compute_hash(content)
        source = str(file_path)

        # Check if already indexed with same content
        if source in self._source_hashes and self._source_hashes[source] == content_hash:
            logger.debug("File %s unchanged, skipping.", file_path)
            existing = self._document_records.get(source)
            return existing

        # Use source path as doc_id for consistency
        record = self.index_document(
            content=content,
            source=source,
            doc_id=source,
            metadata=metadata,
        )
        self._source_hashes[source] = content_hash
        return record

    def index_directory(
        self,
        dir_path: str | Path,
        extensions: Optional[Set[str]] = None,
        recursive: bool = True,
        metadata: Optional[dict] = None,
    ) -> List[DocumentRecord]:
        """Index all files in a directory.

        Args:
            dir_path: Path to the directory.
            extensions: Set of file extensions to include (e.g., {'.txt', '.md'}).
                       If None, includes all text files.
            recursive: Whether to search subdirectories.
            metadata: Optional metadata to attach to all documents.

        Returns:
            List of DocumentRecord objects for indexed files.
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            logger.error("Directory not found: %s", dir_path)
            return []

        default_extensions = {".txt", ".md", ".rst", ".csv", ".json", ".xml", ".html", ".py", ".js"}
        ext_filter = extensions or default_extensions

        records = []
        pattern = "**/*" if recursive else "*"

        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in ext_filter:
                record = self.index_file(file_path, metadata=metadata)
                if record:
                    records.append(record)

        logger.info("Indexed %d files from directory %s.", len(records), dir_path)
        return records

    def sync_directory(
        self,
        dir_path: str | Path,
        extensions: Optional[Set[str]] = None,
        recursive: bool = True,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Synchronize index with directory contents.

        Indexes new/modified files and removes deleted files from the index.

        Args:
            dir_path: Path to the directory.
            extensions: Set of file extensions to include.
            recursive: Whether to search subdirectories.
            metadata: Optional metadata.

        Returns:
            Dict with keys 'added', 'updated', 'removed', 'unchanged'.
        """
        dir_path = Path(dir_path)
        stats = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0}

        # Get current files
        default_extensions = {".txt", ".md", ".rst", ".csv", ".json", ".xml", ".html", ".py", ".js"}
        ext_filter = extensions or default_extensions
        pattern = "**/*" if recursive else "*"
        current_sources: Set[str] = set()

        for file_path in dir_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in ext_filter:
                source = str(file_path)
                current_sources.add(source)

                old_hash = self._source_hashes.get(source)
                content = file_path.read_text(encoding="utf-8")
                new_hash = self._compute_hash(content)

                if old_hash is None:
                    # New file
                    self.index_file(file_path, metadata=metadata)
                    stats["added"] += 1
                elif old_hash != new_hash:
                    # Modified file
                    self.index_file(file_path, metadata=metadata)
                    stats["updated"] += 1
                else:
                    stats["unchanged"] += 1

        # Remove files that no longer exist
        indexed_sources = set(self._source_hashes.keys())
        for source in indexed_sources - current_sources:
            if source.startswith(str(dir_path)):
                self.remove_document(source)
                del self._source_hashes[source]
                stats["removed"] += 1

        logger.info(
            "Synced directory %s: added=%d, updated=%d, removed=%d, unchanged=%d",
            dir_path,
            stats["added"],
            stats["updated"],
            stats["removed"],
            stats["unchanged"],
        )
        return stats
