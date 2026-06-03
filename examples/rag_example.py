#!/usr/bin/env python3
"""
RAG Pipeline Example - Nexus-LLM
==================================
Demonstrates a Retrieval-Augmented Generation (RAG) pipeline
with document ingestion, indexing, and query-time retrieval.
"""

from nexus_llm import InferenceEngine, Conversation
from nexus_llm.rag import (
    DocumentStore,
    ChunkingStrategy,
    EmbeddingEngine,
    VectorIndex,
    RAGPipeline,
    RetrievalConfig,
)


def main():
    # --- Step 1: Set up the embedding engine ---
    embedder = EmbeddingEngine(
        model_name="nexus-embedding-large",
        device="auto",
        batch_size=32,
        normalize_embeddings=True,
    )

    # --- Step 2: Configure document chunking ---
    chunking_strategy = ChunkingStrategy(
        method="recursive",
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function="token",
    )

    # --- Step 3: Create the document store and vector index ---
    doc_store = DocumentStore(
        storage_path="./data/document_store",
        chunking_strategy=chunking_strategy,
    )

    vector_index = VectorIndex(
        embedding_engine=embedder,
        index_type="faiss",         # Options: faiss, hnswlib, chromadb
        metric="cosine",
        dimension=1024,
        index_path="./data/vector_index",
    )

    # --- Step 4: Ingest documents ---
    print("Ingesting documents...")

    # From files
    doc_ids = doc_store.ingest_files(
        paths=[
            "./docs/product_manual.pdf",
            "./docs/faq.md",
            "./docs/api_reference.html",
        ],
        metadata={
            "source": "internal_docs",
            "version": "2.1",
        },
    )
    print(f"Ingested {len(doc_ids)} documents")

    # From raw text
    doc_ids += doc_store.ingest_text(
        text="Nexus-LLM supports multiple model architectures including "
             "transformer-based and Mamba-based models. The system can "
             "handle up to 128K context length with Flash Attention 2.",
        metadata={"source": "knowledge_base", "topic": "features"},
    )

    # Build the vector index from ingested chunks
    print("Building vector index...")
    chunks = doc_store.get_all_chunks()
    vector_index.build(chunks)
    print(f"Indexed {len(chunks)} chunks")

    # --- Step 5: Create the RAG pipeline ---
    retrieval_config = RetrievalConfig(
        top_k=5,                         # Number of chunks to retrieve
        similarity_threshold=0.7,        # Minimum similarity score
        reranking=True,                  # Enable cross-encoder reranking
        rerank_top_k=3,                  # Chunks after reranking
        max_context_tokens=2048,         # Maximum context window for retrieved text
    )

    engine = InferenceEngine(model_name="nexus-7b-chat", device="auto")

    pipeline = RAGPipeline(
        inference_engine=engine,
        document_store=doc_store,
        vector_index=vector_index,
        retrieval_config=retrieval_config,
    )

    # --- Step 6: Query the RAG pipeline ---
    print("\n--- Querying RAG Pipeline ---\n")

    queries = [
        "What is the maximum context length supported?",
        "How do I configure the API server?",
        "What are the supported model architectures?",
    ]

    for query in queries:
        result = pipeline.query(query)
        print(f"Question: {query}")
        print(f"Answer: {result.answer}")
        print(f"Sources ({len(result.sources)}):")
        for src in result.sources:
            print(f"  - [{src.score:.3f}] {src.document_id} (chunk {src.chunk_index})")
        print()

    # --- Step 7: Conversational RAG ---
    print("--- Conversational RAG ---\n")

    conversation = Conversation()
    conversation.add_user_message("Tell me about the product's features.")
    result = pipeline.query_conversation(conversation)
    print(f"Answer: {result.answer}")

    conversation.add_user_message("How does it compare to competitors?")
    result = pipeline.query_conversation(conversation)
    print(f"Follow-up Answer: {result.answer}")


if __name__ == "__main__":
    main()
