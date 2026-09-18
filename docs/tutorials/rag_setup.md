# RAG Setup Tutorial

This tutorial guides you through building a Retrieval-Augmented Generation (RAG) pipeline with Nexus-LLM.

## What is RAG?

RAG combines document retrieval with LLM generation. When a user asks a question, the system:
1. Retrieves relevant document chunks from a knowledge base
2. Provides them as context to the LLM
3. Generates a grounded, factual response

This approach reduces hallucinations and allows the model to answer questions about private or up-to-date information.

## Step 1: Set Up the Embedding Engine

Embeddings convert text into vector representations for similarity search:

```python
from nexus_llm.rag import EmbeddingEngine

embedder = EmbeddingEngine(
    model_name="nexus-embedding-large",
    device="auto",
    batch_size=32,
    normalize_embeddings=True,
)
```

## Step 2: Configure Document Chunking

Documents are split into smaller chunks for effective retrieval:

```python
from nexus_llm.rag import DocumentStore, ChunkingStrategy

chunking = ChunkingStrategy(
    method="recursive",       # Split on paragraph/line/sentence boundaries
    chunk_size=512,           # Target chunk size in tokens
    chunk_overlap=64,         # Overlap between chunks for context continuity
)
```

### Chunking Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `recursive` | Splits on structural boundaries (paragraphs, lines, sentences) | General purpose (recommended) |
| `fixed_size` | Splits at exact token counts | When consistency matters |
| `semantic` | Splits based on semantic coherence | High-quality retrieval |

## Step 3: Ingest Documents

```python
doc_store = DocumentStore(
    storage_path="./data/doc_store",
    chunking_strategy=chunking,
)

# From files (PDF, HTML, Markdown, plain text)
doc_ids = doc_store.ingest_files([
    "./docs/product_manual.pdf",
    "./docs/api_reference.html",
    "./docs/faq.md",
])

# From raw text
doc_store.ingest_text(
    text="Nexus-LLM supports 128K context length with Flash Attention 2.",
    metadata={"source": "knowledge_base", "topic": "specs"},
)
```

## Step 4: Build the Vector Index

```python
from nexus_llm.rag import VectorIndex

vector_index = VectorIndex(
    embedding_engine=embedder,
    index_type="faiss",       # Options: faiss, hnswlib, chromadb
    metric="cosine",
    dimension=1024,
)

# Build from document chunks
chunks = doc_store.get_all_chunks()
vector_index.build(chunks)
print(f"Indexed {len(chunks)} chunks")
```

### Vector Store Comparison

| Store | Speed | Persistence | Metadata Filtering | GPU Support |
|-------|-------|-------------|-------------------|-------------|
| FAISS | Fast | Optional | No | Yes |
| HNSWLib | Fast | Optional | No | No |
| ChromaDB | Moderate | Built-in | Yes | No |

## Step 5: Create the RAG Pipeline

```python
from nexus_llm import InferenceEngine
from nexus_llm.rag import RAGPipeline, RetrievalConfig

engine = InferenceEngine(model_name="nexus-7b-chat", device="auto")

retrieval_config = RetrievalConfig(
    top_k=5,                     # Retrieve top 5 chunks
    similarity_threshold=0.65,   # Minimum similarity score
    reranking=True,              # Enable cross-encoder reranking
    rerank_top_k=3,              # Keep top 3 after reranking
    max_context_tokens=2048,     # Max tokens from retrieved context
)

pipeline = RAGPipeline(
    inference_engine=engine,
    document_store=doc_store,
    vector_index=vector_index,
    retrieval_config=retrieval_config,
)
```

## Step 6: Query the Pipeline

```python
result = pipeline.query("What GPUs does Nexus-LLM support?")

print(f"Answer: {result.answer}")
print(f"\nSources:")
for src in result.sources:
    print(f"  [{src.score:.3f}] {src.document_id} (chunk {src.chunk_index})")
    print(f"       {src.text[:100]}...")
```

## Step 7: Incremental Updates

Add new documents without rebuilding the entire index:

```python
# Add a new document
new_id = doc_store.ingest_text(
    text="Version 2.1 adds support for Mamba architectures.",
    metadata={"source": "release_notes", "version": "2.1"},
)

# Update the index incrementally
new_chunks = doc_store.get_chunks(new_id)
vector_index.add(new_chunks)
```

## Step 8: Conversational RAG

Maintain context across multiple turns:

```python
from nexus_llm import Conversation

conversation = Conversation()
conversation.add_user_message("What are the system requirements?")
result = pipeline.query_conversation(conversation)

conversation.add_user_message("Can it run on a laptop?")
result = pipeline.query_conversation(conversation)
```

## Best Practices

1. **Chunk size matters**: 256-512 tokens is the sweet spot for most use cases
2. **Use reranking**: Cross-encoder reranking significantly improves retrieval quality
3. **Include metadata**: Tag documents with source, version, and category for filtering
4. **Set similarity thresholds**: Filter out irrelevant chunks to reduce noise
5. **Limit context length**: Too much context can confuse the model; stick to 2K-4K tokens
6. **Test retrieval quality**: Evaluate precision@k and recall@k on a held-out set
7. **Update incrementally**: For production systems, update the index as documents change

## Next Steps

- **[Evaluation Tutorial](../tutorials/getting_started.md)** - Evaluate RAG pipeline quality
- **[Performance Tuning Guide](../guides/performance_tuning.md)** - Optimize retrieval speed
- **[Deployment Guide](../guides/deployment.md)** - Deploy RAG in production
