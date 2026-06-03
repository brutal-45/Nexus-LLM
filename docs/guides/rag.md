# RAG (Retrieval-Augmented Generation) Guide

Learn how to set up retrieval-augmented generation with Nexus-LLM to give your models access to external knowledge.

---

## What is RAG?

Retrieval-Augmented Generation (RAG) enhances LLM responses by retrieving relevant documents from a knowledge base and including them in the model's context. This allows the model to:

- Answer questions about **proprietary or recent** information not in its training data
- Provide **cited, verifiable** answers with source references
- Reduce **hallucinations** by grounding responses in retrieved facts
- Work with **private documents** without retraining

### How RAG Works in Nexus-LLM

```
User Query
    │
    ▼
┌─────────────┐     ┌──────────────────┐
│  Embed Query │────▶│  Vector Search   │
└─────────────┘     └────────┬─────────┘
                             │
                    Top-K relevant chunks
                             │
                             ▼
                    ┌──────────────────┐
                    │  Build Augmented │
                    │  Prompt          │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  LLM Inference   │
                    └────────┬─────────┘
                             │
                             ▼
                    Generated Response + Sources
```

---

## Step 1: Enable RAG

Edit your configuration file:

```yaml
# config/default.yaml
rag:
  enabled: true
  chunk_size: 512
  chunk_overlap: 64
  retrieval_top_k: 5
  similarity_threshold: 0.7
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  vector_store: "chromadb"      # chromadb, faiss, or qdrant
  persist_directory: "./cache/vector_store"
```

Or use environment variables:

```bash
export NEXUS_RAG_ENABLED=true
export NEXUS_RAG_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
```

---

## Step 2: Index Documents

### Via CLI

```bash
# Index a single file
nexus rag index --file ./documents/report.pdf

# Index a directory of files
nexus rag index --dir ./documents/

# Index with a specific collection
nexus rag index --dir ./documents/ --collection company_docs

# Index with custom chunk settings
nexus rag index --dir ./documents/ --chunk-size 256 --chunk-overlap 32
```

### Via API

```bash
# Upload and index files
curl -X POST http://localhost:8000/api/v1/rag/index \
  -H "Authorization: Bearer nexus_your_api_key" \
  -F "files=@report.pdf" \
  -F "files=@notes.txt" \
  -F "files=@data.json" \
  -F "collection_name=company_docs" \
  -F "chunk_size=256"
```

**Response:**

```json
{
  "collection": "company_docs",
  "documents_indexed": 3,
  "chunks_created": 47,
  "elapsed_seconds": 3.2
}
```

### Supported File Types

| Format | Extension | Notes |
|--------|-----------|-------|
| Plain Text | `.txt` | Direct text extraction |
| Markdown | `.md` | Preserves headers as section boundaries |
| PDF | `.pdf` | Extracts text with layout preservation |
| JSON | `.json` | Flattened to text |
| JSONL | `.jsonl` | Each line is a document |
| CSV | `.csv` | Each row is a document |
| HTML | `.html`, `.htm` | Strips tags, preserves text |
| DOCX | `.docx` | Microsoft Word documents |

### Programmatic Indexing

```python
from nexus_llm.rag import DocumentIndexer

indexer = DocumentIndexer(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    vector_store="chromadb",
    chunk_size=512,
    chunk_overlap=64,
)

# Index files
indexer.index_files(
    paths=["./documents/report.pdf", "./documents/notes.md"],
    collection="my_docs"
)

# Index raw text
indexer.index_text(
    text="Nexus-LLM is a powerful language model framework...",
    metadata={"source": "product_overview", "author": "team"},
    collection="my_docs"
)

# Index from a HuggingFace dataset
indexer.index_dataset(
    dataset_name="wikipedia",
    subset="20231101.en",
    split="train",
    text_column="text",
    collection="wikipedia",
    max_documents=10000
)
```

---

## Step 3: Configure the Retrieval Pipeline

### Embedding Models

Choose an embedding model that balances quality and speed:

| Model | Dimensions | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good | General purpose, fast retrieval |
| `all-mpnet-base-v2` | 768 | Medium | Better | Higher accuracy retrieval |
| `e5-base-v2` | 768 | Medium | Better | Multilingual, instruction-tuned |
| `bge-large-en-v1.5` | 1024 | Slower | Best | Highest accuracy English |
| `multilingual-e5-base` | 768 | Medium | Good | Multilingual documents |

```yaml
rag:
  embedding_model: "sentence-transformers/all-mpnet-base-v2"
```

### Vector Stores

| Store | Persistence | Scalability | Setup | Best For |
|-------|------------|-------------|-------|----------|
| **ChromaDB** | Yes | Medium | Zero-config | Local development, small-medium collections |
| **FAISS** | Optional | Large | Zero-config | High-performance, in-memory search |
| **Qdrant** | Yes | Large | Requires server | Production, distributed deployments |

#### ChromaDB (Default)

```yaml
rag:
  vector_store: "chromadb"
  persist_directory: "./cache/chromadb"
```

#### FAISS

```yaml
rag:
  vector_store: "faiss"
  faiss:
    index_type: "IVFFlat"     # Flat, IVFFlat, IVFPQ, HNSW
    nlist: 100                # Number of clusters (for IVF)
    nprobe: 10                # Clusters to search
    save_index: true
    index_path: "./cache/faiss/index.bin"
```

#### Qdrant

```yaml
rag:
  vector_store: "qdrant"
  qdrant:
    url: "http://localhost:6333"
    collection_name: "nexus_docs"
    api_key: null
    prefer_grpc: false
```

### Chunking Strategies

#### Fixed-Size Chunking

Simple, predictable chunk sizes:

```yaml
rag:
  chunking:
    strategy: "fixed"
    chunk_size: 512       # Tokens per chunk
    chunk_overlap: 64     # Overlap between chunks
```

#### Semantic Chunking

Chunks at natural semantic boundaries:

```yaml
rag:
  chunking:
    strategy: "semantic"
    max_chunk_size: 512
    similarity_threshold: 0.85   # Split when similarity drops below this
```

#### Sentence Chunking

Chunks at sentence boundaries:

```yaml
rag:
  chunking:
    strategy: "sentence"
    sentences_per_chunk: 10
    chunk_overlap: 2       # Overlapping sentences
```

---

## Step 4: Query Documents

### Via CLI

```bash
# Query your indexed documents
nexus rag query "What is the company refund policy?"
```

### Via API

```bash
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Authorization: Bearer nexus_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the refund policy?",
    "collection_name": "company_docs",
    "top_k": 5,
    "similarity_threshold": 0.5
  }'
```

**Response:**

```json
{
  "results": [
    {
      "content": "Refund Policy: Customers may request a full refund within 30 days of purchase...",
      "metadata": {
        "source": "refund_policy.pdf",
        "page": 3,
        "chunk_index": 12
      },
      "score": 0.89
    },
    {
      "content": "For digital products, refunds are processed within 5-7 business days...",
      "metadata": {
        "source": "refund_policy.pdf",
        "page": 4,
        "chunk_index": 15
      },
      "score": 0.82
    }
  ]
}
```

### Via Chat (Automatic RAG)

When RAG is enabled, relevant documents are automatically injected into the chat context:

```bash
./scripts/run.sh --mode chat

You> What is the refund policy?

Assistant> Based on the company documentation:

Customers may request a full refund within 30 days of purchase. For digital
products, refunds are processed within 5-7 business days [Source: refund_policy.pdf, p.3].

To request a refund, contact support@example.com with your order number [Source: refund_policy.pdf, p.4].
```

### Programmatic Query

```python
from nexus_llm.rag import RAGPipeline

rag = RAGPipeline(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    vector_store="chromadb",
    collection="company_docs"
)

# Retrieve relevant documents
results = rag.retrieve("What is the refund policy?", top_k=5)
for result in results:
    print(f"[{result.score:.2f}] {result.content[:100]}...")

# Generate an answer with citations
answer = rag.generate("What is the refund policy?", include_sources=True)
print(answer.text)
for source in answer.sources:
    print(f"  Source: {source.metadata['source']}, page {source.metadata.get('page', '?')}")
```

---

## Step 5: Advanced RAG Techniques

### Hybrid Search

Combine semantic (vector) search with keyword (BM25) search for better retrieval:

```yaml
rag:
  retrieval:
    strategy: "hybrid"
    semantic_weight: 0.7
    keyword_weight: 0.3
    bm25:
      k1: 1.5
      b: 0.75
```

### Re-ranking

Apply a cross-encoder re-ranker after initial retrieval for improved relevance:

```yaml
rag:
  reranking:
    enabled: true
    model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: 20          # Retrieve more candidates
    rerank_top_k: 5    # Return top-K after re-ranking
```

### Parent-Child Chunking

Retrieve small chunks but return the surrounding context:

```yaml
rag:
  chunking:
    strategy: "parent_child"
    parent_chunk_size: 2000
    child_chunk_size: 256
    retrieval_mode: "child"    # Search on child chunks
    return_mode: "parent"      # Return parent chunk for context
```

### Multi-Collection Query

Search across multiple document collections:

```bash
nexus rag query "How do I deploy the application?" \
  --collection docs \
  --collection runbooks \
  --collection faq
```

---

## Managing Collections

### List Collections

```bash
nexus rag list-collections
# OR
curl http://localhost:8000/api/v1/rag/collections \
  -H "Authorization: Bearer nexus_your_api_key"
```

### Delete a Collection

```bash
nexus rag delete-collection --name company_docs
```

### Collection Stats

```bash
nexus rag stats --collection company_docs
```

**Output:**

```
Collection: company_docs
  Documents: 42
  Chunks: 312
  Total tokens: ~156,000
  Embedding model: all-MiniLM-L6-v2
  Created: 2024-01-01T00:00:00Z
  Last updated: 2024-01-15T12:34:56Z
```

---

## Performance Tuning

| Setting | Low Resource | Balanced | High Accuracy |
|---------|-------------|----------|---------------|
| Embedding model | MiniLM-L6 (384d) | mpnet-base (768d) | bge-large (1024d) |
| Chunk size | 256 | 512 | 1024 |
| Chunk overlap | 32 | 64 | 128 |
| Top-K | 3 | 5 | 10 |
| Re-ranking | Off | Optional | On |
| Vector store | FAISS | ChromaDB | Qdrant |
