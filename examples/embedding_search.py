#!/usr/bin/env python3
"""Embedding search example."""
from nexus_llm.embeddings import EmbeddingEngine, EmbeddingStore

engine = EmbeddingEngine()
store = EmbeddingStore()

docs = ['Python is great', 'Machine learning is fun', 'Data science rocks']
for i, doc in enumerate(docs):
    emb = engine.embed(doc)
    store.add(str(i), emb, {'text': doc})

query_emb = engine.embed('programming')
results = store.search(query_emb, top_k=2)
for id, score, meta in results:
    print(f'{meta["text"]}: {score:.3f}')
