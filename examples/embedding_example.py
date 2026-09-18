#!/usr/bin/env python3
"""
Embeddings Example - Nexus-LLM
================================
Demonstrates how to generate and use text embeddings for
similarity search, clustering, and classification.
"""

import numpy as np
from nexus_llm import InferenceEngine
from nexus_llm.embeddings import EmbeddingEngine, EmbeddingUtils


def main():
    # --- Initialize the embedding engine ---
    embedder = EmbeddingEngine(
        model_name="nexus-embedding-large",
        device="auto",
        batch_size=32,
        normalize_embeddings=True,    # L2 normalize for cosine similarity
        max_length=512,
    )

    # --- Generate single embedding ---
    print("=" * 60)
    print("Single Embedding")
    print("=" * 60)

    text = "Nexus-LLM is a powerful framework for building LLM applications."
    embedding = embedder.embed(text)
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    print(f"First 5 values: {embedding[:5]}")

    # --- Batch embedding ---
    print("\n" + "=" * 60)
    print("Batch Embedding")
    print("=" * 60)

    texts = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "Machine learning is revolutionizing technology.",
        "The stock market reached new highs today.",
        "Python is a popular programming language.",
    ]

    embeddings = embedder.embed_batch(texts)
    print(f"Generated {len(embeddings)} embeddings of shape {embeddings[0].shape}")

    # --- Similarity search ---
    print("\n" + "=" * 60)
    print("Semantic Similarity")
    print("=" * 60)

    utils = EmbeddingUtils()

    # Pairwise similarities
    query = "A cat is sitting on something"
    query_embedding = embedder.embed(query)

    similarities = utils.cosine_similarity(query_embedding, embeddings)
    ranked = sorted(zip(texts, similarities), key=lambda x: x[1], reverse=True)

    print(f"Query: {query}")
    print("\nRanked by similarity:")
    for text, score in ranked:
        print(f"  [{score:.4f}] {text}")

    # --- Clustering ---
    print("\n" + "=" * 60)
    print("Embedding Clustering")
    print("=" * 60)

    documents = [
        "Python is great for data science.",
        "R is popular in statistical computing.",
        "JavaScript powers the modern web.",
        "React is a JavaScript framework.",
        "Pandas is a Python data analysis library.",
        "Neural networks learn from data.",
        "Deep learning uses multiple layers.",
        "TypeScript adds types to JavaScript.",
    ]

    doc_embeddings = embedder.embed_batch(documents)
    clusters = utils.cluster(doc_embeddings, n_clusters=3, method="kmeans")

    print("Cluster assignments:")
    for cluster_id in range(3):
        members = [documents[i] for i, c in enumerate(clusters) if c == cluster_id]
        print(f"\n  Cluster {cluster_id}:")
        for doc in members:
            print(f"    - {doc}")

    # --- Visualization (2D projection) ---
    print("\n" + "=" * 60)
    print("2D Projection")
    print("=" * 60)

    coords_2d = utils.reduce_dimensions(
        doc_embeddings,
        method="tsne",
        n_components=2,
    )

    for doc, (x, y) in zip(documents, coords_2d):
        print(f"  ({x:.2f}, {y:.2f}) - {doc[:50]}...")

    # --- Embedding for classification ---
    print("\n" + "=" * 60)
    print("Embedding-based Classification")
    print("=" * 60)

    # Simple few-shot classification using embedding similarity
    categories = {
        "technology": "News about software, hardware, and tech companies",
        "sports": "News about athletic competitions and sporting events",
        "finance": "News about markets, economy, and financial institutions",
    }

    category_embeddings = {
        name: embedder.embed(desc) for name, desc in categories.items()
    }

    new_articles = [
        "Apple announced the new M4 chip with groundbreaking performance.",
        "The Lakers defeated the Celtics in overtime last night.",
        "Federal Reserve signals potential rate cuts in the next quarter.",
    ]

    for article in new_articles:
        article_embedding = embedder.embed(article)
        scores = {
            name: utils.cosine_similarity_pair(article_embedding, cat_emb)
            for name, cat_emb in category_embeddings.items()
        }
        predicted = max(scores, key=scores.get)
        print(f"  [{predicted}] {article[:60]}... (scores: {scores})")


if __name__ == "__main__":
    main()
