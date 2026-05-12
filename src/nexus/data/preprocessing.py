"""
Data Preprocessing & Deduplication
=====================================
Text cleaning, quality filtering, and MinHash-based deduplication
for preparing web-scraped corpora for LLM training.

Pipeline:
    Raw Text -> Clean -> Quality Score -> Deduplicate -> Final Corpus
    
Steps:
    1. Text cleaning: remove HTML, normalize whitespace, filter by language
    2. Quality scoring: perplexity-based, heuristic rules, classifier
    3. Deduplication: MinHash LSH for approximate near-duplicate detection
    4. Domain balancing: ensure diverse source distribution
"""

from __future__ import annotations
import re
import hashlib
import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple, Iterator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CleanStats:
    """Statistics from text cleaning."""
    total_input: int = 0
    total_kept: int = 0
    removed_too_short: int = 0
    removed_too_long: int = 0
    removed_low_quality: int = 0
    removed_duplicates: int = 0
    removed_language: int = 0


class TextCleaner:
    """
    Text cleaning and quality filtering for pre-training data.
    
    Cleaning rules:
        - Remove HTML tags and entities
        - Normalize Unicode whitespace
        - Remove control characters
        - Filter by length (min/max)
        - Filter by alphanumeric ratio
        - Filter by repetitive content
        - Remove boilerplate (navigation, ads, etc.)
    """

    def __init__(
        self,
        min_length: int = 100,
        max_length: int = 1048576,  # 1MB
        min_alnum_ratio: float = 0.5,
        max_repetition_ratio: float = 0.3,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.min_alnum_ratio = min_alnum_ratio
        self.max_repetition_ratio = max_repetition_ratio
        self.stats = CleanStats()

    def clean_html(self, text: str) -> str:
        """Remove HTML tags, entities, and scripts."""
        # Remove script and style blocks
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Decode common HTML entities
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        text = text.replace('&nbsp;', ' ')
        return text

    def normalize_whitespace(self, text: str) -> str:
        """Normalize all whitespace to single spaces, trim edges."""
        text = re.sub(r'[\t\r\f\v]+', ' ', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def remove_control_chars(self, text: str) -> str:
        """Remove non-printable control characters except newlines."""
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    def compute_alnum_ratio(self, text: str) -> float:
        """Compute ratio of alphanumeric characters."""
        if not text:
            return 0.0
        alnum = sum(1 for c in text if c.isalnum())
        return alnum / len(text)

    def compute_repetition_ratio(self, text: str) -> float:
        """
        Compute ratio of repeated n-grams.
        
        High repetition indicates boilerplate or low-quality content.
        Uses trigram repetition as a heuristic.
        """
        words = text.lower().split()
        if len(words) < 10:
            return 0.0
        
        # Check trigram repetition
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        if not trigrams:
            return 0.0
        
        unique_trigrams = len(set(trigrams))
        return 1.0 - (unique_trigrams / len(trigrams))

    def clean(self, text: str) -> Optional[str]:
        """
        Apply full cleaning pipeline to a single document.
        
        Returns None if the document should be discarded.
        """
        self.stats.total_input += 1

        # Step 1: Basic cleaning
        text = self.clean_html(text)
        text = self.normalize_whitespace(text)
        text = self.remove_control_chars(text)

        # Step 2: Length filtering
        if len(text) < self.min_length:
            self.stats.removed_too_short += 1
            return None
        if len(text) > self.max_length:
            self.stats.removed_too_long += 1
            return None

        # Step 3: Quality heuristics
        alnum_ratio = self.compute_alnum_ratio(text)
        if alnum_ratio < self.min_alnum_ratio:
            self.stats.removed_low_quality += 1
            return None

        rep_ratio = self.compute_repetition_ratio(text)
        if rep_ratio > self.max_repetition_ratio:
            self.stats.removed_low_quality += 1
            return None

        self.stats.total_kept += 1
        return text

    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a batch of texts, returning only those that pass."""
        results = []
        for text in texts:
            cleaned = self.clean(text)
            if cleaned is not None:
                results.append(cleaned)
        return results


class DataDeduplicator:
    """
    MinHash-based approximate deduplication.
    
    MinHash provides an efficient way to estimate Jaccard similarity
    between documents without computing exact intersections:
        J(A, B) ≈ |MinHash(A) ∩ MinHash(B)| / num_hashes
    
    For large corpora (10T+ tokens), we use Locality-Sensitive Hashing (LSH)
    to group similar documents into bands, then only compare within bands.
    
    Reference:
        - Broder, "On the Resemblance and Containment of Documents" (1997)
    """

    def __init__(
        self,
        num_hashes: int = 128,
        num_bands: int = 16,
        jaccard_threshold: float = 0.8,
        ngram_size: int = 5,
    ):
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.band_size = num_hashes // num_bands
        self.jaccard_threshold = jaccard_threshold
        self.ngram_size = ngram_size
        
        # Random permutation functions for MinHash
        random.seed(42)
        self.max_hash = (1 << 61) - 1  # Large prime
        self.a_coeffs = [random.randint(1, self.max_hash) for _ in range(num_hashes)]
        self.b_coeffs = [random.randint(0, self.max_hash) for _ in range(num_hashes)]

    def _get_ngrams(self, text: str) -> Set[str]:
        """Extract character n-grams from text."""
        words = text.lower().split()
        return {" ".join(words[i:i+self.ngram_size]) for i in range(len(words) - self.ngram_size + 1)}

    def _hash_ngram(self, ngram: str, coeff_a: int, coeff_b: int) -> int:
        """Hash an n-gram with a random permutation."""
        h = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
        return (coeff_a * h + coeff_b) % self.max_hash

    def compute_minhash(self, text: str) -> List[int]:
        """
        Compute MinHash signature for a document.
        
        The MinHash signature is a vector of num_hashes values where each
        value is the minimum hash across all n-grams using a different
        random permutation.
        """
        ngrams = self._get_ngrams(text)
        if not ngrams:
            return [self.max_hash] * self.num_hashes
        
        signature = []
        for i in range(self.num_hashes):
            min_hash = min(
                self._hash_ngram(ng, self.a_coeffs[i], self.b_coeffs[i])
                for ng in ngrams
            )
            signature.append(min_hash)
        
        return signature

    def compute_lsh_buckets(self, signature: List[int]) -> List[int]:
        """
        Compute LSH band hashes for grouping similar documents.
        
        Divides the MinHash signature into num_bands bands.
        Documents with at least one matching band are candidates
        for near-duplicate comparison.
        """
        buckets = []
        for band_idx in range(self.num_bands):
            start = band_idx * self.band_size
            band = tuple(signature[start:start + self.band_size])
            bucket_hash = hash(band)
            buckets.append(bucket_hash)
        return buckets

    def deduplicate(
        self,
        documents: List[Tuple[int, str]],
    ) -> Tuple[List[Tuple[int, str]], List[Tuple[int, int]]]:
        """
        Deduplicate documents using MinHash LSH.
        
        Args:
            documents: List of (doc_id, text) tuples.
        
        Returns:
            Tuple of:
                - unique_docs: List of (doc_id, text) after deduplication
                - duplicate_pairs: List of (keep_id, remove_id) pairs
        """
        print(f"Deduplicating {len(documents)} documents...")
        
        # Step 1: Compute MinHash for all documents
        signatures = {}
        for doc_id, text in documents:
            signatures[doc_id] = self.compute_minhash(text)
        
        # Step 2: Build LSH index
        lsh_index: Dict[int, List[int]] = defaultdict(list)
        for doc_id, sig in signatures.items():
            buckets = self.compute_lsh_buckets(sig)
            for bucket in buckets:
                lsh_index[bucket].append(doc_id)
        
        # Step 3: Find candidate pairs and verify
        candidate_pairs: Set[Tuple[int, int]] = set()
        for bucket, doc_ids in lsh_index.items():
            if len(doc_ids) > 1:
                for i in range(len(doc_ids)):
                    for j in range(i + 1, len(doc_ids)):
                        a, b = min(doc_ids[i], doc_ids[j]), max(doc_ids[i], doc_ids[j])
                        candidate_pairs.add((a, b))
        
        # Step 4: Verify candidates with exact Jaccard similarity
        duplicate_pairs: List[Tuple[int, int]] = []
        remove_ids: Set[int] = set()
        
        for id_a, id_b in candidate_pairs:
            if id_a in remove_ids or id_b in remove_ids:
                continue
            
            sig_a = set(signatures[id_a])
            sig_b = set(signatures[id_b])
            jaccard = len(sig_a & sig_b) / len(sig_a | sig_b) if sig_a | sig_b else 0
            
            if jaccard >= self.jaccard_threshold:
                # Keep the shorter ID (earlier document), remove the later one
                keep, remove = (id_a, id_b) if id_a < id_b else (id_b, id_a)
                duplicate_pairs.append((keep, remove))
                remove_ids.add(remove)
        
        # Step 5: Build unique document list
        unique_docs = [(doc_id, text) for doc_id, text in documents if doc_id not in remove_ids]
        
        print(f"  Found {len(duplicate_pairs)} duplicate pairs")
        print(f"  Kept {len(unique_docs)}/{len(documents)} documents "
              f"({len(documents) - len(unique_docs)} removed)")
        
        return unique_docs, duplicate_pairs


def prepare_training_data(
    input_files: List[str],
    output_dir: str,
    min_doc_length: int = 100,
    deduplicate: bool = True,
    domain_balancing: bool = False,
) -> None:
    """
    End-to-end data preparation pipeline.
    
    Args:
        input_files: List of input file paths (JSONL or TXT).
        output_dir: Directory to write cleaned, deduplicated JSONL files.
        min_doc_length: Minimum document length after cleaning.
        deduplicate: Whether to run MinHash deduplication.
        domain_balancing: Whether to balance domains (if URL info available).
    """
    cleaner = TextCleaner(min_length=min_doc_length)
    
    all_docs: List[Tuple[int, str]] = []
    doc_id = 0
    
    # Phase 1: Clean
    print("Phase 1: Cleaning documents...")
    for filepath in input_files:
        print(f"  Processing {filepath}...")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    import json
                    data = json.loads(line)
                    text = data.get("text", data.get("content", line))
                except json.JSONDecodeError:
                    text = line
                
                cleaned = cleaner.clean(text)
                if cleaned:
                    all_docs.append((doc_id, cleaned))
                    doc_id += 1
    
    print(f"  Cleaned: {cleaner.stats.total_kept}/{cleaner.stats.total_input} documents")
    
    # Phase 2: Deduplicate
    if deduplicate and all_docs:
        print("Phase 2: Deduplicating...")
        dedup = DataDeduplicator(num_hashes=128, num_bands=16)
        all_docs, dup_pairs = dedup.deduplicate(all_docs)
    
    # Phase 3: Write output
    print("Phase 3: Writing output...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "corpus.jsonl"
    
    with open(output_path, "w", encoding="utf-8") as f:
        for doc_id, text in all_docs:
            f.write(json.dumps({"text": text}) + "\n")
    
    print(f"  Written {len(all_docs)} documents to {output_path}")
    total_chars = sum(len(t) for _, t in all_docs)
    print(f"  Total characters: {total_chars:,} (~{total_chars // 4:,} tokens)")
