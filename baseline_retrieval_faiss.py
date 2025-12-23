import json
import os
import time
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import numpy as np  # Added for array handling if needed

# --- SETTINGS (defaults; can be overridden with CLI args) ---
BASE_DIR = "./long_context"
DEFAULT_PARA_FILE = os.path.join(BASE_DIR, "paragraphs", "paragraphs.jsonl")
DEFAULT_QA_FILE = os.path.join(BASE_DIR, "qa_pairs.json")
INDEX_DIR = os.path.join(BASE_DIR, "indices")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_NAME = "all-MiniLM-L6-v2"

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


@dataclass
class RetrievalResult:
    rank: int
    doc_id: str
    para_id: int
    score: float
    text: str


class BaselineRetriever:
    """
    Dense retrieval baseline:
      - encodes each paragraph/chunk with SentenceTransformers
      - indexes vectors with FAISS IndexFlatL2 OR IndexPQ (Compressed)
      - retrieves top-k nearest vectors for a query
    """
    def __init__(self, model_name: str = MODEL_NAME):
        print(f"Model loading: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index: Optional[faiss.Index] = None
        # FAISS internal id -> metadata
        self.metadata: List[Dict[str, Any]] = []  # {"doc_id","para_id","text"}

    def load_and_index(self, jsonl_path: str, save_name: str = "baseline.faiss", compress: bool = False) -> None:
        """Read paragraphs, encode, build FAISS index, save it to disk."""
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"{jsonl_path} not found.")

        print(f"Reading paragraphs from: {jsonl_path}")
        texts: List[str] = []
        self.metadata = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                self.metadata.append(
                    {
                        "doc_id": data["doc_id"],
                        "para_id": int(data["para_id"]),
                        "text": data["text"],
                    }
                )
                texts.append(data["text"])

        print(f"Total {len(texts)} paragraphs/chunks to embed.")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # ---------------------------------------------------------
        # NEW: Compression Logic (Product Quantization)
        # ---------------------------------------------------------
        if compress:
            print("[INFO] Compression enabled: Training Product Quantizer (PQ)...")
            # For 384 dimensions (MiniLM), we split into m=48 sub-vectors (8 dims each).
            # nbits=8 means each sub-vector is encoded in 1 byte.
            m = 48
            nbits = 8
            self.index = faiss.IndexPQ(self.dimension, m, nbits)
            
            # PQ requires training on the distribution of vectors
            self.index.train(embeddings)
            print("[INFO] Adding vectors to Compressed Index...")
            self.index.add(embeddings)
        else:
            print("Building FAISS index (IndexFlatL2)...")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)

        out_path = os.path.join(INDEX_DIR, save_name)
        faiss.write_index(self.index, out_path)
        print(f"Index saved: {out_path} ({len(embeddings)} vectors)")

    def load_index_from_disk(self, index_name: str, jsonl_path: str):
        """Helper to load existing index without re-embedding."""
        index_path = os.path.join(INDEX_DIR, index_name)
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index {index_path} not found.")
        
        print(f"[INFO] Loading index from {index_path}...")
        self.index = faiss.read_index(index_path)
        
        # Reload metadata
        self.metadata = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        self.metadata.append({
                            "doc_id": data["doc_id"],
                            "para_id": int(data["para_id"]),
                            "text": data["text"],
                        })
                    except: continue

    def search(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """Return top-k nearest paragraphs for a query."""
        if self.index is None:
            raise ValueError("Index not built yet. Call load_and_index() first.")

        query_vec = self.model.encode([query], convert_to_numpy=True)
        # If using cosine similarity:
        # faiss.normalize_L2(query_vec)

        distances, indices = self.index.search(query_vec, k)

        results: List[RetrievalResult] = []
        for rank_idx, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append(
                    RetrievalResult(
                        rank=rank_idx + 1,
                        doc_id=meta["doc_id"],
                        para_id=int(meta["para_id"]),
                        score=float(distances[0][rank_idx]),
                        text=meta["text"],
                    )
                )
        return results


def _binary_rels(pred_pairs: List[Tuple[str, int]], gold_pairs: set) -> List[int]:
    return [1 if p in gold_pairs else 0 for p in pred_pairs]


def precision_at_k(rels: List[int], k: int) -> float:
    if k <= 0:
        return 0.0
    rels_k = rels[:k]
    return sum(rels_k) / k


def recall_hit_at_k(rels: List[int], k: int) -> bool:
    """Your existing 'Recall@k' definition: hit if ANY gold appears in top-k."""
    return any(rels[:k])


def mrr_at_k(rels: List[int], k: int) -> float:
    for i, r in enumerate(rels[:k], start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rels: List[int], k: int, num_gold: int) -> float:
    """
    Binary NDCG@k:
      DCG = sum(rel_i / log2(i+1))
      IDCG = best possible DCG with min(num_gold, k) ones at the front
    """
    rels_k = rels[:k]
    dcg = 0.0
    for i, r in enumerate(rels_k, start=1):
        if r:
            dcg += 1.0 / math.log2(i + 1)

    ideal_ones = min(max(num_gold, 0), k)
    if ideal_ones == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_ones + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_system(
    retriever: BaselineRetriever,
    qa_file: str,
    *,
    k_eval: int = 10,
    out_csv_name: str = "baseline_results.csv",
) -> None:
    """
    Evaluates retrieval.
    - Uses gold_para_ids to build the gold set as (doc_id, para_id).
    - Reports hit-based Recall@1/@3/@10 (same notion you used before).
    - Also reports Precision@1/@3/@10, MRR@10, NDCG@10, and Avg latency.
    """
    print("\nStarting evaluation...")

    with open(qa_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    totals = {
        "n": 0,
        "recall@1": 0,
        "recall@3": 0,
        "recall@10": 0,
        "p@1": 0.0,
        "p@3": 0.0,
        "p@10": 0.0,
        "mrr@10": 0.0,
        "ndcg@10": 0.0,
        "lat_ms_sum": 0.0,
    }

    rows: List[Dict[str, Any]] = []

    for item in tqdm(qa_data, desc="Processing questions"):
        qid = item.get("qid")
        question = item.get("question", "")
        doc_id = item.get("doc_id")
        gold_para_ids = item.get("gold_para_ids", [])

        if not doc_id or not question or not gold_para_ids:
            # Skip invalid / unannotated
            continue

        totals["n"] += 1
        gold_pairs = {(doc_id, int(pid)) for pid in gold_para_ids}

        t0 = time.perf_counter()
        preds = retriever.search(question, k=k_eval)
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0
        totals["lat_ms_sum"] += lat_ms

        pred_pairs = [(p.doc_id, p.para_id) for p in preds]
        rels = _binary_rels(pred_pairs, gold_pairs)

        hit1 = recall_hit_at_k(rels, 1)
        hit3 = recall_hit_at_k(rels, 3)
        hit10 = recall_hit_at_k(rels, 10)

        totals["recall@1"] += int(hit1)
        totals["recall@3"] += int(hit3)
        totals["recall@10"] += int(hit10)

        p1 = precision_at_k(rels, 1)
        p3 = precision_at_k(rels, 3)
        p10 = precision_at_k(rels, 10)
        totals["p@1"] += p1
        totals["p@3"] += p3
        totals["p@10"] += p10

        mrr10 = mrr_at_k(rels, 10)
        totals["mrr@10"] += mrr10

        ndcg10 = ndcg_at_k(rels, 10, num_gold=len(gold_pairs))
        totals["ndcg@10"] += ndcg10

        rows.append(
            {
                "qid": qid,
                "question": question,
                "doc_id": doc_id,
                "gold_para_ids": gold_para_ids,
                "retrieved_top3": pred_pairs[:3],
                "hit@1": hit1,
                "hit@3": hit3,
                "hit@10": hit10,
                "p@1": p1,
                "p@3": p3,
                "p@10": p10,
                "mrr@10": mrr10,
                "ndcg@10": ndcg10,
                "latency_ms": lat_ms,
            }
        )

    n = totals["n"]
    if n == 0:
        print("No valid QA items with gold_para_ids.")
        return

    print("\n" + "=" * 44)
    print(f"RESULTS (N = {n})")
    print("=" * 44)
    print(f"Recall@1  (hit) : {totals['recall@1'] / n:.2%}")
    print(f"Recall@3  (hit) : {totals['recall@3'] / n:.2%}")
    print(f"Recall@10 (hit) : {totals['recall@10'] / n:.2%}")
    print("-" * 44)
    print(f"Precision@1     : {totals['p@1'] / n:.2%}")
    print(f"Precision@3     : {totals['p@3'] / n:.2%}")
    print(f"Precision@10    : {totals['p@10'] / n:.2%}")
    print(f"MRR@10          : {totals['mrr@10'] / n:.4f}")
    print(f"NDCG@10         : {totals['ndcg@10'] / n:.4f}")
    print(f"Avg latency (ms): {totals['lat_ms_sum'] / n:.2f}")
    print("=" * 44)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, out_csv_name)
    df.to_csv(csv_path, index=False)
    print(f"Detailed log saved to: {csv_path}")


def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="FAISS + SentenceTransformer dense retrieval baseline with extra metrics.")
    p.add_argument("--paragraphs", type=str, default=DEFAULT_PARA_FILE, help="Path to paragraphs.jsonl")
    p.add_argument("--qa", type=str, default=DEFAULT_QA_FILE, help="Path to qa_pairs.json")
    p.add_argument("--k", type=int, default=10, help="Top-k to retrieve")
    p.add_argument("--out", type=str, default="baseline_results.csv", help="Output CSV name (written under results/)")
    p.add_argument("--index-name", type=str, default="baseline.faiss", help="FAISS index filename (written under indices/)")
    
    # NEW ARGUMENT
    p.add_argument("--compress", action="store_true", help="Use Product Quantization (PQ) for compressed embeddings")
    p.add_argument("--rebuild", action="store_true", help="Force rebuild index even if it exists")
    
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    retriever = BaselineRetriever()
    
    # Logic to handle rebuild vs load
    if args.rebuild or not os.path.exists(os.path.join(INDEX_DIR, args.index_name)):
        retriever.load_and_index(args.paragraphs, save_name=args.index_name, compress=args.compress)
    else:
        retriever.load_index_from_disk(args.index_name, args.paragraphs)
        
    evaluate_system(retriever, args.qa, k_eval=args.k, out_csv_name=args.out)