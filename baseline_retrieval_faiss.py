"""baseline_retrieval_faiss.py

Dense retrieval baseline using SentenceTransformers + FAISS.

What it does:
1) Loads a chunk collection from a JSONL file (each line: {doc_id, para_id, text, ...}).
2) Embeds every chunk with a SentenceTransformer model.
3) Builds a FAISS index for fast nearest-neighbor search.
4) Evaluates retrieval on QA pairs (qa_pairs.json) using gold_para_ids.
5) Writes a CSV with per-question details (gold ids, top3 predictions, hit flags).

This file is intentionally CLI-friendly so you can point it at different
paragraph/chunk JSONLs and QA files.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


@dataclass
class SearchResult:
    rank: int
    doc_id: str
    para_id: int
    score: float
    text: str


class BaselineRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", metric: str = "cosine"):
        """Create retriever.

        metric:
          - "cosine"  : normalize embeddings, use IndexFlatIP (inner product)
          - "l2"      : no normalization, use IndexFlatL2
        """
        self.model_name = model_name
        self.metric = metric.lower().strip()

        print(f"Model loading: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

        self.index: faiss.Index | None = None
        self.metadata: List[Dict[str, Any]] = []

    def _maybe_normalize(self, x: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            faiss.normalize_L2(x)
        return x

    def load_and_index(self, jsonl_path: str, index_out: str | None = None, batch_size: int = 32):
        """Read chunks, encode, build FAISS index."""
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

        print(f"Reading chunks: {jsonl_path}")
        texts: List[str] = []
        self.metadata.clear()

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Required
                doc_id = str(data.get("doc_id"))
                para_id = int(data.get("para_id"))
                text = str(data.get("text", ""))

                self.metadata.append({"doc_id": doc_id, "para_id": para_id, "text": text})
                texts.append(text)

        print(f"Total {len(texts)} chunks to embed.")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

        self._maybe_normalize(embeddings)

        if self.metric == "cosine":
            print("Building FAISS index (IndexFlatIP; cosine via L2-normalization)...")
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.metric == "l2":
            print("Building FAISS index (IndexFlatL2)...")
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            raise ValueError("metric must be one of: cosine, l2")

        self.index.add(embeddings)

        if index_out:
            os.makedirs(os.path.dirname(index_out), exist_ok=True)
            faiss.write_index(self.index, index_out)
            print(f"Index saved: {index_out} ({len(embeddings)} vectors)")

    def search(self, query: str, k: int = 10) -> List[SearchResult]:
        if self.index is None:
            raise ValueError("Index not built yet. Call load_and_index().")

        query_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        self._maybe_normalize(query_vec)

        scores, indices = self.index.search(query_vec, k)

        results: List[SearchResult] = []
        for rank_idx, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append(
                    SearchResult(
                        rank=rank_idx + 1,
                        doc_id=meta["doc_id"],
                        para_id=int(meta["para_id"]),
                        score=float(scores[0][rank_idx]),
                        text=meta["text"],
                    )
                )
        return results


def evaluate_system(
    retriever: BaselineRetriever,
    qa_path: str,
    out_csv: str,
    k_eval: Tuple[int, int, int] = (1, 3, 10),
):
    """Evaluate retrieval with Recall@K on QA pairs."""

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    k1, k3, k10 = k_eval
    metrics = {f"recall@{k1}": 0, f"recall@{k3}": 0, f"recall@{k10}": 0, "total": 0}

    rows: List[Dict[str, Any]] = []

    for item in tqdm(qa_data, desc="Evaluating"):
        qid = item.get("qid")
        question = item.get("question")
        doc_id = item.get("doc_id")
        gold_para_ids = item.get("gold_para_ids") or []

        if not qid or not question or not doc_id or not gold_para_ids:
            continue

        metrics["total"] += 1
        gold_pairs = {(str(doc_id), int(pid)) for pid in gold_para_ids}

        preds = retriever.search(str(question), k=k10)
        pred_pairs = [(p.doc_id, p.para_id) for p in preds]

        hit1 = any(pair in gold_pairs for pair in pred_pairs[:k1])
        hit3 = any(pair in gold_pairs for pair in pred_pairs[:k3])
        hit10 = any(pair in gold_pairs for pair in pred_pairs[:k10])

        if hit1:
            metrics[f"recall@{k1}"] += 1
        if hit3:
            metrics[f"recall@{k3}"] += 1
        if hit10:
            metrics[f"recall@{k10}"] += 1

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
            }
        )

    total = metrics["total"]
    if total == 0:
        print("No valid QA items with gold_para_ids.")
        return

    print("\n" + "=" * 40)
    print(f"RESULTS (N = {total})")
    print("=" * 40)
    print(f"Recall@{k1: <2}: {metrics[f'recall@{k1}'] / total:.2%}")
    print(f"Recall@{k3: <2}: {metrics[f'recall@{k3}'] / total:.2%}")
    print(f"Recall@{k10}: {metrics[f'recall@{k10}'] / total:.2%}")
    print("=" * 40)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Per-question log saved: {out_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", default="./long_context")
    p.add_argument("--paragraphs", default=None, help="Path to paragraphs.jsonl")
    p.add_argument("--qa", default=None, help="Path to qa_pairs.json")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out_csv", default=None)
    p.add_argument("--index_out", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    base_dir = args.base_dir
    paragraphs = args.paragraphs or os.path.join(base_dir, "paragraphs", "paragraphs.jsonl")
    qa_path = args.qa or os.path.join(base_dir, "qa_pairs.json")

    results_dir = os.path.join(base_dir, "results")
    index_dir = os.path.join(base_dir, "indices")

    out_csv = args.out_csv or os.path.join(results_dir, "baseline_results.csv")
    index_out = args.index_out or os.path.join(index_dir, f"baseline_{args.metric}.faiss")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    retriever = BaselineRetriever(model_name=args.model, metric=args.metric)
    retriever.load_and_index(paragraphs, index_out=index_out, batch_size=args.batch_size)
    evaluate_system(retriever, qa_path=qa_path, out_csv=out_csv, k_eval=(1, 3, 10))


if __name__ == "__main__":
    main()
