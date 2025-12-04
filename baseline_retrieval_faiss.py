import json
import os
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch


# --- SETTINGS ---
BASE_DIR = "./long_context"
PARA_FILE = os.path.join(BASE_DIR, "paragraphs", "paragraphs.jsonl")
QA_FILE = os.path.join(BASE_DIR, "qa_pairs.json")
INDEX_DIR = os.path.join(BASE_DIR, "indices")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_NAME = "all-MiniLM-L6-v2"

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

class BaselineRetriever:
    def __init__(self, model_name=MODEL_NAME):
        print(f"Model loading: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        # FAISS internal id -> metadata
        self.metadata = []  # list of dict: {"doc_id", "para_id", "text"}

    def load_and_index(self, jsonl_path):
        """Read paragraphs, encode, build FAISS index."""
        if not os.path.exists(jsonl_path):
            print(f"ERROR: {jsonl_path} not found.")
            return

        print("Reading paragraphs...")
        texts = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                self.metadata.append(
                    {
                        "doc_id": data["doc_id"],
                        "para_id": data["para_id"],
                        "text": data["text"],
                    }
                )
                texts.append(data["text"])

        print(f"Total {len(texts)} paragraphs to embed.")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        # Optional: normalize for cosine-sim-as-L2 (yorum satırı, istersen aç)
        # faiss.normalize_L2(embeddings)

        print("Building FAISS index (IndexFlatL2)...")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)

        faiss.write_index(self.index, os.path.join(INDEX_DIR, "baseline.faiss"))
        print(f"Index saved with {len(embeddings)} vectors.")

    def search(self, query, k=10):
        """Return top-k nearest paragraphs for a query."""
        if self.index is None:
            raise ValueError("Index not built yet.")

        query_vec = self.model.encode([query], convert_to_numpy=True)
        # Optional: normalize for cosine metric
        # faiss.normalize_L2(query_vec)

        distances, indices = self.index.search(query_vec, k)

        results = []
        for rank_idx, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append(
                    {
                        "rank": rank_idx + 1,
                        "doc_id": meta["doc_id"],
                        "para_id": meta["para_id"],
                        "score": float(distances[0][rank_idx]),
                        "text": meta["text"],
                    }
                )
        return results


def evaluate_system(retriever, qa_file):
    """Evaluate retrieval with Recall@1/5/10."""
    print("\nStarting evaluation...")

    with open(qa_file, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    metrics = {"recall@1": 0, "recall@5": 0, "recall@10": 0, "total": 0}
    results_log = []

    for item in tqdm(qa_data, desc="Processing questions"):
        qid = item["qid"]
        question = item["question"]
        gold_para_ids = item["gold_para_ids"]  # list[int]
        doc_id = item["doc_id"]

        if not gold_para_ids:
            # no ground-truth paragraph ids, skip from metrics
            print(f"WARNING: {qid} has empty gold_para_ids, skipping.")
            continue

        metrics["total"] += 1

        # gold set: (doc_id, para_id) tuples
        gold_pairs = {(doc_id, pid) for pid in gold_para_ids}

        predictions = retriever.search(question, k=10)
        pred_pairs = [(p["doc_id"], p["para_id"]) for p in predictions]

        hit1 = any(pair in gold_pairs for pair in pred_pairs[:1])
        hit5 = any(pair in gold_pairs for pair in pred_pairs[:5])
        hit10 = any(pair in gold_pairs for pair in pred_pairs[:10])

        if hit1:
            metrics["recall@1"] += 1
        if hit5:
            metrics["recall@5"] += 1
        if hit10:
            metrics["recall@10"] += 1

        results_log.append(
            {
                "qid": qid,
                "question": question,
                "doc_id": doc_id,
                "gold_para_ids": gold_para_ids,
                "retrieved_top3": pred_pairs[:3],
                "hit@1": hit1,
                "hit@5": hit5,
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
    print(f"Recall@1  : {metrics['recall@1'] / total:.2%}")
    print(f"Recall@5  : {metrics['recall@5'] / total:.2%}")
    print(f"Recall@10 : {metrics['recall@10'] / total:.2%}")
    print("=" * 40)

    df = pd.DataFrame(results_log)
    csv_path = os.path.join(RESULTS_DIR, "baseline_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Detailed log saved to: {csv_path}")


if __name__ == "__main__":
    retriever = BaselineRetriever()
    retriever.load_and_index(PARA_FILE)
    evaluate_system(retriever, QA_FILE)
