"""
longformer_retriever.py

Fixes applied vs your original:
- `use_global_cls` flag now actually toggles global attention behavior.
- Proper metric naming + implementation:
  - Hit@k (a.k.a. Recall@k in your old code) kept as Hit@k.
  - True Recall@k added (fraction of gold retrieved within top-k).
  - Precision@k unchanged (but now uses relevant count / k).
  - MRR@k unchanged.
  - NDCG@k corrected for multi-relevant binary labels using IDCG = min(gt_count, k).
- Removed `torch.cuda.empty_cache()` inside the embedding loop (it hurts throughput).
- Added optional saving/loading of FAISS index + metadata + config in one directory.
- Added safety checks and clearer CLI.

Expected inputs:
- paragraphs.jsonl: each line is JSON with at least {"doc_id": ..., "para_id": ..., "text": "..."}
- qa_pairs.json: a JSON list; each entry has {"qid", "question", "doc_id", "gold_para_ids": [...]}

Run:
python longformer_retriever.py --paragraphs ./paragraphs.jsonl --qa ./qa_pairs.json --k 10
"""

import os
import json
import time
import math
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import faiss
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# ----------------------------
# Metrics
# ----------------------------
def precision_at_k(rels: List[int], k: int) -> float:
    """Precision@k = (# relevant in top-k) / k."""
    k = min(k, len(rels))
    if k <= 0:
        return 0.0
    return float(sum(rels[:k])) / float(k)


def recall_at_k(rels: List[int], k: int, gt_count: int) -> float:
    """Recall@k = (# relevant in top-k) / (# relevant total)."""
    k = min(k, len(rels))
    if gt_count <= 0 or k <= 0:
        return 0.0
    return float(sum(rels[:k])) / float(gt_count)


def hit_at_k(rels: List[int], k: int) -> float:
    """Hit@k = 1 if any relevant appears in top-k else 0."""
    k = min(k, len(rels))
    if k <= 0:
        return 0.0
    return 1.0 if any(rels[:k]) else 0.0


def mrr_at_k(rels: List[int], k: int) -> float:
    """MRR@k with binary relevance."""
    k = min(k, len(rels))
    for i in range(k):
        if rels[i] == 1:
            return 1.0 / float(i + 1)
    return 0.0


def ndcg_at_k(rels: List[int], k: int, gt_count: int) -> float:
    """
    NDCG@k with binary relevance.
    DCG = sum_{i=1..k} rel_i / log2(i+1)
    IDCG assumes all relevant are ranked first => min(gt_count, k) ones at top.
    """
    k = min(k, len(rels))
    if k <= 0 or gt_count <= 0:
        return 0.0

    dcg = 0.0
    for i in range(k):
        if rels[i] == 1:
            dcg += 1.0 / math.log2(i + 2)  # i=0 -> log2(2)=1

    ideal_ones = min(gt_count, k)
    idcg = 0.0
    for i in range(ideal_ones):
        idcg += 1.0 / math.log2(i + 2)

    return (dcg / idcg) if idcg > 0 else 0.0


# ----------------------------
# Persistence config
# ----------------------------
@dataclass
class IndexConfig:
    model_name: str
    max_length: int
    use_global_cls: bool
    dim: int


# ----------------------------
# Longformer Retriever
# ----------------------------
class LongformerRetriever:
    """
    Uses Longformer encoder to embed text.
    Embedding = mean pooling over last_hidden_state masked by attention_mask.
    FAISS = cosine similarity using normalized vectors + IndexFlatIP.
    """

    def __init__(
        self,
        model_name: str = "allenai/longformer-base-4096",
        device: Optional[str] = None,
        batch_size: int = 2,
        max_length: int = 4096,
        use_global_cls: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.use_global_cls = bool(use_global_cls)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[INFO] Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.dim = int(self.model.config.hidden_size)
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []  # [{"doc_id","para_id","text"}]

    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts to np.ndarray [N, dim] float32."""
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        all_vecs = []
        for i in tqdm(
            range(0, len(texts), self.batch_size),
            desc="Embedding (Longformer)",
            leave=False,
        ):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            if self.use_global_cls:
                global_attention_mask = torch.zeros_like(attention_mask)
                global_attention_mask[:, 0] = 1  # CLS as global token
            else:
                global_attention_mask = None

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
            )

            last_hidden = outputs.last_hidden_state  # [B, T, H]

            # Mean pooling with attention mask
            mask = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
            summed = (last_hidden * mask).sum(dim=1)     # [B, H]
            counts = mask.sum(dim=1).clamp(min=1e-9)     # [B, 1]
            pooled = summed / counts                     # [B, H]

            vecs = pooled.detach().cpu().numpy().astype(np.float32)
            all_vecs.append(vecs)

        return np.vstack(all_vecs)

    def load_paragraphs(self, jsonl_path: str) -> List[str]:
        """Loads paragraphs.jsonl and populates self.metadata; returns list of texts."""
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Paragraphs file not found: {jsonl_path}")

        self.metadata.clear()
        texts: List[str] = []

        bad = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue

                doc_id = rec.get("doc_id")
                para_id = rec.get("para_id")
                text = rec.get("text", "")

                if doc_id is None or para_id is None or not isinstance(text, str) or not text.strip():
                    bad += 1
                    continue

                self.metadata.append(
                    {"doc_id": doc_id, "para_id": int(para_id), "text": text}
                )
                texts.append(text)

        print(f"[INFO] Loaded {len(texts)} passages from {jsonl_path} (skipped={bad})")
        return texts

    def build_faiss(self, embeddings: np.ndarray):
        """Builds cosine-sim FAISS index from embeddings."""
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Embeddings must be [N,{self.dim}] got {embeddings.shape}")

        emb = embeddings.astype(np.float32, copy=False)
        faiss.normalize_L2(emb)
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        self.index = index

    def index_corpus(self, paragraphs_jsonl: str):
        texts = self.load_paragraphs(paragraphs_jsonl)
        emb = self._encode_texts(texts)
        self.build_faiss(emb)
        print(f"[INFO] Index ready. dim={self.dim}, size={len(texts)}")

    def save_index_dir(self, out_dir: str):
        """Saves index + metadata + config to a directory."""
        if self.index is None:
            raise ValueError("Index not built; nothing to save.")
        os.makedirs(out_dir, exist_ok=True)

        faiss_path = os.path.join(out_dir, "index.faiss")
        meta_path = os.path.join(out_dir, "metadata.jsonl")
        cfg_path = os.path.join(out_dir, "config.json")

        faiss.write_index(self.index, faiss_path)

        with open(meta_path, "w", encoding="utf-8") as f:
            for m in self.metadata:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        cfg = IndexConfig(
            model_name=self.model_name,
            max_length=self.max_length,
            use_global_cls=self.use_global_cls,
            dim=self.dim,
        )
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

        print(f"[INFO] Saved index bundle to: {out_dir}")

    def load_index_dir(self, in_dir: str):
        """Loads index + metadata + config from a directory."""
        faiss_path = os.path.join(in_dir, "index.faiss")
        meta_path = os.path.join(in_dir, "metadata.jsonl")
        cfg_path = os.path.join(in_dir, "config.json")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path) and os.path.exists(cfg_path)):
            raise FileNotFoundError(f"Index bundle incomplete in: {in_dir}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # sanity check: ensure current model matches saved config
        if cfg.get("dim") != self.dim:
            raise ValueError(f"Dim mismatch: saved={cfg.get('dim')} current={self.dim}")
        if cfg.get("model_name") != self.model_name:
            print(f"[WARN] Model name differs: saved={cfg.get('model_name')} current={self.model_name}")
        if cfg.get("max_length") != self.max_length:
            print(f"[WARN] max_length differs: saved={cfg.get('max_length')} current={self.max_length}")
        if bool(cfg.get("use_global_cls")) != self.use_global_cls:
            print(f"[WARN] use_global_cls differs: saved={cfg.get('use_global_cls')} current={self.use_global_cls}")

        self.index = faiss.read_index(faiss_path)

        self.metadata.clear()
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.metadata.append(json.loads(line))

        print(f"[INFO] Loaded index bundle from: {in_dir} (size={len(self.metadata)})")

    @torch.no_grad()
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        if self.index is None:
            raise ValueError("Index not built. Call index_corpus() or load_index_dir().")

        k = int(k)
        if k <= 0:
            return []

        q_emb = self._encode_texts([query])  # [1, H]
        faiss.normalize_L2(q_emb)

        scores, idxs = self.index.search(q_emb, k)
        scores = scores[0]
        idxs = idxs[0]

        results = []
        for rank, (score, idx) in enumerate(zip(scores, idxs), start=1):
            if idx < 0 or idx >= len(self.metadata):
                continue
            m = self.metadata[idx]
            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "doc_id": m["doc_id"],
                    "para_id": int(m["para_id"]),
                    "text": m["text"],
                }
            )
        return results


# ----------------------------
# Evaluation
# ----------------------------
def evaluate(
    retriever: LongformerRetriever,
    qa_path: str,
    out_csv: str,
    k_list: List[int] = [1, 3, 10],
):
    if not os.path.exists(qa_path):
        raise FileNotFoundError(f"QA file not found: {qa_path}")

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    totals = 0
    hit_sum = {k: 0.0 for k in k_list}
    recall_sum = {k: 0.0 for k in k_list}
    precision_sum = {k: 0.0 for k in k_list}
    mrr_sum = 0.0
    ndcg_sum = 0.0
    latency_ms_sum = 0.0

    rows = []

    for item in tqdm(qa_data, desc="Evaluating"):
        qid = item.get("qid")
        question = item.get("question", "")
        doc_id = item.get("doc_id")
        gold_para_ids = item.get("gold_para_ids", [])

        if not question or doc_id is None or not gold_para_ids:
            continue

        gold_set = {(doc_id, int(pid)) for pid in gold_para_ids}
        gt_count = len(gold_para_ids)

        t0 = time.perf_counter()
        preds = retriever.search(question, k=max(k_list))
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        pred_pairs = [(p["doc_id"], p["para_id"]) for p in preds]
        rels = [1 if pair in gold_set else 0 for pair in pred_pairs]

        totals += 1
        latency_ms_sum += latency_ms

        for k in k_list:
            hit_sum[k] += hit_at_k(rels, k)
            recall_sum[k] += recall_at_k(rels, k, gt_count=gt_count)
            precision_sum[k] += precision_at_k(rels, k)

        mrr_sum += mrr_at_k(rels, k=max(k_list))
        ndcg_sum += ndcg_at_k(rels, k=max(k_list), gt_count=gt_count)

        rows.append(
            {
                "qid": qid,
                "question": question,
                "doc_id": doc_id,
                "gold_para_ids": gold_para_ids,
                "retrieved_top3": pred_pairs[:3],
                "hit@1": bool(hit_at_k(rels, 1)),
                "hit@3": bool(hit_at_k(rels, 3)),
                "hit@10": bool(hit_at_k(rels, 10)),
                "recall@10": float(recall_at_k(rels, 10, gt_count=gt_count)),
                "latency_ms": round(latency_ms, 3),
            }
        )

    if totals == 0:
        print("[WARN] No valid QA entries found (missing gold_para_ids etc.).")
        return

    print("\n" + "=" * 52)
    print(f"RESULTS (N = {totals})")
    print("=" * 52)
    for k in k_list:
        print(f"Hit@{k:<2}               : {hit_sum[k] / totals:.2%}")
    print("-" * 52)
    for k in k_list:
        print(f"Recall@{k:<2}             : {recall_sum[k] / totals:.2%}")
    print("-" * 52)
    for k in k_list:
        print(f"Precision@{k:<2}          : {precision_sum[k] / totals:.2%}")
    print("-" * 52)
    print(f"MRR@{max(k_list)}              : {mrr_sum / totals:.4f}")
    print(f"NDCG@{max(k_list)}             : {ndcg_sum / totals:.4f}")
    print(f"Avg latency (ms)         : {latency_ms_sum / totals:.2f}")
    print("=" * 52)

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Per-question log saved: {out_csv}")


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Longformer retriever (FAISS cosine) over paragraphs.jsonl + evaluation on qa_pairs.json"
    )
    ap.add_argument("--paragraphs", type=str, required=True, help="Path to paragraphs.jsonl")
    ap.add_argument("--qa", type=str, required=True, help="Path to qa_pairs.json")
    ap.add_argument("--out_csv", type=str, default="./long_context/results/longformer_results.csv", help="CSV output path")

    ap.add_argument("--model", type=str, default="allenai/longformer-base-4096", help="Longformer model name")
    ap.add_argument("--batch_size", type=int, default=4, help="Embedding batch size")
    ap.add_argument("--max_length", type=int, default=4096, help="Max tokens for Longformer")

    ap.add_argument("--no_global_cls", action="store_true", help="Disable global attention on CLS token")

    ap.add_argument("--k", type=int, default=10, help="Top-k retrieval for evaluation")
    ap.add_argument("--save_index_dir", type=str, default="", help="If set, save index bundle to this dir after building")
    ap.add_argument("--load_index_dir", type=str, default="", help="If set, load index bundle from this dir instead of building")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    retriever = LongformerRetriever(
        model_name=args.model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_global_cls=(not args.no_global_cls),
    )

    if args.load_index_dir:
        retriever.load_index_dir(args.load_index_dir)
    else:
        retriever.index_corpus(args.paragraphs)
        if args.save_index_dir:
            retriever.save_index_dir(args.save_index_dir)

    k = int(args.k)
    if k < 1:
        raise SystemExit("--k must be >= 1")

    k_list = [1, 3, 10]
    if k < 10:
        k_list = sorted(set([1, min(3, k), k]))
    else:
        k_list = [1, 3, 10]

    evaluate(
        retriever=retriever,
        qa_path=args.qa,
        out_csv=args.out_csv,
        k_list=k_list,
    )


if __name__ == "__main__":
    main()
