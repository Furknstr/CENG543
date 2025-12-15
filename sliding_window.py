"""sliding_window.py

Creates *sentence-based sliding window* chunks from the cleaned texts produced by
`download_datasets.py` and writes them to:

    <BASE_DIR>/paragraphs/paragraphs.jsonl

It can optionally auto-realign QA gold labels (gold_para_ids) by matching each
QA item's `gold_chunk_snippet` against the newly created sliding chunks.

Key behaviors (as you requested):
- OVERWRITES paragraphs.jsonl (replace if exists)
- MODIFIES qa_pairs.json IN PLACE (no new QA file)
- Writes a retrieval evaluation CSV (Recall@1/@3/@10 + top3 predictions)

Run examples:
  python sliding_window.py --base_dir ./long_context --align_qa --evaluate

  # custom window sizes
  python sliding_window.py --arxiv_window 8 --arxiv_stride 4 --gut_window 6 --gut_stride 3 \
      --align_qa --evaluate

Dependencies:
  pip install sentence-transformers faiss-cpu pandas tqdm torch
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Tuple

from auto_align_gold_ids import align_qa_inplace


# --------------------------
# Sentence splitting
# --------------------------

# A small, practical list for sentence-splitting safety.
_ABBREV = (
    "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.",
    "St.", "No.", "Fig.", "Eq.", "Sec.", "Ch.",
)

_SENT_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“‘(])")


def _normalize_space(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter.

    We first protect a few abbreviations so that "Mr." doesn't split.
    This is not perfect (sentence segmentation is hard), but it's deterministic
    and works well enough for sliding window chunking.
    """
    text = _normalize_space(text)
    if not text:
        return []

    placeholder = "<DOT>"
    for abbr in _ABBREV:
        text = text.replace(abbr, abbr.replace(".", placeholder))

    parts = _SENT_SPLIT_REGEX.split(text)
    sents = [p.strip() for p in parts if p.strip()]

    # restore dots
    sents = [s.replace(placeholder, ".") for s in sents]
    return sents


# --------------------------
# Sliding window chunking
# --------------------------

def text_to_sliding_windows(
    text: str,
    window_size: int,
    stride: int,
    min_chars: int,
) -> List[Dict]:
    """Turn text into overlapping sentence windows.

    Returns list of dicts:
      {para_id, start_sent_id, end_sent_id, text}

    para_id is the CHUNK ID (0..N-1) per document.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    n = len(sentences)

    # If shorter than one window, return a single chunk.
    if n <= window_size:
        chunk_text = " ".join(sentences).strip()
        if len(chunk_text) < min_chars:
            return []
        return [
            {
                "para_id": 0,
                "start_sent_id": 0,
                "end_sent_id": n - 1,
                "text": chunk_text,
            }
        ]

    chunks: List[Dict] = []
    chunk_id = 0
    start = 0
    while start < n:
        end = min(start + window_size, n)
        chunk_text = " ".join(sentences[start:end]).strip()

        if len(chunk_text) >= min_chars:
            chunks.append(
                {
                    "para_id": chunk_id,
                    "start_sent_id": start,
                    "end_sent_id": end - 1,
                    "text": chunk_text,
                }
            )
            chunk_id += 1

        if end == n:
            break
        start += stride

    return chunks


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_paragraphs_jsonl(
    base_dir: str,
    arxiv_ids: List[str],
    gutenberg_id: str,
    arxiv_window: int,
    arxiv_stride: int,
    arxiv_min_chars: int,
    gut_window: int,
    gut_stride: int,
    gut_min_chars: int,
) -> str:
    """Create/replace paragraphs.jsonl using sliding-window chunking."""

    raw_dir = os.path.join(base_dir, "raw_txt")
    para_dir = os.path.join(base_dir, "paragraphs")
    os.makedirs(para_dir, exist_ok=True)

    out_jsonl = os.path.join(para_dir, "paragraphs.jsonl")
    if os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    def append_doc(doc_id: str, text: str, window: int, stride: int, min_chars: int):
        chunks = text_to_sliding_windows(text, window, stride, min_chars)
        if not chunks:
            print(f"[WARN] No chunks for {doc_id} (maybe too short or min_chars too high).")
            return 0

        with open(out_jsonl, "a", encoding="utf-8") as out_f:
            for ch in chunks:
                rec = {
                    "doc_id": doc_id,
                    "para_id": ch["para_id"],
                    "text": ch["text"],
                    "metadata": {
                        "start_sent_id": ch["start_sent_id"],
                        "end_sent_id": ch["end_sent_id"],
                    },
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return len(chunks)

    print("[INFO] Building sliding-window paragraphs.jsonl ...")

    total = 0
    for aid in arxiv_ids:
        p = os.path.join(raw_dir, f"{aid}_clean.txt")
        if not os.path.exists(p):
            print(f"[WARN] Missing cleaned text: {p} (run download_datasets.py first)")
            continue
        total += append_doc(aid, _read_text(p), arxiv_window, arxiv_stride, arxiv_min_chars)

    gut_doc_id = f"gutenberg_{gutenberg_id}"
    gut_path = os.path.join(raw_dir, f"{gut_doc_id}_clean.txt")
    if os.path.exists(gut_path):
        total += append_doc(gut_doc_id, _read_text(gut_path), gut_window, gut_stride, gut_min_chars)
    else:
        print(f"[WARN] Missing cleaned text: {gut_path} (run download_datasets.py first)")

    print(f"[DONE] paragraphs.jsonl written: {out_jsonl} (total chunks: {total})")
    return out_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="./long_context")

    # IDs are kept in sliding_window.py so you can quickly swap doc sets.
    ap.add_argument(
        "--arxiv_ids",
        nargs="+",
        default=["1706.03762", "1810.04805", "1910.01108"],
        help="List of arXiv IDs to include (cleaned txt must exist).",
    )
    ap.add_argument("--gutenberg_id", default="1342")

    ap.add_argument("--arxiv_window", type=int, default=8)
    ap.add_argument("--arxiv_stride", type=int, default=4)
    ap.add_argument("--arxiv_min_chars", type=int, default=150)

    ap.add_argument("--gut_window", type=int, default=6)
    ap.add_argument("--gut_stride", type=int, default=3)
    ap.add_argument("--gut_min_chars", type=int, default=150)

    ap.add_argument("--align_qa", action="store_true", help="Update qa_pairs.json gold_para_ids in place")
    ap.add_argument("--evaluate", action="store_true", help="Run FAISS baseline evaluation after rebuild")

    args = ap.parse_args()

    para_jsonl = build_paragraphs_jsonl(
        base_dir=args.base_dir,
        arxiv_ids=args.arxiv_ids,
        gutenberg_id=args.gutenberg_id,
        arxiv_window=args.arxiv_window,
        arxiv_stride=args.arxiv_stride,
        arxiv_min_chars=args.arxiv_min_chars,
        gut_window=args.gut_window,
        gut_stride=args.gut_stride,
        gut_min_chars=args.gut_min_chars,
    )

    qa_path = os.path.join(args.base_dir, "qa_pairs.json")

    if args.align_qa:
        if not os.path.exists(qa_path):
            raise SystemExit(f"qa_pairs.json not found: {qa_path}")
        align_report = align_qa_inplace(qa_path=qa_path, paragraphs_jsonl=para_jsonl)
        print(f"[DONE] QA aligned in place: {qa_path}")
        print(f"[INFO] Align report: {align_report}")

    if args.evaluate:
        from baseline_retrieval_faiss import run_eval

        run_eval(
            base_dir=args.base_dir,
            paragraphs_jsonl=para_jsonl,
            qa_path=qa_path,
            out_csv=os.path.join(args.base_dir, "results", "sliding_eval.csv"),
            recall_ks=(1, 3, 10),
        )


if __name__ == "__main__":
    main()
