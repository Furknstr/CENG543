"""auto_align_gold_ids.py

Automatically updates `qa_pairs.json` in-place by recomputing each item's
`gold_para_ids` *against the current* paragraphs JSONL file.

Why this is needed:
- When you change chunking strategy (paragraph split -> sliding windows), the
  (doc_id, para_id) mapping changes.
- Your QA file still references old `gold_para_ids`, which makes evaluation look
  artificially bad.

What changes:
- `gold_para_ids` will be replaced with the para_id(s) of the best matching
  sliding-window chunks.

What does NOT change:
- `gold_chunk_snippet` is treated as the anchor/ground-truth text. We DO NOT
  rewrite it automatically.

Should `gold_chunk_snippet` change?
- Usually: NO. Keep it as the quote/snippet you originally annotated from the
  source document.
- Exception: If your snippet came from a *previously bad extraction* (e.g., the
  PDF extractor merged columns incorrectly) or you later edited the texts, then
  the snippet might no longer exist verbatim. In that case, either:
    (a) update the snippet manually, or
    (b) accept a fuzzy match (this script supports fuzzy matching).

Matching strategy used here (robust):
1) Normalize snippet & chunk text (unicode quotes, whitespace, case-fold).
2) Try exact substring match.
3) If not found, use token-overlap (Jaccard) + SequenceMatcher ratio to score
   candidates within the same doc.
4) Return 1..N best para_ids if their score passes a threshold.

Usage:
  python auto_align_gold_ids.py --base_dir ./long_context

This rewrites: <base_dir>/qa_pairs.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, List, Tuple


# ----------------------------
# Normalization helpers
# ----------------------------

_QUOTE_MAP = {
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "’": "'",
    "‘": "'",
    "‚": "'",
    "‛": "'",
    "—": "-",
    "–": "-",
    "…": "...",
}


def _normalize(s: str) -> str:
    if s is None:
        return ""
    # Unicode NFKC, unify quotes/dashes
    s = unicodedata.normalize("NFKC", s)
    for k, v in _QUOTE_MAP.items():
        s = s.replace(k, v)
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.casefold()


def _tokenize(s: str) -> List[str]:
    s = _normalize(s)
    # keep simple word tokens
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", s)


def _jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    A, B = set(a), set(b)
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0


# ----------------------------
# Loading paragraphs
# ----------------------------


def load_paragraphs_by_doc(paragraphs_jsonl: str) -> Dict[str, List[Tuple[int, str]]]:
    """Return: doc_id -> list[(para_id, text)]"""
    if not os.path.exists(paragraphs_jsonl):
        raise FileNotFoundError(f"paragraphs.jsonl not found: {paragraphs_jsonl}")

    by_doc: Dict[str, List[Tuple[int, str]]] = {}
    with open(paragraphs_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj["doc_id"]
            para_id = int(obj["para_id"])
            text = obj["text"]
            by_doc.setdefault(doc_id, []).append((para_id, text))

    # ensure stable ordering
    for d in by_doc:
        by_doc[d].sort(key=lambda x: x[0])

    return by_doc


# ----------------------------
# Core alignment
# ----------------------------


def find_best_para_ids(
    doc_paras: List[Tuple[int, str]],
    snippet: str,
    max_hits: int = 3,
    min_score: float = 0.55,
) -> List[int]:
    """Find para_ids in doc_paras that best match snippet.

    Returns up to `max_hits` para_ids.

    Scoring:
    - If exact substring match found: score=1.0
    - Else: combined score = 0.65*seq_ratio + 0.35*jaccard_token
    """
    sn = _normalize(snippet)
    if not sn:
        return []

    # 1) exact substring match
    exact_hits: List[int] = []
    for pid, txt in doc_paras:
        tn = _normalize(txt)
        if sn in tn:
            exact_hits.append(pid)

    if exact_hits:
        return exact_hits[:max_hits]

    # 2) fuzzy: jaccard + sequence ratio
    s_tokens = _tokenize(snippet)
    scored: List[Tuple[float, int]] = []
    for pid, txt in doc_paras:
        t_tokens = _tokenize(txt)
        jac = _jaccard(s_tokens, t_tokens)
        # SequenceMatcher is expensive; run on normalized strings but keep it bounded
        tn = _normalize(txt)
        # quick prune: if token overlap is extremely low, skip ratio
        if jac < 0.05:
            ratio = 0.0
        else:
            ratio = SequenceMatcher(None, sn, tn).ratio()
        score = 0.65 * ratio + 0.35 * jac
        scored.append((score, pid))

    scored.sort(reverse=True)
    best = [pid for score, pid in scored if score >= min_score][:max_hits]
    return best


def align_qa_inplace(
    base_dir: str,
    paragraphs_jsonl: str | None = None,
    qa_json: str | None = None,
    max_hits: int = 3,
    min_score: float = 0.55,
) -> None:
    """Update qa_pairs.json in place."""
    if paragraphs_jsonl is None:
        paragraphs_jsonl = os.path.join(base_dir, "paragraphs", "paragraphs.jsonl")
    if qa_json is None:
        qa_json = os.path.join(base_dir, "qa_pairs.json")

    by_doc = load_paragraphs_by_doc(paragraphs_jsonl)

    if not os.path.exists(qa_json):
        raise FileNotFoundError(f"qa_pairs.json not found: {qa_json}")

    with open(qa_json, "r", encoding="utf-8") as f:
        qa = json.load(f)

    updated = 0
    missing_docs = 0
    missing_snippets = 0

    for item in qa:
        doc_id = item.get("doc_id")
        snippet = item.get("gold_chunk_snippet", "")
        if not snippet:
            item["gold_para_ids"] = []
            missing_snippets += 1
            continue

        doc_paras = by_doc.get(doc_id)
        if not doc_paras:
            item["gold_para_ids"] = []
            missing_docs += 1
            continue

        new_ids = find_best_para_ids(doc_paras, snippet, max_hits=max_hits, min_score=min_score)
        old_ids = item.get("gold_para_ids")

        # Always write, even if same
        item["gold_para_ids"] = new_ids
        if new_ids != old_ids:
            updated += 1

    with open(qa_json, "w", encoding="utf-8") as f:
        json.dump(qa, f, ensure_ascii=False, indent=2)

    print("[auto_align] Done.")
    print(f"[auto_align] QA items: {len(qa)}")
    print(f"[auto_align] Updated gold_para_ids: {updated}")
    if missing_docs:
        print(f"[auto_align] Missing doc_id in paragraphs: {missing_docs}")
    if missing_snippets:
        print(f"[auto_align] Missing gold_chunk_snippet: {missing_snippets}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="./long_context")
    ap.add_argument("--paragraphs", default=None, help="Optional path to paragraphs.jsonl")
    ap.add_argument("--qa", default=None, help="Optional path to qa_pairs.json")
    ap.add_argument("--max_hits", type=int, default=3)
    ap.add_argument("--min_score", type=float, default=0.55)
    args = ap.parse_args()

    align_qa_inplace(
        base_dir=args.base_dir,
        paragraphs_jsonl=args.paragraphs,
        qa_json=args.qa,
        max_hits=args.max_hits,
        min_score=args.min_score,
    )


if __name__ == "__main__":
    main()
