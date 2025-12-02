    # dataset_faz1.py
# Faz 1: Veri Hazırlığı ve Ground-Truth için altyapı
# Gerekli paketler:
#   pip install arxiv pymupdf requests

import os
import re
import json
import requests
import fitz  # PyMuPDF
import arxiv


# ==========================
# KONFİG
# ==========================

BASE_DIR = "./long_context"  # İstersen değiştir
ARXIV_IDS = [
    "1706.03762",  # Attention Is All You Need
    "1810.04805",  # BERT
    "1910.01108",  # DistilBERT
    "1909.11942",  # ALBERT
    "1907.11692",  # RoBERTa
    "2005.14165",  # GPT-3
    "2010.11929",  # ViT
    "2103.00020",  # CLIP
    "1910.10683",  # T5
    "2106.09685",  # LoRA
]

GUTENBERG_BOOK = {
    "id": "1342",
    "title": "Pride and Prejudice"
}


# ==========================
# DİZİN OLUŞTURMA
# ==========================

def setup_dirs(base_dir: str):
    docs = os.path.join(base_dir, "docs")          # PDF + ham txt
    raw_txt = os.path.join(base_dir, "raw_txt")    # temizlenmiş txt
    paragraphs = os.path.join(base_dir, "paragraphs")  # paragraphs.jsonl
    os.makedirs(docs, exist_ok=True)
    os.makedirs(raw_txt, exist_ok=True)
    os.makedirs(paragraphs, exist_ok=True)
    return {
        "BASE": base_dir,
        "DOCS": docs,
        "RAW": raw_txt,
        "PARA": paragraphs,
    }


# ==========================
# ARXIV İNDİRME
# ==========================

def download_arxiv_pdf(arxiv_id: str, dest_dir: str) -> str | None:
    search = arxiv.Search(id_list=[arxiv_id])
    for result in search.results():
        pdf_path = os.path.join(dest_dir, f"{arxiv_id}.pdf")
        result.download_pdf(filename=pdf_path)
        return pdf_path
    return None


# ==========================
# GUTENBERG İNDİRME
# ==========================

def download_gutenberg_txt(book_id: str, title: str, dest_dir: str) -> str:
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    resp = requests.get(url)
    resp.raise_for_status()
    fname = f"gutenberg_{book_id}_{title.replace(' ', '_')}.txt"
    out_path = os.path.join(dest_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    return out_path


# ==========================
# PDF → METİN
# ==========================

def pdf_to_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text_out = []
    for page in doc:
        blocks = page.get_text("blocks") 
        blocks.sort(key=lambda b: (b[1], b[0])) 
        for b in blocks:
            text_out.append(b[4])
    doc.close()
    return "\n\n".join(text_out)


# ==========================
# TEMİZLEME
# ==========================

def clean_text(text: str) -> str:
    # satır sonlarını normalize et
    text = text.replace("\r", "\n")
    # 3+ boş satırı 2 satıra indir
    text = re.sub(r"\n{3,}", "\n\n", text)
    # satır sonu boşluklarını sil
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    return text


def save_clean_text(raw_text: str, out_path: str):
    cleaned = clean_text(raw_text)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(cleaned)


# ==========================
# PARAGRAF ÇIKARMA
# ==========================

def text_to_paragraphs(text: str, min_len: int = 50) -> list[str]:
    blocks = text.split("\n\n")
    paras = []
    for b in blocks:
        t = b.strip()
        if len(t) >= min_len:
            paras.append(t)
    return paras


def write_paragraphs_jsonl(
    cleaned_txt_path: str,
    doc_id: str,
    out_jsonl_path: str,
    min_len: int = 50
):
    with open(cleaned_txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    paras = text_to_paragraphs(text, min_len=min_len)

    # append modunda yaz
    with open(out_jsonl_path, "a", encoding="utf-8") as out_f:
        for i, p in enumerate(paras):
            rec = {
                "doc_id": doc_id,
                "para_id": i,
                "text": p
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ==========================
# QA FILE
# ==========================

def init_empty_qa(base_dir: str, doc_ids: list) -> str:
    qa_path = os.path.join(base_dir, "qa_pairs.json")
    
    if os.path.exists(qa_path) and os.path.getsize(qa_path) > 5:
        print(f"[INFO] QA file already exists: {qa_path}")
        return qa_path

    qa_skeleton = []
    for doc_id in doc_ids:
        for i in range(1, 6):
            qa_skeleton.append({
                "qid": f"{doc_id}_q{i}",
                "doc_id": doc_id,
                "question": "SORU_BURAYA",
                "answer": "CEVAP_BURAYA",
                "gold_chunk_snippet": "REFERANS_METIN_BURAYA" # Cevabın geçtiği cümleyi buraya yapıştıracaksın
            })
            
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_skeleton, f, ensure_ascii=False, indent=2)
    
    return qa_path


# ==========================
# MAIN PIPELINE (FAZ 1)
# ==========================

def run_faz1():
    dirs = setup_dirs(BASE_DIR)
    docs_dir = dirs["DOCS"]
    raw_dir = dirs["RAW"]
    para_dir = dirs["PARA"]

    print(f"[INFO] Base dir: {dirs['BASE']}")

    # 1) arXiv PDF'leri indir
    pdf_paths: dict[str, str | None] = {}
    for aid in ARXIV_IDS:
        print(f"[INFO] Downloading arXiv {aid} ...")
        path = download_arxiv_pdf(aid, docs_dir)
        pdf_paths[aid] = path
        print(f"      → {path}")

    # 2) Gutenberg kitabını indir
    print(f"[INFO] Downloading Gutenberg book {GUTENBERG_BOOK['id']} ...")
    book_txt = download_gutenberg_txt(
        GUTENBERG_BOOK["id"],
        GUTENBERG_BOOK["title"],
        docs_dir
    )
    print(f"      → {book_txt}")

    # 3) PDF → text ve temizlenmiş txt kaydet
    for aid, pdf_path in pdf_paths.items():
        if pdf_path is None:
            print(f"[WARN] No PDF for {aid}")
            continue
        print(f"[INFO] PDF to text for {aid} ...")
        raw_txt = pdf_to_text(pdf_path)
        out_clean_path = os.path.join(raw_dir, f"{aid}_clean.txt")
        save_clean_text(raw_txt, out_clean_path)
        print(f"      → {out_clean_path}")

    # 4) Kitap txt’sini temizle
    print("[INFO] Cleaning Gutenberg book text ...")
    with open(book_txt, "r", encoding="utf-8") as f:
        book_raw = f.read()
    book_clean_path = os.path.join(
        raw_dir,
        f"gutenberg_{GUTENBERG_BOOK['id']}_clean.txt"
    )
    save_clean_text(book_raw, book_clean_path)
    print(f"      → {book_clean_path}")

    # 5) paragraphs.jsonl üret
    paragraphs_jsonl = os.path.join(para_dir, "paragraphs.jsonl")
    # varsa eskiyi sil
    if os.path.exists(paragraphs_jsonl):
        os.remove(paragraphs_jsonl)

    print("[INFO] Writing paragraphs for arXiv docs ...")
    for aid in ARXIV_IDS:
        clean_path = os.path.join(raw_dir, f"{aid}_clean.txt")
        if not os.path.exists(clean_path):
            print(f"[WARN] Clean text not found for {aid}, skipping.")
            continue
        write_paragraphs_jsonl(
            cleaned_txt_path=clean_path,
            doc_id=aid,
            out_jsonl_path=paragraphs_jsonl,
            min_len=50
        )

    print("[INFO] Writing paragraphs for Gutenberg book ...")
    write_paragraphs_jsonl(
        cleaned_txt_path=book_clean_path,
        doc_id=f"gutenberg_{GUTENBERG_BOOK['id']}",
        out_jsonl_path=paragraphs_jsonl,
        min_len=80  
    )

    all_doc_ids = ARXIV_IDS + [f"gutenberg_{GUTENBERG_BOOK['id']}"]
    
    qa_path = init_empty_qa(dirs["BASE"], all_doc_ids)

    print("[DONE] Faz 1 tamamlandı.")
    print(f"       Docs dir:        {docs_dir}")
    print(f"       Raw txt dir:     {raw_dir}")
    print(f"       Paragraphs json: {paragraphs_jsonl}")
    print(f"       QA file:         {qa_path}")


if __name__ == "__main__":
    run_faz1()
