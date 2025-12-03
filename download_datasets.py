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
    # Eğer dosya zaten varsa indirme
    pdf_path = os.path.join(dest_dir, f"{arxiv_id}.pdf")
    if os.path.exists(pdf_path):
        return pdf_path

    search = arxiv.Search(id_list=[arxiv_id])
    for result in search.results():
        result.download_pdf(filename=pdf_path)
        return pdf_path
    return None


# ==========================
# GUTENBERG İNDİRME
# ==========================

def download_gutenberg_txt(book_id: str, title: str, dest_dir: str) -> str:
    fname = f"gutenberg_{book_id}_{title.replace(' ', '_')}.txt"
    out_path = os.path.join(dest_dir, fname)
    
    # Zaten varsa indirme
    if os.path.exists(out_path):
        return out_path

    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    resp = requests.get(url)
    resp.raise_for_status()
    resp.encoding = 'utf-8' # Encoding'i garantiye al
    
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
        # Blokları yukarıdan aşağıya, soldan sağa sırala (çift sütun desteği)
        blocks.sort(key=lambda b: (b[1], b[0])) 
        for b in blocks:
            # Header/Footer temizliği için basit kontrol (sayfanın en tepesi ve en altı)
            if b[1] < 50 or b[3] > page.rect.height - 50:
                continue
            text_out.append(b[4])
    doc.close()
    return "\n\n".join(text_out)


# ==========================
# TEMİZLEME (GENEL & GUTENBERG)
# ==========================

def clean_text(text: str) -> str:
    """Arxiv makaleleri için genel temizlik."""
    text = text.replace("\r", "\n")
    # Çoklu boş satırları koru ama abartma (paragraf ayrımı için \n\n önemli)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Satır başı/sonu boşlukları temizle
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    return text

def clean_gutenberg_text(text: str) -> str:
    """Kitaplar için özel temizlik: Header/Footer atar."""
    # 1. Satır sonlarını düzelt
    text = text.replace('\r', '\n')
    
    # 2. Gutenberg Header Temizliği
    # Genelde "*** START OF THE PROJECT..." ile başlar
    start_marker = "*** START OF THE PROJECT"
    idx = text.find(start_marker)
    if idx != -1:
        # Marker'dan sonraki satıra geç
        text = text[idx:]
        # Marker satırının bitimini bul
        newline_idx = text.find('\n')
        if newline_idx != -1:
            text = text[newline_idx+1:]
            
    # 3. Gutenberg Footer Temizliği
    end_marker = "*** END OF THE PROJECT"
    idx_end = text.find(end_marker)
    if idx_end != -1:
        text = text[:idx_end]

    # 4. Fazla boşlukları indirgeme
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def save_clean_text(raw_text: str, out_path: str):
    # Eğer dosya zaten varsa tekrar yazma (performans için)
    if os.path.exists(out_path):
        return
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(raw_text)


# ==========================
# PARAGRAF ÇIKARMA
# ==========================

def text_to_paragraphs(text: str, min_len: int = 50) -> list[str]:
    # Çift enter'a göre böl
    blocks = text.split("\n\n")
    paras = []
    for b in blocks:
        # Satır içi enterları boşluğa çevir (tek paragraf tek satır olsun)
        t = b.replace('\n', ' ').strip()
        t = re.sub(r'\s+', ' ', t) # çift boşlukları sil
        
        if len(t) >= min_len:
            paras.append(t)
    return paras


def write_paragraphs_jsonl(
    cleaned_txt_path: str,
    doc_id: str,
    out_jsonl_path: str,
    min_len: int = 50
):
    if not os.path.exists(cleaned_txt_path):
        print(f"❌ HATA: Dosya bulunamadı -> {cleaned_txt_path}")
        return

    with open(cleaned_txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    paras = text_to_paragraphs(text, min_len=min_len)

    if not paras:
        print(f"⚠️ UYARI: {doc_id} dosyasından hiç paragraf çıkmadı! Temizlik fonksiyonunu kontrol et.")

    # append modunda yaz
    with open(out_jsonl_path, "a", encoding="utf-8") as out_f:
        for i, p in enumerate(paras):
            rec = {
                "doc_id": doc_id,
                "para_id": i,
                "text": p
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"✅ {doc_id}: {len(paras)} paragraf eklendi.")


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
        # Her doküman için 5 soru şablonu
        for i in range(1, 6):
            qa_skeleton.append({
                "qid": f"{doc_id}_q{i}",
                "doc_id": doc_id,
                "question": "SORU_BURAYA",
                "answer": "CEVAP_BURAYA",
                "gold_chunk_snippet": "REFERANS_METIN_BURAYA"
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
        print(f"[INFO] Downloading/Checking arXiv {aid} ...")
        path = download_arxiv_pdf(aid, docs_dir)
        pdf_paths[aid] = path

    # 2) Gutenberg kitabını indir
    print(f"[INFO] Downloading/Checking Gutenberg book {GUTENBERG_BOOK['id']} ...")
    book_txt = download_gutenberg_txt(
        GUTENBERG_BOOK["id"],
        GUTENBERG_BOOK["title"],
        docs_dir
    )

    # 3) PDF → text ve temizlenmiş txt kaydet
    for aid, pdf_path in pdf_paths.items():
        if pdf_path is None:
            continue
        # print(f"[INFO] Processing {aid} ...")
        raw_txt = pdf_to_text(pdf_path)
        out_clean_path = os.path.join(raw_dir, f"{aid}_clean.txt")
        save_clean_text(clean_text(raw_txt), out_clean_path)

    # 4) Kitap txt’sini temizle (Gutenberg Temizleyici ile!)
    print("[INFO] Cleaning Gutenberg book text ...")
    with open(book_txt, "r", encoding="utf-8") as f:
        book_raw = f.read()
    
    book_clean_path = os.path.join(
        raw_dir,
        f"gutenberg_{GUTENBERG_BOOK['id']}_clean.txt"
    )
    # Burada özel Gutenberg temizleyicisini kullanıyoruz
    save_clean_text(clean_gutenberg_text(book_raw), book_clean_path)
    print(f"      → {book_clean_path}")

    # 5) paragraphs.jsonl üret
    paragraphs_jsonl = os.path.join(para_dir, "paragraphs.jsonl")
    # Dosyayı sıfırla (overwrite)
    if os.path.exists(paragraphs_jsonl):
        os.remove(paragraphs_jsonl)

    print("[INFO] Writing paragraphs for arXiv docs ...")
    for aid in ARXIV_IDS:
        clean_path = os.path.join(raw_dir, f"{aid}_clean.txt")
        write_paragraphs_jsonl(
            cleaned_txt_path=clean_path,
            doc_id=aid,
            out_jsonl_path=paragraphs_jsonl,
            min_len=50
        )

    print("[INFO] Writing paragraphs for Gutenberg book ...")
    # Kitabın temizlenmiş yolunu kullanarak jsonl'e ekle
    write_paragraphs_jsonl(
        cleaned_txt_path=book_clean_path,
        doc_id=f"gutenberg_{GUTENBERG_BOOK['id']}",
        out_jsonl_path=paragraphs_jsonl,
        min_len=80  # Roman paragrafları genelde biraz daha uzundur
    )

    all_doc_ids = ARXIV_IDS + [f"gutenberg_{GUTENBERG_BOOK['id']}"]
    qa_path = init_empty_qa(dirs["BASE"], all_doc_ids)

    print("\n[DONE] Faz 1 tamamlandı.")
    print(f"       Paragraphs: {paragraphs_jsonl}")
    print(f"       QA Skeleton: {qa_path}")


if __name__ == "__main__":
    run_faz1()