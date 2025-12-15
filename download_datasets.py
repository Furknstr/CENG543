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

# -----------------------------------------------------------------------------
# Notes / Fixes
# -----------------------------------------------------------------------------
# 1) Some arXiv identifiers can contain slashes (e.g., "math/0211159"). If you
#    use such IDs, you MUST sanitize filenames, otherwise Python will treat the
#    slash as a directory separator and you'll get:
#      FileNotFoundError: ... long_context\\docs\\math/0211159.pdf
#    This script therefore saves PDFs using a safe filename ("/" -> "_") while
#    keeping the original arXiv id as doc_id in paragraphs.jsonl.
# 2) The arxiv.Search.results() iterator is deprecated; we use arxiv.Client.


# ==========================
# KONFİG
# ==========================

BASE_DIR = "./long_context"  # İstersen değiştir
ARXIV_IDS = [
    # NLP / ML 
    "1706.03762",  # Attention Is All You Need
    "1810.04805",  # BERT
    "1910.01108",  # DistilBERT
    # Mathematics (3)
    "1805.08392",  # Optimal Transport for Applied Mathematicians
    "2006.16928",  # Deep learning and differential equations
    "2102.09554",  # Neural operators for PDEs
    # Physics (3)
    "1503.07589",  # Gravitational waves from binary black holes
    "1905.03777",  # Quantum supremacy using a programmable superconducting processor
    "2103.01955",  # Black hole information paradox (modern review)
    # Medical / Bio (3)
    "2003.10555",  # Deep learning for medical image analysis
    "2104.07302",  # AI in radiology: challenges and opportunities
    "2201.12345",  # Machine learning in precision medicine
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
    """Download arXiv PDF by id.

    FIX: Old-style arXiv ids can contain '/' (e.g., 'math/0211159'). If we use
    that string as a filename, it becomes a subdirectory and crashes on Windows
    (FileNotFoundError). We therefore sanitize filenames but keep the original
    arXiv id for doc_id elsewhere.

    Also switches to arxiv.Client() to avoid Search.results() deprecation.
    """
    os.makedirs(dest_dir, exist_ok=True)

    safe_id = arxiv_id.replace("/", "_")
    pdf_path = os.path.join(dest_dir, f"{safe_id}.pdf")
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
        return pdf_path

    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    for result in client.results(search):
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
    resp.encoding = 'utf-8'  # Encoding'i garantiye al

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    return out_path


# ==========================
# PDF → METİN
# ==========================

def pdf_to_text(pdf_path: str) -> str:
    """Extract text from a PDF.

    arXiv PDFs often use 2-column layouts. A naive (y,x) block sort can interleave
    columns, harming downstream paragraphing and retrieval. This extractor:
      - filters simple headers/footers
      - splits blocks into left/right columns using the page midpoint
      - outputs left column top-to-bottom, then right column top-to-bottom
    """

    doc = fitz.open(pdf_path)
    text_out: list[str] = []

    for page in doc:
        blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)
        if not blocks:
            continue

        h = page.rect.height
        mid_x = page.rect.width / 2.0

        left: list[tuple] = []
        right: list[tuple] = []

        for b in blocks:
            x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
            if not txt or not txt.strip():
                continue
            # very simple header/footer removal
            if y0 < 50 or y1 > h - 50:
                continue
            # classify by column
            (left if x0 < mid_x else right).append((x0, y0, x1, y1, txt))

        left.sort(key=lambda t: (t[1], t[0]))
        right.sort(key=lambda t: (t[1], t[0]))

        for _, _, _, _, txt in left:
            text_out.append(txt)
        for _, _, _, _, txt in right:
            text_out.append(txt)

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

    # 2) Gutenberg header/footer boundaries vary across books.
    #    Common patterns:
    #      "*** START OF THIS PROJECT GUTENBERG EBOOK ... ***"
    #      "*** START OF THE PROJECT GUTENBERG EBOOK ... ***"
    #    So we match more flexibly.
    start_match = re.search(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, flags=re.IGNORECASE | re.DOTALL)
    if start_match:
        text = text[start_match.end():]

    end_match = re.search(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", text, flags=re.IGNORECASE | re.DOTALL)
    if end_match:
        text = text[:end_match.start()]

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
# PARAGRAF ÇIKARMA: ARXIV
# ==========================

def text_to_paragraphs_arxiv(text: str, min_len: int = 50) -> list[str]:
    """
    ArXiv makaleleri için eski, basit ayırma mantığı.
    """
    # Çift enter'a göre böl
    blocks = text.split("\n\n")
    paras = []
    for b in blocks:
        # Satır içi enterları boşluğa çevir (tek paragraf tek satır olsun)
        t = b.replace('\n', ' ').strip()
        t = re.sub(r'\s+', ' ', t)  # çift boşlukları sil

        if len(t) >= min_len:
            paras.append(t)
    return paras


# ==========================
# PARAGRAF ÇIKARMA: GUTENBERG
# ==========================

ABBREVS = ("Mr.", "Mrs.", "Dr.", "Ms.", "St.", "Col.", "Gen.", "Sr.", "Jr.")


def _normalize_block(b: str) -> str:
    t = b.replace("\n", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _should_merge(prev: str, nxt: str) -> bool:
    """
    Gutenberg wrap çözmek için heuristik:
      - prev sonu . ? ! ile bitmiyorsa → muhtemelen satır sarma → merge
      - prev sonu bilinen kısaltma (Mr., Mrs., ...) ise → merge
      - cümle sonu olsa bile nxt küçük harfle ya da noktalama ile başlıyorsa → merge
      - aksi halde yeni paragraf
    """
    if not prev or not nxt:
        return False

    prev_core = prev.rstrip()
    # Sondaki parantez, tırnak vs. temizle
    prev_core = re.sub(r'[\)\]\}"»’]+$', "", prev_core)
    if not prev_core:
        return True

    # Kısaltma kontrolü (Mr., Mrs. vs.)
    for abbr in ABBREVS:
        if prev_core.endswith(abbr):
            return True

    last = prev_core[-1]

    # Cümle sonu değilse: büyük ihtimalle satır sarma → birleştir
    if last not in ".?!":
        return True

    # Sonraki blok
    nxt_core = nxt.lstrip()
    # Baştaki tırnak/parantezleri at
    nxt_core = re.sub(r'^[\("“\'«]+', "", nxt_core)
    if not nxt_core:
        return True

    first = nxt_core[0]

    # Küçük harfle başlıyorsa → cümlenin devamı
    if first.islower():
        return True

    # Noktalama ile başlıyorsa → devam
    if first in ",;:)]}'»”":
        return True

    # Geri kalan durumlarda yeni paragraf
    return False


def text_to_paragraphs_gutenberg(text: str, min_len: int = 80) -> list[str]:
    """
    Gutenberg romanı için daha akıllı paragraf/chunk çıkarma.
    - \n\n ile bloklara ayır.
    - Satır sarma kaynaklı yanlış bölünmeleri _should_merge ile birleştir.
    - [Illustration ...] bloklarını at.
    """
    raw_blocks = text.split("\n\n")

    merged_blocks: list[str] = []
    current: str | None = None

    for b in raw_blocks:
        t = _normalize_block(b)
        if not t:
            continue

        if current is None:
            current = t
            continue

        if _should_merge(current, t):
            current = current + " " + t
        else:
            if len(current) >= min_len:
                merged_blocks.append(current)
            current = t

    # Son buffer
    if current is not None and len(current) >= min_len:
        merged_blocks.append(current)

    # Illustration vb. çöp paragrafları filtrele
    filtered: list[str] = []
    for p in merged_blocks:
        pt = p.strip()
        if pt.startswith("[Illustration"):
            continue
        filtered.append(p)

    return filtered


# ==========================
# JSONL YAZMA (AYRI FONKSİYONLAR)
# ==========================

def write_paragraphs_jsonl_arxiv(
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

    paras = text_to_paragraphs_arxiv(text, min_len=min_len)

    if not paras:
        print(f"⚠️ UYARI: {doc_id} (arxiv) dosyasından hiç paragraf çıkmadı!")

    # append modunda yaz
    with open(out_jsonl_path, "a", encoding="utf-8") as out_f:
        for i, p in enumerate(paras):
            rec = {
                "doc_id": doc_id,
                "para_id": i,
                "text": p
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ (arxiv) {doc_id}: {len(paras)} paragraf eklendi.")


def write_paragraphs_jsonl_gutenberg(
    cleaned_txt_path: str,
    doc_id: str,
    out_jsonl_path: str,
    min_len: int = 80
):
    if not os.path.exists(cleaned_txt_path):
        print(f"❌ HATA: Dosya bulunamadı -> {cleaned_txt_path}")
        return

    with open(cleaned_txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    paras = text_to_paragraphs_gutenberg(text, min_len=min_len)

    if not paras:
        print(f"⚠️ UYARI: {doc_id} (gutenberg) dosyasından hiç paragraf çıkmadı!")

    # append modunda yaz
    with open(out_jsonl_path, "a", encoding="utf-8") as out_f:
        for i, p in enumerate(paras):
            rec = {
                "doc_id": doc_id,
                "para_id": i,
                "text": p
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"✅ (gutenberg) {doc_id}: {len(paras)} paragraf eklendi.")


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

    # 3) PDF → text ve temizlenmiş txt kaydet (arxiv)
    for aid, pdf_path in pdf_paths.items():
        if pdf_path is None:
            continue
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
        write_paragraphs_jsonl_arxiv(
            cleaned_txt_path=clean_path,
            doc_id=aid,
            out_jsonl_path=paragraphs_jsonl,
            min_len=50
        )

    print("[INFO] Writing paragraphs for Gutenberg book ...")
    # Kitabın temizlenmiş yolunu kullanarak jsonl'e ekle
    write_paragraphs_jsonl_gutenberg(
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
