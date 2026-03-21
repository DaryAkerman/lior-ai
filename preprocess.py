"""
Run this ONCE before starting app.py.
It extracts text from the PDF and creates semantic search embeddings.
Output: pages_data.json + embeddings.npy
"""
import json
import numpy as np
from pathlib import Path

PDF_PATH      = 'info.pdf'
PAGES_FILE    = 'pages_data.json'
EMBEDDINGS_FILE = 'embeddings.npy'
MODEL_NAME    = 'paraphrase-multilingual-MiniLM-L12-v2'


def extract_pages():
    import pymupdf
    print(f"📄 Extracting text from {PDF_PATH} ...")
    pages = []
    with pymupdf.open(PDF_PATH) as doc:
        total = len(doc)
        print(f"   Total pages: {total}")
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                pages.append({'page': i + 1, 'text': text.strip()})
            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"   {i+1}/{total} done...")
    print(f"✓ Extracted text from {len(pages)} pages\n")
    return pages


def create_embeddings(pages):
    from sentence_transformers import SentenceTransformer
    print(f"🔢 Loading embedding model: {MODEL_NAME}")
    print("   (first time downloads ~90 MB – wait a moment)")
    model = SentenceTransformer(MODEL_NAME)

    # Use first 512 characters per page (enough for semantic matching)
    texts = [p['text'][:512] for p in pages]

    print(f"   Embedding {len(texts)} pages ...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True   # pre-normalise for fast cosine similarity
    )
    print(f"✓ Embeddings ready\n")
    return embeddings


def main():
    if Path(PAGES_FILE).exists() and Path(EMBEDDINGS_FILE).exists():
        ans = input("⚠️  Processed data already exists. Re-process? (y/N): ")
        if ans.strip().lower() != 'y':
            print("Skipping – using existing data.")
            return

    pages = extract_pages()

    with open(PAGES_FILE, 'w', encoding='utf-8') as f:
        json.dump(pages, f, ensure_ascii=False)
    print(f"✓ Saved {PAGES_FILE}")

    embeddings = create_embeddings(pages)
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"✓ Saved {EMBEDDINGS_FILE}")

    print("\n🎉 Done! Now run:  python app.py")


if __name__ == '__main__':
    main()
