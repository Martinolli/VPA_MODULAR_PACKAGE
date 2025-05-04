import os
import fitz  # PyMuPDF
import json
from typing import List
from pathlib import Path

# Configuration
SOURCE_PATH = Path("vpa_knowledge_base/sources/anna_coulling_vpa.pdf")
OUTPUT_PATH = Path("vpa_knowledge_base/chunked/anna_coulling_chunks.json")
CHUNK_SIZE = 500  # words
OVERLAP = 100  # words


def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[dict]:
    words = text.split()
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "chunk_id": f"coulling_{chunk_id}",
            "source": "anna_coulling_vpa",
            "text": chunk_text
        })
        start += chunk_size - overlap
        chunk_id += 1
    return chunks


def save_chunks_to_json(chunks: List[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def main():
    print("🔍 Extracting text from PDF...")
    raw_text = extract_text_from_pdf(SOURCE_PATH)
    print(f"✅ Extracted {len(raw_text.split())} words")

    print("✂️ Chunking text with overlap...")
    chunks = chunk_text(raw_text, CHUNK_SIZE, OVERLAP)
    print(f"✅ Created {len(chunks)} chunks")

    print(f"💾 Saving to {OUTPUT_PATH}")
    save_chunks_to_json(chunks, OUTPUT_PATH)
    print("✅ Done.")


if __name__ == "__main__":
    main()
