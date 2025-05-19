import os
import json
from pathlib import Path
from openai import OpenAI

# Load your OpenAI API key securely
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

INPUT_PATH = Path("vpa_knowledge_base/chunked/the_wyckoff_methodology_in_depth.json")
OUTPUT_PATH = Path("vpa_knowledge_base/embeddings/the_wyckoff_methodology_in_depth.json")
MODEL = "text-embedding-3-small"


def load_chunks(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_chunk(text):
    try:
        response = openai.embeddings.create(
            model=MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️ Error embedding text: {e}")
        return None


def main():
    print("📦 Loading chunks...")
    chunks = load_chunks(INPUT_PATH)

    embedded = []
    for i, chunk in enumerate(chunks):
        print(f"🔍 Embedding chunk {i+1}/{len(chunks)}")
        emb = embed_chunk(chunk["text"])
        if emb:
            chunk_entry = {
                "chunk_id": chunk["chunk_id"],
                "source": chunk["source"],
                "text": chunk["text"],
                "embedding": emb
            }
            embedded.append(chunk_entry)

    print(f"💾 Saving {len(embedded)} embedded chunks to {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(embedded, f, ensure_ascii=False, indent=2)

    print("✅ Embedding complete.")


if __name__ == "__main__":
    main()
