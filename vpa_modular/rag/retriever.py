import os
import json
import numpy as np
from pathlib import Path
from openai import OpenAI
from typing import List

# Configurations
EMBEDDING_FILE = Path("vpa_knowledge_base/embeddings/anna_coulling_embeddings.json")
MODEL = "text-embedding-3-small"

# Load API key
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def embed_query(text: str) -> List[float]:
    try:
        response = openai.embeddings.create(
            model=MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Error embedding query: {e}")
        return []


def load_embeddings():
    with open(EMBEDDING_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_top_chunks(query: str, top_k: int = 5):
    print(f"🔍 Retrieving top {top_k} chunks for query: '{query}'")
    query_emb = embed_query(query)
    if not query_emb:
        return []

    chunks = load_embeddings()
    scored = []

    for chunk in chunks:
        score = cosine_similarity(query_emb, chunk["embedding"])
        scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])
    top_chunks = [chunk for _, chunk in scored[:top_k]]

    return top_chunks


# Example test
if __name__ == "__main__":
    query = "What is the importance of volume in price analysis?"
    results = retrieve_top_chunks(query)
    for i, chunk in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk["text"][:500] + "...")
