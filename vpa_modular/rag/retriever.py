import os
import json
import numpy as np
from pathlib import Path
from openai import OpenAI
from typing import List
# Removed invalid import for openai.embeddings_utils

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


def retrieve_top_chunks(query, top_k=5):
    # Embed the query
    client = OpenAI()
    query_embedding = embed_query(query)

    # Load chunks and calculate similarities
    chunks = load_embeddings()
    scored = []
    for chunk in chunks:
        sim = cosine_similarity(query_embedding, chunk["embedding"])
        scored.append((sim, chunk))

    # Sort and return top K
    scored.sort(reverse=True, key=lambda x: x[0])
    top_chunks = scored[:top_k]

    # Print for visual verification (can be removed later)
    print(f"\n🔍 Retrieving top {top_k} chunks for query: '{query}'\n")
    for i, (score, chunk) in enumerate(top_chunks, 1):
        meta = chunk.get("metadata", {})
        print(f"--- Chunk {i} ---")
        print(f"Score: {score:.4f}")
        print(f"Page: {meta.get('page', '?')}, Section: {meta.get('section', '?')}")
        print(chunk['text'][:500] + "...\n")

    # Return only the chunk content and metadata
    return [chunk for _, chunk in top_chunks]


# Example test
if __name__ == "__main__":
    query = "What is the importance of volume in price analysis?"
    results = retrieve_top_chunks(query)
    for i, chunk in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk["text"][:500] + "...")
