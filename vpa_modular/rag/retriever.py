import os
import json
import numpy as np
from pathlib import Path
from openai import OpenAI
from typing import List
# Removed invalid import for openai.embeddings_utils

# Configurations
EMBEDDING_DIR = Path("vpa_knowledge_base/embeddings")
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


def load_all_embeddings():
    all_embeddings = []
    for embedding_file in EMBEDDING_DIR.glob("*.json"):
        with open(embedding_file, "r", encoding="utf-8") as f:
            embeddings = json.load(f)
            all_embeddings.extend(embeddings)
    return all_embeddings


def retrieve_top_chunks(query, top_k=5):
    # Embed the query
    query_embedding = embed_query(query)

    # Load all chunks and calculate similarities
    chunks = load_all_embeddings()
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
        print(f"Source: {chunk.get('source', '?')}")
        print(f"Page: {meta.get('page', '?')}, Section: {meta.get('section', '?')}")
        print(chunk['text'][:500] + "...\n")

    # Return only the chunk content and metadata
    return [chunk for _, chunk in top_chunks]


# Example test
if __name__ == "__main__":
    query = "What is Wickoff Method?"
    results = retrieve_top_chunks(query)
    for i, chunk in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {chunk.get('source', '?')}")
        print(chunk["text"][:500] + "...")