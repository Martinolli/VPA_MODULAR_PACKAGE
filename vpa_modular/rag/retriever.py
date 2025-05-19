import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import openai
import logging
from functools import lru_cache
from tenacity import retry, stop_after_attempt, wait_exponential

# Configurations
DEFAULT_EMBEDDING_DIR = Path("vpa_knowledge_base/embeddings")
DEFAULT_MODEL = "text-embedding-3-small"
openai.api_key = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)

@lru_cache(maxsize=100)
def cached_embed_query(text: str, model: str = DEFAULT_MODEL) -> Optional[np.ndarray]:
    return embed_query(text, model)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def embed_query(text: str, model: str = DEFAULT_MODEL) -> Optional[np.ndarray]:
    try:
        response = openai.embeddings.create(model=model, input=text)
        return np.array(response.data[0].embedding)
    except Exception as e:
        logger.error(f"Error embedding query: {e}")
        raise

def diversity_filter(chunks: List[Dict[str, Any]], similarity_threshold: float = 0.95) -> List[Dict[str, Any]]:
    filtered_chunks = []
    for chunk in chunks:
        if not any(cosine_similarity(chunk['embedding'], fc['embedding']) > similarity_threshold for fc in filtered_chunks):
            filtered_chunks.append(chunk)
    return filtered_chunks


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def load_all_embeddings(embedding_dir: Path = DEFAULT_EMBEDDING_DIR) -> List[Dict[str, Any]]:
    all_embeddings = []
    for embedding_file in embedding_dir.glob("*.json"):
        with open(embedding_file, "r", encoding="utf-8") as f:
            embeddings = json.load(f)
            all_embeddings.extend(embeddings)
    return all_embeddings

def retrieve_top_chunks(
    query: str,
    top_k: int = 5,
    embedding_dir: Path = DEFAULT_EMBEDDING_DIR,
    model: str = DEFAULT_MODEL,
    cache: Dict = {},
    diversity_threshold: float = 0.95
) -> List[Dict[str, Any]]:
    # Cache embeddings to avoid repeated disk reads
    if "embeddings" not in cache:
        cache["embeddings"] = load_all_embeddings(embedding_dir)
    chunks = cache["embeddings"]

    query_embedding = cached_embed_query(query, model)
    if query_embedding is None:
        logger.warning("Query embedding failed. Returning empty result.")
        return []

    # Vectorized similarity calculation
    chunk_embeddings = np.array([chunk["embedding"] for chunk in chunks])
    sims = np.dot(chunk_embeddings, query_embedding) / (
        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
    )
    scored = sorted(zip(sims, chunks), reverse=True, key=lambda x: x[0])
    top_chunks = scored[:top_k*2]  # Get more chunks than needed for diversity filtering

    # Apply diversity filter
    diverse_chunks = diversity_filter([{"score": score, **chunk} for score, chunk in top_chunks], diversity_threshold)
    top_diverse_chunks = diverse_chunks[:top_k]

    for i, chunk in enumerate(top_diverse_chunks, 1):
        meta = chunk.get("metadata", {})
        logger.info(f"--- Chunk {i} ---\nScore: {chunk['score']:.4f}\nSource: {chunk.get('source', '?')}\n"
                    f"Page: {meta.get('page', '?')}, Section: {meta.get('section', '?')}\n"
                    f"{chunk['text'][:200]}...\n")

    return top_diverse_chunks

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    query = "What is Wyckoff Method?"
    results = retrieve_top_chunks(query)
    for i, chunk in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {chunk.get('source', '?')}")
        print(chunk["text"][:500] + "...")
