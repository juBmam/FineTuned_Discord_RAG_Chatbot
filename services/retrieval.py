import json
from pathlib import Path

import numpy as np

from services.embeddings import create_embedding


KNOWLEDGE_INDEX_PATH = Path(
    "data/knowledge_index.json"
)

PERSONALITY_INDEX_PATH = Path(
    "data/personality_index.json"
)


def load_index(index_path: Path):
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found: {index_path}. "
            "Run `python -m services.ingest` first."
        )

    with open(
        index_path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def cosine_similarity(
    vector_a,
    vector_b
) -> float:
    a = np.array(vector_a)
    b = np.array(vector_b)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b)
        / denominator
    )


def retrieve_from_index(
    query: str,
    index_path: Path,
    top_k: int
):
    index = load_index(
        index_path
    )

    query_embedding = create_embedding(
        query
    )

    results = []

    for item in index:
        score = cosine_similarity(
            query_embedding,
            item["embedding"]
        )

        results.append(
            {
                "text": item["text"],
                "source": item["source"],
                "category": item.get("category"),
                "file": item.get("file"),
                "section": item.get("section"),
                "chunk_number": item.get(
                    "chunk_number"
                ),
                "score": score,
            }
        )

    results.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return results[:top_k]


def retrieve_knowledge(
    question: str,
    top_k: int = 5
):
    return retrieve_from_index(
        question,
        KNOWLEDGE_INDEX_PATH,
        top_k
    )


def retrieve_personality(
    question: str,
    top_k: int = 3
):
    return retrieve_from_index(
        question,
        PERSONALITY_INDEX_PATH,
        top_k
    )