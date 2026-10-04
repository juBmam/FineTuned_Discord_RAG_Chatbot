import os

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(
    MODEL_NAME,
    token=os.getenv("HF_TOKEN")
)



def create_embedding(text: str) -> list[float]:
    embedding = model.encode(
        text,
        normalize_embeddings=True
    )

    return embedding.tolist()