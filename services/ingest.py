import json
from pathlib import Path
from pypdf import PdfReader

from services.embeddings import create_embedding


KNOWLEDGE_DIR = Path("data/Knowledge")
PERSONALITY_DIR = Path("data/Personality")

KNOWLEDGE_INDEX_PATH = Path(
    "data/knowledge_index.json"
)

PERSONALITY_INDEX_PATH = Path(
    "data/personality_index.json"
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def extract_pdf_text(
    file_path: Path
) -> str:

    reader = PdfReader(file_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def extract_text_file(
    file_path: Path
) -> str:

    return file_path.read_text(
        encoding="utf-8"
    )


def extract_text(
    file_path: Path
) -> str:

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf_text(file_path)

    if suffix in {
        ".txt",
        ".md"
    }:
        return extract_text_file(file_path)

    raise ValueError(
        f"Unsupported file type: {suffix}"
    )


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[str]:

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        start += (
            chunk_size - overlap
        )

    return chunks


def ingest_directory(
    source_dir: Path,
    index_path: Path,
    category: str
):

    records = []

    source_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for file_path in source_dir.iterdir():

        if file_path.suffix.lower() not in {
            ".pdf",
            ".txt",
            ".md"
        }:
            continue

        print(
            f"Ingesting {category}: "
            f"{file_path.name}"
        )

        text = extract_text(
            file_path
        )

        chunks = chunk_text(
            text
        )

        for chunk_number, chunk in enumerate(
            chunks
        ):

            embedding = create_embedding(
                chunk
            )

            records.append(
                {
                    "category": category,
                    "source": file_path.name,
                    "chunk_number": chunk_number,
                    "text": chunk,
                    "embedding": embedding,
                }
            )

    index_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        index_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            records,
            f
        )

    print(
        f"Saved {len(records)} "
        f"{category} chunks "
        f"to {index_path}"
    )


def ingest():

    ingest_directory(
        KNOWLEDGE_DIR,
        KNOWLEDGE_INDEX_PATH,
        "knowledge"
    )

    ingest_directory(
        PERSONALITY_DIR,
        PERSONALITY_INDEX_PATH,
        "personality"
    )


if __name__ == "__main__":
    ingest()