import json
import re
from pathlib import Path

from pypdf import PdfReader

from services.embeddings import create_embedding


KNOWLEDGE_DIR = Path("data/Knowledge")
PERSONALITY_DIR = Path("data/Personality")
KQM_ROOT = Path("data/kqm_tcl")

KNOWLEDGE_INDEX_PATH = Path(
    "data/knowledge_index.json"
)

PERSONALITY_INDEX_PATH = Path(
    "data/personality_index.json"
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

KQM_SOURCE_DIRS = {
    "character": KQM_ROOT / "docs" / "characters",
    "weapon": KQM_ROOT / "docs" / "equipment" / "weapons",
    "enemy": KQM_ROOT / "docs" / "enemy-data",
}


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
        ".md",
        ".mdx",
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


def clean_kqm_markdown(
    text: str
) -> str:
    # Remove YAML frontmatter
    text = re.sub(
        r"\A---\s*\n.*?\n---\s*\n",
        "",
        text,
        flags=re.DOTALL,
    )

    # Remove Docusaurus / JavaScript import lines
    text = re.sub(
        r"^\s*import\s+.*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Remove simple standalone JSX component tags
    text = re.sub(
        r"^\s*</?[A-Z][^>]*>\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Collapse excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


def split_markdown_sections(
    text: str
) -> list[dict]:
    sections = []

    current_heading = "Overview"
    current_lines = []

    for line in text.splitlines():
        match = re.match(
            r"^(#{1,6})\s+(.+)$",
            line
        )

        if match:
            content = "\n".join(
                current_lines
            ).strip()

            if content:
                sections.append(
                    {
                        "heading": current_heading,
                        "content": content,
                    }
                )

            current_heading = match.group(2).strip()
            current_lines = []

        else:
            current_lines.append(line)

    content = "\n".join(
        current_lines
    ).strip()

    if content:
        sections.append(
            {
                "heading": current_heading,
                "content": content,
            }
        )

    return sections


def iter_kqm_files():
    for category, source_dir in (
        KQM_SOURCE_DIRS.items()
    ):
        if not source_dir.exists():
            print(
                f"Skipping missing KQM directory: "
                f"{source_dir}"
            )
            continue

        for file_path in source_dir.rglob("*"):
            if file_path.suffix.lower() not in {
                ".md",
                ".mdx",
            }:
                continue

            if "_common" in file_path.parts:
                continue

            yield category, file_path


def load_kqm_documents() -> list[dict]:
    documents = []

    for category, file_path in iter_kqm_files():
        raw_text = extract_text_file(
            file_path
        )

        cleaned_text = clean_kqm_markdown(
            raw_text
        )

        sections = split_markdown_sections(
            cleaned_text
        )

        relative_path = file_path.relative_to(
            KQM_ROOT
        )

        for section in sections:
            if not section["content"].strip():
                continue

            documents.append(
                {
                    "category": category,
                    "source": "KQM TCL",
                    "file": str(relative_path),
                    "section": section["heading"],
                    "content": section["content"],
                }
            )

    return documents


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
            ".md",
            ".mdx",
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


def ingest_knowledge():
    records = []

    KNOWLEDGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Existing manually curated knowledge files
    for file_path in KNOWLEDGE_DIR.iterdir():
        if file_path.suffix.lower() not in {
            ".pdf",
            ".txt",
            ".md",
            ".mdx",
        }:
            continue

        print(
            f"Ingesting knowledge: "
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
                    "category": "knowledge",
                    "source": file_path.name,
                    "chunk_number": chunk_number,
                    "text": chunk,
                    "embedding": embedding,
                }
            )

    # KQM TCL character, weapon, and enemy docs
    kqm_documents = load_kqm_documents()

    print(
        f"Loaded {len(kqm_documents)} "
        f"KQM sections"
    )

    for document in kqm_documents:
        chunks = chunk_text(
            document["content"]
        )

        for chunk_number, chunk in enumerate(
            chunks
        ):
            embedding_text = (
                f"Source: {document['source']}\n"
                f"Category: {document['category']}\n"
                f"File: {document['file']}\n"
                f"Section: {document['section']}\n\n"
                f"{chunk}"
            )

            embedding = create_embedding(
                embedding_text
            )

            records.append(
                {
                    "category": document["category"],
                    "source": document["source"],
                    "file": document["file"],
                    "section": document["section"],
                    "chunk_number": chunk_number,
                    "text": chunk,
                    "embedding": embedding,
                }
            )

    KNOWLEDGE_INDEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        KNOWLEDGE_INDEX_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            records,
            f
        )

    print(
        f"Saved {len(records)} "
        f"knowledge chunks "
        f"to {KNOWLEDGE_INDEX_PATH}"
    )


def ingest():
    ingest_knowledge()

    ingest_directory(
        PERSONALITY_DIR,
        PERSONALITY_INDEX_PATH,
        "personality"
    )


if __name__ == "__main__":
    ingest()
