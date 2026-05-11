"""
Document indexer: load files → split → embed → persist to Chroma.

Embedding strategy (fastest available wins):
  1. Google text-embedding-004  — if GOOGLE_API_KEY is set (API call, no local model, instant startup)
  2. HuggingFace all-MiniLM-L6-v2 — fully local fallback (loads PyTorch, slower to start)

NOTE: switching embedding backends on an existing index requires --reindex,
      because the two models produce incompatible vector spaces.
"""
from __future__ import annotations

import os
import pathlib
import shutil

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EXTENSIONS = {".py", ".md", ".txt", ".rst"}
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DEFAULT_PERSIST = "./chroma_db"

# Directories that are never worth indexing
SKIP_DIRS = {
    ".venv", "venv", ".env", "__pycache__", ".git", ".hg", ".svn",
    "node_modules", ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "chroma_db", "dist", "build", ".eggs", "site-packages",
}


def _make_embeddings() -> tuple[Embeddings, str]:
    """Return (embeddings, label). Prefers Google to avoid local model load time."""
    load_dotenv()
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()

    if google_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return (
            GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=google_key,
            ),
            "Google gemini-embedding-001",
        )

    # Local fallback — loads PyTorch + ~90 MB model on first use
    from langchain_huggingface import HuggingFaceEmbeddings
    return (
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        "HuggingFace all-MiniLM-L6-v2 (local)",
    )


def _load_files(dir_path: str) -> list[Document]:
    root = pathlib.Path(dir_path).expanduser().resolve()
    docs: list[Document] = []
    for path in sorted(root.rglob("*")):
        # Skip any path whose ancestors include a blacklisted directory name
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix in EXTENSIONS:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                rel = str(path.relative_to(root))
                docs.append(Document(
                    page_content=text,
                    metadata={"source": rel, "file_type": path.suffix.lstrip(".")},
                ))
            except Exception:
                pass
    return docs


def index_directory(
    dir_path: str,
    persist_dir: str = DEFAULT_PERSIST,
    force_reindex: bool = False,
) -> Chroma:
    """Load, split, embed, and persist documents from dir_path."""
    persist_path = pathlib.Path(persist_dir)

    embeddings, embed_label = _make_embeddings()
    print(f"Embeddings: {embed_label}")

    if persist_path.exists() and not force_reindex:
        print(f"Index already exists at '{persist_dir}' — loading. (Use --reindex to rebuild.)")
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    if persist_path.exists() and force_reindex:
        shutil.rmtree(persist_path)

    print(f"Scanning '{dir_path}' for {', '.join(sorted(EXTENSIONS))} files...")
    raw_docs = _load_files(dir_path)
    if not raw_docs:
        raise ValueError(f"No supported files found in '{dir_path}'.")

    print(f"  Loaded {len(raw_docs)} files. Splitting into chunks...")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    ).split_documents(raw_docs)

    chunks = [c for c in chunks if c.page_content.strip()]
    print(f"  {len(chunks)} chunks. Embedding...")
    vectorstore = Chroma.from_documents(
        chunks, embedding=embeddings, persist_directory=persist_dir
    )
    print(f"  Done. Index saved to '{persist_dir}'.")
    return vectorstore


def load_vectorstore(persist_dir: str = DEFAULT_PERSIST) -> Chroma:
    """Load an existing Chroma index from disk."""
    if not pathlib.Path(persist_dir).exists():
        raise FileNotFoundError(
            f"No Chroma DB at '{persist_dir}'. Run with --reindex first."
        )
    embeddings, _ = _make_embeddings()
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
