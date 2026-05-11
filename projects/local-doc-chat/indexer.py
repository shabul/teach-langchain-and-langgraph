"""
Document indexer: load files → split → embed locally → persist to Chroma.

Uses HuggingFaceEmbeddings (sentence-transformers) so nothing leaves the machine.

pip install sentence-transformers langchain-huggingface langchain-chroma
"""
from __future__ import annotations

import pathlib
from typing import Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EXTENSIONS = {".py", ".md", ".txt", ".rst"}
CHUNK_SIZE = 500      # ~375 tokens — fits 3 chunks comfortably in 11k context
CHUNK_OVERLAP = 50
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_PERSIST = "./chroma_db"


def _make_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def _load_files(dir_path: str) -> list[Document]:
    root = pathlib.Path(dir_path).expanduser().resolve()
    docs: list[Document] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in EXTENSIONS:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                rel = str(path.relative_to(root))
                docs.append(Document(
                    page_content=text,
                    metadata={"source": rel, "file_type": path.suffix.lstrip(".")},
                ))
            except Exception:
                pass  # skip unreadable files
    return docs


def index_directory(
    dir_path: str,
    persist_dir: str = DEFAULT_PERSIST,
    force_reindex: bool = False,
) -> Chroma:
    """Load, split, embed, and persist documents from dir_path."""
    persist_path = pathlib.Path(persist_dir)

    if persist_path.exists() and not force_reindex:
        print(f"Chroma DB already exists at '{persist_dir}'. Use --reindex to rebuild.")
        return load_vectorstore(persist_dir)

    print(f"Scanning '{dir_path}' for {', '.join(EXTENSIONS)} files...")
    raw_docs = _load_files(dir_path)
    if not raw_docs:
        raise ValueError(f"No supported files found in '{dir_path}'.")

    print(f"  Loaded {len(raw_docs)} files. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(raw_docs)
    print(f"  {len(chunks)} chunks. Embedding with '{EMBED_MODEL}' (first run downloads ~90 MB)...")

    embeddings = _make_embeddings()
    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"  Done. Index saved to '{persist_dir}'.")
    return vectorstore


def load_vectorstore(persist_dir: str = DEFAULT_PERSIST) -> Chroma:
    """Load an existing Chroma index from disk."""
    if not pathlib.Path(persist_dir).exists():
        raise FileNotFoundError(
            f"No Chroma DB at '{persist_dir}'. Run with --reindex first."
        )
    return Chroma(persist_directory=persist_dir, embedding_function=_make_embeddings())
