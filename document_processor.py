"""
document_processor.py
─────────────────────
Handles loading and chunking of PDF, DOCX, TXT, and Markdown files.
Uses LangChain community loaders + RecursiveCharacterTextSplitter.
"""

import os
import tempfile
from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)


# ── Supported extensions ──────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".md", ".markdown"}


def get_loader(file_path: str):
    """Return the appropriate LangChain loader for the given file extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext in {".docx", ".doc"}:
        return Docx2txtLoader(file_path)
    elif ext in {".txt", ".md", ".markdown"}:
        return TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_document(file_path: str) -> List[Document]:
    """Load a single document and return a list of LangChain Document objects."""
    loader = get_loader(file_path)
    docs = loader.load()

    # Inject source metadata
    filename = Path(file_path).name
    for doc in docs:
        doc.metadata["source"] = filename
        doc.metadata["file_path"] = file_path

    return docs


def load_uploaded_file(uploaded_file) -> List[Document]:
    """
    Accept a Streamlit UploadedFile object, save it to a temp file,
    load it, and return Document chunks.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        docs = load_document(tmp_path)
        # Override source with the real uploaded filename
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
    finally:
        os.unlink(tmp_path)

    return docs


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.

    The splitter tries to break on paragraphs → sentences → words → characters,
    preserving semantic coherence as much as possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    # Tag each chunk with its index within its source file
    source_counters: dict = {}
    for chunk in chunks:
        src = chunk.metadata.get("source", "unknown")
        source_counters[src] = source_counters.get(src, 0) + 1
        chunk.metadata["chunk_index"] = source_counters[src]

    return chunks


def process_uploaded_file(
    uploaded_file,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """End-to-end: load → chunk a Streamlit UploadedFile."""
    docs = load_uploaded_file(uploaded_file)
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunks


def format_source_citation(doc: Document) -> str:
    """Return a human-readable citation string for a source Document."""
    meta = doc.metadata
    source = meta.get("source", "Unknown source")
    page = meta.get("page")
    chunk = meta.get("chunk_index")

    parts = [f"📄 **{source}**"]
    if page is not None:
        parts.append(f"Page {int(page) + 1}")
    if chunk is not None:
        parts.append(f"Chunk {chunk}")

    return " · ".join(parts)
