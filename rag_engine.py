"""
rag_engine.py
─────────────
DocuMind — Core RAG pipeline:
  • HuggingFace embeddings  (all-MiniLM-L6-v2, local — no extra API key)
  • Pinecone               (vector store & semantic search)
  • Groq + LangChain       (LLM orchestration)
  • ConversationBufferMemory (multi-turn memory)
  • Source-cited responses
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_TEMPLATE = """You are an expert research assistant with deep knowledge of \
document analysis. You answer questions using ONLY the provided context from the \
user's uploaded documents.

Rules:
1. Base every answer strictly on the retrieved context below.
2. If the context doesn't contain enough information, say so clearly — do NOT \
   hallucinate or use outside knowledge.
3. Always be concise yet thorough. Use bullet points or numbered lists when helpful.
4. Cite relevant document sections naturally (e.g. "According to the uploaded document…").
5. If asked for a summary, provide a structured overview with key points.

Context:
────────
{context}
────────
"""

HUMAN_TEMPLATE = "{question}"


# ── RAGEngine ─────────────────────────────────────────────────────────────────
class RAGEngine:
    """
    Encapsulates the full RAG pipeline.

    Usage:
        engine = RAGEngine(groq_api_key="...", pinecone_api_key="...")
        engine.initialize()
        engine.add_documents(chunks)
        result = engine.query("What is the main finding?")
        print(result["answer"])
        for src in result["sources"]:
            print(src.metadata)
    """

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim, MIT licence, runs on CPU
    EMBEDDING_DIM   = 384
    METRIC          = "cosine"

    def __init__(
        self,
        groq_api_key: str,
        pinecone_api_key: str,
        index_name: str = "documind-index",
        groq_model: str = "llama-3.3-70b-versatile",
        pinecone_cloud: str = "aws",
        pinecone_region: str = "us-east-1",
        top_k: int = 4,
        temperature: float = 0.1,
    ) -> None:
        self.groq_api_key     = groq_api_key
        self.pinecone_api_key = pinecone_api_key
        self.index_name       = index_name
        self.groq_model       = groq_model
        self.pinecone_cloud   = pinecone_cloud
        self.pinecone_region  = pinecone_region
        self.top_k            = top_k
        self.temperature      = temperature

        # Populated in initialize()
        self.embeddings:    Optional[HuggingFaceEmbeddings]     = None
        self.vector_store:  Optional[PineconeVectorStore]        = None
        self.chain:         Optional[ConversationalRetrievalChain] = None
        self.memory:        Optional[ConversationBufferMemory]   = None
        self._pc:           Optional[Pinecone]                   = None
        self._initialized   = False

    # ── Public API ────────────────────────────────────────────────────────────

    def initialize(self, status_callback=None) -> None:
        """
        Boot the engine: load embeddings, connect to Pinecone, build chain.
        `status_callback(message: str)` is called with progress updates.
        """
        def _status(msg: str) -> None:
            if status_callback:
                status_callback(msg)

        _status("⚙️  Loading embedding model (all-MiniLM-L6-v2)…")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        _status("🌲  Connecting to Pinecone…")
        self._pc = Pinecone(api_key=self.pinecone_api_key)
        self._ensure_index()

        self.vector_store = PineconeVectorStore(
            index=self._pc.Index(self.index_name),
            embedding=self.embeddings,
            text_key="text",
        )

        _status("🤖  Initialising Groq LLM…")
        llm = ChatGroq(
            api_key=self.groq_api_key,
            model_name=self.groq_model,
            temperature=self.temperature,
            streaming=True,
        )

        _status("🧠  Building conversational chain…")
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
            HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE),
        ])

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.top_k},
            ),
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt},
            verbose=False,
        )

        self._initialized = True
        _status("✅  RAG engine ready.")

    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
        status_callback=None,
    ) -> int:
        """
        Embed and upsert document chunks into Pinecone.
        Returns the number of chunks ingested.
        """
        self._assert_initialized()

        total = len(documents)
        ingested = 0

        for i in range(0, total, batch_size):
            batch = documents[i : i + batch_size]
            self.vector_store.add_documents(batch)
            ingested += len(batch)
            if status_callback:
                status_callback(f"📥  Indexed {ingested}/{total} chunks…")
            time.sleep(0.1)   # be gentle with the API

        return ingested

    def query(self, question: str) -> Dict:
        """
        Run a RAG query against the indexed documents.

        Returns:
            {
                "answer":  str,
                "sources": List[Document],
                "question": str,
            }
        """
        self._assert_initialized()

        result = self.chain.invoke({"question": question})

        # De-duplicate sources by (source, page)
        seen: set = set()
        unique_sources: List[Document] = []
        for doc in result.get("source_documents", []):
            key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", ""),
            )
            if key not in seen:
                seen.add(key)
                unique_sources.append(doc)

        return {
            "question": question,
            "answer":   result["answer"],
            "sources":  unique_sources,
        }

    def clear_memory(self) -> None:
        """Reset conversation history."""
        if self.memory:
            self.memory.clear()

    def clear_index(self) -> None:
        """Delete all vectors from the Pinecone index."""
        self._assert_initialized()
        self._pc.Index(self.index_name).delete(delete_all=True)

    def get_index_stats(self) -> Dict:
        """Return Pinecone index statistics."""
        self._assert_initialized()
        return self._pc.Index(self.index_name).describe_index_stats()

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_index(self) -> None:
        """Create the Pinecone serverless index if it doesn't exist yet."""
        existing = [idx.name for idx in self._pc.list_indexes()]
        if self.index_name not in existing:
            self._pc.create_index(
                name=self.index_name,
                dimension=self.EMBEDDING_DIM,
                metric=self.METRIC,
                spec=ServerlessSpec(
                    cloud=self.pinecone_cloud,
                    region=self.pinecone_region,
                ),
            )
            # Wait for the index to be ready
            timeout = 60
            elapsed = 0
            while elapsed < timeout:
                desc = self._pc.describe_index(self.index_name)
                if desc.status.get("ready", False):
                    break
                time.sleep(2)
                elapsed += 2

    def _assert_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "RAGEngine is not initialised. Call engine.initialize() first."
            )
