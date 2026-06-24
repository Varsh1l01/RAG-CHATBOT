# 🧠 DocuMind
### Groq × Pinecone × LangChain × HuggingFace Embeddings

A production-ready Retrieval-Augmented Generation (RAG) pipeline that lets you
chat with your own documents — PDFs, Word files, plain text, and Markdown — with
full conversational memory and source-cited responses.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Streamlit UI                         │
│           (upload · chat · source citations)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │    Document Processor   │
          │  PDF · DOCX · TXT · MD  │
          │  RecursiveCharSplitter  │
          └────────────┬────────────┘
                       │ chunks (LangChain Documents)
          ┌────────────▼────────────┐
          │  HuggingFace Embeddings │
          │   all-MiniLM-L6-v2      │
          │      (384 dims)         │
          └────────────┬────────────┘
                       │ vectors
          ┌────────────▼────────────┐
          │       Pinecone          │
          │  Serverless Vector DB   │
          │   cosine similarity     │
          └────────────┬────────────┘
                       │ top-k chunks
          ┌────────────▼────────────┐
          │   LangChain RAG Chain   │
          │  ConversationalRetrieval│
          │   + BufferMemory        │
          └────────────┬────────────┘
                       │ prompt + context
          ┌────────────▼────────────┐
          │       Groq LLM          │
          │  llama-3.1-70b /        │
          │  mixtral-8x7b / etc.    │
          └─────────────────────────┘
```

---

## Features

| Feature | Details |
|---|---|
| **Document formats** | PDF, DOCX, DOC, TXT, MD, Markdown |
| **Embeddings** | `all-MiniLM-L6-v2` — local, no API cost |
| **Vector store** | Pinecone Serverless (auto-created) |
| **LLM** | Groq (llama-3.1-70b, mixtral, gemma2) |
| **Memory** | `ConversationBufferMemory` — full chat history |
| **Source citations** | Every answer cites source file + page + chunk |
| **Chunking** | Recursive character splitting (configurable) |
| **Anti-hallucination** | Strict system prompt; only answers from context |

---

## Quick Start

### 1 — install

```bash
git clone <repo>
cd documind
py -3.11 -m venv .venv
.venv\Scripts\activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your keys:
#   GROQ_API_KEY    → https://console.groq.com/keys
#   PINECONE_API_KEY → https://app.pinecone.io/
```

### 3 — Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Usage

1. **Enter API keys** in the sidebar (or set them in `.env`)
2. **Click "Initialise Engine"** — loads embeddings and connects to Pinecone
3. **Upload documents** (PDF, DOCX, TXT, or MD) in the right panel
4. **Click "Index Documents"** to embed and store them in Pinecone
5. **Start chatting** — ask anything about your documents

---

## Configuration

| Setting | Default | Description |
|---|---|---|
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVER_TOP_K` | `4` | Chunks retrieved per query |
| `PINECONE_INDEX_NAME` | `documind-index` | Pinecone index name |
| `PINECONE_REGION` | `us-east-1` | Pinecone serverless region |

---

## Available Groq Models

| Model | Context | Best For |
|---|---|---|
| `llama-3.3-70b-versatile` | 128k | Best quality, complex Q&A |
| `llama-3.1-8b-instant` | 128k | Fast, lower latency |
| `meta-llama/llama-4-scout-17b-16e-instruct` | 128k | Latest Llama 4, experimental |
| `qwen/qwen3-32b` | 128k | Strong alternative, reasoning |

---

## Project Structure

```
documind/
├── app.py                  # Streamlit UI
├── rag_engine.py           # Core RAG pipeline (Groq + Pinecone + LangChain)
├── document_processor.py   # Document loading & chunking
├── requirements.txt
├── .env.example
└── README.md
```

---

## Key Design Decisions

- **Local embeddings**: HuggingFace `all-MiniLM-L6-v2` runs on CPU with no API key
  or cost, and produces high-quality 384-dim vectors.
- **Pinecone Serverless**: Auto-created on first run; sub-second similarity search.
- **Recursive splitter**: Tries paragraph → sentence → word → character breaks,
  preserving semantic coherence far better than fixed-size splits.
- **Strict system prompt**: The LLM is explicitly told to answer ONLY from the
  retrieved context, drastically reducing hallucinations.
- **De-duplicated sources**: Source citations are de-duplicated by (file, page)
  so the UI stays clean.

---

## License

MIT
