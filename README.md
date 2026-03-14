# Research Intelligence Platform

AI-powered research assistant combining semantic search, RAG-based Q&A, research gap detection, and automated presentation generation.

## Features

- **Semantic Search** — Find papers by meaning, not keywords
- **Research Gap Detection** — Identify unexplored areas via HDBSCAN clustering
- **RAG Q&A** — Ask questions grounded in retrieved literature
- **PDF Analysis** — Upload a paper and detect continuation opportunities
- **Auto PPT Generation** — Generate research slides instantly

## Tech Stack

| Layer | Tool |
|---|---|
| Backend | FastAPI |
| Frontend | Streamlit |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector DB | Qdrant |
| Clustering | HDBSCAN |
| LLM | Ollama (Mistral, local) |
| Slides | python-pptx |

## Setup
```bash
git clone https://github.com/yourusername/research-intelligence.git
cd research-intelligence
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Start Qdrant:**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Start Ollama:**
```bash
ollama run mistral
```

## Run
```bash
uvicorn backend.main:app --reload   # API → http://localhost:8000
streamlit run frontend/app.py       # UI  → http://localhost:8501
```

## How It Works

1. Papers are embedded and stored in Qdrant
2. User query → vector similarity search → top-K chunks retrieved
3. Chunks passed to Ollama → grounded answer returned
4. HDBSCAN detects outlier embeddings → potential research gaps

## Dataset

Papers sourced from PubMed Central, CORE, and arXiv — cleaned, chunked, and embedded.

## License

MIT
