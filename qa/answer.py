# qa/answer.py

import requests
import os
from typing import List, Dict

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("OLLAMA_MODEL", "phi")


SYSTEM_PROMPT = """
You are a research assistant.

Rules:
- Answer ONLY using the provided context.
- Do NOT use outside knowledge.
- If the answer is not present, say: "Not found in the provided literature."
- Cite paper_ids explicitly at the end.
"""


def build_context(chunks: List[Dict]) -> str:
    context_blocks = []
    for c in chunks:
        block = (
            f"[Paper ID: {c['paper_id']} | Section: {c['section']}]\n"
            f"{c['text'][:600]}"
        )
        context_blocks.append(block)
    return "\n\n".join(context_blocks)



def generate_answer(question: str, chunks: List[Dict]) -> Dict:
    context = build_context(chunks)

    prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{question}

Answer:
"""

    response = requests.post(
    OLLAMA_URL,
    json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 200,   # HARD STOP
            "temperature": 0.2,
            "num_ctx": 2048,
        }
    },
    timeout=300   # allow slow CPUs
    )


    response.raise_for_status()
    answer_text = response.json()["response"].strip()

    cited_papers = list({c["paper_id"] for c in chunks})

    return {
        "answer": answer_text,
        "citations": cited_papers
    }
