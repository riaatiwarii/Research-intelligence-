# qa/rag_qa.py

from qa.retrieve import retrieve_chunks
from qa.answer import generate_answer


def rag_qa(question: str, top_k: int = 8):
    chunks = retrieve_chunks(question, top_k=4)

    if not chunks:
        return {
            "answer": "No relevant literature found.",
            "citations": []
        }

    result = generate_answer(question, chunks)
    return result
