from fastapi import APIRouter
from qa.rag_qa import rag_qa

router = APIRouter()

@router.post("/")
def ask(question: str):
    return rag_qa(question)
