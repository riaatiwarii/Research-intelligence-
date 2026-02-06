# api/main.py
from fastapi import FastAPI
from qa.rag_qa import answer

app = FastAPI()

@app.get("/ask")
def ask(q: str):
    return {"answer": answer(q)}
