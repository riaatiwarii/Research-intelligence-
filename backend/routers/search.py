from fastapi import APIRouter

router = APIRouter()

@router.post("/")
def search(query: str):
    return {
        "query": query,
        "results": []
    }
