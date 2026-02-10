from fastapi import APIRouter

router = APIRouter()

@router.post("/")
def detect_gaps():
    return {
        "status": "gap detection not wired yet"
    }
