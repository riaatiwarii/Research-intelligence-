from fastapi import APIRouter

router = APIRouter()

@router.post("/")
def generate_ppt():
    return {
        "status": "ppt generation not implemented yet"
    }
