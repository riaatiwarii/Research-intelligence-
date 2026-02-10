from fastapi import FastAPI
from backend.routers import qa, search, gaps, ppt

app = FastAPI(title="Research Intelligence API")

app.include_router(qa.router, prefix="/qa")
app.include_router(search.router, prefix="/search")
app.include_router(gaps.router, prefix="/gaps")
app.include_router(ppt.router, prefix="/ppt")
