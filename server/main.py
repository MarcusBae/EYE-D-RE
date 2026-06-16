"""
server/main.py
==============
EYE-D FastAPI 서버 진입점.

실행:
  uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from server.routers import reid, health, retail

app = FastAPI(
    title="EYE-D Re-ID API",
    version="0.1.0",
    description="Person Re-Identification API",
)

app.include_router(health.router)
app.include_router(reid.router, prefix="/reid")
app.include_router(retail.router, prefix="/retail")
