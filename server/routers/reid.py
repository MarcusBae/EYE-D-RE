"""
server/routers/reid.py
======================
Re-ID 관련 API 엔드포인트.
pipeline.reid_merger.OSNetExtractor를 사용하여 특징 추출 및 검색을 수행합니다.
"""

from fastapi import APIRouter

router = APIRouter(tags=["reid"])


@router.get("/status")
def reid_status():
    """Re-ID 엔진 상태 확인."""
    return {"engine": "OSNet", "status": "ready"}
