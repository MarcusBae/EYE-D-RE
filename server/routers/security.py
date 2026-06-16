"""침입탐지 API — zones와 검출 위치를 대조 (읽기전용)."""
from typing import List
from fastapi import APIRouter
from server.db import get_conn
from server.verticals.security import is_restricted, intrusion_check

router = APIRouter(tags=["security"])

def _load_zones():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT zone_id, name, camera_id, polygon, purpose FROM zones")
            return cur.fetchall()
    finally:
        conn.close()

@router.get("/zones")
def list_zones():
    zs = _load_zones()
    return {"zones": [{"zone_id": z["zone_id"], "name": z["name"], "purpose": z["purpose"],
                       "restricted": is_restricted(z), "has_polygon": bool(z["polygon"])} for z in zs]}

@router.post("/check")
def check_intrusion(bbox: List[float]):
    """검출 bbox [x1,y1,x2,y2]가 제한구역 침입인지 판정."""
    return intrusion_check(bbox, _load_zones())
