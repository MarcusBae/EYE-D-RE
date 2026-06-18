"""Smart Retail API — Re-ID 코어 위에서 고객 분석 신호 제공 (읽기전용)."""
from fastapi import APIRouter, HTTPException
from server.db import get_conn
from server.verticals.retail import retail_signal

router = APIRouter(tags=["retail"])

@router.get("/profile/{global_id}")
def customer_profile(global_id: int):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT global_id, visit_count, customer_tier, is_vip, display_name "
                        "FROM persons WHERE global_id=%s", (global_id,))
            p = cur.fetchone()
    finally:
        conn.close()
    if not p:
        raise HTTPException(404, f"global_id {global_id} 없음")
    return retail_signal(p)

@router.get("/top-customers")
def top_customers(limit: int = 10):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT global_id, visit_count, customer_tier, is_vip "
                        "FROM persons ORDER BY visit_count DESC, global_id LIMIT %s", (limit,))
            rows = cur.fetchall()
    finally:
        conn.close()
    return {"top_customers": [retail_signal(r) for r in rows]}
