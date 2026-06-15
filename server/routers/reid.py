"""server/routers/reid.py — Re-ID API: 임베딩 추출 + DB 코사인 검색 + 판정(읽기전용)."""
import sys
from collections import OrderedDict
import numpy as np
import cv2
import yaml
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from server.db import get_conn

ROOT = Path(__file__).resolve().parent.parent.parent
router = APIRouter(tags=["reid"])
_extractor = None

def get_extractor():
    global _extractor
    if _extractor is None:
        sys.path.insert(0, str(ROOT))
        from pipeline.reid_merger import OSNetExtractor
        cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml", encoding="utf-8"))
        rc = cfg.get("reid", {})
        wp = rc.get("weights_path")
        if wp and not Path(wp).is_absolute():
            wp = str(ROOT / wp)
        _extractor = OSNetExtractor(model_name=rc.get("model_name", "osnet_x1_0"),
            pretrained=rc.get("pretrained", True), weights_path=wp,
            device=rc.get("device", "auto"))
    return _extractor

def _embed(file_bytes):
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "이미지를 디코딩할 수 없습니다.")
    return np.asarray(get_extractor().extract_image_feature(img), dtype=np.float32).ravel()

def _search(feat, top_k):
    vec = "[" + ",".join(f"{x:.6f}" for x in feat) + "]"
    sql = """
        SELECT d.global_id, p.display_name, p.customer_tier, p.is_vip, p.visit_count,
               ROUND((1 - (d.embedding_identity <=> %s::vector))::numeric, 4)::float8 AS similarity
        FROM detections d JOIN persons p ON p.global_id = d.global_id
        ORDER BY d.embedding_identity <=> %s::vector
        LIMIT %s;
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (vec, vec, top_k))
            return cur.fetchall()
    finally:
        conn.close()

@router.get("/status")
def reid_status():
    return {"engine": "OSNet", "status": "ready", "model_loaded": _extractor is not None}

@router.post("/extract")
async def extract_embedding(file: UploadFile = File(...)):
    feat = _embed(await file.read())
    return {"dim": int(feat.shape[0]), "embedding": feat.tolist()}

@router.post("/match")
async def match_person(file: UploadFile = File(...), top_k: int = 10, threshold: float = 0.6):
    """쿼리 이미지 -> 임베딩 -> top_k 검색 -> 인물별 투표·최고유사도 -> 임계 판정."""
    feat = _embed(await file.read())
    rows = _search(feat, top_k)
    agg = OrderedDict()
    for r in rows:
        g = r["global_id"]
        if g not in agg:
            agg[g] = {"global_id": g, "display_name": r["display_name"],
                      "customer_tier": r["customer_tier"], "is_vip": r["is_vip"],
                      "visit_count": r["visit_count"], "votes": 0, "max_similarity": 0.0}
        agg[g]["votes"] += 1
        agg[g]["max_similarity"] = max(agg[g]["max_similarity"], r["similarity"])
    candidates = sorted(agg.values(), key=lambda x: (x["votes"], x["max_similarity"]), reverse=True)
    best = candidates[0] if candidates else None
    if best and best["max_similarity"] >= threshold:
        decision = {"status": "matched", "global_id": best["global_id"],
                    "display_name": best["display_name"], "customer_tier": best["customer_tier"],
                    "is_vip": best["is_vip"], "similarity": best["max_similarity"], "votes": best["votes"]}
    else:
        bs = best["max_similarity"] if best else 0.0
        decision = {"status": "unknown", "reason": f"best similarity {bs} < threshold {threshold}"}
    return {"decision": decision, "threshold": threshold, "top_k": top_k, "candidates": candidates}
