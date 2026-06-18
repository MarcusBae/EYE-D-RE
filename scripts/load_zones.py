#!/usr/bin/env python3
"""configs/zones.yaml -> DB zones 테이블 적재. 기본 dry-run(미리보기), --apply 로 반영.
없는 camera_id 는 cameras 에 자동 등록. zone_id 단위 upsert(--replace-all 시 전체 교체)."""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 프로젝트 루트
import yaml
from server.db import get_conn
from server.verticals.security import is_restricted

REQUIRED = ("zone_id", "name", "camera_id", "purpose", "polygon")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="configs/zones.yaml")
    ap.add_argument("--apply", action="store_true", help="실제 DB 반영(기본 dry-run)")
    ap.add_argument("--replace-all", action="store_true", help="기존 zones 전체 삭제 후 적재")
    a = ap.parse_args()

    data = yaml.safe_load(Path(a.config).read_text(encoding="utf-8")) or {}
    zones = data.get("zones") or []
    if not zones: sys.exit(f"[중단] {a.config} 에 zones 없음")
    for i, z in enumerate(zones):
        for k in REQUIRED:
            if k not in z: sys.exit(f"[중단] zones[{i}] '{k}' 누락")
        if not isinstance(z["polygon"], list) or len(z["polygon"]) < 3:
            sys.exit(f"[중단] {z['zone_id']} polygon 은 [x,y] 3개 이상")

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # 없는 카메라 자동 등록
            cur.execute("SELECT camera_id FROM cameras")
            existing = {r["camera_id"] for r in cur.fetchall()}
            new_cams = sorted({z["camera_id"] for z in zones} - existing)
            for c in new_cams:
                cur.execute("INSERT INTO cameras (camera_id) VALUES (%s) ON CONFLICT (camera_id) DO NOTHING", (c,))
            if new_cams: print(f"[카메라 자동등록] {new_cams}")

            if a.replace_all:
                cur.execute("DELETE FROM zones"); print("[replace-all] 기존 zones 전체 삭제")
            for z in zones:
                cur.execute("DELETE FROM zones WHERE zone_id=%s", (z["zone_id"],))
                cur.execute("INSERT INTO zones (zone_id,name,camera_id,polygon,purpose) "
                            "VALUES (%s,%s,%s,%s::jsonb,%s)",
                            (z["zone_id"], z["name"], z["camera_id"],
                             json.dumps(z["polygon"]), z["purpose"]))
                print(f"  [{'제한' if is_restricted(z) else '허용'}] {z['zone_id']:14} "
                      f"({z['camera_id']}) pts={len(z['polygon'])} purpose={z['purpose']}")
            cur.execute("SELECT count(*) AS n FROM zones"); total = cur.fetchone()["n"]
        if a.apply:
            conn.commit(); print(f"=== 반영됨 · zones 총 {total}건 ===")
        else:
            conn.rollback(); print(f"=== DRY-RUN(미반영) · 적용 시 {total}건 · 반영은 --apply ===")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
