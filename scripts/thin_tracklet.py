"""
scripts/thin_tracklet.py
=========================
특정 트랙렛 폴더의 프레임을 균등하게 솎아내기.
이미지가 너무 많은 개별 트랙렛을 빠르게 정리할 때 사용.

사용법:
    python scripts/thin_tracklet.py data/filtered/c1_t10/track_0067 --keep 1/2
    python scripts/thin_tracklet.py data/filtered/c1_t10/track_0067 --keep 1/3 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent


def parse_fraction(s: str) -> float:
    """'1/2', '1/3' 등의 분수 문자열을 float으로 변환."""
    try:
        if "/" in s:
            num, den = s.split("/")
            return int(num) / int(den)
        return float(s)
    except Exception:
        raise argparse.ArgumentTypeError(f"올바른 분수 형식이 아닙니다: '{s}'  (예: 1/2, 1/3)")


def main():
    parser = argparse.ArgumentParser(description="단일 트랙렛 프레임 솎아내기")
    parser.add_argument("tracklet_dir",
                        help="트랙렛 폴더 경로 (예: data/filtered/c1_t10/track_0067)")
    parser.add_argument("--keep", type=parse_fraction, required=True,
                        metavar="1/N",
                        help="유지할 비율 (예: 1/2=절반 유지, 1/3=1/3 유지)")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 삭제 없이 결과만 출력")
    args = parser.parse_args()

    if not (0 < args.keep < 1):
        sys.exit("[ERROR] --keep 값은 0~1 사이여야 합니다 (예: 1/2, 1/3)")

    track_dir = Path(args.tracklet_dir)
    if not track_dir.is_absolute():
        track_dir = project_root / track_dir
    if not track_dir.exists():
        sys.exit(f"[ERROR] 폴더 없음: {track_dir}")

    meta_path = track_dir / "metadata.json"
    if not meta_path.exists():
        sys.exit(f"[ERROR] metadata.json 없음: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    crop_files    = meta.get("crop_files", [])
    frame_indices = meta.get("frame_indices", [])
    bboxes        = meta.get("bboxes", [])
    confidences   = meta.get("confidences", [])

    if not crop_files:
        sys.exit("[ERROR] 프레임 정보가 없습니다.")

    # 실제 존재하는 파일만 대상으로 (두 번 실행 시 불일치 방지)
    crop_files = [f for f in crop_files if (track_dir / f).exists()]
    if not crop_files:
        sys.exit("[ERROR] 실제 존재하는 이미지가 없습니다.")

    # 모든 필드를 crop_files 길이에 맞춰 동기화
    n = len(crop_files)
    frame_indices = frame_indices[:n]
    bboxes        = bboxes[:n]
    confidences   = confidences[:n]

    total = n
    keep_count = max(1, round(total * args.keep))

    # 균등 간격으로 keep_count개 선택
    keep_idx = [round(i * (total - 1) / (keep_count - 1)) for i in range(keep_count)] \
               if keep_count > 1 else [0]
    keep_idx = sorted(set(keep_idx))  # 중복 제거

    keep_files   = set(crop_files[i] for i in keep_idx)
    remove_files = [f for f in crop_files if f not in keep_files]

    print(f"[INFO] 트랙렛 : {track_dir.parent.name}/{track_dir.name}")
    print(f"[INFO] {total}장 × {args.keep:.4g} → {len(keep_idx)}장 유지 / {len(remove_files)}장 삭제")

    if not remove_files:
        print("[INFO] 삭제할 프레임 없음.")
        return

    if args.dry_run:
        print("[DRY-RUN] 삭제 예정 파일:")
        for f in remove_files:
            print(f"  {f}")
        return

    # 파일 삭제
    deleted = 0
    for fname in remove_files:
        p = track_dir / fname
        if p.exists():
            p.unlink()
            deleted += 1

    # metadata 갱신
    new_crops  = [crop_files[i] for i in keep_idx]
    new_frames = [frame_indices[i] for i in keep_idx] if frame_indices else []
    new_bboxes = [bboxes[i] for i in keep_idx] if bboxes else []
    new_confs  = [confidences[i] for i in keep_idx] if confidences else []

    meta["crop_files"]    = new_crops
    meta["frame_indices"] = new_frames
    meta["bboxes"]        = new_bboxes
    meta["confidences"]   = new_confs
    meta["length"]        = len(new_crops)
    meta["avg_conf"]      = float(np.mean(new_confs)) if new_confs else 0.0
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[완료] {deleted}장 삭제, metadata.json 갱신")


if __name__ == "__main__":
    main()
