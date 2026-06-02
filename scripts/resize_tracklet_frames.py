"""
scripts/resize_tracklet_frames.py
===================================
트랙렛 프레임을 256×128 (H×W) 로 일괄 리사이즈 (in-place).
OSNet / TransReID / CLIP-ReID 등 주요 Re-ID 모델 공통 입력 크기.

사용법:
    python scripts/resize_tracklet_frames.py --dry-run   # 미리 확인
    python scripts/resize_tracklet_frames.py             # 실제 변환
    python scripts/resize_tracklet_frames.py --workers 8 # 병렬 수 조정
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent

TARGET_H, TARGET_W = 256, 128  # Re-ID 표준 입력 크기


def resize_one(jpg_path: str) -> tuple[str, bool]:
    """단일 이미지 리사이즈. 반환: (경로, 성공 여부)"""
    try:
        img = cv2.imread(jpg_path)
        if img is None:
            return jpg_path, False
        h, w = img.shape[:2]
        if h == TARGET_H and w == TARGET_W:
            return jpg_path, True  # 이미 맞는 크기
        resized = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(jpg_path, resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return jpg_path, True
    except Exception:
        return jpg_path, False


def main():
    parser = argparse.ArgumentParser(description=f"트랙렛 이미지 {TARGET_H}×{TARGET_W} 리사이즈")
    parser.add_argument("--tracklets-dir", default="data/tracklets-1-original-reduced")
    parser.add_argument("--workers", type=int, default=4, help="병렬 프로세스 수 (기본: 4)")
    parser.add_argument("--dry-run", action="store_true", help="실제 변환 없이 대상만 출력")
    args = parser.parse_args()

    tracklet_dir = Path(args.tracklets_dir)
    if not tracklet_dir.is_absolute():
        tracklet_dir = project_root / tracklet_dir
    if not tracklet_dir.exists():
        sys.exit(f"[ERROR] 디렉토리 없음: {tracklet_dir}")

    jpg_paths = sorted(str(p) for p in tracklet_dir.glob("*/track_*/frame_*.jpg"))
    if not jpg_paths:
        sys.exit("[ERROR] 이미지 파일이 없습니다.")

    total_mb_before = sum(Path(p).stat().st_size for p in jpg_paths) / 1024 / 1024
    print(f"[INFO] 대상: {len(jpg_paths):,}장  ({total_mb_before:.0f} MB)")
    print(f"[INFO] 목표 크기: {TARGET_H}×{TARGET_W} (H×W)")

    if args.dry_run:
        print("[DRY-RUN] 실제 변환 없음")
        return

    failed = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(resize_one, p): p for p in jpg_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="리사이즈"):
            path, ok = future.result()
            if not ok:
                failed.append(path)

    total_mb_after = sum(Path(p).stat().st_size for p in jpg_paths) / 1024 / 1024

    print()
    print("=" * 45)
    print(f"  변환 완료 : {len(jpg_paths) - len(failed):,}장")
    if failed:
        print(f"  실패      : {len(failed)}장")
    print(f"  용량 변화 : {total_mb_before:.0f} MB → {total_mb_after:.0f} MB")
    print("=" * 45)


if __name__ == "__main__":
    main()
