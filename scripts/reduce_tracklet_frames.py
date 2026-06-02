"""
scripts/reduce_tracklet_frames.py
===================================
트랙렛 내 프레임을 실제 frame 번호 기준으로 간격을 두어 삭제.
각 트랙렛의 frame_stride를 읽어 min_gap_frames를 자동 계산.
metadata.json도 함께 갱신.

동작:
  - 첫 프레임 유지
  - 이후 직전 유지 프레임과 frame 번호 차이가 min_gap 미만이면 삭제

사용법:
    python scripts/reduce_tracklet_frames.py --stride 5 --dry-run  # 미리 확인
    python scripts/reduce_tracklet_frames.py --stride 5            # 실제 삭제
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent


def select_keep_indices(frame_indices: list[int], min_gap: int) -> list[int]:
    """frame 번호 기준으로 min_gap 이상 차이나는 프레임만 유지."""
    if not frame_indices:
        return []
    keep = [0]
    for i in range(1, len(frame_indices)):
        if frame_indices[i] - frame_indices[keep[-1]] >= min_gap:
            keep.append(i)
    return keep


def reduce_tracklet(track_dir: Path, stride: int, dry_run: bool) -> tuple[int, int]:
    meta_path = track_dir / "metadata.json"
    if not meta_path.exists():
        return 0, 0

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    crop_files = meta.get("crop_files", [])
    frame_indices = meta.get("frame_indices", [])
    bboxes = meta.get("bboxes", [])
    confidences = meta.get("confidences", [])

    if not crop_files or not frame_indices:
        return 0, 0

    # 트랙렛의 frame_stride로 실제 gap 계산
    # (stride=5, frame_stride=6 → 30 video frames = 1.25초 @ 24fps)
    frame_stride = meta.get("frame_stride", 6)
    min_gap = stride * frame_stride

    keep_idx = select_keep_indices(frame_indices, min_gap)
    keep_files = set(crop_files[i] for i in keep_idx)
    remove_files = [f for f in crop_files if f not in keep_files]

    if not dry_run:
        for fname in remove_files:
            p = track_dir / fname
            if p.exists():
                p.unlink()

        new_crops = [crop_files[i] for i in keep_idx]
        new_frames = [frame_indices[i] for i in keep_idx]
        new_bboxes = [bboxes[i] for i in keep_idx] if bboxes else []
        new_confs = [confidences[i] for i in keep_idx] if confidences else []

        meta["crop_files"] = new_crops
        meta["frame_indices"] = new_frames
        meta["bboxes"] = new_bboxes
        meta["confidences"] = new_confs
        meta["length"] = len(new_crops)
        meta["avg_conf"] = float(np.mean(new_confs)) if new_confs else 0.0
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return len(crop_files), len(remove_files)


def main():
    parser = argparse.ArgumentParser(description="트랙렛 프레임 간격 축소 (frame 번호 기준)")
    parser.add_argument("--tracklets-dir", default="data/tracklets-1-original-reduced")
    parser.add_argument("--stride", type=int, default=5,
                        help="트래킹 샘플링 stride의 배수 (기본: 5 → frame_stride×5 간격 유지)")
    parser.add_argument("--dry-run", action="store_true", help="실제 삭제 없이 결과만 출력")
    args = parser.parse_args()

    tracklet_dir = Path(args.tracklets_dir)
    if not tracklet_dir.is_absolute():
        tracklet_dir = project_root / tracklet_dir

    if not tracklet_dir.exists():
        sys.exit(f"[ERROR] 디렉토리 없음: {tracklet_dir}")

    track_dirs = sorted(tracklet_dir.glob("*/track_*"))
    if not track_dirs:
        sys.exit("[ERROR] 트랙렛 폴더를 찾을 수 없습니다.")

    # 첫 트랙렛에서 실제 gap 미리 출력
    sample_meta_path = track_dirs[0] / "metadata.json"
    if sample_meta_path.exists():
        sample_meta = json.loads(sample_meta_path.read_text())
        frame_stride = sample_meta.get("frame_stride", 6)
        src_fps = sample_meta.get("src_fps", 24.0)
        min_gap = args.stride * frame_stride
        interval_sec = min_gap / src_fps
        print(f"[INFO] frame_stride={frame_stride}, stride={args.stride} "
              f"→ min_gap={min_gap} frames = {interval_sec:.2f}초 간격")

    print(f"[INFO] 트랙렛 {len(track_dirs)}개  |  {'DRY-RUN' if args.dry_run else '실제 삭제'}")

    total_before, total_removed = 0, 0
    for track_dir in tqdm(track_dirs, desc="처리 중"):
        before, removed = reduce_tracklet(track_dir, args.stride, args.dry_run)
        total_before += before
        total_removed += removed

    print()
    print("=" * 45)
    print(f"  처리 전 : {total_before:,}장")
    print(f"  삭제    : {total_removed:,}장")
    print(f"  남은 것 : {total_before - total_removed:,}장")
    if args.dry_run:
        print("\n  ※ --dry-run: 실제 변경 없음")
    print("=" * 45)


if __name__ == "__main__":
    main()
