"""
pipeline/tracklet_splitter.py
==============================
하나의 tracklet 폴더 안에 두 사람의 crop이 섞인 경우를 감지하고
시간순 전환점(split point)을 찾아 두 개의 tracklet으로 분리한다.

알고리즘
  1. tracklet 내 모든 crop에서 Re-ID 특징 추출
  2. 가능한 모든 시간 분할점 i 에 대해
       - 앞 그룹(0..i)의 평균 특징 벡터
       - 뒷 그룹(i+1..n)의 평균 특징 벡터
     두 그룹 간 cosine distance 계산
  3. distance가 최대인 분할점을 전환점으로 선택
  4. 최대 distance > split_threshold 이면 분리 수행
     - metadata.json 복사 후 frame_indices / crop_files 갱신
     - 폴더 이름: {원본}_a, {원본}_b
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy.spatial.distance import cosine


def _load_feats(crops: list[Path], extractor) -> np.ndarray:
    """crop 파일 목록 → (N, D) 특징 행렬."""
    feats = []
    for p in crops:
        img = cv2.imread(str(p))
        if img is None:
            feats.append(None)
            continue
        f = extractor.extract_batch_features([img])[0]
        norm = np.linalg.norm(f)
        feats.append(f / (norm + 1e-8) if norm > 0 else f)
    valid = [f for f in feats if f is not None]
    return np.array(valid) if valid else np.zeros((0, 512))


def _find_split_point(feats: np.ndarray) -> tuple[int, float]:
    """
    시간순 분할점 탐색.
    반환: (best_i, max_dist)  — best_i 이후부터 두 번째 그룹.
    """
    n = len(feats)
    best_i, best_dist = 0, 0.0
    for i in range(1, n - 1):
        mean_a = feats[:i].mean(axis=0)
        mean_b = feats[i:].mean(axis=0)
        d = cosine(mean_a, mean_b)
        if d > best_dist:
            best_dist = d
            best_i = i
    return best_i, best_dist


def split_tracklet(
    tracklet_dir: Path,
    extractor,
    split_threshold: float = 0.30,
    min_crops_per_part: int = 3,
    dry_run: bool = False,
) -> Optional[tuple[Path, Path]]:
    """
    단일 tracklet을 검사하고 필요 시 분리.

    반환
      (path_a, path_b)  분리 성공
      None              분리 불필요 또는 실패
    """
    name = tracklet_dir.name
    meta_path = tracklet_dir / "metadata.json"
    if not meta_path.exists():
        print(f"[SPLIT] {name}  skip: metadata.json 없음")
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    crop_files: list[str] = meta.get("crop_files", [])
    frame_indices: list[int] = meta.get("frame_indices", [])

    if len(crop_files) < min_crops_per_part * 2:
        print(f"[SPLIT] {name}  skip: crop 수 부족 ({len(crop_files)} < {min_crops_per_part * 2})")
        return None

    crops = [tracklet_dir / c for c in crop_files]
    feats = _load_feats(crops, extractor)

    if len(feats) < min_crops_per_part * 2:
        print(f"[SPLIT] {name}  skip: 유효 특징 수 부족 ({len(feats)} < {min_crops_per_part * 2})")
        return None

    split_i, max_dist = _find_split_point(feats)

    if max_dist < split_threshold:
        print(f"[SPLIT] {name}  skip: max_dist={max_dist:.3f} < threshold={split_threshold}")
        return None

    # 각 파트 crop 수 확인
    if split_i < min_crops_per_part or (len(feats) - split_i) < min_crops_per_part:
        print(
            f"[SPLIT] {name}  skip: 분리 후 파트 크기 미달"
            f"  (앞={split_i}, 뒤={len(feats) - split_i}, 최소={min_crops_per_part})"
        )
        return None

    print(
        f"[SPLIT] {tracklet_dir.name}  max_dist={max_dist:.3f} > {split_threshold}"
        f"  → split at crop idx {split_i}/{len(feats)}"
    )

    if dry_run:
        return None

    # ── 분리 실행 ──────────────────────────────────────────────────
    parent = tracklet_dir.parent
    dir_a = parent / f"{tracklet_dir.name}_a"
    dir_b = parent / f"{tracklet_dir.name}_b"

    for d in (dir_a, dir_b):
        d.mkdir(exist_ok=True)

    crops_a = crop_files[:split_i]
    crops_b = crop_files[split_i:]
    frames_a = frame_indices[:split_i] if frame_indices else []
    frames_b = frame_indices[split_i:] if frame_indices else []

    def _write_part(dst: Path, crops: list[str], frames: list[int]):
        for c in crops:
            src = tracklet_dir / c
            if src.exists():
                shutil.copy2(src, dst / c)
        part_meta = {**meta, "crop_files": crops, "frame_indices": frames}
        (dst / "metadata.json").write_text(
            json.dumps(part_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    _write_part(dir_a, crops_a, frames_a)
    _write_part(dir_b, crops_b, frames_b)

    # 원본 제거
    shutil.rmtree(tracklet_dir)

    return dir_a, dir_b


def split_all(
    filtered_root: str | Path,
    extractor,
    split_threshold: float = 0.30,
    min_crops_per_part: int = 3,
    dry_run: bool = False,
) -> dict:
    """
    filtered_root 하위 모든 tracklet을 순회하며 분리 처리.

    반환: {"total": int, "split": int, "skipped": int}
    """
    root = Path(filtered_root)
    total = split = skipped = 0

    for slot_dir in sorted(root.iterdir()):
        if not slot_dir.is_dir():
            continue
        for track_dir in sorted(slot_dir.iterdir()):
            if not track_dir.is_dir():
                continue
            total += 1
            result = split_tracklet(
                track_dir, extractor, split_threshold, min_crops_per_part, dry_run
            )
            if result:
                split += 1
            else:
                skipped += 1

    return {"total": total, "split": split, "skipped": skipped}
