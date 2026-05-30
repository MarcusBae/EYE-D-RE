"""
tracklet_io.py
==============
Tracklet 저장 / 로드 / 탐색 유틸리티.

디렉토리 구조:
    tracklets/
      c{cam}_t{slot}/
        tracking_stats.json
        track_{id:04d}/
          metadata.json
          frame_{frame_idx:06d}.jpg   # crop 이미지들
          ...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


def save_tracklet(
    tracklet_id: int,
    crops: Optional[List[np.ndarray]],
    metadata: dict,
    output_dir: str,
    jpg_quality: int = 92,
) -> Path:
    """
    하나의 tracklet을 디렉토리에 저장.

    Parameters
    ----------
    tracklet_id : int
        Track ID (영상 내 ByteTrack ID)
    crops : list of ndarray or None
        각 frame의 person crop (BGR). metadata['frame_indices'] 와 길이 동일해야 함.
        None이면 crop은 저장하지 않고 metadata만 기록.
    metadata : dict
        Tracklet 메타데이터 (length, frame_indices, bboxes, confidences, ...)
    output_dir : str
        저장 루트 (예: data/tracklets/c1_t1)
    jpg_quality : int

    Returns
    -------
    Path : tracklet 디렉토리 경로
    """
    output_dir = Path(output_dir)
    tdir = output_dir / f"track_{int(tracklet_id):04d}"
    tdir.mkdir(parents=True, exist_ok=True)

    # crop 저장
    if crops is not None:
        if len(crops) != len(metadata.get("frame_indices", [])):
            raise ValueError(
                f"crops 개수({len(crops)})와 frame_indices "
                f"({len(metadata.get('frame_indices', []))}) 불일치"
            )
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]
        crop_files = []
        for fidx, crop in zip(metadata["frame_indices"], crops):
            if crop is None or crop.size == 0:
                continue
            fname = f"frame_{int(fidx):06d}.jpg"
            cv2.imwrite(str(tdir / fname), crop, encode_params)
            crop_files.append(fname)
        metadata["crop_files"] = crop_files

    # metadata 저장 (numpy 타입 제거)
    md = _to_json_safe(metadata)
    with open(tdir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(md, f, indent=2, ensure_ascii=False)

    return tdir


def _to_json_safe(obj):
    """numpy 타입을 python native로 변환."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def load_tracklet(tracklet_dir: str, load_crops: bool = False) -> Dict:
    """
    tracklet 디렉토리에서 metadata 및 (선택적으로) crop 이미지를 로드.

    Returns
    -------
    dict : metadata + 'crops'(list of ndarray) 가 포함될 수 있음
    """
    tracklet_dir = Path(tracklet_dir)
    meta_path = tracklet_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json 없음: {tracklet_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["tracklet_dir"] = str(tracklet_dir)

    if load_crops:
        crop_files = meta.get("crop_files", [])
        crops = []
        for fname in crop_files:
            img = cv2.imread(str(tracklet_dir / fname))
            crops.append(img)
        meta["crops"] = crops

    return meta


def list_tracklets(base_dir: str) -> List[Dict]:
    """
    base_dir 아래 모든 tracklet을 찾아 metadata 리스트로 반환.

    base_dir 구조 예:
        data/tracklets/
          c1_t1/ track_0001/ ...
          c1_t2/ track_0001/ ...
          ...
    """
    base_dir = Path(base_dir)
    results = []
    if not base_dir.exists():
        return results

    for video_dir in sorted(base_dir.iterdir()):
        if not video_dir.is_dir():
            continue
        for track_dir in sorted(video_dir.glob("track_*")):
            if not track_dir.is_dir():
                continue
            meta_path = track_dir / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["tracklet_dir"] = str(track_dir)
                meta["video_dir"] = str(video_dir)
                results.append(meta)
            except Exception as e:
                print(f"[WARN] {meta_path} 로드 실패: {e}")
    return results


def get_tracklet_stats(tracklets: List[Dict]) -> Dict:
    """전체 tracklet 리스트의 요약 통계."""
    if not tracklets:
        return {"count": 0}

    lengths = [t["length"] for t in tracklets]
    confs = [t.get("avg_conf", 0.0) for t in tracklets]

    # 카메라/슬롯별 집계
    by_cam_slot: Dict[str, int] = {}
    for t in tracklets:
        key = f"c{t['camera_id']}_t{t['time_slot']}"
        by_cam_slot[key] = by_cam_slot.get(key, 0) + 1

    return {
        "count": len(tracklets),
        "length_min": int(min(lengths)),
        "length_max": int(max(lengths)),
        "length_mean": float(np.mean(lengths)),
        "length_median": float(np.median(lengths)),
        "avg_conf_min": float(min(confs)),
        "avg_conf_max": float(max(confs)),
        "avg_conf_mean": float(np.mean(confs)),
        "by_cam_slot": by_cam_slot,
    }
