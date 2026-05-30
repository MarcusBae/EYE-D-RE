"""
video_utils.py
==============
영상 I/O, Google Drive 마운트, 프레임 추출 유틸리티.

CCTV 영상(720p+, 25~30fps, AVI/MOV/MKV/MP4)을 5fps 샘플링하여
`data/frames/{cam}_{slot}/` 디렉토리에 jpg로 저장합니다.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import yaml
from tqdm import tqdm


# -------------------------------------------------------------
# Colab / Google Drive
# -------------------------------------------------------------
def is_colab() -> bool:
    """현재 환경이 Google Colab인지 확인."""
    try:
        import google.colab # noqa: F401
        return True
    except ImportError:
        return False


def mount_gdrive(mount_point: str = "/content/drive") -> Optional[str]:
    """
    Colab 환경에서 Google Drive를 마운트합니다.
    이미 마운트되어 있다면 그대로 사용합니다.

    Returns
    -------
    str | None : 마운트된 경로 (Colab이 아니면 None)
    """
    if not is_colab():
        print("[INFO] Colab 환경이 아닙니다. mount_gdrive() 무시.")
        return None

    if os.path.ismount(mount_point):
        print(f"[INFO] 이미 마운트됨: {mount_point}")
        return mount_point

    from google.colab import drive # type: ignore
    drive.mount(mount_point)
    print(f"[INFO] Google Drive 마운트 완료: {mount_point}")
    return mount_point


# -------------------------------------------------------------
# Config Loader
# -------------------------------------------------------------
def load_config(config_path: str = "configs/config.yaml") -> dict:
    """YAML config 파일을 dict로 로드."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config 파일을 찾을 수 없습니다: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -------------------------------------------------------------
# Video Info
# -------------------------------------------------------------
def get_video_info(video_path: str) -> Dict:
    """
    영상 파일의 메타정보 추출 (FPS, 총 프레임, 해상도).
    AVI/MOV/MKV/MP4 등 OpenCV가 지원하는 모든 포맷에서 동작.

    OpenCV가 일부 코덱(예: HEVC)을 읽지 못할 경우:
    pip install imageio-ffmpeg 후 ffmpeg 백엔드를 사용하도록 변경 권장.
    """
    video_path = str(video_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"영상 파일이 없습니다: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(
            f"OpenCV가 영상을 열지 못했습니다: {video_path}\n"
            "→ 코덱 문제일 수 있습니다. ffmpeg로 mp4(H.264)로 변환 후 재시도해보세요:\n"
            f" ffmpeg -i '{video_path}' -c:v libx264 -crf 23 '{video_path}.mp4'"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total / fps if fps > 0 else 0.0
    cap.release()

    return {
        "video_path": video_path,
        "fps": round(float(fps), 3),
        "total_frames": total,
        "width": width,
        "height": height,
        "duration_sec": round(duration, 2),
    }


# -------------------------------------------------------------
# Frame Extraction
# -------------------------------------------------------------
def extract_frames(
    video_path: str,
    output_dir: str,
    target_fps: float = 5.0,
    camera_id: int = 0,
    time_slot: int = 0,
    image_format: str = "jpg",
    jpg_quality: int = 95,
    overwrite: bool = False,
) -> Dict:
    """
    영상에서 target_fps로 프레임을 샘플링하여 저장합니다.

    파일명 규칙: `c{cam}_t{slot}_f{frame_idx:06d}.{ext}`
    - frame_idx는 원본 영상의 frame index (sampling 이전 기준)

    Parameters
    ----------
    video_path : str
        영상 파일 경로
    output_dir : str
        프레임 저장 디렉토리 (자동 생성)
    target_fps : float
        샘플링 FPS (기본 5)
    camera_id : int
        카메라 ID (1, 2, 3)
    time_slot : int
        시간 슬롯 (1, 2)
    image_format : str
        "jpg" | "png"
    jpg_quality : int
        JPEG 압축 품질 (1-100)
    overwrite : bool
        True면 기존 출력 폴더의 파일을 덮어씀

    Returns
    -------
    dict : 추출 통계 + 메타데이터 (output_dir/meta.json 에도 저장됨)
    """
    video_path = str(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = get_video_info(video_path)
    src_fps = info["fps"]
    total_frames = info["total_frames"]

    if src_fps <= 0:
        raise RuntimeError(f"원본 FPS를 알 수 없습니다: {video_path}")
    
    # 샘플링 step: 원본 1프레임 단위로 진행하며 target_fps에 맞춰 sampling
    # step=src_fps / target_fps (실수 단위 누적 방식 사용)
    if target_fps <= 0 or target_fps >= src_fps:
        step = 1.0  # 원본 fps 이하라면 매 프레임 사용
    else:
        step = src_fps / target_fps

    # 이미 결과가 있고 overwrite가 False면 스킵
    existing = list(output_dir.glob(f"c{camera_id}_t{time_slot}_f*.{image_format}"))
    if existing and not overwrite:
        print(f"[SKIP] 이미 {len(existing)}장 추출됨: {output_dir} (overwrite=False)")
        meta_path = output_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    encode_params = []
    if image_format.lower() in ("jpg", "jpeg"):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]
    elif image_format.lower() == "png":
        encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]

    saved_count = 0
    next_target = 0.0
    frame_idx = 0
    saved_frames: List[int] = []

    pbar = tqdm(total=total_frames, desc=f" extract c{camera_id}_t{time_slot}", leave=False)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= next_target:
            fname = f"c{camera_id}_t{time_slot}_f{frame_idx:06d}.{image_format}"
            out_path = output_dir / fname
            cv2.imwrite(str(out_path), frame, encode_params)
            saved_count += 1
            saved_frames.append(frame_idx)
            next_target += step

        frame_idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()

    meta = {
        "video_path": video_path,
        "camera_id": camera_id,
        "time_slot": time_slot,
        "src_fps": src_fps,
        "target_fps": float(target_fps),
        "src_total_frames": total_frames,
        "extracted_frames": saved_count,
        "step": step,
        "width": info["width"],
        "height": info["height"],
        "duration_sec": info["duration_sec"],
        "image_format": image_format,
        "first_frame_idx": saved_frames[0] if saved_frames else None,
        "last_frame_idx": saved_frames[-1] if saved_frames else None,
    }

    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return meta


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def load_videos_from_config(config: dict, video_dir: Optional[str] = None) -> List[Dict]:
    """
    config.yaml의 `videos` 매핑을 사용하여 영상 목록을 만듭니다.

    Returns
    -------
    List[dict] : [{"path": ..., "camera": 1, "slot": 1, "filename": ...}, ...]
    """
    if video_dir is None:
        video_dir = config["data"]["video_dir"]
    video_dir = Path(video_dir)

    items = []
    for fname, meta in config["videos"].items():
        path = video_dir / fname
        items.append({
            "path": str(path),
            "filename": fname,
            "camera": int(meta["camera"]),
            "slot": int(meta["slot"]),
            "exists": path.exists(),
        })
    return items


def print_video_summary(config: dict) -> None:
    """영상 목록 + 존재 여부 + 메타정보 요약 출력."""
    items = load_videos_from_config(config)
    print(f"\n{'='*78}")
    print(f"{'파일명':<20} {'cam':>4} {'slot':>5} {'존재':>5} {'FPS':>7} {'프레임':>8} {'해상도':>12} {'길이(s)':>9}")
    print("-" * 78)
    for it in items:
        if it["exists"]:
            try:
                info = get_video_info(it["path"])
                print(f"{it['filename']:<20} {it['camera']:>4} {it['slot']:>5} {'OK':>5} "
                      f"{info['fps']:>7.2f} {info['total_frames']:>8} "
                      f"{info['width']}x{info['height']:>5} {info['duration_sec']:>9.1f}")
            except Exception as e:
                print(f"{it['filename']:<20} {it['camera']:>4} {it['slot']:>5} {'ERR':>5} "
                      f"{str(e)[:40]}")
        else:
            print(f"{it['filename']:<20} {it['camera']:>4} {it['slot']:>5} {'MISS':>5} "
                  f"(파일 없음)")
    print("=" * 78)
