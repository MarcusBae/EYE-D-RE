"""
tracker.py
==========
BoT-SORT 래퍼 (Ultralytics 내장).

Ultralytics의 ``YOLO.track()`` 메서드는 ByteTrack/BoT-SORT를 내장하고 있어
별도 의존성 없이 동작합니다.
BoT-SORT는 IoU 기반 motion 매칭에 Re-ID 외형 특징을 추가하여
두 사람이 겹칠 때 발생하는 ID switch를 억제합니다.
영상 한 개에 대해 tracking 결과를 frame별로 모아 tracklet 단위로 정리합니다.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .detector import auto_device
from .tracklet_io import save_tracklet


class PersonTracker:
    """
    Ultralytics YOLO + BoT-SORT를 이용한 person tracking.

    영상 한 개를 입력받아:
    1. frame 단위로 검출 + 트래킹 수행
    2. (track_id, frame_idx, bbox, conf) 튜플 누적
    3. tracklet 단위로 묶어서 ``tracklet_io.save_tracklet()`` 으로 저장

    Notes
    -----
    - 트래킹은 sampling 된 frame(예: 5fps 추출본)이 아니라
      **원본 영상 자체**에 적용합니다. (Ultralytics가 자체 frame buffer 관리)
      필요시 `frame_stride` 인자로 매 N프레임만 사용하도록 조정 가능.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        tracker_yaml: str = "configs/botsort.yaml",
        conf: float = 0.5,
        iou: float = 0.45,
        imgsz: int = 640,
        classes: Optional[List[int]] = None,
        device: str = "auto",
        verbose: bool = False,
    ):
        from ultralytics import YOLO

        self.device = auto_device(device)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.classes = classes if classes is not None else [0]
        self.tracker_yaml = tracker_yaml
        self.verbose = verbose

        print(f"[Tracker] 모델 로드: {model_path} (device={self.device}, tracker={tracker_yaml})")
        self.model = YOLO(model_path)

    # -----------------------------------------------------
    def track_video(
        self,
        video_path: str,
        camera_id: int,
        time_slot: int,
        output_dir: str,
        frame_stride: int = 1,
        save_crops: bool = True,
    ) -> Dict:
        """
        영상 1개에 대해 트래킹 → tracklet 단위 저장.

        Parameters
        ----------
        video_path : 영상 파일 경로
        camera_id : 카메라 ID
        time_slot : 시간 슬롯
        output_dir : tracklet 저장 디렉토리
                     (`output_dir/c{cam}_t{slot}/track_{id:04d}/` 구조로 저장됨)
        frame_stride : 매 N프레임만 트래킹 (5fps 등가 효과). 1=모든 프레임.
        save_crops : tracklet 각 frame의 person crop 이미지 저장 여부

        Returns
        -------
        dict : 트래킹 통계
        """
        video_path = str(video_path)
        output_dir = Path(output_dir)
        out_video_dir = output_dir / f"c{camera_id}_t{time_slot}"
        out_video_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS)

        # track_id → list of dict(frame_idx, bbox, conf, crop)
        tracklets: Dict[int, List[dict]] = defaultdict(list)

        frame_idx = 0
        processed = 0
        pbar = tqdm(total=total_frames, desc=f"  track c{camera_id}_t{time_slot}", leave=False)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                # Ultralytics tracking (persist=True로 ID 유지)
                results = self.model.track(
                    frame,
                    conf=self.conf,
                    iou=self.iou,
                    imgsz=self.imgsz,
                    classes=self.classes,
                    device=self.device,
                    persist=True,
                    tracker=self.tracker_yaml,
                    verbose=self.verbose,
                )
                r = results[0]

                if r.boxes is not None and r.boxes.id is not None:
                    ids = r.boxes.id.cpu().numpy().astype(int)
                    bboxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                    confs = r.boxes.conf.cpu().numpy().astype(np.float32)

                    for tid, bbox, conf in zip(ids, bboxes, confs):
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        # bbox clamping
                        h, w = frame.shape[:2]
                        x1c, y1c = max(0, x1), max(0, y1)
                        x2c, y2c = min(w, x2), min(h, y2)
                        if x2c <= x1c or y2c <= y1c:
                            continue
                        
                        entry = {
                            "frame_idx": int(frame_idx),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "conf": float(conf),
                        }
                        if save_crops:
                            entry["_crop"] = frame[y1c:y2c, x1c:x2c].copy()
                        
                        tracklets[int(tid)].append(entry)
                processed += 1
            frame_idx += 1
            pbar.update(1)
        pbar.close()
        cap.release()

        # tracklet 단위 저장
        saved_tracks = 0
        for tid, entries in tracklets.items():
            if len(entries) == 0:
                continue
            
            crops = [e.pop("_crop") for e in entries] if save_crops else None
            metadata = {
                "track_id": int(tid),
                "camera_id": int(camera_id),
                "time_slot": int(time_slot),
                "src_video": video_path,
                "src_fps": float(src_fps),
                "frame_stride": int(frame_stride),
                "length": len(entries),
                "frame_indices": [e["frame_idx"] for e in entries],
                "bboxes": [e["bbox"] for e in entries],
                "confidences": [e["conf"] for e in entries],
                "avg_conf": float(np.mean([e["conf"] for e in entries])),
            }
            save_tracklet(tid, crops, metadata, out_video_dir)
            saved_tracks += 1

        stats = {
            "video_path": video_path,
            "camera_id": camera_id,
            "time_slot": time_slot,
            "total_frames": total_frames,
            "processed_frames": processed,
            "frame_stride": frame_stride,
            "num_tracklets": saved_tracks,
            "output_dir": str(out_video_dir),
        }

        # video 단위 stats 저장
        with open(out_video_dir / "tracking_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        return stats
