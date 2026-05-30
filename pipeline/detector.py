"""
detector.py
===========
YOLOv8n person detector (Ultralytics).

CCTV 영상의 프레임 또는 영상 전체에서 사람(class 0)을 검출합니다.
이 모듈은 단일 이미지/배치 검출을 담당하며,
트래킹은 `tracker.py`에서 model.track()을 사용합니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def auto_device(device: str = "auto") -> str:
    """device='auto'일 때 cuda 가능하면 'cuda', 아니면 'cpu' 반환."""
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device


class PersonDetector:
    """
    YOLOv8 기반 Person Detector.

    Examples
    --------
    >>> det = PersonDetector("yolov8n.pt", conf=0.5)
    >>> bboxes, confs, cls = det.detect(img_bgr)
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
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
        self.classes = classes if classes is not None else [0] # person only
        self.verbose = verbose

        print(f"[Detector] 모델 로드: {model_path} (device={self.device})")
        self.model = YOLO(model_path)
        self.model_path = model_path

    # -----------------------------------------------------
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        단일 이미지(BGR ndarray)에서 사람 검출.

        Returns
        -------
        bboxes : (N, 4) xyxy float32
        confs : (N,) float32
        cls : (N,) int32
        """
        results = self.model.predict(
            image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            verbose=self.verbose,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32))

        bboxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = r.boxes.conf.cpu().numpy().astype(np.float32)
        cls = r.boxes.cls.cpu().numpy().astype(np.int32)
        return bboxes, confs, cls

    # -----------------------------------------------------
    def detect_batch(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """이미지 리스트를 배치로 검출."""
        results = self.model.predict(
            images,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            verbose=self.verbose,
        )
        outputs = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                outputs.append((np.zeros((0, 4), dtype=np.float32),
                                np.zeros((0,), dtype=np.float32),
                                np.zeros((0,), dtype=np.int32)))
                continue
            bboxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            confs = r.boxes.conf.cpu().numpy().astype(np.float32)
            cls = r.boxes.cls.cpu().numpy().astype(np.int32)
            outputs.append((bboxes, confs, cls))
        return outputs

    # -----------------------------------------------------
    def visualize(self, image: np.ndarray, bboxes: np.ndarray, confs: np.ndarray) -> np.ndarray:
        """검출 결과를 시각화한 BGR 이미지 반환."""
        import cv2
        img = image.copy()
        for (x1, y1, x2, y2), c in zip(bboxes.astype(int), confs):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"person {c:.2f}"
            cv2.putText(img, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return img
