"""
quality_filter.py
=================
Tracklet 보수적 품질 필터링 + 시각적 검증용 thumbnail grid 생성.

수동 검증이 불가능한 환경에서는 false positive(잘못된 tracklet)를
강하게 걸러내는 것이 cross-camera 라벨링 품질에 결정적입니다.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .tracklet_io import list_tracklets


# -------------------------------------------------------------
class TrackletQualityFilter:
    """
    Tracklet 품질 필터.

    필터 기준 (모두 AND 조건):
    - length >= min_length
    - avg_conf >= min_avg_conf
    - 평균 bbox height >= min_bbox_h
    - 평균 bbox width >= min_bbox_w
    - 평균 aspect ratio (h/w) <= max_aspect_ratio
    - 평균 bbox area >= min_area
    """

    def __init__(
        self,
        min_length: int = 30,
        min_avg_conf: float = 0.7,
        min_bbox_h: int = 80,
        min_bbox_w: int = 40,
        max_aspect_ratio: float = 4.0,
        min_area: float = 5000.0,
    ):
        self.min_length = int(min_length)
        self.min_avg_conf = float(min_avg_conf)
        self.min_bbox_h = int(min_bbox_h)
        self.min_bbox_w = int(min_bbox_w)
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.min_area = float(min_area)

    # -----------------------------------------------------
    def evaluate(self, meta: Dict) -> Tuple[bool, List[str]]:
        """
        하나의 tracklet metadata에 대해 통과 여부와 탈락 사유를 반환.

        Returns
        -------
        (passed, reasons) : (bool, list of failure reasons — passed=True면 빈 리스트)
        """
        reasons: List[str] = []

        length = int(meta.get("length", 0))
        if length < self.min_length:
            reasons.append(f"length<{self.min_length}({length})")
            
        avg_conf = float(meta.get("avg_conf", 0.0))
        if avg_conf < self.min_avg_conf:
            reasons.append(f"avg_conf<{self.min_avg_conf:.2f}({avg_conf:.2f})")
            
        bboxes = np.array(meta.get("bboxes", []), dtype=np.float32)
        if len(bboxes) == 0:
            reasons.append("no_bboxes")
        else:
            widths = bboxes[:, 2] - bboxes[:, 0]
            heights = bboxes[:, 3] - bboxes[:, 1]
            mean_w = float(np.mean(widths))
            mean_h = float(np.mean(heights))
            mean_area = float(np.mean(widths * heights))
            
            # aspect ratio: h/w (사람은 보통 2~3)
            valid = widths > 0
            if valid.any():
                ar = (heights[valid] / widths[valid]).mean()
            else:
                ar = 0.0

            if mean_h < self.min_bbox_h:
                reasons.append(f"bbox_h<{self.min_bbox_h}({mean_h:.0f})")
            if mean_w < self.min_bbox_w:
                reasons.append(f"bbox_w<{self.min_bbox_w}({mean_w:.0f})")
            if ar > self.max_aspect_ratio:
                reasons.append(f"aspect>{self.max_aspect_ratio:.1f}({ar:.1f})")
            if mean_area < self.min_area:
                reasons.append(f"area<{self.min_area:.0f}({mean_area:.0f})")
                
        return (len(reasons) == 0, reasons)

    # -----------------------------------------------------
    def filter_all(
        self,
        tracklet_dir: str,
        output_dir: str,
        copy_crops: bool = True,
        verbose: bool = True,
    ) -> Dict:
        """
        tracklet_dir 아래 모든 tracklet에 필터 적용 → output_dir로 복사.

        Returns
        -------
        dict : 필터링 통계 (전체/통과/탈락, 탈락 사유 카운트)
        """
        tracklet_dir = Path(tracklet_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_tracks = list_tracklets(str(tracklet_dir))
        passed_meta = []
        failed_reasons: Dict[str, int] = {}

        for meta in tqdm(all_tracks, desc="quality filter", disable=not verbose):
            passed, reasons = self.evaluate(meta)
            if passed:
                passed_meta.append(meta)
                if copy_crops:
                    self._copy_tracklet(meta, output_dir)
            else:
                for r in reasons:
                    key = r.split("(")[0]
                    failed_reasons[key] = failed_reasons.get(key, 0) + 1

        stats = {
            "total": len(all_tracks),
            "passed": len(passed_meta),
            "failed": len(all_tracks) - len(passed_meta),
            "pass_rate": (len(passed_meta) / len(all_tracks)) if all_tracks else 0.0,
            "failure_reasons": dict(sorted(failed_reasons.items(), key=lambda x: -x[1])),
            "filter_params": {
                "min_length": self.min_length,
                "min_avg_conf": self.min_avg_conf,
                "min_bbox_h": self.min_bbox_h,
                "min_bbox_w": self.min_bbox_w,
                "max_aspect_ratio": self.max_aspect_ratio,
                "min_area": self.min_area,
            },
        }

        with open(output_dir / "filter_stats.json", "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"\n[Filter] 전체 {stats['total']} → 통과 {stats['passed']} ({stats['pass_rate']*100:.1f}%)")
            print("[Filter] 탈락 사유 분포:")
            for k, v in stats["failure_reasons"].items():
                print(f" - {k:<20} : {v}")
        return stats

    # -----------------------------------------------------
    @staticmethod
    def _copy_tracklet(meta: Dict, output_dir: Path) -> None:
        """통과한 tracklet을 그대로 output_dir에 복사."""
        src = Path(meta["tracklet_dir"])
        rel = src.relative_to(src.parent.parent) # c{cam}_t{slot}/track_xxxx
        dst = output_dir / rel
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            shutil.copy2(item, dst / item.name)


# -------------------------------------------------------------
# Thumbnail Grid (시각적 검증 보조)
# -------------------------------------------------------------
def make_thumbnail_grid(
    tracklets: List[Dict],
    output_path: str,
    n_per_track: int = 4,
    thumb_size: Tuple[int, int] = (128, 256),
    max_tracks: int = 80,
    title: str = "Tracklet Thumbnails",
) -> str:
    """
    각 tracklet에서 균등 간격 N장씩 샘플링 → grid 이미지 저장.

    Parameters
    ----------
    tracklets : list of metadata dict (tracklet_dir 포함)
    output_path : 출력 이미지 경로 (.jpg or .png)
    n_per_track : 각 tracklet당 thumbnail 수
    thumb_size : (width, height) per thumbnail
    max_tracks : 한 grid에 표시할 최대 tracklet 수
    title : 그림 제목

    Returns
    -------
    str : output_path
    """
    import matplotlib.pyplot as plt

    tracklets = tracklets[:max_tracks]
    n_tracks = len(tracklets)
    if n_tracks == 0:
        print("[WARN] tracklet이 없습니다.")
        return output_path

    tw, th = thumb_size
    fig, axes = plt.subplots(
        n_tracks, n_per_track,
        figsize=(n_per_track * 1.6, n_tracks * 1.6),
        squeeze=False,
    )

    for row, meta in enumerate(tracklets):
        tdir = Path(meta["tracklet_dir"])
        crop_files = sorted(tdir.glob("frame_*.jpg"))
        if not crop_files:
            for col in range(n_per_track):
                axes[row, col].axis("off")
            continue

        # 균등 간격 샘플링
        if len(crop_files) <= n_per_track:
            sampled = crop_files
        else:
            idxs = np.linspace(0, len(crop_files) - 1, n_per_track).astype(int)
            sampled = [crop_files[i] for i in idxs]
            
        for col in range(n_per_track):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(sampled):
                img = cv2.imread(str(sampled[col]))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (tw, th))
                    ax.imshow(img)
                    if col == 0:
                        cam = meta.get("camera_id", "?")
                        slot = meta.get("time_slot", "?")
                        tid = meta.get("track_id", "?")
                        length = meta.get("length", "?")
                        ax.set_title(f"c{cam}_t{slot}\nid={tid} L={length}", fontsize=7, loc="left")
                        
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[Thumbnail] 저장: {output_path} ({n_tracks} tracklets × {n_per_track})")
    return output_path
