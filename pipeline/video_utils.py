"""
video_utils.py
==============
영상 I/O, Google Drive 마운트, 프레임 추출 유틸리티.

CCTV 영상(720p+, 25~30fps, AVI/MOV/MKV/MP4)을 5fps 샘플링하여
`data/frames/{cam}_{slot}/` 디렉토리에 jpg로 저장합니다.

주요 추가 기능
--------------
iter_annotated_frames()
    영상 파일을 프레임 단위로 읽으면서 YOLOv8 탐지 + BoT-SORT 추적 +
    OSNet Re-ID 를 온라인으로 적용하고 어노테이션된 BGR 프레임을 yield 합니다.
    Streamlit 실시간 재생 등에서 직접 사용할 수 있도록 제너레이터 형태로 설계됐습니다.
"""

from __future__ import annotations

import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm


# ── 색상 팔레트 (BGR, Re-ID global_id 에 매핑) ────────────────────────
REID_PALETTE_BGR: list[tuple[int, int, int]] = [
    ( 56,  56, 255), (151, 157, 255), (131, 212, 255), ( 23, 221, 100),
    (184, 186,   0), (193, 164,  17), (255,  89,  44), (232, 120, 164),
    (200, 120, 255), (  0, 165, 255), (170, 205, 102), (250, 206, 135),
    ( 56, 255,  56), (255, 200,  56), ( 56, 200, 255), (255,  56, 200),
]


def _reid_color(gid: int) -> tuple[int, int, int]:
    return REID_PALETTE_BGR[abs(int(gid)) % len(REID_PALETTE_BGR)]


def _draw_box(
    frame: np.ndarray,
    bbox: list[int],
    gid: int,
    track_id: int,
    conf: float,
    reid_ready: bool,
) -> None:
    """bbox + 레이블을 프레임에 직접 그린다 (in-place)."""
    x1, y1, x2, y2 = bbox
    col = _reid_color(gid) if reid_ready else (160, 160, 160)
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)

    if reid_ready:
        label = f"ID:{gid}  {conf:.2f}"
    else:
        label = f"T:{track_id}  {conf:.2f}"

    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    lx, ly = x1, max(y1 - th - bl - 4, 0)
    cv2.rectangle(frame, (lx, ly), (lx + tw + 4, ly + th + bl + 4), col, -1)
    cv2.putText(
        frame, label, (lx + 2, ly + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )


# ── 온라인 Re-ID 상태 ─────────────────────────────────────────────────
class _OnlineReID:
    """
    단일 영상 스트림 내 온라인 인물 재식별 상태 관리.

    동작 방식
    ---------
    1. 각 track_id 별로 크롭 이미지를 최대 ``buf_size`` 장 유지.
    2. 누적 프레임 수가 ``min_frames`` 에 도달할 때마다 OSNet 특징 추출 후
       기존 global_id 갤러리와 cosine 유사도 비교.
    3. 유사도 > ``sim_thresh`` 이면 동일인 → 기존 ID 배정 + 갤러리 피처 이동평균 갱신.
    4. 유사도 ≤ ``sim_thresh`` 이면 신규 인물 → 새 ID 부여 + 갤러리 추가.
    5. 이후 동일 track_id 는 이미 확정된 global_id 를 재사용.
       새로운 갤러리 인물이 등록되면 미확정 트랙들도 재평가.
    """

    def __init__(
        self,
        extractor,
        sim_thresh: float = 0.75,
        min_frames: int = 6,
        buf_size: int = 24,
        sample_n: int = 8,
        reeval_every: int = 30,
        init_gallery: Optional[Dict[int, np.ndarray]] = None,
        # ── 매칭 전략 ──────────────────────────────────────────────
        distance_metric: str = "cosine",     # "cosine" | "l2"
        match_strategy: str = "nearest",     # "nearest" | "mutual_topk" | "mean"
        mutual_topk_k: int = 3,
        mean_k: int = 5,                     # AQE 에서 평균낼 top-k 수
        # ── k-reciprocal Re-ranking (Zhong et al. 2017) ────────────
        use_rerank: bool = False,
        rerank_k1: int = 20,
        rerank_k2: int = 6,
        rerank_lambda: float = 0.3,
        rerank_min_gallery: int = 5,
    ):
        self.extractor        = extractor
        self.sim_thresh       = sim_thresh
        self.min_frames       = min_frames
        self.buf_size         = buf_size
        self.sample_n         = sample_n
        self.reeval_every     = reeval_every
        self.distance_metric  = distance_metric
        self.match_strategy   = match_strategy
        self.mutual_topk_k    = mutual_topk_k
        self.mean_k           = mean_k
        self.use_rerank       = use_rerank
        self.rerank_k1        = rerank_k1
        self.rerank_k2        = rerank_k2
        self.rerank_lambda    = rerank_lambda
        self.rerank_min_gallery = rerank_min_gallery

        # track_id → {"crops": deque, "count": int, "gid": int|None, "last_eval": int}
        self._trk: dict[int, dict] = {}
        # global_id → 갤러리 피처 (L2 정규화된 512-dim)
        self._gallery: dict[int, np.ndarray] = {}
        # init_gallery에서 주입된 person IDs (demo_data 등)
        self._init_gallery_ids: set[int] = set()
        self._next_gid = 1

        if init_gallery:
            for pid, feat in init_gallery.items():
                self._gallery[pid] = feat.astype(np.float32).copy()
                self._init_gallery_ids.add(pid)
            self._next_gid = max(init_gallery.keys()) + 1

    # ── 공개 API ─────────────────────────────────────────────────────

    def update(self, frame_idx: int, detections: list[tuple[int, list[int], float]],
               frame: np.ndarray, W: int, H: int) -> dict[int, int]:
        """
        Parameters
        ----------
        frame_idx   : 현재 프레임 인덱스
        detections  : [(track_id, [x1,y1,x2,y2], conf), ...]
        frame       : BGR 이미지 (크롭 추출용)
        W, H        : 영상 너비·높이

        Returns
        -------
        dict[track_id, global_id]  — 현재 프레임의 track_id → global_id 매핑
        """
        # 현재 프레임에 없는 track_id 는 stale 상태로 유지
        active_tids = {d[0] for d in detections}

        for tid, bbox, conf in detections:
            cx1 = max(0, bbox[0]); cy1 = max(0, bbox[1])
            cx2 = min(W, bbox[2]); cy2 = min(H, bbox[3])
            if cx2 > cx1 and cy2 > cy1:
                crop = frame[cy1:cy2, cx1:cx2].copy()
            else:
                crop = None

            if tid not in self._trk:
                self._trk[tid] = {
                    "crops": deque(maxlen=self.buf_size),
                    "count": 0,
                    "gid": None,
                    "last_eval": -1,
                }
            t = self._trk[tid]
            if crop is not None:
                t["crops"].append(crop)
            t["count"] += 1

            # Re-ID 실행 조건: 충분한 프레임 + (미확정이거나 주기 도달)
            need_eval = (
                t["count"] >= self.min_frames
                and (
                    t["gid"] is None
                    or (t["count"] - t["last_eval"]) >= self.reeval_every
                )
            )
            if need_eval and len(t["crops"]) > 0:
                feat = self._extract_feat(list(t["crops"]))
                gid  = self._match_or_create(feat)
                t["gid"]      = gid
                t["last_eval"] = t["count"]

        return {
            tid: (self._trk[tid]["gid"] or 0)
            for tid in active_tids
            if tid in self._trk
        }

    def num_persons(self) -> int:
        return len(self._gallery)

    def all_assignments(self) -> dict[int, int]:
        return {tid: t["gid"] for tid, t in self._trk.items() if t["gid"] is not None}

    # ── 내부 헬퍼 ────────────────────────────────────────────────────

    def _extract_feat(self, crops: list[np.ndarray]) -> np.ndarray:
        n = min(self.sample_n, len(crops))
        idx = np.linspace(0, len(crops) - 1, n, dtype=int)
        sampled = [crops[i] for i in idx]
        feats = self.extractor.extract_batch_features(sampled)
        f = feats.mean(axis=0).astype(np.float32)
        norm = np.linalg.norm(f)
        return f / max(norm, 1e-8)

    def _new_gid(self, feat: np.ndarray) -> int:
        gid = self._next_gid
        self._next_gid += 1
        self._gallery[gid] = feat.copy()
        return gid

    def _update_gallery(self, gid: int, feat: np.ndarray) -> None:
        updated = 0.7 * self._gallery[gid] + 0.3 * feat
        n = np.linalg.norm(updated)
        self._gallery[gid] = updated / max(n, 1e-8)

    def _compute_scores(self, feat: np.ndarray, vecs: np.ndarray) -> np.ndarray:
        """유사도 점수 반환 (높을수록 유사). L2 정규화된 벡터 가정."""
        if self.distance_metric == "cosine":
            return (feat @ vecs.T).astype(np.float32)
        # l2: 음수 L2 거리 (높을수록 가까움)
        return -np.linalg.norm(vecs - feat[None], axis=1).astype(np.float32)

    def _match_threshold(self) -> float:
        if self.distance_metric == "cosine":
            return self.sim_thresh
        # L2 로 변환: 단위 벡터에서 l2 = sqrt(2 - 2*cos)
        return -float(np.sqrt(max(0.0, 2.0 - 2.0 * self.sim_thresh)))

    def _mutual_topk_match(
        self, feat: np.ndarray, vecs: np.ndarray, scores: np.ndarray
    ) -> Tuple[int, float]:
        """Probe와 갤러리 엔트리가 서로의 top-k 안에 있을 때만 매칭."""
        k = min(self.mutual_topk_k, len(vecs))
        probe_topk = set(np.argsort(scores)[-k:].tolist())

        best_i, best_s = -1, -np.inf
        for i in probe_topk:
            # 갤러리[i] 기준으로 probe가 top-k 안에 드는지 확인
            other_mask = np.ones(len(vecs), dtype=bool)
            other_mask[i] = False
            probe_sim = float(vecs[i] @ feat)
            rank = int(np.sum(vecs[i] @ vecs[other_mask].T > probe_sim)) if other_mask.any() else 0
            if rank < k:
                s = float(scores[i])
                if s > best_s:
                    best_s, best_i = s, i

        if best_i < 0:   # 상호 top-k 없으면 nearest 로 폴백
            best_i = int(np.argmax(scores))
            best_s = float(scores[best_i])
        return best_i, best_s

    def _aqe_match(
        self, feat: np.ndarray, vecs: np.ndarray, scores: np.ndarray
    ) -> Tuple[int, float]:
        """Average Query Expansion: probe를 top-k 갤러리 피처 평균으로 확장 후 재비교."""
        k = min(self.mean_k, len(vecs))
        top_k_idx = np.argsort(scores)[-k:]
        # 확장 쿼리: probe + top-k 갤러리 피처의 평균, L2 정규화
        expanded = feat.copy()
        for i in top_k_idx:
            expanded = expanded + vecs[i]
        norm = np.linalg.norm(expanded)
        expanded = expanded / max(norm, 1e-8)
        # 확장된 쿼리로 재비교
        new_scores = self._compute_scores(expanded, vecs)
        best_i = int(np.argmax(new_scores))
        best_s = float(new_scores[best_i])
        return best_i, best_s

    def _rerank_match(
        self, feat: np.ndarray, gids: List[int], vecs: np.ndarray
    ) -> int:
        """Zhong et al. 2017 k-reciprocal re-ranking 기반 매칭."""
        n = len(gids)
        # index 0 = probe, 1..n = gallery
        all_f = np.vstack([feat[None], vecs]).astype(np.float32)  # [N, d]
        N = n + 1

        k1 = min(self.rerank_k1, n)
        k2 = min(self.rerank_k2, n)

        # 코사인 거리 행렬 [N, N]
        sims_mat = np.clip(all_f @ all_f.T, -1.0, 1.0)
        dist_mat = 1.0 - sims_mat
        np.fill_diagonal(dist_mat, 0.0)

        # 각 포인트의 정렬된 이웃 인덱스 (자기 자신 제외: 1:k+1)
        sort_idx = np.argsort(dist_mat, axis=1)  # [N, N]

        def k_recip(i: int, k: int) -> set:
            fwd = set(sort_idx[i, 1:k + 1].tolist())
            return {j for j in fwd if i in set(sort_idx[j, 1:k + 1].tolist())}

        # V 행렬: soft k-reciprocal membership [N, N]
        V = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            R_i = k_recip(i, k1)
            # 확장: 절반 k1 범위에서 2/3 이상 겹치면 병합
            R_exp = set(R_i)
            for j in R_i:
                half = max(1, k1 // 2)
                R_j = k_recip(j, half)
                if R_j and len(R_j & R_i) >= (2.0 / 3) * len(R_j):
                    R_exp |= R_j
            # k2 local query expansion
            w = 1.0 / (k2 + 1)
            for j in R_exp:
                for m in sort_idx[j, 0:k2 + 1].tolist():   # j 포함
                    V[i, m] += w

        # probe (0) vs gallery (1..n) Jaccard 거리
        pv = V[0]
        gv = V[1:]
        num = np.minimum(pv[None], gv).sum(axis=1)
        den = np.maximum(pv[None], gv).sum(axis=1) + 1e-8
        j_dist = 1.0 - num / den

        orig_dist = dist_mat[0, 1:]
        final_dist = (1.0 - self.rerank_lambda) * j_dist + self.rerank_lambda * orig_dist

        best_i = int(np.argmin(final_dist))
        best_d = float(final_dist[best_i])
        dist_thresh = 1.0 - self.sim_thresh

        if best_d <= dist_thresh:
            gid = gids[best_i]
            self._update_gallery(gid, feat)
            return gid
        return self._new_gid(feat)

    def _match_or_create(self, feat: np.ndarray) -> int:
        if not self._gallery:
            return self._new_gid(feat)

        gids = list(self._gallery.keys())
        vecs = np.stack([self._gallery[g] for g in gids])

        # Re-ranking 경로 (갤러리가 충분히 쌓인 경우)
        if self.use_rerank and len(gids) >= self.rerank_min_gallery:
            return self._rerank_match(feat, gids, vecs)

        # 표준 매칭
        scores = self._compute_scores(feat, vecs)
        if self.match_strategy == "mutual_topk":
            best_i, best_s = self._mutual_topk_match(feat, vecs, scores)
        elif self.match_strategy == "mean":
            best_i, best_s = self._aqe_match(feat, vecs, scores)
        else:
            best_i = int(np.argmax(scores))
            best_s = float(scores[best_i])

        if best_s >= self._match_threshold():
            gid = gids[best_i]
            self._update_gallery(gid, feat)
            return gid
        return self._new_gid(feat)


# ── 메인 제너레이터 ──────────────────────────────────────────────────
def iter_annotated_frames(
    video_path: str,
    yolo_model,
    extractor,
    *,
    device: str = "cpu",
    conf: float = 0.55,
    iou: float = 0.45,
    imgsz: int = 640,
    tracker_yaml: str = "configs/botsort.yaml",
    frame_stride: int = 1,
    reid_sim_thresh: float = 0.75,
    min_frames_for_reid: int = 6,
    show_unconfirmed: bool = True,
    init_gallery: Optional[Dict[int, np.ndarray]] = None,
    start_frame: int = 0,
    ctrl: Optional[dict] = None,
    distance_metric: str = "cosine",
    match_strategy: str = "nearest",
    mutual_topk_k: int = 3,
    mean_k: int = 5,
    use_rerank: bool = False,
    rerank_k1: int = 20,
    rerank_k2: int = 6,
    rerank_lambda: float = 0.3,
    rerank_min_gallery: int = 5,
) -> Generator[tuple[np.ndarray, dict], None, None]:
    """
    영상을 프레임 단위로 읽으면서 탐지·추적·Re-ID 를 온라인으로 수행하고
    어노테이션된 BGR 프레임을 yield 합니다.

    Parameters
    ----------
    video_path        : 입력 영상 파일 경로
    yolo_model        : ultralytics.YOLO 인스턴스 (persist=True 상태로 호출됨)
    extractor         : pipeline.reid_merger.OSNetExtractor 인스턴스
    device            : 추론 장치 ("cpu" | "cuda" | "auto")
    conf              : YOLOv8 검출 신뢰도 임계값
    iou               : YOLOv8 NMS IoU 임계값
    imgsz             : YOLOv8 추론 입력 크기
    tracker_yaml      : BoT-SORT 설정 파일 경로
    frame_stride      : N 프레임마다 1회 추론 (1 = 모든 프레임)
    reid_sim_thresh   : Re-ID 매칭 cosine 유사도 임계값 (높을수록 엄격)
    min_frames_for_reid: Re-ID 실행 전 누적해야 할 최소 크롭 수
    show_unconfirmed  : Re-ID 미확정 트랙렛도 회색 bbox 로 표시할지 여부

    Yields
    ------
    (annotated_frame: np.ndarray[H,W,3 BGR], stats: dict)

    stats keys
    ----------
    frame_idx       : 원본 프레임 인덱스
    processed_idx   : 처리된 프레임 카운터 (frame_stride 적용 후)
    total_frames    : 영상 전체 프레임 수
    fps             : 원본 영상 FPS
    num_detections  : 현재 프레임 탐지 수
    num_persons     : 누적 등록된 Re-ID 인물 수
    tid_to_gid      : {track_id: global_id} — 현재 프레임 매핑
    """
    # "auto" → 실제 디바이스로 변환 (Ultralytics 는 "auto" 를 허용하지 않음)
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    src_fps      = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    reid = _OnlineReID(
        extractor           = extractor,
        sim_thresh          = reid_sim_thresh,
        min_frames          = min_frames_for_reid,
        init_gallery        = init_gallery,
        distance_metric     = distance_metric,
        match_strategy      = match_strategy,
        mutual_topk_k       = mutual_topk_k,
        mean_k              = mean_k,
        use_rerank          = use_rerank,
        rerank_k1           = rerank_k1,
        rerank_k2           = rerank_k2,
        rerank_lambda       = rerank_lambda,
        rerank_min_gallery  = rerank_min_gallery,
    )

    frame_idx     = max(0, start_frame)
    processed_idx = 0

    if frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    try:
        while True:
            # ── FF / REW 제어 (ctrl["seek_to"] 에 목표 프레임 설정) ────
            if ctrl is not None:
                seek_to = ctrl.get("seek_to", -1)
                if seek_to >= 0:
                    target = int(max(0, min(seek_to, total_frames - 1)))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                    frame_idx = target
                    ctrl["seek_to"] = -1
                    # YOLO 트래커 상태 초기화 (seek 후 ID 혼선 방지)
                    try:
                        if getattr(yolo_model, "predictor", None) is not None:
                            yolo_model.predictor = None
                    except Exception:
                        pass

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride != 0:
                frame_idx += 1
                continue

            # ── YOLOv8 + BoT-SORT ─────────────────────────────────────
            results = yolo_model.track(
                frame,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=[0],
                device=device,
                persist=True,
                tracker=tracker_yaml,
                verbose=False,
            )
            r = results[0]

            detections: list[tuple[int, list[int], float]] = []
            if r.boxes is not None and r.boxes.id is not None:
                ids_arr    = r.boxes.id.cpu().numpy().astype(int)
                bboxes_arr = r.boxes.xyxy.cpu().numpy()
                confs_arr  = r.boxes.conf.cpu().numpy()
                for tid, bbox, c in zip(ids_arr, bboxes_arr, confs_arr):
                    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                    detections.append((int(tid), [x1, y1, x2, y2], float(c)))

            # ── 온라인 Re-ID 업데이트 ─────────────────────────────────
            tid_to_gid = reid.update(frame_idx, detections, frame, W, H)

            # ── 어노테이션 ────────────────────────────────────────────
            ann = frame.copy()
            for tid, bbox, conf_val in detections:
                gid        = tid_to_gid.get(tid, 0)
                reid_ready = (gid > 0)
                if reid_ready or show_unconfirmed:
                    _draw_box(ann, bbox, gid, tid, conf_val, reid_ready)

            # ── 오버레이 텍스트 ───────────────────────────────────────
            overlay = (
                f"Frame {frame_idx}/{total_frames}  "
                f"Det:{len(detections)}  "
                f"Persons(Re-ID):{reid.num_persons()}"
            )
            cv2.putText(ann, overlay, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(ann, overlay, (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            stats = {
                "frame_idx":           frame_idx,
                "processed_idx":       processed_idx,
                "total_frames":        total_frames,
                "fps":                 src_fps,
                "num_detections":      len(detections),
                "num_persons":         reid.num_persons(),
                "tid_to_gid":          dict(tid_to_gid),
                "matched_gallery_ids": reid._init_gallery_ids & set(tid_to_gid.values()),
            }

            yield ann, stats

            frame_idx     += 1
            processed_idx += 1
    finally:
        cap.release()


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
