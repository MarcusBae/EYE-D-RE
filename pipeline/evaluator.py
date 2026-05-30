"""
pipeline/evaluator.py
=====================
Market-1501 표준 프로토콜 기반 Re-ID 평가 엔진.

평가 프로토콜:
- Query 이미지마다 Gallery 전체와 cosine distance 계산
- Good match  : 동일 pid, 다른 cam
- Junk        : 동일 pid, 동일 cam (랭킹에서 제외)
- mAP / CMC@k 산출
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


_FNAME_RE = re.compile(r"(\d+)_c(\d+)s(\d+)_(\d+)_00\.jpg")


def parse_market_filename(filename: str) -> Tuple[int, int, int, int]:
    """파일명에서 (pid, cam, slot, frame) 파싱."""
    m = _FNAME_RE.match(Path(filename).name)
    if not m:
        raise ValueError(f"Market-1501 파일명 형식 불일치: {filename}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def load_market_split(split_dir: str) -> Tuple[List[Path], np.ndarray, np.ndarray]:
    """
    Market-1501 split 폴더에서 이미지 경로, pid 배열, cam 배열 반환.

    Returns
    -------
    (paths, pids, cams) : 각각 길이 N의 리스트/배열
    """
    split_dir = Path(split_dir)
    paths, pids, cams = [], [], []
    for fpath in sorted(split_dir.glob("*.jpg")):
        try:
            pid, cam, _, _ = parse_market_filename(fpath.name)
        except ValueError:
            continue
        paths.append(fpath)
        pids.append(pid)
        cams.append(cam)
    return paths, np.array(pids), np.array(cams)


def extract_features(
    paths: List[Path],
    extractor,
    batch_size: int = 64,
) -> np.ndarray:
    """
    이미지 경로 리스트에서 L2 정규화된 특징 벡터 행렬 추출.

    Returns
    -------
    np.ndarray : shape (N, feat_dim)
    """
    all_feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="  feature extraction"):
        batch_paths = paths[i : i + batch_size]
        crops = []
        for p in batch_paths:
            img = cv2.imread(str(p))
            if img is not None and img.size > 0:
                crops.append(img)
            else:
                crops.append(np.zeros((256, 128, 3), dtype=np.uint8))
        feats = extractor.extract_batch_features(crops, batch_size=batch_size)
        all_feats.append(feats)
    return np.concatenate(all_feats, axis=0)


def compute_distance_matrix(
    query_feats: np.ndarray, gallery_feats: np.ndarray
) -> np.ndarray:
    """L2 정규화된 벡터 간 cosine distance 행렬 계산. shape: (Nq, Ng)"""
    sim = query_feats @ gallery_feats.T
    dist = 1.0 - sim
    return np.clip(dist, 0.0, 2.0)


def compute_map_cmc(
    distmat: np.ndarray,
    query_pids: np.ndarray,
    query_cams: np.ndarray,
    gallery_pids: np.ndarray,
    gallery_cams: np.ndarray,
    max_rank: int = 10,
) -> Tuple[float, np.ndarray]:
    """
    Market-1501 표준 프로토콜로 mAP와 CMC@1~max_rank 계산.

    Junk(동일 pid + 동일 cam)는 랭킹에서 제거 후 평가.

    Returns
    -------
    (mAP, cmc) : mAP float, cmc shape (max_rank,)
    """
    num_q = distmat.shape[0]
    all_ap: List[float] = []
    all_cmc: List[np.ndarray] = []

    for q_idx in range(num_q):
        q_pid = query_pids[q_idx]
        q_cam = query_cams[q_idx]

        order = np.argsort(distmat[q_idx])
        g_pids = gallery_pids[order]
        g_cams = gallery_cams[order]

        # good: 동일 pid, 다른 cam
        good_mask = (g_pids == q_pid) & (g_cams != q_cam)
        # junk: 동일 pid, 동일 cam
        junk_mask = (g_pids == q_pid) & (g_cams == q_cam)

        if good_mask.sum() == 0:
            continue

        # junk 제거 후 keep 인덱스
        keep = ~junk_mask
        good_keep = good_mask[keep]

        num_good = int(good_keep.sum())
        if num_good == 0:
            continue

        # AP 계산
        hit_positions = np.where(good_keep)[0]  # 0-indexed
        ap = sum(
            (i + 1) / (pos + 1) for i, pos in enumerate(hit_positions)
        ) / num_good
        all_ap.append(ap)

        # CMC: 첫 번째 정답 위치
        cmc = np.zeros(max_rank, dtype=np.float32)
        first_hit = hit_positions[0]
        if first_hit < max_rank:
            cmc[first_hit:] = 1.0
        all_cmc.append(cmc)

    if not all_ap:
        return 0.0, np.zeros(max_rank)

    mAP = float(np.mean(all_ap))
    cmc = np.mean(all_cmc, axis=0)
    return mAP, cmc


class MarketEvaluator:
    """Market-1501 포맷 데이터셋에 대한 Re-ID Zero-shot 평가기."""

    def __init__(self, config: Dict):
        self.market_dir = Path(config.get("market1501", {}).get("output_dir", "data/market1501"))
        eval_cfg = config.get("evaluation", {})
        self.batch_size = eval_cfg.get("batch_size", 64)
        self.top_k = eval_cfg.get("top_k", [1, 5, 10])
        self.output_dir = Path(eval_cfg.get("output_dir", "data/eval_results"))
        self.max_rank = max(self.top_k)

    def run(self, extractor) -> Dict:
        """전체 평가 파이프라인 실행. extractor는 OSNetExtractor 인스턴스."""
        print("[Eval] Query / Gallery 로드 중...")
        q_paths, q_pids, q_cams = load_market_split(self.market_dir / "query")
        g_paths, g_pids, g_cams = load_market_split(self.market_dir / "bounding_box_test")

        print(f"[Eval] Query  : {len(q_paths)}장, {len(set(q_pids))}개 ID")
        print(f"[Eval] Gallery: {len(g_paths)}장, {len(set(g_pids))}개 ID")

        print("[Eval] Query 특징 추출 중...")
        q_feats = extract_features(q_paths, extractor, self.batch_size)

        print("[Eval] Gallery 특징 추출 중...")
        g_feats = extract_features(g_paths, extractor, self.batch_size)

        print("[Eval] 거리 행렬 계산 중...")
        distmat = compute_distance_matrix(q_feats, g_feats)

        print("[Eval] mAP / CMC 계산 중...")
        mAP, cmc = compute_map_cmc(distmat, q_pids, q_cams, g_pids, g_cams, self.max_rank)

        results = {
            "mAP": round(mAP * 100, 2),
            "CMC": {f"Rank-{k}": round(float(cmc[k - 1]) * 100, 2) for k in self.top_k},
            "num_query": len(q_paths),
            "num_gallery": len(g_paths),
            "num_query_ids": int(len(set(q_pids))),
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "zeroshot_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results
