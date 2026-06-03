"""
pipeline/evaluator.py
=====================
Market-1501 표준 프로토콜 기반 Re-ID 평가 엔진.

평가 프로토콜:
- Query 이미지마다 Gallery 전체와 cosine distance 계산
- Good match  : 동일 pid, 다른 cam
- Junk        : 동일 pid, 동일 cam (랭킹에서 제외)
- mAP / CMC@k 산출

Matching 전략:
- single        : 이미지 1장당 특징 1개 (기본)
- mean          : 같은 트랙렛 이미지 특징 평균 (Mean Pooling)
- conf_weighted : 검출 신뢰도 가중 평균 (curated metadata 필요)
- tta           : 원본 + 좌우반전 특징 평균 (Test-Time Augmentation)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from .reranking import re_ranking


_FNAME_RE = re.compile(r"(\d+)_c(\d+)s(\d+)_(\d+)_00\.jpg")

MATCHING_STRATEGIES = ("single", "mean", "conf_weighted", "tta")


def parse_market_filename(filename: str) -> Tuple[int, int, int, int]:
    """파일명에서 (pid, cam, slot, frame) 파싱."""
    m = _FNAME_RE.match(Path(filename).name)
    if not m:
        raise ValueError(f"Market-1501 파일명 형식 불일치: {filename}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def load_market_split(split_dir: str) -> Tuple[List[Path], np.ndarray, np.ndarray]:
    """Market-1501 split 폴더에서 이미지 경로, pid 배열, cam 배열 반환."""
    split_dir = Path(split_dir)
    paths, pids, cams = [], [], []
    skipped_count = 0
    total_count = 0
    for fpath in sorted(split_dir.glob("*.jpg")):
        total_count += 1
        try:
            pid, cam, _, _ = parse_market_filename(fpath.name)
        except ValueError:
            skipped_count += 1
            continue
        paths.append(fpath)
        pids.append(pid)
        cams.append(cam)
    if skipped_count > 0:
        print(f"[WARN] {split_dir.name}: 파일명 형식 불일치 {skipped_count}개 제외 "
              f"(총 {total_count}개 중)")
    return paths, np.array(pids), np.array(cams)


def load_confidence_map(curated_dir: str) -> dict:
    """
    curated 메타데이터에서 (pid, cam, slot, frame) → confidence 룩업 테이블 생성.
    conf_weighted 전략에서 사용.
    """
    conf_map = {}
    for meta_path in Path(curated_dir).glob("*/track_*/metadata.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        gid = meta.get("global_id")
        cam = meta.get("camera_id")
        slot = meta.get("time_slot")
        crop_files = meta.get("crop_files", [])
        confidences = meta.get("confidences", [])
        for i, fname in enumerate(crop_files):
            m = re.search(r"(\d+)", Path(fname).stem)
            if m and i < len(confidences):
                frame_num = int(m.group(1))
                conf_map[(gid, cam, slot, frame_num)] = float(confidences[i])
    return conf_map


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-8 else v


def extract_features(
    paths: List[Path],
    extractor,
    batch_size: int = 64,
) -> np.ndarray:
    """이미지 경로 리스트에서 L2 정규화된 특징 벡터 행렬 추출 (single 전략)."""
    all_feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="  feature extraction"):
        batch_paths = paths[i : i + batch_size]
        crops = []
        for p in batch_paths:
            img = cv2.imread(str(p))
            crops.append(img if img is not None and img.size > 0
                         else np.zeros((256, 128, 3), dtype=np.uint8))
        feats = extractor.extract_batch_features(crops, batch_size=batch_size)
        all_feats.append(feats)
    return np.concatenate(all_feats, axis=0)


def extract_features_tta(
    paths: List[Path],
    extractor,
    batch_size: int = 64,
) -> np.ndarray:
    """TTA: 원본 + 좌우반전 특징 평균 후 L2 재정규화."""
    all_feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="  feature extraction (TTA)"):
        batch_paths = paths[i : i + batch_size]
        crops_orig, crops_flip = [], []
        for p in batch_paths:
            img = cv2.imread(str(p))
            if img is not None and img.size > 0:
                crops_orig.append(img)
                crops_flip.append(cv2.flip(img, 1))
            else:
                dummy = np.zeros((256, 128, 3), dtype=np.uint8)
                crops_orig.append(dummy)
                crops_flip.append(dummy)
        f_orig = extractor.extract_batch_features(crops_orig, batch_size=batch_size)
        f_flip = extractor.extract_batch_features(crops_flip, batch_size=batch_size)
        avg = (f_orig + f_flip) / 2.0
        norms = np.linalg.norm(avg, axis=1, keepdims=True)
        avg = avg / np.maximum(norms, 1e-8)
        all_feats.append(avg)
    return np.concatenate(all_feats, axis=0)


def pool_tracklet_features(
    paths: List[Path],
    pids: np.ndarray,
    cams: np.ndarray,
    base_feats: np.ndarray,
    strategy: str,
    conf_map: dict = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    이미지 단위 특징을 트랙렛 단위로 집계.

    같은 (pid, cam, slot) = 같은 트랙렛으로 묶어 pooling.

    Returns
    -------
    (tracklet_feats, tracklet_pids, tracklet_cams)
    """
    # (pid, cam, slot) → 인덱스 목록
    groups: Dict[Tuple, List[int]] = defaultdict(list)
    slots = []
    for path in paths:
        _, _, slot, _ = parse_market_filename(path.name)
        slots.append(slot)

    for idx, (path, pid, cam, slot) in enumerate(zip(paths, pids, cams, slots)):
        groups[(int(pid), int(cam), int(slot))].append(idx)

    t_feats, t_pids, t_cams = [], [], []

    for (pid, cam, slot), indices in sorted(groups.items()):
        feats = base_feats[indices]  # (n_frames, D)

        if strategy == "mean":
            pooled = feats.mean(axis=0)

        elif strategy == "conf_weighted" and conf_map is not None:
            weights = []
            for idx in indices:
                _, _, _, frame = parse_market_filename(paths[idx].name)
                w = conf_map.get((pid, cam, slot, frame), 1.0)
                weights.append(w)
            weights = np.array(weights, dtype=np.float32)
            weights /= weights.sum() if weights.sum() > 0 else 1.0
            pooled = (feats * weights[:, None]).sum(axis=0)

        else:
            pooled = feats.mean(axis=0)

        t_feats.append(_l2_normalize(pooled))
        t_pids.append(pid)
        t_cams.append(cam)

    return np.array(t_feats), np.array(t_pids), np.array(t_cams)


def compute_distance_matrix(
    query_feats: np.ndarray, gallery_feats: np.ndarray
) -> np.ndarray:
    """L2 정규화된 벡터 간 cosine distance 행렬. shape: (Nq, Ng)"""
    sim = query_feats @ gallery_feats.T
    return np.clip(1.0 - sim, 0.0, 2.0)


def compute_map_cmc(
    distmat: np.ndarray,
    query_pids: np.ndarray,
    query_cams: np.ndarray,
    gallery_pids: np.ndarray,
    gallery_cams: np.ndarray,
    max_rank: int = 10,
) -> Tuple[float, np.ndarray]:
    """Market-1501 표준 프로토콜로 mAP와 CMC@1~max_rank 계산."""
    all_ap, all_cmc = [], []

    for q_idx in range(distmat.shape[0]):
        q_pid = query_pids[q_idx]
        q_cam = query_cams[q_idx]

        order = np.argsort(distmat[q_idx])
        g_pids = gallery_pids[order]
        g_cams = gallery_cams[order]

        good_mask = (g_pids == q_pid) & (g_cams != q_cam)
        junk_mask = (g_pids == q_pid) & (g_cams == q_cam)

        if good_mask.sum() == 0:
            continue

        keep = ~junk_mask
        good_keep = good_mask[keep]
        num_good = int(good_keep.sum())
        if num_good == 0:
            continue

        hit_positions = np.where(good_keep)[0]
        ap = sum((i + 1) / (pos + 1) for i, pos in enumerate(hit_positions)) / num_good
        all_ap.append(ap)

        cmc = np.zeros(max_rank, dtype=np.float32)
        first_hit = hit_positions[0]
        if first_hit < max_rank:
            cmc[first_hit:] = 1.0
        all_cmc.append(cmc)

    if not all_ap:
        return 0.0, np.zeros(max_rank)

    return float(np.mean(all_ap)), np.mean(all_cmc, axis=0)


class MarketEvaluator:
    """Market-1501 포맷 데이터셋에 대한 Re-ID 평가기."""

    def __init__(self, config: Dict):
        self.market_dir = Path(config.get("market1501", {}).get("output_dir", "data/market1501"))
        self.curated_dir = Path(config.get("data", {}).get("filtered_dir", "data/curated"))
        eval_cfg = config.get("evaluation", {})
        self.batch_size = eval_cfg.get("batch_size", 64)
        self.top_k = eval_cfg.get("top_k", [1, 5, 10])
        self.output_dir = Path(eval_cfg.get("output_dir", "outputs/eval_results"))
        self.max_rank = max(self.top_k)

    def run(self, extractor, matching: str = "single", rerank: bool = False) -> Dict:
        """
        평가 파이프라인 실행.

        Parameters
        ----------
        matching : 'single' | 'mean' | 'conf_weighted' | 'tta'
        rerank   : True이면 k-reciprocal re-ranking 후처리 적용
        """
        if matching not in MATCHING_STRATEGIES:
            raise ValueError(f"지원하지 않는 matching 전략: {matching}. "
                             f"선택 가능: {MATCHING_STRATEGIES}")

        print(f"[Eval] Matching 전략: {matching}")
        print("[Eval] Query / Gallery 로드 중...")
        q_paths, q_pids, q_cams = load_market_split(self.market_dir / "query")
        g_paths, g_pids, g_cams = load_market_split(self.market_dir / "bounding_box_test")

        print(f"[Eval] Query  : {len(q_paths)}장, {len(set(q_pids))}개 ID")
        print(f"[Eval] Gallery: {len(g_paths)}장, {len(set(g_pids))}개 ID")

        if len(q_paths) == 0 or len(g_paths) == 0:
            raise FileNotFoundError(
                f"Query 또는 Gallery 이미지가 없습니다. "
                f"format_market1501.py를 먼저 실행하세요. ({self.market_dir.resolve()})"
            )

        # 특징 추출
        feat_fn = extract_features_tta if matching == "tta" else extract_features

        print("[Eval] Query 특징 추출 중...")
        q_feats_raw = feat_fn(q_paths, extractor, self.batch_size)

        print("[Eval] Gallery 특징 추출 중...")
        g_feats_raw = feat_fn(g_paths, extractor, self.batch_size)

        # 트랙렛 풀링 (mean / conf_weighted)
        if matching in ("mean", "conf_weighted"):
            conf_map = None
            if matching == "conf_weighted":
                print("[Eval] curated 메타데이터에서 신뢰도 로드 중...")
                conf_map = load_confidence_map(str(self.curated_dir))
                print(f"[Eval] 신뢰도 엔트리: {len(conf_map)}개")

            print("[Eval] 트랙렛 풀링 중...")
            q_feats, q_pids, q_cams = pool_tracklet_features(
                q_paths, q_pids, q_cams, q_feats_raw, matching, conf_map)
            g_feats, g_pids, g_cams = pool_tracklet_features(
                g_paths, g_pids, g_cams, g_feats_raw, matching, conf_map)
            print(f"[Eval] 풀링 후 — Query: {len(q_feats)}개 트랙렛, "
                  f"Gallery: {len(g_feats)}개 트랙렛")
        else:
            q_feats, g_feats = q_feats_raw, g_feats_raw

        if rerank:
            print("[Eval] Re-ranking 적용 중 (k1=20, k2=6, λ=0.3)...")
            distmat = re_ranking(q_feats, g_feats)
        else:
            print("[Eval] 거리 행렬 계산 중...")
            distmat = compute_distance_matrix(q_feats, g_feats)

        print("[Eval] mAP / CMC 계산 중...")
        mAP, cmc = compute_map_cmc(distmat, q_pids, q_cams, g_pids, g_cams, self.max_rank)

        results = {
            "matching": matching,
            "rerank": rerank,
            "mAP": round(mAP * 100, 2),
            "CMC": {f"Rank-{k}": round(float(cmc[k - 1]) * 100, 2) for k in self.top_k},
            "num_query": len(q_feats),
            "num_gallery": len(g_feats),
            "num_query_ids": int(len(set(q_pids))),
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"{matching}_rerank" if rerank else matching
        out_path = self.output_dir / f"results_{suffix}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return results
