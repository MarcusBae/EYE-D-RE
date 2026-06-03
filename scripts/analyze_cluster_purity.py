"""
scripts/analyze_cluster_purity.py
==================================
클러스터(Global ID) 내 트랙렛 간 cosine 유사도를 분석하여 오병합 의심 클러스터를 탐지합니다.

핵심 아이디어:
  - 같은 Global ID 내 트랙렛들의 대표 임베딩 간 pairwise cosine 유사도 계산
  - 유사도가 낮은 쌍이 많을수록 서로 다른 사람이 섞인 오병합 클러스터
  - 임계값(--sim-threshold) 미만 쌍의 비율을 "오염도"로 정의

사용법:
    python scripts/analyze_cluster_purity.py --config configs/config.yaml
    python scripts/analyze_cluster_purity.py --config configs/config.yaml \\
        --sim-threshold 0.6 --out outputs/purity_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from pipeline.reid_merger import OSNetExtractor

MAX_FRAMES_PER_TRACKLET = 8


def collect_tracklets(filtered_dir: Path) -> dict[int, list[dict]]:
    """global_id → [{"tdir": Path, "meta": dict}] 매핑 구성"""
    clusters: dict[int, list] = {}
    for meta_path in sorted(filtered_dir.glob("*/track_*/metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        gid = meta.get("global_id")
        if gid is None:
            continue
        clusters.setdefault(gid, []).append({
            "tdir": meta_path.parent,
            "meta": meta,
        })
    return clusters


def representative_feature(tdir: Path, extractor: OSNetExtractor) -> np.ndarray | None:
    """트랙렛 디렉토리에서 균등 샘플링 후 평균 L2 정규화 임베딩 반환"""
    imgs_paths = sorted(tdir.glob("*.jpg"))
    if not imgs_paths:
        return None

    if len(imgs_paths) > MAX_FRAMES_PER_TRACKLET:
        indices = np.linspace(0, len(imgs_paths) - 1, MAX_FRAMES_PER_TRACKLET, dtype=int)
        imgs_paths = [imgs_paths[i] for i in indices]

    crops = []
    for p in imgs_paths:
        img = cv2.imread(str(p))
        if img is not None and img.size > 0:
            crops.append(img)

    if not crops:
        return None

    feats = extractor.extract_batch_features(crops)
    mean_feat = np.mean(feats, axis=0)
    norm = np.linalg.norm(mean_feat)
    return mean_feat / norm if norm > 0 else None


def analyze_cluster(tracklets: list[dict], extractor: OSNetExtractor,
                    sim_threshold: float) -> dict | None:
    """
    클러스터 내 모든 트랙렛 쌍의 cosine 유사도 계산 후 통계 반환.
    트랙렛이 1개뿐이면 None 반환 (쌍 비교 불가).
    """
    feats = []
    for t in tracklets:
        feat = representative_feature(t["tdir"], extractor)
        if feat is not None:
            feats.append((t, feat))

    if len(feats) < 2:
        return None

    tracklet_info = [
        f"{t['tdir'].parent.name}/{t['tdir'].name}" for t, _ in feats
    ]
    feat_matrix = np.stack([f for _, f in feats])  # (N, D)

    # pairwise cosine 유사도 (이미 L2 정규화됨 → 내적)
    sim_matrix = feat_matrix @ feat_matrix.T
    n = len(feats)
    pair_sims = [
        float(sim_matrix[i, j])
        for i, j in combinations(range(n), 2)
    ]

    contaminated_pairs = sum(1 for s in pair_sims if s < sim_threshold)
    total_pairs = len(pair_sims)

    return {
        "num_tracklets": n,
        "num_pairs": total_pairs,
        "sim_min": round(float(min(pair_sims)), 4),
        "sim_mean": round(float(np.mean(pair_sims)), 4),
        "sim_p10": round(float(np.percentile(pair_sims, 10)), 4),
        "contaminated_pairs": contaminated_pairs,
        "contamination_rate": round(contaminated_pairs / total_pairs, 4),
        "tracklets": tracklet_info,
    }


def print_report(results: list[dict], sim_threshold: float):
    total = len(results)
    contaminated = [r for r in results if r["contamination_rate"] > 0]

    print()
    print("=" * 60)
    print("  클러스터 내 유사도 분석 결과")
    print("=" * 60)
    print(f"  분석된 클러스터 수 : {total}개")
    print(f"  유사도 임계값      : {sim_threshold:.2f}  (이 미만 = 오병합 의심)")
    print(f"  오염 클러스터 수   : {len(contaminated)}개 "
          f"({100 * len(contaminated) / total:.1f}%)")
    print()
    print(f"  {'Global ID':>10}  {'트랙렛':>5}  {'min sim':>8}  "
          f"{'mean sim':>9}  {'오염 쌍':>7}  {'오염율':>7}")
    print("  " + "-" * 56)
    for r in results[:30]:  # 상위 30개만 출력
        flag = " !" if r["contamination_rate"] > 0.3 else ""
        print(f"  {r['global_id']:>10}  {r['num_tracklets']:>5}  "
              f"{r['sim_min']:>8.3f}  {r['sim_mean']:>9.3f}  "
              f"{r['contaminated_pairs']:>5}/{r['num_pairs']:<5}"
              f"  {r['contamination_rate']:>6.1%}{flag}")
    if len(results) > 30:
        print(f"  ... (나머지 {len(results) - 30}개는 JSON 파일 참조)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="클러스터 내 유사도 분석 (오병합 탐지)")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--filtered-dir", default=None,
                        help="filtered 데이터 경로 (미지정 시 config에서 읽음)")
    parser.add_argument("--sim-threshold", type=float, default=0.6,
                        help="cosine 유사도 하한 — 이 미만 쌍을 오병합으로 간주 (기본: 0.6)")
    parser.add_argument("--out", default="outputs/purity_report.json",
                        help="결과 JSON 저장 경로 (기본: outputs/purity_report.json)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"[ERROR] config 파일 없음: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    filtered_dir = Path(
        args.filtered_dir
        or config.get("quality_filter", {}).get("output_dir", "data/filtered")
    )
    if not filtered_dir.exists():
        sys.exit(f"[ERROR] filtered 디렉토리 없음: {filtered_dir.resolve()}")

    reid_cfg = config.get("reid", {})
    hac_threshold = reid_cfg.get("clustering", {}).get("threshold", 0.25)
    # cosine distance → similarity 변환
    hac_sim = 1.0 - hac_threshold
    print(f"[INFO] HAC 병합 임계값: distance {hac_threshold:.3f} "
          f"(유사도 {hac_sim:.3f}) — 분석 임계값: {args.sim_threshold:.3f}")

    print(f"[INFO] filtered 경로: {filtered_dir.resolve()}")
    clusters = collect_tracklets(filtered_dir)
    print(f"[INFO] Global ID {len(clusters)}개 발견")

    extractor = OSNetExtractor(
        model_name=reid_cfg.get("model_name", "osnet_x1_0"),
        pretrained=reid_cfg.get("pretrained", True),
        device=reid_cfg.get("device", "auto"),
    )

    results = []
    for gid in tqdm(sorted(clusters), desc="클러스터 분석"):
        stats = analyze_cluster(clusters[gid], extractor, args.sim_threshold)
        if stats is None:
            continue
        stats["global_id"] = gid
        results.append(stats)

    # 오염율 내림차순, 같으면 min_sim 오름차순 정렬
    results.sort(key=lambda r: (-r["contamination_rate"], r["sim_min"]))

    print_report(results, args.sim_threshold)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[완료] 전체 결과 저장: {out_path.resolve()}")


if __name__ == "__main__":
    main()
