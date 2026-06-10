"""
scripts/tune_threshold.py
=========================
data/demo_data/crops/*/feature.npy 에 저장된 특징 벡터를 불러와
HAC 클러스터링 threshold 값을 변경하며 결과를 빠르게 비교.

특징 추출을 재실행하지 않으므로 즉시 실행 가능.

사용법
------
  # 기본: threshold 0.10 ~ 0.50 구간을 0.05 간격으로 스캔
  python scripts/tune_threshold.py

  # 범위·간격 지정
  python scripts/tune_threshold.py --min 0.10 --max 0.40 --step 0.05

  # 특정 threshold 하나만 자세히 보기 (--debug: 거리 행렬 출력)
  python scripts/tune_threshold.py --threshold 0.20 --debug

  # 데이터 디렉터리 지정
  python scripts/tune_threshold.py --crops data/demo_data/crops
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial.distance import cdist


def load_features(crops_dir: Path) -> tuple[list[str], np.ndarray]:
    """crops/{pid}/feature.npy 를 모두 로드."""
    entries = sorted(crops_dir.iterdir())
    labels, feats = [], []
    for d in entries:
        fp = d / "feature.npy"
        if d.is_dir() and fp.exists():
            feats.append(np.load(str(fp)))
            labels.append(d.name)          # 폴더명 = 이전 person id
    if not feats:
        sys.exit(f"[ERROR] feature.npy 파일이 없습니다: {crops_dir}")
    return labels, np.stack(feats)


def cluster(feats: np.ndarray, threshold: float) -> np.ndarray:
    """L2 정규화 후 HAC(average linkage, cosine) 클러스터링."""
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    fn = feats / np.maximum(norms, 1e-8)
    if len(fn) == 1:
        return np.array([1])
    return fclusterdata(fn, t=threshold,
                        criterion="distance", metric="cosine", method="average")


def show_debug(labels: list[str], feats: np.ndarray, threshold: float,
               cluster_ids: np.ndarray):
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    fn = feats / np.maximum(norms, 1e-8)
    dist = cdist(fn, fn, metric="cosine")

    col_w = max(len(l) for l in labels)
    print(f"\n{'─'*60}")
    print(f"  pairwise cosine 거리 행렬  (threshold={threshold:.2f})")
    print(f"{'─'*60}")
    header = " " * (col_w + 2) + "  ".join(f"{l:>6}" for l in labels)
    print(header)
    for i, li in enumerate(labels):
        row = "  ".join(
            f"\033[1;31m{dist[i,j]:6.3f}\033[0m" if dist[i,j] < threshold and i != j
            else f"{dist[i,j]:6.3f}"
            for j in range(len(labels))
        )
        print(f"  {li:{col_w}}  {row}")

    print(f"\n  클러스터 구성 (threshold={threshold:.2f})")
    from collections import defaultdict
    groups: dict[int, list[str]] = defaultdict(list)
    for lbl, cid in zip(labels, cluster_ids):
        groups[int(cid)].append(lbl)
    for cid, members in sorted(groups.items()):
        mark = " ←병합" if len(members) > 1 else ""
        print(f"  cluster {cid:3d}: {members}{mark}")


def scan(labels: list[str], feats: np.ndarray,
         t_min: float, t_max: float, t_step: float):
    """threshold 범위를 스캔해 요약 테이블 출력."""
    print(f"\n{'threshold':>10}  {'클러스터 수':>10}  {'최대 크기':>9}  {'분포'}")
    print("─" * 60)
    t = t_min
    while t <= t_max + 1e-9:
        ids = cluster(feats, t)
        from collections import Counter
        cnt = Counter(int(c) for c in ids)
        n_clusters = len(cnt)
        max_size = max(cnt.values())
        sizes = sorted(cnt.values(), reverse=True)
        bar = " ".join(str(s) for s in sizes)
        print(f"  {t:8.2f}    {n_clusters:8d}    {max_size:8d}  [{bar}]")
        t = round(t + t_step, 10)


def main():
    parser = argparse.ArgumentParser(description="HAC threshold 튜닝 (feature 재추출 없음)")
    parser.add_argument("--crops", default="data/demo_data/crops",
                        help="feature.npy 가 들어있는 crops 폴더 (기본: data/demo_data/crops)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="단일 threshold 지정 시 --debug 모드로 상세 출력")
    parser.add_argument("--min",  type=float, default=0.10, help="스캔 최솟값 (기본: 0.10)")
    parser.add_argument("--max",  type=float, default=0.50, help="스캔 최댓값 (기본: 0.50)")
    parser.add_argument("--step", type=float, default=0.05, help="스캔 간격 (기본: 0.05)")
    parser.add_argument("--debug", action="store_true",
                        help="거리 행렬 + 클러스터 구성 상세 출력 (--threshold 와 함께)")
    args = parser.parse_args()

    crops_dir = Path(args.crops)
    if not crops_dir.exists():
        sys.exit(f"[ERROR] 폴더 없음: {crops_dir}")

    labels, feats = load_features(crops_dir)
    print(f"[INFO] feature 로드: {len(labels)}개  ({crops_dir})")
    print(f"[INFO] feature 차원: {feats.shape[1]}-d")

    if args.threshold is not None:
        ids = cluster(feats, args.threshold)
        from collections import Counter
        cnt = Counter(int(c) for c in ids)
        print(f"\n  threshold={args.threshold:.2f} → {len(cnt)}개 클러스터")
        if args.debug:
            show_debug(labels, feats, args.threshold, ids)
    else:
        scan(labels, feats, args.min, args.max, args.step)
        print("\n  ※ 특정 threshold 상세 보기:")
        print("    python scripts/tune_threshold.py --threshold 0.20 --debug")


if __name__ == "__main__":
    main()
