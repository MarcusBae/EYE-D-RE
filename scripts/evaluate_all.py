#!/usr/bin/env python
"""
scripts/evaluate_all.py
========================
정해진 모델·가중치·매칭·리랭킹 조합을 모두 실행하고
결과를 마크다운 표로 출력.

실행 예:
    python scripts/evaluate_all.py
    python scripts/evaluate_all.py --config configs/config.yaml
    python scripts/evaluate_all.py --skip-existing   # 이미 완료된 항목 건너뜀
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from pipeline.reid_merger import OSNetExtractor
from pipeline.evaluator import MarketEvaluator

# ── 평가 조합 정의 ────────────────────────────────────────────────
# (번호, 모델표시명, model_name, 가중치표시명, weights_path, matching, rerank)
COMBOS = [
    (1,  "OSNet x1.0",     "osnet_x1_0",     "Market-1501", "models/osnet_x1_0_market1501.pth",     "single", False),
    (2,  "OSNet x1.0",     "osnet_x1_0",     "Market-1501", "models/osnet_x1_0_market1501.pth",     "mean",   False),
    (3,  "OSNet x1.0",     "osnet_x1_0",     "Market-1501", "models/osnet_x1_0_market1501.pth",     "mean",   True),
    (4,  "OSNet x1.0",     "osnet_x1_0",     "ImageNet",    "models/osnet_x1_0_imagenet.pth",       "single", False),
    (5,  "OSNet x1.0",     "osnet_x1_0",     "ImageNet",    "models/osnet_x1_0_imagenet.pth",       "mean",   True),
    (6,  "OSNet-AIN x1.0", "osnet_ain_x1_0", "ImageNet",    "models/osnet_ain_x1_0_imagenet.pth",   "mean",   False),
    (7,  "OSNet-AIN x1.0", "osnet_ain_x1_0", "ImageNet",    "models/osnet_ain_x1_0_imagenet.pth",   "mean",   True),
    (8,  "OSNet-AIN x1.0", "osnet_ain_x1_0", "Market-1501*","models/osnet_ain_x1_0_market1501.pth", "mean",   True),
    (9,  "OSNet-AIN x1.0", "osnet_ain_x1_0", "MSMT17",      "models/osnet_ain_x1_0_msmt17.pth",     "mean",   False),
    (10, "OSNet-AIN x1.0", "osnet_ain_x1_0", "MSMT17",      "models/osnet_ain_x1_0_msmt17.pth",     "mean",   True),
]
# * Market-1501(구버전): IN 없는 osnet_x1_0 구조 → _load_model이 자동 변환


def run_one(combo: tuple, config: dict, extractor_cache: dict) -> dict:
    idx, model_label, model_name, weights_label, weights_path, matching, rerank = combo

    cache_key = (model_name, weights_path)
    if cache_key not in extractor_cache:
        extractor_cache[cache_key] = OSNetExtractor(
            model_name=model_name,
            pretrained=True,
            weights_path=weights_path,
            device=config.get("reid", {}).get("device", "auto"),
        )
    extractor = extractor_cache[cache_key]

    cfg = dict(config)
    cfg["reid"] = dict(config.get("reid", {}))
    cfg["reid"]["model_name"]   = model_name
    cfg["reid"]["weights_path"] = weights_path

    evaluator = MarketEvaluator(cfg)
    results   = evaluator.run(extractor, matching=matching, rerank=rerank)
    return results


def fmt(val: float) -> str:
    return f"**{val:.2f}%**" if val >= 70 else f"{val:.2f}%"


def main():
    parser = argparse.ArgumentParser(description="EYE-D 전체 조합 평가")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--combos", nargs="*", type=int, default=None,
                        help="실행할 조합 번호 목록 (예: --combos 1 3 10). 미지정 시 전체")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    combos = COMBOS
    if args.combos:
        combos = [c for c in COMBOS if c[0] in args.combos]

    rows   = []
    errors = []
    extractor_cache: dict = {}

    print(f"\n총 {len(combos)}개 조합 평가 시작\n")

    for combo in combos:
        idx, model_label, model_name, weights_label, weights_path, matching, rerank = combo
        rerank_str   = "Re-ranking" if rerank else "—"
        matching_str = "Mean Pooling" if matching == "mean" else matching.capitalize()

        wp = Path(weights_path)
        if not wp.exists():
            print(f"  [{idx:2d}] SKIP — 가중치 파일 없음: {weights_path}")
            rows.append((idx, model_label, weights_label, rerank_str, matching_str,
                         "—", "—", "—", "—", "—", "—", "skip"))
            continue

        print(f"  [{idx:2d}] {model_label} | {weights_label} | {matching_str} | {rerank_str} ...",
              end=" ", flush=True)
        t0 = time.time()
        try:
            results = run_one(combo, config, extractor_cache)
            elapsed = time.time() - t0
            cmc     = results["CMC"]
            r1  = cmc.get("Rank-1",  0.0)
            r5  = cmc.get("Rank-5",  0.0)
            r10 = cmc.get("Rank-10", 0.0)
            map_ = results["mAP"]
            query   = f"{results['num_query']} / {results['num_query_ids']} ID"
            gallery = f"{results['num_gallery']}장"
            print(f"Rank-1={r1:.2f}%  mAP={map_:.2f}%  ({elapsed:.0f}s)")
            rows.append((idx, model_label, weights_label, rerank_str, matching_str,
                         fmt(r1), fmt(r5), fmt(r10), fmt(map_), query, gallery, "done"))
        except Exception as e:
            print(f"ERROR: {e}")
            errors.append((idx, str(e)))
            rows.append((idx, model_label, weights_label, rerank_str, matching_str,
                         "—", "—", "—", "—", "—", "—", f"error"))

    # ── 마크다운 표 출력 ──────────────────────────────────────────
    header = ("| # | 모델 | 가중치 | 후처리 | Matching "
              "| Rank-1 | Rank-5 | Rank-10 | mAP "
              "| Query | Gallery |")
    sep    = ("| :--- | :--- | :--- | :--- | :--- "
              "| ---: | ---: | ---: | ---: "
              "| ---: | ---: |")
    print()
    print("## 평가 결과\n")
    print(header)
    print(sep)
    for (idx, model_label, weights_label, rerank_str, matching_str,
         r1, r5, r10, map_, query, gallery, status) in rows:
        print(f"| {idx} | {model_label} | {weights_label} | {rerank_str} | {matching_str} "
              f"| {r1} | {r5} | {r10} | {map_} "
              f"| {query} | {gallery} |")

    if errors:
        print("\n### 오류 목록")
        for idx, msg in errors:
            print(f"- [{idx}] {msg}")


if __name__ == "__main__":
    main()
