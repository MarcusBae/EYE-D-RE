#!/usr/bin/env python
"""
scripts/evaluate_zeroshot.py
=============================
Phase 2: 사전학습된 OSNet으로 Market-1501 포맷 데이터셋을 Zero-shot 평가.

실행 예:
  python scripts/evaluate_zeroshot.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from pipeline.reid_merger import OSNetExtractor
from pipeline.evaluator import MarketEvaluator


def main():
    parser = argparse.ArgumentParser(description="EYE-D — Phase 2 Zero-shot Evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] 설정 파일 없음: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    reid_cfg = config.get("reid", {})

    print("=" * 55)
    print("  Phase 2: Zero-shot Re-ID Evaluation")
    print("=" * 55)
    print(f"  모델  : {reid_cfg.get('model_name', 'osnet_x1_0')} (pretrained)")
    print(f"  데이터: {config.get('market1501', {}).get('output_dir', 'data/market1501')}")
    print("=" * 55)

    # OSNet 로드
    try:
        extractor = OSNetExtractor(
            model_name=reid_cfg.get("model_name", "osnet_x1_0"),
            pretrained=reid_cfg.get("pretrained", True),
            weights_path=reid_cfg.get("weights_path", None),
            device=reid_cfg.get("device", "auto"),
        )
    except RuntimeError as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        sys.exit(1)

    # 평가 실행
    evaluator = MarketEvaluator(config)
    results = evaluator.run(extractor)

    # 결과 출력
    print()
    print("=" * 55)
    print("  평가 결과 (Zero-shot)")
    print("=" * 55)
    print(f"  Query   : {results['num_query']}장 / {results['num_query_ids']}개 ID")
    print(f"  Gallery : {results['num_gallery']}장")
    print(f"  mAP     : {results['mAP']:.2f}%")
    for rank_key, val in results["CMC"].items():
        print(f"  {rank_key:<8}: {val:.2f}%")
    print("=" * 55)
    print(f"  결과 저장: {evaluator.output_dir / 'zeroshot_results.json'}")


if __name__ == "__main__":
    main()
