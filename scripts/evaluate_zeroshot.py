#!/usr/bin/env python
"""
scripts/evaluate_zeroshot.py
=============================
Phase 2: 사전학습된 OSNet으로 Market-1501 포맷 데이터셋을 Zero-shot 평가.

실행 예:
    #1 
    python scripts/evaluate_zeroshot.py --model osnet_x1_0 --weights models/osnet_x1_0_imagenet.pth         --matching single
    
    #2 
    python scripts/evaluate_zeroshot.py --model osnet_x1_0 --weights models/osnet_ain_x1_0_market1501.pth   --matching single

    #3
    python scripts/evaluate_zeroshot.py --model osnet_x1_0 --weights models/osnet_x1_0_imagenet.pth         -- matching single --rerank

    #4



  python scripts/evaluate_zeroshot.py --config configs/config.yaml
  python scripts/evaluate_zeroshot.py --config configs/config.yaml --matching mean --rerank

  # 모델·가중치 직접 지정 (config.yaml 값 덮어씀)
  python scripts/evaluate_zeroshot.py --matching mean --rerank \
      --model osnet_ain_x1_0 --weights models/osnet_ain_x1_0_msmt17.pth
  python scripts/evaluate_zeroshot.py --matching mean --rerank \
      --model osnet_x1_0     --weights models/osnet_x1_0_market1501.pth
  python scripts/evaluate_zeroshot.py --matching mean --rerank \
      --model osnet_ain_x1_0 --weights models/osnet_ain_x1_0_imagenet.pth
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
from pipeline.evaluator import MarketEvaluator, MATCHING_STRATEGIES


def main():
    parser = argparse.ArgumentParser(description="EYE-D — Phase 2 Zero-shot Evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--matching",
        type=str,
        default="single",
        choices=MATCHING_STRATEGIES,
        help=f"Matching 전략 (기본: single). 선택: {MATCHING_STRATEGIES}",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="k-reciprocal re-ranking 후처리 적용 (k1=20, k2=6, λ=0.3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="모델 이름 (예: osnet_x1_0, osnet_ain_x1_0). 미지정 시 config.yaml 값 사용",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="가중치 파일 경로 (예: models/osnet_ain_x1_0_msmt17.pth). 미지정 시 config.yaml 값 사용",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] 설정 파일 없음: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    reid_cfg = config.get("reid", {})
    if args.model:
        reid_cfg["model_name"] = args.model
    if args.weights:
        reid_cfg["weights_path"] = args.weights

    model_name   = reid_cfg.get("model_name", "osnet_x1_0")
    weights_path = reid_cfg.get("weights_path") or "ImageNet pretrained"
    rerank_str   = "rerank" if args.rerank else "no-rerank"
    print("=" * 70)
    print("  Phase 2: Zero-shot Re-ID Evaluation")
    print(f"  모델·가중치 : {model_name}  |  {weights_path}")
    print(f"  옵션        : matching={args.matching}  |  {rerank_str}"
          f"  |  data={config.get('market1501', {}).get('output_dir', 'data/market1501')}")
    print("=" * 70)

    try:
        extractor = OSNetExtractor(
            model_name=reid_cfg.get("model_name", "osnet_x1_0"),
            pretrained=reid_cfg.get("pretrained", True),
            weights_path=reid_cfg.get("weights_path", None),
            device=reid_cfg.get("device", "auto"),
        )
    except (RuntimeError, FileNotFoundError) as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        sys.exit(1)

    evaluator = MarketEvaluator(config)
    results = evaluator.run(extractor, matching=args.matching, rerank=args.rerank)

    cmc_str = "  ".join(f"{k}={v:.2f}%" for k, v in results["CMC"].items())
    suffix  = f"{args.matching}_rerank" if args.rerank else args.matching
    print()
    print("=" * 70)
    print(f"  [{model_name}  |  {weights_path}]")
    print(f"  Query {results['num_query']}개/{results['num_query_ids']} ID  |  "
          f"Gallery {results['num_gallery']}개  |  "
          f"mAP {results['mAP']:.2f}%  |  {cmc_str}")
    print(f"  결과 저장: {evaluator.output_dir / f'results_{suffix}.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
