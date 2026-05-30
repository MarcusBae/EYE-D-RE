"""
scripts/filter_tracklets.py
===========================
추출된 tracklet들을 품질 조건(길이, 해상도, 신뢰도 등)에 따라 필터링하여 정제된 데이터셋을 생성하고,
시각적 검증을 위한 썸네일 그리드를 시각화하는 실행 스크립트.

실행 예:
  python scripts/filter_tracklets.py --config configs/config.yaml --visualize
"""

import argparse
import sys
from pathlib import Path

# pipeline 패키지를 임포트하기 위해 path 추가
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.video_utils import load_config
from pipeline.tracklet_io import list_tracklets
from pipeline.quality_filter import TrackletQualityFilter, make_thumbnail_grid


def main():
    parser = argparse.ArgumentParser(description="EYE-D — Quality Filtering")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--visualize", action="store_true", help="Generate visual thumbnail grid for passed tracklets")
    args = parser.parse_args()

    config = load_config(args.config)

    # 경로 및 하이퍼파라미터 로딩
    tracklet_dir = config["data"]["tracklet_dir"]
    filtered_dir = config["data"]["filtered_dir"]

    q_cfg = config["quality_filter"]
    
    # 필터 생성
    q_filter = TrackletQualityFilter(
        min_length=q_cfg["min_track_length"],
        min_avg_conf=q_cfg["min_avg_confidence"],
        min_bbox_h=q_cfg["min_bbox_height"],
        min_bbox_w=q_cfg["min_bbox_width"],
        max_aspect_ratio=q_cfg["max_aspect_ratio"],
        min_area=q_cfg["min_bbox_area"]
    )

    print(f"\n[Run] Tracklet 품질 필터링 시작 (소스: {tracklet_dir} -> 목적지: {filtered_dir})")
    
    # 모든 tracklet 필터링 진행 및 복사
    stats = q_filter.filter_all(
        tracklet_dir=tracklet_dir,
        output_dir=filtered_dir,
        copy_crops=True,
        verbose=True
    )

    # 썸네일 시각화 생성 여부
    if args.visualize:
        print("\n[Run] 시각화용 썸네일 그리드 생성을 시작합니다...")
        passed_tracks = list_tracklets(filtered_dir)
        grid_out = Path(filtered_dir) / "passed_tracklets_grid.png"
        make_thumbnail_grid(
            tracklets=passed_tracks,
            output_path=str(grid_out),
            n_per_track=4,
            max_tracks=50, # 최대 50개 tracklet 그리드 렌더링
            title="Filter-Passed Tracklets (Grid View)"
        )

    print("\n[Finished] 필터링 작업 완료.")


if __name__ == "__main__":
    main()
