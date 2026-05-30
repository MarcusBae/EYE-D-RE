"""
scripts/run_tracking.py
=======================
모든 CCTV 영상에 대해 YOLOv8 + ByteTrack 트래킹을 실행하여 tracklet을 추출하는 스크립트.

실행 예:
  python scripts/run_tracking.py --config configs/config.yaml --frame-stride 6
"""

import argparse
import sys
from pathlib import Path

# pipeline 패키지를 임포트하기 위해 path 추가
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.video_utils import load_config, load_videos_from_config
from pipeline.tracker import PersonTracker


def main():
    parser = argparse.ArgumentParser(description="EYE-D — Tracking")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument(
        "--frame-stride", 
        type=int, 
        default=6, 
        help="Frame stride for tracking. (e.g., 30fps video with stride 6 simulates 5fps tracking)"
    )
    parser.add_argument("--no-crops", action="store_true", help="Do not save crop images (only metadata)")
    args = parser.parse_args()

    config = load_config(args.config)
    video_items = load_videos_from_config(config)

    # 파라미터 로드
    model_path = config["detection"]["model"]
    conf_thresh = config["detection"]["conf_threshold"]
    iou_thresh = config["detection"]["iou_threshold"]
    classes = config["detection"]["classes"]
    imgsz = config["detection"]["imgsz"]
    device = config["detection"]["device"]
    tracker_yaml = config["tracking"]["tracker_yaml"]
    tracklet_dir = config["data"]["tracklet_dir"]

    # Tracker 로드
    tracker = PersonTracker(
        model_path=model_path,
        tracker_yaml=tracker_yaml,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=imgsz,
        classes=classes,
        device=device,
        verbose=False
    )

    print(f"\n[Run] 객체 추적 & Tracklet 추출을 시작합니다... (stride={args.frame_stride})")
    
    success_count = 0
    total_tracklets = 0

    for item in video_items:
        if not item["exists"]:
            print(f"[MISSING] 영상 없음: {item['filename']} -> 스킵")
            continue

        print(f"\n - {item['filename']} 트래킹 중...")
        try:
            stats = tracker.track_video(
                video_path=item["path"],
                camera_id=item["camera"],
                time_slot=item["slot"],
                output_dir=tracklet_dir,
                frame_stride=args.frame_stride,
                save_crops=not args.no_crops
            )
            print(f"   -> 완료: {stats['num_tracklets']}개 tracklet 저장됨 (경로: {stats['output_dir']})")
            success_count += 1
            total_tracklets += stats["num_tracklets"]
        except Exception as e:
            print(f"[ERR] {item['filename']} 트래킹 실패: {e}")

    print(f"\n[Finished] 작업 완료. (성공 영상={success_count}, 추출된 총 tracklet={total_tracklets})")


if __name__ == "__main__":
    main()
