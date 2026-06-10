"""
scripts/extract_frames.py
=========================
모든 CCTV 영상에서 프레임을 샘플링(5fps)하여 저장하는 실행 스크립트.

실행 예:
  python scripts/extract_frames.py --config configs/config.yaml
"""

import argparse
import sys
from pathlib import Path

# pipeline 패키지를 임포트하기 위해 path 추가
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.video_utils import (
    load_config,
    load_videos_from_config,
    extract_frames,
    print_video_summary
)


def main():
    parser = argparse.ArgumentParser(description="EYE-D — Frame Extraction")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--video", type=str, default=None, help="Specific video filename to extract (e.g., cam1_t1.avi)")
    parser.add_argument("--fps", type=int, default=None, help="Target FPS for frame extraction (defaults to config value)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing frames")
    args = parser.parse_args()

    config = load_config(args.config)
    print_video_summary(config)

    video_items = load_videos_from_config(config)
    
    if args.video:
        filtered_items = [item for item in video_items if item["filename"] == args.video]
        if not filtered_items:
            print(f"[Error] 지정한 비디오 '{args.video}'가 설정 파일에 존재하지 않습니다.")
            sys.exit(1)
        video_items = filtered_items
    
    # data/frames 폴더 경로
    frame_dir = Path(config["data"]["frame_dir"])
    target_fps = args.fps if args.fps is not None else config["frame_extraction"]["target_fps"]
    img_fmt = config["frame_extraction"]["image_format"]
    quality = config["frame_extraction"]["jpg_quality"]

    print(f"\n[Run] 프레임 추출을 시작합니다... (target_fps={target_fps}, format={img_fmt})")
    
    missing_count = 0
    extracted_count = 0

    for item in video_items:
        if not item["exists"]:
            print(f"[MISSING] 영상 없음: {item['filename']} -> 스킵")
            missing_count += 1
            continue

        vpath = item["path"]
        cam = item["camera"]
        slot = item["slot"]
        
        # c{cam}_t{slot} 별 개별 폴더
        out_subdir = frame_dir / f"c{cam}_t{slot}"

        try:
            meta = extract_frames(
                video_path=vpath,
                output_dir=str(out_subdir),
                target_fps=target_fps,
                camera_id=cam,
                time_slot=slot,
                image_format=img_fmt,
                jpg_quality=quality,
                overwrite=args.overwrite
            )
            print(f" - {item['filename']} -> {meta['extracted_frames']}장 완료 (L={meta['duration_sec']}s)")
            extracted_count += 1
        except Exception as e:
            print(f"[ERR] {item['filename']} 처리 실패: {e}")

    print(f"\n[Finished] 작업 완료. (성공={extracted_count}, 스킵/오류={missing_count})")


if __name__ == "__main__":
    main()
