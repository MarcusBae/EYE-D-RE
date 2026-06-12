"""
scripts/run_tracking.py
=======================
모든 CCTV 영상에 대해 YOLOv8 + ByteTrack 트래킹을 실행하여 tracklet을 추출하는 스크립트.

실행 예:
  python scripts/run_tracking.py --config configs/config.yaml --frame-stride 6
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

# pipeline 패키지를 임포트하기 위해 path 추가
sys.path.append(str(Path(__file__).parent.parent))

from pipeline.video_utils import load_config, load_videos_from_config
from pipeline.tracker import PersonTracker, VIDEO_PROGRESS_FILE

CHECKPOINT_FILENAME = "tracking_checkpoint.json"


# ── 영상 간 체크포인트 (완료된 영상 목록) ──────────────────────────────────

def _load_checkpoint(tracklet_dir: str) -> dict | None:
    path = Path(tracklet_dir) / CHECKPOINT_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_checkpoint(tracklet_dir: str, checkpoint: dict) -> None:
    path = Path(tracklet_dir) / CHECKPOINT_FILENAME
    Path(tracklet_dir).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)


def _remove_checkpoint(tracklet_dir: str) -> None:
    path = Path(tracklet_dir) / CHECKPOINT_FILENAME
    if path.exists():
        path.unlink()


def _checkpoint_matches(checkpoint: dict, config_path: str, frame_stride: int, save_crops: bool) -> bool:
    return (
        checkpoint.get("config") == config_path
        and checkpoint.get("frame_stride") == frame_stride
        and checkpoint.get("save_crops") == save_crops
    )


# ── 영상 내 체크포인트 (트래킹 중간 진행 상황) ──────────────────────────

def _load_video_progress(tracklet_dir: str, item: dict) -> dict | None:
    path = Path(tracklet_dir) / f"c{item['camera']}_t{item['slot']}" / VIDEO_PROGRESS_FILE
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _clear_video_dir(tracklet_dir: str, item: dict) -> None:
    """영상 출력 디렉토리 전체 삭제 (새로 시작 시 이전 부분 결과 제거)."""
    video_dir = Path(tracklet_dir) / f"c{item['camera']}_t{item['slot']}"
    if video_dir.exists():
        shutil.rmtree(video_dir)
        print(f"   [초기화] 이전 부분 결과 삭제: {video_dir}")


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
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="처리된 프레임 N개마다 영상 내 체크포인트 저장. 0=비활성 (기본: 500)"
    )
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

    save_crops = not args.no_crops

    # ── 영상 간 체크포인트 확인 ───────────────────────────────────────────
    checkpoint = _load_checkpoint(tracklet_dir)
    completed: set[str] = set()
    accumulated_stats: dict = {}

    if checkpoint is not None:
        if _checkpoint_matches(checkpoint, args.config, args.frame_stride, save_crops):
            done = checkpoint.get("completed", [])
            if done:
                print(f"\n[체크포인트 발견] 이전 실행에서 {len(done)}개 영상이 완료되었습니다:")
                for fname in done:
                    print(f"   - {fname}")
                answer = input("\n이어서 실행하시겠습니까? [y/N] ").strip().lower()
                if answer == "y":
                    completed = set(done)
                    accumulated_stats = checkpoint.get("stats", {})
                    print(f"[Resume] {len(completed)}개 영상을 건너뜁니다.")
                else:
                    print("[Fresh] 처음부터 새로 실행합니다.")
                    _remove_checkpoint(tracklet_dir)
            else:
                _remove_checkpoint(tracklet_dir)
        else:
            print("[체크포인트 발견] 설정이 달라 이전 체크포인트를 무시하고 새로 시작합니다.")
            _remove_checkpoint(tracklet_dir)

    # 새 체크포인트 초기화 (resume 아닌 경우)
    if not completed and not accumulated_stats:
        _save_checkpoint(tracklet_dir, {
            "config": args.config,
            "frame_stride": args.frame_stride,
            "save_crops": save_crops,
            "completed": [],
            "stats": {},
        })

    # ── 트래커 로드 ────────────────────────────────────────────────────────
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

    print(f"\n[Run] 객체 추적 & Tracklet 추출을 시작합니다... (stride={args.frame_stride}, checkpoint_every={args.checkpoint_every})")

    success_count = len(completed)
    total_tracklets = sum(s.get("num_tracklets", 0) for s in accumulated_stats.values())

    for item in video_items:
        filename = item["filename"]

        if filename in completed:
            print(f"\n - {filename} [건너뜀 — 이미 완료]")
            continue

        if not item["exists"]:
            print(f"[MISSING] 영상 없음: {filename} -> 스킵")
            continue

        # ── 영상 내 체크포인트 확인 ─────────────────────────────────────
        video_progress = _load_video_progress(tracklet_dir, item)
        resume_from = None

        if video_progress:
            fi = video_progress["frame_idx"]
            fc = video_progress["flushed_count"]
            print(f"\n - {filename}")
            print(f"   [영상 내 체크포인트] 프레임 {fi} 까지 처리됨 (저장된 tracklet {fc}개)")
            answer = input("   이어서 실행하시겠습니까? [y/N] ").strip().lower()
            if answer == "y":
                resume_from = video_progress
            else:
                _clear_video_dir(tracklet_dir, item)
        else:
            print(f"\n - {filename} 트래킹 중...")

        try:
            stats = tracker.track_video(
                video_path=item["path"],
                camera_id=item["camera"],
                time_slot=item["slot"],
                output_dir=tracklet_dir,
                frame_stride=args.frame_stride,
                save_crops=save_crops,
                checkpoint_every=args.checkpoint_every,
                resume_from=resume_from,
            )
            print(f"   -> 완료: {stats['num_tracklets']}개 tracklet 저장됨 (경로: {stats['output_dir']})")
            success_count += 1
            total_tracklets += stats["num_tracklets"]

            # 영상 간 체크포인트 업데이트
            completed.add(filename)
            accumulated_stats[filename] = stats
            _save_checkpoint(tracklet_dir, {
                "config": args.config,
                "frame_stride": args.frame_stride,
                "save_crops": save_crops,
                "completed": list(completed),
                "stats": accumulated_stats,
            })

        except Exception as e:
            print(f"[ERR] {filename} 트래킹 실패: {e}")

    print(f"\n[Finished] 작업 완료. (성공 영상={success_count}, 추출된 총 tracklet={total_tracklets})")

    # 모든 영상 처리 완료 시 체크포인트 정리
    total_existing = sum(1 for item in video_items if item["exists"])
    if success_count >= total_existing:
        _remove_checkpoint(tracklet_dir)
        print("[체크포인트 삭제] 모든 영상 처리 완료.")


if __name__ == "__main__":
    main()
