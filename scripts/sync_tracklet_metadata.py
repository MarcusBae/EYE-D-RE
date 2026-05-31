"""
scripts/sync_tracklet_metadata.py
===================================
트랙렛 폴더에서 이미지를 수동 삭제한 뒤 metadata.json을 실제 파일 기준으로 재동기화합니다.

사용법:
    # 특정 트랙렛 하나
    python scripts/sync_tracklet_metadata.py data/raw_tracklets/c2_t10/track_0042

    # 카메라/슬롯 전체
    python scripts/sync_tracklet_metadata.py data/raw_tracklets/c2_t10

    # raw_tracklets 전체
    python scripts/sync_tracklet_metadata.py data/raw_tracklets

    # 삭제 없이 불일치만 보고 (dry-run)
    python scripts/sync_tracklet_metadata.py data/raw_tracklets --dry-run
"""

import argparse
import json
import sys
from pathlib import Path


MIN_FRAMES = 3  # 이 미만으로 남은 트랙렛은 폴더째 삭제


def sync_one(tdir: Path, dry_run: bool) -> str:
    """
    트랙렛 디렉토리 1개를 동기화.
    반환값: "ok" | "updated" | "deleted" | "skip"
    """
    meta_path = tdir / "metadata.json"
    if not meta_path.exists():
        return "skip"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    existing = {p.name for p in tdir.glob("*.jpg")}
    original_count = len(meta.get("frame_indices", []))

    # (frame_idx, bbox, conf) 세트에서 이미지가 존재하는 것만 남김
    kept = [
        (fi, bb, cf)
        for fi, bb, cf in zip(
            meta.get("frame_indices", []),
            meta.get("bboxes", []),
            meta.get("confidences", []),
        )
        if f"frame_{fi:06d}.jpg" in existing
    ]

    if len(kept) == original_count:
        return "ok"

    if len(kept) < MIN_FRAMES:
        # 트랙렛이 너무 짧아졌으면 폴더째 삭제
        if not dry_run:
            import shutil
            shutil.rmtree(tdir)
        return "deleted"

    # metadata 갱신
    import numpy as np
    confs = [cf for _, _, cf in kept]
    meta["frame_indices"] = [fi for fi, _, _ in kept]
    meta["bboxes"] = [bb for _, bb, _ in kept]
    meta["confidences"] = confs
    meta["length"] = len(kept)
    meta["avg_conf"] = float(sum(confs) / len(confs))

    if not dry_run:
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return "updated"


def collect_tracklet_dirs(root: Path) -> list[Path]:
    """주어진 경로가 트랙렛 폴더인지, 상위 폴더인지 판별해서 트랙렛 목록 반환"""
    if (root / "metadata.json").exists():
        return [root]
    return sorted(p.parent for p in root.glob("*/track_*/metadata.json"))


def main():
    parser = argparse.ArgumentParser(description="트랙렛 metadata 재동기화")
    parser.add_argument("path", help="트랙렛 디렉토리 또는 상위 디렉토리")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 변경 없이 불일치 현황만 출력")
    parser.add_argument("--min-frames", type=int, default=MIN_FRAMES,
                        help=f"트랙렛 최소 프레임 수 (기본: {MIN_FRAMES}, 미만 시 폴더 삭제)")
    args = parser.parse_args()

    global MIN_FRAMES
    MIN_FRAMES = args.min_frames

    root = Path(args.path)
    if not root.exists():
        sys.exit(f"[ERROR] 경로를 찾을 수 없습니다: {root}")

    tracklet_dirs = collect_tracklet_dirs(root)
    if not tracklet_dirs:
        sys.exit(f"[ERROR] 트랙렛 디렉토리가 없습니다: {root}")

    print(f"{'[DRY-RUN] ' if args.dry_run else ''}트랙렛 {len(tracklet_dirs)}개 동기화 시작\n")

    counts = {"ok": 0, "updated": 0, "deleted": 0, "skip": 0}
    for tdir in tracklet_dirs:
        result = sync_one(tdir, args.dry_run)
        counts[result] += 1
        if result in ("updated", "deleted"):
            tag = "[삭제]" if result == "deleted" else "[갱신]"
            print(f"  {tag} {tdir.parent.name}/{tdir.name}")

    print()
    print("=" * 40)
    print(f"  정상(변경 없음) : {counts['ok']}개")
    print(f"  metadata 갱신   : {counts['updated']}개")
    print(f"  폴더 삭제       : {counts['deleted']}개  (프레임 {MIN_FRAMES}개 미만)")
    print(f"  건너뜀          : {counts['skip']}개  (metadata.json 없음)")
    if args.dry_run:
        print("\n  ※ --dry-run 모드: 실제 변경 없음")
    print("=" * 40)


if __name__ == "__main__":
    main()
