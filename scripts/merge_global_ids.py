"""
scripts/merge_global_ids.py
============================
같은 사람인데 다른 global_id로 나뉜 트랙렛을 수동으로 병합.
--src 의 global_id를 --dst 로 흡수시킵니다.

사용법:
    # global_id 7번을 3번으로 병합 (7번 트랙렛들이 3번으로 바뀜)
    python scripts/merge_global_ids.py --src 7 --dst 3

    # 확인만 (실제 변경 없음)
    python scripts/merge_global_ids.py --src 7 --dst 3 --dry-run

    # 특정 디렉토리 지정
    python scripts/merge_global_ids.py --src 7 --dst 3 --filtered-dir data/filtered
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="두 global_id 수동 병합")
    parser.add_argument("--src", type=int, required=True,
                        help="흡수될 global_id (이 ID가 dst로 바뀜)")
    parser.add_argument("--dst", type=int, required=True,
                        help="유지될 global_id")
    parser.add_argument("--filtered-dir", default="data/filtered",
                        help="트랙렛 디렉토리 (기본: data/filtered)")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 변경 없이 대상만 출력")
    args = parser.parse_args()

    if args.src == args.dst:
        sys.exit("[ERROR] --src 와 --dst 가 같습니다.")

    filtered_dir = Path(args.filtered_dir)
    if not filtered_dir.is_absolute():
        filtered_dir = project_root / filtered_dir
    if not filtered_dir.exists():
        sys.exit(f"[ERROR] 디렉토리 없음: {filtered_dir}")

    # src global_id를 가진 트랙렛 탐색
    meta_paths = sorted(filtered_dir.glob("*/track_*/metadata.json"))
    targets = []
    for mp in meta_paths:
        meta = json.loads(mp.read_text(encoding="utf-8"))
        if meta.get("global_id") == args.src:
            targets.append((mp, meta))

    if not targets:
        sys.exit(f"[ERROR] global_id={args.src} 인 트랙렛이 없습니다.")

    print(f"[INFO] global_id {args.src} → {args.dst} 병합")
    print(f"[INFO] 대상 트랙렛 {len(targets)}개:")
    for mp, meta in targets:
        tdir = mp.parent
        print(f"  {tdir.parent.name}/{tdir.name}  "
              f"(cam={meta.get('camera_id')}, slot={meta.get('time_slot')}, "
              f"length={meta.get('length')})")

    if args.dry_run:
        print("\n[DRY-RUN] 실제 변경 없음")
        return

    # global_id 변경
    for mp, meta in targets:
        meta["global_id"] = args.dst
        mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[완료] {len(targets)}개 트랙렛의 global_id를 {args.src} → {args.dst} 변경")
    print(f"  gather_by_person.py 를 재실행하면 결과에 반영됩니다.")


if __name__ == "__main__":
    main()
