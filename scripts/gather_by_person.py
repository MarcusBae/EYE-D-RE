"""
scripts/gather_by_person.py
============================
Global ID별로 모든 카메라/슬롯의 트랙렛 이미지를 한 폴더로 수집합니다.
수동 편집 작업을 인물 단위로 할 수 있도록 도와줍니다.

출력 구조:
    data/persons/
    ├── person_0001/
    │   ├── source_map.json          ← 역추적용 매핑 파일
    │   ├── c2_t1_tk0003_frame000114.jpg
    │   ├── c2_t1_tk0003_frame001482.jpg
    │   └── c3_t2_tk0017_frame005208.jpg
    └── person_0002/
        └── ...

사용법:
    python scripts/gather_by_person.py --tracklets-dir data/tracklets-2-manual-edit
    python scripts/gather_by_person.py --tracklets-dir data/tracklets-2-manual-edit \\
        --out data/persons --global-id 5  # 특정 인물만
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


def encode_filename(cam: int, slot: int, track_id: int, orig_name: str) -> str:
    return f"c{cam}_t{slot}_tk{track_id:04d}_{orig_name}"


def gather(tracklets_dir: Path, out_dir: Path, global_id_filter: int = None):
    meta_paths = sorted(tracklets_dir.glob("*/track_*/metadata.json"))
    if not meta_paths:
        sys.exit(f"[ERROR] 트랙렛이 없습니다: {tracklets_dir}")

    # global_id → 트랙렛 목록 수집
    clusters: dict[int, list] = {}
    for mp in meta_paths:
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        gid = meta.get("global_id")
        if gid is None:
            continue
        if global_id_filter is not None and gid != global_id_filter:
            continue
        clusters.setdefault(gid, []).append((mp.parent, meta))

    if not clusters:
        sys.exit(f"[ERROR] Global ID가 없습니다. merge_ids.py를 먼저 실행하세요.")

    print(f"[INFO] Global ID {len(clusters)}개 발견")

    total_copied = 0
    for gid, tracklets in sorted(clusters.items()):
        person_dir = out_dir / f"person_{gid:04d}"
        person_dir.mkdir(parents=True, exist_ok=True)

        source_map = {}  # 새 파일명 → 원본 절대 경로

        for tdir, meta in tracklets:
            cam = meta.get("camera_id", 0)
            slot = meta.get("time_slot", 0)
            track_id = meta.get("track_id", 0)
            crop_files = meta.get("crop_files", [])

            for fname in crop_files:
                src = tdir / fname
                if not src.exists():
                    continue
                new_name = encode_filename(cam, slot, track_id, fname)
                dst = person_dir / new_name
                shutil.copy2(str(src), str(dst))
                source_map[new_name] = str(src.resolve())
                total_copied += 1

        # source_map 저장
        (person_dir / "source_map.json").write_text(
            json.dumps(source_map, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        img_count = len([f for f in person_dir.iterdir() if f.suffix == ".jpg"])
        print(f"  person_{gid:04d}: {img_count}장  ({len(tracklets)}개 트랙렛)")

    print(f"\n[완료] 총 {total_copied}장 → {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="인물별 이미지 수집 (수동 편집용)")
    parser.add_argument("--tracklets-dir", required=True,
                        help="트랙렛 루트 디렉토리 (예: data/tracklets-2-manual-edit)")
    parser.add_argument("--out", default="data/persons",
                        help="출력 디렉토리 (기본: data/persons)")
    parser.add_argument("--global-id", type=int, default=None,
                        help="특정 Global ID만 수집 (미지정 시 전체)")
    args = parser.parse_args()

    tracklets_dir = Path(args.tracklets_dir)
    if not tracklets_dir.exists():
        sys.exit(f"[ERROR] 경로 없음: {tracklets_dir}")

    gather(tracklets_dir, Path(args.out), args.global_id)


if __name__ == "__main__":
    main()
