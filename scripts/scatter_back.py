"""
scripts/scatter_back.py
========================
gather_by_person.py로 수집 후 수동 편집한 결과를
원본 트랙렛 폴더에 반영하고 metadata.json을 재동기화합니다.

동작:
    1. source_map.json과 현재 person 폴더의 jpg를 비교
    2. person 폴더에서 삭제된 이미지 → 원본 트랙렛 폴더에서도 삭제
    3. 영향받은 트랙렛의 metadata.json 재동기화
    4. 프레임 수가 min_frames 미만으로 남은 트랙렛 폴더 삭제

사용법:
    python scripts/scatter_back.py --persons-dir data/persons
    python scripts/scatter_back.py --persons-dir data/persons --dry-run
    python scripts/scatter_back.py --persons-dir data/persons --global-id 5
"""

import argparse
import json
import sys
from pathlib import Path

MIN_FRAMES = 3


def sync_metadata(tdir: Path) -> str:
    """트랙렛 폴더의 metadata.json을 실제 jpg 기준으로 재동기화."""
    meta_path = tdir / "metadata.json"
    if not meta_path.exists():
        return "no_meta"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    existing = {p.name for p in tdir.glob("*.jpg")}

    kept = [
        (fi, bb, cf)
        for fi, bb, cf in zip(
            meta.get("frame_indices", []),
            meta.get("bboxes", []),
            meta.get("confidences", []),
        )
        if f"frame_{fi:06d}.jpg" in existing
    ]

    # crop_files 필드 기반 동기화 (frame_indices 대신 crop_files 쓰는 경우)
    crop_files = meta.get("crop_files", [])
    if crop_files:
        kept_crops = [f for f in crop_files if (tdir / f).exists()]
        if len(kept_crops) == len(crop_files):
            return "ok"
        if len(kept_crops) < MIN_FRAMES:
            import shutil
            shutil.rmtree(tdir)
            return "deleted"
        confs = []
        frames = []
        bboxes_kept = []
        for f in kept_crops:
            try:
                idx = crop_files.index(f)
                frames.append(meta["frame_indices"][idx])
                bboxes_kept.append(meta["bboxes"][idx])
                confs.append(meta["confidences"][idx])
            except (ValueError, IndexError):
                pass
        meta["crop_files"] = kept_crops
        meta["frame_indices"] = frames
        meta["bboxes"] = bboxes_kept
        meta["confidences"] = confs
        meta["length"] = len(kept_crops)
        meta["avg_conf"] = float(sum(confs) / len(confs)) if confs else 0.0
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return "updated"

    return "ok"


def scatter(persons_dir: Path, dry_run: bool, global_id_filter: int = None):
    person_dirs = sorted(persons_dir.iterdir()) if persons_dir.exists() else []
    person_dirs = [p for p in person_dirs if p.is_dir() and p.name.startswith("person_")]

    if global_id_filter is not None:
        person_dirs = [p for p in person_dirs if p.name == f"person_{global_id_filter:04d}"]

    if not person_dirs:
        sys.exit(f"[ERROR] person 폴더가 없습니다: {persons_dir}")

    print(f"{'[DRY-RUN] ' if dry_run else ''}person 폴더 {len(person_dirs)}개 처리\n")

    total_deleted = 0
    affected_tracklets = set()

    for person_dir in person_dirs:
        source_map_path = person_dir / "source_map.json"
        if not source_map_path.exists():
            print(f"  [SKIP] {person_dir.name}: source_map.json 없음")
            continue

        source_map: dict = json.loads(source_map_path.read_text(encoding="utf-8"))
        current_jpgs = {f.name for f in person_dir.glob("*.jpg")}

        deleted_count = 0
        for new_name, orig_path_str in source_map.items():
            if new_name not in current_jpgs:
                # person 폴더에서 삭제됨 → 원본도 삭제
                orig_path = Path(orig_path_str)
                if orig_path.exists():
                    if not dry_run:
                        orig_path.unlink()
                    deleted_count += 1
                    affected_tracklets.add(orig_path.parent)

        total_deleted += deleted_count
        if deleted_count > 0:
            print(f"  {person_dir.name}: {deleted_count}장 원본 삭제")

    # 영향받은 트랙렛 metadata 재동기화
    if not dry_run and affected_tracklets:
        print(f"\n[INFO] 영향받은 트랙렛 {len(affected_tracklets)}개 metadata 재동기화...")
        sync_stats = {"ok": 0, "updated": 0, "deleted": 0}
        for tdir in sorted(affected_tracklets):
            result = sync_metadata(tdir)
            sync_stats[result] = sync_stats.get(result, 0) + 1
            if result in ("updated", "deleted"):
                tag = "[삭제]" if result == "deleted" else "[갱신]"
                print(f"  {tag} {tdir.parent.name}/{tdir.name}")

        print(f"\n  metadata 갱신: {sync_stats.get('updated', 0)}개")
        print(f"  폴더 삭제 (프레임 부족): {sync_stats.get('deleted', 0)}개")

    print()
    print("=" * 45)
    print(f"  원본에서 삭제된 이미지 : {total_deleted}장")
    print(f"  영향받은 트랙렛       : {len(affected_tracklets)}개")
    if dry_run:
        print("\n  ※ --dry-run 모드: 실제 변경 없음")
    print("=" * 45)


def main():
    parser = argparse.ArgumentParser(description="수동 편집 결과를 원본 트랙렛에 반영")
    parser.add_argument("--persons-dir", default="data/persons",
                        help="person 폴더 루트 (기본: data/persons)")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 변경 없이 결과만 출력")
    parser.add_argument("--global-id", type=int, default=None,
                        help="특정 Global ID만 처리")
    args = parser.parse_args()

    persons_dir = Path(args.persons_dir)
    if not persons_dir.exists():
        sys.exit(f"[ERROR] 경로 없음: {persons_dir}")

    scatter(persons_dir, args.dry_run, args.global_id)


if __name__ == "__main__":
    main()
