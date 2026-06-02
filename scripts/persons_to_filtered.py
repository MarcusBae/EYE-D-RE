"""
scripts/persons_to_filtered.py
================================
data/persons/ 폴더를 수동으로 편집한 후,
그 결과를 data/filtered/ 형식(트랙렛 구조 + metadata.json with global_id)으로 재구성합니다.

동작 방식:
  1. persons/ 하위 각 person_XXXX/ 폴더를 하나의 Global ID로 취급합니다.
  2. 각 폴더 내 source_map.json 을 이용해 이미지의 원본 트랙렛 경로를 역추적합니다.
  3. 원본 트랙렛 단위로 묶어 filtered/ 에 복사하고 metadata.json 의 global_id 를 갱신합니다.
  4. persons 폴더에서 삭제된 이미지는 filtered 에도 포함하지 않습니다.

수동 편집 지침:
  - 같은 사람을 하나의 person_XXXX 폴더로 이미지를 직접 이동/복사하세요.
  - 다른 사람이 섞인 경우 해당 이미지를 다른 person_XXXX 폴더로 이동하거나 삭제하세요.
  - 폴더 이름(person_XXXX)이 Global ID가 됩니다. 번호는 임의로 변경 가능합니다.
  - person_-001/ 폴더는 미분류(오병합 의심) 이미지용으로 자동 무시됩니다.

사용법:
    python scripts/persons_to_filtered.py
    python scripts/persons_to_filtered.py --persons-dir data/persons --out data/filtered
    python scripts/persons_to_filtered.py --dry-run   # 실제 복사 없이 예상 결과만 출력
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path


def load_source_map(person_dir: Path) -> dict[str, str]:
    """source_map.json 로드. 없으면 빈 딕셔너리 반환."""
    sm_path = person_dir / "source_map.json"
    if not sm_path.exists():
        return {}
    try:
        return json.loads(sm_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def person_name_to_id(name: str) -> int | None:
    """
    person_0001 → 1
    person_-001 → -1  (미분류 폴더, 무시 대상)
    인식 불가 → None
    """
    m = re.match(r"^person_(-?\d+)_?$", name)
    if m:
        return int(m.group(1))
    return None


def build_tracklet_map(
    person_dir: Path,
    global_id: int,
    source_map: dict[str, str],
) -> dict[str, dict]:
    """
    person 폴더에 현재 남아 있는 이미지들을 원본 트랙렛별로 그룹화.

    반환:
        { orig_tracklet_abs_path: { "files": [파일명, ...], "global_id": int } }
    """
    # source_map: 새 파일명(c2_t1_tk0003_frame000114.jpg) → 원본 절대 경로
    existing_images = {f.name for f in person_dir.iterdir() if f.suffix in (".jpg", ".png", ".jpeg")}

    tracklet_groups: dict[str, list[str]] = {}
    unmapped: list[str] = []

    for img_name in sorted(existing_images):
        orig_path = source_map.get(img_name)
        if orig_path is None:
            unmapped.append(img_name)
            continue
        orig_file = Path(orig_path)
        tracklet_dir_str = str(orig_file.parent)
        tracklet_groups.setdefault(tracklet_dir_str, []).append(orig_file.name)

    if unmapped:
        print(f"  [WARN] person_{global_id:04d}: source_map에 없는 이미지 {len(unmapped)}장 무시됨 "
              f"(수동 추가 파일이면 source_map을 직접 보완하세요)")

    return {
        tdir: {"files": sorted(fnames), "global_id": global_id}
        for tdir, fnames in tracklet_groups.items()
    }


def copy_tracklet(
    src_tracklet_dir: Path,
    dst_tracklet_dir: Path,
    files_to_copy: list[str],
    global_id: int,
    dry_run: bool,
) -> int:
    """트랙렛 이미지와 metadata.json 을 dst_tracklet_dir 에 복사하고 global_id 기입."""
    if not dry_run:
        dst_tracklet_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for fname in files_to_copy:
        src = src_tracklet_dir / fname
        if not src.exists():
            print(f"    [SKIP] 원본 없음: {src}")
            continue
        if not dry_run:
            shutil.copy2(str(src), str(dst_tracklet_dir / fname))
        copied += 1

    # metadata.json 복사 후 global_id 갱신
    meta_src = src_tracklet_dir / "metadata.json"
    if meta_src.exists():
        try:
            meta = json.loads(meta_src.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        meta["global_id"] = global_id
        # 실제 복사된 파일 목록만 crop_files 에 반영
        meta["crop_files"] = sorted(files_to_copy)
        meta["num_frames"] = len(files_to_copy)
        if not dry_run:
            (dst_tracklet_dir / "metadata.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    else:
        print(f"    [WARN] metadata.json 없음: {src_tracklet_dir}")

    return copied


def resolve_cam_slot_track(tracklet_abs: str, out_dir: Path) -> Path:
    """
    원본 트랙렛 절대 경로로부터 filtered/ 하위 상대 경로(cx_ty/track_NNNN)를 결정합니다.
    예: /home/.../data/tracklets/c2_t1/track_0003 → filtered/c2_t1/track_0003
    """
    p = Path(tracklet_abs)
    # 마지막 두 파트: cx_ty / track_NNNN
    cam_slot = p.parent.name   # 예: c2_t1
    track    = p.name          # 예: track_0003
    return out_dir / cam_slot / track


def run(persons_dir: Path, out_dir: Path, dry_run: bool):
    person_dirs = sorted(
        [p for p in persons_dir.iterdir() if p.is_dir() and p.name.startswith("person")],
        key=lambda p: p.name,
    )

    if not person_dirs:
        sys.exit(f"[ERROR] persons 폴더가 없습니다: {persons_dir}")

    print(f"[INFO] persons 디렉토리: {persons_dir.resolve()}")
    print(f"[INFO] filtered 출력: {out_dir.resolve()}")
    print(f"[INFO] dry-run: {dry_run}")
    print()

    if not dry_run:
        # 기존 출력 디렉토리 초기화 여부 확인
        if out_dir.exists():
            print(f"[WARN] '{out_dir}' 폴더가 이미 존재합니다.")
            ans = input("  기존 내용을 삭제하고 새로 생성할까요? [y/N] ").strip().lower()
            if ans == "y":
                shutil.rmtree(out_dir)
                print(f"  → 삭제 완료")
            else:
                print("  → 기존 폴더 유지 (이미지가 덮어씌워질 수 있습니다)")

    total_persons = 0
    total_tracklets = 0
    total_images = 0
    skipped_persons = []

    for pdir in person_dirs:
        global_id = person_name_to_id(pdir.name)
        if global_id is None:
            print(f"[SKIP] 폴더명 인식 불가: {pdir.name}")
            continue
        if global_id < 0:
            print(f"[SKIP] 미분류 폴더 무시: {pdir.name}")
            continue

        source_map = load_source_map(pdir)
        tracklet_map = build_tracklet_map(pdir, global_id, source_map)

        if not tracklet_map:
            skipped_persons.append(pdir.name)
            print(f"  {pdir.name}: 이미지 없음 (건너뜀)")
            continue

        n_imgs = sum(len(v["files"]) for v in tracklet_map.values())
        print(f"  {pdir.name}  →  global_id={global_id}  "
              f"({len(tracklet_map)}개 트랙렛, {n_imgs}장)")

        for tracklet_abs, info in sorted(tracklet_map.items()):
            src_dir = Path(tracklet_abs)
            dst_dir = resolve_cam_slot_track(tracklet_abs, out_dir)

            if dry_run:
                print(f"    [DRY] {src_dir.parent.name}/{src_dir.name}  "
                      f"→  {dst_dir.relative_to(out_dir)}  "
                      f"({len(info['files'])}장)")
            else:
                n = copy_tracklet(src_dir, dst_dir, info["files"], global_id, dry_run=False)
                print(f"    {src_dir.parent.name}/{src_dir.name}  →  "
                      f"{dst_dir.relative_to(out_dir)}  ({n}장 복사)")
            total_tracklets += 1
            total_images += len(info["files"])

        total_persons += 1

    print()
    print("=" * 60)
    if dry_run:
        print(f"  [DRY-RUN 결과] 처리 예정: "
              f"{total_persons}명 / {total_tracklets}개 트랙렛 / {total_images}장")
    else:
        print(f"  [완료] {total_persons}명 / {total_tracklets}개 트랙렛 / {total_images}장")
        print(f"  → {out_dir.resolve()}")
    if skipped_persons:
        print(f"  건너뛴 폴더: {', '.join(skipped_persons)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="persons/ 수동 편집 결과를 filtered/ 트랙렛 구조로 역변환"
    )
    parser.add_argument(
        "--persons-dir", default="data/persons",
        help="persons 루트 디렉토리 (기본: data/persons)",
    )
    parser.add_argument(
        "--out", default="data/filtered",
        help="출력 디렉토리 (기본: data/filtered)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="실제 복사 없이 예상 결과만 출력",
    )
    args = parser.parse_args()

    persons_dir = Path(args.persons_dir)
    if not persons_dir.exists():
        sys.exit(f"[ERROR] persons 디렉토리 없음: {persons_dir}")

    run(persons_dir, Path(args.out), args.dry_run)


if __name__ == "__main__":
    main()
