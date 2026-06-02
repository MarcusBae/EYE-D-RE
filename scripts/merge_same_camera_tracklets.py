"""
scripts/merge_same_camera_tracklets.py
========================================
동일 cam/slot 내에서 시간이 겹치지 않고 외형이 유사한 트랙렛을 병합.

트래커가 잠깐 놓쳤다 재획득한 경우(= 같은 사람인데 track_id가 나뉜 경우)를
복원하는 스크립트. split_switched_tracklets.py 실행 후에 사용.

병합 조건 (AND):
  1. 같은 cam/slot 디렉토리
  2. frame_indices 가 시간상 겹치지 않음
  3. 두 트랙렛 대표 임베딩의 cosine 유사도 >= --sim-threshold
  4. 두 트랙렛 사이 frame gap <= --max-gap-frames

사용법:
    python scripts/merge_same_camera_tracklets.py --config configs/config.yaml --dry-run
    python scripts/merge_same_camera_tracklets.py --config configs/config.yaml
    python scripts/merge_same_camera_tracklets.py --config configs/config.yaml --sim-threshold 0.80
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from pipeline.reid_merger import OSNetExtractor


SIM_THRESHOLD  = 0.75   # 대표 임베딩 cosine 유사도 하한
MAX_GAP_FRAMES = 300    # 두 트랙렛 사이 최대 허용 frame gap (24fps×stride6 → 약 75초)
MIN_FRAMES     = 5      # 병합 후 유지할 최소 프레임 수


# ── 대표 임베딩 추출 ───────────────────────────────────────────
def extract_representative(tdir: Path, crop_files: list[str],
                            extractor: OSNetExtractor) -> np.ndarray | None:
    """트랙렛 전체 프레임의 평균 L2 정규화 임베딩."""
    imgs = []
    for fname in crop_files:
        img = cv2.imread(str(tdir / fname))
        if img is not None and img.size > 0:
            imgs.append(img)
    if not imgs:
        return None
    feats = extractor.extract_batch_features(imgs)   # (N, D) L2 정규화
    mean = feats.mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


# ── 시간 겹침 / gap 계산 ──────────────────────────────────────
def time_overlap(frames_a: list[int], frames_b: list[int]) -> bool:
    return not (max(frames_a) < min(frames_b) or max(frames_b) < min(frames_a))


def frame_gap(frames_a: list[int], frames_b: list[int]) -> int:
    """시간상 앞선 트랙렛 끝과 뒤 트랙렛 시작의 frame 수 차이."""
    if max(frames_a) < min(frames_b):
        return min(frames_b) - max(frames_a)
    return min(frames_a) - max(frames_b)


# ── 실제 병합 ─────────────────────────────────────────────────
def merge_two(meta_a: dict, tdir_a: Path,
              meta_b: dict, tdir_b: Path,
              dry_run: bool) -> dict:
    """
    meta_b를 meta_a에 병합. 파일은 tdir_a로 이동, tdir_b 삭제.
    반환: 갱신된 meta_a dict
    """
    # 프레임 순서 기준 합치기
    pairs = list(zip(meta_a["frame_indices"], meta_a["crop_files"],
                     meta_a["bboxes"], meta_a["confidences"]))
    pairs += list(zip(meta_b["frame_indices"], meta_b["crop_files"],
                      meta_b["bboxes"], meta_b["confidences"]))
    pairs.sort(key=lambda x: x[0])

    new_frames, new_crops, new_bboxes, new_confs = zip(*pairs)
    new_frames = list(new_frames)
    new_crops  = list(new_crops)
    new_bboxes = list(new_bboxes)
    new_confs  = list(new_confs)

    if not dry_run:
        # tdir_b의 프레임 파일을 tdir_a로 이동
        for fname in meta_b["crop_files"]:
            src = tdir_b / fname
            dst = tdir_a / fname
            if src.exists():
                shutil.move(str(src), str(dst))

        # tdir_b 폴더 제거
        shutil.rmtree(tdir_b, ignore_errors=True)

        # meta_a 갱신
        meta_a["crop_files"]   = new_crops
        meta_a["frame_indices"] = new_frames
        meta_a["bboxes"]       = new_bboxes
        meta_a["confidences"]  = new_confs
        meta_a["length"]       = len(new_crops)
        meta_a["avg_conf"]     = float(np.mean(new_confs))
        (tdir_a / "metadata.json").write_text(
            json.dumps(meta_a, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    meta_a["crop_files"]    = new_crops
    meta_a["frame_indices"] = new_frames
    meta_a["bboxes"]        = new_bboxes
    meta_a["confidences"]   = new_confs
    meta_a["length"]        = len(new_crops)
    return meta_a


# ── cam/slot 단위 병합 처리 ───────────────────────────────────
def process_cam_slot(cam_slot_dir: Path, extractor: OSNetExtractor,
                     sim_threshold: float, max_gap: int,
                     min_frames: int, dry_run: bool) -> tuple[int, int]:
    """
    한 cam/slot 내 트랙렛들을 분석해 병합 후보를 찾고 실행.
    반환: (병합된 쌍 수, 삭제된 트랙렛 수)
    """
    track_dirs = sorted(cam_slot_dir.glob("track_*"))
    if len(track_dirs) < 2:
        return 0, 0

    # 모든 트랙렛 메타 + 대표 임베딩 로드
    tracklets: list[dict] = []
    for tdir in track_dirs:
        mp = tdir / "metadata.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text(encoding="utf-8"))
        if meta.get("length", 0) < min_frames:
            continue
        emb = extract_representative(tdir, meta.get("crop_files", []), extractor)
        if emb is None:
            continue
        tracklets.append({"meta": meta, "tdir": tdir, "emb": emb})

    if len(tracklets) < 2:
        return 0, 0

    # 병합된 트랙렛 추적 (greedy)
    merged_into: dict[int, int] = {}   # idx → 흡수된 대상 idx

    def root(i: int) -> int:
        while i in merged_into:
            i = merged_into[i]
        return i

    merge_count = deleted_count = 0

    # 유사도 행렬 계산 후 내림차순 정렬
    n = len(tracklets)
    candidates = []
    for i in range(n):
        for j in range(i + 1, n):
            fa = tracklets[i]["meta"]["frame_indices"]
            fb = tracklets[j]["meta"]["frame_indices"]
            if time_overlap(fa, fb):
                continue
            gap = frame_gap(fa, fb)
            if gap > max_gap:
                continue
            sim = float(tracklets[i]["emb"] @ tracklets[j]["emb"])
            if sim >= sim_threshold:
                candidates.append((sim, gap, i, j))

    candidates.sort(key=lambda x: (-x[0], x[1]))  # 유사도 높고 gap 짧은 순

    for sim, gap, i, j in candidates:
        ri, rj = root(i), root(j)
        if ri == rj:
            continue  # 이미 같은 그룹

        ti = tracklets[ri]
        tj = tracklets[rj]

        # 시간 순서로 앞선 것이 흡수하는 쪽
        if max(ti["meta"]["frame_indices"]) > min(tj["meta"]["frame_indices"]):
            ri, rj = rj, ri
            ti, tj = tj, ti

        if dry_run:
            tqdm.write(
                f"  [병합 후보] {ti['tdir'].name} + {tj['tdir'].name}"
                f"  sim={sim:.3f}  gap={gap}f"
            )
        else:
            tqdm.write(
                f"  [병합] {ti['tdir'].name} ← {tj['tdir'].name}"
                f"  sim={sim:.3f}  gap={gap}f"
            )

        updated_meta = merge_two(
            ti["meta"], ti["tdir"],
            tj["meta"], tj["tdir"],
            dry_run,
        )
        tracklets[ri]["meta"] = updated_meta
        merged_into[rj] = ri
        merge_count += 1
        deleted_count += 1

    return merge_count, deleted_count


# ── 메인 ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="동일 cam/slot 내 트랙렛 병합")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--tracklets-dir", default=None)
    parser.add_argument("--sim-threshold", type=float, default=SIM_THRESHOLD,
                        help=f"대표 임베딩 cosine 유사도 하한 (기본: {SIM_THRESHOLD})")
    parser.add_argument("--max-gap-frames", type=int, default=MAX_GAP_FRAMES,
                        help=f"허용 최대 frame gap (기본: {MAX_GAP_FRAMES})")
    parser.add_argument("--min-frames", type=int, default=MIN_FRAMES,
                        help=f"처리 대상 최소 트랙렛 길이 (기본: {MIN_FRAMES})")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 변경 없이 병합 후보만 출력")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tracklets_dir = Path(
        args.tracklets_dir
        or config.get("data", {}).get("tracklet_dir", "data/tracklets-1-original-reduced")
    )
    if not tracklets_dir.is_absolute():
        tracklets_dir = project_root / tracklets_dir
    if not tracklets_dir.exists():
        sys.exit(f"[ERROR] 트랙렛 디렉토리 없음: {tracklets_dir}")

    reid_cfg = config.get("reid", {})
    extractor = OSNetExtractor(
        model_name=reid_cfg.get("model_name", "osnet_x1_0"),
        pretrained=reid_cfg.get("pretrained", True),
        weights_path=reid_cfg.get("weights_path"),
        device=reid_cfg.get("device", "auto"),
    )

    cam_slot_dirs = sorted(d for d in tracklets_dir.iterdir() if d.is_dir())
    print(f"[INFO] cam/slot {len(cam_slot_dirs)}개  |  "
          f"sim_threshold={args.sim_threshold}  "
          f"max_gap={args.max_gap_frames}  "
          f"min_frames={args.min_frames}")
    if args.dry_run:
        print("[DRY-RUN] 실제 변경 없음\n")

    total_merges = total_deleted = 0
    for cam_slot_dir in cam_slot_dirs:
        track_count = sum(1 for d in cam_slot_dir.iterdir() if d.is_dir() and d.name.startswith("track_"))
        print(f"\n[{cam_slot_dir.name}] 트랙렛 {track_count}개 처리 중...")
        merges, deleted = process_cam_slot(
            cam_slot_dir, extractor,
            args.sim_threshold, args.max_gap_frames,
            args.min_frames, args.dry_run,
        )
        total_merges  += merges
        total_deleted += deleted
        print(f"[{cam_slot_dir.name}] 완료 ✓  (병합 {merges}쌍, 삭제 {deleted}개)")

    print()
    print("=" * 50)
    print(f"  병합된 쌍       : {total_merges}쌍")
    print(f"  삭제된 트랙렛   : {total_deleted}개")
    if args.dry_run:
        print("\n  ※ --dry-run: 실제 변경 없음")
    print("=" * 50)


if __name__ == "__main__":
    main()
