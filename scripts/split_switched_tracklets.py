"""
scripts/split_switched_tracklets.py
=====================================
트랙렛 내부에서 연속 프레임 간 임베딩 유사도가 급격히 떨어지는 지점을
ID switch로 판단하여 트랙렛을 분리합니다.

동작 원리:
  1. 트랙렛의 모든 프레임 임베딩을 추출
  2. 슬라이딩 앵커(현재 구간 평균 임베딩)와 각 프레임 유사도 비교
  3. --sim-threshold 미만이 --min-switch-frames 프레임 연속되면 분리
  4. --min-frames 미만의 짧은 조각은 버림

사용법:
    conda activate eye-d
    python scripts/split_switched_tracklets.py --config configs/config.yaml
    python scripts/split_switched_tracklets.py --config configs/config.yaml --dry-run
    python scripts/split_switched_tracklets.py --config configs/config.yaml --sim-threshold 0.55
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


# ── 기본 파라미터 ──────────────────────────────────────────────
MIN_FRAMES = 8          # 분리 후 유지할 최소 프레임 수
SIM_THRESHOLD = 0.70    # 앵커 대비 유사도 하한 (이 미만 = 이물질 프레임)
MIN_SWITCH_FRAMES = 3   # 연속 N프레임 이상 유사도 미달 시 분리


# ── 임베딩 추출 ────────────────────────────────────────────────
def extract_all_features(tdir: Path, crop_files: list[str],
                         extractor: OSNetExtractor) -> np.ndarray | None:
    """트랙렛 내 모든 프레임의 L2 정규화 임베딩 반환. shape: (N, D)"""
    imgs, valid_idx = [], []
    for i, fname in enumerate(crop_files):
        img = cv2.imread(str(tdir / fname))
        if img is not None and img.size > 0:
            imgs.append(img)
            valid_idx.append(i)

    if len(imgs) < 2:
        return None, valid_idx

    feats = extractor.extract_batch_features(imgs)   # (N, D), L2 정규화됨
    return feats, valid_idx


# ── 분리 지점 탐지 ─────────────────────────────────────────────
def find_split_points(feats: np.ndarray,
                      sim_threshold: float,
                      min_switch_frames: int,
                      fixed_anchor: bool = False) -> list[int]:
    """
    분리 지점(인덱스) 반환.

    fixed_anchor=False (기본, 슬라이딩 앵커):
        앵커 = 현재 구간 전체의 running mean.
        점진적 변화에 유연하지만 서서히 바뀌는 ID switch는 놓칠 수 있음.

    fixed_anchor=True (고정 앵커):
        앵커 = 각 구간의 첫 프레임 임베딩으로 고정.
        점진적 드리프트(밝기 변화, 서서히 다른 사람)도 탐지 가능.
        단, 같은 사람이라도 포즈/조명 변화가 크면 과분리 위험.

    반환값: 각 구간의 시작 인덱스 리스트 (항상 0 포함)
    """
    n = len(feats)
    segments = [0]
    anchor = feats[0].copy()  # 현재 구간 기준 임베딩 (앵커)
    below_count = 0

    for i in range(1, n):
        sim = float(feats[i] @ anchor)   # cosine sim (이미 L2 정규화)
        if sim < sim_threshold:
            below_count += 1
            if below_count >= min_switch_frames:
                split_at = i - min_switch_frames + 1
                if split_at > segments[-1]:
                    segments.append(split_at)
                    # 새 구간 앵커 = 분리 시작 프레임으로 리셋 (공통)
                    anchor = feats[i].copy()
                    below_count = 0
        else:
            below_count = 0
            if not fixed_anchor:
                # 슬라이딩 앵커: 현재 구간 전체 평균으로 갱신
                seg_start = segments[-1]
                seg_feats = feats[seg_start : i + 1]
                mean = seg_feats.mean(axis=0)
                norm = np.linalg.norm(mean)
                anchor = mean / norm if norm > 0 else mean
            # fixed_anchor=True 면 앵커 갱신 없이 구간 첫 프레임 유지

    return segments


# ── 트랙렛 분리 실행 ───────────────────────────────────────────
def split_tracklet(tdir: Path, meta: dict, segments: list[int],
                   valid_idx: list[int], next_track_id: int,
                   min_frames: int, dry_run: bool) -> tuple[int, list[str]]:
    """
    분리 지점에 따라 트랙렛을 여러 폴더로 나눔.
    첫 구간은 원본 폴더 유지, 이후 구간은 새 track_id 폴더 생성.

    Returns: (사용된 새 track_id 수, 생성된 폴더 이름 목록)
    """
    crop_files = meta["crop_files"]
    frame_indices = meta["frame_indices"]
    bboxes = meta["bboxes"]
    confidences = meta["confidences"]

    # valid_idx: 실제 로드된 프레임의 원본 인덱스
    # segments: valid_idx 기준 구간 시작점
    segments_end = segments[1:] + [len(valid_idx)]

    cam_dir = tdir.parent
    new_ids_used = 0
    created = []

    for seg_num, (seg_start, seg_end) in enumerate(zip(segments, segments_end)):
        orig_indices = valid_idx[seg_start:seg_end]
        if not orig_indices:
            continue

        seg_crops = [crop_files[i] for i in orig_indices]
        seg_frames = [frame_indices[i] for i in orig_indices]
        seg_bboxes = [bboxes[i] for i in orig_indices]
        seg_confs = [confidences[i] for i in orig_indices]

        if seg_num == 0:
            # 첫 구간: 원본 폴더 유지, 불필요한 이미지 삭제
            seg_dir = tdir
            seg_track_id = meta["track_id"]
        else:
            # 이후 구간: 새 폴더 생성
            seg_track_id = next_track_id + new_ids_used
            new_ids_used += 1
            seg_dir = cam_dir / f"track_{seg_track_id:04d}"
            if not dry_run:
                seg_dir.mkdir(exist_ok=True)
                for fname in seg_crops:
                    src = tdir / fname
                    dst = seg_dir / fname
                    if src.exists():
                        shutil.move(str(src), str(dst))

        # metadata 갱신
        seg_meta = dict(meta)
        seg_meta["track_id"] = seg_track_id
        seg_meta["length"] = len(seg_crops)
        seg_meta["crop_files"] = seg_crops
        seg_meta["frame_indices"] = seg_frames
        seg_meta["bboxes"] = seg_bboxes
        seg_meta["confidences"] = seg_confs
        seg_meta["avg_conf"] = float(np.mean(seg_confs))

        if not dry_run:
            (seg_dir / "metadata.json").write_text(
                json.dumps(seg_meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        created.append(seg_dir.name)

    # 첫 구간 이외의 이미지가 원본 폴더에 남아있으면 삭제
    if not dry_run and new_ids_used > 0:
        keep = set(crop_files[i] for i in valid_idx[:segments[1]])
        for p in tdir.glob("*.jpg"):
            if p.name not in keep:
                p.unlink(missing_ok=True)

    return new_ids_used, created


# ── 전체 파이프라인 ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="트랙렛 내부 ID switch 분리")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--tracklets-dir", default=None,
                        help="트랙렛 루트 디렉토리 (미지정 시 config에서 읽음)")
    parser.add_argument("--sim-threshold", type=float, default=SIM_THRESHOLD,
                        help=f"앵커 대비 cosine 유사도 하한 (기본: {SIM_THRESHOLD})")
    parser.add_argument("--min-switch-frames", type=int, default=MIN_SWITCH_FRAMES,
                        help=f"연속 미달 프레임 수 기준 (기본: {MIN_SWITCH_FRAMES})")
    parser.add_argument("--min-frames", type=int, default=MIN_FRAMES,
                        help=f"분리 후 유지할 최소 프레임 수 (기본: {MIN_FRAMES})")
    parser.add_argument("--fixed-anchor", action="store_true",
                        help="앵커를 구간 첫 프레임으로 고정 (점진적 드리프트 탐지에 유리)")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 변경 없이 분리 후보만 출력")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tracklets_dir = Path(
        args.tracklets_dir
        or config.get("tracking", {}).get("output_dir", "data/tracklets")
    )
    if not tracklets_dir.exists():
        sys.exit(f"[ERROR] 트랙렛 디렉토리 없음: {tracklets_dir.resolve()}")

    reid_cfg = config.get("reid", {})
    extractor = OSNetExtractor(
        model_name=reid_cfg.get("model_name", "osnet_x1_0"),
        pretrained=reid_cfg.get("pretrained", True),
        weights_path=reid_cfg.get("weights_path", None),
        device=reid_cfg.get("device", "auto"),
    )

    # 현재 최대 track_id 파악
    all_meta_paths = sorted(tracklets_dir.glob("*/track_*/metadata.json"))
    if not all_meta_paths:
        sys.exit(f"[ERROR] 트랙렛이 없습니다: {tracklets_dir}")

    next_id = max(
        json.loads(p.read_text()).get("track_id", 0) for p in all_meta_paths
    ) + 1

    anchor_mode = "고정(fixed)" if args.fixed_anchor else "슬라이딩(sliding)"
    print(f"[INFO] 트랙렛 {len(all_meta_paths)}개  |  "
          f"sim_threshold={args.sim_threshold}  "
          f"min_switch_frames={args.min_switch_frames}  "
          f"min_frames={args.min_frames}  "
          f"anchor={anchor_mode}")
    print(f"[INFO] 신규 track_id 시작: {next_id}")
    if args.dry_run:
        print("[DRY-RUN] 실제 변경 없음\n")

    stats = {"scanned": 0, "split": 0, "new_tracklets": 0}

    # cam/slot 단위로 그룹화
    from itertools import groupby
    cam_slot_groups = groupby(all_meta_paths, key=lambda p: p.parent.parent)

    for cam_slot_dir, group in cam_slot_groups:
        group_paths = list(group)
        print(f"\n[{cam_slot_dir.name}] 트랙렛 {len(group_paths)}개 처리 중...")

        for meta_path in tqdm(group_paths, desc=cam_slot_dir.name, leave=False):
            tdir = meta_path.parent
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            crop_files = meta.get("crop_files", [])

            if len(crop_files) < args.min_frames * 2:
                stats["scanned"] += 1
                continue

            feats, valid_idx = extract_all_features(tdir, crop_files, extractor)
            if feats is None or len(feats) < 2:
                stats["scanned"] += 1
                continue

            segments = find_split_points(feats, args.sim_threshold, args.min_switch_frames,
                                         fixed_anchor=args.fixed_anchor)
            stats["scanned"] += 1

            if len(segments) <= 1:
                continue

            new_ids, created = split_tracklet(
                tdir, meta, segments, valid_idx, next_id,
                args.min_frames, args.dry_run
            )

            stats["split"] += 1
            stats["new_tracklets"] += new_ids
            next_id += new_ids

            tqdm.write(
                f"  [분리] {tdir.name}  "
                f"{len(segments)}구간 → {len(created)}개 트랙렛 생성"
            )

        print(f"[{cam_slot_dir.name}] 완료 ✓")

    print()
    print("=" * 50)
    print(f"  검사한 트랙렛  : {stats['scanned']}개")
    print(f"  분리된 트랙렛  : {stats['split']}개")
    print(f"  생성된 신규 ID : {stats['new_tracklets']}개")
    if args.dry_run:
        print("\n  ※ --dry-run 모드: 실제 변경 없음")
    print("=" * 50)


if __name__ == "__main__":
    main()
