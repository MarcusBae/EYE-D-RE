"""
scripts/fiftyone_curator.py
============================
FiftyOne 기반 트랙렛 프레임 큐레이션 툴.

metadata.json 무시 — 실제 존재하는 frame_*.jpg 파일만 로드.
OSNet 임베딩으로 각 트랙렛 내 아웃라이어 프레임을 자동 점수화.

사용 흐름:
  1. 최초 실행 (임베딩 계산 포함, 시간 소요):
       python scripts/fiftyone_curator.py --config configs/config.yaml

  2. 브라우저에서 아웃라이어 프레임 확인 → "delete" 태그로 표시

  3. 태그된 프레임 실제 파일 삭제:
       python scripts/fiftyone_curator.py --config configs/config.yaml --delete-tagged

  4. 재실행 (임베딩 캐시 재사용, 빠름):
       python scripts/fiftyone_curator.py --config configs/config.yaml --reload
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

DATASET_NAME_FULL = "eye-d-tracklets"
DATASET_NAME_SAMPLE = "eye-d-tracklets-sample"
EMBED_CACHE = project_root / "data" / ".fiftyone_embeddings.npz"


# ── 데이터 로드 ────────────────────────────────────────────────
def load_samples_from_filesystem(tracklet_dir: Path) -> list[dict]:
    """
    metadata.json 무시하고 실제 파일시스템 기준으로 샘플 목록 생성.
    반환: [{"filepath": ..., "track_key": ..., "camera_id": ..., ...}, ...]
    """
    samples = []
    for cam_slot_dir in sorted(tracklet_dir.iterdir()):
        if not cam_slot_dir.is_dir():
            continue
        # 폴더명 파싱: c1_t3 → camera_id=1, time_slot=3
        parts = cam_slot_dir.name.split("_")
        if len(parts) != 2:
            continue
        try:
            cam_id = int(parts[0][1:])
            slot_id = int(parts[1][1:])
        except ValueError:
            continue

        for track_dir in sorted(cam_slot_dir.iterdir()):
            if not track_dir.is_dir() or not track_dir.name.startswith("track_"):
                continue
            try:
                track_id = int(track_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue

            frames = sorted(track_dir.glob("frame_*.jpg"))
            if not frames:
                continue

            track_key = f"c{cam_id}_t{slot_id}/{track_dir.name}"
            for frame_path in frames:
                samples.append({
                    "filepath": str(frame_path),
                    "track_key": track_key,
                    "camera_id": cam_id,
                    "time_slot": slot_id,
                    "track_id": track_id,
                    "frame_name": frame_path.name,
                })

    return samples


# ── 임베딩 계산 ────────────────────────────────────────────────
def compute_or_load_embeddings(
    sample_list: list[dict],
    config: dict,
    force: bool = False,
) -> np.ndarray:
    """OSNet으로 임베딩 계산. 캐시가 있으면 재사용."""
    filepaths = [s["filepath"] for s in sample_list]

    if not force and EMBED_CACHE.exists():
        cache = np.load(EMBED_CACHE, allow_pickle=True)
        cached_paths = list(cache["filepaths"])
        cached_embeds = cache["embeddings"]

        # 캐시와 현재 파일 목록이 완전히 일치하면 재사용
        if cached_paths == filepaths:
            print(f"[캐시] 임베딩 로드: {EMBED_CACHE.name} ({len(filepaths)}장)")
            return cached_embeds

        # 일부만 캐시에 있으면 새 파일만 추가 계산
        cached_map = dict(zip(cached_paths, cached_embeds))
        new_paths = [p for p in filepaths if p not in cached_map]
        if new_paths:
            print(f"[캐시] {len(cached_map)}장 재사용, {len(new_paths)}장 새로 계산")
            new_embeds = _run_extractor(new_paths, config)
            for p, e in zip(new_paths, new_embeds):
                cached_map[p] = e
        else:
            print(f"[캐시] 임베딩 전체 재사용")

        embeddings = np.stack([cached_map[p] for p in filepaths])
        _save_cache(filepaths, embeddings)
        return embeddings

    print(f"[임베딩] OSNet으로 {len(filepaths)}장 계산 중...")
    embeddings = _run_extractor(filepaths, config)
    _save_cache(filepaths, embeddings)
    return embeddings


def _run_extractor(filepaths: list[str], config: dict) -> np.ndarray:
    from pipeline.reid_merger import OSNetExtractor

    reid_cfg = config.get("reid", {})
    extractor = OSNetExtractor(
        model_name=reid_cfg.get("model_name", "osnet_x1_0"),
        pretrained=reid_cfg.get("pretrained", True),
        weights_path=reid_cfg.get("weights_path"),
        device=reid_cfg.get("device", "auto"),
    )
    batch_size = reid_cfg.get("batch_size", 64)

    imgs, valid_idx = [], []
    for i, fp in enumerate(tqdm(filepaths, desc="이미지 로드")):
        img = cv2.imread(fp)
        if img is not None:
            imgs.append(img)
            valid_idx.append(i)

    all_embeds = np.zeros((len(filepaths), 512), dtype=np.float32)
    if imgs:
        feats = extractor.extract_batch_features(imgs, batch_size=batch_size)
        for out_i, orig_i in enumerate(valid_idx):
            all_embeds[orig_i] = feats[out_i]

    return all_embeds


def _save_cache(filepaths: list[str], embeddings: np.ndarray):
    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(EMBED_CACHE, filepaths=filepaths, embeddings=embeddings)
    print(f"[캐시] 임베딩 저장 완료: {EMBED_CACHE.name}")


# ── 아웃라이어 점수 계산 ───────────────────────────────────────
def compute_outlier_scores(
    sample_list: list[dict],
    embeddings: np.ndarray,
) -> np.ndarray:
    """
    각 트랙렛 내부에서 중앙값 임베딩과의 코사인 유사도를 계산.
    점수가 낮을수록(0에 가까울수록) 아웃라이어.
    """
    scores = np.ones(len(sample_list), dtype=np.float32)

    # track_key별로 인덱스 그룹화
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(sample_list):
        groups.setdefault(s["track_key"], []).append(i)

    for track_key, idxs in groups.items():
        if len(idxs) < 2:
            continue
        embs = embeddings[idxs]  # (N, D)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        embs_norm = embs / norms

        median_emb = np.median(embs_norm, axis=0)
        median_norm = np.linalg.norm(median_emb)
        if median_norm > 0:
            median_emb /= median_norm

        sims = embs_norm @ median_emb  # (N,)
        for local_i, global_i in enumerate(idxs):
            scores[global_i] = float(sims[local_i])

    return scores


# ── FiftyOne 데이터셋 구성 ─────────────────────────────────────
def build_or_reload_dataset(
    sample_list: list[dict],
    embeddings: np.ndarray,
    outlier_scores: np.ndarray,
    dataset_name: str,
    reload: bool = False,
):
    import fiftyone as fo
    import fiftyone.brain as fob

    # 기존 데이터셋 삭제 후 재생성
    if dataset_name in fo.list_datasets():
        if reload:
            fo.delete_dataset(dataset_name)
        else:
            print(f"[FiftyOne] 기존 데이터셋 '{dataset_name}' 로드")
            return fo.load_dataset(dataset_name)

    print(f"[FiftyOne] 데이터셋 생성 중 ({len(sample_list)}장)...")
    dataset = fo.Dataset(dataset_name, persistent=True)

    fo_samples = []
    for i, s in enumerate(tqdm(sample_list, desc="샘플 등록")):
        sample = fo.Sample(filepath=s["filepath"])
        sample["track_key"] = s["track_key"]
        sample["camera_id"] = s["camera_id"]
        sample["time_slot"] = s["time_slot"]
        sample["track_id"] = s["track_id"]
        sample["frame_name"] = s["frame_name"]
        sample["outlier_score"] = float(outlier_scores[i])
        # 아웃라이어 점수가 낮을수록 의심스러운 프레임
        sample["is_suspicious"] = bool(outlier_scores[i] < 0.70)
        fo_samples.append(sample)

    dataset.add_samples(fo_samples)

    # 임베딩 기반 시각화 인덱스 생성 (UMAP 등 뷰를 위해)
    print("[FiftyOne] 유사도 인덱스 생성 중...")
    fob.compute_similarity(
        dataset,
        embeddings=embeddings,
        brain_key="osnet_sim",
    )

    dataset.save()
    print(f"[FiftyOne] 완료: {len(dataset)}장 등록")
    return dataset


# ── 태그된 프레임 삭제 ─────────────────────────────────────────
def delete_tagged_frames(dataset):
    import fiftyone as fo

    tagged = dataset.match_tags("delete")
    count = len(tagged)
    if count == 0:
        print("[삭제] 'delete' 태그 프레임 없음")
        return

    print(f"[삭제] 'delete' 태그된 프레임 {count}장 삭제 시작...")
    paths_to_delete = tagged.values("filepath")

    deleted = 0
    for fp in tqdm(paths_to_delete, desc="파일 삭제"):
        p = Path(fp)
        if p.exists():
            p.unlink()
            deleted += 1

    # FiftyOne 데이터셋에서도 제거
    dataset.delete_samples(tagged)
    dataset.save()

    # 임베딩 캐시 무효화 (다음 실행에서 재계산)
    if EMBED_CACHE.exists():
        EMBED_CACHE.unlink()
        print("[캐시] 임베딩 캐시 삭제 (다음 실행 시 재계산)")

    print(f"[삭제] 완료: {deleted}/{count}장 삭제")


# ── 메인 ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FiftyOne 트랙렛 큐레이션")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--tracklets-dir", default=None)
    parser.add_argument("--delete-tagged", action="store_true",
                        help="'delete' 태그된 프레임을 실제로 삭제하고 종료")
    parser.add_argument("--reload", action="store_true",
                        help="FiftyOne 데이터셋을 처음부터 다시 만들기")
    parser.add_argument("--force-embed", action="store_true",
                        help="임베딩 캐시 무시하고 강제 재계산")
    parser.add_argument("--sample", type=int, default=None, metavar="N",
                        help="처음 N개 트랙렛만 로드 (동작 확인용)")
    parser.add_argument("--cam-slot", default=None, metavar="CAM_SLOT",
                        help="특정 cam/slot만 로드 (예: c1_t10)")
    parser.add_argument("--port", type=int, default=5151)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    tracklet_dir = Path(
        args.tracklets_dir
        or config.get("data", {}).get("tracklet_dir", "data/tracklets-2-manual-edit")
    )
    if not tracklet_dir.is_absolute():
        tracklet_dir = project_root / tracklet_dir
    if not tracklet_dir.exists():
        sys.exit(f"[ERROR] 트랙렛 디렉토리 없음: {tracklet_dir}")

    import fiftyone as fo

    # --delete-tagged: 앱 열지 않고 삭제만 수행
    is_sample = bool(args.sample or args.cam_slot)
    dataset_name = DATASET_NAME_SAMPLE if is_sample else DATASET_NAME_FULL

    if args.delete_tagged:
        if dataset_name not in fo.list_datasets():
            sys.exit(f"[ERROR] 데이터셋 '{dataset_name}'이 없습니다. 먼저 일반 실행하세요.")
        dataset = fo.load_dataset(dataset_name)
        delete_tagged_frames(dataset)
        return

    # 일반 실행: 로드 → 임베딩 → 앱 실행
    print(f"[INFO] 트랙렛 디렉토리: {tracklet_dir}")
    sample_list = load_samples_from_filesystem(tracklet_dir)
    print(f"[INFO] 실제 파일 기준 {len(sample_list)}장 발견")

    # --cam-slot 필터
    if args.cam_slot:
        sample_list = [s for s in sample_list if s["track_key"].startswith(args.cam_slot + "/")]
        print(f"[INFO] cam-slot '{args.cam_slot}' 필터 적용 → {len(sample_list)}장")

    # --sample N: 처음 N개 트랙렛만
    if args.sample:
        all_keys = list(dict.fromkeys(s["track_key"] for s in sample_list))
        keep_keys = set(all_keys[: args.sample])
        sample_list = [s for s in sample_list if s["track_key"] in keep_keys]
        print(f"[INFO] 샘플 {args.sample}개 트랙렛 → {len(sample_list)}장")

    embeddings = compute_or_load_embeddings(
        sample_list, config, force=args.force_embed
    )
    outlier_scores = compute_outlier_scores(sample_list, embeddings)

    suspicious_count = int((outlier_scores < 0.70).sum())
    print(f"[INFO] 의심 프레임 (score < 0.70): {suspicious_count}장")

    dataset = build_or_reload_dataset(
        sample_list, embeddings, outlier_scores,
        dataset_name=dataset_name, reload=args.reload
    )

    print()
    print("=" * 55)
    print("  FiftyOne 앱 사용 방법")
    print("=" * 55)
    print("  1. 브라우저에서 http://localhost:{} 열기".format(args.port))
    print("  2. 'is_suspicious = True' 필터로 의심 프레임 확인")
    print("  3. 삭제할 프레임 선택 → 우클릭 → Tag: 'delete'")
    print("  4. Ctrl+C로 앱 종료 후:")
    print("     python scripts/fiftyone_curator.py --delete-tagged")
    print("=" * 55)

    session = fo.launch_app(dataset, port=args.port)
    session.wait()


if __name__ == "__main__":
    main()
