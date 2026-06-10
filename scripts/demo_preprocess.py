"""
scripts/demo_preprocess.py
===========================
demo_app.py 가 읽는 data/demo_data/ 를 생성하는 전처리 스크립트.

출력물
  data/demo_data/
  ├── manifest.json          인물별 등장 구간·썸네일 경로 목록
  └── crops/{pid:04d}/
      ├── thumb.jpg          대표 썸네일 (중앙 프레임)
      └── feature.npy        OSNet 512-d 평균 특징 벡터

타임스탬프 계산
  각 프레임의 절대 시각 = configs/config.yaml videos[파일명].start_time
                          + frame_index / src_fps
  start_time 은 파일마다 다를 수 있으므로 반드시 config 를 먼저 수정한 뒤 실행.

──────────────────────────────────────────────────────────────
[모드 1] 기존 트랙렛 사용 (빠름, 권장)
  data/curated/ 가 이미 준비된 상태에서 manifest 만 (재)생성할 때 사용.

    # 전체 슬롯 처리
    python scripts/demo_preprocess.py --config configs/config.yaml

    # 특정 슬롯만 처리 (cam_slot 폴더명 기준)
    python scripts/demo_preprocess.py --config configs/config.yaml \
        --slots c1_t8 c2_t8 c3_t8

──────────────────────────────────────────────────────────────
[모드 2] 새 영상 파일에서 전체 파이프라인 실행
  영상 파일을 직접 지정하면 트래킹 → 품질 필터 → Re-ID → 매칭까지 자동 수행.
  처리 완료 후 임시 폴더(_tracklets, _filtered)는 자동 삭제됨.

    python scripts/demo_preprocess.py --config configs/config.yaml \
        --videos data/raw_videos/cam1_t8.avi \
                 data/raw_videos/cam2_t8.avi \
                 data/raw_videos/cam3_t8.avi

──────────────────────────────────────────────────────────────
주요 옵션
  --config     configs/config.yaml 경로 (기본: configs/config.yaml)
  --out        출력 루트 (기본: data/demo_data)
  --tracklets-dir  트랙렛 소스 디렉터리 (기본: data/curated)
  --slots      처리할 cam_slot 폴더 이름 목록 (모드 1, 미지정 시 전체)
  --videos     처리할 영상 파일 경로 목록 (모드 2)
  --threshold  HAC 병합 임계값 cosine distance (기본: config 값)
  --debug      pairwise 거리 행렬 및 클러스터 구성 출력
  --match-only 캐시된 특징 벡터로 매칭만 재실행 (파라미터 튜닝용, 아래 참고)

──────────────────────────────────────────────────────────────
[매칭 파라미터 튜닝] --match-only 활용법

  특징 추출(수 분)은 건너뛰고 run_matching() 만 반복 실행할 수 있음.
  전체 파이프라인을 한 번 실행하면 특징 벡터가 캐시로 저장됨:
    data/demo_data/_cache_feats.npy
    data/demo_data/_cache_tracklets.json

  이후 --match-only 로 매칭만 즉시 재실행:

    # threshold 값만 바꿔서 결과 확인
    python scripts/demo_preprocess.py --match-only --threshold 0.25

    # 거리 행렬 + 클러스터 구성 상세 출력
    python scripts/demo_preprocess.py --match-only --threshold 0.20 --debug

    # 더 느슨하게 병합 (같은 사람으로 묶이는 트랙렛 증가)
    python scripts/demo_preprocess.py --match-only --threshold 0.35 --debug

  threshold 가이드:
    낮을수록 (0.10~0.20) 보수적 — 다른 사람으로 분리
    높을수록 (0.30~0.45) 관대함 — 같은 사람으로 병합
    현재 config 기본값: reid.clustering.threshold

──────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from pipeline.reid_merger import OSNetExtractor
from pipeline.tracklet_io import list_tracklets

def build_cam_slot_start(config: dict) -> dict[tuple[int, int], str]:
    """config videos 섹션에서 (camera, slot) → start_time 룩업 테이블 생성."""
    table: dict[tuple[int, int], str] = {}
    for info in config.get("videos", {}).values():
        key = (info["camera"], info["slot"])
        if "start_time" in info:
            table[key] = info["start_time"]
    return table


# ── 유틸 ──────────────────────────────────────────────────────────

def frame_to_timestamp(frame_idx: int, fps: float, start_time: str) -> str:
    start = datetime.strptime(start_time, "%H:%M:%S")
    ts = start + timedelta(seconds=frame_idx / fps)
    return ts.strftime("%H:%M:%S")


def pick_thumbnail(tdir: Path, crop_files: list) -> Path | None:
    mid = crop_files[len(crop_files) // 2]
    p = tdir / mid
    return p if p.exists() else None


def build_appearance(tdir: Path, meta: dict,
                     cam_slot_start: dict[tuple[int, int], str]) -> dict:
    fps = meta.get("src_fps", 24.0)
    slot = meta.get("time_slot", 0)
    cam = meta.get("camera_id", 0)
    crops = meta.get("crop_files", [])
    frames = meta.get("frame_indices", [])
    origin = cam_slot_start.get((cam, slot), "00:00:00")
    start_t = frame_to_timestamp(frames[0], fps, origin) if frames else "??:??:??"
    end_t   = frame_to_timestamp(frames[-1], fps, origin) if frames else "??:??:??"
    return {
        "camera": cam,
        "slot": slot,
        "start_time": start_t,
        "end_time": end_t,
        "tracklet_dir": str(tdir),
        "crop_files": crops,
        "num_frames": len(crops),
    }


# ── 모드 2: 영상 → 트랙렛 ────────────────────────────────────────

def get_video_cam_slot(video_path: Path, config: dict) -> tuple[int, int] | None:
    """config.yaml videos 섹션에서 파일명으로 (camera_id, time_slot) 조회."""
    name = video_path.name
    for vname, info in config.get("videos", {}).items():
        if vname == name:
            return info["camera"], info["slot"]
    return None


def run_tracking(video_path: Path, camera_id: int, time_slot: int,
                 config: dict, out_dir: Path, frame_stride: int = 6) -> Path:
    """영상 1개를 트래킹해 out_dir/{cam_slot}/ 에 트랙렛 저장."""
    from pipeline.tracker import PersonTracker
    det_cfg = config.get("detection", {})
    trk_cfg = config.get("tracking", {})
    tracker = PersonTracker(
        model_path=det_cfg.get("model", "yolov8n.pt"),
        tracker_yaml=trk_cfg.get("tracker_yaml", "configs/botsort.yaml"),
        conf=det_cfg.get("conf_threshold", 0.5),
        iou=det_cfg.get("iou_threshold", 0.45),
        imgsz=det_cfg.get("imgsz", 640),
        classes=det_cfg.get("classes", [0]),
        device=det_cfg.get("device", "auto"),
    )
    print(f"  [트래킹] {video_path.name}  (cam={camera_id}, slot={time_slot})")
    tracker.track_video(
        video_path=str(video_path),
        camera_id=camera_id,
        time_slot=time_slot,
        output_dir=str(out_dir),
        frame_stride=frame_stride,
        save_crops=True,
    )
    return out_dir / f"c{camera_id}_t{time_slot}"


def run_quality_filter(tracklet_dir: Path, filtered_dir: Path, config: dict) -> Path:
    """품질 필터를 적용해 통과한 트랙렛만 filtered_dir 에 복사."""
    from pipeline.quality_filter import TrackletQualityFilter
    qf_cfg = config.get("quality_filter", {})
    filt = TrackletQualityFilter(
        min_length=qf_cfg.get("min_track_length", 6),
        min_avg_conf=qf_cfg.get("min_avg_confidence", 0.7),
        min_bbox_h=qf_cfg.get("min_bbox_height", 64),
        min_bbox_w=qf_cfg.get("min_bbox_width", 32),
        max_aspect_ratio=qf_cfg.get("max_aspect_ratio", 4.0),
        min_area=qf_cfg.get("min_bbox_area", 2048),
    )
    filtered_dir.mkdir(parents=True, exist_ok=True)
    stats = filt.filter_all(
        tracklet_dir=str(tracklet_dir),
        output_dir=str(filtered_dir),
        copy_crops=True,
        verbose=False,
        sample_failures=10,
    )
    passed = stats.get("passed", 0)
    total  = stats.get("total", 0)
    print(f"  [필터] {tracklet_dir.name}: {passed}/{total}개 통과")
    if passed < total:
        samples = stats.get("failed_samples", [])
        print(f"  [필터] 탈락 예시 (최대 10개):")
        for s in samples:
            tid = s.get("track_id", "?")
            reasons = ", ".join(s.get("reasons", []))
            print(f"    track_{tid:04d}: {reasons}")
    return filtered_dir


def run_matching(tracklets: list, feats: np.ndarray, threshold: float,
                 debug: bool = False) -> list:
    """HAC 클러스터링으로 person_id 배정."""
    from scipy.cluster.hierarchy import fclusterdata
    from scipy.spatial.distance import cdist

    # [단계 1] 특징 벡터 L2 정규화 (L2 Normalization)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    feats_norm = feats / np.maximum(norms, 1e-8)

    if debug:
        dist_mat = cdist(feats_norm, feats_norm, metric="cosine")

        def _tid(t: dict) -> str:
            p = Path(t["tracklet_dir"])
            return f"{p.parent.name}/{p.name}"

        print(f"\n[DEBUG run_matching] 트랙렛 {len(tracklets)}개 / threshold={threshold}")
        print(f"[DEBUG] 특징 벡터 norm 범위: min={norms.min():.4f}  max={norms.max():.4f}")
        print("[DEBUG] pairwise cosine distance matrix:")
        labels_w = max(len(_tid(t)) for t in tracklets)
        header = " " * (labels_w + 2) + "  ".join(
            _tid(t)[-10:] for t in tracklets
        )
        print(header)
        for i, t in enumerate(tracklets):
            row = "  ".join(f"{dist_mat[i,j]:.3f}" for j in range(len(tracklets)))
            print(f"  {_tid(t):<{labels_w}}  {row}")
        print(f"[DEBUG] threshold={threshold} 기준으로 병합될 쌍 (dist < threshold):")
        merged = [(i, j) for i in range(len(tracklets))
                  for j in range(i+1, len(tracklets))
                  if dist_mat[i, j] < threshold]
        if merged:
            for i, j in merged:
                print(f"    {_tid(tracklets[i])} ↔ {_tid(tracklets[j])}  dist={dist_mat[i,j]:.4f}")
        else:
            print("    (없음 — 모든 트랙렛이 별도 클러스터)")

    # [단계 2] 계층적 병합 군집화 (HAC Clustering) 수행
    labels = fclusterdata(feats_norm, t=threshold,
                          criterion="distance", metric="cosine", method="average")

    if debug:
        from collections import Counter
        cluster_counts = Counter(int(l) for l in labels)
        print(f"[DEBUG] 클러스터 결과: {len(cluster_counts)}개 클러스터")
        for cid, cnt in sorted(cluster_counts.items()):
            members = [_tid(tracklets[i])
                       for i, l in enumerate(labels) if int(l) == cid]
            print(f"    cluster {cid}: {cnt}개 트랙렛 → {members}")

    return [int(l) for l in labels]


# ── 공통: 특징 추출 ───────────────────────────────────────────────

def extract_features(tracklets: list, extractor: OSNetExtractor,
                     save_samples_dir: Path | None = None) -> np.ndarray:
    if save_samples_dir is not None:
        save_samples_dir.mkdir(parents=True, exist_ok=True)

    feats = []
    for t in tqdm(tracklets, desc="  특징 추출"):
        tdir  = Path(t["tracklet_dir"])
        crops = t.get("crop_files", [])
        max_n = 8
        if len(crops) > max_n:
            idx   = np.linspace(0, len(crops) - 1, max_n + 2, dtype=int)[1:-1]
            crops = [crops[i] for i in idx]
        imgs = [cv2.imread(str(tdir / c)) for c in crops]
        imgs = [img for img in imgs if img is not None]
        if imgs:
            f = extractor.extract_batch_features(imgs)
            feats.append(f.mean(axis=0))
        else:
            feats.append(np.zeros(512, dtype=np.float32))

        if save_samples_dir is not None:
            dst = save_samples_dir / tdir.parent.name / tdir.name
            dst.mkdir(parents=True, exist_ok=True)
            for c in crops:
                src = tdir / c
                if src.exists():
                    shutil.copy2(src, dst / src.name)

    return np.array(feats)


# ── demo_data 저장 ────────────────────────────────────────────────

def save_demo_data(tracklets: list, person_ids: list, feats: np.ndarray,
                   out_dir: Path, tracklet_root: str,
                   cam_slot_start: dict[tuple[int, int], str] | None = None):
    css = cam_slot_start or {}
    persons: dict[int, dict] = {}
    for t, pid, feat in zip(tracklets, person_ids, feats):
        if pid not in persons:
            persons[pid] = {"id": pid, "appearances": [],
                            "feat_sum": np.zeros(512), "feat_count": 0}
        persons[pid]["appearances"].append(build_appearance(Path(t["tracklet_dir"]), t, css))
        persons[pid]["feat_sum"] += feat
        persons[pid]["feat_count"] += 1

    manifest_persons = []
    for pid, info in sorted(persons.items()):
        crop_dir = out_dir / "crops" / f"{pid:04d}"
        crop_dir.mkdir(parents=True, exist_ok=True)

        thumb_dst = None
        for app in info["appearances"]:
            tdir = Path(app["tracklet_dir"])
            if app["crop_files"]:
                src = pick_thumbnail(tdir, app["crop_files"])
                if src and src.exists():
                    thumb_dst = str((crop_dir / "thumb.jpg").resolve())
                    shutil.copy2(src, thumb_dst)
                    break

        mean_feat = info["feat_sum"] / max(info["feat_count"], 1)
        norm = np.linalg.norm(mean_feat)
        if norm > 0:
            mean_feat /= norm
        np.save(str(crop_dir / "feature.npy"), mean_feat)

        manifest_persons.append({
            "id": pid,
            "thumb": thumb_dst,
            "appearances": info["appearances"],
        })

    manifest = {
        "tracklet_root": tracklet_root,
        "num_persons": len(manifest_persons),
        "persons": manifest_persons,
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_persons


# ── main ──────────────────────────────────────────────────────────

def main():
    import yaml

    parser = argparse.ArgumentParser(description="EYE-D 데모 사전 처리")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--videos", nargs="*", default=None,
                        help="[모드 2] 동영상 파일 경로 목록 (전체 파이프라인 실행)")
    parser.add_argument("--tracklets-dir", default=None,
                        help="[모드 1] 트랙렛 루트 (미지정 시 config.filtered_dir)")
    parser.add_argument("--slots", nargs="*", default=None,
                        help="[모드 1] 사용할 슬롯 (예: c1_t4 c2_t4)")
    parser.add_argument("--frame-stride", type=int, default=6,
                        help="[모드 2] 트래킹 프레임 간격 (기본: 6)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="매칭 임계값 — 미지정 시 config.yaml reid.clustering.threshold 사용")
    parser.add_argument("--debug", action="store_true",
                        help="클러스터링 단계에서 pairwise 거리 행렬 출력")
    parser.add_argument("--save-samples", default=None, metavar="DIR",
                        help="특징 추출에 사용된 8장을 DIR/{cam_slot}/{track}/ 에 저장 (시각화용)")
    parser.add_argument("--match-only", action="store_true",
                        help="캐시된 특징 벡터로 매칭만 재실행 (파라미터 튜닝용). "
                             "--threshold, --debug 와 함께 사용")
    parser.add_argument("--out", default="data/demo_data")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    reid_cfg   = config.get("reid", {})
    if args.threshold is None:
        args.threshold = float(reid_cfg.get("clustering", {}).get("threshold", 0.30))
        print(f"[INFO] threshold = {args.threshold} (config.yaml 기준)")
    out_dir    = Path(args.out)

    # ── --match-only: 캐시된 특징으로 매칭만 재실행 ──────────────
    if args.match_only:
        cache_feats = out_dir / "_cache_feats.npy"
        cache_meta  = out_dir / "_cache_tracklets.json"
        if not cache_feats.exists() or not cache_meta.exists():
            sys.exit(f"[ERROR] 캐시 없음. 먼저 전체 파이프라인을 실행하세요.\n"
                     f"  필요 파일: {cache_feats}, {cache_meta}")
        feats      = np.load(str(cache_feats))
        tracklets  = json.loads(cache_meta.read_text(encoding="utf-8"))
        cam_slot_start = build_cam_slot_start(config)
        print(f"[INFO] 캐시 로드: {len(tracklets)}개 트랙렛, feats={feats.shape}")
        print(f"[INFO] threshold={args.threshold}, debug={args.debug}")
        person_ids = run_matching(tracklets, feats, args.threshold, debug=args.debug)
        persons    = save_demo_data(tracklets, person_ids, feats, out_dir,
                                    tracklets[0]["tracklet_dir"].rsplit("/", 2)[0] if tracklets else "",
                                    cam_slot_start=cam_slot_start)
        print(f"\n완료: {len(persons)}명 / {len(tracklets)}개 트랙렛")
        return

    # 출력 폴더 초기화
    if out_dir.exists():
        ans = input(f"[WARN] '{out_dir}' 이미 존재합니다. 삭제하고 새로 생성할까요? [y/N] ").strip().lower()
        if ans == "y":
            shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True)
            (out_dir / "crops").mkdir()
        else:
            print("[INFO] 기존 폴더를 그대로 사용합니다.")
            (out_dir / "crops").mkdir(exist_ok=True)
    else:
        out_dir.mkdir(parents=True)
        (out_dir / "crops").mkdir()

    # Re-ID 특징 추출기 로드
    extractor = OSNetExtractor(
        model_name=reid_cfg.get("model_name", "osnet_x1_0"),
        pretrained=reid_cfg.get("pretrained", True),
        weights_path=reid_cfg.get("weights_path"),
        device=reid_cfg.get("device", "auto"),
    )

    # ── 모드 분기 ──────────────────────────────────────────────────
    if args.videos:
        # ── 모드 2: 영상 파일 → 전체 파이프라인 ──────────────────
        print("[INFO] 모드 2: 영상 파일에서 전체 파이프라인 실행")
        video_paths = [Path(v) for v in args.videos]

        for vp in video_paths:
            if not vp.exists():
                sys.exit(f"[ERROR] 영상 파일 없음: {vp}")
            info = get_video_cam_slot(vp, config)
            if info is None:
                sys.exit(f"[ERROR] '{vp.name}' 이 configs/config.yaml videos 섹션에 없습니다.")

        tmp_track  = out_dir / "_tracklets"
        tmp_filter = out_dir / "_filtered"

        for _d in (tmp_track, tmp_filter):
            if _d.exists():
                ans = input(f"[WARN] '{_d}' 이미 존재합니다. 삭제하고 새로 시작할까요? [y/N] ").strip().lower()
                if ans == "y":
                    shutil.rmtree(_d)
                    print(f"[INFO] 폴더 삭제: {_d}")
                else:
                    print(f"[INFO] 기존 폴더 유지: {_d}")

        print("\n[STEP 1] 트래킹")
        for vp in video_paths:
            cam, slot = get_video_cam_slot(vp, config)
            run_tracking(vp, cam, slot, config, tmp_track, args.frame_stride)

        print("\n[STEP 2] 품질 필터")
        for slot_dir in sorted(tmp_track.iterdir()):
            if slot_dir.is_dir():
                run_quality_filter(slot_dir, tmp_filter / slot_dir.name, config)

        print("\n[STEP 2-b] 혼합 트랙렛 분리 (tracklet splitter)")
        from pipeline.tracklet_splitter import split_all as _split_all
        _split_cfg = config.get("reid", {}).get("clustering", {})
        _split_thresh = float(_split_cfg.get("split_threshold",
                              _split_cfg.get("threshold", 0.30)))
        _split_stats = _split_all(
            tmp_filter, extractor,
            split_threshold=_split_thresh,
            min_crops_per_part=3,
        )
        print(f"  [splitter] 전체 {_split_stats['total']}개 중 "
              f"{_split_stats['split']}개 분리, {_split_stats['skipped']}개 유지")

        tracklet_root = str(tmp_filter)

    else:
        # ── 모드 1: 기존 트랙렛 사용 ─────────────────────────────
        print("[INFO] 모드 1: 기존 트랙렛 사용")
        tracklet_root = args.tracklets_dir or config["data"]["filtered_dir"]

    print(f"\n[INFO] 트랙렛 경로: {tracklet_root}")

    # 트랙렛 로드 (슬롯 필터)
    tracklets = list_tracklets(tracklet_root)
    if args.slots and not args.videos:
        tracklets = [t for t in tracklets
                     if Path(t["tracklet_dir"]).parent.name in args.slots]
    if not tracklets:
        sys.exit("[ERROR] 트랙렛이 없습니다.")
    print(f"[INFO] 트랙렛 {len(tracklets)}개")

    # 특징 추출
    print("\n[STEP 3] Re-ID 특징 추출")
    samples_dir = Path(args.save_samples) if args.save_samples else None
    feats = extract_features(tracklets, extractor, save_samples_dir=samples_dir)
    if samples_dir:
        print(f"[INFO] 대표 이미지 저장: {samples_dir}")

    # 캐시 저장 (--match-only 재실행용)
    np.save(str(out_dir / "_cache_feats.npy"), feats)
    (out_dir / "_cache_tracklets.json").write_text(
        json.dumps(tracklets, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[INFO] 특징 캐시 저장: {out_dir}/_cache_feats.npy")

    # person_id 결정
    has_global_id = all(t.get("global_id") is not None for t in tracklets)
    if has_global_id and not args.videos:
        print("[INFO] 기존 global_id 사용")
        person_ids = [t["global_id"] for t in tracklets]
    else:
        print(f"[INFO] HAC 클러스터링 (threshold={args.threshold})")
        person_ids = run_matching(tracklets, feats, args.threshold, debug=args.debug)

    # demo_data 저장
    print("\n[STEP 4] demo_data 저장")
    persons = save_demo_data(tracklets, person_ids, feats, out_dir, tracklet_root,
                             cam_slot_start=build_cam_slot_start(config))

    print()
    print("=" * 50)
    print(f"  완료: {len(persons)}명 / {len(tracklets)}개 트랙렛")
    print(f"  저장: {(out_dir / 'manifest.json').resolve()}")
    print("=" * 50)
    print()
    print("앱 실행:")
    print(f"  streamlit run scripts/demo_app.py")


if __name__ == "__main__":
    main()
