#!/usr/bin/env python
"""
merge_ids.py
============
CCTV 트랙렛들의 OSNet 외형 피처를 추출하고, 
동시성 제약 조건이 가미된 HAC(Hierarchical Agglomerative Clustering)를 사용해
Global ID 병합하는 배치 실행 스크립트.

입력 (Input):
    1. 설정 파일 (configs/config.yaml): 데이터 경로 및 클러스터링 임계값 정보 등
    2. 품질 필터 통과 트랙렛 폴더 (data/filtered/{camera_id}_{time_slot}/track_{NNNN}/):
        - frame_XXXXXX.jpg: 보행자 크롭 이미지들
        - metadata.json: 트랙렛 정보 (기본 카메라, 시간대 등 포함)
    3. Re-ID 모델 가중치 (OSNet 모델 가중치)

출력 (Output):
    1. 업데이트된 metadata.json:
        - 각 트랙렛 폴더 안의 metadata.json 파일에 "global_id" 필드(동일인 식별 ID)가 추가 및 기록됩니다.
        - 제외된 트랙렛에는 global_id가 -1로 설정됩니다.
    2. 콘솔 통계:
        - 병합 전후의 ID 개수 비교, ID 병합 압축률, 상위 Global ID별 카메라 분산 매핑 통계 등 출력
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

# 프로젝트 루트 경로 추가 (src 모듈 임포트용)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from pipeline.tracklet_io import list_tracklets
from pipeline.reid_merger import CrossCameraMerger


def _load_feature_cache(cache_path: Path, tracklets: list) -> list | None:
    """캐시가 존재하고 트랙렛 목록이 일치하면 특징 벡터 리스트 반환, 아니면 None."""
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=True)
        cached_dirs = data["tracklet_dirs"].tolist()
        current_dirs = [t["tracklet_dir"] for t in tracklets]
        if cached_dirs != current_dirs:
            print("[INFO] 캐시 트랙렛 목록 불일치 → 재추출")
            return None
        print(f"[INFO] 특징 캐시 로드: {cache_path}  ({len(tracklets)}개 트랙렛)")
        return list(data["features"])
    except Exception as e:
        print(f"[WARN] 캐시 로드 실패 ({e}) → 재추출")
        return None


def _save_feature_cache(cache_path: Path, features: list, tracklets: list) -> None:
    """특징 벡터 리스트를 npz 파일로 저장."""
    features_arr = np.empty(len(features), dtype=object)
    for i, f in enumerate(features):
        features_arr[i] = f
    np.savez(
        cache_path,
        features=features_arr,
        tracklet_dirs=np.array([t["tracklet_dir"] for t in tracklets]),
    )
    print(f"[INFO] 특징 캐시 저장: {cache_path}  ({len(features)}개 트랙렛)")


def main():
    parser = argparse.ArgumentParser(description="Cross-camera ID Merger (Phase 1E)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to global configuration YAML file"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="캐시를 무시하고 특징 벡터를 강제 재추출"
    )
    args = parser.parse_args()

    # 1. Config 로드
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] 설정을 찾을 수 없습니다: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    filtered_dir = config.get("data", {}).get("filtered_dir", "data/filtered")
    cluster_cfg = config.get("reid", {}).get("clustering", {})
    threshold = cluster_cfg.get("threshold", 0.5)
    allow_cross_slot = cluster_cfg.get("allow_cross_slot", False)

    print(f"[INFO] 작업 대상 폴더    : {filtered_dir}")
    print(f"[INFO] HAC threshold     : {threshold}")
    print(f"[INFO] allow_cross_slot  : {allow_cross_slot}")
    print()
    print(f"[INFO] 1단계: 정제 완료된 트랙렛 폴더 스캔: {filtered_dir}")

    # 2. 정제 완료된 트랙렛 리스트 수집
    tracklets = list_tracklets(filtered_dir)
    if not tracklets:
        print(f"[WARNING] '{filtered_dir}' 내에 트랙렛이 존재하지 않습니다. 먼저 filter_tracklets.py를 실행하세요.")
        sys.exit(0)

    print(f"[INFO] 총 {len(tracklets)}개의 정제된 트랙렛이 탐색되었습니다.")

    # 3. Merger 초기화 및 피처 추출 (캐시 활용)
    merger = CrossCameraMerger(config)

    cache_path = Path(filtered_dir) / ".feature_cache.npz"
    raw_features = None

    if not args.no_cache:
        raw_features = _load_feature_cache(cache_path, tracklets)

    if raw_features is None:
        try:
            raw_features = merger.extract_tracklet_representative_features(tracklets)
        except Exception as e:
            print(f"[ERROR] 특징 추출 중 오류 발생: {e}")
            print("[TIP] PyTorch Hub 연결 오류인 경우, 인터넷 연결을 확인하거나 torchreid 패키지를 가상환경에 설치해 주세요.")
            sys.exit(1)
        _save_feature_cache(cache_path, raw_features, tracklets)

    # [P2-1] 이미지 로드 불가 트랙렛 분리
    valid_tracklets, valid_features, skipped_tracklets = merger.filter_valid_tracklets(tracklets, raw_features)

    # 4. 거리 행렬 계산
    print("[INFO] 2단계: 트랙렛 간 유사도 거리 행렬 계산 중...")
    dist_matrix = merger.compute_distance_matrix(valid_features)

    # 5. 동시성 제약 조건 적용
    print("[INFO] 3단계: 동일 카메라 및 시간대 동시성 제약(Must-not-link Constraint) 적용 중...")
    constrained_dist_matrix = merger.apply_must_not_link_constraints(dist_matrix, valid_tracklets)

    # 6. HAC 클러스터링 실행
    print("[INFO] 4단계: 계층적 군집화(HAC) 실행 중...")
    labels = merger.run_hac_clustering(constrained_dist_matrix)

    # [P2-2] HAC 후 제약 위반 클러스터 강제 분리
    print("[INFO] 4-1단계: Must-not-link 위반 클러스터 강제 분리 중...")
    labels = merger.enforce_must_not_link(labels, valid_tracklets)

    # 7. 클러스터 검증
    print("[INFO] 5단계: 클러스터 병합 결과 제약조건 검증 중...")
    is_valid = merger.verify_clustering_results(labels, valid_tracklets)
    if is_valid:
        print("[SUCCESS] 클러스터 검증 완료: 모든 동시성 제약 조건이 완벽히 준수되었습니다!")
    else:
        print("[WARNING] 클러스터 검증 결과 일부 제약 조건이 위반되었을 가능성이 있습니다. 로그를 확인하세요.")

    # 8. 최종 Global ID 메타데이터 저장
    print("[INFO] 6단계: 각 트랙렛의 metadata.json에 Global ID 기록 중...")
    merger.update_tracklet_metadata_with_global_id(valid_tracklets, labels)
    merger.mark_skipped_tracklets(skipped_tracklets)

    # 9. 결과 요약 통계 출력
    num_original_tracks = len(tracklets)
    num_global_ids = len(set(labels))
    compression_ratio = (1.0 - (num_global_ids / num_original_tracks)) * 100

    print("\n" + "="*50)
    print("🎯 Cross-camera ID 병합 (Phase 1E) 완료 요약")
    print("="*50)
    print(f"* 정제 트랙렛(로컬 ID) 총수  : {num_original_tracks} 개")
    print(f"* 병합 후 Global ID 총수     : {num_global_ids} 개")
    print(f"* ID 병합 압축률             : {compression_ratio:.1f} %")
    
    # 카메라별/슬롯별 병합 분포 출력
    cam_slot_counts = {}
    for t, label in zip(valid_tracklets, labels):
        key = f"c{t['camera_id']}_t{t['time_slot']}"
        cam_slot_counts.setdefault(label, []).append(key)

    print("\n* 주요 Global ID 매핑 정보 (상위 10개):")
    sorted_global_ids = sorted(
        [(gid, paths) for gid, paths in cam_slot_counts.items()],
        key=lambda x: len(x[1]),
        reverse=True
    )
    for gid, paths in sorted_global_ids[:10]:
        unique_cams = set([p.split('_')[0] for p in paths])
        print(f"  - Global ID {gid:03d} : 총 {len(paths)}회 매핑 ({', '.join(sorted(set(paths)))}) [카메라 {len(unique_cams)}대 분산]")
        
    print("="*50)
    print("[SUCCESS] Cross-camera ID 병합이 성공적으로 끝났습니다. 다음 단계는 Phase 1F(Market-1501 변환)입니다.")


if __name__ == "__main__":
    main()
