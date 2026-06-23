#!/usr/bin/env python3
"""
scripts/profile_pipeline.py
===========================
EYE-D-RE 파이프라인의 처리 속도 및 RTF(Real-Time Factor)를 프로파일링하는 스크립트.

사용 예:
  # 기본 설정 (data/raw_videos/cam1_t1.avi 비디오의 첫 300프레임 벤치마크)
  python scripts/profile_pipeline.py --config configs/config.yaml

  # 전체 영상 벤치마크 및 프레임 스트라이드 변경
  python scripts/profile_pipeline.py --config configs/config.yaml --video data/raw_videos/cam1_t1.avi --frame-stride 2 --max-frames 1000
"""

import argparse
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# 프로젝트 루트 경로 등록
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from pipeline.tracker import PersonTracker
from pipeline.quality_filter import TrackletQualityFilter
from pipeline.reid_merger import OSNetExtractor, CrossCameraMerger
from pipeline.tracklet_io import list_tracklets


def trim_video(input_path: Path, output_path: Path, max_frames: int) -> tuple[int, float, float]:
    """영상을 지정된 프레임 수만큼 잘라서 임시 영상으로 복사하여 반환."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상 파일을 열 수 없습니다: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        count += 1
        
    cap.release()
    writer.release()
    
    duration = count / fps if fps > 0 else 0.0
    return count, fps, duration


def main():
    parser = argparse.ArgumentParser(description="EYE-D 파이프라인 성능 프로파일러")
    parser.add_argument("--config", default="configs/config.yaml", help="설정 파일 경로")
    parser.add_argument("--video", default="data/raw_videos/cam1_t1.avi", help="프로파일링할 비디오 경로")
    parser.add_argument("--max-frames", type=int, default=300, help="프로파일링용 테스트 프레임 수 제한 (CPU 환경 분석용)")
    parser.add_argument("--frame-stride", type=int, default=6, help="트래킹 프레임 스킵 간격")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[오류] 비디오 파일을 찾을 수 없습니다: {video_path}")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[오류] 설정 파일을 찾을 수 없습니다: {config_path}")
        sys.exit(1)

    import yaml
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print(" 📊 EYE-D-RE 파이프라인 성능 프로파일링 시작")
    print("=" * 60)
    
    # ── 시스템 정보 수집 ──
    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    gpu_model = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"* 실행 OS        : {platform.system()} ({platform.machine()})")
    print(f"* 프로세서       : {platform.processor()}")
    print(f"* 연산 디바이스  : {device_name}")
    if torch.cuda.is_available():
        print(f"* GPU 모델       : {gpu_model}")
    print(f"* PyTorch 버전   : {torch.__version__}")
    print("-" * 60)

    # ── 임시 작업 경로 설정 ──
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_video = tmp_dir / "trimmed_temp.avi"
    tmp_track_dir = tmp_dir / "tracklets"
    tmp_filter_dir = tmp_dir / "filtered"
    tmp_track_dir.mkdir()
    tmp_filter_dir.mkdir()

    try:
        # ── 0단계: 비디오 트리밍 (벤치마크 시간 단축용) ──
        print(f"[준비] 영상 트리밍 중 ({args.max_frames} 프레임 제한)...")
        t_start = time.perf_counter()
        actual_frames, fps, video_duration = trim_video(video_path, tmp_video, args.max_frames)
        t_prep = time.perf_counter() - t_start
        print(f"  - 완료: {actual_frames} 프레임 / 재생시간 {video_duration:.2f}초 (FPS: {fps:.2f})")
        print(f"  - 트리밍 소요 시간: {t_prep:.3f}초")
        print("-" * 60)

        # ── 1단계: Object Detection & Tracking (YOLOv8 + ByteTrack) ──
        det_cfg = config.get("detection", {})
        trk_cfg = config.get("tracking", {})
        
        print("[Step 1] 객체 탐지 및 로컬 추적 (YOLO + ByteTrack) 구동 중...")
        t_start = time.perf_counter()
        
        tracker = PersonTracker(
            model_path=det_cfg.get("model", "yolov8n.pt"),
            tracker_yaml=trk_cfg.get("tracker_yaml", "configs/botsort.yaml"),
            conf=det_cfg.get("conf_threshold", 0.5),
            iou=det_cfg.get("iou_threshold", 0.45),
            imgsz=det_cfg.get("imgsz", 640),
            classes=det_cfg.get("classes", [0]),
            device=det_cfg.get("device", "auto"),
            verbose=False
        )
        
        tracker.track_video(
            video_path=str(tmp_video),
            camera_id=1,
            time_slot=1,
            output_dir=str(tmp_track_dir),
            frame_stride=args.frame_stride,
            save_crops=True,
            checkpoint_every=0  # 벤치마크 중 파일 디스크 I/O 병목 제거
        )
        t_tracking = time.perf_counter() - t_start
        
        # 탐지/추적 결과 파악
        tracklet_slot_dir = tmp_track_dir / "c1_t1"
        raw_tracklets = list_tracklets(str(tmp_track_dir))
        raw_tracklets_count = len(raw_tracklets)
        
        # 1프레임당 소요 시간 계산 (매 Stride마다 1회 추론하므로 추론한 프레임 기준)
        inferred_frames = (actual_frames + args.frame_stride - 1) // args.frame_stride
        tracking_fps = inferred_frames / t_tracking if t_tracking > 0 else 0.0
        ms_per_frame = (t_tracking / inferred_frames) * 1000 if inferred_frames > 0 else 0.0
        
        print(f"  - 소요 시간: {t_tracking:.3f}초")
        print(f"  - 처리 속도: {tracking_fps:.2f} FPS (프레임당 {ms_per_frame:.1f} ms)")
        print(f"  - 추출된 로컬 트랙렛 수: {raw_tracklets_count}개")
        print("-" * 60)

        # ── 2단계: 품질 필터링 (Quality Filter) ──
        qf_cfg = config.get("quality_filter", {})
        print("[Step 2] 트랙렛 품질 필터링 적용 중...")
        t_start = time.perf_counter()
        
        filt = TrackletQualityFilter(
            min_length=qf_cfg.get("min_track_length", 6),
            min_avg_conf=qf_cfg.get("min_avg_confidence", 0.7),
            min_bbox_h=qf_cfg.get("min_bbox_height", 64),
            min_bbox_w=qf_cfg.get("min_bbox_width", 32),
            max_aspect_ratio=qf_cfg.get("max_aspect_ratio", 4.0),
            min_area=qf_cfg.get("min_bbox_area", 2048),
        )
        
        filt.filter_all(
            tracklet_dir=str(tracklet_slot_dir),
            output_dir=str(tmp_filter_dir / "c1_t1"),
            copy_crops=True,
            verbose=False
        )
        t_filter = time.perf_counter() - t_start
        
        filtered_tracklets = list_tracklets(str(tmp_filter_dir))
        filtered_tracklets_count = len(filtered_tracklets)
        pass_rate = (filtered_tracklets_count / raw_tracklets_count * 100) if raw_tracklets_count > 0 else 0.0
        ms_per_filter = (t_filter / max(raw_tracklets_count, 1)) * 1000
        
        print(f"  - 소요 시간: {t_filter:.3f}초 (트랙렛당 {ms_per_filter:.2f} ms)")
        print(f"  - 필터 통과 결과: {filtered_tracklets_count}/{raw_tracklets_count}개 통과 ({pass_rate:.1f}%)")
        print("-" * 60)

        # ── 3단계: Re-ID 특징 추출 (OSNet Feature Extraction) ──
        reid_cfg = config.get("reid", {})
        print("[Step 3] Re-ID 특징 추출 (OSNet) 구동 중...")
        
        t_start = time.perf_counter()
        extractor = OSNetExtractor(
            model_name=reid_cfg.get("model_name", "osnet_x1_0"),
            pretrained=reid_cfg.get("pretrained", True),
            weights_path=reid_cfg.get("weights_path"),
            device=reid_cfg.get("device", "auto"),
        )
        t_model_load = time.perf_counter() - t_start
        print(f"  - Re-ID 모델 로드 완료 ({t_model_load:.3f}초)")
        
        # 총 이미지 크롭 수 계산
        total_crops = sum(len(t.get("crop_files", [])) for t in filtered_tracklets)
        # 각 트랙렛별 최대 8장씩 샘플링하여 특징 추출
        sampled_crops = sum(min(len(t.get("crop_files", [])), 8) for t in filtered_tracklets)
        
        t_start = time.perf_counter()
        feats = []
        for t in filtered_tracklets:
            tdir = Path(t["tracklet_dir"])
            crops = t.get("crop_files", [])
            if len(crops) > 8:
                idx = np.linspace(0, len(crops) - 1, 8, dtype=int)
                crops = [crops[k] for k in idx]
            imgs = [cv2.imread(str(tdir / c)) for c in crops]
            imgs = [im for im in imgs if im is not None]
            if imgs:
                f = extractor.extract_batch_features(imgs, batch_size=reid_cfg.get("batch_size", 64))
                feats.append(f.mean(axis=0))
            else:
                feats.append(np.zeros(512, dtype=np.float32))
                
        t_extraction = time.perf_counter() - t_start
        feats_arr = np.array(feats)
        
        crop_fps = sampled_crops / t_extraction if t_extraction > 0 else 0.0
        ms_per_crop = (t_extraction / max(sampled_crops, 1)) * 1000
        
        print(f"  - 소요 시간: {t_extraction:.3f}초")
        print(f"  - 처리 속도: {crop_fps:.2f} crops/sec (크롭 한 장당 {ms_per_crop:.2f} ms)")
        print(f"  - 총 추출된 특징 행렬 크기: {feats_arr.shape}")
        print("-" * 60)

        # ── 4단계: ID 병합 및 매칭 (HAC Clustering) ──
        print("[Step 4] Cross-Camera ID 병합 (HAC 클러스터링) 구동 중...")
        t_start = time.perf_counter()
        
        merger = CrossCameraMerger(config)
        
        # feats는 리스트 포맷으로 전달
        features_list = [f for f in feats]
        valid_t, valid_f, skipped_t = merger.filter_valid_tracklets(filtered_tracklets, features_list)
        
        dist_matrix = merger.compute_distance_matrix(valid_f)
        constrained_dist_matrix = merger.apply_must_not_link_constraints(dist_matrix, valid_t)
        labels = merger.run_hac_clustering(constrained_dist_matrix)
        labels = merger.enforce_must_not_link(labels, valid_t)
        
        t_matching = time.perf_counter() - t_start
        
        num_global_ids = len(set(labels)) if len(labels) > 0 else 0
        ms_per_match = (t_matching / max(filtered_tracklets_count, 1)) * 1000
        
        print(f"  - 소요 시간: {t_matching:.3f}초 (트랙렛당 {ms_per_match:.2f} ms)")
        print(f"  - 최종 병합 결과: {filtered_tracklets_count}개 트랙렛 → {num_global_ids}명 식별")
        print("=" * 60)

        # ── 종합 결과 리포트 ──
        total_pipeline_time = t_tracking + t_filter + t_extraction + t_matching
        rtf = total_pipeline_time / video_duration if video_duration > 0 else 0.0
        overall_fps = actual_frames / total_pipeline_time if total_pipeline_time > 0 else 0.0
        
        print(" 📈 프로파일링 종합 리포트")
        print("=" * 60)
        print(f"* 테스트 비디오  : {video_path.name}")
        print(f"* 비디오 재생시간: {video_duration:.2f} 초")
        print(f"* 총 처리 프레임 : {actual_frames} 프레임")
        print(f"* 총 처리 시간   : {total_pipeline_time:.3f} 초")
        print("-" * 60)
        print(f"* 실시간 처리 배수 (RTF)      : {rtf:.3f}")
        if rtf < 1.0:
            print(f"  └> 결과: 실시간 처리 가능! (실시간 대비 {1/rtf:.1f}배 빠름)")
        else:
            print(f"  └> 결과: 실시간 처리 불가능 (실시간 대비 {rtf:.1f}배 느림)")
            
        print(f"* 파이프라인 평균 속도 (E2E) : {overall_fps:.2f} FPS")
        print("-" * 60)
        
        # 표 형식 출력
        print(f"{'파이프라인 단계':<28} | {'소요 시간 (초)':<12} | {'점유율 (%)':<8} | {'상세 속도 지표':<25}")
        print("-" * 65)
        steps = [
            ("1. Detection & Tracking (YOLO)", t_tracking, f"{tracking_fps:.1f} FPS"),
            ("2. Quality Filtering", t_filter, f"{ms_per_filter:.2f} ms/tracklet"),
            ("3. Re-ID Feature Ext. (OSNet)", t_extraction, f"{crop_fps:.1f} crops/sec"),
            ("4. HAC Clustering (Merger)", t_matching, f"{ms_per_match:.2f} ms/tracklet"),
        ]
        for name, duration, detail in steps:
            share = (duration / total_pipeline_time * 100) if total_pipeline_time > 0 else 0.0
            print(f"{name:<28} | {duration:<12.3f} | {share:<8.1f} | {detail:<25}")
        print("=" * 60)

    finally:
        # 임시 폴더 클린업
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
