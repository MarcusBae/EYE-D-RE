#!/usr/bin/env python3
"""
scripts/profile_latency.py
===========================
EYE-D-RE 파이프라인의 Latency(지연 시간) 및 처리 속도(Throughput)를 
단계별(Decode -> Detect -> Track -> Embed -> Cluster)로 상세 프로파일링하고, 
결과 보고서 및 시각적 바 차트를 생성하는 스크립트.

사용 예:
  python scripts/profile_latency.py --config configs/config.yaml --video data/raw_videos/cam1_t3.avi --max-frames 500 --frame-stride 4
"""

from __future__ import annotations

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

from pipeline.tracker import PersonTracker, _save_tracklets_to_disk
from pipeline.quality_filter import TrackletQualityFilter
from pipeline.reid_merger import OSNetExtractor, CrossCameraMerger
from pipeline.tracklet_io import list_tracklets


def generate_ascii_bar(shares: list[float], labels: list[str], width: int = 80) -> str:
    """각 단계의 점유율에 맞는 ASCII/Unicode 블록 가로 바 차트를 생성."""
    # 블록 문자 정의
    block_chars = ["░", "▒", "▓", "█", "█"]
    colors = [
        ("\033[94m", "\033[0m"), # Light Blue
        ("\033[91m", "\033[0m"), # Red
        ("\033[93m", "\033[0m"), # Yellow
        ("\033[92m", "\033[0m"), # Green
        ("\033[95m", "\033[0m"), # Magenta
    ]
    
    bar_str = ""
    legend_str = "\n  "
    
    accumulated = 0.0
    for idx, (share, label) in enumerate(zip(shares, labels)):
        if share <= 0:
            continue
        char_count = max(1, round((share / 100.0) * width))
        # 바의 마지막 문자 수 보정
        if idx == len(shares) - 1:
            current_len = len(bar_str.replace("\033[94m","").replace("\033[91m","").replace("\033[93m","").replace("\033[92m","").replace("\033[95m","").replace("\033[0m",""))
            if current_len < width:
                char_count += (width - current_len)
        
        color_start, color_end = colors[idx % len(colors)]
        # 블록 패턴
        pattern = block_chars[idx % len(block_chars)] * char_count
        bar_str += f"{color_start}{pattern}{color_end}"
        
        # 레전드 구성
        legend_str += f"{color_start}■{color_end} {label} ({share:.1f}%)   "
        if (idx + 1) % 3 == 0:
            legend_str += "\n  "
            
    return f"  [{bar_str}]\n{legend_str}"


def main():
    parser = argparse.ArgumentParser(description="EYE-D 파이프라인 Latency 상세 프로파일러")
    parser.add_argument("--config", default="configs/config.yaml", help="설정 파일 경로")
    parser.add_argument("--video", default="data/raw_videos/cam1_t3.avi", help="프로파일링할 비디오 경로")
    parser.add_argument("--max-frames", type=int, default=300, help="프로파일링할 최대 프레임 수")
    parser.add_argument("--frame-stride", type=int, default=4, help="트래킹 프레임 스킵 간격")
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

    # ── 임시 작업 경로 설정 ──
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_track_dir = tmp_dir / "tracklets"
    tmp_filter_dir = tmp_dir / "filtered"
    tmp_track_dir.mkdir()
    tmp_filter_dir.mkdir()

    print("=" * 70)
    print(" 📊 EYE-D-RE 파이프라인 Latency 및 Throughput 상세 프로파일링")
    print("=" * 70)
    print(f"* 비디오 소스    : {video_path.name}")
    print(f"* 최대 프레임    : {args.max_frames} frames")
    print(f"* 프레임 스트라이드: {args.frame_stride}")
    print("-" * 70)

    try:
        # ── 비디오 로딩 ──
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_limit = min(args.max_frames, total_frames_in_video)

        # ── 1단계: Object Detection & Tracking (YOLOv8 + ByteTrack) 상세 시간 측정 ──
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
            verbose=False
        )

        out_video_dir = tmp_track_dir / "c1_t1"
        out_video_dir.mkdir(parents=True, exist_ok=True)

        tracklets = {}
        id_offset = 0
        max_used_id = 0

        # 상세 타이머 변수
        total_decode_time = 0.0
        total_preprocess_time = 0.0
        total_inference_time = 0.0
        total_postprocess_time = 0.0

        print("\n[Step 1 & 2] Frame Decoding & YOLO + Tracking 수행 중...")
        for frame_idx in range(frame_limit):
            # 1. Decode & Preprocess (OpenCV frame read)
            t_dec_start = time.perf_counter()
            ret, frame = cap.read()
            t_dec = time.perf_counter() - t_dec_start
            total_decode_time += t_dec

            if not ret:
                break

            if frame_idx % args.frame_stride == 0:
                # 2. YOLO Detection & Tracking
                t_track_start = time.perf_counter()
                results = tracker.model.track(
                    frame,
                    conf=tracker.conf,
                    iou=tracker.iou,
                    imgsz=tracker.imgsz,
                    classes=tracker.classes,
                    device=tracker.device,
                    persist=True,
                    tracker=tracker.tracker_yaml,
                    verbose=False
                )
                t_track_end = time.perf_counter() - t_track_start
                
                # Ultralytics 속도 정보 (ms단위)
                speed = results[0].speed
                prep = speed.get('preprocess', 0.0) / 1000.0
                infer = speed.get('inference', 0.0) / 1000.0
                post = t_track_end - prep - infer
                if post < 0:
                    post = speed.get('postprocess', 0.0) / 1000.0

                total_preprocess_time += prep
                total_inference_time += infer
                total_postprocess_time += post

                # BBox 크롭 및 트랙렛 데이터 수집
                r = results[0]
                if r.boxes is not None and r.boxes.id is not None:
                    ids = r.boxes.id.cpu().numpy().astype(int)
                    bboxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                    confs = r.boxes.conf.cpu().numpy().astype(np.float32)

                    for tid, bbox, conf in zip(ids, bboxes, confs):
                        actual_tid = int(tid) + id_offset
                        if actual_tid > max_used_id:
                            max_used_id = actual_tid

                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        h, w = frame.shape[:2]
                        x1c, y1c = max(0, x1), max(0, y1)
                        x2c, y2c = min(w, x2), min(h, y2)
                        if x2c <= x1c or y2c <= y1c:
                            continue

                        entry = {
                            "frame_idx": int(frame_idx),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "conf": float(conf),
                            "_crop": frame[y1c:y2c, x1c:x2c].copy()
                        }
                        tracklets.setdefault(actual_tid, []).append(entry)

        cap.release()

        # 메모리에 저장된 트랙렛을 디스크에 저장
        _save_tracklets_to_disk(
            tracklets, 1, 1, str(video_path), src_fps,
            args.frame_stride, True, out_video_dir
        )

        raw_tracklets = list_tracklets(str(tmp_track_dir))
        raw_tracklets_count = len(raw_tracklets)

        # ── 2단계: 품질 필터링 (Quality Filter) ──
        qf_cfg = config.get("quality_filter", {})
        filt = TrackletQualityFilter(
            min_length=qf_cfg.get("min_track_length", 6),
            min_avg_conf=qf_cfg.get("min_avg_confidence", 0.7),
            min_bbox_h=qf_cfg.get("min_bbox_height", 64),
            min_bbox_w=qf_cfg.get("min_bbox_width", 32),
            max_aspect_ratio=qf_cfg.get("max_aspect_ratio", 4.0),
            min_area=qf_cfg.get("min_bbox_area", 2048),
        )
        
        filt.filter_all(
            tracklet_dir=str(out_video_dir),
            output_dir=str(tmp_filter_dir / "c1_t1"),
            copy_crops=True,
            verbose=False
        )
        filtered_tracklets = list_tracklets(str(tmp_filter_dir))
        filtered_tracklets_count = len(filtered_tracklets)

        # ── 3단계: Re-ID 특징 추출 (OSNet Feature Extraction) 시간 측정 ──
        reid_cfg = config.get("reid", {})
        extractor = OSNetExtractor(
            model_name=reid_cfg.get("model_name", "osnet_x1_0"),
            pretrained=reid_cfg.get("pretrained", True),
            weights_path=reid_cfg.get("weights_path"),
            device=reid_cfg.get("device", "auto"),
        )

        sampled_crops = sum(min(len(t.get("crop_files", [])), 8) for t in filtered_tracklets)

        print("[Step 3] OSNet 특징 추출 수행 중...")
        t_embed_start = time.perf_counter()
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

        t_extraction = time.perf_counter() - t_embed_start

        # ── 4단계: ID 병합 및 매칭 (HAC Clustering) 시간 측정 ──
        print("[Step 4] Cross-Camera ID 병합 (HAC) 수행 중...")
        t_hac_start = time.perf_counter()
        
        merger = CrossCameraMerger(config)
        features_list = [f for f in feats]
        valid_t, valid_f, skipped_t = merger.filter_valid_tracklets(filtered_tracklets, features_list)
        
        num_clusters = 0
        if len(valid_f) > 0:
            dist_matrix = merger.compute_distance_matrix(valid_f)
            constrained_dist_matrix = merger.apply_must_not_link_constraints(dist_matrix, valid_t)
            labels = merger.run_hac_clustering(constrained_dist_matrix)
            labels = merger.enforce_must_not_link(labels, valid_t)
            num_clusters = len(set(labels))
        
        t_matching = time.perf_counter() - t_hac_start

        # ── Latency 및 Throughput 수치화 계산 ──
        # 총 소요 시간 합산
        stage1_time = total_decode_time + total_preprocess_time
        stage2_time = total_inference_time
        stage3_time = total_postprocess_time
        stage4_time = t_extraction
        stage5_time = t_matching

        total_pipeline_time = stage1_time + stage2_time + stage3_time + stage4_time + stage5_time

        # ms / frame 단위 기여도 계산
        lat_dec = (stage1_time / frame_limit) * 1000
        lat_det = (stage2_time / frame_limit) * 1000
        lat_trk = (stage3_time / frame_limit) * 1000
        lat_emb = (stage4_time / frame_limit) * 1000
        lat_hac = (stage5_time / frame_limit) * 1000
        total_lat_ms = (total_pipeline_time / frame_limit) * 1000

        # 각 단계 점유율 (%)
        share_dec = (stage1_time / total_pipeline_time) * 100
        share_det = (stage2_time / total_pipeline_time) * 100
        share_trk = (stage3_time / total_pipeline_time) * 100
        share_emb = (stage4_time / total_pipeline_time) * 100
        share_hac = (stage5_time / total_pipeline_time) * 100

        # 처리량 (Throughput)
        tp_dec = 1000.0 / lat_dec if lat_dec > 0 else 0.0
        tp_det = 1000.0 / lat_det if lat_det > 0 else 0.0
        tp_trk = 1000.0 / lat_trk if lat_trk > 0 else 0.0
        tp_emb = sampled_crops / stage4_time if stage4_time > 0 else 0.0
        tp_hac_ms = (t_matching * 1000)

        # ── 5. 결과 시각화 및 터미널 출력 ──
        device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        gpu_model = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        
        print("\n" + "=" * 80)
        print(" 시스템 정보 요약")
        print("=" * 80)
        print(f" * 실행 OS        : {platform.system()} ({platform.machine()})")
        print(f" * CPU 프로세서   : {platform.processor()}")
        print(f" * 연산 디바이스  : {device_name}")
        if torch.cuda.is_available():
            print(f" * GPU 모델       : {gpu_model}")
    print(f" * PyTorch 버전   : {torch.__version__}")
        print("=" * 80)

        print("\n" + "=" * 80)
        print(" Latency 점유율 바 차트 (Decode -> Detect -> Track -> Embed -> Cluster)")
        print("=" * 80)
        shares = [share_dec, share_det, share_trk, share_emb, share_hac]
        labels = ["Decode", "Detect", "Track", "Embed", "HAC"]
        bar_chart = generate_ascii_bar(shares, labels, width=65)
        print(bar_chart)
        print("=" * 80)

        batch_size = reid_cfg.get("batch_size", 64)
        details_stage4 = f"per crop · batch={batch_size}"
        details_stage5 = f"N = {filtered_tracklets_count} tracklets"
        print(f" {'#':<2} | {'Stage / Operator':<22} | {'Details':<25} | {'Latency (ms)':<12} | {'Share':<6} | {'Throughput':<15}")
        print("-" * 92)
        print(f" 01 | {'Decode & Pre-process':<22} | {'OpenCV · BGR->RGB · resize':<25} | {lat_dec:<12.1f} | {share_dec:<5.1f}% | {tp_dec:<4.0f} fps")
        print(f" 02 | {'YOLOv8 Detection':<22} | {'person class only':<25} | {lat_det:<12.1f} | {share_det:<5.1f}% | {tp_det:<4.0f} fps")
        print(f" 03 | {'BoT-SORT Tracking':<22} | {'IoU · ReID · Kalman':<25} | {lat_trk:<12.1f} | {share_trk:<5.1f}% | {tp_trk:<4.0f} fps")
        print(f" 04 | {'OSNet Embedding':<22} | {details_stage4:<25} | {lat_emb:<12.1f} | {share_emb:<5.1f}% | {tp_emb:<4.0f} crops/s")
        print(f" 05 | {'HAC Clustering':<22} | {details_stage5:<25} | {lat_hac:<12.1f} | {share_hac:<5.1f}% | {filtered_tracklets_count} trk / {tp_hac_ms:.1f} ms")
        print("-" * 92)
        print(f" {'':<2}   {'Total Pipeline Latency':<22}   {'':<25}   {total_lat_ms:<12.1f}   {'100%':<6}   {1000.0/total_lat_ms:<4.1f} fps (E2E)")
        print("=" * 80)

        # 주요 병목 및 안내 구문 출력
        det_embed_share = share_det + share_emb
        print(f"\n | Detection + Embedding = {det_embed_share:.1f}% — 추가 가속은 YOLO 경량화 또는 OSNet 배치 처리에서 가장 큰 효과 기대.")
        print(f" | 프레임 스트라이드 적용 상태로 평균 실시간 배율 (RTF): {total_pipeline_time / (frame_limit / src_fps):.3f}")
        print("=" * 80)

        # ── 6. 파일 보고서 출력 (outputs/latency_report.md) ──
        report_dir = project_root / "outputs"
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / "latency_report.md"
        
        markdown_content = f"""# EYE-D-RE 파이프라인 Latency 프로파일링 보고서

본 보고서는 `{video_path.name}` 영상의 `{frame_limit}`개 프레임을 기준으로 파이프라인 단계별 소요 시간과 처리 성능을 측정하여 작성되었습니다.

## 1. 시스템 정보
- **OS**: {platform.system()} ({platform.machine()})
- **CPU**: {platform.processor()}
- **GPU**: {gpu_model} (디바이스: {device_name})
- **PyTorch**: {torch.__version__}
- **OpenCV**: {cv2.__version__}

## 2. 프로파일링 요약
- **총 프레임 수**: {frame_limit} frames
- **E2E 전체 파이프라인 지연**: {total_lat_ms:.2f} ms / frame
- **E2E 파이프라인 처리량**: {1000.0/total_lat_ms:.2f} fps
- **실시간 계수 (RTF)**: {total_pipeline_time / (frame_limit / src_fps):.3f} (1.0 미만 시 실시간 가동 가능)

## 3. 단계별 Latency 분포
| # | Stage / Operator | Details | Latency (ms/frame) | Share (%) | Throughput |
| :--- | :--- | :--- | :---: | :---: | :--- |
| 01 | **Decode & Pre-process** | OpenCV · BGR->RGB · resize | {lat_dec:.2f} | {share_dec:.1f}% | {tp_dec:.0f} fps |
| 02 | **YOLOv8 Detection** | person class only | {lat_det:.2f} | {share_det:.1f}% | {tp_det:.0f} fps |
| 03 | **BoT-SORT Tracking** | IoU · ReID · Kalman | {lat_trk:.2f} | {share_trk:.1f}% | {tp_trk:.0f} fps |
| 04 | **OSNet Embedding** | per crop · batch={reid_cfg.get("batch_size", 64)} | {lat_emb:.2f} | {share_emb:.1f}% | {tp_emb:.0f} crops/s |
| 05 | **HAC Clustering** | N = {filtered_tracklets_count} tracklets | {lat_hac:.2f} | {share_hac:.1f}% | {filtered_tracklets_count} trk / {tp_hac_ms:.1f} ms |
| | **Total Pipeline** | | **{total_lat_ms:.2f}** | **100%** | **{1000.0/total_lat_ms:.2f} fps** |

## 4. 분석 결과 해석
- **핵심 병목 구간**: **YOLOv8 Detection 및 OSNet Embedding** 두 단계가 전체 연산의 **{det_embed_share:.1f}%**를 차지하고 있습니다.
- **최적화 가이드**: 추가 성능 향상을 위해 YOLO 모델 경량화(예: `yolov8n.pt` 사용 확인) 및 OSNet의 추론 배치 크기(`reid.batch_size` 조율) 및 TensorRT 가속 적용을 검토할 것을 권장합니다.
"""
        report_path.write_text(markdown_content, encoding="utf-8")
        print(f"\n[Finished] 상세 지연시간 보고서가 저장되었습니다: {report_path.resolve()}")

    finally:
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
