# EYE-D Project

## ⚙️ 설치

### 로컬 (Linux / macOS / WSL)

```bash
conda activate eye-d
pip install -r requirements.txt
```

---

## 🚀 사용 방법

### 영상 파일 배치

```
EYE-D-RE/
└── data/
    └── raw_videos/
        ├── cam1_t1.avi (또는 .mp4/.mov/.mkv)
        ├── cam1_t2.avi
        ├── cam2_t1.avi
        ├── cam2_t2.avi
        ├── cam3_t1.avi
        └── cam3_t2.avi
```

### CLI 스크립트 실행 순서

```bash
# 1단계: 객체 추적 (BoT-SORT, AVI 직접 처리)
python scripts/run_tracking.py --config configs/config.yaml --frame-stride 6

# 2단계: 트랙렛 내부 ID switch 자동 분리
python scripts/split_switched_tracklets.py --config configs/config.yaml --dry-run  # 먼저 확인
python scripts/split_switched_tracklets.py --config configs/config.yaml

# 3단계: [선택] 수동 편집 — 자동이 못 잡은 케이스 처리
#         data/tracklets/ 내 잘못된 frame_*.jpg 삭제 후
python scripts/sync_tracklet_metadata.py data/tracklets

# 4단계: 품질 필터링
python scripts/filter_tracklets.py --config configs/config.yaml

# 5단계: Cross-camera ID 병합 (OSNet + 제약 HAC)
python scripts/merge_ids.py --config configs/config.yaml

# 6단계: [선택] Global ID 썸네일 시트로 병합 결과 육안 검수
python scripts/export_global_id_sheet.py <global_id>

# 7단계: Market-1501 포맷 변환
python scripts/format_market1501.py --config configs/config.yaml

# 8단계: Zero-shot 평가
python scripts/evaluate_zeroshot.py --config configs/config.yaml
```

---

## 📂 폴더 구조

```
EYE-D-RE/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── config.yaml                    # 전역 설정 (영상 매핑, 하이퍼파라미터)
│   ├── botsort.yaml                   # Ultralytics BoT-SORT 설정 (현재 사용)
│   └── bytetrack.yaml                 # Ultralytics ByteTrack 설정 (레거시)
├── notebooks/
│   ├── 01_frame_extraction.ipynb
│   ├── 02_bytetrack_tracking.ipynb
│   ├── 03_quality_filtering.ipynb
│   └── 04_image_viewer.ipynb          # Filtered tracklet / Market-1501 / Query 뷰어
├── pipeline/
│   ├── detector.py                    # YOLOv8 person detector
│   ├── tracker.py                     # BoT-SORT 래퍼 (Ultralytics 내장)
│   ├── tracklet_io.py                 # tracklet 저장/로드
│   ├── quality_filter.py              # 품질 필터 + thumbnail
│   ├── reid_merger.py                 # OSNet 특징 추출 + 제약 HAC 병합
│   ├── market_formatter.py            # Market-1501 포맷 변환
│   ├── evaluator.py                   # mAP / CMC 평가 엔진
│   └── video_utils.py                 # 영상 I/O
├── scripts/
│   ├── run_tracking.py                # 트래킹 실행
│   ├── split_switched_tracklets.py    # 트랙렛 내부 ID switch 분리
│   ├── sync_tracklet_metadata.py      # 수동 편집 후 metadata 재동기화
│   ├── filter_tracklets.py            # 품질 필터링
│   ├── merge_ids.py                   # Cross-camera ID 병합
│   ├── export_global_id_sheet.py      # Global ID 썸네일 시트 생성
│   ├── format_market1501.py           # Market-1501 포맷 변환
│   └── evaluate_zeroshot.py           # Zero-shot 평가
└── data/                              # (.gitignore)
    ├── raw_videos/
    ├── tracklets/
    ├── filtered/
    └── market1501/
```

---

## 🛣️ 로드맵

| Phase | 내용 | 상태 |
|---|---|---|
| **1B. 트래킹** | YOLOv8 + BoT-SORT (Re-ID 기반 ID switch 억제) | ✅ 완료 |
| **1C. 품질 필터링** | 길이·신뢰도·크기 기준 필터 | 다음 단계 |
| **1E. Cross-camera ID 병합** | OSNet feature + 제약 HAC 클러스터링 | 다음 단계 |
| **1F. Market-1501 포맷 변환** | train/query/gallery split | 다음 단계 |
| **2. Zero-shot 평가** | mAP / Rank-1/5/10 측정 | 다음 단계 |
| **3. Fine-tuning** | Triplet + CE loss, PK sampling | 다음 단계 |
| **4. 성능 비교/Ablation** | 카메라 쌍별 mAP, t-SNE, CMC | 다음 단계 |
| **5. 실시간 데모** | 3-cam Gradio 데모 + FAISS gallery | 다음 단계 |

---

## ⚠️ 핵심 설계 원칙

1. **고신뢰 검출만 사용**: confidence >= 0.5, person class only
2. **고품질 tracklet만 통과**: 길이 >= 30 frame, 평균 conf >= 0.7
3. **너무 작은/왜곡된 bbox 제거**: min size, aspect ratio 제한
4. **모든 단계 메타데이터 보존**: 추후 재라벨링/디버깅 용이
5. **트랙렛 품질 보장**: BoT-SORT + 자동 ID switch 분리 + 수동 편집 3단계 검수

---

## 📝 라이선스 & 참고

- YOLOv8: Ultralytics AGPL-3.0
- BoT-SORT: Ultralytics 내장 구현
- OSNet: torchreid (MIT)
- 본 프로젝트 코드: 자유 사용 가능 (학습/연구 목적)
