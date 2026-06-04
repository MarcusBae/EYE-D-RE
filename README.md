# EYE-D Project

Multi-camera Person Re-Identification 데이터셋 구축 및 평가 파이프라인.

---

## ⚙️ 설치

```bash
conda activate eye-d
pip install -r requirements.txt
```

---

## 🚀 파이프라인 실행 순서

### 1단계: 객체 추적

```bash
# YOLOv8n + BoT-SORT, 6프레임마다 추론 (≈ 4fps 효과)
python scripts/run_tracking.py --config configs/config.yaml --frame-stride 6
# 출력: data/tracklets/{cam_slot}/track_NNNN/ (crop 이미지 + metadata.json)
```

### 2단계: 프레임 수 축소

```bash
# frame 번호 기준 간격 조밀한 프레임 삭제, metadata 자동 갱신
python scripts/reduce_tracklet_frames.py --stride 5 --dry-run  # 미리 확인
python scripts/reduce_tracklet_frames.py --stride 5
```

### 3단계: 이미지 리사이즈

```bash
# 256×128 (H×W) in-place 변환 — OSNet/TransReID 공통 입력 크기
python scripts/resize_tracklet_frames.py --dry-run
python scripts/resize_tracklet_frames.py
```

### 4단계: 트랙렛 내부 ID switch 분리

```bash
# OSNet 임베딩 유사도 급락 지점을 경계로 트랙렛 분리
python scripts/split_switched_tracklets.py --config configs/config.yaml --dry-run
python scripts/split_switched_tracklets.py --config configs/config.yaml \
  --sim-threshold 0.70 --min-switch-frames 2 --min-frames 5
```

### 5단계: 동일 cam/slot 내 동일인 재병합

```bash
# 과분리된 트랙렛 쌍을 외형 유사도 기준으로 복원
python scripts/merge_same_camera_tracklets.py --config configs/config.yaml --dry-run
python scripts/merge_same_camera_tracklets.py --config configs/config.yaml \
  --sim-threshold 0.75
```

### 6단계: [선택] 반자동 클리닝

```bash
# FiftyOne UI에서 아웃라이어 프레임 시각 검수
pip install fiftyone
python scripts/fiftyone_curator.py --config configs/config.yaml --sample 20
python scripts/fiftyone_curator.py --config configs/config.yaml
python scripts/fiftyone_curator.py --config configs/config.yaml --delete-tagged
```

### 7단계: 품질 필터링

```bash
# min_track_length=6, avg_conf>=0.7, bbox 64×32px 이상
# 통과 트랙렛만 data/filtered/ 로 복사 (원본 data/tracklets/ 유지)
python scripts/filter_tracklets.py --config configs/config.yaml
```

### 8단계: Cross-camera ID 병합

```bash
# OSNet 임베딩 + 동시성 제약 HAC 클러스터링 → global_id 부여
python scripts/merge_ids.py --config configs/config.yaml
# 출력: 각 트랙렛 metadata.json 에 "global_id" 필드 추가
```

### 9단계: 수동 큐레이션 루프 (오병합 교정)

```bash
# 9-1. global_id별 썸네일 시트 생성 (오병합 육안 확인)
python scripts/export_global_id_sheet.py 32  # global_id=32 확인
# 출력: outputs/sheets/global_0032_sheet.jpg

# 9-2. persons 폴더로 이미지 수집 (수동 편집용)
python scripts/gather_by_person.py --tracklets-dir data/filtered --out data/persons

# 9-3. data/persons/ 에서 이미지 이동·삭제로 오병합 교정
#      person_-001/ 폴더에 미분류 이미지 격리

# 9-4. 교정 결과를 curated 구조로 역변환
python scripts/persons_to_filtered.py --persons-dir data/persons --out data/curated
```

### 10단계: Market-1501 포맷 변환

```bash
# global_id → person ID, 70/30 train/test 분할 (disjoint)
# Query: 2개 이상 카메라 등장 인물만 선발 (cross-camera 평가 보장)
python scripts/format_market1501.py --config configs/config.yaml
# 출력: data/market1501-v1/{bounding_box_train, bounding_box_test, query}/
```

### 11단계: Zero-shot 평가

```bash
# 기본 (single matching)
python scripts/evaluate_zeroshot.py --config configs/config.yaml

# Matching 전략 변경
python scripts/evaluate_zeroshot.py --config configs/config.yaml --matching mean
python scripts/evaluate_zeroshot.py --config configs/config.yaml --matching conf_weighted

# Re-ranking 후처리 추가
python scripts/evaluate_zeroshot.py --config configs/config.yaml --rerank
python scripts/evaluate_zeroshot.py --config configs/config.yaml --matching mean --rerank

# 출력: outputs/eval_results/results_{matching}[_rerank].json
```

---

## 📂 폴더 구조

```
EYE-D-RE/
├── README.md
├── requirements.txt
├── configs/
│   ├── config.yaml                    # 전역 설정 (경로, 하이퍼파라미터)
│   ├── botsort.yaml                   # BoT-SORT 설정 (현재 사용)
│   └── bytetrack.yaml                 # ByteTrack 설정 (레거시)
├── pipeline/
│   ├── detector.py                    # YOLOv8 person detector
│   ├── tracker.py                     # BoT-SORT 래퍼
│   ├── tracklet_io.py                 # tracklet 저장/로드
│   ├── quality_filter.py              # 품질 필터 + thumbnail
│   ├── reid_merger.py                 # OSNet 특징 추출 + 제약 HAC 병합
│   ├── market_formatter.py            # Market-1501 포맷 변환
│   ├── evaluator.py                   # mAP / CMC 평가 엔진 (multi-strategy)
│   ├── reranking.py                   # k-reciprocal re-ranking (Zhong et al.)
│   └── video_utils.py                 # 영상 I/O
├── scripts/
│   ├── run_tracking.py                # 트래킹 실행
│   ├── reduce_tracklet_frames.py      # 프레임 수 축소
│   ├── resize_tracklet_frames.py      # 이미지 리사이즈 (256×128)
│   ├── split_switched_tracklets.py    # 트랙렛 내부 ID switch 분리
│   ├── merge_same_camera_tracklets.py # 동일 cam/slot 동일인 재병합
│   ├── fiftyone_curator.py            # FiftyOne 반자동 클리닝
│   ├── thin_tracklet.py               # 개별 트랙렛 프레임 비율 축소
│   ├── sync_tracklet_metadata.py      # 수동 편집 후 metadata 재동기화
│   ├── filter_tracklets.py            # 품질 필터링
│   ├── merge_ids.py                   # Cross-camera ID 병합
│   ├── gather_by_person.py            # global_id별 이미지 수집 (수동 편집용)
│   ├── persons_to_filtered.py         # persons 편집 결과 → curated 역변환
│   ├── export_global_id_sheet.py      # Global ID 썸네일 시트 생성
│   ├── format_market1501.py           # Market-1501 포맷 변환
│   ├── evaluate_zeroshot.py           # Zero-shot 평가 (--matching, --rerank)
│   ├── analyze_cluster_purity.py      # 클러스터 순도 분석
│   ├── merge_global_ids.py            # global_id 수동 병합
│   ├── find_multi_person.py           # 다중 인물 트랙렛 탐지
│   └── scatter_back.py                # curated → filtered 역전파
├── notebooks/
│   └── 04_image_viewer.ipynb          # tracklet / Market-1501 / Query 뷰어
├── outputs/
│   ├── eval_results/                  # 평가 결과 JSON
│   └── sheets/                        # global_id 썸네일 시트
└── data/                              # (.gitignore)
    ├── raw_videos/                    # 원본 AVI 영상
    ├── tracklets/                     # 자동 트래킹 결과 (전체)
    ├── filtered/                      # 품질 필터 통과본 (자동 global_id)
    ├── curated/                       # 수동 검수 완료본 (persons 기반 global_id)
    ├── persons/                       # 수동 편집용 person 뷰
    └── market1501-v1/                 # Market-1501 포맷 데이터셋
```

---

## 🛣️ 로드맵

| Phase | 내용 | 상태 |
|---|---|---|
| **트래킹** | YOLOv8n + BoT-SORT | ✅ 완료 |
| **전처리** | 프레임 축소 + 리사이즈 | ✅ 완료 |
| **ID switch 정제** | 자동 분리 + 동일인 재병합 | ✅ 완료 |
| **품질 필터링** | 길이·신뢰도·크기 기준 필터 | ✅ 완료 |
| **Cross-camera ID 병합** | OSNet + 제약 HAC + 수동 큐레이션 | ✅ 완료 |
| **Market-1501 변환** | cross-camera query/gallery 분리 | ✅ 완료 |
| **Zero-shot 평가** | mAP / Rank-k, multi-strategy | ✅ 완료 |
| **성능 개선** | Re-ranking, Mean Pooling, OSNet-AIN | 진행 중 |
| **Fine-tuning** | Triplet + CE loss | 다음 단계 |
| **데모** | 오프라인 동선 분석 / 인물 탐색 | 다음 단계 |

---

## 📊 현재 성능 (Zero-shot, 2026-06-04)

| 조합 | Rank-1 | mAP | 비고 |
|---|---|---|---|
| OSNet x1.0, Market-1501, Single | 51.61% | 26.23% | Baseline |
| OSNet x1.0, Market-1501, Re-ranking + Mean | **45.16%** | **54.56%** | 최우수 |

- 데이터: `data/curated` (43명 / 1,216장, 수동 검수)
- 평가: cross-camera query/gallery 분리, Query 31장 / Gallery 430장

---

## ⚠️ 핵심 설계 원칙

1. **고신뢰 검출만 사용**: confidence ≥ 0.5, person class only
2. **고품질 트랙렛만 통과**: 길이 ≥ 6프레임, 평균 conf ≥ 0.7
3. **수동 큐레이션 루프**: 자동 클러스터링 후 persons 폴더에서 오병합 교정
4. **Cross-camera 평가 기준**: 2개 이상 카메라 등장 인물만 query 선발
5. **모든 단계 메타데이터 보존**: 재라벨링·디버깅 용이

---

## 📝 라이선스 & 참고

- YOLOv8: Ultralytics AGPL-3.0
- BoT-SORT: Ultralytics 내장 구현
- OSNet / torchreid: MIT
- k-reciprocal Re-ranking: Zhong et al., CVPR 2017
