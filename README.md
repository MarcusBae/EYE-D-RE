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
# ─────────────────────────────────────────────
# 1단계: 객체 추적 (BoT-SORT)
# ─────────────────────────────────────────────
# data/raw_videos/ 의 AVI 파일을 읽어 사람을 탐지·추적.
# 결과는 data/tracklets/ 하위에 cam/slot별 폴더로 저장됨.
# (예: data/tracklets/c1_t3/track_0012/frame_*.jpg + metadata.json)
# --frame-stride 6 = 24fps 영상에서 6프레임마다 1장 처리 (= 4fps 효과)
python scripts/run_tracking.py --config configs/config.yaml --frame-stride 6

# ─────────────────────────────────────────────
# 2단계: 프레임 수 축소
# ─────────────────────────────────────────────
# 각 트랙렛에서 frame 번호 기준으로 간격이 좁은 프레임을 삭제.
# --stride 5: frame_stride(6) × 5 = 30 video frames = 1.25초 간격만 유지.
# 205,000장 → 약 41,000장으로 축소. metadata.json 자동 갱신.
# 대상 폴더: data/tracklets-1-original-reduced (config와 무관, 스크립트 기본값)
python scripts/reduce_tracklet_frames.py --stride 5 --dry-run  # 삭제 예정 장수 확인
python scripts/reduce_tracklet_frames.py --stride 5

# ─────────────────────────────────────────────
# 3단계: 이미지 리사이즈
# ─────────────────────────────────────────────
# 모든 프레임을 256×128 (H×W) 로 in-place 변환.
# OSNet, TransReID, CLIP-ReID 등 주요 Re-ID 모델의 공통 입력 크기.
# 128×64 (Market-1501 표준)로 줄이면 나중에 ViT 모델 사용 시 upscale 손실 발생.
python scripts/resize_tracklet_frames.py --dry-run  # 용량 변화 미리 확인
python scripts/resize_tracklet_frames.py

# ─────────────────────────────────────────────
# 4단계: 트랙렛 내부 ID switch 자동 분리
# ─────────────────────────────────────────────
# 하나의 트랙렛 안에 다른 사람이 섞인 경우를 탐지해 분리.
# OSNet 임베딩을 슬라이딩 앵커와 비교하여 유사도가 급락하는 지점을 경계로 설정.
# --sim-threshold 0.70: 앵커 대비 cosine 유사도가 0.70 미만이면 이물질로 판단
# --min-switch-frames 2: 2프레임 연속 이상해야 분리 (1이면 과분리 위험)
# 짧은 구간도 삭제하지 않고 별도 트랙렛으로 생성 (이후 필터링에서 처리)
python scripts/split_switched_tracklets.py --config configs/config.yaml --dry-run
python scripts/split_switched_tracklets.py --config configs/config.yaml \
  --sim-threshold 0.70 --min-switch-frames 2 --min-frames 5

# ─────────────────────────────────────────────
# 5단계: 동일 cam/slot 내 동일인 트랙렛 재병합
# ─────────────────────────────────────────────
# 4단계 과분리 또는 tracker가 동일인을 잠시 놓쳤다 재획득한 경우를 복원.
# 같은 cam/slot 안에서 시간이 겹치지 않고(시간 비겹침 필수) 외형이 유사한
# 트랙렛 쌍을 병합. 시간상 앞선 트랙렛 폴더로 뒤 트랙렛을 흡수.
# --sim-threshold 0.75: 대표 임베딩 cosine 유사도 기준 (높을수록 보수적)
# --max-gap-frames 300: 두 트랙렛 사이 허용 최대 frame 간격 (약 75초)
python scripts/merge_same_camera_tracklets.py --config configs/config.yaml --dry-run
python scripts/merge_same_camera_tracklets.py --config configs/config.yaml \
  --sim-threshold 0.75

# ─────────────────────────────────────────────
# 6단계: [선택] FiftyOne 반자동 클리닝
# ─────────────────────────────────────────────
# 자동으로 잡지 못한 아웃라이어 프레임을 브라우저 UI에서 시각 검수.
# OSNet 임베딩으로 각 트랙렛 내 아웃라이어 점수를 계산하여 표시.
# 사용 흐름:
#   1) 아래 명령으로 앱 실행 → http://localhost:5151 접속
#   2) is_suspicious=True 필터로 의심 프레임 확인
#   3) 삭제할 프레임 선택 후 태그: "delete"
#   4) Ctrl+C 종료 후 --delete-tagged 로 실제 파일 삭제
# --sample 20: 처음 20개 트랙렛만 로드 (동작 확인용)
pip install fiftyone  # 최초 1회
python scripts/fiftyone_curator.py --config configs/config.yaml --sample 20
python scripts/fiftyone_curator.py --config configs/config.yaml          # 전체 실행
python scripts/fiftyone_curator.py --config configs/config.yaml --delete-tagged  # 태그 파일 삭제

# ─────────────────────────────────────────────
# 6-1단계: [선택] 특정 트랙렛 프레임 수 줄이기
# ─────────────────────────────────────────────
# 이미지가 너무 많은 특정 트랙렛만 골라서 균등하게 솎아낼 때 사용.
# --keep 1/2: 전체의 절반만 균등 간격으로 유지 (나머지 삭제)
# --keep 2/3: 전체의 2/3 유지, 1/3 삭제
# 실행 후 metadata.json 자동 갱신.
python scripts/thin_tracklet.py data/filtered/c1_t10/track_0067 --keep 1/2 --dry-run
python scripts/thin_tracklet.py data/filtered/c1_t10/track_0067 --keep 1/2

# ─────────────────────────────────────────────
# 6-2단계: [선택] data/filtered 수동 편집 후 metadata 동기화
# ─────────────────────────────────────────────
# data/filtered/ 내 frame_*.jpg 를 직접 삭제했을 경우 metadata.json 을 실제 파일 기준으로 갱신.
# thin_tracklet.py 는 metadata 자동 갱신하므로 이 단계 불필요.
# 직접 파일 탐색기 등으로 삭제한 경우에만 실행.
# data/filtered/ 는 filter_tracklets.py 로 언제든 재생성 가능하므로 부담 없이 편집 가능.
python scripts/sync_tracklet_metadata.py data/filtered

# ─────────────────────────────────────────────
# 7단계: 품질 필터링
# ─────────────────────────────────────────────
# config.yaml의 quality_filter 기준으로 불량 트랙렛 제거.
# 기준: 길이 >= 6프레임 (stride 5 적용 후 0.8fps 기준 약 7.5초),
#       평균 confidence >= 0.7, bbox 높이 >= 64px, 너비 >= 32px,
#       aspect ratio <= 4.0, bbox 면적 >= 2048px²
# ※ reduce_tracklet_frames --stride 5 미적용 원본 데이터 사용 시
#   min_track_length를 30으로 되돌릴 것.
# 통과한 트랙렛만 data/filtered/ 로 복사 (원본 data/tracklets/ 유지).
python scripts/filter_tracklets.py --config configs/config.yaml

# ─────────────────────────────────────────────
# 8단계: Cross-camera ID 병합
# ─────────────────────────────────────────────
# data/filtered/ 의 트랙렛들을 카메라 간 동일인으로 묶어 Global ID 부여.
# OSNet으로 트랙렛별 대표 임베딩 추출 → 코사인 거리 행렬 계산
# → 동시성 제약 HAC 클러스터링 (같은 cam/slot의 다른 트랙 = must-not-link)
# 결과: 각 트랙렛 metadata.json 에 "global_id" 필드 추가.
python scripts/merge_ids.py --config configs/config.yaml

# ─────────────────────────────────────────────
# 9단계: [선택] Global ID 썸네일 시트 육안 검수
# ─────────────────────────────────────────────
# 특정 global_id 에 묶인 모든 트랙렛의 썸네일을 한 장으로 합성해 저장.
# global_id 목록은 8단계 실행 시 콘솔에 출력되며,
# metadata.json의 "global_id" 필드에도 저장됨.
# 출력 파일: output/sheets/global_id_<번호>.jpg
python scripts/export_global_id_sheet.py 0   # global_id=0 검수
python scripts/export_global_id_sheet.py 1   # global_id=1 검수

# ─────────────────────────────────────────────
# 10단계: Market-1501 포맷 변환
# ─────────────────────────────────────────────
# data/filtered/ 의 트랙렛을 Market-1501 디렉토리 구조로 변환.
# 파일명 규칙: PPPP_CCS_FFFFF.jpg (P=person_id, C=cam_id, S=seq, F=frame)
# config.yaml train_ratio=0.7 기준으로 train / query / gallery 자동 분할.
# 결과: data/market1501/bounding_box_train/, query/, bounding_box_test/
python scripts/format_market1501.py --config configs/config.yaml

# ─────────────────────────────────────────────
# 11단계: Zero-shot 평가
# ─────────────────────────────────────────────
# fine-tuning 없이 pretrained OSNet으로 mAP / Rank-1/5/10 측정.
# query 이미지 1장을 gallery 전체와 비교해 동일인 검색 정확도 평가.
# 결과: data/eval_results/ 에 JSON 및 CMC 곡선 저장.
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
│   ├── reduce_tracklet_frames.py      # 프레임 수 축소 (frame 번호 기준 간격)
│   ├── resize_tracklet_frames.py      # 이미지 리사이즈 (256×128)
│   ├── split_switched_tracklets.py    # 트랙렛 내부 ID switch 분리
│   ├── merge_same_camera_tracklets.py # 동일 cam/slot 내 동일인 재병합
│   ├── fiftyone_curator.py            # FiftyOne 반자동 클리닝
│   ├── thin_tracklet.py               # 특정 트랙렛 프레임 비율 축소 (1/2, 1/3 등)
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
| **1B-2. 전처리** | 프레임 축소(1.25초 간격) + 256×128 리사이즈 | ✅ 완료 |
| **1C. ID switch 정제** | 자동 분리 + 동일인 재병합 + FiftyOne 반자동 클리닝 | 진행 중 |
| **1D. 품질 필터링** | 길이·신뢰도·크기 기준 필터 | 다음 단계 |
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
