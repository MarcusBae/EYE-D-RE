# 개발 노트

---

## 목차

**1. 파이프라인**
- [1.1 추가 영상 확보 시 파이프라인 처리 전략 (2026-05-30)](#11-추가-영상-확보-시-파이프라인-처리-전략-2026-05-30)
- [1.2 데이터셋 준비 파이프라인 — 폴더 중심 흐름도 (2026-06-03)](#12-데이터셋-준비-파이프라인-폴더-중심-흐름도-2026-06-03)
- [1.3 `curated` 생성 규칙 (persons → filtered 역변환) (2026-06-03)](#13-curated-생성-규칙-persons-filtered-역변환-2026-06-03)
- [1.4 파이프라인 단계별 처리 프로세스 상세 (1단계 ~ 6단계)](#14-파이프라인-단계별-처리-프로세스-상세-1단계-6단계)

**2. 데이터 & 설정**
- [2.1 Re-ID 파일 규격 및 Slot(슬롯)의 개념 (2026-05-30)](#21-re-id-파일-규격-및-slot슬롯의-개념-2026-05-30)
- [2.2 Re-ID 글로벌 설정 파라미터 설명 (`configs/config.yaml`) (2026-05-30)](#22-re-id-글로벌-설정-파라미터-설명-configsconfigyaml-2026-05-30)
- [2.3 신규 비디오 파일 이름 매핑 리스트 (Before / After) (2026-05-30)](#23-신규-비디오-파일-이름-매핑-리스트-before-after-2026-05-30)

**3. 알고리즘 & 이론**
- [3.1 HAC 클러스터링 및 동시성 제약 조건 원리 (2026-05-30)](#31-hac-클러스터링-및-동시성-제약-조건-원리-2026-05-30)
- [3.2 Re-ID에서의 L2 정규화 (L2 Normalization) 개념 및 원리 (2026-05-30)](#32-re-id에서의-l2-정규화-l2-normalization-개념-및-원리-2026-05-30)
- [3.3 동일 인물의 다중 카메라 재등장 시 Global ID 병합 가능성 분석 (2026-05-30)](#33-동일-인물의-다중-카메라-재등장-시-global-id-병합-가능성-분석-2026-05-30)

**4. 코드 분석**
- [4.1 파이썬 패키지 내 `__init__.py` 역할과 필요성 (2026-05-30)](#41-파이썬-패키지-내-__init__py-역할과-필요성-2026-05-30)
- [4.2 OSNet 이미지 전처리 파이프라인 (`self.transform`) 분석 (2026-05-30)](#42-osnet-이미지-전처리-파이프라인-selftransform-분석-2026-05-30)
- [4.3 단일 이미지 특징 추출 메서드 (`extract_image_feature`) 흐름 분석 (2026-05-30)](#43-단일-이미지-특징-추출-메서드-extract_image_feature-흐름-분석-2026-05-30)
- [4.4 NumPy에서 형상 표기 `(512,)` 의 의미 분석 (2026-05-30)](#44-numpy에서-형상-표기-512-의-의미-분석-2026-05-30)

**5. 실험 결과**
- [5.1 Zero-shot 평가 결과 — curated / market1501-v1 (2026-06-03)](#51-zero-shot-평가-결과-curated-market1501-v1-2026-06-03)
- [5.2 Re-ID 평가 절차 상세](#52-re-id-평가-절차-상세)
- [5.3 성능 개선 방향 — 선택지 조합 분석](#53-성능-개선-방향--선택지-조합-분석)
- [5.4 배포 설계 — 매칭 임계값 (threshold)](#54-배포-설계--매칭-임계값-threshold)

**6. 트러블슈팅**
- [6.1 CPU 연산 성능 최적화 패치 (2026-05-30)](#61-cpu-연산-성능-최적화-패치-2026-05-30)
- [6.2 Phase 1E~1F 코드 검토 결과 — 발견된 문제점 (2026-05-30)](#62-phase-1e1f-코드-검토-결과-발견된-문제점-2026-05-30)
- [6.3 torchreid pretrained_urls는 ImageNet 가중치만 포함 (2026-06-03)](#63-torchreid-pretrained_urls는-imagenet-가중치만-포함-2026-06-03)

**7. 기타**
- [7.1 Runpod Cmds (2026-05-31)](#71-runpod-cmds-2026-05-31)

**8. 데모**
- [8.1 시연 시나리오 (제품 데모)](#81-시연-시나리오-제품-데모)


---


## 1. 파이프라인

### 1.1 추가 영상 확보 시 파이프라인 처리 전략 (2026-05-30)

#### 1. 개별 영상만 처리 (1~3단계)
- **대상**: 프레임 추출, 객체 추적, 품질 필터링.
- **원리**: 각 비디오 파일은 독립적이므로, 신규 비디오 데이터에 대해서만 트랙렛을 생성 및 정제하여 `data/filtered`에 추가.

#### 2. 전체 데이터 재실행 (4단계 - `merge_ids.py`)
- **대상**: Cross-camera ID 병합.
- **이유**:
  - **클러스터링 일관성**: 전체 트랙렛(기존+신규) 간의 거리 행렬을 재구성해야 신규 인물과 기존 인물의 맵핑이 올바르게 이루어짐.
  - **동시성 제약**: 동일 시간대 및 카메라 겹침 조건의 충돌 대조 필요.
  - **ID 체계 통일**: 글로벌 ID 부여의 연속성과 고유성 보장.

---

### 1.2 데이터셋 준비 파이프라인 — 폴더 중심 흐름도 (2026-06-03)

AVI 원본 영상에서 Market-1501 학습 데이터까지의 전체 폴더 변환 과정.

```
data/raw_videos/
├── cam1_t1.avi  cam2_t1.avi  cam3_t1.avi
└── ... (카메라×슬롯 조합, 총 29개 영상)
        │
        │  [1단계] extract_frames.py
        │  5fps 샘플링, JPG 저장
        ▼
data/frames/cams/{cam_slot}/
├── 000001.jpg
└── ...
        │
        │  [2단계] run_tracking.py  (--frame-stride 6)
        │  YOLOv8n + BoT-SORT, 6프레임마다 추론
        ▼
data/tracklets/{cam_slot}/track_NNNN/
├── frame_XXXXXX.jpg  (person crop)
└── metadata.json     (track_id, camera_id, time_slot, bboxes, confidences ...)
        │
        │  [보조] split_switched_tracklets.py  (필요 시)
        │  트랙렛 내 ID switch 구간을 임베딩 유사도로 감지해 분리
        │  → data/tracklets-1-original (원본 백업)
        │  → data/tracklets-2-manual-edit (수동 편집 병행)
        │
        │  [보조] reduce_tracklet_frames.py  (--stride 5)
        │  프레임 간격이 너무 조밀하면 솎아내기 (metadata 갱신)
        │
        │  [3단계] filter_tracklets.py
        │  품질 필터: min_track_length=6, min_avg_confidence=0.7
        │             min_bbox 64×32, max_aspect_ratio=4.0
        ▼
data/filtered/{cam_slot}/track_NNNN/
├── frame_XXXXXX.jpg
└── metadata.json  (품질 통과 트랙렛만, global_id 아직 없음)
        │
        │  [4단계] merge_ids.py
        │  OSNet 특징 추출 → HAC 클러스터링 (threshold=0.20, cosine, average)
        │  동시성 제약 (same cam+slot, 프레임 겹침) Must-not-link 적용
        ▼
data/filtered/{cam_slot}/track_NNNN/
└── metadata.json  ← global_id 추가됨
        │
        │  [수동 큐레이션 루프]
        │
        │  gather_by_person.py  --tracklets-dir data/filtered
        │  global_id별로 이미지를 person 폴더에 집결 + source_map.json 생성
        ▼
data/persons/person_XXXX/
├── c{cam}_t{slot}_tk{id}_frame_XXXXXX.jpg
└── source_map.json   (새 파일명 → 원본 절대경로 역매핑)
        │
        │  ← 수동 편집: 오병합 이미지 이동·삭제, person_-001로 분리
        │
        │  persons_to_filtered.py  --out data/curated
        │  source_map으로 원본 트랙렛 역추적, global_id 재할당
        ▼
data/curated/{cam_slot}/track_NNNN/
├── frame_XXXXXX.jpg  (수동 검수 반영)
└── metadata.json     (global_id = person 폴더 번호)
        │
        │  [보조] thin_tracklet.py  (개별 트랙렛 이미지 과다 시)
        │  균등 솎아내기 (예: --keep 1/2)
        │
        │  merge_ids.py  (curated 기반 재실행)
        │  수동 검수 반영된 global_id 재클러스터링
        ▼
data/curated/{cam_slot}/track_NNNN/
└── metadata.json  ← 최종 global_id 기록
        │
        │  [5단계] format_market1501.py
        │  global_id → person ID, 70/30 train/test 분할
        │  파일명: {pid:04d}_c{cam}s{slot}_{frame:06d}_00.jpg
        ▼
data/market1501-v1/
├── bounding_box_train/   (전체 ID의 70%)
├── bounding_box_test/    (전체 ID의 30%, gallery)
└── query/                (test ID 중 카메라별 대표 1장씩)
```

#### 현재 버전 현황 (2026-06-03)

| 폴더 | 설명 | 이미지 수 |
| :--- | :--- | ---: |
| `data/tracklets` | 전체 트랙렛 (품질 필터 전) | — |
| `data/filtered` | 품질 필터 통과본 (자동 global_id) | 1,381장 |
| `data/persons` | 수동 검수용 person 뷰 | 1,272장 / 43명 |
| `data/curated` | 수동 검수 반영 최신본 | 1,216장 / 43명 |
| `data/market1501-v1` | 학습 데이터셋 (이전 버전) | — |

---

### 1.3 `curated` 생성 규칙 (persons → filtered 역변환) (2026-06-03)

`data/persons/` 폴더를 수동 편집한 뒤 `scripts/persons_to_filtered.py --out data/curated` 으로 생성.

#### 전체 흐름

```
data/tracklets (+ global_id)
    ↓  gather_by_person.py
data/persons/person_XXXX/   ← 수동 편집 (이미지 이동·삭제)
    ↓  persons_to_filtered.py --out data/curated
data/curated/{cam_slot}/track_NNNN/
```

#### 생성 규칙 요약

| 항목 | 규칙 |
| :--- | :--- |
| **Global ID** | 폴더명 `person_XXXX`의 숫자가 그대로 `global_id`로 사용됨 |
| **미분류 폴더 제외** | `person_-001/` 폴더는 오병합 의심 이미지 보관용으로 자동 무시 |
| **삭제된 이미지 제외** | persons 폴더에서 수동으로 지운 이미지는 curated에 포함하지 않음 |
| **원본 경로 역추적** | `source_map.json` (키: `c{cam}_t{slot}_tk{id}_{orig}` → 값: 원본 절대 경로) 를 이용해 원본 트랙렛 경로를 복원 |
| **출력 구조** | 원본 트랙렛 경로 기준으로 `curated/{cam_slot}/track_NNNN/` 에 복사 |
| **metadata.json 갱신** | `global_id` 삽입, `crop_files`·`num_frames`를 실제 남아있는 파일 기준으로 갱신 |
| **미매핑 이미지** | `source_map.json`에 없는 이미지(수동 추가 파일)는 경고 출력 후 무시 |

#### 현황 (2026-06-03 기준)

- `data/persons/`: 45개 person 폴더, **1,272장**
- `data/curated/`: **1,354장** (persons에 없던 이미지가 일부 포함 — `filtered` 또는 `filtered_` 경유 가능)

---

### 1.4 파이프라인 단계별 처리 프로세스 상세 (1단계 ~ 6단계)

전체 인물 재식별(Person Re-ID) 파이프라인의 각 단계별 구체적인 내부 처리 원리와 입출력 구조는 다음과 같습니다.

#### 1단계: 프레임 추출 (`extract_frames.py`) : 동영상 -> 정지 영상 추출
* **역할**: 원시 비디오 녹화본 파일(`data/raw_videos/*.avi`)로부터 이미지 프레임을 추출합니다.
* **연산 최적화**: 설정파일(`configs/config.yaml`) 내 `frame_extraction.target_fps: 5` 값에 의거하여, 원본 약 25~30fps 동영상을 초당 5프레임(5fps) 수준으로 균등 서브샘플링하여 디스크 용량을 최적화하고 불필요한 중복 연산을 차단합니다.
* **입출력**:
  * **Input**: `data/raw_videos/*.avi`
  * **Output**: `data/frames/cams/[카메라_슬롯명]/*.jpg` (예: `cam1_t1/000001.jpg`)

#### 2단계: 객체 추적 (`run_tracking.py`) : 객체 추적해서 Tracklets 생성 
* **역할**: 추출된 각 이미지 프레임 시퀀스를 스캔하며 YOLOv8n 검출기와 ByteTrack 다중 객체 추적 알고리즘을 사용해 사람을 감지하고 동일인의 이동 동선(Tracklet)으로 바운딩 박스를 묶어 추적합니다.
* **연산 최적화**: 명령 프롬프트 상에서 `--frame-stride 6` 설정을 추가로 부여하여, 매 6번째 프레임만 추적 추론(Inference) 연산을 가동함으로써 배치 처리 속도를 크게 높여 CPU/GPU 연산 지연을 줄입니다.
* **입출력**:
  * **Input**: `data/frames/cams/[카메라_슬롯명]/*.jpg`
  * **Output**: 각 카메라/슬롯별 임시 폴더에 생성되는 추적 상태 메타데이터와 바운딩 박스 크롭 이미지들.


#### 3단계: 품질 필터링 및 썸네일 그리드 시각화 (`filter_tracklets.py`) : trakclets 선별해서 filtered로 이동
* **역할**: 바운딩 박스가 오검출되었거나(정밀도 낮은 물체), 궤적 길이가 너무 짧은 프레임(노이즈성 깜빡임) 등 품질 미달 트랙렛을 엄격히 여과하여 고품질의 순수 인물 크롭 데이터만 필터링합니다.
* **필터 가드 레일**: `min_track_length: 30` (최소 6초 이상 지속되는 인물만 보존), `min_avg_confidence: 0.7` (YOLO의 평균 검출 점수 0.7 이상) 등을 적용하며, `--visualize` 옵션을 통해 각 인물 트랙렛별 그리드 형태의 썸네일 요약 이미지를 보조 생성합니다.
* **입출력**:
  * **Input**: 2단계에서 형성된 추적 트랙렛 임시 데이터
  * **Output**: `data/filtered/[카메라_슬롯명]/track_[트랙ID]/` (내부 구조: `metadata.json`, `*.jpg` 인물 크롭 컷, 시각화 폴더)

#### 4단계: Cross-camera ID 병합 (`merge_ids.py`)
* **역할**: 각 카메라(`cam1~cam3`)와 서로 다른 시간대(`t1~t10`)에서 획득되어 `data/filtered`에 저장된 수많은 독립적인 로컬 트랙렛들의 인물 크롭 이미지를 OSNet 모델에 입력하여 512차원 특징 벡터(Embedding)를 추출한 뒤, HAC(Hierarchical Agglomerative Clustering) 알고리즘으로 군집화하여 고유한 단 하나의 글로벌 ID(Global ID)로 매핑합니다.
* **동시성 제약 조건 (Hard Constraint)**: 동일 시간대 및 동일 카메라 내에서 프레임 구간이 중첩되는 두 트랙렛은 물리적으로 같은 사람일 수 없으므로, 해당 쌍의 코사인 거리를 무조건 무한대(`999.0`)로 강제 마스킹하여 모순된 ID 병합을 차단합니다.
* **입출력**:
  * **Input**: `data/filtered` 하위 트랙렛 내부의 크롭 이미지 파일셋
  * **Output**: 각 트랙렛의 `metadata.json` 파일에 신규 추가 및 저장되는 `"global_id"` 값.

#### 5단계: Market-1501 데이터셋 포맷 변환 (`format_market1501.py`)
* **역할**: 글로벌 ID 매핑이 완료된 `data/filtered` 내부의 인물 크롭 이미지들을 Re-ID 표준 벤치마크 규격인 Market-1501 폴더 포맷으로 재구성하고 파일명을 정형화하여 복사합니다.
* **격리 분할 (Disjoint Split)**: 인물 글로벌 ID셋을 무작위로 뒤섞은 후 `train_ratio: 0.7` 설정에 따라 70%는 훈련용(`bounding_box_train`)으로, 30%는 평가용(`bounding_box_test` 및 `query`)으로 분할합니다. 이때 훈련셋과 평가셋에 동일 인물 ID가 겹쳐서 분할되지 않도록 disjoint 구조를 완벽히 유지합니다.
* **입출력**:
  * **Input**: `data/filtered` 디렉토리 전체 및 각 트랙렛 메타데이터 속 `global_id`
  * **Output**: `data/market1501/` 내 3개 디렉토리에 복사되는 `[ID:04d]_c[카메라ID]s[슬롯ID]_[프레임번호]_00.jpg` 형태의 이미지들.

#### 6단계: Zero-shot 평가 (`evaluate_zeroshot.py`)
* **역할**: 파인튜닝 학습 단계를 거치기 전, 사전 학습된 순수 OSNet Baseline 가중치만을 사용하여 `data/market1501` 데이터셋 상의 인물 이미지 매칭 성능을 모의 측정합니다.
* **평가 지표**: 각 Query 이미지별로 Gallery 이미지들을 비교 정렬하여 동일 인물 이미지가 올바른 순위 내에 포함되는지 확인하여 mAP(Mean Average Precision), Rank-1, Rank-5, Rank-10 정확도를 최종 도출합니다.
* **입출력**:
  * **Input**: `data/market1501/` 데이터 폴더 전체
  * **Output**: `data/eval_results/zeroshot_results.json`에 성능 기록 및 콘솔 CMC 곡선 정확도 로그 출력.

## 2. 데이터 & 설정

### 2.1 Re-ID 파일 규격 및 Slot(슬롯)의 개념 (2026-05-30)

#### 1. Slot (Time Slot)의 정의
- **정의**: 비디오가 촬영된 독립적인 시간대 혹은 세션(Sequence) 구분을 의미합니다.
- **역할**: 동일한 카메라(`camera_id`)에서 촬영된 데이터이더라도 촬영된 시간적 흐름이 다른 경우(예: 오전 촬영 세션 `t1`과 오후 촬영 세션 `t2`), 이를 구분해 주는 고유 넘버입니다.
- **물리적 제약 조건 활용**: 동시성 제약 조건(Must-not-link) 검사 시, 두 트랙렛이 **동일 카메라(`camera_id`)와 동일 시간대 슬롯(`time_slot`)**에 속할 때에만 프레임의 중첩 여부를 검사합니다. 카메라가 같더라도 슬롯이 다르면 동일 인물이 다시 등장한 것일 수 있으므로 동시성 충돌을 일으키지 않습니다.

#### 2. Market-1501 파일명 규격과의 매핑
- **규격**: `[personID:04d]_c[camera_id]s[slot]_[frameID:06d]_00.jpg`
- **매핑**: 
  - `c[camera_id]`: 카메라 ID (예: `c1`, `c2`)
  - `s[slot]`: Sequence 번호에 해당하는 값으로, 본 프로젝트의 `time_slot` 값을 숫자로 매핑해 사용합니다. (예: `s1`, `s2`)

---

### 2.2 Re-ID 글로벌 설정 파라미터 설명 (`configs/config.yaml`) (2026-05-30)

`reid` 설정 그룹은 4단계(`scripts/merge_ids.py`)에서 다중 카메라 트랙렛들을 하나의 고유 글로벌 ID로 병합할 때 적용되는 신경망 모델 및 클러스터링 하이퍼파라미터들로 구성됩니다.

#### 1. 특징 추출기 모델 설정
- **`model_name: "osnet_x1_0"`**:
  - 사람의 외형 특징(임베딩 벡터)을 추출하기 위해 사용하는 OSNet 백본의 크기를 선택합니다.
  - `osnet_x1_0`이 가중치 크기가 크고 정확도가 높은 표준 모델이며, 모바일/엣지향 경량화가 필요할 시 `osnet_x0_75`, `osnet_x0_5` 등으로 대체 가능합니다.
- **`pretrained: true`**:
  - 모델 로딩 시 Re-ID 벤치마크 데이터셋(기본값 MSMT17 등)으로 미리 학습된 사전 가중치(Pre-trained weights)를 불러옵니다. 이 값이 `true`여야 파인튜닝 전에도 유의미한 외형 유사도를 계산할 수 있습니다.
- **`device: "auto"`**:
  - 임베딩 추출 추론 연산을 가속할 디바이스를 설정합니다. (`cuda`, `cpu`, `auto` 지원)
  - `auto` 설정 시 시스템에 사용 가능한 NVIDIA GPU가 존재하면 자동으로 CUDA 가속을 적용하고, 그렇지 않으면 CPU 추론으로 기동합니다.
- **`batch_size: 64`**:
  - 특징 추출 단계에서 한 번에 묶어서 추론을 실행할 크롭 이미지 배치(Batch) 크기입니다. 연산 장치 메모리 상황에 맞게 32~128 사이로 조절합니다.

#### 2. HAC(계층적 병합 군집) 설정
- **`clustering.threshold: 0.5`**:
  - 두 인물 클러스터 간의 **Cosine Distance**를 기준으로 같은 사람으로 판정하여 병합할지 여부를 결정하는 핵심 기준선입니다. (Cosine Distance 범위: 0.0 ~ 2.0)
  - 값이 **높을수록**(예: 0.5 이상) 거리가 멀고 덜 닮은 사람들도 동일인으로 쉽게 병합하여 최종 글로벌 ID 수가 줄어듭니다(과병합 위험).
  - 값이 **낮을수록**(예: 0.2~0.3) 외형 특징이 매우 흡사한 경우에만 엄격하게 합치므로 최종 글로벌 ID 수가 늘어납니다(보수적 결합).
- **`clustering.metric: "cosine"`**:
  - 두 특징 벡터 간의 거리를 측정하는 공식입니다. Re-ID에서는 이미지의 밝기 변화 등에 덜 민감하고 방향성 유사도를 정밀하게 대변하는 코사인 거리(`cosine`)를 주로 사용합니다.
- **`clustering.linkage: "average"`**:
  - 군집과 군집 사이의 통합 거리를 구하는 결합 방식(Linkage Method)입니다.
  - `average`(평균 연결법)는 두 군집의 모든 벡터 쌍 간의 평균 거리를 계산하므로 단일 노이즈 값에 의해 클러스터가 오병합되는 현상을 크게 줄여줍니다.

---

### 2.3 신규 비디오 파일 이름 매핑 리스트 (Before / After) (2026-05-30)

`data/raw_videos` 내 새로 추가된 시간대별 영상 파일들에 대해, 파이프라인 정합성을 맞추기 위해 정의된 카메라 및 시간 슬롯 변환 리스트입니다.

#### 1. 변환 규칙
- **카메라 번호**: `00` -> `cam1`, `02` -> `cam2`, `03` -> `cam3` (기존 카메라인덱스 1, 2, 3과 동기화)
- **시간 슬롯**: 12:00부터 16:30까지 30분 간격으로 순차적 일련번호(`t1` ~ `t10`) 부여

#### 2. 매핑 테이블
| 시간대 | Before (현재 파일명) | After (변경 후 파일명) | 비고 |
| :--- | :--- | :--- | :--- |
| **12:00** | `12000000.avi` | `cam1_t1.avi` | 12:00:00 (Slot 1) - Cam 1 |
| | `12000002.avi` | `cam2_t1.avi` | 12:00:00 (Slot 1) - Cam 2 |
| | `12000003.avi` | `cam3_t1.avi` | 12:00:00 (Slot 1) - Cam 3 |
| **12:30** | `12300002.avi` | `cam2_t2.avi` | 12:30:00 (Slot 2) - Cam 2 (※ `00` 카메라 없음) |
| | `12300003.avi` | `cam3_t2.avi` | 12:30:00 (Slot 2) - Cam 3 |
| **13:00** | `13000000.avi` | `cam1_t3.avi` | 13:00:00 (Slot 3) - Cam 1 |
| | `13000002.avi` | `cam2_t3.avi` | 13:00:00 (Slot 3) - Cam 2 |
| | `13000003.avi` | `cam3_t3.avi` | 13:00:00 (Slot 3) - Cam 3 |
| **13:30** | `13300000.avi` | `cam1_t4.avi` | 13:30:00 (Slot 4) - Cam 1 |
| | `13300002.avi` | `cam2_t4.avi` | 13:30:00 (Slot 4) - Cam 2 |
| | `13300003.avi` | `cam3_t4.avi` | 13:30:00 (Slot 4) - Cam 3 |
| **14:00** | `14000000.avi` | `cam1_t5.avi` | 14:00:00 (Slot 5) - Cam 1 |
| | `14000002.avi` | `cam2_t5.avi` | 14:00:00 (Slot 5) - Cam 2 |
| | `14000003.avi` | `cam3_t5.avi` | 14:00:00 (Slot 5) - Cam 3 |
| **14:30** | `14300000.avi` | `cam1_t6.avi` | 14:30:00 (Slot 6) - Cam 1 |
| | `14300002.avi` | `cam2_t6.avi` | 14:30:00 (Slot 6) - Cam 2 |
| | `14300003.avi` | `cam3_t6.avi` | 14:30:00 (Slot 6) - Cam 3 |
| **15:00** | `15000000.avi` | `cam1_t7.avi` | 15:00:00 (Slot 7) - Cam 1 |
| | `15000002.avi` | `cam2_t7.avi` | 15:00:00 (Slot 7) - Cam 2 |
| | `15000003.avi` | `cam3_t7.avi` | 15:00:00 (Slot 7) - Cam 3 |
| **15:30** | `15300000.avi` | `cam1_t8.avi` | 15:30:00 (Slot 8) - Cam 1 |
| | `15300002.avi` | `cam2_t8.avi` | 15:30:00 (Slot 8) - Cam 2 |
| | `15300003.avi` | `cam3_t8.avi` | 15:30:00 (Slot 8) - Cam 3 |
| **16:00** | `16000000.avi` | `cam1_t9.avi` | 16:00:00 (Slot 9) - Cam 1 |
| | `16000002.avi` | `cam2_t9.avi` | 16:00:00 (Slot 9) - Cam 2 |
| | `16000003.avi` | `cam3_t9.avi` | 16:00:00 (Slot 9) - Cam 3 |
| **16:30** | `16300000.avi` | `cam1_t10.avi` | 16:30:00 (Slot 10) - Cam 1 |
| | `16300002.avi` | `cam2_t10.avi` | 16:30:00 (Slot 10) - Cam 2 |
| | `16300003.avi` | `cam3_t10.avi` | 16:30:00 (Slot 10) - Cam 3 |

```
Slot 1 (12:00): cam1_t1.avi, cam2_t1.avi, cam3_t1.avi
Slot 2 (12:30): cam2_t2.avi, cam3_t2.avi (※ cam1_t2 촬영본은 부재)
Slot 3 (13:00): cam1_t3.avi, cam2_t3.avi, cam3_t3.avi
Slot 4 (13:30): cam1_t4.avi, cam2_t4.avi, cam3_t4.avi
Slot 5 (14:00): cam1_t5.avi, cam2_t5.avi, cam3_t5.avi
Slot 6 (14:30): cam1_t6.avi, cam2_t6.avi, cam3_t6.avi
Slot 7 (15:00): cam1_t7.avi, cam2_t7.avi, cam3_t7.avi
Slot 8 (15:30): cam1_t8.avi, cam2_t8.avi, cam3_t8.avi
Slot 9 (16:00): cam1_t9.avi, cam2_t9.avi, cam3_t9.avi
Slot 10 (16:30): cam1_t10.avi, cam2_t10.avi, cam3_t10.avi
```

---


## 3. 알고리즘 & 이론

### 3.1 HAC 클러스터링 및 동시성 제약 조건 원리 (2026-05-30)

#### 1. 계층적 군집화 (HAC, Hierarchical Agglomerative Clustering)
- **방식**: 상향식(Bottom-up) 클러스터링. 모든 트랙렛을 단일 클러스터로 시작해 임계값(Threshold) 이하의 가장 유사한 두 그룹을 반복 병합.
- **거리 측정**: Cosine Distance (1 - Cosine Similarity) 활용.
- **연결법 (Linkage)**: Average Linkage 적용. 두 클러스터의 모든 멤버 간 평균 거리를 기준으로 병합하여 이상치 노이즈에 강인함.

#### 2. 동시성 제약 조건 (Must-not-link Constraint)
- **물리 제약**: 동일한 인물이 같은 시각에 다른 장소(카메라)에 존재하거나, 동일 비디오 프레임에 동시에 두 명의 개별 인물로 잡힐 수 없음.
- **원리**: 
  1. 두 트랙렛의 카메라 및 시간 슬롯(`cam`, `time_slot`) 일치 여부 판별.
  2. 프레임 구간(Start-End Frame)이 상호 중첩되는지 검사.
  3. 중첩 발생 시 두 트랙렛 간의 거리(Distance)를 무조건 무한대(`999.0`)로 강제 변경.
  4. HAC 병합 기준선(임계값 0.5)을 초과하도록 유도하여 **물리적 모순이 있는 동일 ID 병합을 원천 차단**.

---

### 3.2 Re-ID에서의 L2 정규화 (L2 Normalization) 개념 및 원리 (2026-05-30)

Re-ID(인물 재식별) 및 임베딩 신경망 연산에서 특징 벡터(Feature Vector) 추출 후 적용되는 **L2 정규화(L2 Normalization)**의 수학적 원리와 필요성.

#### 1. L2 정규화의 정의
L2 정규화는 특징 벡터 $\mathbf{x} = [x_1, x_2, \dots, x_d]^T$ 의 **유클리드 노름(Euclidean Norm, L2 Norm)** 크기가 $1$이 되도록 각 원소를 나누어 크기를 맞추는 연산.

$$\mathbf{x}_{\text{norm}} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2} = \frac{\mathbf{x}}{\sqrt{\sum_{i=1}^d x_i^2}}$$

이 변환을 거친 모든 특징 벡터는 다차원 공간 상에서 원점으로부터의 거리가 항상 $1$인 **초구면(Hypersphere)** 표면 위에 고르게 위치한다. 

#### 2. 코사인 유사도(Cosine Similarity)와의 수학적 관계
두 임베딩 특징 벡터 $\mathbf{u}$ 와 $\mathbf{v}$ 사이의 각도 $\theta$를 측정하는 코사인 유사도의 공식은 다음과 같습니다.

$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

* **L2 정규화 완료 시**: $\|\mathbf{u}\|_2 = 1$, $\|\mathbf{v}\|_2 = 1$이 되므로, 분모의 크기 연산이 완전히 생략됩니다.
* **연산 단순화**: 코사인 유사도는 오직 두 벡터의 **단순 내적(Dot Product)**만으로 완벽히 대체됩니다.
  $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v}$$
* **코사인 거리(Cosine Distance)**: Re-ID 병합 거리 행렬 계산에 사용되는 코사인 거리는 아래와 같이 표현됩니다.
  $$\text{Cosine Distance}(\mathbf{u}, \mathbf{v}) = 1 - (\mathbf{u} \cdot \mathbf{v})$$

#### 3. Re-ID 시스템에서 L2 정규화가 필수적인 이유
* **조명 및 환경 변화 강인성 (Scale Invariance)**:
  - 이미지의 밝기, 노출, 명암비, 카메라 렌즈 성능의 차이에 따라 임베딩 벡터의 절대적인 값의 크기(Magnitude)가 왜곡되어 커지거나 작아질 수 있다.
  - L2 정규화는 이러한 **크기 편향을 소거**하고 인물의 의상 색상 및 외형 패턴의 상대적 비율인 **방향성(Direction)** 정보만을 정확히 비교하게 돕습니다.
* **기하학적 일관성 확보**:
  - 만약 정규화를 수행하지 않으면, 픽셀 값의 합이나 특정 활성화 함수 값에 따라 벡터의 노름 크기가 제각각 달라집니다.
  - 이 경우 임계값(Threshold)을 `0.25` 등 일관된 상수로 지정하여 병합 여부를 제어하는 클러스터링(HAC 등) 알고리즘이 올바르게 작동할 수 없습니다.


---

### 3.3 동일 인물의 다중 카메라 재등장 시 Global ID 병합 가능성 분석 (2026-05-30)

**시나리오**: 어떤 사람이 cam1에 잠시 등장 후 cam2에 잠시 등장, 다시 cam1에 잠시 등장한 경우에도 올바르게 동일 global ID로 병합되는가?

#### 결론: 조건부 가능 — 두 가지 장벽이 있음

**시나리오 분해**

```
cam1_t1/track_A  (등장 1: frame 100~160)
cam2_t1/track_B  (등장 2: frame 200~260)
cam1_t1/track_C  (등장 3: frame 400~460)
```

---

#### 장벽 1 — Quality Filter (Phase 1C)

`min_track_length: 30` 기준. `frame_stride=6`으로 트래킹하면 원본 30프레임 = 5개 트랙렛 프레임에 불과.

| 원본 등장 시간 | stride=6 기준 트랙렛 length | 통과 여부 |
|---|---|---|
| ~1초 (24프레임) | 4 | 탈락 |
| ~4초 (96프레임) | 16 | 탈락 |
| **~8초 (192프레임)** | **32** | **통과** |

**"잠시"가 8초 미만이면 Quality Filter에서 전부 탈락한다.** 클러스터링까지 도달하지 못함.

---

#### 장벽 2 — Must-not-link 제약 (Phase 1E)

`track_A`와 `track_C`는 같은 `cam1_t1`이지만 **프레임이 겹치지 않음** (100~160 vs 400~460). 현재 코드는 이 쌍을 차단하지 않음.

```python
# apply_must_not_link_constraints
if max(f1_min, f2_min) <= min(f1_max, f2_max):  # 100~160 vs 400~460 → False
    constrained_matrix[i, j] = 999.0  # 실행 안 됨 → 병합 허용
```

크로스 카메라(`track_A` vs `track_B`)는 카메라가 다르므로 애초에 체크 대상이 아님.
**제약 조건 관점에서는 세 트랙렛 모두 병합 가능.**

---

#### 장벽 3 — threshold 0.25

같은 사람이라도 카메라 간 조명·각도 차이로 cosine distance가 0.25를 초과할 수 있음.

```
실제 Re-ID 논문 기준:
  동일 카메라, 인접 시간  → distance ≈ 0.05~0.15  (쉬움)
  다른 카메라, 조명 변화  → distance ≈ 0.15~0.35  (어려움)
  다른 카메라, 각도 변화  → distance ≈ 0.25~0.50  (매우 어려움)
```

threshold 0.25는 보수적이라 **크로스 카메라 같은 사람도 다른 ID로 분리될 가능성**이 있음.

---

#### 최종 정리

| 조건 | 결과 |
|---|---|
| 각 등장이 8초 이상 | Quality Filter 통과 가능 |
| 프레임 겹침 없는 재등장 (cam1→cam1) | must-not-link 차단 없음, 병합 가능 |
| 크로스 카메라 (cam1↔cam2) | threshold 0.25에서 운에 따라 갈림 |

**핵심 취약점**: threshold를 단일 값으로 고정하면 크로스 카메라 병합은 항상 불안정함. 실제 데이터로 `merge_ids.py` 실행 후 결과를 보고 threshold를 0.25~0.35 사이에서 조정하는 것이 현실적.

---

#### 장벽 1, 2, 3은 모두 통과해야 Global ID 획득

세 장벽을 **순서대로 모두 통과**해야 최종 Global ID가 부여된다.

```
cam1_t1/track_A
cam2_t1/track_B  →  [장벽1] Quality Filter  →  [장벽2] Must-not-link  →  [장벽3] threshold  →  Global ID 부여
cam1_t1/track_C
```

어느 하나라도 탈락하면:

| 장벽 | 탈락 시 결과 |
|---|---|
| **1. Quality Filter** | `data/filtered`에 복사 안 됨 → 이후 단계 진입 불가, global_id 없음 |
| **2. Must-not-link** | 클러스터링은 되지만 동일 ID 병합이 차단됨 → 각자 다른 global_id |
| **3. threshold** | 거리가 기준 초과 → HAC가 다른 클러스터로 분리 → 다른 global_id |

장벽 1에서 탈락한 트랙렛은 `data/filtered`에 아예 없으므로 global_id 자체가 부여되지 않는다. 장벽 2, 3은 클러스터링 단계에서 작동하므로 `data/filtered`까지는 들어오지만 서로 다른 global_id를 받게 된다.

---


## 4. 코드 분석

### 4.1 파이썬 패키지 내 `__init__.py` 역할과 필요성 (2026-05-30)

#### 1. 디렉토리의 패키지화
- **정의**: 해당 디렉토리가 파이썬의 임포트 가능한 패키지(Package)임을 명시.
- **참고**: Python 3.3+ 버전부터는 `__init__.py`가 없어도 암시적 네임스페이스 패키지로 인식되나, 모호성 제거 및 명시적 임포트를 위해 생성이 표준 관례임.

#### 2. 패키지 수준의 진입점 API 정의 (캡슐화)
- **임포트 경로 간소화**: 
  - 외부에서 내부 세부 파일 경로를 알지 못해도 쉽게 임포트하도록 유도.
  - 예: `from src.reid_merger import CrossCameraMerger` 대신 `from src import CrossCameraMerger`로 직접 호출 가능하게 매핑.
- **공개 스코프 제어**: `__all__ = [...]`을 선언하여 패키지 외부로 노출할 클래스/함수를 명시적으로 통제 및 보호.

#### 3. 초기화 코드 실행
- 패키지가 최초 임포트될 때 실행되어야 하는 초기 설정 코드(버전 선언 `__version__ = "0.1.0"`, 의존성 검사 등) 정의 창구.

---

### 4.2 OSNet 이미지 전처리 파이프라인 (`self.transform`) 분석 (2026-05-30)

OSNet 특징 추출기(`OSNetExtractor`) 내부에서 입력 인물 크롭 이미지를 신경망에 입력하기 전 전처리하는 파이프라인의 구성 요소별 의미와 설계 사유.

```python
self.transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

#### 1. `transforms.Compose([...])`
여러 개의 개별 전처리 변환(Transformations)을 하나의 체인(Chain)으로 묶어, 입력 데이터에 순차적으로 적용해주는 컨테이너 역할을 합니다.

#### 2. `transforms.Resize((256, 128))`
* **역할**: 다양한 해상도로 크롭된 인물 바운딩 박스 이미지를 **세로 256픽셀, 가로 128픽셀**로 리사이즈(Resize)합니다.
* **설계 사유**: 
  - 서 있는 사람의 외형 구조(머리, 상체, 하체, 신발 등)는 일반적으로 **2:1 비율의 세로형 직사각형**에 속합니다.
  - OSNet을 비롯한 대부분의 Person Re-ID 모델은 이 최적 비율(256x128)을 표준 입력 스펙(Standard Input Size)으로 가정하고 설계되어 있어, 이미지 내 공간 정보의 종횡비를 보존하며 피처를 조밀하게 추출할 수 있습니다.

#### 3. `transforms.ToTensor()`
* **역할**: `[0, 255]` 범위를 갖는 PIL Image나 `HWC` (Height, Width, Channel) 구조의 numpy array를, `[0.0, 1.0]` 범위의 실수형 및 `CHW` (Channel, Height, Width) 구조의 PyTorch 텐서(Tensor)로 변환합니다.
* **설계 사유**: 
  - PyTorch CNN 신경망 레이어가 연산을 수행하기 위해 반드시 받아야 하는 데이터 포맷 규격입니다.

#### 4. `transforms.Normalize(mean=[...], std=[...])`
* **역할**: ImageNet의 채널별 통계치(평균, 표준편차)를 이용하여 각 픽셀 채널별로 **Z-Score 표준화(Standardization)**를 적용합니다.
  $$x' = \frac{x - \text{mean}}{\text{std}}$$
  - `mean=[0.485, 0.456, 0.406]`: Red, Green, Blue 채널별 평균값
  - `std=[0.229, 0.224, 0.225]`: Red, Green, Blue 채널별 표준편차값
* **설계 사유**:
  - 학습에 사용된 OSNet 사전 학습 가중치(Pre-trained Weights)는 수백만 장의 대규모 데이터셋(ImageNet, MSMT17 등)의 평균과 편차 분포 하에서 수렴하도록 학습되었습니다.
  - 추론 시에도 입력 이미지의 픽셀 분포 데이터를 이와 동일한 통계적 범위로 정렬해주어야만 가중치의 왜곡 없이 본래 의도된 고차원 공간 상의 특징(Feature)을 오차 없이 완벽하게 추출해낼 수 있습니다.


---

### 4.3 단일 이미지 특징 추출 메서드 (`extract_image_feature`) 흐름 분석 (2026-05-30)

`reid_merger.py` 내부 `OSNetExtractor` 클래스에 정의된 `extract_image_feature` 메서드의 라인별 구현 목적과 작동 원리 분석.

```python
@torch.no_grad()
def extract_image_feature(self, bgr_image: np.ndarray) -> np.ndarray:
    """단일 이미지(BGR)에서 L2 정규화된 512차원 특징 벡터 추출."""
    # BGR -> RGB 변환 및 PIL Image 생성
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_image)

    # 텐서 변환 및 배치 차원 추가
    img_t = self.transform(pil_img).unsqueeze(0).to(self.device)

    # 특징 추출 및 L2 정규화
    feat = self.model(img_t)  # shape: [1, 512]
    feat = feat / feat.norm(p=2, dim=1, keepdim=True)
    return feat.cpu().numpy()[0]
```

#### 1. `@torch.no_grad()` 데코레이터
* **설명**: 해당 함수가 실행되는 동안 PyTorch의 자동 미분 엔진(Autograd)을 비활성화합니다.
* **목적**: 학습이 아닌 단순 **추론(Inference) 단계**이므로 역전파(Backpropagation)에 필요한 역방향 그래프 계산 메모리를 생성하지 않아 연산 속도가 빨라지고 GPU/RAM 메모리 사용량을 크게 절약합니다.

#### 2. BGR ➡️ RGB 색상 공간 변환 및 PIL 래핑
* **코드**: `cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)`, `Image.fromarray(...)`
* **목적**: 
  - OpenCV는 이미지를 읽어 들일 때 기본 채널 순서로 **BGR** 방식을 사용합니다.
  - 하지만 PyTorch 및 OSNet 모델은 **RGB** 순서로 학습되었으므로 이를 일치시키기 위해 채널을 전환한 뒤, torchvision 전처리 파이프라인(`self.transform`)의 입력 요건인 PIL Image 객체 형태로 감싸줍니다.

#### 3. 텐서 변환 및 배치 차원 확장 (Unsqueeze)
* **코드**: `self.transform(pil_img).unsqueeze(0).to(self.device)`
* **목적**:
  - `self.transform`을 적용해 크기 조절 및 표준화를 거치면 이미지 텐서의 형태(Shape)는 `[C, H, W]` (3차원)가 됩니다.
  - PyTorch의 CNN 레이어는 무조건 배치(Batch) 형태로 입력 데이터를 입력받아야 하므로, 0번째 축에 가상의 배치 차원을 삽입하는 `unsqueeze(0)` 연산을 취해 `[1, C, H, W]` (4차원)로 규격을 확장합니다.
  - 최종적으로 이 텐서를 모델이 탑재된 연산 가속 장치(GPU인 CUDA 또는 CPU) 메모리로 업로드합니다.

#### 4. 모델 전방 추론 및 L2 정규화
* **코드**: `feat = self.model(img_t)`, `feat / feat.norm(p=2, dim=1, ...)`
* **목적**:
  - 준비된 배치 텐서를 OSNet 모델에 전달하여 512차원 특징 벡터 $1\times512$ 텐서를 획득합니다.
  - 벡터의 L2 Euclidean Norm 값을 구해 스스로를 나누어 줌으로써 벡터 길이를 1로 고정하는 **L2 정규화**를 실시간 연산하여 코사인 거리 비교의 기하학적 정합성을 갖추도록 변환합니다.

#### 5. 넘파이(NumPy) 반환
* **코드**: `feat.cpu().numpy()[0]`
* **목적**:
  - 연산 장치(GPU) 상의 텐서를 CPU 메모리로 다운로드한 다음, 파이썬 환경의 넘파이 배열(`ndarray`) 형식으로 컴파일합니다.
  - 1개 배치 결과물만 존재하므로 첫 번째 원소(`[0]`)를 추출하여 최종 `(512,)` 차원의 1D 어레이로 정제해 반환합니다.


---

### 4.4 NumPy에서 형상 표기 `(512,)` 의 의미 분석 (2026-05-30)

NumPy(넘파이) 다차원 배열(`ndarray`) 및 PyTorch 연산 완료 후 사용되는 차원(Shape) 표현식 중 **`(512,)`**의 언어적 정의와 튜플 표기 방식에 대한 설명.

#### 1. 쉼표(`,`)가 붙는 문법적 이유
파이썬 언어 명세상, 괄호로 단순히 감싼 `(512)`는 튜플이 아닌 **연산식의 우선순위 괄호**로 해석되어 단순 정수 `512`와 같아집니다.
따라서, 요소가 1개뿐인 **튜플(Tuple) 자료형**임을 컴파일러에 명시하기 위해 끝에 쉼표를 찍어 **`(512,)`**로 표기하도록 약속되어 있습니다.

#### 2. `[1, 512]`(2차원) vs `(512,)`(1차원) 비교
* **`[1, 512]` 또는 `(1, 512)` (2차원 행렬, Matrix)**:
  * 형태: `[[x1, x2, ..., x512]]` (이중 리스트 구조)
  * 차원: 1개의 행(Row)과 512개의 열(Column)로 구성된 2D 평면 공간을 나타냅니다.
* **`(512,)` (1차원 벡터, Vector)**:
  * 형태: `[x1, x2, ..., x512]` (단일 리스트 구조)
  * 차원: 중첩이 없는 1D 선형 방향으로 나열된 단일 축(Axis 0) 데이터입니다.

#### 3. 임베딩 벡터에서의 차원 축소 흐름
인공신경망은 한 번에 다량의 입력을 받아 병렬 처리하도록 고안되었기 때문에, 단일 이미지 1장만 추론하더라도 항상 배치 차원이 존재하여 출력 결과는 2차원(`shape: [1, 512]`) 텐서가 됩니다. 

```python
# 1. 모델이 예측한 원본 결과 (Batch가 포함된 2D Matrix)
feat = self.model(img_t)       # PyTorch Tensor: shape [1, 512]
feat_np = feat.cpu().numpy()   # NumPy Array: shape (1, 512)

# 2. [0] 인덱스를 취해 Batch 축 제거 (1D Vector로 변환)
final_feat = feat_np[0]        # NumPy Array: shape (512,)
```

배치 차원(0번째 축)의 첫 번째 데이터만 꺼내옴으로써 바깥쪽 리스트 구조를 걷어내고, 거리 비교나 다른 연산에 사용하기 쉽도록 **`(512,)` 차원의 1D 벡터**로 정제하여 최종 반환하는 구조입니다.




## 5. 실험 결과

### 5.1 Zero-shot 평가 결과 — curated / market1501-v1 (2026-06-03)

#### 실험 조건

**데이터**

| 항목 | 값 |
| :--- | :--- |
| 데이터셋 | `data/market1501-v1` |
| 기반 | `data/curated` (수동 검수, 43명 / 1,216장) |
| Global ID 출처 | persons 폴더 번호 (수동 검수 기반, 1~43) |
| train / test 분할 | 70% / 30% (disjoint, seed=42) |

**모델 스택**

| 역할 | 모델 | 가중치 | 비고 |
| :--- | :--- | :--- | :--- |
| 객체 검출 | YOLOv8n | COCO pretrained | `yolov8n.pt` |
| 단일 카메라 추적 | BoT-SORT | YOLO 백본 재사용 (`auto`) | `configs/botsort.yaml` |
| Cross-cam 특징 추출 (`merge_ids.py`) | OSNet x1.0 | ImageNet pretrained | `osnet_x1_0_imagenet.pth` — Market-1501 가중치 미사용 (`weights_path: null`) |
| Zero-shot 평가 (`evaluate_zeroshot.py`) | OSNet x1.0 | 실험별 상이 | 아래 결과 비교 참조 |

**HAC 클러스터링 파라미터**

| 파라미터 | 값 |
| :--- | :--- |
| threshold | 0.20 (cosine distance) |
| linkage | average |
| must-not-link | 동일 cam+slot 프레임 겹침 쌍 차단 |

#### 평가 결과 비교

**[이전] 잘못된 query/gallery 구성** (같은 카메라 내 비교, 무효):

| 지표 | ImageNet pretrained | Market-1501 pretrained |
| :--- | ---: | ---: |
| Rank-1 | 72.73% | 54.55% |
| Rank-5 | 100.0% | 63.64% |
| Rank-10 | 100.0% | 90.91% |
| mAP | 52.23% | 51.12% |
| Query 수 | 25장 (19 ID) | 25장 (19 ID) |
| Gallery 수 | 151장 | 151장 |
| 평가 방식 | ❌ 동일 카메라 내 query/gallery 분리 | ❌ 동일 |

**[현재] 수정된 cross-camera 평가** (2026-06-03, `market_formatter.py` 프로토콜 수정):

| 지표 | ImageNet pretrained | Market-1501 pretrained |
| :--- | ---: | ---: |
| **Rank-1** | 54.55% | **81.82%** |
| Rank-5 | 63.64% | **100.0%** |
| Rank-10 | 81.82% | **100.0%** |
| **mAP** | 49.54% | **55.03%** |
| Query 수 | 11장 (5 ID) | 11장 (5 ID) |
| Gallery 수 | 176장 | 176장 |
| 평가 방식 | ✅ cross-camera query/gallery 분리 | ✅ 동일 |

#### 핵심 해석

- **이전 결과가 왜 무효였는가**:
  - query와 gallery가 동일 카메라에서 나와 배경·조명·각도가 거의 동일 → trivially easy
  - Rank-5 = 100%는 평가가 쉬운 문제였다는 신호였음

- **수정 후 Market-1501이 ImageNet보다 높아진 이유**:
  - Cross-camera 비교로 평가 난이도가 올라가자 Re-ID 전용 학습(Market-1501)의 강점이 드러남
  - ImageNet은 범용 특징이라 cross-camera 외형 변화에 덜 강인함
  - **이전에 ImageNet이 더 좋게 보인 것은 평가 방식의 오류**였음

- **Query 수 감소 (25장 → 11장, 19 ID → 5 ID)**:
  - 수정된 프로토콜은 2개 이상 카메라에 등장한 인물만 query 선발
  - test ID 중 14개는 단일 카메라만 등장 → gallery에만 포함
  - 데이터셋 특성상 cross-camera 등장 인물이 적음을 반영

- **통계적 신뢰도 여전히 낮음**: query 11장 / 5 ID — 수치 변동 폭이 큼

- **이전 베이스라인 (HAC 재클러스터링, 무효)**: Rank-1 81.82% / mAP 55.03% — Query 5 ID / Gallery 176장
- **현재 베이스라인 (persons 수동 검수 기반)**: **Rank-1 51.61% / mAP 26.23%** — Query 13 ID / Gallery 430장

#### 다음 방향

파인튜닝 가치 있음. 단, 학습 데이터(~44 ID / ~1,000장)가 작으므로 아래 전략 권장:

| 항목 | 권장 |
| :--- | :--- |
| 학습 방식 | feature extractor 동결 후 마지막 레이어만 fine-tune |
| Loss | Triplet Loss 또는 ArcFace 계열 |
| Augmentation | Random Flip, Color Jitter, Random Erasing |
| 대안 | 데이터 추가 수집 후 재클러스터링 → 데이터 규모 확보 우선 |

---


### 5.2 Re-ID 평가 절차 상세

#### 전체 흐름

```
market1501-v1/
├── query/            ← 각 ID의 대표 이미지 (카메라별 1장)
├── bounding_box_test/  ← gallery (test ID 전체 이미지)
└── bounding_box_train/ ← 평가에 미사용 (학습용)

        │
        │  1. 특징 추출 (OSNet)
        ▼
query_feats   : (Nq, 512)  — query 이미지별 L2 정규화 임베딩
gallery_feats : (Ng, 512)  — gallery 이미지별 L2 정규화 임베딩

        │
        │  2. 거리 행렬 계산
        ▼
distmat : (Nq, Ng)  — cosine distance = 1 − (q_feat · g_feat)

        │
        │  3. 각 query에 대해 gallery 거리 순 정렬
        ▼
ranked_gallery : [가장 유사한 순서로 정렬된 gallery 이미지 목록]

        │
        │  4. Good / Junk 분류 후 mAP·CMC 계산
        ▼
결과 : mAP, Rank-1/5/10
```

#### Good / Junk 정의 (Market-1501 표준 프로토콜)

| 분류 | 조건 | 처리 |
| :--- | :--- | :--- |
| **Good** | 동일 PID + **다른** 카메라 | 정답으로 인정 |
| **Junk** | 동일 PID + **같은** 카메라 | 랭킹에서 제외 (AP 계산 대상 아님) |
| **Distractor** | 다른 PID | 오답 |

> Junk를 제외하는 이유: 같은 카메라에서 찍힌 동일인은 거의 동일한 이미지라 trivially easy — 평가 난이도를 cross-camera 매칭으로 한정하기 위함.

#### Rank-k (CMC) 계산

```
각 query에 대해:
  → Junk 제거 후 gallery 거리 순 정렬
  → 상위 k개 안에 Good 이미지가 1장이라도 있으면 → 정답
  → Rank-1: 1등이 Good인 비율
  → Rank-5: 상위 5개 안에 Good이 있는 비율
```

현재 결과 (query 25장 / gallery 151장):
- Rank-1 = 72.73% → 25장 중 **18장**이 1등 매칭 성공
- Rank-5 = 100.0% → 전체 query가 상위 5개 안에 정답 포함

#### mAP (Mean Average Precision) 계산

각 query에 대해 AP(Average Precision) 를 계산한 뒤 평균:

```
AP = (1/R) × Σ [ precision@k × relevant(k) ]
   = 정답이 나타난 위치별 precision의 평균

예) Good 이미지가 gallery 순위 2, 5번째에 있는 경우:
    AP = (1/2) × (1/2 + 2/5) = (1/2) × 0.9 = 0.45
```

- Rank-1이 높아도 mAP가 낮은 경우 → 1등은 맞추지만 나머지 Good 이미지가 뒤쪽에 분산됨
- 현재 Rank-1(72.7%) vs mAP(52.2%) 갭이 큰 이유: 동일인이 여러 트랙렛(여러 카메라·시간대)에 걸쳐 있어 일부만 상위 랭크에 오름

#### 현재 평가 구성의 한계

| 항목 | 현재 | 이상적 |
| :--- | :--- | :--- |
| Query 수 | 25장 (19 ID) | ID당 카메라별 1장 |
| Gallery 수 | 151장 | 수천 장 이상 |
| Query 선택 방식 | `format_market1501.py` 자동 배분 | 카메라별 엄격 분리 |
| 통계적 신뢰도 | 낮음 (샘플 적음) | ID 100개+ 권장 |

---

### 5.3 성능 개선 방향 — 선택지 조합 분석

현재 베이스라인: **Rank-1 72.73% / mAP 52.23%** (OSNet x1.0, ImageNet pretrained, zero-shot)

#### 축별 선택지 및 장단점

**데이터**

| 선택지 | 장점 | 단점 | 예상 효과 |
| :--- | :--- | :--- | :--- |
| 현재 유지 (43 ID / 1,216장) | 추가 비용 없음 | ID 수 부족으로 파인튜닝 효과 제한 | — |
| **추가 영상 확보** | 파인튜닝 효과 극대화, ID 수 증가 → 일반화 향상 | 촬영·라벨링 비용 | Rank-1 +10~20%p 가능 |

**모델**

| 선택지 | 장점 | 단점 | 예상 효과 |
| :--- | :--- | :--- | :--- |
| osnet_x0_25 (경량) | 추론 속도 빠름, 엣지 배포 적합 | 특징 품질 낮음, 소규모 데이터엔 과소적합 위험 | 현재 대비 Rank-1 -10~15%p |
| **osnet_x1_0 (현재)** | 균형잡힌 성능·속도, 검증된 Re-ID 백본 | — | 베이스라인 |
| TransReID (ViT 기반) | 최고 수준 Re-ID 성능, 글로벌 어텐션 구조 | 연산량 많음, 파인튜닝 데이터 더 필요, 구현 복잡 | 충분한 데이터 시 Rank-1 +10~15%p |

**학습 방법**

| 선택지 | 장점 | 단점 | 예상 효과 |
| :--- | :--- | :--- | :--- |
| **범용 사전학습 (현재)** | 학습 불필요, 즉시 사용 가능 | 실환경 도메인 불일치 | 베이스라인 |
| Re-ranking (후처리) | 학습 없이 mAP 향상, 구현 간단 | 갤러리 규모 커질수록 연산 증가 (O(n²)) | mAP +5~15%p, Rank-1 소폭 향상 |
| 도메인 파인튜닝 | 실환경 특성 반영, Rank-1 개선 기대 | 소규모 데이터(43 ID) → 과적합 위험, 신중한 하이퍼파라미터 필요 | Rank-1 +5~15%p (데이터 규모 의존) |

**Matching 전략**

| 선택지 | 장점 | 단점 | 예상 효과 |
| :--- | :--- | :--- | :--- |
| Single Detection (현재) | 구현 단순, 빠름 | 단일 프레임 노이즈에 취약 | 베이스라인 |
| **Mean Pooling** | 트랙렛 내 다중 프레임 평균 → 안정적인 특징 | 저품질 프레임이 평균을 오염 | Rank-1 +3~7%p |
| Confidence-weighted Pooling | 검출 신뢰도 높은 프레임에 가중치 → 노이즈 최소화 | 신뢰도 점수 활용 구조 필요 | Rank-1 +5~10%p |

**임계값 전략**

| 선택지 | 장점 | 단점 | 예상 효과 |
| :--- | :--- | :--- | :--- |
| 0.85 고정 (현재) | 오매칭 최소화 | 미매칭(False Negative) 많음 | 베이스라인 |
| 0.73 고정 (권장) | 매칭 커버리지 확대 | 오매칭 소폭 증가 | Recall 향상 |
| 영상별 적응형 | 카메라·조명별 최적 임계값 자동 적용 | 보정 데이터 필요, 구현 복잡 | 가장 균형잡힌 Precision-Recall |

---

#### 권장 조합 시나리오

| # | 모델 | 가중치 | 후처리 | Matching | Rank-1 | Rank-5 | Rank-10 | mAP | Query | Gallery | 상태 |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **Baseline** | OSNet x1.0 | Market-1501 | — | Single | **51.61%** | **67.74%** | **77.42%** | **26.23%** | 31 / 13 ID | 430장 | done |
| **ref-old** | OSNet x1.0 | Market-1501 | — | Single | ~~81.82%~~ | ~~100.0%~~ | ~~100.0%~~ | ~~55.03%~~ | 11 / 5 ID | 176장 | done (HAC오병합, 무효) |
| **①** | OSNet x1.0 | Market-1501 | — | Mean Pooling | 38.71% | 64.52% | 87.10% | 43.22% | 31 / 13 ID | 49트랙렛 | done |
| **②** | OSNet x1.0 | Market-1501 | — | Conf-weighted | 38.71% | 61.29% | 87.10% | 42.98% | 31 / 13 ID | 49트랙렛 | done |
| **③** | OSNet x1.0 | Market-1501 | — | TTA (좌우 반전) | 51.61% | 67.74% | 74.19% | 27.13% | 31 / 13 ID | 430장 | done (역효과) |
| **④** | OSNet x1.0 | Market-1501 | Re-ranking | Single | 41.94% | 61.29% | 64.52% | 36.01% | 31 / 13 ID | 430장 | done |
| **⑤** | OSNet x1.0 | Market-1501 | Re-ranking | Mean Pooling | 45.16% | 74.19% | 96.77% | **54.56%** | 31 / 13 ID | 49트랙렛 | done |
| **⑥** | OSNet x1.0 | Market-1501 | Re-ranking | Conf-weighted | 41.94% | 70.97% | 93.55% | 49.61% | 31 / 13 ID | 49트랙렛 | done |
| **⑦** | OSNet x1.0 | Market-1501 | Re-ranking | TTA + Mean Pooling | — | — | — | — | — | — | — |
| **⑧-a** | OSNet-AIN x1.0 | ImageNet | — | Single | 38.71% | 61.29% | 83.87% | 27.05% | 31 / 13 ID | 430장 | done |
| **⑧-b** | OSNet-AIN x1.0 | ImageNet | — | Mean Pooling | 45.16% | 77.42% | 96.77% | 49.29% | 31 / 13 ID | 49트랙렛 | done |
| **⑧-c** | OSNet-AIN x1.0 | ImageNet | Re-ranking | Mean Pooling | 48.39% | 74.19% | 87.10% | 52.09% | 31 / 13 ID | 49트랙렛 | done |
| **⑧-d** | OSNet-AIN x1.0 | Market-1501 | — | Single | — | — | — | — | — | — | - |
| **⑧-e** | OSNet-AIN x1.0 | Market-1501 | — | Mean Pooling | — | — | — | — | — | — | - |
| **⑧-f** | OSNet-AIN x1.0 | Market-1501 | Re-ranking | Mean Pooling | — | — | — | — | — | — | - |
| **⑨** | OSNet x1.0 | 파인튜닝 (마지막 블록) | Re-ranking | Mean Pooling | — | — | — | — | — | — | — |
| **⑩** | OSNet x1.0 | 파인튜닝 (전체) | Re-ranking | Mean Pooling | — | — | — | — | — | — | — |
| **⑪** | OSNet-AIN x1.0 | 파인튜닝 | Re-ranking | Conf-weighted | — | — | — | — | — | — | — |
| **⑫** | OSNet x1.0 | 파인튜닝 | Re-ranking | Conf-weighted | — | — | — | — | — | — | — |
| **⑬** | TransReID | 파인튜닝 | Re-ranking | Conf-weighted | — | — | — | — | — | — | — |

**결과 해석 (2026-06-03, persons 수동 검수 기반 재실험)**

| # | Matching | Rank-1 변화 | mAP 변화 | Gallery | 판정 |
| :--- | :--- | ---: | ---: | :--- | :--- |
| ① | Mean Pooling | -12.90%p | +16.99%p | 430장 → 49트랙렛 | 조건부 유효 |
| ② | Conf-weighted | -12.90%p | +16.75%p | 430장 → 49트랙렛 | 조건부 유효 |
| ③ | TTA | +0.00%p | +0.90%p | 430장 유지 | 미미, 제외 |
| ④ | Re-ranking | -9.67%p | +9.78%p | 430장 유지 | 부분 유효 |
| **⑤** | **Re-ranking + Mean** | **-6.45%p** | **+28.33%p** | **49트랙렛** | **최우수** |
| ⑥ | Re-ranking + Conf-weighted | -9.67%p | +23.38%p | 49트랙렛 | 유효 |

- **①②**: Gallery가 이미지→트랙렛 단위로 집계되면서 Rank-1은 소폭 하락하지만 mAP 크게 개선. 트랙렛 평균 특징이 전체 순위 품질을 높임. Gallery 크기 차이로 직접 비교는 어려움.
- **③**: TTA 효과 미미 (mAP +0.9%p). 역효과는 없지만 연산 비용 대비 이득 없음 → 제외.
- **④**: Re-ranking 단독 시 mAP +9.78%p 개선, Rank-1 소폭 하락. Gallery 430장에서 k-reciprocal 계산 안정적.
- **⑤**: **현재 최우수 조합** — Re-ranking + Mean Pooling. mAP 54.56%로 Baseline(26.23%) 대비 +28.33%p. 트랙렛 단위 안정적 특징 + Re-ranking 순위 보정의 시너지.
- **⑤ > ⑥**: Mean이 Conf-weighted보다 일관되게 우수. 균등 평균이 신뢰도 가중 평균보다 안정적.
- **⑧ OSNet-AIN (ImageNet)**: Market-1501 가중치 미제공, ImageNet pretrained만 사용. Single 기준 OSNet x1.0 ImageNet(Rank-1 54.55%→38.71%)보다 낮음 — AIN 구조가 ImageNet 도메인 특화, Re-ID 도메인에서 Market-1501 pretrained 없이는 불리. Re-ranking + Mean(⑧-c, mAP 52.09%)은 ⑤(54.56%)에 근접하지만 미달. **OSNet-AIN의 장점은 Market-1501 pretrained 가중치 사용 시 발휘될 가능성** — 별도 다운로드 후 재실험 필요.

**그룹 요약**

| 그룹 | 시나리오 | 설명 |
| :--- | :--- | :--- |
| **A. 추론 방식** | ①②③ | 학습 없음, Matching 전략만 변경 — ✅ 완료 |
| **B. 후처리** | ④ | Re-ranking 단독 적용 — done |
| **C. 추론+후처리** | ⑤⑥ | Re-ranking + Mean/Conf-weighted 조합 — done |
| **D. 모델 교체** | ⑧-a~f | OSNet-AIN ImageNet(done) / Market-1501 가중치(미실험) |
| **E. 파인튜닝** | ⑨⑩⑪ | 현재 데이터로 학습 |
| **F. 데이터 확보** | ⑫⑬ | 추가 촬영 필요 |

> 예상 수치는 유사 규모 Re-ID 연구 기준 추정값, query 5 ID 기준으로 실제 편차 클 수 있음
> **권장 순서**: D → E 순으로 진행 (A, B, C 완료)

---

### 5.4 배포 설계 — 매칭 임계값 (threshold)

> 아래 항목은 모델 성능이 확정된 후, **실서비스 배포 직전** 단계에서 결정합니다.
> 평가(mAP/Rank-k)는 threshold 없이 전체 순위로 판단하므로 평가 단계에서는 불필요.

**임계값의 역할**

```
distance < threshold  →  "같은 사람" (매칭 성공)
distance ≥ threshold  →  "다른 사람" (미매칭)
```

**전략별 비교**

| 전략 | 값 | 특성 | 적합 상황 |
| :--- | :--- | :--- | :--- |
| 보수적 고정 | 0.85 | 오탐(FP) 최소화, 누락(FN) 증가 | 오인식 비용이 큰 경우 |
| 권장 고정 | 0.73 | 균형점, 일반적 권장 | 범용 운영 |
| 영상별 적응형 | 자동 보정 | 카메라·조명 환경별 최적화 | 다중 카메라 환경 |

**보정 방법**

모델 확정 후 validation 세트로 precision-recall 곡선을 그려 운영 목적에 맞는 임계값 선택:
```python
for threshold in np.arange(0.0, 2.0, 0.05):
    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    # F1-score 최대 지점 또는 목표 precision 달성 지점 선택
```

---

## 6. 트러블슈팅

### 6.1 CPU 연산 성능 최적화 패치 (2026-05-30)

#### 1. 문제 상황
- **현상**: `merge_ids.py` 가동 시 Re-ID 특징 추출 단계에서 극심한 연산 지연 (536개 트랙렛 기준 약 4시간 40분 예상).
- **원인**: 트랙렛당 수십~수백 장에 달하는 크롭 이미지 전체를 전수 추론(Inference)하여 CPU 병목 유발.

#### 2. 해결 방안 (균등 샘플링 - Uniform Sampling)
- **방식**: 트랙렛 타임라인을 고르게 대변할 수 있도록 최대 8장만 균등 선택하여 대표 피처 평균화.
- **구현 (`src/reid_merger.py`)**: `max_frames_per_tracklet = 8` 기준 `np.linspace` 기반 샘플링 인덱스 추출.
  ```python
  if len(crop_files) > max_frames_per_tracklet:
      indices = np.linspace(0, len(crop_files) - 1, max_frames_per_tracklet, dtype=int)
      crop_files = [crop_files[idx] for idx in indices]
  ```

#### 3. 개선 결과
- **연산량**: OSNet 연산 횟수 **80% 이상 절감** (21,440회 → 4,288회 이하).
- **소요 시간**: 전체 병합 수행 시간 수 시간에서 **1~2분 내외**로 극적 단축.
- **신뢰성**: 인물 프레임의 전/중/후반부 외형을 모두 포괄하여 Re-ID 특징 고유 품질 유지.

---

### 6.2 Phase 1E~1F 코드 검토 결과 — 발견된 문제점 (2026-05-30)

#### [P0] `market_formatter.py:103` — `camera` 키 오류 (버그, 치명적)

- **현상**: Market-1501 변환 후 생성된 모든 파일명의 카메라 번호가 `c0`으로 고정됨.
- **원인**: `t.get("camera", "c0")`으로 읽지만, `tracker.py`가 메타데이터에 저장하는 실제 키는 `camera_id`(int). `"camera"` 키가 없으므로 항상 기본값 `"c0"` → 숫자 `0`으로 폴백.
- **영향**: Re-ID의 핵심인 cross-camera 구분이 완전히 손상됨. 현재 생성된 `data/market1501` 전체가 무효.
- **수정**:
  ```python
  # 수정 전
  cam = self._extract_number(t.get("camera", "c0"))
  # 수정 후
  cam = self._extract_number(t.get("camera_id", 0))
  ```

---

#### [P1] `configs/config.yaml:66` — HAC threshold 0.5가 너무 높음 (설계 문제)

- **현상**: 536개 트랙렛이 고작 20개 global ID로 병합됨 (평균 26.8개 트랙렛/ID). 모든 global ID에 동일 camera+slot에서 여러 개(최대 13개)의 트랙렛이 포함됨.
- **원인**: cosine distance 기준 `threshold: 0.5`는 두 벡터의 각도 차이가 약 60도 이내면 같은 사람으로 판정. 외형이 상당히 다른 사람들도 병합되어 레이블 노이즈 폭발.
- **영향**: Re-ID 모델 학습 시 서로 다른 사람을 같은 ID로 제시 → 학습 붕괴 위험.
- **권장 조치**: `threshold: 0.2~0.3`으로 낮추고, 결과 global ID 수가 촬영 공간의 예상 등장 인원(50~150명 수준)에 근접하는지 확인 후 재실행.

---

#### [P1] `market_formatter.py:132` — Query 선택 방식이 Re-ID 평가 표준과 불일치 (설계 문제)

- **현상**: test tracklet의 첫 번째 이미지(index 0)를 query, 나머지를 gallery로 분배. 현재 query/gallery가 동일 tracklet, 동일 카메라, 동일 위치에서 촬영됨.
- **원인**: Market-1501 표준은 query는 특정 카메라, gallery는 다른 카메라에서 선택해야 cross-camera 검색 성능을 평가할 수 있음. 같은 tracklet에서 나눈 query/gallery는 외형이 거의 동일하여 평가가 trivially easy해짐.
- **추가 이상**: query에 동일 PID 이미지가 최대 35장(PID 0009) 포함 — 진짜 Market-1501은 query를 person×camera당 1장으로 제한.
- **권장 조치**: test ID의 트랙렛들을 카메라별로 분리하여, 한 카메라 → query, 나머지 카메라 → gallery로 배분하는 방식으로 재설계.

---

#### [P2] `reid_merger.py:187` — 이미지 로드 실패 시 zero-vector 할당 (잠재적 오염)

- **현상**: crop 이미지를 하나도 읽지 못한 트랙렛에 `np.zeros(512)`를 특징 벡터로 할당.
- **원인**: zero vector는 L2 norm = 0이므로 정규화 불가. cosine 거리 계산 시 다른 모든 벡터와의 거리가 `1.0`으로 동일하게 계산되어 의미 없는 클러스터 연결을 유발.
- **권장 조치**: 이미지 로드 실패 트랙렛은 특징 추출 목록에서 제외하고, 클러스터링 후 별도 처리(전 단계 tracklet_dir 제거 또는 global_id = -1 마킹).

---

#### [P2] `reid_merger.py:251` — Must-not-link를 999.0으로 처리하는 방식의 신뢰성 한계 (설계 문제)

- **현상**: 동일 cam+slot에서 여러 트랙렛이 같은 global ID에 포함되는 사례 다수 발생 (비-겹침 시간대이더라도 클러스터 과병합 증거).
- **원인**: average linkage HAC에서 클러스터 간 거리는 멤버 쌍 거리의 **평균**으로 계산됨. 제약 쌍의 거리를 999.0으로 올려도, 해당 클러스터 내 다른 낮은 거리 쌍들의 평균에 희석되면 999.0의 억제 효과가 약해질 수 있음.
- **권장 조치**: threshold 조정(P1 수정)이 우선이며, 장기적으로는 scipy HAC 대신 must-not-link를 hard constraint로 보장하는 COP-KMeans 또는 직접 구현한 constrained HAC로 전환 고려.

---


### 6.3 torchreid `pretrained_urls`는 ImageNet 가중치만 포함 (2026-06-03)

#### 현상

`configs/config.yaml`의 `reid.weights_path: null` 상태에서 `merge_ids.py` 및 `evaluate_zeroshot.py` 실행 시 torchreid가 자동으로 내려받는 가중치가 **ImageNet pretrained 가중치**임을 뒤늦게 확인.

#### 원인

torchreid 라이브러리 내부 `pretrained_urls` 딕셔너리에는 OSNet 계열 전 모델에 대해 **ImageNet pretrained 가중치 URL만** 등록되어 있음:

```python
# torchreid/reid/models/osnet.py
pretrained_urls = {
    'osnet_x1_0':     'https://drive.google.com/uc?id=1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY',  # ImageNet
    'osnet_x0_75':    'https://drive.google.com/uc?id=1uwA9fElHOk3ZogwbeY5GkLI6QPTX70Hq',  # ImageNet
    ...
}
```

Market-1501 등 Re-ID 벤치마크로 학습된 가중치는 **[torchreid Model Zoo 페이지](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO)에서 별도 수동 다운로드** 필요.

#### 구분 방법

```python
import torch
state = torch.load('가중치파일.pth', map_location='cpu')
sd = state.get('state_dict', state.get('model', state))
print(sd['classifier.weight'].shape)
# [1000, 512] → ImageNet pretrained (1000 classes)
# [ 751, 512] → Market-1501 pretrained (751 IDs)
# [ 702, 512] → DukeMTMC pretrained (702 IDs)
```

#### 조치

- Market-1501 가중치: [torchreid Model Zoo 페이지](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO)에서 `osnet_x1_0` → Market-1501 항목 직접 다운로드
- 다운로드 후 `configs/config.yaml`의 `reid.weights_path`에 절대 경로 지정
- `~` 홈 디렉토리 축약은 Python `Path`가 자동 확장하지 않으므로 **반드시 절대 경로** 사용

#### 실험 결과 비교 (zero-shot, 동일 데이터셋)

| 가중치 | Rank-1 | Rank-5 | mAP |
| :--- | ---: | ---: | ---: |
| ImageNet pretrained | **72.73%** | **100.0%** | **52.23%** |
| Market-1501 pretrained | 54.55% | 63.64% | 51.12% |

→ Market-1501 가중치가 오히려 낮은 이유: 공개 데이터셋과 촬영 환경(카메라, 조명, 복장) 도메인 차이로 인해 ImageNet의 범용 특징이 우리 데이터에 더 유리하게 작동.

---

## 7. 기타

### 7.1 Runpod Cmds (2026-05-31)

* ssh -i ~/.ssh/id_ed25519 xhj5hlgd3z8d7f-64410ffb@ssh.runpod.io

* git clone https://github.com/MarcusBae/EYE-D-RE.git

* pip install gdown

* gdown "https://drive.google.com/file/d/1uy3GSywVl_PmfpswEDYMty3P5FjzygzW/view?usp=drive_link" -O market1501.zip

* gdown "https://drive.google.com/file/d/1ibuCdnEhGmXDT0C4gX32H-tsFTKxdNBa/view?usp=drive_link" -O filtered.zip

* gdown "https://drive.google.com/file/d/1v3qPm4TLrjppJJRKBLX7YjEBLxOSFHc-/view?usp=drive_link" -O tracklets.zip

* python scripts/evaluate_zeroshot.py --config configs/config.yaml

* jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser &

---

## 8. 데모

### 8.1 시연 시나리오 (제품 데모)

---

### 단계 1 — PC 2대 시연

#### 장비 구성

```
Ubuntu PC #1  ── 카메라 A ── 검출 + 트래킹 + Re-ID 특징 추출
Ubuntu PC #2  ── 카메라 B ── 검출 + 트래킹 + Re-ID 특징 추출
      └──────────── LAN ─────────────┘
                매칭 결과 공유 + UI 출력
```

#### 시나리오 A — 동선 분석 (오프라인)

**컨셉**: "녹화된 영상을 분석해 누가 어디를 지나갔는지 자동으로 정리"

**데모 흐름**
1. 카메라 A, B 녹화 영상 업로드
2. 전체 영상 자동 처리 (검출 → 트래킹 → Re-ID → cross-camera 매칭)
3. 인물별 동선 타임라인 생성 및 출력

```
[입력]
  cam_A.mp4  cam_B.mp4
       ↓ 자동 분석
[결과 — 인물 동선 리포트]
  Person #3  09:42 cam_A → 09:45 cam_B → 09:58 cam_A
  Person #7  09:51 cam_A → 10:03 cam_B
  Person #12 10:11 cam_B (단일 카메라)
```

**비즈니스 메시지**: 사후 분석·보안 조사·공간 이용 패턴 파악 (실시간 인프라 불필요)

#### 시나리오 B — 관심 인물 추적 (오프라인)

**컨셉**: "녹화 영상에서 특정 인물이 어디에 등장했는지 찾아드립니다"

**데모 흐름**
1. 카메라 A, B 녹화 영상 업로드
2. 영상 재생 중 관심 인물 선택 (클릭 또는 ID 입력)
3. 전체 영상에서 해당 인물 등장 구간 자동 탐색
4. 결과 — 등장 시각, 카메라, 유사도 목록 출력

```
[입력]
  cam_A.mp4  cam_B.mp4  + 관심 인물 지정 (Person #3)
       ↓ 자동 탐색
[결과 — 등장 구간 목록]
  09:42~09:47  cam_A  유사도 0.91  ← 등장
  09:45~09:52  cam_B  유사도 0.87  ← 재등장 (cross-camera)
  [해당 구간 썸네일 + 영상 클립 미리보기]
```

**비즈니스 메시지**: 용의자·VIP 사후 추적, 보안 영상 검색 시간 단축

#### 구현 우선순위 (단계 1)

| 순위 | 항목 | 내용 |
| :--- | :--- | :--- |
| 1 | 실시간 파이프라인 | 카메라 → 검출 → 트래킹 → Re-ID → 매칭 |
| 2 | UI / 시각화 | OpenCV 오버레이 또는 웹 대시보드 |
| 3 | 네트워크 통신 | PC 간 crop 전송 + 매칭 결과 공유 |

#### 미결 결정 사항 (단계 1)

| 항목 | 선택지 |
| :--- | :--- |
| 카메라 입력 | USB 웹캠 / IP 카메라 / 녹화 영상 재생 |
| 매칭 트리거 | 자동(임계값 초과) / 수동(버튼) |
| 결과 UI | OpenCV 오버레이 / 웹 대시보드 / 터미널 |

---

### 단계 2 — PC 2대 + Jetson Orin Nano 시연

#### 장비 구성

```
Ubuntu PC #1  ── 카메라 A ── 검출 + 트래킹
Ubuntu PC #2  ── 카메라 B ── 검출 + 트래킹
Jetson Orin Nano ── 카메라 C (또는 PC crop 수신)
                     검출 + 트래킹 + Re-ID 전담
      └──────────────── LAN ─────────────────┘
              3-node 매칭 결과 통합 + UI 출력
```

#### 시나리오 C — 3-카메라 확장 추적

**컨셉**: "카메라가 늘어나도 Jetson 한 대만 추가하면 확장 가능"

**데모 흐름**
1. 단계 1과 동일한 A/B 카메라 추적 진행
2. Jetson 연결 카메라 C 화면 추가
3. A → B → C 이동 경로를 하나의 ID로 연속 추적
4. "Jetson 추가 시 카메라 1대 더 커버" 실시간 시연

**비즈니스 메시지**: 노드 단위 확장 가능한 분산 Re-ID 아키텍처

#### 시나리오 D — PC vs Jetson 성능 비교

**컨셉**: "클라우드 없이 현장에서 바로 동작, 저전력으로"

```
[PC #1]                    [Jetson Orin Nano]
  ~30 FPS                    ~15 FPS (TensorRT)
  GPU 서버급                  소비전력 10W
  높은 설치 비용               카메라 옆 현장 설치
         ↓                          ↓
         동일 Re-ID 결과 나란히 비교
```

**비즈니스 메시지**: 별도 서버 인프라 없이 엣지에서 즉시 배포 가능

#### 구현 우선순위 (단계 2)

| 순위 | 항목 | 내용 |
| :--- | :--- | :--- |
| 1 | Jetson 환경 구성 | 의존성 설치, 모델 동작 확인 |
| 2 | TensorRT 최적화 | YOLOv8n + OSNet INT8/FP16 변환 |
| 3 | 3-node 통신 | PC #1, PC #2, Jetson 간 결과 동기화 |
| 4 | 통합 UI | 3-카메라 뷰 + 동선 로그 |

#### 미결 결정 사항 (단계 2)

| 항목 | 선택지 |
| :--- | :--- |
| Jetson 역할 | 카메라 1대 전담 처리 / Re-ID 추론 서버 |
| 최적화 수준 | FP16 / INT8 (정확도 trade-off 확인 필요) |
| 통신 프로토콜 | gRPC / ZeroMQ / REST API |

