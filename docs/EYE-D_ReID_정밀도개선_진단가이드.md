# EYE-D Re-ID 정밀도 개선 — 오병합/과분할 진단 가이드 (v2, 실제 코드 반영)

작성: 2026-06-09 / 코드 검증 갱신: 2026-06-10 / 대상: feat/init (origin 271b738)
목적: 수동 검수에서 발견된 ID 오류("같은 사람 다른 ID" / "2명을 1명으로")를 실제 코드·파라미터에
매핑하고, 이미 존재하는 도구와 개선 지점을 정리한 코드 학습 로드맵.

> v2 변경점: 1차 가이드의 "추정"을 실제 코드(`pipeline/reid_merger.py`,
> `scripts/split_switched_tracklets.py`, `scripts/find_multi_person.py`, `configs/config.yaml`)로
> 검증·교체. 핵심: 제안했던 진단 도구 중 일부는 **이미 구현돼 있음**.

---

## 1. 두 오류 유형 — 정의와 핵심 원리

| 유형 | 현상 | Re-ID 용어 | 본질 |
|---|---|---|---|
| **A형** | 같은 사람이 여러 ID | 과분할 / under-merge | 합쳐야 할 것을 못 합침 |
| **B형** | 두 사람이 한 ID | 과병합 / over-merge | 나눠야 할 것을 합침 |

병합 임계를 낮추면 A형↓·B형↑, 높이면 반대 — 임계 하나로 동시에 못 잡는다. 정밀도는 **제약(동시성)
+ ID-switch 탐지 + 임계/linkage**의 조합으로 결정된다.

---

## 2. 실제 코드 구조 (검증됨)

| 단계 | 파일 | 핵심 |
|---|---|---|
| 1. 추적 | `scripts/run_tracking.py`, `pipeline/tracker.py`, `pipeline/detector.py`, `configs/botsort.yaml` | YOLOv8n + BoT-SORT(IoU+Re-ID 외형). conf 0.5, iou 0.45, imgsz 640. **frame_stride≈6(≈4fps)이 ID switch·분할 증폭** |
| 4. ID switch 분리 | `scripts/split_switched_tracklets.py` (302줄) | 슬라이딩 앵커 대비 유사도 급락 분리 |
| 5. 동일 cam 재병합 | `scripts/merge_same_camera_tracklets.py` (286줄) | `--sim-threshold 0.75` (내부 미검증) |
| 7. 품질 필터 | `scripts/filter_tracklets.py`, `pipeline/quality_filter.py` | len≥6, conf≥0.7, bbox≥64×32 |
| 8. cross-camera 병합 | `scripts/merge_ids.py`(131줄, 오케스트레이터) → `pipeline/reid_merger.py`(474줄, 로직) | must-not-link + HAC |
| 9. 수동 큐레이션 | `export_global_id_sheet.py`, `gather_by_person.py`, `persons_to_filtered.py` | 썸네일 시트 + 인물 폴더 |
| 10/11 | `format_market1501.py`, `evaluate_zeroshot.py` (+ `pipeline/reranking.py`) | 데이터셋 변환 / 평가 |
| **진단(기존)** | **`find_multi_person.py`(97줄), `analyze_cluster_purity.py`** | **B형 탐지 — 이미 존재** |

`pipeline/reid_merger.py` = 두 클래스: ① OSNet **특징 추출기**(`extract_image/batch_features`)
② **병합기**(`compute_distance_matrix`, `apply_must_not_link_constraints`, `run_hac_clustering`,
`enforce_must_not_link`, `verify_clustering_results`, `_has_conflict`).

---

## 3. 8단계 병합 로직 해부 — `_has_conflict` + HAC

### must-not-link 정의 (`_has_conflict(t1, t2)`)
True = 같은 사람일 수 없음(병합 금지). 세 갈래:
1. `allow_cross_slot=False`(기본값, `reid_merger.py:197`)이고 **다른 time_slot** → 무조건 충돌.
   = 시간대를 넘는 병합을 통째로 차단 (ID 046 과병합 방지 가드).
2. 다른 카메라 **또는** 다른 슬롯 → 충돌 아님(병합 허용). cross-camera 같은 슬롯 = 정상 병합 대상.
3. 같은 카메라 + 같은 슬롯 + **프레임 인덱스 겹침** → 충돌. (동시 존재 = 다른 사람)

적용은 2회 — 거리행렬(`apply_must_not_link_constraints`)에서 ∞ 처리 + HAC 후
(`enforce_must_not_link`) 위반 클러스터 강제 분리 + `verify_clustering_results` 검증. **견고함.**

### HAC (`run_hac_clustering`)
scipy `linkage(method=self.linkage_method)` + `fcluster(t=self.threshold, criterion="distance")`.
- **config 실제값:** `threshold: 0.20` (cosine distance, 이전 0.12 → **상향 = 더 쉽게 병합**),
  `linkage: "average"`.
- time_slot 구조: 녹화 세션 10개(t1~t10, 12:00~16:30), 슬롯당 cam1/2/3 동시 촬영.

---

## 4. 두 오류를 코드 지점에 매핑 (핵심)

### A형 "같은 사람을 다른 ID로" — 세 원인
- **A-slot (가장 큼·구조적):** `allow_cross_slot=False` → 같은 사람이 다른 슬롯에 재등장하면
  **무조건 새 global_id.** 임계 문제 아님. 4시간 반 동안 같은 배우가 여러 슬롯에 나오면 슬롯 수만큼
  ID가 쪼개진다. → 큐레이션 43명 중 일부는 동일인의 다른 슬롯일 수 있음(인원 부풀림 + 다양성 손실).
- **A-cross-camera:** HAC `threshold 0.20` / average linkage가 일부 카메라 간 동일인엔 여전히 엄격.
- **A-within-camera:** 5단계 `--sim-threshold 0.75`가 엄격하면 카메라 내 동일인 미병합.

### B형 "2명을 1명으로" — 네 경로
- **B1-공간 (느슨한 박스에 두 사람):** `find_multi_person.py`가 담당 (§5 참고).
- **B1-시간 (추적 ID swap, 급변):** `split_switched_tracklets.py`가 담당 (§5 참고).
- **B2-동시발생 (같은 cam·슬롯·시간 겹침):** must-not-link가 **구조적으로 방지** ✓.
- **B2-닮은 타인 (다른 cam/시간):** must-not-link 안 걸림 → 오직 HAC `threshold 0.20`+linkage 의존.
  **임계를 0.12→0.20으로 올린 만큼 이 경로의 과병합 위험이 커졌을 수 있음 — 점검 1순위.**

---

## 5. 이미 있는 진단 도구 — 동작과 개선점

### `find_multi_person.py` — B1-공간 탐지 (97줄)
동작: crop마다 YOLO 재검출 → 사람 클래스 ≥2면 그 **프레임을 `data/remove/`로 이동(`shutil.move`)**
+ `output/multi_person_removed.json` 저장. 대상 `data/tracklets-2-manual-edit` 하드코딩.
개선점:
1. **오탐 위험 — 판정이 `person_count>=2`뿐.** 배경 행인·반사·가장자리 신체도 카운트. → 2번째 박스의
   **면적 비율·중심 위치 필터** 추가.
2. **파괴적** — 플래그와 동시에 자동 이동. 오탐 시 멀쩡한 프레임 손실. → **리포트 우선, 검수 후 이동.**
3. **프레임 단위만** — 산발적 다중검출(박스 느슨) vs 트랙렛 다수 프레임 다중(진짜 두 사람) 미구분.
   → 트랙렛당 다중검출 비율 지표 추가.
4. `target_dir` 하드코딩 → 파라미터화.

### `split_switched_tracklets.py` — B1-시간 분리 (302줄)
동작: 슬라이딩 윈도우 평균(앵커) 대비 유사도가 `SIM_THRESHOLD=0.70` 미만으로
`MIN_SWITCH_FRAMES=3` 연속 → 분리. `MIN_FRAMES=8` 미만 조각 폐기. (`--sim-threshold` CLI override)
평가: **hysteresis 있음(3연속)** → 단일 프레임 노이즈에 강함. 앵커 슬라이딩 → 점진 변화 적응.
개선점/실험: `SIM_THRESHOLD 0.70` 민감도 sweep (낮추면 B1↑잡지만 A형 오분할↑). 닮은 옷·완만한
전환은 이 방식이 놓침 → `find_multi_person`/큐레이션과 병행.

### `analyze_cluster_purity.py` — 클러스터 순도 (내부 미검증)
보고서 §3-2(global_id 내부 임베딩 재분할로 두 사람 의심) 기능일 가능성. **다음 정독 대상.**

---

## 6. 측정 — 검수 결과를 수치로

- **Fragmentation rate (A형):** 정답 인물 1명당 평균 global_id 수 (1.0이 이상).
- **Collision rate (B형):** global_id 1개당 평균 정답 인물 수 (1.0이 이상).
- 수동 검수를 ground-truth로 고정, 임계(0.20)·linkage·`allow_cross_slot`를 바꿔가며 두 곡선 비교.
- ⚠️ 현 평가셋 13명/31쿼리로 너무 작아 측정 노이즈 큼 → **정밀도 측정도 데이터 확보 전제.**

---

## 7. 개선 우선순위 (이번 주 이후)

1. **B2-닮은타인 점검:** `threshold 0.12→0.20` 상향 이후 과병합이 늘었는지 — `analyze_cluster_purity.py`
   결과로 확인. 필요시 0.20 재조정 또는 linkage 검토.
2. **`find_multi_person.py` 정밀화:** 면적/위치 필터 + 리포트 우선(비파괴) + 트랙렛 비율.
3. **A-slot(cross-slot) 정책 결정:** A형의 최대 원인. 자동 병합은 위험하니 **9단계 큐레이션에서
   슬롯 간 동일인 연결을 반자동 보조**(예: 슬롯 경계 후보쌍 시트화). `allow_cross_slot=True` + 더 엄격한
   임계 + 검증으로 푸는 안도 실험 가능.
4. **진단→큐레이션 연결:** `find_multi_person`·`analyze_cluster_purity` 출력을
   `export_global_id_sheet`에 합쳐 "의심 우선" 검수로 전환.
5. **상류(추적) 손질:** `frame_stride`를 줄이면(촘촘한 추적) A1·B1이 근원에서 감소(연산 비용↑). `configs/botsort.yaml`의 track_buffer·매칭 임계도 연속성 손잡이. nano 검출기(yolov8n) 업그레이드도 검출 누락 감소에 기여. *실제 stride 값 먼저 확인.*
6. **(데이터 확보 후)** OSNet-AIN fine-tuning으로 분리력 자체 향상. *지금 43명은 과적합 위험, 시기상조.*

---

## 8. 코드 학습 체크리스트 — 진행 현황

- [x] `merge_ids.py` / `reid_merger.py` — must-not-link(hard, _has_conflict) + HAC(0.20/average) 확인
- [x] `allow_cross_slot` 기본값 = False (cross-slot 병합 차단)
- [x] `split_switched_tracklets.py` — 0.70 / 3연속(hysteresis) / 8min 확인
- [x] `find_multi_person.py` — 개선점 4종 도출
- [x] `analyze_cluster_purity.py` — global_id 내 트랙렛 대표임베딩 pairwise cosine, sim<0.6 쌍 비율="오염도", 읽기전용 purity_report.json (B2-닮은타인 탐지)
- [x] `merge_same_camera_tracklets.py` — 같은 cam/slot + 시간 비겹침 + sim≥0.75 + gap≤300(≈75s) AND 병합 (A1 복구, must-not-link의 반대). 보수적 설계
- [x] `pipeline/tracker.py` — BoT-SORT(IoU+Re-ID) ID switch 1차 억제, but frame_stride≈6이 A1·B1 증폭. botsort.yaml에 연속성 임계

---

## 9. 실측 튜닝 결과 — 정답(43) 기반 평가 (2026-06-10)

팀장 보관본 `data/curated`(233 트랙렛, **고유 global_id 43개 = 수동 큐레이션 정답**)를 ground-truth로
삼아, 병합 결과를 쌍(pair) 단위로 평가. **merge recall** = 정답상 같은 사람 쌍 중 예측도 같이 묶은
비율(높을수록 덜 쪼갬), **merge precision** = 예측상 같은 쌍 중 실제로 같은 사람 비율(높을수록 덜
잘못 합침). 모든 실험은 `/tmp` 샌드박스에서 수행(원본 무손상).

### 베이스라인 (현재 설정: threshold 0.20, allow_cross_slot off)
- 예측 ID **107** vs 정답 43 → **recall 27.6% / precision 97.5%**
- 해석: 잘못 합치는 일은 거의 없으나(과병합 7쌍), **같은 사람의 72%를 갈라놓음** = 심한 과분할.

### 임계값 단독 스윕 (cross_slot off) — 막다른 길
| threshold | 예측ID | recall | precision |
|---|---|---|---|
| 0.20 | 107 | 27.6% | 97.5% |
| 0.30 | 73 | 35.1% | 70.9% |
| 0.40 | 48 | 35.5% | 45.8% |

→ 임계를 올려도 **recall이 ~35%에서 천장**(과분할 대부분이 cross-slot이라 임계로 못 고침),
precision만 붕괴. **임계 단독 조정은 해답이 아님.**

### allow_cross_slot ON — 핵심 해법
| 설정 | 예측ID | recall | precision |
|---|---|---|---|
| 0.20, off (기존) | 107 | 27.6% | 97.5% |
| **0.20, ON** | 81 | **57.4%** | **97.4%** |
| **0.25, ON** | 56 | **74.1%** | 90.3% |

→ **threshold 0.20에서 cross_slot만 켜면 recall이 2배(27.6→57.4%)인데 precision은 무손실(97.4%).**
사실상 부작용 없는 개선. 0.25까지 올리면 recall 74%·ID 56개로 정답 43에 근접(precision 90%로 소폭
하락, 과병합 77쌍 등장).

### 권장 조정
- **안전(precision 우선): `allow_cross_slot=True`, threshold 0.20** — recall 2배, 과병합 거의 무증가.
- **적극(정답 근접): `allow_cross_slot=True`, threshold 0.25** — 단 과병합 77쌍은 적용 전 육안 점검 필요.
- 과거 cross_slot을 끈 이유(ID 046 과병합)는 실측상 0.20-ON에서 과병합 7→15쌍으로 미미 → **우려가
  과했고, 적정 임계면 cross_slot ON이 명백히 이득.**

### ⚠️ 운영 주의 (이번에 배운 교훈)
`merge_ids.py`는 `config.data.filtered_dir`(현재 **`data/curated`**)에 global_id를 **in-place로 덮어씀.**
큐레이션 결과를 날릴 수 있으니 — ① 실행 전 그 폴더 백업 필수, ② config 값 먼저 확인, ③ 권장: 별도
출력 폴더에 쓰도록 개선. (실험은 항상 샌드박스 복사본에서.)

> 분석 도구: `scripts/find_oversplit.py`(과분할 후보 탐지), `scripts/eval_merge.py`(정답 대비 평가).

---

## 10. End-to-end · Fine-tuning 검증 + 방법론 주의 (2026-06-11)

### 10-1. End-to-end — 자기-라벨 평가의 순환 함정
개선 병합 데이터로 market1501 평가셋을 재생성해 평가하면 점수가 오르지만, **모델이 좋아진 게
아니다.** 세 데이터셋(single matching):

| 평가셋 (라벨) | 병합 ID | Query ID | mAP | Rank-1 |
|---|---|---|---|---|
| 사람-43 (독립 라벨) | 43 | 13 | 58.3% | 74.2% |
| 자동 0.20-off | 107 | 11 | 74.4% | 83.3% |
| 자동 0.25-on | 56 | 9 | 79.0% | 95.5% |

Query ID는 줄어드는데(13→11→9) 점수는 오른다(74→95). 자동 병합이 모델 기준으로 더 묶을수록,
그 라벨로 같은 모델을 평가하니 **점수가 부풀고 평가셋은 작고 쉬워진다(순환 평가).** → **신뢰할
수 있는 성능 수치는 독립 라벨(사람-43)뿐이다.** 병합 품질은 정답 대비 recall/precision(§9)으로 판단.

### 10-2. Fine-tuning — 30 IDs에선 불안정
`finetune.py`(osnet_ain_x1_0 마지막 블록, Triplet+CE, Market AIN 가중치에서 시작)로 사람-43
train(30 ID)에 학습 → 동일 harness(euclidean) 평가:

| | mAP | Rank-1 | Rank-5 |
|---|---|---|---|
| zero-shot (학습 안 함) | 54.4% | 77.4% | 87.1% |
| ft 5 epoch | **59.2%** | 77.4% | **93.5%** |
| ft 10 epoch | 47.5% | 58.1% | 74.2% |
| ft 30 epoch | 52.4% | 77.4% | 83.9% |

5 epoch은 개선(mAP +4.8)이지만 10 epoch에서 급락 → **에폭 몇 개 차이로 출렁이는 불안정 상태.**
30 train IDs로는 단일 실행을 신뢰할 수 없다(시드 평균 필요). **결론: fine-tuning은 "무익"이 아니라
"데이터 부족으로 불안정". 조기 중단(light fine-tuning)은 데이터 확보 후 안정적 이득이 기대된다.**

### 10-3. 방법론 주의 (팀 공유 권장)
1. **순환 평가 금지:** 자동 병합 라벨로 같은 모델의 성능을 자랑하면 안 된다. 성능은 *독립(사람)
   라벨* 평가셋으로만 보고.
2. **거리척도 일치:** `evaluate_zeroshot`(cosine)와 `finetune` harness(euclidean)는 수치를 직접
   비교하면 안 된다 — 같은 평가틀끼리만.
3. **merge_ids in-place 위험:** §9 운영주의 참조 (실행 전 입력 폴더 백업 필수).

> **종합:** 성능·과분할·fine-tuning 모든 층위에서 병목은 **데이터 양**. 측정 틀(정답 라벨 43,
> eval harness, recall/precision)은 이미 확립 → 데이터가 늘면 같은 절차로 즉시 재측정 가능.

---

## 부록 — 한 줄 요약

> "2명을 1명으로"는 **B1-공간(`find_multi_person`) / B1-시간(`split_switched_tracklets`) /
> B2-동시발생(must-not-link, 해결됨) / B2-닮은타인(HAC 0.20)** 네 갈래. "같은 사람 다른 ID"는
> 대부분 **cross-slot 가드(`allow_cross_slot=False`)의 구조적 분리**. 진단 도구는 이미 있으니
> 새로 만들기보다 **정밀화 + 큐레이션 연결**이 핵심. fine-tuning은 데이터 확보 후.
