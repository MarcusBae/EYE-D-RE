"""
scripts/video_reid_app.py
=========================
동영상 파일을 재생하면서 인물 탐지·추적·Re-ID 결과를 실시간으로 오버레이하는
Streamlit 앱.

pipeline/video_utils.iter_annotated_frames() 제너레이터를 사용하여
영상의 각 프레임에 대해 YOLOv8n + BoT-SORT + OSNet Re-ID 를 온라인으로 수행하고
어노테이션된 프레임을 Streamlit 화면에 즉시 표시합니다.

실행:
    streamlit run scripts/video_reid_app.py
    streamlit run scripts/video_reid_app.py -- --config configs/config.yaml
"""

from __future__ import annotations

import base64
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.video_utils import iter_annotated_frames, REID_PALETTE_BGR

# ── 페이지 설정 ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="EYE-D | Video Re-ID",
    page_icon="🎬",
    layout="wide",
)

# ── 모델 캐시 ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="YOLOv8 + BoT-SORT 모델 로딩 중…")
def _load_yolo(model_path: str):
    from ultralytics import YOLO
    return YOLO(model_path)


@st.cache_resource(show_spinner="OSNet Re-ID 특징 추출기 로딩 중…")
def _load_extractor(model_name: str, pretrained: bool,
                    weights_path: Optional[str], device: str):
    from pipeline.reid_merger import OSNetExtractor
    return OSNetExtractor(
        model_name=model_name,
        pretrained=pretrained,
        weights_path=weights_path or None,
        device=device,
    )


@st.cache_data
def _load_config(config_path: str) -> dict:
    import yaml
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _rgb(gid: int) -> tuple[int, int, int]:
    b, g, r = REID_PALETTE_BGR[abs(int(gid)) % len(REID_PALETTE_BGR)]
    return (r, g, b)


def _hex(gid: int) -> str:
    r, g, b = _rgb(gid)
    return f"#{r:02x}{g:02x}{b:02x}"


@st.cache_data(show_spinner=False)
def _load_demo_gallery(demo_data_dir: str) -> dict:
    """demo_data의 persons를 읽어 {pid: {"b64": str, "feat": ndarray}} 반환."""
    manifest = Path(demo_data_dir) / "manifest.json"
    if not manifest.exists():
        return {}
    with open(manifest, encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for p in data.get("persons", []):
        pid = int(p["id"])
        feat_path  = Path(demo_data_dir) / "crops" / f"{pid:04d}" / "feature.npy"
        thumb_path = Path(demo_data_dir) / "crops" / f"{pid:04d}" / "thumb.jpg"
        if not feat_path.exists() or not thumb_path.exists():
            continue
        feat = np.load(str(feat_path)).astype(np.float32)
        norm = float(np.linalg.norm(feat))
        feat = feat / max(norm, 1e-8)
        img = cv2.imread(str(thumb_path))
        if img is None:
            continue
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf.tobytes()).decode()
        result[pid] = {"b64": b64, "feat": feat}
    return result


def _render_demo_gallery(demo_gallery: dict, matched_ids: set) -> str:
    """모든 썸네일을 flex 그리드로 렌더링. matched_ids에 있으면 크게 강조."""
    if not demo_gallery:
        return ""
    parts = [
        "<div style='display:flex;flex-wrap:wrap;gap:4px;padding:6px;"
        "background:#0e1117;border-radius:6px'>"
    ]
    for pid, info in sorted(demo_gallery.items()):
        matched = pid in matched_ids
        h       = 96 if matched else 44
        border  = f"3px solid {_hex(pid)}" if matched else "1px solid #333"
        glow    = f"0 0 8px {_hex(pid)}99" if matched else "none"
        lc      = "#fff" if matched else "#555"
        parts.append(
            f"<div style='text-align:center'>"
            f"<img src='data:image/jpeg;base64,{info['b64']}' "
            f"style='height:{h}px;width:auto;display:block;"
            f"border:{border};border-radius:3px;box-shadow:{glow}'/>"
            f"<div style='font-size:9px;color:{lc};margin-top:1px'>P{pid}</div>"
            f"</div>"
        )
    parts.append("</div>")
    return "".join(parts)


# ── 사이드바 ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎬 EYE-D Video Re-ID")
    st.caption("실시간 인물 탐지 · 추적 · 재식별")
    st.divider()

    config_path = st.text_input("설정 파일", value="configs/config.yaml")
    try:
        cfg = _load_config(config_path)
        st.success("설정 로드 완료", icon="✅")
    except Exception:
        cfg = {}
        st.warning("설정 파일 없음 — 기본값 사용", icon="⚠️")

    det_cfg  = cfg.get("detection",  {})
    reid_cfg = cfg.get("reid",       {})
    trk_cfg  = cfg.get("tracking",   {})

    st.subheader("🔧 모델")
    model_path    = st.text_input("YOLOv8 가중치",           value=det_cfg.get("model", "yolov8n.pt"))
    tracker_yaml  = st.text_input("Tracker 설정 (yaml)",     value=trk_cfg.get("tracker_yaml", "configs/botsort.yaml"))
    osnet_weights = st.text_input("OSNet 가중치 (빈칸=ImageNet)",
                                  value=reid_cfg.get("weights_path") or "")
    device = st.selectbox("연산 장치", ["auto", "cpu", "cuda"], index=0)

    st.subheader("⚙️ 파라미터")
    conf_thr     = st.slider("검출 신뢰도",       0.10, 0.90,
                              float(det_cfg.get("conf_threshold", 0.55)), 0.05)
    iou_thr      = st.slider("NMS IoU",           0.10, 0.90,
                              float(det_cfg.get("iou_threshold",  0.45)), 0.05)
    frame_stride = st.slider("프레임 스트라이드", 1, 12, 3,
                              help="N 프레임마다 1회 추론. 클수록 빠르지만 추적 정밀도 감소")
    reid_thresh  = st.slider("Re-ID 유사도 임계값", 0.40, 0.98, 0.75, 0.01,
                              help="cosine 유사도. 높을수록 동일인 판정이 엄격해짐")
    min_frames   = st.slider("Re-ID 최소 프레임",  2, 20, 6,
                              help="이 값 이상 크롭이 모여야 Re-ID 실행")
    show_unconf  = st.checkbox("미확정 트랙 표시 (회색)", value=True)

    st.divider()
    st.subheader("🔬 매칭 전략")
    distance_metric = st.selectbox(
        "거리 척도", ["cosine", "l2"],
        help="cosine: 각도 기반 / l2: 유클리드 (단위 벡터에서 동등하지만 체감 차이 있음)")
    match_strategy = st.selectbox(
        "매칭 전략",
        ["nearest", "mutual_topk", "mean"],
        format_func=lambda x: {
            "nearest":    "Nearest Neighbor",
            "mutual_topk": "Mutual Top-K (상호 검증)",
            "mean":        "Mean / AQE (평균 쿼리 확장)",
        }[x],
        help="Mean(AQE): probe를 top-k 갤러리 피처 평균으로 확장 후 재비교 → 클러스터 중심에 가까울수록 매칭 강화")
    mutual_topk_k = 3
    mean_k = 5
    if match_strategy == "mutual_topk":
        mutual_topk_k = st.slider("상호 Top-K (k)", 2, 10, 3)
    elif match_strategy == "mean":
        mean_k = st.slider("AQE Top-K (k)", 2, 10, 5,
                            help="probe 확장에 사용할 갤러리 피처 수")

    use_rerank = st.checkbox(
        "k-reciprocal Re-ranking 사용",
        value=False,
        help="Zhong et al. 2017. Jaccard 거리 기반 재순위로 정확도 향상 (FPS 소폭 하락)")
    rerank_k1, rerank_k2, rerank_lambda, rerank_min_gallery = 20, 6, 0.3, 5
    if use_rerank:
        rerank_min_gallery = st.slider(
            "Re-ranking 최소 갤러리 크기", 3, 20, 5,
            help="등록 인물이 이 수 이상 쌓여야 re-ranking 활성화")
        rerank_k1 = st.slider("k1 (k-reciprocal 범위)", 5, 30, 20)
        rerank_k2 = st.slider("k2 (local expansion)", 2, 10, 6)
        rerank_lambda = st.slider(
            "λ (원본 거리 가중치)", 0.0, 1.0, 0.3, 0.05,
            help="0 = 순수 Jaccard 거리 / 1 = 원본 cosine 거리")

    st.divider()
    st.subheader("💾 어노테이션 영상 저장")
    save_video = st.checkbox("처리 후 MP4 저장", value=True)

# ── 메인 영역 ────────────────────────────────────────────────────────
st.header("🎬 실시간 인물 탐지 · 추적 · Re-ID")

# ── 영상 파일 선택 ────────────────────────────────────────────────────
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

def _scan_videos(search_dirs: list[str]) -> list[Path]:
    found = []
    for d in search_dirs:
        p = Path(d)
        if p.is_dir():
            for ext in VIDEO_EXTS:
                found.extend(sorted(p.glob(f"*{ext}")))
    return found

video_dir = cfg.get("data", {}).get("video_dir", "data/raw_videos")
scan_dirs = [video_dir, "data", "."]
candidates = _scan_videos(scan_dirs)

sel_col, manual_col = st.columns([3, 2])
with sel_col:
    if candidates:
        options = ["— 선택하세요 —"] + [str(p) for p in candidates]
        chosen = st.selectbox(
            "영상 파일 선택",
            options,
            index=0,
            help=f"{', '.join(scan_dirs)} 폴더를 자동 스캔한 결과입니다.",
        )
        video_src = chosen if chosen != "— 선택하세요 —" else None
    else:
        st.warning("스캔된 영상 파일이 없습니다. 아래 경로를 직접 입력하세요.", icon="⚠️")
        video_src = None

with manual_col:
    manual_path = st.text_input(
        "또는 경로 직접 입력",
        value="",
        placeholder="data/raw_videos/cam1_t1.avi",
    )
    if manual_path.strip():
        video_src = manual_path.strip()

if video_src and not Path(video_src).exists():
    st.error(f"파일 없음: `{video_src}`")
    video_src = None
elif video_src:
    st.caption(f"선택된 파일: `{video_src}`")

st.divider()

# 실행 / 중지 버튼
run_col, stop_col = st.columns([3, 1])
with run_col:
    run_btn  = st.button("▶ 분석 시작", type="primary",
                          disabled=(not video_src or not Path(str(video_src)).exists()),
                          use_container_width=True)
with stop_col:
    stop_btn = st.button("⏹ 중지", use_container_width=True)

if stop_btn:
    st.session_state["stop_flag"] = True

if run_btn:
    st.session_state["stop_flag"]           = False
    st.session_state["result_video"]        = None
    st.session_state["persons"]             = {}
    st.session_state["matched_gallery_ids"] = set()

# ── 참조 갤러리 (demo_data 썸네일) ──────────────────────────────────
DEMO_DATA_DIR = str(ROOT / "data" / "demo_data")
demo_gallery  = _load_demo_gallery(DEMO_DATA_DIR)
demo_gallery_slot = None

if demo_gallery:
    with st.expander(f"🖼️ 참조 갤러리 — {len(demo_gallery)}명", expanded=True):
        st.caption("영상 속 인물이 갤러리의 누군가와 매칭되면 해당 썸네일이 크게 표시됩니다.")
        demo_gallery_slot = st.empty()
        _init_matched = st.session_state.get("matched_gallery_ids", set())
        demo_gallery_slot.markdown(
            _render_demo_gallery(demo_gallery, _init_matched),
            unsafe_allow_html=True,
        )

# ── 라이브 처리 영역 ─────────────────────────────────────────────────
if run_btn and video_src and Path(video_src).exists():

    # 모델 로드
    try:
        yolo_model = _load_yolo(model_path)
        extractor  = _load_extractor(
            reid_cfg.get("model_name", "osnet_x1_0"),
            bool(reid_cfg.get("pretrained", True)),
            osnet_weights.strip() or None,
            device,
        )
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        st.stop()

    # ── UI 레이아웃 ───────────────────────────────────────────────────
    vid_col, info_col = st.columns([3, 1])

    with vid_col:
        st.subheader("📺 실시간 처리 화면")
        frame_slot   = st.empty()   # 라이브 프레임 표시
        progress_bar = st.progress(0.0)
        status_text  = st.empty()

    with info_col:
        st.subheader("📊 실시간 통계")
        metric_frame  = st.empty()
        metric_det    = st.empty()
        metric_person = st.empty()
        metric_fps    = st.empty()
        st.divider()
        st.subheader("👥 탐지 인물")
        gallery_slot  = st.empty()  # 인물 갤러리 (실시간 갱신)

    # ── 어노테이션 영상 저장용 VideoWriter ────────────────────────────
    writer   = None
    out_path = None

    if save_video:
        tmp_out   = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        out_path  = tmp_out.name
        tmp_out.close()

    # ── 인물 갤러리 상태 ─────────────────────────────────────────────
    # gid → {"thumb_bgr": ndarray, "first_frame": int, "last_frame": int, "count": int}
    persons_state: dict[int, dict] = {}
    t_start = time.time()

    # demo_gallery features → init_gallery for Re-ID seeding
    init_gallery = (
        {pid: info["feat"] for pid, info in demo_gallery.items()}
        if demo_gallery else None
    )

    # ── 제너레이터 루프 ───────────────────────────────────────────────
    for ann_frame, stats in iter_annotated_frames(
        video_path          = video_src,
        yolo_model          = yolo_model,
        extractor           = extractor,
        device              = device,
        conf                = conf_thr,
        iou                 = iou_thr,
        imgsz               = 640,
        tracker_yaml        = tracker_yaml,
        frame_stride        = frame_stride,
        reid_sim_thresh     = reid_thresh,
        min_frames_for_reid = min_frames,
        show_unconfirmed    = show_unconf,
        init_gallery        = init_gallery,
        distance_metric     = distance_metric,
        match_strategy      = match_strategy,
        mutual_topk_k       = mutual_topk_k,
        mean_k              = mean_k,
        use_rerank          = use_rerank,
        rerank_k1           = rerank_k1,
        rerank_k2           = rerank_k2,
        rerank_lambda       = rerank_lambda,
        rerank_min_gallery  = rerank_min_gallery,
    ):
        if st.session_state.get("stop_flag", False):
            status_text.warning("⏹ 사용자가 중지했습니다.")
            break

        fi        = stats["frame_idx"]
        total     = stats["total_frames"]
        n_det     = stats["num_detections"]
        n_persons = stats["num_persons"]
        tid_to_gid = stats["tid_to_gid"]

        # ── 라이브 프레임 표시 ────────────────────────────────────────
        frame_rgb = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)
        frame_slot.image(
            Image.fromarray(frame_rgb),
            use_container_width=True,
        )

        # ── 진행률 + 통계 ─────────────────────────────────────────────
        progress_bar.progress(min(fi / total, 1.0))
        elapsed    = time.time() - t_start
        proc_fps   = (stats["processed_idx"] + 1) / max(elapsed, 1e-3)
        status_text.caption(
            f"프레임 {fi} / {total}  ·  경과 {elapsed:.1f}s  ·  처리 {proc_fps:.1f} fps"
        )

        metric_frame.metric("현재 프레임",       f"{fi} / {total}")
        metric_det.metric("현재 프레임 탐지",    f"{n_det}명")
        metric_person.metric("누적 Re-ID 인물",  f"{n_persons}명")
        metric_fps.metric("처리 속도",           f"{proc_fps:.1f} fps")

        # ── 인물 갤러리 상태 업데이트 ────────────────────────────────
        for gid in set(tid_to_gid.values()):
            if gid <= 0:
                continue
            if gid not in persons_state:
                persons_state[gid] = {
                    "first_frame": fi,
                    "last_frame":  fi,
                    "count":       0,
                }
            ps = persons_state[gid]
            ps["last_frame"] = fi
            ps["count"]     += 1

        # 갤러리 카드 렌더링
        if persons_state:
            card_lines = []
            for gid, ps in sorted(persons_state.items()):
                col_hex = _hex(gid)
                start_s = ps["first_frame"] / max(stats["fps"], 1.0)
                last_s  = ps["last_frame"]  / max(stats["fps"], 1.0)
                card_lines.append(
                    f"<div style='border-left:4px solid {col_hex};"
                    f"padding:4px 8px;margin-bottom:4px;background:#1e1e1e;"
                    f"border-radius:4px'>"
                    f"<b style='color:{col_hex}'>ID {gid}</b>"
                    f"<br><span style='color:#9ca3af;font-size:0.8em'>"
                    f"{start_s:.1f}s ~ {last_s:.1f}s · {ps['count']}회"
                    f"</span></div>"
                )
            gallery_slot.markdown("".join(card_lines), unsafe_allow_html=True)

        # ── 참조 갤러리 썸네일 업데이트 ──────────────────────────────
        if demo_gallery_slot is not None:
            matched_gids = stats.get("matched_gallery_ids", set())
            st.session_state["matched_gallery_ids"] = matched_gids
            demo_gallery_slot.markdown(
                _render_demo_gallery(demo_gallery, matched_gids),
                unsafe_allow_html=True,
            )

        # ── VideoWriter 에 프레임 저장 ────────────────────────────────
        if save_video and out_path:
            if writer is None:
                h, w = ann_frame.shape[:2]
                out_fps = max(stats["fps"] / frame_stride, 1.0)
                fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
                writer  = cv2.VideoWriter(out_path, fourcc, out_fps, (w, h))
            writer.write(ann_frame)

    # ── 루프 종료 후 처리 ─────────────────────────────────────────────
    if writer is not None:
        writer.release()

    progress_bar.progress(1.0)
    status_text.success(
        f"✅ 완료! {stats['processed_idx']+1}프레임 처리 / "
        f"Re-ID 인물 {stats['num_persons']}명"
    )

    # 결과 저장 (세션 유지용)
    st.session_state["result_video"]  = out_path
    st.session_state["persons"]       = persons_state
    st.session_state["last_stats"]    = stats

# ── 처리 완료 후: 어노테이션 영상 재생 + 인물 갤러리 ──────────────────
result_video = st.session_state.get("result_video")
persons      = st.session_state.get("persons", {})

if result_video and Path(result_video).exists() and not run_btn:
    st.divider()
    st.subheader("🎞️ 어노테이션 결과 영상")

    res_col, gal_col = st.columns([3, 2])
    with res_col:
        # H.264 재인코딩 시도 (브라우저 호환)
        try:
            import subprocess
            h264_path = result_video.replace(".mp4", "_h264.mp4")
            subprocess.run(
                ["ffmpeg", "-y", "-i", result_video,
                 "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                 "-movflags", "+faststart", h264_path],
                check=True, capture_output=True, timeout=300,
            )
            play_path = h264_path
        except Exception:
            play_path = result_video

        with open(play_path, "rb") as f:
            st.video(f.read())
        st.caption(f"저장 경로: `{play_path}`")

    with gal_col:
        st.subheader(f"👥 탐지 인물 — {len(persons)}명")
        last_stats = st.session_state.get("last_stats", {})
        fps_val    = last_stats.get("fps", 25.0)

        for gid, ps in sorted(persons.items()):
            col_hex  = _hex(gid)
            start_s  = ps["first_frame"] / max(fps_val, 1.0)
            end_s    = ps["last_frame"]  / max(fps_val, 1.0)
            duration = end_s - start_s
            with st.container(border=True):
                st.markdown(
                    f"<span style='background:{col_hex};padding:3px 10px;"
                    f"border-radius:4px;color:white;font-weight:bold'>"
                    f"Re-ID  ID: {gid}</span>",
                    unsafe_allow_html=True,
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("첫 등장", f"{start_s:.1f}s")
                c2.metric("마지막", f"{end_s:.1f}s")
                c3.metric("체류", f"{duration:.1f}s")
                st.caption(f"탐지 횟수: {ps['count']}회")

elif not run_btn and not result_video:
    st.info("영상을 선택하고 **▶ 분석 시작** 버튼을 누르세요.")

# ── 하단 ─────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**EYE-D Video Re-ID** · YOLOv8n + BoT-SORT + OSNet x1.0  |  "
    "`streamlit run scripts/video_reid_app.py`"
)
