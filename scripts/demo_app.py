"""
scripts/demo_app.py
====================
EYE-D 데모 Streamlit 앱.

실행:
    streamlit run scripts/demo_app.py
    streamlit run scripts/demo_app.py -- --data demo_data
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

# ── 설정 ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EYE-D | Person Re-ID Demo",
    page_icon="👁",
    layout="wide",
)

THUMB_W = 100  # px


# ── 데이터 로드 ───────────────────────────────────────────────────
@st.cache_data
def load_manifest(data_dir: str) -> dict:
    path = Path(data_dir) / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_image(path: str | None) -> Image.Image | None:
    if path and Path(path).exists():
        return Image.open(path).convert("RGB")
    return None


def load_feature(data_dir: str, pid: int) -> np.ndarray | None:
    p = Path(data_dir) / "crops" / f"{pid:04d}" / "feature.npy"
    return np.load(str(p)) if p.exists() else None


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ── 사이드바 ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("👁 EYE-D Demo")
    st.caption("Multi-Camera Person Re-Identification")
    st.divider()
    data_dir = st.text_input("Demo 데이터 경로", value="data/demo_data")
    st.divider()
    scenario = st.radio(
        "시나리오 선택",
        ["A — 동선 분석", "B — 관심 인물 탐색"],
        index=0,
    )

# ── 데이터 로드 ───────────────────────────────────────────────────
manifest = load_manifest(data_dir)

if not manifest:
    st.warning(f"**{data_dir}/manifest.json** 을 찾을 수 없습니다.")
    st.code("python scripts/demo_preprocess.py --config configs/config.yaml")
    st.stop()

persons = manifest.get("persons", [])
if not persons:
    st.error("등록된 인물이 없습니다.")
    st.stop()

# ── 시나리오 A: 동선 분석 ─────────────────────────────────────────
if scenario.startswith("A"):
    st.header("📊 시나리오 A — 동선 분석")
    st.caption("녹화 영상에 등장한 모든 인물의 카메라별 이동 경로를 자동으로 정리합니다.")
    st.divider()

    # 카메라 목록 수집
    all_cams = sorted({
        app["camera"]
        for p in persons
        for app in p["appearances"]
    })
    cam_labels = {c: f"CAM {c}" for c in all_cams}

    # 요약 지표
    multi_cam = [p for p in persons if len({a["camera"] for a in p["appearances"]}) > 1]
    col1, col2, col3 = st.columns(3)
    col1.metric("총 등장 인물", len(persons))
    col2.metric("다중 카메라 등장", len(multi_cam))
    col3.metric("카메라 수", len(all_cams))
    st.divider()

    # 인물별 동선 카드
    for person in persons:
        pid = person["id"]
        apps = person["appearances"]
        cams_seen = sorted({a["camera"] for a in apps})
        is_cross = len(cams_seen) > 1

        with st.expander(
            f"{'🔴 ' if is_cross else '⚪ '}Person #{pid:04d}  "
            f"| 등장 카메라: {', '.join(f'CAM {c}' for c in cams_seen)}  "
            f"| 총 {len(apps)}개 구간",
            expanded=is_cross,
        ):
            thumb_col, timeline_col = st.columns([1, 4])

            with thumb_col:
                img = load_image(person.get("thumb"))
                if img:
                    st.image(img, width=THUMB_W, caption=f"Person #{pid:04d}")
                else:
                    st.write("(썸네일 없음)")

            with timeline_col:
                rows = []
                for app in sorted(apps, key=lambda a: a["start_time"]):
                    try:
                        fmt = "%H:%M:%S"
                        s = datetime.strptime(app["start_time"], fmt)
                        e = datetime.strptime(app["end_time"], fmt)
                        sec = int((e - s).total_seconds())
                        dwell = f"{sec // 60:02d}:{sec % 60:02d}초"
                    except Exception:
                        dwell = "-"
                    rows.append({
                        "카메라": f"CAM {app['camera']}",
                        "등장 시각": app["start_time"],
                        "종료 시각": app["end_time"],
                        "체류 시간": dwell,
                        "프레임 수": app["num_frames"],
                    })
                st.table(rows)

                if is_cross:
                    path_str = " → ".join(
                        f"CAM {a['camera']} ({a['start_time']})"
                        for a in sorted(apps, key=lambda a: a["start_time"])
                    )
                    st.success(f"Cross-camera 이동 경로: {path_str}")

# ── 시나리오 B: 관심 인물 탐색 ───────────────────────────────────
else:
    st.header("🔍 시나리오 B — 관심 인물 탐색")
    st.caption("특정 인물을 선택하면 전체 영상에서 해당 인물이 등장한 구간을 찾아드립니다.")
    st.divider()

    # 인물 갤러리
    st.subheader("① 관심 인물 선택")
    n_cols = 8
    cols = st.columns(n_cols)
    selected_id = st.session_state.get("selected_pid", None)

    for i, person in enumerate(persons):
        pid = person["id"]
        with cols[i % n_cols]:
            img = load_image(person.get("thumb"))
            if img:
                st.image(img, width=THUMB_W)
            label = f"#{pid:04d}"
            if st.button(label, key=f"btn_{pid}",
                         type="primary" if pid == selected_id else "secondary"):
                st.session_state["selected_pid"] = pid
                st.rerun()

    st.divider()

    # 탐색 결과
    selected_id = st.session_state.get("selected_pid", None)
    if selected_id is None:
        st.info("위에서 관심 인물을 선택하세요.")
        st.stop()

    query_person = next((p for p in persons if p["id"] == selected_id), None)
    if not query_person:
        st.error("선택한 인물 정보를 찾을 수 없습니다.")
        st.stop()

    query_feat = load_feature(data_dir, selected_id)

    st.subheader(f"② 탐색 결과 — Person #{selected_id:04d}")

    q_col, r_col = st.columns([1, 4])
    with q_col:
        img = load_image(query_person.get("thumb"))
        if img:
            st.image(img, width=THUMB_W, caption="Query")

    with r_col:
        if query_feat is None:
            st.warning("이 인물의 특징 벡터가 없습니다.")
            st.stop()

        # 다른 인물과 유사도 계산
        results = []
        for person in persons:
            if person["id"] == selected_id:
                continue
            feat = load_feature(data_dir, person["id"])
            if feat is None:
                continue
            sim = cosine_sim(query_feat, feat)
            for app in person["appearances"]:
                results.append({
                    "person_id": person["id"],
                    "thumb": person.get("thumb"),
                    "similarity": sim,
                    "camera": app["camera"],
                    "start_time": app["start_time"],
                    "end_time": app["end_time"],
                })

        # 쿼리 인물 자신의 등장 구간도 포함 (유사도 1.0)
        for app in query_person["appearances"]:
            results.append({
                "person_id": selected_id,
                "thumb": query_person.get("thumb"),
                "similarity": 1.0,
                "camera": app["camera"],
                "start_time": app["start_time"],
                "end_time": app["end_time"],
            })

        results.sort(key=lambda x: x["similarity"], reverse=True)

        # 결과 표시
        THRESHOLD = 0.75
        matched = [r for r in results if r["similarity"] >= THRESHOLD]
        st.caption(f"유사도 {THRESHOLD:.2f} 이상 구간: **{len(matched)}개** 탐지")

        for r in matched:
            is_self = r["person_id"] == selected_id
            label = "✅ 본인 등장" if is_self else f"🔶 유사 인물 (Person #{r['person_id']:04d})"
            with st.container(border=True):
                rc1, rc2, rc3 = st.columns([1, 2, 3])
                with rc1:
                    img = load_image(r.get("thumb"))
                    if img:
                        st.image(img, width=THUMB_W)
                with rc2:
                    st.markdown(f"**{label}**")
                    st.metric("유사도", f"{r['similarity']:.3f}")
                with rc3:
                    st.markdown(f"**카메라**: CAM {r['camera']}")
                    st.markdown(f"**등장**: {r['start_time']} ~ {r['end_time']}")

# ── 하단 정보 ─────────────────────────────────────────────────────
st.divider()
st.caption(f"데이터: `{data_dir}/manifest.json`  |  총 {len(persons)}명 등록")
