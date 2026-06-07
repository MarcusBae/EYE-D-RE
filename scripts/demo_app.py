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
from datetime import datetime, timedelta
from pathlib import Path

import math

import numpy as np
import plotly.graph_objects as go
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


# ── 카메라 흐름 그래프 ────────────────────────────────────────────
def _build_camera_flow(persons: list) -> tuple[dict, dict]:
    """카메라 간 이동 엣지와 노드별 인물 집합을 반환."""
    flow: dict[tuple, list] = {}
    node_pids: dict = {}
    for p in persons:
        apps = sorted(p["appearances"], key=lambda a: a["start_time"])
        for a in apps:
            node_pids.setdefault(a["camera"], set()).add(p["id"])
        cams = [a["camera"] for a in apps]
        for i in range(len(cams) - 1):
            s, d = cams[i], cams[i + 1]
            if s != d:
                flow.setdefault((s, d), []).append(p["id"])
    return flow, node_pids


def render_camera_graph(
    persons: list,
    all_cams: list,
    highlight_pid: int | None = None,
) -> go.Figure:
    flow, node_pids = _build_camera_flow(persons)
    cams = sorted(all_cams)
    n = len(cams)
    R = 1.0
    NODE_R = 0.19

    angles = [2 * math.pi * i / max(n, 1) - math.pi / 2 for i in range(n)]
    pos = {c: (R * math.cos(a), R * math.sin(a)) for c, a in zip(cams, angles)}

    # 하이라이트 대상 추출
    hl_edges: set[tuple] = set()
    hl_cams: set = set()
    if highlight_pid is not None:
        hp = next((p for p in persons if p["id"] == highlight_pid), None)
        if hp:
            apps = sorted(hp["appearances"], key=lambda a: a["start_time"])
            seq = [a["camera"] for a in apps]
            hl_cams = set(seq)
            for i in range(len(seq) - 1):
                if seq[i] != seq[i + 1]:
                    hl_edges.add((seq[i], seq[i + 1]))

    dim = highlight_pid is not None
    fig = go.Figure()

    # 엣지 (방향 화살표)
    for (s, d), pids in flow.items():
        x0, y0 = pos[s]
        x1, y1 = pos[d]
        dx, dy = x1 - x0, y1 - y0
        dist = math.sqrt(dx ** 2 + dy ** 2) + 1e-9
        ax0 = x0 + dx / dist * NODE_R
        ay0 = y0 + dy / dist * NODE_R
        ax1 = x1 - dx / dist * NODE_R
        ay1 = y1 - dy / dist * NODE_R
        cnt = len(pids)
        is_hl = (s, d) in hl_edges

        if dim:
            arrow_color  = "rgba(249,115,22,0.92)" if is_hl else "rgba(96,165,250,0.18)"
            arrow_width  = max(2.5, cnt * 2.2)     if is_hl else max(1.0, cnt * 0.7)
            bg_color     = "rgba(124,45,18,0.92)"  if is_hl else "rgba(30,58,138,0.25)"
            border_color = "rgba(251,146,60,0.85)" if is_hl else "rgba(96,165,250,0.15)"
            font_color   = "white"                 if is_hl else "rgba(255,255,255,0.3)"
        else:
            arrow_color  = "rgba(96,165,250,0.85)"
            arrow_width  = max(1.8, cnt * 1.6)
            bg_color     = "rgba(30,58,138,0.85)"
            border_color = "rgba(96,165,250,0.5)"
            font_color   = "white"

        fig.add_trace(go.Scatter(
            x=[(ax0 + ax1) / 2], y=[(ay0 + ay1) / 2],
            mode="markers",
            marker=dict(size=14, color="rgba(0,0,0,0)"),
            hovertext=f"CAM {s} → CAM {d} : {cnt}명",
            hoverinfo="text",
            showlegend=False,
        ))
        fig.add_annotation(
            x=ax1, y=ay1, ax=ax0, ay=ay0,
            xref="x", yref="y", axref="x", ayref="y",
            text=f"<b>{cnt}</b>",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=arrow_width,
            arrowcolor=arrow_color,
            font=dict(size=11, color=font_color),
            bgcolor=bg_color,
            bordercolor=border_color,
            borderwidth=1,
            borderpad=3,
        )

    # 노드
    for cam in cams:
        x, y = pos[cam]
        cnt = len(node_pids.get(cam, set()))
        is_hl = cam in hl_cams

        if dim:
            node_color   = "#c2410c"               if is_hl else "#0f2550"
            border_color = "#fb923c"               if is_hl else "#1d4ed8"
            text_color   = "white"                 if is_hl else "rgba(255,255,255,0.35)"
            sub_color    = "#fdba74"               if is_hl else "rgba(156,163,175,0.35)"
            size         = 74                      if is_hl else 62
        else:
            node_color   = "#1d4ed8"
            border_color = "#93c5fd"
            text_color   = "white"
            sub_color    = "#9ca3af"
            size         = 68

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=size, color=node_color,
                        line=dict(width=2.5, color=border_color)),
            text=[f"CAM {cam}"],
            textposition="middle center",
            textfont=dict(color=text_color, size=13),
            hovertemplate=f"<b>CAM {cam}</b><br>{cnt}명 탐지<extra></extra>",
            showlegend=False,
        ))
        fig.add_annotation(
            x=x, y=y - NODE_R - 0.09,
            text=f"{cnt}명",
            showarrow=False,
            font=dict(size=11, color=sub_color),
            xref="x", yref="y",
        )

    margin = 0.5
    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-R - margin, R + margin]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-R - margin, R + margin], scaleanchor="x"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="#1f2937", font_color="white"),
    )
    return fig


# ── 카메라 흐름 애니메이션 ───────────────────────────────────────
def render_animated_flow(persons: list, all_cams: list) -> go.Figure:
    fmt = "%H:%M:%S"

    # 전체 시간 범위 수집
    all_dt: list[datetime] = []
    for p in persons:
        for app in p["appearances"]:
            try:
                all_dt.append(datetime.strptime(app["start_time"], fmt))
                all_dt.append(datetime.strptime(app["end_time"], fmt))
            except Exception:
                pass
    if not all_dt:
        fig = go.Figure()
        fig.add_annotation(text="타임스탬프 데이터 없음", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(size=14, color="#9ca3af"))
        return fig

    t_min, t_max = min(all_dt), max(all_dt)
    STEP = 60  # 1분 간격
    n_steps = max(2, int((t_max - t_min).total_seconds() / STEP) + 2)
    time_steps = [t_min + timedelta(seconds=i * STEP) for i in range(n_steps)]

    # 노드 위치
    cams = sorted(all_cams)
    R, NODE_R = 1.0, 0.19
    angles = [2 * math.pi * i / max(len(cams), 1) - math.pi / 2 for i in range(len(cams))]
    pos = {c: (R * math.cos(a), R * math.sin(a)) for c, a in zip(cams, angles)}

    flow, _ = _build_camera_flow(persons)
    edges = list(flow.keys())
    E = len(edges)

    # 타임스텝별 카메라 점유 인원
    def occ_at(t: datetime) -> dict:
        occ = {c: 0 for c in cams}
        for p in persons:
            for app in p["appearances"]:
                try:
                    s = datetime.strptime(app["start_time"], fmt)
                    e = datetime.strptime(app["end_time"], fmt)
                    if s <= t <= e:
                        occ[app["camera"]] += 1
                except Exception:
                    pass
        return occ

    # 이동 중인 엣지 (end_time ~ next start_time ±2분 윈도우)
    TRANSIT = timedelta(seconds=120)

    def active_at(t: datetime) -> set:
        active: set = set()
        for p in persons:
            apps_s = sorted(p["appearances"], key=lambda a: a["start_time"])
            for i in range(len(apps_s) - 1):
                a1, a2 = apps_s[i], apps_s[i + 1]
                if a1["camera"] == a2["camera"]:
                    continue
                try:
                    e1 = datetime.strptime(a1["end_time"], fmt)
                    s2 = datetime.strptime(a2["start_time"], fmt)
                    if e1 - TRANSIT <= t <= s2 + TRANSIT:
                        active.add((a1["camera"], a2["camera"]))
                except Exception:
                    pass
        return active

    all_occs   = [occ_at(t)    for t in time_steps]
    all_active = [active_at(t) for t in time_steps]
    max_occ = max((max(occ.values(), default=0) for occ in all_occs), default=1) or 1

    # ── Trace 빌더 ────────────────────────────────────────────────
    # 구조: [0..E-1] 엣지 라인, [E] 노드 원, [E+1] 노드 카운트 라벨
    def _edge(src: int, dst: int, is_active: bool) -> go.Scatter:
        x0, y0 = pos[src]; x1, y1 = pos[dst]
        dx, dy = x1 - x0, y1 - y0
        d = math.sqrt(dx**2 + dy**2) + 1e-9
        ax0, ay0 = x0 + dx/d*NODE_R, y0 + dy/d*NODE_R
        ax1, ay1 = x1 - dx/d*NODE_R, y1 - dy/d*NODE_R
        return go.Scatter(
            x=[ax0, ax1, None], y=[ay0, ay1, None],
            mode="lines",
            line=dict(
                color="rgba(249,115,22,0.92)" if is_active else "rgba(96,165,250,0.18)",
                width=4.5 if is_active else 1.2,
            ),
            showlegend=False, hoverinfo="skip",
        )

    def _nodes(occ: dict) -> go.Scatter:
        v = [min(occ.get(c, 0) / max_occ, 1.0) for c in cams]
        return go.Scatter(
            x=[pos[c][0] for c in cams],
            y=[pos[c][1] for c in cams],
            mode="markers+text",
            marker=dict(
                size=[max(44, 36 + occ.get(c, 0) * 9) for c in cams],
                color=[f"rgb({int(29+vi*196)},{int(78+vi*48)},{int(216-vi*116)})"
                       for vi in v],
                line=dict(width=2.5, color="#93c5fd"),
            ),
            text=[f"CAM {c}" for c in cams],
            textposition="middle center",
            textfont=dict(color="white", size=12),
            hovertext=[f"CAM {c}: {occ.get(c, 0)}명" for c in cams],
            hoverinfo="text",
            showlegend=False,
        )

    def _labels(occ: dict) -> go.Scatter:
        return go.Scatter(
            x=[pos[c][0] for c in cams],
            y=[pos[c][1] - NODE_R - 0.10 for c in cams],
            mode="text",
            text=[f"{occ.get(c, 0)}명" for c in cams],
            textfont=dict(size=11, color="#9ca3af"),
            showlegend=False, hoverinfo="skip",
        )

    def _frame_data(occ: dict, active: set) -> list:
        return [_edge(s, d, (s, d) in active) for s, d in edges] + [_nodes(occ), _labels(occ)]

    # ── 초기 데이터 & 프레임 ─────────────────────────────────────
    init_data = _frame_data(all_occs[0], all_active[0])
    frames = [
        go.Frame(
            data=_frame_data(all_occs[i], all_active[i]),
            traces=list(range(E + 2)),
            name=time_steps[i].strftime("%H:%M"),
        )
        for i in range(n_steps)
    ]

    # 정적 방향 화살표 (layout annotations — 색 고정, 방향 표시용)
    dir_annots = []
    for s, d in edges:
        x0, y0 = pos[s]; x1, y1 = pos[d]
        dx, dy = x1 - x0, y1 - y0
        dist = math.sqrt(dx**2 + dy**2) + 1e-9
        dir_annots.append(dict(
            x=x1 - dx/dist*NODE_R, y=y1 - dy/dist*NODE_R,
            ax=(x0+x1)/2, ay=(y0+y1)/2,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2,
            arrowwidth=1.5, arrowcolor="rgba(96,165,250,0.28)", text="",
        ))

    # ── 슬라이더 & 재생 버튼 ─────────────────────────────────────
    slider = {
        "active": 0,
        "steps": [
            {"args": [[f.name], {"frame": {"duration": 300, "redraw": True},
                                  "mode": "immediate",
                                  "transition": {"duration": 150}}],
             "label": f.name, "method": "animate"}
            for f in frames
        ],
        "x": 0.05, "len": 0.9, "y": -0.04,
        "currentvalue": {"prefix": "시각: ", "visible": True, "xanchor": "center",
                         "font": {"size": 13, "color": "#e5e7eb"}},
        "transition": {"duration": 150},
        "pad": {"b": 10},
    }
    updatemenus = [{
        "type": "buttons", "showactive": False,
        "x": 0.0, "y": -0.15, "xanchor": "left",
        "buttons": [
            {"label": "▶ 재생", "method": "animate",
             "args": [None, {"frame": {"duration": 500, "redraw": True},
                             "fromcurrent": True,
                             "transition": {"duration": 200}}]},
            {"label": "⏸ 정지", "method": "animate",
             "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]},
        ],
    }]

    margin = 0.5
    return go.Figure(
        data=init_data,
        frames=frames,
        layout=go.Layout(
            height=530,
            margin=dict(l=10, r=10, t=20, b=100),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-R-margin, R+margin]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-R-margin, R+margin], scaleanchor="x"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            sliders=[slider],
            updatemenus=updatemenus,
            annotations=dir_annots,
            hoverlabel=dict(bgcolor="#1f2937", font_color="white"),
        ),
    )


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

    view_mode = st.radio(
        "보기 모드",
        ["📋 테이블 보기", "🗺️ 그래프 보기", "▶ 애니메이션"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if view_mode == "🗺️ 그래프 보기":
        st.subheader("카메라 이동 흐름")

        # 인물 하이라이트 선택
        sorted_persons = sorted(
            persons,
            key=lambda p: (-(len({a["camera"] for a in p["appearances"]}) > 1), p["id"]),
        )
        options: dict[str, int | None] = {"— 전체 보기 —": None}
        for p in sorted_persons:
            apps = sorted(p["appearances"], key=lambda a: a["start_time"])
            seq = [a["camera"] for a in apps]
            deduped: list = [seq[0]]
            for c in seq[1:]:
                if c != deduped[-1]:
                    deduped.append(c)
            icon = "🔴" if len(set(seq)) > 1 else "⚪"
            label = f"{icon} Person #{p['id']:04d}  ( {' → '.join(f'CAM {c}' for c in deduped)} )"
            options[label] = p["id"]

        sel_label = st.selectbox("인물 동선 하이라이트", list(options.keys()), index=0)
        highlight_pid = options[sel_label]

        if highlight_pid is not None:
            hp = next(p for p in persons if p["id"] == highlight_pid)
            apps = sorted(hp["appearances"], key=lambda a: a["start_time"])
            path_str = " → ".join(
                f"CAM {a['camera']} ({a['start_time']})" for a in apps
            )
            st.info(f"**Person #{highlight_pid:04d}** 이동 경로: {path_str}")

        st.caption(
            "노드 = 카메라 · 화살표 = 이동 방향 · 화살표 위 숫자 = 이동 인원 수"
            + ("  |  🟠 = 선택 인물 동선" if highlight_pid is not None else "")
        )
        st.plotly_chart(
            render_camera_graph(persons, all_cams, highlight_pid=highlight_pid),
            use_container_width=True,
        )
        st.divider()

    elif view_mode == "▶ 애니메이션":
        st.subheader("카메라 인원 흐름 애니메이션")
        st.caption(
            "노드 크기·색 = 해당 시각 인원 수 (클수록 많음) · "
            "주황 엣지 = 이동 발생 구간 (±2분 윈도우) · "
            "회색 화살표 = 이동 방향"
        )
        st.plotly_chart(render_animated_flow(persons, all_cams), use_container_width=True)
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
