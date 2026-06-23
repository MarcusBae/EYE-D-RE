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
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import math

import cv2
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
    config_path_sb = st.text_input("Config 경로", value="configs/config.yaml")
    st.divider()
    scenario = st.radio(
        "시나리오 선택",
        ["A — 동선 데이터 분석", "B — 동선 확인", "C — 관심 인물 탐색"],
        index=0,
    )

# ── 데이터 로드 ───────────────────────────────────────────────────
# 사용자 레이블(A/B/C) → 내부 코드(pipeline/movement/person) 매핑
# A=동선 데이터 분석(파이프라인), B=동선 확인, C=관심 인물 탐색
_s = {"A": "pipeline", "B": "movement", "C": "person"}.get(scenario[0], "pipeline")

manifest = load_manifest(data_dir)

if _s != "pipeline":
    if not manifest:
        st.warning(f"**{data_dir}/manifest.json** 을 찾을 수 없습니다.")
        st.code("python scripts/demo_preprocess.py --config configs/config.yaml")
        st.stop()

    persons = manifest.get("persons", [])
    if not persons:
        st.error("등록된 인물이 없습니다.")
        st.stop()
else:
    persons = manifest.get("persons", []) if manifest else []

# ── 시나리오 B: 동선 확인 ─────────────────────────────────────────
if _s == "movement":
    st.header("📊 시나리오 B — 동선 확인")
    st.caption("썸네일을 클릭해 인물을 선택하면 해당 인물의 카메라별 이동 경로를 확인합니다.")
    st.divider()

    # ① 인물 갤러리 ───────────────────────────────────────────────
    st.subheader("① 인물 선택")

    if "b_selected_pids" not in st.session_state:
        st.session_state["b_selected_pids"] = set()
    _b_sel: set = st.session_state["b_selected_pids"]

    _ba, _bb, _bc = st.columns([1, 1, 6])
    with _ba:
        if st.button("전체 선택", use_container_width=True):
            st.session_state["b_selected_pids"] = {p["id"] for p in persons}
            st.rerun()
    with _bb:
        if st.button("전체 해제", use_container_width=True, disabled=not _b_sel):
            st.session_state["b_selected_pids"] = set()
            st.rerun()

    _n_cols_b = 8
    _cols_b = st.columns(_n_cols_b)
    for _bi, _bp in enumerate(persons):
        _bpid = _bp["id"]
        _bsel = _bpid in _b_sel
        with _cols_b[_bi % _n_cols_b]:
            _bimg = load_image(_bp.get("thumb"))
            if _bimg:
                st.image(_bimg, width=THUMB_W)
            if st.button(
                f"#{_bpid:04d}",
                key=f"b_btn_{_bpid}",
                type="primary" if _bsel else "secondary",
                use_container_width=True,
            ):
                if _bsel:
                    st.session_state["b_selected_pids"].discard(_bpid)
                else:
                    st.session_state["b_selected_pids"].add(_bpid)
                st.rerun()
            if _bsel:
                if st.button("🗑", key=f"b_del_{_bpid}", use_container_width=True,
                             help="이 인물을 데이터에서 삭제"):
                    st.session_state["b_del_confirm"] = _bpid
                    st.rerun()

    # 삭제 확인 대화 ─────────────────────────────────────────────
    if "b_del_confirm" in st.session_state:
        _dpid = st.session_state["b_del_confirm"]
        st.warning(
            f"⚠️ **Person #{_dpid:04d}** 를 삭제하시겠습니까?  "
            f"manifest에서 제거되고 crops 폴더도 삭제됩니다."
        )
        _dc1, _dc2, _ = st.columns([1, 1, 6])
        with _dc1:
            if st.button("삭제 확인", type="primary", use_container_width=True):
                _manifest_path = Path(data_dir) / "manifest.json"
                _mdata = json.loads(_manifest_path.read_text(encoding="utf-8"))
                _mdata["persons"] = [p for p in _mdata["persons"] if p["id"] != _dpid]
                _mdata["num_persons"] = len(_mdata["persons"])
                _manifest_path.write_text(
                    json.dumps(_mdata, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                _crops_del = Path(data_dir) / "crops" / f"{_dpid:04d}"
                if _crops_del.exists():
                    shutil.rmtree(str(_crops_del))
                st.session_state["b_selected_pids"].discard(_dpid)
                del st.session_state["b_del_confirm"]
                load_manifest.clear()
                st.rerun()
        with _dc2:
            if st.button("취소", use_container_width=True):
                del st.session_state["b_del_confirm"]
                st.rerun()

    st.divider()

    # ② 동선 보기 ─────────────────────────────────────────────────
    if not _b_sel:
        st.info("위에서 인물 썸네일을 클릭해 선택하면 동선이 표시됩니다.")
        st.stop()

    _sel_persons = [p for p in persons if p["id"] in _b_sel]
    _all_cams_b  = sorted({app["camera"] for p in persons for app in p["appearances"]})
    _multi_cam_b = [p for p in _sel_persons if len({a["camera"] for a in p["appearances"]}) > 1]

    _bc1, _bc2, _bc3 = st.columns(3)
    _bc1.metric("선택 인물", len(_sel_persons))
    _bc2.metric("다중 카메라 등장", len(_multi_cam_b))
    _bc3.metric("전체 카메라 수", len(_all_cams_b))
    st.divider()

    view_mode = st.radio(
        "보기 모드",
        ["📋 테이블 보기", "🗺️ 그래프 보기", "▶ 애니메이션"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if view_mode == "🗺️ 그래프 보기":
        st.subheader("카메라 이동 흐름")
        st.caption("노드 = 카메라 · 화살표 = 이동 방향 · 숫자 = 선택 인물 이동 수")
        st.plotly_chart(
            render_camera_graph(_sel_persons, _all_cams_b),
            use_container_width=True,
        )
        st.divider()

    elif view_mode == "▶ 애니메이션":
        st.subheader("카메라 인원 흐름 애니메이션")
        st.caption(
            "노드 크기·색 = 해당 시각 인원 수 · "
            "주황 엣지 = 이동 발생 구간 (±2분 윈도우) · "
            "회색 화살표 = 이동 방향"
        )
        st.plotly_chart(render_animated_flow(_sel_persons, _all_cams_b), use_container_width=True)
        st.divider()

    # 선택 인물별 동선 카드
    for person in _sel_persons:
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

# ── 시나리오 C: 관심 인물 탐색 ───────────────────────────────────
elif _s == "person":
    st.header("🔍 시나리오 C — 관심 인물 탐색")
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

# ── 시나리오 A: 동선 데이터 분석 ─────────────────────────────────
elif _s == "pipeline":
    st.header("⚙️ 시나리오 A — 동선 데이터 분석")
    st.caption(
        "영상 파일에서 추적 → 품질 필터 → Re-ID 특징 추출 → HAC 클러스터링을 실행해 "
        "demo_data에 인물 정보를 추가합니다."
    )
    st.divider()

    VIDEO_EXTS_C = {".mp4", ".avi", ".mov", ".mkv"}
    ROOT_C = Path(__file__).resolve().parent.parent

    # ① 영상 파일 선택 ──────────────────────────────────────────────
    st.subheader("① 영상 파일 선택")

    # 기본 비디오 폴더 입력 받기
    video_dir_input = st.text_input("영상이 있는 폴더 경로", value="data/raw_videos", key="c_video_dir_input")
    
    # 해당 폴더 내의 비디오 파일들 탐색
    video_files = []
    p = Path(video_dir_input)
    if p.exists() and p.is_dir():
        for ext in VIDEO_EXTS_C:
            video_files.extend(sorted(p.glob(f"*{ext}")))
    
    video_file_names = [f.name for f in video_files]
    
    # 기존 선택된 파일 목록 로드
    current_selected = st.session_state.get("c_selected_files", [])
    
    # multiselect UI
    if video_file_names:
        # 기존 선택된 파일 중 현재 폴더에 속하는 파일들의 이름 추출
        prev_selected_names = [Path(f).name for f in current_selected if Path(f).parent == p.resolve() or Path(f).parent == p]
        
        selected_names = st.multiselect(
            "선택할 영상 파일들 (다중 선택 가능)",
            options=video_file_names,
            default=prev_selected_names,
            help="폴더 내에서 분석할 영상을 선택하세요."
        )
        
        # 선택된 파일들 전체 경로 리스트 구축
        new_selected = []
        # 1. multiselect에서 선택한 파일들 경로 추가
        for name in selected_names:
            new_selected.append(str((p / name).resolve()))
        # 2. multiselect에 표시되지 않는 파일들(다른 폴더 경로의 파일들)은 그대로 유지
        for f in current_selected:
            if Path(f).parent != p.resolve() and Path(f).parent != p:
                new_selected.append(f)
                
        st.session_state["c_selected_files"] = new_selected
    else:
        st.warning(f"'{video_dir_input}' 폴더에 지원하는 영상 파일({', '.join(VIDEO_EXTS_C)})이 존재하지 않습니다.")

    # 추가 파일 입력 및 전체 지우기 행
    add_col1, add_col2, clr_col = st.columns([4, 2, 2])
    with add_col1:
        custom_file_input = st.text_input("개별 파일 경로 추가 (필요 시)", placeholder="예: /absolute/path/to/video.mp4")
    with add_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True) # 버튼 위치 맞춤
        if st.button("➕ 추가", use_container_width=True, disabled=not custom_file_input):
            custom_path = Path(custom_file_input)
            if custom_path.exists() and custom_path.is_file():
                prev = st.session_state.get("c_selected_files", [])
                full_path_str = str(custom_path.resolve())
                if full_path_str not in prev:
                    st.session_state["c_selected_files"] = prev + [full_path_str]
                    st.success(f"추가 완료: {custom_path.name}")
                    st.rerun()
            else:
                st.error("올바르지 않거나 존재하지 않는 파일 경로입니다.")
    with clr_col:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True) # 버튼 위치 맞춤
        if st.button("🗑 전체 지우기", use_container_width=True,
                     disabled=not st.session_state.get("c_selected_files")):
            st.session_state["c_selected_files"] = []
            st.rerun()

    # 선택된 파일 목록 시각화
    selected_c: list[str] = st.session_state.get("c_selected_files", [])
    if selected_c:
        st.caption("선택된 파일 목록:")
        with st.container(border=True):
            for _fi, _fp in enumerate(selected_c):
                _rc1, _rc2 = st.columns([8, 1])
                with _rc1:
                    st.caption(f"**{Path(_fp).name}**  `{_fp}`")
                with _rc2:
                    if st.button("✕", key=f"c_rm_{_fi}"):
                        st.session_state["c_selected_files"].pop(_fi)
                        st.rerun()
    else:
        st.caption("선택된 영상 파일이 없습니다.")

    # ② 파라미터 ────────────────────────────────────────────────────
    st.subheader("② 파라미터")
    pc1, pc2 = st.columns(2)
    with pc1:
        c_stride = st.slider("프레임 스트라이드", 1, 12, 2,
                              help="N 프레임마다 1회 탐지. ID switch 줄이려면 1~2 권장, 속도 우선이면 6")
        _thresh_default = 0.30
        try:
            import yaml as _yaml_thresh
            _cfg_thresh = _yaml_thresh.safe_load(Path(config_path_sb).read_text(encoding="utf-8"))
            _thresh_default = float(_cfg_thresh.get("reid", {}).get("clustering", {}).get("threshold", 0.30))
        except Exception:
            pass
        c_thresh = st.slider("HAC 클러스터링 임계값", 0.10, 0.50, _thresh_default, 0.05,
                              help="낮을수록 보수적 (다른 사람으로 처리). config.yaml reid.clustering.threshold")
    with pc2:
        c_append = st.checkbox(
            "기존 demo_data에 추가 (덮어쓰지 않음)", value=True,
            help="체크 시 기존 인물 데이터를 보존하고 새 인물을 추가합니다."
        )

    with st.expander("🔧 품질 필터 파라미터 조정", expanded=False):
        st.caption("트랙렛이 0개 통과될 때는 아래 값을 낮추거나 0으로 설정하세요.")
        qf_c1, qf_c2, qf_c3 = st.columns(3)
        with qf_c1:
            qf_min_length = st.slider("최소 트랙 길이 (프레임)", 0, 30, 6,
                                      help="이 프레임 수 미만 트랙렛 제거")
            qf_min_conf   = st.slider("최소 평균 신뢰도", 0.0, 1.0, 0.7, 0.05,
                                      help="평균 탐지 신뢰도가 이 값 미만이면 제거")
        with qf_c2:
            qf_min_h    = st.slider("최소 bbox 높이 (px)", 0, 256, 64)
            qf_min_w    = st.slider("최소 bbox 너비 (px)", 0, 128, 32)
        with qf_c3:
            qf_max_ratio = st.slider("최대 종횡비 (H/W)", 1.0, 12.0, 4.0, 0.5,
                                     help="이 값 초과 트랙렛 제거 (너무 좁은 bbox)")
            qf_min_area  = st.slider("최소 bbox 면적 (px²)", 0, 8192, 2048, 128)

    # ③ 실행 버튼 ───────────────────────────────────────────────────
    if selected_c:
        st.info(
            f"선택된 영상 **{len(selected_c)}개**: "
            + ", ".join(Path(v).name for v in selected_c[:5])
            + (f" 외 {len(selected_c)-5}개" if len(selected_c) > 5 else "")
        )
    run_c = st.button("▶ 파이프라인 실행", type="primary", disabled=not selected_c)

    # ④ 파이프라인 실행 ─────────────────────────────────────────────
    if run_c and selected_c:
        # sys.path 준비
        if str(ROOT_C) not in sys.path:
            sys.path.insert(0, str(ROOT_C))
        scripts_dir_c = ROOT_C / "scripts"
        if str(scripts_dir_c) not in sys.path:
            sys.path.insert(0, str(scripts_dir_c))

        import yaml as _yaml

        cfg_path_c = Path(config_path_sb)
        if not cfg_path_c.exists():
            st.error(f"Config 파일을 찾을 수 없습니다: {cfg_path_c}")
            st.stop()

        with open(cfg_path_c, encoding="utf-8") as _f:
            cfg_c = _yaml.safe_load(_f)

        try:
            from pipeline.tracker import PersonTracker as _PersonTracker
            from pipeline.reid_merger import OSNetExtractor as _OSNetExtractor
            from pipeline.tracklet_io import list_tracklets as _list_tracklets
            from pipeline.quality_filter import TrackletQualityFilter as _TQF
            from demo_preprocess import (
                get_video_cam_slot as _get_cam_slot,
                build_cam_slot_start as _build_css,
                run_matching as _run_matching,
                save_demo_data as _save_demo_data,
            )
        except ImportError as _ie:
            st.error(f"모듈 임포트 실패: {_ie}")
            st.stop()

        out_dir_c = Path(data_dir)
        out_dir_c.mkdir(parents=True, exist_ok=True)

        # 영상 유효성 검사
        invalid_c = [v for v in selected_c if _get_cam_slot(Path(v), cfg_c) is None]
        for _v in invalid_c:
            st.warning(f"⚠️ {Path(_v).name} — config.yaml videos 섹션에 없어 건너뜁니다.")
        valid_c = [v for v in selected_c if v not in invalid_c]
        if not valid_c:
            st.error("처리할 영상이 없습니다. config.yaml의 videos 섹션을 확인하세요.")
            st.stop()

        cam_slot_start_c = _build_css(cfg_c)
        reid_cfg_c = cfg_c.get("reid", {})
        det_cfg_c  = cfg_c.get("detection", {})
        trk_cfg_c  = cfg_c.get("tracking", {})
        qf_cfg_c   = cfg_c.get("quality_filter", {})
        n_vids = len(valid_c)

        overall_bar = st.progress(0.0, text="준비 중...")

        try:
            # 기존 manifest 백업 (복구 안전망)
            _manifest_src = out_dir_c / "manifest.json"
            _manifest_bak = out_dir_c / "manifest.json.bak"
            if _manifest_src.exists():
                shutil.copy2(_manifest_src, _manifest_bak)

            tmp_dir_c = Path(tempfile.mkdtemp())
            tmp_track_c  = tmp_dir_c / "tracklets"
            tmp_filter_c = tmp_dir_c / "filtered"
            tmp_track_c.mkdir(); tmp_filter_c.mkdir()

            # ── Step 1: 모델 로드 ─────────────────────────────────
            with st.status("⚙️ Step 1: 모델 로드 중...", expanded=False) as _s1:
                _extractor_c = _OSNetExtractor(
                    model_name=reid_cfg_c.get("model_name", "osnet_ain_x1_0"),
                    pretrained=reid_cfg_c.get("pretrained", True),
                    weights_path=reid_cfg_c.get("weights_path"),
                    device=reid_cfg_c.get("device", "auto"),
                )
                _tracker_c = _PersonTracker(
                    model_path=det_cfg_c.get("model", "yolov8n.pt"),
                    tracker_yaml=trk_cfg_c.get("tracker_yaml", "configs/botsort.yaml"),
                    conf=det_cfg_c.get("conf_threshold", 0.5),
                    iou=det_cfg_c.get("iou_threshold", 0.45),
                    imgsz=det_cfg_c.get("imgsz", 640),
                    classes=det_cfg_c.get("classes", [0]),
                    device=det_cfg_c.get("device", "auto"),
                )
                _s1.update(label="✅ Step 1: 모델 로드 완료", state="complete")
            overall_bar.progress(0.08, text="Step 1 완료: 모델 로드")

            # ── Step 2: 트래킹 ────────────────────────────────────
            # 전체 프레임 수 사전 집계 (progress % 계산용)
            _frames_per_vid: list[int] = []
            for _vp_str in valid_c:
                _cap_tmp = cv2.VideoCapture(_vp_str)
                _frames_per_vid.append(int(_cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT)))
                _cap_tmp.release()
            _total_all_frames = max(sum(_frames_per_vid), 1)

            with st.status(f"🎬 Step 2: 트래킹 (0/{n_vids})", expanded=True) as _s2:
                _track_bar = st.progress(0.0, text="0.0%  (0 / 0 프레임)")
                _frames_done_before = 0

                for _i, _vp_str in enumerate(valid_c):
                    _vp = Path(_vp_str)
                    _vid_frames = _frames_per_vid[_i]
                    st.write(f"처리 중: **{_vp.name}** ({_vid_frames:,} 프레임)")
                    _cam_c, _slot_c = _get_cam_slot(_vp, cfg_c)

                    _done_snap = _frames_done_before  # capture for closure

                    def _make_cb(_done_before, _total_all, _bar):
                        def _cb(_cur, _tot):
                            _done = _done_before + _cur
                            _pct = _done / _total_all
                            _bar.progress(
                                min(_pct, 1.0),
                                text=f"{_pct*100:.1f}%  ({_done:,} / {_total_all:,} 프레임)",
                            )
                        return _cb

                    _tracker_c.track_video(
                        video_path=str(_vp),
                        camera_id=_cam_c,
                        time_slot=_slot_c,
                        output_dir=str(tmp_track_c),
                        frame_stride=c_stride,
                        save_crops=True,
                        progress_callback=_make_cb(_done_snap, _total_all_frames, _track_bar),
                    )
                    _frames_done_before += _vid_frames
                    _s2.update(label=f"🎬 Step 2: 트래킹 ({_i+1}/{n_vids})")

                _track_bar.progress(1.0, text=f"100.0%  ({_total_all_frames:,} / {_total_all_frames:,} 프레임)")
                _s2.update(label=f"✅ Step 2: 트래킹 완료 ({n_vids}개)", state="complete")
            overall_bar.progress(0.40, text="Step 2 완료: 트래킹")

            # ── Step 3: 품질 필터 ─────────────────────────────────
            _filt_c = _TQF(
                min_length=qf_min_length,
                min_avg_conf=qf_min_conf,
                min_bbox_h=qf_min_h,
                min_bbox_w=qf_min_w,
                max_aspect_ratio=qf_max_ratio,
                min_area=qf_min_area,
            )
            with st.status("🔍 Step 3: 품질 필터링...", expanded=True) as _s3:
                _passed_total = 0
                _slot_dirs = sorted(d for d in tmp_track_c.iterdir() if d.is_dir())
                for _sd in _slot_dirs:
                    _fout = tmp_filter_c / _sd.name
                    _fout.mkdir(parents=True, exist_ok=True)
                    _stats = _filt_c.filter_all(
                        tracklet_dir=str(_sd),
                        output_dir=str(_fout),
                        copy_crops=True,
                        verbose=False,
                    )
                    _p = _stats.get("passed", 0) if isinstance(_stats, dict) else 0
                    _t = _stats.get("total",  0) if isinstance(_stats, dict) else 0
                    _passed_total += _p
                    st.write(f"  {_sd.name}: {_p}/{_t}개 통과")
                _s3.update(
                    label=f"✅ Step 3: 품질 필터 완료 — {_passed_total}개 통과",
                    state="complete",
                )
            overall_bar.progress(0.55, text="Step 3 완료: 품질 필터")

            # ── Step 4: 특징 추출 ─────────────────────────────────
            _tracklets_c = _list_tracklets(str(tmp_filter_c))
            if not _tracklets_c:
                st.error(
                    "품질 필터를 통과한 트랙렛이 없습니다. "
                    "아래 '🔧 품질 필터 파라미터 조정'에서 값을 낮추고 다시 실행하세요."
                )
                raise RuntimeError("no tracklets")

            _n_t = len(_tracklets_c)
            with st.status(f"🧠 Step 4: Re-ID 특징 추출 (0/{_n_t})", expanded=True) as _s4:
                _feat_bar = st.progress(0.0)
                _feats_c: list[np.ndarray] = []
                for _j, _t in enumerate(_tracklets_c):
                    _tdir = Path(_t["tracklet_dir"])
                    _crops = _t.get("crop_files", [])
                    if len(_crops) > 8:
                        _idx = np.linspace(0, len(_crops) - 1, 8, dtype=int)
                        _crops = [_crops[k] for k in _idx]
                    _imgs = [cv2.imread(str(_tdir / c)) for c in _crops]
                    _imgs = [im for im in _imgs if im is not None]
                    if _imgs:
                        _f = _extractor_c.extract_batch_features(_imgs)
                        _feats_c.append(_f.mean(axis=0))
                    else:
                        _feats_c.append(np.zeros(512, dtype=np.float32))
                    _feat_bar.progress((_j + 1) / _n_t)
                    _s4.update(label=f"🧠 Step 4: Re-ID 특징 추출 ({_j+1}/{_n_t})")
                _feats_arr = np.array(_feats_c)
                _s4.update(
                    label=f"✅ Step 4: 특징 추출 완료 ({_n_t}개 트랙렛)",
                    state="complete",
                )
            overall_bar.progress(0.75, text="Step 4 완료: 특징 추출")

            # ── Step 5: HAC 클러스터링 ────────────────────────────
            with st.status("🔗 Step 5: 인물 ID 배정 (HAC 클러스터링)...", expanded=False) as _s5:
                _pids_c = _run_matching(_tracklets_c, _feats_arr, c_thresh, config=cfg_c)
                _n_new_persons = len(set(_pids_c))
                _s5.update(
                    label=f"✅ Step 5: 클러스터링 완료 — {_n_new_persons}명 식별",
                    state="complete",
                )
            overall_bar.progress(0.88, text="Step 5 완료: 클러스터링")

            # ── Step 6: 저장 ──────────────────────────────────────
            with st.status("💾 Step 6: demo_data 저장...", expanded=False) as _s6:
                # 기존 인물 로드 (append 모드): 백업에서 읽어 경로 문제 회피
                _existing_persons: list = []
                _id_offset = 0
                if c_append:
                    _src = _manifest_bak if _manifest_bak.exists() else _manifest_src
                    if _src.exists():
                        _existing_m = json.loads(_src.read_text(encoding="utf-8"))
                        _existing_persons = _existing_m.get("persons", [])
                        _id_offset = max((p["id"] for p in _existing_persons), default=0)

                _pids_offset = [pid + _id_offset for pid in _pids_c]
                _new_persons = _save_demo_data(
                    _tracklets_c, _pids_offset, _feats_arr,
                    out_dir_c, str(tmp_filter_c), cam_slot_start_c,
                )

                # append 모드면 항상 머지 (existing이 비어있어도 안전하게 처리)
                if c_append and _existing_persons:
                    _merged = _existing_persons + _new_persons
                    _merged_manifest = {
                        "tracklet_root": str(out_dir_c),
                        "num_persons": len(_merged),
                        "persons": _merged,
                    }
                    (out_dir_c / "manifest.json").write_text(
                        json.dumps(_merged_manifest, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    _total = len(_merged)
                else:
                    _total = len(_new_persons)

                _s6.update(
                    label=(
                        f"✅ Step 6: 저장 완료 "
                        f"— 신규 {len(_new_persons)}명 / 전체 {_total}명"
                    ),
                    state="complete",
                )
            overall_bar.progress(1.0, text="완료!")

            st.success(
                f"파이프라인 완료! "
                f"신규 **{len(_new_persons)}명** 추가 (전체 **{_total}명** 등록)"
            )
            st.balloons()
            load_manifest.clear()

            # ── 신규 인물 썸네일 갤러리 ───────────────────────────
            if _new_persons:
                st.subheader(f"🖼️ 신규 생성 인물 썸네일 ({len(_new_persons)}명)")
                _THUMB_COLS = 8
                _tcols = st.columns(_THUMB_COLS)
                for _ti, _tp in enumerate(_new_persons):
                    with _tcols[_ti % _THUMB_COLS]:
                        _timg = load_image(_tp.get("thumb"))
                        if _timg:
                            st.image(_timg, width=THUMB_W)
                        else:
                            st.markdown(
                                "<div style='width:100px;height:120px;background:#1f2937;"
                                "border:1px solid #374151;border-radius:4px;display:flex;"
                                "align-items:center;justify-content:center;"
                                "color:#6b7280;font-size:11px'>No img</div>",
                                unsafe_allow_html=True,
                            )
                        _n_apps = len(_tp.get("appearances", []))
                        _cams_t = sorted({a["camera"] for a in _tp.get("appearances", [])})
                        st.caption(
                            f"**#{_tp['id']:04d}**  "
                            f"{_n_apps}구간 · CAM {','.join(str(c) for c in _cams_t)}"
                        )

        except RuntimeError:
            pass  # already shown st.error above
        except Exception as _exc:
            st.error(f"파이프라인 실패: {_exc}")
            st.code(traceback.format_exc())
        finally:
            if "tmp_dir_c" in dir():
                shutil.rmtree(str(tmp_dir_c), ignore_errors=True)

# ── 하단 정보 ─────────────────────────────────────────────────────
st.divider()
st.caption(f"데이터: `{data_dir}/manifest.json`  |  총 {len(persons)}명 등록")
