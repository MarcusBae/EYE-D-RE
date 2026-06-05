"""
EYE-D — PT용 발표 웹페이지 (Streamlit)
"""
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

DEFAULT_DATA_DIR = os.environ.get("EYE_D_DATA", "/mnt/d/EYE-D/data/curated")

st.set_page_config(page_title="EYE-D 발표", layout="wide")

st.markdown("""<style>
[data-testid="stHeader"]{background:transparent}
.block-container{padding-top:3.6rem;max-width:1150px}
.eyed-hero{font-size:2.4rem;font-weight:800;line-height:1.2;margin:.6rem 0 .2rem}
.eyed-sub{color:#667;font-size:1.12rem;margin-bottom:1rem}
.eyed-card{background:#f6f8fb;border:1px solid #e6eaf2;border-radius:14px;padding:16px 20px;margin:8px 0;height:100%}
.eyed-card h4{margin:0 0 6px 0;color:#2a4d8f;font-size:1.05rem}
.eyed-card p{margin:0;color:#333}
.eyed-step{display:inline-block;background:#2a4d8f;color:#fff;border-radius:12px;padding:14px 14px;font-weight:700;text-align:center;min-width:120px;font-size:.95rem}
.eyed-arrow{color:#2a4d8f;font-size:1.5rem;margin:0 4px}
.eyed-tag{display:inline-block;background:#eaf0fb;color:#2a4d8f;border-radius:20px;padding:5px 14px;margin:4px;font-size:.95rem;font-weight:600}
.eyed-jcard{border-left:5px solid #d9534f;background:#fbf7f7;border-radius:10px;padding:12px 18px;margin:10px 0}
.eyed-jtitle{font-weight:800;color:#b53;font-size:1.05rem;margin-bottom:4px}
.eyed-jcard div{margin:2px 0}
.eyed-jlesson{color:#2a7d4f;font-weight:600;margin-top:4px}
</style>""", unsafe_allow_html=True)


def card(title, body):
    st.markdown(f'<div class="eyed-card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def journey(title, problem, fix, lesson):
    st.markdown(
        f'<div class="eyed-jcard"><div class="eyed-jtitle">{title}</div>'
        f'<div>⚠️ <b>난관</b> · {problem}</div>'
        f'<div>🛠️ <b>대응</b> · {fix}</div>'
        f'<div class="eyed-jlesson">💡 교훈 · {lesson}</div></div>',
        unsafe_allow_html=True)


@st.cache_data(show_spinner=True)
def scan_tracklets(data_dir: str):
    root = Path(data_dir)
    by_gid = defaultdict(list)
    if not root.exists():
        return None, f"경로를 찾을 수 없습니다: {root}"
    for meta_path in root.rglob("metadata.json"):
        if not meta_path.parent.name.startswith("track_"):
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        imgs = sorted(meta_path.parent.glob("*.jpg"))
        if not imgs or meta.get("global_id") is None:
            continue
        by_gid[meta["global_id"]].append({
            "images": imgs, "cam": meta.get("camera_id", "?"),
            "slot": meta.get("time_slot", "?"), "track": meta.get("track_id", "?"),
            "n": len(imgs),
        })
    return dict(by_gid), None


def n_cam(tl):
    return len({t["cam"] for t in tl})


def page_summary():
    st.markdown('<div class="eyed-hero">EYE-D · 한 장 요약</div>', unsafe_allow_html=True)
    st.markdown('<div class="eyed-sub">엣지 기반 인물 재식별(Re-ID)로 여러 카메라의 같은 사람을 자동으로 연결합니다</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("표준 성능 (Market-1501)", "Rank-1 74.19%", "mAP 79.37%", delta_color="off")
    m2.metric("정제 데이터", "43명", "다중 카메라 동일인", delta_color="off")
    m3.metric("확장 분야", "4 갈래", "침입탐지·리테일·예술·시니어케어", delta_color="off")
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        card("무엇을", "YOLO 검출 → 추적 → OSNet 임베딩 → 제약 병합으로 <b>카메라를 넘어 동일인을 식별</b>합니다.")
        card("성과", "표준 평가로 성능을 검증하고, over-merge를 <b>육안 검수로 잡아</b> 신뢰성을 확보했습니다.")
    with c2:
        card("적용", "지능형 침입 탐지 / 스마트 리테일 / arttrace / <b>시니어 주택 케어</b>.")
        card("현황·다음", "모델은 검증 완료. <b>다음 도약은 데이터 확보</b>(신규 촬영)에 달려 있습니다.")
    st.caption("자세한 내용은 좌측 목차 1~8을 참고하세요. 이 화면을 그대로 캡처하면 1페이지 요약 핸드아웃이 됩니다.")


def page_intro():
    st.markdown('<div class="eyed-hero">EYE-D</div>', unsafe_allow_html=True)
    st.markdown('<div class="eyed-sub">엣지 게이트웨이 기반 지능형 침입 탐지 · 인물 재식별(Re-ID) 시스템</div>', unsafe_allow_html=True)
    st.markdown("> **여러 CCTV에 흩어져 찍힌 같은 사람을, 사람 손을 거치지 않고 자동으로 알아보고 동선을 복원합니다.**")
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1: card("무엇을", "서로 다른 카메라·시간대 영상에서 <b>동일 인물</b>을 식별·추적합니다.")
    with c2: card("어떻게", "엣지에서 영상 처리 후 외형을 벡터로 변환, 카메라를 넘어 <b>자동 매칭</b>.")
    with c3: card("왜", "수동 대조는 불가능. <b>실시간·선제 대응</b>으로 안전과 운영 가치를 만듭니다.")


def page_problem():
    st.markdown('<div class="eyed-hero">해결하는 문제</div>', unsafe_allow_html=True)
    st.markdown('<div class="eyed-sub">다중 카메라 환경에서 사람의 흐름은 카메라 경계에서 끊긴다</div>', unsafe_allow_html=True)
    st.markdown("""
| 기존 방식 | 한계 |
|---|---|
| 사람이 직접 CCTV 대조 | 카메라·시간이 많아질수록 사실상 불가능 |
| 단순 동작 감지 센서 | 정상/이상 구별 불가, 누구인지 모름 |
| 얼굴 인식 | 마스크·모자·뒷모습에서 무력, 프라이버시 부담 |
| 사후 열람형 CCTV | 실시간·선제 개입 불가 |
""")
    st.info("EYE-D는 **얼굴이 아닌 체형·외형**으로 사람을 구분하고, **카메라를 넘어 같은 사람을 잇습니다.**")


def page_arch():
    st.markdown('<div class="eyed-hero">시스템 구조</div>', unsafe_allow_html=True)
    st.markdown('<div class="eyed-sub">엣지에서 영상을 처리하고, 원시 영상은 외부로 보내지 않습니다 (프라이버시)</div>', unsafe_allow_html=True)
    steps = ["① 검출<br>YOLOv8n", "② 추적<br>BoT-SORT", "③ 임베딩<br>OSNet-AIN 512d", "④ 병합<br>제약 클러스터링", "⑤ 평가<br>Market-1501"]
    html = '<div style="display:flex;align-items:center;flex-wrap:wrap;gap:2px;margin:10px 0 20px">'
    for i, s in enumerate(steps):
        html += f'<div class="eyed-step">{s}</div>'
        if i < len(steps) - 1:
            html += '<span class="eyed-arrow">→</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: card("처리 흐름", "사람을 <b>검출</b>하고 등장을 <b>추적</b>해 묶고, 외형을 <b>512차원 벡터</b>로 바꿔 카메라를 넘어 <b>같은 사람으로 병합</b>합니다.")
    with c2: card("기술 스택", "YOLOv8n · BoT-SORT · OSNet-AIN(ONNX) · 제약 HAC · PostgreSQL+pgvector · FastAPI 서버")


def page_model():
    st.markdown('<div class="eyed-hero">모델 기능 · 성능</div>', unsafe_allow_html=True)
    st.markdown('<div class="eyed-sub">공개 표준 평가(Market-1501)로 검증된 동일인 식별 성능</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Rank-1 정확도", "74.19%")
    m2.metric("mAP", "79.37%")
    m3.metric("임베딩 차원", "512-d")
    st.write("")
    df = pd.DataFrame({"정확도(%)": [74.19, 79.37]}, index=["Rank-1", "mAP"])
    st.bar_chart(df, height=260, color="#2a4d8f")
    st.caption("구성: OSNet-AIN + Market-1501 가중치, zero-shot, Re-ranking + Mean (실험 ⑧-f). Rank-1=첫 후보 정답 확률, mAP=검색 품질 종합.")
    c1, c2 = st.columns(2)
    with c1: card("핵심 기능", "카메라 간 <b>동일인 식별</b>, 트랙렛 평균 풀링, k-reciprocal <b>Re-ranking</b>으로 검색 정확도 향상.")
    with c2: card("데이터 품질관리", "<b>over-merge를 육안 검수로 발견·수정</b>하고, 시간대 교차 병합을 막는 안전장치를 코드에 반영.")


def page_demo(data_dir):
    st.markdown('<div class="eyed-hero">라이브 데모 — 동일인 식별</div>', unsafe_allow_html=True)
    by_gid, err = scan_tracklets(data_dir)
    if err:
        st.error(err)
        st.info("사이드바 하단 '데모 데이터' 에서 경로를 확인하세요 (WSL: /mnt/d/EYE-D/data/curated).")
        return
    if not by_gid:
        st.warning("global_id 가 부여된 트랙렛이 없습니다. 병합 결과(curated) 폴더를 가리키는지 확인하세요.")
        return
    people = [(g, t, n_cam(t)) for g, t in by_gid.items() if n_cam(t) >= 2]
    people.sort(key=lambda x: (x[2], len(x[1])), reverse=True)
    total = len(by_gid)
    multi = sum(1 for _, t in by_gid.items() if n_cam(t) >= 2)
    a, b, c = st.columns(3)
    a.metric("식별된 인물", f"{total}명")
    b.metric("2대+ 카메라 동일인", f"{multi}명")
    c.metric("표준 평가", "Rank-1 74.19% / mAP 79.37%")
    if not people:
        st.warning("여러 카메라에 등장한 인물이 없습니다.")
        return
    choice = st.selectbox("인물 선택", options=people,
                          format_func=lambda p: f"ID {p[0]} — 카메라 {p[2]}대 / 트랙렛 {len(p[1])}개")
    gid, tlist, ncam = choice
    st.subheader(f"🎯 Global ID {gid} — {ncam}개 카메라에서 동일인으로 식별됨")
    ordered = sorted(tlist, key=lambda t: (t["slot"] if isinstance(t["slot"], int) else 0, t["cam"]))
    st.markdown("**동선:** " + "  →  ".join(f"cam{t['cam']}(t{t['slot']})" for t in ordered))
    by_c = defaultdict(list)
    for t in tlist:
        by_c[t["cam"]].append(t)
    cols = st.columns(max(len(by_c), 1))
    for col, cam in zip(cols, sorted(by_c.keys(), key=lambda c: str(c))):
        with col:
            st.markdown(f"### 📷 카메라 {cam}")
            for t in by_c[cam]:
                st.caption(f"t{t['slot']} · track_{t['track']} · {t['n']}장")
                thumbs = t["images"][:: max(1, len(t["images"]) // 6)][:6]
                imgs = [Image.open(p).convert("RGB") for p in thumbs if p.exists()]
                if imgs:
                    st.image(imgs, width=70)


def page_dataset(data_dir):
    st.markdown('<div class="eyed-hero">정제 데이터 현황</div>', unsafe_allow_html=True)
    st.markdown('<div class="eyed-sub">검수를 거쳐 라벨이 정리된 curated 데이터셋</div>', unsafe_allow_html=True)
    by_gid, err = scan_tracklets(data_dir)
    if err or not by_gid:
        st.warning("데이터를 불러오지 못했습니다. 사이드바 '데모 데이터' 경로를 확인하세요.")
        return
    tracklets = sum(len(t) for t in by_gid.values())
    images = sum(tr["n"] for t in by_gid.values() for tr in t)
    a, b, c = st.columns(3)
    a.metric("인물(Global ID)", f"{len(by_gid)}명")
    b.metric("트랙렛", f"{tracklets}개")
    c.metric("정제 프레임", f"{images:,}장")
    st.write("")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**카메라 등장 수별 인물 분포**")
        cc = Counter(n_cam(t) for t in by_gid.values())
        df1 = pd.DataFrame({"인물 수": [cc[k] for k in sorted(cc)]}, index=[f"{k}대" for k in sorted(cc)])
        st.bar_chart(df1, height=240, color="#2a4d8f")
    with g2:
        st.markdown("**카메라별 정제 프레임 수**")
        ibc = Counter()
        for t in by_gid.values():
            for tr in t:
                ibc[tr["cam"]] += tr["n"]
        df2 = pd.DataFrame({"프레임 수": [ibc[k] for k in sorted(ibc, key=str)]}, index=[f"카메라 {k}" for k in sorted(ibc, key=str)])
        st.bar_chart(df2, height=240, color="#5b8def")
    st.info("**모델은 됐고, 이제 데이터다.** 정제 인물 풀을 넓힐수록 정밀도와 적용 범위가 함께 커집니다.")


def page_journey():
    st.markdown('<div class="eyed-hero">추진 경과 · 역경 극복</div>', unsafe_allow_html=True)
    st.markdown('<div class="eyed-sub">매끄럽지 않았습니다. 막힌 지점마다 원인을 찾아 넘었습니다</div>', unsafe_allow_html=True)
    journey("Docker 드라이브 이전 사고", "C→D 이전 중 정리 명령으로 DB 볼륨이 통째로 사라짐.",
            "빈 DB 재구축, 정리 전 pg_dump 백업을 표준 절차로 못박음.", "되돌릴 수 없는 작업 전엔 백업부터.")
    journey("8GB RAM의 벽 (OOM)", "30분 혼잡 영상 추적에서 메모리 초과로 중단.",
            "WSL 스왑 확대 + 영상 15분 트림 + 프레임 간격 확대로 처리량 조절.", "자원 제약은 핑계가 아니라 설계 조건.")
    journey("조용한 실패 — 가중치 0/552", "AIN 모델 가중치가 하나도 로딩되지 않음(이름 규칙 불일치).",
            "접두어를 보정해 550/552 정상 로딩, 부풀려진 수치를 바로잡음.", "에러 없이 틀리는 실패는 수치로만 잡힌다.")
    journey("보이지 않던 과병합(over-merge)", "다른 사람이 한 ID로 병합(시간대 교차)되어 지표가 부풀려짐.",
            "육안 검수로 발견, 시간대 교차 병합을 막는 가드를 코드에 추가.", "육안 검수가 지표를 구했다.")
    journey("파인튜닝의 역설", "추가 학습이 오히려 성능을 떨어뜨림(소량 데이터 과적합).",
            "zero-shot ⑧-f를 운영 모델로 채택, 병목이 데이터임을 규명.", "모델보다 데이터가 결정한다.")
    journey("데이터 소진", "가용 영상으로는 43명에서 인물 풀이 한계.",
            "신규 촬영 필요를 보고, 데모·제안서로 시스템 가치를 입증하는 방향으로 전환.", "한계를 인정하고 가치를 증명하는 쪽으로.")


def page_vision():
    st.markdown('<div class="eyed-hero">비전 · 확장 · 로드맵</div>', unsafe_allow_html=True)
    st.markdown('<div class="eyed-sub">같은 Re-ID 코어 위에서 여러 분야로, 그리고 예방 건강 파트너로</div>', unsafe_allow_html=True)
    st.markdown("##### 같은 코어, 네 갈래 적용")
    st.markdown(
        '<span class="eyed-tag">① 지능형 침입 탐지</span>'
        '<span class="eyed-tag">② 스마트 리테일 — VIP·단골 알림 / 체류시간 분석</span>'
        '<span class="eyed-tag">③ arttrace — 관객 행동 기반 멀티모달 예술</span>'
        '<span class="eyed-tag">④ 시니어 주택 케어 — 낙상·이탈 / 동선·생활패턴 / 방문자 관리 / 케어 알림</span>',
        unsafe_allow_html=True)
    st.write("")
    st.markdown("##### 단계별 로드맵")
    st.markdown("""
| 단계 | 시기 | 핵심 |
|---|---|---|
| **Phase A** | 2026 | 시니어 주택 1개 동 파일럿 · 데이터 거버넌스 확립 |
| **Phase B** | 2027 | 3~5개 단지 확산 · 스마트워치 활동량 결합 |
| **Phase C** | 2028~ | 표준화 · 개인 의료 데이터 연동 연구 |
| **Phase D** | 2029~ | 예방 건강 플랫폼 · 건축 설계 기준 데이터 제공 |
""")
    st.success("생활 데이터로 공간을 설계하는 패러다임 — 데이터의 주인은 언제나 사용자 자신입니다.")


PAGES = {
    "0 · 한 장 요약": page_summary,
    "1 · 프로젝트 소개": page_intro,
    "2 · 해결 문제": page_problem,
    "3 · 시스템 구조": page_arch,
    "4 · 모델 기능·성능": page_model,
    "5 · 라이브 데모 (Re-ID)": "demo",
    "6 · 정제 데이터 현황": "dataset",
    "7 · 추진 경과·역경": page_journey,
    "8 · 비전·확장·로드맵": page_vision,
}

with st.sidebar:
    st.markdown("## 📑 EYE-D 발표")
    sel = st.radio("목차", list(PAGES.keys()), label_visibility="collapsed")
    st.divider()
    with st.expander("⚙️ 데모 데이터", expanded=(sel.startswith(("5", "6")))):
        data_dir = st.text_input("데이터 폴더", value=DEFAULT_DATA_DIR)
        if st.button("🔄 다시 스캔"):
            scan_tracklets.clear()

target = PAGES[sel]
if target == "demo":
    page_demo(data_dir)
elif target == "dataset":
    page_dataset(data_dir)
else:
    target()
