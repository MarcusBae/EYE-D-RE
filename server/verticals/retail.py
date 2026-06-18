"""Smart Retail 버티컬 로직 — Re-ID 코어(persons)를 소비해 고객 등급·알림 신호 생성.
Re-ID가 채운 visit_count를 '비즈니스 신호'로 변환한다. (순수 규칙, DB 의존 없음)"""

# (방문횟수 임계, 등급) — 높은 것부터
TIER_RULES = [(10, "vip"), (4, "regular"), (2, "returning"), (0, "new")]

def classify_tier(visit_count: int) -> str:
    for th, tier in TIER_RULES:
        if (visit_count or 0) >= th:
            return tier
    return "new"

def retail_signal(person: dict) -> dict:
    vc = person.get("visit_count", 0) or 0
    tier = classify_tier(vc)
    alerts = []
    if tier == "vip":
        alerts.append("⭐ VIP 고객")
    elif tier == "regular":
        alerts.append("단골 고객")
    return {"global_id": person.get("global_id"), "visit_count": vc,
            "computed_tier": tier, "stored_tier": person.get("customer_tier"),
            "is_vip": tier == "vip", "alerts": alerts}
