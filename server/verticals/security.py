"""침입탐지 버티컬 — detections(위치)와 zones(구역)를 대조해 제한구역 진입 신호 생성.
순수 기하 규칙: 점이 다각형 안인가(point-in-polygon) + 구역이 제한구역인가(purpose)."""

RESTRICTED_PURPOSES = {"restricted", "no_entry", "staff_only", "금지구역"}

def is_restricted(zone: dict) -> bool:
    return (zone.get("purpose") or "").lower() in RESTRICTED_PURPOSES

def point_in_polygon(x, y, polygon) -> bool:
    """ray-casting: 점(x,y)이 다각형 polygon([[x,y],...]) 내부인지."""
    if not polygon or len(polygon) < 3:
        return False
    inside = False
    n = len(polygon); j = n - 1
    for i in range(n):
        xi, yi = polygon[i]; xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def intrusion_check(bbox, zones) -> dict:
    """검출 bbox 중심이 어떤 제한구역 안이면 침입."""
    cx, cy = bbox_center(bbox)
    for z in zones:
        if is_restricted(z) and point_in_polygon(cx, cy, z.get("polygon") or []):
            return {"is_intrusion": True, "zone_id": z.get("zone_id"),
                    "zone_name": z.get("name"), "alert": f"🚨 제한구역 침입: {z.get('name')}"}
    return {"is_intrusion": False, "zone_id": None, "alert": None}
