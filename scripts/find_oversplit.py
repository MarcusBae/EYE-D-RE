import json, sys, argparse
from pathlib import Path
from itertools import combinations
from collections import Counter
import cv2, numpy as np, yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from pipeline.reid_merger import OSNetExtractor

MAXF = 8  # 트랙렛당 샘플 프레임 수

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/config.yaml")
ap.add_argument("--filtered-dir", default="data/filtered")
ap.add_argument("--sim-threshold", type=float, default=0.70)
ap.add_argument("--out", default="outputs/oversplit_report.json")
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
rc = cfg.get("reid", {})
ext = OSNetExtractor(model_name=rc.get("model_name", "osnet_x1_0"),
                     pretrained=rc.get("pretrained", True),
                     weights_path=rc.get("weights_path", None),
                     device=rc.get("device", "auto"))

frames_by_gid, meta_by_gid = {}, {}
for mp in sorted(Path(args.filtered_dir).glob("*/track_*/metadata.json")):
    try:
        m = json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        continue
    gid = m.get("global_id")
    if gid is None:
        continue
    fr = sorted(mp.parent.glob("frame_*.jpg")) or sorted(mp.parent.glob("*.jpg"))
    if not fr:
        continue
    for i in np.linspace(0, len(fr)-1, min(MAXF, len(fr))).astype(int):
        img = cv2.imread(str(fr[i]))
        if img is not None and img.size > 0:
            frames_by_gid.setdefault(gid, []).append(img)
    gm = meta_by_gid.setdefault(gid, {"slots": set(), "cams": set()})
    gm["slots"].add(m.get("time_slot")); gm["cams"].add(m.get("camera_id"))

gids, cents = [], []
for gid, imgs in frames_by_gid.items():
    if not imgs:
        continue
    f = np.asarray(ext.extract_batch_features(imgs), dtype=np.float32)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
    c = f.mean(axis=0); c = c / (np.linalg.norm(c) + 1e-9)
    gids.append(gid); cents.append(c)
cents = np.stack(cents)
sim = cents @ cents.T

rows = []
for i, j in combinations(range(len(gids)), 2):
    s = float(sim[i, j])
    if s < args.sim_threshold:
        continue
    a, b = gids[i], gids[j]; ma, mb = meta_by_gid[a], meta_by_gid[b]
    same_slot = bool(ma["slots"] & mb["slots"])
    shared_cam = bool(ma["cams"] & mb["cams"])
    if same_slot and not shared_cam:
        kind = "A2(같은슬롯·다른카메라)"
    elif not same_slot:
        kind = "cross-slot(다른시간대)"
    else:
        kind = "같은슬롯·같은카메라"
    rows.append({"gid_a": a, "gid_b": b, "sim": round(s, 3), "kind": kind,
                 "a": {"slots": sorted(ma["slots"]), "cams": sorted(ma["cams"])},
                 "b": {"slots": sorted(mb["slots"]), "cams": sorted(mb["cams"])}})

rows.sort(key=lambda r: -r["sim"])
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(rows, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

print("=" * 60)
print(f"과분할 후보 (유사도 >= {args.sim_threshold}) : {len(rows)}쌍")
for k, v in Counter(r["kind"] for r in rows).items():
    print(f"  - {k}: {v}쌍")
print("=" * 60)
print(f"{'gid_a':>6} {'gid_b':>6} {'sim':>6}  유형")
for r in rows[:30]:
    print(f"{r['gid_a']:>6} {r['gid_b']:>6} {r['sim']:>6.3f}  {r['kind']}")
print(f"\n저장: {args.out}")
