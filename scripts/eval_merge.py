import json, sys
from pathlib import Path
from itertools import combinations
gt = json.load(open("/tmp/gt_curated43.json"))
pdir = Path(sys.argv[1])
pred = {str(mp.parent.relative_to(pdir)): json.loads(mp.read_text(encoding="utf-8")).get("global_id")
        for mp in pdir.glob("*/track_*/metadata.json")}
keys = [k for k in gt if k in pred]
nT = len(set(gt[k] for k in keys)); nP = len(set(pred[k] for k in keys))
tt=pp=tp=os_=om=0
for a,b in combinations(keys,2):
    t = gt[a]==gt[b]; p = pred[a]==pred[b]
    tt += t; pp += p; tp += (t and p); os_ += (t and not p); om += ((not t) and p)
print(f"트랙렛 {len(keys)} | 정답 ID {nT} → 예측 ID {nP}")
print(f"과분할 쌍(같은데 갈림): {os_}  | merge recall {tp/tt:.1%} (높을수록 덜 쪼갬)")
print(f"과병합 쌍(다른데 묶임): {om}  | merge precision {tp/pp:.1%} (높을수록 덜 잘못합침)")
