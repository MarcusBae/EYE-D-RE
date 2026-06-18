"""scripts/ingest_video_tracked.py — 트래킹 기반 영상 자동 등록 (+ 품질 필터).
YOLO+BoT-SORT로 같은 사람 연속 track_id → 트랙당 대표 crop 1장만 /ingest.
품질 필터: 최소 트랙 길이(min_len 프레임) + 최소 crop 면적(min_area px) 미만 트랙은 등록 제외.
사용법: python scripts/ingest_video_tracked.py <video> [max_frames=300] [threshold=0.35] [commit=false] [min_len=5] [min_area=4000]"""
import sys, cv2, requests
from collections import defaultdict
sys.path.insert(0, ".")
from ultralytics import YOLO

def main():
    video    = sys.argv[1]
    maxf     = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    thr      = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
    commit   = sys.argv[4] if len(sys.argv) > 4 else "false"
    min_len  = int(sys.argv[5]) if len(sys.argv) > 5 else 5
    min_area = int(sys.argv[6]) if len(sys.argv) > 6 else 4000
    server = "http://127.0.0.1:8000"
    print(f"영상: {video} (트래킹+품질필터 min_len={min_len}f min_area={min_area}px thr={thr} commit={commit})")
    model = YOLO("yolov8n.pt")
    best  = {}                 # track_id -> (area, jpg_bytes)
    count = defaultdict(int)   # track_id -> 등장 프레임 수
    fi = 0
    for r in model.track(source=video, stream=True, persist=True,
                         classes=[0], conf=0.5, tracker="botsort.yaml", verbose=False):
        if fi >= maxf: break
        fi += 1
        if r.boxes is None or r.boxes.id is None: continue
        frame = r.orig_img
        ids  = r.boxes.id.cpu().numpy().astype(int)
        xyxy = r.boxes.xyxy.cpu().numpy()
        for (x1,y1,x2,y2), tid in zip(xyxy, ids):
            x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
            crop = frame[max(0,y1):y2, max(0,x1):x2]
            if crop.size == 0: continue
            count[tid] += 1
            area = (x2-x1)*(y2-y1)
            if tid not in best or area > best[tid][0]:
                ok, buf = cv2.imencode(".jpg", crop)
                if ok: best[tid] = (area, buf.tobytes())
    kept, skipped = [], []
    for tid in best:
        (kept if (count[tid] >= min_len and best[tid][0] >= min_area) else skipped).append(tid)
    print(f"전체 트랙 {len(best)}  ->  등록 {len(kept)}, 제외 {len(skipped)} (단편/저품질)")
    for tid in sorted(skipped):
        print(f"  [제외] track {tid}: len={count[tid]}f area={best[tid][0]}px")
    stats = defaultdict(int)
    for tid in sorted(kept):
        area, jpg = best[tid]
        try:
            rr = requests.post(f"{server}/reid/ingest",
                params={"camera_id":"CAM_01","tracklet_id":f"vid-track{tid}","threshold":thr,"commit":commit},
                files={"file":("c.jpg", jpg, "image/jpeg")}, timeout=90)
            d = rr.json(); stats[d["decision"]] += 1
            print(f"  [등록] track {tid:>3}: {d['decision']:<11} gid={d['global_id']} sim={d.get('nearest_similarity')} (len={count[tid]}f)")
        except Exception as e:
            stats["error"] += 1; print(f"  track {tid}: ERROR {e}")
    print(f"=== 결과 === 등록 {len(kept)} · existing {stats['existing']} · new {stats['new_person']} · 제외 {len(skipped)} · 에러 {stats['error']}")

if __name__ == "__main__":
    main()
