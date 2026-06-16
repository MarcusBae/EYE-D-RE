"""scripts/ingest_video_tracked.py — 트래킹 기반 영상 자동 등록.
YOLO+BoT-SORT로 같은 사람에게 연속 track_id 부여 → 트랙당 대표 crop 1장만 /ingest.
사용법: python scripts/ingest_video_tracked.py <video> [max_frames=300] [threshold=0.35] [commit=false]"""
import sys, cv2, requests
from collections import defaultdict
sys.path.insert(0, ".")
from ultralytics import YOLO

def main():
    video  = sys.argv[1]
    maxf   = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    thr    = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
    commit = sys.argv[4] if len(sys.argv) > 4 else "false"
    server = "http://127.0.0.1:8000"
    print(f"영상: {video} (트래킹, 최대 {maxf}프레임, threshold={thr}, commit={commit})")
    model = YOLO("yolov8n.pt")
    best = {}   # track_id -> (area, jpg_bytes)  : 트랙별 가장 큰 crop 보관
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
            area = (x2-x1)*(y2-y1)
            if tid not in best or area > best[tid][0]:
                ok, buf = cv2.imencode(".jpg", crop)
                if ok: best[tid] = (area, buf.tobytes())
    print(f"트랙(고유 인물) 수: {len(best)}  (프레임 {fi}개 처리)")
    stats = defaultdict(int)
    for tid, (area, jpg) in sorted(best.items()):
        try:
            rr = requests.post(f"{server}/reid/ingest",
                params={"camera_id":"CAM_01","tracklet_id":f"vid-track{tid}","threshold":thr,"commit":commit},
                files={"file":("c.jpg", jpg, "image/jpeg")}, timeout=90)
            d = rr.json(); stats[d["decision"]] += 1
            print(f"  track {tid:>3}: {d['decision']:<11} global_id={d['global_id']} sim={d.get('nearest_similarity')}")
        except Exception as e:
            stats["error"] += 1; print(f"  track {tid}: ERROR {e}")
    print(f"=== 결과 === 트랙 {len(best)} · existing {stats['existing']} · new {stats['new_person']} · 에러 {stats['error']}")

if __name__ == "__main__":
    main()
