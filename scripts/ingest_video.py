"""scripts/ingest_video.py — 영상 자동 등록 파이프라인.
영상 -> 프레임 샘플 -> YOLO 사람 검출 -> crop -> 서버 /reid/ingest 호출.
사용법: python scripts/ingest_video.py <video> [stride=30] [max_frames=8] [threshold=0.5] [commit=false]
(서버가 떠 있어야 함: uvicorn server.main:app)"""
import sys, cv2, requests
sys.path.insert(0, ".")
from pipeline.detector import PersonDetector

def main():
    video  = sys.argv[1]
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    maxf   = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    thr    = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    commit = sys.argv[5] if len(sys.argv) > 5 else "false"
    server = "http://127.0.0.1:8000"
    print(f"영상: {video} (매 {stride}프레임, 최대 {maxf}, threshold={thr}, commit={commit})")
    det = PersonDetector(model_path="yolov8n.pt", conf=0.5, device="auto", verbose=False)
    cap = cv2.VideoCapture(video)
    fi = processed = 0
    stats = {"frames":0,"persons":0,"existing":0,"new_person":0,"errors":0}
    while processed < maxf:
        ret, frame = cap.read()
        if not ret: break
        if fi % stride == 0:
            stats["frames"] += 1; npers = 0
            bb, cf, cls = det.detect(frame)
            for (x1,y1,x2,y2), c in zip(bb, cls):
                if c != 0: continue
                x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
                crop = frame[max(0,y1):y2, max(0,x1):x2]
                if crop.size == 0: continue
                ok, buf = cv2.imencode(".jpg", crop)
                if not ok: continue
                try:
                    r = requests.post(f"{server}/reid/ingest",
                        params={"camera_id":"CAM_01","tracklet_id":f"vid-f{fi}","threshold":thr,"commit":commit},
                        files={"file":("c.jpg", buf.tobytes(), "image/jpeg")}, timeout=90)
                    d = r.json(); stats["persons"]+=1; npers+=1
                    stats[d["decision"]] = stats.get(d["decision"],0)+1
                except Exception:
                    stats["errors"] += 1
            print(f"  frame {fi:>5}: {npers}명")
            processed += 1
        fi += 1
    cap.release()
    print(f"=== 결과 === 프레임 {stats['frames']} · 인물 {stats['persons']} · "
          f"existing {stats['existing']} · new {stats['new_person']} · 에러 {stats['errors']}")

if __name__ == "__main__":
    main()
