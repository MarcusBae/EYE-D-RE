import os
import sys
import glob
import json
import shutil
from pathlib import Path
import cv2
import numpy as np

# 프로젝트 루트를 sys.path에 추가하여 pipeline 임포트 가능하도록 설정
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from pipeline.detector import PersonDetector

def main():
    # 1. 대상 디렉토리 결정 (수동 편집 버전만 탐색하며, 없을 시 즉시 종료)
    target_dir = Path("data/tracklets-2-manual-edit")
    if not target_dir.exists():
        print(f"[오류] 수동 편집 버전 폴더가 존재하지 않아 작업을 종료합니다. (경로: {target_dir})")
        return
        
    print(f"[INFO] 탐색 대상 디렉토리: {target_dir.resolve()}")
    
    # 2. Detector 로드 (yolov8n.pt 사용)
    # GPU(CUDA)를 자동 탐색하여 실행 속도를 높입니다.
    detector = PersonDetector(model_path="yolov8n.pt", conf=0.5, device="auto", verbose=False)
    
    # 3. 이미지 수집 (모든 c*_t* 폴더 내의 track_* 폴더 내 이미지 수집)
    img_paths = sorted(list(target_dir.glob("c*_t*/track_*/frame_*.jpg")))
    total_imgs = len(img_paths)
    print(f"[INFO] 총 검사 대상 이미지 수: {total_imgs}개")
    
    if total_imgs == 0:
        print("[경고] 검사할 이미지 파일이 없습니다.")
        return

    multi_person_list = []
    
    # 4. 스캔 시작
    print("스캔을 시작합니다...")
    for idx, img_path in enumerate(img_paths):
        # 200개 단위로 진행률 표시
        if (idx + 1) % 200 == 0:
            print(f"진행 중: {idx + 1}/{total_imgs} ({(idx + 1)/total_imgs*100:.1f}%)")
            
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        bboxes, confs, cls = detector.detect(img)
        
        # 사람 클래스(0) 개수 세기
        person_count = sum(1 for c in cls if c == 0)
        
        if person_count >= 2:
            # 2명 이상 검출된 경우
            cam_slot = img_path.parent.parent.name
            track_id = img_path.parent.name
            rel_path = img_path.relative_to(target_dir.parent)
            
            multi_person_list.append({
                "image_path": str(img_path.resolve()),
                "relative_path": str(rel_path),
                "cam_slot": cam_slot,
                "track_id": track_id,
                "person_count": person_count,
                "bboxes": bboxes.tolist(),
                "confidences": confs.tolist()
            })
            
            # 이미지 파일 이동 실행 (data/remove 폴더로 이동)
            try:
                dest_dir = Path("data/remove") / cam_slot / track_id
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_path = dest_dir / img_path.name
                shutil.move(str(img_path), str(dest_path))
                print(f"  [이동 완료] {cam_slot}/{track_id}/{img_path.name} -> data/remove/{cam_slot}/{track_id}/{img_path.name} ({person_count}명 검출)")
            except Exception as e:
                print(f"  [이동 실패] {cam_slot}/{track_id}/{img_path.name} -> 에러: {e}")
            
    # 5. 결과 리포트 출력 및 저장
    print("\n" + "="*60)
    print(f"[스캔 완료] 두 사람 이상이 포함된 이미지 총 {len(multi_person_list)}개 발견")
    print("="*60)
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_json = output_dir / "multi_person_removed.json"
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(multi_person_list, f, indent=4, ensure_ascii=False)
        
    print(f"[INFO] 결과 파일 저장 완료: {output_json.resolve()}")

if __name__ == "__main__":
    main()
