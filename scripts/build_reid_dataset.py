import os
import json
import shutil
import re
from pathlib import Path
import numpy as np

def parse_track_path(track_path_str):
    # 예: "c1_t3/track_0012" -> cam=1, slot=3, track_dir="track_0012"
    m = re.match(r"c(\d+)_t(\d+)/(track_\d+)", track_path_str)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3)

def build_dataset(map_file_path, output_dir, num_samples=15):
    project_root = Path(__file__).resolve().parent.parent
    src_base_dir = project_root / "data/tracklets-2-manual-edit"
    output_dir = Path(output_dir)
    train_dir = output_dir / "bounding_box_train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(map_file_path):
        print(f"[오류] 매핑 JSON 파일이 존재하지 않습니다: {map_file_path}")
        return
        
    with open(map_file_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)
        
    print(f"[INFO] {len(id_map)}명의 인물 매핑 정보를 로드했습니다.")
    total_copied = 0
    
def build_dataset_from_metadata(src_base_dir, output_dir, num_samples=15):
    """
    각 트랙릿 폴더 내 metadata.json의 global_id를 직접 탐색하여
    Re-ID 데이터셋(Market-1501 포맷)을 구축합니다.
    """
    project_root = Path(__file__).resolve().parent.parent
    output_dir = Path(output_dir)
    train_dir = output_dir / "bounding_box_train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] metadata.json 기반 자동 수집 시작: {src_base_dir.relative_to(project_root)}")
    
    # 1. 모든 metadata.json을 스캔하여 global_id별 트랙 목록 수집
    global_groups = {}
    for meta_path in sorted(src_base_dir.glob("c*_t*/track_*/metadata.json")):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
            
        gid = meta.get("global_id")
        if gid is None or gid < 0:
            continue
            
        global_groups.setdefault(gid, []).append((meta_path.parent, meta))
        
    print(f"[INFO] 발견된 총 Global ID 개수: {len(global_groups)}개")
    total_copied = 0
    
    for gid in sorted(global_groups.keys()):
        tracks = global_groups[gid]
        print(f"\n👤 Global ID: {gid:04d} 처리 중 (트랙릿 {len(tracks)}개 발견)...")
        
        for tdir, meta in tracks:
            cam = meta.get("camera_id", "?")
            slot = meta.get("time_slot", "?")
            
            img_paths = sorted(list(tdir.glob("*.jpg")))
            if not img_paths:
                continue
                
            # 샘플링할 개수 결정
            actual_samples = min(num_samples, len(img_paths))
            indices = np.linspace(0, len(img_paths) - 1, actual_samples, dtype=int)
            sampled_paths = [img_paths[i] for i in indices]
            
            print(f"  └─ 📷 Cam {cam} (Slot {slot}): {len(img_paths)}장 중 {actual_samples}장 샘플링 완료")
            
            for idx, img_path in enumerate(sampled_paths):
                frame_match = re.search(r"frame_(\d+)\.jpg", img_path.name)
                if frame_match:
                    frame_num = int(frame_match.group(1))
                else:
                    frame_num = idx
                    
                # Market-1501 파일명 규격 생성
                new_filename = f"{gid:04d}_c{cam}s{slot}_{frame_num:06d}_00.jpg"
                dest_path = train_dir / new_filename
                
                shutil.copy2(img_path, dest_path)
                total_copied += 1
                
    print(f"\n✨ Re-ID Custom 데이터셋 빌드 완료!")
    print(f"   출력 경로: {train_dir.relative_to(project_root)}")
    print(f"   총 생성 이미지 수: {total_copied}장")


def build_dataset(map_file_path, output_dir, num_samples=15):
    project_root = Path(__file__).resolve().parent.parent
    src_base_dir = project_root / "data/tracklets-2-manual-edit"
    output_dir = Path(output_dir)
    train_dir = output_dir / "bounding_box_train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(map_file_path):
        print(f"[정보] 매핑 JSON이 없습니다. metadata.json에 기입된 정보를 직접 사용합니다.")
        build_dataset_from_metadata(src_base_dir, output_dir, num_samples)
        return
        
    with open(map_file_path, "r", encoding="utf-8") as f:
        id_map = json.load(f)
        
    print(f"[INFO] {len(id_map)}명의 인물 매핑 정보를 로드했습니다.")
    total_copied = 0
    
    for global_id_str, cam_tracks in id_map.items():
        try:
            pid = int(global_id_str)
        except ValueError:
            print(f"[경고] ID 형식이 올바르지 않습니다: {global_id_str}")
            continue
            
        print(f"\n👤 Global ID: {pid:04d} 처리 중...")
        for cam_key, track_rel_path in cam_tracks.items():
            parsed = parse_track_path(track_rel_path)
            if not parsed:
                print(f"  └─ [경고] 경로 파싱 실패: {track_rel_path}")
                continue
                
            cam, slot, track_dir_name = parsed
            track_full_path = src_base_dir / f"c{cam}_t{slot}" / track_dir_name
            
            if not track_full_path.exists():
                print(f"  └─ ❌ 폴더 없음: {track_full_path.relative_to(project_root)}")
                continue
                
            # 이미지 수집
            img_paths = sorted(list(track_full_path.glob("frame_*.jpg")))
            if not img_paths:
                print(f"  └─ ⚠️ 이미지 없음: {track_full_path.relative_to(project_root)}")
                continue
                
            # 샘플링할 개수 결정
            actual_samples = min(num_samples, len(img_paths))
            indices = np.linspace(0, len(img_paths) - 1, actual_samples, dtype=int)
            sampled_paths = [img_paths[i] for i in indices]
            
            print(f"  └─ 📷 Cam {cam} (Slot {slot}): {len(img_paths)}장 중 {actual_samples}장 샘플링 완료")
            
            for idx, img_path in enumerate(sampled_paths):
                # 프레임 번호 파싱
                frame_match = re.search(r"frame_(\d+)\.jpg", img_path.name)
                if frame_match:
                    frame_num = int(frame_match.group(1))
                else:
                    frame_num = idx
                    
                # Market-1501 파일명 규격 생성: [pid:04d]_c[cam]s[slot]_[frame:06d]_00.jpg
                new_filename = f"{pid:04d}_c{cam}s{slot}_{frame_num:06d}_00.jpg"
                dest_path = train_dir / new_filename
                
                shutil.copy2(img_path, dest_path)
                total_copied += 1
                
    print(f"\n✨ Re-ID Custom 데이터셋 빌드 완료!")
    print(f"   출력 경로: {train_dir.relative_to(project_root)}")
    print(f"   총 생성 이미지 수: {total_copied}장")

if __name__ == "__main__":
    # 예시 실행용 설정 (필요시 CLI 인자나 직접 변경 가능)
    project_root = Path(__file__).resolve().parent.parent
    map_json = project_root / "configs/global_id_map.json"
    out_dir = project_root / "data/reid_dataset_custom"
    
    build_dataset(map_json, out_dir, num_samples=15)
