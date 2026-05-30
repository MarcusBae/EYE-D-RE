import os
import re
import shutil

def rename_videos():
    dir_path = "/home/torious/projects/tmp/EYE-D-Restart/data/raw_videos"
    backup_dir = os.path.join(dir_path, "backup_old")
    
    # 1. 매핑 정의
    cam_map = {"00": 1, "02": 2, "03": 3}
    time_map = {
        "120000": 1,
        "123000": 2,
        "130000": 3,
        "133000": 4,
        "140000": 5,
        "143000": 6,
        "150000": 7,
        "153000": 8,
        "160000": 9,
        "163000": 10
    }
    
    # Target pattern: 8 digits followed by .avi (6 digits time + 2 digits camera)
    file_pattern = re.compile(r"^(\d{6})(\d{2})\.avi$")
    
    # 2. 파일 목록 조회 및 변환 대상 분류
    all_files = os.listdir(dir_path)
    rename_targets = []
    existing_cams = []
    
    for f in all_files:
        match = file_pattern.match(f)
        if match:
            time_part, cam_part = match.groups()
            if cam_part in cam_map and time_part in time_map:
                rename_targets.append((f, time_part, cam_part))
        elif f.startswith("cam") and f.endswith(".avi"):
            existing_cams.append(f)
            
    # 3. 기존 cam*_t*.avi 파일이 있는 경우 백업 폴더로 이동 (이름 충돌 방지)
    if existing_cams:
        os.makedirs(backup_dir, exist_ok=True)
        print(f"기존 파일 백업 중... ({len(existing_cams)}개)")
        for f in existing_cams:
            src_path = os.path.join(dir_path, f)
            dst_path = os.path.join(backup_dir, f)
            shutil.move(src_path, dst_path)
            print(f"백업 이동: {f} -> backup_old/{f}")
            
    # 4. rename 수행
    print(f"\n파일 이름 변환 시작... ({len(rename_targets)}개)")
    for old_name, time_part, cam_part in rename_targets:
        cam_num = cam_map[cam_part]
        slot_num = time_map[time_part]
        new_name = f"cam{cam_num}_t{slot_num}.avi"
        
        src_path = os.path.join(dir_path, old_name)
        dst_path = os.path.join(dir_path, new_name)
        
        os.rename(src_path, dst_path)
        print(f"변환 완료: {old_name} -> {new_name}")

if __name__ == "__main__":
    rename_videos()
