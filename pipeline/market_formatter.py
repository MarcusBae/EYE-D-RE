import os
import shutil
import random
import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
from tqdm import tqdm
from .tracklet_io import list_tracklets

class MarketFormatter:
    """
    정제된 트랙렛 데이터셋을 Re-ID 표준 포맷인 Market-1501 형식으로 변환.
    - 전체 global_id를 중복 없이 Train/Test ID로 분할 (Disjoint Split)
    - Test ID에 속하는 이미지는 Query(대표 1장)와 Gallery(나머지)로 분배
    - 파일명을 [personID:04d]_c[cam]s[slot]_[frameID:06d]_00.jpg 규격으로 변환 및 복사
    """
    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config.get("data", {})
        self.market_config = config.get("market1501", {})
        
        self.filtered_dir = Path(self.data_config.get("filtered_dir", "data/filtered"))
        self.output_dir = Path(self.market_config.get("output_dir", "data/market1501"))
        self.train_ratio = self.market_config.get("train_ratio", 0.7)
        self.seed = config.get("runtime", {}).get("seed", 42)
        
    def _extract_number(self, text) -> int:
        """문자열에서 숫자만 추출 (예: 'c1' -> 1, 't2' -> 2, 정수인 경우 그대로 반환)"""
        if isinstance(text, int):
            return text
        if text is None:
            return 0
        text_str = str(text)
        match = re.search(r'\d+', text_str)
        return int(match.group()) if match else 0
        
    def _parse_frame_num(self, filename) -> int:
        """이미지 파일명에서 프레임 번호 추출 (예: '000125.jpg' -> 125)"""
        if isinstance(filename, int):
            return filename
        filename_str = str(filename)
        name_without_ext = Path(filename_str).stem
        match = re.search(r'\d+', name_without_ext)
        return int(match.group()) if match else 0


    def format_dataset(self) -> Dict:
        # 1. 정제된 트랙렛 스캔
        tracklets = list_tracklets(str(self.filtered_dir))
        if not tracklets:
            raise ValueError(f"정제된 트랙렛을 찾을 수 없습니다: {self.filtered_dir}")

        # 2. global_id 검사 및 고유 ID 수집
        valid_tracklets = []
        global_ids = set()
        
        for t in tracklets:
            gid = t.get("global_id")
            if gid is None or gid == -1:
                raise ValueError(
                    f"트랙렛 '{t['tracklet_dir']}'에 global_id가 지정되지 않았습니다. "
                    f"먼저 scripts/merge_ids.py를 실행하여 ID 병합을 완료해야 합니다."
                )
            global_ids.add(gid)
            valid_tracklets.append(t)
            
        print(f"[INFO] 발견된 고유 글로벌 ID 수: {len(global_ids)}")
        print(f"[INFO] 총 유효 트랙렛 수: {len(valid_tracklets)}")

        # 3. Train/Test ID 분할 (Disjoint Split)
        sorted_gids = sorted(list(global_ids))
        random.seed(self.seed)
        random.shuffle(sorted_gids)
        
        split_idx = int(len(sorted_gids) * self.train_ratio)
        train_ids = set(sorted_gids[:split_idx])
        test_ids = set(sorted_gids[split_idx:])
        
        print(f"[INFO] 분할 결과 - Train ID: {len(train_ids)}개, Test ID: {len(test_ids)}개")

        # 4. 출력 디렉토리 초기화
        train_out = self.output_dir / "bounding_box_train"
        test_out = self.output_dir / "bounding_box_test"
        query_out = self.output_dir / "query"
        
        for d in [train_out, test_out, query_out]:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        # 5. 복사 및 파일명 규격화 변환
        stats = {
            "train_ids_count": len(train_ids),
            "test_ids_count": len(test_ids),
            "copied_train_images": 0,
            "copied_test_images": 0,
            "copied_query_images": 0
        }

        for t in tqdm(valid_tracklets, desc="Formatting to Market-1501"):
            tdir = Path(t["tracklet_dir"])
            gid = t["global_id"]
            cam = self._extract_number(t.get("camera", "c0"))
            slot = self._extract_number(t.get("time_slot", "t0"))
            crop_files = sorted(t.get("crop_files", []))
            
            if not crop_files:
                continue

            # Train ID 분기 (전체 이미지 -> Train 폴더)
            if gid in train_ids:
                for fname in crop_files:
                    src_path = tdir / fname
                    if not src_path.exists():
                        continue
                    
                    frame_num = self._parse_frame_num(fname)
                    dst_name = f"{gid:04d}_c{cam}s{slot}_{frame_num:06d}_00.jpg"
                    shutil.copy2(src_path, train_out / dst_name)
                    stats["copied_train_images"] += 1
            
            # Test ID 분기 (첫 프레임 -> Query, 나머지 -> Gallery/Test 폴더)
            else:
                for idx, fname in enumerate(crop_files):
                    src_path = tdir / fname
                    if not src_path.exists():
                        continue
                        
                    frame_num = self._parse_frame_num(fname)
                    dst_name = f"{gid:04d}_c{cam}s{slot}_{frame_num:06d}_00.jpg"
                    
                    if idx == 0:
                        shutil.copy2(src_path, query_out / dst_name)
                        stats["copied_query_images"] += 1
                    else:
                        shutil.copy2(src_path, test_out / dst_name)
                        stats["copied_test_images"] += 1

        print(f"[INFO] Market-1501 데이터셋 변환 성공: {self.output_dir}")
        return stats
