import argparse
import sys
from pathlib import Path
import yaml

# 로컬 src 모듈 경로 추가
sys.path.append(str(Path(__file__).resolve().parent.parent))
from pipeline.market_formatter import MarketFormatter

def main():
    parser = argparse.ArgumentParser(description="Convert filtered tracklets to Market-1501 dataset format")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to global config file"
    )
    args = parser.parse_args()

    # 1. Config 로드
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[ERROR] 설정을 찾을 수 없습니다: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. 기존 출력 폴더 확인 및 삭제
    import shutil
    output_dir = Path(config.get("market1501", {}).get("output_dir", "data/market1501"))
    if output_dir.exists():
        ans = input(f"[WARN] '{output_dir}' 폴더가 이미 존재합니다. 삭제하고 새로 생성할까요? [y/N] ").strip().lower()
        if ans != "y":
            print("취소되었습니다.")
            sys.exit(0)
        shutil.rmtree(output_dir)
        print(f"[INFO] '{output_dir}' 삭제 완료")

    # 3. Formatter 초기화 및 실행
    print("[INFO] Market-1501 데이터셋 포맷 변환 시작...")
    formatter = MarketFormatter(config)
    
    try:
        stats = formatter.format_dataset()
    except ValueError as ve:
        print(f"[ERROR] 변환 실패: {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] 예상치 못한 변환 오류 발생: {e}")
        sys.exit(1)

    # 3. 최종 결과 요약 리포트 출력
    print("\n" + "="*50)
    print("      🎉 Market-1501 포맷 변환 완료 리포트      ")
    print("="*50)
    print(f" - Train 고유 ID 개수      : {stats['train_ids_count']} 개")
    print(f" - Test 고유 ID 개수       : {stats['test_ids_count']} 개")
    print(f" - 복사된 Train 이미지 수  : {stats['copied_train_images']} 장 (bounding_box_train)")
    print(f" - 복사된 Gallery 이미지 수: {stats['copied_test_images']} 장 (bounding_box_test)")
    print(f" - 복사된 Query 이미지 수  : {stats['copied_query_images']} 장 (query)")
    print("-"*50)
    print(f" - 총 이미지 파일 개수     : {stats['copied_train_images'] + stats['copied_test_images'] + stats['copied_query_images']} 장")
    print("="*50)

if __name__ == "__main__":
    main()
