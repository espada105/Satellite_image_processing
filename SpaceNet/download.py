"""
SpaceNet SN8_floods 데이터셋 다운로드 스크립트

AWS S3에서 SpaceNet 데이터를 다운로드합니다.
AWS CLI가 설치되어 있어야 합니다.

사용법:
    python download.py --output_dir ./data/raw --datasets all
    python download.py --output_dir ./data/raw --datasets training
    python download.py --output_dir ./data/raw --datasets germany louisiana-east
"""

import argparse
import subprocess
import os
from pathlib import Path
import sys

# SpaceNet S3 버킷 경로
S3_BUCKET = "s3://spacenet-dataset/spacenet/SN8_floods/tarballs/"

# 전체 데이터셋 목록
ALL_DATASETS = {
    "germany": "Germany_Training_Public.tar.gz",
    "louisiana-east": "Louisiana-East_Training_Public.tar.gz",
    "louisiana-west": "Louisiana-West_Test_Public.tar.gz"
}

TRAINING_DATASETS = ["germany", "louisiana-east"]
TEST_DATASETS = ["louisiana-west"]


def check_aws_cli():
    """AWS CLI가 설치되어 있는지 확인"""
    try:
        result = subprocess.run(
            ["aws", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ AWS CLI 확인: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ AWS CLI가 설치되어 있지 않습니다.")
        print("설치 방법:")
        print("  Windows: https://aws.amazon.com/cli/")
        print("  Linux: sudo apt-get install awscli")
        print("  Mac: brew install awscli")
        return False


def download_dataset(dataset_name: str, output_dir: Path, s3_bucket: str = S3_BUCKET):
    """
    단일 데이터셋 다운로드
    
    Args:
        dataset_name: 데이터셋 이름 (germany, louisiana-east, louisiana-west)
        output_dir: 다운로드할 디렉토리
        s3_bucket: S3 버킷 경로
    """
    if dataset_name not in ALL_DATASETS:
        print(f"❌ 알 수 없는 데이터셋: {dataset_name}")
        print(f"사용 가능한 데이터셋: {', '.join(ALL_DATASETS.keys())}")
        return False
    
    filename = ALL_DATASETS[dataset_name]
    s3_path = f"{s3_bucket}{filename}"
    output_path = output_dir / filename
    
    # 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 다운로드 시작: {dataset_name}")
    print(f"   S3 경로: {s3_path}")
    print(f"   저장 위치: {output_path}")
    
    try:
        # AWS CLI를 사용하여 다운로드
        # --no-sign-request: AWS 계정 없이 공개 데이터 접근
        result = subprocess.run(
            [
                "aws", "s3", "cp",
                s3_path,
                str(output_path),
                "--no-sign-request"
            ],
            check=True,
            capture_output=True,
            text=True
        )
        
        print(f"✅ 다운로드 완료: {filename}")
        print(f"   파일 크기: {output_path.stat().st_size / (1024**3):.2f} GB")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 다운로드 실패: {filename}")
        print(f"   오류: {e.stderr}")
        return False


def list_available_datasets(s3_bucket: str = S3_BUCKET):
    """S3에서 사용 가능한 데이터셋 목록 조회"""
    print(f"\n🔍 데이터셋 목록 조회: {s3_bucket}")
    
    try:
        result = subprocess.run(
            [
                "aws", "s3", "ls",
                s3_bucket,
                "--no-sign-request"
            ],
            check=True,
            capture_output=True,
            text=True
        )
        
        print("사용 가능한 데이터셋:")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 목록 조회 실패: {e.stderr}")
        return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="SpaceNet SN8_floods 데이터셋 다운로드",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 모든 데이터셋 다운로드
  python download.py --output_dir ./data/raw --datasets all
  
  # 훈련 데이터만 다운로드
  python download.py --output_dir ./data/raw --datasets training
  
  # 특정 데이터셋 다운로드
  python download.py --output_dir ./data/raw --datasets germany louisiana-east
        """
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/raw",
        help="다운로드할 디렉토리 (기본값: ./data/raw)"
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", "training", "testing"] + list(ALL_DATASETS.keys()),
        help="다운로드할 데이터셋 (기본값: all)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="사용 가능한 데이터셋 목록만 조회"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SpaceNet SN8_floods 데이터셋 다운로드")
    print("=" * 60)
    
    # AWS CLI 확인
    if not check_aws_cli():
        sys.exit(1)
    
    # 목록만 조회하는 경우
    if args.list:
        list_available_datasets()
        return
    
    # 다운로드할 데이터셋 결정
    datasets_to_download = []
    
    if "all" in args.datasets:
        datasets_to_download = list(ALL_DATASETS.keys())
    elif "training" in args.datasets:
        datasets_to_download = TRAINING_DATASETS
    elif "testing" in args.datasets:
        datasets_to_download = TEST_DATASETS
    else:
        datasets_to_download = args.datasets
    
    # 출력 디렉토리 설정
    output_dir = Path(args.output_dir).resolve()
    
    print(f"\n📂 출력 디렉토리: {output_dir}")
    print(f"📦 다운로드할 데이터셋: {', '.join(datasets_to_download)}")
    
    # 각 데이터셋 다운로드
    success_count = 0
    total_count = len(datasets_to_download)
    
    for dataset in datasets_to_download:
        if download_dataset(dataset, output_dir):
            success_count += 1
    
    # 결과 요약
    print("\n" + "=" * 60)
    print(f"다운로드 완료: {success_count}/{total_count}")
    print("=" * 60)
    
    if success_count == total_count:
        print("\n✅ 모든 데이터셋이 성공적으로 다운로드되었습니다!")
        print(f"\n다운로드된 파일 위치: {output_dir}")
        print("\n압축 해제 방법:")
        print(f"  cd {output_dir}")
        print("  tar -xzf Germany_Training_Public.tar.gz")
        print("  tar -xzf Louisiana-East_Training_Public.tar.gz")
        print("  tar -xzf Louisiana-West_Test_Public.tar.gz")
    else:
        print(f"\n⚠️  일부 데이터셋 다운로드 실패 ({total_count - success_count}개)")
        sys.exit(1)


if __name__ == "__main__":
    main()

