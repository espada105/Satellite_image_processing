"""
SpaceNet SN8_floods 데이터 처리 파이프라인

1. 압축 해제 (.tar.gz)
2. 이미지 및 라벨 구조 확인
3. 메타데이터 추출
4. 이미지 전처리 (리사이즈, 형식 변환)
5. 캡션 생성 (선택)
6. 데이터베이스 준비용 구조로 정리
"""

import argparse
import tarfile
import json
from pathlib import Path
from typing import Dict, List, Optional
import shutil
from PIL import Image
import sys
import os

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.utils.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    METADATA_DIR
)


def extract_tar_files(tar_dir: Path, extract_dir: Path, remove_tar: bool = False):
    """
    tar.gz 파일들을 압축 해제
    
    Args:
        tar_dir: tar.gz 파일들이 있는 디렉토리
        extract_dir: 압축 해제할 디렉토리
        remove_tar: 압축 해제 후 tar.gz 파일 삭제 여부
    """
    tar_files = list(tar_dir.glob("*.tar.gz"))
    
    if not tar_files:
        print("❌ 압축 해제할 tar.gz 파일을 찾을 수 없습니다.")
        return False
    
    print(f"📦 {len(tar_files)}개의 tar.gz 파일 발견")
    
    for tar_file in tar_files:
        print(f"\n📥 압축 해제 중: {tar_file.name}")
        
        # 각 데이터셋별 디렉토리 생성
        dataset_name = tar_file.stem.replace("_Public", "")
        extract_path = extract_dir / dataset_name
        
        try:
            with tarfile.open(tar_file, 'r:gz') as tar:
                tar.extractall(extract_path)
                print(f"✅ 압축 해제 완료: {extract_path}")
            
            if remove_tar:
                tar_file.unlink()
                print(f"🗑️  원본 파일 삭제: {tar_file.name}")
        
        except Exception as e:
            print(f"❌ 압축 해제 실패: {tar_file.name}")
            print(f"   오류: {e}")
            continue
    
    return True


def explore_spacenet_structure(extracted_dir: Path):
    """
    SpaceNet 데이터셋 구조 탐색
    
    SpaceNet SN8_floods 구조 예시:
    Germany_Training_Public/
    ├── annotated/
    │   └── flooding_annotations/
    │       └── *.geojson
    ├── RGB-PanSharpen/
    │   └── *.tif (RGB 이미지)
    └── ...
    """
    print("\n🔍 데이터셋 구조 탐색 중...")
    
    structure_info = {}
    
    for dataset_dir in extracted_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        print(f"\n📁 {dataset_dir.name}:")
        structure_info[dataset_dir.name] = {}
        
        # 주요 디렉토리 찾기
        subdirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        for subdir in subdirs:
            print(f"  - {subdir.name}/")
            
            # 이미지 파일 찾기
            image_files = []
            for ext in ['*.tif', '*.tiff', '*.jpg', '*.png']:
                image_files.extend(list(subdir.rglob(ext)))
            
            if image_files:
                structure_info[dataset_dir.name][subdir.name] = len(image_files)
                print(f"    → {len(image_files)}개 이미지 파일")
    
    return structure_info


def extract_metadata_from_spacenet(extracted_dir: Path, output_dir: Path):
    """
    SpaceNet 데이터셋에서 메타데이터 추출
    
    Args:
        extracted_dir: 압축 해제된 데이터 디렉토리
        output_dir: 메타데이터를 저장할 디렉토리
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_list = []
    
    print("\n📋 메타데이터 추출 중...")
    
    # SpaceNet SN8 구조: PRE-event, POST-event 디렉토리
    image_patterns = ['PRE-event', 'POST-event', 'RGB-PanSharpen', 'RGB', 'images']
    
    for dataset_dir in extracted_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        print(f"\n처리 중: {dataset_name}")
        
        # 여러 이미지 디렉토리 처리 (PRE-event, POST-event 등)
        image_dirs = []
        for pattern in image_patterns:
            potential_dir = dataset_dir / pattern
            if potential_dir.exists() and potential_dir.is_dir():
                image_dirs.append((pattern, potential_dir))
        
        if not image_dirs:
            print(f"  ⚠️  이미지 디렉토리를 찾을 수 없습니다.")
            continue
        
        print(f"  📁 발견된 이미지 디렉토리: {', '.join([d[0] for d in image_dirs])}")
        
        # 모든 이미지 디렉토리에서 파일 수집
        all_image_files = []
        for pattern, img_dir in image_dirs:
            image_files = list(img_dir.rglob("*.tif")) + list(img_dir.rglob("*.tiff")) + \
                         list(img_dir.rglob("*.png")) + list(img_dir.rglob("*.jpg"))
            all_image_files.extend(image_files)
            print(f"    {pattern}: {len(image_files)}개 이미지")
        
        print(f"  📸 총 {len(all_image_files)}개 이미지 발견")
        
        for img_path in all_image_files:
            # 이미지 정보 추출
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                
                # 파일명에서 정보 추출 (예: sn8_Germany_Train_AOI_03_06_img123.tif)
                image_id = img_path.stem
                
                # PRE-event 또는 POST-event 구분
                event_type = None
                if "PRE-event" in str(img_path):
                    event_type = "PRE-event"
                elif "POST-event" in str(img_path):
                    event_type = "POST-event"
                
                metadata = {
                    "image_id": f"{dataset_name}_{image_id}",
                    "original_path": str(img_path.relative_to(extracted_dir)),
                    "image_path": str(img_path),
                    "dataset": dataset_name,
                    "location": _extract_location_from_name(dataset_name),
                    "event_type": event_type,  # PRE-event 또는 POST-event
                    "width": width,
                    "height": height,
                    "format": img_path.suffix[1:].upper(),
                    "satellite_type": "SpaceNet",
                    "date": None,
                    "resolution": "0.31m/pixel",  # SpaceNet SN8 해상도
                    "description": f"SpaceNet SN8_floods - {dataset_name} - {event_type or 'unknown'}"
                }
                
                metadata_list.append(metadata)
            
            except Exception as e:
                print(f"  ⚠️  이미지 처리 실패: {img_path.name} - {e}")
                continue
        
        # 데이터셋별 메타데이터 저장
        metadata_file = output_dir / f"{dataset_name}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_list[-len(image_files):], f, ensure_ascii=False, indent=2)
        
        print(f"  ✅ 메타데이터 저장: {metadata_file}")
    
    return metadata_list


def _extract_location_from_name(dataset_name: str) -> Dict:
    """데이터셋 이름에서 위치 정보 추출"""
    location_map = {
        "Germany_Training": {
            "name": "Germany",
            "coordinates": {"lat": 51.1657, "lon": 10.4515}  # 독일 중심부
        },
        "Louisiana-East_Training": {
            "name": "Louisiana-East",
            "coordinates": {"lat": 30.4581, "lon": -90.1406}
        },
        "Louisiana-West_Test": {
            "name": "Louisiana-West",
            "coordinates": {"lat": 30.2241, "lon": -93.2144}
        }
    }
    
    for key, location in location_map.items():
        if key in dataset_name:
            return location
    
    return {"name": dataset_name, "coordinates": {"lat": None, "lon": None}}


def preprocess_images(
    extracted_dir: Path,
    output_dir: Path,
    target_size: tuple = (512, 512),
    format: str = "JPEG",
    max_images: Optional[int] = None
):
    """
    이미지 전처리 (리사이즈, 형식 변환)
    
    Args:
        extracted_dir: 압축 해제된 데이터 디렉토리
        output_dir: 전처리된 이미지를 저장할 디렉토리
        target_size: 리사이즈할 크기 (width, height)
        format: 저장할 이미지 형식 (JPEG, PNG)
        max_images: 최대 처리할 이미지 수 (None이면 전체)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🖼️  이미지 전처리 시작 (크기: {target_size[0]}x{target_size[1]}, 형식: {format})")
    
    # SpaceNet SN8 구조: PRE-event, POST-event 디렉토리
    image_patterns = ['PRE-event', 'POST-event', 'RGB-PanSharpen', 'RGB', 'images']
    processed_count = 0
    
    for dataset_dir in extracted_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        print(f"\n처리 중: {dataset_name}")
        
        # 여러 이미지 디렉토리 찾기
        image_dirs = []
        for pattern in image_patterns:
            potential_dir = dataset_dir / pattern
            if potential_dir.exists() and potential_dir.is_dir():
                image_dirs.append((pattern, potential_dir))
        
        if not image_dirs:
            print(f"  ⚠️  이미지 디렉토리를 찾을 수 없습니다.")
            continue
        
        print(f"  📁 발견된 이미지 디렉토리: {', '.join([d[0] for d in image_dirs])}")
        
        # 출력 디렉토리 생성
        dataset_output_dir = output_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 모든 이미지 디렉토리에서 파일 수집
        all_image_files = []
        for pattern, img_dir in image_dirs:
            image_files = list(img_dir.rglob("*.tif")) + list(img_dir.rglob("*.tiff")) + \
                         list(img_dir.rglob("*.png")) + list(img_dir.rglob("*.jpg"))
            all_image_files.extend((pattern, img) for img in image_files)
        
        if max_images:
            all_image_files = all_image_files[:max_images]
        
        print(f"  📸 {len(all_image_files)}개 이미지 처리 예정")
        
        for idx, (pattern, img_path) in enumerate(all_image_files, 1):
            if max_images and processed_count >= max_images:
                break
            
            try:
                # 이미지 로드 및 전처리
                with Image.open(img_path) as img:
                    # RGB 변환 (필요한 경우)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # 리사이즈
                    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                    
                    # 저장 (event_type 포함)
                    output_filename = f"{dataset_name}_{pattern}_{img_path.stem}.{format.lower()}"
                    output_path = dataset_output_dir / output_filename
                    
                    img_resized.save(output_path, format=format, quality=95)
                
                processed_count += 1
                
                if idx % 50 == 0:
                    print(f"  진행: {idx}/{len(all_image_files)} (전체: {processed_count}개)")
            
            except Exception as e:
                print(f"  ⚠️  이미지 처리 실패: {img_path.name} - {e}")
                continue
    
    print(f"\n✅ 이미지 전처리 완료: 총 {processed_count}개 처리됨")
    return processed_count


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="SpaceNet SN8_floods 데이터 처리 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
처리 단계:
  1. 압축 해제
  2. 구조 탐색
  3. 메타데이터 추출
  4. 이미지 전처리

예제:
  # 전체 파이프라인 실행
  python process_spacenet_data.py --tar_dir ./data/raw --extract_dir ./data/extracted

  # 압축 해제만
  python process_spacenet_data.py --tar_dir ./data/raw --extract_dir ./data/extracted --step extract

  # 전처리만 (샘플 100개)
  python process_spacenet_data.py --extract_dir ./data/extracted --step preprocess --max_images 100
        """
    )
    
    parser.add_argument(
        "--tar_dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help="tar.gz 파일들이 있는 디렉토리"
    )
    
    parser.add_argument(
        "--extract_dir",
        type=str,
        default="./data/extracted",
        help="압축 해제할 디렉토리"
    )
    
    parser.add_argument(
        "--step",
        choices=["all", "extract", "explore", "metadata", "preprocess"],
        default="all",
        help="실행할 단계 (기본값: all)"
    )
    
    parser.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="이미지 리사이즈 크기 (기본값: 512 512)"
    )
    
    parser.add_argument(
        "--format",
        choices=["JPEG", "PNG"],
        default="JPEG",
        help="저장할 이미지 형식 (기본값: JPEG)"
    )
    
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="최대 처리할 이미지 수 (기본값: 전체)"
    )
    
    parser.add_argument(
        "--remove_tar",
        action="store_true",
        help="압축 해제 후 tar.gz 파일 삭제"
    )
    
    args = parser.parse_args()
    
    tar_dir = Path(args.tar_dir)
    extract_dir = Path(args.extract_dir)
    
    print("=" * 60)
    print("SpaceNet SN8_floods 데이터 처리 파이프라인")
    print("=" * 60)
    
    # 1. 압축 해제
    if args.step in ["all", "extract"]:
        print("\n" + "=" * 60)
        print("1단계: 압축 해제")
        print("=" * 60)
        extract_tar_files(tar_dir, extract_dir, args.remove_tar)
    
    # 2. 구조 탐색
    if args.step in ["all", "explore"] and extract_dir.exists():
        print("\n" + "=" * 60)
        print("2단계: 데이터셋 구조 탐색")
        print("=" * 60)
        explore_spacenet_structure(extract_dir)
    
    # 3. 메타데이터 추출
    if args.step in ["all", "metadata"] and extract_dir.exists():
        print("\n" + "=" * 60)
        print("3단계: 메타데이터 추출")
        print("=" * 60)
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        extract_metadata_from_spacenet(extract_dir, METADATA_DIR)
    
    # 4. 이미지 전처리
    if args.step in ["all", "preprocess"] and extract_dir.exists():
        print("\n" + "=" * 60)
        print("4단계: 이미지 전처리")
        print("=" * 60)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        preprocess_images(
            extract_dir,
            PROCESSED_DATA_DIR,
            target_size=tuple(args.target_size),
            format=args.format,
            max_images=args.max_images
        )
    
    print("\n" + "=" * 60)
    print("✅ 모든 처리 단계 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

