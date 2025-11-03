"""
이미지 캡션 생성 스크립트

BLIP-2 모델을 사용하여 전처리된 위성 이미지에 대한 캡션을 자동 생성합니다.

사용법:
    python generate_captions.py --image_dir ./data/processed --output_file ./data/captions/captions.json
    python generate_captions.py --image_dir ./data/processed --output_file ./data/captions/captions.json --batch_size 8 --max_images 100
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
from PIL import Image
import sys
import os

# PyTorch 및 transformers 임포트 (에러 처리)
try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  필요한 라이브러리가 설치되지 않았습니다: {e}")
    print("\n설치 방법:")
    print("  pip install torch torchvision transformers pillow tqdm")
    TRANSFORMERS_AVAILABLE = False
    sys.exit(1)

from tqdm import tqdm

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.utils.config import (
    PROCESSED_DATA_DIR,
    CAPTIONS_DIR
)


def load_caption_model(model_name: str = "Salesforce/blip-image-captioning-large", device: str = None):
    """
    BLIP 캡션 생성 모델 로드
    
    Args:
        model_name: 사용할 모델 이름
        device: 사용할 디바이스 (None이면 자동 선택)
    
    Returns:
        processor, model
    """
    print(f"🤖 모델 로딩 중: {model_name}")
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"   디바이스: {device}")
    
    try:
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
        model.eval()
        
        print(f"✅ 모델 로드 완료")
        return processor, model, device
    
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        raise


def generate_caption_for_image(
    image_path: Path,
    processor,
    model,
    device: str,
    max_length: int = 50,
    num_beams: int = 5
) -> str:
    """
    단일 이미지에 대한 캡션 생성
    
    Args:
        image_path: 이미지 파일 경로
        processor: BLIP processor
        model: BLIP model
        device: 디바이스
        max_length: 최대 생성 길이
        num_beams: 빔 서치 개수
    
    Returns:
        생성된 캡션 텍스트
    """
    try:
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        
        # 캡션 생성
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=False
            )
        
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption.strip()
    
    except Exception as e:
        print(f"⚠️  이미지 처리 실패: {image_path.name} - {e}")
        return None


def generate_captions_batch(
    image_paths: List[Path],
    processor,
    model,
    device: str,
    batch_size: int = 8,
    max_length: int = 50
) -> List[Dict]:
    """
    배치로 이미지 캡션 생성
    
    Args:
        image_paths: 이미지 파일 경로 리스트
        processor: BLIP processor
        model: BLIP model
        device: 디바이스
        batch_size: 배치 크기
        max_length: 최대 생성 길이
    
    Returns:
        캡션 정보 리스트
    """
    results = []
    
    print(f"\n📝 캡션 생성 시작 (총 {len(image_paths)}개 이미지)")
    
    # 배치 처리
    for i in tqdm(range(0, len(image_paths), batch_size), desc="캡션 생성"):
        batch = image_paths[i:i + batch_size]
        batch_results = []
        
        for image_path in batch:
            caption = generate_caption_for_image(
                image_path,
                processor,
                model,
                device,
                max_length=max_length
            )
            
            if caption:
                # 윈도우/절대경로 혼합 환경에서도 안전하게 상대경로 계산
                try:
                    rel_path = image_path.resolve().relative_to(PROCESSED_DATA_DIR.resolve())
                except Exception:
                    rel_path = image_path

                batch_results.append({
                    "image_id": image_path.stem,
                    "image_path": str(rel_path).replace('\\', '/'),
                    "caption": caption
                })
        
        results.extend(batch_results)
    
    return results


def find_all_images(image_dir: Path) -> List[Path]:
    """
    모든 이미지 파일 찾기
    
    Args:
        image_dir: 이미지 디렉토리
    
    Returns:
        이미지 파일 경로 리스트
    """
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(image_dir.rglob(f"*{ext}")))
        image_files.extend(list(image_dir.rglob(f"*{ext.upper()}")))
    
    return sorted(image_files)


def load_metadata_for_captions(captions: List[Dict], metadata_dir: Path) -> List[Dict]:
    """
    메타데이터 정보를 캡션에 추가
    
    Args:
        captions: 캡션 리스트
        metadata_dir: 메타데이터 디렉토리
    
    Returns:
        메타데이터가 추가된 캡션 리스트
    """
    # 메타데이터 파일 로드
    metadata_files = list(metadata_dir.glob("*_metadata.json"))
    metadata_dict = {}
    
    for metadata_file in metadata_files:
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
                for meta in metadata_list:
                    # 이미지 경로를 기반으로 매칭
                    if 'image_path' in meta:
                        metadata_dict[meta['image_id']] = meta
        except Exception as e:
            print(f"⚠️  메타데이터 로드 실패: {metadata_file.name} - {e}")
    
    # 캡션에 메타데이터 추가
    enriched_captions = []
    for caption_item in captions:
        image_id = caption_item['image_id']
        
        # 메타데이터 찾기 (여러 방법으로 시도)
        metadata = None
        
        # 직접 매칭
        if image_id in metadata_dict:
            metadata = metadata_dict[image_id]
        else:
            # 부분 매칭 시도
            for meta_id, meta in metadata_dict.items():
                if image_id in meta_id or meta_id in image_id:
                    metadata = meta
                    break
        
        enriched_item = caption_item.copy()
        if metadata:
            enriched_item['metadata'] = {
                'location': metadata.get('location', {}),
                'event_type': metadata.get('event_type'),
                'dataset': metadata.get('dataset'),
                'resolution': metadata.get('resolution')
            }
        
        enriched_captions.append(enriched_item)
    
    return enriched_captions


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="이미지 캡션 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 전체 이미지 캡션 생성
  python generate_captions.py --image_dir ./data/processed --output_file ./data/captions/captions.json

  # 테스트용 (100개만)
  python generate_captions.py --image_dir ./data/processed --output_file ./data/captions/captions_test.json --max_images 100

  # GPU 사용, 배치 크기 조정
  python generate_captions.py --image_dir ./data/processed --output_file ./data/captions/captions.json --batch_size 16 --device cuda
        """
    )
    
    parser.add_argument(
        "--image_dir",
        type=str,
        default=str(PROCESSED_DATA_DIR),
        help="전처리된 이미지 디렉토리 (기본값: ./data/processed)"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(CAPTIONS_DIR / "captions.json"),
        help="캡션 출력 파일 (기본값: ./data/captions/captions.json)"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="Salesforce/blip-image-captioning-large",
        help="사용할 BLIP 모델 이름"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="배치 크기 (기본값: 8)"
    )
    
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="최대 처리할 이미지 수 (기본값: 전체)"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="캡션 최대 길이 (기본값: 50)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help="사용할 디바이스 (기본값: 자동 선택)"
    )
    
    parser.add_argument(
        "--include_metadata",
        action="store_true",
        help="메타데이터 정보를 캡션에 포함"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("이미지 캡션 생성")
    print("=" * 60)
    
    # 디렉토리 및 파일 경로 설정
    image_dir = Path(args.image_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 이미지 파일 찾기
    print(f"\n🔍 이미지 파일 찾는 중: {image_dir}")
    all_images = find_all_images(image_dir)
    
    if not all_images:
        print("❌ 이미지 파일을 찾을 수 없습니다!")
        sys.exit(1)
    
    print(f"✅ {len(all_images)}개 이미지 파일 발견")
    
    # 최대 이미지 수 제한
    if args.max_images:
        all_images = all_images[:args.max_images]
        print(f"📌 제한 적용: {len(all_images)}개 이미지 처리")
    
    # 모델 로드
    processor, model, device = load_caption_model(args.model_name, args.device)
    
    # 캡션 생성
    captions = generate_captions_batch(
        all_images,
        processor,
        model,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    print(f"\n✅ 캡션 생성 완료: {len(captions)}개")
    
    # 메타데이터 추가 (선택적)
    if args.include_metadata:
        print("\n📋 메타데이터 추가 중...")
        from scripts.utils.config import METADATA_DIR
        captions = load_metadata_for_captions(captions, METADATA_DIR)
        print("✅ 메타데이터 추가 완료")
    
    # 결과 저장
    print(f"\n💾 캡션 저장 중: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 캡션 저장 완료!")
    
    # 통계 출력
    print("\n" + "=" * 60)
    print("생성 완료 통계")
    print("=" * 60)
    print(f"처리된 이미지: {len(captions)}개")
    print(f"저장 위치: {output_file}")
    
    # 샘플 캡션 출력
    if captions:
        print(f"\n📝 샘플 캡션 (처음 3개):")
        for i, item in enumerate(captions[:3], 1):
            print(f"\n{i}. 이미지: {item['image_id']}")
            print(f"   캡션: {item['caption']}")


if __name__ == "__main__":
    main()

