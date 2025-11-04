"""
벡터 임베딩 생성 스크립트

생성된 캡션 텍스트를 벡터 임베딩으로 변환합니다.
sentence-transformers를 사용하여 텍스트를 벡터로 변환합니다.

사용법:
    python generate_embeddings.py --captions_file ./data/captions/captions.json --output_file ./data/embeddings/embeddings.npy
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import sys
import os

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.utils.config import (
    CAPTIONS_DIR,
    EMBEDDINGS_DIR
)

# sentence-transformers 임포트
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  sentence-transformers가 설치되지 않았습니다: {e}")
    print("\n설치 방법:")
    print("  pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    sys.exit(1)


def load_captions(captions_file: Path) -> List[Dict]:
    """
    캡션 JSON 파일 로드
    
    Args:
        captions_file: 캡션 JSON 파일 경로
    
    Returns:
        캡션 리스트
    """
    print(f"📖 캡션 파일 로드 중: {captions_file}")
    
    with open(captions_file, 'r', encoding='utf-8') as f:
        captions = json.load(f)
    
    print(f"✅ {len(captions)}개 캡션 로드 완료")
    return captions


def generate_embeddings(
    captions: List[Dict],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    device: str = None
) -> np.ndarray:
    """
    캡션 텍스트를 벡터 임베딩으로 변환
    
    Args:
        captions: 캡션 리스트
        model_name: 사용할 임베딩 모델 이름
        batch_size: 배치 크기
        device: 사용할 디바이스 (None이면 자동 선택)
    
    Returns:
        벡터 임베딩 배열 (n_samples, embedding_dim)
    """
    print(f"\n🤖 임베딩 모델 로드 중: {model_name}")
    
    if device is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"   디바이스: {device}")
    
    # 모델 로드
    model = SentenceTransformer(model_name, device=device)
    
    # 임베딩 차원 확인
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"   임베딩 차원: {embedding_dim}")
    
    # 캡션 텍스트 추출
    caption_texts = [item['caption'] for item in captions]
    
    print(f"\n📝 벡터 임베딩 생성 시작 (총 {len(caption_texts)}개)")
    
    # 배치로 임베딩 생성
    embeddings = model.encode(
        caption_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"✅ 임베딩 생성 완료: {embeddings.shape}")
    
    return embeddings


def save_embeddings(
    embeddings: np.ndarray,
    captions: List[Dict],
    output_file: Path
):
    """
    임베딩을 파일로 저장
    
    Args:
        embeddings: 벡터 임베딩 배열
        captions: 캡션 리스트 (메타데이터 포함)
        output_file: 출력 파일 경로
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 임베딩 저장 중: {output_file}")
    
    # NumPy 배열로 저장
    np.save(output_file, embeddings)
    
    # 메타데이터 파일도 저장 (JSON)
    metadata_file = output_file.with_suffix('.json')
    
    metadata = {
        "embedding_file": str(output_file),
        "embedding_dim": embeddings.shape[1],
        "num_samples": embeddings.shape[0],
        "model": "all-MiniLM-L6-v2",
        "captions_count": len(captions),
        "sample_captions": captions[:3] if captions else []
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 임베딩 저장 완료")
    print(f"   NumPy 배열: {output_file} ({embeddings.nbytes / (1024**2):.2f} MB)")
    print(f"   메타데이터: {metadata_file}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="캡션 텍스트를 벡터 임베딩으로 변환",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 전체 캡션 임베딩 생성
  python generate_embeddings.py --captions_file ./data/captions/captions.json --output_file ./data/embeddings/embeddings.npy

  # 테스트용 (100개만)
  python generate_embeddings.py --captions_file ./data/captions/captions.json --output_file ./data/embeddings/test_embeddings.npy --max_samples 100

  # GPU 사용, 배치 크기 증가
  python generate_embeddings.py --captions_file ./data/captions/captions.json --output_file ./data/embeddings/embeddings.npy --batch_size 64 --device cuda
        """
    )
    
    parser.add_argument(
        "--captions_file",
        type=str,
        default=str(CAPTIONS_DIR / "captions.json"),
        help="캡션 JSON 파일 경로 (기본값: ./data/captions/captions.json)"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default=str(EMBEDDINGS_DIR / "embeddings.npy"),
        help="출력 임베딩 파일 경로 (기본값: ./data/embeddings/embeddings.npy)"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="사용할 임베딩 모델 이름 (기본값: all-MiniLM-L6-v2)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="배치 크기 (기본값: 32)"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="최대 처리할 샘플 수 (기본값: 전체)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help="사용할 디바이스 (기본값: 자동 선택)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("벡터 임베딩 생성")
    print("=" * 60)
    
    # 캡션 로드
    captions_file = Path(args.captions_file)
    if not captions_file.exists():
        print(f"❌ 캡션 파일을 찾을 수 없습니다: {captions_file}")
        sys.exit(1)
    
    captions = load_captions(captions_file)
    
    # 샘플 제한 적용
    if args.max_samples:
        captions = captions[:args.max_samples]
        print(f"📌 제한 적용: {len(captions)}개 샘플 처리")
    
    # 임베딩 생성
    embeddings = generate_embeddings(
        captions,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # 저장
    output_file = Path(args.output_file)
    save_embeddings(embeddings, captions, output_file)
    
    # 통계 출력
    print("\n" + "=" * 60)
    print("생성 완료 통계")
    print("=" * 60)
    print(f"처리된 캡션: {len(captions)}개")
    print(f"임베딩 차원: {embeddings.shape[1]}차원")
    print(f"임베딩 크기: {embeddings.shape[0]} x {embeddings.shape[1]}")
    print(f"저장 위치: {output_file}")
    print(f"파일 크기: {embeddings.nbytes / (1024**2):.2f} MB")


if __name__ == "__main__":
    main()

