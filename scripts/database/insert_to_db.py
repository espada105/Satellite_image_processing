"""
데이터베이스에 이미지와 임베딩 삽입 스크립트

캡션과 벡터 임베딩을 PostgreSQL + pgvector 데이터베이스에 저장합니다.

사용법:
    python insert_to_db.py --captions_file ./data/captions/captions.json --embeddings_file ./data/embeddings/embeddings.npy
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import sys
import os
from tqdm import tqdm

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.database.setup_database import get_db_connection
from scripts.utils.config import (
    POSTGRES_CONFIG,
    PROCESSED_DATA_DIR
)

try:
    import psycopg2
    from pgvector.psycopg2 import register_vector
    PSYCOPG2_AVAILABLE = True
except ImportError:
    print("⚠️  psycopg2가 설치되지 않았습니다.")
    print("설치 방법: pip install psycopg2-binary pgvector")
    PSYCOPG2_AVAILABLE = False
    sys.exit(1)


def load_captions_and_embeddings(
    captions_file: Path,
    embeddings_file: Path
) -> tuple:
    """
    캡션과 임베딩 파일 로드
    
    Args:
        captions_file: 캡션 JSON 파일 경로
        embeddings_file: 임베딩 NumPy 파일 경로
    
    Returns:
        (captions, embeddings) 튜플
    """
    print(f"📖 캡션 파일 로드 중: {captions_file}")
    with open(captions_file, 'r', encoding='utf-8') as f:
        captions = json.load(f)
    
    print(f"📊 임베딩 파일 로드 중: {embeddings_file}")
    embeddings = np.load(embeddings_file)
    
    print(f"✅ 로드 완료: {len(captions)}개 캡션, {embeddings.shape} 임베딩")
    
    # 개수 일치 확인
    if len(captions) != embeddings.shape[0]:
        print(f"⚠️  경고: 캡션 개수({len(captions)})와 임베딩 개수({embeddings.shape[0]})가 일치하지 않습니다.")
        min_len = min(len(captions), embeddings.shape[0])
        captions = captions[:min_len]
        embeddings = embeddings[:min_len]
        print(f"   {min_len}개로 제한하여 처리합니다.")
    
    return captions, embeddings


def insert_to_database(
    captions: List[Dict],
    embeddings: np.ndarray,
    batch_size: int = 100,
    skip_existing: bool = True
):
    """
    캡션과 임베딩을 데이터베이스에 삽입
    
    Args:
        captions: 캡션 리스트
        embeddings: 벡터 임베딩 배열
        batch_size: 배치 크기
        skip_existing: 기존 레코드 건너뛰기 여부
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    print(f"\n💾 데이터베이스 삽입 시작 (총 {len(captions)}개)")
    
    inserted_count = 0
    skipped_count = 0
    error_count = 0
    
    try:
        # 배치로 삽입
        for i in tqdm(range(0, len(captions), batch_size), desc="DB 삽입"):
            batch_captions = captions[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            for caption_item, embedding in zip(batch_captions, batch_embeddings):
                try:
                    # 이미지 경로 구성
                    image_path = PROCESSED_DATA_DIR / caption_item['image_path']
                    if not image_path.exists():
                        # 상대 경로로 시도
                        image_path = Path(caption_item['image_path'])
                    
                    # 메타데이터 추출
                    metadata = caption_item.get('metadata', {})
                    location = metadata.get('location', {}) if isinstance(metadata, dict) else {}
                    location_name = location.get('name', '') if isinstance(location, dict) else ''
                    coordinates = location.get('coordinates', {}) if isinstance(location, dict) else {}
                    latitude = coordinates.get('lat') if isinstance(coordinates, dict) else None
                    longitude = coordinates.get('lon') if isinstance(coordinates, dict) else None
                    
                    # 이벤트 타입 추출
                    event_type = metadata.get('event_type', '') if isinstance(metadata, dict) else ''
                    
                    # 기존 레코드 확인
                    if skip_existing:
                        cursor.execute(
                            "SELECT id FROM satellite_images WHERE image_id = %s",
                            (caption_item['image_id'],)
                        )
                        if cursor.fetchone():
                            skipped_count += 1
                            continue
                    
                    # 데이터 삽입
                    insert_query = """
                    INSERT INTO satellite_images 
                    (image_id, image_path, caption, embedding, metadata, location_name, latitude, longitude)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (image_id) DO UPDATE SET
                        caption = EXCLUDED.caption,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                    """
                    
                    # 메타데이터 JSON 준비
                    metadata_json = {
                        'event_type': event_type,
                        'dataset': metadata.get('dataset', '') if isinstance(metadata, dict) else '',
                        'resolution': metadata.get('resolution', '') if isinstance(metadata, dict) else ''
                    }
                    
                    cursor.execute(
                        insert_query,
                        (
                            caption_item['image_id'],
                            str(image_path),
                            caption_item.get('caption', ''),
                            embedding.tolist(),  # numpy array를 리스트로 변환
                            json.dumps(metadata_json),
                            location_name,
                            float(latitude) if latitude else None,
                            float(longitude) if longitude else None
                        )
                    )
                    
                    inserted_count += 1
                
                except Exception as e:
                    error_count += 1
                    if error_count <= 5:  # 처음 5개 오류만 출력
                        print(f"\n⚠️  삽입 오류: {caption_item.get('image_id', 'unknown')} - {e}")
                    continue
            
            # 배치마다 커밋
            conn.commit()
        
        print(f"\n✅ 데이터베이스 삽입 완료!")
        print(f"   삽입: {inserted_count}개")
        print(f"   건너뜀: {skipped_count}개")
        print(f"   오류: {error_count}개")
    
    finally:
        cursor.close()
        conn.close()


def verify_database():
    """데이터베이스에 저장된 데이터 확인"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT COUNT(*) FROM satellite_images")
        count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM satellite_images WHERE embedding IS NOT NULL")
        with_embedding = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM satellite_images WHERE metadata IS NOT NULL")
        with_metadata = cursor.fetchone()[0]
        
        print(f"\n📊 데이터베이스 현황:")
        print(f"   총 레코드: {count}개")
        print(f"   임베딩 포함: {with_embedding}개")
        print(f"   메타데이터 포함: {with_metadata}개")
        
        # 샘플 데이터 조회
        cursor.execute("""
            SELECT image_id, caption, location_name 
            FROM satellite_images 
            LIMIT 3
        """)
        samples = cursor.fetchall()
        
        if samples:
            print(f"\n📝 샘플 데이터:")
            for sample in samples:
                print(f"   - {sample[0]}: {sample[1][:50]}...")
    
    finally:
        cursor.close()
        conn.close()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="캡션과 임베딩을 데이터베이스에 삽입",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 전체 데이터 삽입
  python insert_to_db.py --captions_file ./data/captions/captions.json --embeddings_file ./data/embeddings/embeddings.npy

  # 테스트용 (100개만)
  python insert_to_db.py --captions_file ./data/captions/captions.json --embeddings_file ./data/embeddings/embeddings.npy --max_samples 100

  # 기존 레코드 덮어쓰기
  python insert_to_db.py --captions_file ./data/captions/captions.json --embeddings_file ./data/embeddings/embeddings.npy --no-skip-existing
        """
    )
    
    parser.add_argument(
        "--captions_file",
        type=str,
        default="./data/captions/captions.json",
        help="캡션 JSON 파일 경로"
    )
    
    parser.add_argument(
        "--embeddings_file",
        type=str,
        default="./data/embeddings/embeddings.npy",
        help="임베딩 NumPy 파일 경로"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="배치 크기 (기본값: 100)"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="최대 삽입할 샘플 수 (기본값: 전체)"
    )
    
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="기존 레코드도 덮어쓰기"
    )
    
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="삽입하지 않고 데이터베이스 현황만 확인"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("데이터베이스 삽입")
    print("=" * 60)
    
    # 데이터베이스 현황만 확인
    if args.verify_only:
        verify_database()
        return
    
    # 파일 경로 확인
    captions_file = Path(args.captions_file)
    embeddings_file = Path(args.embeddings_file)
    
    if not captions_file.exists():
        print(f"❌ 캡션 파일을 찾을 수 없습니다: {captions_file}")
        sys.exit(1)
    
    if not embeddings_file.exists():
        print(f"❌ 임베딩 파일을 찾을 수 없습니다: {embeddings_file}")
        sys.exit(1)
    
    # 데이터 로드
    captions, embeddings = load_captions_and_embeddings(captions_file, embeddings_file)
    
    # 샘플 제한 적용
    if args.max_samples:
        captions = captions[:args.max_samples]
        embeddings = embeddings[:args.max_samples]
        print(f"📌 제한 적용: {len(captions)}개 샘플 처리")
    
    # 데이터베이스 삽입
    insert_to_database(
        captions,
        embeddings,
        batch_size=args.batch_size,
        skip_existing=not args.no_skip_existing
    )
    
    # 검증
    verify_database()
    
    print("\n" + "=" * 60)
    print("✅ 모든 작업 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

