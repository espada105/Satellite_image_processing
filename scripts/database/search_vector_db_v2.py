"""
벡터 데이터베이스 검색 함수 (pgvector 없이)

코사인 유사도를 사용하여 벡터 검색을 수행합니다.
"""

import psycopg
import numpy as np
import json
import sys
import os
from typing import List, Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.utils.config import POSTGRES_CONFIG


def get_db_connection():
    """데이터베이스 연결 반환"""
    conn_string = f"host={POSTGRES_CONFIG['host']} port={POSTGRES_CONFIG['port']} user={POSTGRES_CONFIG['user']} password={POSTGRES_CONFIG['password']} dbname={POSTGRES_CONFIG['database']}"
    return psycopg.connect(conn_string)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """코사인 유사도 계산"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def search_by_embedding(
    query_embedding: np.ndarray,
    top_k: int = 24,
    threshold: float = 0.0
) -> List[Dict]:
    """
    벡터 임베딩으로 유사 이미지 검색
    
    Args:
        query_embedding: 쿼리 벡터 임베딩 (numpy array, 384차원)
        top_k: 반환할 이미지 개수
        threshold: 최소 유사도 임계값
    
    Returns:
        검색 결과 리스트 (유사도 높은 순)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # 모든 임베딩 가져오기
        cursor.execute("""
            SELECT id, image_id, image_path, caption, embedding_array, metadata
            FROM satellite_images
            WHERE embedding_array IS NOT NULL
        """)
        
        all_results = cursor.fetchall()
        
        if not all_results:
            return []
        
        # 유사도 계산
        query_vec = np.array(query_embedding).flatten()
        similarities = []
        
        for row in all_results:
            img_id, image_id, image_path, caption, embedding_array, metadata = row
            
            if embedding_array is None:
                continue
            
            # 배열을 numpy 배열로 변환
            db_vec = np.array(embedding_array)
            
            # 차원 확인
            if query_vec.shape != db_vec.shape:
                continue
            
            # 코사인 유사도 계산
            similarity = cosine_similarity(query_vec, db_vec)
            
            if similarity >= threshold:
                similarities.append({
                    'id': img_id,
                    'image_id': image_id,
                    'image_path': image_path,
                    'caption': caption,
                    'metadata': json.loads(metadata) if isinstance(metadata, str) else metadata,
                    'similarity': float(similarity)
                })
        
        # 유사도 높은 순으로 정렬
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 상위 K개 반환
        return similarities[:top_k]
    
    finally:
        cursor.close()
        conn.close()


def search_by_text(
    query_text: str,
    top_k: int = 24,
    threshold: float = 0.0
) -> List[Dict]:
    """
    텍스트 쿼리로 유사 이미지 검색
    
    Args:
        query_text: 검색할 텍스트
        top_k: 반환할 이미지 개수
        threshold: 최소 유사도 임계값
    
    Returns:
        검색 결과 리스트
    """
    from sentence_transformers import SentenceTransformer
    
    # 임베딩 모델 로드 (캡션 생성 시 사용한 것과 동일)
    from scripts.utils.config import EMBEDDING_MODEL
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # 쿼리 텍스트를 벡터로 변환
    query_embedding = model.encode(query_text, convert_to_numpy=True)
    
    # 벡터 검색 수행
    return search_by_embedding(query_embedding, top_k=top_k, threshold=threshold)


def search_by_metadata(
    filters: Dict,
    top_k: int = 100
) -> List[Dict]:
    """
    메타데이터로 이미지 검색
    
    Args:
        filters: 필터 조건 딕셔너리
            - event_type: 'PRE-event' 또는 'POST-event'
            - dataset: 데이터셋 이름
            - location_name: 위치 이름
        top_k: 반환할 이미지 개수
    
    Returns:
        검색 결과 리스트
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        query = "SELECT id, image_id, image_path, caption, metadata, location_name FROM satellite_images WHERE 1=1"
        params = []
        
        # 필터 조건 추가
        if 'event_type' in filters:
            query += " AND metadata->>'event_type' = %s"
            params.append(filters['event_type'])
        
        if 'dataset' in filters:
            query += " AND metadata->>'dataset' = %s"
            params.append(filters['dataset'])
        
        if 'location_name' in filters:
            query += " AND location_name = %s"
            params.append(filters['location_name'])
        
        query += f" LIMIT {top_k}"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        return [
            {
                'id': row[0],
                'image_id': row[1],
                'image_path': row[2],
                'caption': row[3],
                'metadata': json.loads(row[4]) if isinstance(row[4], str) else row[4],
                'location_name': row[5]
            }
            for row in results
        ]
    
    finally:
        cursor.close()
        conn.close()


def hybrid_search(
    query_text: Optional[str] = None,
    query_embedding: Optional[np.ndarray] = None,
    metadata_filters: Optional[Dict] = None,
    top_k: int = 24,
    similarity_weight: float = 0.7,
    metadata_weight: float = 0.3
) -> List[Dict]:
    """
    하이브리드 검색 (벡터 검색 + 메타데이터 필터링)
    
    Args:
        query_text: 텍스트 쿼리
        query_embedding: 벡터 임베딩 (query_text가 없을 때 사용)
        metadata_filters: 메타데이터 필터
        top_k: 반환할 이미지 개수
        similarity_weight: 유사도 가중치
        metadata_weight: 메타데이터 매칭 가중치
    
    Returns:
        검색 결과 리스트
    """
    # 벡터 검색 결과
    if query_text:
        vector_results = search_by_text(query_text, top_k=top_k * 2)
    elif query_embedding is not None:
        vector_results = search_by_embedding(query_embedding, top_k=top_k * 2)
    else:
        vector_results = []
    
    # 메타데이터 필터 적용
    if metadata_filters:
        metadata_results = search_by_metadata(metadata_filters, top_k=top_k * 2)
        metadata_ids = {r['id'] for r in metadata_results}
        
        # 벡터 결과에 메타데이터 필터 적용
        filtered_results = []
        for result in vector_results:
            if result['id'] in metadata_ids:
                # 메타데이터 매칭 보너스
                result['similarity'] = result['similarity'] * similarity_weight + metadata_weight
                filtered_results.append(result)
        
        # 정렬 및 상위 K개 반환
        filtered_results.sort(key=lambda x: x['similarity'], reverse=True)
        return filtered_results[:top_k]
    
    return vector_results[:top_k]

