"""벡터 데이터베이스 검색 함수"""
import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.utils.config import POSTGRES_CONFIG


def get_db_connection():
    """데이터베이스 연결 반환"""
    # 연결 문자열 직접 구성 (Windows 인코딩 문제 회피)
    conn_string = f"host={POSTGRES_CONFIG['host']} port={POSTGRES_CONFIG['port']} user={POSTGRES_CONFIG['user']} password={POSTGRES_CONFIG['password']} dbname={POSTGRES_CONFIG['database']}"
    conn = psycopg2.connect(conn_string)
    register_vector(conn)
    return conn


def search_by_text(query_embedding: np.ndarray, top_k: int = 10):
    """
    텍스트 쿼리의 벡터 임베딩으로 유사 이미지 검색
    
    Args:
        query_embedding: 쿼리 텍스트의 벡터 임베딩 (numpy array)
        top_k: 반환할 이미지 개수
    
    Returns:
        검색 결과 리스트
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # 코사인 유사도로 검색
        query = """
        SELECT 
            image_id,
            image_path,
            caption,
            metadata,
            1 - (embedding <=> %s) as similarity
        FROM satellite_images
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s
        LIMIT %s
        """
        
        cursor.execute(query, (query_embedding.tolist(), query_embedding.tolist(), top_k))
        results = cursor.fetchall()
        
        # 결과 포맷팅
        formatted_results = []
        for row in results:
            formatted_results.append({
                "image_id": row[0],
                "image_path": row[1],
                "caption": row[2],
                "metadata": row[3] if isinstance(row[3], dict) else json.loads(row[3]),
                "similarity": float(row[4])
            })
        
        return formatted_results
    
    finally:
        cursor.close()
        conn.close()


def search_by_location(lat: float, lon: float, radius_km: float = 10.0, top_k: int = 10):
    """
    지리공간 기반 이미지 검색
    
    Args:
        lat: 위도
        lon: 경도
        radius_km: 검색 반경 (킬로미터)
        top_k: 반환할 이미지 개수
    
    Returns:
        검색 결과 리스트
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # PostGIS 사용 여부 확인
        cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'postgis');")
        has_postgis = cursor.fetchone()[0]
        
        if has_postgis:
            # PostGIS를 사용한 지리공간 검색
            query = """
            SELECT 
                image_id,
                image_path,
                caption,
                metadata,
                ST_Distance(
                    ST_SetSRID(ST_MakePoint(longitude, latitude)::geography, 4326),
                    ST_SetSRID(ST_MakePoint(%s, %s)::geography, 4326)
                ) / 1000.0 as distance_km
            FROM satellite_images
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
                AND ST_DWithin(
                    ST_SetSRID(ST_MakePoint(longitude, latitude)::geography, 4326),
                    ST_SetSRID(ST_MakePoint(%s, %s)::geography, 4326),
                    %s * 1000
                )
            ORDER BY distance_km
            LIMIT %s
            """
            
            cursor.execute(query, (lon, lat, lon, lat, radius_km, top_k))
        else:
            # 간단한 유클리드 거리 계산 (PostGIS 없을 때)
            query = """
            SELECT 
                image_id,
                image_path,
                caption,
                metadata,
                SQRT(
                    POW(69.1 * (latitude - %s), 2) + 
                    POW(69.1 * (longitude - %s) * COS(latitude / 57.3), 2)
                ) as distance_km
            FROM satellite_images
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
                AND SQRT(
                    POW(69.1 * (latitude - %s), 2) + 
                    POW(69.1 * (longitude - %s) * COS(latitude / 57.3), 2)
                ) <= %s
            ORDER BY distance_km
            LIMIT %s
            """
            
            cursor.execute(query, (lat, lon, lat, lon, radius_km, top_k))
        
        results = cursor.fetchall()
        
        # 결과 포맷팅
        formatted_results = []
        for row in results:
            formatted_results.append({
                "image_id": row[0],
                "image_path": row[1],
                "caption": row[2],
                "metadata": row[3] if isinstance(row[3], dict) else json.loads(row[3]),
                "distance_km": float(row[4])
            })
        
        return formatted_results
    
    finally:
        cursor.close()
        conn.close()


def search_hybrid(
    query_embedding: np.ndarray,
    lat: float = None,
    lon: float = None,
    radius_km: float = 10.0,
    top_k: int = 10,
    similarity_threshold: float = 0.7
):
    """
    텍스트 유사도와 지리공간 정보를 결합한 하이브리드 검색
    
    Args:
        query_embedding: 쿼리 벡터 임베딩
        lat: 위도 (선택)
        lon: 경도 (선택)
        radius_km: 지리공간 검색 반경
        top_k: 반환할 이미지 개수
        similarity_threshold: 최소 유사도 임계값
    
    Returns:
        검색 결과 리스트
    """
    # 텍스트 기반 검색
    text_results = search_by_text(query_embedding, top_k=top_k * 2)
    
    # 지리공간 필터링 (제공된 경우)
    if lat and lon:
        location_results = search_by_location(lat, lon, radius_km, top_k=top_k * 2)
        
        # 두 결과를 결합 및 점수 계산
        location_ids = {r["image_id"]: r for r in location_results}
        
        combined_results = []
        for result in text_results:
            image_id = result["image_id"]
            
            # 유사도 임계값 체크
            if result["similarity"] < similarity_threshold:
                continue
            
            # 지리공간 정보 추가
            if image_id in location_ids:
                result["distance_km"] = location_ids[image_id]["distance_km"]
                result["has_location_match"] = True
            else:
                result["has_location_match"] = False
            
            combined_results.append(result)
        
        # 최종 점수로 정렬 (유사도 우선, 거리는 보너스)
        combined_results.sort(
            key=lambda x: (
                x["similarity"],
                -x.get("distance_km", float("inf")) if x.get("has_location_match") else 0
            ),
            reverse=True
        )
        
        return combined_results[:top_k]
    
    else:
        # 텍스트 검색만 사용
        return [r for r in text_results if r["similarity"] >= similarity_threshold][:top_k]


if __name__ == "__main__":
    # 테스트 코드
    from sentence_transformers import SentenceTransformer
    
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    query_text = "도시 지역의 위성 이미지"
    query_emb = encoder.encode(query_text)
    
    print("텍스트 기반 검색 테스트:")
    results = search_by_text(query_emb, top_k=5)
    for r in results:
        print(f"- {r['image_id']}: 유사도={r['similarity']:.3f}")
    
    print("\n지리공간 검색 테스트 (서울):")
    results = search_by_location(37.5665, 126.9780, radius_km=50, top_k=5)
    for r in results:
        print(f"- {r['image_id']}: 거리={r['distance_km']:.2f}km")

