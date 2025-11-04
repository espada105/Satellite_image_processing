"""
데이터베이스 검색 테스트 스크립트

저장된 데이터를 검색하고 조회하는 기능을 테스트합니다.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import psycopg
import json
import numpy as np
from scripts.utils.config import POSTGRES_CONFIG

def get_db_connection():
    """데이터베이스 연결"""
    conn_string = f"host={POSTGRES_CONFIG['host']} port={POSTGRES_CONFIG['port']} user={POSTGRES_CONFIG['user']} password={POSTGRES_CONFIG['password']} dbname={POSTGRES_CONFIG['database']}"
    return psycopg.connect(conn_string)

def test_basic_query():
    """기본 쿼리 테스트"""
    print("=" * 60)
    print("테스트 1: 기본 데이터 조회")
    print("=" * 60)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # 전체 레코드 수
        cursor.execute("SELECT COUNT(*) FROM satellite_images")
        total = cursor.fetchone()[0]
        print(f"\n✅ 총 레코드 수: {total}개")
        
        # 샘플 데이터 조회
        cursor.execute("""
            SELECT image_id, caption, location_name, created_at 
            FROM satellite_images 
            LIMIT 5
        """)
        samples = cursor.fetchall()
        
        print(f"\n📝 샘플 데이터 (5개):")
        for i, (img_id, caption, location, created) in enumerate(samples, 1):
            print(f"\n  {i}. {img_id}")
            print(f"     캡션: {caption[:60]}...")
            print(f"     위치: {location or 'N/A'}")
            print(f"     생성일: {created}")
        
        return True
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def test_text_search():
    """텍스트 검색 테스트"""
    print("\n" + "=" * 60)
    print("테스트 2: 캡션 텍스트 검색")
    print("=" * 60)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # "golf course" 검색
        search_term = "golf course"
        print(f"\n🔍 검색어: '{search_term}'")
        
        cursor.execute("""
            SELECT image_id, caption, location_name
            FROM satellite_images
            WHERE caption ILIKE %s
            LIMIT 5
        """, (f"%{search_term}%",))
        
        results = cursor.fetchall()
        print(f"\n✅ 검색 결과: {len(results)}개")
        
        for i, (img_id, caption, location) in enumerate(results, 1):
            print(f"\n  {i}. {img_id}")
            print(f"     캡션: {caption}")
            print(f"     위치: {location or 'N/A'}")
        
        return True
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def test_metadata_search():
    """메타데이터 검색 테스트"""
    print("\n" + "=" * 60)
    print("테스트 3: 메타데이터 필터링")
    print("=" * 60)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # POST-event 이미지만 검색
        print(f"\n🔍 필터: event_type = 'POST-event'")
        
        cursor.execute("""
            SELECT image_id, caption, metadata
            FROM satellite_images
            WHERE metadata->>'event_type' = 'POST-event'
            LIMIT 5
        """)
        
        results = cursor.fetchall()
        print(f"\n✅ 검색 결과: {len(results)}개")
        
        for i, (img_id, caption, metadata) in enumerate(results, 1):
            print(f"\n  {i}. {img_id}")
            print(f"     캡션: {caption[:60]}...")
            if metadata:
                meta = json.loads(metadata) if isinstance(metadata, str) else metadata
                print(f"     이벤트 타입: {meta.get('event_type', 'N/A')}")
        
        # 데이터셋별 통계
        print(f"\n📊 데이터셋별 통계:")
        cursor.execute("""
            SELECT 
                metadata->>'dataset' as dataset,
                COUNT(*) as count
            FROM satellite_images
            WHERE metadata->>'dataset' IS NOT NULL
            GROUP BY metadata->>'dataset'
            ORDER BY count DESC
        """)
        
        stats = cursor.fetchall()
        for dataset, count in stats:
            print(f"  - {dataset or 'N/A'}: {count}개")
        
        return True
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cursor.close()
        conn.close()

def test_embedding_array():
    """임베딩 배열 테스트"""
    print("\n" + "=" * 60)
    print("테스트 4: 임베딩 배열 조회")
    print("=" * 60)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # 임베딩이 있는 데이터 조회
        cursor.execute("""
            SELECT image_id, array_length(embedding_array, 1) as embedding_dim
            FROM satellite_images
            WHERE embedding_array IS NOT NULL
            LIMIT 5
        """)
        
        results = cursor.fetchall()
        print(f"\n✅ 임베딩이 있는 레코드: {len(results)}개")
        
        for i, (img_id, dim) in enumerate(results, 1):
            print(f"  {i}. {img_id}: 임베딩 차원 = {dim}")
        
        # 간단한 유사도 계산 (코사인 유사도)
        print(f"\n🔍 간단한 벡터 유사도 계산 테스트:")
        cursor.execute("""
            SELECT image_id, embedding_array
            FROM satellite_images
            WHERE embedding_array IS NOT NULL
            LIMIT 2
        """)
        
        results = cursor.fetchall()
        if len(results) >= 2:
            img1_id, emb1 = results[0]
            img2_id, emb2 = results[1]
            
            # numpy 배열로 변환
            vec1 = np.array(emb1)
            vec2 = np.array(emb2)
            
            # 코사인 유사도 계산
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            similarity = dot_product / (norm1 * norm2) if (norm1 * norm2) > 0 else 0
            
            print(f"  이미지 1: {img1_id}")
            print(f"  이미지 2: {img2_id}")
            print(f"  코사인 유사도: {similarity:.4f}")
        
        return True
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cursor.close()
        conn.close()

def test_location_search():
    """위치 기반 검색 테스트"""
    print("\n" + "=" * 60)
    print("테스트 5: 위치 정보 검색")
    print("=" * 60)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # 위치 정보가 있는 데이터 조회
        cursor.execute("""
            SELECT image_id, location_name, latitude, longitude
            FROM satellite_images
            WHERE location_name IS NOT NULL 
               OR latitude IS NOT NULL
            LIMIT 5
        """)
        
        results = cursor.fetchall()
        print(f"\n✅ 위치 정보가 있는 레코드: {len(results)}개")
        
        for i, (img_id, location, lat, lon) in enumerate(results, 1):
            print(f"\n  {i}. {img_id}")
            print(f"     위치: {location or 'N/A'}")
            if lat and lon:
                print(f"     좌표: ({lat}, {lon})")
            else:
                print(f"     좌표: N/A")
        
        return True
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        return False
    finally:
        cursor.close()
        conn.close()

def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("데이터베이스 검색 기능 테스트")
    print("=" * 60)
    
    results = []
    
    # 각 테스트 실행
    results.append(("기본 조회", test_basic_query()))
    results.append(("텍스트 검색", test_text_search()))
    results.append(("메타데이터 검색", test_metadata_search()))
    results.append(("임베딩 배열", test_embedding_array()))
    results.append(("위치 검색", test_location_search()))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✅ 성공" if success else "❌ 실패"
        print(f"  {test_name}: {status}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    print(f"\n총 {total}개 테스트 중 {passed}개 성공, {total - passed}개 실패")
    
    if passed == total:
        print("\n🎉 모든 테스트 통과!")
    else:
        print("\n⚠️  일부 테스트 실패")

if __name__ == "__main__":
    main()

