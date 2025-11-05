"""
문제점 종합 분석 스크립트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.database.search_vector_db_v2 import search_by_text, search_by_embedding
from scripts.utils.config import EMBEDDING_MODEL
import psycopg
from scripts.utils.config import POSTGRES_CONFIG
import time
import numpy as np

print("=" * 60)
print("문제점 종합 분석")
print("=" * 60)

# 1. 데이터베이스 통계
print("\n1️⃣ 데이터베이스 상태 확인")
conn_string = f"host={POSTGRES_CONFIG['host']} port={POSTGRES_CONFIG['port']} user={POSTGRES_CONFIG['user']} password={POSTGRES_CONFIG['password']} dbname={POSTGRES_CONFIG['database']}"
conn = psycopg.connect(conn_string)
cursor = conn.cursor()

cursor.execute('SELECT COUNT(*) FROM satellite_images')
total = cursor.fetchone()[0]
print(f"  총 이미지 수: {total:,}개")

cursor.execute('SELECT COUNT(*) FROM satellite_images WHERE embedding_array IS NOT NULL')
with_emb = cursor.fetchone()[0]
print(f"  임베딩 있는 이미지: {with_emb:,}개 ({with_emb/total*100:.1f}%)")

cursor.execute("SELECT array_length(embedding_array, 1) FROM satellite_images WHERE embedding_array IS NOT NULL LIMIT 1")
dim_result = cursor.fetchone()
if dim_result:
    print(f"  임베딩 차원: {dim_result[0]}차원")

# 인덱스 확인
cursor.execute("""
    SELECT indexname, indexdef 
    FROM pg_indexes 
    WHERE tablename = 'satellite_images'
""")
indexes = cursor.fetchall()
print(f"  인덱스 수: {len(indexes)}개")
if indexes:
    for idx_name, idx_def in indexes:
        print(f"    - {idx_name}: {idx_def[:80]}...")
else:
    print("  ⚠️  벡터 검색용 인덱스 없음!")

conn.close()

# 2. 검색 성능 테스트
print("\n2️⃣ 검색 성능 테스트")
test_query = "golf course"

start_time = time.time()
results = search_by_text(test_query, top_k=10)
elapsed = time.time() - start_time

print(f"  쿼리: '{test_query}'")
print(f"  검색 시간: {elapsed:.3f}초")
print(f"  결과 수: {len(results)}개")
if results:
    print(f"  최고 유사도: {results[0]['similarity']:.3f}")
    print(f"  평균 유사도: {sum(r['similarity'] for r in results)/len(results):.3f}")

# 3. 임베딩 모델 확인
print("\n3️⃣ 임베딩 모델 정보")
print(f"  모델: {EMBEDDING_MODEL}")
print(f"  모델 타입: sentence-transformers")

# 4. 캡션 품질 문제 확인
print("\n4️⃣ 캡션 품질 문제")
conn = psycopg.connect(conn_string)
cursor = conn.cursor()
cursor.execute("""
    SELECT caption, COUNT(*) as cnt 
    FROM satellite_images 
    GROUP BY caption 
    HAVING COUNT(*) > 1 
    ORDER BY cnt DESC 
    LIMIT 5
""")
duplicates = cursor.fetchall()
if duplicates:
    print(f"  중복 캡션 샘플 (상위 5개):")
    for caption, cnt in duplicates:
        print(f"    - {cnt}회: {caption[:60]}...")
else:
    print("  중복 없음")

cursor.execute("""
    SELECT COUNT(*) FROM (
        SELECT caption, COUNT(*) as cnt 
        FROM satellite_images 
        GROUP BY caption 
        HAVING COUNT(*) > 1
    ) as dup
""")
dup_count = cursor.fetchone()[0]
print(f"  중복된 캡션 종류: {dup_count}개")

cursor.execute("SELECT COUNT(DISTINCT caption) FROM satellite_images")
unique_count = cursor.fetchone()[0]
print(f"  고유 캡션 수: {unique_count}개")
print(f"  중복률: {(1 - unique_count/total)*100:.1f}%")

# 5. 검색 알고리즘 문제
print("\n5️⃣ 검색 알고리즘 문제")
print("  ⚠️  전체 스캔 방식: 모든 데이터를 메모리에 로드하여 계산")
print("  ⚠️  인덱스 없음: O(n) 시간 복잡도")
print("  ⚠️  스케일링 불가: 데이터가 많아지면 매우 느려짐")

# 6. 평가 기준 문제
print("\n6️⃣ 평가 기준 문제")
print("  ⚠️  키워드 기반 평가: 의미적 유사도를 제대로 반영하지 못함")
print("  ⚠️  임베딩 기반 검색인데 키워드 매칭으로 평가하는 모순")
print("  ⚠️  'flooded'와 'water'는 의미적으로 관련이 있지만 키워드 매칭으로는 실패")

# 7. 개선 제안
print("\n7️⃣ 개선 제안")
print("  ✅ pgvector 확장 설치 및 벡터 인덱스 생성")
print("  ✅ BLIP-2 프롬프트 최적화 (더 구체적이고 다양한 캡션 생성)")
print("  ✅ 더 큰 임베딩 모델 사용 (all-mpnet-base-v2 등)")
print("  ✅ 의미적 유사도 기반 평가 방법 도입")
print("  ✅ 검색 결과 다양성 확보 (중복 제거)")

conn.close()

print("\n" + "=" * 60)

