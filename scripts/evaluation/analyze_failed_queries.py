"""실패한 검색 쿼리 상세 분석"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.database.search_vector_db_v2 import search_by_text

# 실패한 쿼리들 상세 분석
queries = [
    ('flooded area', ['flood', 'water']),
    ('residential area', ['residential', 'building', 'house']),
    ('water body', ['water', 'lake', 'river']),
    ('road network', ['road', 'street', 'highway']),
    ('vegetation and trees', ['tree', 'vegetation', 'green']),
]

print('=' * 60)
print('검색 실패 케이스 분석')
print('=' * 60)

for query, keywords in queries:
    print(f'\n🔍 쿼리: "{query}"')
    print(f'   기대 키워드: {keywords}')
    results = search_by_text(query, top_k=5)
    
    print(f'   상위 5개 결과:')
    for i, r in enumerate(results, 1):
        caption = r['caption']
        has_keywords = [kw for kw in keywords if kw in caption.lower()]
        all_keywords = all(kw in caption.lower() for kw in keywords)
        status = '✅' if all_keywords else '❌'
        print(f'   {i}. [{status}] {caption}')
        print(f'      포함 키워드: {has_keywords} | 모든 키워드 포함: {all_keywords}')
        print(f'      유사도: {r["similarity"]:.3f}')

