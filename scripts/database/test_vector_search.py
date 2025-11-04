"""
벡터 검색 기능 테스트
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.database.search_vector_db_v2 import (
    search_by_text,
    search_by_embedding,
    search_by_metadata,
    hybrid_search
)
import numpy as np

def test_text_search():
    """텍스트 검색 테스트"""
    print("=" * 60)
    print("테스트 1: 텍스트로 벡터 검색")
    print("=" * 60)
    
    queries = [
        "golf course",
        "aerial view of buildings",
        "flooded area",
        "rivers and bridges"
    ]
    
    for query in queries:
        print(f"\n🔍 검색어: '{query}'")
        results = search_by_text(query, top_k=3)
        
        print(f"✅ 검색 결과: {len(results)}개")
        for i, result in enumerate(results, 1):
            print(f"\n  {i}. {result['image_id']}")
            print(f"     캡션: {result['caption'][:60]}...")
            print(f"     유사도: {result['similarity']:.4f}")

def test_embedding_search():
    """임베딩 검색 테스트"""
    print("\n" + "=" * 60)
    print("테스트 2: 벡터 임베딩으로 검색")
    print("=" * 60)
    
    # 샘플 벡터 생성 (384차원)
    query_vec = np.random.rand(384).astype(np.float32)
    
    print(f"\n🔍 384차원 랜덤 벡터로 검색")
    results = search_by_embedding(query_vec, top_k=5)
    
    print(f"✅ 검색 결과: {len(results)}개")
    for i, result in enumerate(results, 1):
        print(f"\n  {i}. {result['image_id']}")
        print(f"     캡션: {result['caption'][:60]}...")
        print(f"     유사도: {result['similarity']:.4f}")

def test_metadata_search():
    """메타데이터 검색 테스트"""
    print("\n" + "=" * 60)
    print("테스트 3: 메타데이터 필터링")
    print("=" * 60)
    
    filters = {
        'event_type': 'POST-event'
    }
    
    print(f"\n🔍 필터: {filters}")
    results = search_by_metadata(filters, top_k=5)
    
    print(f"✅ 검색 결과: {len(results)}개")
    for i, result in enumerate(results, 1):
        print(f"\n  {i}. {result['image_id']}")
        print(f"     캡션: {result['caption'][:60]}...")
        if result.get('metadata'):
            print(f"     메타데이터: {result['metadata']}")

def test_hybrid_search():
    """하이브리드 검색 테스트"""
    print("\n" + "=" * 60)
    print("테스트 4: 하이브리드 검색 (텍스트 + 메타데이터)")
    print("=" * 60)
    
    query = "golf course"
    filters = {
        'event_type': 'POST-event'
    }
    
    print(f"\n🔍 검색어: '{query}'")
    print(f"🔍 필터: {filters}")
    
    results = hybrid_search(
        query_text=query,
        metadata_filters=filters,
        top_k=5
    )
    
    print(f"✅ 검색 결과: {len(results)}개")
    for i, result in enumerate(results, 1):
        print(f"\n  {i}. {result['image_id']}")
        print(f"     캡션: {result['caption'][:60]}...")
        print(f"     유사도: {result['similarity']:.4f}")

def main():
    """메인 테스트 함수"""
    print("\n" + "=" * 60)
    print("벡터 검색 기능 테스트")
    print("=" * 60)
    
    try:
        test_text_search()
        test_embedding_search()
        test_metadata_search()
        test_hybrid_search()
        
        print("\n" + "=" * 60)
        print("🎉 모든 테스트 완료!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

