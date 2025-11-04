"""
RAG (Retrieval Augmented Generation) 도구

텍스트 쿼리를 받아 벡터 검색을 수행하고 관련 이미지와 캡션을 반환합니다.
"""

import sys
import os
from typing import List, Dict, Optional
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.database.search_vector_db_v2 import search_by_text, hybrid_search
from scripts.utils.config import PROCESSED_DATA_DIR


class RAGTool:
    """RAG 검색 도구"""
    
    def __init__(self, top_k: int = 5):
        """
        Args:
            top_k: 검색 결과 개수
        """
        self.top_k = top_k
    
    def search(
        self,
        query: str,
        metadata_filters: Optional[Dict] = None,
        top_k: Optional[int] = None
    ) -> Dict:
        """
        텍스트 쿼리로 관련 이미지 검색
        
        Args:
            query: 검색할 텍스트
            metadata_filters: 메타데이터 필터 (선택)
            top_k: 반환할 결과 개수 (기본값: self.top_k)
        
        Returns:
            검색 결과 딕셔너리
        """
        if top_k is None:
            top_k = self.top_k
        
        # 벡터 검색 수행
        if metadata_filters:
            results = hybrid_search(
                query_text=query,
                metadata_filters=metadata_filters,
                top_k=top_k
            )
        else:
            results = search_by_text(query, top_k=top_k)
        
        # 결과 포맷팅
        formatted_results = []
        for result in results:
            image_path = PROCESSED_DATA_DIR / result['image_path']
            
            formatted_results.append({
                'image_id': result['image_id'],
                'image_path': str(image_path),
                'caption': result['caption'],
                'similarity': result['similarity'],
                'metadata': result.get('metadata', {})
            })
        
        return {
            'query': query,
            'results': formatted_results,
            'count': len(formatted_results)
        }
    
    def get_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        검색 결과를 컨텍스트 문자열로 변환
        
        Args:
            query: 검색 쿼리
            top_k: 결과 개수
        
        Returns:
            컨텍스트 문자열
        """
        search_results = self.search(query, top_k=top_k)
        
        context_parts = [f"검색 쿼리: {query}\n"]
        context_parts.append(f"관련 이미지 {len(search_results['results'])}개를 찾았습니다:\n\n")
        
        for i, result in enumerate(search_results['results'], 1):
            context_parts.append(f"{i}. 이미지 ID: {result['image_id']}\n")
            context_parts.append(f"   캡션: {result['caption']}\n")
            context_parts.append(f"   경로: {result['image_path']}\n")
            context_parts.append(f"   유사도: {result['similarity']:.4f}\n\n")
        
        return "".join(context_parts)


def create_rag_tool(top_k: int = 5):
    """
    RAG 도구 생성 (LangChain/LangGraph 호환)
    
    Args:
        top_k: 검색 결과 개수
    
    Returns:
        RAG 도구 함수
    """
    rag = RAGTool(top_k=top_k)
    
    def rag_search(query: str, metadata_filters: Optional[Dict] = None) -> Dict:
        """
        RAG 검색 도구
        
        Args:
            query: 검색할 텍스트
            metadata_filters: 메타데이터 필터
        
        Returns:
            검색 결과
        """
        return rag.search(query, metadata_filters)
    
    return rag_search


if __name__ == "__main__":
    # 테스트
    rag = RAGTool(top_k=3)
    
    print("=" * 60)
    print("RAG 도구 테스트")
    print("=" * 60)
    
    queries = [
        "golf course",
        "flooded area",
        "buildings in urban area"
    ]
    
    for query in queries:
        print(f"\n🔍 검색어: '{query}'")
        results = rag.search(query)
        
        print(f"✅ 검색 결과: {results['count']}개")
        for i, result in enumerate(results['results'], 1):
            print(f"\n  {i}. {result['image_id']}")
            print(f"     캡션: {result['caption'][:60]}...")
            print(f"     유사도: {result['similarity']:.4f}")
        
        # 컨텍스트 생성
        context = rag.get_context(query)
        print(f"\n📄 생성된 컨텍스트 (처음 200자):")
        print(context[:200] + "...")

