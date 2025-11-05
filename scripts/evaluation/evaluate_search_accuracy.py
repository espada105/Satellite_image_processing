"""
검색 정확도 평가 스크립트

벡터 검색의 정확도를 평가합니다:
- Precision@K
- Recall@K
- 평균 유사도
- 검색 결과 분석
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Set
import statistics

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.database.search_vector_db_v2 import search_by_text
from scripts.utils.config import POSTGRES_CONFIG
import psycopg
import numpy as np


def get_db_connection():
    """데이터베이스 연결"""
    conn_string = f"host={POSTGRES_CONFIG['host']} port={POSTGRES_CONFIG['port']} user={POSTGRES_CONFIG['user']} password={POSTGRES_CONFIG['password']} dbname={POSTGRES_CONFIG['database']}"
    return psycopg.connect(conn_string)


def create_test_queries() -> List[Dict]:
    """
    테스트 쿼리 세트 생성
    
    Returns:
        테스트 쿼리 리스트 (query, expected_keywords, category)
    """
    return [
        {
            'query': 'golf course',
            'expected_keywords': ['golf', 'course'],
            'category': 'sports_facility'
        },
        {
            'query': 'flooded area',
            'expected_keywords': ['flood', 'water'],
            'category': 'disaster'
        },
        {
            'query': 'aerial view of buildings',
            'expected_keywords': ['building', 'aerial'],
            'category': 'urban'
        },
        {
            'query': 'rivers and bridges',
            'expected_keywords': ['river', 'bridge'],
            'category': 'infrastructure'
        },
        {
            'query': 'residential area',
            'expected_keywords': ['residential', 'building', 'house'],
            'category': 'urban'
        },
        {
            'query': 'water body',
            'expected_keywords': ['water', 'lake', 'river'],
            'category': 'natural'
        },
        {
            'query': 'road network',
            'expected_keywords': ['road', 'street', 'highway'],
            'category': 'infrastructure'
        },
        {
            'query': 'vegetation and trees',
            'expected_keywords': ['tree', 'vegetation', 'green'],
            'category': 'natural'
        },
        {
            'query': 'urban development',
            'expected_keywords': ['urban', 'city', 'building'],
            'category': 'urban'
        },
        {
            'query': 'agricultural fields',
            'expected_keywords': ['field', 'agricultural', 'farm'],
            'category': 'agriculture'
        }
    ]


def check_relevance(caption: str, expected_keywords: List[str], strict: bool = False) -> bool:
    """
    검색 결과의 관련성 판단 (키워드 기반)
    
    Args:
        caption: 캡션 텍스트
        expected_keywords: 기대 키워드 리스트
        strict: True면 모든 키워드 포함, False면 하나만 포함해도 OK (기본값)
    
    Returns:
        관련성 여부
    """
    caption_lower = caption.lower()
    
    if strict:
        # 엄격한 모드: 모든 키워드가 포함되어야 함
        return all(keyword.lower() in caption_lower for keyword in expected_keywords)
    else:
        # 관대한 모드: 하나만 포함되어도 OK
        return any(keyword.lower() in caption_lower for keyword in expected_keywords)


def calculate_precision_at_k(results: List[Dict], expected_keywords: List[str], k: int, strict: bool = False) -> float:
    """Precision@K 계산"""
    if len(results) == 0:
        return 0.0
    
    top_k = results[:k]
    relevant = sum(1 for r in top_k if check_relevance(r.get('caption', ''), expected_keywords, strict=strict))
    
    return relevant / len(top_k) if top_k else 0.0


def calculate_recall_at_k(results: List[Dict], expected_keywords: List[str], k: int, total_relevant: int = None, strict: bool = False) -> float:
    """Recall@K 계산"""
    if len(results) == 0:
        return 0.0
    
    top_k = results[:k]
    relevant_found = sum(1 for r in top_k if check_relevance(r.get('caption', ''), expected_keywords, strict=strict))
    
    # 전체 관련 문서 수가 제공되지 않으면, 검색된 결과 중 관련 문서 비율로 근사
    if total_relevant is None:
        # 데이터베이스에서 전체 관련 문서 수 조회 (근사치)
        return relevant_found / len(top_k) if top_k else 0.0
    
    return relevant_found / total_relevant if total_relevant > 0 else 0.0


def calculate_mean_reciprocal_rank(results: List[Dict], expected_keywords: List[str], strict: bool = False) -> float:
    """Mean Reciprocal Rank (MRR) 계산"""
    for rank, result in enumerate(results, 1):
        if check_relevance(result.get('caption', ''), expected_keywords, strict=strict):
            return 1.0 / rank
    return 0.0


def evaluate_search_accuracy(test_queries: List[Dict] = None, k_values: List[int] = [1, 3, 5, 10], strict: bool = False) -> Dict:
    """
    검색 정확도 종합 평가
    
    Args:
        test_queries: 테스트 쿼리 리스트
        k_values: 평가할 K 값들
        strict: 엄격한 평가 모드 (모든 키워드 포함해야 함)
    """
    print("=" * 60)
    print("검색 정확도 평가 시작")
    if strict:
        print("⚠️  엄격한 평가 모드: 모든 키워드가 포함되어야 함")
    else:
        print("ℹ️  관대한 평가 모드: 하나의 키워드만 포함되어도 OK")
    print("=" * 60)
    
    if test_queries is None:
        test_queries = create_test_queries()
    
    all_results = []
    all_precisions = {k: [] for k in k_values}
    all_recalls = {k: [] for k in k_values}
    all_mrrs = []
    all_similarities = []
    
    print(f"\n🔍 테스트 쿼리 수: {len(test_queries)}개")
    print(f"📊 평가 K 값: {k_values}\n")
    
    for i, test_query in enumerate(test_queries, 1):
        query = test_query['query']
        expected_keywords = test_query['expected_keywords']
        category = test_query['category']
        
        print(f"[{i}/{len(test_queries)}] 검색: '{query}'")
        
        try:
            # 검색 실행
            results = search_by_text(query, top_k=max(k_values))
            
            if not results:
                print(f"  ⚠️  검색 결과 없음")
                continue
            
            # 유사도 수집
            similarities = [r.get('similarity', 0.0) for r in results]
            all_similarities.extend(similarities)
            
            # 각 K 값에 대한 정확도 계산
            query_result = {
                'query': query,
                'category': category,
                'num_results': len(results),
                'precisions': {},
                'recalls': {},
                'mrr': 0.0,
                'avg_similarity': statistics.mean(similarities) if similarities else 0.0,
                'top_result': results[0] if results else None
            }
            
            for k in k_values:
                precision = calculate_precision_at_k(results, expected_keywords, k, strict=strict)
                recall = calculate_recall_at_k(results, expected_keywords, k, strict=strict)
                
                query_result['precisions'][k] = precision
                query_result['recalls'][k] = recall
                
                all_precisions[k].append(precision)
                all_recalls[k].append(recall)
            
            # MRR 계산
            mrr = calculate_mean_reciprocal_rank(results, expected_keywords, strict=strict)
            query_result['mrr'] = mrr
            all_mrrs.append(mrr)
            
            all_results.append(query_result)
            
            # 간단한 출력
            print(f"  결과: {len(results)}개, P@5: {query_result['precisions'][5]:.2f}, "
                  f"R@5: {query_result['recalls'][5]:.2f}, MRR: {mrr:.3f}")
            
        except Exception as e:
            print(f"  ❌ 오류: {e}")
            continue
    
    # 전체 통계 계산
    overall_stats = {
        'total_queries': len(test_queries),
        'successful_queries': len(all_results),
        'precision_at_k': {
            k: {
                'mean': statistics.mean(precisions) if precisions else 0.0,
                'median': statistics.median(precisions) if precisions else 0.0,
                'std': statistics.stdev(precisions) if len(precisions) > 1 else 0.0
            }
            for k, precisions in all_precisions.items()
        },
        'recall_at_k': {
            k: {
                'mean': statistics.mean(recalls) if recalls else 0.0,
                'median': statistics.median(recalls) if recalls else 0.0,
                'std': statistics.stdev(recalls) if len(recalls) > 1 else 0.0
            }
            for k, recalls in all_recalls.items()
        },
        'mrr': {
            'mean': statistics.mean(all_mrrs) if all_mrrs else 0.0,
            'median': statistics.median(all_mrrs) if all_mrrs else 0.0
        },
        'similarity': {
            'mean': statistics.mean(all_similarities) if all_similarities else 0.0,
            'median': statistics.median(all_similarities) if all_similarities else 0.0,
            'min': min(all_similarities) if all_similarities else 0.0,
            'max': max(all_similarities) if all_similarities else 0.0
        }
    }
    
    return {
        'overall_statistics': overall_stats,
        'query_results': all_results
    }


def print_evaluation_report(results: Dict):
    """평가 결과 출력"""
    print("\n" + "=" * 60)
    print("검색 정확도 평가 결과")
    print("=" * 60)
    
    stats = results['overall_statistics']
    
    print(f"\n📊 전체 통계")
    print(f"  총 쿼리 수: {stats['total_queries']}개")
    print(f"  성공한 쿼리 수: {stats['successful_queries']}개")
    
    print(f"\n🎯 Precision@K")
    for k in sorted(stats['precision_at_k'].keys()):
        p = stats['precision_at_k'][k]
        print(f"  P@{k}: 평균 {p['mean']:.3f}, 중앙값 {p['median']:.3f}, 표준편차 {p['std']:.3f}")
    
    print(f"\n📈 Recall@K")
    for k in sorted(stats['recall_at_k'].keys()):
        r = stats['recall_at_k'][k]
        print(f"  R@{k}: 평균 {r['mean']:.3f}, 중앙값 {r['median']:.3f}, 표준편차 {r['std']:.3f}")
    
    print(f"\n🔄 Mean Reciprocal Rank (MRR)")
    mrr = stats['mrr']
    print(f"  평균: {mrr['mean']:.3f}")
    print(f"  중앙값: {mrr['median']:.3f}")
    
    print(f"\n📉 유사도 분포")
    sim = stats['similarity']
    print(f"  평균: {sim['mean']:.3f}")
    print(f"  중앙값: {sim['median']:.3f}")
    print(f"  범위: {sim['min']:.3f} ~ {sim['max']:.3f}")
    
    # 쿼리별 상세 결과 (상위 5개)
    print(f"\n📝 쿼리별 결과 (상위 5개):")
    query_results = results['query_results']
    sorted_results = sorted(query_results, key=lambda x: x['mrr'], reverse=True)[:5]
    
    for i, qr in enumerate(sorted_results, 1):
        print(f"\n  {i}. '{qr['query']}' ({qr['category']})")
        print(f"     P@5: {qr['precisions'][5]:.2f}, R@5: {qr['recalls'][5]:.2f}, MRR: {qr['mrr']:.3f}")
        print(f"     평균 유사도: {qr['avg_similarity']:.3f}")
        if qr['top_result']:
            top = qr['top_result']
            print(f"     최상위 결과: {top['caption'][:60]}...")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="검색 정확도 평가")
    parser.add_argument(
        '--output',
        type=str,
        default='./data/captions/search_accuracy_report.json',
        help='평가 결과 저장 경로'
    )
    parser.add_argument(
        '--k',
        type=int,
        nargs='+',
        default=[1, 3, 5, 10],
        help='평가할 K 값들'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='엄격한 평가 모드 (모든 키워드가 포함되어야 함)'
    )
    
    args = parser.parse_args()
    
    # 테스트 쿼리 생성
    test_queries = create_test_queries()
    
    # 평가 실행
    results = evaluate_search_accuracy(test_queries, k_values=args.k, strict=args.strict)
    
    # 결과 출력
    print_evaluation_report(results)
    
    # 결과 저장
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 평가 결과 저장: {output_file}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

