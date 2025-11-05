"""
종합 품질 평가 스크립트

캡션 품질과 검색 정확도를 모두 평가합니다.
"""

import sys
import os
from pathlib import Path
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.evaluation.evaluate_caption_quality import evaluate_caption_quality, print_evaluation_report as print_caption_report
from scripts.evaluation.evaluate_search_accuracy import evaluate_search_accuracy, print_evaluation_report as print_search_report
from scripts.utils.config import CAPTIONS_DIR


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="종합 품질 평가")
    parser.add_argument(
        '--captions_file',
        type=str,
        default=str(CAPTIONS_DIR / "captions.json"),
        help='캡션 JSON 파일 경로'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=str(CAPTIONS_DIR),
        help='평가 결과 저장 디렉토리'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("종합 품질 평가 시작")
    print("=" * 60)
    
    # 1. 캡션 품질 평가
    print("\n" + "=" * 60)
    print("1단계: 캡션 품질 평가")
    print("=" * 60)
    
    captions_file = Path(args.captions_file)
    if not captions_file.exists():
        print(f"❌ 캡션 파일을 찾을 수 없습니다: {captions_file}")
        return
    
    caption_results = evaluate_caption_quality(captions_file)
    print_caption_report(caption_results)
    
    caption_output = output_dir / "quality_report.json"
    with open(caption_output, 'w', encoding='utf-8') as f:
        json.dump(caption_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 캡션 평가 결과 저장: {caption_output}")
    
    # 2. 검색 정확도 평가
    print("\n" + "=" * 60)
    print("2단계: 검색 정확도 평가")
    print("=" * 60)
    
    search_results = evaluate_search_accuracy(
        k_values=[1, 3, 5, 10]
    )
    print_search_report(search_results)
    
    search_output = output_dir / "search_accuracy_report.json"
    with open(search_output, 'w', encoding='utf-8') as f:
        json.dump(search_results, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 검색 평가 결과 저장: {search_output}")
    
    # 3. 종합 요약
    print("\n" + "=" * 60)
    print("종합 평가 요약")
    print("=" * 60)
    
    # 캡션 품질 요약
    caption_stats = caption_results['length_statistics']
    vocab_stats = caption_results['vocabulary_statistics']
    
    print(f"\n📝 캡션 품질:")
    print(f"  총 캡션 수: {caption_stats['total_captions']:,}개")
    print(f"  평균 단어 수: {caption_stats['word_count']['mean']:.1f}개")
    print(f"  어휘 다양성: {vocab_stats['vocabulary_diversity']:.3f}")
    print(f"  중복률: {vocab_stats['duplicate_rate']:.1%}")
    
    # 검색 정확도 요약
    search_stats = search_results['overall_statistics']
    
    print(f"\n🔍 검색 정확도:")
    print(f"  성공한 쿼리: {search_stats['successful_queries']}/{search_stats['total_queries']}개")
    print(f"  Precision@5: {search_stats['precision_at_k'][5]['mean']:.3f}")
    print(f"  Recall@5: {search_stats['recall_at_k'][5]['mean']:.3f}")
    print(f"  MRR: {search_stats['mrr']['mean']:.3f}")
    print(f"  평균 유사도: {search_stats['similarity']['mean']:.3f}")
    
    # 종합 리포트 저장
    summary = {
        'caption_quality': {
            'total_captions': caption_stats['total_captions'],
            'avg_word_count': caption_stats['word_count']['mean'],
            'vocabulary_diversity': vocab_stats['vocabulary_diversity'],
            'duplicate_rate': vocab_stats['duplicate_rate']
        },
        'search_accuracy': {
            'successful_queries': search_stats['successful_queries'],
            'total_queries': search_stats['total_queries'],
            'precision_at_5': search_stats['precision_at_k'][5]['mean'],
            'recall_at_5': search_stats['recall_at_k'][5]['mean'],
            'mrr': search_stats['mrr']['mean'],
            'avg_similarity': search_stats['similarity']['mean']
        }
    }
    
    summary_output = output_dir / "evaluation_summary.json"
    with open(summary_output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 종합 요약 저장: {summary_output}")
    print("\n" + "=" * 60)
    print("🎉 평가 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

