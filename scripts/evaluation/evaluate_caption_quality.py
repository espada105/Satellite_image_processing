"""
캡션 품질 평가 스크립트

생성된 캡션의 품질을 평가합니다:
- 캡션 길이 통계
- 단어 다양성 분석
- 캡션 샘플 검토
"""

import sys
import os
import json
from pathlib import Path
from collections import Counter
import statistics
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.utils.config import CAPTIONS_DIR


def load_captions(captions_file: Path) -> List[Dict]:
    """캡션 파일 로드"""
    with open(captions_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_caption_lengths(captions: List[Dict]) -> Dict:
    """캡션 길이 분석"""
    lengths = []
    word_counts = []
    
    for item in captions:
        caption = item.get('caption', '')
        lengths.append(len(caption))
        word_counts.append(len(caption.split()))
    
    return {
        'total_captions': len(captions),
        'char_length': {
            'mean': statistics.mean(lengths) if lengths else 0,
            'median': statistics.median(lengths) if lengths else 0,
            'min': min(lengths) if lengths else 0,
            'max': max(lengths) if lengths else 0,
            'std': statistics.stdev(lengths) if len(lengths) > 1 else 0
        },
        'word_count': {
            'mean': statistics.mean(word_counts) if word_counts else 0,
            'median': statistics.median(word_counts) if word_counts else 0,
            'min': min(word_counts) if word_counts else 0,
            'max': max(word_counts) if word_counts else 0,
            'std': statistics.stdev(word_counts) if len(word_counts) > 1 else 0
        }
    }


def analyze_vocabulary(captions: List[Dict]) -> Dict:
    """어휘 다양성 분석"""
    all_words = []
    unique_captions = set()
    
    for item in captions:
        caption = item.get('caption', '').lower()
        if caption:
            unique_captions.add(caption)
            words = caption.split()
            all_words.extend(words)
    
    word_counter = Counter(all_words)
    total_words = len(all_words)
    unique_words = len(word_counter)
    
    # 가장 빈번한 단어 상위 20개
    top_words = word_counter.most_common(20)
    
    return {
        'total_words': total_words,
        'unique_words': unique_words,
        'vocabulary_diversity': unique_words / total_words if total_words > 0 else 0,
        'unique_captions': len(unique_captions),
        'duplicate_rate': 1 - (len(unique_captions) / len(captions)) if captions else 0,
        'top_words': top_words,
        'avg_words_per_caption': total_words / len(captions) if captions else 0
    }


def analyze_caption_content(captions: List[Dict]) -> Dict:
    """캡션 내용 분석 (키워드 빈도)"""
    # 위성 이미지 관련 키워드
    keywords = {
        'building': ['building', 'buildings', 'structure', 'structures'],
        'water': ['water', 'river', 'lake', 'ocean', 'sea', 'flood', 'flooded'],
        'road': ['road', 'street', 'highway', 'path', 'bridge'],
        'vegetation': ['tree', 'trees', 'forest', 'vegetation', 'green', 'grass'],
        'urban': ['city', 'urban', 'residential', 'commercial'],
        'golf': ['golf', 'golf course', 'course'],
        'aerial': ['aerial', 'satellite', 'overhead', 'view', 'top']
    }
    
    keyword_counts = {category: 0 for category in keywords.keys()}
    
    for item in captions:
        caption = item.get('caption', '').lower()
        for category, terms in keywords.items():
            if any(term in caption for term in terms):
                keyword_counts[category] += 1
    
    total = len(captions)
    keyword_percentages = {
        category: (count / total * 100) if total > 0 else 0
        for category, count in keyword_counts.items()
    }
    
    return {
        'keyword_counts': keyword_counts,
        'keyword_percentages': keyword_percentages
    }


def get_sample_captions(captions: List[Dict], n: int = 10) -> List[Dict]:
    """랜덤 샘플 캡션 선택"""
    import random
    return random.sample(captions, min(n, len(captions)))


def evaluate_caption_quality(captions_file: Path) -> Dict:
    """캡션 품질 종합 평가"""
    print("=" * 60)
    print("캡션 품질 평가 시작")
    print("=" * 60)
    
    # 캡션 로드
    print(f"\n📂 캡션 파일 로드: {captions_file}")
    captions = load_captions(captions_file)
    print(f"✅ 총 {len(captions)}개 캡션 로드 완료")
    
    # 길이 분석
    print("\n📏 캡션 길이 분석 중...")
    length_stats = analyze_caption_lengths(captions)
    
    # 어휘 분석
    print("📚 어휘 다양성 분석 중...")
    vocab_stats = analyze_vocabulary(captions)
    
    # 내용 분석
    print("🔍 캡션 내용 분석 중...")
    content_stats = analyze_caption_content(captions)
    
    # 샘플 선택
    print("📝 샘플 캡션 선택 중...")
    samples = get_sample_captions(captions, n=10)
    
    # 결과 종합
    results = {
        'length_statistics': length_stats,
        'vocabulary_statistics': vocab_stats,
        'content_statistics': content_stats,
        'sample_captions': [
            {
                'image_id': s.get('image_id', 'N/A'),
                'caption': s.get('caption', 'N/A'),
                'char_length': len(s.get('caption', '')),
                'word_count': len(s.get('caption', '').split())
            }
            for s in samples
        ]
    }
    
    return results


def print_evaluation_report(results: Dict):
    """평가 결과 출력"""
    print("\n" + "=" * 60)
    print("캡션 품질 평가 결과")
    print("=" * 60)
    
    # 길이 통계
    length = results['length_statistics']
    print(f"\n📏 캡션 길이 통계")
    print(f"  총 캡션 수: {length['total_captions']}")
    print(f"\n  문자 길이:")
    print(f"    평균: {length['char_length']['mean']:.1f}자")
    print(f"    중앙값: {length['char_length']['median']:.1f}자")
    print(f"    범위: {length['char_length']['min']} ~ {length['char_length']['max']}자")
    print(f"    표준편차: {length['char_length']['std']:.1f}")
    
    print(f"\n  단어 수:")
    print(f"    평균: {length['word_count']['mean']:.1f}개")
    print(f"    중앙값: {length['word_count']['median']:.1f}개")
    print(f"    범위: {length['word_count']['min']} ~ {length['word_count']['max']}개")
    
    # 어휘 통계
    vocab = results['vocabulary_statistics']
    print(f"\n📚 어휘 다양성")
    print(f"  총 단어 수: {vocab['total_words']:,}개")
    print(f"  고유 단어 수: {vocab['unique_words']:,}개")
    print(f"  어휘 다양성: {vocab['vocabulary_diversity']:.3f}")
    print(f"  고유 캡션 수: {vocab['unique_captions']:,}개")
    print(f"  중복률: {vocab['duplicate_rate']:.1%}")
    print(f"  캡션당 평균 단어 수: {vocab['avg_words_per_caption']:.1f}개")
    
    print(f"\n  가장 빈번한 단어 (상위 10개):")
    for word, count in vocab['top_words'][:10]:
        print(f"    '{word}': {count}회 ({count/vocab['total_words']*100:.1f}%)")
    
    # 내용 분석
    content = results['content_statistics']
    print(f"\n🔍 키워드 분석")
    for category, percentage in content['keyword_percentages'].items():
        count = content['keyword_counts'][category]
        print(f"  {category}: {count}개 ({percentage:.1f}%)")
    
    # 샘플 캡션
    print(f"\n📝 샘플 캡션 (10개):")
    for i, sample in enumerate(results['sample_captions'], 1):
        print(f"\n  {i}. [{sample['image_id']}]")
        print(f"     캡션: {sample['caption']}")
        print(f"     길이: {sample['char_length']}자, {sample['word_count']}단어")


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="캡션 품질 평가")
    parser.add_argument(
        '--captions_file',
        type=str,
        default=str(CAPTIONS_DIR / "captions.json"),
        help='캡션 JSON 파일 경로'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=str(CAPTIONS_DIR / "quality_report.json"),
        help='평가 결과 저장 경로'
    )
    
    args = parser.parse_args()
    
    captions_file = Path(args.captions_file)
    if not captions_file.exists():
        print(f"❌ 캡션 파일을 찾을 수 없습니다: {captions_file}")
        return
    
    # 평가 실행
    results = evaluate_caption_quality(captions_file)
    
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

