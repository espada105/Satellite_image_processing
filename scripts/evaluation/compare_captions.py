"""
이전 캡션과 개선된 캡션 비교
"""

import json
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def load_captions(file_path: Path):
    """캡션 파일 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_captions(captions, name):
    """캡션 분석"""
    total = len(captions)
    caption_texts = [c['caption'] for c in captions]
    unique_captions = len(set(caption_texts))
    duplicate_rate = (1 - unique_captions / total) * 100 if total > 0 else 0
    
    # 길이 분석
    word_counts = [len(c.split()) for c in caption_texts]
    char_counts = [len(c) for c in caption_texts]
    
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    avg_chars = sum(char_counts) / len(char_counts) if char_counts else 0
    
    # 키워드 분석
    keywords = {
        'road': sum(1 for c in caption_texts if 'road' in c.lower() or 'highway' in c.lower()),
        'water': sum(1 for c in caption_texts if 'water' in c.lower() or 'river' in c.lower() or 'lake' in c.lower()),
        'building': sum(1 for c in caption_texts if 'building' in c.lower() or 'house' in c.lower() or 'town' in c.lower()),
        'vegetation': sum(1 for c in caption_texts if 'tree' in c.lower() or 'forest' in c.lower() or 'field' in c.lower() or 'vegetation' in c.lower()),
    }
    
    return {
        'name': name,
        'total': total,
        'unique': unique_captions,
        'duplicate_rate': duplicate_rate,
        'avg_words': avg_words,
        'avg_chars': avg_chars,
        'keywords': keywords,
        'sample_captions': caption_texts[:5]
    }

def main():
    print("=" * 80)
    print("이전 캡션 vs 개선된 캡션 비교")
    print("=" * 80)
    
    # 파일 경로
    old_file = Path("data/captions/captions.json")
    new_file = Path("data/captions/captions_blip_improved.json")
    
    if not old_file.exists():
        print(f"❌ 이전 캡션 파일을 찾을 수 없습니다: {old_file}")
        return
    
    if not new_file.exists():
        print(f"❌ 개선된 캡션 파일을 찾을 수 없습니다: {new_file}")
        return
    
    # 캡션 로드
    print("\n📖 캡션 파일 로드 중...")
    old_captions = load_captions(old_file)
    new_captions = load_captions(new_file)
    
    # 분석
    print("📊 분석 중...")
    old_stats = analyze_captions(old_captions, "이전 캡션")
    new_stats = analyze_captions(new_captions, "개선된 캡션")
    
    # 비교 결과 출력
    print("\n" + "=" * 80)
    print("📊 비교 결과")
    print("=" * 80)
    
    print(f"\n{'항목':<25} {'이전':<20} {'개선 후':<20} {'변화':<15}")
    print("-" * 80)
    
    # 총 개수
    print(f"{'총 캡션 수':<25} {old_stats['total']:<20} {new_stats['total']:<20} {new_stats['total'] - old_stats['total']:>+14}")
    
    # 고유 캡션
    unique_diff = new_stats['unique'] - old_stats['unique']
    unique_pct = (unique_diff / old_stats['unique'] * 100) if old_stats['unique'] > 0 else 0
    print(f"{'고유 캡션 수':<25} {old_stats['unique']:<20} {new_stats['unique']:<20} {unique_pct:>+13.1f}%")
    
    # 중복률
    dup_diff = old_stats['duplicate_rate'] - new_stats['duplicate_rate']
    dup_pct = (dup_diff / old_stats['duplicate_rate'] * 100) if old_stats['duplicate_rate'] > 0 else 0
    print(f"{'중복률 (%)':<25} {old_stats['duplicate_rate']:<20.1f} {new_stats['duplicate_rate']:<20.1f} {dup_pct:>+13.1f}% 감소")
    
    # 평균 단어 수
    word_diff = new_stats['avg_words'] - old_stats['avg_words']
    word_pct = (word_diff / old_stats['avg_words'] * 100) if old_stats['avg_words'] > 0 else 0
    print(f"{'평균 단어 수':<25} {old_stats['avg_words']:<20.1f} {new_stats['avg_words']:<20.1f} {word_pct:>+13.1f}%")
    
    # 평균 문자 수
    char_diff = new_stats['avg_chars'] - old_stats['avg_chars']
    char_pct = (char_diff / old_stats['avg_chars'] * 100) if old_stats['avg_chars'] > 0 else 0
    print(f"{'평균 문자 수':<25} {old_stats['avg_chars']:<20.1f} {new_stats['avg_chars']:<20.1f} {char_pct:>+13.1f}%")
    
    print("\n" + "-" * 80)
    print("🔍 키워드 인식률 비교")
    print("-" * 80)
    
    for keyword in ['road', 'water', 'building', 'vegetation']:
        old_count = old_stats['keywords'][keyword]
        new_count = new_stats['keywords'][keyword]
        old_pct = (old_count / old_stats['total'] * 100) if old_stats['total'] > 0 else 0
        new_pct = (new_count / new_stats['total'] * 100) if new_stats['total'] > 0 else 0
        diff_pct = new_pct - old_pct
        print(f"{keyword.capitalize():<25} {old_pct:<20.1f}% {new_pct:<20.1f}% {diff_pct:>+13.1f}%p")
    
    print("\n" + "=" * 80)
    print("📝 샘플 캡션 비교")
    print("=" * 80)
    
    print("\n🔵 이전 캡션 (샘플 5개):")
    for i, cap in enumerate(old_stats['sample_captions'], 1):
        print(f"  {i}. {cap[:100]}{'...' if len(cap) > 100 else ''} ({len(cap.split())}단어)")
    
    print("\n🟢 개선된 캡션 (샘플 5개):")
    for i, cap in enumerate(new_stats['sample_captions'], 1):
        print(f"  {i}. {cap[:100]}{'...' if len(cap) > 100 else ''} ({len(cap.split())}단어)")
    
    print("\n" + "=" * 80)
    print("✅ 비교 완료!")
    print("=" * 80)

if __name__ == "__main__":
    main()

