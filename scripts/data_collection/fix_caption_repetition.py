"""
캡션 반복 문제 후처리 스크립트

반복되는 단어나 구문을 감지하고 제거합니다.
"""

import json
import re
from pathlib import Path
from typing import List, Dict
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def detect_repetition(text: str) -> bool:
    """
    텍스트에 반복 패턴이 있는지 감지
    
    Args:
        text: 캡션 텍스트
    
    Returns:
        반복이 있으면 True
    """
    # 단어 단위로 분리
    words = text.lower().split()
    
    # 같은 단어가 3번 이상 연속으로 나타나는지 확인
    if len(words) < 3:
        return False
    
    for i in range(len(words) - 2):
        if words[i] == words[i+1] == words[i+2]:
            return True
    
    # 구문 반복 감지 (예: "country side of the country side")
    # 3개 이상의 단어가 연속으로 반복되는지 확인
    for length in range(2, min(5, len(words) // 2 + 1)):
        for i in range(len(words) - length * 2 + 1):
            segment1 = words[i:i+length]
            segment2 = words[i+length:i+length*2]
            if segment1 == segment2:
                return True
    
    return False


def fix_repetition(text: str) -> str:
    """
    반복 문제를 수정
    
    Args:
        text: 원본 캡션 텍스트
    
    Returns:
        수정된 캡션 텍스트
    """
    # 단어 단위로 분리
    words = text.split()
    
    if len(words) < 3:
        return text
    
    # 1. 연속된 동일 단어 제거 (2개 이상 연속이면 1개만 유지)
    fixed_words = []
    i = 0
    while i < len(words):
        word = words[i]
        count = 1
        # 연속된 같은 단어 개수 세기
        while i + count < len(words) and words[i + count].lower() == word.lower():
            count += 1
        
        # 2개 이상이면 1개만 유지
        if count >= 2:
            fixed_words.append(word)
            i += count
        else:
            fixed_words.append(word)
            i += 1
    
    # 2. 구문 반복 제거 (예: "country side of the country side")
    # 여러 번 반복될 수 있으므로 반복적으로 처리
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        result_words = []
        i = 0
        found_any = False
        
        while i < len(fixed_words):
            found_repetition = False
            
            # 2-6개 단어 구문이 반복되는지 확인 (더 긴 구문도 감지)
            for length in range(2, min(7, (len(fixed_words) - i) // 2 + 1)):
                if i + length * 2 > len(fixed_words):
                    continue
                
                segment1 = [w.lower() for w in fixed_words[i:i+length]]
                segment2 = [w.lower() for w in fixed_words[i+length:i+length*2]]
                
                if segment1 == segment2:
                    # 반복 구문 발견 - 첫 번째만 유지
                    result_words.extend(fixed_words[i:i+length])
                    # 추가 반복도 확인
                    j = i + length * 2
                    while j + length <= len(fixed_words):
                        segment_next = [w.lower() for w in fixed_words[j:j+length]]
                        if segment_next == segment1:
                            j += length
                        else:
                            break
                    i = j
                    found_repetition = True
                    found_any = True
                    break
            
            if not found_repetition:
                result_words.append(fixed_words[i])
                i += 1
        
        fixed_words = result_words
        
        if not found_any:
            break
        
        iteration += 1
    
    # 공백으로 다시 결합
    result = ' '.join(fixed_words)
    
    # 불필요한 공백 정리
    result = re.sub(r'\s+', ' ', result).strip()
    
    # 마지막 점검: 여전히 반복이 있으면 더 강력하게 제거
    words_final = result.split()
    if len(words_final) > 10:
        # 긴 캡션에서 반복 패턴이 있으면 처음 100단어만 유지
        # (일반적으로 반복은 끝부분에 발생)
        unique_segments = []
        seen = set()
        for i in range(len(words_final)):
            # 3단어 윈도우로 확인
            if i + 3 <= len(words_final):
                segment = ' '.join(words_final[i:i+3]).lower()
                if segment not in seen:
                    seen.add(segment)
                    unique_segments.extend(words_final[i:i+3])
                    break
            else:
                unique_segments.append(words_final[i])
        
        if len(unique_segments) < len(words_final) * 0.5:  # 너무 많이 줄어들면 원본 사용
            result = ' '.join(words_final[:100])  # 최대 100단어만
    
    return result


def process_captions(captions: List[Dict]) -> tuple:
    """
    캡션 리스트를 처리하여 반복 문제 수정
    
    Args:
        captions: 캡션 리스트
    
    Returns:
        (수정된 캡션 리스트, 통계 딕셔너리)
    """
    total = len(captions)
    fixed_count = 0
    repetition_detected = 0
    
    fixed_captions = []
    
    for item in captions:
        original = item['caption']
        
        # 반복 감지
        has_repetition = detect_repetition(original)
        if has_repetition:
            repetition_detected += 1
        
        # 수정 적용
        fixed = fix_repetition(original)
        
        if fixed != original:
            fixed_count += 1
            item['caption'] = fixed
            item['_original_caption'] = original  # 원본 보관 (선택적)
        
        fixed_captions.append(item)
    
    stats = {
        'total': total,
        'repetition_detected': repetition_detected,
        'fixed': fixed_count,
        'fix_rate': (fixed_count / total * 100) if total > 0 else 0
    }
    
    return fixed_captions, stats


def main():
    parser = argparse.ArgumentParser(
        description="캡션 반복 문제 후처리",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 기본 사용
  python fix_caption_repetition.py --input_file ./data/captions/captions_blip_improved.json --output_file ./data/captions/captions_blip_improved_fixed.json
  
  # 원본 보관 없이
  python fix_caption_repetition.py --input_file ./data/captions/captions_blip_improved.json --output_file ./data/captions/captions_blip_improved_fixed.json --no-keep-original
        """
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="입력 캡션 JSON 파일 경로"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="출력 캡션 JSON 파일 경로"
    )
    
    parser.add_argument(
        "--keep-original",
        action="store_true",
        default=False,
        help="원본 캡션을 _original_caption 필드에 보관"
    )
    
    parser.add_argument(
        "--no-keep-original",
        dest="keep_original",
        action="store_false",
        help="원본 캡션 보관하지 않음"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("캡션 반복 문제 후처리")
    print("=" * 60)
    
    # 파일 로드
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    
    if not input_file.exists():
        print(f"❌ 입력 파일을 찾을 수 없습니다: {input_file}")
        sys.exit(1)
    
    print(f"\n📖 캡션 파일 로드 중: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        captions = json.load(f)
    
    print(f"✅ {len(captions)}개 캡션 로드 완료")
    
    # 처리
    print(f"\n🔧 반복 문제 수정 중...")
    fixed_captions, stats = process_captions(captions)
    
    # 원본 보관 옵션 처리
    if not args.keep_original:
        for item in fixed_captions:
            item.pop('_original_caption', None)
    
    # 저장
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n💾 수정된 캡션 저장 중: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(fixed_captions, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 저장 완료!")
    
    # 통계 출력
    print("\n" + "=" * 60)
    print("📊 처리 통계")
    print("=" * 60)
    print(f"  총 캡션 수: {stats['total']}개")
    print(f"  반복 감지: {stats['repetition_detected']}개 ({stats['repetition_detected']/stats['total']*100:.1f}%)")
    print(f"  수정됨: {stats['fixed']}개 ({stats['fix_rate']:.1f}%)")
    
    # 샘플 출력
    print("\n" + "=" * 60)
    print("📝 수정 샘플 (처음 5개)")
    print("=" * 60)
    
    sample_count = 0
    for item in fixed_captions:
        if '_original_caption' in item:
            print(f"\n  원본: {item['_original_caption'][:100]}...")
            print(f"  수정: {item['caption'][:100]}...")
            sample_count += 1
            if sample_count >= 5:
                break
    
    print("\n" + "=" * 60)
    print("✅ 후처리 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

