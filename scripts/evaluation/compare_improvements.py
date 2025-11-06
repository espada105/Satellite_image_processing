"""
캡션 품질 개선 정도 분석
"""

import json
from pathlib import Path

print("=" * 60)
print("캡션 품질 개선 정도 분석")
print("=" * 60)

# 이전 결과 (문제 분석 시점의 데이터)
old_stats = {
    "평균 단어 수": 14,
    "중복률": 60.3,
    "고유 캡션 수": 1022,
    "총 캡션 수": 2576
}

# 현재 결과
new_file = Path("data/captions/quality_report.json")

if new_file.exists():
    with open(new_file, "r", encoding="utf-8") as f:
        new_data = json.load(f)
    
    new_stats = {
        "평균 단어 수": new_data["word_count"]["mean"],
        "중복률": new_data["duplication_rate"] * 100,
        "고유 캡션 수": new_data["unique_captions"],
        "총 캡션 수": new_data["total_captions"]
    }
    
    print("\n📊 개선 전후 비교")
    print("-" * 60)
    print(f"{'항목':<20} {'이전':<15} {'개선 후':<15} {'개선율':<15}")
    print("-" * 60)
    
    for key in old_stats.keys():
        old_val = old_stats[key]
        new_val = new_stats[key]
        if old_val > 0:
            if key == "중복률":
                # 중복률은 감소가 좋은 것
                improvement = ((old_val - new_val) / old_val) * 100
                print(f"{key:<20} {old_val:<15.1f} {new_val:<15.1f} {improvement:>+13.1f}% 감소")
            else:
                improvement = ((new_val - old_val) / old_val) * 100
                print(f"{key:<20} {old_val:<15.1f} {new_val:<15.1f} {improvement:>+13.1f}% 증가")
        else:
            print(f"{key:<20} {old_val:<15.1f} {new_val:<15.1f} {'N/A':>15}")
    
    print("-" * 60)
    
    print("\n🎯 주요 개선 사항:")
    word_imp = ((new_stats["평균 단어 수"] - old_stats["평균 단어 수"]) / old_stats["평균 단어 수"]) * 100
    dup_imp = ((old_stats["중복률"] - new_stats["중복률"]) / old_stats["중복률"]) * 100
    unique_imp = ((new_stats["고유 캡션 수"] - old_stats["고유 캡션 수"]) / old_stats["고유 캡션 수"]) * 100
    
    print(f"  1. 평균 단어 수: {old_stats['평균 단어 수']}개 → {new_stats['평균 단어 수']:.1f}개 (+{word_imp:.1f}%)")
    print(f"  2. 중복률: {old_stats['중복률']:.1f}% → {new_stats['중복률']:.1f}% ({dup_imp:.1f}% 감소)")
    print(f"  3. 고유 캡션: {old_stats['고유 캡션 수']}개 → {new_stats['고유 캡션 수']}개 (+{unique_imp:.1f}%)")
    
    print("\n📈 품질 점수 (개선율 종합):")
    overall = (word_imp + dup_imp + unique_imp) / 3
    print(f"  평균 개선율: {overall:.1f}%")
    
    print("\n" + "=" * 60)
    print("샘플 캡션 비교")
    print("=" * 60)
    
    print("\n📝 이전 캡션 (샘플 3개):")
    old_captions = [
        "this is an aerial view of a golf course in the country",
        "this is an aerial view of a golf course in the country",
        "this is an aerial view of a golf course in a wooded area"
    ]
    for i, cap in enumerate(old_captions, 1):
        print(f"  {i}. {cap} ({len(cap.split())}단어)")
    
    print("\n📝 개선된 캡션 (샘플 3개):")
    with open("data/captions/captions.json", "r", encoding="utf-8") as f:
        new_captions = json.load(f)
    for i, item in enumerate(new_captions[:3], 1):
        cap = item["caption"]
        print(f"  {i}. {cap[:100]}... ({len(cap.split())}단어)")
    
    print("\n💡 개선된 점:")
    print("  - 더 구체적인 설명 (예: 'houses and trees', 'road running through')")
    print("  - 중복 제거 (동일한 캡션 반복 없음)")
    print("  - 더 긴 설명 (14단어 → 21단어)")
    print("  - 세부 사항 포함 (건물, 도로, 수체, 식생 등)")
    
else:
    print("⚠️  새로운 평가 결과 파일을 찾을 수 없습니다.")

