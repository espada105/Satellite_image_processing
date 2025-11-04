"""
pg_hba.conf 파일 자동 수정 및 서비스 재시작 시도

주의: 관리자 권한이 필요할 수 있습니다.
"""
import os
import shutil
from pathlib import Path

pg_hba_path = Path("C:/Program Files/PostgreSQL/17/data/pg_hba.conf")

print("=" * 60)
print("PostgreSQL pg_hba.conf 파일 수정")
print("=" * 60)

if not pg_hba_path.exists():
    print(f"\n❌ 파일을 찾을 수 없습니다: {pg_hba_path}")
    print(f"   PostgreSQL이 다른 경로에 설치되어 있을 수 있습니다.")
    exit(1)

print(f"\n📁 파일 경로: {pg_hba_path}")

# 백업 생성
backup_path = pg_hba_path.with_suffix('.conf.backup')
try:
    shutil.copy2(pg_hba_path, backup_path)
    print(f"✅ 백업 파일 생성: {backup_path}")
except PermissionError:
    print(f"\n❌ 권한 오류: 관리자 권한이 필요합니다.")
    print(f"\n💡 다음 방법을 시도하세요:")
    print(f"   1. 관리자 권한으로 PowerShell 실행")
    print(f"   2. 또는 파일을 직접 메모장으로 열어서 수정")
    print(f"      파일: {pg_hba_path}")
    exit(1)

# 파일 읽기
try:
    with open(pg_hba_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # md5를 trust로 변경
    if 'md5' in content:
        new_content = content.replace('md5', 'trust')
        
        # 파일 쓰기
        with open(pg_hba_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ pg_hba.conf 파일 수정 완료 (md5 → trust)")
        
        # 변경된 내용 확인
        changed_lines = [line for line in content.split('\n') if 'md5' in line]
        if changed_lines:
            print(f"\n변경된 줄:")
            for line in changed_lines[:3]:  # 처음 3줄만 표시
                print(f"  {line.strip()}")
        
        # 서비스 재시작 시도
        print(f"\n🔄 PostgreSQL 서비스 재시작 시도 중...")
        
        import subprocess
        try:
            # 서비스 중지
            result = subprocess.run(
                ['net', 'stop', 'postgresql-x64-17'],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print(f"✅ 서비스 중지 완료")
            else:
                print(f"⚠️  서비스 중지 실패 (이미 중지되었거나 권한 부족)")
                print(f"   출력: {result.stdout}")
        
            # 서비스 시작
            result = subprocess.run(
                ['net', 'start', 'postgresql-x64-17'],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print(f"✅ 서비스 시작 완료")
            else:
                print(f"⚠️  서비스 시작 실패")
                print(f"   출력: {result.stdout}")
                print(f"   오류: {result.stderr}")
                print(f"\n💡 수동으로 서비스를 재시작하세요:")
                print(f"   - 서비스 관리자 (services.msc)에서 재시작")
                print(f"   - 또는 관리자 권한 PowerShell에서:")
                print(f"     Restart-Service postgresql-x64-17")
        
        except subprocess.TimeoutExpired:
            print(f"⚠️  서비스 재시작 시간 초과")
        except FileNotFoundError:
            print(f"⚠️  'net' 명령어를 찾을 수 없습니다.")
            print(f"   수동으로 서비스를 재시작하세요.")
        except Exception as e:
            print(f"⚠️  서비스 재시작 중 오류: {e}")
            print(f"\n💡 수동으로 서비스를 재시작하세요:")
            print(f"   - 서비스 관리자 (services.msc)에서 재시작")
        
        print(f"\n" + "=" * 60)
        print(f"✅ 설정 완료!")
        print(f"=" * 60)
        print(f"\n다음 단계:")
        print(f"1. PostgreSQL 서비스가 재시작되었는지 확인")
        print(f"2. 다음 명령어로 데이터베이스 설정 실행:")
        print(f"   venv/Scripts/python.exe scripts/database/setup_without_password.py")
        
    else:
        print(f"⚠️  파일에 'md5'가 없습니다. 이미 'trust'로 설정되어 있을 수 있습니다.")
        print(f"   다음 명령어로 데이터베이스 설정을 실행해보세요:")
        print(f"   venv/Scripts/python.exe scripts/database/setup_without_password.py")

except PermissionError:
    print(f"\n❌ 권한 오류: 관리자 권한이 필요합니다.")
    print(f"\n💡 다음 방법을 시도하세요:")
    print(f"   1. 관리자 권한으로 이 스크립트 실행")
    print(f"   2. 또는 파일을 직접 메모장으로 열어서 수정:")
    print(f"      파일: {pg_hba_path}")
    print(f"      수정: 'md5' → 'trust'")
except Exception as e:
    print(f"\n❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()

