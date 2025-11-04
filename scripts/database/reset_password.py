"""
PostgreSQL 비밀번호 재설정 가이드

PostgreSQL 비밀번호를 모르는 경우, 다음 방법으로 재설정할 수 있습니다.
"""

import os
from pathlib import Path

print("=" * 60)
print("PostgreSQL 비밀번호 재설정 가이드")
print("=" * 60)

# PostgreSQL 설치 경로 확인
pg_path = Path("C:/Program Files/PostgreSQL/17")
pg_hba_path = pg_path / "data" / "pg_hba.conf"
pg_data_path = pg_path / "data"

print(f"\n📁 PostgreSQL 설치 경로:")
print(f"   {pg_path}")
print(f"\n📁 설정 파일 경로:")
print(f"   {pg_hba_path}")

if pg_hba_path.exists():
    print(f"\n✅ pg_hba.conf 파일을 찾았습니다!")
    
    print(f"\n" + "=" * 60)
    print("비밀번호 재설정 방법:")
    print("=" * 60)
    
    print(f"\n방법 1: pg_hba.conf 파일 수정 (권장)")
    print(f"\n1. 관리자 권한으로 다음 파일을 엽니다:")
    print(f"   {pg_hba_path}")
    print(f"\n2. 파일 끝부분에서 다음 줄을 찾습니다:")
    print(f"   # IPv4 local connections:")
    print(f"   host    all    all    127.0.0.1/32    md5")
    print(f"\n3. 'md5'를 'trust'로 변경합니다:")
    print(f"   host    all    all    127.0.0.1/32    trust")
    print(f"\n4. PostgreSQL 서비스를 재시작합니다:")
    print(f"   net stop postgresql-x64-17")
    print(f"   net start postgresql-x64-17")
    print(f"\n5. 이 스크립트를 다시 실행하면 비밀번호 없이 연결됩니다.")
    print(f"   그 후 비밀번호를 재설정할 수 있습니다.")
    
    print(f"\n" + "-" * 60)
    print(f"\n방법 2: PowerShell에서 직접 명령 실행")
    print(f"\n다음 명령어를 관리자 권한 PowerShell에서 실행하세요:")
    print(f"\n# 1. pg_hba.conf 파일 열기 (메모장)")
    print(f'   notepad "{pg_hba_path}"')
    print(f"\n# 2. 또는 직접 편집:")
    print(f'   (Get-Content "{pg_hba_path}") -replace "md5", "trust" | Set-Content "{pg_hba_path}"')
    print(f"\n# 3. PostgreSQL 서비스 재시작")
    print(f"   Restart-Service postgresql-x64-17")
    
    print(f"\n" + "-" * 60)
    print(f"\n방법 3: pgAdmin 사용")
    print(f"\n1. pgAdmin 4를 실행합니다")
    print(f"2. 서버에 연결 시 비밀번호를 저장하지 않은 경우")
    print(f"   → 서버를 우클릭 → Properties → Connection 탭에서")
    print(f"   → 비밀번호를 확인하거나 재설정할 수 있습니다")
    
else:
    print(f"\n⚠️  pg_hba.conf 파일을 찾을 수 없습니다.")
    print(f"   PostgreSQL이 다른 경로에 설치되어 있을 수 있습니다.")

print(f"\n" + "=" * 60)
print("참고:")
print("=" * 60)
print(f"\n- 'trust' 모드는 비밀번호 없이 연결을 허용합니다 (보안상 위험)")
print(f"- 비밀번호를 재설정한 후에는 다시 'md5'로 변경하는 것을 권장합니다")
print(f"- pg_hba.conf 파일 수정 후에는 반드시 PostgreSQL 서비스를 재시작해야 합니다")

