"""
pg_hba.conf 파일에서 scram-sha-256을 trust로 변경

이미지에서 확인한 바로는 scram-sha-256을 사용하고 있습니다.
"""
import os
import shutil
from pathlib import Path

pg_hba_path = Path("C:/Program Files/PostgreSQL/17/data/pg_hba.conf")

print("=" * 60)
print("PostgreSQL pg_hba.conf 파일 수정 (scram-sha-256 → trust)")
print("=" * 60)

if not pg_hba_path.exists():
    print(f"\n❌ 파일을 찾을 수 없습니다: {pg_hba_path}")
    exit(1)

print(f"\n📁 파일 경로: {pg_hba_path}")

# 백업 생성
backup_path = pg_hba_path.with_suffix('.conf.backup2')
try:
    if not backup_path.exists():
        shutil.copy2(pg_hba_path, backup_path)
        print(f"✅ 백업 파일 생성: {backup_path}")
    else:
        print(f"✅ 백업 파일이 이미 존재합니다: {backup_path}")
except PermissionError:
    print(f"\n❌ 권한 오류: 관리자 권한이 필요합니다.")
    exit(1)

# 파일 읽기
try:
    with open(pg_hba_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 변경할 줄 찾기 및 변경
    changed = False
    new_lines = []
    
    for i, line in enumerate(lines, 1):
        # IPv4 local connections (127.0.0.1/32)에서 scram-sha-256을 trust로
        if '127.0.0.1/32' in line and 'scram-sha-256' in line and 'replication' not in line:
            new_line = line.replace('scram-sha-256', 'trust')
            new_lines.append(new_line)
            print(f"✅ 줄 {i} 변경: {line.strip()} → {new_line.strip()}")
            changed = True
        # IPv6 local connections (::1/128)에서 scram-sha-256을 trust로
        elif '::1/128' in line and 'scram-sha-256' in line and 'replication' not in line:
            new_line = line.replace('scram-sha-256', 'trust')
            new_lines.append(new_line)
            print(f"✅ 줄 {i} 변경: {line.strip()} → {new_line.strip()}")
            changed = True
        # local connections에서 scram-sha-256을 trust로 (replication 제외)
        elif line.strip().startswith('local') and 'scram-sha-256' in line and 'replication' not in line:
            new_line = line.replace('scram-sha-256', 'trust')
            new_lines.append(new_line)
            print(f"✅ 줄 {i} 변경: {line.strip()} → {new_line.strip()}")
            changed = True
        else:
            new_lines.append(line)
    
    if changed:
        # 파일 쓰기
        with open(pg_hba_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        print(f"\n✅ pg_hba.conf 파일 수정 완료!")
        print(f"\n💡 다음 단계:")
        print(f"   1. PostgreSQL 서비스를 재시작하세요:")
        print(f"      - 서비스 관리자 (services.msc)에서 재시작")
        print(f"      - 또는 관리자 권한 PowerShell: Restart-Service postgresql-x64-17")
        print(f"   2. 서비스 재시작 후 다음 명령어 실행:")
        print(f"      venv/Scripts/python.exe scripts/database/setup_without_password.py")
    else:
        print(f"\n⚠️  변경할 내용이 없습니다.")
        print(f"   이미 'trust'로 설정되어 있거나 'scram-sha-256'이 없습니다.")

except PermissionError:
    print(f"\n❌ 권한 오류: 관리자 권한이 필요합니다.")
    print(f"\n💡 수동으로 파일을 수정하세요:")
    print(f"   1. 관리자 권한으로 메모장 실행")
    print(f"   2. 다음 파일 열기: {pg_hba_path}")
    print(f"   3. 다음 줄을 찾아서:")
    print(f"      host    all    all    127.0.0.1/32    scram-sha-256")
    print(f"   4. 'scram-sha-256'을 'trust'로 변경")
    print(f"   5. 저장 후 PostgreSQL 서비스 재시작")
except Exception as e:
    print(f"\n❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()

