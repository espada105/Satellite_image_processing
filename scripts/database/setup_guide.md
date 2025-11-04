# PostgreSQL 비밀번호 재설정 가이드

## 문제
PostgreSQL 서비스를 재시작할 수 없습니다 (권한 부족).

## 해결 방법

### 방법 1: 관리자 권한 PowerShell 사용

1. **Windows 검색에서 PowerShell 찾기**
2. **우클릭 → "관리자 권한으로 실행"**
3. 다음 명령어 실행:
```powershell
# 디렉토리 이동
cd "C:\Program Files\PostgreSQL\17\data"

# pg_hba.conf 파일 열기 (메모장)
notepad pg_hba.conf

# 또는 직접 편집 (md5 → trust)
(Get-Content pg_hba.conf) -replace "md5", "trust" | Set-Content pg_hba.conf

# PostgreSQL 서비스 재시작
Restart-Service postgresql-x64-17
```

### 방법 2: 서비스 관리자 사용 (GUI)

1. **Windows 키 + R** → `services.msc` 입력
2. **postgresql-x64-17** 서비스 찾기
3. **우클릭 → 다시 시작**

### 방법 3: 명령 프롬프트 (관리자 권한)

1. **Windows 키 + X** → "Windows PowerShell (관리자)" 또는 "명령 프롬프트 (관리자)"
2. 다음 명령어 실행:
```cmd
net stop postgresql-x64-17
net start postgresql-x64-17
```

### 방법 4: pgAdmin 사용 (가장 쉬운 방법)

pgAdmin이 설치되어 있다면:
1. **pgAdmin 4 실행**
2. 서버에 연결 시도
3. 비밀번호 입력 창에서 "비밀번호 저장" 옵션 확인
4. 또는 서버 우클릭 → Properties → Connection 탭에서 비밀번호 확인/변경

### 방법 5: 직접 pg_hba.conf 수정 후 자동 재시작 시도

아래 스크립트를 실행하면 pg_hba.conf를 수정하고 서비스 재시작을 시도합니다:
```bash
venv/Scripts/python.exe scripts/database/modify_pg_hba.py
```

