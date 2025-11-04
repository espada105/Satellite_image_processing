"""
PostgreSQL 비밀번호 찾기/테스트 스크립트
"""
import psycopg

print("=" * 60)
print("PostgreSQL 비밀번호 테스트")
print("=" * 60)

# 시도할 비밀번호 목록
passwords = ["5712", "postgres", "", "admin", "1234"]

host = "localhost"
port = 5432
user = "postgres"
database = "postgres"

for password in passwords:
    try:
        conn_string = f"host={host} port={port} user={user} password={password} dbname={database}"
        conn = psycopg.connect(conn_string, connect_timeout=3)
        
        with conn.cursor() as cursor:
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            print(f"\n✅ 연결 성공!")
            print(f"   비밀번호: {'(비어있음)' if not password else '***'}")
            print(f"   PostgreSQL 버전: {version[:50]}...")
        
        conn.close()
        print(f"\n✅ 올바른 비밀번호를 찾았습니다!")
        break
        
    except psycopg.OperationalError as e:
        if "password" in str(e).lower() or "인증" in str(e):
            print(f"   비밀번호 '{password if password else '(비어있음)'}' 실패")
            continue
        else:
            print(f"   오류: {e}")
            break
    except Exception as e:
        print(f"   예상치 못한 오류: {e}")
        break
else:
    print(f"\n❌ 시도한 비밀번호로 연결할 수 없습니다.")
    print(f"\n💡 PostgreSQL 비밀번호를 확인하세요:")
    print(f"   1. pgAdmin에서 확인")
    print(f"   2. 또는 PostgreSQL 설정 파일에서 확인")
    print(f"   3. 또는 Windows 서비스에서 PostgreSQL 재설정")

