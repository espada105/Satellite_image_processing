"""
PostgreSQL 연결 문제 해결 스크립트

Windows 환경에서 psycopg2 인코딩 문제를 우회하여 연결합니다.
"""

import sys
import os

# 환경 변수 설정
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PGCLIENTENCODING'] = 'UTF8'

try:
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    
    print("=" * 60)
    print("PostgreSQL 연결 테스트 (인코딩 문제 해결 시도)")
    print("=" * 60)
    
    # 연결 정보 입력
    host = input("\nPostgreSQL 호스트 [localhost]: ").strip() or "localhost"
    port = input("PostgreSQL 포트 [5432]: ").strip() or "5432"
    user = input("PostgreSQL 사용자 [postgres]: ").strip() or "postgres"
    password = input("PostgreSQL 비밀번호: ").strip()
    database = input("데이터베이스 이름 [postgres]: ").strip() or "postgres"
    
    print(f"\n연결 시도 중...")
    
    # 연결 문자열 구성 (ASCII만 사용)
    conn_string = f"host={host} port={port} user={user} password={password} dbname={database}"
    
    try:
        conn = psycopg2.connect(conn_string)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # 버전 확인
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"✅ 연결 성공!")
        print(f"\nPostgreSQL 버전:")
        print(f"  {version}")
        
        # pgvector 확장 확인
        cursor.execute("""
            SELECT EXISTS(
                SELECT 1 FROM pg_extension WHERE extname = 'vector'
            );
        """)
        has_pgvector = cursor.fetchone()[0]
        
        if has_pgvector:
            print(f"\n✅ pgvector 확장이 설치되어 있습니다.")
        else:
            print(f"\n⚠️  pgvector 확장이 설치되어 있지 않습니다.")
            print(f"   설치 방법: CREATE EXTENSION vector;")
        
        cursor.close()
        conn.close()
        
        print(f"\n✅ 연결 테스트 성공!")
        print(f"\n이 정보를 .env 파일에 저장하세요:")
        print(f"POSTGRES_HOST={host}")
        print(f"POSTGRES_PORT={port}")
        print(f"POSTGRES_USER={user}")
        print(f"POSTGRES_PASSWORD={password}")
        print(f"POSTGRES_DB=satellite_db")
        
    except psycopg2.OperationalError as e:
        print(f"\n❌ 연결 실패: {e}")
        print(f"\n가능한 원인:")
        print(f"  1. PostgreSQL이 실행 중이 아닙니다")
        print(f"  2. 비밀번호가 잘못되었습니다")
        print(f"  3. 포트가 차단되어 있습니다")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError:
    print("❌ psycopg2가 설치되지 않았습니다.")
    print("설치: pip install psycopg2-binary")

