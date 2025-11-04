"""
PostgreSQL 연결 테스트 스크립트

간단하게 데이터베이스 연결만 테스트합니다.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.utils.config import POSTGRES_CONFIG

try:
    import psycopg2
    print("=" * 60)
    print("PostgreSQL 연결 테스트")
    print("=" * 60)
    print(f"\n연결 정보:")
    print(f"  Host: {POSTGRES_CONFIG['host']}")
    print(f"  Port: {POSTGRES_CONFIG['port']}")
    print(f"  User: {POSTGRES_CONFIG['user']}")
    print(f"  Database: {POSTGRES_CONFIG['database']}")
    
    print(f"\n연결 시도 중...")
    
    # 연결 문자열 직접 구성 (인코딩 문제 회피)
    conn_string = f"host={POSTGRES_CONFIG['host']} port={POSTGRES_CONFIG['port']} user={POSTGRES_CONFIG['user']} password={POSTGRES_CONFIG['password']} dbname=postgres"
    
    # 연결 시도
    conn = psycopg2.connect(conn_string)
    cursor = conn.cursor()
    
    # 버전 확인
    cursor.execute("SELECT version();")
    version = cursor.fetchone()[0]
    print(f"✅ 연결 성공!")
    print(f"\nPostgreSQL 버전:")
    print(f"  {version}")
    
    cursor.close()
    conn.close()
    
except psycopg2.OperationalError as e:
    print(f"\n❌ 연결 실패:")
    print(f"   {e}")
    print(f"\n가능한 원인:")
    print(f"   1. PostgreSQL이 설치되지 않았거나 실행 중이 아닙니다")
    print(f"   2. 연결 정보(호스트, 포트, 사용자, 비밀번호)가 잘못되었습니다")
    print(f"   3. 방화벽이 포트 5432를 차단하고 있습니다")
    print(f"\n해결 방법:")
    print(f"   - PostgreSQL 설치: https://www.postgresql.org/download/")
    print(f"   - 또는 Docker 사용: docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16")
    
except Exception as e:
    print(f"\n❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()

