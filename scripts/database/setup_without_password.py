"""
비밀번호 없이 PostgreSQL 연결 및 설정

pg_hba.conf 파일에서 'md5'를 'trust'로 변경한 후 실행하세요.
"""
import psycopg
from pgvector.psycopg import register_vector

print("=" * 60)
print("PostgreSQL 데이터베이스 설정 (비밀번호 없이)")
print("=" * 60)

host = "localhost"
port = 5432
user = "postgres"
database = "postgres"

print(f"\n연결 정보:")
print(f"  Host: {host}")
print(f"  Port: {port}")
print(f"  User: {user}")
print(f"  (비밀번호 없이 연결 시도)")

print(f"\n연결 시도 중...")

try:
    # 비밀번호 없이 연결 시도 (autocommit 먼저 설정)
    conn_string = f"host={host} port={port} user={user} dbname={database}"
    conn = psycopg.connect(conn_string, connect_timeout=5, autocommit=True)
    
    with conn.cursor() as cursor:
        # 버전 확인
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"✅ 연결 성공!")
        print(f"\nPostgreSQL 버전:")
        print(f"  {version[:80]}...")
        
        # 데이터베이스 존재 확인
        cursor.execute("SELECT datname FROM pg_database WHERE datname = 'satellite_db'")
        db_exists = cursor.fetchone()
        
        if db_exists:
            print(f"\n✅ satellite_db 데이터베이스가 이미 존재합니다.")
        else:
            print(f"\n📝 satellite_db 데이터베이스 생성 중...")
            cursor.execute("CREATE DATABASE satellite_db")
            print(f"✅ satellite_db 데이터베이스가 생성되었습니다.")
        
        # 비밀번호 재설정
        print(f"\n🔑 PostgreSQL 비밀번호 재설정 중...")
        new_password = "postgres"  # 원하는 비밀번호로 변경 가능
        cursor.execute(f"ALTER USER {user} WITH PASSWORD '{new_password}';")
        print(f"✅ 비밀번호가 '{new_password}'로 설정되었습니다.")
        print(f"\n💡 이 비밀번호를 .env 파일에 저장하세요:")
        print(f"   POSTGRES_PASSWORD={new_password}")
    
    conn.close()
    
    # 이제 satellite_db에 연결하여 확장 설치
    print(f"\n📦 pgvector 확장 설치 중...")
    conn_string2 = f"host={host} port={port} user={user} dbname=satellite_db"
    conn2 = psycopg.connect(conn_string2)
    register_vector(conn2)
    
    with conn2.cursor() as cursor2:
        cursor2.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print(f"✅ pgvector 확장이 설치되었습니다.")
        
        # 테이블 생성
        print(f"\n📊 테이블 생성 중...")
        create_table_query = """
        CREATE TABLE IF NOT EXISTS satellite_images (
            id SERIAL PRIMARY KEY,
            image_id VARCHAR(255) UNIQUE NOT NULL,
            image_path TEXT NOT NULL,
            caption TEXT,
            embedding vector(384),
            metadata JSONB,
            location_name VARCHAR(255),
            latitude DECIMAL(10, 8),
            longitude DECIMAL(11, 8),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            image_date DATE
        );
        """
        cursor2.execute(create_table_query)
        print(f"✅ satellite_images 테이블이 생성되었습니다.")
        
        # 인덱스 생성
        cursor2.execute("""
            CREATE INDEX IF NOT EXISTS satellite_images_embedding_idx 
            ON satellite_images 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        print(f"✅ 벡터 인덱스가 생성되었습니다.")
        
        cursor2.execute("""
            CREATE INDEX IF NOT EXISTS satellite_images_metadata_idx 
            ON satellite_images USING GIN (metadata);
        """)
        print(f"✅ 메타데이터 인덱스가 생성되었습니다.")
    
    conn2.commit()
    conn2.close()
    
    print(f"\n" + "=" * 60)
    print(f"✅ 모든 설정이 완료되었습니다!")
    print(f"=" * 60)
    print(f"\n⚠️  보안을 위해 pg_hba.conf 파일을 다시 'md5'로 변경하세요!")
    
except psycopg.OperationalError as e:
    print(f"\n❌ 연결 실패: {e}")
    if "password" in str(e).lower() or "인증" in str(e):
        print(f"\n💡 pg_hba.conf 파일에서 'md5'를 'trust'로 변경했는지 확인하세요.")
        print(f"   파일 경로: C:\\Program Files\\PostgreSQL\\17\\data\\pg_hba.conf")
        print(f"   변경 후 PostgreSQL 서비스를 재시작하세요:")
        print(f"   net stop postgresql-x64-17")
        print(f"   net start postgresql-x64-17")
    elif "could not connect" in str(e).lower():
        print(f"\n💡 PostgreSQL 서비스가 실행 중인지 확인하세요.")
        
except Exception as e:
    print(f"\n❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()

