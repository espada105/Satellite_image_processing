"""
pgvector 없이 PostgreSQL 데이터베이스 설정

pgvector가 설치되어 있지 않은 경우, 기본 테이블만 생성합니다.
벡터 검색 기능은 나중에 pgvector 설치 후 추가할 수 있습니다.
"""
import psycopg

print("=" * 60)
print("PostgreSQL 데이터베이스 설정 (pgvector 없이)")
print("=" * 60)

host = "localhost"
port = 5432
user = "postgres"
password = "postgres"
database = "postgres"

print(f"\n연결 정보:")
print(f"  Host: {host}")
print(f"  Port: {port}")
print(f"  User: {user}")

print(f"\n연결 시도 중...")

try:
    # 연결 시도
    conn_string = f"host={host} port={port} user={user} password={password} dbname={database}"
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
    
    conn.close()
    
    # 이제 satellite_db에 연결하여 테이블 생성
    print(f"\n📊 테이블 생성 중...")
    conn_string2 = f"host={host} port={port} user={user} password={password} dbname=satellite_db"
    conn2 = psycopg.connect(conn_string2, autocommit=True)
    
    with conn2.cursor() as cursor2:
        # pgvector 없이 테이블 생성 (embedding은 TEXT로 저장)
        create_table_query = """
        CREATE TABLE IF NOT EXISTS satellite_images (
            id SERIAL PRIMARY KEY,
            image_id VARCHAR(255) UNIQUE NOT NULL,
            image_path TEXT NOT NULL,
            caption TEXT,
            embedding TEXT,  -- 임시로 TEXT로 저장 (나중에 vector 타입으로 변경)
            embedding_array REAL[],  -- 벡터를 배열로 저장
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
        
        # 메타데이터 인덱스
        cursor2.execute("""
            CREATE INDEX IF NOT EXISTS satellite_images_metadata_idx 
            ON satellite_images USING GIN (metadata);
        """)
        print(f"✅ 메타데이터 인덱스가 생성되었습니다.")
        
        # 이미지 경로 인덱스
        cursor2.execute("""
            CREATE INDEX IF NOT EXISTS satellite_images_image_id_idx 
            ON satellite_images (image_id);
        """)
        print(f"✅ 이미지 ID 인덱스가 생성되었습니다.")
        
        # 임베딩 배열 인덱스 (벡터 검색 대신 사용)
        cursor2.execute("""
            CREATE INDEX IF NOT EXISTS satellite_images_embedding_array_idx 
            ON satellite_images USING GIN (embedding_array);
        """)
        print(f"✅ 임베딩 배열 인덱스가 생성되었습니다.")
    
    conn2.close()
    
    print(f"\n" + "=" * 60)
    print(f"✅ 기본 설정이 완료되었습니다!")
    print(f"=" * 60)
    print(f"\n⚠️  참고:")
    print(f"   - pgvector 확장이 설치되지 않아 벡터 타입을 사용하지 않습니다")
    print(f"   - 임베딩은 TEXT와 REAL[] 배열로 저장됩니다")
    print(f"   - 나중에 pgvector를 설치하면 vector 타입으로 마이그레이션할 수 있습니다")
    print(f"\n💡 다음 단계:")
    print(f"   - 데이터 삽입 스크립트 실행:")
    print(f"     venv/Scripts/python.exe scripts/database/insert_to_db.py")
    
except psycopg.OperationalError as e:
    print(f"\n❌ 연결 실패: {e}")
    if "password" in str(e).lower() or "인증" in str(e):
        print(f"\n💡 비밀번호가 잘못되었습니다.")
        print(f"   현재 비밀번호: {password}")
    elif "could not connect" in str(e).lower():
        print(f"\n💡 PostgreSQL 서비스가 실행 중인지 확인하세요.")
        
except Exception as e:
    print(f"\n❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()

