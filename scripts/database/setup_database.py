"""PostgreSQL 데이터베이스 및 pgvector 확장 설정"""
import psycopg2
from psycopg2 import sql
from pgvector.psycopg2 import register_vector
import sys
import os

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from scripts.utils.config import POSTGRES_CONFIG


def create_database():
    """데이터베이스 생성 (없는 경우)"""
    # postgres 데이터베이스에 연결하여 새 DB 생성
    conn_config = POSTGRES_CONFIG.copy()
    conn_config["database"] = "postgres"
    
    try:
        conn = psycopg2.connect(**conn_config)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # 데이터베이스 존재 확인
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (POSTGRES_CONFIG["database"],)
        )
        
        if cursor.fetchone():
            print(f"데이터베이스 '{POSTGRES_CONFIG['database']}'가 이미 존재합니다.")
        else:
            cursor.execute(
                sql.SQL("CREATE DATABASE {}").format(
                    sql.Identifier(POSTGRES_CONFIG["database"])
                )
            )
            print(f"데이터베이스 '{POSTGRES_CONFIG['database']}'가 생성되었습니다.")
        
        cursor.close()
        conn.close()
    except psycopg2.Error as e:
        print(f"데이터베이스 생성 오류: {e}")
        raise


def setup_extensions():
    """pgvector 확장 설치"""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        register_vector(conn)
        cursor = conn.cursor()
        
        # pgvector 확장 설치
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("pgvector 확장이 설치되었습니다.")
        
        # PostGIS 확장 설치 (선택사항, 지리공간 검색용)
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
            print("PostGIS 확장이 설치되었습니다.")
        except psycopg2.Error:
            print("PostGIS 확장 설치 실패 (선택사항이므로 계속 진행합니다.)")
        
        conn.commit()
        cursor.close()
        conn.close()
    except psycopg2.Error as e:
        print(f"확장 설치 오류: {e}")
        raise


def create_tables():
    """위성 이미지 인덱스 테이블 생성"""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        register_vector(conn)
        cursor = conn.cursor()
        
        # 위성 이미지 테이블 생성
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
        
        cursor.execute(create_table_query)
        print("satellite_images 테이블이 생성되었습니다.")
        
        # 인덱스 생성
        create_index_query = """
        CREATE INDEX IF NOT EXISTS satellite_images_embedding_idx 
        ON satellite_images 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        """
        
        cursor.execute(create_index_query)
        print("벡터 인덱스가 생성되었습니다.")
        
        # 메타데이터 인덱스
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS satellite_images_metadata_idx 
            ON satellite_images USING GIN (metadata);
        """)
        print("메타데이터 인덱스가 생성되었습니다.")
        
        # 위치 인덱스 (PostGIS 사용 시)
        try:
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS satellite_images_location_idx 
                ON satellite_images 
                USING GIST (ST_SetSRID(ST_MakePoint(longitude, latitude), 4326));
            """)
            print("지리공간 인덱스가 생성되었습니다.")
        except psycopg2.Error:
            print("지리공간 인덱스 생성 실패 (PostGIS 미설치 가능)")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ 데이터베이스 설정이 완료되었습니다!")
        
    except psycopg2.Error as e:
        print(f"테이블 생성 오류: {e}")
        raise


def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("PostgreSQL 데이터베이스 설정 시작")
    print("=" * 50)
    
    try:
        # 1. 데이터베이스 생성
        create_database()
        
        # 2. 확장 설치
        setup_extensions()
        
        # 3. 테이블 생성
        create_tables()
        
        print("\n" + "=" * 50)
        print("모든 설정이 성공적으로 완료되었습니다!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

