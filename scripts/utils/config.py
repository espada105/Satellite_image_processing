"""공통 설정 파일"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 데이터 경로
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
QA_DATASET_DIR = DATA_DIR / "qa_dataset"

# 모델 경로
MODELS_DIR = PROJECT_ROOT / "models"
BASE_MODEL_DIR = MODELS_DIR / "llava-base"
FINETUNED_MODEL_DIR = MODELS_DIR / "llava-satellite-7b-lora"

# 데이터베이스 설정
POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    "database": os.getenv("POSTGRES_DB", "satellite_db")
}

# 모델 설정
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CAPTION_MODEL = os.getenv("CAPTION_MODEL", "Salesforce/blip-image-captioning-large")
LLAVA_MODEL = os.getenv("LLAVA_MODEL", "liuhaotian/llava-v1.5-7b")

# API 설정
MULTIMODAL_API_URL = os.getenv("MULTIMODAL_API_URL", "http://localhost:8001")
AGENT_API_URL = os.getenv("AGENT_API_URL", "http://localhost:8002")

# OpenAI 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Langfuse 설정
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# GPU 설정
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")

