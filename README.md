# 위성 이미지 처리 AI 프로젝트

위성 이미지를 활용한 지리공간 기반 멀티모달 AI 시스템 구축 프로젝트입니다.


## 📋 프로젝트 구조

```
satellite-ai-project/
├── docs/                    # 상세 문서
│   ├── 프로젝트_구축_계획서.md
│   ├── 단계별_실행_가이드.md
│   └── 프로젝트_구조_가이드.md
├── scripts/                 # 실행 스크립트
│   ├── data_collection/    # 데이터 수집
│   ├── database/            # DB 설정 및 검색
│   ├── finetuning/          # 모델 파인튜닝
│   └── utils/               # 유틸리티
├── api/                     # API 서버
├── agents/                  # AI 에이전트
├── docker/                  # Docker 설정
└── k8s/                     # Kubernetes 매니페스트
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론 (또는 프로젝트 디렉토리로 이동)
cd satellite-ai-project

# Python 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일을 편집하여 API 키 등을 설정
```

### 2. 데이터베이스 설정

```bash
# PostgreSQL 설치 (필요시)
# Ubuntu/Debian:
sudo apt-get install postgresql postgresql-contrib
sudo apt-get install postgresql-16-pgvector

# 데이터베이스 생성 및 설정
python scripts/database/setup_database.py
```

### 3. 단계별 실행

#### 1단계: 데이터 수집 및 RAG 구축

```bash
# 데이터 다운로드 (예: Sentinel-2)
python scripts/data_collection/download_sentinel2.py

# 이미지 전처리
python scripts/data_collection/preprocess_images.py

# 캡션 생성 및 임베딩
python scripts/data_collection/generate_captions.py
python scripts/data_collection/generate_embeddings.py

# 데이터베이스에 삽입
python scripts/database/insert_to_db.py
```

#### 2단계: 모델 파인튜닝

```bash
# QA 데이터셋 생성
python scripts/finetuning/generate_qa_dataset.py

# 모델 파인튜닝
python scripts/finetuning/finetune_llava.py

# 모델 평가
python scripts/finetuning/evaluate_model.py
```

#### 3단계: API 서버 및 에이전트

```bash
# 멀티모달 서버 실행
cd api
uvicorn multimodal_server:app --host 0.0.0.0 --port 8001

# 에이전트 서버 실행 (새 터미널)
cd agents
uvicorn main:app --host 0.0.0.0 --port 8002
```


## 🔧 기술 스택

### 데이터베이스
- **PostgreSQL 15+**: 관계형 데이터베이스
- **pgvector**: 벡터 검색 확장
- **PostGIS**: 지리공간 검색 (선택)

### AI/ML
- **PyTorch**: 딥러닝 프레임워크
- **Transformers**: Hugging Face 모델 라이브러리
- **PEFT/LoRA**: 파인튜닝 최적화
- **sentence-transformers**: 텍스트 임베딩

### 에이전트 및 API
- **LangGraph**: 에이전트 워크플로우
- **LangChain**: LLM 통합
- **FastAPI**: REST API 서버
- **vLLM**: 고속 LLM 서빙 (선택)

### 배포 및 모니터링
- **Docker**: 컨테이너화
- **Kubernetes**: 오케스트레이션
- **Langfuse**: 추적 및 모니터링

## 📊 프로젝트 타임라인

| 단계 | 작업 | 예상 기간 |
|------|------|-----------|
| 1단계 | 데이터 수집 및 RAG 구축 | 5-8일 |
| 2단계 | 멀티모달 모델 파인튜닝 | 7-11일 |
| 3단계 | AI 에이전트 및 API 서버 | 8-9일 |
| 4단계 | 프로덕션 배포 및 모니터링 | 9-10일 |
| **총계** | | **29-38일 (약 4-6주)** |

## 🎯 성공 기준

- ✅ 최소 500개의 위성 이미지가 벡터DB에 인덱싱됨
- ✅ 파인튜닝된 모델의 성능이 베이스 모델 대비 10% 이상 향상
- ✅ 사용자 질문에 대한 전체 워크플로우가 5초 내 응답
- ✅ Kubernetes 클러스터에 성공적으로 배포
- ✅ Langfuse 대시보드에서 모든 추적 정보 확인 가능

## 💻 하드웨어 요구사항

- **GPU**: NVIDIA GPU (최소 16GB VRAM, 권장: A100 또는 V100)
- **RAM**: 최소 32GB, 권장 64GB
- **저장공간**: 최소 500GB (모델 파일 + 데이터셋)

## 🔐 환경 변수 설정

```bash
# 데이터베이스
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=satellite_db

# OpenAI API
OPENAI_API_KEY=your-openai-api-key

# Langfuse
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
```

## 📝 간단 요약

- 데이터: SpaceNet SN8(PRE/POST-event) 타일 → 전처리 이미지/메타데이터, BLIP-2 캡션, Sentence-Transformers 임베딩(384d), PostgreSQL에 메타+임베딩(REAL[]) 저장
- 프레임워크: FastAPI(서버), LangGraph/LangChain(에이전트), sentence-transformers(임베딩), psycopg3(DB), NumPy. pgvector 없이 코사인 유사도 검색

## 🚀 서버 실행/사용

- 실행:
```bash
venv/Scripts/python.exe -m uvicorn scripts.api.server:app --host 0.0.0.0 --port 8000
```

- 접속: `http://127.0.0.1:8000`
- 사용: 입력창에 질문 → 답변 + 관련 이미지(기본 6개)와 캡션/유사도 표시
- API 예시(선택):
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"golf course near river","top_k":6}'
```

