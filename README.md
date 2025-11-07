# 위성 이미지 검색 AI Platform

위성 이미지를 활용한 지리공간 기반 멀티모달 AI 시스템 구축 프로젝트입니다.

## 메인 페이지
<img width="1918" height="908" alt="{5276CD66-B9D9-4682-B5A3-98654FF810A8}" src="https://github.com/user-attachments/assets/98467100-a64d-4741-9388-0cc482baa183" />

## 위성 이미지 검색 기본 페이지
<img width="1918" height="907" alt="{865D1ED5-DF5D-4B86-AC49-8D2EB94F45D7}" src="https://github.com/user-attachments/assets/36143394-d402-4eed-848c-7a23befc89ae" />

## 위성 이미지 검색 결과 페이지
<img width="1919" height="909" alt="{5A7F8550-31A1-481F-844D-4FE3E1D527B9}" src="https://github.com/user-attachments/assets/7282abe9-cfd1-4a26-9b06-4166cd8349c8" />

## 위성 이미지 검색 결과 확대
<img width="1915" height="906" alt="{0A868A41-0449-4E24-B5D9-D691063BF956}" src="https://github.com/user-attachments/assets/c20d5b2b-63da-4a56-965f-53e72bea0efe" />


## 📋 프로젝트 구조

```
Satellite_image_processing/
├── scripts/                      # 실행 스크립트
│   ├── data_collection/         # 데이터 수집 및 처리
│   │   ├── process_spacenet_data.py    # SpaceNet 데이터 처리
│   │   ├── generate_captions.py        # 이미지 캡션 생성 (BLIP, CUDA 병렬 처리)
│   │   ├── generate_embeddings.py      # 텍스트 임베딩 생성
│   │   └── fix_caption_repetition.py   # 캡션 반복 제거 후처리
│   ├── database/                # 데이터베이스 설정 및 검색
│   │   ├── setup_database.py          # PostgreSQL + pgvector 설정
│   │   ├── insert_to_db.py            # 데이터 삽입
│   │   └── search_vector_db_v2.py     # 벡터 유사도 검색
│   ├── agent/                   # AI 에이전트
│   │   └── satellite_agent.py         # LangGraph 기반 RAG 에이전트
│   ├── rag/                     # RAG 도구
│   │   └── rag_tool.py                # 벡터 검색 + LLM 통합
│   ├── api/                     # API 서버
│   │   ├── server.py                  # FastAPI 서버
│   │   └── web/                       # 웹 UI
│   │       ├── static/
│   │       │   ├── app.js            # 프론트엔드 JavaScript
│   │       │   ├── style.css         # 스타일시트
│   │       │   └── earth.jpg         # Hero 섹션 배경 이미지
│   │       └── templates/
│   ├── evaluation/              # 평가 스크립트
│   │   ├── evaluate_caption_quality.py
│   │   └── evaluate_search_accuracy.py
│   └── utils/                   # 유틸리티
│       ├── config.py                 # 설정 파일
│       └── check_cuda.py            # CUDA 확인
├── data/                        # 데이터 디렉토리
│   ├── raw/                     # 원본 데이터 (SpaceNet tar.gz)
│   ├── processed/               # 전처리된 이미지
│   ├── captions/                # 생성된 캡션 JSON
│   ├── embeddings/              # 생성된 임베딩
│   └── metadata/                # 이미지 메타데이터
├── SpaceNet/                    # SpaceNet 다운로드 스크립트
│   └── download.py
└── requirements.txt             # Python 의존성
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론 (또는 프로젝트 디렉토리로 이동)
cd Satellite_image_processing

# Python 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 데이터베이스 설정

```bash
# PostgreSQL 설치 (필요시)
# Ubuntu/Debian:
sudo apt-get install postgresql postgresql-contrib
sudo apt-get install postgresql-16-pgvector

# Windows: PostgreSQL 공식 사이트에서 설치
# https://www.postgresql.org/download/windows/

# 데이터베이스 생성 및 설정
python scripts/database/setup_database.py
```

### 3. 데이터 수집 및 처리

#### 3.1 SpaceNet 데이터 다운로드

```bash
# SpaceNet SN8 데이터 다운로드
python SpaceNet/download.py
```

#### 3.2 데이터 전처리

```bash
# SpaceNet 데이터 압축 해제 및 전처리
python scripts/data_collection/process_spacenet_data.py \
    --input_dir ./data/raw \
    --output_dir ./data/processed \
    --metadata_dir ./data/metadata
```

#### 3.3 캡션 생성

```bash
# GPU 사용 (CUDA 병렬 처리, FP16 최적화)
python scripts/data_collection/generate_captions.py \
    --image_dir ./data/processed \
    --output_file ./data/captions/captions.json \
    --batch_size 8 \
    --max_length 120 \
    --min_length 20 \
    --temperature 0.8 \
    --repetition_penalty 1.3 \
    --use_fp16

# CPU 사용 (느리지만 메모리 절약)
python scripts/data_collection/generate_captions.py \
    --image_dir ./data/processed \
    --output_file ./data/captions/captions.json \
    --device cpu \
    --batch_size 4
```

#### 3.4 캡션 반복 제거 (선택)

```bash
# 생성된 캡션의 반복 구문 제거
python scripts/data_collection/fix_caption_repetition.py \
    --input_file ./data/captions/captions.json \
    --output_file ./data/captions/captions_fixed.json
```

#### 3.5 임베딩 생성

```bash
# 텍스트 임베딩 생성 (sentence-transformers)
python scripts/data_collection/generate_embeddings.py \
    --captions_file ./data/captions/captions.json \
    --output_file ./data/embeddings/embeddings.npy \
    --batch_size 64
```

#### 3.6 데이터베이스에 삽입

```bash
# 캡션 및 임베딩을 PostgreSQL에 삽입
python scripts/database/insert_to_db.py \
    --captions_file ./data/captions/captions.json \
    --embeddings_file ./data/embeddings/embeddings.npy
```

### 4. 웹 서버 실행

```bash
# FastAPI 서버 실행
# Windows:
venv\Scripts\python.exe -m uvicorn scripts.api.server:app --host 0.0.0.0 --port 8000

# Linux/Mac:
python -m uvicorn scripts.api.server:app --host 0.0.0.0 --port 8000

# 개발 모드 (자동 리로드)
python -m uvicorn scripts.api.server:app --host 0.0.0.0 --port 8000 --reload
```

브라우저에서 `http://127.0.0.1:8000` 접속

## 🎯 주요 기능

### 1. 이미지 캡션 생성
- **모델**: BLIP (Salesforce/blip-image-captioning-large)
- **최적화**: CUDA 병렬 처리, FP16 메모리 최적화
- **품질 개선**: 
  - 샘플링 활성화 (`do_sample=True`)
  - 반복 방지 (`repetition_penalty=1.3`)
  - 길이 제어 (`max_length=120`, `min_length=20`)
  - 온도 조절 (`temperature=0.8`)

### 2. 벡터 검색
- **임베딩 모델**: sentence-transformers (all-MiniLM-L6-v2, 384차원)
- **검색 방식**: 코사인 유사도 기반 벡터 검색
- **하이브리드 검색**: 텍스트 검색 + 메타데이터 필터링 지원

### 3. AI 에이전트
- **프레임워크**: LangGraph + LangChain
- **기능**: RAG 기반 질의응답, 위성 이미지 검색 및 분석
- **LLM**: OpenAI GPT (설정 가능)

### 4. 웹 UI
- **Hero 섹션**: 배경 이미지와 진입 버튼
- **2열 레이아웃**: 이미지 그리드 (왼쪽) + 채팅 창 (오른쪽)
- **모달 이미지 뷰어**: 이미지 클릭 시 전체 화면 표시
- **반응형 디자인**: 다양한 화면 크기 지원

## 🔧 기술 스택

### 데이터베이스
- **PostgreSQL 15+**: 관계형 데이터베이스
- **pgvector**: 벡터 검색 확장
- **psycopg2**: PostgreSQL Python 드라이버

### AI/ML
- **PyTorch**: 딥러닝 프레임워크
- **Transformers**: Hugging Face 모델 라이브러리
  - BLIP (이미지 캡션 생성)
  - InstructBLIP, BLIP-2 (선택적)
- **sentence-transformers**: 텍스트 임베딩 (all-MiniLM-L6-v2)

### 에이전트 및 API
- **LangGraph**: 에이전트 워크플로우 관리
- **LangChain**: LLM 통합 및 도구 체인
- **FastAPI**: REST API 서버
- **OpenAI API**: GPT 모델 (에이전트 응답 생성)

### 프론트엔드
- **HTML/CSS/JavaScript**: 웹 UI
- **반응형 디자인**: 모바일/데스크톱 지원

## 📊 데이터셋

### SpaceNet SN8 Floods Dataset
- **데이터셋**: SpaceNet 8 - Flood Detection Challenge
- **지역**: 
  - Louisiana-East (Training)
  - Louisiana-West (Test)
  - Germany (Training)
- **이벤트 타입**: PRE-event, POST-event
- **이미지 형식**: GeoTIFF 타일
- **용도**: 홍수 탐지, 위성 이미지 분석

## 🔐 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하거나 환경 변수를 설정하세요:

```bash
# 데이터베이스
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DB=satellite_db

# OpenAI API (에이전트용)
OPENAI_API_KEY=your-openai-api-key

# Langfuse (선택, 모니터링용)
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
```

또는 `scripts/utils/config.py`에서 직접 설정할 수 있습니다.

## 💻 하드웨어 요구사항

### 최소 사양
- **GPU**: NVIDIA GPU (최소 6GB VRAM, 권장: 8GB+)
  - 테스트 환경: RTX 3060 Ti 8GB
- **RAM**: 최소 16GB, 권장 32GB
- **저장공간**: 최소 100GB (모델 파일 + 데이터셋)

### 권장 사양
- **GPU**: NVIDIA GPU (16GB+ VRAM)
- **RAM**: 32GB+
- **저장공간**: 500GB+

## 📝 API 사용 예시

### 텍스트 검색

```bash
curl -X POST http://127.0.0.1:8000/search/text \
  -H "Content-Type: application/json" \
  -d '{
    "query": "flooded area with buildings",
    "top_k": 24,
    "threshold": 0.0
  }'
```

### 하이브리드 검색

```bash
curl -X POST http://127.0.0.1:8000/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "roads in urban area",
    "metadata_filters": {
      "event_type": "PRE-event"
    },
    "top_k": 24
  }'
```

### RAG 챗봇

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find images of flooded areas near buildings",
    "top_k": 24
  }'
```
