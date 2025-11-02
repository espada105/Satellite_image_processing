# 위성 이미지 처리 AI 프로젝트

위성 이미지를 활용한 지리공간 기반 멀티모달 AI 시스템 구축 프로젝트입니다.

## 🌟 프로젝트 개요

이 프로젝트는 다음과 같은 5가지 핵심 기술을 통합하여 구현합니다:

1. **도메인 특화 LLM**: 위성 이미지 분석에 특화된 멀티모달 모델 파인튜닝
2. **지리공간 RAG**: PGVector 기반 벡터 데이터베이스로 위성 이미지 검색
3. **멀티모달 처리**: 이미지-텍스트 이해 및 분석
4. **AI 에이전트**: LangGraph 기반 복잡한 워크플로우 자동화
5. **프로덕션 모니터링**: Langfuse를 통한 추적 및 성능 모니터링

## 📋 프로젝트 구조

```
satellite-ai-project/
├── data/                    # 데이터 디렉토리
│   ├── raw/                 # 원본 위성 이미지
│   ├── processed/           # 전처리된 이미지
│   ├── metadata/            # 메타데이터 JSON
│   ├── captions/            # 생성된 캡션
│   ├── embeddings/          # 벡터 임베딩
│   └── qa_dataset/          # QA 데이터셋
├── models/                  # 모델 디렉토리
│   ├── llava-base/          # 베이스 모델
│   ├── llava-satellite-7b-lora/  # 파인튜닝된 모델
│   └── embeddings/          # 임베딩 모델
├── scripts/                 # 실행 스크립트
│   ├── data_collection/     # 데이터 수집
│   ├── database/            # DB 설정 및 검색
│   ├── finetuning/          # 모델 파인튜닝
│   └── utils/               # 유틸리티
├── api/                     # API 서버
├── agents/                  # AI 에이전트
│   └── tools/               # 에이전트 도구
├── docker/                  # Docker 설정
├── k8s/                     # Kubernetes 매니페스트
├── notebooks/               # Jupyter 노트북
├── tests/                   # 테스트 코드
└── docs/                    # 문서

```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
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
# 데이터베이스 생성 및 설정
python scripts/database/setup_database.py
```

## 📚 문서

- **[프로젝트 구축 계획서](../프로젝트_구축_계획서.md)**: 각 단계별 상세 계획 및 구현 방법
- **[단계별 실행 가이드](../단계별_실행_가이드.md)**: 실제 실행 명령어 및 체크리스트
- **[프로젝트 구조 가이드](../프로젝트_구조_가이드.md)**: 디렉토리 구조 및 파일 설명

## 📊 프로젝트 타임라인

| 단계 | 작업 | 예상 기간 |
|------|------|-----------|
| 1단계 | 데이터 수집 및 RAG 구축 | 5-8일 |
| 2단계 | 멀티모달 모델 파인튜닝 | 7-11일 |
| 3단계 | AI 에이전트 및 API 서버 | 8-9일 |
| 4단계 | 프로덕션 배포 및 모니터링 | 9-10일 |
| **총계** | | **29-38일 (약 4-6주)** |

## 🔧 기술 스택

- **데이터베이스**: PostgreSQL + pgvector
- **AI/ML**: PyTorch, Transformers, PEFT/LoRA
- **에이전트**: LangGraph, LangChain
- **API**: FastAPI
- **배포**: Docker, Kubernetes
- **모니터링**: Langfuse

---

**참고**: 자세한 내용은 상위 디렉토리의 문서를 참고하세요.

