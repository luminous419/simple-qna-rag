# Simple Q&A RAG System

Python 기반 고급 RAG(Retrieval-Augmented Generation) 문서 질의응답 시스템

## 개요

이 시스템은 PDF 및 텍스트 문서를 벡터 데이터베이스에 저장하고, 하이브리드 검색(Sparse + Dense) 및 Re-ranking을 통해 사용자의 질문에 대해 정확한 답변을 생성합니다.

**핵심 기능:**
- 3-Stage Retrieval (Hybrid Search + Re-ranking)
- Intent Classification (질문 의도 분류)
- Intent-based Prompt Template (의도별 맞춤 프롬프트)
- LLM 기반 Agent Router (웹검색 vs 문서 QA 자동 라우팅)
- Web UI 및 CLI 인터페이스

## 주요 특징

### 🔍 3-Stage Retrieval 파이프라인

#### Stage 1: Hybrid Search (Sparse + Dense)
- **BM25 (Sparse Retrieval)**: 키워드 기반 검색, 50개 후보 추출
- **FAISS (Dense Retrieval)**: 의미 기반 검색, 50개 후보 추출
- **RRF (Reciprocal Rank Fusion)**: 두 결과를 융합하여 상위 50개 선택

#### Stage 2: MMR (Maximal Marginal Relevance)
- 유사한 문서 중복 제거로 **다양성 확보**
- lambda=0.5 (관련성 vs 다양성 밸런스)
- 50개 → 20개 (중복 제거 후 다양한 문서 선택)

#### Stage 3: Re-ranking
- **Cross-Encoder (BAAI/bge-reranker-v2-m3)**: 문서를 정밀 재정렬
- 20개 → 최종 10개 문서만 LLM에 전달

### 🎯 Intent Classification (질문 의도 분류)

사용자 질문의 의도를 자동으로 분류하여 최적의 답변 형식을 제공합니다.

#### 지원 의도 유형 (6가지)
| 의도 | 설명 | 예시 |
|------|------|------|
| `explanation` | 개념 설명 | "RAG에서 MMR이 뭐야?" |
| `comparison` | 비교 | "FAISS와 Elasticsearch를 비교해줘" |
| `procedure` | 절차 설명 | "Python에서 FAISS 설치하는 방법을 알려줘" |
| `yesno` | 예/아니오 질문 | "LangChain은 무료인가요?" |
| `other` | 기타 | "코드를 JSON으로 보여줘" |
| `uncertain` | 불명확 | "그게 뭐였지?" |

#### Intent Classifier 모델
- **임베딩**: BAAI/bge-m3 (1024차원)
- **분류기**: Linear Classification Head (Softmax)
- **학습 데이터**: 1,200개 한국어 예시 (라벨당 약 200개)

### 📝 Intent-based Prompt Template

분류된 의도에 따라 자동으로 최적화된 프롬프트 템플릿이 선택됩니다:

| 의도 | 템플릿 특징 |
|------|-------------|
| `explanation` | 핵심 요약 → 개념별 단락 → 결론 형식 |
| `comparison` | Markdown 비교표 생성 |
| `procedure` | 단계별 번호 매기기, 필수 요소/주의사항 포함 |
| `yesno` | "예/아니오" 명확한 답변 + 간략한 설명 |
| `other/uncertain` | 기본 템플릿 (유연한 답변 형식) |

### 🌐 Agent 기반 Query Router (웹검색 vs 문서 QA)

사용자 질문의 의미를 LLM이 직접 판단하여 웹검색과 문서 QA 중 적절한 경로로 자동 라우팅합니다 (`agent.py`).

- **LLM 기반 도구 선택**: `ChatOllama.bind_tools()`로 `web_search`/`document_qa` 두 도구를 바인딩하고, LLM이 질문의 의미를 보고 도구를 선택. 키워드가 전혀 없는 질문("FAISS와 Elasticsearch를 비교해줘")도 올바르게 문서 QA로, 명시적 웹검색 요청은 정제된 검색어와 함께 웹검색으로 라우팅됨
- **단발성 라우팅 방식**: 표준 LangChain `AgentExecutor`(ReAct 루프)를 쓰지 않고, LLM에게는 "어느 도구를 쓸지 + (웹검색 시) 정제된 검색어"만 맡김. 두 도구가 이미 완결된 최종 답변(sources 포함)을 반환하므로, Agent가 도구 결과를 다시 요약하면 포맷이 깨지고 LLM 호출이 중복되기 때문
- **웹검색 (`web_search.py`)**: DuckDuckGo(`ddgs`)를 통해 검색을 수행하고 결과(URL/제목/요약)를 RAG 응답과 동일한 형식(`answer`, `sources`, `success`)으로 포맷팅
- **폴백**: Agent 호출이 실패하거나 도구를 선택하지 못하면 키워드 기반 라우터(`query_router.py`)로, 웹검색이 실패하면 문서 QA로 자동 재시도
- `config.py`의 `USE_WEB_SEARCH`로 기능 전체를 켜고 끌 수 있음
- `tools.py`는 `agent.py`가 사용하는 도구 정의(이름/설명/실행 함수)를 제공

### 🤖 모델 선정

#### 임베딩 모델 (BAAI/bge-m3)
- **8192 토큰 지원**: 매우 긴 문맥 처리 가능
- **멀티언어 지원**: 한국어, 영어 등 100+ 언어 지원
- **높은 성능**: MTEB 벤치마크에서 우수한 성능
- **도메인 일반화**: 법률, 금융, 기술 등 다양한 문체 대응

#### LLM 모델 (gpt-oss:20b via Ollama)
- **20B 파라미터**: 높은 추론 능력
- **긴 컨텍스트**: 충분한 문서 처리 능력
- **한국어 지원**: 자연스러운 한국어 답변
- **로컬 실행**: Ollama를 통한 프라이버시 보호

## 시스템 요구사항

- Python 3.11+
- Ollama (로컬 LLM 실행용)
- 8GB+ RAM 권장

## 설치

### 1. Python 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. Ollama 설치 및 모델 다운로드

#### Ollama 설치
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# https://ollama.com/download 에서 다운로드
```

#### LLM 모델 다운로드
```bash
ollama pull gpt-oss:20b
```

#### Ollama 실행
```bash
ollama serve
```

> **참고**: Ollama는 백그라운드에서 계속 실행되어야 합니다.

### 3. Intent Classifier 학습 (선택사항)

Intent Classifier를 사용하려면 학습을 수행하세요:

```bash
python train_intent_classifier.py
```

> **참고**: 학습하지 않아도 기본 템플릿으로 동작합니다.

## 사용 방법

### 1. 문서 준비

`data` 디렉토리에 PDF 또는 텍스트 파일을 넣습니다:

```bash
data/
├── document1.pdf
├── document2.pdf
└── notes.txt
```

### 2. 문서 등록 (벡터스토어 생성)

```bash
python document_register.py
```

이 프로그램은:
- `data` 디렉토리의 모든 PDF와 텍스트 파일을 로드
- 문서를 적절한 크기의 청크로 분할
- 각 청크를 임베딩하여 FAISS 벡터스토어에 저장
- **실시간 진행률 표시**: 1초마다 진행 상황과 경과 시간 출력

### 3. 문서 질의

#### 방법 1: 웹 인터페이스 (권장)

```bash
python web_server.py
```

브라우저에서 http://localhost:8000 접속

**웹 UI 특징:**
- 깔끔한 채팅 인터페이스
- 실시간 답변 표시
- 참고 문서 출처 확인

#### 방법 2: CLI 인터페이스

```bash
python document_query_cli.py
```

**CLI 특징:**
- 터미널에서 대화형 질의
- 상세한 검색 과정 로그 출력

### 4. 종료 명령어 (CLI)

다음 단어 중 하나를 입력하면 프로그램이 종료됩니다:
- `종료`, `끝`, `stop`, `quit`, `exit`, `finish`

또는 `Ctrl+C`를 눌러 종료할 수 있습니다.

## 프로젝트 구조

```
simple-qna-rag/
├── config.py                  # 설정 파일 (모델, 경로, 검색 파라미터)
├── rag_engine.py              # RAG 코어 엔진 (싱글톤)
├── document_register.py       # 문서 등록 (임베딩 + 벡터스토어 생성)
├── document_query_cli.py      # 문서 질의 CLI (rag_engine 사용)
├── web_server.py              # FastAPI 웹 서버
├── agent.py                   # LLM 기반 Agent 라우터 (웹검색/문서 QA 자동 선택)
├── query_router.py            # 키워드 기반 라우터 (Agent 실패 시 폴백)
├── web_search.py              # DuckDuckGo 웹검색 모듈
├── tools.py                   # LangChain Tool 정의 (agent.py가 사용)
├── prompt_templates.py        # Intent별 프롬프트 템플릿
├── intent_classifier.py       # Intent 분류 추론 모듈
├── train_intent_classifier.py # Intent Classifier 학습
├── generate_intent_dataset.py # Intent 학습 데이터 생성 스크립트
├── test_web_search_simple.py       # 웹검색 모듈 단위 테스트
├── test_web_search_integration.py  # Query Router 통합 테스트
├── requirements.txt           # Python 패키지 의존성
├── README.md                  # 이 파일
├── .gitignore                 # Git 무시 파일 설정
├── data/                      # 문서 저장 디렉토리 (Git에서 제외)
│   ├── *.pdf
│   └── *.txt
├── vectorstore/               # FAISS 벡터 데이터베이스 (Git에서 제외)
│   ├── index.faiss
│   └── index.pkl
├── intent_dataset/            # Intent 분류 학습 데이터
│   ├── train.jsonl            # 학습 데이터 (1,200개)
│   └── dev.jsonl              # 검증 데이터 (42개)
├── intent-bge-m3-softmax/     # 학습된 Intent Classifier
│   ├── classifier_head.pt     # 분류기 헤드 가중치 (~26KB)
│   └── config.json            # 모델 설정
└── templates/                 # 웹 UI 템플릿
    └── index.html
```

## API 엔드포인트

### 웹 서버 API

| 엔드포인트 | 메소드 | 설명 |
|-----------|--------|------|
| `/` | GET | 메인 웹 UI |
| `/rag` | POST | RAG 질의 API |
| `/health` | GET | 헬스 체크 |

#### POST /rag

**Request:**
```json
{
  "question": "RAG 시스템이 뭔가요?"
}
```

**Response:**
```json
{
  "answer": "RAG(Retrieval-Augmented Generation)은...",
  "sources": [
    {
      "index": 1,
      "source": "document1.pdf",
      "page": 3,
      "content": "..."
    }
  ],
  "success": true
}
```

> **참고**: 내부적으로 `/rag`는 `agent.route_query()`를 호출하여 LLM이 질문의 의미를 보고 문서 QA 또는 웹검색(DuckDuckGo) 중 하나를 선택하도록 라우팅합니다. 웹검색으로 라우팅된 경우 `sources`의 `source` 필드에는 문서 파일명 대신 검색 결과 URL이 담깁니다.

## 설정 커스터마이징

`config.py` 파일에서 다음 설정을 변경할 수 있습니다:

### 임베딩 모델
```python
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"  # 또는 다른 HuggingFace 모델
NORMALIZE_EMBEDDINGS = True  # L2 정규화 (Cosine 유사도 최적화)
```

### LLM 모델
```python
OLLAMA_MODEL = "gpt-oss:20b"  # 또는 qwen2.5:7b, llama3.1:8b 등
```

### 문서 처리
```python
CHUNK_SIZE = 1000      # 문서 청크 크기
CHUNK_OVERLAP = 200    # 청크 간 오버랩
```

### 하이브리드 검색
```python
USE_HYBRID_SEARCH = True  # 하이브리드 검색 활성화
BM25_TOP_K = 50          # BM25 검색 결과 수
DENSE_TOP_K = 50         # FAISS 검색 결과 수
RRF_TOP_K = 20           # RRF 융합 후 선택 수
RRF_CONSTANT = 60        # RRF 상수
```

### Re-ranking
```python
USE_RERANKER = True      # Re-ranker 활성화
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_TOP_K = 10      # 최종 문서 수
```

### MMR (다양성 확보)
```python
USE_MMR = True           # MMR 활성화 (하이브리드 검색 비활성화 시)
MMR_FETCH_K = 100        # 초기 후보 수
MMR_K = 20               # MMR 선택 수
MMR_LAMBDA = 0.5         # 다양성 vs 관련성 밸런스
```

### 웹검색 (Query Router)
```python
USE_WEB_SEARCH = True        # 웹검색 기능 활성화 여부 (false 시 항상 문서 QA만 사용)
WEB_SEARCH_MAX_RESULTS = 3   # 최대 검색 결과 수
WEB_SEARCH_TIMEOUT = 10      # 검색 타임아웃 (초)
WEB_SEARCH_REGION = "kr-kr"  # 검색 지역 (kr-kr: 한국)
```

## Intent Classifier 학습

### 학습 데이터 형식

`intent_dataset/train.jsonl`:
```json
{"text": "RAG에서 MMR이 뭐야?", "label": "explanation"}
{"text": "FAISS와 Elasticsearch를 비교해줘", "label": "comparison"}
{"text": "Python에서 FAISS 설치하는 방법을 알려줘", "label": "procedure"}
```

### 학습 실행

```bash
python train_intent_classifier.py
```

**출력:**
```
============================================================
Intent Classifier 학습 시작
============================================================
모델: BAAI/bge-m3
디바이스: cpu (또는 cuda)
배치 크기: 32
에폭: 3
...
✅ 학습 완료!
Best Dev F1: 0.95+
모델 저장 위치: intent-bge-m3-softmax
============================================================
```

### 모델 저장 구조

학습 완료 후 `intent-bge-m3-softmax/` 디렉토리에 저장:
- `classifier_head.pt`: 분류기 가중치 (~26KB)
- `config.json`: 라벨 매핑, 임베딩 모델 이름 등

> **참고**: 임베딩 모델(BAAI/bge-m3)은 HuggingFace Hub에서 로드하므로 로컬에 저장하지 않습니다.

## 대안 모델

### 임베딩 모델
- `intfloat/multilingual-e5-large`: 512 토큰, 멀티언어
- `jhgan/ko-sroberta-multitask`: 한국어 특화

### LLM 모델
- `qwen2.5:7b`: 128K 컨텍스트, 뛰어난 한국어 지원
- `llama3.1:8b`: 128K 컨텍스트, 좋은 한국어 지원
- `gemma2:9b`: 효율적, 빠른 응답
- `mistral:7b`: 균형잡힌 성능

다른 모델 설치:
```bash
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
ollama pull gemma2:9b
```

### Re-ranker 모델
- `BAAI/bge-reranker-v2-m3`: 멀티언어, 8192 토큰 (기본값)
- `cross-encoder/ms-marco-MiniLM-L-6-v2`: 영어 특화, 빠른 속도

## 문제 해결

### 1. Ollama 연결 실패
```
❌ LLM 초기화 실패: Connection refused
```

**해결 방법:**
```bash
# Ollama 서비스 시작
ollama serve
```

### 2. 모델이 없음
```
Error: model 'gpt-oss:20b' not found
```

**해결 방법:**
```bash
ollama pull gpt-oss:20b
```

### 3. 메모리 부족
큰 모델 사용 시 메모리가 부족한 경우, 더 작은 모델 사용:
```bash
ollama pull qwen2.5:3b  # 더 작은 버전
```

`config.py`에서 변경:
```python
OLLAMA_MODEL = "qwen2.5:3b"
```

### 4. 임베딩 모델 다운로드 느림
첫 실행 시 HuggingFace에서 임베딩 모델을 다운로드합니다. 시간이 걸릴 수 있으니 기다려주세요.

### 5. GPU 사용
GPU를 사용하려면 `config.py` 또는 코드에서 `device`를 변경:
```python
# document_register.py, rag_engine.py에서
model_kwargs={'device': 'cuda'}  # 'cpu'를 'cuda'로 변경
```

### 6. Intent Classifier 모델 없음
```
⚠️ Intent Classifier 모델을 찾을 수 없습니다. 기본 템플릿을 사용합니다.
```

**해결 방법:**
```bash
python train_intent_classifier.py
```

## 성능 최적화 팁

1. **하이브리드 검색**: BM25 + FAISS로 키워드와 의미 검색 병행
2. **Re-ranking**: Cross-Encoder로 검색 정확도 향상
3. **Intent Classification**: 질문 유형에 맞는 프롬프트로 답변 품질 향상
4. **RRF 파라미터 조정**: `RRF_CONSTANT`를 조정하여 융합 방식 변경
5. **GPU 사용**: CUDA 지원 GPU가 있다면 임베딩에 GPU 사용
6. **청크 크기 조정**: 문서 특성에 맞게 `CHUNK_SIZE` 조정
7. **배치 크기**: `document_register.py`의 `batch_size`를 조정하여 속도 향상

## 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Interface                              │
│                 ┌──────────────┐  ┌──────────────┐                  │
│                 │   Web UI     │  │    CLI       │                  │
│                 │ (FastAPI)    │  │              │                  │
│                 └──────┬───────┘  └──────┬───────┘                  │
└────────────────────────┼─────────────────┼──────────────────────────┘
                         │                 │
                         ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Agent Router (LLM 기반, gpt-oss:20b tool calling)       │
│         질문 의미 판단 → document_qa / web_search 도구 선택           │
│         (Agent 실패 시 키워드 기반 query_router.py로 폴백)             │
└────────────────────────┬─────────────────────┬──────────────────────┘
                         │                      │
                document_qa 선택           web_search 선택
                         │                      │
                         ▼                      ▼
┌─────────────────────────────────────┐   ┌──────────────────────────┐
│              RAG Engine             │   │  Web Search (DuckDuckGo) │
│  ┌───────────┐  ┌───────────┐  ┌───┐│   │   ddgs                   │
│  │  Intent   │  │ Retrieval │  │LLM││   │   (URL/제목/요약 반환)     │
│  │Classifier │  │ Pipeline  │  │   ││   └──────────────────────────┘
│  │ (BGE-M3)  │  │           │  │   ││
│  └─────┬─────┘  └─────┬─────┘  └─┬─┘│
│        │              │          │  │
│        ▼              ▼          ▼  │
│  ┌───────────┐ ┌───────────────┐┌──┐│
│  │  Template │ │ BM25 + FAISS  ││Re││
│  │  Selector │ │  + RRF Fusion ││sp││
│  └───────────┘ └───────┬───────┘└──┘│
└────────────────────────┼────────────┘
                         ▼
                    ┌───────────────────────┐
                    │   FAISS VectorStore   │
                    │   + BM25 Index        │
                    └───────────────────────┘
```

## 기술 스택

- **LangChain**: RAG 파이프라인 구축
- **FAISS**: Dense 벡터 검색 (IndexFlatIP)
- **BM25**: Sparse 키워드 검색
- **Sentence Transformers**: 임베딩 및 Re-ranking
- **Ollama**: 로컬 LLM 실행
- **HuggingFace**: 임베딩 모델
- **FastAPI**: 웹 서버
- **PyTorch**: Intent Classifier 학습
- **ddgs**: 웹검색 (Agent Router)

## 라이선스

MIT License

## 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.
