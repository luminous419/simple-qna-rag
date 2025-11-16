# Simple Q&A RAG System

Python 기반 고급 RAG(Retrieval-Augmented Generation) 문서 질의응답 시스템

## 개요

이 시스템은 PDF 및 텍스트 문서를 벡터 데이터베이스에 저장하고, 하이브리드 검색(Sparse + Dense) 및 Re-ranking을 통해 사용자의 질문에 대해 정확한 답변을 생성합니다.

## 주요 특징

### 🔍 3-Stage Retrieval 파이프라인

#### Stage 1: Hybrid Search (Sparse + Dense)
- **BM25 (Sparse Retrieval)**: 키워드 기반 검색, 50개 후보 추출
- **FAISS (Dense Retrieval)**: 의미 기반 검색, 50개 후보 추출
- **RRF (Reciprocal Rank Fusion)**: 두 결과를 융합하여 상위 20개 선택

#### Stage 2: Re-ranking
- **Cross-Encoder (BAAI/bge-reranker-v2-m3)**: 20개 문서를 정밀 재정렬
- 최종 상위 5개 문서만 LLM에 전달

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

**실행 예시:**
```
============================================================
📚 문서 등록 프로그램 시작
============================================================
📂 문서 로딩 중: ./data
✅ PDF 파일 3개 로드 완료
✅ 텍스트 파일 1개 로드 완료
📊 총 4개 문서 로드 완료
✂️  문서 분할 중 (청크 크기: 1000, 오버랩: 200)
✅ 156개 청크로 분할 완료
🔧 임베딩 모델 초기화 중: BAAI/bge-m3
   - L2 정규화: True
💾 벡터스토어 생성 중 (FAISS IndexFlatIP): ./vectorstore
📊 총 156개 청크 임베딩 시작...

진행: .................... 100% (156/156 청크) - 62.8초 경과

🔧 FAISS 인덱스 생성 중...
💾 벡터스토어 저장 중...
✅ 벡터스토어 생성 완료: 156개 청크 저장됨
   - 저장 위치: ./vectorstore
   - 인덱스 타입: FAISS IndexFlatIP (정규화된 벡터 + Inner Product)

============================================================
✅ 문서 등록 완료!
============================================================
📊 통계:
  - 총 문서 수: 4
  - 총 청크 수: 156
  - 저장 위치: ./vectorstore
  - 임베딩 모델: BAAI/bge-m3

💡 이제 document_query.py를 실행하여 질의할 수 있습니다.
```

### 3. 문서 질의

```bash
python document_query.py
```

이 프로그램은:
- 저장된 FAISS 벡터스토어 로드
- BM25 인덱스 생성 (하이브리드 검색용)
- Ollama LLM 초기화
- 대화형 인터페이스 제공

**실행 예시:**
```
============================================================
📚 문서 질의 프로그램 시작
============================================================
📂 벡터스토어 로딩 중: ./vectorstore
✅ 벡터스토어 로드 완료 (FAISS IndexFlatIP)

🔧 BM25 검색기 생성 중...
✅ BM25 인덱스 생성 완료 (문서 156개)

🔧 LLM 초기화 중: gpt-oss:20b
✅ LLM 초기화 완료

🔍 Stage 1 - Hybrid Search 설정
   - Dense (FAISS): 50개
   - Sparse (BM25): 50개
   - RRF 융합 후: 20개
🔍 Stage 2 - Re-ranker 활성화
   - 모델: BAAI/bge-reranker-v2-m3
   - 최종 문서 수: 5
   - 파이프라인: BM25+FAISS(50+50개) → RRF(20개) → Re-rank(5개)
✅ QA 체인 설정 완료

============================================================
💬 문서 Q&A 시스템에 오신 것을 환영합니다!
============================================================
ℹ️  종료하려면 '종료', '끝', 'stop', 'quit', 'finish' 중 하나를 입력하세요.
============================================================

🤔 질문을 입력하세요: 이 문서의 주요 내용은 무엇인가요?

🔍 검색 및 답변 생성 중...
🔀 RRF 융합 완료:
   - BM25: 50개
   - Dense: 50개
   - 고유 문서: 87개
   - 최종 선택: 20개
🔄 Re-ranking 완료: 20개 → 5개 (상위 5개)
   [1] Score: 0.9876 | data/document1.pdf
   [2] Score: 0.9654 | data/document2.pdf
   [3] Score: 0.9432 | data/document1.pdf
   [4] Score: 0.9211 | data/notes.txt
   [5] Score: 0.8987 | data/document2.pdf

============================================================
📝 답변:
============================================================
제공된 문서에 따르면, 주요 내용은 다음과 같습니다:
1. RAG 시스템의 구조와 작동 원리
2. 하이브리드 검색을 통한 정확도 향상 방법
3. Cross-Encoder Re-ranking의 효과

------------------------------------------------------------
📚 참고 문서:
------------------------------------------------------------
  [1] data/document1.pdf (페이지 3)
  [2] data/document2.pdf (페이지 5)
  [3] data/document1.pdf (페이지 1)
  [4] data/notes.txt
  [5] data/document2.pdf (페이지 2)

============================================================

🤔 질문을 입력하세요: 종료

👋 문서 Q&A 시스템을 종료합니다. 감사합니다!
```

### 종료 명령어

다음 단어 중 하나를 입력하면 프로그램이 종료됩니다:
- `종료`
- `끝`
- `stop`
- `quit`
- `exit`
- `finish`

또는 `Ctrl+C`를 눌러 종료할 수 있습니다.

## 프로젝트 구조

```
simple-qna-rag/
├── config.py                # 설정 파일 (모델, 경로, 프롬프트 등)
├── document_register.py     # 문서 등록 프로그램 (임베딩 + 벡터스토어 생성)
├── document_query.py        # 문서 질의 프로그램 (하이브리드 검색 + Re-ranking)
├── requirements.txt         # Python 패키지 의존성
├── README.md               # 이 파일
├── .gitignore              # Git 무시 파일 설정
├── data/                   # 문서 저장 디렉토리 (Git에서 제외)
│   ├── *.pdf
│   └── *.txt
└── vectorstore/            # 벡터 데이터베이스 (Git에서 제외)
    ├── index.faiss
    └── index.pkl
```

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
RERANKER_TOP_K = 5       # 최종 문서 수
```

### MMR (다양성 확보)
```python
USE_MMR = True           # MMR 활성화 (하이브리드 검색 비활성화 시)
MMR_FETCH_K = 100        # 초기 후보 수
MMR_K = 20               # MMR 선택 수
MMR_LAMBDA = 0.5         # 다양성 vs 관련성 밸런스
```

### 프롬프트 템플릿
```python
PROMPT_TEMPLATE = """당신은 주어진 문서 내용을 바탕으로 정확하게 답변하는 AI 어시스턴트입니다.

지침:
1. 주어진 문맥(Context)의 정보만을 사용하여 답변하세요.
2. 문맥에 답이 없으면 "제공된 문서에서 관련 정보를 찾을 수 없습니다"라고 답변하세요.
...
"""
```

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
# document_register.py, document_query.py에서
model_kwargs={'device': 'cuda'}  # 'cpu'를 'cuda'로 변경
```

## 성능 최적화 팁

1. **하이브리드 검색**: BM25 + FAISS로 키워드와 의미 검색 병행
2. **Re-ranking**: Cross-Encoder로 검색 정확도 향상
3. **RRF 파라미터 조정**: `RRF_CONSTANT`를 조정하여 융합 방식 변경
4. **GPU 사용**: CUDA 지원 GPU가 있다면 임베딩에 GPU 사용
5. **청크 크기 조정**: 문서 특성에 맞게 `CHUNK_SIZE` 조정
6. **배치 크기**: `document_register.py`의 `batch_size`를 조정하여 속도 향상

## 기술 스택

- **LangChain**: RAG 파이프라인 구축
- **FAISS**: Dense 벡터 검색 (IndexFlatIP)
- **BM25**: Sparse 키워드 검색
- **Sentence Transformers**: 임베딩 및 Re-ranking
- **Ollama**: 로컬 LLM 실행
- **HuggingFace**: 임베딩 모델

## 라이선스

MIT License

## 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.
