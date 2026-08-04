# Simple Q&A RAG

로컬 문서와 웹 검색 결과를 기반으로 질문에 답하는 한국어 중심의 RAG(Retrieval-Augmented Generation) 애플리케이션입니다. 문서 데이터는 로컬에서 임베딩하고, Ollama로 실행되는 LLM이 답변을 생성합니다.

프로젝트의 방향과 현재 위치는 [Roadmap](docs/Roadmap.md), 알려진 미해결 문제는 [Problem](docs/Problem.md)에서 관리합니다.

## 목적

- PDF와 텍스트 문서를 검색 가능한 지식베이스로 변환합니다.
- 키워드 검색과 의미 검색을 결합해 관련 문서를 찾습니다.
- 질문 유형에 맞는 형식으로 문서 근거 답변을 생성합니다.
- 최신 정보가 필요한 질문은 웹 검색으로 자동 라우팅합니다.
- 모델과 문서를 가능한 한 로컬에서 처리해 데이터 통제권을 유지합니다.

## 주요 기능

### 문서 검색과 답변

- PDF/TXT 문서 로딩 및 청크 분할
- BAAI/bge-m3 임베딩과 FAISS 벡터 인덱스
- BM25 + Dense Search + RRF 기반 Hybrid Retrieval
- MMR 기반 검색 결과 다양화
- BAAI/bge-reranker-v2-m3 Cross-Encoder 재정렬
- 검색 문서와 페이지를 포함한 출처 반환

### 질문 처리

- explanation, comparison, procedure, yesno, other, uncertain 의도 분류
- 질문 의도별 답변 프롬프트
- Ollama `gpt-oss:20b` 기반 답변 생성
- LLM tool calling을 이용한 문서 QA/웹 검색 라우팅
- Agent 장애 시 키워드 라우터, 웹 검색 장애 시 문서 QA 폴백

### 인터페이스와 안전성

- FastAPI 기반 Web UI 및 JSON API
- 터미널 기반 문서 QA CLI
- 웹 검색 결과의 Markdown 정화와 XSS 회귀 테스트
- 프런트엔드 라이브러리의 로컬 vendor 및 잠금 버전 관리

## 아키텍처

```text
사용자 질문
    |
    v
Agent Router
    |-- web_search  ---> DuckDuckGo 검색 결과
    |
    `-- document_qa
            |
            v
       Intent 분류
            |
            v
BM25 + FAISS -> RRF -> MMR -> Re-ranker
            |
            v
      Ollama 답변 생성
```

Agent가 실패하면 키워드 라우터를 사용합니다. 웹 검색까지 실패하면 원본 질문으로 문서 QA를 재시도합니다.

## 개발 환경 구성

### 요구사항

- Python 3.11 권장
- Node.js 22.22.2 이상 권장: 프런트엔드 테스트와 vendor 동기화에만 필요
- Ollama
- `gpt-oss:20b`를 실행할 수 있는 메모리와 디스크 공간

설치된 `simple-qna-rag-*` 제품 명령은 다른 current directory에서도 실행할 수 있습니다. `evaluation` 명령과 저장소 관리 명령은 프로젝트 루트에서 실행하십시오. runtime 환경변수와 CLI 경로 override는 M2.5 Phase 4에서 추가합니다.

### 1. Python 환경

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Windows PowerShell에서는 다음 명령으로 가상환경을 활성화합니다.

```powershell
.venv\Scripts\Activate.ps1
```

첫 실행 시 Hugging Face 모델을 다운로드하므로 시간이 걸릴 수 있습니다.

### 2. 프런트엔드 테스트 환경

운영 페이지에서 사용하는 `marked`와 DOMPurify 파일은 저장소의 `web/static/vendor/`에 포함됩니다. 라이브러리를 갱신하거나 프런트엔드 테스트를 실행하려면 다음 명령을 사용합니다.

```bash
npm ci
```

`npm ci`의 `postinstall` 단계가 잠금된 npm 패키지의 배포 파일을 `web/static/vendor/`에 동기화합니다. 필요하면 직접 실행할 수 있습니다.

```bash
npm run sync-vendor
```

### 3. Ollama

Ollama를 설치한 다음 모델을 준비합니다.

```bash
ollama pull gpt-oss:20b
ollama serve
```

기본 연결 주소와 모델은 [config.py](src/simple_qna_rag/config.py)의 `OLLAMA_BASE_URL`, `OLLAMA_MODEL`에서 설정합니다.

### 4. 문서 인덱스 생성

`runtime/documents/` 디렉터리에 PDF 또는 TXT 문서를 넣고 인덱스를 생성합니다.

```bash
simple-qna-rag-index
```

생성된 FAISS 인덱스는 기본적으로 `runtime/vectorstore/`에 저장됩니다. `runtime/` 전체는 Git에서 제외됩니다.

주의: 현재 등록 명령은 기존 `runtime/vectorstore/`를 삭제하고 전체 인덱스를 다시 생성합니다. 중요한 인덱스는 실행 전에 별도로 백업하십시오.

필요하면 CLI 또는 환경변수로 runtime 경로를 재정의할 수 있습니다.

```bash
simple-qna-rag-index --documents-dir /path/to/documents --vectorstore-dir /path/to/vectorstore

export SIMPLE_QNA_RAG_DOCUMENTS_DIR=/path/to/documents
export SIMPLE_QNA_RAG_VECTORSTORE_DIR=/path/to/vectorstore
export SIMPLE_QNA_RAG_MODEL_DIR=/path/to/intent-model
```

경로 우선순위는 `CLI > environment > repository default`입니다.

## 실행 방법

### Web UI와 API

```bash
simple-qna-rag-web
```

브라우저에서 <http://localhost:8000>에 접속합니다.

주요 API:

- `GET /`: Web UI
- `POST /rag`: 질문 처리
- `GET /health`: 애플리케이션 상태 확인

요청 예시:

```bash
curl -X POST http://localhost:8000/rag \
  -H 'Content-Type: application/json' \
  -d '{"question":"RAG에서 MMR이 무엇인가요?"}'
```

응답에는 `answer`, `sources`, `success`, `search_type`, `intent`가 포함됩니다.

### CLI

CLI는 웹 검색 Agent를 거치지 않고 문서 QA 엔진을 직접 사용합니다.

```bash
simple-qna-rag-query
```

`종료`, `끝`, `stop`, `quit`, `exit`, `finish` 중 하나를 입력하면 종료합니다.

## 테스트 방법

### 전체 오프라인 검증

일반 Pull Request와 로컬 회귀 검증은 Ollama, 웹 검색, `runtime/documents/`, `runtime/vectorstore/` 없이 실행됩니다.

```bash
python -m pip check
python -c "from simple_qna_rag.web.server import app"
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
pytest -q
npm ci
npm test
npm run sync-vendor
git diff --exit-code -- web/static/vendor/
git diff --check
```

GitHub Actions의 `python-tests`와 `frontend-tests`도 같은 오프라인 경계를 사용합니다. CI는 Ollama, DDGS, 모델 가중치 다운로드, 로컬 corpus/vectorstore 또는 secret을 요구하지 않습니다.

### 골든 평가셋 검증

평가셋의 schema, 최소 사례 수, category·intent 구성과 정답 규칙을 검사합니다.

```bash
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
```

현재 골든셋 작성 규칙과 metric 정의는 [evaluation/README.md](evaluation/README.md)를 참고하십시오.

### 개별 evaluator 실행

Retrieval과 Answer 평가는 실제 `runtime/documents/`, `runtime/vectorstore/`, embedding/reranker와 Ollama를 사용합니다. Routing의 live 모드도 Ollama가 필요합니다. 상세 결과는 기본적으로 Git에서 제외되는 `evaluation/reports/` 아래 JSON과 Markdown으로 생성됩니다.

```bash
python -m evaluation.retrieval \
  --dataset evaluation/datasets/golden.jsonl \
  --output evaluation/reports/retrieval

RUN_LIVE_LLM_TESTS=1 python -m evaluation.routing \
  --dataset evaluation/datasets/golden.jsonl \
  --mode live \
  --output evaluation/reports/routing

python -m evaluation.answers \
  --dataset evaluation/datasets/golden.jsonl \
  --output evaluation/reports/answers
```

Routing은 모델 호출 없이 파싱·집계·리포팅을 확인할 수 있는 offline 모드도 제공합니다.

```bash
python -m evaluation.routing \
  --dataset evaluation/datasets/golden.jsonl \
  --mode offline \
  --output evaluation/reports/routing-offline
```

### 통합 live baseline

통합 명령은 dataset validation → Retrieval → live Routing → Answer 순서로 실행합니다. 실제 모델과 vectorstore를 사용하므로 명시적인 opt-in이 필요합니다.

```bash
RUN_LIVE_LLM_TESTS=1 python -m evaluation.baseline \
  --dataset evaluation/datasets/golden.jsonl \
  --output evaluation/reports
```

실행 전 다음을 확인하십시오.

- Ollama가 실행 중이고 [config.py](src/simple_qna_rag/config.py)의 모델이 설치돼 있음
- `runtime/documents/`와 `runtime/vectorstore/index.faiss`, `runtime/vectorstore/index.pkl`이 준비돼 있음
- 실행 중 corpus와 vectorstore를 변경하지 않음
- 비교 목적이라면 `git status`가 clean이고 기준선과 dataset/corpus/vectorstore fingerprint가 같음

빠른 환경 점검에는 `--limit 1`을 사용할 수 있지만 제한 실행 결과를 정식 baseline으로 확정하면 안 됩니다. `--tag`, `--skip-routing`, `--skip-answers`도 지원합니다.

사용자가 승인한 M2 최초 기준선은 다음 파일에 고정돼 있습니다.

- [기계 판독용 기준선](evaluation/baselines/m2_initial.json)
- [사람 판독용 기준선](evaluation/baselines/m2_initial.md)

timestamped report에는 질문과 모델 답변이 포함될 수 있으므로 `evaluation/reports/`는 commit하지 않습니다. 고정 기준선에는 비교에 필요한 집계 수치와 fingerprint만 포함합니다.

### 라이브 Agent 라우팅 회귀 테스트

Ollama와 `gpt-oss:20b`가 실행 중일 때만 사용합니다.

```bash
RUN_LIVE_LLM_TESTS=1 pytest tests/integration/test_agent_routing.py -v
```

이 테스트는 LLM 출력의 확률적 변동을 고려해 라우팅 정확도 80% 이상을 통과 기준으로 사용합니다.

## 배포 방법

### 현재 지원 범위

현재 프로젝트는 단일 호스트의 로컬 또는 내부 데모 배포를 지원합니다. Docker 이미지, 프로세스 관리자, 자동 배포, 다중 worker 운영 구성은 아직 제공하지 않습니다.

배포 호스트에서 다음 항목을 먼저 준비해야 합니다.

1. Python 의존성
2. Ollama와 `gpt-oss:20b`
3. `runtime/documents/`에서 생성한 `runtime/vectorstore/`
4. 저장소에 포함된 `web/templates/`와 `web/static/`

서버 실행:

```bash
simple-qna-rag-web --host 0.0.0.0 --port 8000
```

현재 RAG 엔진은 프로세스마다 대형 모델과 인덱스를 메모리에 로드하므로 worker 수를 무작정 늘리지 마십시오. 외부에 공개할 경우 애플리케이션 앞에 TLS를 종료하는 reverse proxy, 인증, 요청 크기 제한, rate limiting을 별도로 구성해야 합니다.

프로덕션 배포 자동화와 운영 준비는 [Roadmap](docs/Roadmap.md)의 Production Readiness 마일스톤에서 다룹니다.

## 주요 설정

[config.py](src/simple_qna_rag/config.py)에서 다음 항목을 조정할 수 있습니다.

- 임베딩, LLM, reranker 모델
- 데이터 및 벡터스토어 경로
- 청크 크기와 오버랩
- BM25/Dense/RRF/MMR/Re-ranker top-k
- 웹 검색 활성화, 결과 수, 타임아웃, 지역

검색·모델 동작 설정은 Python 상수로 관리됩니다. 문서, vectorstore와 intent model 경로는 `SIMPLE_QNA_RAG_*` 환경변수 또는 CLI 옵션으로 재정의할 수 있습니다.

## 프로젝트 구조

```text
.
├── pyproject.toml              # Python package와 CLI entry point
├── src/simple_qna_rag/         # 제품 Python package
│   ├── cli/                    # query/index/web 명령
│   └── web/                    # FastAPI 애플리케이션
├── web/                        # HTML 템플릿과 프런트엔드/vendor 자산
├── tests/                      # unit/integration/frontend 테스트
├── training/                   # Intent 학습 코드와 dataset
├── models/                     # 버전 관리되는 Intent 모델 artifact
├── runtime/                    # Git 제외 문서와 vectorstore
├── evaluation/                 # 골든셋, evaluator, 리포팅과 승인 기준선
├── docs/                       # 로드맵, 문제, 마일스톤과 리뷰 문서
├── .github/workflows/ci.yml    # Python 및 frontend 오프라인 CI
└── README.md                   # 프로젝트 진입 문서
```

M2.5 구조 이전과 GitHub Actions 검증, 사용자 최종 승인을 완료했습니다. 디렉터리별 책임은 [Repository Structure](docs/architecture/Repository_Structure.md)를 참조하십시오.

## 라이선스

MIT License. 자세한 내용은 [LICENSE](LICENSE)를 참조하십시오.
