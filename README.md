# Simple Q&A RAG

로컬 문서와 웹 검색 결과를 기반으로 질문에 답하는 한국어 중심의 RAG(Retrieval-Augmented Generation) 애플리케이션입니다. 문서 데이터는 로컬에서 임베딩하고, Ollama로 실행되는 LLM이 답변을 생성합니다.

프로젝트의 방향과 현재 위치는 [Roadmap.md](Roadmap.md), 알려진 미해결 문제는 [Problem.md](Problem.md)에서 관리합니다.

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
- Node.js 20 이상: 프런트엔드 테스트와 vendor 동기화에만 필요
- Ollama
- `gpt-oss:20b`를 실행할 수 있는 메모리와 디스크 공간

모든 명령은 프로젝트 루트에서 실행해야 합니다. 현재 데이터, 벡터스토어, 템플릿 경로가 프로젝트 루트 기준 상대 경로로 설정되어 있습니다.

### 1. Python 환경

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell에서는 다음 명령으로 가상환경을 활성화합니다.

```powershell
.venv\Scripts\Activate.ps1
```

첫 실행 시 Hugging Face 모델을 다운로드하므로 시간이 걸릴 수 있습니다.

### 2. 프런트엔드 테스트 환경

운영 페이지에서 사용하는 `marked`와 DOMPurify 파일은 저장소의 `static/vendor/`에 포함됩니다. 라이브러리를 갱신하거나 프런트엔드 테스트를 실행하려면 다음 명령을 사용합니다.

```bash
npm ci
```

`npm ci`의 `postinstall` 단계가 잠금된 npm 패키지의 배포 파일을 `static/vendor/`에 동기화합니다. 필요하면 직접 실행할 수 있습니다.

```bash
npm run sync-vendor
```

### 3. Ollama

Ollama를 설치한 다음 모델을 준비합니다.

```bash
ollama pull gpt-oss:20b
ollama serve
```

기본 연결 주소와 모델은 [config.py](config.py)의 `OLLAMA_BASE_URL`, `OLLAMA_MODEL`에서 설정합니다.

### 4. 문서 인덱스 생성

`data/` 디렉터리에 PDF 또는 TXT 문서를 넣고 인덱스를 생성합니다.

```bash
python document_register.py
```

생성된 FAISS 인덱스는 기본적으로 `vectorstore/`에 저장됩니다. `data/`와 `vectorstore/`는 Git에서 제외됩니다.

주의: 현재 등록 명령은 기존 `vectorstore/`를 삭제하고 전체 인덱스를 다시 생성합니다. 중요한 인덱스는 실행 전에 별도로 백업하십시오.

## 실행 방법

### Web UI와 API

```bash
python web_server.py
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
python document_query_cli.py
```

`종료`, `끝`, `stop`, `quit`, `exit`, `finish` 중 하나를 입력하면 종료합니다.

## 테스트 방법

### 기본 Python 테스트

외부 네트워크와 Ollama 없이 mock 기반 라우팅, 폴백, 웹 검색 포맷을 검증합니다.

```bash
pytest -q
```

### 라이브 Agent 라우팅 테스트

Ollama와 `gpt-oss:20b`가 실행 중일 때만 사용합니다.

```bash
RUN_LIVE_LLM_TESTS=1 pytest test_agent_routing.py -v
```

이 테스트는 LLM 출력의 확률적 변동을 고려해 라우팅 정확도 80% 이상을 통과 기준으로 사용합니다.

### 프런트엔드 보안 테스트

```bash
npm ci
npm test
```

Vitest와 jsdom으로 XSS 정화, 안전한 링크, 출처 렌더링을 검증합니다.

### 전체 로컬 검증

```bash
pytest -q
npm test
git diff --check
```

현재 자동 테스트는 mock과 DOM 단위 테스트 중심입니다. 실제 문서 검색과 답변 품질을 평가하는 End-to-End 기준선은 향후 마일스톤에서 구축할 예정입니다.

## 배포 방법

### 현재 지원 범위

현재 프로젝트는 단일 호스트의 로컬 또는 내부 데모 배포를 지원합니다. Docker 이미지, 프로세스 관리자, 자동 배포, 다중 worker 운영 구성은 아직 제공하지 않습니다.

배포 호스트에서 다음 항목을 먼저 준비해야 합니다.

1. Python 의존성
2. Ollama와 `gpt-oss:20b`
3. `data/`에서 생성한 `vectorstore/`
4. 저장소에 포함된 `templates/`와 `static/`

서버 실행:

```bash
uvicorn web_server:app --host 0.0.0.0 --port 8000
```

현재 RAG 엔진은 프로세스마다 대형 모델과 인덱스를 메모리에 로드하므로 worker 수를 무작정 늘리지 마십시오. 외부에 공개할 경우 애플리케이션 앞에 TLS를 종료하는 reverse proxy, 인증, 요청 크기 제한, rate limiting을 별도로 구성해야 합니다.

프로덕션 배포 자동화와 운영 준비는 [Roadmap.md](Roadmap.md)의 Production Readiness 마일스톤에서 다룹니다.

## 주요 설정

[config.py](config.py)에서 다음 항목을 조정할 수 있습니다.

- 임베딩, LLM, reranker 모델
- 데이터 및 벡터스토어 경로
- 청크 크기와 오버랩
- BM25/Dense/RRF/MMR/Re-ranker top-k
- 웹 검색 활성화, 결과 수, 타임아웃, 지역

현재 설정은 Python 상수로 관리됩니다. 환경변수 기반 설정은 향후 운영 개선 대상입니다.

## 프로젝트 구조

```text
.
├── agent.py                    # LLM 기반 도구 라우팅
├── config.py                   # 애플리케이션 설정
├── document_register.py        # 문서 인덱스 생성
├── document_query_cli.py       # 문서 QA CLI
├── intent_classifier.py        # 질문 의도 분류
├── prompt_templates.py         # 의도별 답변 프롬프트
├── query_router.py             # Agent 장애 시 키워드 폴백
├── rag_engine.py               # 검색 및 답변 생성 파이프라인
├── tools.py                    # Agent 도구 정의
├── web_search.py               # DuckDuckGo 웹 검색
├── web_server.py               # FastAPI 애플리케이션
├── templates/                  # HTML 템플릿
├── static/                     # 프런트엔드 코드와 vendor 파일
├── frontend_tests/             # 프런트엔드 보안 테스트
├── intent_dataset/             # Intent 학습/검증 데이터
├── Roadmap.md                  # 비전, 마일스톤, 현재 위치
├── Problem.md                  # 알려진 미해결 문제
├── Development_M2_Quality_Baseline_Requirement.md
│                               # M2 요구사항과 수용 기준
└── Development_M2_Quality_Baseline_Plan.md
                                # M2 단계별 개발 계획
```

## 라이선스

MIT License. 자세한 내용은 [LICENSE](LICENSE)를 참조하십시오.
