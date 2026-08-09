# M4 Production Readiness 상세 설계

상태: **초안 v4 — [Design Review Iteration 1](Design_Review_Iteration_1.md) MAJOR
M-01~M-07/MINOR m-01~m-03/TRIVIAL t-01, [Iteration 2](Design_Review_Iteration_2.md)
MAJOR M2-01~M2-06/MINOR m2-01~m2-04, [Iteration 3](Design_Review_Iteration_3.md)
MAJOR M3-01~M3-07/MINOR m3-01~m3-02 모두 반영, 구현 대기**
근거: [Requirement.md](Requirement.md), [Plan.md](Plan.md), [milestone_dev_orchestration_guide.md](../../../milestone_dev_orchestration_guide.md)
모델 제약 확인: 본 문서는 Claude Sonnet 5(`claude-sonnet-5`)로 작성했다(guide §모델 사용 제약 준수).
리뷰 대응 매핑은 §14(Iteration 1)/§15(Iteration 2)/§16(Iteration 3)를 참조한다.

## 0. 문서 사용법

이 문서는 제품 코드를 변경하지 않는다. Phase 0~7 구현자가 그대로 옮겨 적을 수
있는 파일 경로, 함수 시그니처, schema, 상태 전이, 오류 코드, metric label,
설정 우선순위, lock 도구/버전, executor 알고리즘, health cache, index 원자적
포인터/롤백, container 파일, runbook 목차, 테스트 fixture/명령을 확정한다.
모호한 지점은 현재 코드(§1)를 근거로 기본값을 정했고, 실측이 기준을 깨뜨릴
때만 사유를 남기고 변경한다(Requirement §7).

## 1. 현재 시스템과 결함 매핑

### 1.1 컴포넌트 맵 (현재)

```
web/server.py (FastAPI, @app.on_event("startup"), 전역 rag_engine)
  -> agent.route_query() -> _decide_tool() -> {web_search_tool | rag_tool}
       -> tools.py: web_search_function -> web_search.search_and_format() -> DDGS
       -> tools.py: rag_function -> rag_engine.get_rag_engine().query()
            -> RAGEngine._retrieve_documents() (BM25/FAISS/RRF/MMR/rerank, 모두 동기)
            -> RAGEngine.generate_answer() (OllamaLLM, 동기, sync_client_kwargs.timeout=600s)
cli/index_documents.py -> FAISS.save_local(VECTORSTORE_PATH) 직접 덮어쓰기
```

### 1.2 결함 -> M4 요구사항 매핑

| 결함(현재 코드) | 위치 | M4 요구사항 |
|---|---|---|
| 상수+부분 env parser 혼재, 검증 계약 없음 | `src/simple_qna_rag/config.py:1-278` | M4-REQ-001 |
| `requirements.txt` 넓은 범위, lock 없음 | `requirements.txt`, `.github/workflows/ci.yml:26` | M4-REQ-002 |
| `print()` 전용 로그 (예: `route_query` 진입/종료) | `agent.py:96,101,224,228,231,239,244`, `web/server.py:43,48,96,101` | M4-REQ-003 |
| 운영 metric 없음 | 전역 | M4-REQ-004 |
| `/health`가 엔진 초기화 실패와 무관하게 200 가능 | `web/server.py:106-112` | M4-REQ-005 |
| `async def rag_query`가 동기 `route_query()`를 직접 await 없이 호출 (event loop 점유) | `web/server.py:71-103` | M4-REQ-006 |
| 요청 크기/CORS/trusted host/오류 schema 없음 | `web/server.py` 전체 | M4-REQ-007 |
| `save_local()` 직접 활성 경로 덮어쓰기, manifest 없음 | `cli/index_documents.py:249-251` | M4-REQ-008 |
| Dockerfile/runbook 없음 | 저장소 루트 | M4-REQ-009 |
| M4 전용 gate/traceability 도구 없음 | `evaluation/` | M4-REQ-010 |

## 2. 공통 설계 계약

### 2.1 candidate ID와 report 규범 (M3 패턴 재사용)

M4 candidate ID 정규식은 M3와 동일한 형태를 `m4-` prefix로 재사용한다:
`^m4-(?:final|p[0-7][a-z]?(?:-[a-z0-9]+)+)$` (`evaluation/reporting.py`의
`_CANDIDATE_ID_RE`를 파라미터화하거나 병렬 상수 `_M4_CANDIDATE_ID_RE`로 추가).
예: `m4-p1-settings-lock`, `m4-p5-index-lifecycle`, `m4-final`.

리포트 디렉터리 규범은 M3와 동일하게 `evaluation/reports/m4/<candidate-id>/`
아래 `logs/`, `fingerprint.json`, `<phase>_report.md`를 둔다.

### 2.2 새 모듈 배치

```
src/simple_qna_rag/
  settings.py                 # Phase 1 — 신규
  errors.py                   # Phase 2 — 신규(오류 taxonomy)
  observability/
    __init__.py
    logging.py                 # Phase 2 — 신규(JSON logging, redaction)
    metrics.py                  # Phase 2 — 신규(metric registry, label allowlist)
  net_budget.py                # Phase 3 — 신규(DeadlineBudget, upstream connect/read/write/pool 예산 파생, §6.6b)
  web/
    server.py                   # Phase 3 — lifespan, health, 미들웨어로 개편, create_app() 팩토리(§9.0)
    concurrency.py               # Phase 3 — 신규(QueryExecutor, §6.4)
    schemas.py                   # Phase 3 — 신규(ErrorResponse 등)
    body_limit.py                 # Phase 3 — 신규(raw ASGI BodySizeLimitMiddleware, §6.6a)
  testing/
    __init__.py
    mock_engine.py                # Phase 3 — 신규, CI 전용(§9.0). 프로덕션 Docker target에 미포함
  index/
    __init__.py
    manifest.py                  # Phase 5 — 신규(IndexManifest, content_digest/version_id, §8.2)
    lifecycle.py                 # Phase 5 — 신규(build/import/activate/rollback/retention, staging 공유, §8.3-8.4)
  cli/
    index_documents.py           # Phase 5 — build 서브커맨드로 개편
    index_lifecycle.py           # Phase 5 — 신규(activate/rollback/import-legacy/retention 서브커맨드)
    web_testonly.py               # Phase 3/6 — 신규, CI 전용(§9.0)
evaluation/
  m4_fingerprint.py              # Phase 0/7 — 신규(dependency/settings/index fingerprint)
  m4_evidence.py                  # Phase 0/7 — 신규(공통 evidence.json 원자적 write/read 및 artifact 경로 검증, §10.1a)
  run_m4_gates.py                 # Phase 7 — 신규(14 gate 고정 DAG 단일 실행기, §10.1c)
  run_static_regression_gate.py   # Phase 7 — 신규(pytest+npm+vendor diff+markdown link+git diff --check 취합, §10.1a)
  m4_gate.py                     # Phase 7 — 신규(§5.1 자동 gate 판정 + evidence fail-closed 검증 + 리포트)
deploy/
  Dockerfile                     # Phase 6 — 신규(runtime/test 멀티스테이지, §9.1)
  docker-compose.yml             # Phase 6 — 신규
  nginx.conf.example             # Phase 6 — 신규(조건부 공개 profile)
scripts/
  scan_image_layers.py           # Phase 6 — 신규(§9.4 M2-05 layer별 경로/known-secret 바이트 스캔)
docs/operations/
  Runbook.md                     # Phase 6 — 신규
  # settings 인벤토리는 별도 문서를 만들지 않는다 — §4.3의 `Settings`
  # dataclass 자체가 단일 원본이다(§4.3b, m2-04/M3-07).
requirements/
  lock-linux-py311.txt           # Phase 1 — 신규(hash-verified)
  TOOL_VERSIONS.md                # Phase 1 — 신규
.dockerignore                    # 저장소 루트, Phase 6 — 신규(§9.2, build context root)
```

`config.py`는 Phase 1 이후 `settings.py`의 `get_settings()` 위에 얇은 facade로
남는다(REQ-001.1의 전환 기간 호환). 기존 공개 상수(`VECTORSTORE_PATH`,
`USE_MMR` 등) import는 그대로 동작하되 내부적으로
`_s = get_settings(); VECTORSTORE_PATH = str(_s.vectorstore_dir)` 형태로
재바인딩한다. Phase 2~5의 `_env_bool`/`_env_enum` 플래그(`MMR_VECTOR_SOURCE`,
`ROUTING_SIGNAL_OVERRIDE`, `ANSWER_TEMPLATE_MODE`, `BM25_TOKENIZER`)도
`settings.py`로 이관하되 **환경변수 이름은 절대 바꾸지 않는다**(REQ-006).

## 3. Phase 0 — M3 기준·위험 고정

### 3.1 산출물

- `evaluation/reports/m4/m4-p0-baseline-check/phase0_report.md` — M2 Phase 0
  스타일을 재사용. `evaluation/fingerprint.py`를 그대로 호출해 4개
  fingerprint(dataset/corpus/index.faiss/index.pkl)가 M3 baseline과 일치함을
  기록한다.
- 현재 `pip list --format=json`/`npm ls --json`, `pip check` 결과를 파일로
  저장(`logs/pip_freeze.json`, `logs/npm_ls.json`, `logs/pip_check.log`).
- mock 2초 query 중 기존 `/health` latency와 1/2/4 동시 요청 처리량/RSS/thread
  수 측정 스크립트: `evaluation/m4_fingerprint.py --pre-change-diagnostics`
  (Phase 0 전용 1회성 플래그, Phase 3 이후 제거 예정 — 코드 주석에 명시).
- 호출 그래프는 본 문서 §1.1을 그대로 인용.
- 아래 §3.2(live 12 case) 및 §3.3(label allowlist)을 이 Phase에서 확정한다
  (Plan §Phase 0 작업 5).

### 3.2 live 12 case ID (고정, `evaluation/datasets/golden.jsonl` 기준)

Requirement §5.1 "document/web route, 긴 answer, abstention 포함"을 만족하는
12건을 실제 M3 answers 리포트(`evaluation/reports/m3/m3-final-approved/answers/`)의
답변 길이 실측으로 선정했다. 4 client × 3 query로 고정 분배한다.

| client | case 1 | case 2 | case 3 | 근거 |
|---|---|---|---|---|
| c1 | `dq-rag-pipeline-001` | `ws-000` | `ua-000` | 최장 답변(1687자, M3 실측) + web + abstention |
| c2 | `dq-retriever-001` | `ws-001` | `ua-001` | document explanation + web + abstention |
| c3 | `dq-sparse-vs-dense-001` | `rr-ws-seoul-temp-001` | `ua-002` | comparison + web(routing-regression) + abstention |
| c4 | `dq-vectorstore-keyword-yn-001` | `dq-realestate-procedure-001` | `dq-agent-arch-001` | yesno + procedure + 장문 백서 근거 |

12개 ID 상수는 `evaluation/m4_fingerprint.py::LIVE_12_CASE_IDS`(정렬된
tuple)와 `evaluation/m4_gate.py`에서 단일 원본으로 import한다. 변경 시 이
표를 함께 갱신한다.

### 3.3 오류/metric label allowlist 초안

§6.3(오류 코드), §5.4(metric label)에서 확정하는 값의 초안을 여기서 먼저
동결한다 — Phase 2/3 구현이 이 표를 변경 없이 그대로 사용한다.

### 3.4 공식 profile

Linux x86_64 CPU, Python 3.11, Node `>=22.22.2 <23`, 단일 Uvicorn worker,
`concurrency=2`/`queue=4`(초기 기본값, Phase 4에서 하향만 가능).

## 4. Phase 1 — dependency lock과 typed settings

### 4.1 lock 도구·버전

- 도구: **pip-tools** `pip-compile`, 버전 고정 `pip-tools==7.4.1`
  (`requirements/TOOL_VERSIONS.md`에 `pip==24.0`, `pip-tools==7.4.1`,
  생성 시 Python `3.11.x`를 기록). Torch는 CPU wheel을 위해
  `--extra-index-url https://download.pytorch.org/whl/cpu`를 컴파일 옵션에
  고정한다(macOS 개발 profile은 이 index를 쓰지 않으므로 §4.2에서 별도
  문서화).
- 생성 명령(공식 Linux CPU profile, 컨테이너 안에서 실행):
  ```bash
  python -m pip install pip==24.0 pip-tools==7.4.1
  pip-compile --generate-hashes --resolver=backtracking \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --output-file=requirements/lock-linux-py311.txt requirements.txt
  ```
- 갱신 절차: `requirements.txt` 변경 시 위 명령 재실행 -> diff 리뷰 ->
  `python -m pip install --require-hashes -r requirements/lock-linux-py311.txt`로
  fresh install 검증 -> 2회 연속 실행 hash 동일성 확인(§4.4) 후 커밋.
- direct/transitive 분리: `requirements.txt`(direct, 의도 선언 유지) vs
  `requirements/lock-linux-py311.txt`(전체 해시 고정, pip-tools가 자동
  구분해 주석으로 `# via` origin을 남김).
- macOS 개발 profile 차이: `requirements/lock-macos-dev.txt`는 **선택
  사항**이며 hash 검증 없이 개발 편의용으로만 존재한다고 명시. CI/container
  공식 설치는 Linux lock만 사용한다.

### 4.2 CI 반영 (`.github/workflows/ci.yml`)

`python-tests` job의 install 단계를 교체:

```yaml
- name: Install locked Python dependencies
  run: python -m pip install --require-hashes -r requirements/lock-linux-py311.txt
- name: Install project package
  run: python -m pip install -e . --no-deps
- name: Check Python dependencies
  run: python -m pip check
```

`package.json`에 `"engines": {"node": ">=22.22.2 <23"}`를 추가하고
`frontend-tests` job에 `node-version: "22.22.2"` 명시(현재 `"22"`만 지정되어
있어 22.17.0 등 하위 버전으로 뜰 위험을 막는다).

### 4.3 `settings.py` — typed settings 계약

```python
# src/simple_qna_rag/settings.py
from __future__ import annotations
import hashlib, json, os
from dataclasses import dataclass, asdict, fields
from enum import Enum
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ENV_PREFIX = "SIMPLE_QNA_RAG_"

class SettingsError(Exception):
    """검증 실패. CLI 진입점은 이 예외를 잡아 `sys.exit(2)`로 변환한다."""

class ReadinessPolicy(str, Enum):
    STRICT = "strict"
    DEGRADED = "degraded"

@dataclass(frozen=True)
class Settings:
    # 경로 (기존 resolve_runtime_path 계약 그대로 재사용)
    documents_dir: Path
    vectorstore_dir: Path
    model_dir: Path
    index_root: Path                      # 신규: runtime/index
    # 모델/검색 (기존 상수 이관, 이름 변경 없음)
    ollama_base_url: str
    ollama_model: str
    embedding_model_name: str
    use_web_search: bool
    # 운영 신규
    log_level: str                        # "DEBUG"|"INFO"|"WARNING"|"ERROR"
    log_format: str                       # "json"|"text"
    request_max_bytes: int                # 기본 16384
    question_max_bytes: int               # 기본 4000
    query_timeout_seconds: float          # 기본 90.0
    concurrency_limit: int                # 기본 2
    queue_limit: int                      # 기본 4
    shutdown_grace_seconds: float         # 기본 30.0
    readiness_policy: ReadinessPolicy     # 기본 strict
    ollama_probe_timeout_seconds: float   # 기본 1.0
    ollama_probe_ttl_seconds: float       # 기본 5.0
    trusted_hosts: tuple[str, ...]        # 기본 ("localhost","127.0.0.1")
    cors_allow_origins: tuple[str, ...]   # 기본 ()
    cors_allow_credentials: bool          # 기본 False
    metrics_bind_all: bool                # 기본 False(loopback 전용 의미: 문서화만, 바인딩 자체는 배포 책임)
    # 신규 — M-04(§6.6) upstream 예산 파생용
    upstream_connect_timeout_seconds: float   # 기본 10.0 — connect 상한(고정 cap)
    upstream_min_read_timeout_seconds: float  # 기본 5.0 — remaining budget이 작아도 유지하는 최소 read 여유
    upstream_safety_margin_seconds: float     # 기본 2.0 — query deadline 이전에 응답 조립을 마치기 위한 예약분
    # 기존 config.py `_env_bool`/`_env_enum` 플래그 이관 — 환경변수 이름은
    # 절대 바꾸지 않는다(§2.2, m2-04). 정확한 전체 목록/type/default/검증은
    # §4.3b 인벤토리 표가 단일 원본이며, 이 dataclass는 그 표를 필드 하나
    # 빠짐없이 그대로 옮긴 것이다(M3-07 — 이전 초안은 이 4개만 우선
    # 반영하고 나머지를 "Phase 1 착수 시 grep"으로 미뤄, retrieval K/
    # chunk size·overlap/hybrid·MMR·reranker 값/web 검색 max·timeout·region
    # 등 실제 `config.py`가 갖고 있던 다수 운영값을 빠뜨렸다).
    answer_template_mode: str          # SIMPLE_QNA_RAG_ANSWER_TEMPLATE_MODE, 기존 허용값 유지
    routing_signal_override: bool      # SIMPLE_QNA_RAG_ROUTING_SIGNAL_OVERRIDE — 기존 config.py 실제 타입은
                                        # bool(`_env_bool`)이다. 이전 초안의 `str | None`은 실제 코드와
                                        # 처음부터 달랐다(Review M3-07 근거 2, 직접 수정).
    mmr_vector_source: str             # SIMPLE_QNA_RAG_MMR_VECTOR_SOURCE, 기존 허용값 유지
    bm25_tokenizer: str                # SIMPLE_QNA_RAG_BM25_TOKENIZER, 기존 허용값 유지
    routing_corpus_topic_hint: bool    # SIMPLE_QNA_RAG_ROUTING_CORPUS_TOPIC_HINT
    # 신규 추가 — §4.3b 표가 기계적으로 열거한 나머지 전체 필드(M3-07)
    retrieval_k: int
    use_mmr: bool
    mmr_fetch_k: int
    mmr_k: int
    mmr_lambda: float
    normalize_embeddings: bool
    use_hybrid_search: bool
    bm25_top_k: int
    dense_top_k: int
    rrf_top_k: int
    rrf_constant: int
    use_reranker: bool
    reranker_model: str
    reranker_top_k: int
    web_search_max_results: int
    web_search_timeout_seconds: float
    web_search_region: str
    mmr_vector_validation_sample: int
    mmr_vector_cosine_floor: float
    routing_corpus_topic_hint_max_items: int
    chunk_size: int
    chunk_overlap: int

    @classmethod
    def load(cls, environ: dict[str, str] | None = None) -> "Settings": ...
    def redacted_dict(self) -> dict: ...   # password/token/URL userinfo -> "<redacted>"
    def canonical_json(self) -> str: ...   # sort_keys, ensure_ascii=False, separators=(",", ":")
    def sha256(self) -> str: ...           # hashlib.sha256(canonical_json(redacted_dict()).encode()).hexdigest() — §4.3a 참조

def get_settings() -> Settings:
    """프로세스 전역 캐시. 최초 호출 시 1회 `Settings.load()`, 이후 동일 객체
    반환(immutable). 테스트는 `settings._reset_cache_for_tests()`로 초기화한다."""
```

**검증 규칙(§4.1 gate의 유효/경계/상호모순 fixture 대상):**

1. bool: `_env_bool`(기존 `{"1","true","yes","on"}` casefold) 규칙 그대로
   재사용.
2. enum: `_env_enum` 그대로, 미허용 값은 `SettingsError`.
3. URL: `ollama_base_url`은 `urlsplit()`으로 scheme이 `http`/`https`인지
   검사. userinfo가 있으면 redaction 대상으로 표시만 하고 거부하지 않는다.
4. 정수/duration: `concurrency_limit>=1`, `queue_limit>=0`,
   `query_timeout_seconds>0`, `shutdown_grace_seconds>=0`,
   `request_max_bytes>=question_max_bytes`(상호 제약 — 위반 시
   `SettingsError("request_max_bytes는 question_max_bytes 이상이어야 합니다")`).
5. 상호모순: `cors_allow_origins==("*",) and cors_allow_credentials is True`
   -> `SettingsError`("wildcard origin과 credentials 동시 허용 금지").
6. 경로: 기존 `resolve_runtime_path()`(environment > 신규 default > legacy)와
   양쪽 존재 시 fail-closed `RuntimeError`를 그대로 재사용하되 `settings.py`
   내부로 이동, `config.py`가 재-export한다.
7. **upstream 예산(M-04, §6.6과 합의):** `query_timeout_seconds >
   upstream_connect_timeout_seconds + upstream_min_read_timeout_seconds +
   upstream_safety_margin_seconds`를 만족하지 않으면 `SettingsError(
   "query_timeout_seconds가 upstream 예산 합계보다 커야 합니다")`. 기본값
   (90.0 > 10.0+5.0+2.0=17.0)은 항상 통과하지만, 운영자가 `query_timeout_seconds`를
   공격적으로 낮출 때 upstream 호출이 query 예산보다 먼저 끝나도록 시작 시점에
   강제한다.

### 4.3a 설정 hash의 의미와 stdout/stderr 계약 (m-03 대응)

`Settings.sha256()`은 **항상 `redacted_dict()`의 canonical JSON**에 대해
계산한다(`raw_dict()`가 아니다) — 단일하고 명확한 정의로 다음 두 필요를
모두 충족한다: (1) settings hash를 fingerprint report/manifest/evidence에
평문 secret 유출 위험 없이 기록할 수 있다, (2) 경로·timeout·flag 등
비밀이 아닌 운영 설정이 바뀌면 hash도 바뀌므로 report 간 설정 drift를
감지하는 본래 목적(REQ-002.4 "settings hash"조회, index manifest의
`settings_hash` 필드 포함)을 만족한다.

**명시적 한계:** secret 값 자체가 바뀌어도 redaction 후 모양(`<redacted>`)이
동일하면 `sha256()`은 변하지 않는다 — 즉 이 hash는 **secret 회전 감지
도구가 아니다**. M4는 secret 회전 감지를 별도 요구하지 않으므로(Requirement에
없음) 이 한계를 문서화하는 것으로 충분하며, keyed/HMAC 기반 별도 secret
fingerprint는 M4 범위에 추가하지 않는다(과설계 방지). 필요해지면 M5에서
별도 설계한다.

**stdout/stderr 계약(§4.4와 합의):** `--check-config`는 stdout에
`redacted_dict()`의 canonical JSON을, stderr에 **같은 redacted 값의**
`sha256()`을 출력한다 — stdout과 stderr는 항상 같은 redaction 경계를
공유하며, 어느 스트림에도 원본 secret이 나타나지 않는다.

**precedence:** `default < 환경변수 < 명시적 CLI 인자`. CLI(`--host`,
`--documents-dir` 등)는 `main()`에서 os.environ에 먼저 써넣은 뒤
`Settings.load()`를 호출하는 기존 패턴(REQ-006 호환, `cli/web.py:22-27`)을
유지한다 — 이러면 우선순위 구현이 이미 존재하는 "os.environ 주입 후 로드"
한 경로로 자연히 성립한다.

### 4.3b typed settings 완전 인벤토리와 직접 `os.environ` 조회 0건 (m2-04 대응)

**문제였던 부분:** §2.2는 "기존 공개 상수 import는 그대로 동작"하고
`MMR_VECTOR_SOURCE`/`ROUTING_SIGNAL_OVERRIDE`/`ANSWER_TEMPLATE_MODE`/
`BM25_TOKENIZER` 같은 `_env_bool`/`_env_enum` 플래그도 이관한다고
서술했지만, §4.3의 `Settings` dataclass는 실제로 이 필드들과 hybrid/MMR/
reranker/chunk 관련 나머지 `config.py` 소비 필드를 담지 않았다 —
구현자가 "무엇을 옮겨야 하는지" 추측해야 하는 공백이었다(Review m2-04).
Requirement는 template/routing/retrieval flag를 최소 설정으로 요구한다
([Requirement.md:79-93](Requirement.md)).

**해결(M3-07 — 더 이상 미루지 않고 여기서 확정한다):**

1. **완전 인벤토리는 §4.3의 `Settings` dataclass 자체다 — 더 이상 "Phase 1
   착수 시 grep"으로 미루지 않는다.** `src/simple_qna_rag/config.py`
   전체를 실측 대조해 `_env_bool`/`_env_enum` 소비 필드 6개
   (`answer_template_mode`/`routing_signal_override`/`mmr_vector_source`/
   `bm25_tokenizer`/`routing_corpus_topic_hint`, 그리고 하드코딩된 나머지
   운영값 — retrieval K, MMR fetch/k/lambda, hybrid search 관련
   BM25/dense/RRF 값, reranker 모델/top-k, web 검색 max/timeout/region,
   MMR 벡터 검증 표본/코사인 하한, corpus hint 최대 항목, chunk size/overlap)를
   §4.3 dataclass에 **모두** 반영했다 — `Settings` dataclass의 필드
   목록 자체가 인벤토리이며 별도 `docs/operations/settings_inventory.md`
   표를 유지보수 대상으로 새로 만들지 않는다(단일 원본 원칙 — dataclass와
   문서 표가 시간이 지나며 어긋나는 이전 설계의 위험을 dataclass
   하나로 없앤다). `routing_signal_override`는 이전 초안의 `str | None`이
   실제 `config.py`의 `bool` 타입과 처음부터 달랐던 것도 함께 바로잡았다
   (Review M3-07 근거 2). `COLLECTION_NAME`/`MMR_EMBED_CACHE_MAX_ITEMS`/
   `INTENT_CONFIDENCE_FLOOR`는 실측 결과 현재 어떤 코드에서도 소비되지
   않는 죽은 상수라 Settings 필드로 승격하지 않고 `config.py` facade의
   미사용 상수로만 남긴다(env를 읽지 않으므로 §4.3b 2번 gate와 무관).
   `TEMPLATES_DIR`/`STATIC_DIR`/`PROMPT_TEMPLATE`은 env를 전혀 소비하지
   않는 구조적 상수/콘텐츠이므로 이 인벤토리(운영자가 튜닝하는 값)
   범위 밖으로 명시적으로 제외한다.
2. **직접 `os.environ` 조회 0건을 정적 테스트로 강제한다.** 신규
   `tests/unit/test_no_direct_environ_access.py`가
   `src/simple_qna_rag/`(단, `settings.py` 자신은 제외 — 유일하게
   환경변수를 읽는 지점)를 AST로 파싱해 `os.environ`, `os.getenv`,
   `os.environ.get` 호출이 **0건**임을 assert한다(AST 검사를 쓰는 이유 —
   문자열 grep은 주석/문자열 리터럴의 오탐과 誤탐 누락을 모두
   일으키므로, `ast.walk()`로 실제 `Attribute`/`Call` 노드만 매칭한다).
   `config.py`(Phase 1 이후 `settings.py` 위 facade)를 포함한 모든
   제품 모듈이 값을 얻는 유일한 경로는 `get_settings()`뿐이라는 계약을
   코드 리뷰가 아니라 CI gate로 강제한다 — 이 테스트가 §5.1 `settings`
   gate의 evidence 일부다(§10.1b).

**결정론적 테스트:** `tests/unit/test_settings.py`가 `dataclasses.fields(Settings)`
개수가 §4.3 dataclass에 실제로 나열된 필드 수(M3-07 기준 30개)와 일치하는지
(회귀 감시 — 필드가 코드 리뷰 없이 조용히 추가/삭제되면 실패), 그리고
config.py의 대응 상수(`retrieval_k`, `use_mmr` 등)를 개별 대입해 값이
동일하게 이관됐는지 assert한다. `tests/unit/
test_no_direct_environ_access.py`가 위 AST 검사를 수행한다.

### 4.4 `--check-config`

`cli/web.py`에 `--check-config` 플래그 추가. 있으면 `Settings.load()`만
호출하고(모델/Ollama/FAISS import 없음) `redacted_dict()`를 canonical JSON으로
stdout에 출력한 뒤 `sha256()`(§4.3a — redacted 값 기준, 원본 secret 없음)를
stderr에 출력하고 `sys.exit(0)`. 검증 실패는 `SettingsError` 메시지를
stderr에 출력하고 `sys.exit(2)`.

```bash
simple-qna-rag-web --check-config
pytest -q tests/unit/test_settings.py tests/integration/test_cli_entrypoints.py
```

## 5. Phase 2 — structured logging과 bounded metrics

### 5.1 로그 이벤트 schema

`observability/logging.py`가 stdlib `logging`에 `JsonFormatter`를 붙인다.
필수 필드: `timestamp`(ISO8601 UTC), `level`, `event`, `service`(`"simple-qna-rag"`),
`version`(`importlib.metadata.version("simple-qna-rag")`), `request_id`,
`duration_ms`(완료 event만), `outcome`, `error_type`. 개발용 `log_format=text`는
`logging.Formatter("%(asctime)s %(levelname)s %(message)s")`로 폴백.

이벤트 이름 allowlist(단일 원본, `observability/logging.py::EVENT_NAMES`,
M2-03/M3-03 대응 — §5.3이 실제로 내보내는 이벤트와 이 allowlist를 정확히
일치시킨다):

```
request_started, request_completed,
stage_completed,               # stage in §5.2 STAGE 목록(상위 4개)
retrieval_substage_completed,  # substage in §5.3 RETRIEVAL_SUBSTAGE(6개)
fallback_triggered,            # reason in §5.3 FALLBACK_REASON(3개)
readiness_probe, index_activate, index_rollback, index_build,
shutdown_draining, shutdown_complete,
log_sink_error                 # §5.6 방어 이벤트
```

**`stage_started`/`retrieval_substage_started`는 이 allowlist에 넣지
않는다(M3-03, Iteration 2 잔여 결함 해소).** Requirement는 stage/substage의
완료·오류 연결만 요구하고 시작 이벤트를 요구하지 않으며([Requirement.md](Requirement.md)),
§5.3의 `ObservationSink` Protocol에도 시작 이벤트 메서드가 없고 실제
구현(`ProductObservationSink`)도 completed 이벤트만 낸다 — allowlist에만
남아 있던 두 값은 "언제 누가 내는지" 정의되지 않은 채 스키마에만 존재해
정확한 started/completed 쌍을 기대하는 후속 구현과 갈라질 위험이 있었다
(Review M3-03). started 이벤트가 필요해지면(예: 장시간 stage의 진행 상황
표시) 별도 리뷰에서 명시적으로 추가한다 — 지금은 완전히 삭제해 스키마와
구현을 1:1로 맞춘다.

### 5.2 request context 전파

`contextvars.ContextVar[str | None]("request_id", default=None)`.
`RequestContextMiddleware`(Starlette `BaseHTTPMiddleware`)가:

1. `observability/request_id.py::resolve_request_id()`(§6.6a에서 정의,
   `^[A-Za-z0-9_-]{1,64}$` 검증 후 유효하면 보존·없거나 무효하면
   `uuid4().hex`로 생성)를 호출한다 — `BodySizeLimitMiddleware`(§6.6a,
   m2-02)와 **같은 함수**를 공유해 413으로 조기 종료되는 요청과 정상
   처리되는 요청의 request ID 계산 규칙이 항상 일치한다.
2. contextvar에 설정, `request_started` 로그(`event="request_started"`,
   `route`, `method`, `path`).
3. 응답 header `X-Request-ID`에 값 반환.
4. `finally`에서 `request_completed`(`duration_ms`, `outcome`,
   `error_type|None`) 로그 — try/except로 감싸 §5.6을 만족.
5. 정확히 한 쌍 보장: `request_started`/`request_completed`를 같은
   `try/finally` 블록 안에서 발행(중간 예외로 스킵 불가).

### 5.3 stage 로그 — 기존 `RetrievalTrace`와의 관계, retrieval sub-stage/fallback (M-05/M2-03 대응)

`RetrievalTrace`(내부 6단계 `query_embed|bm25|dense|rrf|mmr|reranker`)는
**평가용 계측 계약을 그대로 유지**하고 필드/이름을 변경하지 않는다
(M3-REQ-002 계약 보존). 제품 관측용 stage는 상위 4개와, retrieval 내부만
별도로 세분화한 6개 sub-stage 두 층으로 집계한다 — 상위 latency
히스토그램의 cardinality를 낮게 유지하면서(§5.4) retrieval 내부
병목/오류를 구분하기 위함(REQ-003.3, Review M-05):

```python
STAGE = ("routing", "web_search", "retrieval", "generation")            # 상위 4개, latency histogram
RETRIEVAL_SUBSTAGE = ("query_embed", "bm25", "dense", "rrf", "mmr", "reranker")  # RetrievalTrace 이름과 1:1
SEARCH_TYPE = ("web_search", "document_qa")                              # agent.py route_query()의 최종 search_type
FALLBACK_REASON = ("agent_error", "web_search_failed", "no_tool_selected")  # agent.py의 3개 실제 폴백 분기
```

`FALLBACK_REASON` 값은 현재 `agent.py::route_query()`의 실제 분기와
정확히 대응한다(코드는 변경하지 않고 이 3개 지점에 로그/metric 호출만
추가한다):

1. `agent_error` — `_decide_tool()` 호출 자체가 예외를 던져 `keyword_fallback_route()`로
   폴백(`agent.py` "Agent 라우팅 실패, 키워드 라우터로 폴백" 분기).
2. `web_search_failed` — `web_search_tool.func()`가 `success=False`를 반환해
   `document_qa`로 재시도(`agent.py` "웹검색 실패, document_qa로 재시도" 분기).
3. `no_tool_selected` — LLM이 도구를 선택하지 못해 `keyword_fallback_route()`로
   폴백(`agent.py` "Agent가 도구를 선택하지 못함" 분기).

#### `ObservationSink` — 제품 계측 seam과 평가 trace 공유(M2-03)

이전 설계는 "제품 query가 trace를 생성하는지, callback sink를 주입하는지,
평가 trace와 중복 측정을 어떻게 피하는지" 정의하지 않았고(Review M2-03),
`safe_log`/`safe_observe` 적용 범위도 `web/server.py` 호출로만 서술해 실제
stage/fallback hook이 있는 `agent.py`/`rag_engine.py`를 빠뜨렸다. 아래
`ObservationSink`로 두 문제를 함께 닫는다.

**M3-03 대응 — 실제 `RetrievalStageTrace`/`RetrievalTrace` 계약과의 호환을
설계 수준에서 확정한다.** 이전 의사코드는 `trace.record(name, duration_ms,
outcome)`을 호출했지만 현재 `rag_engine.py::RetrievalTrace`에는 `record()`
메서드가 없고 `stages: list[RetrievalStageTrace]`만 있으며,
`rag_engine.py::RetrievalStageTrace`도 `name`/`latency_ms`/`candidate_count`
세 필드뿐이라 `outcome`을 받을 자리가 없다(그대로 구현하면 첫 retrieval
호출에서 `AttributeError`). 아래 세 가지로 이 불일치를 완전히 없앤다: (1)
`RetrievalStageTrace`는 **필드를 하나도 추가하지 않고 3필드 그대로** 계속
쓴다, (2) `outcome`은 trace에 절대 저장하지 않고 **sink 호출에만** 투영한다,
(3) sink를 호출하는 모든 지점(`stage`/`retrieval_substage`/`fallback` 셋
다)이 `safe_sink_call()` 하나만 거친다.

```python
# src/simple_qna_rag/observability/__init__.py (신규, Phase 2)
from typing import Protocol

class ObservationSink(Protocol):
    def stage(self, name: str, *, duration_ms: float, outcome: str) -> None: ...
    def retrieval_substage(self, name: str, *, duration_ms: float, outcome: str) -> None: ...
    def fallback(self, reason: str) -> None: ...

def safe_sink_call(fn, *args, **kwargs) -> None:
    """§5.6/M3-03 — `ObservationSink`의 세 메서드를 호출하는 **유일한**
    경로. `web/server.py`/`agent.py`/`rag_engine.py` 어디서 sink를
    호출하든 반드시 이 함수를 거친다 — `ProductObservationSink`뿐 아니라
    테스트가 주입하는 임의의(고장난) sink 구현체가 예외를 던져도 여기서
    삼키고 `log_sink_error`만 stderr에 남긴다. 이전 설계는
    `ProductObservationSink` 내부만 `safe_log`/`safe_observe`로 감싸
    자기 자신은 안전했지만, `agent.py`/`rag_engine.py`가 `sink.stage(...)`/
    `sink.fallback(...)`을 **직접** 호출하고 있어 non-Product(테스트) sink가
    예외를 던지면 그 예외가 `route_query()`/`RAGEngine.query()`까지 그대로
    전파됐다(Review M3-03 — "sink 메서드 자체가 예외여도 route_query/
    RAGEngine 결과 정상"이라는 테스트 계약이 이 경로로는 통과할 수
    없었다). 세 호출 지점을 이 함수 하나로 통일해 그 공백을 구조적으로
    없앤다."""
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - 관측 경로 예외는 제품 흐름에 전파하지 않는다
        _stderr_log.error("log_sink_error: sink=%r error=%r", getattr(fn, "__self__", fn), exc)

class ProductObservationSink:
    """유일한 production 구현. 세 메서드 내부는 `safe_log`/`safe_observe`로
    "로깅/메트릭 라이브러리 자체의 실패"를 방어한다(§5.6) — 이는
    `safe_sink_call()`이 방어하는 "sink 구현체 자체의 버그"와는 다른
    층이며, 두 방어가 함께 있어야 어떤 sink를 어떻게 호출해도 제품 흐름이
    깨지지 않는다."""
    def stage(self, name, *, duration_ms, outcome):
        safe_log(_logger, "INFO", "stage_completed", stage=name, duration_ms=duration_ms, outcome=outcome)
        safe_observe(STAGE_DURATION_SECONDS.labels(stage=name).observe, duration_ms / 1000)
        if outcome != "success":
            safe_observe(STAGE_ERRORS_TOTAL.labels(stage=name).inc)

    def retrieval_substage(self, name, *, duration_ms, outcome):
        safe_log(_logger, "INFO", "retrieval_substage_completed", substage=name, duration_ms=duration_ms, outcome=outcome)
        safe_observe(RETRIEVAL_SUBSTAGE_TOTAL.labels(substage=name).inc)
        safe_observe(RETRIEVAL_SUBSTAGE_DURATION_SECONDS_TOTAL.labels(substage=name).inc, duration_ms / 1000)
        if outcome != "success":
            safe_observe(RETRIEVAL_SUBSTAGE_ERRORS_TOTAL.labels(substage=name).inc)

    def fallback(self, reason):
        safe_log(_logger, "WARNING", "fallback_triggered", reason=reason)
        safe_observe(FALLBACK_TOTAL.labels(reason=reason).inc)
```

**주입 지점:** `RAGEngine.__init__(self, ..., observation_sink:
ObservationSink = ProductObservationSink())`와
`agent.route_query(question, *, budget, observation_sink:
ObservationSink = ProductObservationSink())`가 생성자/함수 인자로 주입받는다
(기존 공개 시그니처에 기본값 있는 keyword-only 인자 추가이므로 REQ-006
호환). `web/server.py`는 별도 sink를 만들지 않고 이 기본값을 그대로
쓰며(단일 production 경로), 테스트만 fake sink를 주입해 호출 여부/인자를
단언한다.

**`RetrievalTrace`와의 단일 측정 — 이중 계측 금지, 기존 3필드 스키마
그대로 보존(M2-03/M3-03 핵심 수정):** retrieval 6개 sub-stage 각각을
감싸는 하나의 헬퍼가 시간을 **한 번만** 잰 뒤 두 소비자에게 같은 값을
전달한다 — 개별 timer를 두 번 만들지 않는다. 기존 `_retrieve_documents()`의
내부 `stage()` 헬퍼가 이미 하던 일(성공한 결과에만 `len(result)`로
`candidate_count`를 매기고, query embedding처럼 문서 리스트가 아닌
반환값은 별도 규칙을 쓰는 것)을 그대로 일반화한다 — 임의 반환형에
`len()`을 강제 적용해 `candidate_count`를 깨뜨리지 않는다:

```python
def _measure_substage(
    name: str, fn, *, candidate_count_of, trace: "RetrievalTrace | None", sink: ObservationSink
):
    """RetrievalStageTrace(name, latency_ms, candidate_count) 3필드를 그대로
    생성하는 유일한 helper(M3-03) — 필드를 추가하지 않는다. `outcome`은
    trace에 절대 저장하지 않고 sink 호출에만 투영한다. `candidate_count_of`는
    호출자가 넘기는 `result -> int` 함수로, 문서 리스트 stage는 `len`을,
    query embedding처럼 리스트가 아닌 반환값은 `lambda _: 0`을 넘겨 기존
    `_retrieve_documents()`의 두 규칙을 그대로 보존한다."""
    start = time.perf_counter()
    try:
        result = fn()
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        # 실패한 stage는 기존 동작과 동일하게 RetrievalStageTrace에 항목을
        # 남기지 않는다(기존 `_retrieve_documents()`도 예외 시 append 이전에
        # 그대로 전파됨) — 다만 sink에는 반드시 outcome="error"로 알린다.
        safe_sink_call(sink.retrieval_substage, name, duration_ms=duration_ms, outcome="error")
        raise
    duration_ms = (time.perf_counter() - start) * 1000
    if trace is not None:
        trace.stages.append(RetrievalStageTrace(name, duration_ms, candidate_count_of(result)))
    safe_sink_call(sink.retrieval_substage, name, duration_ms=duration_ms, outcome="success")
    return result
```

제품 경로는 `trace=None`으로 호출해 `RetrievalTrace` 객체 생성 비용이
전혀 들지 않는 기존 zero-cost 계약을 그대로 유지하면서도(`rag_engine.py::RetrievalTrace`,
`rag_engine.py::RAGEngine._retrieve_documents` 계약 보존), `sink.retrieval_substage(...)`는
항상 호출돼 제품 관측이 빠짐없이 남는다. 평가 harness는 `trace`를 넘겨
같은 측정값을 `RetrievalTrace`에도 채운다 — 두 경로가 서로 다른 시계로
각자 재는 대신 하나의 `time.perf_counter()` 구간을 공유한다.
`_retrieve_documents()`의 6개 호출 지점 중 `query_embed`만
`candidate_count_of=lambda _: 0`을 넘기고 나머지(`bm25`/`dense`/`rrf`/
`mmr`/`reranker`)는 `candidate_count_of=len`을 넘긴다 — 기존 코드의 두
분기(공용 `stage()` 헬퍼가 `len(result)`를 쓰는 경로, query embedding이
별도로 `candidate_count=0`을 쓰는 경로)를 그대로 유지한다.

`agent.route_query()`와 `rag_engine.RAGEngine.query()`는
`safe_sink_call(sink.stage, name, duration_ms=..., outcome=...)` 호출로
상위 4단계 완료를 남긴다. `retrieval` stage의 `duration_ms`는
`RetrievalTrace(name="total")`이 이미 계산한 값을 재사용해 이중 측정하지
않는다(trace가 없는 제품 경로에서는 `time.perf_counter()`로 감싼다). §5.3
1~3에 나열한 3개 fallback 분기는 `safe_sink_call(sink.fallback, reason)`
호출로 남긴다 — `web/server.py`의 요청 수명주기 로그(§5.2)를 제외한
sink로 향하는 모든 호출이 예외 없이 `safe_sink_call()` 하나만 거친다.

### 5.4 metric registry

의존성: `prometheus-client`(작은 dependency, REQ-004.1 허용 범위) 신규 추가,
`requirements.txt`/lock에 반영. **전용 `CollectorRegistry` 인스턴스**를
`observability/metrics.py::REGISTRY`로 만들어 전역 default registry를
오염시키지 않는다(테스트 격리).

**label allowlist(단일 원본, §5.3의 4개 tuple을 그대로 import):**

```python
ROUTE = ("rag",)
OUTCOME = ("success", "client_error", "server_error", "rejected", "timeout")
STAGE = ("routing", "web_search", "retrieval", "generation")
RETRIEVAL_SUBSTAGE = ("query_embed", "bm25", "dense", "rrf", "mmr", "reranker")
SEARCH_TYPE = ("web_search", "document_qa")
FALLBACK_REASON = ("agent_error", "web_search_failed", "no_tool_selected")
ERROR_TYPE = tuple(e.value for e in ErrorCode)  # §6.3, 10개
```

**metric 정의:**

| 이름 | 종류 | label | 용도 |
|---|---|---|---|
| `qna_rag_requests_total` | Counter | route, outcome | 총수/성공/거절/오류 |
| `qna_rag_requests_in_progress` | Gauge | route | 진행 중 |
| `qna_rag_request_duration_seconds` | Histogram(7개 명시 boundary, m2-03) | route, outcome | request latency |
| `qna_rag_stage_duration_seconds` | Histogram(7개 명시 boundary, m2-03) | stage | 상위 4단계 latency |
| `qna_rag_stage_errors_total` | Counter | stage | 상위 4단계별 오류 수(M-05 신규) |
| `qna_rag_retrieval_substage_total` | Counter | substage | retrieval 내부 6단계 실행 횟수(M-05 신규) |
| `qna_rag_retrieval_substage_duration_seconds_total` | Counter | substage | retrieval 내부 6단계 누적 소요 시간(초, M-05 신규 — Histogram이 아니라 합산 Counter로 cardinality를 억제하고 `.../_total ÷ qna_rag_retrieval_substage_total`로 평균을 구한다) |
| `qna_rag_retrieval_substage_errors_total` | Counter | substage | retrieval 내부 6단계별 오류 수(M-05 신규) |
| `qna_rag_search_type_total` | Counter | search_type | route_query 최종 분기(web_search/document_qa) 횟수(M-05 신규) |
| `qna_rag_fallback_total` | Counter | reason | agent 폴백 발생 횟수(M-05 신규) |
| `qna_rag_errors_total` | Counter | error_type | 안정된 오류 유형별 오류 수 |
| `qna_rag_ready` | Gauge(0/1, no label) | — | readiness |
| `qna_rag_queue_depth` | Gauge(no label) | — | executor waiting 수 |
| `qna_rag_executor_running` | Gauge(no label) | — | executor running 수 |
| `qna_rag_executor_orphaned` | Gauge(no label) | — | timeout/cancel 후에도 thread에서 계속 도는 작업 수(§6.4) |
| `qna_rag_index_version_info` | Gauge(value=1) | version | 활성 index version — activate/rollback마다 이전 label을 `.remove()`한 뒤 새로 `.set()`해 **항상 정확히 1개 series만** 유지한다(M-05, 아래 cardinality 절 참조) |
| `qna_rag_build_info` | Gauge(value=1) | git_sha, version | build provenance |

**cardinality 예산(§5.1 gate 상한 150, M2-03 근거로 재계산 — m2-03 대응):**
Iteration 1의 "×10 근사"는 실제 `prometheus_client.generate_latest()`가
찍는 sample 수와 다르다(Review m2-03) — Histogram은 boundary가 `N`개면
`le="+Inf"` 포함 `N+1`개의 `_bucket` sample에 `_sum`/`_count`가 더해지고,
Counter/Histogram 모두 `_created` sample이 추가될 수 있다. 이 계산은 두
가지를 고정해 모호함을 없앤다: (1) `requirements/lock-linux-py311.txt`가
`prometheus-client` 버전을 정확히 고정하고(§4.1과 동일한 lock 원칙), (2)
`observability/metrics.py` 모듈 최상단에서
**`prometheus_client.disable_created_metrics()`를 호출**해 `_created`
sample 생성을 명시적으로 끈다(M3-07 수정 — read-only spike로
`prometheus_client==0.26.0`의 `prometheus_client.metrics` 소스를 직접
확인한 결과, `_use_created` 플래그를 끄는 방법은 `PROMETHEUS_DISABLE_CREATED_SERIES`
env var 뿐 아니라 **공개 non-env 함수**
`prometheus_client.metrics::disable_created_metrics()`도 있다 —
`__init__.py`가 `disable_created_metrics`/`enable_created_metrics`를
top-level export한다. env var 경로는 §4.3b 2번이 강제하는 "`settings.py`
밖 `os.environ` 접근 0건" AST gate와 정면 충돌했지만(Review M3-07 근거
4), 이 함수 호출은 `os.environ`을 전혀 건드리지 않으므로 그 gate와
충돌하지 않는다 — 이 설정 없이는 `_created`가 버전/설정에 따라 있을
수도 없을 수도 있어 상한 계산 자체가 재현 불가능해진다).

Counter/Gauge(`_created` 비활성화 후 label 조합당 정확히 1 series):
`requests_total`(route×outcome=5), `requests_in_progress`(1),
`errors_total`(10), `stage_errors_total`(4), `retrieval_substage_total`(6),
`retrieval_substage_duration_seconds_total`(6),
`retrieval_substage_errors_total`(6), `ready`(1), `queue_depth`(1),
`executor_running`(1), `executor_orphaned`(1), `index_version_info`(**1
고정**, 아래 참조), `build_info`(1), `search_type_total`(2),
`fallback_total`(3). 소계 **49**.

Histogram(label 조합당 `boundary 개수 + 1(+Inf) + _sum + _count` sample,
`_created` 비활성화 후 실제 client 계약): boundary를 **7개**로 명시
고정하면 조합당 `7 + 1 + 1 + 1 = 10` sample이다 —
`request_duration_seconds`(route×outcome=5×10=50),
`stage_duration_seconds`(상위 STAGE=4×10=40). 소계 **90**.

합계 **139** ≤ 150. boundary를 8개로 뒀다면 조합당 11 sample이 되어
소계 99, 합계 148로 여유가 단 2 sample만 남아 향후 label/metric 추가
여지가 사실상 없었다 — 정확한 sample 계약으로 재계산한 뒤 boundary를
7개로 낮춰 안전 여유(11 sample)를 확보했다(m2-03 "필요하면 bucket/metric을
줄여 150 이하 여유를 확보하라" 반영). retrieval sub-stage는 Histogram이
아니라 Counter 3개로 투영했기 때문에(만약 6-value substage에도 Histogram을
썼다면 6×10=60이 추가되어 예산을 초과했을 것) 상세 진단과 예산 준수를
동시에 만족한다.

**`index_version_info`가 반복 activate/rollback에도 1로 고정되는 이유:**
이전 설계는 "프로세스 수명 중 최대 1회 rollback"을 가정해 ≤2로 계산했으나
Phase 5는 반복 activate/rollback을 지원하므로 이 가정은 깨진다(Review
M-05). `index/lifecycle.py`의 activate/rollback 성공 경로 마지막 단계에서
`metrics.py::set_active_index_version(new_version_id)`를 호출하도록 하고,
이 함수는 내부에 저장한 이전 `version_id`가 있으면
`INDEX_VERSION_INFO.remove(old_version_id)`를 먼저 호출한 뒤
`INDEX_VERSION_INFO.labels(version=new_version_id).set(1)`한다 — 그 결과
어떤 시점에도 label 값 1개만 존재한다.

**테스트:** `tests/unit/test_metrics_cardinality.py`가 (1) 1,000개 고유
request_id/질문 주입 후 `REGISTRY.collect()` 총 sample 수 ≤150을, (2) 서로
다른 100개 `version_id`로 activate/rollback을 100회 반복한 뒤
`qna_rag_index_version_info`의 sample이 정확히 1개이고 그 값이 마지막
activate 대상과 일치함을, (3) 전체 series 수가 여전히 ≤150임을 assert한다
(M-05 "100회 lifecycle cardinality 상한" 요구). `request_id`, 질문, source,
파일명, exception message는 어떤 label에도 사용하지 않는다. 이 테스트는
`generate_latest()`가 실제로 만드는 raw sample을 세므로(근사 계산이
아님), lock된 `prometheus-client` 버전과 `disable_created_metrics()`
호출(M3-07)이 흔들리면 CI에서 즉시 드러난다(m2-03).

### 5.5 `/metrics` endpoint

`web/server.py`에 `GET /metrics` 추가, `prometheus_client.generate_latest(REGISTRY)`
반환(`content_type=CONTENT_TYPE_LATEST`). 기본 바인딩은 앱 bind와 동일하되
운영 문서(§8 runbook)에서 loopback/신뢰망 전용을 권고하고, 외부 노출 시
reverse proxy에서 차단하도록 명시한다(코드 레벨 강제는 하지 않음 — REQ가
"기본은 loopback/신뢰망 전용"이라고만 명시).

### 5.6 로그/metric 오류 방어

`observability/logging.py::safe_log(logger, level, event, **fields)`와
`observability/metrics.py::safe_observe(metric_fn, *args)`가 내부에서
try/except로 모든 예외를 삼키고 `log_sink_error` stderr 폴백 1줄만 남긴다 —
이 둘은 "로깅/메트릭 라이브러리 자체의 실패"를 방어한다.
`observability/__init__.py::safe_sink_call(fn, *args, **kwargs)`(§5.3,
M3-03 신규)는 그와 다른 층인 "sink 구현체 자체의 실패"를 방어한다 —
`web/server.py`의 요청 수명주기 로그(§5.2)는 `safe_log`/`safe_observe`를
직접 호출하고, `agent.py`/`rag_engine.py`의 `stage`/`retrieval_substage`/
`fallback` 세 호출은 예외 없이 모두 `safe_sink_call()`을 거친다(M3-03 —
이전 설계는 `ProductObservationSink` 내부만 안전했을 뿐 `agent.py`/
`rag_engine.py`가 sink 메서드를 직접 호출해, 테스트가 주입하는 고장난
sink의 예외가 그대로 전파될 수 있었다). 어떤 모듈도
`logging`/`prometheus_client`를 이 경로 밖에서 직접 호출하지 않는다
(REQ-003.5).

**결정론적 테스트(신규 `tests/integration/test_observation_sink.py`,
M2-03/M3-03 대응):** fake sink를 주입해 (1) retrieval 6개 sub-stage 각각
정상 완료 시 `stage`가 아니라 `retrieval_substage`가 정확히 6회 호출되고
각 호출의 `RetrievalStageTrace`(trace를 넘긴 경우)가 3필드
(`name`/`latency_ms`/`candidate_count`) 그대로이며 `query_embed`만
`candidate_count==0`, 나머지는 `len(result)`와 일치함, (2) §5.3의 3개
fallback 분기 각각에서 `fallback(reason)`이 대응하는 `FALLBACK_REASON`
값으로 정확히 호출됨, (3) sink의 `stage`/`retrieval_substage`/`fallback`
**세 메서드 모두 각각 개별적으로** 예외를 던지도록 monkeypatch해도
`route_query()`/`RAGEngine.query()`가 정상 결과를 반환함(`safe_sink_call()`이
세 호출 지점 모두를 동일하게 격리함을 개별 검증 — 이전 설계는 이
계약을 하나로 뭉뚱그려 서술해 `stage`/`fallback` 직접 호출 지점의 결함을
가렸다), (4) `RetrievalStageTrace`에 `outcome` 필드가 존재하지 않음(스키마
불변 회귀 감시)을 assert한다.

## 6. Phase 3 — health, lifecycle, blocking과 입력 경계

### 6.1 애플리케이션 생명주기 상태 머신

```python
class LifecycleState(str, Enum):
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    STOPPED = "stopped"
```

전이: `STARTING -[엔진+인덱스 검증 성공]-> READY`,
`READY -[lifespan shutdown 시작]-> DRAINING`,
`DRAINING -[진행 중 요청 0 또는 grace 만료]-> STOPPED`.
`STARTING -[엔진 초기화 실패]-> STARTING 유지`(readiness는 계속 503, liveness는
200 — 컨테이너가 무한 재시작 루프에 빠지지 않고 로그로 원인을 볼 수 있게 함).

FastAPI `lifespan` context manager(`@asynccontextmanager`)로 구현, 기존
`@app.on_event("startup")`을 대체한다. `app.state.lifecycle = LifecycleState.STARTING`,
`app.state.inflight = 0`(진행 중 `/rag` 요청 카운터, `QueryExecutor`가 §6.4의
ticket 전이와 같은 lock 아래 증감).

### 6.1a graceful shutdown — drain 알고리즘과 executor 종료 순서 (m-01 대응)

`app.state.drain_complete: asyncio.Event`를 lifespan 시작 시 생성한다.
shutdown 절차는 `lifespan`의 `yield` 이후 블록에서 순서대로 실행된다:

```python
# lifespan 함수의 yield 이후(shutdown) 블록
app.state.lifecycle = LifecycleState.DRAINING
app.state.query_executor.begin_drain()                             # 1
try:                                                                # 2
    await asyncio.wait_for(app.state.drain_complete.wait(),
                            timeout=settings.shutdown_grace_seconds)
except asyncio.TimeoutError:
    pass  # grace 만료 — 진행 중인 작업을 기다리지 않고 다음 단계로
app.state.query_executor.shutdown_pool(wait=False, cancel_futures=True)  # 3
app.state.lifecycle = LifecycleState.STOPPED                       # 4
```

1. **대기 중(QUEUED) waiter 즉시 거절, idle이면 즉시 완료(m2-01).**
   `QueryExecutor.begin_drain()`은 `_draining=True`로 설정하고 내부 FIFO
   deque(§6.4)의 모든 ticket에 `ticket.rejected = True`를 표시한 뒤
   `ticket.event.set()`으로 깨운다 — 깨어난 `run()` 호출은 `ticket.rejected`를
   보고 즉시 `NotReadyError`를 던진다(실행을 시작하지 않으므로 `_running`을
   건드리지 않는다). 그 직후 `begin_drain()`은 `_check_drain_locked()`를
   호출해 이 시점에 이미 `_running == 0`(idle)이면 `drain_complete`를 즉시
   `set()`한다 — 이전 설계는 이 즉시-set 경로가 없어 idle 상태에서도 매번
   `shutdown_grace_seconds` 전체를 대기했다(Review m2-01). 이 시점 이후 신규
   `/rag` 요청은 §6.1의 lifecycle 체크가 executor보다 먼저 거절한다(코드
   순서상 executor에 도달하지 않음).
2. **진행 중(RUNNING) 작업은 grace period까지만 기다린다.** `_running`이
   0이 되고 `lifecycle==DRAINING`인 순간 §6.4의 `_finalize` 경로가
   `app.state.drain_complete.set()`을 호출한다(`_check_drain_locked()` 내부,
   이미 획득한 lock 아래). grace 만료 시 대기를 포기하고 다음 단계로 진행 —
   진행 중인 thread pool 작업은 강제 종료하지 않는다(§6.4와 동일한 "강제
   중단 불가" 한계).
3. **thread pool은 논블로킹으로 shutdown한다.** `query_executor.shutdown_pool(
   wait=False, cancel_futures=True)`는 §6.4의 public API로 내부
   `ThreadPoolExecutor.shutdown(...)`을 감싼다 — 이전 설계는 executor가 노출
   하지 않는 `query_executor.pool` 속성을 직접 호출해 구현하면
   `AttributeError`가 나는 상태였다(Review m2-01). `wait=False`이므로 이
   호출은 즉시 반환한다. `cancel_futures=True`는 아직 시작하지 않은(아직
   pool 내부 큐에 있는) future만 취소한다 — 이미 실행 중인 future는 계속
   실행된다(Python 표준 라이브러리 계약, 강제 중단 아님).
4. `STOPPED` 전이 후 프로세스가 실제로 종료될 때까지 시간차가 있을 수
   있다 — 그 사이 grace 만료 후에도 여전히 실행 중이던 abandoned 작업이
   완료되며 `_schedule_finalize`가 `asyncio.ensure_future(self._finalize(ticket))`로
   새 task를 스케줄할 때 이미 닫힌 loop를 대상으로 호출될 수 있다(§6.4
   M3-01 — 이전 설계는 여기서 `call_soon_threadsafe()`를 한 번 더 거쳤으나,
   `loop.run_in_executor()`의 future는 `asyncio.Future`이므로 완료 콜백은
   이미 loop thread 위에서 실행돼 그 추가 hop이 thread-safety 이득 없이
   지연만 더했다 — 개정된 설계는 이 hop을 제거해 지연 구간 자체를
   줄인다). 이 경우 `RuntimeError`를 잡아 stdlib
   `logging.getLogger(...).error(...)`(asyncio 문맥이 필요 없는 순수
   thread-safe 호출)로 stderr에 한 줄만 남기고 무시한다 — 프로세스 종료
   자체를 막지 않는다.

**bounded 프로세스 종료가 아니라는 한계(Review m2-01 근거 3):**
`ThreadPoolExecutor.shutdown(wait=False)`는 이미 실행 중인 thread를 강제
종료하지 않고, CPython은 인터프리터 종료 시점에 non-daemon executor thread를
join한다 — 즉 `STOPPED` 전이 이후에도 실제 프로세스 종료는 남은 worker가
끝날 때까지 지연될 수 있다. 이 한계를 코드나 문서 어디서도 "grace 이후
프로세스가 bounded 시간 안에 죽는다"고 서술하지 않으며,
`docs/operations/Runbook.md#incident-triage`(§9.5)가 process supervisor
(K8s `terminationGracePeriodSeconds` 경과 후 SIGKILL, systemd
`TimeoutStopSec` 등)에 의한 hard-stop을 M4 프로세스 자체 책임 밖의 필수
운영 조건으로 명시한다.

**shutdown state table:**

| lifecycle | 신규 `/rag` 요청 | QUEUED waiter | RUNNING 작업 | `inflight==0` 도달 시점 |
|---|---|---|---|---|
| READY | 정상 admit | FIFO 대기 | 정상 실행 | — |
| DRAINING(진입 직후) | lifecycle 체크가 executor 이전에 `not_ready` 즉시 거절 | `begin_drain()`이 즉시 깨워 `not_ready`로 거절 | 계속 실행(강제 중단 안 함) | 이미 idle이면 즉시(`begin_drain()` 내부), 아니면 `drain_complete.wait()`로 대기 시작 |
| DRAINING(grace 만료) | 계속 거절 | (이미 비어 있음) | 계속 실행 — `shutdown_pool(wait=False, cancel_futures=True)`만 호출 | 대기 없이 STOPPED로 진행 |
| STOPPED | 프로세스 종료 절차 | — | finalize 콜백은 best-effort(닫힌 loop면 RuntimeError를 잡아 무시) | — |

**결정론적 테스트(신규 `tests/integration/test_shutdown_drain.py`):**

| 테스트 | 강제 방법 | 검증 |
|---|---|---|
| `test_queued_waiter_rejected_on_drain` | `threading.Event`로 실행 중 작업을 붙잡아 concurrency를 포화시킨 뒤 추가 요청을 QUEUED로 만들고 shutdown 트리거 | QUEUED 요청이 `not_ready`로 즉시 반환, deque 즉시 0 |
| `test_idle_drain_completes_immediately` | shutdown 트리거 시점에 진행 중 작업이 0개(`_running == 0`) | `begin_drain()` 호출 직후 `drain_complete`가 이미 set 상태 — `asyncio.wait_for`가 대기 없이 즉시 반환(grace 전체를 기다리지 않음, m2-01) |
| `test_inflight_finishes_within_grace` | grace보다 짧게 걸리는 fake worker | 정상 응답 반환 후 STOPPED |
| `test_grace_timeout_still_stops` | grace보다 오래 걸리는(`threading.Event`로 무기한 대기) fake worker, 짧은 `shutdown_grace_seconds` | grace 만료 후 대기 없이 STOPPED 전이, orphan 카운트 유지 |
| `test_shutdown_pool_public_api` | lifespan 코드가 `query_executor.shutdown_pool(...)`만 호출(내부 `_pool` 필드 미참조) | `AttributeError` 없이 `ThreadPoolExecutor.shutdown(wait=False, cancel_futures=True)`가 정확히 1회 호출됨(m2-01) |
| `test_finalize_callback_after_loop_closed` | 이벤트 루프를 닫은 뒤 `threading.Barrier`로 붙잡아 둔 fake worker를 완료시켜 콜백을 유발 | `RuntimeError`가 잡혀 프로세스가 죽지 않고 stderr 로그 1줄만 남음 |

### 6.2 `/health/live`, `/health/ready`, deprecated `/health`

```python
@app.get("/health/live")
async def health_live():
    return {"status": "live"}  # 항상 200, 외부 의존성 조회 없음
```

`/health/ready` 응답 schema:

```json
{"status": "ready", "reason": null,
 "checks": {"settings": "ok", "engine": "ok", "index": "ok", "ollama": "ok"}}
```

reason 코드(안정, 하나만 채택 — 우선순위 순):

```
starting, engine_not_initialized, index_invalid,
ollama_unreachable, draining
```

판정 로직(순서대로 첫 실패에서 short-circuit, 무거운 query 실행 없음):

1. `lifecycle == DRAINING` -> `reason="draining"`, 503.
2. `lifecycle == STARTING` -> `reason="starting"`, 503.
3. `RAGEngine` 미초기화 -> `reason="engine_not_initialized"`, 503.
4. index manifest 검증 실패(§7.5) -> `reason="index_invalid"`, 503.
5. `readiness_policy == strict`이고 Ollama probe 실패 -> `reason="ollama_unreachable"`, 503.
   `degraded` policy면 이 단계를 건너뛰고 `checks.ollama="degraded"`로 표시,
   200을 허용(문서화된 향후 profile, M4 기본은 `strict`).
6. 모두 통과 -> `status="ready"`, 200.

**Ollama readiness probe:** `httpx.get(f"{ollama_base_url}/api/tags",
timeout=ollama_probe_timeout_seconds)`. 결과를 `(ok: bool, checked_at: float)`
로 프로세스 메모리에 TTL cache(`ollama_probe_ttl_seconds=5.0`)한다 — cache
적중 시 네트워크 호출 없이 즉시 반환(§5.1 gate: cached p95 ≤50ms). cache
miss/만료 시에만 실제 확률 호출(uncached p95 ≤250ms 목표를 위해
`ollama_probe_timeout_seconds=1.0`으로 짧게 유지).

**deprecated `/health`:** 기존 body(`status`, `rag_engine_initialized`)를
그대로 반환하되 응답 header에 `Deprecation: true`,
`Link: <.../docs/operations/Runbook.md#health>; rel="deprecation"` 추가.

### 6.3 오류 코드 taxonomy (`src/simple_qna_rag/errors.py`)

```python
class ErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    QUESTION_TOO_LONG = "question_too_long"
    BODY_TOO_LARGE = "body_too_large"
    NOT_READY = "not_ready"
    OVERLOADED = "overloaded"
    TIMEOUT = "timeout"
    UPSTREAM_OLLAMA_ERROR = "upstream_ollama_error"
    UPSTREAM_WEB_SEARCH_ERROR = "upstream_web_search_error"
    INDEX_UNAVAILABLE = "index_unavailable"
    INTERNAL_ERROR = "internal_error"
```

실패 응답 schema(`web/schemas.py::ErrorResponse`, **새 HTTP 4xx/5xx 실패에만
적용** — REQ-007.2):

```json
{"error": {"code": "overloaded", "message": "요청이 많아 잠시 후 다시 시도하세요.",
           "request_id": "…", "retryable": true}}
```

코드 -> HTTP status/retryable 매핑:

| code | status | retryable | Retry-After |
|---|---|---|---|
| invalid_request | 400 | false | — |
| question_too_long | 400 | false | — |
| body_too_large | 413 | false | — |
| not_ready | 503 | true | 2 |
| overloaded | 503 | true | 2 |
| timeout | 503 | true | 5 |
| upstream_ollama_error | 502 | true | — |
| upstream_web_search_error | 502 | true | — |
| index_unavailable | 503 | true | 5 |
| internal_error | 500 | false | — |

**호환성 경계:** `route_query()`가 정상적으로 실행되어 반환한 dict(
`answer/sources/success/search_type/intent`, `success=False` 포함)는 지금과
동일하게 HTTP 200으로 그대로 반환한다 — 이 계약은 절대 바꾸지 않는다. 위
`ErrorResponse`는 `route_query()` 실행 자체에 도달하지 못한 경우(입력 검증
실패, not-ready, overload, timeout, 5xx)에만 쓴다.

### 6.4 bounded concurrency — `QueryExecutor` (`web/concurrency.py`)

**M-01 전면 재설계 — 설계 목표(Review M-01에 대한 직접 대응):** (1) FIFO를 `asyncio.Condition`의
비공개 wait-queue 순서에 의존하지 않고 명시적 `deque` ticket으로 보장한다,
(2) timeout을 queue 진입 시점부터의 **absolute deadline**(`asyncio.timeout_at`)으로
측정해 "제출 시점부터 대기+실행 합산"이라는 계약(REQ-006.2)과 실제 구현을
일치시킨다, (3) slot 반환을 `released: bool` 플래그로 lock 아래 정확히 한
함수에서 한 번만 수행해 이중 반환/음수 `_running`을 구조적으로 막는다,
(4) cancel과 timeout을 완전히 동일한 "abandoned" 전이로 취급해 `_orphaned`
증감이 항상 짝을 이루게 한다.

**M2-01 추가 수정(Iteration 2 대응) — 위 (3)이 실제로 닫지 못했던 두 누수
지점을 없앤다:** (a) 제품 함수(`fn`)가 예외를 던지는 정상적인 production
실패 경로에서 `await asyncio.shield(future)`가 그 예외를 그대로 전파할 때
`except TimeoutError`/`except CancelledError` 어느 분기에도 걸리지 않고
`else`의 release 호출도 실행되지 않아 slot이 영구 누수됐다. (b) future가
정상 완료된 뒤 `else` 블록의 `await self._transition_done(...)`이 lock을
얻는 도중 caller가 취소되면, 그 취소는 `try` 블록만 감싸는 앞선
`except CancelledError`가 잡지 못해 같은 방식으로 누수됐다. 해결책은
release를 **`run()`의 제어 흐름과 완전히 분리해 future 자체의 완료
콜백에만 매인 단일 `_finalize()` 함수**로 옮기는 것이다 — `run()`은
future 생성 직후(어떤 await도 하기 전에) 콜백을 등록하기만 하고, 이후
자신이 무엇을 반환/전파하는지와 무관하게 `_finalize()`가 worker 종료
시점에 정확히 한 번 호출되어 slot을 반환한다. 동시에 `ticket.deadline`을
**항상 `loop.time()` 기준으로만** 계산해(테스트용 fake clock 파라미터
제거) `asyncio.timeout_at(ticket.deadline)`을 QUEUED/RUNNING 두 단계 모두
동일하게 사용한다 — deadline 계산과 만료 판정이 서로 다른 시계를 참조하던
Iteration 1 잔여 버그를 제거한다. 결정론적 테스트는 `timeout_cm_factory`
자체를 barrier 기반 fake로 주입해 real sleep 없이 만료 시점을 강제한다.

```python
import asyncio, collections, functools, itertools, logging
from dataclasses import dataclass, field
from enum import Enum

from .net_budget import DeadlineBudget   # §6.6b — 단일 monotonic deadline 계약(M2-02)

_stderr_log = logging.getLogger("simple_qna_rag.concurrency")  # 순수 stdlib, 이벤트 루프 불필요

class Overloaded(Exception):
    def __init__(self, running: int, waiting: int):
        self.running, self.waiting = running, waiting

class QueryTimeoutError(Exception):
    def __init__(self, phase: str):  # "queued" | "executing"
        self.phase = phase

class NotReadyError(Exception):
    """shutdown drain 중 begin_drain()이 대기 중인 ticket을 깨울 때 던진다(§6.1a)."""

class _TicketState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    ABANDONED = "abandoned"   # RUNNING 중 timeout/cancel — thread는 계속 실행 중, 아직 release 전(M2-01)
    DONE = "done"

@dataclass
class _Ticket:
    seq: int                                   # 명시적 FIFO 순서
    deadline: float                            # loop.time() 기준 absolute deadline(M2-01: 항상 loop clock)
    event: asyncio.Event = field(default_factory=asyncio.Event)  # slot 부여/거절 시 set
    state: _TicketState = _TicketState.QUEUED
    released: bool = False                     # 정확히 한 번만 True로 전이(exactly-once release)
    rejected: bool = False                     # reject_queued()/begin_drain()이 표시

class QueryExecutor:
    """상태 머신: 모든 ticket은 QUEUED -> RUNNING -> DONE(정상/예외 무관) 또는
    QUEUED -> (거절, ticket 생성 안 됨) 또는 QUEUED -> RUNNING -> ABANDONED -> DONE
    (timeout/cancel) 중 정확히 하나의 경로를 거친다. `_running`은 ticket이
    RUNNING 또는 ABANDONED인 동안(실제 thread pool worker를 점유하는 동안) 항상
    1을 유지하고, DONE 전이에서만 -1 한다 — "타임아웃 이후에도 실행 중인 작업이
    slot을 정직하게 점유한다"는 REQ-006.3 요구사항을 그대로 코드 구조로
    표현한다.

    **release는 worker의 결과(성공/예외/취소)와 완전히 독립적으로 정확히 한 번
    일어난다(M2-01):** `run()` 코루틴의 제어 흐름이 아니라 `future`
    (`loop.run_in_executor` 반환값)에 등록한 완료 콜백만이 release 경로다
    (`_finalize`). `run()`은 caller에게 무엇을 반환/전파할지만 결정하며 slot을
    직접 반환하지 않는다.
    """
    def __init__(self, concurrency_limit: int, queue_limit: int,
                 timeout_seconds: float, thread_pool: ThreadPoolExecutor,
                 timeout_cm_factory=asyncio.timeout_at):
        self._concurrency_limit = concurrency_limit
        self._queue_limit = queue_limit
        self._timeout_seconds = timeout_seconds
        self._pool = thread_pool          # max_workers=concurrency_limit
        # 테스트 주입용 — production 기본값은 항상 loop clock을 쓰는
        # asyncio.timeout_at. 별도 fake clock 파라미터는 두지 않는다: ticket.deadline과
        # 실제 대기 primitive가 서로 다른 시계를 참조하면 deadline 계산과 만료
        # 판정이 어긋난다(Iteration 1 잔여 버그, M2-01). 결정론적 테스트는 이
        # factory 자체를 barrier 기반 fake로 교체해 real sleep 없이 만료 시점을
        # 강제한다.
        self._timeout_cm_factory = timeout_cm_factory
        self._deque: "collections.deque[_Ticket]" = collections.deque()
        self._running = 0
        self._orphaned = 0                # ABANDONED 상태 ticket 수(§5.4 qna_rag_executor_orphaned)
        self._seq = itertools.count()
        self._lock = asyncio.Lock()       # _deque/_running/_orphaned/ticket.state/released 전체를 보호
        self._loop: asyncio.AbstractEventLoop | None = None
        self._drain_complete: asyncio.Event | None = None   # lifespan이 주입(§6.1a)
        self._draining = False

    async def run(self, fn, *args):
        loop = asyncio.get_running_loop()
        self._loop = self._loop or loop
        ticket = _Ticket(seq=next(self._seq), deadline=loop.time() + self._timeout_seconds)

        async with self._lock:
            if self._draining:
                raise NotReadyError()
            if self._running >= self._concurrency_limit:
                if len(self._deque) >= self._queue_limit:
                    raise Overloaded(self._running, len(self._deque))
                self._deque.append(ticket)             # FIFO: 항상 오른쪽에 추가, 왼쪽에서 pop
            else:
                self._grant_locked(ticket)              # 즉시 실행 (아래 §wake_next와 동일 전이)

        if ticket.state is _TicketState.QUEUED:
            try:
                async with self._timeout_cm_factory(ticket.deadline):
                    await ticket.event.wait()
            except TimeoutError:
                never_granted = await self._pop_from_queue_or_none(ticket)
                if never_granted:
                    raise QueryTimeoutError(phase="queued")
                # 경쟁: deadline 만료와 거의 동시에 _wake_next_locked()가 이미 이
                # ticket에 slot을 부여했다(_pop_from_queue_or_none()이 False를
                # 반환) — 이중 처리하지 않고 정상 RUNNING으로 이어간다(아래로
                # fall-through, 타임아웃-완료 경쟁의 핵심 케이스, 결정론적
                # 테스트 참조). 이 케이스는 M3-01 이전에도 올바르게
                # 처리됐었다 — 아래 CancelledError 분기와 달리 future를 이미
                # 정상적으로 제출해야 하므로 여기서 finalize하지 않는다.
            except asyncio.CancelledError:
                never_granted = await self._pop_from_queue_or_none(ticket)
                if never_granted:
                    raise
                # M3-01 근거 1 — grant와 caller cancel이 정확히 경쟁한 케이스:
                # `_pop_from_queue_or_none()`이 False를 반환했다는 것은
                # `_wake_next_locked()`가 이미 이 ticket을 RUNNING으로 전이시켜
                # (`_running += 1`) slot을 부여했다는 뜻이다. caller는 결과를
                # 더 이상 원하지 않으므로(cancel) future를 제출하지 않고 지금
                # 바로 이 자리에서 release한다 — future/완료 콜백이 이 ticket에
                # 대해 존재한 적이 없으므로 이 호출이 유일한 소유자다(M3-01
                # ownership 계약). `asyncio.shield()`로 감싸 이 release 호출
                # 자체가 (반복적인) cancel로 다시 중단되어도 끝까지
                # 실행되도록 보장한다(M3-01 수정안 2, `_finalize`는 `released`
                # 가드가 있으므로 몇 번 취소 재시도가 들어와도 안전하게
                # 재진입 가능).
                await asyncio.shield(self._finalize(ticket))
                raise
            if ticket.rejected:
                raise NotReadyError()

        # 여기 도달하면 ticket.state == RUNNING (slot 보유 확정)이고 future도
        # ticket도 아직 아무도 finalize하지 않았다 — 이 지점부터 future 제출
        # 완료(콜백 등록)까지가 M3-01이 요구하는 "cancellation-free" 임계
        # 구간이다. ticket.deadline과 정확히 같은 값으로 DeadlineBudget을
        # 만들어 sync worker에 전달한다 — 동일한 하나의 loop.time() 값에서
        # 파생된 단일 deadline이 asyncio 대기(timeout_at)와 동기 upstream
        # 호출(§6.6b) 양쪽에 그대로 쓰인다(M2-02, 서로 다른 시계를 혼용하지
        # 않는다).
        budget = DeadlineBudget(deadline=ticket.deadline)
        callback_registered = False
        try:
            future = loop.run_in_executor(
                self._pool, functools.partial(fn, *args, budget=budget))
            # future 생성 직후, 어떤 await도 하기 전에 콜백을 등록한다 — 이
            # 두 줄 사이에는 await가 없으므로 caller cancellation이 끼어들
            # 지점 자체가 없다(M3-01 — "slot grant와 future submit/callback
            # registration 사이에 cancellation point가 없다"는 사실에 기대는
            # 대신, 아래 finally의 `callback_registered` ownership token으로
            # 이 구간에서 예외적으로 뭔가 실패하더라도 release가 보장되게
            # 만든다).
            future.add_done_callback(functools.partial(self._schedule_finalize, ticket))
            callback_registered = True
        finally:
            if not callback_registered:
                # run_in_executor 자체가 동기적으로 실패했거나(예: pool이 이미
                # shutdown) add_done_callback이 실패한 경우 — future의 완료
                # 콜백은 절대 호출되지 않으므로 "future callback 또는 caller
                # cleanup 중 정확히 하나가 책임을 인수한다"는 ownership 계약에
                # 따라 여기가 유일한 소유자다(M3-01 수정안 2~3). 이 finally
                # 블록 자체가 caller cancellation으로 중단되더라도 release가
                # 반드시 끝까지 실행되도록, 별도 task로 만들어 shield한다 —
                # `await self._finalize(ticket)`를 직접 쓰면 이 await 자체가
                # 취소될 수 있어(M3-01 근거 2) 여전히 release가 중간에
                # 끊어질 수 있었다.
                await asyncio.shield(self._finalize(ticket))

        try:
            async with self._timeout_cm_factory(ticket.deadline):
                return await asyncio.shield(future)
        except TimeoutError:
            await self._mark_abandoned(ticket)
            raise QueryTimeoutError(phase="executing")
        except asyncio.CancelledError:
            await self._mark_abandoned(ticket)
            raise
        # future가 던진 product 예외(제품 함수의 정상적인 실패 경로, 예:
        # route_query() 내부 예외)는 asyncio.shield(future)가 그대로 전파한다 —
        # 위 두 except 어디에도 걸리지 않고 caller까지 그대로 올라간다. 이
        # 경로에서도 release는 이미 등록된 future 콜백(`_finalize`)이 worker
        # 종료 시점에 수행하므로 여기서 별도 처리가 필요 없다(M2-01 근거 1).

    def _grant_locked(self, ticket: _Ticket) -> None:
        """호출자가 이미 self._lock을 보유한 상태에서만 호출. QUEUED/신규 ticket에
        slot을 부여하는 유일한 지점(신규 admit과 wake_next_locked이 공유)."""
        self._running += 1
        ticket.state = _TicketState.RUNNING
        ticket.event.set()

    def _wake_next_locked(self) -> None:
        if self._deque and self._running < self._concurrency_limit:
            self._grant_locked(self._deque.popleft())

    async def _pop_from_queue_or_none(self, ticket: _Ticket) -> bool:
        """QUEUED 대기가 timeout 또는 caller cancel로 끝나는 시점에 lock 아래
        정확히 한 번 호출하는 단일 진입점(M3-01 — 이전에는 timeout/cancel
        두 분기가 각자 다른 `async with self._lock:` 블록에서 `ticket in
        self._deque`를 따로 검사해, cancel 분기만 "이미 grant된" 경쟁을
        처리하지 못했다). True를 반환하면 이 ticket은 아직 grant되지 않아
        deque에서 방금 제거됐다는 뜻이고 — 어떤 slot도 점유한 적이 없다.
        False를 반환하면 `_wake_next_locked()`가 이미 이 ticket을 RUNNING으로
        전이시킨 뒤라는 뜻이다(deadline 만료/cancel과 grant의 경쟁). 이
        함수는 순수 관측/제거만 수행하고 release는 하지 않는다 — timeout과
        cancel 두 호출자가 "이미 grant된" 경우에 서로 다른 정책(timeout은
        정상 실행 계속, cancel은 즉시 release)을 적용해야 하므로 정책
        결정은 호출자(`run()`)에 남긴다."""
        async with self._lock:
            if ticket in self._deque:
                self._deque.remove(ticket)
                return True
            return False

    async def _mark_abandoned(self, ticket: _Ticket) -> None:
        """run()이 caller에게 timeout/cancel을 반환/전파하기로 결정한 시점에
        호출한다. **release는 하지 않는다** — 순수 관측(§5.4
        qna_rag_executor_orphaned) 목적이며, `_finalize`가 이미 먼저 실행돼
        `released=True`이면 아무것도 하지 않는다(worker가 거의 동시에 정상
        완료된 경쟁 상황, `test_timeout_completion_race` 참조)."""
        async with self._lock:
            if ticket.released:
                return
            if ticket.state is _TicketState.RUNNING:
                ticket.state = _TicketState.ABANDONED
                self._orphaned += 1

    def _schedule_finalize(self, ticket: _Ticket, future: "asyncio.Future") -> None:
        """future의 완료 콜백 — worker의 성공/예외/취소 여부와 무관하게 future가
        끝나면 `asyncio.Future` 계약상 항상 정확히 한 번 호출된다.

        **M3-01 수정 — `call_soon_threadsafe`의 불필요한 한 턴 지연을
        제거한다.** `loop.run_in_executor()`가 반환하는 `future`는
        `concurrent.futures.Future`가 아니라 **`asyncio.Future`**다 — 표준
        라이브러리가 내부적으로 `asyncio.futures.wrap_future()`로 감싸면서
        worker thread의 완료를 `loop.call_soon_threadsafe()`로 이미 한 번
        건너와 이 asyncio.Future의 상태를 채우고, 그 상태 변경이 등록된
        `add_done_callback` 콜백들을 `loop.call_soon()`으로 스케줄한다
        (`asyncio.Future`의 문서화된 계약). 즉 **이 함수 자체가 호출되는
        시점은 이미 loop thread 위**다 — 이전 설계가 여기서 다시
        `loop.call_soon_threadsafe(_resume)`로 한 턴을 더 미룬 것은
        thread-safety 이득이 전혀 없이 지연만 추가했고, 그 지연 구간에 loop가
        멈추면(§6.1a shutdown) 예외도 기록되지 못하고 finalize도 실행되지
        않는 원인이었다(Review M3-01 근거 3, 원문 인용: "완료 callback은
        asyncio.Future.add_done_callback()에 의해 이미 loop thread에서
        실행되는데 다시 call_soon_threadsafe(_resume)로 한 turn 뒤로
        미룬다"). 이 함수는 동기 콜백이라 직접 `await`할 수 없으므로
        `_finalize` 코루틴을 task로 만들어 스케줄하는 것은 여전히
        필요하지만, 그 스케줄 자체는 지금 바로(추가 hop 없이) 수행한다."""
        try:
            asyncio.ensure_future(self._finalize(ticket))
        except RuntimeError:
            # loop가 이미 닫힘(§6.1a shutdown 이후) — best-effort 로그만 남기고
            # 프로세스 종료를 막지 않는다. 이 함수가 호출된다는 것 자체가
            # 이 순간 loop가 최소 한 번은 더 콜백을 처리할 수 있는 상태였다는
            # 뜻이므로(위 설명), 이 예외는 스케줄된 새 task가 실제로 실행되기
            # *전에* loop가 완전히 close()된 극히 드문 경계 케이스만
            # 포착한다 — `test_finalize_callback_after_loop_closed`(§6.1a)와
            # 신규 `callback_queued_then_loop_shutdown`(아래 결정론적 race
            # 테스트 표)이 이 두 경계를 각각 검증한다.
            _stderr_log.error("finalize callback: event loop already closed")

    async def _finalize(self, ticket: _Ticket) -> None:
        """모든 종료 경로(worker 성공, worker 예외, future 콜백을 통한 취소/timeout
        확정, run_in_executor 자체 실패)가 수렴하는 **유일한** release 지점
        (M2-01 수정안 1~2). `released` 가드가 lock 아래 첫 줄에 있으므로 몇 번
        호출돼도 실제 상태 전이와 카운터 변경은 정확히 한 번만 일어난다."""
        async with self._lock:
            if ticket.released:
                return
            ticket.released = True
            if ticket.state is _TicketState.ABANDONED:
                self._orphaned -= 1
            ticket.state = _TicketState.DONE
            self._running -= 1
            self._wake_next_locked()
            self._check_drain_locked()

    def _check_drain_locked(self) -> None:
        if self._draining and self._running == 0 and self._drain_complete is not None:
            self._drain_complete.set()

    def reject_queued(self, error_cls) -> None:
        """대기 중(QUEUED) ticket만 즉시 거절한다. shutdown 절차 전체는
        `begin_drain()`(§6.1a, m2-01)을 사용한다 — 이 메서드는 그 내부
        구현으로만 남는다."""
        self._draining = True
        while self._deque:
            ticket = self._deque.popleft()
            ticket.rejected = True
            ticket.event.set()

    def begin_drain(self) -> None:
        """§6.1a shutdown 1단계 — draining 진입, QUEUED 즉시 거절, 이미
        idle(`_running == 0`)이면 `drain_complete`도 즉시 set한다(m2-01 —
        이전에는 이 즉시-set 경로가 없어 idle 상태에서도 매번
        `shutdown_grace_seconds` 전체를 기다렸다). 동기 함수, `reject_queued()`와
        동일하게 이벤트 루프 스레드 안에서만 호출된다는 계약으로 보호한다(단일
        event loop, 재진입 없음 — await 지점이 없으므로 실행 도중 다른
        코루틴이 끼어들 수 없다)."""
        self.reject_queued(NotReadyError)
        self._check_drain_locked()

    def shutdown_pool(self, *, wait: bool = False, cancel_futures: bool = True) -> None:
        """public accessor — lifespan이 존재하지 않는 `query_executor.pool`
        속성에 직접 접근해 `AttributeError`를 유발하던 문제를 제거한다(m2-01).
        내부 `ThreadPoolExecutor.shutdown(...)`을 그대로 감싼다."""
        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)

    def bind_drain_event(self, event: asyncio.Event) -> None:
        self._drain_complete = event
```

**exactly-once release 증명 스케치(M3-01 개정 — ownership token):** release는
오직 `_finalize()` 한 함수에서만 `ticket.released`를 `True`로 설정한다.
`_finalize`를 호출하는 경로는 정확히 **셋**뿐이다 — (1) QUEUED 대기가
caller cancel로 끝났는데 그 순간 이미 `_wake_next_locked()`가 grant해 버린
경쟁(`run()`의 `CancelledError` 분기, `asyncio.shield()`로 감쌈, M3-01
신규), (2) `run_in_executor`/`add_done_callback` 자체의 동기 실패 직후
`callback_registered` 플래그가 여전히 False인 채로 `run()`의 `finally`
블록이 직접 호출(`asyncio.shield()`로 감쌈, M3-01 개정 — 이전에는 이
`await`가 shield 없이 노출돼 있어 근거 2의 누수를 재현할 수 있었다), (3)
future의 완료 콜백(`_schedule_finalize` -> `_finalize`, worker 성공/예외/
`future.cancel()` 모두 이 콜백 한 곳으로 수렴). 이 세 경로는 서로 배타적이다
— 어떤 ticket이든 (1)이 실행됐다면 future 자체가 만들어지지 않으므로 (3)의
콜백이 등록될 수 없고, (2)가 실행됐다면 `callback_registered=False`였다는
뜻이므로 마찬가지로 (3)이 등록되지 않는다. `run()`의 timeout/cancel
처리(`_mark_abandoned`)는 관측용 상태 전이만 수행하며 `released`를 절대
건드리지 않는다.

**ownership token은 두 개의 명시적 플래그로 구현된다** — `_pop_from_queue_or_none()`이
반환하는 `never_granted`(QUEUED 단계 소유권 판정)와 `run()` 지역 변수
`callback_registered`(RUNNING 단계 소유권 판정). 두 플래그 모두 **lock 아래
읽고 쓰는 갱신과, 그 갱신 직후 곧바로 이어지는 분기 사이에 추가
await 지점이 없다** — 즉 "플래그를 확인했다"와 "그 확인에 따라 정확히 한
경로만 release를 수행한다" 사이에 다른 코루틴이 끼어들어 플래그의 의미를
무효화할 시간이 없다. 세 호출 경로 모두 `_finalize` 첫 줄에서 lock 아래
`if ticket.released: return`을 추가로 실행하므로(방어적 이중 잠금 —
ownership token이 논리적으로 정확히 하나만 release를 호출하도록 보장하지만,
`released` 가드는 그 논리에 버그가 있더라도 실제 카운터 훼손을 막는 마지막
방어선이다), 어떤 순서로 몇 번 호출돼도 실제 상태 전이(`_running -= 1`,
`_orphaned` 조정, `_wake_next_locked()`, `_check_drain_locked()`)는 정확히
한 번만 일어난다. `run()`의 제어 흐름(정상 반환, product 예외 전파, timeout,
cancel)은 이 보장에 전혀 관여하지 않으므로 Iteration 1/2/3에서 발견된 누수
지점(worker 예외, 완료 직후 cancel, grant-cancel 경쟁, submit 실패 경로의
비-shield await)이 구조적으로 재발할 수 없다.

**`begin_drain()`/`reject_queued()`의 lock 우회에 대한 명시적 불변식(M3-01
— Review 근거 4에 대한 답):** 위 세 release 경로와 `run()`의 admit
critical section은 모두 `async with self._lock:`을 쓰지만,
`begin_drain()`/`reject_queued()`는 동기 함수로서 lock을 얻지 않는다. 이
둘은 서로 다른 종류의 안전성에 기대므로 혼동하지 않아야 한다 — `async with
self._lock:` 블록들이 필요한 이유는 "여러 코루틴에 걸쳐 시간적으로 떨어진
여러 critical section이 서로의 중간 상태를 관찰하지 못하게" 막기 위함이지,
개별 critical section 내부에 스레드 동시성이 있어서가 아니다(단일 event
loop thread이므로 GIL 경쟁 자체가 없다). `begin_drain()`은 (a) 자기 자신의
실행 범위 안에 `await`가 전혀 없고(동기 함수), (b) lifespan shutdown
코루틴이 다른 어떤 await도 거치지 않고 직접 호출하므로, 이 함수가 실행되는
전체 구간 동안 Python 인터프리터는 다른 코루틴/콜백으로 전혀 전환할 수
없다 — 이는 정의상 "그 구간 전체를 하나의 원자적 critical section으로
간주"하는 것과 동치이며, `self._lock`을 uncontended 상태에서 얻어 쥐고 있는
것과 관찰 가능한 차이가 없다. 따라서 이 두 함수는 "모든 상태 전이는 lock
아래에서 일어난다"는 불변식의 **예외가 아니라 그 불변식이 성립하는 특수
사례**로 명시한다 — 이 성질은 `_deque`/`_running`/`ticket.state`를
건드리는 새 동기 헬퍼를 추가할 때마다(이번 개정의 `_pop_from_queue_or_none`
제외 — 이 함수는 `async with self._lock:`을 명시적으로 쓴다) 반드시 같은
방식으로 재검증해야 하는 설계 규칙으로 남긴다.

**state table:**

| 상태 | ticket.state | `_running` 변화 | 신규 요청 결과 |
|---|---|---|---|
| idle | — | 0 -> 1(즉시 admit) | 즉시 실행 |
| 여유 | — | <limit -> +1 | 즉시 실행 |
| 포화, 대기 여유 | QUEUED | 불변 | FIFO deque에 추가, absolute deadline까지 대기 |
| 포화, 대기 포화 | — | 불변 | `Overloaded` -> 503 + `Retry-After: 2`(ticket 생성 안 됨) |
| QUEUED 중 deadline 만료 | QUEUED->(제거) | 불변 | `QueryTimeoutError(phase="queued")` |
| QUEUED 중 만료와 grant 경쟁 | QUEUED->RUNNING | +1(정상 admit 경로) | 정상 실행으로 이어짐(타임아웃 아님) |
| QUEUED 중 caller cancel | QUEUED->(제거) | 불변 | `CancelledError` 전파, 어떤 slot도 점유하지 않았음 |
| QUEUED 중 caller cancel과 grant 경쟁(M3-01 신규) | QUEUED->RUNNING->DONE(`_finalize`, `run()`이 shield로 직접 호출) | +1 즉시, -1 즉시(같은 `run()` 호출 안에서 동기적으로 이어짐) | `CancelledError` 전파, future/콜백은 생성된 적 없음, slot은 caller가 알기도 전에 이미 반환됨 |
| RUNNING 중 정상 완료 | RUNNING->DONE(`_finalize`) | -1 | 결과 반환 |
| RUNNING 중 worker 예외 | RUNNING->DONE(`_finalize`) | -1 | 예외가 caller까지 그대로 전파(M2-01 근거 1) |
| RUNNING 중 정상 완료 직후 caller cancel | RUNNING->DONE(`_finalize`, cancel과 무관하게 진행) | -1 | `CancelledError`가 caller에 전파되더라도 slot은 정상 반환(M2-01 근거 2) |
| RUNNING 중 실행 timeout | RUNNING->ABANDONED(`_mark_abandoned`, 즉시)->DONE(`_finalize`, 지연) | 즉시 불변, thread 종료 시 -1 | `QueryTimeoutError(phase="executing")`, `_orphaned+1`(즉시), thread 종료 시 `_orphaned-1` |
| RUNNING 중 caller cancel | RUNNING->ABANDONED(즉시)->DONE(지연) | 즉시 불변, thread 종료 시 -1 | `CancelledError` 전파, `_orphaned+1`(즉시), thread 종료 시 `_orphaned-1` |
| run_in_executor 자체 실패 | RUNNING->DONE(`run()`이 `_finalize` 직접 호출) | -1(즉시) | 원 예외 전파(M2-01 수정안 3) |
| draining | — | — | `begin_drain()`이 QUEUED를 즉시 `NotReadyError`로 거절, idle이면 `drain_complete` 즉시 set(§6.1a); 신규 제출은 §6.1의 lifecycle 체크가 먼저 막음 |

`timeout_seconds`는 **제출 시점부터**(대기+실행 합산) 측정한다(REQ-006.2
"query 전체 timeout") — `ticket.deadline`이 ticket 생성 시 한 번만 계산되고
QUEUED/RUNNING 두 단계 모두 동일한 absolute deadline을 공유하므로 원 설계의
"wait() 이후에야 wait_for가 시작"되던 이중 시계 버그가 구조적으로 사라진다.
`ticket.deadline`은 항상 `loop.time()` 기준으로만 계산하고(별도 주입 가능한
fake clock 파라미터는 두지 않는다) QUEUED/RUNNING 두 단계 모두 같은
`asyncio.timeout_at(ticket.deadline)` 호출로 대기하므로, deadline 계산에
쓰인 시계와 실제 만료를 판정하는 시계가 항상 하나로 일치한다(M2-01 —
이전에는 fake clock이 `ticket.deadline` 계산에만 쓰이고
`asyncio.timeout(remaining)`은 실제 event-loop clock을 썼다). 결정론적
테스트는 시계를 흉내 내는 대신 `timeout_cm_factory` 자체를 barrier 기반
구현으로 교체해 만료 시점을 강제한다. 타임아웃된 스레드는 강제 종료할 수
없다는 한계를 `docs/operations/Runbook.md#incident-triage`와 코드 docstring에
명시하고, `_orphaned` 값을 `qna_rag_executor_orphaned`(Gauge, §5.4)로
노출한다. executor 포화가 orphan 누적으로 지속되는 경우, readiness는 별도
reason code를 추가하지 않고(REQ-005.4 "무거운 query 실행 없이" 판정) 여전히
`ready`를 반환하되 개별 `/rag` 요청이 `Overloaded`(503)로 계속 거절되는
것으로 과부하를 표현한다 — liveness/readiness는 순간 부하와 독립적으로
안정적이어야 한다는 REQ-005.1/.2와 일치하는 의도적 선택이다.

**결정론적 race 테스트(신규 `tests/integration/test_web_concurrency.py`,
M-01 "5. barrier로 강제하는 결정론적 테스트" 및 M2-01 수정안 4 대응):**

| 테스트 | 강제 방법 | 검증 |
|---|---|---|
| `test_timeout_completion_race` | `timeout_cm_factory`를 barrier 기반 fake로 주입해 만료 시점을 제어하고, worker 완료를 `threading.Event`로 정확히 그 시점에 맞춰 release — real sleep에 의존하지 않음 | `_running`이 음수가 되지 않고, `_finalize`가 정확히 1회만 `released=True`로 전이(카운터로 assert), `_mark_abandoned`와의 실행 순서와 무관 |
| `test_worker_exception_releases_slot` | 동기 `fn`이 `ValueError`를 raise(제품 `route_query()`가 예외를 던지는 실제 production 실패 경로 재현, M2-01 근거 1) | 예외가 caller까지 그대로 전파되고, future 완료 콜백(`_finalize`)이 `_running`을 정확히 1 감소시키며 다음 FIFO ticket이 깨어남 |
| `test_cancel_immediately_after_normal_completion` | future를 정상 완료(`threading.Event`)시킨 직후, `_finalize`가 lock을 얻기 전 시점에 `run()` Task를 `task.cancel()`(barrier로 순서 고정, M2-01 근거 2) | 취소가 caller에게 `CancelledError`로 전파되더라도 slot은 `_finalize`가 정상 반환(누수 없음, `released` 정확히 1회 전이) |
| `test_repeated_cancel_during_finalize_lock_wait` | `_finalize`가 lock을 잡고 있는 동안(barrier로 지연) 같은 ticket에 대해 여러 차례 `task.cancel()`을 반복 전송 | 반복 취소가 `_finalize`의 단일 실행이나 `released` 가드를 깨지 않고, 카운터가 정확히 한 번만 변경됨 |
| `test_run_in_executor_failure_releases_slot` | `thread_pool.submit`(`loop.run_in_executor`가 호출)이 동기적으로 `RuntimeError`를 raise하도록 monkeypatch(pool shutdown과의 경합 재현, M2-01 수정안 3) | 콜백이 등록되지 않았음에도 `run()`이 직접 `_finalize`를 호출해 slot이 즉시 반환됨, 원 예외가 caller에 전파 |
| `test_queue_timeout_grant_race` | deadline 직전에 `_wake_next_locked()`가 먼저 실행되도록 두 코루틴을 `asyncio.Event`로 순서 고정 | ticket이 QUEUED 제거가 아니라 RUNNING으로 이어지고 정상 결과 반환(타임아웃 아님) |
| `test_queue_cancel` | 대기 중인 `run()` Task를 `task.cancel()` | ticket이 deque에서 제거되고 `_running`/`_orphaned` 불변, `_finalize`가 호출되지 않음(슬롯을 점유한 적이 없으므로) |
| `test_executing_cancel` | 실행 중(RUNNING) `run()` Task를 `task.cancel()`, 내부 fn은 `threading.Event`로 계속 실행 | `CancelledError` 전파, `_mark_abandoned`로 `_orphaned+1` 즉시 반영, thread 완료 시 future 콜백이 `_finalize`를 호출해 `released=True`/`_orphaned-1`/`_running-1` 정확히 1회 |
| `test_finalize_callback_after_loop_closed` | `_loop`를 닫은 뒤 `threading.Barrier(2)`로 붙잡아 둔 worker를 완료시켜 future 완료 콜백(`_schedule_finalize`) 유발 | `RuntimeError`가 잡혀 `_stderr_log.error(...)` 1회 호출, 프로세스/테스트 크래시 없음 |
| `queue_grant_then_cancel_before_submit`(M3-01 신규, 위 state table 신규 행과 동일 시나리오) | `asyncio.Event` 두 개로 순서를 고정 — ①`_wake_next_locked()`가 QUEUED ticket을 RUNNING으로 전이(`ticket.event.set()`)한 직후, ②`run()`이 `ticket.event.wait()`에서 깨어나기 *전에* 바깥에서 `task.cancel()`을 전송해 `wait()` 자체가 `CancelledError`로 끝나도록 강제 | `_pop_from_queue_or_none()`이 False를 반환(ticket이 이미 deque 밖)하고, `run()`이 future를 전혀 생성하지 않은 채 `asyncio.shield(self._finalize(ticket))`을 호출해 `_running`이 즉시 -1(순누적 0), `released=True`가 정확히 1회, `_orphaned` 불변(RUNNING을 거쳤지만 ABANDONED로 전이한 적은 없으므로) |
| `cancel_during_submit_failure_finalize`(M3-01 신규) | `thread_pool.submit`이 동기적으로 예외를 던지도록 monkeypatch(`test_run_in_executor_failure_releases_slot`과 동일 강제)한 상태에서, `run()`의 `finally` 블록이 `asyncio.shield(self._finalize(ticket))`을 await하는 도중 같은 Task에 `task.cancel()`을 반복 전송(barrier로 정확한 타이밍 고정) | `CancelledError`가 `finally` 블록을 빠져나가는 `run()` 자신에게는 전파되더라도, `asyncio.shield()`로 감싼 내부 `_finalize` task는 취소되지 않고 끝까지 실행돼 `released=True`/`_running-1`이 정확히 1회 관측됨(shield가 없던 이전 설계라면 이 취소가 release 자체를 끊었을 것) |
| `callback_queued_then_loop_shutdown`(M3-01 신규, Review 근거 3 직접 검증) | 정상적으로 future가 완료돼 `_schedule_finalize`가 `asyncio.ensure_future(self._finalize(ticket))`를 호출한 *직후*, 그 새 task가 첫 실행 기회를 얻기 *전에* 이벤트 루프를 닫도록 스케줄 순서를 barrier로 고정 | 이전 설계(`call_soon_threadsafe`로 한 턴 더 미루던 버전)라면 이 타이밍에서 `_finalize`가 영영 실행되지 않았을 것 — 개정된 설계는 `_schedule_finalize`가 이미 loop thread 위에서 실행되므로 `asyncio.ensure_future()` 호출 자체는 항상 성공하고, 이후 실제 task 실행이 지연되는 잔여 위험은 §6.1a에 이미 문서화된 "bounded 프로세스 종료가 아니라는 한계"로 흡수됨을 확인(테스트는 `RuntimeError`가 발생하지 않고 `ensure_future`가 정상 호출됨을 assert) |
| `test_fifo_order_explicit` | queue_limit 내에서 5개 ticket을 임의 순서로 `run()` 호출 후 slot을 하나씩 해제 | 실행 순서가 정확히 제출 순서(`seq`)와 동일함을 assert — `asyncio.Condition`의 비공개 순서가 아니라 `deque`가 순서를 만든다는 것을 직접 검증 |

**`web/server.py` 통합:**

```python
@app.post("/rag", response_model=QueryResponse, responses={503: {"model": ErrorResponse}})
async def rag_query(request: QueryRequest, http_request: Request):
    if app.state.lifecycle is not LifecycleState.READY:
        return error_response(ErrorCode.NOT_READY, request_id, 503)
    try:
        result = await app.state.query_executor.run(route_query, request.question)
    except Overloaded:
        return error_response(ErrorCode.OVERLOADED, request_id, 503, retry_after=2)
    except QueryTimeoutError:
        return error_response(ErrorCode.TIMEOUT, request_id, 503, retry_after=5)
    except NotReadyError:
        return error_response(ErrorCode.NOT_READY, request_id, 503, retry_after=2)
    return QueryResponse(**result)
```

health/static/metrics 라우트는 `QueryExecutor`를 전혀 거치지 않으므로 query
포화 중에도 독립적으로 응답한다(REQ-006.5).

### 6.5 singleton thread-safety

`rag_engine.get_rag_engine()`에 모듈 레벨 `threading.Lock()`을 도입해
double-checked locking으로 감싼다:

```python
_init_lock = threading.Lock()

def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is not None:
        return _rag_engine
    with _init_lock:
        if _rag_engine is None:
            engine = RAGEngine()
            if not engine.initialize():
                raise RuntimeError("RAG 엔진 초기화 실패")
            _rag_engine = engine
    return _rag_engine
```

lifespan startup이 이 함수를 await 전에 한 번 완료시키므로 런타임 질의
경로는 실제로 경합하지 않지만, 방어적으로 lock을 유지한다(REQ-006.4).
`ThreadPoolExecutor(max_workers=concurrency_limit)` 크기 자체가 모델 호출의
동시성 상한이 된다. Phase 4에서 `CrossEncoder.predict()`/
`HuggingFaceEmbeddings.embed_*()`를 `concurrency_limit=2`로 동시 실행해
RSS 급증이나 예외가 관측되면, 해당 호출만 `threading.Lock()`으로 직렬화하는
옵션을 코드에 추가한다(이벤트 루프는 막지 않음 — 이미 thread executor
내부이므로 lock 대기가 이벤트 루프에 영향 없음). Phase 3 시점에는 이 lock을
추가하지 않고 measurement-first로 남긴다.

### 6.6 입력·네트워크 경계

- 질문: `question.strip()` 길이 1자 이상, UTF-8 인코딩 바이트 수
  `<= question_max_bytes`(4000). 위반 시 `ErrorCode.QUESTION_TOO_LONG`
  (0자는 `INVALID_REQUEST`).
- trusted host: `starlette.middleware.trustedhost.TrustedHostMiddleware(
  allowed_hosts=settings.trusted_hosts)`.
- CORS: `settings.cors_allow_origins`가 비어 있으면 `CORSMiddleware`를
  아예 추가하지 않는다(현재와 동일 — CORS 없음이 기본). 설정 시에만 추가.

#### 6.6a body 크기 제한 — raw ASGI `receive` wrapper (m-02 대응)

`request.stream()`을 FastAPI 라우트 안에서 래핑하면 이미 Starlette/Pydantic이
body 파싱을 시작한 뒤라 너무 늦다. `BodySizeLimitMiddleware`는 **순수 ASGI
미들웨어**(Starlette `BaseHTTPMiddleware` 상속이 아님 — 그것도 내부적으로
`receive`를 소비/버퍼링해 동일한 문제를 재현한다)로 구현해 `receive` 자체를
감싼다. 이 미들웨어는 §5.2 `RequestContextMiddleware`보다 **바깥**에 있으므로
request context가 아직 만들어지지 않은 채로 413을 보내야 한다 — 그래도
M4-REQ-007.2("실패 응답에도 request ID")를 만족하도록, 그리고 하위 앱이
이미 응답을 시작한 뒤에는 두 번째 `http.response.start`를 보내지 않도록
아래 두 가지를 명시적으로 처리한다(m2-02):

```python
# src/simple_qna_rag/observability/request_id.py (신규, Phase 2 — §5.2와 공유)
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

def resolve_request_id(headers: "Iterable[tuple[bytes, bytes]]") -> str:
    """`X-Request-ID` inbound header를 검증해 재사용하거나 새로 만든다 — §5.2
    `RequestContextMiddleware`와 `BodySizeLimitMiddleware`(m2-02)가 **같은
    함수**를 호출하므로, 413으로 조기 종료되는 요청과 정상 처리되는 요청이
    항상 동일한 규칙으로 request ID를 얻는다(한쪽만 다른 로직을 쓰다가
    어긋나는 것을 구조적으로 방지)."""
    for k, v in headers:
        if k == b"x-request-id":
            candidate = v.decode("latin-1")
            if _REQUEST_ID_RE.match(candidate):
                return candidate
            break
    return uuid.uuid4().hex

class _BodyTooLarge(Exception):
    pass

class BodySizeLimitMiddleware:
    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        request_id = resolve_request_id(scope.get("headers", ()))
        response_started = False

        async def tracking_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        async def send_413():
            if response_started:
                # 하위 앱이 이미 응답을 시작한 뒤라 두 번째 http.response.start를
                # 보내는 것은 ASGI 계약 위반이다 — 새 응답을 만들지 않고 그냥
                # 반환한다(서버가 연결을 정리, m2-02 근거: 이전 설계는 "일반적
                # 처리 순서"에만 의존했다).
                return
            body = ErrorResponse(code=ErrorCode.BODY_TOO_LARGE,
                                  request_id=request_id).to_json_bytes()
            await tracking_send({"type": "http.response.start", "status": 413,
                                  "headers": [(b"content-type", b"application/json"),
                                              (b"x-request-id", request_id.encode())]})
            await tracking_send({"type": "http.response.body", "body": body})

        length_headers = [v for (k, v) in scope.get("headers", ()) if k == b"content-length"]
        if len(length_headers) > 1:
            return await send_413()                     # 중복 Content-Length
        if length_headers:
            try:
                declared = int(length_headers[0])
            except ValueError:
                return await send_413()                   # 비정수 Content-Length
            if declared < 0 or declared > self.max_bytes:
                return await send_413()                    # 음수 또는 header 자체가 초과

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.disconnect":
                return message                             # disconnect는 그대로 통과
            received += len(message.get("body", b""))
            if received > self.max_bytes:
                raise _BodyTooLarge()                       # header보다 실제 body가 더 큰 경우
            return message

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            await send_413()
```

`send_413()`은 이제 `response_started` 플래그로 하위 앱이 이미
`http.response.start`를 보냈는지 **직접 추적**해 판단한다(m2-02 —
"body를 다 읽기 전에는 하위 앱이 응답을 시작하지 않는다"는 암묵적 순서
가정 대신, 모든 `send()` 호출을 `tracking_send`로 감싸 실제로 관측한다).
`resolve_request_id()`가 만든 request ID는 정상 경로에서도 §5.2
`RequestContextMiddleware`가 같은 함수로 다시 계산해 재사용하므로(같은
inbound header, 같은 검증 규칙) 413 응답과 이후 로그의 request ID가
어긋나지 않는다. `app.add_middleware`가 아니라 `app =
BodySizeLimitMiddleware(app, max_bytes=settings.request_max_bytes)`로
**가장 바깥(raw ASGI app 자체를 감싸는 방식)**에 적용해 FastAPI/Starlette
미들웨어 스택보다 먼저 실행되게 한다.

**결정론적 테스트(신규 `tests/integration/test_body_size_limit.py`, m2-02
수정안 대응):**

| 테스트 | 시나리오 | 기대 |
|---|---|---|
| `test_duplicate_content_length` | 동일 요청에 `Content-Length` header 2개 | 즉시 413, body 미접근, 응답 body의 `error.request_id`가 `X-Request-ID` 응답 header와 일치 |
| `test_non_integer_content_length` | `Content-Length: abc` | 413(500 아님) |
| `test_negative_content_length` | `Content-Length: -1` | 413 |
| `test_declared_small_actual_large` | header는 예산 이내, 실제 스트리밍 body가 예산 초과(청크 전송) | 스트리밍 도중 413로 중단, request ID 포함 |
| `test_no_content_length_chunked_ok` | Transfer-Encoding: chunked, 실제 크기 정상 | 정상 처리(200), 같은 `resolve_request_id()` 규칙으로 만든 request ID가 `RequestContextMiddleware` 로그와 일치 |
| `test_disconnect_during_body` | body 수신 중 ASGI `receive()`가 `http.disconnect` 반환 | 예외/500 없이 정상 disconnect 처리 |
| `test_response_already_started_no_double_start` | 하위 app이 `http.response.start`를 이미 보낸 뒤(예: streaming 응답 도중) `limited_receive`가 `_BodyTooLarge`를 던지도록 강제 | `tracking_send`가 `response_started=True`를 관측해 `send_413()`이 두 번째 `http.response.start`를 보내지 않고 조용히 반환(ASGI 계약 위반 없음) |
| `test_request_id_shared_with_request_context_middleware` | 유효한 `X-Request-ID` header를 단 요청이 정상 처리(200)됨 | body limiter와 `RequestContextMiddleware`가 각자 계산한 request ID가 동일(같은 `resolve_request_id()` 호출) |

#### 6.6b 외부 호출 timeout — 단일 `DeadlineBudget`으로 전 구간 강제 (M-04/M2-02 대응)

이전 설계는 Ollama `read=600.0`을 그대로 유지해 90초 query timeout 이후에도
두 worker가 최대 600초 점유될 수 있었다(Review M-04). Iteration 1에서 추가한
`compute_upstream_timeout()`은 connect/read/write/pool **각 단계**의 timeout만
계산했을 뿐 호출 전체를 감싸는 overall 경계가 없었고, `remaining`이 작을 때
예산보다 큰 read timeout을 만들 수 있었으며, router `ChatOllama`/answer
`OllamaLLM`이 모두 프로세스 singleton이라 매 호출마다 다른 timeout을 주입할
구체적인 seam이 없었다(Review M2-02). 아래로 이 세 가지를 모두 고친다.

**단일 `DeadlineBudget` — request 진입부터 sync 호출까지 하나의 monotonic
deadline만 사용한다:**

```python
# src/simple_qna_rag/net_budget.py (신규, Phase 3)
import time
from dataclasses import dataclass

class DeadlineExceededError(Exception):
    """budget.require_remaining()이 상류 호출 시작 직전 예산 소진을 감지하면 던진다."""

@dataclass(frozen=True)
class DeadlineBudget:
    """§6.4 QueryExecutor가 ticket.deadline과 **동일한 loop.time() 값**으로 생성해
    sync worker에 전달하는 단일 deadline. M4는 표준 asyncio event loop 구현
    (`SelectorEventLoop`/`ProactorEventLoop`, §3.4 공식 profile)만 지원하며, 이
    구현에서 `loop.time()`은 `time.monotonic()`과 동일한 시계다(CPython
    `asyncio.base_events.BaseEventLoop.time()`의 문서화된 계약) — 따라서 이
    deadline 값은 async 쪽(`asyncio.timeout_at`)과 sync 쪽(`time.monotonic()`)
    양쪽에서 그대로 비교 가능하다. 커스텀 event loop policy는 M4 범위 밖이다.
    """
    deadline: float

    def remaining(self, now: float | None = None) -> float:
        return max(0.0, self.deadline - (now if now is not None else time.monotonic()))

    def require_remaining(self, safety_margin: float = 0.0) -> float:
        """매 upstream 호출 **직전** 호출한다. 남은 예산이 safety_margin 이하이면
        네트워크를 시작하지 않고 즉시 예외를 던진다(M2-02 수정안 — 이전
        설계는 remaining==0에서도 connect=0/read=min_read로 호출을 허용했다).
        통과하면 안전마진을 제외한 실제 가용 예산을 반환한다."""
        r = self.remaining() - safety_margin
        if r <= 0.0:
            raise DeadlineExceededError(f"remaining={r:.3f}s <= 0")
        return r

def compute_upstream_timeout(budget: DeadlineBudget, connect_cap: float,
                              min_read: float, safety_margin: float) -> httpx.Timeout:
    """호출 전: `budget.require_remaining(safety_margin)`을 반드시 먼저 호출해
    remaining<=0을 걸러낸 뒤에만 이 함수를 호출한다(관례 강제 — 이 함수는
    remaining>0을 전제한다). connect+read가 remaining을 넘지 않도록 상한을
    맞춘다(이전 버그: `read = max(min_read, remaining - connect)`는 remaining이
    min_read보다 작을 때 예산을 초과하는 read timeout을 만들었다)."""
    remaining = budget.require_remaining(safety_margin)
    connect = min(connect_cap, remaining)
    read = max(0.0, remaining - connect)
    if read < min_read and remaining > min_read:
        # 남는 예산이 있다면 read에 최소 여유를 우선 배정하고 connect를 줄인다 —
        # 그래도 connect+read <= remaining 불변식은 항상 유지한다.
        connect = remaining - min_read
        read = min_read
    return httpx.Timeout(connect=connect, read=read, write=read, pool=connect)
```

**전파 경로 — request부터 router/answer/DDGS까지 같은 `budget` 인스턴스 하나만
쓴다(M2-02):** §6.4 `QueryExecutor.run()`이 `ticket.deadline`과 동일한 값으로
`DeadlineBudget`을 생성해 `fn(*args, budget=budget)`으로 sync worker에 주입한다
(REQ-006 호환 — 함수 시그니처 추가 keyword-only 인자이며 기존 공개 `/rag`
응답 schema는 바뀌지 않는다). `route_query(question, *, budget)` ->
`agent._decide_tool(question, budget)`(router LLM 호출) ->
`rag_engine.RAGEngine.query(question, budget)`(answer LLM 호출) ->
`tools.web_search_function(query, budget)`(DDGS 호출)까지 **새 budget을 다시
만들지 않고 같은 객체를 그대로 전달**한다 — 하위 호출이 상위 호출의 남은
예산을 그대로 물려받으므로 "router가 예산을 다 쓰면 answer/DDGS는 자동으로
호출 자체가 막힌다"는 계약이 코드 구조로 성립한다.

**upstream 호출 직전 3단계 계약(모든 호출 지점 — router LLM, answer LLM,
DDGS — 동일하게 적용, M2-02 "per-phase cap + 호출 전체 watchdog"):**

1. `budget.require_remaining(settings.upstream_safety_margin_seconds)`를 호출해
   remaining이 이미 소진됐으면 네트워크를 시작하지 않고 즉시
   `DeadlineExceededError`를 던진다(기존 `route_query()`/`rag_engine.query()`
   try/except 계약이 이를 잡아 `success=False`로 정상 반환, §7.5 fault
   injection 표 그대로 유지).
2. 통과하면 `compute_upstream_timeout(budget, connect_cap, min_read,
   safety_margin)`으로 **per-phase cap**(connect/read/write/pool, 항상
   `connect+read <= remaining`)을 계산해 그 호출에만 적용한다.
3. **호출 전체 watchdog:** httpx의 네 값은 redirect/스트리밍 처리를 포함한
   wall-clock 총합을 보장하지 않으므로(Review M2-02 근거), 두 계층을 함께
   둔다 — (a) §6.4 `QueryExecutor`의 `ticket.deadline`(=`budget.deadline`)이
   worker 함수 전체에 대한 executor 수준 절대 상한으로 항상 유지되고(진행 중
   thread를 강제 종료하지는 못하지만 caller에게는 `QueryTimeoutError`가
   제때 반환됨, §6.4), (b) 응답을 스트리밍으로 소비하는 경로가 있다면 청크
   `n`개마다(또는 매 청크) `budget.expired()`를 재검사해, 서버가 개별 read
   timeout보다 짧은 간격으로 바이트를 흘리며 응답을 질질 끄는 경우에도
   전체 소비가 `budget.deadline`을 넘기지 못하게 한다.

**router/answer LLM singleton과 request-scoped timeout — 실제 API 확인 결과
(M3-02 대응, §13-5 가정을 여기서 확정한다):** 아래는 lock된
`langchain-ollama==0.3.10`/`ollama==0.6.0`/`ddgs==9.14.4`(현재 venv에
설치된 버전, `requirements.txt`의 `langchain-ollama>=0.2.0`/`ddgs>=9.0.0`
범위 안)에 대해 read-only executable spike로 직접 실행/확인한 결과다 —
추측이나 문서 인용이 아니다.

1. **`model_copy(update=...)`는 실제 httpx client를 바꾸지 않는다 —
   Iteration 2가 채택했던 "1차 방법"은 동작하지 않는 것으로 확인됐다.**
   `ChatOllama`/`OllamaLLM` 둘 다 `_client`/`_async_client`를
   `@model_validator(mode="after") def _set_clients(self)`(정확한 근거:
   `langchain_ollama.chat_models::ChatOllama._set_clients`,
   `langchain_ollama.llms::OllamaLLM._set_clients` — 두 클래스가 동일한
   패턴)에서 **생성자 시점에 한 번만** 만든다:
   ```python
   self._client = Client(host=cleaned_url, **sync_client_kwargs)
   ```
   Pydantic v2의 `BaseModel.model_copy(update=...)`는 필드 값만 얕게
   갱신할 뿐 `model_validator`를 다시 실행하지 않는다(Pydantic v2
   문서화된 계약, private attribute도 얕은 참조로 그대로 복사됨). 실측:
   ```python
   llm = ChatOllama(model="x", base_url="...", sync_client_kwargs={"timeout": 5.0})
   copy = llm.model_copy(update={"sync_client_kwargs": {"timeout": 999.0}})
   copy._client is llm._client        # True  — 같은 객체, 재생성 안 됨
   copy._client._client.timeout       # Timeout(timeout=5.0) — 원본 값 그대로, 999.0 미반영
   ```
   즉 `model_copy`로 `client_kwargs`/`sync_client_kwargs`를 바꿔도 실제
   호출에 쓰이는 httpx timeout은 **원본 인스턴스 생성 시점 값에 고정된
   채로 절대 바뀌지 않는다** — 이 경로로는 매 호출 timeout을 주입할 수
   없다.
2. **유일하게 동작하는 방법은 매 요청 재생성이다 — 비용은 낮다.**
   `ChatOllama.model_fields["validate_model_on_init"].default`와
   `OllamaLLM.model_fields["validate_model_on_init"].default` 모두 실측
   `False`다 — 즉 생성자가 `validate_model(self._client, self.model)`(모델
   존재 여부를 실제로 Ollama에 물어보는 네트워크 호출)를 기본적으로
   건너뛴다. 따라서 `ChatOllama(...)`/`OllamaLLM(...)`을 요청마다 새로
   만드는 비용은 순수 in-process 객체 생성(`httpx.Client()` 인스턴스화)뿐이며
   네트워크 I/O가 없다. singleton은 이제 **model name/base_url/temperature
   등 불변 설정의 캐시**로만 쓰고, 실제 호출 직전 그 설정 +
   `compute_upstream_timeout(budget, ...)`로 계산한 `httpx.Timeout`을
   `sync_client_kwargs={"timeout": timeout}`으로 지정한 **새 인스턴스**를
   만들어 그 호출에만 쓴다.
3. **router의 `RunnableBinding`도 매 요청 재구성이 곧 정답이다 —
   기존 계획대로 `.bound`를 몰래 바꿀 필요가 없다.** `ChatOllama.bind_tools()`는
   (`langchain_ollama.chat_models::ChatOllama.bind_tools`) 내부에서
   `formatted_tools = [convert_to_openai_tool(t) for t in tools];
   return super().bind(tools=formatted_tools, **kwargs)`만 수행한다 —
   네트워크 호출이 없고, 반환된 `RunnableBinding`의 `.bound`는 **호출
   시점의 `self` 그 자체**(실측: `bound.bound is llm`)다. 따라서 "기존
   `RunnableBinding`의 bound model을 in-place로 교체하는 API"를 찾을
   필요가 애초에 없다 — `agent.py::_get_router_llm(budget)`을 매 요청
   `ChatOllama(model=..., base_url=..., sync_client_kwargs={"timeout":
   ...}).bind_tools([web_search_tool, rag_tool])`로 새로 호출하면 매번
   새 `RunnableBinding`이 만들어지고, tool 목록은 매 호출 동일하게
   `convert_to_openai_tool`로 재계산되므로(입력이 불변이라 출력도 항상
   동일) 실질적으로 캐시된 것과 같은 결과를 얻으면서 timeout만 요청별로
   달라진다.
4. **answer 경로는 `generate_answer(...)`에 `llm`을 명시 인자로 받는다.**
   `RAGEngine.generate_answer(self, question, context, template_str, *,
   llm: "BaseLanguageModel | None" = None)`으로 시그니처를 바꾼다(기본값
   `None`이면 `self.llm`을 그대로 쓰는 기존 동작 100% 보존 — 평가
   harness처럼 budget 없이 직접 호출하는 기존 caller와 호환, REQ-006).
   `llm = llm or self.llm`으로 시작해 `qa_chain = (... | llm |
   StrOutputParser())`을 조립한다. `RAGEngine.query(question, *,
   budget: "DeadlineBudget | None" = None)`이 `budget`이 주어지면
   `budget.require_remaining(safety_margin)` 통과 후
   `OllamaLLM(model=self.llm.model, base_url=self.llm.base_url,
   temperature=self.llm.temperature, sync_client_kwargs={"timeout":
   compute_upstream_timeout(budget, ...)})`을 새로 만들어
   `self.generate_answer(question, context, template_str, llm=request_llm)`로
   전달한다. `budget=None`이면(기존 evaluator 호출 경로) 이 블록 전체를
   건너뛰고 `self.llm`을 그대로 쓴다 — 시그니처 추가만으로 기존 호출자를
   깨지 않는다.
5. **잔여 한계 — 스트리밍 총 소요시간은 여전히 하드 개런티가 없다.**
   `ChatOllama.invoke()`(non-streaming으로 보이는 공개 API)조차 내부적으로
   `_chat_stream_with_aggregation()`을 거쳐 항상 HTTP 스트리밍 응답을
   집계한다(실측: `langchain_ollama.chat_models::ChatOllama._generate`
   소스가 `final_chunk = self._chat_stream_with_aggregation(...)`로
   시작). httpx의 `read` timeout은 **각 개별 소켓 read**를 개별적으로
   제한할 뿐 스트림 전체의 누적 소요시간을 제한하지 않으므로, 서버가 매
   청크를 `read` timeout보다 짧은 간격으로 흘리면(트리클링) 개별 read는
   전부 timeout 안에 들어오면서도 총 응답 시간은 `budget.deadline`을
   넘길 수 있다. `ChatOllama`/`OllamaLLM` 공개 API에는 청크마다 콜백을
   끼워 넣어 소비를 중단시킬 supported hook이 없다(`callbacks`/
   `run_manager`는 관측용이며 스트림을 중단시키는 계약이 아니다). 이
   경우 caller는 §6.4 `QueryExecutor`의 `ticket.deadline`
   (`asyncio.timeout_at`)으로 정시에 `QueryTimeoutError`를 받지만(=caller
   응답은 bounded), 그 sync worker 자신은 스트림이 실제로 끝날 때까지
   thread를 점유한 채 계속 돈다 — `_orphaned` gauge(§5.4)로 관측되고
   `docs/operations/Runbook.md#incident-triage`에 "LLM 응답 트리클링으로
   인한 orphan 누적" 항목을 추가해 명시적으로 문서화한다(아래 6번
   subprocess 경계를 router/answer에는 기본 적용하지 않는 이유이기도
   하다 — connect/read별 httpx timeout이 대부분의 실패 모드를 이미
   막고, Ollama는 신뢰 경계 안의 로컬/사내 서비스이므로 DDGS만큼
   적대적인 입력을 가정할 필요가 낮다).
6. **DDGS는 애초에 connect/read 분리도, per-call timeout override도
   지원하지 않고, 내부 스레드 풀이 timeout을 무시하고 blocking-join할 수
   있다 — 지원 불가 API이므로 종료 가능한 subprocess 경계를 채택한다.**
   `DDGS.__init__(self, proxy=None, timeout: int | None = 5, *,
   verify=True, api_url=None, spawn_api=False)`(정확한 근거:
   `ddgs.ddgs::DDGS.__init__`)는 `timeout`을 **생성자에서 한 번만** 받아
   각 검색 엔진 인스턴스에 그대로 넘길 뿐(`ddgs.ddgs::DDGS._get_engines`),
   `.text(query, **kwargs)`/`_search_sync(...)`(`ddgs.ddgs::DDGS.text`,
   `ddgs.ddgs::DDGS._search_sync`)의 `**kwargs`에는 timeout이 없다 — 매
   호출 budget에 맞춰 timeout을 바꾸려면 `DDGS(timeout=...)` 인스턴스
   자체를 매번 새로 만들어야 한다(이는 지원되는 사용법이므로 채택).
   **더 심각한 문제는 총 소요시간 상한이 없다는 것이다:** 실제 설치된
   `ddgs==9.14.4`의 `_search_sync()` 소스를 직접 읽으면, 엔진별 검색을
   `with ThreadPoolExecutor(max_workers=..., thread_name_prefix="DDGS")
   as executor:` 블록 안에서 실행하고 `wait(futures, timeout=self._timeout,
   return_when="FIRST_EXCEPTION")`로 배치 단위 대기 시간만 제한한다 —
   그러나 이 `with` 블록을 빠져나갈 때 Python 표준 라이브러리 계약상
   `executor.__exit__()`가 인자 없는(`wait=True` 기본값)
   `shutdown(wait=True)`를 호출하므로, `max_results`에 먼저 도달해
   for 루프를 `break`하거나 모든 엔진을 순회한 뒤에도 **아직 완료되지
   않은 제출된 future가 있으면 `text()` 호출 자체가 그 스레드가 끝날
   때까지 무한정 블록**된다 — `self._timeout`은 이 최종 `shutdown(wait=True)`
   대기에는 전혀 적용되지 않는다. 이 라이브러리는 outer caller가 주입할
   수 있는 monotonic clock이나 취소 hook을 전혀 제공하지 않으므로("동일
   injected monotonic clock" 요구를 만족시킬 지점 자체가 라이브러리
   안에 없다), in-process 코드로는 `search_web()` 호출이 QueryExecutor의
   thread pool worker를 절대 시간 안에 반환한다고 보장할 수 없다.

   **subprocess boundary 구체 설계 (신규 `net_budget.py::run_in_killable_subprocess`,
   Phase 3):**
   ```python
   # src/simple_qna_rag/net_budget.py 추가
   import multiprocessing as mp

   _SPAWN_CTX = mp.get_context("spawn")  # ASGI 이벤트 루프/스레드 상태를
                                          # 자식이 상속하지 않도록 fork 대신 spawn 고정

   class SubprocessDeadlineExceeded(Exception):
       """budget 안에 자식 프로세스가 결과를 반환하지 못해 강제 종료했을 때."""

   def _run_target(fn, args, kwargs, out_queue: "mp.Queue") -> None:
       try:
           out_queue.put(("ok", fn(*args, **kwargs)))
       except Exception as exc:  # noqa: BLE001 - 자식의 모든 예외를 부모로 그대로 전달
           out_queue.put(("error", exc))

   def run_in_killable_subprocess(
       fn, args: tuple, kwargs: dict, *, budget: "DeadlineBudget",
       safety_margin: float, grace_seconds: float = 1.0,
       now_fn: "Callable[[], float]" = time.monotonic,   # 테스트 주입용(M3-02)
   ):
       """fn(*args, **kwargs)을 별도 프로세스에서 실행하고 budget 안에 결과가
       없으면 프로세스를 강제 종료해 **호출자는 항상 bounded 시간 안에
       반환**되도록 보장한다(DDGS처럼 내부적으로 timeout을 무시하고
       blocking-join하는 API를 감싸는 유일한 방법, M3-02). 부모는 자식의
       내부 스레드 풀 상태를 전혀 기다리지 않는다 — OS 프로세스
       join/kill만 사용하므로 자식이 무엇을 하고 있든 강제로 반환할 수
       있다."""
       remaining = budget.require_remaining(safety_margin)
       out_queue: "mp.Queue" = _SPAWN_CTX.Queue()
       proc = _SPAWN_CTX.Process(target=_run_target, args=(fn, args, kwargs, out_queue), daemon=True)
       started_at = now_fn()
       proc.start()
       try:
           kind, payload = out_queue.get(timeout=remaining)
       except Exception:  # queue.Empty 포함 — remaining 안에 결과 없음
           proc.terminate()                      # SIGTERM
           proc.join(grace_seconds)
           if proc.is_alive():
               proc.kill()                       # SIGKILL — 반드시 반환을 보장
               proc.join()
           raise SubprocessDeadlineExceeded(
               f"remaining={remaining:.3f}s 안에 자식 프로세스가 반환하지 않아 종료함"
           )
       proc.join(max(0.0, remaining - (now_fn() - started_at)))
       if proc.is_alive():
           proc.kill()
           proc.join()
       if kind == "error":
           raise payload
       return payload
   ```
   `tools.py::web_search_function(query, *, budget)`이 기존 직접
   `search_web(query)` 호출을 `run_in_killable_subprocess(search_web,
   (query,), {}, budget=budget, safety_margin=settings.upstream_safety_margin_seconds)`로
   교체한다 — `search_web()` 자신의 코드(내부에서 여전히
   `DDGS(timeout=min(WEB_SEARCH_TIMEOUT, remaining))`을 그대로 사용)는
   변경하지 않고 그대로 자식 프로세스 안에서 실행된다. 재시도 횟수는
   기존과 동일하게 명시적 **0**을 유지한다(REQ-007.4, 코드 상수
   `DDGS_RETRY_COUNT = 0`, `web_search.py:48` 기존 동작과 일치 —
   subprocess 경계는 반환 시간만 보장할 뿐 재시도 정책과는 무관).
   프로세스 spawn 비용(리눅스 기준 수십 ms)은 이미 네트워크 바운드인
   DDGS 호출(수백 ms~수 초) 대비 상대적으로 작고, 이 경계가 없으면
   worker 반환 자체를 보장할 수 없으므로(위 근거) 이 비용을 감수하는
   것으로 채택한다.

**결정론적 통합 테스트(신규 `tests/integration/test_upstream_deadline.py`,
M3-02 수정안 대응 — 모든 client 단계의 예산 적용과 worker 반환을 spike로
확인한 API 형태 그대로 검증한다):**

| 테스트 | 강제 방법 | 검증 |
|---|---|---|
| `test_remaining_exhausted_blocks_call_start` | `budget.deadline`을 이미 지난 값으로 고정 | `require_remaining()`이 네트워크 호출 전에 `DeadlineExceededError`를 던짐, mock transport에 요청이 0회 도달 |
| `test_compute_upstream_timeout_never_exceeds_remaining` | 다양한 `remaining < min_read` 조합을 property 기반으로 대입 | `connect + read <= remaining`이 항상 성립(이전 버그 재발 방지) |
| `test_connect_stall_bounded_by_budget` | fake TCP 서버가 connect를 절대 응답하지 않음(connect stall) | worker가 `budget.deadline` 이전에 `httpx.ConnectTimeout`으로 반환 |
| `test_read_stall_bounded_by_budget` | fake 서버가 connect 후 응답 바이트를 전송하지 않음(read stall) | worker가 `budget.deadline` 이전에 `httpx.ReadTimeout`으로 반환 |
| `test_stream_trickle_bounded_by_budget` | fake 서버가 개별 read timeout보다 짧은 간격으로 1바이트씩 무한히 흘림(트리클링) | 청크 단위 `budget.expired()` 검사로 개별 read timeout을 우회하더라도 전체 소비가 `budget.deadline` 이전에 중단됨 |
| `test_budget_propagates_unchanged_through_router_and_answer` | router LLM 호출과 answer LLM 호출 각각에 전달된 `budget` 객체의 `id()`를 캡처 | 두 호출 지점에서 관측한 `budget`이 동일 객체(재계산되지 않음)이고, router가 예산을 소비한 뒤 answer 호출 시점의 `remaining()`이 그만큼 줄어 있음 |
| `test_model_copy_does_not_rebuild_client`(M3-02, 회귀 방지) | 위 실측 스크립트를 그대로 단위 테스트화 — `model_copy(update={"sync_client_kwargs": ...})` 후 `copy._client is original._client`와 `copy._client._client.timeout`을 확인 | 이 assertion이 언젠가 뒤집혀(즉 `model_copy`가 실제로 rebuild하도록 `langchain-ollama`가 바뀌어) 실패하면 그 자체가 "재생성 대신 model_copy로 되돌려도 된다"는 신호이므로 이 테스트가 CI에서 실패로 드러나야 한다(의도적 회귀 감시 테스트) |
| `test_router_binding_rebuilt_per_request` | 연속된 두 번의 `agent._get_router_llm(budget_a)`/`agent._get_router_llm(budget_b)` 호출에서 반환된 `RunnableBinding.bound._client._client.timeout`을 비교 | 서로 다른 budget마다 다른 timeout 값을 가진 서로 다른 `ChatOllama` 인스턴스가 만들어짐(캐시 재사용 없음), `bound_tools`(`.kwargs["tools"]`)는 매번 동일 |
| `test_answer_llm_injected_into_chain` | `RAGEngine.generate_answer(..., llm=fake_llm)` 직접 호출과, `RAGEngine.query(question, budget=budget)`를 통한 간접 호출 각각에서 실제 체인에 들어간 llm 객체를 캡처 | 전자는 `fake_llm`이, 후자는 request-scoped `OllamaLLM` 인스턴스(= `self.llm`이 아님)가 체인에 들어감; `budget=None` 호출은 여전히 `self.llm`이 그대로 쓰임(하위 호환) |
| `test_ddgs_worker_returns_within_budget_plus_grace` | `search_web`을 대체할 fake target이 `budget.deadline`을 훨씬 넘겨 `time.sleep(...)`(자식 프로세스 안에서 실제로 블록)하도록 주입 | `run_in_killable_subprocess()` 호출이 `remaining + grace_seconds` 이내에 `SubprocessDeadlineExceeded`로 반환하고(호출측 wall-clock 측정), 자식 프로세스가 실제로 종료됨(`proc.is_alive() is False`) — DDGS 내부 스레드 풀의 실제 지연 시간과 무관하게 QueryExecutor worker가 항상 bounded 시간 안에 반환됨을 증명 |
| `test_ddgs_worker_normal_result_passthrough` | fake target이 budget 안에 정상 반환 | `run_in_killable_subprocess()`가 그 반환값을 그대로 전달, 프로세스가 정상 종료(exitcode 0) |

### 6.7 결정론적 ASGI 테스트 (신규 `tests/integration/test_web_concurrency.py`)

| fixture | 목적 |
|---|---|
| `slow_fake_route_query(delay=2.0)` | 고정 지연 함수 — 부하/timeout 시나리오 |
| `never_return_route_query(release_event)` | `threading.Event`가 set될 때까지 반환하지 않음 — orphan slot 계측 검증 |
| `flaky_ollama_probe(fail_n_times)` | readiness flap 검증 |
| `failing_startup_engine()` | STARTING에서 벗어나지 못하는 경로 검증 |
| `fake_timeout_cm(barrier)` | §6.4 `QueryExecutor(timeout_cm_factory=...)` 주입용 — barrier/event로 만료 시점을 직접 제어해 real sleep이나 loop clock과의 불일치 없이 deadline 경계를 결정론적으로 재현(M2-01) |

## 7. Phase 4 — 부하·장애 검증과 tuning

### 7.1 harness

신규 `evaluation/m4_load.py` — in-process ASGI mock(`httpx.AsyncClient(app=app,
base_url="http://test")`)과 실제 HTTP live 클라이언트를 공유 인터페이스
(`run_wave(client, requests) -> WaveResult`)로 감싼다. 출력: p50/p95/p99,
queue/service/total latency 분해, RSS peak(`resource.getrusage(RUSAGE_SELF).ru_maxrss`),
thread 수(`threading.active_count()`), rejection 수, orphan 수.

### 7.2 결정론적 gate 실행

```bash
python -m evaluation.m4_load mock --profile smoke-200ms --count 40 --concurrency 2
python -m evaluation.m4_load mock --profile smoke-2s --count 8 --concurrency 2 --queue 4
python -m evaluation.m4_load mock --profile timeout-cancel-burst
```

`accepted + rejected + timeout == count`를 항상 assert(§5.1 "미분류 outcome
0"). smoke-2s는 §6.4 state table의 "포화, 대기 포화" 행을 8건 동시 요청으로
실제로 때린다: 2 running + 4 waiting + 2 rejected(1초 내 503) 조합.

### 7.3 live 12-case + M3 14 gate 재실행

```bash
RUN_LIVE_LLM_TESTS=1 python -m evaluation.m4_load live \
  --case-ids-file evaluation/live_12_case_ids.json --clients 4 --candidate-id m4-p4-tuning
RUN_LIVE_LLM_TESTS=1 python -m evaluation.baseline \
  --dataset evaluation/datasets/golden.jsonl --output evaluation/reports/m4/m4-p4-tuning \
  --candidate-id m4-p4-tuning
```

두 번째 명령은 기존 M3 14 gate(`evaluate_gates()`, `evaluation/compare.py`)를
그대로 재사용한다 — 새 gate 코드를 만들지 않는다(REQ-005/§5.1 M3 품질 항).

### 7.4 concurrency 1 vs 2 결정 규칙

기본은 2. §7.2/§7.3 결과에서 다음 중 하나라도 관측되면 `concurrency_limit`
기본값을 1로 낮추고 `Design.md` 본 절과 `settings.py` 기본값 주석에 사유를
기록한다: (a) RSS peak가 concurrency=1 대비 1.5배를 초과, (b) 동시 2 실행
중 model 호출 예외/불일치 결과 발생, (c) live 12-case p95가 concurrency=1
warm 대비 2.5배를 초과. 외부 queue/worker 도입은 하지 않는다(Plan §위험).

### 7.5 fault injection 목록

| 대상 | 방법 | 기대 동작 |
|---|---|---|
| Ollama timeout | `OllamaLLM.invoke`를 `httpx.ReadTimeout` raise로 monkeypatch | `route_query()`가 예외를 잡아 `success=False` 반환(기존 `rag_engine.query()` try/except 계약 유지), executor는 정상 slot 반환 |
| DDGS 실패 | `search_web`을 `[]` 반환으로 monkeypatch | 기존 `document_qa` 재시도 경로(`agent.py:230-235`) 그대로 동작 |
| executor 포화 | §7.2 smoke-2s | 503 + `Retry-After`, orphan 0 |
| client disconnect | ASGI `receive()`가 `{"type":"http.disconnect"}` 반환하도록 fixture | 서버 측 스레드는 완료까지 실행(§6.4 한계 문서화), 응답 전송 생략, slot은 완료 시 정상 반환 |
| logging 실패 | 로그 핸들러 `emit()`을 예외 raise로 monkeypatch | 요청은 정상 결과 반환(§5.6 `safe_log`) |

## 8. Phase 5 — versioned index lifecycle

### 8.1 디렉터리 레이아웃

```
runtime/index/
  versions/<version_id>/index.faiss
  versions/<version_id>/index.pkl
  versions/<version_id>/manifest.json
  staging/<staging_id>/...            # build와 import-legacy가 공유하는 경로. 서비스가 읽지 않음
  CURRENT                             # 텍스트 파일 1줄: version_id
  .lock                                # fcntl.flock 대상 — build/import/activate/rollback/retention 공통(§8.3a)
```

`version_id` 형식과 순환 참조 제거는 §8.2에서 정의한다. **포인터는
심볼릭 링크가 아니라 파일**로 구현한다(컨테이너 bind mount/개발 OS
차이에서 심볼릭 링크 안정성이 낮기 때문 — Plan §5.1의 "pointer 교체"
요구를 원자적 파일 교체로 충족).

기존 `runtime/vectorstore/`(M3 index.faiss/index.pkl, manifest 없음)는
**수정하지 않고 그대로 둔다**. `runtime/index/`가 없거나 `CURRENT`가
가리키는 manifest 검증에 실패하면 서비스는 **fail-closed**한다 — legacy로
자동 폴백하지 않는다(§8.6, Review M-02).

### 8.2 manifest schema와 순환 없는 version ID (`index/manifest.py::IndexManifest`, M-03 대응)

**문제였던 순환 참조:** 이전 설계는 `version_id`를 "manifest SHA-256 앞
8자"로 정의하면서 그 manifest 자체에 `version_id` 필드를 포함했다 —
hash를 계산하려면 완성된 manifest(= `version_id` 포함)가 필요하고,
`version_id`를 만들려면 hash가 필요한 순환이었다.

**해소 방법 — `content_digest`를 `version_id`를 포함하지 않는 명시적
payload로 분리한다:**

```python
# content_digest 계산에 포함되는 필드 — "무엇을 빌드했는가"만 표현하고
# 신원/시각/자기참조 필드(version_id, created_at_utc, schema_version,
# source, legacy_import_note)는 제외한다.
_CONTENT_FIELDS = (
    "corpus_manifest_sha256", "source_count", "chunk_count",
    "embedding_model", "embedding_model_revision", "normalize_embeddings",
    "chunk_size", "chunk_overlap", "faiss_index_type", "faiss_dimension",
    "settings_hash", "lock_sha256", "index_faiss_sha256", "index_pkl_sha256",
)

def compute_content_digest(fields: dict) -> str:
    payload = {k: fields[k] for k in _CONTENT_FIELDS}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                            separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()

def compute_version_id(content_digest: str, created_at_utc: str) -> str:
    compact = created_at_utc.replace("-", "").replace(":", "")  # "20260808T120000Z"
    return f"{compact}-{content_digest[:8]}"
```

`content_digest`는 `version_id`를 절대 참조하지 않으므로 자기참조가
없다. `version_id = compute_version_id(content_digest, created_at_utc)`를
확정한 뒤에야 `version_id`를 포함한 전체 manifest를 조립·직렬화한다.
manifest 파일 자체의 외부 hash(`evaluation/m4_fingerprint.py`의
`index_manifest_sha256`, §10.1)는 **완성된 manifest.json 바이트에 대해
별도로** 계산하며 manifest 내부 필드로 다시 저장하지 않는다 — "manifest가
자기 자신의 hash를 담는" 두 번째 순환도 이렇게 방지한다.

```json
{
  "schema_version": "1.0.0",
  "version_id": "20260808T120000Z-3f7a21b0",
  "content_digest": "3f7a21b0...(64자 전체)",
  "created_at_utc": "2026-08-08T12:00:00Z",
  "corpus_manifest_sha256": "…",
  "source_count": 18,
  "chunk_count": 1234,
  "embedding_model": "BAAI/bge-m3",
  "embedding_model_revision": "unknown_legacy | <hf revision>",
  "normalize_embeddings": true,
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "faiss_index_type": "IndexFlatIP",
  "faiss_dimension": 1024,
  "settings_hash": "…",
  "lock_sha256": "…",
  "index_faiss_sha256": "…",
  "index_pkl_sha256": "…",
  "source": "build",
  "legacy_import_note": null
}
```

직렬화: `json.dumps(manifest, sort_keys=True, ensure_ascii=False,
separators=(",", ":"))` — canonical hash 재현성 보장(기존
`evaluation/reporting.py` 스타일 재사용). `content_digest`를 manifest에
저장해 두면 activate 시 재계산 없이 §8.4의 "대상 이미 존재" idempotency
비교에 바로 쓸 수 있다(다시 계산해 저장값과 대조하는 검증도 activate
3단계에서 함께 수행한다 — 저장된 값을 무조건 신뢰하지 않는다).

### 8.3 CLI 서브커맨드 — build/import 공통 staging (M-03 대응)

`simple-qna-rag-index`(기존 `cli/index_documents.py`, 하위 호환 유지)는
`build` 서브커맨드로 재해석되고, 신규 `cli/index_lifecycle.py`가
`simple-qna-rag-index-lifecycle {activate|rollback|import-legacy|retention}`을
제공한다(entry point는 `pyproject.toml` `[project.scripts]`에 추가).

```
simple-qna-rag-index build [--documents-dir DIR]
simple-qna-rag-index-lifecycle import-legacy --from runtime/vectorstore   # §8.7(M3-04) — 승인 hash는 CLI 인자로 받지 않는다(아래 참조)
simple-qna-rag-index-lifecycle activate --staging-id <staging_id>
simple-qna-rag-index-lifecycle rollback --to <version_id>
simple-qna-rag-index-lifecycle retention [--dry-run] [--keep 2]
```

**M3-04 — `import-legacy`에는 더 이상 `--expected-*` 인자가 없다.**
Iteration 2/3은 승인된 SHA-256을 CLI 인자(또는 별도 파일 경로)로 받게
했으나, 이는 "운영자가 승인한 fingerprint"를 CLI 호출 시점에 **누구나
같이 조작해 넘길 수 있는 값**으로 만들어(파일과 그 파일의 hash를 공격자가
함께 공급하면 검증을 통과) provenance를 전혀 입증하지 못했다(Review
M3-04). §8.7이 이 인자들을 완전히 제거하고, 승인 hash의 유일한 원본을
git-committed된 M3 baseline 파일 하나로 고정한다.

**`build`와 `import-legacy`는 반드시 같은 산출물 모양을 `staging/<staging_id>/`에
만든다** — `index.faiss`, `index.pkl`(build는 새로 생성, import-legacy는
`shutil.copy2`로 `runtime/vectorstore/`에서 원본 미변경 복사), 그리고 두
경우 모두 §8.2의 `content_digest`/`version_id`를 포함한 **완성된**
`manifest.json`을 staging에 직접 써서 마무리한다(import-legacy는
`embedding_model_revision="unknown_legacy"`, `source="legacy_import"`로
채운다). `staging_id`는 `uuid4().hex`(build_id 개념을 대체 — 아직
`version_id`가 확정되지 않은 시점의 임시 이름이므로 content와 무관해도
된다). 이후 **build와 import-legacy 모두 동일한 하나의
`activate(staging_id)` 경로**만 거쳐 `versions/`로 승격된다(§8.4) — 두
산출 경로가 서로 다른 activation 계약을 갖던 이전 설계의 불일치를
제거한다. `staging/`이나 `versions/`, `CURRENT`는 build/import 단계에서
서로의 결과를 절대 직접 건드리지 않는다(REQ-008.3 "다른 staging
디렉터리").

**exit code 계약(§8.4):**

```
0 정상(신규 activate 또는 §8.4 "동일 content_digest" idempotent 성공 포함)
2 settings 오류
3 manifest 검증 실패(schema mismatch, hash mismatch, embedding/chunk 불일치,
  또는 §8.4의 "동일 version_id·다른 content_digest" 충돌)
4 lock 경합(동시 build/import/activate/rollback/retention, §8.3a)
5 fault(IO 오류, disk full 등)
6 CorpusManifestError(source ID 충돌) — index/retrieval/answer/baseline CLI 공통
```

`CorpusManifestError`(기존 `evaluation/reporting.py`가 던짐, Problem.md P2)를
모든 CLI(`index_documents.py`, `evaluation/retrieval.py`, `evaluation/answers.py`,
`evaluation/baseline.py`)의 `main()`에서 동일하게 잡아 exit 6 + traceback
비노출 메시지로 통일한다.

### 8.3a lock 범위 — build를 포함한 단일 OS lock (M-03 대응)

Requirement M4-REQ-008.4는 "동시 build/activate는 OS-level lock으로
하나만 허용"을 요구하지만, 이전 설계는 activate 절차 4단계에서만 lock을
얻고 build의 lock 구간을 정의하지 않았다(Review M-03). `runtime/index/.lock`
하나가 **모든** lifecycle 변경 작업(build, import-legacy, activate,
rollback, retention)을 상호 배제한다 — 명령 시작과 동시에 non-blocking으로
획득하고, 실패하면 즉시 exit 4로 종료한다(대기하지 않음, "예측 가능한
종료 코드"):

| operation | lock 획득 시점 | lock 보유 구간 | 실패 시 |
|---|---|---|---|
| `build` | 명령 시작(문서 읽기 이전) | embedding 계산+`staging/`에 파일/manifest 완성까지 전체 | exit 4 |
| `import-legacy` | 명령 시작 | `staging/`로 복사+manifest 작성 완료까지 전체 | exit 4 |
| `activate` | 명령 시작(§8.4 1단계 이전) | §8.4 절차 전체(1~10단계) | exit 4 |
| `rollback` | 명령 시작 | §8.4의 활성화 하위 절차 전체 | exit 4 |
| `retention` | 명령 시작 | 대상 판별+삭제 전체 | exit 4 |

**trade-off를 명시한다:** build가 embedding 계산 동안(수 분 가능) lock을
독점하므로 그동안 activate/rollback은 대기하지 않고 즉시 실패한다 — M4는
단일 운영자·비-CI 반복 시나리오를 가정하므로(Requirement §7) 이 직렬화는
허용 가능한 단순함으로 채택한다. 향후 동시 운영이 필요해지면 lock을
"build 단계"와 "활성화 단계"로 분리하는 재설계가 필요하며, 이는 M4 범위
밖(M5 조건부 후속)이다.

### 8.4 원자적 activate 절차 — 충돌/멱등성 정책 포함 (M-03 대응)

1. staging 디렉터리의 `index.faiss`/`index.pkl` 각각 `os.fsync(fd)`.
2. staging 디렉터리 자체 fsync(디렉터리 엔트리 flush, POSIX `os.open(dir, O_RDONLY)` + `os.fsync`).
3. manifest 재검증: `content_digest`를 §8.2 함수로 다시 계산해 저장된 값과
   비교, `version_id`를 재계산해 저장된 값과 비교, **staging 디렉터리와
   `index.faiss`/`index.pkl`의 owner/mode를 dir_fd 기반 `fstat()`으로
   재검증**(§8.7 M3-04 신규 — import/build 시점 이후 staging 권한이
   바뀌지 않았음을 activate 시점에 다시 확인, 검사 규칙은 §8.7 4번과
   동일), FAISS `load_local(allow_dangerous_deserialization=True)` load
   smoke test — 모두 성공해야 다음 단계로 진행.
4. `.lock`을 `fcntl.flock(fd, LOCK_EX | LOCK_NB)`으로 획득(§8.3a — 이미
   명령 시작 시 획득했다면 이 단계는 재확인만).
5. **대상 존재 여부 확인(신규 — 이전 설계에 없던 정책):**
   `versions/<version_id>`가 이미 존재하면 —
   - 기존 `versions/<version_id>/manifest.json`의 `content_digest`가
     staging 것과 **동일** -> 이미 활성화된 것과 같은 내용이므로 rename을
     건너뛰고(중복 쓰기 없음) 7단계로 진행 — **idempotent 성공, exit 0**.
   - `content_digest`가 **다름**(8자 prefix 충돌, 확률적으로 극히 낮지만
     반드시 처리) -> `versions/`를 건드리지 않고 즉시 **exit 3**.
   대상이 없으면 정상 신규 경로로 6단계 진행.
6. `os.rename(staging_dir, versions_dir / version_id)`(동일 파일시스템,
   원자적).
7. `versions/` 부모 디렉터리 fsync.
8. `CURRENT.tmp`에 `version_id` 기록, fsync, `os.replace(CURRENT.tmp, CURRENT)`
   (원자적, 같은 파일시스템).
9. `runtime/index/` 디렉터리 fsync.
10. `metrics.py::set_active_index_version(version_id)` 호출(§5.4 — 이전
    label 제거 후 신규 label set) 후 lock 해제.

3~5단계 사이 어디서 중단돼도 `CURRENT`는 이전(정상) 버전을 계속 가리킨다 —
서비스가 절반만 쓰인 버전을 읽을 방법이 없다. 6단계 이후 9단계 이전 중단은
`versions/`에 참조되지 않는 완전한 새 버전 디렉터리를 남기지만(orphan),
활성 pointer는 변하지 않는다 — `retention`이 이런 orphan을 정리 대상으로
식별한다(dry-run 기본). staging 정리: activate 성공(6단계 rename) 또는
idempotent 성공(5단계) 이후 `staging/<staging_id>/`는 더 이상 존재하지
않거나(rename됨) 더 이상 참조되지 않는 중복이므로, activate 마지막에
정리하거나 `retention`이 "activate된 지 오래된 미참조 staging"으로 함께
정리한다(dry-run 기본, 명령 자체는 실패시키지 않음).

`rollback --to <version_id>`는 대상이 이미 `versions/`에 존재하므로 1~2단계
없이 3단계(manifest/hash 재검증)와 7~10단계만 수행한다 — 5단계의 "동일
content_digest" 판정은 rollback에는 적용하지 않는다(대상이 이미 `versions/`에
있는 것이 rollback의 전제 조건이므로 자연히 idempotent).

### 8.5 fault injection 지점(§Requirement 5.1 "index" 행, M-02/M-03 확장)

| # | 주입 지점 | 방법 | 검증 |
|---|---|---|---|
| 1 | staging 쓰기 중단(§8.4 step 1 이전) | `os.kill(pid, SIGKILL)` 시뮬레이션(subprocess로 build 실행 후 kill) | `CURRENT`/`versions/` 불변, staging에 부분 파일만 남음 |
| 2 | rename 이후 CURRENT 쓰기 이전(step 6~8 사이) | 테스트에서 monkeypatch로 7~8단계 사이 예외 강제 | `CURRENT`가 이전 버전 유지, 새 버전은 orphan으로 존재 |
| 3 | disk full | `tmp_path`를 quota 제한 파일시스템(tmpfs 소용량) 또는 `os.fsync`를 `OSError(ENOSPC)` raise로 monkeypatch | staging 쓰기 실패, exit 5, `versions/`/`CURRENT` 불변 |
| 4 | hash mismatch | activate 직전 staging 파일 1바이트 변조 | exit 3, `CURRENT` 불변 |
| 5 | 동시 build/activate(§8.3a) | build가 lock을 보유한 동안 별도 프로세스가 activate 호출; 역방향도 검증 | 두 번째 프로세스 exit 4, 첫 프로세스 정상 완료, 어느 순서든 동일 |
| 6 | 설정 불일치 | 서비스 시작 시 `embedding_model` 설정을 manifest와 다르게 설정 | `/health/ready` `reason=index_invalid`, 프로세스는 살아있음(crash 아님) |
| 7 | `CURRENT`/manifest 부재(M-02) | fresh `runtime/index/`(빈 디렉터리) 또는 `runtime/index/` 자체 없음 상태로 서비스 기동 | legacy `runtime/vectorstore/` 자동 로드 없음, `/health/ready` `reason=index_invalid`로 fail-closed, 프로세스는 죽지 않음 |
| 8 | activate 대상 이미 존재, 동일 content(§8.4 step 5) | 같은 staging을 두 번 activate 시도 | 두 번째 호출도 exit 0(idempotent), `versions/`/`CURRENT` 파일 hash 불변 |
| 9 | activate 대상 이미 존재, 다른 content(§8.4 step 5) | `version_id`(8자 prefix) 충돌을 강제 monkeypatch로 재현 | exit 3, 기존 `versions/<version_id>` 불변 |

100회 정상 build->activate/rollback 반복 테스트(`tests/integration/test_index_lifecycle_stress.py`)
에서 매 회 `CURRENT` 내용이 유효한 `versions/` 항목을 가리키는지, 부분
pointer(존재하지 않는 version_id 참조)가 0건인지, `qna_rag_index_version_info`
series가 항상 1개인지(§5.4) assert한다.

### 8.6 서비스 startup 검증 — fail-closed, legacy 자동 폴백 제거 (M-02 대응)

**이전 설계의 문제:** `runtime/index/`가 없으면 manifest 검증 없이 기존
`runtime/vectorstore/index.pkl`을 자동 직접 로드했다 — 이는 (1) pickle
신뢰 경계를 다시 열고(REQ-008.2 위반), (2) 같은 문서의 readiness 정의(index
manifest 검증 완료가 ready 조건, §6.2)와 동시에 만족할 수 없는 모순이었다
(Review M-02).

**수정된 계약 — 자동 폴백 없음:** `rag_engine.RAGEngine._load_vectorstore()`
앞단에 `index/lifecycle.py::resolve_active_index()`를 추가한다:

1. `runtime/index/CURRENT`가 존재하면: 가리키는 `versions/<id>/manifest.json` 로드
   -> `schema_version` 지원 여부 확인 -> `content_digest`/`index.faiss`/`index.pkl`
   재해시 후 manifest 저장값과 비교 -> `embedding_model`/`normalize_embeddings`/
   `chunk_size`/`chunk_overlap`을 현재 `Settings`와 비교. 모두 통과하면 이
   경로의 파일을 로드.
2. `runtime/index/CURRENT`가 없거나, 있어도 위 검증 중 하나라도 실패하면
   —  **legacy `runtime/vectorstore/`를 절대 자동으로 읽지 않는다.**
   `IndexInvalidError`를 던지고 `RAGEngine.initialize()`는 `False`를
   반환한다(REQ 그대로), `/health/ready`는 `index_invalid`로 계속
   503이다(REQ-008.5 fail-closed, 프로세스는 죽지 않음).

**legacy 복구 경로는 명시적 두 가지뿐이다(REQ-006 M3 rollback 경로 보존과
양립):**

- **서비스 버전 rollback**: M4 서비스 자체를 M3 실행 방식(레거시
  `VECTORSTORE_PATH` 직접 로드 코드 경로가 존재했던 이전 배포/커밋)으로
  되돌리는 것 — 이것이 "M3 index 경로 지정으로 복구"의 실제 의미이며,
  **살아있는 M4 프로세스가 조건 없이 자동으로 taking하는 경로가 아니다.**
- **명시적 one-shot `import-legacy`**(§8.7): 운영자가 승인된 fingerprint를
  알고 있는 상태에서 실행해 manifest/version을 생성한 뒤, 별도
  `activate`로 명시적으로 승격해야만 M4가 그 index를 읽는다. 이 경로를
  거치기 전에는 M4가 legacy pickle을 절대 로드하지 않는다.

### 8.7 legacy import — 승인된 fingerprint와 경로 경계를 강제하는 trust boundary (M-02/M2-04/M3-04 대응)

**이전 설계의 문제(Review M2-04, M3-04로 재확인):** `import-legacy --from
<path>`가 받은 두 파일을 CLI 스스로 hash해 그 값으로 manifest를 채웠다 —
자체 계산한 hash는 무결성 표식일 뿐 "운영자가 승인한 fingerprint"라는
승인 증거가 아니다. Iteration 2가 추가한 `--expected-faiss-sha256`/
`--expected-pkl-sha256` CLI 인자도 **같은 문제를 해소하지 못했다** —
공격자나 잘못된 자동화가 파일과 그 파일의 hash를 **같은 명령 호출에서
함께** 공급하면 두 값은 항상 일치하므로, "인자로 기대 hash를 받는다"는
사실 자체는 provenance를 전혀 입증하지 않는다(Review M3-04 근거). 또한
`O_NOFOLLOW`는 열리는 **마지막 경로 요소만** 보호할 뿐, 그 이전의
`resolve()`/`realpath()` 검사와 실제 `os.open()` 호출 사이에 **부모
디렉터리 자체**가 rename/symlink로 교체되면 open은 새 부모 아래의
동명 파일을 아무 경고 없이 연다 — "TOCTOU 창을 없앤다"는 이전 설계의
주장은 parent component에 대해 성립하지 않았다.

**수정된 계약 — 세 가지 축을 모두 activate 이전에 강제한다: (A) 승인
source를 CLI 인자가 아니라 git-committed 파일 하나로 고정, (B) 승인 root부터
목표 파일까지 **모든** path component를 dir_fd 기반 `openat`으로
내려가 parent-swap TOCTOU까지 닫음, (C) 열린 fd의 `fstat()`으로
owner/mode/inode를 검증.**

```
simple-qna-rag-index-lifecycle import-legacy --from runtime/vectorstore
```

**(A) 승인 hash의 단일 원본 — `evaluation/baselines/m3_initial.json`
(git-committed, code review를 거친 파일).** 이 파일은 M3 완료 시점에
이미 커밋됐고 `.reproducibility.vectorstore_fingerprint.index_faiss_sha256`/
`.index_pkl_sha256` 두 필드에 M3가 승인한 `index.faiss`/`index.pkl`의
SHA-256을 담고 있다(실측 확인 — 현재 저장소의 실제 값:
`index_faiss_sha256=c52fb288f3bc780d681ab68d62dd5b4c545843ee9dc53ef5528bd50db4d69820`,
`index_pkl_sha256=3f7217a2825f141bbea4c75dbf2b15738c3bcd50e3c0900e25a6a4a7ed91bb00`).
`import-legacy`는 이제 hash를 CLI 인자로 **전혀 받지 않는다** — 대신
`index/lifecycle.py::_M3_APPROVED_FAISS_SHA256`/`_M3_APPROVED_PKL_SHA256`
두 모듈 상수(구현 시점에 위 커밋된 JSON 값을 그대로 복사해 하드코딩,
바로 위 주석에 원본 JSON 경로/필드를 남긴다)와 비교한다. 이렇게 하면
승인 hash는 **git 커밋 히스토리를 통해서만** 바뀔 수 있다 — 런타임
파일시스템을 조작할 수 있는 공격자가 "파일+그 파일의 hash"를 함께
공급해 검증을 우회하는 이전 경로 자체가 사라진다(그 값이 CLI 인자도,
런타임에 읽는 별도 파일도 아니라 배포된 코드 자체에 박혀 있으므로).
**신선도 보증:** `tests/unit/test_legacy_import_approved_hash_matches_baseline.py`가
전체 저장소 checkout이 있는 dev/CI 환경에서 `evaluation/baselines/m3_initial.json`을
다시 읽어 그 두 필드가 `index/lifecycle.py`의 하드코딩 상수와 **바이트
단위로 일치**하는지 assert한다 — baseline 파일이 갱신됐는데 코드
상수를 잊고 안 바꾸면 이 테스트가 즉시 실패해 drift를 잡는다. 이
approval 상수는 `src/`에 있으므로 최소 production 이미지(§9.1,
`evaluation/`을 포함하지 않는다)에도 항상 존재한다 — `import-legacy`
실행에 `evaluation/` 디렉터리가 배포 환경에 있을 필요가 없다.
2. **승인된 root 아래 정규화된 고정 파일명만 허용(경로 containment) —
   dir_fd 기반 openat으로 parent component까지 전부 보호(M3-04 핵심
   수정).** `approved_legacy_import_root`(기존과 동일, 기본
   `runtime/vectorstore`, `index/lifecycle.py`의 모듈 상수)를 먼저
   `root_fd = os.open(approved_legacy_import_root, os.O_RDONLY |
   os.O_DIRECTORY)`로 연다. `--from`이 이 root와 다른 경로를 가리키면
   `--from`부터 root까지의 **각 중간 component**를 문자열 경로 재해석 없이
   `os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
   dir_fd=parent_fd)`로 한 단계씩 내려가며 매번 새 `parent_fd`를 얻는다
   (표준 라이브러리가 지원하는 `dir_fd` 인자 — `os.open`/`os.stat` 계열이
   POSIX `openat(2)`/`fstatat(2)`로 위임한다). **이 방식은 "이전에
   resolve()로 확인한 경로 문자열을 다시 여는" 단계 자체가 없으므로,
   확인과 open 사이에 부모 디렉터리가 rename/symlink로 교체되는 TOCTOU
   창이 구조적으로 존재하지 않는다** — 매 단계가 직전 단계에서 이미 연
   fd를 기준으로만 다음 이름을 찾으므로, 공격자가 심볼릭 경로 문자열을
   아무리 바꿔도 이미 열린 fd가 가리키는 실제 inode는 바뀌지 않는다.
   `--from`이 root 자신이거나 root의 직계 하위가 아니면(예: `..`을
   포함하거나 root 바깥을 가리키면) 첫 component부터 열기 실패 -> 즉시
   exit 3, 어떤 파일도 열지 않는다. 최종 디렉터리 fd 안에서도 정확히
   `index.faiss`/`index.pkl` 두 고정 파일명만 `os.open(name, os.O_RDONLY |
   os.O_NOFOLLOW, dir_fd=final_dir_fd)`로 연다 — 임의 파일명이나 다른
   확장자를 허용하지 않는다.
3. **symlink는 각 component에서 즉시 거부되고, `os.path.realpath()`
   비교는 더 이상 쓰지 않는다.** 이전 설계의 "`realpath()` 결과와 원본
   `Path` 비교" 규칙은 절대/상대 경로 정규화 형식이 정의되지 않아 정상
   경로까지 다르다고 오판할 수 있는 모호한 규칙이었다(Review M3-04).
   dir_fd 기반 `O_NOFOLLOW` open은 이 비교 자체를 불필요하게 만든다 —
   경로의 **어느 component든** symlink면(중간 디렉터리든 마지막
   파일이든) 그 단계의 `os.open(..., dir_fd=...)` 호출이 `ELOOP`로 즉시
   실패하므로, "심볼릭 링크가 있는지"를 별도로 판정할 필요 없이 open
   성공 자체가 "경로 전체에 symlink가 없었다"는 증명이 된다.
4. **`fstat()`으로 owner/mode/inode를 검증한다(M3-04 신규 — 이전에는
   경로 검사만 하고 소유자/권한을 전혀 확인하지 않았다).** root 및 각
   중간 디렉터리 fd, 그리고 최종 두 파일 fd 각각에 대해
   `st = os.fstat(fd, dir_fd=...)`(파일은 `os.fstat(fd)`)를 호출해:
   - 디렉터리는 `stat.S_ISDIR(st.st_mode)`, 파일은
     `stat.S_ISREG(st.st_mode)`(O_DIRECTORY/일반 open이 이미 강제하지만
     방어적으로 재확인),
   - `st.st_uid`가 현재 프로세스의 실효 UID(`os.geteuid()`, 즉 이
     lifecycle CLI를 실행하는 운영자 계정)와 같은가 — 다른 uid가 소유한
     디렉터리/파일은 "운영자가 승인"한 것으로 볼 수 없으므로 거부,
   - `st.st_mode & (stat.S_IWGRP | stat.S_IWOTH) == 0` — group/other
     write 권한이 있으면(다른 계정이 내용을 바꿀 수 있었다는 뜻) 거부.
   하나라도 위반하면 즉시 exit 3, 다음 component/파일을 열지 않는다.
   `st.st_ino`/`st.st_dev`(파일의 경우)는 manifest의 `source_inode`
   보조 필드로 기록해 감사 로그에 남긴다(승인 판정 자체에는 쓰지 않음 —
   inode는 재사용될 수 있으므로 hash가 유일한 content 판정 기준이다).
5. **hash 불일치는 deserialize 이전에 실패한다.** 4번까지 통과한 fd로
   읽은 바이트의 SHA-256이 (A)의 하드코딩 승인 상수와 일치하는지
   **`staging/`에 쓰기 전에** 먼저 검사한다 — 불일치 시 즉시 exit 3,
   `staging/`에 어떤 파일도 만들지 않고 `FAISS.load_local`을 호출하지
   않는다. 일치해야만 `staging/<staging_id>/`(§8.3 — build와 동일한
   위치, `versions/`에 바로 쓰지 않는다)에 복사하고,
   `index_faiss_sha256`/`index_pkl_sha256`를 그 확인된 값으로 채우고
   `embedding_model_revision="unknown_legacy"`, `source="legacy_import"`로,
   §8.2의 `content_digest`/`version_id`를 계산해 완성된 `manifest.json`을
   같은 staging에 쓴다.

**staging owner/mode 재검증(M3-04 신규 — §8.4 activate 3단계 확장).**
import(또는 build)와 activate 사이에 `staging/<staging_id>/`의 소유자/권한이
바뀔 수 있다는 이전 설계의 공백을 닫는다. §8.4의 manifest 재검증
단계에서 `staging/<staging_id>/` 디렉터리와 그 안의 `index.faiss`/
`index.pkl`을 **여기서도 dir_fd 기반으로 다시 연다**(문자열 경로
재오픈이 아니라, activate가 시작한 시점에 `runtime/index/staging/`을
연 root fd로부터 `staging_id` 하나만 내려가는 짧은 openat 체인 — staging
트리는 root 바로 아래 1단계뿐이므로 §8.7 3~4번과 동일한 fstat
owner/mode 검사를 그대로 적용한다). 이 검사가 실패하면(예: import 직후
누군가 staging 파일의 권한을 world-writable로 바꿔 두었다면) exit 3로
activate 자체를 거부한다 — "import 시점에는 안전했다"는 사실이 "activate
시점에도 안전하다"를 보장하지 않는다는 것을 명시적으로 재확인하는
단계다.

**자동으로 activate하지 않는다** — 별도
`simple-qna-rag-index-lifecycle activate --staging-id <staging_id>`
호출로만 `versions/`에 승격되며(§8.4, build 산출물과 완전히 동일한
경로), activate 3단계의 manifest/hash 재검증(§8.4 step 3, 위 staging
owner/mode 재검증 포함)이 여기서 이미 확인한 값을 다시 한번
재확인한다(신뢰 경계가 두 지점에서 이중으로 성립). import 직후 자체
activate 옵션(`--activate` flag, 기본 False)을 제공해 한 명령으로
승격까지 수행할 수도 있으나, 내부적으로는 동일한 `activate(staging_id)`
함수를 호출한다(코드 경로 중복 없음).

**결정론적 테스트(신규 `tests/integration/test_import_legacy_trust_boundary.py`,
M2-04/M3-04 수정안 대응):** 아래 각 시나리오에서 `FAISS.load_local` mock 호출
횟수가 **0**임을 assert한다(활성 pickle에 도달하기 전에 항상 차단됨을
직접 증명):

| 시나리오 | 강제 방법 | 기대 |
|---|---|---|
| 승인 root 밖 임의 경로 | `--from /tmp/attacker-controlled` | 첫 openat component부터 실패 -> exit 3, staging 미생성 |
| 최종 파일 symlink escape | `runtime/vectorstore/index.pkl`을 root 밖 파일을 가리키는 symlink로 교체 | 최종 `os.open(..., O_NOFOLLOW, dir_fd=...)`이 `ELOOP`로 실패 -> exit 3 |
| **parent 디렉터리 symlink escape(M3-04 신규 — 이전 설계가 막지 못하던 경로)** | `runtime/vectorstore` 자체(또는 root와 대상 파일 사이의 중간 디렉터리)를 root 밖을 가리키는 symlink로 교체 | root/중간 component openat 단계에서 `O_NOFOLLOW`가 `ELOOP`로 실패 -> exit 3, 최종 파일에 도달하지 못함 |
| **parent 디렉터리 TOCTOU 교체(M3-04 신규)** | 승인 root를 먼저 정상적으로 연 뒤(테스트가 root fd를 쥐고 있는 상태), 별도 프로세스가 실제 파일시스템의 `runtime/vectorstore` 디렉터리 엔트리를 다른 디렉터리로 rename 교체 | 이미 연 `root_fd`는 원래 디렉터리의 inode를 계속 가리키므로(POSIX 계약) 그 이후 `dir_fd=root_fd`로 여는 모든 openat이 원래 디렉터리 기준으로 완결되고, 교체된 새 디렉터리의 내용은 이번 실행에 전혀 영향을 주지 않음 |
| **owner 불일치(M3-04 신규)** | 대상 파일/디렉터리의 소유자를 현재 프로세스 UID가 아닌 다른 값으로 monkeypatch(`os.fstat` 결과 mock) | `fstat` owner 검사 실패 -> exit 3 |
| **world-writable 권한(M3-04 신규)** | 대상 디렉터리 mode에 `S_IWOTH` 설정 | `fstat` mode 검사 실패 -> exit 3 |
| 기대 hash 불일치 | 하드코딩된 승인 상수와 다른 내용으로 대상 파일을 변조 | exit 3, staging 미생성, `FAISS.load_local` 0회 |
| **승인 상수-baseline drift(M3-04 신규, 별도 단위 테스트)** | `evaluation/baselines/m3_initial.json`의 hash 필드를 임시로 변경 | `test_legacy_import_approved_hash_matches_baseline.py`가 실패 — 코드 상수와 committed baseline이 어긋났음을 CI에서 즉시 검출 |
| **staging owner/mode 재검증(M3-04 신규)** | 정상 import 이후 activate 직전 `staging/<staging_id>/index.faiss`의 mode를 world-writable로 변경 | activate가 §8.4 3단계 확장 검사에서 exit 3, `versions/`/`CURRENT` 불변 |
| 정상 승인 경로 | 올바른 root/파일명/hash/owner/mode | exit 0, staging에 manifest 생성, activate 전까지 `FAISS.load_local` 0회 |

## 9. Phase 6 — container, security, runbook

### 9.0 test-only dependency injection (M-06 대응 — production settings/env backdoor 제거)

이전 설계는 `SIMPLE_QNA_RAG_MOCK_ENGINE` 환경변수로 프로덕션과 동일한
entrypoint/이미지에서 엔진을 mock으로 바꿨다 — 이는 production 환경에서도
같은 이름의 env var 하나로 실제 RAG 엔진을 우회할 수 있는 숨은 backdoor였다
(Review M-06). 대신 **두 계층**으로 분리한다:

1. **앱 팩토리 DI(코드 계층):** `web/server.py`가
   `create_app(engine_provider: Callable[[], RAGEngine] | None = None) -> FastAPI`를
   export한다. `engine_provider=None`(기본, 프로덕션 entrypoint
   `cli/web.py`가 항상 이렇게 호출)이면 기존 `get_rag_engine()` singleton
   경로를 그대로 쓴다. `Settings`나 환경변수 어디에도 "mock으로 전환"
   플래그가 없다 — DI는 오직 `create_app()`을 직접 호출하는 코드에서만
   선택된다.
2. **CI 전용 image stage(빌드 계층):** `deploy/Dockerfile`에 프로덕션
   `runtime` 스테이지와 완전히 분리된 `test` 스테이지를 추가한다. `test`
   스테이지만 `src/simple_qna_rag/testing/`(신규,
   `MockRAGEngine.initialize()->True`, `.query()->` 고정 canned 응답)과
   `cli/web_testonly.py::main()`(`create_app(engine_provider=lambda:
   MockRAGEngine())` 호출 후 uvicorn 구동, 콘솔 스크립트
   `simple-qna-rag-web-testonly`)을 포함한다. **프로덕션 `runtime`
   스테이지(기본 build target)는 `testing/` 패키지와 `web_testonly.py`를
   소스 수준에서 전혀 포함하지 않는다**(§9.1 builder의 `COPY --exclude`) —
   진짜 trust boundary는 **이미지 태그(빌드 target) 선택**이다. `docker build`
   (target 미지정, 기본 `runtime`)로 만든 이미지는 mock 코드 자체가 파일로
   존재하지 않으므로 어떤 실행 인자를 줘도 mock 경로를 탈 수 없다. CI가
   `--target test`로 **별도로** 빌드한 이미지(`qna-rag:ci-test`, §9.4)만
   mock 코드를 포함하며, 이 이미지의 **기본 `ENTRYPOINT` 자체가 이미
   `simple-qna-rag-web-testonly`**다 — `--target test`를 선택하는 행위
   자체가 "mock을 쓰겠다"는 명시적 의사표시이므로, 그 이미지 안에서 다시
   `--entrypoint`를 요구하지 않는다(M2-05 — 이전 설계는 "test 스테이지가
   이미 mock ENTRYPOINT를 기본값으로 갖는다"는 Dockerfile의 실제 동작과
   "명시적 --entrypoint일 때만 mock을 쓴다"는 §9.0 서술이 서로 모순됐다.
   두 계층 중 실제 신뢰 경계는 이미지 태그이므로 이 서술을 그 하나로
   통일한다).

### 9.1 `deploy/Dockerfile` (M2-05 대응 — clean build/실제 site-packages 경로/정상 packaging)

**이전 설계의 다섯 가지 문제(M2-05에서 셋을 고쳤고, M3-05 spike/실행
검증으로 나머지 둘을 추가로 발견):** (1) `pip --target /opt/venv`는
패키지를 `/opt/venv/` 바로 아래에 평탄하게 설치하는데 test stage는 mock
파일을 `/opt/venv/lib/python3.11/site-packages/...`에 복사해
`PYTHONPATH`가 가리키는 경로와도, 실제 설치 경로와도 맞지 않았다. (2)
파일을 손으로 복사만 해서는 `simple-qna-rag-web-testonly` console
script(entry point)가 생성되지 않는다. (3) `COPY pyproject.toml README.md
LICENSE ./`가 root `.dockerignore`의 `*.md`와 충돌해 build 자체가
실패했다(§9.2에서 해결). **(4, M3-05 신규) `test` stage가 파일의 마지막
stage였다 — Docker는 `--target` 없이 빌드하면 항상 파일에서 마지막으로
정의된 stage를 만든다는 계약이 있으므로, `docker build ... -t
qna-rag:ci .`는 이전 설계 그대로라면 실제로는 mock 코드와
`simple-qna-rag-web-testonly` ENTRYPOINT를 포함한 test 이미지를
만들었을 것이다(Review M3-05, Docker 공식 문서화된 계약).** **(5, M3-05
신규) builder가 `requirements.txt`를 build context에서 전혀 COPY하지
않는다** — 현재 `pyproject.toml`을 실측 확인하면
`[tool.setuptools.dynamic] dependencies = { file = ["requirements.txt"] }`로
선언돼 있어, `pip install --no-deps .`가 `--no-deps`로 실제 의존성
설치는 건너뛰더라도 setuptools가 PEP 517 메타데이터를 만들려면 그
시점에 `requirements.txt` 파일이 **반드시 디스크에 존재**해야 한다 —
lock 파일만 다른 basename(`lock-linux-py311.txt`)으로 복사하고
`requirements.txt` 자체는 복사하지 않았으므로, 이전 설계 그대로라면
`pip install --no-deps .`가 메타데이터 생성 단계에서
`FileNotFoundError`로 실패했을 것이다.

아래는 (1)(2)를 **표준 venv + `pip install`**로, (3)을 `.dockerignore`의
`!README.md` 예외로, (4)를 **명시적 `production` stage를 파일의 진짜
마지막 stage로 둠**으로써, (5)를 **`requirements.txt`를 builder에도
COPY**함으로써 해결한다.

```dockerfile
# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS builder
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements/lock-linux-py311.txt .
RUN pip install --require-hashes -r lock-linux-py311.txt
# M3-05 수정 5 — pyproject.toml의 `[tool.setuptools.dynamic]
# dependencies = { file = ["requirements.txt"] }`가 이 파일을 요구한다.
# 실제 의존성 설치는 위 lock 파일로 이미 끝났으므로 아래 `pip install
# --no-deps .`는 이 파일 내용을 재설치에 쓰지 않지만, PEP 517 메타데이터
# 생성 자체가 이 파일의 존재를 전제한다 — 없으면 빌드가 실패한다.
COPY requirements.txt pyproject.toml README.md LICENSE ./
# runtime 대상에는 testing/ 서브패키지와 web_testonly.py 소스를 절대 포함하지
# 않는다(M-06 trust boundary) — BuildKit COPY --exclude로 명시적으로 뺀다.
COPY --exclude=src/simple_qna_rag/testing --exclude=src/simple_qna_rag/cli/web_testonly.py \
     src/ src/
RUN pip install --no-deps .

FROM python:3.11-slim AS runtime
RUN groupadd -g 10001 qnarag && useradd -u 10001 -g qnarag -m -s /usr/sbin/nologin qnarag
ENV PATH=/opt/venv/bin:$PATH
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY web/ /app/web/
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health/live', timeout=2)" || exit 1
ENTRYPOINT ["simple-qna-rag-web"]
CMD ["--host", "0.0.0.0", "--port", "8000"]

# CI 전용 — 프로덕션 이미지 태그에는 절대 포함되지 않는다(아래 production
# stage가 파일의 마지막 stage이므로 --target 없는 build는 항상 production을
# 만든다 — M3-05 수정 4). `docker build --target test`로 명시적으로만
# 빌드된다(§9.4). builder의 venv(기본 의존성만 설치된 상태)를 기반으로
# testing/ 소스를 다시 COPY하고 **`pip install .`을 재실행**한다 — 이번에는
# testing/과 web_testonly.py가 모두 포함된 채로 정상 설치되므로
# `pyproject.toml`의 `[project.scripts]`가 `simple-qna-rag-web-testonly`
# launcher를 실제로 생성한다(M2-05 수정안 — 손으로 파일만 복사하던 이전
# 방식 대체).
FROM builder AS test-builder
COPY src/simple_qna_rag/testing/ src/simple_qna_rag/testing/
COPY src/simple_qna_rag/cli/web_testonly.py src/simple_qna_rag/cli/web_testonly.py
RUN pip install --no-deps --force-reinstall .

FROM runtime AS test
USER root
COPY --from=test-builder /opt/venv /opt/venv
USER 10001:10001
# --target test로 빌드를 선택하는 것 자체가 명시적 의사표시이므로, 이 이미지
# 안에서 다시 --entrypoint override를 요구하지 않는다(§9.0 정책 통일).
ENTRYPOINT ["simple-qna-rag-web-testonly"]

# M3-05 수정 4 — 파일의 **진짜 마지막 stage**. `runtime`을 그대로
# re-tag하는 것 외에 아무 것도 하지 않는다(레이어 추가 없음, `runtime`과
# byte-identical 이미지). `test`/`test-builder`보다 파일에서 뒤에 오는
# 이유는 오직 하나 — Docker BuildKit은 `--target`이 없으면 파일에서
# **마지막으로 선언된** stage를 기본 산출물로 삼기 때문에, 이 stage가
# 파일의 끝에 있어야 "그냥 `docker build .`"가 항상 production을
# 만든다는 보장이 stage 나열 순서와 무관하게 실수로 깨지지 않는다.
# 그럼에도 §9.3/§9.4/§10.3의 모든 production build 명령은 방어적으로
# `--target production`을 **명시**한다(m3-05: "final stage production
# 또는 모든 prod build --target 강제" 중 이 설계는 둘 다 채택한다).
FROM runtime AS production
```

### 9.2 `.dockerignore`(저장소 루트, M-06 대응 — 경로 정정)

`docker build -f deploy/Dockerfile ... .`와 compose의 `context: ..`(§9.3)는
build context를 **저장소 루트**로 지정한다. Docker/BuildKit이 이 build에서
실제로 읽는 ignore 파일은 **build context 루트의 `.dockerignore`**이거나
BuildKit이 지원하는 Dockerfile 전용 `deploy/Dockerfile.dockerignore`(Dockerfile과
같은 디렉터리)뿐이다 — 이전 설계처럼 `deploy/.dockerignore`에만 파일을
두면 이 build에서 전혀 적용되지 않는다(Review M-06, 실제로 daemon에
`runtime/`, `.env`, 리포트가 그대로 전송될 위험). 따라서 파일은 **저장소
루트**에 `.dockerignore`로 둔다:

```
# /.dockerignore (저장소 루트)
runtime/
.env
.env.*
.git
.github
.idea
.claude
build/
venv/
__pycache__/
*.pyc
evaluation/reports/
models/
node_modules/
docs/
*.md
!README.md
```

`*.md`로 모든 Markdown을 막은 뒤 `!README.md`로 그 파일 하나만 다시
포함한다 — Docker/BuildKit의 `.dockerignore`는 뒤에 나오는 부정(`!`) 패턴이
앞선 패턴보다 우선하므로, root의 `README.md`(builder가
`COPY pyproject.toml README.md LICENSE ./`로 요구하는 파일, §9.1)만 build
context에 전송되고 `docs/`나 하위 디렉터리의 다른 `*.md`는 여전히
제외된다(M2-05 — 이전 설계는 이 예외가 없어 clean build가 COPY 단계에서
바로 실패했다).

### 9.3 `deploy/docker-compose.yml`(loopback 기본 profile)

```yaml
services:
  qna-rag:
    build:
      context: ..
      dockerfile: deploy/Dockerfile
      target: production   # M3-05 — §9.1의 명시적 마지막 stage를 방어적으로도 고정 지정
    user: "10001:10001"
    read_only: true
    tmpfs: ["/tmp"]
    cap_drop: ["ALL"]
    security_opt: ["no-new-privileges:true"]
    environment:
      SIMPLE_QNA_RAG_OLLAMA_BASE_URL: "http://host.docker.internal:11434"
    volumes:
      - ../runtime/documents:/app/runtime/documents:ro
      - ../runtime/index:/app/runtime/index:ro
      - ../models:/app/models:ro
    ports: ["127.0.0.1:8000:8000"]
    mem_limit: "6g"
    cpus: "2.0"
```

index build/activate는 별도 쓰기 volume을 마운트하는 **다른 compose
서비스/일회성 `docker run --entrypoint simple-qna-rag-index-lifecycle`**로
분리한다 — serving 컨테이너는 항상 `runtime/index`를 read-only로 마운트한다
(REQ-009.2).

### 9.4 CI container job (`.github/workflows/ci.yml` 신규 job `container`, M-06/M3-05 대응)

```yaml
container:
  runs-on: ubuntu-latest
  permissions:
    contents: read
    actions: write        # upload-artifact
  steps:
    - uses: actions/checkout@v4
    - name: Build production image
      # M3-05 수정 4 — §9.1의 마지막 stage(production)에 의존하지 않고
      # --target을 명시해 이중으로 고정한다.
      run: docker build -f deploy/Dockerfile --target production -t qna-rag:ci .
    - name: Build CI-only test image (mock DI, §9.0)
      run: docker build -f deploy/Dockerfile --target test -t qna-rag:ci-test .
    - name: Non-root UID check
      # ENTRYPOINT override 없이 `qna-rag:ci id -u`를 실행하면 `id -u`가
      # 기본 ENTRYPOINT(simple-qna-rag-web)의 인자로 전달되어 "id -u" 자체가
      # 실행되지 않는다 — --entrypoint로 명시 override해야 한다(Review M-06).
      run: test "$(docker run --rm --entrypoint id qna-rag:ci -u)" = "10001"
    - name: Import/config check (no live Ollama, REQ-009.4)
      run: |
        docker run --rm --entrypoint python qna-rag:ci -c \
          "import simple_qna_rag; import simple_qna_rag.web.server"
        docker run --rm qna-rag:ci --check-config
    - name: Read-only rootfs + drop-caps + mock readiness/liveness smoke
      # m3-01 MINOR 대응 — `docker exec` 중 하나가 실패해도 `docker stop`에
      # 반드시 도달하도록 `trap`으로 정리하고, 고정 `sleep 5` 대신 준비될
      # 때까지 bounded poll한다(고정 sleep은 느린 러너에서 flaky하고
      # 빠른 러너에서 불필요하게 느리다).
      run: |
        trap 'docker rm -f smoke >/dev/null 2>&1 || true' EXIT
        docker run --rm --read-only --tmpfs /tmp --cap-drop=ALL \
          --security-opt no-new-privileges:true \
          -d --name smoke qna-rag:ci-test
        for i in $(seq 1 30); do
          if docker exec smoke python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health/live', timeout=1)" 2>/dev/null; then
            break
          fi
          if [ "$i" -eq 30 ]; then
            echo "::error::container did not become live within 30x1s poll"
            exit 1
          fi
          sleep 1
        done
        docker exec smoke python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health/ready', timeout=2)"
        docker stop smoke
    - name: Layer content scan (production image only, M2-05/M3-05)
      run: |
        docker save qna-rag:ci -o /tmp/img.tar
        mkdir -p /tmp/img && tar -xf /tmp/img.tar -C /tmp/img
        python scripts/scan_image_layers.py /tmp/img \
          --forbidden-path-prefix runtime/documents/ \
          --forbidden-path-prefix runtime/vectorstore/ \
          --forbidden-path-prefix evaluation/reports/ \
          --forbidden-path-prefix .git/ \
          --forbidden-exact .env \
          --known-secret-file tests/fixtures/known_secret_canary.bin \
          --workdir-prefix app
    - name: Write container evidence and gate result JSON (M3-05 신규 — 이전
        설계는 이 job이 layer scan에서 끝나 write_evidence/artifact upload가
        없었다, Review M3-05)
      if: always()
      run: |
        python -m evaluation.m4_evidence write --gate container \
          --exit-code "${{ job.status == 'success' && '0' || '1' }}" \
          --ci-commit-sha "${{ github.sha }}" \
          --ci-run-id "${{ github.run_id }}" \
          --ci-run-attempt "${{ github.run_attempt }}" \
          --image-digest "$(docker inspect --format='{{.Id}}' qna-rag:ci)" \
          --candidate-id "${{ env.M4_CANDIDATE_ID || 'm4-final' }}" \
          --evidence-dir evaluation/reports/m4/${{ env.M4_CANDIDATE_ID || 'm4-final' }}/evidence
    - name: Upload container evidence artifact (M3-05 신규)
      if: always()
      uses: actions/upload-artifact@v4
      with:
        # §10.1a가 이미 정의한 고정 이름 규칙 — Phase 7 통합 단계가 이
        # 정확한 이름으로만 다운로드하므로, 이름이 다른 실행의 artifact를
        # 조용히 집어올 수 없다.
        name: m4-container-evidence-${{ env.M4_CANDIDATE_ID || 'm4-final' }}-${{ github.run_id }}
        path: evaluation/reports/m4/${{ env.M4_CANDIDATE_ID || 'm4-final' }}/evidence/container.json
        retention-days: 14
```

**`scripts/scan_image_layers.py`(신규, Phase 6) — outer archive를 풀고 layer
단위로 실제 내용을 검사한다(M2-05, 경로 정규화는 M3-05 신규):** 이전
설계의 `docker save ... | tar -xO > /tmp/img.tar`는 outer OCI/Docker save
archive(JSON, manifest, 각 layer의 압축 tar가 이어붙은 형태)를 유효한
단일 tar처럼 다뤄 `tar -tf`로 읽으려 한 것이라 layer 내용을 신뢰성 있게
파싱하지 못했고, 파일 **경로**만 grep했을 뿐 known-secret **바이트
내용**은 전혀 검사하지 않았다. 수정된 스크립트는 (1) `docker save -o`로
받은 outer archive를 통째로 풀어 `manifest.json`에서 실제 layer tar 목록을
읽고, (2) **각 layer tar를 개별적으로** `tarfile.open()`으로 열어 모든
멤버를 순회하며 `--forbidden-path-prefix`/`--forbidden-exact` 목록과
대조하고, (3) 일반 파일은 내용을 읽어 `--known-secret-file`(build context에
심어 둔 canary 바이트 fixture)과 바이트 단위로 비교한다. layer 단위로
따로 검사하는 이유는 OverlayFS의 특성상 어떤 파일이 뒤 layer에서
"삭제"돼도 앞 layer의 tar 안에는 원본 바이트가 그대로 남기 때문이다 —
최종 파일시스템만 보는 검사는 이런 "삭제됐지만 layer에는 남은" 유출을
놓친다. 위반이 하나라도 있으면 위반 목록(layer id, 멤버 경로)을 출력하고
0이 아닌 코드로 종료하며, 이 실패 자체가 `evidence/container.json`
(§10.1a)에 그대로 기록된다.

**경로 canonicalization 계약(M3-05 신규 — 이전에는 정의되지 않았다).**
`tar` 아카이브 멤버 경로는 보통 leading slash 없이 저장되고(예:
`app/runtime/documents/secret.txt`), `WORKDIR /app`(§9.1)이 앞에
붙는다 — `--forbidden-path-prefix runtime/documents/`를 있는 그대로
비교하면 실제 유출 경로 `app/runtime/documents/...`를 놓친다(Review
M3-05). `scan_image_layers.py`는 각 멤버 이름에서 선행 `./`와 `/`를
제거한 뒤, **정규화된 이름 자체**와 **`--workdir-prefix`(기본 `app`,
§9.1 `WORKDIR`과 반드시 일치) + `/` + 정규화된 이름** 두 형태 모두를
`--forbidden-path-prefix`/`--forbidden-exact` 목록과 대조한다 — 어느
한쪽이라도 일치하면 위반으로 기록한다. 이렇게 하면 forbidden prefix
목록 자체는 `WORKDIR`을 신경 쓰지 않고 저장소 상대 경로로만 표현할 수
있다.

**positive/negative control(M3-05 신규 — 이전에는 CI가 실제 이미지를
스캔해 "위반 없음"을 확인할 뿐, 스캐너가 실제로 위반을 찾아낼 수
있다는 것 자체를 증명하지 않았다).** 신규
`tests/unit/test_scan_image_layers.py`가 Docker 빌드 없이 `tarfile`로
직접 만든 두 synthetic layer로 스캐너 함수를 단위 테스트한다: (1)
**positive control** — `app/runtime/documents/secret.txt`(forbidden
prefix에 걸림)와 `--known-secret-file` fixture와 바이트가 동일한
`app/leftover.bin`을 담은 layer tar를 스캔하면 **반드시 실패**(0이 아닌
종료 코드, 두 위반이 모두 결과에 포함)해야 한다는 것을 assert한다 — 이
테스트가 실패하면(즉 스캐너가 이 명백한 위반을 놓치면) 스캐너 자체가
고장났다는 뜻이므로 CI가 이 케이스를 잡아야 한다. (2) **negative
control** — 정상 파일(`app/web/static/style.css` 등, forbidden 목록과
무관하고 known-secret과 다른 내용)만 담은 clean layer tar를 스캔하면
**반드시 성공**(exit 0, 위반 0건)해야 한다. 이 두 케이스가 함께 있어야
"스캐너가 아무것도 위반으로 잡지 않는 것"이 "이미지가 깨끗해서"인지
"스캐너가 고장나서 항상 통과시켜서"인지 구분할 수 있다(§9.4 CI job의
실제 이미지 스캔은 매번 클린 결과만 나오므로 그 자체로는 스캐너의
탐지 능력을 증명하지 못한다 — 이 unit test가 그 증명을 담당한다).

`qna-rag:ci`(`--target production` 명시, §9.1)는 항상 실제
`create_app()`(mock provider 없음)을 쓴다 — CI가 readiness/liveness를
mock으로 확인해야 하는 단계에서만 별도 빌드한 `qna-rag:ci-test`
(`--target test`, §9.0)를 쓰고, 그 이미지의 layer/실행 경로는 production
`qna-rag:ci` 태그와 완전히 분리되어 있으므로 layer content scan은
production 이미지만 검사한다. `--check-config`는 `Settings.load()`만
수행하고 모델/Ollama import가 없으므로 기본 `ENTRYPOINT`
(`simple-qna-rag-web`)에 인자로 그대로 전달해도 안전하다(entrypoint
override 불필요 — `id`처럼 다른 바이너리를 실행할 때만 override가
필요하다). 실제 Ollama/model image test는 별도 `workflow_dispatch` 전용
`container-live` job으로 분리한다.

### 9.5 `docs/operations/Runbook.md` 목차(Phase 6 산출물, 여기서는 목차만 확정)

```
1. 사전 준비 (model preflight: ollama pull gpt-oss:20b, BGE-M3/reranker 캐시 확인)
2. host 실행 (venv + lock 설치 + --check-config)
3. container 실행 (compose up, healthcheck 확인)
4. index 운영 (build/activate/rollback/retention 명령, 쓰기 volume 분리 이유)
5. graceful stop (SIGTERM -> draining -> grace 30s -> STOPPED)
6. backup/restore (runtime/index/versions, runtime/documents tar)
7. 로그/메트릭 접근 (loopback /metrics, JSON 로그 grep 예시)
8. 공개 노출 profile (reverse proxy TLS/auth/rate limit 필수 조건)
9. incident triage 표: 증상 -> 원인 후보 -> 확인 명령 -> 조치
```

## 10. Phase 7 — clean 통합 검증과 M4 baseline

### 10.1 `evaluation/m4_fingerprint.py`

`evaluation/fingerprint.py`(dataset/corpus/index.faiss/index.pkl)를 그대로
호출하고 다음을 추가 수집한다: `settings_hash`(`Settings.load().sha256()`,
§4.3a — redacted 값 기준), `lock_sha256`(`requirements/lock-linux-py311.txt`
파일 hash), `dependency_versions`(`importlib.metadata`로 핵심 package 정렬
목록 + canonical SHA-256), `index_manifest_sha256`(활성 버전 manifest.json
**파일 바이트** 자체의 hash — manifest 내부 필드가 아니다, §8.2), `image_digest`
(container 실행 시 `docker inspect --format='{{.Id}}'`, host 실행 시 null),
`host_info`(`os.cpu_count()`, `platform.uname()`, `/proc/meminfo`의
`MemTotal` 파싱 — 새 의존성 없이 stdlib만 사용), `worker_config`
(`concurrency_limit`, `queue_limit`, 단일 uvicorn worker 고정),
`warmup_cases`, `profile`(`"mock"|"live"`).

### 10.1a gate evidence schema — 공통 `evidence.json` 계약 (M-07 대응)

**문제였던 부분:** 이전 설계는 gate 데이터 소스를 "pytest 결과", "CI 로그
exit code"처럼 서술적으로만 매핑해 파일 형식·경로·schema·SHA-256·
candidate/profile 일치 검증을 정의하지 않았다. 최종 검증 명령은 각 도구를
개별 shell 명령으로 실행할 뿐 그 결과를 `m4_gate.py`가 읽을 artifact로
저장하지 않아, `m4_gate.py`가 실제로 앞선 명령이 수행됐는지 확인할 방법이
없었다(Review M-07).

**해결 — 모든 gate runner가 공통 `evidence.json`을 원자적으로 쓴다.**
신규 `evaluation/m4_evidence.py::write_evidence(...)`가 유일한 작성
경로다:

```python
# evaluation/m4_evidence.py
class UnsafeArtifactPathError(Exception):
    """result_artifact_path가 candidate root containment 검사를 통과하지
    못하면 던진다(M2-06/M3-06 — write와 read 양쪽에서 동일 함수로 검사)."""

class ResultBindingMismatchError(Exception):
    """result JSON 내부의 candidate/run/gate binding이 evidence wrapper의
    값과 다르면 던진다(M3-06 신규 — 아래 참조)."""

def _open_contained(result_artifact_path: str, root_fd: int) -> int:
    """§10.1b M2-06/M3-06 대응 — evidence가 다른 실행의 artifact를 참조하지
    못하게 한다. 이전 구현은 `(candidate_root / path).resolve()`한 뒤
    `resolved.is_symlink()`를 검사했는데, `Path.resolve()`는 경로의 모든
    symlink를 이미 따라가 최종 실제 경로로 바꾸므로 그 결과에 대한
    `is_symlink()`는 사실상 항상 False다(죽은 코드, Review M3-06 실측
    확인) — symlink를 탐지한 적이 없었다는 뜻이다. 이 함수는 M3-04의
    legacy import와 동일한 패턴으로 **dir_fd 기반 openat**을 써서 이
    결함을 근본적으로 없앤다: `result_artifact_path`의 각 component를
    `candidate_root`를 연 `root_fd`로부터 한 단계씩
    `os.open(name, os.O_RDONLY | (os.O_DIRECTORY if not last else 0) |
    os.O_NOFOLLOW, dir_fd=parent_fd)`로 내려가며 새 fd를 얻는다 — 어떤
    component든 symlink면 그 단계의 open이 `ELOOP`로 즉시 실패한다.
    `result_artifact_path`가 절대 경로이거나 `..` component를 포함하면
    애초에 component 분해 단계에서 거부한다(문자열 정규화에 의존하지
    않고 `Path(...).parts`에 `os.pardir`/절대 anchor가 있는지 직접
    검사). 반환값은 최종 파일의 **열린 fd**다 — 호출자는 이 fd로만
    읽는다(다시 경로를 열지 않는다)."""
    parts = Path(result_artifact_path).parts
    if Path(result_artifact_path).is_absolute() or os.pardir in parts:
        raise UnsafeArtifactPathError(f"unsafe path: {result_artifact_path}")
    fd = root_fd
    opened: list[int] = []
    try:
        for i, name in enumerate(parts):
            is_last = i == len(parts) - 1
            flags = os.O_RDONLY | os.O_NOFOLLOW | (0 if is_last else os.O_DIRECTORY)
            fd = os.open(name, flags, dir_fd=fd)
            opened.append(fd)
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise UnsafeArtifactPathError(f"not a regular file: {result_artifact_path}")
        return fd
    except OSError as exc:
        for f in opened:
            os.close(f)
        raise UnsafeArtifactPathError(f"containment/openat failed: {result_artifact_path}: {exc}") from exc

def write_result_json(*, gate: str, candidate_id: str, run_id: str,
                       payload: dict, path: Path) -> None:
    """M3-06 신규 — 모든 gate runner가 자신의 result JSON을 쓸 때 반드시
    거치는 헬퍼. `payload`에 `_binding: {"candidate_id", "run_id", "gate"}`를
    자동으로 주입해, 개별 gate runner 구현이 이 필드를 깜빡 빠뜨릴 수
    없게 한다 — result JSON **내용 자체**가 어떤 candidate/run/gate에
    속하는지 스스로 증언하므로, evidence wrapper의 `result_artifact_path`가
    가리키는 파일을 다른 run에서 만든 동일 schema의 결과 파일로 바꿔치기해도
    (설령 evidence의 `result_artifact_sha256`을 공격자가 그 새 파일에 맞춰
    다시 계산해 넣었더라도) 파일 내용 자체의 `_binding`이 evidence
    wrapper의 값과 달라 `write_evidence()`/`m4_gate.py` 양쪽에서 검출된다."""
    full = {**payload, "_binding": {"candidate_id": candidate_id, "run_id": run_id, "gate": gate}}
    canonical = json.dumps(full, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(canonical)
    os.replace(tmp, path)

def write_evidence(*, gate: str, command: str, started_at_utc: str,
                    ended_at_utc: str, exit_code: int, profile: str,
                    candidate_id: str, run_id: str,
                    result_artifact_path: str,
                    evidence_dir: Path, candidate_root: Path) -> Path:
    fp = get_current_fingerprint()  # evaluation.m4_fingerprint 재사용 — git/settings/lock/index
    root_fd = os.open(candidate_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        result_fd = _open_contained(result_artifact_path, root_fd)
    finally:
        os.close(root_fd)
    try:
        result_bytes = os.read(result_fd, os.fstat(result_fd).st_size)
    finally:
        os.close(result_fd)
    result_json = json.loads(result_bytes)
    binding = result_json.get("_binding") or {}
    if (binding.get("candidate_id"), binding.get("run_id"), binding.get("gate")) != (candidate_id, run_id, gate):
        # M3-06 신규 — result JSON의 자기 신원과 evidence wrapper가 주장하는
        # 신원이 어긋난다. "다른 run artifact로 교체" 공격/버그가 여기서
        # write 시점에 이미 fail-closed된다(read 시점 §10.1b 검사와 이중 방어).
        raise ResultBindingMismatchError(
            f"{result_artifact_path} binding={binding} != expected "
            f"(candidate_id={candidate_id!r}, run_id={run_id!r}, gate={gate!r})"
        )
    payload = {
        "schema_version": "1.0.0",
        "gate": gate,
        "run_id": run_id,                      # 같은 Phase 7 실행 배치를 식별(§10.1b)
        "command": command,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "exit_code": exit_code,
        "profile": profile,                     # "mock" | "live" | "static"
        "candidate_id": candidate_id,
        "git_sha": fp.git_sha, "git_dirty": fp.git_dirty,
        "settings_sha256": fp.settings_hash,
        "lock_sha256": fp.lock_sha256,
        "index_manifest_sha256": fp.index_manifest_sha256,  # gate가 index와 무관하면 null
        "result_artifact_path": result_artifact_path,
        "result_artifact_sha256": hashlib.sha256(result_bytes).hexdigest(),
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    target = evidence_dir / f"{gate}.json"
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(canonical)
    os.replace(tmp, target)   # 원자적 write — 부분 기록된 evidence가 존재하지 않음
    return target
```

읽는 쪽(§10.1b `m4_gate.py`)도 **같은 `_open_contained()`/`write_result_json`
binding 규칙**을 재사용한다 — write와 read가 서로 다른 함수로 각자
containment를 구현하다가 어긋나는 것을 구조적으로 막는다(M2-06 원 취지
그대로, M3-06이 구현을 openat 기반으로 교체).

각 gate 산출 위치: `evaluation/reports/m4/<candidate-id>/evidence/<gate>.json`.

**gate별 작성 주체:**

- **정적 회귀(pytest + npm + vendor diff + markdown link + git diff --check,
  M2-06 static composite):** 신규 `evaluation/run_static_regression_gate.py`
  하나가 다섯 하위 명령을 **모두** subprocess로 순서대로 실행하고 각각의
  종료 코드/stdout hash를 하나의 `static_regression_result.json`에
  취합한다 — 이전 설계는 `run_pytest_gate.py`가 pytest만 기록한다고
  정의하면서 evidence 문서는 npm/vendor/markdown-link/git-diff까지
  "취합"한다고 서술해 그 넷의 결과 schema와 취합 주체가 없었다:

  ```python
  # evaluation/run_static_regression_gate.py 내부 결과 schema
  {
    "pytest": {"exit_code": 0, "junit_xml_sha256": "…"},
    "npm_test": {"exit_code": 0, "stdout_sha256": "…"},
    "vendor_sync_diff": {"exit_code": 0},   # `npm run sync-vendor` 후 `git diff --exit-code -- web/static/vendor`
    "markdown_links": {"exit_code": 0, "checked_file_count": N},
    "git_diff_check": {"exit_code": 0}      # 저장소 전체 `git diff --check`
  }
  ```

  다섯 하위 명령 중 하나라도 0이 아닌 종료 코드면 이 스크립트 자체가
  0이 아닌 코드로 종료하고, `write_evidence(gate="static_regression", ...,
  exit_code=<취합 결과>)`를 호출한다 — "일반 `pytest -q`를 별도 shell에서
  실행"하던 이전 설계의 공백을 메운다. **pytest 하위 명령은 M4
  orchestrator self-test를 명시적으로 제외한다(M3-06 — §10.1c 참조):**
  `pytest -q -m "not m4_orchestrator_self_test"` (`tests/integration/
  test_run_m4_gates_self_test.py`의 모든 테스트에
  `@pytest.mark.m4_orchestrator_self_test`를 붙이고 `pytest.ini`에 마커를
  등록) — 이 gate 자신이 다시 `run_m4_gates`를 실행하는 테스트를 자기
  자신 안에 포함시켜 무한 재귀/프로세스 폭증을 일으키지 않는다(Review
  M3-06 근거 1, 원문 인용: "static_regression runner가 전체 pytest -q를
  실행하고, 그 전체 suite에는 test_run_m4_gates_self_test.py가 포함된다").
  제외된 self-test는 별도 `pytest -q -m m4_orchestrator_self_test`(정적
  회귀 gate 밖, 일반 CI unit test 단계)로 실행된다 — §10.1c가 그 내부
  구현을 정의한다.
- **dependency/settings/logging/metrics/health/event loop/bounded
  load/정상 부하:** `evaluation/m4_load.py`와 각 pytest 대상 스위트를
  동일하게 `run_pytest_gate.py` 또는 `m4_load.py`가 자체적으로
  `write_evidence()`를 결과 JSON 생성 직후 호출하도록 구현한다(결과
  JSON 자체가 `result_artifact_path`, `write_result_json()`으로 생성).
- **container — local/live 산출물과 완전히 분리된 별도 inbox에서
  assemble된다(M3-06 신규, 아래 근거 참조):** §9.4 CI job 마지막 단계에서
  `python -m evaluation.m4_evidence write --gate container --exit-code $? \
  --ci-commit-sha "$GITHUB_SHA" --ci-run-id "$GITHUB_RUN_ID" \
  --ci-run-attempt "$GITHUB_RUN_ATTEMPT" \
  --image-digest "$(docker inspect --format='{{.Id}}' qna-rag:ci)" ...`를
  호출해 그 job 안에서 evidence를 기록한다(CI 산출물(artifact)로 업로드,
  §9.4). Phase 7 통합은 이 artifact를 **local 13-gate evidence 디렉터리와
  분리된** `evaluation/reports/m4/<candidate-id>/container_inbox/`(§10.1c
  `assemble_candidate_evidence()`가 매 실행 시작 시 empty 상태를 강제)로
  `gh run download --name m4-container-evidence-<candidate-id>-<run-id>
  --dir .../container_inbox`로 받은 뒤, assemble 단계에서 candidate
  evidence root로 copy+rehash한다(§10.1c). `ci_commit_sha`/`ci_run_id`/
  `ci_run_attempt`/`image_digest` 네 필드가 이 gate의 attestation을
  이룬다(M2-06/M3-06) — §10.1b의 fail-closed 검증이 이 필드를 현재
  `m4_gate.py` 실행의 commit과, 그리고 `--ci-run-id`가 명시적으로
  주어졌다면 그 값과 대조한다(§10.1b 아래 표 참조 — local Phase 7 실행처럼
  consumer run이 CI run과 다를 수 있는 구조와 양립하도록 강제 일치
  대상을 완화했다, Review M3-06 근거 6). artifact 이름은
  `m4-container-evidence-<candidate-id>-<run-id>`로 고정하고, 이름이
  다르면 다운로드 자체가 실패해 조용히 다른 실행의 산출물을 집어오지
  못한다.
- **M3 품질/단일 요청 성능:** 기존 `evaluate_gates()`/`compare.py` 실행 직후
  동일하게 `write_evidence()`를 호출하는 얇은 wrapper
  (`evaluation/run_compare_gate.py`)를 추가한다 — `evaluate_gates()`
  자체는 재구현하지 않는다(REQ-005/§7.3 그대로).
- **index:** §8.5 fault injection/100회 stress 테스트 결과 JSON을
  `tests/integration/test_index_lifecycle_stress.py`가 생성하고 같은
  wrapper 패턴으로 evidence를 남긴다.

### 10.1b fail-closed evidence 검증 — `evaluation/m4_gate.py` (M-07 대응)

Requirement §5.1의 모든 행을 하나의 `M4_GATES` 리스트(`evaluation/compare.py`의
`Gate`/`evaluate_gates()` 패턴 재사용, 새 gate 판정 알고리즘 자체는
재구현하지 않음)로 정의한다. `m4_gate.py --candidate-id <id> --run-id <id>`는
각 gate 이름에 대해 `evidence/<gate>.json`을 읽고 **다음을 모두 통과해야만**
그 gate의 실제 결과(PASS/FAIL)를 신뢰한다 — 하나라도 실패하면 그 gate는
`UNKNOWN`으로 fail-closed 처리한다(REQ-010.2 "미측정/schema/fingerprint
mismatch는 pass가 아니다"):

| 검증 | 실패 시 상태 |
|---|---|
| `evidence/<gate>.json` 파일이 존재하는가 | `NOT_RUN` |
| `schema_version`이 `m4_gate.py`가 아는 버전인가, JSON 파싱 성공 | `UNKNOWN`("schema mismatch") |
| `run_id`가 이번 `m4_gate.py` 호출에 전달된 `--run-id`와 같은가(§10.3 — Phase 7 전체 실행이 하나의 `run_id`를 공유하도록 환경변수 `EVIDENCE_RUN_ID`로 모든 gate 명령에 주입) | `UNKNOWN`("stale evidence, 다른 run") |
| `candidate_id`가 `--candidate-id`와 같은가 | `UNKNOWN`("candidate mismatch") |
| `git_sha`/`git_dirty`/`settings_sha256`/`lock_sha256`/`index_manifest_sha256`가 `m4_gate.py`가 지금 다시 계산한 `m4_fingerprint`와 같은가(index가 무관한 gate는 `index_manifest_sha256`을 비교하지 않는다) | `UNKNOWN`("fingerprint mismatch") |
| `result_artifact_path`가 candidate root 기준 dir_fd openat containment를 통과하는가(`m4_evidence.py::_open_contained`, `write_evidence()`와 동일 함수를 여기서도 호출해 write/read 양쪽에서 같은 규칙을 강제한다, M2-06/M3-06 — 이전 `resolve()` 후 `is_symlink()` 검사는 `resolve()`가 이미 symlink를 모두 따라간 뒤라 항상 False였던 죽은 코드였다, §10.1a 참조) | `UNKNOWN`("unsafe artifact path") |
| `result_artifact_path`가 실제 존재하고 그 파일을 다시 SHA-256 해시한 값이 `result_artifact_sha256`과 같은가 | `UNKNOWN`("result hash mismatch — 손상/변조 가능성") |
| **result JSON 내부 `_binding.{candidate_id,run_id,gate}`가 evidence wrapper 자신의 값과 일치하는가(M3-06 신규)** — hash가 일치하더라도 내용 자체가 스스로 다른 candidate/run/gate에 속한다고 증언하면 그 result 파일은 이 evidence의 것이 아니다 | `UNKNOWN`("result binding mismatch — 다른 run의 artifact") |
| `exit_code == 0`인가 | `FAIL`(여기서 멈추고 아래 세부 판정으로 넘어가지 않음) |
| `gate == "container"`인 경우: evidence의 `ci_commit_sha`가 `git_sha`와 같은가(항상 강제) — 그리고 **`m4_gate.py --ci-run-id`가 명시적으로 주어진 경우에만**(전형적 CI 트리거 실행) `ci_run_id`가 그 값과 같은지 추가로 검사한다. 로컬/후속 workflow에서 `--ci-run-id`를 생략하면 대신 **`--accept-container-evidence-run-id <id>` 필수 인자**로 운영자가 "이 특정 CI run의 container evidence를 신뢰한다"를 명시적으로 지정해야 하며 그 값이 evidence의 `ci_run_id`와 같아야 한다(M3-06 — Review M3-06 근거 6: consumer run이 항상 producer run과 같아야 한다는 이전 제약은 Phase 7이 로컬/후속 workflow에서 실행될 수 있는 구조와 양립하지 않았다. 둘 중 하나도 만족하지 않으면 검증 실패). `image_digest`가 비어 있지 않은가도 함께 검사(§9.4/§10.1a "CI attestation", M2-06) | `UNKNOWN`("container attestation mismatch") |
| 위를 모두 통과 | `result_artifact_path`의 실제 결과 JSON을 읽어 §5.1 각 행의 원 판정 로직(기존 `Gate`/`evaluate_gates()` 패턴, threshold 비교)을 적용해 `PASS`/`FAIL` 산출 |

`overall_pass`는 14개 gate 전부가 `PASS`일 때만 true — 하나라도
`FAIL`/`UNKNOWN`/`NOT_RUN`이면 false(REQ-010.2, 반올림 없이 raw count 비교).
이 판정은 하나의 내부 `GateVerdict` dataclass 리스트로만 표현되고,
`m4_gate.py --output X.json`은 그 리스트를 JSON으로, 같은 호출에서 항상
함께 `X.md`를 같은 리스트로부터 렌더링한다(**별도 코드 경로 없음** — JSON과
Markdown이 다른 값을 낼 가능성 자체를 구조적으로 제거, REQ-010.2 "같은
결과").

| gate | evidence 소스(§10.1a) |
|---|---|
| 정적 회귀 | `evidence/static_regression.json`(pytest JUnit XML + npm/vendor/markdown-link/git-diff exit code 취합) |
| dependency | `evidence/dependency.json`(lock install exit code + 2회 hash 비교 결과 JSON) |
| settings | `evidence/settings.json`(`tests/unit/test_settings.py` + `--check-config` redaction fixture) |
| logging | `evidence/logging.json`(`tests/integration/test_logging_contract.py`) |
| metrics | `evidence/metrics.json`(`tests/unit/test_metrics_cardinality.py`, §5.4 100회 activation 케이스 포함) |
| health | `evidence/health.json`(`tests/integration/test_health_state_table.py`) |
| event loop | `evidence/event_loop.json`(`m4_load.py mock --profile blocking-2s`) |
| bounded load | `evidence/bounded_load.json`(`m4_load.py mock --profile smoke-2s`) |
| 정상 부하 | `evidence/normal_load.json`(`m4_load.py mock --profile smoke-200ms`) |
| live smoke | `evidence/live_smoke.json`(`m4_load.py live`, §7.3) |
| M3 품질 | `evidence/m3_quality.json`(`evaluate_gates()` 재사용) |
| 단일 요청 성능 | `evidence/single_request_perf.json`(`compare.py` concurrency=1 vs M3) |
| index | `evidence/index.json`(§8.5 fault injection + 100회 stress) |
| container | `evidence/container.json`(§9.4 CI job) |

### 10.1c `evaluation/run_m4_gates.py` — 단일 고정 gate DAG (M2-06/M3-06 대응)

**문제였던 부분(M2-06):** 이전 설계는 "최종 검증 명령"(옛 §10.3)이 npm,
locked install, config check, index pytest, Docker build 등을 evidence
wrapper **밖에서** 사람이 shell로 직접 실행했다.

**M3-06이 추가로 발견한 세 가지 자기모순(Review M3-06):** (1)
`static_regression` gate가 실행하는 전체 `pytest -q`에
`test_run_m4_gates_self_test.py`가 포함돼 있었고, 그 self-test는 다시
`run_m4_gates`(그 안의 `static_regression`이 다시 전체 pytest 실행)를
호출해 무한 재귀/프로세스 폭증을 일으켰다. (2) "처음부터 비어 있는
fresh 디렉터리"를 주장하면서 `container` gate는 "CI에서 미리 다운로드된
evidence의 존재만 검사"했다 — 디렉터리가 정말 비어 있으면 container
evidence가 없고, 미리 다운로드해 두면 더 이상 fresh empty가 아니라는
모순. (3) 최종 명령이 `run_m4_gates`(그 안에 `live_smoke`가 이미
포함) 실행 후 evidence wrapper **밖에서** `m4_load live`를 다시
호출해 live LLM 호출을 중복 실행했다.

**해결 — 세 축으로 각각 답한다.**

1. **재귀 제거 — self-test를 static gate의 pytest selection에서
   제외하고, self-test 자체는 fake runner registry를 주입해 subprocess를
   전혀 만들지 않는다(§10.1a 이미 기술).** `run_m4_gates.main()`이
   `gate_dag: tuple[GateSpec, ...] = GATE_DAG`를 키워드 인자로 받도록
   설계해, self-test는 실제 subprocess를 실행하는 `runner`가 아니라
   `write_result_json()`+`write_evidence()`만 즉시 호출하는 **in-process
   fake runner**로 구성한 `FAKE_GATE_DAG`를 직접 주입한다 — 실제
   `pytest`/`docker`/`m4_load` 어떤 subprocess도 만들지 않으므로
   재귀할 대상 자체가 없다(이중 방어: pytest marker 제외 + 애초에
   self-test가 실제 gate를 실행하지 않는 설계).
2. **fresh dir와 container evidence의 모순 해소 — local/live 산출물
   root와 container 산출물 inbox를 물리적으로 분리한 뒤, 별도의 명시적
   atomic assemble 단계로 합친다.** `GATE_DAG`에서 `container`를
   **제거**한다 — 이 DAG는 더 이상 container evidence의 존재를
   "확인"하지 않는다. 대신:
   - `run_m4_gates.py`는 `evidence_dir`(candidate root 아래, local
     12개 + live 1개 = 13개 gate 전용)가 호출 시작 시 **반드시
     비어 있거나 존재하지 않아야** 함을 강제한다(이미 파일이 있으면
     즉시 실패 — "fresh dir" 계약을 코드로 강제).
   - container evidence는 **별도 디렉터리**
     `evaluation/reports/m4/<candidate-id>/container_inbox/`로
     `gh run download --name m4-container-evidence-<candidate-id>-<run-id>
     --dir .../container_inbox`(§9.4/§10.1a)로만 채워진다 — 이 디렉터리도
     다운로드 직전에는 비어 있어야 한다.
   - 신규 `evaluation.m4_evidence::assemble_candidate_evidence(inbox_dir,
     evidence_dir, *, gate="container")` 함수가 `container_inbox/`의
     `container.json`과 그것이 가리키는 result artifact를
     **copy+rehash**해 `evidence_dir/container.json`(및 그 result
     artifact)로 옮긴다 — **목적지(`evidence_dir/container.json`)가 이미
     존재하면 즉시 실패**(원자적 assemble, 덮어쓰기 없음). copy 후
     다시 SHA-256을 계산해 원본 `result_artifact_sha256`과 대조하므로
     다운로드 손상도 여기서 걸러진다.
   - `run_m4_gates.py --assemble-container-from <container_inbox_dir>`
     플래그가 주어지면 13개 gate 실행 직후 자동으로 이 assemble을
     호출한다(생략하면 `container` gate는 §10.1b에서 `NOT_RUN` —
     mock-only 로컬 검증에서는 자연스럽게 생략).
3. **live 중복 실행 제거 — `--live-mode {run,import,skip}`로 단일화.**
   `live_smoke` `GateSpec`의 `runner`가 내부적으로 `evaluation.m4_load
   live`를 호출하는 **유일한** 지점이 된다. `run`(기본, 실제 Phase 7
   최종 검증)은 이 gate 안에서 정확히 한 번 live 호출을 수행한다.
   `import <verified_live_evidence_dir>`는 실행 대신 이미 검증된 이전
   live evidence를 `assemble_candidate_evidence()`와 동일한
   copy+rehash+목적지-empty 규칙으로 가져온다(재실행 없이 신뢰할 수
   있는 이전 live 증거를 재사용하는 명시적 경로 — "몰래 재사용"이
   아니라 CLI 플래그로 드러난 선택). `skip`(self-test/mock-only 로컬
   개발 전용)은 이 gate를 아예 실행하지 않아 §10.1b가 `NOT_RUN` 처리한다
   — **`run`/`import`/`skip` 중 정확히 하나만 선택되므로 evidence
   wrapper 밖에서 별도로 `m4_load live`를 다시 호출할 이유 자체가
   없다**(§10.3이 그 중복 호출을 제거한다).

```python
# evaluation/run_m4_gates.py (신규, Phase 7)
@dataclass(frozen=True)
class GateSpec:
    name: str
    depends_on: tuple[str, ...]
    runner: Callable[[RunContext], int]   # 각 gate 전용 subprocess 래퍼, 반환값은 exit code

GATE_DAG: tuple[GateSpec, ...] = (
    GateSpec("dependency", (), run_dependency_gate),
    GateSpec("settings", (), run_settings_gate),
    GateSpec("static_regression", ("dependency",), run_static_regression_gate),
    GateSpec("logging", ("dependency",), run_logging_gate),
    GateSpec("metrics", ("dependency",), run_metrics_gate),
    GateSpec("health", ("settings",), run_health_gate),
    GateSpec("event_loop", ("settings",), run_event_loop_gate),
    GateSpec("bounded_load", ("settings",), run_bounded_load_gate),
    GateSpec("normal_load", ("settings",), run_normal_load_gate),
    GateSpec("index", ("dependency",), run_index_gate),
    GateSpec("live_smoke", ("bounded_load", "normal_load"), run_live_smoke_gate),  # M3-06: --live-mode로 run/import/skip 분기
    GateSpec("m3_quality", ("dependency",), run_m3_quality_gate),
    GateSpec("single_request_perf", ("m3_quality",), run_single_request_perf_gate),
    # M3-06 — container는 이 DAG에서 완전히 제거됐다(더 이상 "존재만 확인"하지
    # 않는다). §10.1c 본문의 `assemble_candidate_evidence()`가 별도 단계로
    # 처리한다.
)

def main(*, gate_dag: tuple[GateSpec, ...] = GATE_DAG,
         live_mode: str = "run",                       # "run" | "import" | "skip"
         assemble_container_from: "Path | None" = None,
         evidence_dir: Path, candidate_root: Path) -> int:
    """의존성 위상 정렬 순으로 각 GateSpec.runner를 실행한다. 선행 gate가
    실패(exit != 0)하면 그 gate에 의존하는 후속 gate는 스스로 시도하지 않고
    evidence 자체를 만들지 않는다(해당 gate는 §10.1b에서 `NOT_RUN`으로
    fail-closed 처리됨 — "실행이 스킵됐다"와 "실행했지만 값이 없다"를
    섞지 않는다). 모든 실행은 같은 `EVIDENCE_RUN_ID`/`M4_CANDIDATE_ID`를
    공유하고, `evidence_dir`은 항상 candidate 디렉터리 아래 고정 경로다.
    시작 시 `evidence_dir`이 비어 있지 않으면 즉시 실패한다(fresh dir
    강제, M3-06). `gate_dag`를 키워드로 주입할 수 있게 한 것은 오직
    self-test가 실제 subprocess를 만들지 않는 fake runner로 이 함수
    전체를 in-process 재사용하기 위함이다(M3-06 — 아래 self-test 참조)."""
```

`python -m evaluation.run_m4_gates --candidate-id m4-final --run-id
<uuid> --evidence-dir evaluation/reports/m4/m4-final/evidence --live-mode
run --assemble-container-from evaluation/reports/m4/m4-final/container_inbox`
한 번 호출로 13개 local/live gate의 canonical result JSON과 evidence가
**처음부터 비어 있는(fresh) 디렉터리**에서 전부 생성되고, 그 직후 이미
별도로 다운로드해 둔 `container_inbox/`의 container evidence가
assemble된다 — Phase 7 최종 검증은 항상 새 candidate 디렉터리에서 이
명령 하나를 실행하는 것으로 시작한다(재사용된 이전 디렉터리의 잔여
evidence가 섞여 들어오는 경우를 원천적으로 배제하며, container 산출물도
별도 inbox를 거치므로 "존재를 가정"하지 않는다).

**self-test(신규 `tests/integration/test_run_m4_gates_self_test.py`,
`@pytest.mark.m4_orchestrator_self_test`로 마킹돼 static_regression gate의
pytest selection에서 제외됨, M2-06/M3-06 수정안 "fake runner registry +
삭제/변조 self-test"):**

| 테스트 | 절차 | 검증 |
|---|---|---|
| `test_fresh_dir_all_evidence_created` | **fake runner registry**(각 `GateSpec.runner`를 `write_result_json()`+`write_evidence()`만 즉시 호출하는 in-process stub으로 교체한 `FAKE_GATE_DAG`)를 `run_m4_gates.main(gate_dag=FAKE_GATE_DAG, live_mode="skip", ...)`로 빈 임시 디렉터리에서 1회 실행 | 13개 evidence 파일이 모두 생성되고 각각 schema validation 통과, `subprocess.Popen`/`subprocess.run`이 monkeypatch로 감시돼 **0회** 호출됐음을 assert(재귀할 실제 프로세스 자체가 없었음을 직접 증명) |
| `test_non_empty_evidence_dir_rejected`(M3-06 신규) | `evidence_dir`에 미리 더미 파일 하나를 만든 뒤 `main()` 호출 | 즉시 실패(비어 있지 않은 dir 거부), 어떤 gate도 실행되지 않음 |
| `test_delete_one_evidence_fails_closed` | 위 정상 실행 직후 evidence 파일 1개를 삭제하고 `m4_gate.py` 재실행 | 그 gate가 `NOT_RUN`, `overall_pass=false` |
| `test_tamper_one_evidence_fails_closed` | evidence 파일 1개의 `result_artifact_sha256`을 다른 값으로 변조 | 그 gate가 `UNKNOWN`("result hash mismatch"), `overall_pass=false` |
| `test_replace_artifact_with_other_run_fails_closed` | 한 gate의 `result_artifact_path`가 가리키는 파일을 **`_binding`은 그대로 두고** 다른 `run_id`로 생성된 결과 파일 내용으로 교체(hash는 자연히 달라짐) | §10.1b "result hash mismatch" 검사가 `UNKNOWN`으로 처리 |
| `test_replace_artifact_and_rehash_still_fails_closed`(M3-06 신규 — 이전 시나리오보다 강한 공격) | 다른 run의 결과 파일로 교체하면서 evidence의 `result_artifact_sha256`도 새 파일에 맞춰 다시 계산해 넣음(hash 검사만으로는 통과) | `_binding.run_id`가 evidence wrapper의 `run_id`와 달라 `write_evidence()`/`m4_gate.py` 양쪽에서 `ResultBindingMismatchError`/`UNKNOWN`("result binding mismatch")로 걸림 — hash 재계산만으로는 우회할 수 없음을 직접 증명 |
| `test_container_assemble_requires_empty_destination`(M3-06 신규) | `container_inbox/`에 정상 container evidence를 두고 `evidence_dir/container.json`에도 미리 파일을 만든 뒤 `assemble_candidate_evidence()` 호출 | 목적지가 이미 존재해 즉시 실패, 기존 파일 불변(덮어쓰기 없음) |
| `test_live_mode_run_import_skip_mutually_exclusive`(M3-06 신규) | `--live-mode run`과 `--live-mode import <dir>`를 각각 별도로 실행 | `run`은 fake live 호출이 정확히 1회, `import`는 live 호출이 0회이며 대신 assemble이 1회 호출됨 — 같은 실행에서 두 경로가 동시에 실행되지 않음(§10.3의 "밖에서 또 실행" 중복이 구조적으로 불가능해짐을 증명) |

### 10.2 (§10.1a/§10.1b로 이전됨 — 이전 절 번호 유지 목적의 참조 전용, 내용 없음)

### 10.3 최종 검증 명령(Plan §Phase 7 그대로, M-07/M2-06/M3-06 대응)

```bash
export EVIDENCE_RUN_ID="$(python -c 'import uuid; print(uuid.uuid4().hex)')"
export M4_CANDIDATE_ID="m4-final"
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
# container evidence를 §9.4 CI job이 만든 artifact에서 먼저 받아 별도
# inbox(fresh, container_inbox/)를 채운다 — evidence_dir 자체는 아래
# run_m4_gates가 시작할 때 여전히 완전히 비어 있다(M3-06).
gh run download "$CONTAINER_CI_RUN_ID" \
  --name "m4-container-evidence-$M4_CANDIDATE_ID-$CONTAINER_CI_RUN_ID" \
  --dir "evaluation/reports/m4/$M4_CANDIDATE_ID/container_inbox"
# 아래 한 호출이 §10.1c GATE_DAG 전체(13개 local/live gate)를 fresh
# evidence 디렉터리에서 순서대로 실행해 evidence를 원자적으로 쓰고,
# 마지막에 위에서 받은 container_inbox를 evidence_dir로 assemble한다 —
# npm/lock install/check-config/index pytest/부하 mock/**live LLM 호출**을
# 더 이상 evidence wrapper 밖에서 손으로 실행하지 않는다(M2-06/M3-06 —
# 이전 설계는 여기서 `m4_load live`를 이 명령과 별개로 한 번 더 호출해
# live 호출이 중복됐었다. `--live-mode run`이 이 명령 **하나** 안에서
# 정확히 한 번만 실행한다).
python -m evaluation.run_m4_gates --candidate-id "$M4_CANDIDATE_ID" --run-id "$EVIDENCE_RUN_ID" \
  --evidence-dir "evaluation/reports/m4/$M4_CANDIDATE_ID/evidence" \
  --live-mode run \
  --assemble-container-from "evaluation/reports/m4/$M4_CANDIDATE_ID/container_inbox"
docker build -f deploy/Dockerfile --target production -t qna-rag:verify .   # M3-05 — --target 명시, 저장소 루트 .dockerignore(§9.2) 적용됨 — production 이미지 태그 검증용, container gate evidence 자체는 위에서 이미 assemble됨
python -m evaluation.m4_gate --candidate-id "$M4_CANDIDATE_ID" --run-id "$EVIDENCE_RUN_ID" \
  --ci-run-id "$CONTAINER_CI_RUN_ID" \
  --output evaluation/baselines/m4_initial.json
git diff --check
```

모든 gate 명령이 같은 `EVIDENCE_RUN_ID`를 공유하므로 `m4_gate.py`는 §10.1b의
"stale evidence" 검사로 이전 실행이나 부분 실행의 evidence가 섞여 들어오는
것을 fail-closed로 걸러낸다. `$CONTAINER_CI_RUN_ID`는 §9.4 CI job이 방금
production 이미지를 빌드/스캔/검증한 그 CI 실행의 run ID다 — 로컬에서
Phase 7을 실행하는 사람이 그 값을 알고 있어야 이 명령을 완주할 수 있다는
점 자체가 "어떤 CI 실행의 container evidence를 신뢰하는지"를 암묵적
가정이 아니라 명시적 인자로 강제한다(§10.1b M3-06 수정).

### 10.4 M4 baseline 산출물

`evaluation/baselines/m4_initial.json`(기계 판독)과 `.md`(사람 요약)를
M3와 동일 형식으로 생성한다. `m4_initial.md`는 M3 baseline을 참조하고
(REQ-002.5) 새 fingerprint(settings/lock/dependency/index manifest)를
추가한다. `Roadmap.md`는 `overall_pass=true`가 확인된 뒤에만 M4를 완료로
갱신한다(사람 승인 없이 자동 판정, REQ-010.4).

## 11. 병렬/직렬 경계 (Plan §2 재확인)

- Phase 2(로그/metric schema 설계)와 Phase 5(manifest 설계, §8.2)는
  **문서 리뷰 단계**에서 병렬 가능 — 서로 다른 파일(`observability/`,
  `index/`)이라 코드 충돌이 없다.
- 같은 production 파일(`web/server.py`, `rag_engine.py`, `config.py`)에 대한
  실제 구현 커밋과 공식 성능 측정 실행(§7.3, §10.3)은 **직렬화**한다 — 두
  Phase가 동시에 `web/server.py`를 수정하면 §6.4 executor와 §5.2 미들웨어
  통합 순서가 꼬인다.
- Phase 6(container)은 Phase 1(lock)과 Phase 3(health 계약)이 **고정된
  뒤**에만 시작한다(Dockerfile이 lock 파일과 `/health/live` 경로를
  참조하기 때문).
- index fault injection(§8.5)과 부하 실행(§7)은 `tmp_path`/전용 staging
  디렉터리에서 수행하고 공유 `runtime/` artifact를 건드리지 않는다(Plan §2
  그대로).

## 12. 호환성 보존 체크리스트

- [x] `/rag` 성공 응답 schema(`answer/sources/success/search_type/intent`)
      불변 — §6.3.
- [x] `simple-qna-rag-web`/`-query`/`-index` entry point 이름 불변,
      `-index-lifecycle`만 신규 추가 — §8.3.
- [x] `SIMPLE_QNA_RAG_MMR_VECTOR_SOURCE`, `SIMPLE_QNA_RAG_ROUTING_SIGNAL_OVERRIDE`,
      `SIMPLE_QNA_RAG_ANSWER_TEMPLATE_MODE` 등 M3 rollback flag/classifier
      artifact 미제거 — §2.2.
- [x] M3 14 gate/`evaluate_gates()` 재사용, 재구현 없음 — §7.3, §10.1b.
- [x] M3 baseline 두 파일(`m3_initial.json/.md`) byte 불변 — 어떤 Phase도
      `evaluation/baselines/m3_initial.*`를 쓰지 않는다.
- [x] `milestone_dev_orchestration_guide.md` rename은 이번 세션에서 이미
      완료된 상태로 두고 추가로 건드리지 않는다.
- [x] `runtime/vectorstore/`(M3 index) 원본 미변경 — §8.1, §8.7.
- [x] M3 legacy index로의 복구 경로는 명시적 서비스 버전 rollback 또는
      명시적 `import-legacy`+`activate`로만 가능 — §8.6(M-02, "자동
      폴백"이 아니라 "명시적 복구 경로"로 REQ-006 M3 rollback 보존 문구를
      해석한다).

## 13. Design Review에서 확인이 필요한 가정(초안 상태 명시)

1. lock 도구를 `pip-tools`로, metric 라이브러리를 `prometheus-client`로
   고른 것은 "작은 dependency" 허용 범위 내에서의 기본값이다(§4.1, §5.4) —
   Codex 리뷰에서 대안(uv/pip-compile 대안, 자체 구현 registry) 요구 시
   변경 가능.
2. index pointer를 심볼릭 링크 대신 파일로 설계한 이유(§8.1)는 컨테이너
   bind mount 이식성이며, 대안 검토 여지가 있다.
3. `/rag` 실패 시 timeout/overload 상태 코드를 모두 503으로 통일한 것(§6.3)
   은 클라이언트 처리 단순화를 위한 선택이며, timeout에 504를 쓰는 대안도
   가능하다.
4. §8.3a에서 build와 activate가 하나의 OS lock을 공유하도록 확정해 build
   진행 중 activate/rollback이 즉시 exit 4로 실패할 수 있다 — 단일
   운영자 가정 하의 trade-off이며, 동시 운영 요구가 생기면 M5에서
   lock을 분리하는 재설계가 필요하다(Review M-03).
5. **(M3-02로 확정, 더 이상 가정이 아니다)** §6.6b는 이전에 request-scoped
   upstream timeout 주입에 `model_copy(update={"client_kwargs": ...})`가
   동작한다고 "가정"했었다. lock된 `langchain-ollama==0.3.10`으로 read-only
   spike를 실행해 이 가정이 **거짓임을 실측으로 확인**했다(`model_copy`
   후에도 `copy._client is llm._client`, timeout 미반영) — §6.6b는 이제
   "매 호출 재생성"을 유일한 채택 정책으로 명시하며, `validate_model_on_init`
   기본값이 `False`임을 함께 확인해 재생성 비용이 낮음을 근거로 남겼다.
   DDGS는 애초에 이런 선택지 자체가 없어(생성자 전용 `timeout`, 내부
   `ThreadPoolExecutor.shutdown(wait=True)`가 timeout을 우회해 무기한 블록
   가능) subprocess 경계(§6.6b `run_in_killable_subprocess`)를 채택했다.

## 14. Design Review Iteration 1 대응 매핑

[Design_Review_Iteration_1.md](Design_Review_Iteration_1.md)의 모든
MAJOR/MINOR/TRIVIAL 발견사항과 본 문서의 대응 절을 1:1로 연결한다 —
Iteration 2 리뷰가 누락 여부를 빠르게 확인할 수 있도록 하기 위함이다.

| 발견 | 요지 | 대응 절 |
|---|---|---|
| M-01 | executor FIFO/absolute deadline/exactly-once release, 결정론적 race 테스트 | §6.4(전면 재설계), §6.1a |
| M-02 | manifest 없는 legacy pickle 자동 로드 제거, fail-closed migration | §8.1, §8.6, §8.5(#7), §12 |
| M-03 | version ID 순환 참조, build/import 공통 staging, build+activate lock/충돌 계약 | §8.2, §8.3, §8.3a, §8.4(step 5), §8.5(#5,#8,#9), §8.7 |
| M-04 | query deadline에서 파생된 upstream connect/read/write/pool/overall 예산 | §4.3(규칙 7), §6.6b |
| M-05 | retrieval 6 sub-stage/fallback observability, 100회 lifecycle cardinality 상한 | §5.3, §5.4 |
| M-06 | 실제 적용되는 root `.dockerignore`, 올바른 entrypoint override, test-only DI | §9.0, §9.1, §9.2, §9.4 |
| M-07 | 공통 gate evidence schema, fail-closed aggregator, JSON/Markdown 단일 판정 | §10.1a, §10.1b, §10.3 |
| m-01 | shutdown drain 순서/executor 종료 순서 | §6.1a |
| m-02 | raw ASGI body receive wrapper | §6.6a |
| m-03 | settings hash 의미, secret-safe 정책 | §4.3a, §4.4 |
| t-01 | `else_done := None` 잔여 문구 제거 | §6.4(재작성으로 자연 제거) |

## 15. Design Review Iteration 2 대응 매핑

[Design_Review_Iteration_2.md](Design_Review_Iteration_2.md)의 모든
MAJOR/MINOR 발견사항과 본 문서의 대응 절을 1:1로 연결한다. Iteration 1
발견(§14)은 "해결"로 재확인됐거나(M-03, m-03, t-01) 아래 M2-01~M2-06/
m2-01~m2-04로 이어져 재작업됐다(M-01→M2-01, M-04→M2-02, M-05→M2-03,
M-02→M2-04, M-06→M2-05, M-07→M2-06, m-01→m2-01, m-02→m2-02).

| 발견 | 요지 | 대응 절 |
|---|---|---|
| M2-01 | future 완료 콜백에 매인 단일 `_finalize` release, `QUEUED/RUNNING/ABANDONED/DONE` 실제 enum, `run_in_executor` 실패 동기 release, 동일-clock `asyncio.timeout_at`, barrier 기반 결정론적 race 테스트 8종 | §6.4(핵심 재작성), §6.7 |
| M2-02 | request~router~answer~DDGS 단일 `DeadlineBudget`(monotonic), `require_remaining()` 사전 차단, per-phase cap + 스트림 watchdog, singleton→request-scoped LLM seam, connect/read/stream stall 통합 테스트 | §6.4(budget 생성), §6.6b(전면 재작성), §13-5 |
| M2-03 | `retrieval_substage_*`/`fallback_triggered` allowlisted event 추가, `ObservationSink`를 agent.py/rag_engine.py에 주입, `RetrievalTrace`와 단일 측정 공유, safe wrapper 적용 범위 확장 | §5.1, §5.3(핵심 재작성), §5.6 |
| M2-04 | `--expected-faiss-sha256`/`--expected-pkl-sha256` 필수, 승인 root containment, `O_NOFOLLOW` symlink/TOCTOU 방어, hash 검증이 deserialize보다 선행 | §8.3, §8.7(핵심 재작성) |
| M2-05 | 표준 venv 기반 builder, `COPY --exclude`로 testing/ 소스 배제, test stage 재설치로 실제 console script 생성, `.dockerignore` `!README.md` 예외, ENTRYPOINT 정책 이미지 태그로 통일, layer별 outer archive 해제 + known-secret 바이트 스캔 | §9.0, §9.1(핵심 재작성), §9.2, §9.4 |
| M2-06 | `run_m4_gates.py` 단일 고정 DAG, static composite(pytest+npm+vendor+link+diff) 취합, artifact 상대 경로 containment/symlink 거부, CI attestation(commit/run/image digest), fresh dir 생성 + 삭제/변조 self-test | §10.1a, §10.1b, §10.1c(신규), §10.3 |
| m2-01 | `begin_drain()`/`shutdown_pool()` public API, idle 즉시 완료, bounded 종료가 아니라는 한계 명시 | §6.1a, §6.4 |
| m2-02 | body limiter와 `RequestContextMiddleware`가 공유하는 `resolve_request_id()`, `tracking_send`로 response-start 실측 후 안전 처리 | §5.2, §6.6a |
| m2-03 | 실제 Prometheus sample 계약(`_created` 비활성화, boundary 7개)으로 재계산한 139/150 예산 | §5.4 |
| m2-04 | Phase 1 필수 산출물로서의 완전 settings 인벤토리 표, 직접 `os.environ` 조회 0건 AST 정적 gate | §4.3(신규 4개 필드), §4.3b(신규) |

## 16. Design Review Iteration 3 대응 매핑

[Design_Review_Iteration_3.md](Design_Review_Iteration_3.md)의 모든
MAJOR/MINOR 발견사항과 본 문서의 대응 절을 1:1로 연결한다. 실제 lock된
`langchain-ollama==0.3.10`/`ollama==0.6.0`/`ddgs==9.14.4`/
`prometheus-client==0.26.0`(신규 도입 예정) API를 read-only executable
spike로 직접 실행/확인한 뒤 반영했다(M3-02, M3-07).

| 발견 | 요지 | 대응 절 |
|---|---|---|
| M3-01 | ownership token(`_pop_from_queue_or_none`/`callback_registered`) 기반 cancellation-free critical section, `asyncio.shield()`로 감싼 단일 finalize, `call_soon_threadsafe` 불필요 hop 제거, 3종 barrier 테스트(`queue_grant_then_cancel_before_submit`/`cancel_during_submit_failure_finalize`/`callback_queued_then_loop_shutdown`) | §6.1a, §6.4(핵심 재작성) |
| M3-02 | spike로 확인한 `model_copy` 미반영 사실, router/answer LLM 매 요청 재생성+`bind_tools`, `generate_answer(...,llm=...)` 명시 인자, DDGS 종료 가능 subprocess 경계(`run_in_killable_subprocess`) | §6.6b(전면 재작성), §13-5 |
| M3-03 | 기존 `RetrievalStageTrace(name,latency_ms,candidate_count)` 3필드 그대로 생성하는 단일 `_measure_substage` helper, outcome은 sink 호출에만 투영, `safe_sink_call()`로 stage/substage/fallback 세 호출 통일, `stage_started`/`retrieval_substage_started` allowlist 삭제 | §5.1, §5.3(핵심 재작성), §5.6 |
| M3-04 | 승인 hash를 committed `evaluation/baselines/m3_initial.json`(exact hash 실측 확인) 단일 원본으로 고정, dir_fd/openat `O_DIRECTORY\|O_NOFOLLOW` component traversal, fstat owner/mode/inode 검증, staging owner/mode activate 재검증 | §8.3, §8.4, §8.7(핵심 재작성) |
| M3-05 | 명시적 `production` stage를 파일의 실제 마지막 stage로 고정 + 모든 production build `--target` 명시, `requirements.txt` builder COPY 추가, CI trap cleanup+bounded poll+evidence write/upload, 경로 canonicalization, scanner positive/negative control unit test | §9.1(핵심 재작성), §9.3, §9.4(핵심 재작성) |
| M3-06 | static gate pytest selection에서 self-test 제외 + fake runner registry로 subprocess 없는 self-test, local/live/container evidence root 분리 후 원자적 assemble(목적지 empty 강제), `--live-mode {run,import,skip}`로 live 중복 실행 제거, dir_fd 기반 `_open_contained`, result JSON `_binding` 상호 검증, container `ci_run_id` 정합성 완화 | §10.1a, §10.1b, §10.1c(핵심 재작성), §10.3(재작성) |
| M3-07 | `Settings` dataclass 자체를 완전 인벤토리 단일 원본으로 확정(retrieval/MMR/hybrid/reranker/web 검색 값 포함 30개 필드), `routing_signal_override` 타입 `bool`로 정정, spike로 확인한 `prometheus_client.disable_created_metrics()`(non-env 공개 API)로 교체 | §4.3(전체 필드 확정), §4.3b(재작성), §5.4 |
| m3-01 | Docker smoke `trap 'docker rm -f smoke \|\| true' EXIT` cleanup, 고정 `sleep 5` 대신 bounded poll(30x1s) | §9.4 |
| m3-02 | 소스 근거를 `module::symbol` 형태로 표기(§6.6b/§8.7의 `ChatOllama._set_clients`/`DDGS._search_sync` 등), line 번호는 보조로만 사용 | §6.6b, §8.7 |
