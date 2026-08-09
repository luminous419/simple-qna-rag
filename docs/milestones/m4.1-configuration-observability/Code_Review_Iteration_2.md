# M4.1 구현 독립 코드 리뷰 — Iteration 2

검토일: 2026-08-08  
검토자: Codex (독립 구현 Gate 리뷰어)

검토 대상은 `milestone_dev_orchestration_guide.md`,
`Code_Review_Iteration_1.md`, 현재 worktree의 tracked/untracked 전체 구현,
`Traceability.md`, 그리고 live report
`evaluation/reports/m4_regression/baseline_20260808T130426002445Z.md`/`.json`이다.
제품 코드와 구현 원문은 수정하지 않았고 본 리뷰 문서만 추가했다.

## 1. Gate 판정

**FAIL — 통합·인수 단계 진입 불가**

- 점수: **8.9 / 10.0**
- 발견사항: **CRITICAL 0 / MAJOR 2 / MINOR 0 / TRIVIAL 0**
- 9.7 Gate: **미통과**
- 통합·인수 단계 진입: **거절**
- 사용자 결정 필요: **없음**. 요구사항 선택의 모호성이 아니라 실행 가능한
  CI 배치/보안 수정과 기존 evaluator 계약 보수 문제다. 다만 M4.1 범위를 넘어
  기존 M3 evaluator를 고칠 권한이 없다면 리더가 선행 evaluator-fix 작업을
  별도로 배정해야 한다.

Iteration 1의 secret redaction, 실제 output surface, 제품 metrics wiring,
logging negative matrix, 3 CLI `main()` wiring, temp cleanup은 코드와 targeted
tests에서 폐쇄됐다. 그러나 새 self-hosted live job은 현재 형태로는 vectorstore를
보존하지 못하고 PR의 신뢰되지 않은 코드를 self-hosted runner에서 실행한다.
또한 제공된 실제 live report는 14개 중 6개가 `pass=None`이고
`overall_pass=False`다. Requirement의 `UNKNOWN`/`NOT_RUN` 불승인 원칙에 따라
환경 또는 기존 evaluator gap이라는 이유만으로 exit 1을 면제하지 않는다.

## 2. 발견사항

### CRITICAL

없음.

### MAJOR

#### CR-I2-MAJ-01 — self-hosted live CI job은 현재 vectorstore를 지우며 PR 코드 실행에도 안전하지 않다

`.github/workflows/ci.yml`의 `m3-live-regression-gate`는
`on: pull_request`와 `push: master` 모두에서 `[self-hosted, ollama-m3]`를
사용한다. 그러나 `actions/checkout@v4`의 기본 clean 동작과 저장소의
`.gitignore`(`runtime/`, `vectorstore/`)를 함께 적용하면 runner workspace에
미리 둔 `runtime/vectorstore/index.faiss`와 `index.pkl`은 checkout 과정에서
제거된다. workflow 주석의 “committed runtime/vectorstore” 주장과 달리 두
파일은 `git ls-files runtime/vectorstore` 결과가 0건이다. 별도 외부 경로
override나 복원 step도 없으므로 현재 문서화된 provisioning만으로 wrapper의
첫 `_vectorstore_fingerprint()`부터 실패한다.

더 심각하게 `pull_request`에서 repository checkout 후 dependency install과
Python 코드를 self-hosted runner에서 곧바로 실행한다. PR head가 신뢰되지 않은
경우 runner host/Ollama/로컬 데이터에 임의 코드 실행 권한을 주는 배치다.
`permissions: contents: read`는 host-level 코드 실행 위험을 제거하지 않는다.
timeout/concurrency도 없어 고가의 24분 이상 live run이 push/PR마다 중복될 수
있다. 따라서 CR-I1-MAJ-04의 “CI 연결”은 선언상 추가됐지만 실행 가능성과
안전성 기준에서는 **미폐쇄**다.

필수 수정:

1. vectorstore는 checkout workspace 밖의 runner 전용 read-only 경로에 두고
   `SIMPLE_QNA_RAG_VECTORSTORE_PATH`로 명시하거나, 검증된 immutable artifact를
   checkout 후 복원한다. 실행 전 두 canonical 파일 존재/해시 preflight를 둔다.
2. untrusted `pull_request`에서는 self-hosted live job을 실행하지 않는다.
   protected `master` push, 승인된 `workflow_dispatch`, protected environment 등
   신뢰 경계를 명시하고 fork/PR 코드가 runner에서 실행되지 않게 한다.
3. `timeout-minutes`, `concurrency`, Ollama model/version/endpoint health preflight,
   report artifact 경로를 명시하고 실제 workflow run 성공 증거를 보존한다.

#### CR-I2-MAJ-02 — 실제 live 결과가 6개 UNKNOWN이므로 14-gate 인수 증거가 아니다

제공된 live report는 모든 evaluator stage가 success이고 품질 값도 좋아 보이지만
Gate는 8 PASS / 6 `None`, `overall_pass=False`다. 원인은 다음과 같이 코드와 raw
report로 재현됐다.

- Retrieval 3개 latency gate: raw report의 `warmup.performed=false` 때문에
  `evaluation.compare._retrieval_gate_inputs()`가 실제 존재하는 latency
  mean/p95/MMR mean 값을 모두 `None`으로 강제한다.
- Answer 2개 latency gate: 동일하게 `warmup.performed=false`라서 실제
  mean/p95 값을 `None`으로 강제한다.
- Document routing recall 1개: single-run raw report에는
  `document_route_correct=61`이 존재하지만 `_routing_gate_inputs()`는 aggregate가
  없을 때 비율 `document_qa.recall=1.0`을 읽은 뒤 count와 비교할 수 없다며
  `None`으로 버리고 이미 존재하는 raw count를 사용하지 않는다.

이 로직은 이번 M4.1 diff에서 변경되지 않은 기존 evaluator gap이다. 그러나
`scripts/run_m4_regression_gate.py`는 `run_baseline()`을 기본
`warmup_cases=0`으로 호출하므로 latency 5개를 구조적으로 UNKNOWN으로 만들고,
single-run routing의 raw count도 evaluator가 버린다. 즉 “범위 밖 기존 결함”은
원인 귀속일 뿐 Gate 면제가 아니다. wrapper의 올바른 결과는 exit 1이며,
Requirement §4의 모든 필수 Gate PASS 조건상 CR-I1-MAJ-04는 여전히 미폐쇄다.

필수 수정: M3 evaluator 선행 보수로 (a) 인수 실행에서 양수 warmup을 명시하고
실제 수행을 검증하며, (b) single-run의 `document_route_correct` raw count를
CountGate 입력으로 사용한다. 그 후 live wrapper를 다시 실행해 14/14
`pass=True`, `overall_success=True`, `overall_pass=True`, exit 0, JSON/Markdown
parity, baseline/vectorstore pre/post 불변을 한 실행에서 보존해야 한다.
기존 evaluator 변경을 M4.1 제품 코드 범위에 섞기 어렵다면 별도 선행 수정으로
분리할 수 있지만, 그 증거 전에는 통합 진입할 수 없다.

### MINOR

없음.

### TRIVIAL

없음.

## 3. Iteration 1 폐쇄 재검증

| ID | Iteration 2 판정 | 독립 증거 |
|---|---|---|
| CR-I1-MAJ-01 | **폐쇄** | `FieldSpec.annotation` 기반 default-redact 정책. credential URL/prompt/token adversarial subprocess 출력에서 marker 0건; bounded bool/int/float/Literal만 값 공개 |
| CR-I1-MAJ-02 | **폐쇄** | audit 기본값 `REPLACE`, 잔존 artifact 71건 모두 CLI/sanctioned sink. 실제 agent→keyword fallback→DDGS와 실제 `RAGEngine.query()` retrieval/generation 동적 capture에서 query/title/summary/URL/doc/answer/exception/절대경로 marker 0건 |
| CR-I1-MAJ-03 | **폐쇄** | `agent.route_query()`와 `RAGEngine.query()`에 routing/web/retrieval/generation duration/error 및 web/MMR fallback sink가 연결됨. 실제 FastAPI `/rag` 1,000건 test가 scrape sample≤150, request counter=1000, 네 stage histogram count와 web fallback counter 증가를 검증 |
| CR-I1-MAJ-04 | **미폐쇄** | wrapper는 `None`을 PASS로 오인하지 않고 exit 1을 내도록 구현됐으나 실제 report가 6 UNKNOWN이고 CI job도 실행 가능·안전하지 않음(CR-I2-MAJ-01/02) |
| CR-I1-MIN-01 | **폐쇄** | required key default/strict raise, route/stage/error/level/status/duration type·enum clamp/strict raise negative matrix 실행 PASS |
| CR-I1-MIN-02 | **폐쇄** | subprocess에서 query/index/web의 실제 `main()`을 호출. invalid 시 엔진/문서 로드 경계 미호출, valid CLI override가 process Settings와 server boundary까지 도달함을 검증 |
| CR-I1-TRI-01 | **폐쇄** | trap이 `${tmp_a}.body`/`${tmp_b}.body`를 함께 제거 |

## 4. Requirement/Gate 영향

| Requirement | 판정 | 근거 |
|---|---|---|
| REQ-001 | 부분 PASS | lock 관련 targeted check는 유효하나 권위 있는 clean Linux CI run 증거는 이번 리뷰 입력에 없음 |
| REQ-002 | PASS(구현/targeted) | adversarial secret 0 및 실제 3 CLI wiring 확인 |
| REQ-003 | PASS(구현/targeted) | 실제 payload surface 0, structured logging negative/response matrix 확인 |
| REQ-004 | PASS(구현/targeted) | 실제 1,000 request scrape와 product counters/histograms 확인 |
| REQ-005 | 부분 PASS | 구현 테스트는 locked venv에서 수집 가능하나 전체 clean CI 결과는 별도 필요 |
| REQ-006 | **FAIL** | 실제 M3 결과 6 UNKNOWN/overall false, CI live job 불실행 가능·unsafe |

Traceability의 CR-I1-MAJ-04 “폐쇄” 표기는 실제 증거와 불일치한다. queued job은
PASS 증거가 아니며, runner 미등록·vectorstore 미존재·UNKNOWN gate를 환경
예외로 자동 승인할 수 없다.

## 5. 검증 명령과 결과

| 명령 | 결과 |
|---|---|
| `venv/bin/pytest -q tests/integration/test_check_config_cli.py tests/integration/test_output_surface_capture.py tests/integration/test_metrics_live_traffic.py tests/integration/test_request_logging_matrix.py tests/integration/test_cli_main_wiring_matrix.py tests/integration/test_m3_regression_gate.py tests/unit/test_logging_callsite_disposition.py tests/unit/test_observability_logging.py tests/unit/test_observability_metrics.py` | **PASS — 100 passed** |
| 동일 명령을 ambient `pytest`로 실행 | **NOT_RUN/collection error** — shared env의 구 FastAPI가 `email-validator>=2`를 요구. `venv`의 FastAPI 0.121.2/Pydantic 2.12.4로 targeted 재실행해 PASS했지만 clean CI 전체 PASS를 대신하지 않음 |
| adversarial `--check-config` (`TOP_SECRET_VALUE`, URL password, token marker) + `rg` | **PASS — secret marker 0건**, 세 필드 모두 redacted metadata |
| `venv/bin/python scripts/logging_callsite_audit.py --check` | **PASS**; artifact는 KEEP_CLI 71, REPLACE 0 |
| `venv/bin/python scripts/generate_field_spec.py --check` | **PASS** |
| `git diff --check` | **PASS** |
| `venv/bin/python scripts/run_m4_regression_gate.py` | **NOT_RUN/exit 2** — opt-in 미설정. 자동 면제하지 않음 |
| live report JSON/raw evaluator report 조사 | **FAIL — 8 PASS / 6 UNKNOWN, overall_pass=false**; 원인은 warmup 0과 single-run raw routing count 미사용 |
| `git ls-files runtime/vectorstore` + `.gitignore` 조사 | tracked 파일 0; `runtime/` ignored. 현재 checkout job의 “committed vectorstore” 전제 불성립 |

## 6. 통합 재진입 조건

1. CR-I2-MAJ-01의 CI 신뢰 경계와 외부 read-only vectorstore provisioning을
   수정하고 실제 self-hosted workflow 성공 증거를 남긴다.
2. CR-I2-MAJ-02의 기존 evaluator gap을 선행 수정한 뒤 동일 wrapper 실행에서
   14/14 PASS와 exit 0을 확보한다.
3. clean locked Linux CI 전체 suite, `pip check`, Node/npm job까지 PASS임을
   확인한다.

이 세 조건 전에는 9.7 Gate, 통합·인수 단계, M4.2 진입을 승인하지 않는다.
