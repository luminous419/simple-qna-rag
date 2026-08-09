# M4.1 구현 독립 코드 리뷰 — Iteration 1

검토일: 2026-08-08  
검토자: Codex (독립 구현 Gate 리뷰어)

검토 대상: [milestone 개발 가이드](../../../milestone_dev_orchestration_guide.md),
[Requirement.md](Requirement.md), [Plan.md](Plan.md), [Design.md](Design.md),
[Traceability.md](Traceability.md),
[Design_Review_Resume_Cycle_3.md](Design_Review_Resume_Cycle_3.md), 현재 worktree의
tracked/untracked 구현 전체와 신규 테스트.

제품 코드와 구현 원문은 수정하지 않았다. 본 리뷰 문서만 추가했다.

## 1. Gate 판정

**FAIL — 통합·인수 단계 진입 불가**

- 점수: **7.8 / 10.0**
- 발견사항: **CRITICAL 0 / MAJOR 4 / MINOR 2 / TRIVIAL 1**
- 9.7 Gate: **미통과**
- 통합·인수 단계 진입: **거절**

typed Settings/facade, lock 재현성, 102-sample schema 상한, lifespan health의
기본 구조는 잘 구현됐고 관련 격리 테스트도 통과한다. 그러나 secret 출력,
실제 요청 경로의 payload 출력 잔존, 선언만 있고 제품에 연결되지 않은 stage/
fallback metrics, 실행되지 않은 M3 live 14-gate 때문에 REQ-002.5,
REQ-003.2/.3, REQ-004.1, REQ-006 및 “모든 필수 Gate PASS” 조건을 만족하지
못한다. `UNKNOWN`/`NOT_RUN`은 완료가 아니라는 Requirement §4에 따라 현재
상태를 환경 예외만으로 승인할 수 없다.

## 2. 발견사항

### CRITICAL

없음.

### MAJOR

#### CR-I1-MAJ-01 — `--check-config`가 credential과 prompt 값을 평문 출력한다

`cli/web.py::_run_check_config()`는 `Path` 값만 redaction하고 나머지 36개
필드를 그대로 JSON에 넣는다. 따라서 환경에서 설정 가능한 `OLLAMA_BASE_URL`의
userinfo credential과 `PROMPT_TEMPLATE`의 임의 민감 문자열이 stdout으로
노출된다. 다음 실행에서 실제로 둘 다 검출됐다.

```bash
SIMPLE_QNA_RAG_PROMPT_TEMPLATE='credential=TOP_SECRET_VALUE {context} {question}' \
SIMPLE_QNA_RAG_OLLAMA_BASE_URL='http://user:password@localhost:11434' \
python -m simple_qna_rag.cli.web --check-config \
  | rg -n 'TOP_SECRET_VALUE|password|credential'
```

출력은 `OLLAMA_BASE_URL`의 `password`와 `PROMPT_TEMPLATE`의
`TOP_SECRET_VALUE`를 그대로 포함했다. 현 테스트는 출력 **값**에 secret을
주입하지 않고 기본 출력에 `password|api_key|secret|credential`이라는 철자가
없는지만 확인하므로 결함을 가린다. REQ-002.5와 보안 Gate 위반이다.

필수 수정: field 단위 공개 정책을 schema에 선언하고 안전한 allowlist만 값으로
출력하거나 모든 문자열을 redacted metadata로 출력해야 한다. credential-bearing
URL, prompt, token-like 값의 adversarial subprocess 테스트를 추가해야 한다.

#### CR-I1-MAJ-02 — output-surface 감사가 제품 요청 경로의 164개 호출을 전부 `KEEP_CLI`로 오분류한다

Design §6.1은 `KEEP_CLI`를 `cli/*.py` 사용자 stdout과 로컬 uvicorn access
log로 제한하고, `agent.py`, `rag_engine.py`, web 요청 처리 경로는
`REPLACE`하도록 고정한다. 그러나 `logging_callsite_audit.py::_classify()`는
몇 개 파일/`RAGEngine.query`만 특별 처리한 뒤 **나머지 모든 파일을
`KEEP_CLI`로 반환**한다. 생성 artifact의 disposition은 `KEEP_CLI` 164건,
그 외 0건이다.

그 결과 실제 제품 호출인 다음 사례도 CI가 승인한다.

- `web_search.search_web`: 검색 query, 결과 title, exception 원문 출력
- `query_router.route_query`: 원 질문에서 추출된 검색어 출력
- `rag_engine.initialize` 및 하위 helper: 절대 vectorstore 경로와 exception
  원문 출력
- `intent_classifier.load_intent_classifier`: 모델 절대경로 출력

특히 `agent.route_query()`의 keyword fallback은 `query_router.route_query()`를
호출하므로 web `/rag` 요청에서도 검색어가 stdout에 노출될 수 있다.
`test_output_surface_capture.py`는 `simple_qna_rag.agent.route_query` 전체를
lambda로 mock하여 이 실제 fallback/web/retrieval 경로를 전혀 실행하지 않기
때문에 통과한다. REQ-003.3과 Design §6.1을 위반하며 “범위 밖 print 잔존”은
Gate 관점에서 허용할 수 없다.

필수 수정: disposition 정책을 설계와 일치시키고 실제 agent → fallback →
web/retrieval 경로를 민감 payload로 실행하는 동적 테스트를 추가한다. CLI
entrypoint에만 존재하는 사용자 대면 출력과 제품 library/request-path 출력을
구분해야 한다.

#### CR-I1-MAJ-03 — stage/error/fallback metrics와 structured events가 제품 경로에 연결되지 않았다

`observability/metrics.py`는 7개 collector family와 clamp 함수는 만들지만,
제품 코드에서 `rag_stage_duration_seconds`, `rag_stage_errors_total`,
`rag_fallback_total`을 호출하는 곳이 없다. 검색 결과상 이 심볼들의 제품 사용은
registry 생성뿐이다. `agent.py`의 web fallback과 `rag_engine.py`의 stored-vector
fallback도 log만 하거나 print할 뿐 해당 counter를 증가시키지 않는다.

마찬가지로 `web_search`/`retrieval` structured event는 allowlist에만 있고 제품
발행자가 없으며, `generation` event는 실패 시 `duration_ms=0.0`으로만
기록된다. `readiness` event도 선언됐지만 lifespan은 `startup`만 발행한다.
현재 cardinality 테스트는 제품 요청 1,000건이 아니라 registry를 직접 조작해
가능한 조합을 인위적으로 채우므로 “bounded schema”만 증명하고 REQ-004.1의
실제 제공 여부를 증명하지 않는다.

필수 수정: app registry를 안전한 observation sink로 요청 파이프라인에 주입해
routing/web/retrieval/generation duration/error 및 두 fallback을 실제로
계측하고, 실제 1,000개 고유 payload 요청 후 scrape sample ≤150과 기대 counter
증가를 검증한다. M3 `RetrievalTrace`는 변경하지 않는 별도 projection seam을
유지해야 한다.

#### CR-I1-MAJ-04 — M3 live 14-gate가 CI에도 없고 이번 검증에서도 `NOT_RUN`이다

`scripts/run_m4_regression_gate.py`와 mock 중심 integration test는 존재하지만
`.github/workflows/ci.yml`은 wrapper 자체를 호출하지 않는다. 직접 실행 결과도
`RUN_LIVE_LLM_TESTS=1` 미설정으로 exit 2였으며 routing/answers를 포함한 실제
14-gate 결과는 생성되지 않았다. baseline file에 대한 `git diff`만으로 M3 품질
보존을 증명할 수 없고, wrapper unit/integration mock은 실제 evaluator·Ollama·
runtime vectorstore 조합의 회귀를 대체하지 못한다.

설계는 exit 2를 환경 제약과 gate 실패를 구분하는 값으로 허용하지만,
Requirement §4는 `UNKNOWN`/`NOT_RUN`을 완료로 인정하지 않는다. 따라서 이번
Iteration에서 `RUN_LIVE_LLM_TESTS=1` 미실행은 **개발 중 진단상의 환경 제약일
수는 있어도 통합·인수 진입 Gate에는 MAJOR blocker**다.

필수 수정: credential/runner 정책을 정해 CI 또는 권위 있는 인수 환경에서
`RUN_LIVE_LLM_TESTS=1 python scripts/run_m4_regression_gate.py`를 실행하고,
exit 0, 14개 gate PASS, JSON/Markdown parity, baseline/vectorstore pre/post
불변 증거를 보존한다. CI에 wrapper step 또는 동일한 필수 인수 job을 연결한다.

### MINOR

#### CR-I1-MIN-01 — logging positive schema의 필수 key/type/enum이 런타임에서 강제되지 않는다

`_EVENT_KEYS`는 required/optional을 구분하지만 `_build_record()`는 이 구분을
allowed-key 합집합 생성에만 사용한다. 필수 key 누락, 잘못된 `route`, `stage`,
`error_code`, `level`, `status_code` 타입도 그대로 기록될 수 있다. 테스트도
각 정상 예시만 확인하고 negative schema matrix가 없다. Design §6.2의
“positive schema”와 fixture schema 100% 수용 기준을 충족하도록 strict 경로뿐
아니라 제품 non-strict 경로에도 안전한 default/clamp/drop 정책을 명시하고
검증해야 한다.

#### CR-I1-MIN-02 — Traceability/신규 CLI 테스트가 주장하는 실행 범위보다 좁다

Traceability REQ-002.4/.006.1은 세 CLI의 valid/invalid/unknown/override matrix와
초기화 미호출을 연결하지만 `test_cli_entrypoints.py`의 신규 probe는 query/index
두 모듈에서 `load_settings_or_exit()`만 직접 호출한다. 실제 `main()`을 실행하지
않고 parser 결과도 override mapping으로 변환하지 않으며 “engine/index
constructor not called” assertion도 없다. web 기본 serve 행의 subprocess
exit/override 연결도 실제 서버 시작 경계에서 검증하지 않는다. 현재 테스트는
Settings helper에는 유효하지만 entrypoint regression 증거로는 과장됐다.

### TRIVIAL

#### CR-I1-TRI-01 — lock 검증 script가 임시 `.body` 파일을 trap에서 제거하지 않는다

`compile_lock.sh`의 trap은 `tmp_a`/`tmp_b`만 지우고 `${tmp_a}.body`와
`${tmp_b}.body`를 남긴다. 재현성 판정에는 영향이 없지만 CI/local `/tmp`에
불필요한 artifact가 누적된다.

## 3. 요구사항 추적성 판정

| Requirement | 판정 | 독립 검증 |
|---|---|---|
| REQ-001 | **부분 PASS** | hash lock 102 package, CPU/no-nvidia 정적 검사, `compile_lock.sh --verify` PASS, snapshot schema tests PASS. Linux clean locked install/`pip check`/Node engine/`npm ci`는 실제 CI 결과가 아직 없으므로 최종 PASS 아님 |
| REQ-002 | **FAIL** | frozen typed Settings 41-field 및 legacy facade 테스트 PASS. 다만 `--check-config` credential/prompt 평문 노출(MAJ-01), 실제 3-CLI matrix 증거 과장(MIN-02) |
| REQ-003 | **FAIL** | request start/end 기본 matrix와 handler failure test는 존재. 실제 fallback/web/retrieval 출력 누출 및 잘못된 audit(MAJ-02), event 발행 누락(MAJ-03), positive schema 미강제(MIN-01) |
| REQ-004 | **FAIL** | fresh registry의 이론/합성 sample 102 및 created-series 0은 PASS. 실제 제품 stage/error/fallback 계측이 없어 REQ-004.1 FAIL(MAJ-03) |
| REQ-005 | **부분 PASS** | lifespan 기반 settings/engine/bootstrap 상태표, live/ready, deprecated alias 구현과 격리 테스트가 타당함. 로컬 FastAPI dependency 불일치 때문에 관련 테스트 4파일은 이번 전체 suite에서 collection 불가했고 locked Linux 결과 필요 |
| REQ-006 | **FAIL** | facade 및 mock wrapper test는 PASS. 실제 live M3 14-gate NOT_RUN, CI 미연결(MAJ-04); 따라서 regression/JSON-Markdown acceptance 미확정 |

## 4. 테스트 유효성·mock 과잉 판정

- `test_settings*`, dependency lock/snapshot, request ID, health 상태표의 pure/DI
  tests는 대상 계약을 직접 검증하며 유효하다.
- metrics 102 test는 collector cardinality 상한에는 유효하지만 실제 제품
  observation 연결을 검증하지 않는다. 직접 collector를 모두 조작하는 방식이
  제품 계측 누락을 가렸다.
- output capture와 request logging matrix는 `agent.route_query` 전체를 mock해
  민감 출력 위험이 가장 큰 fallback/web/retrieval 구현을 우회한다. mock 경계가
  지나치게 넓다.
- M3 wrapper test는 public API delegation/exit 계산 단위에는 유효하지만 live
  evaluator를 대체하지 않는다. 실제 gate는 별도 필수 증거다.
- CLI entrypoint probe는 `main()` 대신 settings bootstrap helper를 직접 부르므로
  이름과 달리 실제 entrypoint wiring을 충분히 검증하지 않는다.

## 5. 실행 결과

| 명령 | 결과 |
|---|---|
| `pytest -q` | **환경 제약 FAIL** — 4 integration module collection error. 로컬 env의 구 FastAPI가 Pydantic `EmailStr` schema 생성 중 설치되지 않은 `email-validator>=2`를 요구. 구현 lock은 FastAPI 0.141.1이므로 locked Linux CI 재검증 필요 |
| 선택 suite: `tests/unit` + M3 wrapper/check-config/CLI integration | **PASS — 521 passed**, warning 1 |
| `npm test` | **PASS — 1 file, 9 tests** |
| `npm run sync-vendor` + `git diff --exit-code -- web/static/vendor` | **PASS** |
| `python -m evaluation.dataset validate evaluation/datasets/golden.jsonl` | **PASS — 76 cases** |
| `bash scripts/compile_lock.sh --verify` | **PASS — 102 packages, two resolves identical, committed drift 0** |
| `python scripts/generate_field_spec.py --check` | **PASS** |
| `python scripts/logging_callsite_audit.py --check` | 형식상 **PASS**, 그러나 정책 자체가 MAJ-02로 무효 |
| `python scripts/check_markdown_links.py` | **PASS — 70 files, 302 links, failures 0** |
| `git diff --check` | **PASS** |
| `python scripts/run_m4_regression_gate.py` | **NOT_RUN/exit 2 — `RUN_LIVE_LLM_TESTS=1` 미설정** |

`pip check`는 현재 shared local environment의 기존 package 충돌을 여러 건
보고했다(torch/torchvision, langchain 계열, protobuf 등). 이는 clean locked
install 결과가 아니므로 구현 lock의 실패로 단정하지 않았으며, REQ-001.3의
권위 있는 증거는 Linux CI clean install 결과여야 한다.

## 6. 다음 Iteration 폐쇄 조건

1. check-config 출력 정책을 schema 기반 allowlist/redaction으로 수정하고 실제
   credential/prompt adversarial fixture를 통과시킨다.
2. output audit의 기본 분류를 설계대로 고치고, 실제 fallback/web/retrieval
   동적 capture에서 질문·검색어·답변·문서·exception·절대경로가 0건임을 증명한다.
3. stage/error/fallback metrics와 web/retrieval/generation events를 실제 제품
   sink에 연결하고 실제 요청 기반 1,000-payload scrape를 검증한다.
4. logging required key/type/enum negative matrix와 실제 세 CLI main wiring
   matrix를 추가한다.
5. clean Linux locked CI 전체 PASS와 `RUN_LIVE_LLM_TESTS=1` M3 14-gate exit 0
   증거를 확보한다.

위 항목을 폐쇄하기 전에는 M4.1 통합·인수 단계나 M4.2 구현으로 진행하면 안 된다.
