# M4.1 Configuration & Observability 상세 설계 리뷰 — Iteration 1

검토일: 2026-08-08  
검토자: Codex (독립 상세 설계 리뷰)  
대상: [Requirement.md](Requirement.md), [Plan.md](Plan.md),
[Design.md](Design.md), [Traceability.md](Traceability.md), 현행 제품 코드·테스트·CI  
상위 결정: [M4 복구 결정](../m4-production-readiness/Recovery_Decision.md)

## 1. 판정

**FAIL — 구현 단계 진입 불가**

- 점수: **4.7 / 10.0**
- 발견사항: **CRITICAL 0 / MAJOR 8 / MINOR 3 / TRIVIAL 2**
- 9.7 Gate: **미통과** (`CRITICAL/MAJOR 0`, MINOR 최소화, 9.7 이상 조건 불충족)

M4.1로 범위를 분리한 방향, 41개 공개 데이터 심볼의 실측, FastAPI lifespan
실패 특성, Prometheus created-series API 조사 자체는 유효하다. 그러나 settings 실패
정책과 health 상태표가 서로 양립하지 않고, typed settings의 실제 환경 파싱·CLI
실패 경로가 확정되지 않았으며, payload-safe logging과 필수 metrics를 구현할 수 있는
완전한 symbol/schema가 없다. lock 재현 spike도 관측 자체는 재현되지만 새 runtime
dependency와 lock 도구가 입력에 고정되지 않았고, M3 14-gate 및 JSON/Markdown 자동
증거 계약은 설계되지 않았다.

## 2. 검토 방법과 독립 재검증

- `milestone_dev_orchestration_guide.md`를 먼저 전부 읽고, Recovery Decision과 M4.1
  Requirement/Plan/Design/Traceability를 전부 검토했다.
- `config.py`, 세 CLI, `web/server.py`, `agent.py`, `rag_engine.py`, evaluation report
  경로, 현행 테스트와 `.github/workflows/ci.yml`을 symbol 단위로 대조했다.
- AST로 `config.py`의 public uppercase 데이터 심볼을 다시 세어 **41개**임을 확인했다.
  이 숫자 자체에 대한 Design §1.4의 주장은 맞다.
- 현행 저장소에서 `rg`로 확인한 `print()`는 `src/simple_qna_rag` 전체 **185건,
  9개 파일**이다. Design §2의 “71건, 3개 파일”은 전체 제품 logging inventory가 아니다.
- 설치된 `uv 0.8.15`로 Design §1.1의 lock 명령을 격리된 `/tmp` 출력에 재실행했다.
  102 packages/2,499 lines, `torch==2.13.0+cpu`, nvidia package 0은 재현됐다. 그러나
  생성 lock에는 `prometheus-client`가 없었고, 출력 파일 header까지 포함한 hash는
  `0a24f9...1421`로 문서의 축약 hash와 직접 비교할 canonical 절차가 없었다.
- 현재 환경의 `prometheus_client.disable_created_metrics()` public symbol과 Pydantic
  2.x는 확인했다. 이는 API 존재만 증명하며 lock된 공식 profile의 dependency 선언과
  동작을 증명하지 않는다.

## 3. CRITICAL

없음.

## 4. MAJOR

### M1-01 — invalid settings의 exit-2 정책과 `settings_invalid` health 상태가 양립하지 않는다

**근거**

- Requirement M4.1-REQ-002.4는 invalid enum/range/conflict가 모델·index 초기화 전에
  **exit code 2**로 실패해야 한다고 요구한다(Requirement:29).
- 동시에 REQ-005.1/.2와 Design §6.1은 settings가 invalid여도 앱이 떠서
  `/health/live` 200, `/health/ready` 503 `settings_invalid`를 반환한다고 정의한다
  (Design:277-281).
- Design의 lifespan 설명은 engine 예외만 catch한다고 명시하고 settings 실패를 누가,
  어느 경계에서 catch하는지 정의하지 않는다(Design:283-287).
- 현행 `web/server.py`는 module import 시 `agent`, `config`, `rag_engine`을 먼저 import하고
  `STATIC_DIR`/`TEMPLATES_DIR`로 앱을 구성한다(`web/server.py:18-33`). 설계대로 facade
  상수가 `get_settings()`를 호출하면 invalid settings는 lifespan 전에 module import를
  중단하므로 health route 자체가 존재할 수 없다.

**영향**

REQ-002.4와 REQ-005.1/.2를 동시에 만족하는 실행 상태 전이가 없다. 구현자는 web은
계속 serve할지 exit 2로 종료할지 임의로 선택해야 하고, acceptance 상태표 한 행은
반드시 실패한다.

**구체 수정안**

프로세스별 정책을 명시적으로 분리한다. 권고안은 (1) `--check-config`, query, index는
invalid settings에 exit 2, (2) web은 최소 bootstrap 설정만으로 app factory를 만들고
lifespan에서 전체 settings 오류를 typed/sanitized 상태로 저장해 live 200/ready 503을
제공하는 것이다. 이 예외가 요구사항의 “exit 2” 대상에서 제외됨을 Requirement에
명시하고, `create_app(settings_loader, engine_factory)`의 import-free 상태 전이와 web
subprocess exit/status 테스트를 Design에 추가해야 한다. 모든 프로세스를 exit 2로
정할 경우 `settings_invalid` health 행을 삭제해야 한다.

### M1-02 — Settings 단일 원본은 선언 형식만 있고 실제 env parsing/default/type 계약이 없다

**근거**

- Design은 41개 필드를 `Field(..., json_schema_extra={"env": ...})`로 둔다고만 한다
  (Design:150-159). `json_schema_extra`는 Pydantic `BaseModel`이 환경변수를 읽게 하는
  alias나 settings source가 아니다. `Settings.from_env()`가 이 metadata를 읽어 어떤
  값 변환·unknown key·빈 문자열·경로 resolution 규칙을 적용하는지 알고리즘이 없다.
- 41개 각각의 type/default/env alias/validator/consumer 표가 없다. 특히 현재 env-backed
  값은 path 3개와 M3 flags 4개이고, `PROJECT_ROOT`, template/static path처럼 파생값인
  필드의 생성 순서와 `resolve_runtime_path` conflict 처리도 정의되지 않았다.
- `SETTINGS_CONSUMERS`의 key-set equality만으로 consumer 정확성이나 facade 1:1 mapping은
  증명되지 않는다. 임의 문자열 tuple도 통과한다(Design:177-182, 215-217).
- `_env_bool`, `_env_enum`, `resolve_runtime_path`를 직접 env reader인 채 facade에
  유지한다는 Design:193-195는 “제품 모듈 direct env read 0” acceptance와 충돌한다.

**영향**

REQ-002.1/.2/.6의 단일 원본이 성립하지 않고, 구현자가 현재 M3 bool semantics,
legacy path conflict, 환경 alias를 서로 다르게 해석할 수 있다. schema count가 맞아도
값과 facade mapping은 틀릴 수 있다.

**구체 수정안**

machine-readable `FieldSpec` 또는 `Settings.model_fields` metadata 한 곳에 각 필드의
Python type, exact default/default factory, env name, parser, validator, derived dependency,
consumer를 완전 열거한다. `from_env(environ)`의 deterministic pseudocode와 unknown/empty/
bool/path 규칙을 고정하고, facade export mapping도 같은 spec에서 생성한다. AST 검사기는
실제 consumer set과 facade export set을 산출해 metadata와 비교하며, compatibility helper는
env를 읽지 않는 pure parser로 바꾸고 기존 import 이름만 유지해야 한다.

### M1-03 — 세 CLI의 validation/override/exit-code 경로가 현실 코드와 맞지 않는다

**근거**

- Design §1.4는 web/query만 env-write-before-import로 분석했고 index CLI를 빠뜨렸다.
  `cli/index_documents.py`는 module top-level에서 config 상수를 import한 뒤 main에서
  module global을 직접 바꾼다. 세 entry point가 동일한 Settings 생성 경로를 쓰지 않는다.
- `SettingsError(exit_code=2)` 객체를 정의하는 것만으로 process exit code가 2가 되지
  않는다. query는 initialization 예외를 catch해 exit 1로 바꾸고, index도 processing
  예외를 exit 1로 바꾸며, web의 delayed import 예외는 별도 mapping이 없다.
- `--check-config`는 web CLI에만 추가되며, 정상 실행 path가 같은 parser와 validation
  결과를 재사용하는지 정의되지 않았다. cache reset 후 facade의 이미 대입된 module
  상수가 갱신되지 않는 문제도 테스트되지 않는다.

**영향**

REQ-002.3/.4와 REQ-006.1의 세 CLI 호환을 입증할 수 없다. 같은 invalid env가 CLI마다
exit 1/2/traceback 또는 무시로 갈라질 수 있고, override가 import order에 의존한다.

**구체 수정안**

세 CLI 모두 `parse args -> build environ/override mapping -> Settings.from_env(mapping) ->
consumer factory` 순서로 통일하고 `SettingsError`를 최외곽 `main()`에서 exit 2로 변환한다.
index의 top-level config import/global mutation을 제거하는 구체 symbol 변경을 설계한다.
세 CLI 각각 valid/invalid/override subprocess matrix와 “engine/index constructor not called”
assertion을 추가한다. cache/reset은 제품 API에서 제거하거나 facade refresh semantics까지
명시하고 테스트한다.

### M1-04 — payload-safe logging 전환이 기존 payload 출력 경로를 제거하지 않는다

**근거**

- 현행 `src/simple_qna_rag`에는 `print()` 185건이 9개 파일에 있고,
  `web/server.py:96`은 질문 원문, `agent.py:228`은 검색어 원문을 출력한다.
- Design §2는 71건/3개 파일만 inventory하고, §5.2는 `/rag`와 `route_query`에 새 이벤트를
  “추가”한다고 할 뿐 기존 print를 제거·변환·격리하는 migration 표가 없다.
- 금지 key에만 `ValueError`를 던지면 다른 key 이름(`query`, `message`, `detail`)으로
  payload가 들어가는 것을 막지 못한다. 반대로 요청 경로에서 instrumentation 실수로
  `ValueError`가 나면 “logging failure가 요청을 실패시키지 않는다”는 REQ-003.4와도
  충돌한다.
- client 제공 `X-Request-Id`를 길이/문자 제한 없이 그대로 재사용한다(Design:225-227).
  이는 bounded field 계약과 로그 위조 방지에 필요한 제한이 없다.

**영향**

설계를 그대로 구현해도 질문·검색 원문이 stdout에 남아 REQ-003.3을 즉시 위반한다.
또한 logging 호출 자체가 제품 요청을 5xx로 만들거나, unbounded request ID가 로그와
metric-adjacent context에 유입될 수 있다.

**구체 수정안**

제품의 모든 stdout/stderr/logging callsite inventory와 disposition(구조화 event로 교체,
CLI user-facing output으로 명시적 유지, 삭제)을 표로 만든다. 허용 key+typed value 기반의
positive schema를 쓰고 문자열 길이 clamp/sanitization을 적용한다. runtime `log_event`는
절대 요청에 예외를 전파하지 않고 내부 failure counter/fallback stderr constant만 남기며,
strict validator는 테스트·개발 전용 별도 함수로 둔다. request ID는 UUID 생성 또는
허용 문자/최대 길이 검증 후 reject/regenerate한다. success/4xx/5xx/startup 전 경로의
stdout/stderr capture에서 금지 payload 0을 검사한다.

### M1-05 — metrics dependency와 family/label schema가 구현 가능한 수준으로 확정되지 않았다

**근거**

- `requirements.txt`에는 `prometheus-client`가 없다. Design §1.2가 조사한 0.23.1은 현재
  환경의 우연한 설치 상태이며, 독립 재생성한 공식 lock에도 이 package가 포함되지 않았다.
  그런데 Design §3은 requirements 입력에 이를 direct dependency로 추가하지 않는다.
- “Counter/Histogram/Gauge”, route 2/status 3/stage 4/error 4/reason 4만 있을 뿐 metric
  family 이름, type, label key 조합, enum 값, bucket 경계, fallback 표현이 없다
  (Design:238-245). REQ-004.1의 request/response status, stage error, route/**fallback**을
  어느 family가 충족하는지 알 수 없다.
- 67-sample spike의 script·registry·정확한 family schema가 저장소나 문서에 없어 결과를
  재현할 수 없다. global default registry를 쓰면 반복 import/test contamination과 기존
  process collector sample을 count할지 여부도 정의되지 않는다.

**영향**

clean locked install에서 metrics module import가 실패할 수 있고, 구현자가 임의 schema를
선택해 150 상한이나 fallback coverage를 깨뜨릴 수 있다. spike 숫자는 acceptance evidence로
감사할 수 없다.

**구체 수정안**

`prometheus-client==<검증 버전>`을 direct input에 명시하고 lock 재생성 대상으로 포함한다.
family별 `name | type | labels | exact allowed values | buckets | maximum samples` 표와 상한
계산식을 Design에 둔다. 모든 wrapper는 injected `CollectorRegistry`를 지원하게 하고,
spike script/fixture를 저장해 empty registry에서 1,000 unique payload 후 `collect()`의
실제 sample을 count한다. fallback counter/label과 endpoint content type/status 테스트도
명시한다.

### M1-06 — lock 생성 도구·canonical 비교·CI artifact 계약이 재현성을 닫지 못한다

**근거**

- Plan은 lock 도구/version 고정을 요구하지만 Design은 `uv` version과 CI 설치 방법을
  정하지 않았다. 현행 CI에도 uv setup step이 없다.
- lock 재생성 비교가 “header 제외 body diff”라고만 되어 있어 어떤 line을 제거하고
  무엇을 hash하는지 canonicalizer symbol이 없다. 독립 실행의 full-file hash는 output
  filename header 때문에 달라질 수 있다.
- `dependency_snapshot.py`는 stdout JSON만 정의하면서 “CI artifact로 첨부”한다고 하지만
  workflow output path, upload action, artifact name, retention/failure behavior가 없다.
  Traceability는 REQ-001.4 테스트를 명시적으로 `없음`으로 둔다.
- package count의 정의(locked distribution lines, package-lock packages, workspace root
  포함 여부)와 snapshot schema/version이 없다.

**영향**

clean CI에서 reproduce step 자체가 실행 불가하거나 uv update에 따라 lock이 변할 수 있고,
REQ-001.4의 canonical JSON은 생성돼도 검증·보존되지 않을 수 있다.

**구체 수정안**

고정 uv version 설치 step과 단일 `scripts/compile_lock.sh` 또는 Python wrapper를 설계하고,
고정 temp output name 및 명시적 canonicalization 규칙으로 두 번 생성한 결과를 비교한다.
snapshot에 schema version, official profile, tool versions, normalized package lists/counts,
두 lock hash를 정의하고 unit test 후 고정 artifact path로 upload한다. CI cache key도
`requirements.lock`으로 변경한다.

### M1-07 — 필수 logging/health acceptance matrix가 테스트 계획에서 누락됐다

**근거**

- Requirement acceptance는 success/4xx/5xx/startup fixture schema 100%, request start/end
  누락 0을 요구한다(Requirement:82).
- Design의 logging unit test는 schema/금지 key/handler failure만 다루고, integration health
  test는 engine 성공/실패와 health response만 본다(Design:262-271). `/rag` validation 422,
  uninitialized/engine error 5xx, `route_query` exception, client cancellation 등에서 start/end
  pair를 검사하는 테스트가 없다.
- startup/readiness “오류” event의 exact emission point와 request ID가 없는 process event에서
  required request_id를 어떻게 표현할지도 정의되지 않았다.

**영향**

REQ-003.1/.2 수용 기준을 통과했다는 자동 판정을 만들 수 없고, 가장 중요한 오류 경로에서
event pair가 빠질 가능성이 높다.

**구체 수정안**

ASGI middleware를 request start/end의 단일 소유자로 지정하고 정상, 404, 422, engine-not-
ready, handler exception 각각 status와 정확히 한 쌍을 검증한다. startup/readiness는
`request_id=null`을 허용하는 별도 process-event schema로 분리하거나 requirement를
명확히 수정한다. event capture용 injected sink와 table-driven acceptance matrix를
Design/Traceability에 연결한다.

### M1-08 — M3 14-gate 회귀와 JSON/Markdown 단일 판정 모델이 설계되지 않았다

**근거**

- Requirement acceptance는 “M3 14 gate와 baseline bytes 보존”을 요구하고 REQ-006.4는
  JSON/Markdown 결과가 같은 판정 모델에서 생성돼야 한다(Requirement:67,85).
- Design §8은 `pytest`, dataset validation 등 Plan 명령을 재기재할 뿐 M3 14-gate를
  어떤 기존 CLI/profile/input으로 실행하고 live dependency를 어떻게 충족하며 어떤
  result를 PASS로 판정하는지 없다.
- Traceability REQ-006.4는 symbol/test/evidence를 모두 “구현 단계 결정/미착수”로
  남겼다(Traceability:36). 이는 상세 설계 완료 상태 및 REQ별 design symbol 연결
  요구와 직접 충돌한다.
- baseline “bytes 보존”도 대상 두 파일의 pre/post hash 또는 `git diff --exit-code --
  evaluation/baselines/m3_initial.*` 중 무엇인지 정의하지 않는다. runtime vectorstore
  불변 검사 역시 path/fingerprint와 pre/post 비교가 없다.

**영향**

M4.1 완료 Gate의 핵심 회귀·자동 증거를 실행할 경로가 없고, M3 live 환경이 없는 CI에서
UNKNOWN을 잘못 PASS로 처리할 위험이 있다. Requirement → Design → Test → Evidence
추적성도 완결되지 않는다.

**구체 수정안**

M3 baseline evaluator의 기존 pure `gate_evaluation` model을 재사용하는 구체 command,
official inputs, live-required job 조건, result JSON path와 `overall_pass == true` 판정을
정의한다. Markdown은 그 동일 payload/model의 renderer로만 만들고 parity test를 둔다.
baseline 파일은 사전 고정 hash와 git diff를, runtime vectorstore는 시작/종료 fingerprint를
비교하며 mutation 없이 실패하게 한다. 실행 불가능한 live gate는 UNKNOWN으로 두고
M4.1 완료를 금지한다.

## 5. MINOR

### m1-01 — Phase 번호가 Plan과 Design 사이에서 어긋난다

**근거:** Plan은 Phase 1이 dependency+settings, Phase 2가 logging+metrics인데 Design은
dependency를 Phase 1, settings와 logging/metrics를 모두 각각 “Phase 2”라고 표기한다
(Design:116,146,221).  
**영향:** 구현 순서와 phase evidence 이름이 모호하다.  
**수정안:** Plan에 맞춰 Design section 번호와 Traceability evidence phase를 하나로 맞춘다.

### m1-02 — `/health` deprecated alias의 HTTP deprecation 계약이 부족하다

**근거:** 기존 response shape 유지만 있고 `Deprecation`/`Sunset` header 또는 문서화
방식과 제거 release 식별자가 없다. 또한 기존 `/health`는 engine이 없어도 `status:
healthy`였는데 ready 판정에서 파생하면 `status` 값이 바뀔 수 있다.  
**영향:** “응답 shape 유지”는 만족해도 기존 semantic 호환과 한 release 종료점을
검증할 수 없다.  
**수정안:** exact body matrix, headers, removal version을 고정하고 기존 상태 semantics를
보존할지 명시한다.

### m1-03 — `disable_created_metrics()`의 process-global side effect 소유권이 불명확하다

**근거:** metrics module import 시 global API를 호출하면 같은 process의 다른 library와
테스트 registry까지 영향을 받는다. Design은 “deployment 초기화 또는 public API” 중
module import side effect를 택하지만 호출 idempotency/ownership을 다루지 않는다.  
**영향:** import order와 test isolation에 따라 unrelated collector 결과가 달라질 수 있다.  
**수정안:** app bootstrap의 명시적 idempotent `configure_metrics()`가 collector 생성 전에
한 번 호출되도록 하고, unit test는 fresh registry/subprocess에서 순서를 검증한다.

## 6. TRIVIAL

### t1-01 — section cross-reference가 틀렸다

Design:53은 metrics wrapper를 “§4.2”라고 가리키지만 실제 정의는 §5.1이다. 구현자가
잘못된 facade section을 참조하지 않도록 링크를 고친다.

### t1-02 — 오탈자

Design:302의 `patttern`을 `pattern`으로 고친다.

## 7. 범위·호환성 판정

- **M3 호환성:** 이름 보존 의도는 있으나 exit code, bool parsing, facade cache,
  index CLI override, `/health` semantics, M3 14-gate 실행 계약이 닫히지 않아 미입증이다.
- **M4.2 범위 침범:** concurrency/timeout 구현은 포함하지 않았고 readiness reason seam만
  두어 대체로 경계를 지켰다. 다만 M4.2가 enum을 “추가만” 하면 된다는 설명은 metric
  cardinality 재계산·schema versioning 없이 안전하다고 단정할 수 없다.
- **M4.3 범위 침범:** index lifecycle/container 구현은 포함하지 않아 경계를 지켰다.
  현재 index CLI 설정 호환 수정은 기존 M3 동작 보존에 필요한 M4.1 범위이며 M4.3
  lifecycle 설계가 아니다.
- **spike 근거:** uv resolution, 41-field count, FastAPI lifespan 특성,
  Prometheus public API 방향은 현실적이다. 그러나 metrics 67-sample script와 exact schema,
  consumer inventory 산출물, lock canonical hash 절차가 보존되지 않아 독립 감사 가능한
  executable evidence로는 불충분하다.

## 8. 다음 Iteration의 최소 폐쇄 조건

1. invalid settings의 web serve-vs-exit 정책을 Requirement와 상태표에서 하나로 정합화한다.
2. 41개 `FieldSpec`, `from_env` 알고리즘, facade/consumer 생성과 세 CLI exit matrix를
   완전하게 설계한다.
3. 모든 제품 출력 callsite migration과 positive logging schema를 제시한다.
4. `prometheus-client` direct pin, exact metric family 표, 보존된 cardinality spike를 추가한다.
5. pinned uv/lock canonicalizer/snapshot artifact workflow를 실행 가능한 CI로 확정한다.
6. M3 14-gate, baseline/vectorstore immutability, JSON/Markdown parity의 command·symbol·test·
   evidence를 Traceability에 연결한다.
7. 위 수정 후 CRITICAL/MAJOR 0과 9.7 이상을 다시 독립 판정하기 전 구현으로 진행하지 않는다.

