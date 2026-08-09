# M4.1 상세 설계 독립 리뷰 — Iteration 2

검토일: 2026-08-08  
검토 대상: `milestone_dev_orchestration_guide.md`, M4.1 `Requirement.md`,
`Plan.md`, `Design.md`, `Traceability.md`, `Design_Review_Iteration_1.md`  
검토 방식: 문서 상호 대조와 현행 제품 코드·테스트·CI·evaluation API의 symbol 단위 재검증  
구현 변경: 없음

## 1. 판정

- **점수: 7.6/10**
- **Gate: FAIL** (`9.7/10` 미만, MAJOR 5건)
- **CRITICAL 0 / MAJOR 5 / MINOR 3 / TRIVIAL 0**
- Iteration 1 폐쇄 상태: **완전 폐쇄 6건, 부분 폐쇄 3건, 재개방 4건**
- 다음 단계: 설계를 다시 개정하고 독립 재리뷰하기 전 구현으로 진행하면 안 된다.

설계는 Iteration 1보다 상당히 구체화됐지만, 핵심 실행 계약 일부가 현행 API와
직접 불일치하거나 서로 모순된다. 특히 settings/CLI, metrics fallback, M3 gate runner는
현재 문서대로 구현하면 수용 조건을 만족할 수 없다.

## 2. 독립 재검증 근거

- `config.py`의 public uppercase 설정 값은 AST 기준 **41개**이며
  `resolve_runtime_path`를 더한 공개 대상 42개라는 기초 조사는 재현됐다.
- 현행 env-backed 이름은 `SIMPLE_QNA_RAG_VECTORSTORE_DIR`,
  `SIMPLE_QNA_RAG_DOCUMENTS_DIR`, `SIMPLE_QNA_RAG_MODEL_DIR`와 M3 flag 4개다.
  CLI는 override를 `os.environ`에 쓴 뒤 지연 import하고, index CLI만 top-level
  config import를 사용한다.
- `.github/workflows/ci.yml`은 현재 `requirements.txt` install/cache를 사용하며
  lock/uv/snapshot artifact 단계는 아직 없다. 이는 구현 전 상태로서 결함은 아니지만,
  Design의 신규 계약이 실제 workflow에 적용 가능한지를 대조하는 기준으로 사용했다.
- 실제 gate API는 `evaluation.compare.evaluate_gates(payload)`이고 반환값에
  `overall_pass`가 있다. 실제 report API는
  `evaluation.reporting.write_report(payload, output_dir, name,
  render_markdown=None)`이며 `format` 인자를 받지 않는다.
- runtime vectorstore는 `runtime/vectorstore`에 존재하며 baseline JSON은 이미
  `index.faiss`/`index.pkl` hash를 기록한다.

## 3. CRITICAL

없음.

## 4. MAJOR

### M2-01 — web bootstrap이 invalid path에서 health route 생존을 보장하지 못한다 (M1-01 부분 재개방)

**근거**

- Design §3.2는 `load_bootstrap()`이 세 path를 읽고 validator가 없어 실패하지
  않는다고 주장하지만 `create_app()`은 곧바로 `StaticFiles(directory=...)`를 mount한다.
  Starlette의 기본 directory check는 경로가 없거나 디렉터리가 아니면 app factory
  단계에서 실패할 수 있다.
- Requirement REQ-002.4의 web 예외는 invalid settings에서도 `/health/live`와
  `/health/ready`가 존재해야 성립한다. bootstrap path를 검증 밖에 뺀 것만으로 route
  생존이 증명되지 않는다.
- Requirement는 “PROJECT_ROOT 기반 고정 상수” bootstrap을 말하지만 Design은
  `PROJECT_ROOT/STATIC_DIR/TEMPLATES_DIR 3개 path만 읽는다`고 하여 env 입력인지 고정
  파생값인지도 불명확하다.

**영향**

path 설정 오류가 settings-invalid 503이 아니라 import/app 생성 실패로 돌아가
M1-01의 원래 실패 형태를 재발시킬 수 있다.

**필수 수정**

bootstrap 입력과 실패 정책을 정확히 고정하고, missing/not-directory static/template
각 경우에도 health route가 뜨는 subprocess/TestClient 증거를 설계한다. 또는 static/template은
사용자 설정에서 완전히 제외하고 package/repository 기준의 검증된 고정 경로만 사용해야 한다.

### M2-02 — `FieldSpec`/`from_env` 계약이 내부적으로 실행 불가능하다 (M1-02 재개방)

**근거**

- §4.1의 8-column `FieldSpec`에는 `derive` callable이 없는데 §4.3 알고리즘은
  `spec.derive(values)`를 호출한다.
- `default: object` 하나로 literal default와 default factory를 구분하지 못하며,
  `py_type: type`은 `Literal`, union, parameterized collection 같은 Pydantic annotation을
  일반적으로 표현하지 못한다.
- “Settings pydantic 모델을 tuple에서 생성”한다고 했지만 `create_model` 또는 동등한
  생성 symbol, frozen config, validator 결합 방법이 없다. 반대로 정적 Settings class를
  별도로 쓰면 FIELD_SPECS가 type/default의 단일 원본이 아니다.
- §4.2는 41개 전수를 구현 단계로 미루고 대표 예시에만 의존하며, Traceability §3도
  validator/default 모순 여부를 구현 후 확인한다고 명시한다. 상세 설계 Gate에서
  field/type/default/env/validator/consumer 전수 계약은 아직 닫히지 않았다.

**영향**

REQ-002.1/.2/.6의 단일 원본과 결정론적 생성 경로를 구현자가 임의로 보완해야 한다.
이는 M1-02의 핵심 폐쇄 조건을 충족하지 않는다.

**필수 수정**

`derive`, default factory, annotation, validators를 실제로 표현하는 완전한 spec type과
Settings 생성 symbol을 정의하고, 41개 전수 산출물을 설계 review artifact로 먼저
생성해 현재 defaults가 모든 cross-field validator를 통과함을 증명한다.

### M2-03 — CLI override 흐름이 기존 환경 전체를 버린다 (M1-03 재개방)

**근거**

- §5.1은 `build overrides mapping -> Settings.from_env(overrides)`를 세 CLI 공통
  흐름으로 고정한다.
- §4.3의 `from_env(environ)`은 인자가 주어지면 `dict(environ)`만 사용하고
  `os.environ`과 merge하지 않는다.
- 따라서 CLI flag 하나를 주면 다른 기존 env 설정과 M3 rollback flag가 모두 default로
  되돌아간다. 반대로 override가 없을 때 빈 mapping을 넘기면 모든 env가 사라진다.
- §5.1의 “env 설정 후 지연 import” 방식과 typed override mapping 방식도 동시에
  적혀 있어 어느 것이 정식 경로인지 불명확하다.

**영향**

REQ-002.3과 REQ-006.1의 환경변수/rollback 호환을 깨뜨리고, 같은 프로세스 입력이 CLI
flag 존재 여부에 따라 전혀 다른 Settings를 만든다.

**필수 수정**

base environment와 CLI override를 별도 인자로 받아 `base <- env <- explicit CLI`
우선순위를 한 symbol에서 merge하도록 고정한다. subprocess matrix에 기존 env와 다른
CLI override를 동시에 주어 비-overridden 값과 네 M3 flag가 보존되는 케이스를 추가한다.

### M2-04 — metrics schema가 필수 fallback metric을 제공하지 않는다 (M1-05 부분 재개방)

**근거**

- REQ-004.1은 `route/fallback` metrics를 모두 요구한다.
- §7.2의 다섯 family는 request, duration, stage error, readiness뿐이며 fallback
  counter/family/label이 전혀 없다.
- Iteration 1 M1-05의 구체 수정안도 fallback counter/label을 명시했지만 폐쇄 표와
  Traceability REQ-004.1은 누락을 PASS로 간주한다.

**영향**

문서 그대로 구현하면 REQ-004.1이 자동으로 실패하며, fallback family 추가 후에는
§7.3 sample 상한식과 67-sample 주장도 다시 계산해야 한다.

**필수 수정**

bounded fallback family의 name/type/labels/allowed values를 추가하고 created-series를
포함한 이론 최대 sample 및 1,000-request 실측을 다시 산출한다. 수식은 “관측된 조합”이
아니라 가능한 label 조합 전부에 대한 안전 상한도 별도로 제시해야 한다.

### M2-05 — M3 14-gate wrapper가 실제 evaluation API로 실행될 수 없다 (M1-08 재개방)

**근거**

- §10.1은 `write_report(format="json")`과 `write_report(format="markdown")`를
  각각 호출한다고 설계한다. 실제 `write_report`에는 `format` 인자가 없고 한 번의
  호출로 JSON/Markdown 두 파일을 함께 쓴다.
- 실제 Markdown gate renderer는 `evaluation.baseline`의 baseline payload 구조에
  `gate_evaluation`을 포함해 렌더링한다. 단순히 `evaluate_gates()` 반환값만
  `write_report()`에 넘기면 기존 M3 report와 같은 판정 표시 계약이 아니다.
- “공식 dataset/baseline 입력”이라고만 하고 retrieval/routing/answers의 정확한 report
  path 또는 이를 새로 생성하는 command를 고정하지 않았다. `evaluate_gates()`는
  baseline 파일 자체가 아니라 이 세 evaluator payload를 요구한다.
- live gate를 UNKNOWN으로 둔다고 했지만 현재 `evaluate_gates()`는 missing metric을
  `pass: null`로 표현할 뿐 별도 status `UNKNOWN`을 생성하지 않는다.

**영향**

신규 wrapper는 문서대로는 `TypeError`가 나거나 잘못된 payload를 렌더링한다.
REQ-006.4와 M3 14-gate 완료 증거가 생성되지 않으므로 M4.1 Gate를 닫을 수 없다.

**필수 수정**

실제 함수 signature와 renderer를 사용한 정확한 pseudocode, 공식 세 input 생성/선택
규칙, output path, exit status를 고정한다. missing input의 표현도 현행 `pass: null`과
용어를 맞추고, 기존 baseline renderer 또는 명시적인 신규 renderer를 지정한다.

## 5. MINOR

### m2-01 — `/health` deprecation 종료점이 여전히 placeholder다 (m1-02 부분 재개방)

§11.2의 `Sunset: <M4.2 목표일, Roadmap 확정 후 채움>`은 exact HTTP header가 아니다.
또한 “한 release”의 시작/종료 version이 없다. 구현자가 날짜를 정할 수 없도록 RFC에
맞는 고정 날짜와 제거 version을 Requirement/Roadmap과 함께 확정해야 한다.

### m2-02 — logging callsite audit가 “모든 출력”을 증명하지 않는다 (M1-04 부분 폐쇄)

§6.1 audit는 AST `print(...)`만 세므로 `logging.*`, `sys.stdout/stderr.write`, handler,
uvicorn access/error logger 같은 출력 표면을 탐지하지 않는다. 또한 line number를 artifact
key로 쓰면 앞선 편집만으로 disposition이 drift한다. callsite 종류 전체를 검사하고 AST의
안정적인 file/symbol/node identity를 쓰거나, acceptance에서 실제 stdout/stderr/log capture를
전체 경로에 적용해야 한다.

### m2-03 — `configure_metrics()`의 “정확히 한 번”과 idempotent 재호출 계약이 충돌한다

§7.4는 bootstrap에서 정확히 한 번 호출한다고 하면서 재호출 시 collector를 재사용한다고
한다. process-global `disable_created_metrics()`와 여러 app/fresh registry의 소유권도
명확히 분리되지 않는다. process 설정 1회와 registry별 collector 생성 N회를 서로 다른
symbol로 나누면 테스트 격리와 실제 수명주기가 명확해진다.

## 6. Iteration 1 지적별 폐쇄 재판정

| ID | 판정 | 재검증 결과 |
|---|---|---|
| M1-01 | 부분 폐쇄 | web exit-vs-health 정책은 정합화됐으나 bootstrap path 실패에서 health 생존 미보장(M2-01) |
| M1-02 | 재개방 | spec에 `derive`가 없고 Settings 생성/전수 field 계약 미완(M2-02) |
| M1-03 | 재개방 | override mapping이 base env를 대체함(M2-03) |
| M1-04 | 부분 폐쇄 | positive schema와 두 payload callsite는 개선됐으나 audit가 print만 포괄(m2-02) |
| M1-05 | 재개방 | 필수 fallback metric 누락(M2-04) |
| M1-06 | 폐쇄 | uv pin, canonical body, snapshot path/artifact 계약이 구현 가능한 수준으로 구체화됨 |
| M1-07 | 폐쇄 | middleware 단일 소유자와 5경로 acceptance matrix가 명시됨 |
| M1-08 | 재개방 | 실제 `write_report` API 및 evaluator input 구조와 불일치(M2-05) |
| m1-01 | 폐쇄 | Plan Phase 0~3 귀속이 정리됨 |
| m1-02 | 부분 폐쇄 | body/header 방향은 정해졌으나 Sunset/removal 값이 placeholder(m2-01) |
| m1-03 | 폐쇄 | import side effect 대신 bootstrap 호출점과 idempotency 테스트가 지정됨(표현 정리는 m2-03) |
| t1-01 | 폐쇄 | metrics 참조가 §7로 정정됨 |
| t1-02 | 폐쇄 | 오탈자 재발 없음 |

## 7. 새 모순·누락과 범위 판정

- **새 모순:** `FieldSpec` 정의에는 없는 `derive` 호출, `from_env(overrides)`의 base env
  손실, `write_report(format=...)`의 실제 API 불일치가 새로 확인됐다.
- **새 누락:** REQ-004.1의 fallback metric, exact M3 evaluator inputs, invalid bootstrap
  path health matrix가 빠졌다.
- **M4.2 범위:** queue/orphan 문자열 seam만 둔 것은 허용 범위다. 다만 §12의
  “반환값에 추가만”은 상태 우선순위, metrics allowlist, cardinality를 함께 바꿔야 하므로
  API 안정성을 과장한다. 실제 concurrency/timeout 구현 침범은 없다.
- **M4.3 범위:** index lifecycle/container 구현 침범은 없다. runtime vectorstore의
  read-only fingerprint는 M4.1 회귀 보존에 필요한 범위다. 다만 M3 gate를 새로 assemble하는
  표현은 M4.3의 최종 single-workflow assembly와 구분해 “M4.1 회귀 check”로 한정해야 한다.

## 8. 다음 Iteration 최소 폐쇄 조건

1. bootstrap path 실패에서도 live/ready route가 존재하는 상태 전이와 테스트를 고정한다.
2. 실행 가능한 완전한 `FieldSpec`과 41개 전수 generated artifact를 설계 증거로 제시한다.
3. base env와 CLI override merge 우선순위 및 혼합 subprocess matrix를 확정한다.
4. fallback metric을 추가하고 전체 sample 이론 상한/실측을 다시 계산한다.
5. 현행 `evaluate_gates`/`write_report` signature와 실제 evaluator payload에 맞춰 M3
   regression wrapper를 다시 설계한다.
6. `/health` Sunset/removal 값을 확정하고 logging audit 출력 표면을 확대한다.
7. 수정 후 CRITICAL/MAJOR 0, MINOR 최소화, **9.7/10 이상**을 독립 재리뷰로 확인한다.
