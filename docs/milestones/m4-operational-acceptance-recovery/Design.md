# M4 Operational Acceptance Recovery 상세 설계

상태: **상세 설계 — Gate PASS(Recovery Cycle 1 Iteration 3, 9.8/10.0) — 구현 완료(pre-merge)**
요구사항: [Requirement.md](Requirement.md)
계획: [Plan.md](Plan.md)
추적표: [Traceability.md](Traceability.md)
중단/정책 결정 보고서: [Stop_Report.md](Stop_Report.md)
기준 revision: `adda1759754b56b514b3ab6252c2dc1032e03d28`(`master`, PR #18 merge,
M4.3 deterministic PASS 보존)
작성자: Claude Code Sonnet 5 (상세 설계자, `milestone_dev_orchestration_guide.md`
"에이전트의 역할")
이 개정(iteration 5)은 [Design_Review_Iteration_4.md](Design_Review_Iteration_4.md)의
DR-I4-MAJ-01/DR-I4-MAJ-02/DR-I4-MIN-01만 닫는다 — §5.3(워크플로 stub 자체)과
§8.4(배너 문구)의 리터럴은 이전 iteration과 동일하게 유지한다. §3.1a의
이름-기반 "보호 심볼" allowlist 메커니즘(`PROTECTED_SYMBOLS`/
`_module_source_segments`/`audit_protected_symbols`, iteration 2~4에서 세 차례
수정됐던 바로 그 메커니즘)은 구조적으로 fail-open이라는 DR-I4-MAJ-01의
지적을 받아들여 **완전히 대체**한다 — 개별 심볼/바인딩-종류를 나열하는 대신
base revision 전체 top-level statement 시퀀스를 대상으로 하는 단일
whole-file allowed-delta oracle(`audit_exact_allowed_delta`, §3.1a)로
교체하고, 그 오라클을 exercise하는 §7.1 테스트 전체를 다시 작성한다.
§5.3의 "불가능한 broad self-hosted grep" 감사 명령 문구(DR-I4-MIN-01)도
§7.3의 파싱된 구조/실행-표면 검사를 가리키도록 고친다. 다른 모든 절과
DR-I1~DR-I3의 폐쇄 판정 자체는 그대로 보존한다 — §13~§15가 그 이력을
기록하고, 새로 추가한 §16이 이번 iteration의 폐쇄를 기록하며, §13~§15의
"iteration 4 최신 메커니즘" 포인터는 iteration 5의 새 메커니즘/테스트
이름을 가리키도록 갱신했다(판정 자체는 변경되지 않음 — 새 메커니즘은
기존 메커니즘이 잡던 모든 뮤턴트를 구조적으로 포함해서 잡는 상위집합이다).

이 개정(Recovery Cycle 1, Iteration 2)은
[Design_Review_Recovery_Cycle_1_Iteration_1.md](Design_Review_Recovery_Cycle_1_Iteration_1.md)의
DR-RC1-I1-MAJ-01 하나만 닫는다 — 그 밖의 모든 절, iteration 5(위 문단)가
도입한 whole-file default-deny 오라클의 비교 알고리즘, 세 pin된 델타, 그리고
DR-I1~DR-I4의 폐쇄 판정은 전혀 바꾸지 않는다. DR-RC1-I1-MAJ-01은 그 오라클
자체가 아니라 오라클이 소비하는 **슬라이스 생성 방법**의 결함이었다 —
`ast.get_source_segment(source, node)`는 decorated `ClassDef`/`FunctionDef`/
`AsyncFunctionDef`의 `class`/`def`/`async def` 토큰부터만 슬라이스를 만들어
decorator 줄을 비교 대상 밖에 남겼다. §3.1a에 `_statement_source_slice`
헬퍼를 신설해 decorator가 있는 세 노드 종류만 슬라이스 시작 좌표를 첫
decorator까지 확장하고, §7.1에 decorator 추가/제거/수정/재정렬 뮤턴트(정의
자체는 손대지 않은 채로, `assemble` 포함) 및 decorated class/async 합성
케이스를 추가한다. 다른 모든 절과 §13~§16의 폐쇄 판정은 그대로 보존하며,
새로 추가한 §17이 이번 개정의 폐쇄를 기록한다.

이 개정(Recovery Cycle 1, Iteration 3)은
[Design_Review_Recovery_Cycle_1_Iteration_2.md](Design_Review_Recovery_Cycle_1_Iteration_2.md)의
DR-RC1-I2-MAJ-01 하나만 닫는다 — 그 밖의 모든 절, §3.1a의 whole-file
default-deny 비교 알고리즘과 iteration 2가 도입한 decorator-span 확장, 세
pin된 델타, 그리고 DR-I1~DR-I4·DR-RC1-I1의 폐쇄 판정은 전혀 바꾸지 않는다.
DR-RC1-I2-MAJ-01은 그 비교 알고리즘 자체가 아니라 `audit_exact_allowed_delta`가
소비하는 `base_source`/`current_source` **문자열이 어떻게 만들어지는지**의
결함이었다 — `_top_level_statement_slices`는 `ast.parse(source).body`의
모듈 docstring 노드부터 시작하므로, 그 노드 이전의 shebang 줄과 PEP 263
encoding cookie 줄(또는 그 삽입/삭제)은 애초에 비교 대상 밖에 있었다.
새로 추가한 §3.1b에 `_source_preamble`/`audit_exact_allowed_delta_bytes`
두 함수를 신설해, base/current를 raw bytes로 읽어 그 맨 앞의 shebang·
encoding cookie·BOM만 byte-exact로 먼저 비교하고, 그 경계가 같을 때만
`tokenize.detect_encoding`이 그 bytes로부터 실제로 검출하는 encoding으로
디코드해 (한 글자도 바꾸지 않은) §3.1a `audit_exact_allowed_delta`에
넘긴다. §7.1에 shebang 수정/삭제/삽입, encoding cookie 삽입(non-ASCII
재현 포함)/수정/삭제, BOM 삽입, BOM-cookie 상충 fail-closed, 그리고 두
양성 경계 테스트를 추가한다. 다른 모든 절과 §13~§17의 폐쇄 판정은 그대로
보존하며, 새로 추가한 §18이 이번 개정의 폐쇄를 기록한다.

이 문서는 코드를 작성하지 않는다. `scripts/assemble_m4_evidence.py`,
`scripts/check_m4_baseline.py`, `.github/workflows/ci.yml`, 기존 테스트
3개 파일 + 신규 문서 감사 테스트 1개 파일, 문서/런북의 실제 변경을 구현
phase가 그대로 채택할 수 있는 수준까지 구체화한다. 인용한 현재 코드
라인은 이 세션이 직접 읽은 `scripts/assemble_m4_evidence.py`(366줄)와
`scripts/check_m4_baseline.py`(189줄) 기준이다.

## 0. 범위, 근거, 설계 원칙

### 0.1 이 설계가 바꾸는 것과 바꾸지 않는 것

바꾸는 것: M4 baseline의 **선언적 상태 스키마**(v1 → v2)와 그 스키마를 만들고
검사하는 두 스크립트, `m3-live-regression-gate` job의 **trigger 계약**, 관련
정적 테스트 3개 파일 + 신규 문서 감사 테스트 1개 파일, 지원 범위를
설명하는 문서 문구, 그리고 역사적 M4.1 runbook 상단의 superseded 배너
(본문은 무변경).

바꾸지 않는 것: M4.3가 이미 구현한 producer receipt 스키마
(`m43-producer-receipt-v1`), payload 검증 로직(`_check_identity`,
`_verify_payloads`, `_parse_and_verify_m43_payload`, `_evaluate_producer`),
M4.3 negative control, `index/`·`rag_engine.py`·`web/`·`cli/`의 제품 코드,
`python-tests`/`frontend-tests`/`container`/`m43-deterministic` 네 job의
내부 step. 이 설계는 **문서화된 정책 변경 + 스키마 마이그레이션**이며 M4.3가
이미 통과한 결정론적 evidence 파이프라인을 재구현하지 않는다
(Requirement.md §1, Plan.md §1 guardrail).

### 0.2 두 판정 경계 (Requirement §2 재확인)

```text
hosted_release_ready         = f(python-tests, frontend-tests, container,
                                  m43-deterministic 네 producer의 같은-run evidence)
native_linux_release_ready   = false  (상수, HOSTED_OCI 정책 아래 항상)
full_production_release_ready = false (상수, 위와 동일)
overall_release_ready        = full_production_release_ready  (호환 alias)
```

`hosted_release_ready`만 evidence-derived 변수다. 나머지 세 필드는 정책
상수이며 어떤 producer 결과로도 true가 될 수 없다 — `check_m4_baseline.py`가
이 세 필드를 **고정 리터럴 검사**(§4.3)로 강제하지, "evidence가 없으므로
false"라는 추론으로 얻지 않는다. 이 구분이 이 설계 전체의 핵심 불변식이다.

### 0.3 설계 원칙

1. **`NOT_ADOPTED`는 타입 수준에서 `PASS`가 될 수 없다.** v2 gate enum은
   정확히 `{"PASS", "FAIL", "NOT_ADOPTED"}`이고, `m3_live_regression`/
   `m41_operational` 두 키는 이 enum 안에서도 **"NOT_ADOPTED" 리터럴과
   정확히 같아야 한다"는 고정값 검사**를 추가로 받는다(§4.3 step 3). enum
   멤버십 검사와 고정값 검사가 이중으로 걸리므로, "PASS"·"WAIVED"·다른
   임의 문자열이 그 자리에 들어가면 최소 하나(대개 둘 다)가 항상 거부한다.
2. **checker는 어떤 self-report도 신뢰하지 않는다 — gate 값뿐 아니라
   신원(identity)과 파생 alias까지.** v1 checker가 이미 갖고 있던 원칙
   (`check_m4_baseline.py` 모듈 docstring, L4-9)을 v2에서도 유지하고
   범위를 넓힌다 — `deterministic_status`/`hosted_release_ready`/
   `operational_status`/`M4.1_BLOCKED`/`overall_release_ready`는
   candidate가 자체 보고한 값이 아니라 `producers[job].status`로부터
   checker가 독립적으로 재계산한 값과 비교되고, 여기에 더해 `git_sha`/
   `workflow_run`은 operator가 지정한 기대 신원과, `image_digest`/
   `m43_deterministic_receipt_sha256`는 같은 `producers[job].payload_hashes`
   로부터 checker가 재계산한 값과 각각 비교된다(§4.3, DR-I1-MAJ-02 폐쇄).
   이 identity/alias 재계산은 **candidate JSON 문서 내부의 일관성과
   operator가 요청한 신원과의 일치**만 증명한다 — 원본 payload 바이트를
   다시 내려받아 재해싱하지는 않는다; 그 바이트 검증은 assembler가 CI
   시점에 이미 수행했다(§4.7의 트러스트 경계 각주가 이 경계를 정확히
   기술한다).
3. **v1과 v2는 별도 코드 경로다.** 마이그레이션은 "v1을 읽어 v2로 변환"이
   아니라 "assembler는 이제 v2만 쓰고, checker는 schema 태그로 완전히 다른
   함수로 분기한다." v1 후보는 v1의 예전 대수 그대로 검사되며, 그 결과에
   `hosted_release_ready` 같은 v2 개념을 절대 주입하지 않는다
   (Requirement §M4-OAR-REQ-003.2). **`--allow-legacy-v1` 하나만으로
   REQ-003.2가 지정한 frozen 상태(§4.4)가 무조건 강제된다** — 별도의
   `--expect-operational-blocked` 플래그는 이 강제를 켜는 스위치가 아니라
   이미 강제된 상태를 재확인하는 잉여 CLI assertion일 뿐이다(DR-I1-MAJ-01
   폐쇄).
4. **fail-closed는 코드가 아니라 정확-key-set 비교와 고정값 비교로
   증명한다.** 기존 M4.3 checker가 이미 쓰는 관례(`set(candidate) !=
   REQUIRED_TOP_KEYS` 형태의 `==` 비교, 부분집합이 아님)를 v2 전체에도
   그대로 적용한다.
5. **워크플로 변경은 "미래 재활성화를 문서로 남기되 오늘은 절대 실행하지
   않는다"는 원칙을 따른다.** `m3-live-regression-gate`는 job으로서는
   남지만, ordinary push/PR에서는 절대 스케줄되지 않고, 명시적
   `workflow_dispatch` opt-in에서도 checkout/secret/environment
   승인/self-hosted 라벨 없이 즉시 종료되는 정보성 job이 된다(§5). 이
   무실행 계약은 파싱된 구조뿐 아니라 워크플로 소스 텍스트 자체에 대한
   금지 문자열 검사로도 증명한다(§7.3, DR-I1-MAJ-03 폐쇄).
6. **rollback은 schema/checker와 workflow를 독립적으로 되돌린다.**
   workflow rollback은 어떤 실패 조합에서도 ordinary push/PR에서
   self-hosted job이 다시 스케줄되는 §5.1 상태로 되돌아가지 않는다 —
   §11의 rollback matrix가 이 불변식을 세 실패 시나리오 모두에서
   증명한다(DR-I1-MAJ-05 폐쇄).
7. **역사적 문서는 삭제·재작성하지 않고 "superseded/non-executable"
   배너로만 표시한다.** `CI_Acceptance_Runbook.md`의 본문(과거 receipt·
   조사 기록)은 그대로 남기되, 상단에 현재 유효하지 않다는 배너를
   추가하고 이 배너의 존재 자체를 문서 감사 테스트의 allowlist 조건으로
   쓴다(§7.4, §8.4, DR-I1-MAJ-04 폐쇄).

## 1. Symbol Inventory

| 파일 | 상태 | 변경 요약 |
|---|---|---|
| `scripts/assemble_m4_evidence.py` | MODIFIED | §3 — v2 상수/`support_policy`/네 readiness 필드 추가. §3.1a의 whole-file allowed-delta oracle(`audit_exact_allowed_delta`)이 base revision 대비 **전체 top-level statement 시퀀스**를 비교해, 명시적으로 pin된 신규 v2 상수 삽입·`_build_baseline` 교체·`main()` exit 표현식 교체 세 가지 외의 모든 statement가 문자 그대로 동일함을 증명한다 — 이름/바인딩-종류를 나열하는 allowlist가 아니라 default-deny 방식이므로 줄 범위나 "byte-for-byte" 수사, 심볼 카테고리 열거 어느 쪽도 쓰지 않는다(DR-I1-MIN-01/DR-I2-MAJ-02/DR-I3-MAJ-01/DR-I3-MAJ-02/DR-I4-MAJ-01/DR-I4-MAJ-02 폐쇄). |
| `scripts/check_m4_baseline.py` | MODIFIED | §4 — v1/v2 schema-dispatch, v2 전용 검사 함수(신원/alias 재계산 포함, DR-I1-MAJ-02), v1 전용 frozen-blocked 무조건 강제(DR-I1-MAJ-01), 공유 producer-algebra 헬퍼, 신규 CLI 플래그(`--allow-legacy-v1`, `--expect-hosted-release-ready`, `--expect-hosted-not-ready`, `--expect-sha`, `--expect-run-id`, `--expect-run-attempt`, `--expect-workflow-path`, `--expect-event`). |
| `.github/workflows/ci.yml` | MODIFIED | §5 — `workflow_dispatch.inputs` 추가, `m3-live-regression-gate` job 전체 재정의, `m4-assemble`의 checker 호출 인자 변경. `python-tests`/`frontend-tests`/`container`/`m43-deterministic`/`m4-assemble`의 `needs`/step 순서는 불변. |
| `tests/unit/test_assemble_m4_evidence.py` | MODIFIED | §7.1 — 기존 6개 테스트 함수는 그대로 두고 v2 전용 테스트를 추가. |
| `tests/unit/test_check_m4_baseline.py` | MODIFIED | §7.2 — 기존 `_valid_candidate`/`test_strict_schema_and_algebra_matrix`를 v1-legacy 전용으로 이름 변경·보존하고, v2 전용 fixture/테스트(신원·alias·frozen-blocked mutant 포함)를 추가. |
| `tests/unit/test_ci_workflow_contract.py` | MODIFIED | §7.3 — `test_protected_live_gate_trigger_runner_environment_unchanged`를 대체하고, exact-shape/source-level/dependency-closure 계약 테스트를 추가. |
| `tests/unit/test_doc_audit_no_active_native_runner_procedure.py` | **NEW** | §7.4 — historical-vs-active 문서 감사를 배너 allowlist 기반으로 자동화(DR-I1-MAJ-04 폐쇄). |
| `README.md` | MODIFIED (문서만) | §8.2 — Ollama 안내를 development-only로 라벨링. |
| `docs/operations/deployment_runbook.md` | MODIFIED (문서만) | §8.3 — v2 artifact 다운로드/신원-바인딩 checker 절차 추가(§6.1), "유일한 정상 절차" 문구, no-SLA 문구 강화. |
| `docs/operations/recovery_runbook.md` | MODIFIED (문서만) | §8.3 — 상단에 hosted-only 인증 경계 참조 추가. |
| `docs/Roadmap.md`, `docs/Problem.md` | 이미 수정됨(감사만) | §8.1 — 이 세션 시작 시점에 이미 target 문구로 갱신돼 있음을 확인(diff 불필요). |
| `docs/milestones/m4.1-configuration-observability/CI_Acceptance_Runbook.md` | MODIFIED (문서만, 배너 삽입만) | §8.4 — 본문은 역사적 기록으로 그대로 두되, 상단에 superseded/non-executable 배너를 삽입해 현재 유효한 절차가 아님을 명시(DR-I1-MAJ-04 폐쇄). Plan.md §2 각주가 보존을 지시한 것은 이 문서가 아니라 아래 스크립트다. |
| `scripts/ci_acceptance_contract.py`, `scripts/preflight_ollama.py`, 관련 테스트 | **범위 밖** | §8.5 — Plan.md §2 각주가 명시한 대로 역사적 도구로 보존, 이번 milestone에서 수정 금지(코드 변경 없음). |

## 2. Baseline schema v2 정의

### 2.1 top-level key 차집합

v1 top-level(현행, `check_m4_baseline.py` L28-34의 `REQUIRED_TOP_KEYS`와
동일 — 이후 `REQUIRED_TOP_KEYS_V1`로 이름만 변경):

```text
schema, schema_version, generated_at, git_sha, workflow_run,
m3_fingerprint_reference, dependency_snapshot_sha256, settings_hash,
image_digest, m43_deterministic_receipt_sha256, producers, gates,
deterministic_status, operational_status, "M4.1_BLOCKED", overall_release_ready
```

v2 top-level(`REQUIRED_TOP_KEYS_V2`) = 위 15개 + 4개:

```text
support_policy, hosted_release_ready, native_linux_release_ready,
full_production_release_ready
```

`schema="m4-baseline-v2"`, `schema_version="2.0.0"`.

두 집합 모두 `==`(exact-set) 비교로만 쓴다. v2 후보에 v1 필드가 하나라도
빠지거나 v1 후보에 v2 필드가 하나라도 섞이면(mixed schema) 두 dispatch
분기 어느 쪽도 이 exact-set을 통과하지 못한다(§4.3/§4.4 첫 단계).

### 2.2 gates 하위 구조 v2

`gates` 딕셔너리의 key 집합은 v1과 동일(`REQUIRED_GATE_KEYS`, 6개:
`python_tests, frontend_tests, container, m43_deterministic,
m3_live_regression, m41_operational`) — 이름 자체를 바꾸면 기존 M4.3
producer→gate 매핑 코드와 이름 규칙 각주(`check_m4_baseline.py` L3484-3486
상당의 "하이픈 vs 언더스코어" 주석)를 건드리게 되므로 유지한다. 값의
허용 집합만 v2에서 좁아진다.

```python
GATE_ENUM_V2 = frozenset({"PASS", "FAIL", "NOT_ADOPTED"})
DETERMINISTIC_GATE_KEYS = frozenset({           # v1과 동일, 공유
    "python_tests", "frontend_tests", "container", "m43_deterministic",
})
FIXED_NOT_ADOPTED_GATE_KEYS = frozenset({"m3_live_regression", "m41_operational"})
```

`GATE_ENUM_V1`(기존 `GATE_ENUM`을 이름만 변경, 값 불변)은
`{"NOT_RUN", "SKIPPED", "UNKNOWN", "BLOCKED", "PASS", "FAIL"}`로 그대로
둔다 — v1 legacy 경로 전용이며 `NOT_ADOPTED`를 포함하지 않는다. 이 비대칭
자체가 "v1은 NOT_ADOPTED 개념을 모른다"는 요구사항(REQ-003.2 "compatibility
mode MUST NOT migrate... call v1 hosted-ready")을 구조적으로 강제한다.

### 2.3 `support_policy` 객체

```python
SUPPORT_POLICY_SCHEMA = "m4-support-policy-v1"
SUPPORT_POLICY_FIXED: dict[str, str] = {
    "schema": SUPPORT_POLICY_SCHEMA,
    "adopted_scope": "HOSTED_OCI",
    "native_linux_ollama": "NOT_ADOPTED",
    "decision_date": "2026-08-15",
}
SUPPORT_POLICY_KEYS = frozenset(SUPPORT_POLICY_FIXED)
```

네 값 모두 Requirement.md `M4-OAR-REQ-001.4`이 지정한 리터럴이며 런타임
계산이 아니다. `decision_date`는 정책 승인일 `2026-08-15`를 코드 상수로
고정한다(M4.3의 `EXPECTED_M43_NODE_IDS` 같은 "review-pinned literal" 관례와
동일한 패턴).

### 2.4 readiness algebra — assembler가 쓰고 checker가 재계산하는 공식

```text
deterministic_status = PASS iff (python_tests, frontend_tests, container,
                                  m43_deterministic 네 gate 모두 PASS) else FAIL
hosted_release_ready = (deterministic_status == "PASS")
native_linux_release_ready = False                       # 상수
full_production_release_ready = False                     # 상수
overall_release_ready = full_production_release_ready     # alias, 항상 False
operational_status = "NOT_ADOPTED"                          # 상수
"M4.1_BLOCKED" = False                                       # 상수 — REQ-001.5
```

`deterministic_status`의 공식 자체는 v1과 동일하다(네 deterministic
producer만 본다 — `m3_live_regression`/`m41_operational`은 애초에
`DETERMINISTIC_GATE_KEYS`에 없으므로 이 계산에 참여하지 않는다, 기존
`check_m4_baseline.py` L140-142와 동일 로직 유지). 바뀌는 것은
**deterministic_status 다음에 오는 파생 필드들의 이름과 값**이다: v1은
`operational_status`(BLOCKED 기반)와 `overall_release_ready`(deterministic
AND operational)를 계산했고, v2는 `operational_status`를 상수
`NOT_ADOPTED`로 고정하고 `overall_release_ready`를 `hosted_release_ready`가
아니라 항상 `False`인 `full_production_release_ready`의 alias로 재정의한다
— 즉 **hosted readiness가 아무리 true여도 overall/full/native는 절대
그 값을 물려받지 않는다.**

## 3. `scripts/assemble_m4_evidence.py` 변경

### 3.1 신규/변경 상수 (파일 상단, 기존 상수 블록 뒤에 추가)

```python
BASELINE_SCHEMA_V2 = "m4-baseline-v2"
BASELINE_SCHEMA_VERSION_V2 = "2.0.0"

SUPPORT_POLICY_SCHEMA = "m4-support-policy-v1"
SUPPORT_POLICY_DECISION_DATE = "2026-08-15"
SUPPORT_POLICY_FIXED = {
    "schema": SUPPORT_POLICY_SCHEMA,
    "adopted_scope": "HOSTED_OCI",
    "native_linux_ollama": "NOT_ADOPTED",
    "decision_date": SUPPORT_POLICY_DECISION_DATE,
}
```

`REQUIRED_PRODUCERS`, `RECEIPT_SCHEMA`, `RECEIPT_TOP_KEYS`,
`SEMANTIC_STATUS_ENUM`, `PAYLOAD_ENTRY_KEYS`, `REQUIRED_PAYLOADS`,
`KNOWN_PAYLOAD_FILENAMES`, `EXPECTED_M43_NODE_IDS`, `M43_*` 상수군은 §3.1a의
보호 심볼 목록에 속하며 변경하지 않는다(Plan.md §2 "keep... producers and
their verification unchanged").

### 3.1a Whole-file allowed-delta oracle — base 전체 top-level statement 시퀀스 비교로 "불변"을 증명한다 (DR-I1-MIN-01, DR-I2-MAJ-02, DR-I3-MAJ-01, DR-I3-MAJ-02, DR-I4-MAJ-01, DR-I4-MAJ-02)

**Iteration 2 리뷰가 지적한 오류:** 이전 iteration은 보호 대상을
"base revision L35-327"이라는 **하나의 연속 줄 범위**로 정의했다. 그러나
그 범위는 base revision에서 `_build_baseline`(L276-312)을 포함한다 —
§3.2가 바로 그 함수를 v2 반환값을 만들도록 통째로 교체하라고 요구하므로,
"L35-327을 건드리는 hunk는 전부 거부"라는 규칙은 이 설계 스스로 요구하는
필수 변경을 스스로 거부하는 자기모순이었다(DR-I2-MAJ-02). 또한 "그 범위는
`_build_baseline` 이전에서 끝난다"는 서술도 틀렸다 — `_dependency_snapshot_sha256`/
`_settings_hash`(base L315-327)는 `_build_baseline` **이후**에 있으면서도
같은 범위 안에 있었다. 마지막으로 `grep -E '^@@'`는 hunk 헤더를 사람이
읽도록 출력할 뿐, "겹침 여부"를 실제로 판정하는 코드가 아니었다 — 판정은
리뷰어의 수작업 해석에 맡겨져 있었다.

이 "줄 범위 대신 이름 붙은 심볼을 보호"하는 방향 자체는 iteration 2에서
iteration 4까지 세 차례에 걸쳐 다듬어졌다 — iteration 3은 뮤턴트가 AST
노드 슬라이스 **밖**에 공백을 추가해 감사를 통과시키는 결함(DR-I3-MAJ-01)과,
이름이 일치하는 **첫** top-level statement에서만 멈춰 뒤에 오는 재대입/
재정의로 실제 런타임 값이 우회되는 결함(DR-I3-MAJ-02)을 고쳤다(두 수정의
전체 서술은 §15에 감사 기록으로 보존한다).

**Iteration 4 리뷰가 지적한 오류 — 이름/바인딩-종류를 열거하는 접근 자체가
구조적으로 fail-open함(DR-I4-MAJ-01, DR-I4-MAJ-02).** 세 차례의 수정을
거치고도 이 접근은 "보호 심볼 26개의 `FunctionDef`/`Assign`/`AnnAssign`
바인딩"만 검사했다 — Python이 이름을 top-level에서 재바인딩할 수 있는
경로는 그보다 훨씬 많다(`import`/`from ... import ... as`, `class`,
`AsyncFunctionDef`, `for`/컴프리헨션 대상, `with ... as`, `except ... as`,
named expression 등). 이 경로 중 하나로 보호 심볼을 재바인딩하면
`_module_source_segments`가 애초에 그 statement를 찾지 않으므로
`audit_protected_symbols`는 위반을 하나도 보고하지 않는다(DR-I4-MAJ-01).
같은 근본 원인이 두 번째 결함도 만든다 — 감사는 26개의 **이름**만
비교했을 뿐, `import`·`assemble()`·`main()`의 exit 줄이 아닌 나머지·
새로 추가된 이름 없는 statement처럼 목록에 없는 나머지 파일 내용은 전혀
비교하지 않았다. 그 결과 실제 v2 patch를 포함하는 것은 증명했지만,
`assemble`을 임의로 재작성하거나 새 import·새 함수·`main()`의 다른 줄을
바꾼 patch까지 통과시켰다(DR-I4-MAJ-02) — "positive 포함"과 "negative
배제"를 구별하지 못한 것이다. 두 결함의 공통 원인은 같다: **이름이
붙은 대상만 나열하는 allowlist는, 그 나열이 아무리 세밀해져도 나열되지
않은 것에 대해서는 구조적으로 fail-open이다.**

**정정: 이름/바인딩-종류 열거를 전부 폐기하고, base 전체 top-level
statement 시퀀스에 대한 단일 whole-file default-deny 오라클로
교체한다.** 새 메커니즘은 어떤 이름이 무엇에 바인딩되는지 전혀 묻지
않는다 — base 소스의 `ast.parse(...).body`가 만드는 top-level statement
목록을 있는 그대로 순서대로 가져와, 그중 명시적으로 pin된 세 개의 델타
(신규 v2 상수 5개 삽입, `_build_baseline` 전체 교체, `main()` 전체 교체)
만 적용한 "허용되는 current" 시퀀스를 기계적으로 구성한 뒤, 실제
`current_source`의 top-level statement 시퀀스가 그것과 정확히
(개수·순서·문자열 모두) 같은지 비교한다. 이 비교는 statement의 **종류**를
전혀 구분하지 않으므로 — import든 class든 함수든 대입이든 loop든
with/try 블록이든 이름 없는 expression statement든 — pin된 세 델타가
아닌 자리에 나타나는 어떤 top-level statement도(신규 추가·기존 변형·재정렬
불문) 예외 없이 거부된다. DR-I3-MAJ-01/DR-I3-MAJ-02가 고쳤던 "슬라이스
안쪽 공백 삽입"과 "재바인딩을 통한 우회"도 이 비교의 특수 사례로
자동으로 잡힌다(둘 다 "어떤 top-level statement가 base와 다르거나 base에
없다"는 하나의 일반 규칙에 포함된다) — 별도의 뮤턴트-위치 규칙이나
바인딩-개수 규칙이 더 이상 필요 없다.

**Design Review Recovery Cycle 1, Iteration 1이 지적한 오류(DR-RC1-I1-MAJ-01)
— 슬라이스 생성 자체가 decorator를 빠뜨림.** 위에서 정정한 whole-file
비교 알고리즘은 "이름 열거" 결함(DR-I4-MAJ-01/02)을 없앴지만, 그 알고리즘이
소비하는 슬라이스를 만드는 방법(`ast.get_source_segment(source, node)`)이
그 자체로 불완전했다. Python 3.8부터 decorated `ClassDef`/`FunctionDef`/
`AsyncFunctionDef` 노드의 `lineno`/`col_offset`은 `class`/`def`/`async def`
토큰에서 시작하고, decorator는 별도의 `node.decorator_list`에만 기록된다 —
`ast.get_source_segment`는 노드 자신의 시작/끝 좌표만 쓰므로 decorator 줄은
반환된 슬라이스 밖에 남는다. 그 결과 `assemble` 바로 앞에 `@staticmethod`나
`@(lambda f: (lambda *a, **k: {}))` 같은 임의의 실행 가능한 decorator를
추가해 런타임 동작을 바꿔도 `_top_level_statement_slices`가 반환하는 문자열
목록은 base와 완전히 동일해 `audit_exact_allowed_delta`가 `[]`를 반환한다 —
decorator 추가/제거/수정/재정렬 전부가 이 오라클 밖에서 일어나는 감사되지
않은 실행 가능 접두부였다.

**정정: 슬라이스 시작 좌표를 decorator까지 확장한다.** decorator가 하나
이상 있는 `ClassDef`/`FunctionDef`/`AsyncFunctionDef`만, 슬라이스의 시작
좌표를 노드 자신이 아니라 **가장 이른(첫) decorator 표현식이 위치한 줄에서,
그 줄의 `@` 문자 위치**로 바꾸고, 끝 좌표는 그대로 노드 자신의
`end_lineno`/`end_col_offset`을 쓴다(decorator는 노드 자신의 끝 좌표에
영향을 주지 않는다 — decorator가 몇 개든, 몇 줄에 걸치든 노드의 본문 끝은
그대로다). decorator가 없는 statement, 그리고 `ClassDef`/`FunctionDef`/
`AsyncFunctionDef`가 아닌 모든 statement 종류(import/대입/loop/with/try/
named-expression 등 §3.1a의 나머지 전부)는 기존과 완전히 동일하게
`ast.get_source_segment(source, node)`를 그대로 쓴다. `@` 문자의 열 위치는
decorator 표현식의 `col_offset`으로 **가정하지 않고**, 그 줄의 텍스트에서
`col_offset` 이전 구간을 `rfind("@", ...)`로 역탐색해 구한다 — module
top-level statement는 들여쓰기가 없어 보통 `@`가 열 0이지만, `@`와 decorator
표현식 사이에 공백이 있는 경우까지 좌표 산술을 가정하지 않고 정확히
처리한다.

```python
import ast

ALLOWED_DELTA_BASE_REVISION = "adda1759754b56b514b3ab6252c2dc1032e03d28"

_DECORATABLE_NODE_TYPES = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _statement_source_slice(source: str, node: ast.stmt) -> str:
    """node 하나의 완전한 원본 텍스트 슬라이스를 반환한다. node가
    decorator가 하나 이상 있는 ClassDef/FunctionDef/AsyncFunctionDef가
    아니면 `ast.get_source_segment(source, node)`와 완전히 동일하다.
    decorator가 있으면(DR-RC1-I1-MAJ-01), 시작 좌표를 node 자신
    (`class`/`def`/`async def` 토큰)이 아니라 `node.decorator_list`의
    **첫** decorator 표현식이 있는 줄에서 그 줄의 '@' 문자 위치(좌표
    산술을 가정하지 않고 역탐색으로 구함)로 확장한다 — 그 뒤로 이어지는
    모든 decorator 줄과 `class`/`def`/`async def` 줄부터 node 자신의
    `end_lineno`/`end_col_offset`까지가 전부 슬라이스에 포함되므로,
    decorator 추가/제거/수정/재정렬 어느 쪽이든 반환 문자열이 달라진다."""
    decorator_list = getattr(node, "decorator_list", None) or []
    if not isinstance(node, _DECORATABLE_NODE_TYPES) or not decorator_list:
        return ast.get_source_segment(source, node)

    lines = source.splitlines(keepends=True)
    first_decorator = decorator_list[0]
    decorator_line = lines[first_decorator.lineno - 1]
    at_index = decorator_line.rfind("@", 0, first_decorator.col_offset)
    start_lineno = first_decorator.lineno
    start_col = at_index if at_index != -1 else 0
    end_lineno = node.end_lineno
    end_col = node.end_col_offset

    if start_lineno == end_lineno:
        return lines[start_lineno - 1][start_col:end_col]
    first_line = lines[start_lineno - 1][start_col:]
    middle_lines = "".join(lines[start_lineno:end_lineno - 1])
    last_line = lines[end_lineno - 1][:end_col]
    return first_line + middle_lines + last_line


def _top_level_statement_slices(source: str) -> list[str]:
    """base/current 소스를 파싱해 **모든** top-level statement의 정확한
    원본 텍스트 슬라이스를 파일에 나타나는 순서 그대로 반환한다.
    `ast.parse(source).body`를 그대로 순회하므로 statement의 종류를 전혀
    구분하지 않는다 — import(단순/`as`-aliased 모두), class, 동기/비동기
    def, 단순/annotated/augmented 대입, for/while/with(별칭 포함)/try
    (`except ... as` 별칭 포함) 블록, 최상위 named-expression을 포함한
    bare expression statement 등 Python이 module 최상위에 둘 수 있는 모든
    statement가 동일하게 "그 자리의 슬라이스 문자열 하나"로 취급된다.
    decorated ClassDef/FunctionDef/AsyncFunctionDef는 `_statement_source_slice`가
    가장 이른 decorator부터 포함하므로(DR-RC1-I1-MAJ-01), 반환되는 슬라이스는
    항상 그 statement의 완전한 실행 가능 소스 구간이며 어떤 decorator나
    실행 가능한 접두부도 비교 대상 밖에 남지 않는다. 주석과 빈 줄은 AST
    statement를 만들지 않으므로 애초에 이 목록의 어떤 원소도 되지 않는다 —
    decorator와 정의 사이에, 또는 top-level statement 사이에 주석/빈 줄이
    몇 줄 끼어들어도 그 사실만으로 statement 목록이 달라지지 않으며,
    주석은 실행 가능한 syntax를 감출 수 없다(§7.1
    `test_audit_exact_allowed_delta_comment_and_blank_line_insertions_between_statements_are_invisible`
    가 이 성질을 직접 증명한다)."""
    tree = ast.parse(source)
    return [_statement_source_slice(source, node) for node in tree.body]
```

이 정정은 세 pin된 델타 중 어느 것에도 영향을 주지 않는다 —
`PINNED_BUILD_BASELINE_OLD_SLICE`/`PINNED_MAIN_OLD_SLICE`는 base revision의
`_build_baseline`/`main()`을 그대로 옮긴 것이고, `PINNED_BUILD_BASELINE_NEW_SLICE`/
`PINNED_MAIN_NEW_SLICE`/`PINNED_NEW_CONSTANT_SLICES`는 §3.2/§3.3/§3.1의
코드 블록을 그대로 옮긴 것인데, base·v2 어느 쪽에도 이 네 함수/다섯 상수
자리에 decorator가 없다(base 전체에 top-level decorator가 하나도 없음은
`git show adda1759754b56b514b3ab6252c2dc1032e03d28:scripts/assemble_m4_evidence.py`를
`^@`로 확인 가능). 따라서 다섯 pin된 상수 리터럴은 문자 그대로 변경되지
않고, 바뀌는 것은 오직 `_top_level_statement_slices`가 (pin되지 않은
`assemble`을 포함해) base의 나머지 모든 statement에 decorator가 실제로
있을 때 그 decorator까지 포함해 슬라이스를 만드는지 여부뿐이다.

**세 가지 pin된 델타.** `PINNED_BUILD_BASELINE_OLD_SLICE`는 base
revision의 `_build_baseline` 함수 텍스트(아래 base 발췌와 문자 그대로
동일 — §3.2가 "기존 `_build_baseline`(L276-312)"이라고 부르는 바로 그
함수), `PINNED_BUILD_BASELINE_NEW_SLICE`는 §3.2 코드 블록의 함수 텍스트와
문자 그대로 동일하다. `PINNED_MAIN_OLD_SLICE`는 base revision의 `main()`
함수 텍스트(아래 base 발췌와 문자 그대로 동일), `PINNED_MAIN_NEW_SLICE`는
§3.3 코드 블록의 `main()`이되 문서 지면상 `...`로 줄인 argparse 블록
자리에 base와 동일한 열 개의 `parser.add_argument`/`args = parser.parse_args(argv)`
줄이 축약 없이 그대로 들어간, 마지막 `return` 한 줄만 base와 다른 완전한
텍스트다.

```python
# base revision `_build_baseline` 발췌 (참고용 — PINNED_BUILD_BASELINE_OLD_SLICE와
# 문자 그대로 동일):
#
# def _build_baseline(producers: dict, deterministic_status: str, args) -> dict:
#     gates = {}
#     for job, gate_key in {
#         "python-tests": "python_tests", "frontend-tests": "frontend_tests",
#         "container": "container", "m43-deterministic": "m43_deterministic",
#     }.items():
#         gates[gate_key] = "PASS" if producers[job]["status"] == "OK" else "FAIL"
#     gates["m3_live_regression"] = "NOT_RUN"
#     gates["m41_operational"] = "BLOCKED"
#
#     m43_receipt_sha = None
#     if producers["m43-deterministic"]["status"] == "OK":
#         m43_receipt_sha = producers["m43-deterministic"]["payload_hashes"].get("m43.json")
#     image_digest = None
#     if producers["container"]["status"] == "OK":
#         image_digest = producers["container"]["payload_hashes"].get("container_smoke.json")
#
#     return {
#         "schema": "m4-baseline-v1", "schema_version": "1.0.0",
#         "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
#         "git_sha": args.expected_sha,
#         "workflow_run": {
#             "run_id": args.expected_run_id, "run_attempt": args.expected_run_attempt,
#             "workflow_path": args.expected_workflow_path, "event_name": args.expected_event,
#         },
#         "m3_fingerprint_reference": "evaluation/baselines/m3_initial.json",
#         "dependency_snapshot_sha256": _dependency_snapshot_sha256(),
#         "settings_hash": _settings_hash(),
#         "image_digest": image_digest,
#         "m43_deterministic_receipt_sha256": m43_receipt_sha,
#         "producers": producers,
#         "gates": gates,
#         "deterministic_status": deterministic_status,
#         "operational_status": "BLOCKED",
#         "M4.1_BLOCKED": True,
#         "overall_release_ready": False,
#     }

# base revision `main()` 발췌 (참고용 — PINNED_MAIN_OLD_SLICE와 문자
# 그대로 동일):
#
# def main(argv: list[str] | None = None) -> int:
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument("--fresh-dir", required=True)
#     parser.add_argument("--expected-sha", required=True)
#     parser.add_argument("--expected-run-id", default="local")
#     parser.add_argument("--expected-run-attempt", default="1")
#     parser.add_argument("--expected-workflow-path", default=".github/workflows/ci.yml")
#     parser.add_argument("--expected-event", default="local")
#     parser.add_argument("--needs-result", action="append", default=[])
#     parser.add_argument("--evidence", action="append", default=[])
#     parser.add_argument("--output", default=None)
#     args = parser.parse_args(argv)
#
#     baseline = assemble(args)
#     text = json.dumps(baseline, sort_keys=True, ensure_ascii=False, indent=2)
#     output = Path(args.output) if args.output else Path(args.fresh_dir) / "m4-baseline.json"
#     output.parent.mkdir(parents=True, exist_ok=True)
#     output.write_text(text + "\n", encoding="utf-8")
#     print(text)
#     return 0 if baseline["deterministic_status"] == "PASS" else 1
# PINNED_MAIN_NEW_SLICE는 위와 문자 그대로 동일하되 마지막 줄만
# `return 0 if baseline["hosted_release_ready"] else 1`이다(§3.3).

PINNED_NEW_CONSTANTS_ANCHOR_SLICE = (
    'M43_NEGATIVE_KEYS = frozenset({"executed", "expected_to_fail", '
    '"actual_exit_code", "result"})'
)
# 위 anchor statement(기존 15개 상수 중 마지막)가 base에 정확히 한 번만
# 존재해야 삽입 지점이 유일하게 결정된다(§7.1 fixture 사전조건). anchor
# **바로 뒤**에, 정확히 이 순서로만 삽입이 허용되는 신규 top-level
# statement 다섯 개(§3.1과 문자 그대로 동일) — 이 목록에 없는 위치·순서·
# 개수의 삽입은 전부 거부된다.
PINNED_NEW_CONSTANT_SLICES = (
    'BASELINE_SCHEMA_V2 = "m4-baseline-v2"',
    'BASELINE_SCHEMA_VERSION_V2 = "2.0.0"',
    'SUPPORT_POLICY_SCHEMA = "m4-support-policy-v1"',
    'SUPPORT_POLICY_DECISION_DATE = "2026-08-15"',
    'SUPPORT_POLICY_FIXED = {\n'
    '    "schema": SUPPORT_POLICY_SCHEMA,\n'
    '    "adopted_scope": "HOSTED_OCI",\n'
    '    "native_linux_ollama": "NOT_ADOPTED",\n'
    '    "decision_date": SUPPORT_POLICY_DECISION_DATE,\n'
    '}',
)


def audit_exact_allowed_delta(base_source: str, current_source: str) -> list[str]:
    """빈 리스트면 감사 통과. base의 top-level statement 시퀀스에 정확히
    세 가지 pin된 델타(신규 상수 5개 삽입, `_build_baseline` 전체 교체,
    `main()` 전체 교체)만 적용해 "허용되는 current"의 statement 시퀀스를
    구성하고, 실제 current의 statement 시퀀스가 그것과 정확히 같은지
    (개수·순서·문자열 전부) 비교한다. 세 델타 중 하나에 해당하지 않는
    base statement는 current에서 문자 그대로 동일해야 하고, 그 세 델타가
    삽입/치환하는 자리가 아닌 곳에 나타나는 어떤 추가 statement도(종류
    불문) 거부된다 — 이것이 DR-I4-MAJ-02가 요구한 "완전한 base-to-v2 AST
    델타"이자 DR-I4-MAJ-01이 요구한 "임의의 미승인 top-level statement/
    델타에 대한 fail-closed" 양쪽을 동시에 만족하는 단일 메커니즘이다."""
    base_slices = _top_level_statement_slices(base_source)
    if base_slices.count(PINNED_BUILD_BASELINE_OLD_SLICE) != 1:
        return ["pinned_build_baseline_old_slice_not_uniquely_present_in_base"]
    if base_slices.count(PINNED_MAIN_OLD_SLICE) != 1:
        return ["pinned_main_old_slice_not_uniquely_present_in_base"]
    if base_slices.count(PINNED_NEW_CONSTANTS_ANCHOR_SLICE) != 1:
        return ["pinned_new_constants_anchor_not_uniquely_present_in_base"]

    expected_slices: list[str] = []
    for slice_ in base_slices:
        if slice_ == PINNED_BUILD_BASELINE_OLD_SLICE:
            expected_slices.append(PINNED_BUILD_BASELINE_NEW_SLICE)
        elif slice_ == PINNED_MAIN_OLD_SLICE:
            expected_slices.append(PINNED_MAIN_NEW_SLICE)
        else:
            expected_slices.append(slice_)
        if slice_ == PINNED_NEW_CONSTANTS_ANCHOR_SLICE:
            expected_slices.extend(PINNED_NEW_CONSTANT_SLICES)

    try:
        current_slices = _top_level_statement_slices(current_source)
    except SyntaxError:
        return ["current_source_not_parsable"]

    if current_slices == expected_slices:
        return []
    return [_first_divergence_violation(expected_slices, current_slices)]


def _first_divergence_violation(expected: list[str], actual: list[str]) -> str:
    """판정에는 관여하지 않는 진단 전용 헬퍼 — 위 리스트 비교가 이미 통과/
    거부를 확정한 뒤, 실패한 경우에만 첫 번째로 어긋나는 위치를 사람이
    읽을 수 있는 위반 코드로 보고한다."""
    for index, (expected_slice, actual_slice) in enumerate(zip(expected, actual)):
        if expected_slice != actual_slice:
            return f"top_level_statement_changed:index={index}"
    if len(actual) > len(expected):
        return f"unapproved_new_top_level_statement:index={len(expected)}"
    return f"missing_top_level_statement:index={len(actual)}"
```

`base_source`/`current_source`(둘 다 `str`)는 이 함수의 계약이며, 이
함수 자체는 그 두 문자열이 **어떻게 만들어지는지** 전혀 묻지 않는다 —
그 질문(raw bytes를 어떤 encoding으로 디코드할지, 디코드 이전의 shebang/
encoding-cookie 접두부를 어떻게 다룰지)은 DR-RC1-I2-MAJ-01을 닫는 §3.1b가
전담한다. §3.1b 이전에는 이 두 문자열이 각각 `git show
<base-revision>:scripts/assemble_m4_evidence.py`와 작업 트리 파일을
그대로 읽어 만든 것으로 서술돼 있었는데, 그 "그대로 읽는다"는 서술이
정확히 디코딩 경계를 생략한 지점이었다(§3.1b가 상세히 기술). 이 비교는
구현 phase가 실제로 실행하는 **테스트**(§7.1
`test_audit_exact_allowed_delta_bytes_positive_actual_v2_file`, §3.1b가
신설하는 raw-bytes 진입점을 통해)이지, 사람이 diff를 눈으로 판독하는
절차가 아니다. `git diff <base> --
scripts/assemble_m4_evidence.py`는 여전히 사람이 변경 개요를 훑어보는
보조 도구로 쓸 수 있지만, 더 이상 "감사 명령"으로서의 판정 권한을 갖지
않는다 — 판정 권한은 §3.1b `audit_exact_allowed_delta_bytes`(그리고 이
함수가 내부적으로 위임하는 `audit_exact_allowed_delta`)의 반환값(빈
리스트인지 아닌지)에만 있다. 그 반환값이 비어 있지 않으면 구현자
체크리스트(§12) 항목이 실패한다. 추가로, §7.1의 기존 6개 회귀 테스트
(수정하지 않음)가 v2 구현에서도 그대로 통과하는 것 자체가 "이 statement들의
관찰 가능한 동작이 바뀌지 않았다"는 의미적 보증이다 — whole-file 소스
비교가 정적 증거, 기존 6개 테스트가 동적 증거로 서로를 보완한다.

**이 메커니즘이 DR-I4-MAJ-01의 "완전한 scope-binding 분석"을 대체하는
방법.** `_top_level_statement_slices`는 statement의 종류를 전혀 검사하지
않으므로, `from attacker import REQUIRED_PRODUCERS`, `class
_evaluate_producer: ...`, `for REQUIRED_PRODUCERS in ...:`, `with ... as
_settings_hash:`, 두 번째 `async def _check_identity(...)`, 이름조차
재사용하지 않는 `(REQUIRED_PRODUCERS := ...)`류의 bare named-expression
statement — 이 모두는 "`PINNED_NEW_CONSTANTS_ANCHOR_SLICE` 바로 뒤에
정확히 다섯 개의 pin된 슬라이스만 삽입 가능"이라는 단일 규칙을 위반하는
**미승인 추가 top-level statement**로 동일하게 거부된다(§7.1의
`test_audit_exact_allowed_delta_rejects_*` 표가 이 바이패스들을 개별
mutant로 증명한다).
바인딩 종류를 열거해 그중 위험한 것을 판별하던 이전 접근과 달리, 이
오라클은 열거되지 않은 새 종류의 top-level statement가 나타나도 자동으로
안전한 쪽(거부)으로 fail한다 — "무엇을 막을지 나열하는" allowlist가
아니라 "base에 있던 것과 pin된 세 델타만 허용하는" default-deny이기
때문이다.

### 3.1b Preamble byte/token-aware file-loading 경계 (DR-RC1-I2-MAJ-01)

**Design Review Recovery Cycle 1, Iteration 2가 지적한 오류.** §3.1a의
`_top_level_statement_slices`는 `ast.parse(source).body`에서 시작한다 —
즉 모듈 docstring AST 노드부터가 비교 대상이고, 그 노드 **이전**의 소스
텍스트는 이 함수가 만드는 슬라이스 목록의 어떤 원소도 되지 않는다. base
revision(`adda1759754b56b514b3ab6252c2dc1032e03d28`)의 실제 첫 줄은
shebang `#!/usr/bin/env python3`이고, encoding cookie는 없으며, 둘째
줄부터 non-ASCII em dash(`—`, U+2014)를 포함한 모듈 docstring이 바로
시작한다(`git show <base-revision>:scripts/assemble_m4_evidence.py`의
첫 두 줄로 직접 확인 가능 — §7.1
`test_source_preamble_matches_pinned_base_preamble_bytes`가 이 사실 자체를
고정한다). 따라서 다음 두 변형이 whole-file 오라클 밖에서 일어난다:

```python
#!/usr/bin/env -S python3 -O
# coding: latin-1
```

shebang은 일반 주석이 아니라 **실행 경계**다 — 이 파일은 실행 가능하고,
직접 호출(`./scripts/assemble_m4_evidence.py`)은 인터프리터 선택/옵션을
이 줄에 위임한다. encoding declaration은 일반 주석이 아니라 **디코딩
경계**다 — PEP 263에 따라 CPython은 이 줄(또는 그 부재)에 따라 소스
바이트를 문자열로 디코드하는 방법을 정한다. base 파일의 UTF-8 바이트를
그대로 두고 둘째 줄에 `# coding: latin-1`만 삽입하면, CPython이 실제로
실행할 모듈 docstring은 mojibake로 바뀌고(em dash가 `â\x80\x94`류의
문자로 깨짐), `argparse.ArgumentParser(description=__doc__)`가 만드는
CLI 도움말 출력도 함께 바뀐다. 그럼에도 `audit_exact_allowed_delta(base_source,
current_source)`는 이미 디코드된 `str` 두 개를 받으므로 — 그 디코드
단계가 cookie를 실제로 반영했는지와 무관하게 — cookie는 그 함수 안에서는
그저 한 줄의 텍스트일 뿐이고, `ast.parse`는 여전히 유효한 모듈을 만들어
`tree.body`의 top-level statement 시퀀스는 base와 정확히 동일하게
남는다. shebang 삭제/수정도 마찬가지로 보이지 않는다 —
`ast.get_source_segment`가 반환하는 모듈 docstring 슬라이스는 그 노드의
시작/끝 좌표 사이의 **문자열**만 담고 줄 번호 자체는 담지 않으므로,
shebang이 있든 없든 그 슬라이스 문자열은 문자 그대로 동일하다.

**정정: `base_source`/`current_source`를 `str`이 아니라 raw bytes로
읽고, 그 bytes 맨 앞에서 shebang과 PEP 263 encoding cookie만 정확히
골라내 byte-exact로 먼저 비교한 뒤, 그 경계가 같을 때만
`tokenize.detect_encoding`(CPython 자신의 PEP 263 구현)이 그 bytes로부터
실제로 검출하는 encoding으로 디코드해서 (한 글자도 바꾸지 않은) §3.1a
`audit_exact_allowed_delta`에 넘긴다.** 이 경계 함수는 shebang/encoding
cookie가 아닌 어떤 주석·빈 줄도 preamble로 취급하지 않는다 — 예를 들어
cookie가 없는 base 파일에서 shebang 바로 다음 줄(둘째 줄)은 모듈
docstring의 시작이지 cookie가 아니므로 preamble에 포함되지 않고, 이미
확립된 "주석/빈 줄은 AST statement를 만들지 않는다"는 성질 아래 통상
statement 영역의 inert gap으로 남는다(§3.1a 마지막 문단). 즉 이 경계
함수는 "encoding cookie 검출을 위해 CPython 자신이 실제로 검사하는 최대
두 줄"이라는 PEP 263 자신의 범위 밖으로 preamble 정의를 넓히지 않는다 —
임의의 파일 앞부분 주석까지 "실행 경계"로 과잉 확장하지 않는다.

```python
import io
import re
import tokenize

_SHEBANG_PREFIX = b"#!"
_UTF8_BOM = b"\xef\xbb\xbf"
_ENCODING_COOKIE_RE = re.compile(rb"^[ \t\f]*#.*coding[:=][ \t]*([-_.a-zA-Z0-9]+)")

PINNED_BASE_PREAMBLE_BYTES = b"#!/usr/bin/env python3\n"
# base revision(adda175...)의 정확한 첫 줄 그대로. 그 파일에는 PEP 263
# encoding cookie가 없으므로(둘째 줄부터 바로 모듈 docstring이 시작한다),
# pin된 preamble은 이 shebang 한 줄뿐이다. §7.1
# `test_source_preamble_matches_pinned_base_preamble_bytes`가
# `_source_preamble(base_bytes) == PINNED_BASE_PREAMBLE_BYTES`를 직접
# 확인한다.


def _source_preamble(raw: bytes) -> bytes | None:
    """raw 소스 bytes 맨 앞에서, 실제 실행/디코딩 경계로 의미 있는 것만
    ── 선택적 UTF-8 BOM, 그 뒤 파일의 실제 첫 두 바이트가 정확히 `#!`인
    경우에만 인정하는 shebang 줄, PEP 263이 정의하는 위치(shebang이
    있으면 그 다음 줄, 없으면 첫 줄)에 있는 encoding cookie 줄 ──만 골라
    원본 바이트 그대로 이어붙여 반환한다. `tokenize.detect_encoding`이
    cookie 검출을 위해 실제로 읽는 최대 두 줄(`consumed`)을 후보로
    쓰되, 그중 shebang도 cookie도 아닌 줄(예: cookie 없는 base 파일의
    둘째 줄인 모듈 docstring 첫 줄)은 preamble에서 제외한다 — 그런 줄은
    이미 statement 영역의 inert gap이다. `tokenize.detect_encoding`은
    `raw`를 BOM을 벗기지 않은 원본 그대로 받는다 — BOM을 미리 벗기고
    호출하면 CPython 자신이 정의하는 "BOM은 UTF-8을 함의하는데 cookie가
    다른 encoding을 선언하는" 상충 조건 자체를 그 함수가 볼 수 없게 되어
    탐지하지 못한다(BOM 유무는 반환된 `encoding`이 `\"utf-8-sig\"`인지,
    또는 `raw.startswith(_UTF8_BOM)`인지로 별도 확인한다). `raw`가
    encoding 검출 단계에서부터 파싱 불가능하면(예: 방금 말한 BOM-cookie
    상충, PEP 263이 SyntaxError로 정의하는 조건) `None`을 반환해 호출자가
    fail-closed 처리하게 한다."""
    try:
        _, consumed = tokenize.detect_encoding(io.BytesIO(raw).readline)
    except SyntaxError:
        return None
    bom = _UTF8_BOM if raw.startswith(_UTF8_BOM) else b""
    has_shebang = bool(consumed) and consumed[0].startswith(_SHEBANG_PREFIX)
    cookie_index = 1 if has_shebang else 0
    has_cookie = (
        cookie_index < len(consumed)
        and _ENCODING_COOKIE_RE.match(consumed[cookie_index]) is not None
    )
    kept_line_count = (1 if has_shebang else 0) + (1 if has_cookie else 0)
    return bom + b"".join(consumed[:kept_line_count])


def audit_exact_allowed_delta_bytes(base_bytes: bytes, current_bytes: bytes) -> list[str]:
    """§3.1a `audit_exact_allowed_delta`의 raw-bytes 진입점 — 감사에
    실제로 쓰이는 함수는 이제 이 함수다(§7.1
    `test_audit_exact_allowed_delta_bytes_positive_actual_v2_file`, §12
    구현자 체크리스트). `base_bytes`는 `git show <base-revision>:
    scripts/assemble_m4_evidence.py`의 stdout bytes를 디코드하지 않고
    그대로, `current_bytes`는 작업 트리 파일을 바이너리 모드로 그대로
    읽는다. 두 preamble이 byte-exact로 다르면(shebang 수정/삭제/삽입,
    encoding cookie 삽입/수정/삭제, BOM 삽입/삭제 전부 포함) 그 자리에서
    거부하고 §3.1a의 statement-시퀀스 비교는 아예 실행하지 않는다 — 이미
    이 시점에서 실행/디코딩 경계 자체가 base와 달라졌기 때문이다.
    preamble이 같으면(따라서 검출되는 encoding도 같으므로) 각자 자신의
    encoding으로 전체 bytes를 디코드해 §3.1a `audit_exact_allowed_delta`
    (한 글자도 바뀌지 않음)에 그대로 넘긴다."""
    base_preamble = _source_preamble(base_bytes)
    if base_preamble is None:
        return ["base_source_encoding_conflict"]
    current_preamble = _source_preamble(current_bytes)
    if current_preamble is None:
        return ["current_source_encoding_conflict"]
    if current_preamble != base_preamble:
        return ["preamble_shebang_or_encoding_declaration_changed"]

    base_encoding, _ = tokenize.detect_encoding(io.BytesIO(base_bytes).readline)
    current_encoding, _ = tokenize.detect_encoding(io.BytesIO(current_bytes).readline)
    try:
        base_text = base_bytes.decode(base_encoding)
    except UnicodeDecodeError:
        return ["base_source_undecodable"]
    try:
        current_text = current_bytes.decode(current_encoding)
    except UnicodeDecodeError:
        return ["current_source_undecodable"]

    return audit_exact_allowed_delta(base_text, current_text)
```

**이 partition이 정확히 증명하는 것과 증명하지 않는 것.** 파일은 정확히
두 개의 서로소(disjoint) 영역으로 나뉜다: (1) **preamble** —
`_source_preamble`이 반환하는, 선택적 BOM + 선택적 shebang 줄 + 선택적
encoding cookie 줄로만 이루어진 byte-exact 접두부. (2) **statement
영역** — 그 뒤의 나머지 전부로, `ast.parse(...).body`가 만드는
top-level statement 시퀀스 각각이 (decorator가 있으면 그 decorator부터
시작하는, §3.1a `_statement_source_slice`가 만드는) 완전한 소스 슬라이스로
비교된다. 파일의 모든 바이트는 이 두 영역 중 하나에 속한다 — 그러나
**비교 대상**이 되는 것은 이 두 영역 자체가 아니라 (a) preamble의 정확한
바이트열과 (b) statement 영역 안의 AST top-level statement들이 만드는
슬라이스 문자열들뿐이다. 주석과 빈 줄은 — 지금까지와 똑같이 — preamble의
두 후보 위치(shebang 줄, cookie 줄) 중 하나가 아닌 한 어떤 AST statement도
만들지 않으므로 비교 대상 밖에 남는다: statement 사이, decorator 사이,
preamble 바로 뒤 첫 statement 앞(예: cookie 없는 base 파일에서 shebang
다음에 오는 임의의 주석 줄이 있다면 그 줄 자체 — §7.1
`test_audit_exact_allowed_delta_bytes_accepts_leading_non_cookie_comment_as_inert`가
이 경우를 직접 증명한다), 마지막 statement 뒤(EOF 근처)의 주석/빈 줄
전부가 그렇다. **따라서 이 메커니즘은 "current 파일이 base+세 델타와
byte-for-byte 동일하다"를 증명하지 않는다** — 그런 수사는 §3.1a가 이미
명시적으로 피하고 있으며(§1 Symbol Inventory 표), 이 정정도 그 수사를
도입하지 않는다. 이 메커니즘이 증명하는 것은 정확히 다음 두 가지뿐이다:
(i) 실행/디코딩 경계(shebang, encoding cookie, BOM)가 base와 문자 그대로
동일하다, 그리고 (ii) 그 경계가 동일하므로 같은 encoding으로 디코드된 두
텍스트에서 모든 top-level statement(모듈 docstring 포함, decorator
포함)가 pin된 세 델타를 제외하고 문자 그대로 동일하다. inert 주석/빈
줄이 두 파일 사이에서 달라져도(그 자리가 preamble도 statement 슬라이스
안쪽도 아닌 한) 감사는 그 차이를 보지 않는다 — 이것은 결함이 아니라
§3.1a가 처음부터 선언한 "주석은 실행 가능한 syntax를 감출 수 없다"는
성질의 직접적 귀결이며, 이번 정정은 그 성질이 적용되는 경계를 preamble의
두 줄까지로 정확하게 넓혔을 뿐이다.

이 정정은 §3.1a의 `_statement_source_slice`/`_top_level_statement_slices`/
`audit_exact_allowed_delta`/`_first_divergence_violation`이나 세 pin된
델타 리터럴 중 어느 것도 바꾸지 않는다 — base revision의 preamble이
shebang 한 줄뿐이고 다섯 pin된 상수/`_build_baseline`/`main()` 중 어느
것도 preamble 안에 있지 않으므로, 새 경계 함수가 preamble 동일성을
확인한 뒤에는 기존 로직이 정확히 이전과 동일하게 실행된다. 이는 §17이
기록한 대로 decorator-span 확장이 세 pin된 델타 리터럴과 statement 비교
알고리즘을 전혀 바꾸지 않았던 것과 동일한 패턴이다 — 이번 정정도 오직
"어디까지가 preamble이고 어디부터가 statement 시퀀스인가"라는 file-loading
경계 하나만 넓힌다.

### 3.2 `_build_baseline` — v2 전용으로 재작성

기존 `_build_baseline`(L276-312)을 아래로 교체한다. `producers` 계산·
`gates`의 네 deterministic 값 계산·`m43_receipt_sha`/`image_digest` 추출·
`_dependency_snapshot_sha256`/`_settings_hash` 호출은 **그대로 재사용**하고,
반환 dict의 마지막 6개 필드만 v2 값으로 바뀐다.

```python
def _build_baseline(producers: dict, deterministic_status: str, args) -> dict:
    gates = {}
    for job, gate_key in {
        "python-tests": "python_tests", "frontend-tests": "frontend_tests",
        "container": "container", "m43-deterministic": "m43_deterministic",
    }.items():
        gates[gate_key] = "PASS" if producers[job]["status"] == "OK" else "FAIL"
    gates["m3_live_regression"] = "NOT_ADOPTED"      # 변경: 이전 "NOT_RUN"
    gates["m41_operational"] = "NOT_ADOPTED"          # 변경: 이전 "BLOCKED"

    m43_receipt_sha = None
    if producers["m43-deterministic"]["status"] == "OK":
        m43_receipt_sha = producers["m43-deterministic"]["payload_hashes"].get("m43.json")
    image_digest = None
    if producers["container"]["status"] == "OK":
        image_digest = producers["container"]["payload_hashes"].get("container_smoke.json")

    hosted_release_ready = deterministic_status == "PASS"
    full_production_release_ready = False

    return {
        "schema": BASELINE_SCHEMA_V2, "schema_version": BASELINE_SCHEMA_VERSION_V2,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_sha": args.expected_sha,
        "workflow_run": {
            "run_id": args.expected_run_id, "run_attempt": args.expected_run_attempt,
            "workflow_path": args.expected_workflow_path, "event_name": args.expected_event,
        },
        "m3_fingerprint_reference": "evaluation/baselines/m3_initial.json",
        "dependency_snapshot_sha256": _dependency_snapshot_sha256(),
        "settings_hash": _settings_hash(),
        "image_digest": image_digest,
        "m43_deterministic_receipt_sha256": m43_receipt_sha,
        "producers": producers,
        "gates": gates,
        "deterministic_status": deterministic_status,
        "support_policy": dict(SUPPORT_POLICY_FIXED),
        "operational_status": "NOT_ADOPTED",              # 변경: 이전 "BLOCKED"
        "M4.1_BLOCKED": False,                              # 변경: 이전 True
        "hosted_release_ready": hosted_release_ready,       # 신규
        "native_linux_release_ready": False,                # 신규, 상수
        "full_production_release_ready": full_production_release_ready,  # 신규, 상수
        "overall_release_ready": full_production_release_ready,  # 변경: alias 재정의
    }
```

`assemble(args)`(L330-340)는 **완전히 불변** — 이 함수는 `_build_baseline`을
호출할 뿐 스키마를 모른다. `deterministic_status` 계산(L339,
`"PASS" if all(p["status"] == "OK" for p in producers.values()) else "FAIL"`)도
불변이다.

### 3.3 `main()` — exit code 기준 필드 변경

```python
def main(argv: list[str] | None = None) -> int:
    ...  # argparse 블록 불변
    baseline = assemble(args)
    text = json.dumps(baseline, sort_keys=True, ensure_ascii=False, indent=2)
    output = Path(args.output) if args.output else Path(args.fresh_dir) / "m4-baseline.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if baseline["hosted_release_ready"] else 1   # 변경: 이전 deterministic_status
```

값 자체는 `deterministic_status == "PASS"`와 수학적으로 동일하다
(`hosted_release_ready`의 정의가 그 식이므로). 그럼에도 필드를 바꾸는
이유를 설계 근거로 남긴다: v2에서 "이 baseline이 뭔가에 대해 ready한가"를
묻는 최상위 질문의 정답 필드는 이제 `hosted_release_ready`이고,
`deterministic_status`는 그것을 만드는 내부 값이다. exit code가 참조하는
필드를 상위 개념으로 옮겨 두면, 향후 `deterministic_status`의 정의가
바뀌어도(예: producer 집합이 늘어나는 미래 milestone) `main()`의 exit
code 의미는 "이 CI 실행이 hosted 릴리스 가능 evidence를 만들었는가"로
안정적으로 유지된다.

### 3.4 CLI argv는 불변

`--fresh-dir`, `--expected-sha`, `--expected-run-id`,
`--expected-run-attempt`, `--expected-workflow-path`, `--expected-event`,
`--needs-result`, `--evidence`, `--output` — 신규 플래그 없음. workflow의
`assemble-m4` step 호출 인자(`.github/workflows/ci.yml` 현재 L256-271)는
그대로 유지된다.

## 4. `scripts/check_m4_baseline.py` 변경

### 4.1 상수 재배치 (V1/V2 분리 + 공유)

```python
# 공유 (기존 이름·값 그대로, producers/payload 하위 구조는 스키마와 무관)
REQUIRED_GATE_KEYS = frozenset({
    "python_tests", "frontend_tests", "container", "m43_deterministic",
    "m3_live_regression", "m41_operational",
})
DETERMINISTIC_GATE_KEYS = frozenset({
    "python_tests", "frontend_tests", "container", "m43_deterministic",
})
PRODUCER_TO_GATE_KEY = {
    "python-tests": "python_tests", "frontend-tests": "frontend_tests",
    "container": "container", "m43-deterministic": "m43_deterministic",
}
REQUIRED_PRODUCER_KEYS = frozenset(PRODUCER_TO_GATE_KEY)
PRODUCER_STATUS_ENUM = frozenset({
    "OK", "MISSING", "FAILED_OR_SKIPPED", "DUPLICATE_PRODUCER",
    "IDENTITY_MISMATCH", "PATH_TRAVERSAL", "MALFORMED", "PAYLOAD_INVALID",
})
PRODUCER_STATUS_SCHEMA = { ... }            # 기존 값 그대로 (L51-60)
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")   # 불변
PRODUCER_EXPECTED_PAYLOAD_FILENAMES = { ... }  # 기존 값 그대로 (L62-66)

# v1 전용 (기존 이름에 _V1 접미사)
BASELINE_SCHEMA_V1 = "m4-baseline-v1"
BASELINE_SCHEMA_VERSION_V1 = "1.0.0"
GATE_ENUM_V1 = frozenset({"NOT_RUN", "SKIPPED", "UNKNOWN", "BLOCKED", "PASS", "FAIL"})
REQUIRED_TOP_KEYS_V1 = frozenset({
    "schema", "schema_version", "generated_at", "git_sha", "workflow_run",
    "m3_fingerprint_reference", "dependency_snapshot_sha256", "settings_hash",
    "image_digest", "m43_deterministic_receipt_sha256", "producers", "gates",
    "deterministic_status", "operational_status", "M4.1_BLOCKED",
    "overall_release_ready",
})

# v2 전용
BASELINE_SCHEMA_V2 = "m4-baseline-v2"
BASELINE_SCHEMA_VERSION_V2 = "2.0.0"
GATE_ENUM_V2 = frozenset({"PASS", "FAIL", "NOT_ADOPTED"})
FIXED_NOT_ADOPTED_GATE_KEYS = frozenset({"m3_live_regression", "m41_operational"})
REQUIRED_TOP_KEYS_V2 = REQUIRED_TOP_KEYS_V1 | frozenset({
    "support_policy", "hosted_release_ready", "native_linux_release_ready",
    "full_production_release_ready",
})
WORKFLOW_RUN_KEYS = frozenset({"run_id", "run_attempt", "workflow_path", "event_name"})
SUPPORT_POLICY_SCHEMA = "m4-support-policy-v1"
SUPPORT_POLICY_FIXED = {
    "schema": SUPPORT_POLICY_SCHEMA, "adopted_scope": "HOSTED_OCI",
    "native_linux_ollama": "NOT_ADOPTED", "decision_date": "2026-08-15",
}
SUPPORT_POLICY_KEYS = frozenset(SUPPORT_POLICY_FIXED)
```

`_payload_manifest_sha256`(L69-70)는 불변.

### 4.2 공유 헬퍼 — producer 구조 검증과 gate-algebra 재계산

기존 `check()`(L73-164)의 앞부분(gate enum 검사 이전은 제외, producers
검증 ~ `expected_gates_from_producers` 계산까지, L92-131)을 v1/v2가
공유하는 헬퍼로 뽑는다. **로직은 한 글자도 바꾸지 않고 함수 경계만
새로 긋는다** — 이 부분이 producer→gate 매핑을 계산하는 유일한 지점이고,
그 계산 자체는 스키마 버전과 무관하기 때문이다.

```python
def _validate_producers_and_recompute_gates(
    candidate: dict,
) -> tuple[dict[str, str] | None, list[str]]:
    """기존 check() L92-131과 동일 로직. producers 구조를 검증하고
    네 deterministic gate_key -> "PASS"/"FAIL"의 재계산 값을 반환한다.
    반환 issues가 비어있지 않으면 첫 번째 원소는 dict가 아니다(None)."""
    issues: list[str] = []
    producers = candidate.get("producers")
    if not isinstance(producers, dict) or set(producers) != REQUIRED_PRODUCER_KEYS:
        return None, ["producer_key_set_mismatch"]
    expected_gates_from_producers: dict[str, str] = {}
    for job, gate_key in PRODUCER_TO_GATE_KEY.items():
        entry = producers[job]
        if not isinstance(entry, dict) or "status" not in entry:
            issues.append(f"producer_schema_invalid:{job}")
            continue
        status = entry["status"]
        if status not in PRODUCER_STATUS_ENUM:
            issues.append(f"producer_status_unknown:{job}={status!r}")
            continue
        if set(entry) != PRODUCER_STATUS_SCHEMA[status]:
            issues.append(f"producer_variant_schema_mismatch:{job}:status={status}:keys={sorted(entry)}")
            continue
        if status == "OK":
            receipt_sha = entry["receipt_sha256"]
            if not isinstance(receipt_sha, str) or not _HEX64_RE.fullmatch(receipt_sha):
                issues.append(f"producer_receipt_sha256_malformed:{job}")
                continue
            payload_hashes = entry["payload_hashes"]
            expected_filenames = PRODUCER_EXPECTED_PAYLOAD_FILENAMES[job]
            if not isinstance(payload_hashes, dict) or set(payload_hashes) != expected_filenames:
                issues.append(f"producer_payload_filename_set_mismatch:{job}")
                continue
            if any(not isinstance(k, str) or not isinstance(v, str) or not _HEX64_RE.fullmatch(v)
                   for k, v in payload_hashes.items()):
                issues.append(f"producer_payload_hashes_malformed:{job}")
                continue
            manifest_sha = entry["payload_manifest_sha256"]
            if not isinstance(manifest_sha, str) or not _HEX64_RE.fullmatch(manifest_sha):
                issues.append(f"producer_payload_manifest_sha256_malformed:{job}")
                continue
            if manifest_sha != _payload_manifest_sha256(payload_hashes):
                issues.append(f"producer_payload_manifest_sha256_mismatch:{job}")
                continue
        expected_gates_from_producers[gate_key] = "PASS" if status == "OK" else "FAIL"
    if issues:
        return None, issues
    return expected_gates_from_producers, []
```

### 4.3 `_check_v2` — 신규

```python
def _check_v2(candidate: dict, *, expect_hosted_release_ready: bool,
              expect_hosted_not_ready: bool,
              expect_sha: str | None = None, expect_run_id: str | None = None,
              expect_run_attempt: str | None = None,
              expect_workflow_path: str | None = None,
              expect_event: str | None = None) -> tuple[bool, list[str]]:
    issues: list[str] = []
    top_keys = set(candidate)
    if top_keys != REQUIRED_TOP_KEYS_V2:
        return False, [f"top_level_key_mismatch:missing={sorted(REQUIRED_TOP_KEYS_V2 - top_keys)}"
                       f",extra={sorted(top_keys - REQUIRED_TOP_KEYS_V2)}"]

    # DR-I1-MAJ-02 — identity binding. This never re-fetches or re-hashes
    # original payload bytes (that happens only inside the assembler at CI
    # time, before this artifact is uploaded, per §4.7); it only proves that
    # the candidate's own declared identity is internally well-typed and, if
    # the operator supplied expected values, matches them.
    git_sha = candidate.get("git_sha")
    if not isinstance(git_sha, str) or not git_sha:
        issues.append(f"git_sha_not_nonempty_string:{git_sha!r}")
    workflow_run = candidate.get("workflow_run")
    if not isinstance(workflow_run, dict) or set(workflow_run) != WORKFLOW_RUN_KEYS:
        issues.append("workflow_run_key_set_mismatch")
    else:
        for key in WORKFLOW_RUN_KEYS:
            if not isinstance(workflow_run[key], str) or not workflow_run[key]:
                issues.append(f"workflow_run_field_not_nonempty_string:{key}")
    if issues:
        return False, issues

    if expect_sha is not None and git_sha != expect_sha:
        issues.append(f"identity_sha_mismatch:expected={expect_sha!r},got={git_sha!r}")
    if expect_run_id is not None and workflow_run["run_id"] != expect_run_id:
        issues.append(f"identity_run_id_mismatch:expected={expect_run_id!r},got={workflow_run['run_id']!r}")
    if expect_run_attempt is not None and workflow_run["run_attempt"] != expect_run_attempt:
        issues.append("identity_run_attempt_mismatch:expected="
                      f"{expect_run_attempt!r},got={workflow_run['run_attempt']!r}")
    if expect_workflow_path is not None and workflow_run["workflow_path"] != expect_workflow_path:
        issues.append("identity_workflow_path_mismatch:expected="
                      f"{expect_workflow_path!r},got={workflow_run['workflow_path']!r}")
    if expect_event is not None and workflow_run["event_name"] != expect_event:
        issues.append(f"identity_event_mismatch:expected={expect_event!r},got={workflow_run['event_name']!r}")
    if issues:
        return False, issues

    gates = candidate["gates"]
    if not isinstance(gates, dict) or set(gates) != REQUIRED_GATE_KEYS:
        return False, ["gate_key_set_mismatch"]
    for name, value in gates.items():
        if not isinstance(value, str) or value not in GATE_ENUM_V2:
            issues.append(f"unknown_gate_enum_v2:{name}={value!r}")
    if issues:
        return False, issues

    # step 3 — NOT_ADOPTED 고정값 강제. GATE_ENUM_V2 멤버십만으로는
    # m3_live_regression="PASS"(enum에 존재)가 통과할 수 있으므로,
    # 이 두 키에 한해 별도로 리터럴 일치를 요구한다(§0.3-1의 이중 방어).
    for key in FIXED_NOT_ADOPTED_GATE_KEYS:
        if gates[key] != "NOT_ADOPTED":
            issues.append(f"gate_not_adopted_fixed_value_violation:{key}={gates[key]!r}")
    if issues:
        return False, issues

    expected_gates_from_producers, producer_issues = _validate_producers_and_recompute_gates(candidate)
    if producer_issues:
        return False, producer_issues

    for gate_key, expected in expected_gates_from_producers.items():
        if gates.get(gate_key) != expected:
            issues.append(f"gate_producer_algebra_mismatch:{gate_key}:"
                          f"producers_imply={expected},gates_say={gates.get(gate_key)!r}")
    if issues:
        return False, issues

    # DR-I1-MAJ-02 — recompute the two top-level provenance aliases from the
    # SAME producers dict already validated above, never trust the
    # candidate's own top-level `image_digest`/`m43_deterministic_receipt_sha256`.
    # `_validate_producers_and_recompute_gates` already proved each producer
    # entry's shape matches `PRODUCER_STATUS_SCHEMA[status]` exactly, so an
    # "OK" entry is guaranteed to have `payload_hashes` as a dict here.
    container_entry = candidate["producers"]["container"]
    expected_image_digest = (
        container_entry["payload_hashes"].get("container_smoke.json")
        if container_entry["status"] == "OK" else None
    )
    if candidate.get("image_digest") != expected_image_digest:
        issues.append("image_digest_alias_mismatch:expected="
                      f"{expected_image_digest!r},got={candidate.get('image_digest')!r}")

    m43_entry = candidate["producers"]["m43-deterministic"]
    expected_m43_receipt_sha = (
        m43_entry["payload_hashes"].get("m43.json")
        if m43_entry["status"] == "OK" else None
    )
    if candidate.get("m43_deterministic_receipt_sha256") != expected_m43_receipt_sha:
        issues.append("m43_deterministic_receipt_sha256_alias_mismatch:expected="
                      f"{expected_m43_receipt_sha!r},"
                      f"got={candidate.get('m43_deterministic_receipt_sha256')!r}")
    if issues:
        return False, issues

    expected_deterministic = "PASS" if all(
        expected_gates_from_producers[k] == "PASS" for k in DETERMINISTIC_GATE_KEYS
    ) else "FAIL"
    if candidate.get("deterministic_status") != expected_deterministic:
        issues.append("deterministic_status_algebra_mismatch")

    expected_hosted_ready = expected_deterministic == "PASS"
    hosted_ready = candidate.get("hosted_release_ready")
    if not isinstance(hosted_ready, bool) or hosted_ready != expected_hosted_ready:
        issues.append("hosted_release_ready_algebra_mismatch")

    native_ready = candidate.get("native_linux_release_ready")
    if native_ready is not False:
        issues.append(f"native_linux_release_ready_not_false:{native_ready!r}")

    full_ready = candidate.get("full_production_release_ready")
    if full_ready is not False:
        issues.append(f"full_production_release_ready_not_false:{full_ready!r}")

    overall_ready = candidate.get("overall_release_ready")
    if overall_ready != full_ready:
        issues.append("overall_release_ready_alias_mismatch")

    if candidate.get("operational_status") != "NOT_ADOPTED":
        issues.append(f"operational_status_not_not_adopted:{candidate.get('operational_status')!r}")

    blocked = candidate.get("M4.1_BLOCKED")
    if blocked is not False:
        issues.append(f"m41_blocked_not_false:{blocked!r}")

    support_policy = candidate.get("support_policy")
    if not isinstance(support_policy, dict) or set(support_policy) != SUPPORT_POLICY_KEYS:
        issues.append("support_policy_key_set_mismatch")
    elif support_policy != SUPPORT_POLICY_FIXED:
        for field, expected_value in SUPPORT_POLICY_FIXED.items():
            if support_policy.get(field) != expected_value:
                issues.append(f"support_policy_field_mismatch:{field}="
                              f"{support_policy.get(field)!r},expected={expected_value!r}")
    if issues:
        return False, issues

    if expect_hosted_release_ready and hosted_ready is not True:
        issues.append("expected_hosted_release_ready_not_satisfied")
    if expect_hosted_not_ready and hosted_ready is not False:
        issues.append("expected_hosted_not_ready_not_satisfied")
    return (not issues, issues)
```

### 4.4 `_check_v1_legacy` — 기존 `check()`를 이름만 바꿔 그대로 이동

기존 `check()`(L73-164) 본문을 아무 로직 변경 없이 `_check_v1_legacy`로
옮기고, `_validate_producers_and_recompute_gates` 공유 헬퍼를 호출하도록만
바꾼다(호출 결과는 기존 인라인 코드와 완전히 동일한 값을 만든다).
`REQUIRED_TOP_KEYS` → `REQUIRED_TOP_KEYS_V1`, `GATE_ENUM` → `GATE_ENUM_V1`로
참조만 바꾼다. `operational_status`/`overall_release_ready` 대수, `M4.1_BLOCKED`
비-bool 검사, `expect_operational_blocked` 처리(L146-163)는 **전부 원문
그대로**다 — 이것이 "v1 compatibility mode는 원래 대수를 유지한다"
(REQ-003.2)는 요구사항을 코드로 증명하는 방법이다.

**DR-I1-MAJ-01 폐쇄 — frozen-blocked 상태는 `--allow-legacy-v1`만으로
무조건 강제된다.** 원문 그대로 옮긴 대수 검사는 "v1 후보 자신의 gate 값이
서로 내적으로 정합적인가"만 증명한다 — `GATE_ENUM_V1`에는 `"PASS"`가
포함되므로, `m3_live_regression`/`m41_operational` 두 gate에 `"PASS"`를
넣고 `operational_status="PASS"`/`overall_release_ready=True`로 맞추면
그 내적 정합성 검사 자체는 통과한다. 그러나 Requirement
M4-OAR-REQ-003.2가 정의하는 legacy 호환 의미는 "v1 대수가 내적으로
정합적이면 무엇이든 허용"이 아니라 **정확히 하나의 고정 상태**(live
`NOT_RUN`, M4.1 `BLOCKED`, `M4.1_BLOCKED=true`, `overall_release_ready=false`)
다. 따라서 gate enum 검사 직후, producer 검증보다 먼저, 이 다섯 값을
**무조건**(플래그와 무관하게) 강제하는 블록을 추가한다:

```python
def _check_v1_legacy(candidate: dict, *, expect_operational_blocked: bool = False) -> tuple[bool, list[str]]:
    issues: list[str] = []
    top_keys = set(candidate)
    if top_keys != REQUIRED_TOP_KEYS_V1:
        return False, [f"top_level_key_mismatch:missing={sorted(REQUIRED_TOP_KEYS_V1 - top_keys)}"
                       f",extra={sorted(top_keys - REQUIRED_TOP_KEYS_V1)}"]

    gates = candidate["gates"]
    if not isinstance(gates, dict) or set(gates) != REQUIRED_GATE_KEYS:
        return False, ["gate_key_set_mismatch"]
    for name, value in gates.items():
        if not isinstance(value, str) or value not in GATE_ENUM_V1:
            issues.append(f"unknown_gate_enum:{name}={value!r}")
    if issues:
        return False, issues

    # Unconditional frozen-blocked legacy contract (REQ-003.2). NOT gated
    # behind expect_operational_blocked — --allow-legacy-v1 alone must
    # enforce this, because the historical artifact meaning IS this exact
    # fixed state, not "whatever v1's internally self-consistent algebra
    # happens to compute." Runs before producer validation so a candidate
    # cannot use a fabricated producers dict to distract from this check.
    if gates["m3_live_regression"] != "NOT_RUN":
        issues.append(f"v1_legacy_live_regression_not_not_run:{gates['m3_live_regression']!r}")
    if gates["m41_operational"] != "BLOCKED":
        issues.append(f"v1_legacy_m41_operational_not_blocked:{gates['m41_operational']!r}")
    if candidate.get("operational_status") != "BLOCKED":
        issues.append(f"v1_legacy_operational_status_not_blocked:{candidate.get('operational_status')!r}")
    if candidate.get("M4.1_BLOCKED") is not True:
        issues.append(f"v1_legacy_m41_blocked_not_true:{candidate.get('M4.1_BLOCKED')!r}")
    if candidate.get("overall_release_ready") is not False:
        issues.append(f"v1_legacy_overall_release_ready_not_false:{candidate.get('overall_release_ready')!r}")
    if issues:
        return False, issues

    expected_gates_from_producers, producer_issues = _validate_producers_and_recompute_gates(candidate)
    if producer_issues:
        return False, producer_issues

    for gate_key, expected in expected_gates_from_producers.items():
        if gates.get(gate_key) != expected:
            issues.append(f"gate_producer_algebra_mismatch:{gate_key}:"
                          f"producers_imply={expected},gates_say={gates.get(gate_key)!r}")
    if issues:
        return False, issues

    expected_deterministic = "PASS" if all(
        expected_gates_from_producers[k] == "PASS" for k in DETERMINISTIC_GATE_KEYS
    ) else "FAIL"
    if candidate.get("deterministic_status") != expected_deterministic:
        issues.append("deterministic_status_algebra_mismatch")

    expected_operational = "PASS" if (gates["m41_operational"] == "PASS"
                                       and gates["m3_live_regression"] == "PASS") else "BLOCKED"
    if candidate.get("operational_status") != expected_operational:
        issues.append("operational_status_algebra_mismatch")

    expected_ready = (expected_deterministic == "PASS" and expected_operational == "PASS")
    if candidate.get("overall_release_ready") != expected_ready:
        issues.append("overall_release_ready_algebra_mismatch")

    for bool_key in ("M4.1_BLOCKED", "overall_release_ready"):
        if not isinstance(candidate.get(bool_key), bool):
            issues.append(f"non_boolean_field:{bool_key}={candidate.get(bool_key)!r}")

    # `expect_operational_blocked` is now a REDUNDANT compatibility CLI
    # assertion, not the switch that activates frozen-blocked semantics —
    # the unconditional block above already enforces the same three fields
    # (plus the two gate values) regardless of this flag. It is kept only so
    # existing `--expect-operational-blocked` call sites keep parsing and
    # keep asserting the same claim explicitly; removing the flag would not
    # weaken `--allow-legacy-v1`'s guarantees.
    if expect_operational_blocked:
        if candidate.get("operational_status") != "BLOCKED" or \
                candidate.get("M4.1_BLOCKED") is not True or \
                candidate.get("overall_release_ready") is not False:
            issues.append("expected_operational_blocked_not_satisfied")
    return (not issues, issues)
```

### 4.5 `check()` 디스패처 — schema/version 태그로만 분기

```python
def check(candidate: dict, *, allow_legacy_v1: bool = False,
          expect_hosted_release_ready: bool = False,
          expect_hosted_not_ready: bool = False,
          expect_operational_blocked: bool = False,
          expect_sha: str | None = None, expect_run_id: str | None = None,
          expect_run_attempt: str | None = None,
          expect_workflow_path: str | None = None,
          expect_event: str | None = None) -> tuple[bool, list[str]]:
    if not isinstance(candidate, dict):
        return False, ["candidate_not_object"]
    schema = candidate.get("schema")
    version = candidate.get("schema_version")
    if schema == BASELINE_SCHEMA_V2 and version == BASELINE_SCHEMA_VERSION_V2:
        return _check_v2(candidate, expect_hosted_release_ready=expect_hosted_release_ready,
                          expect_hosted_not_ready=expect_hosted_not_ready,
                          expect_sha=expect_sha, expect_run_id=expect_run_id,
                          expect_run_attempt=expect_run_attempt,
                          expect_workflow_path=expect_workflow_path, expect_event=expect_event)
    if schema == BASELINE_SCHEMA_V1 and version == BASELINE_SCHEMA_VERSION_V1:
        if not allow_legacy_v1:
            return False, ["legacy_v1_schema_requires_allow_legacy_v1_flag"]
        return _check_v1_legacy(candidate, expect_operational_blocked=expect_operational_blocked)
    return False, ["unknown_or_unsupported_schema"]
```

`schema`/`schema_version`은 **쌍으로만** 유효하다 — `schema="m4-baseline-v2"`
이면서 `schema_version="1.0.0"`처럼 짝이 어긋나면 두 `if` 모두 거짓이 되어
`unknown_or_unsupported_schema`로 떨어진다(§2.1이 말한 "mixed schema"
방어가 여기서 실제로 일어나는 지점).

### 4.6 CLI — `main()`과 플래그 조합 규칙

```python
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--allow-legacy-v1", action="store_true",
                         help="Accept schema=m4-baseline-v1 and check it under "
                              "its original (pre-M4-OAR) fail-closed algebra.")
    parser.add_argument("--expect-hosted-release-ready", action="store_true")
    parser.add_argument("--expect-hosted-not-ready", action="store_true")
    parser.add_argument("--expect-operational-blocked", action="store_true",
                         help="Legacy v1-only redundant compatibility assertion; requires "
                              "--allow-legacy-v1. Does not activate frozen-blocked semantics — "
                              "--allow-legacy-v1 alone already enforces them (DR-I1-MAJ-01).")
    parser.add_argument("--expect-sha", default=None,
                         help="v2-only identity binding: candidate.git_sha must equal this.")
    parser.add_argument("--expect-run-id", default=None,
                         help="v2-only identity binding: candidate.workflow_run.run_id must equal this.")
    parser.add_argument("--expect-run-attempt", default=None,
                         help="v2-only identity binding: candidate.workflow_run.run_attempt must equal this.")
    parser.add_argument("--expect-workflow-path", default=None,
                         help="v2-only identity binding: candidate.workflow_run.workflow_path must equal this.")
    parser.add_argument("--expect-event", default=None,
                         help="v2-only identity binding: candidate.workflow_run.event_name must equal this.")
    parser.add_argument("--require-identity-binding", action="store_true",
                         help="Post-merge mode (DR-I1-MAJ-02): makes all five --expect-sha/"
                              "--expect-run-id/--expect-run-attempt/--expect-workflow-path/"
                              "--expect-event flags mandatory alongside --expect-hosted-release-ready. "
                              "The pre-merge fixture check (Plan.md §5, no real run to bind to yet) "
                              "does NOT set this flag; the post-merge runbook procedure (§8.3 §6.1) "
                              "always does.")
    args = parser.parse_args(argv)

    _IDENTITY_FLAG_NAMES = ("expect_sha", "expect_run_id", "expect_run_attempt",
                            "expect_workflow_path", "expect_event")

    if args.expect_operational_blocked and not args.allow_legacy_v1:
        parser.error("--expect-operational-blocked requires --allow-legacy-v1")
    if args.allow_legacy_v1 and (args.expect_hosted_release_ready or args.expect_hosted_not_ready
                                  or args.require_identity_binding
                                  or any(getattr(args, f) is not None for f in _IDENTITY_FLAG_NAMES)):
        parser.error("--expect-hosted-release-ready/--expect-hosted-not-ready/--require-identity-binding/"
                      "--expect-sha/--expect-run-id/--expect-run-attempt/--expect-workflow-path/"
                      "--expect-event are incompatible with --allow-legacy-v1 (v1 has no "
                      "hosted_release_ready or checker-verified identity fields)")
    if args.expect_hosted_release_ready and args.expect_hosted_not_ready:
        parser.error("--expect-hosted-release-ready and --expect-hosted-not-ready are mutually exclusive")
    # DR-I1-MAJ-02 — the post-merge hosted-ready assertion MUST be
    # identity-bound; a bare --expect-hosted-release-ready with no identity
    # flags would let an operator point the checker at a baseline copied
    # from a different run/SHA and still get a clean PASS. This requirement
    # is opt-in via --require-identity-binding (rather than implied by
    # --expect-hosted-release-ready alone) so the pre-merge fixture command
    # in Plan.md §5 — which has no real workflow run to bind to yet — keeps
    # working unmodified.
    if args.require_identity_binding:
        if not args.expect_hosted_release_ready:
            parser.error("--require-identity-binding requires --expect-hosted-release-ready")
        missing = [f"--{name.replace('_', '-')}" for name in _IDENTITY_FLAG_NAMES
                   if getattr(args, name) is None]
        if missing:
            parser.error("--require-identity-binding requires all five identity flags; missing: "
                          + ", ".join(missing))

    try:
        candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"ok": False, "issues": [f"candidate_unreadable:{type(exc).__name__}"]}),
              file=sys.stderr)
        return 1

    ok, issues = check(candidate, allow_legacy_v1=args.allow_legacy_v1,
                        expect_hosted_release_ready=args.expect_hosted_release_ready,
                        expect_hosted_not_ready=args.expect_hosted_not_ready,
                        expect_operational_blocked=args.expect_operational_blocked,
                        expect_sha=args.expect_sha, expect_run_id=args.expect_run_id,
                        expect_run_attempt=args.expect_run_attempt,
                        expect_workflow_path=args.expect_workflow_path, expect_event=args.expect_event)
    if not ok:
        print(json.dumps({"ok": False, "issues": issues}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "issues": []}))
    return 0
```

이 다섯 `parser.error(...)` 호출 지점 — (1) `--expect-operational-blocked`는
`--allow-legacy-v1` 필요, (2) `--allow-legacy-v1`은 hosted/신원-바인딩
관련 플래그 전부와 상호배제, (3) `--expect-hosted-release-ready`/
`--expect-hosted-not-ready` 상호배제, (4) `--require-identity-binding`는
`--expect-hosted-release-ready` 필요, (5) `--require-identity-binding`는
다섯 신원 플래그 전부 필요(누락분을 한 메시지에 모아 보고) — 는
argparse 관례상 stderr에 usage를 출력하고 **exit code 2**로 즉시
종료한다(파이썬 표준 `argparse` 동작, 이 저장소의
`scripts/ci_acceptance_contract.py`가 이미 쓰는 동일 관례 —
`CI_Acceptance_Runbook.md` §2.2의 "exit=2" 예시와 대칭을 이룬다).

#### exit code 표

| 상황 | exit code |
|---|---|
| `--candidate` 파일 없음/JSON 파싱 실패 | 1 |
| flag 조합이 위 다섯 규칙 중 하나를 위반 | 2 |
| candidate schema/version이 v1도 v2도 아님, 또는 v1인데 `--allow-legacy-v1` 없음 | 1 |
| v2 candidate가 §4.3의 어떤 검사라도 실패 | 1 |
| v1 candidate가(`--allow-legacy-v1`와 함께) §4.4 검사 실패 | 1 |
| 모든 검사 통과 (+ 지정된 `--expect-*`가 있다면 그것도 만족) | 0 |

### 4.7 왜 CI의 `m4-assemble` step은 `--expect-hosted-*`를 지정하지 않는가

`hosted_release_ready`는 v1의 `operational_status=BLOCKED`처럼 매 실행마다
같은 값으로 고정된 정책 상수가 **아니다** — 그 run에서 네 producer가 실제로
통과했는지에 따라 true/false가 달라지는 evidence-derived 값이다. CI의
`m4-assemble` job은 `if: always()`로 예정대로 producer 중 하나가 실패한
run에서도 실행되어 baseline을 만든다(현재 워크플로 L216 그대로 유지). 이
step이 만약 `--expect-hosted-release-ready`를 무조건 요구한다면, 의도적으로
실패하는 PR(리뷰 중인 버그 재현, 실패 테스트가 있는 draft 등)마다
`m4-assemble` job 자체가 추가로 빨갛게 되어 "이미 실패한 producer 위에
얹힌 잡음"이 된다 — 반대로 `--expect-hosted-not-ready`를 무조건 요구하면
모든 것이 정상 통과하는 날마다 이 step이 실패한다. 따라서 **`m4-assemble`
step은 스키마/대수 자기-정합성만 검사**하고(§5.3), "이번 run이 실제로
hosted-ready였는가"의 **단언은 pre-merge fixture 테스트(Plan §5)와
post-merge 정확-SHA 검증(Plan §6, §10)로 옮긴다** — 이 두 자리에서는 기대값을
미리 알 수 있기 때문이다(fixture는 저자가 구성, post-merge는 다른 job들의
실제 conclusion을 먼저 확인한 뒤 검사).

같은 이유로 `m4-assemble` step은 §4.6의 다섯 신원-바인딩 `--expect-*`
플래그도 지정하지 않는다 — 그 run 자신의 `github.sha`/`github.run_id`를
그 run 자신이 만든 baseline과 비교하는 것은 항상 참인 동어반복이기
때문이다. 신원 바인딩이 실제로 의미를 갖는 자리는 **다른 시점, 다른
프로세스(운영자의 워크스테이션)가 "이 아티팩트가 내가 요청한 정확히 그
run에서 나왔는가"를 독립적으로 확인하는 post-merge 시점**뿐이며, 그
호출은 §8.3 "6.1 Hosted/OCI baseline verification"에 정의한다.

**트러스트 경계를 정확히 기술한다(DR-I1-MAJ-02 폐쇄):** checker의
identity/alias 검사(§4.3)는 candidate JSON 문서 **내부**의 신원 필드가
잘 정형화돼 있고, 지정된 경우 operator의 기대값과 일치하며, 두 top-level
alias가 같은 문서 안의 `producers[...].payload_hashes`와 산술적으로
일치한다는 것만 증명한다. 이 검사는 원본 payload 바이트(container 이미지
레이어, `m43.json` 내용 등)를 다시 내려받거나 재해싱하지 않는다 — 그
바이트-대-해시 검증은 오직 assembler가 CI job 안에서, 이 아티팩트가
업로드되기 전에 `_verify_payloads`/`_check_identity`로 이미 수행했다
(§0.1 "바꾸지 않는 것"). 따라서 이 checker를 "post-merge에서 payload
바이트까지 독립적으로 재검증하는 도구"라고 부르지 않는다 — 정확한 설명은
"candidate 문서의 신원·alias·대수 자기-정합성을, operator가 지정한 기대
신원과 함께 독립적으로 재계산해 검증하는 도구"다. 두 신뢰 경계(assembler의
바이트 검증 vs. checker의 신원/alias/대수 검증)는 서로 다른 시점에 서로
다른 것을 증명하며, 이 문서 어디에서도 checker가 payload 바이트를
재검증한다고 주장하지 않는다.

## 5. Workflow 계약 변경 — `.github/workflows/ci.yml`

### 5.1 현재 문제의 정확한 위치

현재 L316-319:

```yaml
  m3-live-regression-gate:
    if: |
      github.event_name == 'workflow_dispatch' ||
      (github.event_name == 'push' && github.ref == 'refs/heads/master')
    runs-on: [self-hosted, ollama-m3]
    environment: m3-live-regression
```

`master`로의 **ordinary push마다** 이 job이 스케줄된다. self-hosted runner
수가 0이면(Stop_Report.md §1이 기록한 실제 상태) 이 job은 영원히
`queued`로 남고, run 전체가 terminal conclusion에 도달하지 못한다 — 이것이
M4-OAR-REQ-004.1이 금지하는 정확한 실패 모드다.

### 5.2 `on.workflow_dispatch.inputs` 추가

```yaml
on:
  pull_request:
  push:
    branches: [master]
  workflow_dispatch:
    inputs:
      enable_m3_live_regression:
        description: >-
          NOT_ADOPTED under the current hosted/OCI policy (M4-OAR-REQ-004).
          This input exists only as a documented reactivation marker for a
          future, separately reviewed milestone. Setting it true today still
          resolves as a no-op informational job — it does not run live code,
          checkout the repository, or contact Ollama.
        type: boolean
        default: false
```

### 5.3 `m3-live-regression-gate` job — 전체 재정의

```yaml
  # M4-OAR-REQ-004.2 — NOT_ADOPTED reactivation stub (Requirement.md,
  # Stop_Report.md §3/§5). 이 job은 checkout하지 않고, secret/environment
  # 승인을 요구하지 않고, self-hosted 라벨을 참조하지 않는다. ordinary
  # push/pull_request에서는 이 job의 `if:`가 항상 거짓이라 애초에
  # 스케줄되지 않는다(즉시 skipped 상태로 run이 종료된다 — queued로 남지
  # 않는다). workflow_dispatch에서도 `enable_m3_live_regression` 입력이
  # 명시적으로 true여야만 실행되며, 실행되더라도 아래 단일 step은 정책
  # 문서를 안내하고 exit 0으로 즉시 끝난다. 재활성화는 새 정책 결정,
  # 요구사항/설계 리뷰, threat model, 소유된 native runner, 별도 구현을
  # 요구한다(Stop_Report.md §4) — 이 workflow_dispatch input을 켜는 것만
  # 으로는 아무것도 재활성화되지 않는다.
  m3-live-regression-gate:
    if: github.event_name == 'workflow_dispatch' && inputs.enable_m3_live_regression == true
    runs-on: ubuntu-latest
    timeout-minutes: 1
    steps:
      - name: NOT_ADOPTED — informational reactivation stub, no live execution
        run: |
          echo "::notice::m3-live-regression-gate is NOT_ADOPTED under the current hosted/OCI release policy."
          echo "This run performed no checkout, no secrets, no environment approval, and no self-hosted runner."
          echo "See docs/milestones/m4-operational-acceptance-recovery/Requirement.md and Stop_Report.md for the reactivation path."
          exit 0
```

설계 근거:

1. **`if:` 조건에 `push`/`pull_request` 분기가 없다** — `workflow_dispatch`
   이벤트가 아니면 이 job은 항상 skipped다. GitHub Actions는 `inputs`
   컨텍스트를 모든 이벤트에서 정의하지만 `workflow_dispatch`가 아닌
   이벤트에서는 빈 값이므로 `inputs.enable_m3_live_regression == true`가
   자연히 거짓이 되어 이중으로 안전하다(단축 평가가 아니라 두 조건 모두
   독립적으로 이 job을 막는다).
2. **`runs-on: ubuntu-latest`** — self-hosted 라벨이 이 job의 `runs-on`
   필드 어디에도 남지 않는다. **DR-I4-MIN-01 정정:** 이 사실의 감사
   명령을 워크플로 파일 전체에 대한 broad `grep -c "self-hosted"
   .github/workflows/ci.yml == 0`으로 서술하지 않는다 — 그 명령은 §5.3
   canonical stub 자신의 안전-설명 echo 문구("no self-hosted runner")와
   이 설계 문서(§5.1, §11.1)가 "고쳐야 할 이전 상태"를 인용하는
   `runs-on: [self-hosted, ollama-m3]` 같은 정당한 부정문/인용문에도
   매치하므로, 그 어떤 올바른 구현으로도 통과할 수 없는 impossible한
   audit 명령이다. 이 필드에 대한 실제 감사는 §7.3의 파싱된 구조 검사
   (`test_m3_live_regression_gate_has_no_self_hosted_or_environment`가
   `job["runs-on"] == "ubuntu-latest"`를 정확한 문자열로, 배열이 아니라
   스칼라로 검사)와, 그 필드 자체가 아닌 나머지 raw YAML 표면에 라벨이
   몰래 재도입되는 것을 막는 실행-표면 정밀 정규식
   (`test_m3_live_regression_gate_source_denylist_has_no_forbidden_executable_surfaces`,
   `self_hosted_runner_label` 패턴)이 맡는다 — 둘 다 이미 §7.3에 정의된
   테스트이며 이 항목은 새 감사를 요구하지 않는다.
3. **`environment:` 필드 삭제** — GitHub Environment 승인을 요구하지
   않는다.
4. **`actions/checkout` step 없음** — 저장소 코드를 전혀 체크아웃하지
   않으므로 untrusted 코드 실행 경로 자체가 없다.
5. **`timeout-minutes: 1`** — 정보성 echo만 하는 job이 우발적으로 오래
   걸릴 이유가 없음을 명시적으로 못박는다.

### 5.4 `m4-assemble` — checker 호출 인자 변경

현재 L273-274:

```yaml
      - name: Check M4 baseline state algebra
        run: python scripts/check_m4_baseline.py --candidate assemble/m4-baseline.json --expect-operational-blocked
```

변경 후:

```yaml
      - name: Check M4 baseline state algebra
        run: python scripts/check_m4_baseline.py --candidate assemble/m4-baseline.json
```

`--allow-legacy-v1`도 `--expect-hosted-*`도 지정하지 않는다 — assembler가
쓰는 것은 항상 v2이므로 checker는 기본 경로(v2, §4.3)로 들어가고,
"기대값 없이 스키마/대수 자기-정합성만 검사"하는 §4.7의 설계를 그대로
따른다. `needs: [python-tests, frontend-tests, container,
m43-deterministic]`와 `if: always()`(L215-216)는 이미 요구사항을 만족하므로
불변이다.

### 5.5 Branch protection 권고 (파일 변경 아님, 운영 절차)

Requirement §M4-OAR-REQ-004.3 "SHOULD require only deterministic hosted
checks"에 따라, 저장소 소유자가 GitHub UI에서 "Require status checks to
pass"에 등록해야 할 정확한 check 이름 목록:

```text
python-tests
frontend-tests
container
m43-deterministic
m4-assemble
```

`m3-live-regression-gate`는 이 목록에 포함하지 않는다 — ordinary push에서
skipped 상태로 끝나는 job을 필수 체크로 등록하면 GitHub은 그 커밋을
영원히 "pending"으로 표시할 수 있다(§5.1이 고치려는 것과 같은 부류의
실패 모드). 이 항목은 리포지토리 설정이라 이번 milestone의 diff에
포함되지 않으며, Traceability.md에 완료 여부를 기록하는 체크리스트
항목으로만 남긴다.

## 6. 하위 호환성 — 기존 v1 아티팩트

기존에 실행된 CI run이 만든 `m4-baseline.json`(schema
`m4-baseline-v1`)은 저장소에 커밋되지 않는 GitHub Actions artifact이므로
"immutable 파일을 재작성"할 위험은 없다 — Requirement §M4-OAR-REQ-003.3의
"Historical artifacts remain immutable and interpretable"은 그 artifact가
**앞으로도 `--allow-legacy-v1`로 예전과 동일한 판정을 받을 수 있다**는
의미로 충족한다. 검증 방법:

```bash
python scripts/check_m4_baseline.py --candidate <과거 v1 baseline.json> \
  --allow-legacy-v1
```

이 호출은 §4.4 `_check_v1_legacy`로 들어간다. 실제 과거 M4.1 v1
아티팩트는 (Traceability.md와 보존된 M4.3 evidence가 일관되게 기록하듯)
항상 live `NOT_RUN`/M4.1 `BLOCKED`/`M4.1_BLOCKED=true`/
`overall_release_ready=false`였으므로, `--allow-legacy-v1` **하나만**으로도
§4.4의 무조건 frozen-blocked 검사(DR-I1-MAJ-01)를 그대로 통과한다 —
`--expect-operational-blocked`를 추가로 줄 필요가 없다(줘도 같은 결과다,
§4.4의 잉여 assertion 설명 참고). `--allow-legacy-v1` 없이 같은 파일을
검사하면(`check()`가 §4.5의 `unknown_or_unsupported_schema`가 아니라
정확히 `legacy_v1_schema_requires_allow_legacy_v1_flag`로) exit 1로
거부되는 것도 의도된 동작이다 — "v2 checker의 기본 경로는 v1을 모른다".
반대로, 진짜 과거 아티팩트가 아니라 `m3_live_regression="PASS"`처럼
frozen 값이 조작된 v1-형 candidate는 `--allow-legacy-v1`을 줘도
`v1_legacy_live_regression_not_not_run`으로 거부된다 — §7.2의
`test_v1_legacy_rejects_*` mutant 목록이 이 경계를 정확히 감사한다.

## 7. 테스트/뮤턴트 — 정확한 목록

### 7.1 `tests/unit/test_assemble_m4_evidence.py`

기존 6개(`test_positive_all_producers_ok`,
`test_expected_node_ids_matches_producer_profile_node_ids`,
`test_check_identity_rejects_non_string_semantic_status_without_crashing`,
`test_check_identity_rejects_invalid_string_semantic_status_enum`,
`test_negative_control_matrix`)는 **수정하지 않는다** — `_evaluate_producer`
이하가 불변이므로 이 테스트들은 그대로 v2 구현에서도 통과해야 한다(회귀
가드 역할).

신규 추가(모두 `assembler.assemble(SimpleNamespace(...))`을 `_build_positive_fresh_dir`
기반 args로 호출하거나, `assembler._build_baseline(producers, status, args)`를
직접 호출해 반환 dict를 검사):

**Whole-file allowed-delta 오라클 뮤턴트 생성기 (DR-I4-MAJ-01, DR-I4-MAJ-02).**
아래 헬퍼는 테스트 파일에만 존재한다(어셈블러 코드가 아님). 개별 심볼을
매개변수로 받던 이전 4개 헬퍼(§13~§15가 역사로 보존)를 폐기하고, "base
소스 끝(또는 임의 지점)에 새 top-level statement 하나를 추가"와 "base
소스의 어떤 slice든 안쪽에 공백 한 칸을 삽입"이라는 두 개의 범용 헬퍼로
교체한다 — 새 오라클이 statement의 종류나 이름을 구분하지 않으므로,
뮤턴트 생성기도 더 이상 "assignment-type/function-type 심볼"을 나눌
필요가 없다.

```python
def _append_top_level_statement(base_source: str, statement: str) -> str:
    """base_source 끝에 새 top-level statement 하나를 추가한 완전한 소스
    문자열을 반환한다 — "미승인 추가 statement" 계열 뮤턴트(신규 import/
    class/함수/재바인딩/named-expression 전부)에 공용으로 쓰는 헬퍼."""
    return base_source.rstrip("\n") + "\n\n" + statement.rstrip("\n") + "\n"

def _in_place_whitespace_mutation(base_source: str, target_slice: str) -> str:
    """`target_slice`(base_source 안에서 문자 그대로 정확히 한 번 나타나야
    하는, 임의의 top-level statement 슬라이스)의 첫 `=` 또는 첫 `(` 바로
    뒤에 공백 한 칸을 삽입한 완전한 소스 문자열을 반환한다. 삽입 지점이
    그 statement의 원래 [start, end) 슬라이스 안이므로, 재파싱된 노드의
    `ast.get_source_segment`는 반드시 base 슬라이스와 달라진다(DR-I3-MAJ-01이
    고친 "슬라이스 뒤에 공백을 붙이는" 실수를 반복하지 않는다). 이 헬퍼는
    26개의 이름 붙은 "보호 심볼"이 아니라 base의 **어떤** top-level
    statement에도 적용할 수 있다 — 오라클이 이름을 구분하지 않기 때문에
    뮤턴트 생성기도 구분할 필요가 없다."""
    assert base_source.count(target_slice) == 1, "fixture bug: slice not unique in base"
    anchor = "(" if "(" in target_slice.split("=")[0] else "="
    idx = target_slice.index(anchor)
    mutated_slice = target_slice[: idx + 1] + " " + target_slice[idx + 1 :]
    assert mutated_slice != target_slice
    return base_source.replace(target_slice, mutated_slice, 1)


def _with_decorators(source: str, target_slice: str, decorator_lines: tuple[str, ...]) -> str:
    """`target_slice`(source 안에서 문자 그대로 정확히 한 번 나타나야 하는,
    임의의 top-level class/함수/async 함수 슬라이스) 바로 앞에
    `decorator_lines`를 주어진 순서 그대로 한 줄씩 삽입한 완전한 소스
    문자열을 반환한다 — 줄 순서가 곧 decorator 적용 순서다.
    `decorator_lines`가 빈 tuple이면 `target_slice`를 그대로 반환한다(즉
    decorator가 없는 원상태). 이 하나의 헬퍼로 decorator 추가(빈 tuple →
    1개 이상), 제거(1개 이상 → 빈 tuple), 수정(같은 개수, 다른 표현식),
    재정렬(같은 집합, 다른 순서) 뮤턴트를 base/current 양쪽에 대칭적으로
    구성한다(DR-RC1-I1-MAJ-01)."""
    assert source.count(target_slice) == 1, "fixture bug: slice not unique in source"
    prefix = "".join(line.rstrip("\n") + "\n" for line in decorator_lines)
    return source.replace(target_slice, prefix + target_slice, 1)
```

| 테스트 이름 | 잡는 뮤턴트 |
|---|---|
| `test_assemble_v2_schema_and_version_constants` | `schema`가 실수로 `"m4-baseline-v1"`로 되돌아가거나 `schema_version`이 `"1.0.0"`으로 남는 회귀 |
| `test_assemble_v2_support_policy_exact_fixed_object` | `support_policy`의 네 필드 중 하나라도 하드코딩 오타(`adopted_scope="HOSTED"` 등)가 나는 회귀 |
| `test_assemble_v2_gates_m3_live_regression_and_m41_operational_are_not_adopted` | 두 gate 값이 `"NOT_RUN"`/`"BLOCKED"`로 되돌아가는 v1 잔존 회귀 |
| `test_assemble_v2_m41_blocked_is_false` | `"M4.1_BLOCKED"`가 `True`로 남는 회귀(가장 치명적인 종류 — REQ-001.5) |
| `test_assemble_v2_operational_status_is_not_adopted` | `operational_status`가 `"BLOCKED"`로 남는 회귀 |
| `test_assemble_v2_hosted_release_ready_true_when_all_four_producers_ok` | `_build_positive_fresh_dir` 결과로 `hosted_release_ready is True` 확인(양성 케이스) |
| `test_assemble_v2_hosted_release_ready_false_when_any_producer_not_ok`(4-way parametrize: python-tests/frontend-tests/container/m43-deterministic 각각을 `needs_result="failure"`로 바꿔 재실행) | 네 producer 중 하나만 실패해도 `hosted_release_ready is False`인지 개별 확인(부분 실패가 상쇄되어 true가 되는 회귀 차단) |
| `test_assemble_v2_native_full_overall_always_false_regardless_of_producer_outcome`(all-pass/all-fail 두 fixture 모두) | `native_linux_release_ready`/`full_production_release_ready`/`overall_release_ready`가 hosted 값과 무관하게 항상 `False`인지 — hosted가 True인 날 이 셋도 실수로 True가 새는 것이 가장 위험한 회귀 |
| `test_assemble_v2_overall_release_ready_equals_full_production_release_ready_alias` | alias 정의(`overall == full_production`)가 하드코딩 상수 복붙으로 깨지지 않는지(두 필드를 각각 다른 리터럴로 바꿔보는 회귀 방지) |
| `test_assemble_v2_producers_and_m43_receipt_sha_shape_unchanged` | `producers` 서브 dict와 `m43_deterministic_receipt_sha256`/`image_digest` 추출 로직이 v1과 동일한 키·값을 만드는지 — §3.1a whole-file 오라클이 지키는 불변 표면의 관찰 가능한 출력에 대한 동적 회귀 가드(정적 증거는 §3.1a의 `audit_exact_allowed_delta`가 맡는다; 이 테스트 하나가 "불변"을 증명하는 유일한 근거라고 주장하지 않는다) |
| `test_assemble_v2_main_exit_code_reflects_hosted_release_ready`(all-pass → 0, one-fail → 1) | §3.3 exit code 변경이 실제로 `hosted_release_ready`를 참조하는지 |
| `test_audit_exact_allowed_delta_positive_actual_v2_file`(DR-I4-MAJ-02 양성 케이스) | §3.1a `audit_exact_allowed_delta(base_source, current_source)`를 base revision과 구현 phase가 실제로 완료한 작업 트리 파일에 대해 실행하고 `== []`를 확인 — 합성 fixture가 아니라 §3.2/§3.3/신규 상수가 요구하는 그 변경 자체가 오라클이 정의하는 "정확히 허용된 델타"와 문자 그대로 일치함을 실제 구현물로 직접 증명한다 |
| `test_audit_exact_allowed_delta_positive_synthetic_fixture` | `base_source`에 세 pin된 델타(신규 상수 삽입 + `_build_baseline` 교체 + `main()` 교체)를 이 테스트 안에서 직접 문자열로 적용해 만든 합성 v2 소스로 오라클을 실행하고 `== []` 확인 — 작업 트리 상태와 무관하게 오라클 로직 자체의 자기-정합성을 증명한다(위 실제-파일 테스트와 상호보완) |
| `test_audit_exact_allowed_delta_rejects_new_import_statement` | `_append_top_level_statement(base_source, "import os")` → `unapproved_new_top_level_statement` — 새 import는 이름·바인딩 종류와 무관하게 거부된다(DR-I4-MAJ-01 "imports" 요구) |
| `test_audit_exact_allowed_delta_rejects_import_rebinding_of_protected_name` | `_append_top_level_statement(base_source, "from attacker import REQUIRED_PRODUCERS")` → 거부 — DR-I4-MAJ-01이 지목한 첫 번째 구체적 바이패스 |
| `test_audit_exact_allowed_delta_rejects_class_shadow_of_protected_name` | `_append_top_level_statement(base_source, "class _evaluate_producer:\n    pass")` → 거부 — 두 번째 바이패스 |
| `test_audit_exact_allowed_delta_rejects_for_loop_target_rebinding` | `_append_top_level_statement(base_source, "for REQUIRED_PRODUCERS in ():\n    pass")` → 거부 — 세 번째 바이패스(loop 대상) |
| `test_audit_exact_allowed_delta_rejects_with_alias_rebinding` | `_append_top_level_statement(base_source, "with open('x') as _settings_hash:\n    pass")` → 거부 — 네 번째 바이패스(`with ... as`) |
| `test_audit_exact_allowed_delta_rejects_async_function_shadow` | `_append_top_level_statement(base_source, "async def _check_identity(*a, **k):\n    return None")` → 거부 — `AsyncFunctionDef` 누락 바이패스 |
| `test_audit_exact_allowed_delta_rejects_exception_alias_rebinding` | `_append_top_level_statement(base_source, "try:\n    pass\nexcept Exception as _settings_hash:\n    pass")` → 거부 — `except ... as` 바이패스 |
| `test_audit_exact_allowed_delta_rejects_top_level_named_expression_statement` | `_append_top_level_statement(base_source, "(REQUIRED_PRODUCERS := ('x',))")` → 거부 — 이름 재사용조차 없는 임의의 named-expression statement까지 일반 규칙으로 잡힘을 증명 |
| `test_audit_exact_allowed_delta_rejects_duplicate_assignment_rebinding` | `_append_top_level_statement(base_source, 'REQUIRED_PRODUCERS = ("attacker-job",)')` → 거부(DR-I3-MAJ-02가 다루던 시나리오가 새 오라클 아래에서도 여전히 잡힘을 재확인) |
| `test_audit_exact_allowed_delta_rejects_duplicate_function_rebinding` | `_append_top_level_statement(base_source, "def _evaluate_producer(*a, **k):\n    return None")` → 거부 |
| `test_audit_exact_allowed_delta_rejects_in_place_whitespace_mutation`(parametrize: `import sys`, `REQUIRED_PRODUCERS = (...)` 대입문, `_check_identity` 함수 정의, `if str(_SRC) not in sys.path: ...` 블록 — import/상수/함수/제어문 각 카테고리를 대표하는 4개 서브케이스) | `_in_place_whitespace_mutation(base_source, target_slice)`로 그 statement 슬라이스 **안쪽**에 공백 한 칸을 삽입 → `top_level_statement_changed:index=...` — "26개 보호 심볼"이 아니라 base의 **어떤** statement를 한 글자만 바꿔도 잡힌다는 것을 증명(DR-I4-MAJ-01/02가 요구한 일반화) |
| `test_audit_exact_allowed_delta_rejects_new_executable_statement` | `_append_top_level_statement(base_source, 'print("x")')` → 거부 — 새 실행 가능 statement(DR-I4-MAJ-02 "executable statement" 요구) |
| `test_audit_exact_allowed_delta_rejects_new_unrelated_function` | `_append_top_level_statement(base_source, "def _new_helper():\n    return 1")` → 거부 — 새 함수(DR-I4-MAJ-02 "new unrelated function" 요구) |
| `test_audit_exact_allowed_delta_rejects_assemble_modified` | `assemble()` 함수 슬라이스에 `_in_place_whitespace_mutation` 적용 → 거부 — `assemble`은 세 pin된 델타 어디에도 없으므로 한 글자도 못 바꾼다(DR-I4-MAJ-02 "assemble" 요구) |
| `test_audit_exact_allowed_delta_rejects_main_non_exit_line_modified` | base `main()` 슬라이스에서 `--fresh-dir` 줄처럼 exit 줄이 아닌 다른 줄 하나를 바꾼 뮤턴트 소스를 만들어 실행 → 거부(`top_level_statement_changed`, `main` 위치) — pin된 `PINNED_MAIN_NEW_SLICE`와 정확히 일치하지 않으면 exit 줄 밖의 어떤 변경도 통과하지 못함을 증명(DR-I4-MAJ-02 "non-exit line inside main" 요구) |
| `test_audit_exact_allowed_delta_rejects_main_left_as_base_v1` | base `main()`을 전혀 바꾸지 않은(exit 줄이 여전히 `deterministic_status` 참조) 소스로 실행 → 거부 — "델타를 아예 적용하지 않은 v1 잔존"도 통과하지 못함을 증명(오라클이 "필요한 변경이 일어났는가"와 "그 밖의 변경이 없는가"를 동시에 강제함을 보여주는 회귀 가드) |
| `test_audit_exact_allowed_delta_rejects_build_baseline_arbitrary_rewrite` | `_build_baseline` 슬라이스를 pin된 §3.2 텍스트가 아닌 다른 내용(예: 반환 dict에 필드 하나 추가)으로 교체 → 거부 — "어떤 v2-스러운 재작성이든 통과"가 아니라 정확히 pin된 텍스트만 통과함을 증명(DR-I4-MAJ-02 핵심 요구) |
| `test_audit_exact_allowed_delta_rejects_build_baseline_left_as_base_v1` | `_build_baseline`을 전혀 바꾸지 않은 소스로 실행 → 거부 — 위 main 대칭 케이스 |
| `test_audit_exact_allowed_delta_rejects_new_constants_inserted_at_wrong_location` | pin된 다섯 상수 슬라이스 전체를 anchor 뒤가 아니라 파일 맨 끝에 삽입한 소스로 실행 → 거부 — 삽입 **위치**도 오라클이 강제함을 증명 |
| `test_audit_exact_allowed_delta_rejects_partial_pinned_constants_block` | pin된 다섯 상수 중 `SUPPORT_POLICY_FIXED` 하나를 빼고 나머지 넷만 anchor 뒤에 삽입한 소스로 실행 → 거부 — "일부만 맞으면 통과"가 아니라 다섯 개 전부가 정확히 일치해야 함을 증명 |
| `test_audit_exact_allowed_delta_rejects_missing_statement_removed_from_base` | base의 아무 statement(예: `import re`)를 소스에서 완전히 제거 → 거부(`missing_top_level_statement` 또는 그 이후 인덱스의 `top_level_statement_changed`) — 삭제도 이름 목록과 무관하게 일반적으로 잡힘을 증명 |
| `test_audit_exact_allowed_delta_rejects_current_source_with_syntax_error` | 파싱 불가능한 문자열을 `current_source`로 전달 → `current_source_not_parsable` |

**Decorator-span 뮤턴트 (DR-RC1-I1-MAJ-01).** 아래 표는 §3.1a
`_statement_source_slice`가 decorator를 슬라이스에 포함시키는지, 그리고
그 결과로 `audit_exact_allowed_delta`가 decorator만 바뀐 뮤턴트를 잡는지
증명한다. 정의 자체(함수/클래스 본문)는 어느 케이스에서도 건드리지
않는다 — "otherwise unchanged definition" 앞에 decorator만 붙였다 뗐다
바꿨다 순서를 바꿨다 하는 것이 유일한 변형이다.

| 테스트 이름 | 잡는 뮤턴트 |
|---|---|
| `test_audit_exact_allowed_delta_rejects_decorator_mutations_on_assemble`(4-way parametrize: 추가 `()→("@staticmethod",)`, 제거 `("@staticmethod",)→()`, 수정 `("@decorator_a",)→("@decorator_b",)`, 재정렬 `("@decorator_a","@decorator_b")→("@decorator_b","@decorator_a")`) | 각 서브케이스에서 `audit_exact_allowed_delta(_with_decorators(base_source, ASSEMBLE_SLICE, before), _with_decorators(base_source, ASSEMBLE_SLICE, after))`가 거부됨을 확인 — `ASSEMBLE_SLICE`는 `_top_level_statement_slices(base_source)`에서 얻은, decorator 없는 원본 `assemble` 슬라이스. 첫 서브케이스가 리뷰의 원 재현(`@staticmethod`/`@(lambda f: (lambda *a, **k: {}))` 추가로 슬라이스 목록이 base와 동일해지던 결함)을 직접 반증한다. `assemble`은 세 pin된 델타 어디에도 없으므로 decorator 한 줄도 못 붙는다는 DR-I4-MAJ-02의 요구를 decorator 차원으로 확장 |
| `test_audit_exact_allowed_delta_rejects_decorator_added_to_other_protected_function`(parametrize: `_evaluate_producer`, `_check_identity`) | 해당 함수의 base 슬라이스에 `_with_decorators(..., ("@staticmethod",))`로 decorator를 추가한 소스를 current로 사용 → 거부 — decorator 누락 방지가 `assemble` 하나에 국한되지 않고 base의 모든 top-level 함수에 동일하게 적용됨을 증명(`_statement_source_slice`가 노드 종류로만 판단하고 이름을 보지 않으므로) |
| `test_audit_exact_allowed_delta_rejects_decorator_added_to_synthetic_class` | `_append_top_level_statement(base_source, "class _Shadow:\n    pass")`로 base에 없는 class statement를 먼저 추가해 새 base로 삼고, 그 class 슬라이스에 `_with_decorators(..., ("@some_decorator",))`로 decorator를 추가한 소스를 current로 사용 → 거부 — `ClassDef`도 `FunctionDef`/`AsyncFunctionDef`와 동일하게 decorator span이 슬라이스 시작 좌표에 포함됨을 증명(DR-RC1-I1-MAJ-01의 "ClassDef" 요구) |
| `test_audit_exact_allowed_delta_rejects_decorator_added_to_synthetic_async_function` | 위와 동일하되 `"async def _shadow():\n    return None"`을 추가 statement로 사용, decorator `@some_decorator` 추가 → 거부 — `AsyncFunctionDef`도 동일하게 보호됨을 증명(DR-RC1-I1-MAJ-01의 "AsyncFunctionDef" 요구) |
| `test_statement_source_slice_decorated_function_starts_at_at_symbol_and_includes_all_decorators`(단위 검증, parametrize: decorator 1개, decorator 2개+그 사이 주석·빈 줄 1줄씩) | `_top_level_statement_slices`가 반환한 `assemble` 위치의 슬라이스 문자열이 정확히 `"@staticmethod\ndef assemble(args) -> dict:\n..."`(1개 케이스) 또는 두 decorator 줄과 그 사이 주석/빈 줄을 포함한 전체 텍스트(2개 케이스)로, `"@"`에서 시작해 `assemble` 함수 본문 끝에서 끝남을 오라클을 거치지 않고 헬퍼 반환값 자체로 직접 assert — DR-RC1-I1-MAJ-01이 지적한 "decorator가 슬라이스 밖에 남는다" 결함이 슬라이스 생성 단계에서 직접 고쳐졌다는 것을 오라클의 최종 판정(reject/accept)이 아니라 중간 산출물로 증명 |
| `test_audit_exact_allowed_delta_comment_and_blank_line_insertions_between_statements_are_invisible` | `assemble` 정의 바로 앞에 `"# REQUIRED_PRODUCERS = (\"attacker-job\",)"`처럼 실행 가능한 statement처럼 보이는 주석 한 줄과 빈 줄 하나를 삽입한 문자열을 current로 사용해 `audit_exact_allowed_delta(base_source, current) == []`(양성)를 확인 — 주석/빈 줄은 AST statement를 만들지 않으므로 `_top_level_statement_slices`가 반환하는 목록 자체가 base와 완전히 동일하게 유지된다는 것을 직접 증명한다. comment/blank line이 실행 가능한 syntax를 감출 수 없다는 성질(comments never produce AST nodes)을 보여주는 것이 이 테스트의 유일한 목적이며, `test_audit_exact_allowed_delta_rejects_current_source_with_syntax_error`(위, 무변경 유지)와 대칭으로 "파싱 불가능한 입력은 fail-closed, 파싱 가능하지만 비-statement인 텍스트 변화는 오라클의 판정 표면 밖"이라는 두 경계를 함께 고정한다 |

**Preamble byte/token-aware 뮤턴트 (DR-RC1-I2-MAJ-01).** 아래 표는 §3.1b
`_source_preamble`/`audit_exact_allowed_delta_bytes`가 shebang/encoding
cookie/BOM 변형을 실제로 잡는지, 그리고 그런 변형이 없을 때는 정상적으로
§3.1a `audit_exact_allowed_delta`에 위임되는지 증명한다. `base_bytes`는
`git show adda1759754b56b514b3ab6252c2dc1032e03d28:scripts/assemble_m4_evidence.py`의
raw stdout bytes(디코드하지 않음). 실제 base에는 shebang만 있고 cookie가
없으므로, "cookie 수정"·"cookie 삭제"·"BOM이 이미 있는 상태에서 불변"
계열은 이 표 안에서 직접 만든 합성(synthetic) base bytes를 쓴다(각 행에
명시). 정의 자체(statement 본문)는 어느 케이스에서도 건드리지 않는다 —
preamble만이 유일한 변형이다.

| 테스트 이름 | 잡는 뮤턴트 |
|---|---|
| `test_source_preamble_matches_pinned_base_preamble_bytes` | `_source_preamble(base_bytes) == PINNED_BASE_PREAMBLE_BYTES == b"#!/usr/bin/env python3\n"` — 실제 base 파일의 preamble이 정확히 shebang 한 줄뿐임을 오라클을 거치지 않고 헬퍼 반환값 자체로 직접 assert(cookie 없음도 함께 고정) |
| `test_audit_exact_allowed_delta_bytes_positive_actual_v2_file` | 실제 base_bytes와 구현 phase가 실제로 완료한 작업 트리 파일의 raw bytes로 `audit_exact_allowed_delta_bytes(base_bytes, current_bytes) == []` 확인 — §3.1a `test_audit_exact_allowed_delta_positive_actual_v2_file`(내부 `audit_exact_allowed_delta`용, 그대로 유지)과 별개로, §3.1b가 신설하는 raw-bytes 진입점이 실제 구현물에 대해서도 통과함을 증명하는 새 정본(canonical) 양성 테스트 |
| `test_audit_exact_allowed_delta_bytes_rejects_shebang_modified` | base_bytes의 첫 줄을 `#!/usr/bin/env -S python3 -O\n`으로 바꾸고 나머지는 그대로 둔 current_bytes → `preamble_shebang_or_encoding_declaration_changed` — 리뷰가 제시한 원 재현 |
| `test_audit_exact_allowed_delta_bytes_rejects_shebang_removed` | base_bytes에서 첫 줄(shebang)만 완전히 제거한 current_bytes(둘째 줄이던 모듈 docstring이 이제 첫 줄) → `preamble_shebang_or_encoding_declaration_changed` — statement 시퀀스만으로는 절대 안 잡히던 케이스(모듈 docstring 슬라이스 문자열 자체는 줄 번호와 무관하므로 base와 동일하게 남는다)를 preamble 비교가 직접 잡음을 증명 |
| `test_audit_exact_allowed_delta_bytes_rejects_shebang_inserted_into_no_shebang_base` | 합성 base_bytes(`b'"""doc"""\nimport os\n'`, shebang 없음, preamble `b""`)와, 그 앞에 `#!/usr/bin/env python3\n`를 삽입한 current_bytes → `preamble_shebang_or_encoding_declaration_changed` — "삭제"의 대칭 방향인 "삽입"도 동일하게 잡힘을 증명 |
| `test_audit_exact_allowed_delta_bytes_rejects_encoding_cookie_inserted_with_non_ascii_semantic_reproduction` | 리뷰의 정확한 재현 — base_bytes 둘째 줄에 `# coding: latin-1\n`을 삽입한 current_bytes → `preamble_shebang_or_encoding_declaration_changed`. 이 테스트는 판정 하나만 확인하지 않는다 — 오라클과 별개로, base의 원본 모듈 docstring bytes를 `"utf-8"`로 디코드한 문자열과 (`tokenize.detect_encoding`이 그 current_bytes에서 실제로 검출하는) `"iso-8859-1"`로 디코드한 문자열을 직접 비교해, em dash(U+2014)가 서로 다른 문자로("â" 계열 mojibake) 디코드됨을 별도로 assert한다 — "cookie 변경이 실제로 실행되는 문서 문자열을 바꾼다"는 리뷰의 non-ASCII semantic reproduction 주장을 판정 결과가 아니라 디코드된 텍스트 자체로 직접 증명 |
| `test_audit_exact_allowed_delta_bytes_rejects_encoding_cookie_modified` | 합성 base_bytes(`b"# coding: utf-8\nimport os\n"`, cookie 있음)와 cookie 값만 `latin-1`로 바꾼 current_bytes → `preamble_shebang_or_encoding_declaration_changed` — 실제 base는 cookie가 없어 "삽입"만으로는 "수정"을 재현할 수 없으므로 합성 fixture로 cookie가 이미 있는 상태에서의 값 변경까지 커버 |
| `test_audit_exact_allowed_delta_bytes_rejects_encoding_cookie_removed` | 위와 같은 합성 base_bytes에서 cookie 줄만 제거한 current_bytes(`b"import os\n"`) → `preamble_shebang_or_encoding_declaration_changed` — "삽입"의 대칭 방향 |
| `test_audit_exact_allowed_delta_bytes_rejects_bom_inserted` | base_bytes 그대로에 UTF-8 BOM(`b"\xef\xbb\xbf"`)만 앞에 붙인 current_bytes → `preamble_shebang_or_encoding_declaration_changed`(preamble이 `b"#!/usr/bin/env python3\n"`에서 `b"\xef\xbb\xbf#!/usr/bin/env python3\n"`로 달라짐) |
| `test_audit_exact_allowed_delta_bytes_rejects_bom_plus_conflicting_cookie_fails_closed` | current_bytes = UTF-8 BOM + `#!/usr/bin/env python3\n` + `# coding: latin-1\n` + base의 나머지 — BOM은 UTF-8을 함의하는데 cookie가 다른 encoding을 선언해 PEP 263/CPython 자신이 `SyntaxError: encoding problem: utf-8`로 정의하는 상충 상태. `tokenize.detect_encoding`이 그 예외를 던지고 `_source_preamble`이 `None`을 반환 → `audit_exact_allowed_delta_bytes`가 `current_source_encoding_conflict`를 반환(크래시하거나 조용히 통과시키지 않고 명시적으로 fail-closed함을 증명) |
| `test_audit_exact_allowed_delta_bytes_accepts_identical_bom_present_in_base_and_current` | 합성 base_bytes/current_bytes가 동일한 UTF-8 BOM + cookie 없음 + 동일한 statement 본문을 가짐(`b"\xef\xbb\xbf" + b"import os\nx = 1\n"`, 양쪽 동일) → `== []` — BOM이 진짜로 안 바뀌었으면 그 존재만으로 오탐(spurious reject)하지 않음을 증명(§3.1b "byte-for-byte 동일성을 요구하지 않는다"는 partition 진술이 preamble 쪽에서도 과잉 제약이 아님을 뒷받침) |
| `test_audit_exact_allowed_delta_bytes_accepts_leading_non_cookie_comment_as_inert` | base_bytes/current_bytes 둘 다 (변경하지 않은) 실제 shebang 바로 다음 줄에 cookie가 아닌 평범한 주석(`# TODO: unrelated`) 한 줄을 동일하게 삽입하고 나머지는 base와 동일 → `== []` — cookie 위치에 있지만 실제로 cookie 정규식과 매치하지 않는 주석은 preamble로 오인되지 않고 statement 영역의 통상적인 inert gap으로 남는다는 §3.1b partition 진술을 직접 증명 |

### 7.2 `tests/unit/test_check_m4_baseline.py`

기존 `_valid_candidate()`를 `_valid_v1_legacy_candidate()`로 이름을
바꾸고 내용은 그대로 유지한다(schema `m4-baseline-v1`). 기존
`test_strict_schema_and_algebra_matrix`를
`test_v1_legacy_strict_schema_and_algebra_matrix`로 이름을 바꾸되, 본문의
모든 `checker.check(valid, expect_operational_blocked=True)` 호출을
`checker.check(valid, allow_legacy_v1=True, expect_operational_blocked=True)`로만
바꾼다(그 외 (a)~(n) 14개 서브케이스는 그대로 보존 — v1 대수 회귀 가드).

**신규 — DR-I1-MAJ-01 frozen-blocked mutant 목록.** `_valid_v1_legacy_candidate()`를
`copy.deepcopy`하고 **`allow_legacy_v1=True`만** 주고
(`expect_operational_blocked`는 기본값 `False`, 즉 주지 않음) 아래 필드
하나씩만 깨서 여전히 거부되는지 확인한다 — 이 표가 "frozen 상태는
`--allow-legacy-v1` 하나만으로 강제된다"를 직접 증명한다:

| 테스트 이름 | mutant 필드 | 기대 issue |
|---|---|---|
| `test_v1_legacy_rejects_live_regression_pass_without_expect_flag` | `gates["m3_live_regression"]="PASS"` | `v1_legacy_live_regression_not_not_run` |
| `test_v1_legacy_rejects_live_regression_skipped_without_expect_flag` | `gates["m3_live_regression"]="SKIPPED"` | `v1_legacy_live_regression_not_not_run` |
| `test_v1_legacy_rejects_m41_operational_pass_without_expect_flag` | `gates["m41_operational"]="PASS"` | `v1_legacy_m41_operational_not_blocked` |
| `test_v1_legacy_rejects_m41_blocked_false_without_expect_flag` | `candidate["M4.1_BLOCKED"]=False` | `v1_legacy_m41_blocked_not_true` |
| `test_v1_legacy_rejects_overall_release_ready_true_without_expect_flag` | `candidate["overall_release_ready"]=True` | `v1_legacy_overall_release_ready_not_false` |
| `test_v1_legacy_rejects_operational_status_pass_without_expect_flag` | `candidate["operational_status"]="PASS"` | `v1_legacy_operational_status_not_blocked` |
| `test_v1_legacy_accepts_frozen_state_with_allow_legacy_v1_alone` | (변경 없음, 양성 기준선) | `ok is True` — `--expect-operational-blocked` 없이도 통과 |

신규 `_valid_v2_candidate()` fixture(schema `m4-baseline-v2`, `git_sha`=
고정 40자 hex 문자열, `workflow_run`= `{"run_id": "12345", "run_attempt":
"1", "workflow_path": ".github/workflows/ci.yml", "event_name": "push"}`,
`gates`에 `m3_live_regression`/`m41_operational` = `"NOT_ADOPTED"`,
`support_policy` = `SUPPORT_POLICY_FIXED`, `hosted_release_ready=True`,
`native_linux_release_ready=False`, `full_production_release_ready=False`,
`overall_release_ready=False`, `M4.1_BLOCKED=False`,
`operational_status="NOT_ADOPTED"`, 그리고 `producers`의 `container`/
`m43-deterministic`이 각각 `status="OK"`인 `payload_hashes`를 가지며
top-level `image_digest`/`m43_deterministic_receipt_sha256`이 그
`payload_hashes["container_smoke.json"]`/`payload_hashes["m43.json"]`와
정확히 같은 값을 가리키도록 구성 — 그래야 §4.3의 alias 재계산이 양성
기준선에서 통과한다)와 아래 신규 테스트:

| 테스트 이름 | 잡는 뮤턴트 |
|---|---|
| `test_v2_valid_candidate_passes` | 양성 기준선 — 이후 모든 negative 테스트가 여기서 하나씩만 필드를 깨는 `copy.deepcopy` 패턴을 쓸 수 있게 함 |
| `test_v2_rejects_missing_or_extra_top_level_key`((a)/(b) v1과 대칭) | §2.1 exact-set 검사 회귀 |
| `test_v2_rejects_live_regression_pass_substituted_for_not_adopted` | `gates["m3_live_regression"]="PASS"` — **가장 핵심적인 adversarial 케이스**(§0.3-1이 말하는 "NOT_ADOPTED가 PASS가 될 수 없다"의 직접 반례 시도) |
| `test_v2_rejects_m41_operational_pass_substituted_for_not_adopted` | 위와 대칭, `m41_operational` |
| `test_v2_rejects_waived_value_anywhere_in_gates` | `gates[key]="WAIVED"` — `GATE_ENUM_V2`에 없으므로 `unknown_gate_enum_v2`로 거부 |
| `test_v2_rejects_gate_producer_algebra_mismatch` | 기존 (d)와 동일 패턴, v2 top-key로 |
| `test_v2_rejects_deterministic_status_mismatch` | `deterministic_status="FAIL"`인데 네 producer가 모두 OK |
| `test_v2_hosted_release_ready_algebra_matrix`(all-OK→True 기대, 4-way 각 producer를 MISSING으로 바꿔 →False 기대, 5 서브케이스) | producer 실패 유형별로 `hosted_release_ready` 재계산이 정확한지 |
| `test_v2_rejects_hosted_release_ready_self_report_disagreeing_with_producers` | candidate가 producers는 실패로 두고 `hosted_release_ready=True`라고 거짓 보고 — "self-reported deterministic PASS over failed producer evidence" adversarial case(Plan §4) |
| `test_v2_rejects_true_native_linux_release_ready` | `native_linux_release_ready=True` — 가장 치명적인 단일 필드 조작 시도 |
| `test_v2_rejects_true_full_production_release_ready` | 위와 대칭 |
| `test_v2_rejects_overall_release_ready_disagreeing_with_full_production` | `overall_release_ready=True`이면서 `full_production_release_ready=False`(alias 위조) |
| `test_v2_rejects_operational_status_not_equal_not_adopted`(`"PASS"`, `"BLOCKED"` 두 값 parametrize) | `operational_status` 위조 |
| `test_v2_rejects_m41_blocked_true` | `"M4.1_BLOCKED"=True` — v1 값이 v2에 남는 회귀 |
| `test_v2_rejects_support_policy_wrong_schema_or_scope_or_date_or_native_ollama`(4-way parametrize, 각 필드 하나씩 오염) | §2.3 고정값 검사 |
| `test_v2_rejects_support_policy_extra_or_missing_key` | `support_policy` 자체의 key-set 위반 |
| `test_v2_rejects_git_sha_not_nonempty_string`(`None`/`""`/`123` 세 값 parametrize) | `git_sha` 타입/공백 위반 — DR-I1-MAJ-02 |
| `test_v2_rejects_workflow_run_key_set_mismatch`(누락/추가 두 서브케이스) | `workflow_run` exact-key-set 위반 |
| `test_v2_rejects_workflow_run_field_not_nonempty_string`(`run_id`/`run_attempt`/`workflow_path`/`event_name` 4-way parametrize, 각각 `None`/`123`/`""`) | 신원 필드 타입 위반 |
| `test_v2_identity_flags_absent_by_default_no_regression` | 다섯 `--expect-*` 신원 플래그를 아무것도 주지 않으면 신원 값과 무관하게 유효한 candidate가 그대로 통과(§4.7 `m4-assemble` step 사용 패턴의 회귀 가드) |
| `test_v2_identity_flags_reject_cross_sha_mismatch` | `expect_sha="다른sha"`, candidate의 `git_sha`는 그대로 — `identity_sha_mismatch` |
| `test_v2_identity_flags_reject_cross_run_id_mismatch` | 위와 대칭, `run_id` |
| `test_v2_identity_flags_reject_cross_run_attempt_mismatch` | 위와 대칭, `run_attempt` |
| `test_v2_identity_flags_reject_cross_workflow_path_mismatch` | 위와 대칭, `workflow_path` |
| `test_v2_identity_flags_reject_cross_event_mismatch` | 위와 대칭, `event_name` |
| `test_v2_identity_flags_accept_matching_identity` | 다섯 플래그 모두 candidate와 일치 — 양성 케이스 |
| `test_v2_rejects_image_digest_alias_tampered_while_container_ok` | `producers.container.status="OK"`인데 top-level `image_digest`를 다른 hex64로 바꿈 — `image_digest_alias_mismatch`(가장 핵심적인 provenance 위조 시도) |
| `test_v2_rejects_image_digest_not_null_when_container_not_ok` | `producers.container.status="MISSING"`인데 `image_digest`가 여전히 hex64 값 — 실패한 producer 위에 가짜 provenance가 남는 회귀 |
| `test_v2_rejects_m43_receipt_sha_alias_tampered_while_m43_deterministic_ok` | 위와 대칭, `m43_deterministic_receipt_sha256`/`producers["m43-deterministic"]` |
| `test_v2_rejects_m43_receipt_sha_not_null_when_m43_deterministic_not_ok` | 위와 대칭 |
| `test_v2_expect_hosted_release_ready_flag_satisfied_and_not_satisfied`(양성/음성 두 서브케이스) | `expect_hosted_release_ready=True` 사용 시 §4.3 마지막 두 if 블록 |
| `test_v2_expect_hosted_not_ready_flag_satisfied_and_not_satisfied` | 위와 대칭 |
| `test_v1_candidate_rejected_without_allow_legacy_v1` | `check(v1_candidate)`(플래그 기본값)이 `legacy_v1_schema_requires_allow_legacy_v1_flag`로 거부되는지 — 이것이 "checker accepts v2 by default"의 직접 증명 |
| `test_v1_candidate_with_v2_only_field_injected_still_rejected_under_allow_legacy_v1` | v1 candidate에 `hosted_release_ready` 키를 억지로 추가하면 `--allow-legacy-v1`을 줘도 v1의 exact-key 검사(`REQUIRED_TOP_KEYS_V1`)에서 거부 — "compatibility mode MUST NOT... call v1 hosted-ready"의 구조적 증명 |
| `test_unknown_schema_string_rejected` | `schema="m4-baseline-v3"` 같은 완전히 모르는 값 |
| `test_mismatched_schema_version_pair_rejected`(`schema=v2,version="1.0.0"`와 `schema=v1,version="2.0.0"` 두 조합) | §4.5 "쌍으로만 유효" 방어 |
| `test_main_cli_expect_operational_blocked_without_allow_legacy_v1_exits_2` | argparse 조합 규칙 1 (subprocess 또는 `pytest.raises(SystemExit)`로 `main()` 직접 호출) |
| `test_main_cli_allow_legacy_v1_with_expect_hosted_release_ready_exits_2` | 조합 규칙 2 |
| `test_main_cli_both_hosted_expectation_flags_exits_2` | 조합 규칙 3 |
| `test_main_cli_allow_legacy_v1_with_require_identity_binding_or_expect_sha_exits_2`(6-way parametrize: `--require-identity-binding` + 다섯 신원 플래그 각각) | 조합 규칙 2 확장 — legacy와 신원-바인딩 관련 플래그 전부의 상호배제 |
| `test_main_cli_require_identity_binding_without_expect_hosted_release_ready_exits_2` | 조합 규칙 4(DR-I1-MAJ-02) |
| `test_main_cli_require_identity_binding_without_identity_flags_exits_2`(5-way parametrize, 다섯 신원 플래그 중 하나씩만 누락) | 조합 규칙 5(DR-I1-MAJ-02) — 부분 누락도 거부되는지 |
| `test_main_cli_v2_candidate_no_expectation_flags_exits_0` | 정상 경로 회귀(§4.7의 CI 사용 패턴과 동일) |
| `test_main_cli_expect_hosted_release_ready_alone_exits_0_without_identity_flags`(all-pass→0, one-fail→1) | Plan.md §5 pre-merge fixture 명령이 신원 플래그·`--require-identity-binding` 없이도 그대로 동작하는지의 회귀 가드(신원-바인딩은 opt-in이지 강제 전제조건이 아님을 증명) |
| `test_main_cli_require_identity_binding_exits_0_when_identity_and_hosted_ready_match`(다섯 신원 플래그를 candidate와 일치시켜 `--require-identity-binding`과 함께 전달) | CLI end-to-end, §8.3 §6.1 runbook 명령과 동일 인자 조합 |
| `test_main_cli_require_identity_binding_exits_1_on_cross_sha` | `--require-identity-binding`과 함께 `--expect-sha`만 candidate와 다르게 주면 exit 1(2가 아님 — 이건 조합 규칙 위반이 아니라 candidate 검증 실패) |

### 7.3 `tests/unit/test_ci_workflow_contract.py`

기존 `test_m4_assemble_needs_all_four_hosted_producers`,
`test_container_and_m43_deterministic_jobs_exist_hosted`,
`test_m43_evidence_upload_artifact_steps_use_if_no_files_found_error`는
**수정하지 않는다**(unaffected job들의 계약).

`test_protected_live_gate_trigger_runner_environment_unchanged`를
아래로 **대체**한다(함수 이름도 바꿔 옛 이름이 더 이상 "unchanged"를
주장하지 않게 한다). DR-I1-MAJ-03이 지적한 대로, "체크아웃 없음"·
"self-hosted 라벨 없음" 같은 개별 필드 검사만으로는 두 번째 `run:` step에
`curl`/`${{ secrets.* }}`/저장소 스크립트 호출을 추가하는 뮤턴트를 잡지
못한다 — 아래 표는 (1) **exact-shape 구조 검사**(정확한 key-set, 정확히
한 step, 정확히 하나의 허용된 스크립트)와 (2) **source-level 금지
형태 검사**(파싱된 구조가 아니라 워크플로 raw YAML 텍스트 자체에 대한
검사, 구조 검사를 우회하는 새 필드 추가 자체를 막기 위함)를 모두
포함한다.

**Iteration 2 리뷰가 지적한 오류와 정정(DR-I2-MAJ-01):** 이전 iteration의
금지 목록은 `self-hosted`/`environment:`를 **위치 무관 bare substring**으로
검사했다. 그런데 §5.3의 허용된 stub 스크립트 자신이 안전함을 설명하기
위해 "no self-hosted runner"라는 부정문을 쓴다 — 이 문자열은 `self-hosted`
bare substring과 매치하므로, **금지 목록을 만족해야 하는 유일한 정답
구현(§5.3 그 자체)이 자기 자신의 금지 검사에 걸려 항상 거부당하는**
자기모순이 있었다. 이 모순의 근본 원인은 "정확히 pin된 값을 또다시 raw
substring으로 재검사"하려 한 데 있다 — `job["steps"][0]["run"]`은 이미
`test_m3_live_regression_gate_step_run_exact_allowlisted_script`가 `==`
전체 일치로 완전히 고정하므로, 그 필드의 내용을 raw substring 검사가
다시 검사할 필요가 없다(오히려 다시 검사하면 "안전을 설명하는 부정문"과
"위험한 실행 형태"를 구별하지 못해 거짓 양성을 만든다). 정정한 설계는
두 가지를 동시에 한다: (1) 이미 exact-pin된 `run` 필드 값은 raw
substring 검사 대상에서 제외하고, (2) 그래도 남기는 raw 검사는 bare
단어가 아니라 **위험한 실행 형태**(비밀 보간, YAML key로서의
`environment:`, 실제 runner 라벨, 엔드포인트/모델 토큰, 네트워크 명령,
checkout/fetch/clone 명령, 저장소 스크립트 실행)만 정밀하게 매치한다.
"안전하다"는 설명문이나 정책 문서 이름은 금지 대상이 아니다.

```python
import re
import textwrap

# job key 줄부터 다음 top-level job key 줄 전까지만 자른다(선행 주석은
# 포함하지 않음 — 주석은 이 job의 실행 가능한 표면이 아니라 §5.3의
# 설계 근거 설명이며, `.github/workflows/ci.yml`이 아니라 이 문서에
# 그 근거를 남기는 것으로 충분하다).
M3_GATE_JOB_KEY_LINE = "  m3-live-regression-gate:"

# job["steps"][0]["run"]과 반드시 문자 그대로 동일해야 하는 §5.3의
# 허용된 스크립트. test_m3_live_regression_gate_step_run_exact_allowlisted_script
# 가 이 상수와 `==`로 이미 완전히 고정하므로, 아래 raw denylist 검사에서는
# 이 정확한 문자열을 블록 텍스트에서 제거한 뒤에만 스캔한다 — 이미
# exact-pin된 값을 이중으로 재검사하지 않는다는 원칙(DR-I2-MAJ-01)의
# 직접 구현이다.
M3_GATE_PINNED_RUN_SCRIPT = (
    'echo "::notice::m3-live-regression-gate is NOT_ADOPTED under the current hosted/OCI release policy."\n'
    'echo "This run performed no checkout, no secrets, no environment approval, and no self-hosted runner."\n'
    'echo "See docs/milestones/m4-operational-acceptance-recovery/Requirement.md and Stop_Report.md for the reactivation path."\n'
    'exit 0\n'
)

FORBIDDEN_WORKFLOW_EXECUTABLE_PATTERNS = (
    (r"\$\{\{\s*secrets\.", "secret_interpolation"),
    (r"(?m)^\s*environment:\s", "environment_approval_key"),          # YAML key로서만, 산문 아님
    (r"runs-on:\s*\[?\s*self-hosted", "self_hosted_runner_label"),     # 실제 라벨 재도입만, "no self-hosted" 부정문 아님
    (r"(?i)\bollama\b", "ollama_vendor_token"),
    (r"gpt-oss", "ollama_model_token"),
    (r"\b11434\b", "ollama_default_port"),
    (r"OLLAMA_BASE_URL", "ollama_base_url_env"),
    (r"RUN_LIVE_LLM_TESTS", "live_llm_test_trigger_env"),
    (r"curl\s", "network_fetch_curl"),
    (r"wget\s", "network_fetch_wget"),
    (r"actions/checkout", "checkout_action"),
    (r"checkout@", "checkout_action_pin"),
    (r"git\s+fetch", "git_fetch_command"),
    (r"git\s+clone", "git_clone_command"),
    (r"(?m)^\s*run:.*\bscripts/", "repository_script_execution"),
)

_NEXT_TOP_LEVEL_JOB_KEY_RE = re.compile(r"(?m)^  [A-Za-z0-9_-]+:\s*$")

def _m3_gate_raw_block(workflow_text: str) -> str:
    """M3_GATE_JOB_KEY_LINE부터 다음 top-level job 키 줄(들여쓰기 2칸 +
    영숫자/하이픈/언더스코어 이름 + 콜론으로 끝나는 다음 줄, 또는 파일
    끝) 전까지만 잘라낸다. 선행 주석은 포함하지 않는다 — 슬라이스는
    정확히 job key 줄에서 시작한다."""
    start = workflow_text.index(M3_GATE_JOB_KEY_LINE)
    search_from = start + len(M3_GATE_JOB_KEY_LINE)
    match = _NEXT_TOP_LEVEL_JOB_KEY_RE.search(workflow_text, search_from)
    end = match.start() if match else len(workflow_text)
    return workflow_text[start:end]

_RUN_BLOCK_SCALAR_HEADER_RE = re.compile(r"(?m)^(?P<indent>[ ]*)run:[ \t]*\|[ \t]*\n")

def _m3_gate_denylist_scan_text(workflow_text: str) -> str:
    """raw 블록에서 `run: |` block-scalar로 작성된 스크립트가 실제로
    (공통 들여쓰기를 제거한 뒤) pin된 `M3_GATE_PINNED_RUN_SCRIPT`와 문자
    그대로 같을 때만, 그 들여쓰기 포함 원본 줄들을 제거한 나머지 텍스트를
    반환한다. YAML block-scalar의 각 줄은 파싱된 스칼라 문자열에는 없는
    고정 들여쓰기를 앞에 달고 있으므로, 들여쓰기 없는 pin 문자열을 raw
    블록 텍스트에 그대로 `str.replace`하면 절대 일치하지 않아 아무것도
    지워지지 않는다(DR-I3-MIN-01) — 이전 구현이 바로 이 버그였다. 이
    구현은 `run: |` header 줄보다 더 깊게 들여쓰기된 연속 줄만 스칼라
    본문으로 수집하고, `textwrap.dedent`로 공통 들여쓰기를 제거해서만
    pin 문자열과 비교한다. run 필드가 뮤턴트로 바뀌어 dedent 결과가 pin과
    더 이상 일치하지 않으면 아무것도 제거하지 않고 블록 전체를 그대로
    반환한다 — 그 뮤턴트 텍스트까지 포함해서 스캔되므로 이 축소는 검사를
    약화시키지 않고 강화한다."""
    block = _m3_gate_raw_block(workflow_text)
    header = _RUN_BLOCK_SCALAR_HEADER_RE.search(block)
    if header is None:
        return block
    header_indent = len(header.group("indent"))
    remainder = block[header.end():]
    scalar_lines: list[str] = []
    consumed = 0
    for line in remainder.splitlines(keepends=True):
        content = line[:-1] if line.endswith("\n") else line
        if content.strip() != "":
            indent = len(content) - len(content.lstrip(" "))
            if indent <= header_indent:
                break
        scalar_lines.append(line)
        consumed += len(line)
    raw_scalar_text = "".join(scalar_lines)
    if textwrap.dedent(raw_scalar_text) != M3_GATE_PINNED_RUN_SCRIPT:
        return block
    return block[:header.end()] + remainder[consumed:]
```

**Iteration 3 리뷰가 지적한 오류와 정정(DR-I3-MIN-01):** 이전 iteration의
`_m3_gate_denylist_scan_text`는 `block.replace(M3_GATE_PINNED_RUN_SCRIPT,
"", 1)`을 그대로 썼다. 그러나 raw YAML 블록의 `run: |` block-scalar 각
줄은 파싱된 `job["steps"][0]["run"]` 문자열에는 없는 들여쓰기를 앞에
달고 있다 — 예를 들어 `    echo "::notice::..."`처럼 선행 공백이 있는
반면 `M3_GATE_PINNED_RUN_SCRIPT`는 들여쓰기 없는 스칼라 값 그 자체다.
기계적 재현 결과 YAML은 정확히 pin된 스크립트로 파싱되는데도 `replace`는
raw 블록을 전혀 지우지 못했다 — canonical stub은 §5.3의 부정문
("no self-hosted runner")이 정밀 패턴과 매치하지 않아 우연히 여전히
통과했지만, 문서가 주장한 "exact-pin된 run 필드는 스캔에서 제외된다"는
설계 근거 자체는 거짓이었다. 정정한 구현은 `run: |` header의 들여쓰기
폭을 기준으로 스칼라 본문 줄만 동적으로 수집해 dedent한 뒤에만 pin과
비교하므로, 실제로 그 줄들을 제거한다 — 아래
`test_m3_gate_denylist_scan_text_actually_removes_pinned_scalar`가 이
제거가 공수표가 아님을 직접 assert한다.

| 테스트 이름 | 검증 내용 |
|---|---|
| `test_m3_live_regression_gate_is_workflow_dispatch_opt_in_only` | `job["if"]`에 `workflow_dispatch`와 `enable_m3_live_regression`이 모두 포함되고 `push`/`pull_request` 문자열이 전혀 없음(정규식이 아니라 `"push" not in condition`/`"pull_request" not in condition` 부분 문자열 부재 확인) |
| `test_m3_live_regression_gate_exact_job_key_set` | `set(job) == {"if", "runs-on", "timeout-minutes", "steps"}` — `env`/`environment`/`permissions`/`needs`/`concurrency` 등 새 top-level 키가 하나라도 추가되면 거부(exact-set, 부분집합 아님) |
| `test_m3_live_regression_gate_has_no_self_hosted_or_environment` | `job.get("runs-on") == "ubuntu-latest"`(리스트 아님, 라벨 배열이 아님), `"environment" not in job`(위 exact-key-set 테스트와 이중 방어) |
| `test_m3_live_regression_gate_exactly_one_step_with_exact_step_key_set` | `len(job["steps"]) == 1`이고 `set(job["steps"][0]) == {"name", "run"}` — 두 번째 step 추가, 또는 `uses`/`with`/`env`가 step에 섞이는 시도를 exact-set으로 거부 |
| `test_m3_live_regression_gate_step_run_exact_allowlisted_script` | `job["steps"][0]["run"]`이 `M3_GATE_PINNED_RUN_SCRIPT`와 **정확히** 문자열 일치(부분 포함이 아니라 `==`) — 스크립트에 몰래 한 줄을 추가하는 뮤턴트를 잡는다(§5.3 코드 블록 자체를 pin) |
| `test_m3_gate_denylist_scan_text_actually_removes_pinned_scalar`(DR-I3-MIN-01 양성 케이스) | §5.3의 canonical job 딕셔너리를 실제 들여쓰기가 있는 raw YAML 텍스트(`run: \|` block-scalar, 스크립트 각 줄이 들여쓰기됨)로 렌더링해 `_m3_gate_denylist_scan_text`에 통과시키고, `M3_GATE_PINNED_RUN_SCRIPT`의 각 줄이 반환된 텍스트에 하나도 남아 있지 않음을 개별 assert — "canonical pinned scalar가 실제로 제거된다"는 것을 직접 증명한다(이전 구현은 조용히 아무것도 지우지 못했다, DR-I3-MIN-01) |
| `test_m3_live_regression_gate_source_denylist_has_no_forbidden_executable_surfaces`(parametrize, `FORBIDDEN_WORKFLOW_EXECUTABLE_PATTERNS` 각 항목) | `_m3_gate_denylist_scan_text(workflow_text)`(위 양성 테스트로 실제 제거가 증명된, exact-pin된 run 스크립트를 제외한 나머지 블록 텍스트)에 각 패턴이 매치하지 않음을 개별 assert — 파싱 구조를 우회해 raw YAML 필드에 위험한 실행 형태를 몰래 추가하는 뮤턴트까지 잡는다(구조 검사와 source-level 검사의 이중 방어). exact-pin된 run 필드 자체는 이제 실제로 스캔에서 제외되므로, §5.3의 안전 설명 부정문("no self-hosted runner" 등)이 거짓 양성을 만들지 않는다 |
| `test_m3_live_regression_gate_canonical_stub_literal_satisfies_full_contract_suite`(DR-I2-MAJ-01 양성 케이스) | §5.3에 정의된 job 딕셔너리 리터럴과 `M3_GATE_PINNED_RUN_SCRIPT`를 이 테스트 안에서 실제 들여쓰기가 있는 raw YAML 텍스트로 렌더링해(파일을 파싱하지 않고 설계 스펙 그 자체를 인메모리 fixture로 사용), 위 5개 구조 검사 + denylist 스캔(실제 들여쓰기 제거 포함)을 모두 같은 테스트 함수 안에서 순서대로 실행하고 전부 통과하는지 확인 — "이 설계가 요구하는 구현은 이 설계가 요구하는 계약 전체를 실제로 만족할 수 있다"를 파일 상태와 무관하게 직접 증명한다(이 테스트가 실패하면 계약 자체가 내적으로 불충족 가능하다는 뜻이므로 구현 phase 진입 전에 걸러진다) |
| `test_m3_live_regression_gate_source_denylist_rejects_each_forbidden_executable_surface`(DR-I2-MAJ-01 음성 케이스, DR-I3-MIN-01로 파싱 가능한 실행 필드 주입으로 정정, `FORBIDDEN_WORKFLOW_EXECUTABLE_PATTERNS` 각 항목을 parametrize) | 위 canonical stub의 raw YAML 텍스트에서, `environment_approval_key`/`self_hosted_runner_label` 두 surface는 실제 job-level YAML 키(`environment: prod`를 job 블록에 추가, `runs-on: ubuntu-latest`를 `runs-on: [self-hosted, foo]`로 치환)를 담은 뮤턴트로 만들고, 나머지 12개 surface는 `run: \|` 스칼라 자신의 마지막 줄 뒤에 그 surface를 대표하는 문자열을 담은 새 셸 줄 하나를 (들여쓰기까지 유지해) 추가한(예: `curl http://x`, `echo "${{ secrets.TOKEN }}"`) 뮤턴트로 만든다 — 두 경우 모두 실제로 파싱되고 실행되는 필드이지 YAML 주석이 아니다(DR-I3-MIN-01 "test parsable executable-field mutants rather than injecting dangerous text only into comments"). run 스칼라를 바꾼 경우 dedent 결과가 더 이상 pin과 일치하지 않으므로 `_m3_gate_denylist_scan_text`가 아무것도 제거하지 않고 뮤턴트 텍스트 전체를 그대로 반환하고, 그 반환 텍스트에서 해당 패턴이 매치하는지 확인한다 — 15개 surface 전부를 개별 mutant로 증명한다 |
| `test_workflow_dispatch_input_enable_m3_live_regression_defaults_false` | `workflow["on"]["workflow_dispatch"]["inputs"]["enable_m3_live_regression"]["default"] is False`이고 `["type"] == "boolean"` |
| `test_m4_assemble_check_step_uses_v2_checker_without_legacy_flags` | `m4-assemble`의 "Check M4 baseline state algebra" step `run:` 문자열에 `check_m4_baseline.py`는 있고 `--allow-legacy-v1`/`--expect-operational-blocked`/`--expect-hosted-`/`--expect-sha`/`--expect-run-`/`--expect-workflow-path`/`--expect-event`/`--require-identity-binding`는 없음(§4.7·§5.4 설계 그대로 워크플로가 구현했는지의 회귀 가드) |
| `test_workflow_job_set_is_exactly_five_hosted_jobs_plus_the_opt_in_stub` | `set(workflow["jobs"]) == {"python-tests", "frontend-tests", "container", "m43-deterministic", "m4-assemble", "m3-live-regression-gate"}` — 여섯 번째 job이 몰래 추가되는 것을 exact-set으로 차단 |
| `test_no_ordinary_job_needs_m3_live_regression_gate` | `workflow["jobs"]`의 모든 job(자기 자신 제외)에 대해 `"m3-live-regression-gate"`가 `needs`(문자열 또는 리스트 어느 형태든)에 나타나지 않음 — MAJ-03의 "ordinary job dependency closure에 다섯 개 hosted job만 있어야 한다"를 `m4-assemble.needs`만이 아니라 **모든** job에 대해 증명(기존 `test_m4_assemble_needs_all_four_hosted_producers`와 상호보완, 이 테스트는 대체하지 않고 그대로 둔다) |

### 7.4 `tests/unit/test_doc_audit_no_active_native_runner_procedure.py` — 신규 (DR-I1-MAJ-04)

§8.1의 기존 3-문자열 grep(`overall_release_ready=true|native_linux_release_ready=true|
m3_live_regression=PASS`)은 "정책 성공 주장"만 잡을 뿐, "실행 가능한
절차 지시문"(runner 등록 명령, environment approval 안내, live job 실행
지시, 라벨 없는 Ollama 안내)은 잡지 못한다 — `CI_Acceptance_Runbook.md`
가 바로 그런 실행 가능한 지시문을 담은 채로 `docs/` 트리에 남아 있던
것이 MAJ-04의 근본 원인이다. 이 파일은 그 간극을 메우는 자동화 테스트다.

**스캔 범위를 "runbook 모양 문서"로만 좁힌다 — `docs/**/*.md` 전체를
스캔하지 않는다.** REQ-005.3이 실제로 겨냥하는 것은 "runbook"이지 설계/
요구사항/추적 문서 일반이 아니다. 이 milestone 자신의 Design.md만 해도
"현재 문제"를 설명하려고 `runs-on: [self-hosted, ollama-m3]`를 그대로
인용한다(§5.1, §11.1) — `docs/**/*.md`를 통째로 스캔했다면 이 설계
문서 자신이 오탐(false positive)으로 걸렸을 것이고, README.md의 "##
테스트 방법" 절(§178행대)의 `RUN_LIVE_LLM_TESTS=1` 로컬 pytest 예시도
마찬가지로 걸렸을 것이다(둘 다 개발/분석 문맥이지 "지금 실행 가능한
운영 절차"가 아니다 — 실제 확인: `grep -rn` 결과 `docs/**/*.md` 전체
스캔에서 Design.md·README.md·m2/m3/m4.2/m4.3의 여러 process 문서가
`RUN_LIVE_LLM_TESTS=1`/`runs-on: [self-hosted`를 인용문으로 포함한다).
따라서 스캔 대상을 실제로 "운영자가 따라 할 수 있는 절차 문서"인
`docs/operations/**/*.md`와 파일명에 `Runbook`이 들어간 문서
(`docs/milestones/**/*Runbook*.md`)로만 한정한다 — 정확히 이 범위
안에서는(구현 시점 기준) `CI_Acceptance_Runbook.md`만 금지 패턴과
매치하고, `deployment_runbook.md`/`recovery_runbook.md`는 매치하지
않는다(사전 확인 완료). README.md의 REQ-005.4 development-only 라벨링은
§8.2가 별도로 처리하며, 이 자동화 테스트의 대상이 아니다(README는
runbook이 아니라 로컬 개발 문서이므로 REQ-005.3의 범위 밖).

```python
FORBIDDEN_ACTIVE_PROCEDURE_PATTERNS = (
    r"config\.sh --url",            # self-hosted runner 등록 명령
    r"--labels self-hosted",
    r"runs-on:\s*\[self-hosted",
    r"Required reviewers",          # environment approval 안내
    r"RUN_LIVE_LLM_TESTS=1",        # live job 실행 지시
    r"OLLAMA_BASE_URL=http",        # 라벨 없는 Ollama 엔드포인트 지시
)
SUPERSEDED_BANNER_MARKER = "SUPERSEDED / NON-EXECUTABLE HISTORICAL RECORD"
# 이 allowlist는 "배너가 있는 파일"만 예외로 허용한다 — 파일 이름만으로
# 예외를 주지 않는다(아래 테스트가 배너 부재 시 이 allowlist 자체가
# 무력화되는지도 별도로 검사한다).
ALLOWLISTED_HISTORICAL_FILES = frozenset({
    "docs/milestones/m4.1-configuration-observability/CI_Acceptance_Runbook.md",
})

def _scanned_doc_paths() -> list[Path]:
    # runbook-shaped documents only — see the scope rationale above.
    # Design/Requirement/Plan/Traceability/Stop_Report/Code_Review docs are
    # intentionally NOT scanned; they legitimately quote/discuss these exact
    # strings when explaining what was forbidden and why.
    return sorted({*REPO_ROOT.glob("docs/operations/**/*.md"),
                   *REPO_ROOT.glob("docs/milestones/**/*Runbook*.md")})

def test_no_active_native_runner_procedure_outside_banner_or_allowlist():
    for path in _scanned_doc_paths():
        text = path.read_text(encoding="utf-8")
        hits = [p for p in FORBIDDEN_ACTIVE_PROCEDURE_PATTERNS if re.search(p, text)]
        if not hits:
            continue
        rel = str(path.relative_to(REPO_ROOT))
        if rel in ALLOWLISTED_HISTORICAL_FILES and SUPERSEDED_BANNER_MARKER in text:
            continue  # 배너로 명시적으로 라벨링된 역사적 기록 — 허용
        pytest.fail(f"{rel}: unallowlisted active-procedure pattern(s) {hits}")

_RUNBOOK_REL_PATH = next(iter(ALLOWLISTED_HISTORICAL_FILES))

def _leading_blockquote(text: str) -> str:
    """파일 맨 앞에서 시작하는 연속된 Markdown blockquote 줄(`>`로 시작하는
    줄)만 모아 반환한다. `>`로 시작하지 않는 첫 줄(배너 뒤의 빈 줄 포함)에서
    멈춘다 — 배너가 몇 줄을 차지하든(§8.4의 배너는 12줄) 매직 넘버 없이
    배너 블록 전체를 정확히 잡아낸다."""
    lines = text.splitlines()
    quote_lines: list[str] = []
    for line in lines:
        if line.startswith(">"):
            quote_lines.append(line)
        else:
            break
    return "\n".join(quote_lines)

def test_ci_acceptance_runbook_has_superseded_banner_near_top():
    text = (REPO_ROOT / _RUNBOOK_REL_PATH).read_text(encoding="utf-8")
    banner = _leading_blockquote(text)
    assert banner, "배너 blockquote가 파일의 첫 내용이어야 한다"
    assert SUPERSEDED_BANNER_MARKER in banner
    assert "deployment_runbook.md" in banner  # 새 정상 절차로의 링크

def test_allowlist_without_banner_still_rejected():
    # 배너 텍스트를 제거한 사본에서는 allowlist가 더 이상 예외를 주지
    # 않는다는 것을 회귀로 증명한다(파일 이름만으로 영구 면제되지 않음).
    text = (REPO_ROOT / _RUNBOOK_REL_PATH).read_text(encoding="utf-8")
    stripped = text.replace(SUPERSEDED_BANNER_MARKER, "")
    hits = [p for p in FORBIDDEN_ACTIVE_PROCEDURE_PATTERNS if re.search(p, stripped)]
    assert hits  # 배너 없이는 원래 금지 패턴들이 여전히 매치돼야 함(사전 조건)
```

**Iteration 2 리뷰가 지적한 오류와 정정(DR-I2-MIN-01):** 이전 iteration의
두 번째 테스트는 `text.splitlines()[:10]`이라는 고정 줄 수로 "상단"을
정의했다. 그러나 §8.4가 실제로 삽입하는 배너는 12줄짜리 blockquote이고,
`deployment_runbook.md` 링크는 그 12번째 줄에 있다 — 배너를 파일 맨 위에
정확히 그대로 삽입해도 12번째 줄은 `[:10]` 슬라이스 밖이므로 이 assertion은
실패했다. 정정한 `test_ci_acceptance_runbook_has_superseded_banner_near_top`은
매직 넘버 대신 `_leading_blockquote`로 파일 맨 앞의 연속된 `>` 줄 전체를
동적으로 모아, 배너가 실제로 몇 줄을 차지하든(지금은 12줄, 향후 배너 문구가
늘거나 줄어도) 그 전체 블록 안에서 마커와 링크를 찾는다 — "파일의 첫 내용이
전부 하나의 blockquote이고 그 안에 마커와 링크가 모두 있다"는 것이 실제로
증명하려는 불변식이며, 그 불변식은 배너의 정확한 줄 수와 무관하다.

세 번째 테스트는 "allowlist 항목이라도 배너가 없으면 통과하지 못한다"는
불변식을 직접 증명한다 — 이 테스트가 없으면 "파일 이름이 allowlist에
있다"는 사실만으로 향후 배너가 실수로 삭제돼도 감사가 계속 조용히
통과하는 회귀를 잡지 못한다.

## 8. 문서/런북/UI 표면

### 8.1 이미 반영된 것 — 감사만

이 세션 시작 시점 `git status`가 이미 `docs/Problem.md`,
`docs/Roadmap.md`를 modified로 보고했고, 두 파일을 읽어 다음을 확인했다
(따라서 이 milestone에서 **재수정하지 않는다** — 편집 범위 밖 지시와도
일치):

- `docs/Roadmap.md` L192-268: "M4 Operational Acceptance Recovery — 정책
  변경 승인 / 구현 대기" 절이 이미 `hosted_release_ready`/
  `native_linux_release_ready`/`full_production_release_ready`/legacy
  `overall_release_ready`의 목표 의미를 정확히 서술한다.
- `docs/Problem.md` L132-140: 같은 정책이 "구현 전에는 기존 artifact의
  `M4.1_BLOCKED=true`, `overall_release_ready=false` 의미는 그대로 유지"
  라고 명시한다(§6이 설계하는 v1 호환 모드와 정합).

구현 phase 완료 시 재확인 명령(Traceability.md §5의 기존 관례 그대로):

```bash
python scripts/check_markdown_links.py
git diff --check
rg -n "overall_release_ready=true|native_linux_release_ready=true|m3_live_regression=PASS" \
  docs/milestones/m4-operational-acceptance-recovery docs/Roadmap.md docs/Problem.md
```

**DR-I1-MAJ-04 폐쇄 — 위 3-문자열 grep은 "정책 성공 주장"만 잡고
"실행 가능한 절차 지시문"은 잡지 못한다.** 이 grep을 대체하는 것이
아니라(정책 문구 감사로는 여전히 유효) 보완하는 것으로,
§7.4의 `tests/unit/test_doc_audit_no_active_native_runner_procedure.py`를
Traceability.md §5 재확인 절차에 추가한다:

```bash
pytest -q tests/unit/test_doc_audit_no_active_native_runner_procedure.py
```

이 pytest가 §8.4의 배너 삽입과 함께 "역사적 runbook이 현재 실행 가능한
절차로 오독될 수 없다"를 코드로 증명한다.

### 8.2 `README.md` — Ollama 안내에 development-only 라벨 추가

REQ-005.4 "Existing generic local-development Ollama instructions may
remain only when clearly labeled development-only." 현재 README.md는
"### 3. Ollama"(L107) 절 등에서 라벨 없이 Ollama 설치를 안내한다. `###
3. Ollama` 헤더 바로 앞에 삽입:

```markdown
> **참고 (development-only):** 아래 Ollama 안내는 로컬 개발 전용이다.
> 이 프로젝트가 채택한 hosted/OCI 지원 범위(`support_policy.adopted_scope
> = "HOSTED_OCI"`)에는 포함되지 않으며, native Linux/Ollama 사용은
> unsupported/best-effort이고 release SLA가 없다. 자세한 내용은
> [M4 운영 acceptance recovery 요구사항](docs/milestones/m4-operational-acceptance-recovery/Requirement.md)
> 참고.
```

README.md의 나머지 개발 환경 구성 절차(§64-282 상당, 로컬 오프라인 검증
명령, `RUN_LIVE_LLM_TESTS=1` 안내 등)는 이미 "개발"이라는 문맥에서만
등장하므로 추가 라벨이 필요 없다 — 이 삽입 하나가 REQ-005.4의 "clearly
labeled"를 만족하는 최소 변경이다(README의 나머지 구조·순서는 바꾸지
않는다).

### 8.3 런북 — `deployment_runbook.md`/`recovery_runbook.md`

**`docs/operations/deployment_runbook.md`**: 기존 L1-6 머리말은 이미
"Native Linux/Ollama execution is an operator's manual responsibility —
this cycle does not run it and does not certify it."라고 말한다. 이
문장 뒤에 한 문장을 추가해 REQ-005.3의 "unsupported/best-effort... no
release SLA" 문구를 명시적으로 포함시킨다:

```markdown
Applies to the `deploy/Dockerfile` `production` stage image and the
`INDEX_ROOT` canonical version layout introduced by M4.3 ([Design.md](../milestones/m4.3-artifact-deployment-safety/Design.md)
§7). Native Linux/Ollama execution is an operator's manual responsibility —
this cycle does not run it and does not certify it. Native Linux/Ollama use
is unsupported/best-effort with no release SLA; the certified deployment
target is the hosted Python/frontend service plus the OCI container image
described here (see [M4 Operational Acceptance Recovery Requirement.md](../milestones/m4-operational-acceptance-recovery/Requirement.md)).
```

기존 §6 "Release identity record" 표 뒤(현재 L79 다음)에 새 §6.1을 추가해
"artifact download plus v2 checker" 절차(REQ-005.3)를 문서화한다:

```markdown
## 6.1 Hosted/OCI baseline verification (pre-deployment)

**This is the only current, normative pre-deployment verification
procedure.** The historical M4.1 self-hosted/native-Ollama runbook
(`docs/milestones/m4.1-configuration-observability/CI_Acceptance_Runbook.md`)
is superseded and non-executable (see the banner at its top) — do not use it
to provision a runner, approve an environment, or gate a release.

Before trusting an `image_digest` for deployment, download the `m4-baseline`
artifact from the exact merge-SHA workflow run and verify it independently,
binding the check to that exact run's identity so a baseline copied from a
different run or SHA cannot pass:

```bash
gh run download <RUN_ID> -n m4-baseline -D <fresh-dir>
python scripts/check_m4_baseline.py --candidate <fresh-dir>/m4-baseline.json \
  --expect-hosted-release-ready --require-identity-binding \
  --expect-sha <MERGE_SHA> \
  --expect-run-id <RUN_ID> \
  --expect-run-attempt <RUN_ATTEMPT> \
  --expect-workflow-path .github/workflows/ci.yml \
  --expect-event push
```

`--require-identity-binding` makes all five `--expect-*` flags mandatory
(the CLI exits 2 if any is missing) — this is what turns "download from the
exact merge-SHA workflow run" from a human convention into a fail-closed
CLI contract (DR-I1-MAJ-02). The pre-merge fixture command in Plan.md §5
intentionally omits this flag (there is no real workflow run yet to bind
to); this is the only invocation in the design that sets it.

`<MERGE_SHA>` is the commit the branch-protected merge produced
(`git rev-parse origin/master` after the merge, matching Plan.md §6's
"exact merge SHA"). `<RUN_ATTEMPT>` is normally `1`; use the actual attempt
number from `gh run view <RUN_ID> --json runAttempt` if the run was manually
re-run. A non-zero exit means either the four deterministic producers did
not all pass on that run, the artifact's schema/algebra/provenance aliases
are inconsistent, or the artifact's declared identity does not match the
run being verified — do not deploy the associated `image_digest` in any of
these cases. This check reports only the narrow "hosted/OCI release ready"
claim, bound to the exact requested run; it never certifies native
Linux/Ollama operation (`native_linux_release_ready` and
`full_production_release_ready` are always `false` under the current
policy) and it does not re-verify original payload bytes (see §4.7's trust
boundary — that verification already happened inside the assembler at CI
time, before this artifact was uploaded).
```

**`docs/operations/recovery_runbook.md`**: 진단 표의 "Ollama outage" 행은
정당한 운영 진단 항목이므로 유지하되, 표 앞에 한 문장을 추가해 그 행이
인증된 릴리스 표면의 일부가 아님을 명시한다:

```markdown
Diagnosis and rollback/backup-restore procedures for the M4.3 canonical
index lifecycle ([Design.md](../milestones/m4.3-artifact-deployment-safety/Design.md) §7.6).
The certified deployment surface is hosted Python/frontend plus the OCI
container; the "Ollama outage" row below is a development/operator
diagnostic aid and does not imply Ollama is part of the release-readiness
claim (native Linux/Ollama remain `NOT_ADOPTED`).
```

### 8.4 역사적 M4.1 CI runbook — superseded 배너 삽입 (DR-I1-MAJ-04 폐쇄)

**이전 iteration의 오류 정정:** 이전 설계는 `CI_Acceptance_Runbook.md`를
"Plan.md §2 각주가 보존을 지시한 역사적 도구"로 분류해 편집 금지
처리했다. 그러나 그 각주가 실제로 지정하는 것은 `scripts/
ci_acceptance_contract.py`/`scripts/preflight_ollama.py`와 그 테스트뿐이다
— runbook markdown 문서 자체는 그 각주의 대상이 아니다. 그리고 이
runbook은 "이미 끝난 조사의 기록"이 아니라 **현재도 그대로 실행 가능한
것처럼 서술된 절차**(self-hosted runner 등록 명령, 외부 데이터
디렉터리 provisioning, GitHub Environment 생성/승인, live job 실행,
receipt 검증)를 담고 있다(`CI_Acceptance_Runbook.md`:132-138, 164-195,
323-402, 560-561). Requirement M4-OAR-REQ-005.3은 정확히 이 상태를
금지한다 — "runbooks MUST remove native runner provisioning and live
approval from the release checklist."

**해결: 본문은 그대로, 상단에 non-executable 배너만 추가한다.** 이는
REQ-006(과거 receipt/evidence 재작성 금지)과 충돌하지 않는다 — 배너는
새 문장 하나를 파일 맨 위에 삽입할 뿐, 기존 조사 기록·명령·출력을 단
한 글자도 바꾸거나 지우지 않는다(§7.4의 `test_allowlist_without_banner_still_rejected`가
"배너를 빼면 원래 텍스트에 금지 패턴이 여전히 존재한다"는 것으로 이
본문 무변경을 간접 증명한다). `CI_Acceptance_Runbook.md` 첫 줄
(`# CR-I3-MAJ-01 — CI 운영 증거 조사와 실행 가능한 계약`) 바로 앞에
삽입:

```markdown
> **SUPERSEDED / NON-EXECUTABLE HISTORICAL RECORD (as of 2026-08-15):**
> This runbook documents the M4.1 self-hosted/native-Ollama live
> regression path, which is `NOT_ADOPTED` under the M4 Operational
> Acceptance Recovery policy
> ([Requirement.md](../m4-operational-acceptance-recovery/Requirement.md)).
> The commands, checklists, and provisioning steps below are a historical
> record of what M4.1 built and verified at the time; they are **not** a
> current or executable release procedure. Do not provision a self-hosted
> runner, approve the `m3-live-regression` environment, or run the live
> job using this document. The only current, normative release-readiness
> procedure is
> [`docs/operations/deployment_runbook.md` §6.1](../../operations/deployment_runbook.md#61-hostedoci-baseline-verification-pre-deployment).
```

이 배너가 §7.4 `SUPERSEDED_BANNER_MARKER`("SUPERSEDED / NON-EXECUTABLE
HISTORICAL RECORD")와 정확히 일치하는 문자열을 포함해야 doc-audit
allowlist가 이 파일을 예외로 인정한다 — 배너 문구를 바꾸면 감사
테스트가 즉시 실패해 그 변경을 잡아낸다. 이 삽입 외에 `CI_Acceptance_Runbook.md`의
어떤 줄도 수정하지 않는다.

### 8.5 범위 밖 — 명시적으로 건드리지 않는 것

Plan.md §2 각주: "`scripts/ci_acceptance_contract.py`,
`scripts/preflight_ollama.py`, and their tests remain as historical
tooling unless dead-code removal is separately approved. They MUST NOT
appear in adopted release gates." 이 milestone은 그 재승인을 하지
않으므로:

- `scripts/ci_acceptance_contract.py`, `scripts/preflight_ollama.py`,
  `scripts/preflight_vectorstore.py`, `scripts/run_m4_regression_gate.py`
  — 코드 변경 없음. 이들은 `check_m4_baseline.py`/`assemble_m4_evidence.py`
  가 참조하지 않는 독립 도구이므로 v2 스키마 마이그레이션이 이들의 동작에
  영향을 주지 않는다.
- `scripts/run_m42_acceptance.py`(L271, L368), `scripts/run_m43_acceptance.py`
  (L158)의 `"M4.1_BLOCKED": True` 리터럴 — 이들은 M4.2/M4.3 **자체**
  acceptance report 스키마의 내부 필드이며 `check_m4_baseline.py`의
  `REQUIRED_TOP_KEYS_V1/V2`와 무관한 별도 스키마다. M4 baseline v2가
  `"M4.1_BLOCKED"=False`로 바뀌는 것과 이 두 스크립트가 자기 보고서에서
  `True`를 쓰는 것은 서로 다른 문서를 가리키므로 충돌하지 않는다 —
  변경하지 않는다.

### 8.6 UI/헬스 엔드포인트 — 무변경 근거

```bash
$ grep -rn "release_ready" src/ web/
(결과 없음)
```

REQ-005.2 "Health/readiness UI MUST NOT expose `hosted_release_ready`"는
현재 코드베이스에서 이미 참이다(위 grep이 증거). 이 milestone은 새로
이 필드를 노출하는 코드를 추가하지 않으므로 `src/`/`web/`에 대한 변경이
없다. 구현 phase는 위 grep을 재실행해 이 불변식이 유지됐는지 확인하는
것으로 충분하다(Traceability.md §5의 markdown-link 감사와 같은 자리에
추가할 수 있는 1줄 명령).

## 9. CLI 요약 표

| 명령 | 신규/변경 플래그 | 비고 |
|---|---|---|
| `scripts/assemble_m4_evidence.py` | 없음 | 출력이 항상 v2. exit code가 `hosted_release_ready` 참조로 변경(§3.3, 값은 기존과 수학적으로 동일) |
| `scripts/check_m4_baseline.py` | `--allow-legacy-v1`(신규), `--expect-hosted-release-ready`(신규), `--expect-hosted-not-ready`(신규), `--expect-operational-blocked`(잔존, v1 전용, 이제 잉여 assertion), `--expect-sha`/`--expect-run-id`/`--expect-run-attempt`/`--expect-workflow-path`/`--expect-event`(신규, v2 전용 신원 바인딩, DR-I1-MAJ-02), `--require-identity-binding`(신규, post-merge 전용, 다섯 신원 플래그를 필수로 만드는 opt-in) | §4.6 조합 규칙 5개, 위반 시 exit 2 |

## 10. 검증 절차 — Plan.md §5/§6 매핑

Plan.md §5 "Hosted pre-merge gate"의 마지막 두 명령은 이 설계가 만드는
정확한 인터페이스로 실행된다:

```bash
python scripts/check_m4_baseline.py --candidate <PASS_V2_JSON> --expect-hosted-release-ready
python scripts/check_m4_baseline.py --candidate <FAIL_V2_JSON> --expect-hosted-not-ready
```

`<PASS_V2_JSON>`/`<FAIL_V2_JSON>`은 §7.2의 `_valid_v2_candidate()` fixture를
각각 그대로 쓰거나(PASS) 네 producer 중 하나를 `MISSING`으로 바꾼 변형을
써서(FAIL) 준비한다 — `assemble_m4_evidence.py`를 실제 fresh-dir로 실행해
만들어도 동일한 결과를 재현할 수 있다(§7.1
`test_assemble_v2_hosted_release_ready_true_when_all_four_producers_ok`/
`..._false_when_any_producer_not_ok`가 그 재현 경로 자체를 테스트한다).

Plan.md §6 "Post-merge gate"가 요구하는 정확-merge-SHA 검증은 §8.3
"6.1 Hosted/OCI baseline verification"에 문서화한 절차(`gh run download`
+ `check_m4_baseline.py --expect-hosted-release-ready --require-identity-binding`
와 다섯 `--expect-sha`/`--expect-run-id`/`--expect-run-attempt`/
`--expect-workflow-path`/`--expect-event` 플래그, DR-I1-MAJ-02)로
재현한다. 기대 출력 필드 값은 Plan.md §6이 이미 나열한 것과 동일하다:

```text
deterministic_status=PASS
operational_status=NOT_ADOPTED
m3_live_regression=NOT_ADOPTED
m41_operational=NOT_ADOPTED
M4.1_BLOCKED=false
hosted_release_ready=true
native_linux_release_ready=false
full_production_release_ready=false
overall_release_ready=false
```

## 11. Rollback과 미래 재활성화

### 11.1 이전 iteration의 오류 정정과 rollback 구성요소 분리 (DR-I1-MAJ-05 폐쇄)

**이전 iteration의 오류:** 이전 설계는 workflow rollback을 "§5.1의 이
설계 이전 상태로 되돌린다"로 정의했다. 하지만 §5.1의 상태는 정확히
`runs-on: [self-hosted, ollama-m3]`이고 `if:`에 `push:master`가 포함된
job이다 — 그 상태로 되돌리면 **ordinary master push마다 다시 이 job이
스케줄되고**, self-hosted runner가 0대인 현재 상태(Stop_Report.md §1)에서
그 run은 다시 영원히 `queued`로 남는다. "재활성화가 아니다"라고 부르는
것은 효과를 바꾸지 못한다 — 이는 M4-OAR-REQ-004.1(ordinary run은 항상
terminal conclusion에 도달해야 한다)을 rollback 절차 자신이 위반하는
것이고, Plan.md §7 "do not enable the live job as a workaround"의 정확한
반례다.

**정정: rollback은 하나가 아니라 독립적인 두 구성요소다.**

- **(A) Schema/checker rollback**: `scripts/assemble_m4_evidence.py`/
  `scripts/check_m4_baseline.py`를 이 설계 이전 커밋으로 되돌린다 — v1
  스키마·algebra·`--expect-operational-blocked` 단독 플래그가 그대로
  복원된다.
- **(B) Workflow rollback**: `.github/workflows/ci.yml`의
  `m3-live-regression-gate`를 **§5.1의 예전 정의로는 절대 되돌리지
  않는다.** workflow rollback이 허용하는 결과는 다음 둘 중 하나뿐이다:
  (b-1) job을 완전히 삭제한 상태를 유지, 또는 (b-2) §5.3의 harmless
  hosted no-op stub(`workflow_dispatch` opt-in에서만 실행, checkout/
  secret/environment/self-hosted 없음)을 그대로 유지. 두 선택지 모두
  "ordinary push/PR에서 이 job이 스케줄되지 않는다"는 불변식을 지킨다.

(A)와 (B)는 서로 독립적으로 실행될 수 있다 — schema rollback이 필요한
실패와 workflow rollback이 필요한 실패는 서로 다른 원인이므로, 하나가
실패했다고 다른 하나까지 되돌릴 필요는 없다.

### 11.2 Rollback matrix

| 실패 시나리오 | (A) schema/checker | (B) workflow | ordinary push/PR 항상 terminal? | live/self-hosted 경로 실행 가능? |
|---|---|---|---|---|
| (a) schema-only 실패(v2 checker/assembler 버그) | 이전 커밋으로 되돌림(v1 복원) | 변경 없음 — §5.3 stub 유지 | Yes(§5.3 `if:`가 push/PR에서 항상 거짓) | No |
| (b) workflow-only 실패(YAML 문법/트리거 오류) | 변경 없음(v2 유지) | §5.3 stub으로 되돌리거나 job 삭제 — **§5.1 예전 정의로는 되돌리지 않음** | Yes | No |
| (c) 둘 다 실패 | 이전 커밋으로 되돌림(v1 복원) | §5.3 stub으로 되돌리거나 job 삭제 | Yes | No |

세 시나리오 모두에서 "ordinary push/PR은 항상 terminal conclusion에
도달한다"와 "self-hosted/live 실행 경로가 열리지 않는다"는 두 불변식이
깨지지 않는다 — schema rollback 여부와 무관하게 workflow는 §5.1 상태로
돌아가지 않는다는 것이 이 rollback 설계의 핵심이다.

### 11.3 테스트로 이 불변식을 증명

§7.3의 워크플로 계약 테스트(exact-shape·source-level·dependency-closure
검사 전부)는 **rollback 이후의 워크플로 파일에도 그대로 다시 적용해
합격해야 한다.** 즉 "workflow rollback이 성공했다"의 합격 기준은 새
테스트 스위트를 만드는 것이 아니라, §7.3의 기존 계약 테스트를 rollback
후 `.github/workflows/ci.yml`에 대해 재실행해 전부 PASS하는 것이다 —
예를 들어 `test_m3_live_regression_gate_has_no_self_hosted_or_environment`
가 rollback 후 YAML에서도 PASS해야 "§5.1 예전 정의를 복원하지 않았다"는
것이 코드로 증명된다.

### 11.4 과거 증거 보존

과거 v1 receipt와 M4.3 PASS 증거는 (A)/(B) 어느 rollback 조합과도
무관하게 그대로 보존된다 — 파일 자체를 건드리지 않는 설계이므로
rollback이 그것들을 훼손할 방법이 없다.

재활성화(native Linux/Ollama를 다시 adopted scope로 편입)는 이 milestone의
산출물이 아니다 — 새 정책 결정, 별도 요구사항/설계 리뷰, threat model,
소유된 native runner, 별도 구현을 요구하며, `NOT_ADOPTED`는 설정
토글이나 secret 값 변경만으로 `PASS`가 될 수 없다(Requirement §M4-OAR-REQ-004.4,
Stop_Report.md §4-§5). §5.3의 `workflow_dispatch` 입력은 그 미래
재활성화가 실제로 승인됐을 때 "실행 가능한 하나의 진입점을 어디에 만들지"
미리 문서화해 두는 역할만 하며, 오늘 이 입력을 true로 주는 것은 아무
live 코드도 실행하지 않는다(§5.3 step 5 그대로).

## 12. 구현자 체크리스트

- [ ] `scripts/assemble_m4_evidence.py` — §3.1 상수 추가, §3.2 `_build_baseline`
      교체, §3.3 `main()` exit code 필드 변경. §3.1a `_statement_source_slice`/
      `_top_level_statement_slices`/`audit_exact_allowed_delta`와 §3.1b
      `_source_preamble`/`audit_exact_allowed_delta_bytes`를 base revision과
      구현 완료 후 작업 트리 파일의 raw bytes에 대해 실행해 `audit_exact_allowed_delta_bytes`의
      반환값이 `[]`인지 확인(DR-I2-MAJ-02, DR-I3-MAJ-01/02, DR-I4-MAJ-01/02,
      DR-RC1-I1-MAJ-01, DR-RC1-I2-MAJ-01 — decorator가 있는 statement의
      슬라이스가 첫 decorator부터 시작하는지, 그리고 shebang/encoding
      cookie/BOM이 base와 byte-exact로 동일한지 모두 §7.1의 decorator-span
      테스트와 preamble byte/token-aware 테스트로 함께 확인).
- [ ] `scripts/check_m4_baseline.py` — §4.1 상수 재배치(+ `WORKFLOW_RUN_KEYS`),
      §4.2 공유 헬퍼 추출, §4.3 `_check_v2` 신설(신원 바인딩 + alias
      재계산 포함, DR-I1-MAJ-02), §4.4 `_check_v1_legacy`(기존 `check()`
      본문 이동 + 무조건 frozen-blocked 강제 블록 추가, DR-I1-MAJ-01),
      §4.5 디스패처(신원 kwarg 전달), §4.6 CLI(다섯 신원 플래그 +
      `--require-identity-binding`, 다섯 조합 규칙).
- [ ] `.github/workflows/ci.yml` — §5.2 input, §5.3 job 재정의, §5.4 checker
      호출 인자(변경 없음 — 신원 플래그는 `m4-assemble` step이 아니라
      §8.3 §6.1 post-merge 절차에서만 쓰인다).
- [ ] `tests/unit/test_assemble_m4_evidence.py` — §7.1 표의 53개 테스트 추가
      (11개 v2 assembler 테스트 + whole-file allowed-delta 오라클 양성 2개
      + 오라클 negative mutant 22개(리뷰된 바이패스 8개 — import 재바인딩/
      class shadow/for-loop 대상/`with` 별칭/async def shadow/exception
      별칭/named expression/duplicate rebinding, 무관 import/실행문/함수/
      `assemble`/`main` non-exit 편집 5개, in-place whitespace mutation
      4-way parametrize 1개, main/`_build_baseline` 미변경·임의재작성·
      삽입위치오류·부분삽입·statement삭제·파싱불가 8개) + decorator-span
      뮤턴트 6개(DR-RC1-I1-MAJ-01 — `assemble` 대상 추가/제거/수정/재정렬
      4-way parametrize 1개, 비-pin 보호 함수 일반화 2-way parametrize 1개,
      decorated class/async 합성 케이스 2개, `_statement_source_slice`
      단위 검증 1개, comment/blank-line 무해성 양성 1개) + preamble
      byte/token-aware 경계 뮤턴트 12개(DR-RC1-I2-MAJ-01 — pinned-preamble
      단위 검증 1개, bytes 진입점 실제-파일 양성 1개, shebang 수정/삭제/삽입
      3개, encoding cookie 삽입(non-ASCII semantic reproduction 포함)/수정/
      삭제 3개, BOM 삽입 1개, BOM+cookie 상충 fail-closed 1개, BOM-불변 양성
      1개, cookie-아닌 선행 주석 무해성 양성 1개)), 기존 6개 무변경 확인.
- [ ] `tests/unit/test_check_m4_baseline.py` — §7.2 표의 테스트 전체 추가/
      이름변경: v1 frozen-blocked mutant 7개, v1-legacy 이름변경 1개,
      v2 신원/alias mutant 15개, v2 기존 계열 테스트, CLI 조합규칙 테스트
      (require-identity-binding 관련 포함).
- [ ] `tests/unit/test_ci_workflow_contract.py` — §7.3 표대로 대체 1개 +
      exact-shape/source-level/dependency-closure 신규 테스트(들여쓰기를
      실제로 dedent해 `M3_GATE_PINNED_RUN_SCRIPT`를 제거하는 정밀 denylist
      와 그 제거 자체를 assert하는 양성 테스트, canonical stub 양성
      테스트, 파싱 가능한 실행 필드로 주입하는 surface별 음성 mutant
      테스트 포함, DR-I2-MAJ-01, DR-I3-MIN-01), 기존 3개
      (`test_m4_assemble_needs_all_four_hosted_producers` 등) 무변경 확인.
- [ ] `tests/unit/test_doc_audit_no_active_native_runner_procedure.py`(신규) —
      §7.4 표대로 3개 테스트 작성.
- [ ] `README.md` — §8.2 콜아웃 1개 삽입.
- [ ] `docs/operations/deployment_runbook.md` — §8.3 문장 추가 + §6.1 신설
      (`--require-identity-binding` + 다섯 신원 플래그 포함).
- [ ] `docs/operations/recovery_runbook.md` — §8.3 문장 추가.
- [ ] `docs/milestones/m4.1-configuration-observability/CI_Acceptance_Runbook.md` —
      §8.4의 superseded/non-executable 배너를 파일 맨 위에 삽입(본문은
      그대로 유지, DR-I1-MAJ-04).
- [ ] `docs/Roadmap.md`/`docs/Problem.md` — 재수정 없음(§8.1 감사만).
- [ ] Plan.md §5 hosted pre-merge gate 전체 명령 실행(신원 플래그 없이,
      §10 그대로), §10 fixture 검증.
- [ ] rollback dry-run — §11.2 matrix의 (a)/(b)/(c) 세 시나리오 각각에서
      §7.3 워크플로 계약 테스트를 rollback 후 YAML에 대해 재실행해 전부
      PASS하는지 확인(§11.3).
- [ ] 저장소 branch protection에 §5.5 다섯 체크 등록(리포지토리 설정,
      diff 없음 — Traceability.md에 완료 기록만).

## 13. Design_Review_Iteration_1 폐쇄 matrix

이 표는 두 번째 fresh 리뷰가 각 finding을 즉시 확인할 수 있도록, 여섯
finding 각각을 "정확히 무엇이 바뀌었고 어떤 테스트/명령이 그것을
증명하는가"에 매핑한다. **DR-I1-MAJ-03과 DR-I1-MIN-01의 "증명하는
테스트/명령" 열은 Design_Review_Iteration_2가 각각 DR-I2-MAJ-01/
DR-I2-MAJ-02로, Design_Review_Iteration_3가 DR-I2-MAJ-01을 정상
폐쇄로 확인하면서 동시에 DR-I2-MAJ-02를 DR-I3-MAJ-01/DR-I3-MAJ-02로
재오픈한 뒤 이 iteration(4)이 다시 고친 최신 메커니즘을 반영한다 —
§14/§15가 그 재오픈·재폐쇄 내역을 각각 별도로 기록한다.**

| Finding | 핵심 결함 | 이 iteration의 수정 | 증명하는 설계 절 | 증명하는 테스트/명령 |
|---|---|---|---|---|
| DR-I1-MAJ-01 | `--allow-legacy-v1`만으로는 frozen-blocked이 강제되지 않음(`expect_operational_blocked`가 있어야만 강제) | `_check_v1_legacy`에 무조건(플래그 무관) 다섯 값 강제 블록 추가, producer 검증보다 먼저 실행 | §4.4, §0.3-3, §6 | §7.2 `test_v1_legacy_rejects_*`(7개, `allow_legacy_v1=True`만 주고 `expect_operational_blocked` 생략) |
| DR-I1-MAJ-02 | checker가 `git_sha`/`workflow_run`을 검증하지 않고 `image_digest`/`m43_deterministic_receipt_sha256`를 재계산하지 않음 | `_check_v2`에 신원 타입/값 검증 + 두 alias 재계산 추가, CLI에 다섯 `--expect-*` + `--require-identity-binding` 추가, 트러스트 경계 명시 | §4.3, §4.6, §4.7, §8.3 §6.1 | §7.2 `test_v2_rejects_git_sha_*`/`test_v2_rejects_workflow_run_*`/`test_v2_identity_flags_reject_cross_*`/`test_v2_rejects_image_digest_alias_tampered_*`/`test_v2_rejects_m43_receipt_sha_alias_tampered_*`/`test_main_cli_require_identity_binding_*`; `check_m4_baseline.py --expect-hosted-release-ready --require-identity-binding --expect-sha ... --expect-run-id ... --expect-run-attempt ... --expect-workflow-path ... --expect-event ...`(§8.3 §6.1) |
| DR-I1-MAJ-03 | 워크플로 계약 테스트가 두 번째 `run:` step, `${{ secrets`, 추가 `uses` 등을 잡지 못함 | exact job/step key-set 검사, exact 허용 스크립트 문자열 일치, exact-pin된 run 필드를 제외하고 정밀한 위험-실행-형태만 잡는 source-level denylist, job-set/`needs` closure 검사 추가 | §7.3 | `test_m3_live_regression_gate_exact_job_key_set`/`..._exactly_one_step_with_exact_step_key_set`/`..._step_run_exact_allowlisted_script`/`..._source_denylist_has_no_forbidden_executable_surfaces`/`..._canonical_stub_literal_satisfies_full_contract_suite`/`..._source_denylist_rejects_each_forbidden_executable_surface`/`test_workflow_job_set_is_exactly_five_hosted_jobs_plus_the_opt_in_stub`/`test_no_ordinary_job_needs_m3_live_regression_gate` |
| DR-I1-MAJ-04 | `CI_Acceptance_Runbook.md`가 편집 금지로 분류된 채 실행 가능한 절차처럼 남아 있었고 감사 grep은 정책 문구 3개만 봄 | 본문 무변경 + superseded 배너 삽입, `deployment_runbook.md` §6.1을 "유일한 정상 절차"로 명시, runbook-scoped 자동화 doc-audit 테스트 신설 | §8.4, §8.3 §6.1, §7.4 | `pytest tests/unit/test_doc_audit_no_active_native_runner_procedure.py`(3개: allowlist-outside 거부, 배너 존재, 배너-없으면-거부 회귀) |
| DR-I1-MAJ-05 | rollback이 §5.1의 self-hosted/ordinary-push 상태로 workflow를 되돌림 | rollback을 (A) schema/checker / (B) workflow로 분리, (B)는 §5.1로 절대 되돌아가지 않음(stub 유지 또는 삭제만), (a)/(b)/(c) matrix | §11.1, §11.2, §11.3 | rollback 후 §7.3 전체 테스트 스위트 재실행(특히 `test_m3_live_regression_gate_has_no_self_hosted_or_environment`) |
| DR-I1-MIN-01 | "byte-for-byte 불변" 수사가 output-shape 테스트만으로 뒷받침됨 | 수사 표현 제거, base revision 전체 top-level statement 시퀀스를 대상으로 하는 `audit_exact_allowed_delta`(whole-file AST/source-slice 비교)로 대체(iteration 5에서 이름-기반 목록 자체를 폐기하고 최종 형태로 대체, §16 DR-I4-MAJ-01/02 참고) | §3.1a | §7.1 `test_audit_exact_allowed_delta_positive_actual_v2_file`(양성) + (iteration 5 최신 메커니즘: `test_audit_exact_allowed_delta_rejects_*` 전체, §16 DR-I4-MAJ-01/02 참고) + 기존 6개 회귀 테스트 |

이 표의 모든 행은 §12 구현자 체크리스트의 해당 항목과 1:1로 대응한다 —
구현 phase가 체크리스트를 완료하면 이 표의 모든 finding이 동시에
폐쇄된다.

## 14. Design_Review_Iteration_2 폐쇄 matrix

§13이 iteration 1의 다섯 MAJOR/MINOR 폐쇄를 기록한 것과 같은 형식으로,
이 표는 Design_Review_Iteration_2가 발견한 세 finding(둘은 iteration 1
폐쇄를 재오픈한 것, 하나는 신규 MINOR) 각각을 이 iteration이 정확히
어떻게 고쳤는지 매핑한다.

| Finding | 핵심 결함 | 이 iteration의 수정 | 증명하는 설계 절 | 증명하는 테스트/명령 |
|---|---|---|---|---|
| DR-I2-MAJ-01(DR-I1-MAJ-03 재오픈) | raw source denylist가 `self-hosted`/`environment:`를 위치 무관 bare substring으로 검사해, §5.3 stub 자신의 안전-설명 부정문("no self-hosted runner")이 그 금지 목록에 걸려 **요구되는 유일한 정답 구현이 자기 자신의 계약을 통과할 수 없는** 모순이 있었음 | exact-pin된 `run` 필드를 denylist 스캔 대상에서 제외(`M3_GATE_PINNED_RUN_SCRIPT`를 블록 텍스트에서 제거한 뒤에만 스캔), bare 단어 대신 위험한 실행 형태만 정밀 매치하는 `FORBIDDEN_WORKFLOW_EXECUTABLE_PATTERNS`로 교체, §5.3 stub 리터럴이 전체 계약을 통과한다는 양성 테스트와 surface별 음성 mutant 테스트 추가 | §7.3 | `test_m3_live_regression_gate_source_denylist_has_no_forbidden_executable_surfaces`(15-way parametrize) + `test_m3_live_regression_gate_canonical_stub_literal_satisfies_full_contract_suite`(양성) + `test_m3_live_regression_gate_source_denylist_rejects_each_forbidden_executable_surface`(15-way parametrize, 음성) |
| DR-I2-MAJ-02(DR-I1-MIN-01 재오픈) | 보호 구간을 base 연속 줄 범위(L35-327)로 정의했는데 그 범위가 §3.2가 요구하는 `_build_baseline` 교체(L276-312)를 포함해 자기모순이었고, `grep -E '^@@'`는 판정 로직이 아니라 사람이 읽는 출력이었음 | 줄 범위를 폐기하고 이름 붙은 심볼을 슬라이스 비교하는 중간 단계(iteration 2~4)를 거쳐, iteration 5에서 이름 자체를 폐기하고 base 전체 top-level statement 시퀀스를 pin된 세 델타로만 변형해 비교하는 `audit_exact_allowed_delta` 도입. `_build_baseline`/`main()`은 그 세 델타 중 두 개로 명시적으로 pin되므로 예외 규칙 없이 정확히 그 자리에서만 교체가 허용된다 | §3.1a | §7.1 `test_audit_exact_allowed_delta_positive_actual_v2_file`(양성, 실제 구현물 대상) + (iteration 5에서 `test_audit_exact_allowed_delta_rejects_*` 전체로 최종 정정, §16 DR-I4-MAJ-01/02 참고) |
| DR-I2-MIN-01 | `test_ci_acceptance_runbook_has_superseded_banner_near_top`이 `splitlines()[:10]` 고정 줄 수를 검사했는데, §8.4 배너는 12줄이고 `deployment_runbook.md` 링크가 12번째 줄에 있어 그 슬라이스 밖으로 벗어남 | 매직 넘버 대신 `_leading_blockquote`로 파일 맨 앞의 연속된 `>` blockquote 전체를 동적으로 모아 그 안에서 마커와 링크를 확인 | §7.4, §8.4 | `test_ci_acceptance_runbook_has_superseded_banner_near_top`(정정된 구현) |

이 표의 모든 행 역시 §12 구현자 체크리스트, §13(원래 DR-I1-MAJ-03/
DR-I1-MIN-01 행의 갱신된 "증명하는 테스트/명령" 열)과 1:1로 대응한다.
DR-I2-MAJ-02 행이 참조하던 음성 테스트는 이 iteration에서 다시 결함이
발견되어 교체됐다 — §15가 그 재오픈·재폐쇄 내역을 기록한다.

## 15. Design_Review_Iteration_3 폐쇄 matrix

§13/§14와 같은 형식으로, 이 표는 Design_Review_Iteration_3가 발견한 세
finding(둘은 iteration 1→2를 거쳐 온 보호 심볼 감사 결함의 연속, 하나는
raw YAML 스칼라 제외 메커니즘의 신규 MINOR) 각각을 이 iteration(4)이
정확히 어떻게 고쳤는지 매핑한다.

| Finding | 핵심 결함 | 이 iteration의 수정 | 증명하는 설계 절 | 증명하는 테스트/명령 |
|---|---|---|---|---|
| DR-I3-MAJ-01(DR-I2-MAJ-02/DR-I1-MIN-01 계열의 지속) | 뮤턴트 구성 `base_source.replace(segment, segment + " ", 1)`이 추가한 공백을 AST 노드의 `end_col_offset` **뒤**에 놓아, 재파싱된 노드의 `ast.get_source_segment`가 base와 문자 그대로 동일하게 남았다 — 26개 보호 심볼 전부에서 감사가 위반을 하나도 못 잡음(`WHITESPACE_MUTANT_MISSED`) | 공백을 슬라이스 **안쪽**(첫 `=` 또는 함수의 첫 `(` 바로 뒤)에 삽입하는 방식은 유지하되, iteration 5에서 대상을 "26개 보호 심볼"에서 "base의 임의 top-level statement"로 일반화한 `_in_place_whitespace_mutation`으로 교체(§16 DR-I4-MAJ-01/02 참고) | §3.1a, §7.1 | §7.1 `test_audit_exact_allowed_delta_rejects_in_place_whitespace_mutation`(4-way parametrize, import/상수/함수/제어문 대표 statement) — 각 서브케이스가 `top_level_statement_changed:index=...`를 정확히 반환하는지 확인 |
| DR-I3-MAJ-02(DR-I2-MAJ-02/DR-I1-MIN-01 계열의 지속) | `_module_source_segment`(단수형)가 이름이 일치하는 첫 top-level statement에서 바로 `return`해, base 끝에 같은 이름의 두 번째 대입/`def`를 추가해도 첫 슬라이스는 그대로라 감사를 통과했다 — 이름은 보호됐지만 Python의 "마지막 바인딩이 이긴다" 규칙 때문에 실제 런타임 값은 보호되지 않는 fail-open이었음 | iteration 3은 이름이 일치하는 모든 top-level 바인딩을 복수형으로 수집해 개수를 세는 방식으로 고쳤으나(0개/2개 이상을 별도 위반으로 식별), 이름 열거 자체가 DR-I4-MAJ-01/02로 재오픈됨에 따라 iteration 5는 "이름이 몇 번 바인딩됐는가"가 아니라 "base에 없던 top-level statement가 나타났는가"라는 일반 규칙으로 흡수 — 재바인딩 시도는 pin된 세 델타 자리가 아닌 곳에 나타나는 순간 종류·이름과 무관하게 거부된다 | §3.1a, §7.1 | §7.1 `test_audit_exact_allowed_delta_rejects_duplicate_assignment_rebinding`(`REQUIRED_PRODUCERS = ("attacker-job",)` 재현) + `test_audit_exact_allowed_delta_rejects_duplicate_function_rebinding` + `test_audit_exact_allowed_delta_rejects_missing_statement_removed_from_base` + `test_audit_exact_allowed_delta_positive_actual_v2_file`(양성, 실제 v2 `_build_baseline`/신규 상수/`main()` exit 줄 변경이 그대로 통과함을 증명) |
| DR-I3-MIN-01 | `_m3_gate_denylist_scan_text`가 `block.replace(M3_GATE_PINNED_RUN_SCRIPT, "", 1)`을 썼는데, raw YAML `run: \|` block-scalar의 각 줄은 파싱된 스칼라 문자열에 없는 들여쓰기를 앞에 달고 있어 `replace`가 절대 일치하지 않고 아무것도 지우지 못했음 — canonical stub은 우연히 여전히 통과했지만 "exact-pin된 run 필드는 스캔에서 제외된다"는 설계 근거 자체가 거짓이었음 | `run: \|` header의 들여쓰기 폭을 기준으로 스칼라 본문 줄만 동적으로 수집해 `textwrap.dedent`한 뒤에만 pin과 비교하고, 일치할 때만 그 원본(들여쓰기 포함) 줄들을 실제로 제거 | §7.3 | §7.3 `test_m3_gate_denylist_scan_text_actually_removes_pinned_scalar`(양성, pin된 각 줄이 반환 텍스트에 없음을 직접 assert) + `test_m3_live_regression_gate_source_denylist_rejects_each_forbidden_executable_surface`(15-way parametrize, 파싱 가능한 job-level 키 또는 run 스칼라 추가 줄로 정정된 mutant) |

이 표의 모든 행은 §12 구현자 체크리스트의 해당 항목과 1:1로 대응한다 —
구현 phase가 체크리스트를 완료하면 이 표의 모든 finding이 동시에
폐쇄된다.

## 16. Design_Review_Iteration_4 폐쇄 matrix

§13~§15와 같은 형식으로, 이 표는 Design_Review_Iteration_4가 발견한 두
MAJOR(iteration 2~4를 거쳐 온 §3.1a 이름-기반 allowlist 메커니즘 자체의
구조적 결함)와 한 MINOR(§5.3의 불가능한 broad grep 감사 명령)를 이
iteration(5)이 정확히 어떻게 고쳤는지 매핑한다.

| Finding | 핵심 결함 | 이 iteration의 수정 | 증명하는 설계 절 | 증명하는 테스트/명령 |
|---|---|---|---|---|
| DR-I4-MAJ-01 | §3.1a가 "모든 top-level 바인딩"을 감사한다고 서술했지만 구현은 `FunctionDef`/`Assign`/`AnnAssign`만 인식했다 — `from attacker import REQUIRED_PRODUCERS`, `class _evaluate_producer: ...`, `for REQUIRED_PRODUCERS in ...:` 대상, `with ... as _settings_hash:` 별칭, `AsyncFunctionDef`로 보호 심볼을 재바인딩하면 `_module_source_segments`가 애초에 그 statement를 찾지 않아 `audit_protected_symbols`가 위반을 하나도 보고하지 않았다 | 이름/바인딩-종류 열거를 전부 폐기 — base 소스의 `ast.parse(...).body`가 만드는 top-level statement 시퀀스를 statement 종류 구분 없이 통째로 다루는 `_top_level_statement_slices`/`audit_exact_allowed_delta`로 교체. import·class·동기/비동기 def·loop/컴프리헨션 대상·`with`/`except` 별칭·named expression·그 밖의 어떤 top-level statement든 pin된 세 델타 자리가 아닌 곳에 나타나면 종류·이름과 무관하게 "미승인 추가 statement"로 거부된다 | §3.1a | §7.1 `test_audit_exact_allowed_delta_rejects_import_rebinding_of_protected_name` + `..._rejects_class_shadow_of_protected_name` + `..._rejects_for_loop_target_rebinding` + `..._rejects_with_alias_rebinding` + `..._rejects_async_function_shadow` + `..._rejects_exception_alias_rebinding` + `..._rejects_top_level_named_expression_statement` + `..._rejects_duplicate_assignment_rebinding` + `..._rejects_duplicate_function_rebinding` |
| DR-I4-MAJ-02 | positive 감사가 26개 named 보호 심볼만 비교해 "포함"만 증명했을 뿐 "배제"는 증명하지 못했다 — `import`, 모듈 최상위 실행 statement, `assemble`, `main()`의 exit 줄이 아닌 나머지, 새로 추가된 이름 없는/이름 있는 statement는 어느 쪽도 비교 대상이 아니었으므로, 실제 v2 patch에 임의의 추가 변경을 얹은 patch까지 감사를 통과시킬 수 있었다 | base-to-v2 전체 AST 델타를 기계적으로 비교하는 단일 오라클 도입 — pin된 신규 v2 상수 5개 삽입, `_build_baseline` 전체 교체, `main()` 전체 교체(§3.3의 exit-expression 한 줄 차이) 세 가지만 허용하고, `assemble`을 포함해 그 세 델타에 해당하지 않는 모든 base statement는 문자 그대로 동일해야 하며, 그 세 델타 자리가 아닌 곳의 어떤 추가 statement도 거부된다 | §3.1a | §7.1 `test_audit_exact_allowed_delta_positive_actual_v2_file`(양성) + `..._positive_synthetic_fixture`(양성) + `..._rejects_new_import_statement` + `..._rejects_new_executable_statement` + `..._rejects_new_unrelated_function` + `..._rejects_assemble_modified` + `..._rejects_main_non_exit_line_modified` + `..._rejects_main_left_as_base_v1` + `..._rejects_build_baseline_arbitrary_rewrite` + `..._rejects_build_baseline_left_as_base_v1` + `..._rejects_new_constants_inserted_at_wrong_location` + `..._rejects_partial_pinned_constants_block` + `..._rejects_missing_statement_removed_from_base` + `..._rejects_current_source_with_syntax_error` |
| DR-I4-MIN-01 | §5.3 설계 근거 항목 2가 "`grep -c \"self-hosted\" .github/workflows/ci.yml`이 0이 되는 것이 감사 명령"이라고 서술했지만, canonical stub 자신의 안전-설명 echo("no self-hosted runner")와 이 문서(§5.1, §11.1)가 인용하는 이전 상태(`runs-on: [self-hosted, ollama-m3]`) 모두 이 broad substring과 매치하므로, 요구되는 유일한 정답 구현조차 이 grep을 통과할 수 없는 impossible한 audit 명령이었다 | §5.3 설계 근거 항목 2에서 broad grep 문구를 제거하고, 이미 §7.3에 정의된 파싱된 구조 검사(`job["runs-on"] == "ubuntu-latest"` 정확 일치)와 실행-표면 정밀 정규식(`self_hosted_runner_label` 패턴, exact-pin된 run 필드는 스캔에서 제외)을 감사 근거로 명시 | §5.3 | §7.3 `test_m3_live_regression_gate_has_no_self_hosted_or_environment` + `test_m3_live_regression_gate_source_denylist_has_no_forbidden_executable_surfaces`(`self_hosted_runner_label` 서브케이스) — 두 테스트 모두 §7.3에 이미 정의돼 있으며 이 항목은 새 테스트를 요구하지 않는다 |

이 표의 모든 행은 §12 구현자 체크리스트의 해당 항목과 1:1로 대응한다 —
구현 phase가 체크리스트를 완료하면 이 표의 모든 finding이 동시에
폐쇄된다. §13~§15가 기록한 DR-I1~DR-I3의 폐쇄 판정은 이 iteration에서도
재오픈되지 않는다 — 새 whole-file 오라클은 이전 메커니즘이 잡던 모든
뮤턴트(in-span whitespace 삽입, 중복 바인딩, 심볼 제거)를 이름 구분 없는
일반 규칙의 특수 사례로 포함하는 상위집합이기 때문이다.

## 17. Design_Review_Recovery_Cycle_1_Iteration_1 폐쇄 matrix

§13~§16과 같은 형식으로, 이 표는
[Design_Review_Recovery_Cycle_1_Iteration_1.md](Design_Review_Recovery_Cycle_1_Iteration_1.md)가
발견한 한 MAJOR(§3.1a whole-file 오라클이 소비하는 슬라이스 생성 자체가
decorator를 비교 대상 밖에 남기는 결함)를 이 개정(Recovery Cycle 1,
Iteration 2)이 정확히 어떻게 고쳤는지 매핑한다.

| Finding | 핵심 결함 | 이 개정의 수정 | 증명하는 설계 절 | 증명하는 테스트/명령 |
|---|---|---|---|---|
| DR-RC1-I1-MAJ-01 | decorated `ClassDef`/`FunctionDef`/`AsyncFunctionDef`의 `lineno`와 `ast.get_source_segment`는 `class`/`def`/`async def` 토큰부터 시작해, `_top_level_statement_slices`가 반환하는 슬라이스가 decorator 줄을 포함하지 않았다 — `assemble` 바로 앞에 `@staticmethod`나 `@(lambda f: (lambda *a, **k: {}))` 같은 decorator를 추가해 런타임 동작을 바꿔도 반환되는 슬라이스 목록이 base와 문자 그대로 동일해 `audit_exact_allowed_delta`가 `[]`(통과)를 반환했다. `assemble`은 DR-I4-MAJ-02가 명시적으로 "한 글자도 못 바꾼다"고 요구한 pin되지 않은 statement였으므로, 이 gap은 그 요구를 decorator 경로로 직접 우회했다 | `_statement_source_slice` 헬퍼를 신설해, decorator가 하나 이상 있는 `ClassDef`/`FunctionDef`/`AsyncFunctionDef`만 슬라이스 시작 좌표를 노드 자신이 아니라 첫 decorator가 위치한 줄의 `@` 문자(좌표 산술을 가정하지 않고 역탐색으로 구함)로 확장하고, 끝 좌표는 노드 자신의 `end_lineno`/`end_col_offset`을 그대로 쓴다. decorator가 없는 statement와 세 노드 종류가 아닌 모든 statement는 기존과 완전히 동일하게 `ast.get_source_segment`를 그대로 쓴다. `_top_level_statement_slices`는 이 헬퍼를 통해서만 슬라이스를 만들도록 교체됐다 — 비교 알고리즘(`audit_exact_allowed_delta`, `_first_divergence_violation`)과 세 pin된 델타 리터럴은 전혀 바뀌지 않는다(어느 쪽에도 decorator가 없으므로) | §3.1a | §7.1 `test_audit_exact_allowed_delta_rejects_decorator_mutations_on_assemble`(4-way parametrize: 추가/제거/수정/재정렬, `assemble` 자신을 대상으로 한 리뷰의 원 재현) + `..._rejects_decorator_added_to_other_protected_function`(2-way parametrize, 비-pin 함수 일반화) + `..._rejects_decorator_added_to_synthetic_class`(`ClassDef`) + `..._rejects_decorator_added_to_synthetic_async_function`(`AsyncFunctionDef`) + `test_statement_source_slice_decorated_function_starts_at_at_symbol_and_includes_all_decorators`(슬라이스 헬퍼 단위 검증) + `test_audit_exact_allowed_delta_comment_and_blank_line_insertions_between_statements_are_invisible`(양성 — comment/blank line이 statement를 감추거나 만들어내지 못함을 확인) + 회귀로 그대로 재실행되는 `test_audit_exact_allowed_delta_positive_actual_v2_file`/`..._positive_synthetic_fixture`(base/v2 어느 쪽도 decorator가 없으므로 슬라이스 로직 교체 후에도 `== []` 유지) + `test_audit_exact_allowed_delta_rejects_current_source_with_syntax_error`(회귀 — 파싱 불가능 입력은 여전히 fail-closed) |

이 표의 행은 §12 구현자 체크리스트의 갱신된 항목과 1:1로 대응한다.
§13~§16이 기록한 DR-I1~DR-I4의 폐쇄 판정은 이 개정에서도 재오픈되지
않는다 — decorator span 확장은 `_top_level_statement_slices`가 만드는
슬라이스의 **시작 좌표만**, 그것도 decorator가 있는 `ClassDef`/
`FunctionDef`/`AsyncFunctionDef` 세 노드 종류에 한해 넓히며, 나머지 모든
statement 종류의 슬라이스 계산, whole-file 시퀀스 비교 알고리즘, 세
pin된 델타(`_build_baseline`/`main`/신규 상수 5개 — 어느 쪽도 decorator가
없음)는 전혀 손대지 않는다. 따라서 DR-I4가 닫은 "이름/바인딩-종류 무관
default-deny" 성질은 이 개정에서 좁아지지 않고, decorator라는 이전에
슬라이스 경계 밖에 있던 실행 가능 접두부까지 포함하는 상위집합으로
넓어진다.

## 18. Design_Review_Recovery_Cycle_1_Iteration_2 폐쇄 matrix

§13~§17과 같은 형식으로, 이 표는
[Design_Review_Recovery_Cycle_1_Iteration_2.md](Design_Review_Recovery_Cycle_1_Iteration_2.md)가
발견한 한 MAJOR(§3.1a whole-file 오라클이 소비하는 `base_source`/
`current_source` 문자열이 shebang/encoding-cookie 실행·디코딩 경계를
비교 대상 밖에 남기는 결함)를 이 개정(Recovery Cycle 1, Iteration 3)이
정확히 어떻게 고쳤는지 매핑한다.

| Finding | 핵심 결함 | 이 개정의 수정 | 증명하는 설계 절 | 증명하는 테스트/명령 |
|---|---|---|---|---|
| DR-RC1-I2-MAJ-01 | `_top_level_statement_slices`는 `ast.parse(source).body`의 모듈 docstring 노드부터 시작해, 그 이전의 shebang 줄과 PEP 263 encoding cookie 줄은 애초에 비교 대상 밖이었다. base 파일(shebang만 있고 cookie 없음, docstring에 non-ASCII em dash 포함)에 shebang을 수정/삭제하거나 `# coding: latin-1`을 삽입해도, `ast.parse(current_source)`는 이미 디코드된 `str`을 받으므로 cookie는 그 시점에는 그냥 텍스트일 뿐이고, 모듈 docstring 슬라이스는 (같은 방식으로 디코드된) base와 문자 그대로 동일하게 남아 `audit_exact_allowed_delta`가 `[]`(통과)를 반환했다 — 그러면서도 실제 CPython 실행에서는 그 cookie가 docstring을 mojibake로 바꾸고 `argparse` CLI 출력까지 바꿨다 | §3.1b에 `_source_preamble`/`audit_exact_allowed_delta_bytes` 두 함수를 신설. `base_source`/`current_source`를 `str`이 아니라 raw bytes로 읽어, 그 맨 앞의 shebang(파일의 실제 첫 두 바이트가 정확히 `#!`인 경우만)과 PEP 263 encoding cookie(shebang이 있으면 그 다음 줄, 없으면 첫 줄, `tokenize.detect_encoding`이 실제로 검사하는 범위 안에서만)와 UTF-8 BOM만 골라 byte-exact로 먼저 비교한다. 그 preamble이 다르면(수정/삭제/삽입 어느 방향이든) statement-시퀀스 비교를 실행하지도 않고 즉시 거부한다. preamble이 같으면 `tokenize.detect_encoding`이 그 bytes로부터 실제로 검출하는 encoding으로 전체를 디코드해 (한 글자도 바뀌지 않은) §3.1a `audit_exact_allowed_delta`에 그대로 넘긴다. BOM과 cookie가 상충하는(PEP 263/CPython 자신이 `SyntaxError`로 정의하는) 입력은 `current_source_encoding_conflict`로 명시적으로 fail-closed된다. cookie가 아닌 임의의 선행 주석까지 preamble로 과잉 확장하지 않으며, 이 정정이 "current가 base+세 델타와 byte-for-byte 동일함을 증명한다"고 주장하지도 않는다 — 정확히 (i) preamble byte-exact 동일성과 (ii) 그 위에서의 top-level statement 시퀀스 동일성 두 가지만 증명한다는 partition을 §3.1b가 명시한다 | §3.1b | §7.1 `test_source_preamble_matches_pinned_base_preamble_bytes` + `test_audit_exact_allowed_delta_bytes_positive_actual_v2_file`(양성, 실제 구현물 대상) + `..._rejects_shebang_modified` + `..._rejects_shebang_removed` + `..._rejects_shebang_inserted_into_no_shebang_base` + `..._rejects_encoding_cookie_inserted_with_non_ascii_semantic_reproduction`(리뷰의 정확한 재현 + em dash mojibake 직접 검증) + `..._rejects_encoding_cookie_modified` + `..._rejects_encoding_cookie_removed` + `..._rejects_bom_inserted` + `..._rejects_bom_plus_conflicting_cookie_fails_closed` + `..._accepts_identical_bom_present_in_base_and_current`(양성) + `..._accepts_leading_non_cookie_comment_as_inert`(양성) + 회귀로 그대로 재실행되는 §3.1a `test_audit_exact_allowed_delta_positive_actual_v2_file`/`..._positive_synthetic_fixture`/decorator-span 표 전체/기존 negative 매트릭스 전체(§7.1 상단, 43-to-48 fixture 포함) |

이 표의 행은 §12 구현자 체크리스트의 갱신된 항목과 1:1로 대응한다.
§13~§17이 기록한 DR-I1~DR-I4·DR-RC1-I1의 폐쇄 판정은 이 개정에서도
재오픈되지 않는다 — §3.1b는 §3.1a의 `_statement_source_slice`/
`_top_level_statement_slices`/`audit_exact_allowed_delta`/
`_first_divergence_violation`이나 세 pin된 델타 리터럴 중 어느 것도
바꾸지 않으며, base revision의 preamble이 shebang 한 줄뿐이고 다섯
pin된 상수/`_build_baseline`/`main()` 중 어느 것도 preamble 안에 있지
않으므로, 새 경계 함수가 preamble 동일성을 확인한 뒤에는 §3.1a의 로직이
정확히 이전과 동일하게 실행된다. 따라서 DR-I4가 닫은 "이름/바인딩-종류
무관 default-deny" 성질과 §17이 닫은 "decorator를 포함하는 완전한
statement span" 성질은 이 개정에서 좁아지지 않고, shebang·encoding
cookie·BOM이라는 이전에 statement 경계 밖에 있던 실행/디코딩 접두부까지
포함하는 상위집합으로 넓어진다. no native Linux, Ollama, protected
live, environment-approval, self-hosted, historical-evidence 실행/
뮤테이션 경계(§0.2, §5, §11)는 이 개정에서 전혀 건드리지 않는다 — 변경은
전부 `scripts/assemble_m4_evidence.py`의 정적 self-audit 오라클(§3.1a/
§3.1b)과 그 오라클을 exercise하는 §7.1 테스트에 한정된다.

## 19. Design_Review_Recovery_Cycle_1_Iteration_3 폐쇄 matrix — 구현 phase

[Design_Review_Recovery_Cycle_1_Iteration_3.md](Design_Review_Recovery_Cycle_1_Iteration_3.md)는
**PASS — 9.8/10.0**(CRITICAL 0, MAJOR 0, MINOR 1)을 판정해 구현 착수를
승인했다. 이 표는 구현 phase가 그 유일한 잔존 MINOR(DR-RC1-I3-MIN-01)를
정확히 어떻게 닫았는지 기록한다. 그 밖의 모든 설계 절, §3.1a whole-file
오라클, §3.1b의 preamble byte-exact 비교 알고리즘, 세 pin된 델타, 그리고
§13~§18이 기록한 DR-I1~DR-I4·DR-RC1-I1·DR-RC1-I2의 폐쇄 판정은 전혀
바뀌지 않는다.

| Finding | 핵심 결함 | 구현 phase의 수정 | 증명하는 코드/테스트 |
|---|---|---|---|
| DR-RC1-I3-MIN-01 | `_source_preamble`은 `cookie_index = 1 if has_shebang else 0`으로만 cookie 위치를 판정했다 — line 1이 shebang이 아닌 평범한 comment-only 줄이고 실제 cookie가 line 2에 있는 합성 no-shebang 케이스에서, `consumed[0]`(평범한 주석)만 검사하고 `consumed[1]`(실제 cookie)은 검사하지 않아 `b""`를 반환했다. PEP 263/`tokenize.detect_encoding`은 line 1이 blank-or-comment-only이기만 하면(shebang 여부와 무관하게) line 2의 cookie를 인정한다 | `_source_preamble`을 `tokenize.detect_encoding`과 동일한 2-line 판정 규칙으로 재작성 — `consumed[0]`이 cookie 정규식과 매치하면 `cookie_index=0`, 아니면 `len(consumed) > 1`이고 `consumed[1]`이 매치하면 `cookie_index=1`(어느 쪽이든 `has_shebang`과 무관하게 독립적으로 판정). `kept_line_count`는 `cookie_index`가 있으면 `cookie_index + 1`, 없고 `has_shebang`이면 `1`, 아니면 `0`으로 통합해, shebang-then-cookie·comment-then-cookie·cookie-only·no-cookie 네 경우를 하나의 규칙으로 정확히 처리한다. 실제 pinned base(shebang만, cookie 없음)에 대한 기존 동작은 100% 보존된다(shebang-only 분기가 여전히 `kept_line_count=1`을 만든다) | `tests/unit/test_assemble_m4_evidence.py::test_source_preamble_detects_comment_first_second_line_cookie`(신규 합성 no-shebang comment-first cookie 검출 단위 검증), `..._bytes_rejects_comment_first_second_line_cookie_modified`/`..._removed`(신규 negative 매트릭스, DR-RC1-I3-MIN-01의 "modification/removal matrix" 요구), `..._bytes_accepts_comment_first_second_line_cookie_unchanged`(양성, preamble 동일성 직접 검증) — 기존 §7.1 전체(43-to-48 fixture, shebang-only pinned base 대상 양성/negative 전부)와 §3.1a whole-file 오라클은 회귀 없이 그대로 재실행되어 통과 |

이 표의 수정은 §3.1a(`_statement_source_slice`/`_top_level_statement_slices`/
`audit_exact_allowed_delta`/pin된 세 델타), §3.1b의 preamble byte-exact
비교 순서(preamble 비교 → encoding 검출 → `audit_exact_allowed_delta`
위임), 그리고 실제 pinned base revision(shebang만, cookie 없음)에 대한
`_source_preamble` 반환값 중 어느 것도 바꾸지 않는다 — 바뀐 것은 오직
"cookie가 line 1 자신에 있는지, 아니면 line 1이 inert(shebang이든 평범한
주석이든)해서 line 2를 봐야 하는지"를 판정하는 규칙 하나뿐이며, 그 규칙은
CPython 자신의 `tokenize.detect_encoding` 판정과 이제 정확히 일치한다.
no native Linux, Ollama, protected live, environment-approval,
self-hosted, historical-evidence 실행/뮤테이션 경계(§0.2, §5, §11)는 이
구현 phase에서도 전혀 건드리지 않는다.
