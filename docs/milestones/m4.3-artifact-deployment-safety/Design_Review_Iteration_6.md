# M4.3 Artifact & Deployment Safety 설계 리뷰 — Iteration 6

리뷰 일자: 2026-08-12 (Asia/Seoul)  
리뷰어: Fresh Codex (독립 설계 reviewer)  
대상: explicit-resume Iteration 6 개정 [Design.md](Design.md),
[Requirement.md](Requirement.md), [Plan.md](Plan.md),
[Traceability.md](Traceability.md), [Design Review Iteration 1](Design_Review_Iteration_1.md),
[Iteration 2](Design_Review_Iteration_2.md), [Iteration 3](Design_Review_Iteration_3.md),
[Iteration 4](Design_Review_Iteration_4.md), [Iteration 5](Design_Review_Iteration_5.md),
[개발 orchestration guide](../../../milestone_dev_orchestration_guide.md), 현재
code/tests/[workflow](../../../.github/workflows/ci.yml), M4.1/M4.2 승인·예외 산출물  
판정 범위: pre-merge Design Quality Gate, 최종 허용 Iteration 6

## 판정

**PASS — 9.7/10**

CRITICAL 0건, MAJOR 0건, MINOR 1건이다. PASS 조건인 CRITICAL=0,
MAJOR=0, score 9.7/10 이상을 충족한다. DR-I5-MAJ-01은 원본 receipt
경계에서 닫혔고 이전 CRITICAL/MAJOR도 재발하지 않았다. 아래 MINOR는
malformed input을 PASS로 만들지 않고 assembler를 nonzero로 중단시키는
방어적 타입 처리·진단 oracle의 완결성 문제이므로 구현 Phase를 차단하지
않지만, 구현 시 함께 수정해야 한다.

Native Linux/Ollama/DDGS, protected M3 live, M4.1 live 14-gate는 실행하지
않았다. M4.1 operational은 계속 `BLOCKED`, protected M3 live는 계속
`NOT_RUN`, `operational_status=BLOCKED`, `overall_release_ready=false`다.
PASS receipt를 합성하지 않았고 self-hosted runner/environment approval을
변경하지 않았다.

## Findings

### DR-I6-MIN-01 — `semantic_status`의 unhashable JSON 타입에 대한 typed rejection oracle이 없다

- **Severity:** MINOR
- **Requirement:** M4.3-REQ-004.3, M4.3-REQ-007.5, M4.3-NFR-005
- **근거:** [Design.md](Design.md) §8.2-b `_check_identity`는
  `doc["semantic_status"] not in SEMANTIC_STATUS_ENUM`을 실행하기 전에
  `isinstance(..., str)`를 검사하지 않는다. JSON string인 `"MAYBE"`는
  계획된 `semantic_status_invalid`로 거부되지만, list 또는 object를 넣으면
  `frozenset` membership에서 `TypeError: unhashable type`이 발생해
  `_evaluate_producer`의 typed `IDENTITY_MISMATCH` 결과 대신 assembler
  process가 중단된다. job/schema는 literal equality가 비문자열도 안전하게
  거부하고, run identity는 bool-as-int까지 명시적으로 배제하므로 이 한
  필드만 타입 경계가 덜 완결돼 있다.
- **영향:** 예외는 workflow를 nonzero로 끝내므로 deterministic PASS나 baseline
  candidate를 합성하지 못한다. 따라서 fail-open 또는 identity substitution은
  아니며 MAJOR로 분류하지 않는다. 다만 §8.3의 malformed negative oracle이
  문자열 enum 위반만 다뤄 parser crash와 안정된 typed rejection을 구분하지
  못한다.
- **수정안:** enum membership 전에
  `if not isinstance(doc["semantic_status"], str) or ...`로 검사하고,
  string/list/object/null/bool/number를 묶은 negative fixture가 모두
  `IDENTITY_MISMATCH: semantic_status_invalid`이며 assembler 자체는 정상적으로
  baseline FAIL 결과를 생성하는지 assert한다. 같은 방어 원칙을 baseline
  checker의 producer `status` membership에도 적용하면 구현 시 진단 품질이
  일관된다.

## DR-I5-MAJ-01 closure 집중 검증

| 검증 경계 | Iteration 6 판정 | mechanism/test-oracle 결론 |
|---|---|---|
| top-level exact schema/version | **CLOSED** | `RECEIPT_TOP_KEYS` 10개 exact set과 `RECEIPT_SCHEMA` literal을 dict 축약 전에 검사해 unknown/missing key와 wrong version을 거부한다. |
| job-to-slot binding과 identity types | **CLOSED with MINOR follow-up** | 호출자의 `job`과 receipt `job`을 직접 비교하고 SHA/workflow/event 및 run/run-attempt 값을 타입·값으로 bind하며 bool-as-int를 배제한다. `semantic_status`의 unhashable 타입만 typed reason이 아닌 process failure가 되는 DR-I6-MIN-01이 남는다. |
| semantic enum | **CLOSED for valid JSON scalar domain** | `PASS`/`FAIL`만 허용하고 다른 문자열은 `semantic_status_invalid`다. assembler는 이 자기 보고를 PASS 근거로 신뢰하지 않고 payload bytes의 semantics를 재계산한다. |
| payload list와 entry schema/types/ranges/allowlist | **CLOSED** | `payloads` list를 먼저 요구하고 각 raw entry에 exact 3-key schema, string filename, global allowlist, 64-lower-hex SHA, non-negative integer size를 검사한다. `size_bytes=true`는 int 검사 전에 명시적으로 거부된다. |
| duplicate before collapse | **CLOSED** | raw filename list 길이와 unique set 길이를 dict 변환 전에 비교해 동일 hash duplicate와 상이 hash duplicate를 모두 `payload_duplicate_filename`으로 거부한다. 마지막 값 승리 경로가 없다. |
| exact job payload set | **CLOSED** | duplicate-free raw entries만 mapping으로 만든 뒤 job별 `REQUIRED_PAYLOADS` exact set과 비교한다. global allowlist를 통과한 cross-job filename도 이 두 번째 경계에서 거부된다. |
| actual bytes/semantic binding | **CLOSED** | contained actual file bytes의 size/hash와 container fields 또는 독립 pinned M4.3 typed parser 결과를 재검증한다. receipt의 `semantic_status`만 바꿔 PASS를 만들 수 없다. |
| single canonical mapping propagation | **CLOSED** | `_verify_payloads`만 검증 완료 `payload_hashes`를 생성하고 `_evaluate_producer`는 raw receipt list를 재축약하지 않는다. 그 mapping에서 manifest hash를 한 번 계산해 receipt 선언값과 비교한 뒤 `OK` entry로 assembler→baseline candidate에 전달한다. checker는 candidate의 exact filename set과 manifest hash를 다시 독립 재계산한다. |
| negative oracle defect detection | **CLOSED with MINOR follow-up** | unknown/missing/schema/job/non-list/malformed entry/allowlist/size bool/same-hash duplicate/different-hash duplicate/substitution과 기존 manifest mismatch cases가 결함 지점별 typed reason을 갖는다. semantic status의 list/object 타입 crash를 typed rejection과 구분하는 oracle만 DR-I6-MIN-01로 보강한다. |

## 이전 finding closure와 전체 설계 재검토

- DR-I1-CRIT-01의 trust-before-pickle은 같은 dirfd에서 검증한 bytes를 재오픈
  없이 역직렬화하고 invalid `current`를 legacy fallback으로 내리지 않는다.
  symlink/owner/mode/hash/settings 위반의 dangerous load 0-call oracle도 유지된다.
- legacy import는 실제 M3 승인 hash pair를 코드 상수로 고정하고 tracked
  baseline bytes diff gate를 둔다. arbitrary CLI approval과 placeholder 문제는
  재발하지 않았다.
- lifecycle은 same-filesystem staging, full-write/file-fsync/directory-fsync,
  immutable publish, atomic pointer replace, transition reconcile, rollback primitive
  재사용과 fd-relative cleanup 순서가 구현 가능하다. per-operation immutable
  history record, strict record/order schema와 latest `pre_pointer` 기반 previous
  대수는 first activation 및 crash window를 포함해 닫혀 있다.
- OCI 설계는 production allowlist/static assets, fixed non-root identity,
  read-only rootfs/tmpfs/drop-all/no-new-privileges, 모든 layer traversal/whiteout
  scan, production image에서 test seam 물리적 제외, read-only harness와 무주입
  503 negative를 구분한다. digest-pinned image/index rollback runbook은 검증
  실패 후 mutation 0의 중단점을 명시한다.
- 단일 [workflow](../../../.github/workflows/ci.yml)에 hosted DAG를 추가하는
  설계이며 현재 protected `m3-live-regression-gate`의 trusted trigger,
  `[self-hosted, ollama-m3]`, `m3-live-regression` environment를 변경하지 않는다.
  현재 repository에는 M4.3 lifecycle/container/assembler 구현이 아직 없으므로
  모든 Traceability 상태를 `PLANNED`로 유지한 것이 정확하다.
- producer tagged union, pinned M4.3 node oracle, consumer-fenced watchdog exit,
  baseline producer→gate→deterministic algebra와 exact payload filename/manifest
  결합은 이전 MAJOR의 합성 PASS 경로를 닫는다. M4.1 exception과 M4.2 final
  deterministic report의 범위도 확대하지 않는다.

## Gate 결론과 구현 인계

Iteration 6은 가이드가 허용한 최종 회차이며 이번 설계는 9.7 Gate를 통과했다.
따라서 별도 설계 Iteration 7은 허용되지 않으며 다음 단계는 Plan의 구현 Phase다.
구현자는 DR-I6-MIN-01의 explicit string guard와 non-string parameterized oracle를
동시에 반영하고, Design의 모든 계획 경로를 실제 code/tests/workflow/runbook으로
구현한 뒤 새 독립 code review Gate를 받아야 한다. 이 문서의 PASS는 설계 품질에만
해당하며 아직 존재하지 않는 M4.3 구현·hosted receipt·operational evidence의 PASS가
아니다.

만약 후속 독립 검증에서 이 Iteration 6 판정이 CRITICAL/MAJOR 또는 score 9.7 미만으로
뒤집히면, [orchestration guide](../../../milestone_dev_orchestration_guide.md)의 총 6회
상한에 따라 설계 iteration을 더 진행하지 말고 terminal guide stop을 적용해야 한다.
중단 원인은 최종 회차 Gate 미달, 잔여 문제는 그 검증에서 확인된 finding, 재개 조건은
사용자가 별도 재설계 범위와 새 cycle을 명시적으로 승인하는 것이다.

## 검증

- 읽기 전용 확인: 현재 Git history/working tree, current code/tests/workflow,
  M3 baseline fingerprint, M4.1 operational exception, M4.2 final deterministic report,
  M4.3 Requirement/Plan/Traceability/Design 및 Design Review Iteration 1~5
- 문서 링크 검사: `python scripts/check_markdown_links.py`
- whitespace/patch 검사: `git diff --check`
- 실행하지 않음: Native Linux/Ollama/DDGS, M3 protected live, M4.1 live
  14-gate, self-hosted runner/environment 승인 변경
