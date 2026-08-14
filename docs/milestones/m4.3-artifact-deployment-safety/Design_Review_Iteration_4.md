# M4.3 Artifact & Deployment Safety 설계 리뷰 — Iteration 4

리뷰 일자: 2026-08-12 (Asia/Seoul)  
리뷰어: Fresh Codex (독립 설계 reviewer)  
대상: Iteration 4 개정 [Design.md](Design.md), [Requirement.md](Requirement.md),
[Plan.md](Plan.md), [Traceability.md](Traceability.md),
[Design_Review_Iteration_1.md](Design_Review_Iteration_1.md),
[Design_Review_Iteration_2.md](Design_Review_Iteration_2.md),
[Design_Review_Iteration_3.md](Design_Review_Iteration_3.md),
[milestone_dev_orchestration_guide.md](../../../milestone_dev_orchestration_guide.md),
현재 코드/tests/`.github/workflows/ci.yml`, M4.1/M4.2 승인·예외 산출물  
판정 범위: pre-merge Design Quality Gate

## 판정

**FAIL — 9.1/10**

CRITICAL 0건, MAJOR 2건, MINOR 1건이다. PASS 조건인 CRITICAL=0,
MAJOR=0, score 9.7/10 이상을 충족하지 못한다. Iteration 4는 production
test seam의 물리적 분리, assembler의 독립 node oracle, command/negative
field exact 검사, consumer fence 즉시 종료를 mechanism과 negative oracle
수준으로 유의미하게 닫았다. 그러나 operation history의 `previous` 대수와
baseline producer payload identity가 제시된 pseudocode에서 아직 요구 계약을
충족하지 않으며, pinned baseline 상수도 실제 승인값 대신 placeholder로 남아 있다.

Native Linux/Ollama/DDGS, protected M3 live, M4.1 live 14-gate는 실행하지
않았다. M4.1 operational은 계속 `BLOCKED`, protected M3 live는 계속
`NOT_RUN`, `operational_status=BLOCKED`, `overall_release_ready=false`다.
이 리뷰는 PASS를 합성하지 않았고 self-hosted runner/environment approval을
변경하지 않았다.

## Findings

### DR-I4-MAJ-01 — immutable history의 `previous` 계산이 첫 operation에서 `pre_pointer`를 잃는다

- **Severity:** MAJOR
- **Iteration 3 closure:** DR-I3-MAJ-01 미완전 closure
- **근거:** §4.4-a-1은 operation별 `<op_id>.json`을 full-write/fsync/
  atomic rename/parent-fsync로 커밋하므로 regular-file partial-tail 문제와
  op별 물리적 중복은 구조적으로 제거했다. 하지만
  `_read_previous_from_history()`는 최신 record의 `pre_pointer`가 아니라
  `rows[-2].post_pointer`를 반환하고, record가 하나뿐이면 무조건 `None`을
  반환한다. 최초 activation/import가 `pre_pointer=A`, `post_pointer=B`인
  정상 operation이면 durable record가 정확히 하나인 상태에서 요구되는
  `previous=A`가 아니라 `previous=None`이 된다. 이는 같은 절의 fault oracle
  “operation당 exact-one, `previous=pre_pointer`, `current=post_pointer`”와
  직접 모순이며, 첫 배포 직후 `rollback --to-previous`와 retention의 직전
  version 보호를 잃는다. 또한 `_read_history_rows()`는 record exact-key/type,
  filename의 op_id와 body `op_id` 일치, sequence uniqueness/contiguity를 검사하지
  않아 ordering oracle도 단순 `r["sequence"]` 정렬에 머문다.
- **수정안:** `previous`는 현재 pointer를 만든 최신 committed record의
  `pre_pointer`에서 직접 도출하고, 최신 record의 `post_pointer == current`를
  함께 검증한다. history reader는 exact schema/key/type, filename↔body op_id,
  unique contiguous sequence와 operation enum을 fail-closed로 검증해야 한다.
  empty history, first `A→B`, second `B→C`, rollback `C→B`, sequence duplicate/gap,
  filename/body mismatch를 각각 독립 fixture로 두고 모든 crash window에서
  exact-one/order/`previous=pre`/`current=post`를 assert해야 한다.

### DR-I4-MAJ-02 — producer `OK` tagged union이 payload exact identity를 검증하지 않는다

- **Severity:** MAJOR
- **Iteration 3 closure:** DR-I3-MAJ-04 미완전 closure
- **근거:** §9.2의 `PRODUCER_STATUS_SCHEMA`는 status별 exact key set을
  검사해 `{"status":"OK"}` 합성 PASS를 닫았고 receipt hash와 payload hash의
  형식도 확인한다. 그러나 `OK.payload_hashes`는 job별 **개수**와 key/value의
  문자열/64-hex 형식만 검사한다. `container`에
  `{"a":"<64hex>","b":"<64hex>"}`, `m43-deterministic`에 임의 두
  filename을 넣어도 통과한다. 따라서 baseline candidate가 assembler가 검증한
  `layer_scan.json`/`container_smoke.json` 또는 `m43.json`/
  `m43-negative.json` identity를 실제로 보존한다는 보장이 없고,
  DR-I3-MAJ-04가 요구한 receipt/payload identity summary를 exact schema로
  닫지 못한다. checker가 원본 payload를 재파싱할 필요는 없지만, candidate에
  복사된 identity의 exact filename set은 독립적으로 고정해야 한다.
- **수정안:** count 상수 대신 job별 review-pinned exact filename set을 두고
  `set(payload_hashes) == EXPECTED_PAYLOAD_FILENAMES[job]`를 검사한다.
  가능하면 producer `OK`에 receipt가 선언한 canonical payload-manifest hash도
  포함해 assembler output과 baseline copy의 identity를 결합한다. same-count
  filename substitution, extra+omission 상쇄, 다른 job filename 교환, malformed
  receipt/payload-manifest hash를 필수 negative cases로 추가해야 한다.

### DR-I4-MIN-03 — pinned M3 승인 상수가 실제 기준값이 아니라 placeholder다

- **Severity:** MINOR
- **Iteration 3 closure:** DR-I3-MIN-06 fixture seam은 CLOSED, real-constant closure는 OPEN
- **근거:** §4.7은 `_parse_m3_baseline_fingerprint(raw: bytes)`를 분리해
  tracked bytes positive와 `tmp_path` 변조 사본 negative가 같은 parser를
  사용하도록 했고, §15에
  `git diff --exit-code -- evaluation/baselines/m3_initial.*`도 추가했다.
  이로써 fixture seam과 baseline bytes diff gate는 닫혔다. 그러나 production
  trust root로 제시된 두 상수는 여전히 `"0" * 64`다. 현재 승인 baseline에는
  이미 `index.faiss=c52fb288f3bc780d681ab68d62dd5b4c545843ee9dc53ef5528bd50db4d69820`,
  `index.pkl=3f7217a2825f141bbea4c75dbf2b15738c3bcd50e3c0900e25a6a4a7ed91bb00`가
  존재하므로 설계의 “real constants” closure를 구현 단계로 미룰 이유가 없다.
- **수정안:** §4.7 pseudocode에 위 승인값을 실제 상수로 고정하고 positive
  provenance test와 baseline-byte diff gate를 그대로 유지한다. 구현 시 상수
  복사 외 baseline JSON/Markdown byte 변경이 0인지 별도 diff receipt로 남긴다.

## Iteration 3 closure 재검증

| Iteration 3 finding | Iteration 4 판정 | mechanism/test-oracle 결론 |
|---|---|---|
| DR-I3-MAJ-01 immutable atomic history | **OPEN** | per-operation atomic record는 partial-tail/exact-one을 닫았지만 `previous`가 latest `pre_pointer`가 아닌 second-last record에서 계산되어 first operation과 ordering corruption을 잘못 처리한다(DR-I4-MAJ-01). |
| DR-I3-MAJ-02 production embedding seam | **CLOSED** | seam은 `tests/support/`로 이동해 production layers에 물리적으로 없고, 동일 production image의 read-only harness와 harness 없는 two-env negative OCI 503 oracle이 분리됐다. |
| DR-I3-MAJ-03 independent m43 oracle | **CLOSED** | assembler-owned pinned node set, exact command, repeat/seed, node schema/count/status와 positive/negative exact fields가 producer constant+output 동시 누락 및 field 변조 fixtures로 고정됐다. |
| DR-I3-MAJ-04 producer tagged union | **OPEN** | variant exact-key와 hash 형식은 닫혔지만 payload identity가 exact filenames가 아닌 count만 검사되어 same-count substitution이 통과한다(DR-I4-MAJ-02). |
| DR-I3-MAJ-05 consumer fenced | **CLOSED** | fenced reason을 exact-one journal에 기록한 뒤 `run_loop`이 1을 반환하며 call count=1/no retry/no send/bounded stderr를 단일 oracle로 검증한다. `main()`의 기존 `return run_loop(...)`가 process nonzero를 전달한다. |
| DR-I3-MIN-06 baseline fixture seam/constants | **CLOSED with MINOR follow-up** | bytes parser seam과 real diff gate는 닫혔다. 실제 승인 상수 대신 placeholder가 남은 부분만 DR-I4-MIN-03이다. |

## 전체 설계 재검토

- trust-before-pickle은 동일 dirfd에서 읽고 검증한 `faiss_bytes`/
  `pkl_bytes`를 재오픈 없이 역직렬화하며, `current` symlink/invalid pointer가
  legacy fallback으로 downgrade되지 않는 구조를 유지한다.
- same-filesystem staging, immutable publish, atomic pointer activation,
  rollback primitive 재사용, fd-relative retention/cleanup의 큰 구조는 구현
  가능하다. 다만 history `previous` 오류가 rollback/retention 보호 대수에
  직접 영향을 주므로 DR-I4-MAJ-01 전에는 구현 Phase 진입이 안전하지 않다.
- OCI production allowlist, static assets, non-root/read-only/drop-all 실행,
  layer scan, mock reachability, deployment/recovery runbook의 계약은 구체적이다.
  embedding seam은 production image에 없고 동일 image에 read-only code mount를
  명시적으로 추가한 harness와 무주입 negative activation을 구별한다.
- hosted DAG는 단일 `.github/workflows/ci.yml`에 추가되고 현재 protected
  `m3-live-regression-gate`를 assembler `needs`에 넣지 않는다. 기존
  self-hosted label/environment/trusted-trigger block과 M4.2 merge 계약을
  semantic snapshot 0-diff로 보존하는 방향이 맞다.
- M4 baseline은 deterministic과 operational을 분리하며
  `M4.1_BLOCKED=true`, protected M3 live `NOT_RUN`,
  `overall_release_ready=false`를 보존한다. 다만 `OK.payload_hashes` exact
  identity가 닫히기 전에는 deterministic PASS candidate 자체가 완전한
  fail-closed evidence라고 볼 수 없다.
- Requirement/Plan/Traceability는 기준 revision `648e3ab`, M4.1 operational
  exception, M4.2 deterministic 승인 산출물을 일관되게 참조한다. 이번
  Design-only 개정의 상세 closure가 상위 Traceability의 `PLANNED` 상태를
  임의로 PASS로 바꾸지 않은 점도 적절하다.

## 재검토 진입 조건

두 MAJOR를 pseudocode와 negative test oracle에서 먼저 닫아야 한다. 다음
iteration은 (1) latest record의 `pre_pointer` 기반 previous 대수와 strict
history record/order schema, (2) producer별 exact payload filename set 및
receipt/payload identity 결합을 우선 재검증해야 한다. 실제 M3 승인 hash 두
상수도 placeholder가 아닌 값으로 문서화해야 9.7 Gate를 검토할 수 있다.

## 검증

- 문서 링크 검사: `python scripts/check_markdown_links.py`
- whitespace/patch 검사: `git diff --check`
- 읽기 전용 근거 확인: 현재 코드/tests/`.github/workflows/ci.yml`, M4.1
  `Operational_Acceptance_Exception.md`, M4.2 `Final_Verification_Report.md`,
  승인 `evaluation/baselines/m3_initial.json`, 최근 merge history
- 실행하지 않음: Native Linux/Ollama/DDGS, M3 protected live, M4.1 live
  14-gate, self-hosted runner/environment 승인 변경
