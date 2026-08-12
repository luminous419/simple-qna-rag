# M4.3 Artifact & Deployment Safety 설계 리뷰 — Iteration 3

리뷰 일자: 2026-08-12 (Asia/Seoul)  
리뷰어: Fresh Codex (독립 설계 reviewer)  
대상: Iteration 3 개정 `Design.md`, M4.3 Requirement/Plan/Traceability,
`Design_Review_Iteration_1.md`, `Design_Review_Iteration_2.md`,
`milestone_dev_orchestration_guide.md`, 현재 코드/tests/`.github/workflows/ci.yml`  
판정 범위: pre-merge Design Quality Gate

## 판정

**FAIL — 8.2/10**

CRITICAL 0건, MAJOR 5건, MINOR 1건이다. PASS 조건인 CRITICAL=0,
MAJOR=0, score 9.7/10 이상을 충족하지 못한다. Iteration 2의 Linux mock
bind/reachability, pinned baseline의 runtime-file 제거, 상세 payload의 형식화,
producer→gate 방향의 재계산, bounded watchdog reason은 유의미한 개선이다.
그러나 아래 결함 때문에 DR-I2-MAJ-01/02/03/04/05는 mechanism 또는 test-oracle
수준에서 완전히 닫히지 않았다. DR-I2-MIN-06의 production runtime-path 문제는
닫혔지만 provenance 테스트 서술에는 작은 불일치가 남는다.

Native Linux/Ollama/DDGS, protected M3 live, M4.1 live 14-gate는 실행하지
않았다. M4.1 operational은 계속 `BLOCKED`, protected M3 live는 계속
`NOT_RUN`, `overall_release_ready=false`다. 이 리뷰는 PASS를 합성하지 않았고
self-hosted runner/environment approval도 변경하지 않았다.

## Findings

### DR-I3-MAJ-01 — history exact-once가 regular-file short write/crash에서 성립하지 않는다

- **Severity:** MAJOR
- **Iteration 2 closure:** DR-I2-MAJ-01 미완전 closure
- **근거:** `Design.md` §4.4-a-1의 `_append_history`는 `O_APPEND` regular
  file에 단일 `os.write(fd, record)`를 하면 record가 `PIPE_BUF`보다 작으므로
  partial write가 없다고 가정한다. `PIPE_BUF` 원자성은 pipe/FIFO 계약이지
  regular file의 complete-write 보장이 아니다. 더구나 `_read_history_rows`는
  crash로 생긴 malformed 후행 줄을 무시한 뒤 같은 파일 끝에 새 JSON 줄을
  append한다. partial bytes에 newline이 없으면 재시도 record가 그 bytes에
  바로 이어져 하나의 malformed line이 되므로 새 `op_id`도 영구히 보이지
  않는다. 이후 reconcile은 같은 작업을 계속 append해도 exact-one history와
  `previous=pre`를 복구하지 못한다.
- **수정안:** append-only JSONL을 exact-once commit primitive로 사용하지 말고,
  operation별 immutable record를 temp-write/full-write/fsync/atomic rename/parent
  fsync로 커밋한 뒤 history view를 그 record 집합에서 파생하거나, 전체 history
  snapshot을 atomic replace한다. 반드시 short write(0이 아닌 partial 포함),
  newline 전 crash, partial tail이 이미 있는 재시작을 fault-inject하고 op별
  exact-one, `current=post`, `previous=pre`, receipt exact-one을 검증한다.

### DR-I3-MAJ-02 — deterministic embedding seam은 production-disabled가 아니라 production에서 env로 활성화 가능하다

- **Severity:** MAJOR
- **Iteration 2 closure:** DR-I2-MAJ-02 미완전 closure
- **근거:** `Design.md` §5.1/§5.2-a/§7.5는 hosted Linux bind를
  `0.0.0.0`, `host-gateway`, 독립 ping probe로 구체화해 reachability 문제는
  닫았다. 그러나 `DeterministicTestEmbeddings`를 production image의 `src/`에
  포함하고 production Settings가 두 공개 env var를 모두 받으면 활성화한다.
  두 키는 accidental default activation만 막을 뿐 production deployment에서
  test provider를 활성화하지 못하게 하는 경계가 아니다. 문서의
  “production 경로에서 활성화 불가능” 주장과 실제 분기가 모순이며, 잘못된
  운영 설정으로 의미 없는 32차원 embedding이 production index/query에 쓰일 수 있다.
- **수정안:** test stage 전용 module/build target 또는 production image에서
  존재하지 않는 sealed test-only capability를 사용하고, production image에서
  두 env를 설정해도 bootstrap이 거부되는 negative OCI test를 둔다. hosted smoke는
  test image가 아니라 production과 동일한 runtime filesystem/security boundary를
  보존한 별도 명시적 test harness로 provider를 주입해야 한다. 현재의 default-only
  단위 테스트는 production-disabled oracle이 아니다.

### DR-I3-MAJ-03 — m43 typed parser의 node oracle이 producer와 공유돼 독립 completeness 검사가 아니다

- **Severity:** MAJOR
- **Iteration 2 closure:** DR-I2-MAJ-03 미완전 closure
- **근거:** `Design.md` §8.2-c는 assembler가 `PROFILE_NODE_IDS`를
  `run_m43_acceptance.py`에서 직접 import하도록 한다. runner 버그가 필수 node를
  constant와 output 양쪽에서 함께 누락하면 assembler도 축소된 같은 constant를
  oracle로 사용해 exact-set 검사를 통과시킨다. 이는 producer 상세 결과를
  consumer가 독립 재계산해야 한다는 closure 목적을 위반한다. 또한 exact top-key를
  요구하면서도 `command`의 정확한 값, negative receipt의
  `expected_to_fail is True`, positive receipt의 `expected_to_fail is None` 및
  `actual_exit_code is None`을 검사하지 않아 typed schema가 약속한 identity를
  실제로 검증하지 않는다.
- **수정안:** assembler에 review-pinned required node set/version을 독립 상수로
  두거나 별도 승인 contract module을 producer/consumer가 각각 version/hash로
  검증하게 한다. command identity와 negative-control 모든 필드의 exact type/value를
  검사하고, runner constant 자체에서 node를 제거한 fixture, command 변조,
  `expected_to_fail=false`, positive receipt의 가짜 exit code를 동일 parser가
  `PAYLOAD_INVALID`로 거부하는 테스트를 추가한다.

### DR-I3-MAJ-04 — baseline checker는 producer entry 최소 `status`만으로 합성 PASS를 허용한다

- **Severity:** MAJOR
- **Iteration 2 closure:** DR-I2-MAJ-04 미완전 closure
- **근거:** `Design.md` §9.2는 producer key 집합과 status enum에서 gate를
  재계산하므로 `MISSING + gate=PASS`는 닫았다. 하지만 각 producer entry는
  dict이고 `status`가 있다는 것만 검사한다. 따라서 네 entry를 모두
  `{"status":"OK"}`로 만든 synthetic candidate는 receipt hash,
  `needs_result`, payload hashes/reasons가 전혀 없어도 네 gate와
  `deterministic_status=PASS`를 통과한다. 본문이 주장하는 producer
  “exact-key/schema/status” 검사와 의사코드가 불일치한다.
- **수정안:** assembler가 내는 producer success/failure variant를 tagged union의
  exact schema로 정의하고 checker가 success entry의 receipt SHA-256, required
  payload hash exact set 및 identity summary를 검증한다. 최소-status-only,
  success metadata omission/extra/wrong type, failure variant에 success-only 필드가
  섞인 candidate를 거부하고, 모든 downstream 값은 그 검증된 producer local
  objects에서만 계산한다.

### DR-I3-MAJ-05 — `consumer_fenced` 후 run loop 계속 실행은 fail-closed가 아니다

- **Severity:** MAJOR
- **Iteration 2 closure:** DR-I2-MAJ-05 미완전 closure
- **근거:** `Design.md` §11.1은 reason을 `consumer_fenced`로 bounded하게
  분류하지만 §11.2 테스트 7은 첫 fence 뒤 loop가 두 번째 `check_once`를
  실행하는 것을 의도적으로 요구한다. ownership이 거부된 consumer는 이후
  check/wake 권한이 없으므로 계속 retry하는 것이 안전한 복구가 아니다. 매
  interval마다 동일 journal row도 계속 append되어 저장량이 bounded하지 않으며,
  실행 프로세스는 성공 exit 경로에 남는다. no-send 단위 테스트는 첫 호출만
  다루므로 반복 중 외부 상태가 바뀌었을 때 stale watchdog이 wake를 보내는 것도
  막지 못한다.
- **수정안:** `consumer_fenced`를 terminal ownership loss로 분리해 journal에
  단 한 번 bounded reason을 durable 기록하고 run loop를 nonzero로 종료한다.
  generic transient CLI failure만 bounded retry/backoff 대상으로 둔다. 8-test
  계약은 fence 후 call count=1, terminal send=0, journal exact-one, process
  nonzero와 재시작 전 명시적 rebind 필요를 검증하도록 재구성한다.

### DR-I3-MIN-06 — pinned provenance 테스트의 “임시 복사본 변조” 서술이 고정 경로 코드와 맞지 않는다

- **Severity:** MINOR
- **Iteration 2 closure:** DR-I2-MIN-06 runtime 배포 문제는 CLOSED
- **근거:** §4.7의 production import가 코드 상수만 사용하므로 production
  image에서 `evaluation/baselines/`가 없다는 문제는 구조적으로 닫혔다. 그러나
  §12는 baseline을 “임시 복사본에서 1바이트 변조”한다고 쓰는 반면 제시된
  테스트는 `Path(__file__).resolve().parents[2] / evaluation/...` 고정 경로를
  직접 읽는다. 이 형태로는 임시 복사본을 주입할 seam이 없다. 또한 상수는
  아직 `"0" * 64` placeholder라 실제 값 치환은 구현 전 필수다.
- **수정안:** parser/comparison helper에 명시적 path 또는 bytes를 주입해 임시
  fixture를 실제 사용하는 negative test를 설계하고, positive test는 tracked
  baseline bytes와 실제 두 code constant를 비교한다. placeholder 치환과
  `git diff --exit-code -- evaluation/baselines/m3_initial.*`를 구현 Gate에 명시한다.

## Iteration 2 closure 재검증

| Iteration 2 finding | Iteration 3 판정 | 메커니즘/test-oracle 결론 |
|---|---|---|
| DR-I2-MAJ-01 operation exact-once | **OPEN** | regular-file single-write/partial-tail 가정 때문에 crash algebra가 exact-once가 아니다(DR-I3-MAJ-01). |
| DR-I2-MAJ-02 Linux mock/embedding | **OPEN** | bind, host-gateway, reachability는 닫혔으나 seam이 production에서 env로 활성화된다(DR-I3-MAJ-02). |
| DR-I2-MAJ-03 m43 detailed payload | **OPEN** | typed fields는 늘었지만 required-node oracle을 producer와 공유하고 일부 identity 값을 검사하지 않는다(DR-I3-MAJ-03). |
| DR-I2-MAJ-04 producer→gate | **OPEN** | 방향 재계산은 추가됐으나 producer success schema가 `status` 하나로 축소 가능하다(DR-I3-MAJ-04). |
| DR-I2-MAJ-05 consumer fencing | **OPEN** | bounded reason/no-send 첫 호출은 개선됐으나 fenced process를 계속 실행한다(DR-I3-MAJ-05). |
| DR-I2-MIN-06 pinned provenance | **CLOSED with MINOR follow-up** | runtime 파일 의존은 제거됐다. negative fixture 서술과 실제 test seam만 정정 필요(DR-I3-MIN-06). |

## 전체 설계 재검토

- trust-before-pickle은 검증된 `faiss_bytes`/`pkl_bytes`를 직접 역직렬화하고
  재오픈하지 않는 구조이며, pointer symlink가 legacy fallback으로 downgrade되지
  않는 설계도 유지됐다. 이 부분은 설계 수준에서 양호하다.
- staging publish, fd-relative retention, atomic pointer activation, rollback primitive
  재사용, OCI layer scan/static assets/runbook의 기본 구조는 구현 가능한 수준이다.
  다만 activation history durability가 깨져 retention의 `previous` 보호까지 연쇄
  영향을 받으므로 DR-I3-MAJ-01 전에는 Phase 3 진입이 불가하다.
- hosted DAG는 하나의 `.github/workflows/ci.yml`에 추가되고 protected
  `m3-live-regression-gate`를 `m4-assemble.needs`에 넣지 않는 방향이 맞다. 현재
  workflow의 protected block과 working-tree watchdog terminal-binding delta도
  보존되어 있다. 구현 시 protected block semantic snapshot 0-diff test가 필수다.
- M4 baseline의 `M4.1_BLOCKED=true`, protected M3 live `NOT_RUN`,
  `operational_status=BLOCKED`, `overall_release_ready=false` 분리는 Requirement와
  Traceability에 일관되게 남아 있다. 이 상태를 deterministic PASS와 별도로
  보존해야 하며 live evidence를 생성·대체해서는 안 된다.

## 재검토 진입 조건

5개 MAJOR를 설계에서 먼저 해소하고 정확한 pseudocode와 negative oracle을
Requirement/Traceability에 다시 연결해야 한다. 다음 iteration은 특히 (1) partial
regular-file write 뒤 재시작 exact-once, (2) production image에서 test embedding
활성화 거부, (3) producer와 독립된 pinned node oracle, (4) exact producer variant
schema, (5) consumer fence 즉시 terminal failure를 우선 검증해야 한다.

## 검증

- 문서 링크 검사: `python scripts/check_markdown_links.py`
- whitespace/patch 검사: `git diff --check`
- 현재 코드/workflow 확인: `scripts/orchestration_watchdog.py`,
  `tests/unit/test_orchestration_watchdog.py`, `.github/workflows/ci.yml`
- 실행하지 않음: Native Linux/Ollama/DDGS, M3 protected live, M4.1 live 14-gate,
  self-hosted runner/environment 승인 변경
