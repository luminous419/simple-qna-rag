# M4.3 Artifact & Deployment Safety 설계 리뷰 — Iteration 2

리뷰 일자: 2026-08-12 (Asia/Seoul)  
리뷰어: Fresh Codex (독립 설계 reviewer)  
대상: 개정 `Design.md`, M4.3 Requirement/Plan/Traceability, Iteration 1 리뷰,
`milestone_dev_orchestration_guide.md`, 현재 코드/tests/workflow, M4.1/M4.2 승인·예외 산출물  
판정 범위: pre-merge Design Quality Gate

## 판정

**FAIL — 8.4/10**

CRITICAL 0건, MAJOR 5건, MINOR 1건이다. PASS 조건인 CRITICAL=0,
MAJOR=0, score 9.7/10 이상을 충족하지 못한다. Iteration 1의 1 CRITICAL,
7 MAJOR, 1 MINOR 중 trust-before-pickle byte identity, `current` fail-closed,
pinned legacy approval, fd-relative retention/staging TTL, static COPY와 Linux
host-gateway, payload 파일 hash 확인, exact-key gate 집합은 설계 수준에서
유의미하게 개선됐다. 그러나 아래 결함 때문에 DR-I1-MAJ-05/07/08과
DR-I1-MIN-09는 완전히 닫히지 않았고, watchdog 계약에도 신규 추적성 결함이
남는다.

Native Linux/Ollama/DDGS, protected M3 live, M4.1 live 14-gate는 실행하지
않았다. M4.1 operational blocker와 protected M3 live `NOT_RUN`은 그대로
보존되며, 이 리뷰는 어떤 PASS도 합성하거나 self-hosted runner/environment
승인을 변경하지 않는다.

## Findings

### DR-I2-MAJ-01 — crash reconcile이 history append를 멱등 처리하지 않아 previous 대수가 깨진다

- **Severity:** MAJOR
- **Iteration 1 closure:** DR-I1-MAJ-05 미완전 closure
- **근거:** `Design.md` §4.4의 성공 순서는 journal `pointer_committed` →
  `_append_history` → `_write_receipt_atomic` → journal clear다. §4.4-b의
  reconcile은 `actual_current == post`이면 history/receipt 존재 여부나
  operation ID를 검사하지 않고 `_append_history`를 다시 실행한다. 따라서
  history append/fsync 성공 직후, receipt write 전 또는 journal clear 전에
  crash하면 같은 operation이 두 번 append된다. `_read_previous_from_history`가
  마지막에서 두 번째 성공 항목을 previous로 사용하므로 duplicate `post`
  행은 previous를 실제 pre가 아니라 current와 같게 만들고 rollback 및 retention
  보호 집합을 훼손한다. 설계의 fault matrix는 history append **전** crash만
  다루며 이 창을 검출하지 못한다.
- **수정안:** history 행에 unique `operation_id`, `pre_pointer`, `post_pointer`를
  넣고 append 전 동일 operation 존재를 exact parse로 검사하거나, history도
  journal에서 파생되는 atomic snapshot/commit marker로 만든다. history append
  후·receipt replace 후·journal unlink 전 각 crash 지점에서 재시작을 반복해
  operation당 history/receipt exact-one, previous=pre, current=post를 검증해야 한다.

### DR-I2-MAJ-02 — OCI mock smoke는 host 연결과 embedding runtime 양쪽에서 실행 불가능하다

- **Severity:** MAJOR
- **Iteration 1 closure:** DR-I1-MAJ-06은 static COPY로 closure, DR-I1-MAJ-07은 미완전 closure
- **근거:** §7.5는 `http.server` 기반 mock을 host thread에서 띄운다고만 하고
  container-reachable 주소(`0.0.0.0`)에 bind한다는 계약이 없다. 기본적인
  localhost bind fixture라면 `--add-host=host.docker.internal:host-gateway`가
  있어도 container에서 host loopback listener에 연결할 수 없다. 더 근본적으로
  host에서 만든 BAAI/bge-m3 index를 production container가 query하려면 같은
  embedding model을 container 안에서 초기화해야 하지만, Dockerfile은 model/cache를
  복사하지 않고 runtime은 UID 10001, read-only rootfs, 제한된 `/tmp`로 실행된다.
  §14도 container job의 **host** 다운로드만 언급하며 container 내부 model 획득·cache
  경로·network 계약을 정의하지 않는다. 따라서 `/health/ready` 또는 `/rag`가 model
  초기화 단계에서 실패할 수 있어 목표 Linux hosted smoke의 구현 가능성이 없다.
- **수정안:** mock server exact bind address와 argv를 계약화하고 실제 socket
  reachability를 먼저 검사한다. smoke 전용 deterministic embedding provider를
  production image에 명시적으로 주입 가능한 test seam으로 두거나, pinned model
  artifact/cache를 read-only mount하고 writable cache 위치를 tmpfs로 제공하되 production
  경로에서 임의 test provider가 활성화되지 않도록 설정 경계를 검증한다. exact Docker
  argv unit test와 OCI production image에서 readiness/static/query를 실제로 확인하는
  hosted test가 모두 필요하다.

### DR-I2-MAJ-03 — `m43.json.status == PASS`만 검사해 detailed evidence semantic PASS를 합성할 수 있다

- **Severity:** MAJOR
- **Iteration 1 closure:** DR-I1-MAJ-08 미완전 closure
- **근거:** §8.2의 assembler는 payload hash/size를 재계산하지만
  `m43-deterministic` payload의 의미 검사는 `m43.json` 최상위 `status == "PASS"`
  하나뿐이다. §10이 요구하는 `schema`, seed/repeat, required node exact set,
  각 node의 repeat count/status, evidence-mismatch negative control 결과와
  completeness를 assembler가 재검산하지 않는다. 공격/버그로 만든
  `{"status":"PASS"}` 파일도 receipt가 그 bytes의 hash/size를 정직하게 선언하면
  producer `OK`가 된다. `_check_identity`도 schema/job/semantic_status 값과 exact-key를
  검사하지 않고 required key의 존재만 확인하며, malformed `payloads` entry는
  dict comprehension의 `KeyError`/`TypeError`로 typed FAIL 대신 assembler 자체를
  죽일 수 있다.
- **수정안:** producer별 typed payload parser를 두고 `m43.json`의 exact schema,
  command identity, repeat=10, seed=4303, required node exact set/count/status와 negative
  control expected failure를 독립 재계산한다. producer envelope도 exact-key/type/value,
  `schema`, `job`, `semantic_status`, payload entry schema를 fail-closed로 검증하고 모든
  malformed 입력을 `PAYLOAD_INVALID`로 정규화한다. 최소 JSON `PASS`, node omission,
  repeat/seed 변조, malformed payload entry negative tests를 추가해야 한다.

### DR-I2-MAJ-04 — baseline checker가 producer↔gate 관계를 재계산하지 않아 합성 PASS가 가능하다

- **Severity:** MAJOR
- **Iteration 1 closure:** DR-I1-MIN-09가 MAJOR 경로로 잔존
- **근거:** §9.2는 top-level/gates exact-key와 gate enum을 검사하지만
  `producers`의 exact key/schema/status를 전혀 검사하지 않는다. 주석은
  “producers/gates에서 직접 재계산”한다고 하나 실제
  `expected_deterministic`는 candidate가 자기 보고한 네 gate만 읽는다.
  따라서 `producers={}` 또는 모든 producer가 `MISSING`이어도 네 gate를 `PASS`로
  쓰면 `deterministic_status=PASS`와 checker exit 0이 가능하다. 또한
  `expected_ready`가 독립적으로 재계산한 지역 변수 대신 candidate의
  `deterministic_status`/`operational_status`를 다시 사용해 대수 구현이 자기 보고에
  불필요하게 의존한다.
- **수정안:** required producer exact set과 각 producer typed status를 검사하고,
  gate를 producer status에서 재계산한 뒤 candidate gate와 비교한다. 이후
  deterministic/operational/overall 값을 오직 재계산된 지역 값에서 도출한다.
  producer omission/extra/status mismatch 및 `producer=MISSING + gate=PASS`를
  필수 negative cases로 추가해야 한다.

### DR-I2-MAJ-05 — watchdog 8-test 계약이 `consumer_fenced` 실제 실패 의미를 검증하지 않는다

- **Severity:** MAJOR
- **근거:** §11.2의 exact argv 두 테스트는 `task-list --run ... --from
  term_coord --brief --json`과 `check --terminal term_coord --run ... --peek
  --json`을 고정하고 총 8개 신규 함수도 명시한다. 그러나 stale/non-owner
  coordinator에서 Orca가 반환하는 `consumer_fenced` 오류를 재현하거나 그 exact
  command가 실패로 전파되고 wake를 보내지 않으며 durable journal에 fenced reason을
  남기는 계약은 없다. generic `RuntimeError("orca cli unavailable")`와
  `run_loop` 실패 문자열 검사만으로는 identity fencing 회귀(예: `--from` 누락,
  wrong terminal인데 fake가 성공)를 검출하지 못한다. 더구나 테스트 6의 설명은
  stdout에 `"ok": true`가 없음을 요구하지만 제시한 코드는 return code만 assert한다.
- **수정안:** exact argv를 받은 fake가 stale terminal에 대해 구조화된
  `consumer_fenced` CLI 실패를 반환하도록 하고, `run_json`/`check_once`/`main`/run-loop의
  fail-closed 결과를 검증한다. no-send, exit 2, stdout no-success, stderr typed error,
  journal의 bounded `consumer_fenced` reason을 모두 assert하고도 총 8-test 계약을
  유지하도록 테스트 항목을 재구성해야 한다.

### DR-I2-MIN-06 — pinned baseline의 배포 위치와 “untracked” 검증 서술이 구현과 불일치한다

- **Severity:** MINOR
- **근거:** §4.7은 고정 path와 embedded SHA로 임의 CLI 승인 파일을 제거해 핵심
  trust 문제는 닫았다. 다만 production image는 `evaluation/`을 복사하지 않으므로
  `_PINNED_M3_BASELINE_PATH = _REPO_ROOT/evaluation/baselines/m3_initial.json`은
  설치/배포 형태에 따라 존재하지 않는다. 또한 테스트 이름은
  `rejects_tampered_or_untracked_baseline`이지만 실제 설계에는 git tracked/blob identity
  검사가 없고, `git diff --check`는 dirty/untracked 여부를 검사하지 않는다.
- **수정안:** 승인 hash pair/baseline ID를 package data 또는 코드 상수로 배포하고
  baseline JSON bytes SHA를 provenance 상수로 함께 보존하거나, baseline 파일을 명시적
  package data/COPY allowlist에 넣는다. 테스트/문서 이름은 실제 보장(embedded SHA
  mismatch)과 일치시켜야 한다.

## Iteration 1 closure 재검증

| Iteration 1 finding | Iteration 2 판정 | 메커니즘 수준 근거 |
|---|---|---|
| DR-I1-CRIT-01 trust-before-pickle/dirfd | **CLOSED** | 동일 version dirfd에서 bytes를 끝까지 읽고 hash 검증한 뒤 `deserialize_index`/`pickle.loads`에 그 bytes를 직접 전달한다. verified path 재오픈이 없다. |
| DR-I1-MAJ-02 `current` downgrade | **CLOSED** | `exists()` 없이 `openat(O_NOFOLLOW)` errno를 사용하며 ENOENT만 legacy fallback, ELOOP/non-regular/malformed는 fail-closed다. |
| DR-I1-MAJ-03 legacy approval | **CLOSED with MINOR follow-up** | CLI override 제거와 embedded SHA로 임의 승인 source는 차단했다. 배포/package 위치 정합성은 DR-I2-MIN-06. |
| DR-I1-MAJ-04 retention/staging | **CLOSED** | version/staging 삭제가 opened dirfd-relative no-follow walk이고 UUID/min-age/global-lock/dry-run/apply 정책이 정의됐다. |
| DR-I1-MAJ-05 crash durability | **OPEN** | journal은 추가됐으나 history append 후 crash의 exact-once/idempotency가 없다(DR-I2-MAJ-01). |
| DR-I1-MAJ-06 static production | **CLOSED** | production allowlist에 `web/static`/`web/templates`가 있고 root/static smoke가 설계됐다. |
| DR-I1-MAJ-07 Linux mock connectivity | **OPEN** | host-gateway는 추가됐으나 listener bind와 container embedding runtime이 미정의다(DR-I2-MAJ-02). |
| DR-I1-MAJ-08 detailed evidence | **OPEN** | payload bytes hash는 검증하지만 m43 상세 semantics/envelope fail-closed가 부족하다(DR-I2-MAJ-03). |
| DR-I1-MIN-09 baseline exact algebra | **OPEN / MAJOR** | gate key exactness은 개선됐으나 producer→gate 대수가 검증되지 않는다(DR-I2-MAJ-04). |

## 추적성·구현 가능성 결론

Requirement→Design→test 이름의 표면적 매핑은 §13에 존재하지만, crash exact-once,
container embedding, m43 detailed receipt, producer→gate algebra, consumer fencing은
요구된 실패 의미를 실제 test oracle로 끝까지 연결하지 못한다. 이 다섯 MAJOR를
해결하기 전에는 구현 Phase로 진입하면 안 된다. M4.1 operational은 계속
`BLOCKED`, protected M3 live는 계속 `NOT_RUN`, `overall_release_ready=false`여야 한다.

## 검증

- 문서 링크 검사: `python scripts/check_markdown_links.py`
- whitespace/patch 검사: `git diff --check`
- Orca CLI 문법 확인(읽기 전용): `orca orchestration task-list --help`,
  `orca orchestration check --help`
- 실행하지 않음: Native Linux/Ollama/DDGS, M3 protected live, M4.1 live 14-gate,
  self-hosted runner/environment 승인 변경

