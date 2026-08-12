# M4.3 Artifact & Deployment Safety — Code Review Iteration 2

검토자: Fresh Codex independent code review worker (M4.3 code review iteration 2)  
기준 revision: `648e3abcca7c321e7f4dd13a7fbed1a4f1886c3e`  
판정: **PASS — 9.8/10** (`CRITICAL 0`, `MAJOR 0`, `MINOR 0`, `TRIVIAL 0`)

## 검토 범위와 결론

`milestone_dev_orchestration_guide.md`와 이 디렉터리의 Requirement, Plan,
Design, Design Review Iteration 1~6, Traceability, Implementation Report,
[Code Review Iteration 1](Code_Review_Iteration_1.md),
[Iteration 1 Remediation](Code_Review_Iteration_1_Remediation.md)을 읽고 기준
revision 이후의 전체 working tree를 독립 검토했다. Iteration 1의 세 MAJOR와
한 MINOR는 코드와 독립 실행 oracle 수준에서 모두 닫혔으며, 전체 M4.3 구현의
trust-before-pickle, atomic lifecycle, OCI, evidence/workflow, baseline 및 watchdog
계약에서 새 회귀를 발견하지 못했다.

실제 production image build/layer scan/container smoke와 hosted workflow receipt는
여전히 미실행이다. 이는 기존에 명시된 pre-merge hosted Gate 잔여 조건이며 이번
로컬 code-quality 판정에서 새 finding으로 세지 않는다. Native Linux, Ollama,
DDGS, protected M3 live, M4.1 live gate는 지시대로 실행하지 않았고 self-hosted
runner/environment 승인도 변경하지 않았다.

## Iteration 1 finding closure

### CR-I1-MAJ-01 — CLOSED: 실제 정적 자산 HTTP 의미론과 exit 배선

- `scripts/container_smoke.py::check_static_asset`은 실제
  `GET /static/app.js`를 수행하고 status 200, JavaScript content-type prefix,
  non-empty body를 모두 요구한다. 404, 잘못된 content-type, 빈 body 및 연결 예외는
  모두 `False`다.
- `_ALL_OK_KEYS`에 `static_asset_ok`가 포함되고 `run_smoke()`의 최종 status는
  `compute_all_ok()`로 계산된다. `main()`은 `PASS`만 exit 0, `FAIL`은 exit 1로
  변환한다(`docker_unavailable`의 문서화된 local-only `SKIPPED` 예외는 유지).
- `tests/unit/test_container_smoke_contract.py`가 URL/200/404/content-type/body/
  exception, missing-or-false field의 fail-closed 계산, `main()` exit 0/1을 직접
  고정한다. focused 실행에 포함되어 전부 통과했다.

### CR-I1-MAJ-02 — CLOSED: transition journal strict parser와 무변경 거부

- `index/lifecycle.py::_parse_transition_journal`은 UTF-8 JSON object, exact 7-key
  set, schema literal, phase/operation enum, lowercase 32-hex `op_id`, nullable
  16-hex `pre_pointer`, non-null 16-hex `post_pointer`, timestamp 형식을 모두
  검증한 뒤 frozen `_TransitionRecord`만 반환한다.
- `_reconcile_pending_transition`과 read-only `_diagnose_pending_transition`이 같은
  parser를 재사용하며, reconcile은 parser 반환 전 `current`, history, receipt 또는
  journal을 mutation하지 않는다.
- 18-way malformed matrix와 원래 악성 입력(`schema="wrong"`, traversal-shaped
  `op_id`, `operation="delete"`, null post pointer)이
  `transition_journal_corrupt`로 거부되고 current 불변, history/receipt 미생성,
  journal 보존을 확인한다. traversal lookalike 경계도 별도 oracle로 고정됐다.

### CR-I1-MAJ-03 — CLOSED: member read bound와 declared cap

- `_read_and_verify_member`는 manifest의 expected size에 정확히 `+1`한 값을
  `_read_bounded`에 넘긴다. reader는 각 `os.read` 요청과 누적 반환 bytes 모두를
  그 bound 이내로 제한하고, short 또는 oversize 결과는
  `member_size_mismatch`로 거부한다.
- manifest parser는 bool/non-int/negative뿐 아니라
  `MAX_MEMBER_SIZE_BYTES`(8 GiB)를 넘는 선언도 읽기 전에 거부한다.
- pipe 기반 계속 성장하는 source, 진짜 EOF short read, `os.read` spy의 총 요청량,
  실제 oversize published member 및 truncate oracle이 모두 통과했다.

### CR-I1-MIN-01 — CLOSED: manifest/current limit+1 EOF와 canonical bytes

- manifest는 `MAX_MANIFEST_BYTES + 1`, current는 `_MAX_CURRENT_BYTES + 1`까지
  bounded loop로 읽고 limit 초과를 각각 `manifest_oversize`/
  `current_pointer_malformed`로 거부한다.
- JSON/schema/self-hash parsing 후 raw bytes가 canonical JSON + 단일 newline 또는
  newline만 생략한 canonical bytes와 정확히 일치해야 한다. key order/공백/trailing
  whitespace는 fail-closed다.
- 두 파일의 limit+1 및 non-canonical negative oracle과 manifest의 permitted
  no-newline positive oracle이 모두 통과했다.

## 전체 구현 재검토

- **Trust-before-pickle:** versioned loader는 dirfd + `O_NOFOLLOW`로 고정한 동일
  member bytes를 size/hash/settings 검증한 뒤 직접 deserialize하며 검증 이후 파일을
  재오픈하지 않는다. dangerous `FAISS.load_local`은 genuine `current` ENOENT의
  legacy compatibility loader 한 곳에만 남는다.
- **Atomic lifecycle:** activate/rollback은 동일 `activate()` primitive를 공유한다.
  journal fsync, verified pointer temp-write, atomic replace, parent fsync, exact-once
  per-operation history/receipt와 reconcile 순서가 보존된다. lock, staging publish,
  EXDEV rejection, fd-relative cleanup 및 previous/current 보호 계약도 테스트와
  일치한다.
- **OCI 및 운영 문서:** production stage의 numeric non-root user, test-seam 물리적
  비포함, locked dependencies, runtime security flags와 layer scanner의 traversal/
  whiteout fail-closed 로직을 정적으로 재확인했다. 실제 image 대상 증거만 hosted
  Gate까지 pending이다.
- **Evidence/workflow:** producer receipt exact tagged schema, duplicate/unknown
  filename 거부, canonical payload identity, same run/attempt/SHA binding, exact
  producer set과 `needs` algebra를 assembler/checker가 재검증한다. protected
  `m3-live-regression-gate`의 trigger, `[self-hosted, ollama-m3]` labels 및
  `environment: m3-live-regression` block은 base와 동일하다.
- **Baseline:** `assemble_m4_evidence.py`는 protected M3 live를 `NOT_RUN`, M4.1을
  `BLOCKED`, `M4.1_BLOCKED=true`, `operational_status=BLOCKED`,
  `overall_release_ready=false`로 만든다. checker는 이 algebra를 독립 재계산하며
  synthetic PASS를 허용하지 않는다.
- **Watchdog readiness:** terminal-scoped read-only peek/exact argv가 유지되고,
  `consumer_fenced`는 bounded reason을 한 번 journal한 뒤 nonzero 종료한다. generic
  transient만 interval-paced retry한다.

## 실행 증거

| 명령 | 결과 |
|---|---|
| focused lifecycle/manifest/verification/container/evidence/workflow/watchdog pytest | **128 passed** |
| `venv/bin/python -m pytest tests/unit tests/integration -q` | **1173 passed, 1 skipped** |
| `npm test` | **9 passed** |
| `venv/bin/python scripts/run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303` | **exit 0** |
| `venv/bin/python scripts/check_markdown_links.py` | 최종 실행 PASS |
| `git diff --check` | 최종 실행 PASS |

의도적으로 실행하지 않음: Native Linux/Ollama/DDGS, protected M3 live 14-gate,
M4.1 live 14-gate, 실제 hosted GitHub Actions 및 production Docker image gate.

## Findings, 점수와 Gate

신규 finding은 없다. Severity count는 `CRITICAL 0 / MAJOR 0 / MINOR 0 /
TRIVIAL 0`이다. Iteration 1 remediation은 네 finding의 원인뿐 아니라 해당
regression의 exit/mutation/read-bound/canonical-byte oracle까지 닫았고, 전체 local
deterministic suite에서도 회귀가 없다. 실제 hosted OCI/workflow receipt가 남아 있어
만점은 유보하되 Code Quality Gate 기준(`CRITICAL=0`, `MAJOR=0`, score>=9.7)을
충족하므로 최종 점수 **9.8/10**, Gate **PASS**다.

이 PASS는 M4.3 pre-merge code quality에만 적용된다. M4.1은 **BLOCKED**, protected
M3 live는 **NOT_RUN**, 전체 `overall_release_ready`는 **false**다.
