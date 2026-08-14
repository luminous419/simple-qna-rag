# M4.3 Artifact & Deployment Safety 개발 계획

상태: **구현 완료 — Phase 2~8 로컬 결정론적 검증 PASS, post-merge hosted Gate는 미커밋으로 실행되지 않음**  
요구사항: [Requirement.md](Requirement.md)  
구현 보고서: [Implementation_Report.md](Implementation_Report.md)  
상위 절차: [개발 orchestration guide](../../../milestone_dev_orchestration_guide.md)

## 1. 실행 원칙

작업은 다음 순서를 따른다.

```text
Requirement/Plan -> executable filesystem/container/evidence prototypes
-> Design -> 독립 설계 리뷰 Gate -> 구현 phases
-> 독립 code review -> clean deterministic 검증
-> pre-merge hosted Gate -> Git 작업 -> post-merge hosted Gate
-> protected operational Gate 또는 BLOCKED 종료
```

- Codex는 요구사항·계획과 독립 설계/코드 리뷰를 맡고, Claude Code Sonnet 5는 상세
  설계·구현·승인 후 Git 작업을 맡는다.
- 같은 작성자는 자신의 산출물을 최종 승인하지 않는다. 리뷰는 CRITICAL/MAJOR 0,
  MINOR 최소화, 9.7/10 이상을 요구한다.
- M4.1 운영 blocker와 protected live Gate는 독립 상태로 보존한다. M4.3 결정론적
  증거로 PASS를 합성하거나 self-hosted runner/environment approval을 변경하지 않는다.
- 이 문서 작성 cycle에서는 코드, workflow, watchdog, commit/push/PR을 추가 변경하지
  않는다. 기존 watchdog readiness fix는 향후 구현·테스트·리뷰·commit 범위로 승계한다.

## 2. Phase 0 — 기준선, provenance와 trust boundary 고정

작업:

1. `648e3ab`의 Git tree, M3 baseline bytes, M4.1 exception, M4.2 final deterministic
   receipt와 현재 `git status`를 기록한다.
2. 기존 direct-save/load 경계를 `cli/index_documents.py::create_vectorstore`,
   `rag_engine.py::_load_vectorstore`, settings vectorstore consumer까지 호출 그래프로 만든다.
3. `.github/workflows/ci.yml`의 hosted Python/frontend와 protected
   `m3-live-regression-gate` trigger/runner/environment를 동결 snapshot으로 남긴다.
4. filesystem, pickle, container build context/layers, CI artifact/run identity의 trust
   boundary와 공격·fault matrix를 Requirement ID에 연결한다.
5. root/coordinator가 만든 working-tree `scripts/orchestration_watchdog.py` readiness fix를
   tracked base `e57fe1c`와 대조한다. 현재 CLI `--help`에서 `task-list --run/--from`과
   `check --terminal/--run/--peek` 지원을 확인하고, exact argv assertion과 bound-Run
   dry-run 검증을 M4.3 구현 Task에 포함한다.

완료 조건:

- coordinator readiness fix가 덮어쓰기·누락되지 않고 provenance와 의도된 commit scope에
  기록된다.
- 모든 기존 index write/load, CI producer/consumer와 release blocker가 추적표에 있다.
- M4.3 deterministic PASS와 전체 M4 BLOCKED의 상태 모델이 명시된다.

관련 요구사항: M4.3-REQ-001~009

## 3. Phase 1 — executable prototype와 상세 설계 Gate

문서 설계 전에 production 코드와 분리된 작은 prototype/test로 다음을 검증한다.

1. 같은 filesystem에서 file/dir fsync와 temporary symlink 또는 pointer atomic replace,
   crash 지점별 이전 pointer 불변성
2. root/component/final-file symlink, non-regular file, owner/mode, TOCTOU를 load 전에
   차단하는 contained-open 방식
3. manifest canonicalization/version ID의 비순환성 및 100회 재직렬화 동일성
4. OCI archive outer path와 각 layer의 traversal/whiteout canonicalization, known forbidden
   fixture positive/negative control
5. fresh evidence directory, producer allowlist, same workflow/run/attempt/SHA binding,
   duplicate/missing/malformed evidence를 거부하는 assembler

상세 설계는 검증된 symbol, 상태 전이, syscall 순서, error/exit code와 test seam만 채택한다.
예상 파일은 `Design.md`이며 다음을 반드시 포함한다.

- index root/staging/version/current/lock layout과 permission model
- manifest JSON schema, version ID derivation과 legacy approved-hash source
- build/import/verify/activate/rollback/cleanup 상태 머신과 fsync 순서
- OCI stages/build context/runtime mounts/security options/layer scanner
- workflow DAG, artifact schema, assembler와 baseline state algebra
- deployment/incident/rollback runbook command flow

독립 설계 리뷰는 filesystem 원자성, pickle load-before-verify 여부, rollback 대칭성,
container COPY/layer 현실성, Actions `needs`/artifact provenance, M4.1 blocker 보존과 tests의
결함 검출력을 확인한다. MAJOR가 있거나 score가 9.7 미만이면 구현에 진입하지 않는다.

관련 요구사항: 전부

## 4. 구현 phases

### Phase 2 — manifest, contained verification과 legacy import

작업:

- canonical manifest/schema/version ID와 typed parser를 구현한다.
- operator-owned index root의 contained-open 및 pre-pickle 검증을 구현한다.
- committed M3 baseline hash만 허용하는 read-only legacy import를 staging에 구현한다.
- canonical round-trip, malformed/unknown/non-finite, symlink/owner/mode/hash/TOCTOU와
  `FAISS.load_local` 0-call negative tests를 추가한다.

완료 조건: Requirement의 manifest/legacy Gate 전부 PASS, M3 index/baseline bytes 변화 0.

Rollback: 새 loader/import 경계를 함께 revert한다. 검증을 건너뛰어 legacy direct-load로
돌아가는 fail-open rollback은 금지한다.

### Phase 3 — staging, activation, rollback과 retention

작업:

- operation staging, file/dir fsync, immutable version publish와 root lock을 구현한다.
- 단일 atomic pointer primitive로 activate와 rollback을 구현한다.
- dry-run-first cleanup과 protected-version retention을 구현한다.
- write/fsync/rename/disk-full/contention/crash fault injection 및 100회 activate/rollback을
  실행한다.

완료 조건: 모든 실패에서 pre/post current target와 bytes/hash가 같고 partial/dangling
pointer가 0이다. 성공 receipt는 parent fsync 이후에만 존재한다.

Rollback: 새 service는 검증된 이전 version pointer와 이전 image pair로만 되돌린다.
active directory에 직접 `save_local()`하는 경로는 복구 수단이 아니다.

### Phase 4 — lifecycle CLI, readiness와 deterministic receipt

작업:

- build/import/verify/activate/rollback/list/cleanup CLI와 안정된 exit/error schema를 만든다.
- startup loader가 current manifest/settings/hash를 검사하고 mismatch를 readiness 503의
  bounded reason으로 표현하게 한다.
- operation/result evidence writer와 negative-control parser를 구현한다.
- 기존 M4.2 lifecycle/readiness 우선순위, response/log/metric cardinality를 회귀 검증한다.

완료 조건: invalid artifact는 pickle load 및 service query 0회, receipt 실패는 전체
operation PASS가 아니며 기존 M4.2 suite가 통과한다.

### Phase 5 — OCI container와 layer scanner

작업:

- 명시적 test/production stage의 Linux CPU Dockerfile과 strict `.dockerignore`를 만든다.
- production runtime을 non-root, read-only rootfs, tmpfs, no-new-privileges, drop-all로
  실행한다.
- OCI archive/layer scanner와 traversal/whiteout/secret/runtime fixture tests를 만든다.
- 실제 Ollama 없이 config/import/live/ready/mock query/graceful stop smoke를 실행한다.

완료 조건: 일반 GitHub `ubuntu-latest`와 clean local Docker 환경에서 container Gate
전부 PASS, forbidden content 0. Ollama/model/corpus/index는 image에 포함하지 않는다.

Rollback: digest-pinned 이전 image로 복귀한다. `latest`, root 실행, writable rootfs 또는
security option 제거를 임시 rollback으로 허용하지 않는다.

### Phase 6 — deployment/recovery runbook과 drill

작업:

- `docs/operations/`에 internal deployment와 recovery runbook을 작성한다.
- digest/settings/lock/index identity preflight, deploy, readiness, backup/restore,
  activate/restart와 rollback 명령을 제공한다.
- 임시 root와 mock service로 정상 deploy/rollback 3회 및 manifest corruption, disk full,
  lock contention, readiness failure의 중단점을 실행한다.
- 실제 Ollama/native Linux 절차는 운영자용 명령으로만 기록하고 이 cycle에서 실행하지
  않았음을 명시한다.

완료 조건: 독립 운영자가 문서 순서로 mock drill을 재현하고 시작 image/index pair로
복귀한다. 검증 실패 뒤 후속 mutation은 0이다.

### Phase 7 — 단일 workflow와 fail-closed M4 assembly

작업:

1. 기존 `.github/workflows/ci.yml`의 hosted Python/frontend를 보존하고 `container`,
   `m43-deterministic`, `m4-assemble` hosted jobs를 추가한다.
2. 모든 producer artifact를 workflow/run/attempt/SHA/event에 bind하고 hash manifest와
   stable schema를 업로드한다.
3. assembler는 fresh empty directory에서 allowlisted artifact만 받아 exact-one producer,
   `needs` success, identity/hash/schema를 검사한다.
4. protected `m3-live-regression-gate` block의 trigger, runner labels, environment와 approval
   계약에 semantic diff가 없음을 정적 test로 검증한다.
5. genuine successful deterministic receipt를 변조하는 negative control과 skipped/missing/
   cross-run/duplicate/path traversal cases를 같은 parser로 실패시킨다.

완료 조건: PR hosted Gate와 exact merge SHA hosted Gate가 각각 같은 workflow/run 안에서
완결된다. live job이 skipped/미실행이면 baseline operational 상태는 BLOCKED다.

관련 요구사항: M4.3-REQ-004~009

### Phase 8 — clean 검증과 M4 baseline candidate

clean checkout/locked venv에서 최소 다음을 실행한다(구현 설계에서 실제 runner 경로와
옵션을 확정한다).

```bash
bash scripts/compile_lock.sh --verify
python -m pip check
python -m pytest -q
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
npm ci
npm test
npm run sync-vendor
git diff --exit-code -- web/static/vendor/
python scripts/generate_field_spec.py --check
python scripts/logging_callsite_audit.py --check
python scripts/check_markdown_links.py
python -m compileall -q src scripts tests evaluation
python scripts/run_m42_acceptance.py --profile deterministic --repeat 10 --seed 4202 --output <tmp>/m42.json
python scripts/run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303 --output <tmp>/m43.json
python scripts/run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303 --inject-evidence-mismatch --output <tmp>/m43-negative.json
docker build --target test -f deploy/Dockerfile .
docker build --target production -f deploy/Dockerfile -t simple-qna-rag:m43-candidate .
python scripts/scan_image_layers.py --image simple-qna-rag:m43-candidate
python scripts/assemble_m4_evidence.py --fresh-dir <tmp>/m4-assemble --expected-sha "$(git rev-parse HEAD)"
python scripts/check_m4_baseline.py --candidate <tmp>/m4-assemble/m4-baseline.json --expect-operational-blocked
git diff --check
```

`run_m43_acceptance.py`, `scan_image_layers.py`, `assemble_m4_evidence.py`와
`check_m4_baseline.py`는 계획된 파일이며 Design에서 이름을 확정한다. negative command는
exit 1과 retained failure evidence가 기대 성공이다. live/Ollama/M3 live는 실행하지 않는다.

baseline candidate는 다음을 동시에 만족해야 한다.

- `deterministic_status=PASS`
- `M4.1_BLOCKED=true`
- protected live Gate 상태 `NOT_RUN` 또는 실제 receipt 기반 상태
- `operational_status=BLOCKED`
- `overall_release_ready=false`

## 5. 독립 code review와 Gate

fresh Codex reviewer는 전체 diff/current code를 다음 순서로 검토한다.

1. Requirement/Design/Traceability와 범위 경계
2. manifest/parser 및 pickle/filesystem trust boundary
3. fsync/atomic replace/lock/rollback/retention 실패 원자성
4. container context/layer/runtime 최소 권한
5. CI DAG/evidence identity와 synthetic-PASS 방지
6. runbook 중단점과 image/index pair rollback
7. M4.1 blocker/protected gate 불변 및 M3/M4.2 회귀

각 발견은 severity, `file:line`, 재현 명령, Requirement ID, pre/post-merge Gate를
기록한다. MAJOR가 있으면 개선 Task와 새 독립 iteration을 수행한다. 최종
Pre-merge Code Quality Gate는 CRITICAL/MAJOR 0, score 9.7 이상, deterministic tests와
negative controls 전부 PASS일 때만 닫는다.

## 6. Pre-merge와 post-merge 실행 분리

### Pre-merge

- local/clean deterministic suite와 mock rollback drill
- PR의 일반 hosted Python/frontend/container/m43-deterministic/m4-assemble
- workflow/protected-job 불변 정적 검사
- post-merge checker와 assembler 자체의 fail-closed 검증

### Post-merge

- `master` exact merge SHA의 동일 hosted workflow Gate와 artifact provenance
- M4 baseline candidate 생성 및 deterministic 상태 확정
- 별도 protected M3 live/M4.1 operational receipt 확인

현재 강제 제외 때문에 마지막 protected 항목은 수행하지 않는다. 따라서 구현이 향후
승인·merge되고 hosted Gate가 PASS하더라도 coordinator는 M4.3 deterministic PASS와
`overall_release_ready=false`를 함께 보고해야 한다. 전체 M4 완료, Roadmap 완료 처리와
release-ready 선언은 금지한다.

## 7. 위험과 복구

| 위험 | 통제/복구 |
|---|---|
| manifest/version ID 자기참조 | identity 대상 필드와 실행 metadata 분리, canonical round-trip test |
| pointer는 바뀌었지만 durable하지 않음 | target 검증 후 atomic replace와 parent fsync, 그 뒤 receipt |
| pickle path/TOCTOU | approved root contained-open/fstat/hash 후 load, symlink 전수 negative test |
| rollback 대상과 image/settings 불일치 | release identity bundle 검증 실패 시 mutation 중단 |
| container에 runtime/secret 포함 | strict ignore + 모든 OCI layer scan + known fixture control |
| CI artifact 합성/교차 run | exact-one producer, needs success, run/attempt/SHA/schema/hash binding |
| protected live gate 약화 | workflow block snapshot/semantic test; 변경 발견 시 Gate FAIL |
| M4.1 blocker 은폐 | baseline typed state와 `overall_release_ready=false` invariant |
| watchdog readiness fix 누락/오귀속 | coordinator provenance, exact argv tests, bound-Run dry-run, 독립 review와 commit scope 포함 |

## 8. 완료 조건

M4.3 deterministic cycle은 Traceability의 M4.3 필수 행이 PASS이고 pre-merge/hosted
deterministic evidence가 완결된 경우에만 완료할 수 있다. 전체 M4 release는 M4.1 운영
blocker 및 protected live Gate가 실제 receipt로 해소되거나 사용자가 별도 release-risk를
명시 승인하기 전까지 완료할 수 없다.
