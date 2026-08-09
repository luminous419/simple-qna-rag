# M4.1 구현 독립 코드 리뷰 — Recovery Cycle 2

검토일: 2026-08-09  
검토자: Codex (fresh independent recovery reviewer)

이번 리뷰는 현재 dirty worktree의 구현과 Recovery Cycle 1 이후 변경을 독립
검토했다. 기존 `Code_Review_Iteration_1.md`부터
`Code_Review_Iteration_5.md`, `Code_Review_Recovery_Cycle_1.md`까지는 감사 기록으로
보존했으며 수정하지 않았다. 구현 파일도 수정하지 않았고 이 문서만 추가했다.
주 검토 범위는 `scripts/ci_acceptance_contract.py`의 canonical filesystem
identity 검사, direct-call/실제 CLI argv 회귀 테스트, CR-I5-MAJ-01/02/03의 기존
폐쇄 상태, lock과 전체 회귀 상태다.

## 1. Gate 판정

**PASS — pre-merge 코드 품질 Gate 통과, Git 단계 진입 허용**

- 점수: **9.9 / 10.0**
- 발견사항: **CRITICAL 0 / MAJOR 0 / MINOR 0 / TRIVIAL 0**
- 요구 Gate(CRITICAL/MAJOR 0, MINOR 최소화, score >= 9.7, 필수 테스트 전부
  PASS): **충족**
- pre-merge 판정: **PASS**
- commit/push/PR 단계 진입: **허용**. 단, 이번 리뷰에서는 어느 Git 쓰기 단계도
  수행하지 않았다.

CR-R1-MAJ-01은 폐쇄됐다. `Path.resolve(strict=True)`와 `os.stat()`의
`(st_dev, st_ino)`를 함께 사용하는 구현은 lexical path가 다른 symlink, nested
symlink chain, 같은 inode를 드러내는 mount alias를 동일 filesystem identity로
판정한다. 이 판정은 runner/environment `gh api` 호출보다 먼저 수행되어 구조적
오류를 외부 상태와 무관하게 fail-closed로 거부한다. direct-call과 CLI는 동일한
`run_provisioning()` 경계를 통과하므로 두 표면의 정책 차이도 없다.

## 2. 발견사항

### CRITICAL

없음.

### MAJOR

없음.

### MINOR

없음.

### TRIVIAL

없음.

## 3. CR-R1-MAJ-01 폐쇄 감사

| 검토 항목 | 판정 | 독립 확인 |
|---|---|---|
| 서로 다른 symlink → 같은 target | **폐쇄** | direct-call과 실제 CLI argv 모두 `distinct filesystem identities`로 exit/failure 처리했다. 구조 검사는 `gh api`보다 먼저 실행된다. |
| nested symlink alias | **폐쇄** | symlink가 다른 symlink를 거쳐 실제 target에 도달하는 adversarial direct-call이 동일 identity로 거부됐다. `strict=True` resolve가 전체 chain을 푼다. |
| canonical filesystem identity | **폐쇄** | resolve 결과를 문자열로만 비교하지 않고 `(st_dev, st_ino)`로 비교하므로 동일 inode의 alias도 닫는다. |
| 존재하지 않는 경로 | **회귀 없음** | `_external_dir_identity()`가 `FileNotFoundError`를 `ContractError`의 `does not exist`로 변환하며 direct-call과 CLI가 모두 fail-closed였다. |
| resolve 권한 오류 | **회귀 없음** | 독립 monkeypatch adversarial에서 `PermissionError`를 주입했고 direct-call과 CLI 모두 `could not be resolved` `ContractError`/exit 1로 변환했다. |
| 정상 distinct directory | **회귀 없음** | 서로 다른 두 실제 디렉터리, online runner 응답, required reviewer environment 응답, read-only 판정을 구성해 direct-call과 실제 CLI argv가 모두 정상 통과했다. |

identity 검사와 마지막 read-only 검사 사이에 filesystem이 공격적으로 바뀌는 일반적인
TOCTOU 가능성은 pathname 기반 운영 검사 전체의 한계지만, 이번 결함의 재현 조건인
안정된 symlink/nested alias를 fail-open으로 남기지 않는다. 현재 계약 범위에서 별도
MINOR로 분류할 회귀나 우회는 발견하지 못했다.

## 4. CR-I5 항목별 회귀 감사

| 항목 | 판정 | 독립 확인 |
|---|---|---|
| CR-I5-MAJ-01 | **폐쇄 유지** | `require_job_conclusions`/`--allow-job-conclusion`은 지정한 failed job만 허용하고 다른 job의 기본 `success` 계약은 유지한다. 관련 targeted tests와 전체 suite가 통과했다. |
| CR-I5-MAJ-02 | **폐쇄 유지** | provisioning의 정확한 두 이름 profile, lexical 중복, canonical identity 중복이 함수 경계에서 거부된다. `check_m41_receipt`/`receipt-m41`의 고정 3-job/2-artifact profile, 빈 event 및 누락/중복 방어도 targeted suite에서 통과했다. |
| CR-I5-MAJ-03 | **폐쇄 유지** | canonical lock은 102 packages로 두 번 동일 resolution을 만들었고 committed lock drift가 없었다. 새 venv의 hash-locked install, editable `--no-deps` install, `pip check`, targeted tests가 모두 통과했다. |

## 5. 독립 실행 검증

| 명령/시나리오 | 결과 |
|---|---|
| `venv/bin/python -m pytest -q tests/unit/test_ci_acceptance_contract.py tests/unit/test_dependency_lock.py` | **PASS — 90 passed** |
| `venv/bin/python -m pytest -q` | **PASS — 979 passed, 1 skipped, 1 warning** |
| `bash scripts/compile_lock.sh --verify` | **PASS — 102 packages, 두 resolution 재현 가능, drift 없음** |
| 새 venv → `pip install --require-hashes -r requirements.lock` → `pip install -e . --no-deps` → `pip check` → targeted tests | **PASS — broken requirement 없음, 90 passed** |
| `venv/bin/python scripts/check_markdown_links.py` | **PASS — 리뷰 문서 추가 전 77 files, 326 links, failures 0; 추가 후 재검증도 PASS** |
| `git diff --check` | **PASS — 출력 없음** |
| symlink/nested alias adversarial | **PASS — direct-call 및 CLI fail-closed** |
| missing/권한 오류 adversarial | **PASS — direct-call 및 CLI fail-closed** |
| 정상 distinct directory adversarial | **PASS — direct-call 및 CLI 성공** |

전체 suite의 단일 warning은 기존 `tests/unit/test_routing_signals.py`의 class-scoped
fixture에 대한 pytest 10 deprecation warning이며 이번 filesystem identity 변경의
실패나 새 회귀가 아니다.

## 6. pre-merge 품질과 post-merge 운영 증거 구분

| 구분 | 현재 상태 | 의미 |
|---|---|---|
| pre-merge 코드 품질 | **PASS** | CRITICAL/MAJOR/MINOR 0, 9.9점, 필수 로컬 검증 전부 PASS다. |
| Git 단계 진입 | **허용** | 별도 소유자가 commit/push/PR 절차로 진행할 수 있다. 이번 리뷰어는 이를 수행하지 않았다. |
| post-merge hosted CI | **NOT_RUN** | merge SHA의 locked install, `pip check`, lock verify, hosted job과 artifact receipt는 로컬 Gate와 별개의 원격 증거다. |
| post-merge live provisioning | **NOT_PROVISIONED / NOT_VERIFIED** | required-reviewer environment, online `ollama-m3` runner, runner service account 기준 read-only이면서 filesystem identity가 다른 외부 두 디렉터리를 운영 환경에서 증명해야 한다. |
| post-merge live receipt | **NOT_RUN** | merge SHA/event/branch/workflow, 세 job·두 artifact, 의도적 failed-job의 `if: always()` artifact receipt가 필요하다. |

post-merge 운영 증거가 아직 없다는 사실은 이번 pre-merge 코드 품질 PASS를 뒤집지
않는다. 반대로 이 PASS를 hosted/live 운영 acceptance 완료로 해석해서도 안 된다.
Git 단계 이후에는 [CI Acceptance Runbook](CI_Acceptance_Runbook.md)의 순서와 고정
profile을 사용해 실제 원격 증거를 별도로 수집해야 한다.

## 7. 결론

CR-R1-MAJ-01의 canonical filesystem identity fix는 요구된 symlink/nested alias와
direct-call/CLI 경로를 모두 fail-closed로 닫았고, missing/권한 오류 및 정상 distinct
directory 동작에도 회귀가 없다. CR-I5-MAJ-01/02/03의 기존 폐쇄도 유지된다. 따라서
pre-merge Gate는 PASS이며 Git 단계 진입을 허용한다. 이번 리뷰에서는 구현 파일 수정,
commit, push, PR을 수행하지 않았다.
