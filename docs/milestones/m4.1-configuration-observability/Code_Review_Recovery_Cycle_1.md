# M4.1 구현 독립 코드 리뷰 — Recovery Cycle 1

검토일: 2026-08-09  
검토자: Codex (fresh independent recovery reviewer)

이번 리뷰는 현재 dirty worktree 전체와 CR-I5-MAJ-01/02/03 recovery 변경을
독립 검토했다. 기존 `Code_Review_Iteration_1.md`부터
`Code_Review_Iteration_5.md`까지는 중단/감사 기록으로 보존했으며 수정하지 않았다.
구현 파일도 수정하지 않았고 이 문서만 추가했다. 검토 대상에는 전체 `git diff`,
untracked 구현/테스트/문서, `.github/workflows/ci.yml`,
`scripts/ci_acceptance_contract.py`, 관련 테스트, `requirements.lock`,
`CI_Acceptance_Runbook.md`, `Traceability.md`가 포함된다.

## 1. Gate 판정

**FAIL — pre-merge 코드 품질 Gate 미통과, Git 단계 진입 불가**

- 점수: **9.5 / 10.0**
- 발견사항: **CRITICAL 0 / MAJOR 1 / MINOR 0 / TRIVIAL 0**
- 요구 Gate(CRITICAL/MAJOR 0, score >= 9.7, 필수 테스트 전부 PASS): **미충족**
- pre-merge 판정: **FAIL**
- commit/push/PR 단계 진입: **불허**

CR-I5-MAJ-01과 CR-I5-MAJ-03은 폐쇄됐다. CR-I5-MAJ-02도 receipt의 고정
3-job/2-artifact profile, 누락/중복 상수, 명시적 빈 event 및 provisioning의
동일 `Path` 값 거부까지는 폐쇄됐지만, 서로 다른 경로 문자열이 같은 실제
디렉터리를 가리키는 alias를 허용해 provisioning의 핵심 두-directory 계약이
아직 fail-open이다.

## 2. 발견사항

### CRITICAL

없음.

### MAJOR

#### CR-R1-MAJ-01 — provisioning의 중복 경로 검사가 filesystem identity alias를 허용한다

`scripts/ci_acceptance_contract.py:211`은
`len(set(external_dirs.values()))`로 두 `Path` 객체의 lexical equality만 검사한다.
따라서 `vectorstore=/tmp/.../vectorstore`와
`documents=/tmp/.../documents`가 둘 다 같은 `/tmp/.../shared` 디렉터리를 가리키는
symlink여도 서로 다른 값으로 간주된다. 함수는 runner/environment 검사와 두 dir
검사를 모두 호출하고 성공 경로로 진행한다. CLI도 `_parse_external_dir()`에서
`Path(raw_path)`만 만들고 동일 함수로 넘기므로 direct-call과 CLI 양쪽에 같은
fail-open이 존재한다.

독립 adversarial 실행에서 같은 실제 디렉터리를 가리키는 두 symlink를 전달했고
`ALIAS_ACCEPTED [...]`가 출력되어 두 검사 호출 및 무예외 반환을 재현했다. 이는
함수 docstring의 “서로 다른 이름이 동일 경로를 가리키지 않는지”와 M4.1의 외부
`vectorstore`/`documents` 두 디렉터리 증거를 만족하지 않는다. 특히 operator가
잘못 provision한 symlink/mount alias를 acceptance checker가 정상 profile로 승인할
수 있으므로 CR-I5-MAJ-02는 완전 폐쇄로 볼 수 없다.

폐쇄 조건은 존재하는 디렉터리를 검사하는 시점에 `Path.resolve(strict=True)` 또는
동등한 canonical filesystem identity로 중복을 거부하고, 같은 target을 가리키는
symlink alias에 대한 direct-call 및 실제 CLI argv 음성 테스트를 추가하는 것이다.
플랫폼에서 bind mount까지 동일 identity로 취급해야 한다면 `os.stat()`의
`(st_dev, st_ino)` 비교도 계약에 포함해야 한다.

### MINOR

없음.

### TRIVIAL

없음.

## 3. CR-I5 항목별 폐쇄 감사

| 항목 | 판정 | 독립 확인 |
|---|---|---|
| CR-I5-MAJ-01 | **폐쇄** | `require_job_conclusions`와 `--allow-job-conclusion JOB=failure`가 실제 run/job 모두 `failure`인 경로에서도 지정 job만 허용하고 artifact를 계속 검증한다. 미지정 job은 계속 `success`만 허용하며 unknown/empty mapping도 거부한다. 함수 및 실제 CLI argv 양성/음성 테스트가 존재하고 통과했다. |
| CR-I5-MAJ-02 | **부분 폐쇄 / 재개방** | `check_m41_receipt()`/`receipt-m41`이 3 job과 2 artifact를 caller가 축소하지 못하게 고정하고, 상수 중복 없음·필수 항목 누락·명시적 `expected_events=set()` 거부가 검증된다. provisioning은 키 누락/추가와 완전히 같은 `Path` 값은 거부하지만 filesystem alias 중복을 허용하므로 CR-R1-MAJ-01이 남는다. |
| CR-I5-MAJ-03 | **폐쇄** | canonical lock은 102 packages이며 현재 `starlette==1.6.0`; 2회 compile 기반 `--verify`가 PASS했다. 새 venv의 hash-locked install, editable no-deps install, `pip check`, targeted tests가 모두 exit 0였다. |

## 4. 독립 실행 검증

| 명령 | 결과 |
|---|---|
| `venv/bin/python -m pytest -q tests/unit/test_ci_acceptance_contract.py tests/unit/test_dependency_lock.py` | **PASS — 85 passed** |
| `venv/bin/python -m pytest -q` | **PASS — 974 passed, 1 skipped, 1 warning** |
| `bash scripts/compile_lock.sh --verify` | **PASS — 102 packages, 두 resolution 재현 가능, drift 없음** |
| 새 `python3 -m venv` → `pip install --require-hashes -r requirements.lock` → `pip install -e . --no-deps` → `pip check` → targeted tests | **PASS — 모든 명령 exit 0, targeted 85 passed** |
| `venv/bin/python scripts/check_markdown_links.py` | **PASS — 76 files, 323 links, failures 0** |
| `git diff --check` | **PASS — 출력 없음** |
| symlink alias adversarial provisioning (`vectorstore`와 `documents`가 같은 `shared` target) | **FAIL-OPEN 재현 — `ALIAS_ACCEPTED`, 예외 없이 두 dir 검사 호출** |

전체 suite와 clean locked validation은 강한 회귀/lock 증거이며 CR-I5-MAJ-01/03의
폐쇄를 뒷받침한다. 그러나 필수 적대적 provisioning 검증이 실패했으므로 “모든 필수
테스트 PASS” 조건은 충족되지 않는다.

## 5. pre-merge 품질과 post-merge 운영 증거 구분

| 구분 | 현재 상태 | 의미 |
|---|---|---|
| pre-merge 코드 품질 | **FAIL** | CR-R1-MAJ-01이 남아 Gate 조건(CRITICAL/MAJOR 0, 9.7+)을 충족하지 못한다. |
| Git 단계 진입 | **불허** | 구현 수정, adversarial 테스트 추가, fresh 독립 재리뷰 PASS 전 commit/push/PR 금지다. |
| post-merge hosted CI | **NOT_RUN** | merge SHA의 locked install, `pip check`, lock verify, hosted jobs와 artifact receipt는 로컬 검증과 별개의 원격 증거다. |
| post-merge live provisioning | **NOT_PROVISIONED / NOT_VERIFIED** | required-reviewer environment, online `ollama-m3` runner, 실제로 분리된 read-only 외부 두 디렉터리는 원격 운영 환경에서 증명해야 한다. |
| post-merge live receipt | **NOT_RUN** | merge SHA/event/branch/workflow, 세 job·두 artifact 및 의도적 failed-job의 `if: always()` artifact receipt가 필요하다. |

post-merge 운영 증거가 아직 없는 사실 자체는 이번 pre-merge 점수 감점 사유가 아니다.
반대로 CR-R1-MAJ-01은 원격 환경 없이 로컬에서 결정적으로 재현되는 acceptance contract
결함이므로 pre-merge 품질 Gate에 포함한다.

## 6. 재진입 조건

1. provisioning 경로를 canonical filesystem identity 기준으로 비교해 symlink/path
   alias 중복을 함수 경계에서 fail-closed로 거부한다.
2. direct-call과 CLI argv 모두에 alias 중복 음성 테스트를 추가한다.
3. targeted/full suite, lock verify, 새 clean hash install + `pip check`, markdown link,
   `git diff --check`를 다시 통과한다.
4. fresh 독립 리뷰에서 CRITICAL/MAJOR 0, MINOR 최소화, score >= 9.7을 받아야만 Git
   단계에 진입한다.
5. 이번 리뷰에서는 commit/push/PR을 수행하지 않았다.
