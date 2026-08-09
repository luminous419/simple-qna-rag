# M4.1 구현 독립 코드 리뷰 — Iteration 5

검토일: 2026-08-09  
검토자: Codex (독립 구현 Gate 리뷰어)

검토 범위는 `milestone_dev_orchestration_guide.md`, M4.1 구현 문서와
`Code_Review_Iteration_4.md`, 최신 `scripts/ci_acceptance_contract.py`,
`tests/unit/test_ci_acceptance_contract.py`, `CI_Acceptance_Runbook.md`,
`Traceability.md`, `requirements.txt`/`requirements.lock`, CI workflow 및 현재
tracked/untracked 전체 diff이다. 제품 코드와 구현 문서는 수정하지 않았고 이 리뷰
문서만 추가했다.

## 1. Gate 판정

**FAIL — 코드 품질 Gate 미통과, 다음 Git 단계 진입 불가**

- 점수: **9.1 / 10.0**
- 발견사항: **CRITICAL 0 / MAJOR 3 / MINOR 0 / TRIVIAL 0**
- 9.7 Gate: **미통과**
- 다음 단계 진입: **거절**. 아래 세 MAJOR를 수정하고 clean locked 검증 및 독립
  재리뷰를 통과해야 한다.
- 사용자 결정 필요: **없음**. 기존 요구사항을 구현하는 범위의 결정적 계약/lock
  정합성 수정이다.

CR-I4-MAJ-01/02의 일부 방어는 실제로 추가됐다. 빈 conclusion 문자열은 CLI에서
exit 2로 거부되고 `--skip-conclusion`과 상호 배타이며, run provenance의 SHA/branch/
event/workflow path 및 비어 있는 job/artifact 목록은 함수 경계에서도 거부된다.
그러나 실패-run 경로와 고정 profile의 direct-call parity는 아직 닫히지 않았다.

## 2. 발견사항

### CRITICAL

없음.

### MAJOR

#### CR-I5-MAJ-01 — `--skip-conclusion`으로도 실패한 live job의 always artifact를 검증할 수 없다

`check_run_receipt()`는 전체 run conclusion만 선택적으로 생략하고, 모든
`require_jobs`의 conclusion은 여전히 무조건 `success`여야 한다. Runbook §4 단계 6은
의도적으로 실패시킨 run에 `--skip-conclusion --require-job
m3-live-regression-gate`를 사용하라고 하지만, 그 live job이 실제로 실패하면 artifact
조회 전에 job conclusion 검사에서 exit 1이다. 따라서 CR-I4-MAJ-01의 핵심 목적인
실패 run의 `if: always()` artifact receipt는 문서의 명령으로 폐쇄할 수 없다.

55개 테스트 중 실패 run 테스트는 run conclusion만 `failure`이고 job conclusion은
`success`인 비현실적 조합이라 이 결함을 놓쳤다. 실패-artifact 모드에서는 지정 job의
존재/실행을 검증하되 허용 conclusion을 명시적으로 별도 계약화하거나, job conclusion
검사를 선택적으로 생략하는 좁은 플래그와 실제 CLI 회귀 테스트가 필요하다.

#### CR-I5-MAJ-02 — M4.1 고정 profile이 CLI에만 있고 direct-call 계약은 fail-open이다

CLI는 external dir 이름 `vectorstore`/`documents`를 검사하지만
`run_provisioning()`은 이름을 받지 않고 `list[Path]` 길이만 2 이상인지 본다. 독립
실행에서 같은 경로 `/same`을 두 번 넘겨도 runner/environment/두 dir 검사를 모두
호출하고 성공할 수 있음을 재현했다. 즉 함수 docstring과 Traceability의
“CLI를 우회해도 동일하게 fail-closed” 주장은 사실이 아니다.

같은 문제가 receipt profile에도 남아 있다. 함수와 CLI 모두 job 하나와 artifact
하나만 있으면 필수 목록으로 인정하므로, Runbook이 최종 receipt에 요구하는 세 job
(`python-tests`, `frontend-tests`, `m3-live-regression-gate`)과 두 artifact
(`dependency-snapshot`, `m4-regression-report`) 중 임의의 부분집합으로도 통과한다.
`expected_events=set()` 역시 함수에서 기본 허용 집합으로 되돌아간다. 고정 M4.1
profile을 이름 있는 구조로 함수 경계에 정의하고 CLI가 동일 구조를 전달해야 하며,
중복/누락/빈 event까지 함수 테스트로 거부해야 한다.

#### CR-I5-MAJ-03 — committed lock drift가 현재 CI `python-tests`를 결정적으로 실패시킨다

`bash scripts/compile_lock.sh --verify`를 두 차례 resolution 경로로 재실행한 결과 매번
**FAIL**했고, 생성 lock과 committed lock의 유일한 본문 차이는
`starlette==1.5.0`에서 `starlette==1.5.1` 및 해당 hash 변경이었다. 따라서 새로 보고된
drift는 실제 재현된다. `requirements.txt`가 Starlette를 직접 선언한 것은 아니지만
unbounded transitive resolution을 canonical lock으로 다시 컴파일하는 현재 계약에서는
upstream 릴리스가 곧 drift가 된다.

CI workflow의 hosted `python-tests`가 locked install 뒤
`scripts/compile_lock.sh --verify`를 필수 실행하므로, 이 상태로 Git 단계에 들어가면
post-merge 외부 인프라와 무관하게 hosted CI가 결정적으로 실패한다. 수정 최소 범위는
현재 의존 제약을 유지한다면 `requirements.lock` 재생성과 clean hash install,
`pip check`, 전체 suite, lock verify 재실행이다. 재발 방지를 위해 Starlette를 직접
pin할지는 별도 유지보수 선택이지만, 이번 Gate를 닫기 위해 `requirements.txt` 변경은
필수는 아니다.

### MINOR

없음.

### TRIVIAL

없음.

## 3. 독립 실행 검증

| 검증 | 결과 |
|---|---|
| `venv/bin/python -m pytest -q tests/unit/test_ci_acceptance_contract.py` | **55 passed** |
| `venv/bin/python -m pytest -q` | **928 passed, 1 skipped, 1 warning** |
| `bash scripts/compile_lock.sh --verify` | **FAIL — committed lock drift** |
| 별도 `uv pip compile` 결과와 lock diff | **Starlette 1.5.0 → 1.5.1 및 hash만 변경** |
| direct `run_provisioning(..., external_dirs=[Path('/same'), Path('/same')])` | **통과 경로 재현 — 동일 경로 두 번을 고정 profile로 오인** |
| 실패-run 테스트 구조 검토 | run=`failure`, required job=`success`; 실제 failed job 음성 경로 없음 |

전체 928 tests는 사실이며 일반 제품 회귀가 없다는 강한 증거다. 그러나 테스트 개수는
acceptance 계약의 빠진 상태 조합과 lock drift를 상쇄하지 않는다.

## 4. 코드 품질 Gate와 post-merge 외부 acceptance 분리

| Gate | 현재 판정 | 폐쇄 조건 |
|---|---|---|
| pre-Git 코드 품질 | **FAIL** | CR-I5-MAJ-01/02 수정, lock 재생성, targeted/full/clean-lock 검증, 독립 9.7 리뷰 |
| Git 단계 진입 | **보류** | pre-Git 코드 품질 PASS 후에만 가능 |
| post-merge hosted CI | **NOT_RUN** | merge SHA에서 locked install, lock verify, `pip check`, `python-tests`/`frontend-tests`, dependency artifact 성공 |
| post-merge live provisioning | **NOT_PROVISIONED** | required reviewer가 있는 environment, online `ollama-m3` runner, 두 외부 경로 read-only |
| post-merge live receipt | **NOT_RUN** | merge SHA/event/branch/workflow, 세 job, 두 artifact 및 실패-run always artifact를 수정된 계약으로 검증 |

실제 Actions receipt의 부재는 여전히 로컬 dispatch에서 만들 수 없는 외부 Gate이며 그
자체를 코드 품질 감점으로 계산하지 않았다. 반대로 lock verify 실패는 GitHub 환경을
기다려야 알 수 있는 문제가 아니라 현재 로컬에서 결정적으로 재현되는 CI 코드 품질
결함이므로 pre-Git Gate에 포함한다.

## 5. 재진입 조건

1. 실패한 required job을 포함한 run에서도 `if: always()` artifact를 검증할 수 있게
   receipt 모드를 고치고 실제 CLI argv 회귀 테스트를 추가한다.
2. provisioning/receipt의 M4.1 고정 profile을 함수 경계에도 적용해 CLI/direct-call
   parity와 필수 provenance를 음성 테스트로 증명한다.
3. `requirements.lock`을 현재 canonical resolution으로 재생성하고 clean
   `--require-hashes` install, `pip check`, lock verify와 전체 suite를 통과시킨다.
4. markdown link와 `git diff --check`를 통과한 뒤 fresh 독립 리뷰에서 CRITICAL/MAJOR
   0, 9.7 이상을 받아야 다음 단계에 진입한다.
5. commit/push/PR은 이번 리뷰에서 수행하지 않았다. pre-Git PASS 뒤 별도 Git 단계와
   post-merge 외부 acceptance를 순서대로 수행한다.
