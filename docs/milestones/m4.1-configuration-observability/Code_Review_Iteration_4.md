# M4.1 구현 독립 코드 리뷰 — Iteration 4

검토일: 2026-08-09  
검토자: Codex (독립 구현 Gate 리뷰어)

검토 범위는 `milestone_dev_orchestration_guide.md`, M4.1
`Requirement.md`/`Plan.md`/`Design.md`/`Traceability.md`,
`Code_Review_Iteration_3.md`, `CI_Acceptance_Runbook.md`,
`scripts/ci_acceptance_contract.py`, `tests/unit/test_ci_acceptance_contract.py`와
최신 tracked/untracked 전체 diff이다. 제품 코드와 구현 원문은 수정하지 않았고 이
리뷰 문서만 추가했다.

## 1. Gate 판정

**FAIL — 코드 품질 Gate 미통과, 현재 Git 단계 진입 불가**

- 점수: **9.4 / 10.0**
- 발견사항: **CRITICAL 0 / MAJOR 2 / MINOR 0 / TRIVIAL 0**
- 9.7 Gate: **미통과**
- 통합/Git 단계 진입: **현재 거절**. CR-I4-MAJ-01/02를 수정하고 독립 재리뷰를
  통과하면 진입할 수 있다.
- 사용자 결정 필요: **없음**. 제품 요구사항의 선택 문제가 아니라 acceptance
  계약의 결정적 구현 누락이다. 다만 Git 반영 후 environment/runner/data
  provisioning과 environment 승인은 저장소 운영자의 실행 권한이 반드시 필요하다.

CR-I3-MAJ-01의 실제 Actions receipt는 현재 로컬 변경만으로 생성할 수 없는
**post-merge 외부 인프라 Gate**라는 판단이 맞다. 따라서 receipt의 현재 부재 자체를
pre-Git 코드 품질 결함으로 계속 계산해서는 안 된다. 그러나 이번에 추가된 acceptance
contract/runbook은 그 post-merge Gate를 아직 충분히 fail-closed로 집행하지 못하므로,
도구 결함 두 건이 별도의 코드 품질 MAJOR다.

## 2. 발견사항

### CRITICAL

없음.

### MAJOR

#### CR-I4-MAJ-01 — 실패-run artifact 검증용 CLI 계약이 문서대로 동작하지 않는다

Runbook §2.2/§4는 `--require-conclusion ""`이 전체 run conclusion 검사를
비활성화해 실패 run의 `if: always()` artifact만 검증한다고 명시한다. 그러나 CLI는
빈 문자열을 그대로 `check_run_receipt()`에 전달하고, 구현은 `None`일 때만 검사를
생략한다. 실제 과거 성공 run에 빈 문자열을 전달한 결과도
`conclusion='success', expected ''`로 exit 1이었다.

단위 테스트는 Python API에 직접 `None`을 넘기는 경로만 검사하고 실제 CLI argv의
빈 문자열 변환을 검사하지 않아 이 불일치를 놓쳤다. 명시적인 `--skip-conclusion`
플래그 또는 빈 문자열→`None` 정규화와 CLI 회귀 테스트가 필요하다. 수정 전에는
CR-I3-MAJ-01이 요구한 실패 run artifact receipt를 Runbook 명령으로 폐쇄할 수 없다.

#### CR-I4-MAJ-02 — provisioning/receipt가 필수 증거를 생략하거나 다른 run으로 대체할 수 있다

현재 CLI의 `--external-dir`, `--require-job`, `--require-artifact`는 모두 기본값이
빈 목록이다. 따라서 online runner와 environment만 존재하면 외부 디렉터리를 하나도
검사하지 않은 `provisioning OK`가 가능하고, 성공 conclusion인 임의 run은 job과
artifact를 하나도 검사하지 않은 `receipt OK`가 가능하다. 이는 “단일 실행 가능한
CR-I3-MAJ-01 계약”의 fail-closed 주장과 모순된다.

또한 receipt는 run의 `event`, `head_branch`, `head_sha`, workflow 파일/path를
검증하지 않는다. 호출자가 run ID를 잘못 선택해도 이름이 맞는 과거/다른 ref의 run을
현재 merge receipt로 받아들일 수 있다. 특히 `workflow_dispatch` workflow는 GitHub
기본 브랜치에 workflow가 존재해야 수동 dispatch가 가능하고, 현재 기본 브랜치는
`master`이지만 원격 `master`의 workflow blob에는 신규 live job이 없다. Git 반영 후
영수증은 최소한 기대 event(`push` 또는 `workflow_dispatch`), `master`, merge SHA,
필수 세 job(`python-tests`, `frontend-tests`, `m3-live-regression-gate`)과 필수 artifact를
한 번에 묶어 검증해야 한다. provisioning도 두 외부 경로를 필수화하거나 M4.1 전용
고정 profile을 제공하고 required-reviewer rule의 실제 reviewer가 비어 있지 않음을
검사해야 한다.

### MINOR

없음.

### TRIVIAL

없음.

## 3. CR-I3-MAJ-01 구조적 불가능성 재검증

2026-08-09 실제 `gh`/Git 상태는 Runbook의 핵심 결론과 일치한다.

| 항목 | 실제 상태 | 판정 |
|---|---|---|
| 로컬 HEAD / `origin/master` | 둘 다 `c056342b5dd3fe85b8ea42fe78c24e6c3e0417d3` | 이번 dirty 변경은 원격 ref에 없음 |
| 기본 브랜치 | `master` | `workflow_dispatch`의 기본 workflow 존재 계약 대상 |
| workflow blob | 원격 `e76b6298...`, 로컬 `c0d0ec67...` | 원격 workflow는 신규 live job을 모름 |
| GitHub Environment | `total_count=0` | `m3-live-regression` 및 required reviewer 없음 |
| self-hosted runner | `total_count=0` | `ollama-m3` runner 없음 |
| 최근 원격 CI | 최신 run `31204494509`, 기존 `python-tests`/`frontend-tests`만 성공 | 신규 job/artifact receipt 아님 |

GitHub Actions는 원격 ref의 workflow를 실행하므로 커밋/push 없이 로컬 미추적
workflow의 push run을 만들 수 없다. 수동 dispatch도 default branch에 workflow가
존재해야 하므로 현재 원격 상태에서는 신규 계약을 dispatch할 수 없다. 따라서
CR-I3-MAJ-01은 **코드 준비 후 Git 반영과 운영 provisioning을 거쳐야만 생성 가능한
외부 receipt**이며, 로컬 live report로 대체할 수 없다.

## 4. 코드 품질 Gate와 외부 acceptance Gate 분리

| Gate | 현재 판정 | 다음 조건 |
|---|---|---|
| pre-Git 코드 품질 | **FAIL** | CR-I4-MAJ-01/02 수정, targeted/전체 suite, link/diff, 독립 9.7 재리뷰 |
| Git 단계 진입 | **보류** | 위 코드 품질 Gate PASS 후 가능; 외부 receipt가 pre-commit에 없다는 이유만으로는 보류하지 않음 |
| post-merge hosted CI | **NOT_RUN** | merge SHA의 `python-tests`/`frontend-tests`, locked Linux install, `pip check`, snapshot artifact 성공 |
| post-merge live infra | **NOT_PROVISIONED** | required-reviewer environment, online `ollama-m3` runner, runner 계정 기준 두 외부 경로 read-only, Ollama/model 준비 |
| post-merge live receipt | **NOT_RUN** | trusted master push 또는 default-branch `workflow_dispatch`, approval 전 실행 차단, live job/preflight/14-gate/artifact 성공을 merge SHA와 결합 검증 |

Git 반영 후에는 위 외부 Gate가 하나라도 `UNKNOWN`/`NOT_RUN`이면 M4.1 최종 완료 및
merge acceptance를 선언하면 안 된다. 실패 run의 `always()` artifact 계약을 실제로
검증할 경우에는 안전하게 통제된 실패 dispatch를 별도 수행해야 하며, 성공 receipt와
혼동하지 않아야 한다.

## 5. clean lock / shared venv 증거 재검증

- `tests/unit/test_ci_acceptance_contract.py`: **20 passed**. 다만 위 두 MAJOR의 CLI
  조합/필수 인자/provenance 음성 경로는 포함하지 않는다.
- 기존 project `venv/bin/python -m pip check`: **FAIL 3건**. 원인은 lock에 없는
  `langgraph-prebuilt==1.0.2`와 `langchain-classic==1.0.0`이 요구하는
  `langchain-core>=1.0.0`/새 text-splitters와 현재 locked 0.3 계열의 충돌이다.
- Runbook/Traceability의 별도 clean venv 기록은 hash locked install, editable
  `--no-deps`, `pip check` **PASS**, 전체 suite **893 passed, 1 skipped**, lock verify
  **102 packages/no drift**로 서로 정합하다. clean venv 경로는 현재 보존돼 있지 않아
  이번 리뷰에서 893 전체 suite를 동일 환경으로 재실행하지는 못했지만, shared venv의
  추가 두 패키지가 lock에 없음을 현재 파일과 `pip check`로 재확인했다.

따라서 shared venv 실패를 requirements lock 결함으로 보지 않는 분리는 타당하다.
다만 macOS clean venv는 Linux hosted CI의 대리 증거일 뿐 REQ-001.3 최종 receipt는
아니다.

## 6. 실행 검증

| 명령 | 결과 |
|---|---|
| `venv/bin/python -m pytest -q tests/unit/test_ci_acceptance_contract.py` | **PASS — 20 passed** |
| 실제 과거 run receipt: `python-tests` + `frontend-tests` | **PASS** — API 판독 기본 경로 확인 |
| 같은 run에 `--require-conclusion ""` | **FAIL** — CR-I4-MAJ-01 재현 |
| `venv/bin/python -m pip check` | **FAIL** — lock 밖 shared venv 오염 3건 |
| `python scripts/check_markdown_links.py` (리뷰 작성 전) | **PASS — 74 files, 316 links, failures 0** |
| `git diff --check` (리뷰 작성 전) | **PASS** |
| 실제 `gh` default branch/workflow/environment/runner/recent runs 조회 | **PASS — §3 상태 확인** |

## 7. 재진입 및 최종 폐쇄 조건

1. CR-I4-MAJ-01/02를 수정하고 누락된 CLI/provenance/fail-closed 테스트를 추가한다.
2. 전체 clean locked suite와 정적·link/diff Gate를 재실행하고 독립 리뷰에서
   CRITICAL/MAJOR 0, 9.7 이상을 받는다.
3. 그때 Git 단계에 진입한다. commit/push/PR 자체는 이번 리뷰에서 수행하지 않았다.
4. Git 반영 후 운영자가 environment/runner/read-only data를 provision하고, merge
   SHA에서 hosted 두 job과 live job을 실행한다.
5. 수정된 contract로 event/branch/SHA/jobs/artifacts/provisioning을 모두 검증한 실제
   Actions URL/receipt를 Traceability에 연결한 뒤에만 M4.1 최종 acceptance를 승인한다.
