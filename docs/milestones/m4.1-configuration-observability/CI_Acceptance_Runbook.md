> **SUPERSEDED / NON-EXECUTABLE HISTORICAL RECORD (as of 2026-08-15):**
> This runbook documents the M4.1 self-hosted/native-Ollama live
> regression path, which is `NOT_ADOPTED` under the M4 Operational
> Acceptance Recovery policy
> ([Requirement.md](../m4-operational-acceptance-recovery/Requirement.md)).
> The commands, checklists, and provisioning steps below are a historical
> record of what M4.1 built and verified at the time; they are **not** a
> current or executable release procedure. Do not provision a self-hosted
> runner, approve the `m3-live-regression` environment, or run the live
> job using this document. The only current, normative release-readiness
> procedure is
> [`docs/operations/deployment_runbook.md` §6.1](../../operations/deployment_runbook.md#61-hostedoci-baseline-verification-pre-deployment).

# CR-I3-MAJ-01 — CI 운영 증거 조사와 실행 가능한 계약

조사일: 2026-08-09 (UTC 2026-08-08T16:xx경 `gh` 조회 시각 기준)
작성자: Claude Code (Sonnet 5), M4.1 구현 담당
범위: [Code_Review_Iteration_3.md](Code_Review_Iteration_3.md)의
`CR-I3-MAJ-01` 폐쇄 작업. 본 문서와 `scripts/ci_acceptance_contract.py`
외에는 제품 코드를 확장하지 않았고, commit/push/PR/merge는 수행하지
않았다(작업 지시 범위).

**2026-08-09 개정 — CR-I4-MAJ-01/02 폐쇄**
([Code_Review_Iteration_4.md](Code_Review_Iteration_4.md)):
아래 §2.1/§2.2의 명령과 출력은 그 폐쇄를 반영해 갱신했다. 두 MAJOR는
`scripts/ci_acceptance_contract.py`가 예전에 "필수 증거를 검증하는
계약"이 아니라 "필수 증거를 생략해도 통과할 수 있는 계약"이었다는
공통 원인을 가리켰다: (1) 문서가 약속한 `--require-conclusion ""`
(빈 문자열 → 검사 생략)이 CLI에서 실제로는 그대로 `None`이 아닌 빈
문자열로 전달돼 항상 실패했고, (2) `--external-dir`/`--require-job`/
`--require-artifact`가 모두 기본값 빈 리스트라 아무 증거도 검사하지
않은 채 `provisioning OK`/`receipt OK`가 가능했으며 run의 event/
head_branch/head_sha/workflow path도 전혀 검증하지 않았다. 두 가지
모두 CLI parsing과 `run_provisioning`/`check_run_receipt` 함수 내부
검증을 동시에 고쳐, CLI를 우회해 함수를 직접 호출해도 동일하게
fail-closed이도록 했다(§2.1/§2.2, `tests/unit/test_ci_acceptance_contract.py`
55 tests). 아래 §7의 재현 명령도 새 계약으로 갱신했다.

**2026-08-09 재개정 — CR-I5-MAJ-01/02/03 폐쇄**
([Code_Review_Iteration_5.md](Code_Review_Iteration_5.md)): 이번 재개
사이클은 Iteration 5의 MAJOR 3건만 닫는다(Iteration 1~5의 기존 폐쇄
판정은 감사 이력으로 보존하며 재작성하지 않음). 세 항목:

- **CR-I5-MAJ-01**: `--skip-conclusion`은 전체 run conclusion 검사만
  생략했고 각 `--require-job`의 conclusion은 여전히 무조건 `success`만
  허용했다 — 그래서 §4 step 6이 문서화한 "의도적으로 실패시킨 live
  job의 `if: always()` artifact를 `--skip-conclusion`으로 검증" 절차가
  실제로는 그 job 자체의 실패 때문에 job-conclusion 검사에서 먼저
  거부됐다. `check_run_receipt()`/`receipt` CLI에 새 좁은 계약
  `require_job_conclusions`/`--allow-job-conclusion JOB=CONCLUSION`을
  추가했다 — 명시적으로 나열한 job만 지정한 conclusion 집합을 허용하고
  (기본은 여전히 `{"success"}`), 나머지 job은 그대로 `success`만
  허용한다. `--skip-conclusion`(전체 run)과는 독립적인 별개 파라미터다.
- **CR-I5-MAJ-02**: `provisioning`/`receipt` 모두 CLI가 강제하는
  M4.1 고정 profile(디렉터리 2개 이름, job 3개, artifact 2개)을 함수
  경계에서는 개수만 확인했다 — 직접 호출로 동일 경로를 두 번 넘기거나
  (`run_provisioning(..., external_dirs=[Path('/same'), Path('/same')])`)
  job/artifact 하나만 넘겨도 통과했다. `run_provisioning()`은 이제
  `external_dirs: dict[str, Path]`(이름 있는 구조)를 받아
  `EXTERNAL_DIR_NAMES`와 정확히 일치하는 키인지, 서로 다른 이름이
  동일 경로를 가리키지 않는지까지 확인한다. receipt 쪽은 새 함수
  `check_m41_receipt()`/CLI 서브커맨드 `receipt-m41`을 추가했다 —
  `M41_RECEIPT_JOBS`(`python-tests`/`frontend-tests`/
  `m3-live-regression-gate`)와 `M41_RECEIPT_ARTIFACTS`
  (`dependency-snapshot`/`m4-regression-report`)를 캐일러가 아예 선택할
  수 없는 고정값으로 내부에서 `check_run_receipt()`에 전달한다(제네릭
  `receipt` 서브커맨드는 다른 과거 run 검증에도 쓰이므로 그대로 두고,
  M4.1 merge receipt 전용 좁은 진입점을 별도로 추가). 부수적으로
  `check_run_receipt(expected_events=set())`(빈 set을 명시적으로
  전달)가 `if expected_events else DEFAULT` truthy 검사를 통과해 기본
  허용 집합으로 되돌아가던 fail-open도 함께 닫았다.
- **CR-I5-MAJ-03**: `bash scripts/compile_lock.sh --verify`가
  committed `requirements.lock`과 오늘 canonical resolution 사이의
  drift(`starlette` 버전)를 재현했으므로 `requirements.lock`을
  재생성했다(`scripts/compile_lock.sh`, 인자 없이).

세 항목 모두 `tests/unit/test_ci_acceptance_contract.py`가 55 tests에서
**80 tests**로 확장됐다(§5.2). 아래 §2.2/§4/§7이 새 계약(`--allow-job-
conclusion`, `receipt-m41`)을 반영하도록 함께 갱신됐다.

**2026-08-09 Recovery Cycle 2 — CR-R1-MAJ-01 폐쇄**
([Code_Review_Recovery_Cycle_1.md](Code_Review_Recovery_Cycle_1.md)):
Recovery Cycle 1 리뷰는 CR-I5-MAJ-02를 "부분 폐쇄"로 재개방하며 단일
MAJOR `CR-R1-MAJ-01`을 남겼다(Iteration 1~5와 Recovery Cycle 1 자체는
감사 이력으로 보존하며 재작성하지 않음). `run_provisioning()`의
`external_dirs` distinctness 체크(§CR-I5-MAJ-02)는 `len(set(...values()))`
로 두 `Path` 객체의 lexical equality만 비교했으므로, `vectorstore`와
`documents`가 서로 다른 symlink이면서 둘 다 같은 실제 디렉터리(예:
`/tmp/.../shared`)를 가리키는 경우에도 "서로 다른 값"으로 통과했다 —
독립 adversarial 재현에서 `ALIAS_ACCEPTED`가 출력되어 두 dir 검사가
모두 무예외로 통과함을 확인했다(Recovery Cycle 1 §4).

폐쇄 조치: `scripts/ci_acceptance_contract.py`에 `_external_dir_identity()`
를 추가해 `Path.resolve(strict=True)`로 symlink를 완전히 풀고
`os.stat()`의 `(st_dev, st_ino)`로 canonical filesystem identity를 얻는다
— bind mount로 이중 마운트된 경우까지 동일 identity로 취급한다. 존재하지
않는 경로는 `FileNotFoundError`를, resolve/stat 도중의 권한 오류는
`OSError`를 잡아 기존 계약과 동일한 `ContractError`로 fail-closed
변환한다. `run_provisioning()`은 이 identity 비교를 `gh api` 호출(runner/
environment 체크) 전에, 기존 lexical distinctness 체크 바로 뒤에서
수행해 alias가 있으면 즉시 거부한다 — direct-call과 실제 CLI argv 양쪽
모두에서 동일하게 적용된다(§5.3).

## 1. 결론 — 실제 Actions run 영수증은 이 dispatch 범위에서 구조적으로 불가능

**CR-I3-MAJ-01이 요구하는 세 가지 필수 폐쇄 증거(protected environment
required reviewer, 실제 self-hosted/`ubuntu-latest` run 성공, runner
service account read-only provisioning)는 커밋도 push도 금지된 이
worktree에서 만들어낼 수 없다.** GitHub Actions는 `push`/`workflow_dispatch`
이벤트가 실제로 GitHub 원격 저장소의 어떤 ref에 존재하는 workflow YAML을
읽어 실행하는 서비스이며, 로컬 uncommitted 변경은 그 ref에 존재하지
않으므로 애초에 트리거될 수 없다. 아래는 이를 추측이 아니라 오늘 직접
조회한 값으로 확인한 기록이다.

| 확인 항목 | 명령 | 결과 |
|---|---|---|
| local HEAD vs `origin/master` | `git rev-parse HEAD`, `git rev-parse origin/master` | 둘 다 `c056342b5dd3fe85b8ea42fe78c24e6c3e0417d3` — 동일. 즉 원격에는 이번 M4.1 변경이 전혀 없음 |
| uncommitted 변경 수 | `git status --short --porcelain \| wc -l` | 59개 경로(tracked 수정 22 + untracked 37, `.github/workflows/ci.yml` 포함) |
| 원격 workflow 파일 내용 | `gh api repos/luminous419/simple-qna-rag/contents/.github/workflows/ci.yml --jq .sha` | `e76b62988f75b54b6f0e1b9b2501bb957c5ba51c` — 로컬 `ci.yml`(m3-live-regression-gate job 포함)과 다른 블롭. 즉 원격 CI는 아직 `m3-live-regression-gate` job 자체를 모른다 |
| GitHub Environment | `gh api repos/luminous419/simple-qna-rag/environments` | `{"total_count":0,"environments":[]}` — `m3-live-regression` environment가 아직 존재하지 않음(required reviewer 설정 대상 자체가 없음) |
| self-hosted runner | `gh api repos/luminous419/simple-qna-rag/actions/runners` | `{"total_count":0,"runners":[]}` — `ollama-m3` 라벨의 runner가 등록되어 있지 않음 |
| 과거 run에 artifact 업로드 이력 | `gh api .../actions/runs/<id>/artifacts` (최근 3개 run 확인) | 세 run 모두 `artifacts: []` — `dependency-snapshot`/`m4-regression-report` 업로드 step 자체가 아직 원격에 없음(로컬 미커밋 변경) |

이 네 가지는 서로 독립적으로 같은 결론을 가리킨다: **실제 GitHub Actions
run 영수증은 최소 한 번의 commit + push(+ 이 저장소의 기존 관례상 PR +
merge, 또는 최소 `push:master`) 없이는 존재할 수 없다.** 이는 코드나
workflow 설계의 결함이 아니라 GitHub Actions의 실행 모델 자체이므로, 이
결론을 숨기거나 로컬 실행을 가짜 영수증으로 치환하지 않는다. Code Review
Iteration 3 §4가 이미 지적했듯 로컬 dirty worktree의 live report는
"훌륭한 evaluator 증거이지 GitHub Actions run receipt가 아니다."

## 2. 이번 dispatch에서 실행 가능한 범위로 만든 것

commit/push/PR/merge 없이 가능한 범위는 (a) 실행 가능한 단일 검증
계약을 코드로 만들고 (b) 그 계약을 실제 데이터/실제 `gh api` 응답으로
테스트하는 것이다. 아래 `scripts/ci_acceptance_contract.py`가 그 계약이며,
`provisioning`/`receipt` 두 서브커맨드로 CR-I3-MAJ-01의 필수 폐쇄 증거
세 항목을 각각 담당한다.

### 2.1 provisioning — runner/environment/외부 데이터 read-only

```bash
python scripts/ci_acceptance_contract.py --repo luminous419/simple-qna-rag \
  provisioning \
  --runner-label self-hosted --runner-label ollama-m3 \
  --environment m3-live-regression \
  --external-dir vectorstore=/opt/simple-qna-rag-data/vectorstore \
  --external-dir documents=/opt/simple-qna-rag-data/documents
```

`gh api repos/{repo}/actions/runners`로 `ollama-m3` 라벨의 online
runner를 찾고, `gh api repos/{repo}/environments/{env}`로
`required_reviewers` protection rule 존재와 **그 rule에 실제 reviewer가
1명 이상 설정돼 있는지**(CR-I4-MAJ-02 — 빈 reviewers 목록은 아무도
막지 못하므로 별도로 거부)를 확인하고, 각 external dir에 대해
`os.access(path, os.W_OK)`로 현재 프로세스가 쓰기 불가능한지 확인한다
(runner service account 기준 read-only 계약 — CR-I3-MAJ-01 필수 증거
3번). 이 read-only 체크는 **runner 호스트에서 runner service
account로 실행할 때만** 의미 있는 답을 준다; 개발자 워크스테이션에서
자신의 계정으로 실행하면 그 계정 기준으로는 쓰기가 가능하므로 정확하게
실패를 보고한다(가짜 PASS를 만들지 않음).

**CR-I4-MAJ-02**: `--external-dir`는 이제 `NAME=PATH` 형식이며 M4.1이
실제로 필요로 하는 `vectorstore`/`documents` 두 이름이 **모두** 있어야
한다(`EXTERNAL_DIR_NAMES`, 고정 profile). 하나라도 빠지면 `gh api`를
전혀 호출하지 않고 즉시 exit 2로 거부한다 — 예전에는 `--external-dir`
자체가 선택 인자(기본값 빈 리스트)라 아무 디렉터리도 검사하지 않은
`provisioning OK`가 가능했다. `run_provisioning()` 함수 자체도 동일한
최소 2-디렉터리 요건을 다시 확인하므로, CLI를 거치지 않고 Python에서
직접 호출해도 fail-closed다.

오늘 이 저장소를 대상으로 실행한 실제 결과(고정 profile 강제와 실제
provisioning 상태를 모두 확인):

```
$ python scripts/ci_acceptance_contract.py --repo luminous419/simple-qna-rag \
    provisioning --runner-label self-hosted --runner-label ollama-m3 \
    --environment m3-live-regression \
    --external-dir vectorstore=runtime/vectorstore
usage: ci_acceptance_contract.py [-h] --repo REPO {provisioning,receipt} ...
ci_acceptance_contract.py: error: provisioning requires --external-dir for
['documents'] (M4.1 fixed profile: ('vectorstore', 'documents')); got
['vectorstore']
exit=2

$ python scripts/ci_acceptance_contract.py --repo luminous419/simple-qna-rag \
    provisioning --runner-label self-hosted --runner-label ollama-m3 \
    --environment m3-live-regression \
    --external-dir vectorstore=runtime/vectorstore --external-dir documents=runtime/documents
no online runner with labels ['ollama-m3', 'self-hosted'] registered on
luminous419/simple-qna-rag (found 0 runner(s): [])
exit=1
```

첫 호출은 하나라도 빠진 external dir을 즉시 parse-time에 거부하고,
두 번째 호출(두 이름 모두 제공)은 runner가 없으므로 provisioning
체크는 정직하게 실패한다. 이 결과 자체가 §1의 결론을 코드로 재확인한
것이다.

### 2.2 receipt — provenance + job conclusion + artifact 업로드 검증

```bash
python scripts/ci_acceptance_contract.py --repo luminous419/simple-qna-rag \
  receipt --run-id <RUN_ID> \
  --require-job python-tests --require-job frontend-tests \
  --require-job m3-live-regression-gate \
  --require-job-label m3-live-regression-gate=self-hosted \
  --require-job-label m3-live-regression-gate=ollama-m3 \
  --require-artifact dependency-snapshot \
  --require-artifact m4-regression-report \
  --expected-sha <MERGE_SHA> \
  --expected-branch master \
  --expected-workflow-path .github/workflows/ci.yml
```

`gh api repos/{repo}/actions/runs/{id}`(전체 conclusion과 provenance:
`head_sha`/`head_branch`/`event`/`path`), `.../jobs`(지정한 job
이름들의 conclusion, 선택적으로 그 job의 requested runner label),
`.../artifacts`(만료되지 않은 artifact 존재, 선택적으로 sha256
digest)를 조회해 이 모든 조건을 만족해야 통과한다.

**CR-I4-MAJ-02**: `--require-job`/`--require-artifact`/`--expected-sha`는
이제 필수 인자다(`--expected-branch`는 `master`, `--expected-workflow-path`는
`.github/workflows/ci.yml` 기본값이 있지만 검사 자체는 생략할 수
없다). 예전에는 이 값들이 모두 기본값 빈 리스트/생략 가능이라 job과
artifact를 하나도 검사하지 않은 `receipt OK`, 그리고 event/branch/sha가
다른 과거 run을 현재 merge receipt로 받아들이는 통과가 가능했다.
`check_run_receipt()` 함수 자체도 동일한 필수값 검증을 다시 하므로,
CLI를 거치지 않고 Python에서 직접 호출해도 fail-closed다. `--run-id`로
지정한 run이 merge SHA(`--expected-sha`)·`master`(`--expected-branch`)·
허용된 trigger event(`--expected-event`, 기본 `push`/`workflow_dispatch`)·
정확한 workflow 파일(`--expected-workflow-path`)에서 나온 것인지까지
한 번에 묶어 검증하므로, 이름이 맞는 다른 run을 잘못 골라도 통과하지
않는다.

**CR-I4-MAJ-01**: `--require-conclusion ""`은 더 이상 조용히 빈 문자열로
비교되지 않는다 — 이제 즉시 parse-time 오류다. `if: always()` 계약
(실패한 run에서도 report artifact가 올라오는지)을 검증하려면 명시적인
`--skip-conclusion` 플래그를 사용한다. 이 플래그는 `--require-conclusion`과
동시에 줄 수 없다(둘 다 주면 exit 2) — Iteration 3 §2 필수 증거 2번
후반부("실패 run에서도 report artifact step의 실제 동작을 확인하면
`if: always()` 계약까지 폐쇄할 수 있다")를 이제 명확한 opt-in으로
코드화했다.

**CR-I5-MAJ-01**: `--skip-conclusion`은 전체 run conclusion만 생략할
뿐, 위에서 `--require-job`으로 지정한 각 job의 conclusion은 여전히
기본으로 `success`만 허용한다 — 그래서 §4 step 6처럼 `m3-live-
regression-gate` job 자체가 실패한 run에서는 artifact 조회 전에
job-conclusion 검사에서 거부됐다. 새 `--allow-job-conclusion
JOB=CONCLUSION`(반복 가능)이 명시한 job에 한해 허용 conclusion
집합을 좁게 override한다 — 나열하지 않은 job은 그대로 `success`만
허용된다. `--skip-conclusion`(전체 run)과는 독립적인 파라미터이므로
"의도적으로 실패시킨 run + 그 안의 실패한 필수 live job" 조합을
동시에 표현할 수 있다(§4 step 6).

이 로직이 실제 GitHub API 응답을 올바르게 처리하는지는, 아직 존재하지
않는 `m3-live-regression-gate` run 대신 이 저장소의 **실제 과거 CI run**
으로 오늘 검증했다:

```
$ python scripts/ci_acceptance_contract.py --repo luminous419/simple-qna-rag \
    receipt --run-id 31204494509 \
    --require-job python-tests --require-job frontend-tests \
    --require-artifact m4-regression-report \
    --expected-sha c056342b5dd3fe85b8ea42fe78c24e6c3e0417d3
run 31204494509 missing non-expired artifact(s) ['m4-regression-report']
(found [])
exit=1

$ python scripts/ci_acceptance_contract.py --repo luminous419/simple-qna-rag \
    receipt --run-id 31204494509 \
    --require-job python-tests \
    --require-artifact m4-regression-report \
    --expected-sha 0000000000000000000000000000000000dead
run 31204494509 head_sha='c056342b5dd3fe85b8ea42fe78c24e6c3e0417d3',
expected '0000000000000000000000000000000000dead' — this run is not
evidence for the expected commit
exit=1

$ python scripts/ci_acceptance_contract.py --repo luminous419/simple-qna-rag \
    receipt --run-id 31204494509 \
    --require-job python-tests --require-artifact m4-regression-report \
    --expected-sha c056342b5dd3fe85b8ea42fe78c24e6c3e0417d3 \
    --require-conclusion ""
ci_acceptance_contract.py receipt: error: --require-conclusion cannot be an
empty string; pass --skip-conclusion to intentionally skip the conclusion
check
exit=2
```

첫 호출은 실제 run의 실제 head_sha/branch/event/workflow-path와 job
성공을 모두 정확히 인식한 뒤(이 run에는 아직 없는) artifact 누락으로
정직하게 실패하고, 두 번째 호출은 sha가 다른 run을 정확히 거부하며,
세 번째 호출은 CR-I4-MAJ-01이 재현했던 빈 문자열을 이제 parse-time에
거부함을 보여준다 — 즉 이 스크립트는 가짜 영수증을 만들 수 없는
구조다(모든 판정이 `gh api`의 실제 응답에서만 나온다). `dependency-snapshot`/
`m4-regression-report` artifact 업로드 step은 이번 M4.1 변경에
포함되어 있어 원격에 아직 없으므로, artifact 존재/hash 판정 로직
자체는 `tests/unit/test_ci_acceptance_contract.py`의 mocked `gh api`
응답으로 검증했다(§4).

### 2.3 데이터 preflight — 기존 스크립트를 실제 로컬 데이터로 재확인

`scripts/preflight_vectorstore.py`/`scripts/preflight_ollama.py`는
CR-I2-MAJ-01에서 이미 구현·테스트됐다. 이번 조사에서 실제 로컬
`runtime/vectorstore`와 로컬 Ollama 데몬을 대상으로 다시 실행해 정상
동작을 재확인했다(제품 코드 미변경, 순수 검증):

```
$ python scripts/preflight_vectorstore.py runtime/vectorstore
index.faiss sha256=c52fb288f3bc780d681ab68d62dd5b4c545843ee9dc53ef5528bd50db4d69820
index.pkl sha256=3f7217a2825f141bbea4c75dbf2b15738c3bcd50e3c0900e25a6a4a7ed91bb00
exit=0

$ python scripts/preflight_ollama.py http://localhost:11434 gpt-oss:20b
Ollama version=0.32.0 required_model=gpt-oss:20b
models=['qwen3-coder:30b', 'qwen3.5:9b', 'glm-4.7-flash:latest', 'bge-m3:latest',
'gpt-oss:120b', 'gpt-oss:20b', 'qwen2.5:7b', 'llama2:7b-chat', 'llama3.1:latest']
exit=0
```

sha256 값은 Traceability.md REQ-006.2의 기존 live 증거 기록과 정확히
일치한다 — vectorstore는 이번 조사로 전혀 변경되지 않았다(§5에서 재확인).
M3 14-gate 실 회귀(`run_m4_regression_gate.py`, 24분+ 소요)는 기존
`baseline_20260808T155819908435Z` 증거를 보존하기 위해 이번 조사에서
**재실행하지 않았다** — 재실행은 새 증거를 만들지 않고(동일 로컬
dirty worktree 실행은 여전히 Actions run이 아님) 기존 보존 요건과
충돌할 위험만 있다.

## 3. 원격 provisioning 선행 조건(운영자가 별도로 수행)

이 dispatch가 만들 수 없는 것은 명확히 구분한다 — 아래는 저장소 소유자가
GitHub UI/API로 한 번 수행해야 하는 절차이며, 이 문서는 그 절차를
기록만 하고 실행하지 않는다.

1. **self-hosted runner 등록**: Ollama와 두 외부 데이터 디렉터리에 접근
   가능한 호스트에서 `Settings → Actions → Runners → New self-hosted
   runner`로 등록 토큰을 발급받아 `./config.sh --url
   https://github.com/luminous419/simple-qna-rag --token <TOKEN> --labels
   self-hosted,ollama-m3`을 실행한다.
2. **외부 데이터 디렉터리 provisioning**: runner 호스트에
   `/opt/simple-qna-rag-data/{vectorstore,documents}`를 생성하고 현재
   `runtime/vectorstore`(index.faiss/index.pkl)와 문서 원본을 복사한 뒤,
   runner service account 소유로 바꾸고 쓰기 권한을 제거한다
   (`chown <runner-svc-account> -R /opt/simple-qna-rag-data && chmod -R
   a-w /opt/simple-qna-rag-data`). 이후 `python
   scripts/ci_acceptance_contract.py --repo luminous419/simple-qna-rag
   provisioning --runner-label self-hosted --runner-label ollama-m3
   --environment m3-live-regression --external-dir
   vectorstore=/opt/simple-qna-rag-data/vectorstore --external-dir
   documents=/opt/simple-qna-rag-data/documents`를 **runner service
   account로** 실행해 read-only 확인이 PASS하는지 검증한다(두 이름 모두
   필수 — CR-I4-MAJ-02).
3. **GitHub Environment 생성**: `Settings → Environments → New
   environment → m3-live-regression`, `Required reviewers`에 저장소
   소유자(또는 지정 승인자)를 **최소 1명** 추가한다(빈 목록은
   CR-I4-MAJ-02 이후 `check_environment_protected()`가 거부한다). 생성
   후 `gh api repos/luminous419/simple-qna-rag/environments/m3-live-regression`이
   `protection_rules`에 reviewer가 채워진 `required_reviewers`를
   포함하는지 확인한다(§2.1 명령이 자동으로 검증).

## 4. 후속 Git 단계(문서화만, 이번 dispatch에서 실행하지 않음)

아래는 §3 provisioning이 완료된 뒤, 이 작업 지시가 금지한 commit/push/
PR/merge를 별도 승인된 작업에서 수행할 때 사용할 정확한 명령이다. 이
dispatch는 이 명령들을 실행하지 않았다.

```bash
# 1. 이번 M4.1 작업(59개 변경 경로)을 커밋
git checkout -b feature/m4.1-configuration-observability
git add -A
git commit -m "M4.1: configuration/observability + CR-I3-MAJ-01 acceptance contract"

# 2. push 후 PR 생성(이 저장소의 기존 관례 — 최근 커밋이 모두 PR merge)
git push -u origin feature/m4.1-configuration-observability
gh pr create --base master --title "M4.1: Configuration & Observability" \
  --body "closes CR-I3-MAJ-01 code path; see docs/milestones/m4.1-configuration-observability/CI_Acceptance_Runbook.md"

# 3. 리뷰 승인 후 merge(master push 트리거가 m3-live-regression-gate를
#    깨움 — environment required reviewer가 실제 실행 전 별도 승인을 요구)
gh pr merge --merge

# 4. master push로 트리거된 run을 찾아 대기
gh run list --branch master --workflow CI --limit 1 --json databaseId,status
gh run watch <RUN_ID>
MERGE_SHA=$(git rev-parse origin/master)

# 5. 이 문서가 만든 계약으로 실제 영수증 검증(CR-I3-MAJ-01 필수 폐쇄 증거 1/2).
#    CR-I5-MAJ-02: 세 필수 job과 두 artifact는 이제 receipt-m41이 고정
#    profile(M41_RECEIPT_JOBS/M41_RECEIPT_ARTIFACTS)로 내부에서 강제하므로
#    --require-job/--require-artifact를 나열할 필요도, 실수로 부분집합만
#    넘길 방법도 없다.
python scripts/ci_acceptance_contract.py --repo luminous419/simple-qna-rag \
  receipt-m41 --run-id <RUN_ID> \
  --require-job-label m3-live-regression-gate=self-hosted \
  --require-job-label m3-live-regression-gate=ollama-m3 \
  --expected-sha "$MERGE_SHA" \
  --expected-branch master \
  --expected-workflow-path .github/workflows/ci.yml

# 6. always()-artifact 계약까지 닫으려면 의도적으로 실패시킨 run(예: Ollama
#    endpoint를 잠시 내린 workflow_dispatch)에서도 동일 receipt 검증을
#    --skip-conclusion 으로 실행해 artifact만 확인(CR-I4-MAJ-01 — 빈
#    문자열이 아니라 명시적 플래그로만 conclusion 검사를 생략한다).
#    CR-I5-MAJ-01: 이 run에서는 m3-live-regression-gate job 자체가
#    실패했으므로 --skip-conclusion만으로는 job-conclusion 검사에서
#    거부된다 — --allow-job-conclusion으로 그 job에 한해 'failure'도
#    허용한다고 명시해야 artifact 조회까지 도달한다. 이 step은 generic
#    receipt를 그대로 쓴다(3 job/2 artifact 전체가 아니라 live job과 그
#    artifact만 확인하는 의도적 부분 검증이므로 receipt-m41의 고정
#    profile과는 범위가 다르다).
python scripts/ci_acceptance_contract.py --repo luminous419/simple-qna-rag \
  receipt --run-id <FAILED_RUN_ID> --skip-conclusion \
  --allow-job-conclusion m3-live-regression-gate=failure \
  --require-job m3-live-regression-gate \
  --require-artifact m4-regression-report \
  --expected-sha "$MERGE_SHA" --expected-branch master \
  --expected-event workflow_dispatch \
  --expected-workflow-path .github/workflows/ci.yml
```

## 5. 검증 — Traceability REQ-001.3 정정: shared venv 오염과 lock 결함 분리

Code_Review_Iteration_3.md §6은 완전히 새로운 venv에서 `pip
check`/`pytest -q`가 PASS했다고 기록했지만, 그 venv가 이 저장소의
project `venv/`와 어떻게 다른지, 그리고 project `venv/`의 실패 원인이
무엇인지는 별도로 재확인되지 않았다. 이번 조사에서 둘을 명시적으로
분리했다.

| 대상 | 명령 | 결과 |
|---|---|---|
| project `venv/`(기존, 공유) | `venv/bin/python -m pip check` | **FAIL** — `langgraph-prebuilt 1.0.2 has requirement langchain-core>=1.0.0, but you have langchain-core 0.3.86.`; `langchain-classic 1.0.0`도 동일 계열 충돌 2건. 이 두 패키지는 `requirements.lock`에 없음(범위 밖에서 별도 설치됨) — **shared venv 오염**, lock 결함 아님 |
| 완전히 새로운 venv(`$SCRATCH/clean_venv`, 이번 조사에서 신규 생성, `python3 -m venv`, project `venv/`와 무관) | `pip install --require-hashes -r requirements.lock` → `pip install -e . --no-deps` → `pip check` | **PASS — No broken requirements found.** |
| 동일 clean venv | `pytest -q` | **PASS — 893 passed, 1 skipped**(Code Review Iteration 3의 873 + 본 작업이 추가한 `test_ci_acceptance_contract.py` 20건 = 893, 정합) |
| 동일 clean venv | `bash scripts/compile_lock.sh --verify` | **PASS — 102 packages, reproducible, no drift** |
| 동일 clean venv | `scripts/dependency_snapshot.py` | **PASS**(gitignored `dependency_snapshot.json` 생성) |
| 동일 clean venv | `scripts/generate_field_spec.py --check` | **PASS** |
| 동일 clean venv | `scripts/logging_callsite_audit.py --check` | **PASS** |
| 동일 clean venv | `scripts/check_markdown_links.py` | **PASS — 73 files, 311 links, failures 0**(본 문서 추가로 파일 수 72→73) |
| 동일 clean venv | `python -m evaluation.dataset validate evaluation/datasets/golden.jsonl` | **PASS — valid=true, total=76** |
| repo root | `git diff --check` | **PASS** |
| repo root | `npm test` | **PASS — 9 tests** |
| repo root | `npm run sync-vendor` 후 `git diff --exit-code -- web/static/vendor/` | **PASS — vendor diff 0** |
| repo root | `git diff --exit-code -- evaluation/baselines/m3_initial.{json,md}` | **PASS — 변경 없음** |
| repo root | `runtime/vectorstore/{index.faiss,index.pkl}` sha256 재계산 | `c52fb2...9820`/`3f7217...91bb00` — Traceability.md REQ-006.2 기록과 **불변 확인** |

결론: **shared project `venv/`의 `pip check` 실패는 lock 파일이나 M4.1
구현의 결함이 아니라, 그 venv에 `requirements.lock` 밖에서 설치된
`langchain-classic`/`langgraph-prebuilt` 때문이다.** 완전히 새로운
venv(이번 조사에서 실제로 처음부터 생성)에서는 동일 lock으로 `pip
check`와 전체 suite가 모두 PASS한다. REQ-001.3의 "Linux CI에서만 최종
확정 가능"이라는 제약은 여전히 유효하다 — 이 clean venv도 macOS이므로
`ubuntu-latest` 실행의 근접 대리 증거일 뿐, §1이 설명하는 이유로 실제
Linux hosted 실행 자체는 이번 dispatch 범위에서 만들 수 없다.

### 5.1 CR-I4-MAJ-01/02 폐쇄 검증(2026-08-09, project `venv/`)

CR-I4-MAJ-01/02 수정 후 project `venv/`(기존 공유 venv, 위 표의 shared
venv와 동일)에서 재실행한 targeted/full suite와 감사 스크립트 결과:

| 명령 | 결과 |
|---|---|
| `venv/bin/python -m pytest -q tests/unit/test_ci_acceptance_contract.py` | **PASS — 55 passed**(기존 20건 + CR-I4-MAJ-01/02 fail-closed/adversarial 신규 35건: 빈 문자열 conclusion 거부, `--skip-conclusion` 상호배타, provisioning 2-디렉터리 미만 거부/이름 불일치 거부, receipt의 head_sha/head_branch/event/workflow-path 불일치 거부, require_jobs/require_artifacts 빈 값 거부, job-label/artifact-hash 매칭·불일치·미지정 job/artifact 참조 거부 등) |
| `venv/bin/python -m pytest -q`(전체) | **PASS — 928 passed, 1 skipped**(이전 893 + 신규 35건 = 928, 정합; 1 skipped는 기존과 동일한 환경 제약 예외) |
| `venv/bin/python -m pip check` | **FAIL — 기존과 동일한 shared venv 오염 3건**(§5 표와 동일 원인, 이번 변경으로 새로 발생한 문제 아님) |
| `bash scripts/compile_lock.sh --verify` | **FAIL — `requirements.lock`이 `requirements.txt`에서 drift**: 두 번의 재컴파일 결과는 서로 동일(reproducible)하지만 커밋된 `requirements.lock`과는 다르다. 원인은 `starlette==1.5.0`(committed lock) vs 오늘 재컴파일 시 PyPI가 서빙하는 `starlette==1.5.1`(신규 upstream 릴리스) 하나뿐이며, `requirements.txt`나 이번 CR-I4-MAJ-01/02 코드 변경과는 무관하다. **이 drift는 이번 작업 범위(CR-I4-MAJ-01/02, `scripts/ci_acceptance_contract.py`) 밖이므로 `requirements.lock`을 재생성하지 않았다** — lock 재생성은 별도 승인된 작업으로 처리해야 한다 |
| `venv/bin/python scripts/dependency_snapshot.py` | **PASS** |
| `venv/bin/python scripts/generate_field_spec.py --check` | **PASS** |
| `venv/bin/python scripts/logging_callsite_audit.py --check` | **PASS** |
| `venv/bin/python scripts/check_markdown_links.py` | **PASS — 75 files, 317 links, failures 0** |
| `venv/bin/python -m evaluation.dataset validate evaluation/datasets/golden.jsonl` | **PASS — valid=true, total=76** |
| `git diff --check` | **PASS** |
| `npm test` | **PASS — 9 tests** |
| `npm run sync-vendor` 후 `git diff --exit-code -- web/static/vendor/` | **PASS — vendor diff 0** |
| `git diff --exit-code -- evaluation/baselines/m3_initial.{json,md}` | **PASS — 변경 없음** |
| `runtime/vectorstore/{index.faiss,index.pkl}` sha256 재계산 | `c52fb2...9820`/`3f7217...91bb00` — REQ-006.2 기록과 **불변 확인** |
| 실제 `gh api`로 §2.1/§2.2 명령 재현(§2.1/§2.2 출력 참고) | **PASS** — 실제 과거 run(`31204494509`)과 실제 environment/runner 상태로 CR-I4-MAJ-01(빈 문자열 거부)과 CR-I4-MAJ-02(2-디렉터리 강제, head_sha 불일치 거부, artifact 누락 거부)를 모두 재현 |

`pip check` FAIL과 `compile_lock.sh --verify` drift 모두 이번
CR-I4-MAJ-01/02 코드 변경(`scripts/ci_acceptance_contract.py`,
`tests/unit/test_ci_acceptance_contract.py`)이 원인이 아니며, 둘 다
그 코드를 전혀 건드리지 않고도 재현된다(shared venv에 lock 밖
패키지가 설치돼 있다는 사실, PyPI의 `starlette` 신규 릴리스라는 사실
자체가 원인). 두 항목 모두 CR-I4-MAJ-01/02의 폐쇄 조건이 아니므로 이
작업의 완료 판정에 영향을 주지 않지만, 향후 별도 작업(lock 재생성
또는 shared venv 재생성)이 필요함을 투명하게 기록해 둔다.

### 5.2 CR-I5-MAJ-01/02/03 폐쇄 검증(2026-08-09)

CR-I5-MAJ-01(`require_job_conclusions`/`--allow-job-conclusion`),
CR-I5-MAJ-02(`run_provisioning`의 named `dict[str, Path]` profile,
`check_m41_receipt`/`receipt-m41`), CR-I5-MAJ-03(`requirements.lock`
재생성) 적용 후 재실행한 결과:

| 명령 | 결과 |
|---|---|
| `venv/bin/python -m pytest -q tests/unit/test_ci_acceptance_contract.py` | **PASS — 80 passed**(기존 55건 + CR-I5-MAJ-01/02 신규 25건: 실제 failed-job argv 양성/음성 CLI 회귀 2건 — `test_main_receipt_allow_job_conclusion_permits_actual_failed_job`/`test_main_receipt_exit_1_when_actual_failed_job_without_allow_job_conclusion` — 은 `check_run_receipt`를 mock하지 않고 `_real_gh_runner`만 fake로 대체해 실제 로직을 통과시킴; `require_job_conclusions` 함수 레벨 4건; `check_m41_receipt`/`receipt-m41` 구조·fail-closed 11건(중복 job/artifact 없음 정적 검증 2건 포함); provisioning 동일 경로 재사용 거부 direct-call/CLI 2건; `expected_events=set()` 명시적 거부 1건; provisioning 이름 불일치 1건 등) |
| `venv/bin/python -m pytest -q`(전체, project `venv/`) | **PASS — 974 passed, 1 skipped**(§5.1의 928 대비 순증 46 — 본 작업이 추가한 contract 신규 25건 외 21건은 이번 CR-I5 작업 범위 밖에서 이미 dirty worktree에 존재하던 다른 미커밋 테스트 파일들의 수집 결과이며, CR-I5-MAJ-01/02/03 변경으로 새로 실패하거나 스킵된 테스트는 없음 — 동일 명령을 완전히 새로운 clean venv에서도 재실행해 동일하게 **974 passed, 1 skipped**로 교차 확인) |
| `bash scripts/compile_lock.sh --verify` | **PASS — 102 packages, reproducible, no drift**(CR-I5-MAJ-03 — `requirements.lock` 재생성 후. `starlette`는 오늘 canonical resolution 기준 `1.6.0`으로 갱신됨 — Iteration 5가 관찰한 `1.5.1`에서 다시 상향된 것으로, upstream이 계속 릴리스되는 unbounded transitive 의존성이라는 Iteration 5의 진단과 일치) |
| 완전히 새로운 clean venv(`python3 -m venv`, project `venv/`와 무관, 이번 조사에서 신규 생성) | `pip install --require-hashes -r requirements.lock` → `pip install -e . --no-deps` → `pip check` **PASS — No broken requirements found** |
| 동일 clean venv | `pytest -q` **PASS — 974 passed, 1 skipped** |
| `venv/bin/python scripts/check_markdown_links.py` | **PASS — 76 files, 323 links, failures 0**(본 문서/Traceability.md의 CR-I5 신규 문단이 추가한 링크 포함) |
| `git diff --check` | **PASS** |

`venv/bin/python -m pip check`(project 공유 venv)는 §5/§5.1과 동일한
원인(`langchain-classic`/`langgraph-prebuilt`이 lock 밖에서 설치된
shared venv 오염)으로 여전히 FAIL하지만, 완료 조건이 요구하는 "clean
`requirements.lock --require-hashes` install + `pip check`"는 위 표의
완전히 새로운 clean venv 결과로 충족된다 — lock 재생성이 그 오염을
고치지는 않으며 애초에 lock 결함이 아니었다(§5).

### 5.3 CR-R1-MAJ-01 폐쇄 검증(2026-08-09, Recovery Cycle 2)

`_external_dir_identity()` 추가와 `run_provisioning()`의 canonical
identity 비교 적용, adversarial 테스트 6건(`_external_dir_identity` 함수
레벨 존재/symlink-identity 동치 2건, direct-call symlink alias 거부 1건,
nested symlink chain 거부 1건, 실제 CLI argv symlink alias 거부 1건 —
§CR-R1-MAJ-01 상단 참고) 적용 후 재실행한 결과:

| 명령 | 결과 |
|---|---|
| `venv/bin/python -m pytest -q tests/unit/test_ci_acceptance_contract.py` | **PASS — 85 passed**(§5.2의 80건 + CR-R1-MAJ-01 신규 5건) |
| `venv/bin/python -m pytest -q tests/unit/test_ci_acceptance_contract.py tests/unit/test_dependency_lock.py` | **PASS — 90 passed** |
| `venv/bin/python -m pytest -q`(전체, project `venv/`) | **PASS — 979 passed, 1 skipped**(§5.2의 974 대비 순증 5 — 본 작업이 추가한 CR-R1-MAJ-01 신규 테스트뿐이며, 기존 테스트 중 새로 실패하거나 스킵된 것은 없음) |
| `bash scripts/compile_lock.sh --verify` | **PASS — 102 packages, reproducible, no drift**(`requirements.lock` 미변경 — CR-R1-MAJ-01은 코드/테스트/문서만 수정) |
| `venv/bin/python scripts/check_markdown_links.py` | **PASS — 77 files, 323 links, failures 0** |
| `git diff --check` | **PASS — 출력 없음** |
| symlink alias adversarial provisioning 재현(direct-call, `vectorstore`/`documents`가 같은 `shared` target) | `ContractError: external_dirs values must be distinct filesystem identities ...` — Recovery Cycle 1이 보고한 `ALIAS_ACCEPTED` fail-open이 fail-closed로 전환됨 |
| 동일 시나리오, 실제 CLI argv(`python scripts/ci_acceptance_contract.py ... provisioning --external-dir vectorstore=<symlink> --external-dir documents=<symlink>`) | exit 1, stderr에 `distinct filesystem identities` — direct-call과 CLI 양쪽 parity 확인 |

## 6. Traceability로의 반영

[Traceability.md](Traceability.md) §2에 `CR-I3-MAJ-01` 행을 추가하고
§3 REQ-001.3 항목을 본 문서의 clean-venv 재확인 결과로 갱신했다(같은
커밋 없이, 같은 dispatch 내 문서 변경). 이후 §2에 `CR-I4-MAJ-01`/
`CR-I4-MAJ-02` 행을 추가해 이번 CLI/함수 계약 fail-closed 수정과 §5.1의
targeted/full suite 재실행 결과를 연결했다. 이번 재개 사이클은 §2에
`CR-I5-MAJ-01`/`CR-I5-MAJ-02`/`CR-I5-MAJ-03` 행을 추가해 §5.2의
재실행 결과를 연결한다. Recovery Cycle 2(이번 사이클)는 §2의
`CR-I5-MAJ-02` 행을 "부분 폐쇄로 재개방 → 폐쇄"로 정정하고 새
`CR-R1-MAJ-01` 행을 추가해 §5.3의 재실행·symlink alias adversarial
재현 결과를 연결한다. 다음 독립 코드 리뷰는 §1~§5의 근거로
CR-I3-MAJ-01이 "코드/도구/문서 준비 폐쇄, 운영 증거는 §4의 Git 단계
이후 재확인 필요"임을, CR-I4-MAJ-01/02가 CLI parsing과 라이브러리
함수 양쪽에서 fail-closed임을, CR-I5-MAJ-01/02/03이 실패한
필수 live job의 always()-artifact 검증·M4.1 고정 profile의 CLI/direct-
call parity·committed lock drift를 모두 닫았음을, 그리고 CR-R1-MAJ-01이
그 M4.1 고정 profile 검사를 canonical filesystem identity 기준으로
강화해 symlink/mount alias 우회를 direct-call과 CLI 양쪽에서 fail-closed로
거부함을 재검증할 수 있다.

## 7. 검증 명령 요약(재현용)

```bash
# CR-I3-MAJ-01/CR-I4-MAJ-01/02/CR-I5-MAJ-01/02 신규 도구 단위 테스트
venv/bin/python -m pytest -q tests/unit/test_ci_acceptance_contract.py

# 전체 suite(CR-I5-MAJ-01/02 신규 25건 포함, 974 passed, 1 skipped)
venv/bin/python -m pytest -q

# CR-I5-MAJ-03: lock 재컴파일 재현성 + committed lock drift 확인
bash scripts/compile_lock.sh --verify

# 오늘 조사에서 실행한 실제 gh api 조회(재실행 시 값이 바뀔 수 있음)
gh api repos/luminous419/simple-qna-rag/environments
gh api repos/luminous419/simple-qna-rag/actions/runners
gh api repos/luminous419/simple-qna-rag/contents/.github/workflows/ci.yml --jq .sha
git rev-parse HEAD origin/master

# provisioning 계약(§2.1) — 두 external-dir 이름이 모두 필요(CR-I4-MAJ-02),
# 서로 다른 이름이 같은 경로를 가리키면 함수 경계에서도 거부(CR-I5-MAJ-02)
python scripts/ci_acceptance_contract.py --repo luminous419/simple-qna-rag \
  provisioning --runner-label self-hosted --runner-label ollama-m3 \
  --environment m3-live-regression \
  --external-dir vectorstore=runtime/vectorstore --external-dir documents=runtime/documents

# receipt 계약(§2.2) — job/artifact/expected-sha가 모두 필수(CR-I4-MAJ-02)
python scripts/ci_acceptance_contract.py --repo luminous419/simple-qna-rag \
  receipt --run-id 31204494509 \
  --require-job python-tests --require-job frontend-tests \
  --require-artifact m4-regression-report \
  --expected-sha c056342b5dd3fe85b8ea42fe78c24e6c3e0417d3

# CR-I4-MAJ-01 회귀 재현: 빈 문자열은 이제 exit 2(과거엔 silent success 오검사)
python scripts/ci_acceptance_contract.py --repo luminous419/simple-qna-rag \
  receipt --run-id 31204494509 \
  --require-job python-tests --require-artifact m4-regression-report \
  --expected-sha c056342b5dd3fe85b8ea42fe78c24e6c3e0417d3 \
  --require-conclusion ""

# receipt-m41 계약(§4 step 5) — M4.1 고정 profile(3 job/2 artifact), 부분집합
# 불가(CR-I5-MAJ-02). --allow-job-conclusion으로 특정 job의 실패를 명시적으로
# 허용할 수 있다(CR-I5-MAJ-01, §4 step 6과 동일 메커니즘)
python scripts/ci_acceptance_contract.py --repo luminous419/simple-qna-rag \
  receipt-m41 --run-id 31204494509 \
  --expected-sha c056342b5dd3fe85b8ea42fe78c24e6c3e0417d3
```
