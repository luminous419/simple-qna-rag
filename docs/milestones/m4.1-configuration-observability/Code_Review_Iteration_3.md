# M4.1 구현 독립 코드 리뷰 — Iteration 3

검토일: 2026-08-09  
검토자: Codex (독립 구현 Gate 리뷰어)

검토 대상은 `milestone_dev_orchestration_guide.md`, M4.1
`Requirement.md`/`Plan.md`/`Design.md`/`Traceability.md`,
`Code_Review_Iteration_2.md`, 최신 tracked/untracked 전체 구현 diff와 live report
`evaluation/reports/m4_regression/baseline_20260808T155819908435Z.json`/`.md`이다.
제품 코드와 구현 원문은 수정하지 않았고 본 리뷰 문서만 추가했다.

## 1. Gate 판정

**FAIL — 통합·인수 단계 진입 불가**

- 점수: **9.5 / 10.0**
- 발견사항: **CRITICAL 0 / MAJOR 1 / MINOR 0 / TRIVIAL 0**
- 9.7 Gate: **미통과**
- 통합·인수 단계 진입: **거절**
- 사용자 결정 필요: **없음**. 구현 선택의 모호성은 없으며, 실제 GitHub
  protected environment/self-hosted runner에서 workflow를 실행해 성공 증거를
  남기는 운영 검증만 필요하다.

CR-I2-MAJ-02는 코드, 신규 테스트와 실제 live report로 폐쇄됐다.
CR-I2-MAJ-01도 workflow의 정적 구현 계약은 폐쇄됐으나, Iteration 2가 명시한
“실제 workflow run 성공 증거”가 아직 없다. Requirement의 모든 필수 Gate PASS 및
`UNKNOWN`/`NOT_RUN` 불승인 원칙상 self-hosted infrastructure가 실제로
provision/configure되어 전체 job이 실행됐다는 증거 없이 CI acceptance를 승인할 수
없다.

## 2. 발견사항

### CRITICAL

없음.

### MAJOR

#### CR-I3-MAJ-01 — self-hosted live CI는 실행 가능하게 설계됐지만 실제 인프라 실행 증거가 없다

`.github/workflows/ci.yml`의 `m3-live-regression-gate`는 다음 정적 결함을
올바르게 보수했다.

- `if:` allow-list가 `workflow_dispatch` 또는 `push`의
  `refs/heads/master`만 허용하므로 `pull_request` 이벤트에서는 self-hosted job이
  실행되지 않는다. job-level 조건은 checkout/install보다 먼저 평가된다.
- `environment: m3-live-regression`, checkout 외부의
  `/opt/simple-qna-rag-data/{vectorstore,documents}`, locked install,
  vectorstore canonical two-file 존재/sha256 preflight, Ollama
  version/endpoint/model preflight, `timeout-minutes: 45`, non-cancelling 단일
  concurrency group, `if: always()` report artifact가 서로 연결돼 있다.
- Settings alias는 실제로 `SIMPLE_QNA_RAG_VECTORSTORE_DIR`와
  `SIMPLE_QNA_RAG_DOCUMENTS_DIR`를 소비하며, Ollama model alias도 workflow와
  일치한다. 따라서 이전 checkout-clean 경로 삭제 문제는 코드상 해소됐다.

그러나 protected environment의 required reviewer 설정, `ollama-m3` label의 실제
runner, `/opt/...` 소유권/runner service account의 read-only 권한, documents와
canonical vectorstore의 실제 배치, Ollama/model 상태, 45분 내 종료, artifact
upload 성공은 repository diff로 생성되지 않는다. 제공된 live report는 로컬 dirty
worktree 실행의 훌륭한 evaluator 증거이지 GitHub Actions run receipt가 아니다.
Traceability도 실제 `ubuntu-latest`와 self-hosted job은 다음 push/dispatch에서
확인해야 한다고 인정한다. 따라서 CR-I2-MAJ-01의 **코드 수정은 폐쇄**됐지만 그
항목의 필수 마지막 조건인 실제 workflow 성공은 CR-I3-MAJ-01로 승계한다.

필수 폐쇄 증거:

1. repository의 `m3-live-regression` environment에 required reviewer가 설정된
   화면/API 증거와, 승인 전 job이 self-hosted runner에 배치되지 않는 run 증거.
2. trusted `workflow_dispatch` 또는 protected `master` push 한 건에서 두 preflight,
   locked install, live wrapper(exit 0), timeout/concurrency 정책 및 artifact upload가
   모두 성공한 Actions run URL/로그/artifact.
3. runner service account 기준 두 외부 디렉터리가 workspace 밖이며 read-only임을
   확인한 provisioning evidence. 실패 run에서도 report artifact step의 실제 동작을
   확인하면 `if: always()` 계약까지 폐쇄할 수 있다.

### MINOR

없음.

### TRIVIAL

없음.

## 3. Iteration 2 MAJOR 폐쇄 재검증

| ID | Iteration 3 판정 | 독립 증거 |
|---|---|---|
| CR-I2-MAJ-01 | **구현 폐쇄 / 운영 증거 미폐쇄(CR-I3-MAJ-01)** | PR event 배제 allow-list, protected environment 선언, checkout 외부 Settings alias, two-file hash/Ollama preflight, 45분 timeout, 단일 concurrency, always-upload artifact를 코드·44개 targeted test로 확인. 실제 Actions run/protected environment/runner read-only provisioning 증거는 없음 |
| CR-I2-MAJ-02 | **폐쇄** | wrapper가 `WARMUP_CASES=3`을 전달하고 retrieval/answers raw report의 `warmup.performed`를 재확인. single-run은 top-level `document_route_correct` raw count를 사용하며 aggregate 경로와 14개 threshold는 불변. true/below-threshold/missing-count 테스트와 실제 61/61 gate를 확인 |

`evaluation.compare._routing_gate_inputs()` 변경은 single-run에서 이전에 버리던
recall 비율을 raw count로 대체할 뿐 multi-run median 경로를 건드리지 않는다.
양수 warmup은 같은 engine object에서 3건을 실행하고 측정에서 제외한다. 실제
retrieval/answers report 모두 `same_process=true`,
`engine_object_id_matches=true`, `discarded_from_metrics=true`, executed/succeeded
3/3, failed 0을 기록하므로 기존 M3 측정 semantics를 훼손하지 않는다.

## 4. Live report 독립 검증

`baseline_20260808T155819908435Z` JSON과 Markdown을 원시 stage report 및 현재
파일 해시와 대조한 결과는 다음과 같다.

| 항목 | 판정 | 증거 |
|---|---|---|
| 전체 stage | PASS | validate/retrieval/routing/answers 모두 `status=success`, `overall_success=true` |
| 14 Gate | PASS | JSON item 14개 전부 `pass=true`, `gate_evaluation.overall_pass=true`; Markdown도 동일 ID/metric/threshold/pass 14행 |
| positive warmup | PASS | retrieval/answers 양쪽 performed/same_process/object match/discarded=true, 3/3 성공 |
| single-run routing count | PASS | raw routing report `aggregate=null`, `run_count=1`, `document_route_correct=61`; Gate metric 61, denominator 61, pass true |
| fingerprint invariant | PASS | `checked=true`, `ok=true`, corpus와 vectorstore retrieval/answers match 모두 true |
| vectorstore 현재 불변 | PASS | report와 현재 `runtime/vectorstore/index.faiss`/`index.pkl` SHA-256가 각각 `c52fb2...9820`/`3f7217...91bb00`로 일치 |
| baseline 현재 불변 | PASS | `git diff --exit-code -- evaluation/baselines/m3_initial.{json,md}` exit 0; 현재 SHA-256 `e7e12b...f976`/`cda916...8df0` |
| JSON/Markdown parity | PASS | overall 결과, 14개 gate ID/metric/threshold/pass와 fingerprint 서술이 동일 |

이 report는 CR-I2-MAJ-02 및 로컬 live regression acceptance를 충분히 증명한다.
다만 `git_dirty=true`이고 GitHub run 식별자/URL/artifact provenance가 없으므로
CR-I3-MAJ-01의 CI 운영 증거를 대신하지 않는다.

## 5. Requirement 영향

| Requirement | 판정 | 근거 |
|---|---|---|
| REQ-001 | **부분 PASS** | fresh macOS Python 3.11 venv에서 hash locked install과 `pip check` PASS, lock verify PASS. 실제 Linux `ubuntu-latest` CI receipt는 없음 |
| REQ-002 | PASS | 전체 suite와 schema/audit/link 검증 PASS; 기존 Iteration 2 폐쇄 유지 |
| REQ-003 | PASS | 전체 suite와 logging audit PASS; 기존 payload-safe 증거 유지 |
| REQ-004 | PASS | 전체 suite에서 bounded metrics/live traffic 회귀 PASS |
| REQ-005 | PASS | 전체 suite에서 health/bootstrap matrix PASS |
| REQ-006 | **부분 PASS** | 실제 local live report 14/14, parity, fingerprint/baseline/vectorstore 불변은 PASS. self-hosted CI 실제 실행은 NOT_RUN |

## 6. 검증 명령과 결과

| 명령 | 결과 |
|---|---|
| fresh `/tmp` venv: `pip install --require-hashes -r requirements.lock`; editable `--no-deps`; `pip check` | **PASS — No broken requirements found** |
| `venv/bin/pytest -q` | **PASS — 873 passed, 1 skipped** |
| CR-I2 targeted: regression wrapper/evaluation gates/two preflight test files | **PASS — 44 passed** |
| `bash scripts/compile_lock.sh --verify` | **PASS — 102 packages, reproducible, no drift** |
| dependency snapshot, field spec check, logging callsite audit, golden dataset validate | **PASS** |
| `npm test`; `npm run sync-vendor`; vendor diff | **PASS — 9 tests, vendor diff 0** |
| `scripts/check_markdown_links.py` (리뷰 작성 전) | **PASS — 72 files, 311 links, failures 0** |
| live JSON/raw stage/Markdown/hash 대조 | **PASS — 14/14, overall true/true, warmup/fingerprint/parity/invariance 확인** |
| GitHub Actions self-hosted/protected environment 실행 | **NOT_RUN — run URL/log/artifact/provisioning evidence 없음** |
| `git diff --check` (리뷰 작성 전) | **PASS** |

기존 project `venv`의 `pip check`는 작업 범위 밖에서 설치된
`langchain-classic==1.0.0`/`langgraph-prebuilt==1.0.2` 때문에 실패했으나 두
패키지는 `requirements.lock`에 없다. 완전히 새로운 venv의 locked install 후
`pip check`가 PASS했으므로 lock 결함으로 판정하지 않았다.

## 7. 통합 재진입 조건

CR-I3-MAJ-01의 실제 protected self-hosted workflow 성공 증거와 Linux hosted
clean job 성공 증거를 확보한다. 그 전에는 필수 Gate에 NOT_RUN이 남으므로 9.7
Gate와 통합·인수 단계 진입을 승인하지 않는다. 증거가 확보되고 새 코드 결함이
없다면 코드 수정이나 사용자 제품 결정 없이 재리뷰할 수 있다.
