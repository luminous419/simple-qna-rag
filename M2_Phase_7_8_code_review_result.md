# M2 Phase 7·8 코드 최종 재리뷰 결과

리뷰 일자: 2026-08-04

리뷰 기준:

- `Development_M2_Quality_Baseline_Requirement.md`
- `Development_M2_Quality_Baseline_Development_Plan.md`
- `Development_M2_Quality_Baseline_Design.md`
- `M2_Phase_7_8_dev_detail_plan_and_rule.md`
- 이전 `M2_Phase_7_8_code_review_result.md`의 P1·P2·P3

리뷰 대상:

- 기준 commit: `7b064be` (PR #9 병합)
- 현재 브랜치의 미커밋 개선 사항
- `evaluation/baseline.py`
- `evaluation/retrieval.py`
- `evaluation/routing.py`
- `test_evaluation_baseline.py`
- `test_evaluation_retrieval.py`
- `test_evaluation_routing.py`
- `.github/workflows/ci.yml` 회귀 여부

## 종합 평가

현재 평가: **코드 승인(9.7/10)**

이전 리뷰에서 발견한 Routing report I/O 실패의 orchestration 중단, 직접 API의 live opt-in 우회, Retrieval report 경로 추측, Routing 비공개 renderer 의존이 모두 해결됐습니다. 직전 재리뷰에서 남은 opt-in 테스트 계약 불일치와 Routing report 실패 회귀 테스트 누락도 보완됐습니다.

Phase 7·8 관련 테스트 111건, 전체 Python 테스트 349건과 프런트엔드 테스트 9건이 모두 통과했습니다. dataset validation, vendor 동기화 검사와 `git diff --check`도 정상입니다. 현재 변경에서 Phase 7·8 코드 병합을 막는 결함은 발견되지 않았습니다.

다만 현재 개선 사항은 아직 미커밋 상태이므로 이 변경을 대상으로 한 GitHub Actions 결과는 존재하지 않습니다. 또한 실제 live baseline 실행, 사용자 검토·승인과 `evaluation/baselines/m2_initial.*` 고정은 별도 승인 게이트로 남아 있습니다. 따라서 **코드는 승인**하지만 **Phase 7 전체 완료 및 Phase 9 착수는 live baseline 승인 이후**로 구분해야 합니다.

## 이전 리뷰 사항 최종 확인

| 발견 사항 | 상태 | 최종 확인 결과 |
|---|---|---|
| Routing report I/O 실패가 run 전체를 중단 | 해결 | 단계 실패로 기록하고 Answer 계속 실행, 최종 baseline report 보존 |
| 직접 `run_baseline()` API가 opt-in 우회 | 해결 | API 진입점에서도 evaluator 호출 전 `RuntimeError` |
| 새 opt-in 계약과 과거 테스트 충돌 | 해결 | 새 실패 계약을 검증하도록 테스트 수정 |
| Retrieval report 경로를 directory snapshot으로 추측 | 해결 | evaluator가 실제 JSON/Markdown 경로를 직접 반환 |
| Routing 비공개 Markdown renderer 의존 | 해결 | 공개 `render_routing_markdown()` 계약 사용 |
| Routing report 실패 자동 회귀 테스트 누락 | 해결 | OSError 경로에서 상태·후속 실행·결과 보존 검증 |
| 최초 live baseline 승인 게이트 | 대기 | 코드 결함 아님; 실제 실행과 사용자 승인 필요 |

## 개선 사항 상세 평가

### Routing report I/O 실패 보존

평가: **승인**

Routing 추론 성공 후 metadata/report 생성 과정을 별도 예외 경계로 처리합니다. report 기록이 실패하면 다음 상태를 명시적으로 보존합니다.

- Routing 최종 상태: `failed`
- `evaluation_status`: `success`
- `report_status`: `failed`
- 오류 유형과 메시지
- 출력 경로
- 사용자가 취할 다음 조치
- 평가된 사례 수, 정답 수와 accuracy

이후 Answer 단계는 계속 실행되고 Retrieval과 Answer 결과가 최종 통합 report에 남습니다. 전체 결과는 `overall_success=False`가 되어 CLI non-zero 정책과도 일치합니다.

추가된 회귀 테스트는 `write_report()`가 Routing에서 `OSError`를 던지도록 만들어 다음을 검증합니다.

- Answer evaluator 호출
- Retrieval 성공 결과 보존
- Routing의 `failed`/`evaluation_status=success`/`report_status=failed`
- Answer 성공 결과 보존
- 전체 실패 판정

이전 P1 재현 조건을 수동으로도 다시 실행했으며 정상 동작했습니다.

```text
overall_success=False
routing_status=failed
evaluation_status=success
report_status=failed
stages_called=retrieval,routing,answers
최종 baseline JSON/Markdown 생성
```

### 직접 API live opt-in

평가: **승인**

CLI뿐 아니라 `run_baseline()`도 `RUN_LIVE_LLM_TESTS=1`을 확인합니다. 환경변수가 없으면 dataset, evaluator, 모델 또는 vectorstore 접근 전에 `RuntimeError`를 반환합니다.

테스트도 현재 계약에 맞게 수정됐습니다.

- opt-in 미설정 상태에서 `RuntimeError`
- 오류 메시지에 `RUN_LIVE_LLM_TESTS` 포함
- library API이므로 `SystemExit` 대신 예외 사용
- opt-in이 설정된 정상 orchestration은 기존 테스트에서 계속 검증

CLI의 선행 검사는 자세한 사용자 안내를 제공하고 API의 검사는 우회 경로를 막으므로 이중 방어가 적절합니다.

### Retrieval report 경로 반환

평가: **승인**

`evaluate_retrieval()`이 생성된 `report_json_path`와 `report_markdown_path`를 반환합니다. baseline은 디렉터리 호출 전후 snapshot을 비교하지 않고 반환된 경로를 직접 사용합니다.

이로써 다음 위험이 제거됐습니다.

- 동시 실행이 만든 다른 프로세스의 report 선택
- 서로 다른 stem의 JSON/Markdown 연결
- timing에 따른 report 경로 누락

테스트는 반환 경로와 실제 생성 파일의 일치 및 존재 여부를 검증합니다.

### Routing renderer 공개 계약

평가: **승인**

기존 `_render_routing_markdown()`을 공개 `render_routing_markdown()`으로 변경하고 Routing CLI와 통합 baseline이 동일 함수를 사용합니다. 비공개 구현 의존을 제거하면서 Markdown 로직을 복제하지 않았습니다.

기존 특수문자 escape 테스트도 공개 함수 기준으로 유지됩니다.

## Phase 7 평가

평가: **코드 승인, live baseline 승인 대기**

확인된 계약:

- validate → Retrieval → live Routing → Answer 실행 순서
- dataset 실패 시 후속 단계 `not_run`
- evaluator 실패 후 가능한 단계 계속 실행
- Routing artifact 실패 후 Answer 계속 실행
- `success`/`failed`/`skipped`/`not_run` 구분
- `--skip-routing`, `--skip-answers`, positive `--limit`, `--tag`
- CLI와 직접 API의 명시적 live opt-in
- Retrieval fingerprint의 top-level 승격
- `--skip-answers`에서도 top-level fingerprint 유지
- Retrieval/Answer fingerprint 불일치 시 전체 실패
- Routing null/not-applicable metadata 격리
- 실제 report 경로의 명시적 전달
- JSON 및 사람이 읽을 수 있는 통합 Markdown
- 실패 사례의 Markdown table escape
- import와 `--help`의 live side effect 방지
- 오류 결과 보존과 CLI non-zero 정책

코드 차원의 Phase 7 blocker는 발견되지 않았습니다.

남은 승인 게이트:

1. 현재 개선 사항 commit 및 PR CI 통과
2. 신뢰 가능한 `data/`, `vectorstore/`, Ollama 환경 확인
3. 전체 live baseline 실행
4. 실행 commit과 dirty 상태 확인
5. dataset SHA, corpus manifest SHA와 vectorstore fingerprint 확인
6. Retrieval·Routing·Answer 지표 및 주요 실패 사례 검토
7. worksheet를 포함한 사람 검토
8. 사용자 명시적 승인
9. 승인된 실제 결과만 `evaluation/baselines/m2_initial.json/.md`로 고정

측정된 품질 수치가 낮다는 사실은 M2 실패 조건이 아닙니다. 결과와 실행 환경이 정확히 기록되고 재현·비교 가능하며 사용자가 이를 검토했는지가 완료 조건입니다.

## Phase 8 평가

평가: **코드 승인, 현재 변경에 대한 PR 실행 대기**

workflow 계약은 계속 정상입니다.

- Pull Request와 `master` push trigger
- `contents: read` 최소 권한
- Python 3.11
- requirements 기반 pip cache
- 전체 dependency 설치
- `python -m pip check`
- `python -c "import web_server"`
- `pytest -q`
- golden dataset validation
- Node.js 22
- npm cache와 `npm ci`
- `npm test`
- vendor sync 후 `static/vendor/` diff 확인
- Ollama, DDGS, vectorstore, live evaluator와 secret 미사용

Node 22는 Requirement의 Node 20 이상 조건을 충족하며 현재 frontend dependency의 engine 조건에 맞춘 선택입니다.

로컬에서 workflow의 핵심 명령은 통과했습니다. 다만 현재 개선 사항이 미커밋 상태이므로 이 정확한 코드에 대한 원격 Actions 결과는 아직 없습니다. commit과 PR 생성 후 다음을 확인해야 합니다.

- `python-tests` 성공
- `frontend-tests` 성공
- 모델 가중치 다운로드와 live 서비스 접근 없음
- job별 실행 시간
- 전체 CI 10분 목표 충족 여부

## 검증 결과

실행일: 2026-08-04

```text
pytest -q test_evaluation_baseline.py test_evaluation_retrieval.py \
  test_evaluation_routing.py
111 passed, 1 warning

pytest -q
349 passed, 1 skipped, 1 warning

python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
통과: total 76
document_qa 51 / web_search 15 / boundary 3 / unanswerable 7
Answer 평가 대상 29 / assertion document QA 22 / abstention 7
routing_regression 16

python -m evaluation.baseline --help
exit 0, live dependency 초기화 없음

npm test
1 test file, 9 tests passed

npm run sync-vendor
4개 vendor 파일 동기화

git diff --exit-code -- static/vendor/
통과

git diff --check
통과
```

Python warning은 공유 conda 환경의 `torchvision` image extension 로드 경고이며 Phase 7·8 구현 실패는 아닙니다.

실제 모델을 사용하는 통합 baseline은 이번 코드 재리뷰에서 실행하지 않았습니다. 최초 측정은 현재 변경이 commit되고 CI가 통과한 뒤 현재 로컬 artifact를 대상으로 실행하여 사용자 승인 게이트를 거쳐야 합니다.

## 남은 비차단 권고

- opt-in API 테스트에 evaluator 호출 목록이 비어 있고 output 디렉터리가 생성되지 않았다는 assertion을 추가하면 "어떤 live 작업보다 먼저 차단" 계약을 더 직접적으로 고정할 수 있습니다. 현재 구현 순서와 기존 CLI 테스트로 동작은 확인되므로 승인 blocker는 아닙니다.
- Routing report 실패 테스트에서 최종 baseline JSON/Markdown 존재와 오류 path/next_action까지 assertion하면 결과 보존 계약이 더 강해집니다. 현재 수동 재현과 구현 검토로 확인됐으므로 역시 비차단 항목입니다.

## 결론

이전 리뷰에서 발견된 Phase 7 실행 안정성, 안전한 opt-in, report 경로 소유권과 공개 API 결합 문제가 모두 해결됐습니다. 직전 재리뷰의 테스트 불일치도 수정되어 전체 오프라인 회귀 검증이 통과합니다.

따라서 **M2 Phase 7·8 구현 코드를 승인합니다.**

다음 순서는 현재 변경을 commit하고 PR에서 CI를 확인한 뒤, 실제 live baseline을 실행해 지표와 주요 실패 사례를 사용자에게 제시하는 것입니다. 사용자가 승인한 후에만 `evaluation/baselines/m2_initial.json/.md`를 고정하고 Phase 9를 시작해야 합니다.
