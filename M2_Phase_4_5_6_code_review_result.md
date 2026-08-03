# M2 Phase 4·5·6 코드 최종 재리뷰 결과

리뷰 대상:

- Phase 4: `rag_engine.py`, `evaluation/retrieval.py`, `test_evaluation_retrieval.py`
- Phase 5: `evaluation/routing.py`, `test_evaluation_routing.py`, `test_agent_routing.py`, 골든셋
- Phase 6: `evaluation/answers.py`, `test_evaluation_answers.py`
- 공통 리포팅: `evaluation/reporting.py`, `test_evaluation_reporting.py`
- 이전 재리뷰에서 남긴 P2/P3 조치 사항

## 종합 평가

현재 평가: **승인(9.8/10)**

이전 재리뷰에서 남긴 Markdown table escape, 직접 Python API limit 검증, Routing `excluded_count` 설명 불일치가 모두 해결됐습니다. 앞선 리뷰의 Routing metric, 사람 판독용 Markdown, 정답 단일화 및 Answer 집계 문제도 계속 정상 상태를 유지합니다.

Phase 4·5·6 및 reporting 관련 테스트 146건, 전체 Python 테스트와 프런트엔드 테스트가 모두 통과했습니다. Phase 7 진행을 막는 문제는 발견되지 않았습니다.

## 전체 리뷰 이력 반영 확인

| 발견 사항 | 상태 | 최종 확인 결과 |
|---|---|---|
| Routing no-tool·unknown-route·exception이 PR/F1에서 제외 | 해결 | 실패 pseudo-label을 confusion matrix에 포함하고 기대 route FN으로 계산 |
| evaluator Markdown에 핵심 지표·실패 상세 누락 | 해결 | 세 evaluator 전용 renderer 제공, 실제 metric 값 테스트 통과 |
| Routing 정답 16쌍이 테스트 상수로 중복 | 해결 | 중복 상수 제거, 골든셋 하나만 정답 원천으로 유지 |
| Answer assertion `cases_scored` 과다 집계 | 해결 | scored/no-assertion/failure 분리 및 합계 invariant 적용 |
| CLI가 0 이하 `--limit` 허용 | 해결 | 세 CLI의 positive integer parser 적용 |
| 실패 사례 Markdown table의 pipe·backslash 미처리 | 해결 | 공통 cell escape helper 적용 및 세 renderer 특수문자 테스트 통과 |
| 직접 evaluator API가 0 이하 limit 허용 | 해결 | Retrieval/Answer 공개 함수 진입점에서 `ValueError`; Routing은 cases API라 해당 없음 |
| Routing `excluded_count` 설명이 이전 동작을 가리킴 | 해결 | confusion matrix에는 포함된다는 현재 계약으로 docstring 정정 |

## 최종 개선 검증

### Markdown table 안전성

`evaluation.reporting.escape_markdown_table_cell()`이 다음 처리를 공통으로 수행합니다.

- 줄바꿈과 연속 공백을 단일 공백으로 변환
- backslash escape
- Markdown table delimiter인 pipe escape
- `None`과 숫자 등 non-string 입력의 안전한 문자열 변환

Retrieval·Routing·Answer의 실패 사례 표에서 ID, 질문, route, 실패 유형, 오류 등 모든 동적 cell에 helper를 적용했습니다.

직접 재현 결과:

```text
입력 ID: none|id
입력 질문: a | b\c + 줄바꿈 + next

생성 행:
| none\|id | a \| b\\c next | document_qa |  | no_tool |  |
```

pipe가 새로운 열로 해석되지 않고 backslash 및 줄바꿈도 한 cell 안에서 보존됩니다.

### 직접 API limit 검증

- `evaluate_retrieval(..., limit=0/-1)`은 `ValueError`
- `evaluate_answers(..., limit=0/-1)`은 `ValueError`
- Routing core는 이미 필터된 `cases`를 받으며 limit 인자가 없음
- 세 CLI의 `--limit 0/-1`은 argparse exit 2

따라서 Phase 7이 evaluator API를 직접 사용하더라도 Retrieval과 Answer에서 음수 slice가 발생하지 않습니다. Phase 7 자체 CLI도 같은 positive limit 계약을 적용해야 합니다.

### Routing 계약 정합성

`excluded_count`는 no-tool과 unknown-route 사례 수를 뜻하되, 이 사례들이 confusion matrix에서 제외되는 것은 아니라는 설명으로 수정됐습니다. 실제 동작은 다음과 같습니다.

- no-tool → `no_tool` 예측 열
- unknown route → `unknown_route` 예측 열
- exception → `exception` 예측 열
- 모두 expected route 행에 포함되어 해당 클래스의 false negative로 계산

이전 재현 사례도 계속 정상입니다.

```text
정상 document_qa 1건 + no-tool document_qa 1건

accuracy            = 0.5
document_qa recall  = 0.5
confusion 반영 건수 = 2/2
```

## Phase별 최종 평가

### Phase 4 — Retrieval trace/evaluator

평가: **승인**

- production 검색 경로 하나만 유지하며 계측은 opt-in입니다.
- 네 검색 분기에서 trace 유무에 따른 결과·순서가 동일합니다.
- Recall/MRR/nDCG가 같은 정규화·dedupe source 순위를 사용합니다.
- 단계별 latency/candidate count와 전체 latency를 보존합니다.
- 실패 후 계속 실행하고 JSON 및 Markdown에 결과를 기록합니다.
- Markdown 실패 표의 특수문자도 안전하게 처리합니다.
- corpus manifest와 vectorstore fingerprint 계약을 유지합니다.

### Phase 5 — Routing evaluator/case migration

평가: **승인**

- offline/live 공통 평가 코어와 live 명시적 opt-in이 유지됩니다.
- 정상 route 및 실패 유형 전체가 정확도와 confusion/PR/F1에 일관되게 반영됩니다.
- 골든셋 `routing_regression` 16건이 정답의 유일한 원천입니다.
- Routing Markdown에서 accuracy, PR/F1, confusion, latency와 실패를 읽을 수 있습니다.
- 특수문자가 포함된 임의 route·질문·오류도 표를 깨뜨리지 않습니다.
- corpus/vectorstore 필드는 null이고 사유가 존재합니다.

### Phase 6 — Answer evaluator/worksheet

평가: **승인**

- assertion, abstention, source, intent 및 latency 집계가 요구사항에 부합합니다.
- assertion 보유 성공 사례 수와 제외 사유가 정확히 분리됩니다.
- aggregate Markdown과 사례별 worksheet가 모두 제공됩니다.
- 질의 실패는 정상 점수를 오염시키지 않고 나머지 사례를 계속 평가합니다.
- source chunk content를 결과 문서에 저장하지 않습니다.
- 실패 표와 worksheet가 복잡한 Markdown 입력을 안전하게 처리합니다.
- corpus manifest와 vectorstore fingerprint 계약을 유지합니다.

## 잘 구현된 공통 개선

- `write_report()`가 evaluator별 Markdown callback을 지원하면서 기존 기본 renderer와 호환됩니다.
- Markdown table escape 로직을 세 evaluator에 복제하지 않고 공통 helper로 통일했습니다.
- CLI와 공개 Python API 경계가 모두 잘못된 limit을 방어합니다.
- 회귀 테스트가 결과 존재 여부뿐 아니라 metric 값, confusion 합계, table cell 구조 및 제외 수 invariant를 확인합니다.
- evaluator import와 `--help`가 외부 모델이나 vectorstore를 초기화하지 않습니다.
- 변경은 Phase 4·5·6 evaluator와 공통 reporting/test 범위에 국한됐습니다.

## 남은 비차단 항목

### P3 — 비정상 Markdown I/O 실패 시 예약 JSON 정리 강화

Phase 3 리뷰에서 기록한 기존 방어 개선 항목입니다. `write_report()`는 Markdown 경로 충돌 시 예약 JSON을 정리하지만, Markdown 쓰기 도중 디스크 부족이나 권한 변경처럼 `FileExistsError`가 아닌 예외가 발생하면 JSON이 남을 수 있습니다.

함수는 예외를 반환하므로 성공으로 오인되지는 않으며 Phase 7 진행을 막지 않습니다. 향후 일반 I/O 예외에서도 이 호출이 만든 JSON을 정리하거나 임시 파일 기반 pair write를 적용할 수 있습니다.

## 검증 결과

실행일: 2026-08-03

```text
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
통과: total 76
document_qa 51 / web_search 15 / boundary 3 / unanswerable 7
Answer 평가 대상 29 / assertion document QA 22 / abstention 7
routing_regression 16

python -m evaluation.retrieval --help
exit 0, 외부 모델 초기화 없음

python -m evaluation.routing --help
exit 0, 외부 모델 초기화 없음

python -m evaluation.answers --help
exit 0, 외부 모델 초기화 없음

pytest -q test_evaluation_retrieval.py test_evaluation_routing.py \
  test_evaluation_answers.py test_evaluation_reporting.py
146 passed, 1 warning

pytest -q
301 passed, 1 skipped, 1 warning

npm test -- --run
1 test file, 9 tests passed

git diff --check
통과

Routing metric/Markdown 특수문자 재현
accuracy 0.5 / document_qa recall 0.5 / confusion 2 of 2
pipe·backslash·줄바꿈 escape 확인
```

Python warning은 공유 conda 환경의 `torchvision` image extension 로드 경고이며 Phase 4·5·6 구현 실패는 아닙니다.

실제 모델을 사용하는 Retrieval, live Routing, Answer 평가는 이번 최종 재리뷰에서 실행하지 않았습니다. 실제 품질 수치와 live artifact는 Phase 7의 로컬 baseline 실행 및 사용자 승인 단계에서 확인해야 합니다.

## 결론

Phase 4·5·6에서 발견된 baseline 수치 왜곡, 사람 판독성, 정답 중복, 집계 정확성 및 입력 경계 문제가 모두 해결됐습니다. 현재 구현과 오프라인 테스트는 Phase 7 통합 baseline이 의존할 수 있는 안정적인 상태입니다.

따라서 **M2 Phase 4·5·6을 최종 승인하며 Phase 7 진행을 권고합니다.** Phase 7에서는 세 evaluator의 fingerprint 일치 invariant, 단계 실패 시 결과 보존·non-zero 종료, top-level 재현성 필드와 실제 local baseline 실행을 중점 검증해야 합니다.
