# M2 Phase 2 코드 재리뷰 결과

리뷰 대상:

- `evaluation/datasets/golden.jsonl`
- `evaluation/README.md`
- `data/` 원문과 재생성된 `vectorstore/`
- `evaluation/schema.py`, `evaluation/dataset.py`의 검증 규칙
- `test_agent_routing.py`와 Phase 2·5 개발 계획

## 종합 평가

현재 평가: **조건부 승인(9.4/10)**

이전 리뷰의 P1 3건이 모두 보완됐습니다. 확장자가 없던 경제 PDF는 `.pdf`로 정정됐고 vectorstore에도 포함됐습니다. 비교·절차형 질문의 독립적인 필수 사실은 별도 assertion으로 분리됐으며, Phase 2 산출물인 `evaluation/README.md`도 추가됐습니다.

골든셋 validator, 전체 Python 테스트, 프런트엔드 테스트와 문서 형식 검증이 모두 통과했습니다. 데이터셋과 인덱스에 새 차단 문제는 발견되지 않았으므로 기술적으로 Phase 3 진행이 가능합니다.

단, 개발 계획이 요구하는 두 사람 검토 게이트의 최종 승인 주체는 사용자입니다. 이 리뷰는 source와 assertion의 기술적 검토 결과를 제공하지만, 사용자의 명시적 승인 기록까지 대신하지는 않습니다.

## 이전 리뷰 반영 확인

| 이전 발견 사항 | 상태 | 확인 결과 |
|---|---|---|
| 확장자 없는 경제 PDF가 인덱싱되지 않음 | 해결 | 파일명이 `.pdf`로 변경됐고 재생성된 vectorstore의 18개 source에 포함됨 |
| 경제 관련 3개 사례의 source ID 불일치 | 해결 | `dq-econ-growth-revision-001`, `dq-econ-growth-rate-001`, `dq-econ-consumption-001`이 새 basename 사용 |
| 복합 질문이 독립 사실을 `any_of` 하나에 합침 | 해결 | 비교·절차 사례의 필수 사실을 여러 `AnswerAssertion` 객체로 분리 |
| `evaluation/README.md` 누락 | 해결 | source ID, assertion, abstention, validator 및 사람 검토 가이드 추가 |
| Phase 2 사람 검토 게이트 미완료 | 사용자 승인 필요 | 기술 검토상 승인 가능하며 계획 체크리스트에는 사용자 결정 반영 필요 |
| 라우팅 회귀 16건 미이관 | Phase 5 예정 | 개발 계획상 Phase 5 작업이므로 Phase 2 차단 사항 아님 |

## 발견 사항

### P2 — 사람 검토 게이트의 사용자 승인 기록이 아직 필요함

관련 위치:

- `Development_M2_Quality_Baseline_Development_Plan.md:1010-1011`
- `evaluation/README.md`의 `사람 검토 게이트`

개발 계획은 Phase 2 완료 전에 다음 두 항목을 사용자가 승인하도록 명시합니다.

1. 골든셋 source relevance 검토
2. 골든셋 answer assertion 검토

이번 재리뷰 결과 두 항목 모두 기술적으로 승인 가능한 상태입니다. 모든 document QA 사례가 source를 가지며, source ID가 실제 corpus와 일치합니다. 수정된 비교·절차 assertion의 핵심 문구도 해당 원문에서 확인됐습니다.

다만 계획의 체크리스트는 아직 미체크 상태이며 사용자의 명시적인 승인 기록이 없습니다. 프로젝트가 문서화한 프로세스를 그대로 적용한다면, 사용자가 결과를 확인하고 승인한 뒤 체크리스트를 갱신해야 Phase 2를 형식적으로 완료 처리할 수 있습니다.

권고:

- 사용자가 본 리뷰의 source relevance 및 assertion 검토 결과를 승인합니다.
- 승인 후 개발 계획의 Phase 2 체크박스 두 개를 갱신합니다.

### P2 — 라우팅 회귀 16건의 골든셋 이관은 Phase 5에서 누락 없이 수행해야 함

관련 위치:

- `test_agent_routing.py:27`
- `Development_M2_Quality_Baseline_Development_Plan.md:642-655`

현재 골든셋에는 `routing_regression` 태그가 없고 기존 16개 질문은 `test_agent_routing.py`에 하드코딩되어 있습니다. 이는 개발 계획에서 Phase 5 작업으로 명시했으므로 Phase 2 결함은 아닙니다.

향후 Phase 5에서 태그 기반 loader만 추가하고 사례 이관을 빠뜨리면 평가 대상이 0건이 될 수 있습니다.

권고:

- 기존 질문과 기대 route 16쌍을 의미 변경 없이 골든셋에 이관합니다.
- `routing_regression` 태그 부분집합이 정확히 16건이고 비어 있지 않음을 테스트합니다.
- 이관 후 `ROUTING_CASES`를 삭제해 정답 원천을 골든셋으로 단일화합니다.

### P3 — 파일 확장자와 vectorstore 정합성은 수동 절차에 의존함

이번에는 PDF 파일명을 정정하고 vectorstore도 정상 재생성했습니다. 다만 골든셋 validator는 source가 실제 파일로 존재하는지, 등록 가능한 확장자인지, 현재 vectorstore에 포함됐는지를 검사하지 않습니다. 같은 유형의 오류가 향후 데이터셋 수정에서 재발할 수 있습니다.

`evaluation/README.md`가 이를 명확히 경고하고 있어 Phase 2 진행을 막지는 않습니다.

후속 권고:

- Phase 3 이후 별도의 corpus audit 명령 또는 테스트를 검토합니다.
- 골든셋의 정규화된 source ID 집합이 `data/**/*.pdf`, `data/**/*.txt` 및 인덱스 metadata의 source 집합에 포함되는지 검사합니다.
- vectorstore provenance 부재는 기존 계획에 따라 M2 이후 개선 항목으로 유지합니다.

## source relevance 검토 결과

- `data/`의 등록 가능한 원문은 PDF 15개와 TXT 3개, 총 18개입니다.
- 재생성된 vectorstore는 18개 source, 389개 chunk를 포함합니다.
- 골든셋의 42개 document QA 사례에는 모두 `relevant_sources`가 있습니다.
- 정규화된 골든셋 source ID는 실제 corpus basename과 일치합니다.
- 수정된 경제 PDF가 vectorstore에 포함되고 관련 3개 사례가 `.pdf` source ID를 사용합니다.
- unanswerable 7건은 현재 corpus에서 답을 찾을 수 없는 주제로 유지됐습니다.

평가: **기술 검토 승인 가능**

## answer assertion 검토 결과

이전 리뷰에서 지적한 복합 assertion은 다음과 같이 개선됐습니다.

| 사례 | 개선 결과 |
|---|---|
| `dq-sparse-vs-dense-001` | Sparse와 Dense 특성을 2개 assertion으로 분리 |
| `dq-retriever-vs-reranker-001` | 단일 인코더와 교차 인코더를 분리 |
| `dq-rag-vs-langgraph-001` | 단방향 구조와 순환 구조를 분리 |
| `dq-econ-growth-revision-001` | 수정치 `0.7%`와 하향 폭 `1.0%p`를 분리 |
| `dq-rag-pipeline-001` | 문서 로드, 텍스트 분할, 임베딩을 단계별로 분리 |
| `dq-docloader-steps-001` | source 선택, 수집, 필터링/전처리를 분리 |
| `dq-textsplit-steps-001` | chunk size와 chunk overlap을 분리 |
| `dq-realestate-procedure-001` | 구청 허가와 실거주 의무를 분리 |

분리된 필수 문구는 관련 PDF/TXT의 추출 텍스트에서 확인됐습니다. 각 `any_of`도 독립 사실이 아닌 동의어·표기 변형 위주로 구성됐습니다.

평가: **기술 검토 승인 가능**

## 잘 구현된 부분

- 62개 사례가 schema와 composition validator를 모두 통과합니다.
- 카테고리는 document QA 42건, web search 10건, boundary 3건, unanswerable 7건으로 구성됐습니다.
- Answer 평가 대상 29건, assertion 포함 document QA 22건, abstention 7건으로 최소 조건을 충족합니다.
- comparison, explanation, procedure, yes/no, uncertain intent가 요구 수량을 충족합니다.
- 모든 질문이 한국어를 포함하며 한국어 비율은 100%입니다.
- 독립 assertion 분리로 부분 답변이 만점을 받는 문제가 해소됐습니다.
- README가 향후 골든셋 작성자가 같은 오류를 반복하지 않도록 구체적인 잘못된 예와 올바른 예를 제공합니다.
- source 정규화, retrieval 전용 사례, assertion의 한계, abstention 작성 원칙을 명확히 문서화했습니다.
- 기존 Python 및 프런트엔드 동작에 회귀가 없습니다.

## 검증 결과

실행일: 2026-08-02

```text
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
통과: total 62
document_qa 42 / web_search 10 / boundary 3 / unanswerable 7
answer evaluation 29 / assertion document_qa 22 / abstention 7
한국어 비율 100%

pytest -q
102 passed, 1 skipped, 1 warning

npm test -- --run
1 test file, 9 tests passed

git diff --check
통과

vectorstore/index.pkl metadata 점검
18 sources / 389 chunks
수정된 2025년 한국 경제 전망 PDF 포함

수정 assertion 원문 대조
비교·절차형 핵심 문구가 관련 PDF/TXT 추출 텍스트에 존재함

routing_regression 태그
0건 (개발 계획상 Phase 5 이관 대상 16건)
```

Python 테스트의 warning은 공유 conda 환경의 `torchvision` image extension 로드 경고이며 Phase 2 구현 실패는 아닙니다. Live LLM 라우팅 테스트는 기본 skip 조건 때문에 실제 모델을 호출하지 않았습니다.

## 결론

이전 리뷰에서 Phase 2 승인을 막았던 source 인덱싱, assertion 품질, 작성 가이드 누락 문제가 모두 해결됐습니다. 자동 검증과 원문·인덱스 대조 결과도 정상입니다.

따라서 **기술적으로 Phase 2를 조건부 승인하며 Phase 3 진행을 권고합니다.** 사용자가 source relevance와 answer assertion 검토 결과를 명시적으로 승인하고 개발 계획의 체크리스트를 갱신하면 Phase 2를 최종 완료 처리할 수 있습니다. 라우팅 회귀 사례 이관은 계획대로 Phase 5에서 수행하면 됩니다.
