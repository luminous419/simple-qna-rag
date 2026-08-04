# M2 Quality Baseline 상세 설계 4차 검토

검토 대상: [Development_M2_Quality_Baseline_Design.md](../../milestones/m2-quality-baseline/Design.md)

검토 기준:

- [Development_M2_Quality_Baseline_Requirement.md](../../milestones/m2-quality-baseline/Requirement.md) M2-REQ-001~004, M2-REQ-008, M2-REQ-011, M2-REQ-016
- [Development_M2_Quality_Baseline_Development_Plan.md](../../milestones/m2-quality-baseline/Development_Plan.md) Phase 1 및 Phase 3·4·6 연계 계약
- 하나의 골든 사례에 적용되는 Retrieval/Answer 지표들이 동시에 만족 가능한지 여부

## 종합 평가

현재 평가: **조건부 구현 가능(9.4/10)**

3차 리뷰의 항목은 모두 현재 설계와 상위 개발 계획에 반영됐습니다. `model_validator` import가 현재 코드 블록에 존재하며 import/CLI smoke 검증이 추가됐습니다. Answer eligibility는 `evaluation.schema.is_answer_eval_eligible()`로 단일 원천화됐고, grade mapping 책임 분리와 전체 테스트 기준선 표현도 정리됐습니다.

Phase 1 구현 직전 수준이지만, Answer 정답 필드 두 개가 상충하는 사례를 허용하는 P1 한 건은 골든셋 작성 전에 해결해야 합니다. nDCG의 all-zero grade 경계도 같은 model validator에서 함께 보완할 수 있습니다.

## 이전 리뷰 반영 확인

| 3차 발견 사항 | 상태 | 확인 결과 |
|---|---|---|
| `model_validator` import 누락 | 현재 문서에서 해결 확인 | import 목록에 포함, import smoke 및 CLI smoke 절차 추가 |
| Answer eligibility 중복 정의 | 완료 | `schema.py` 공개 함수 하나를 dataset/answers가 공유하도록 설계와 상위 계획 수정 |
| grade mapping 책임 분산 | 완료 | schema는 무결성, metrics는 계산용 mapping 생성으로 책임 명시 |
| 고정 테스트 pass 숫자 | 완료 | Phase 0 기준선으로만 보존하고 완료 조건은 새 실패 없음으로 변경 |

## 발견 사항

### P1 — assertion과 abstention을 동시에 요구하는 모순 사례가 허용됨

현재 `is_answer_eval_eligible()`과 테스트 설계는 `answer_assertions`가 존재하면서 `expect_abstention=true`인 사례도 유효한 Answer 평가 대상으로 취급합니다. `TestIsAnswerEvalEligible`에도 “둘 다 존재”하는 경우가 포함됩니다.

하지만 두 필드는 서로 다른 기대 동작을 뜻합니다.

- `answer_assertions`: 모델 답변이 하나 이상의 핵심 사실을 포함해야 함
- `expect_abstention=true`: 모델이 문서로 답할 수 없다고 거절해야 함

하나의 답변이 정상적으로 abstain하면 assertion coverage는 실패하고, assertion을 충족하는 답을 생성하면 abstention 정확도는 false negative가 됩니다. 즉 두 자동 지표를 동시에 만족할 수 없는 골든 사례를 schema가 허용합니다. Requirement의 “`answer_assertions` 또는 `expect_abstention=true`”는 eligibility 조건으로는 포괄적 OR이지만, 정답 의미상 두 값의 동시 사용 정책은 정의돼 있지 않습니다.

해결 방향:

- 기본 정책으로 두 조건을 상호 배타적으로 만듭니다.

```python
if self.answer_assertions and self.expect_abstention:
    raise ValueError(
        "answer_assertions와 expect_abstention=true를 동시에 사용할 수 없습니다"
    )
```

- 위 검증을 기존 `model_validator(mode="after")`에 통합하거나 별도 model validator로 추가합니다.
- `is_answer_eval_eligible()`의 truth table은 다음과 같이 구분합니다.
  - assertions만: 유효, eligible
  - abstention만: 유효, eligible
  - 둘 다 없음: 유효할 수 있음, not eligible
  - 둘 다 존재: schema validation 실패
- Requirement M2-REQ-003에 두 필드가 상호 배타적임을 명시합니다.
- Phase 6 evaluator는 방어적으로 둘 다 있는 입력을 받더라도 사례 실패로 기록하고 계속하도록 합니다.

두 값을 동시에 허용해야 하는 실제 사용 사례가 있다면 assertion 점수를 abstention 사례에서 제외하는 등 지표별 eligibility를 별도로 정의해야 합니다. 현재 문서에는 그런 요구가 없으므로 상호 배타적으로 만드는 편이 명확합니다.

### P2 — grade가 전부 0인 `relevance_grades` 사례의 의미와 평가 경로가 불명확

positive grade source를 `relevant_sources`에 포함하도록 수정했지만 다음 사례는 여전히 유효합니다.

```json
{
  "relevant_sources": [],
  "relevance_grades": {
    "irrelevant.pdf": 0
  }
}
```

이 사례는 nDCG 정답 mapping을 갖지만 `relevant_sources`가 비어 있어 Retrieval evaluator eligibility에서 제외됩니다. 만약 evaluator가 직접 nDCG를 계산하면 모든 gain이 0이라 IDCG가 0이 되어 지표 해석도 유효하지 않습니다.

grade 0 source를 positive relevant source 외부에 허용하는 정책 자체는 타당하지만, `relevance_grades`가 비어 있지 않은 사례에는 최소 하나의 positive grade가 있어야 nDCG 평가 사례로 의미가 있습니다.

해결 방향:

- `relevance_grades`가 비어 있지 않다면 최소 하나의 grade 1~3을 요구합니다.
- 결과적으로 grades가 있는 사례는 positive grade의 관계 검증을 통해 `relevant_sources`도 자동으로 하나 이상 갖게 됩니다.
- all-zero mapping을 거부하는 schema 테스트를 추가합니다.
- nDCG 함수 자체도 IDCG가 0인 임의 입력의 반환값 또는 오류 정책을 명시적으로 유지합니다. schema를 우회한 직접 함수 호출 테스트가 필요합니다.

### P2 — schema import smoke test는 실제 배포 파일 누락까지 검증하지 못함

`test_module_imports_without_error()`는 유용하지만 같은 테스트 모듈의 상단에서 이미 `GoldenCase` 등을 import한다면 smoke test 함수가 실행되기 전에 collection이 실패합니다. 실패 자체는 잡히지만 별도 테스트 이름으로 원인이 표시되지는 않습니다.

해결 방향:

- 테스트 모듈에서 schema 심볼을 module-level로 import한다면 별도 smoke test의 목적을 “독립 subprocess import 경로 확인”으로 바꿉니다.
- 최소 하나는 `subprocess.run([sys.executable, "-c", "import evaluation.schema, evaluation.dataset"])` 형태로 새 interpreter import를 검증하거나, 현재 §7의 shell smoke 명령을 필수 검증으로 유지합니다.

이 항목은 구현 blocker가 아니라 테스트 진단 품질 개선입니다.

### P3 — 검토 이력 설명이 본문보다 길어지는 추세

설계 본문에 1~3차 리뷰의 세부 배경과 재검증 기록이 누적돼 실제 구현 계약보다 리뷰 이력이 큰 비중을 차지하기 시작했습니다. 현재 정확성에는 문제가 없지만 이후 리뷰가 반복되면 구현자가 최신 규칙을 찾기 어려워질 수 있습니다.

해결 방향:

- §2~§8에는 최종 결정과 현재 코드·테스트만 남깁니다.
- §9에는 리뷰별 상세 서술 대신 변경 요약 표와 최종 검증 결과만 남깁니다.
- 해결 과정의 상세 근거는 `design_review.md` 또는 Git 이력에서 관리합니다.

## 잘 설계된 부분

- Phase 1 책임과 이후 Phase handoff가 구체적이고 상위 문서와 동기화돼 있습니다.
- source 정규화, 중복, 빈 값, grade 범위와 positive relevance 관계를 다층적으로 검증합니다.
- Answer eligibility가 schema의 공개 함수 하나로 통일됐습니다.
- Pydantic coercion 및 extra field 정책이 골든셋 저작 오류를 조기에 드러냅니다.
- JSONL 파싱·스키마·I/O 오류가 위치 정보와 함께 일관된 오류로 변환됩니다.
- 구성 검증은 실제 evaluator 모집단을 기준으로 여러 오류를 한 번에 보고합니다.
- 테스트와 스캐폴딩 검증이 corpus, vectorstore, 모델 없이 실행 가능합니다.

## 구현 전 권장 순서

1. `answer_assertions`와 `expect_abstention=true`를 상호 배타적으로 검증하고 Requirement를 동기화합니다.
2. non-empty `relevance_grades`에는 최소 하나의 positive grade를 요구합니다.
3. 새 interpreter 기반 import smoke 또는 §7 shell smoke를 필수 검증으로 유지합니다.
4. 구현 계약이 확정되면 리뷰 이력을 압축해 문서 가독성을 회복합니다.

첫 두 항목은 같은 model-level validation 영역의 국소 수정입니다. 반영 후에는 Phase 1을 높은 확신으로 구현할 수 있으며 전체 Phase 구조나 일정 변경은 필요하지 않습니다.
