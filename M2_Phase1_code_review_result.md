# M2 Phase 1 코드 재리뷰 결과

리뷰 대상:

- `evaluation/__init__.py`
- `evaluation/schema.py`
- `evaluation/dataset.py`
- `test_evaluation_schema.py`
- `test_evaluation_dataset.py`
- `requirements.txt`
- 관련 Requirement 및 상세 설계 동기화 내용

## 종합 평가

현재 평가: **승인(9.7/10)**

이전 코드 리뷰에서 확인한 P1 1건, P2 1건, P3 2건이 모두 구현·테스트·Requirement에 반영됐습니다. Phase 1 전용 테스트, 전체 Python 회귀 테스트, 프런트엔드 테스트, CLI/import smoke, 문서 형식 검증이 모두 통과했습니다.

Phase 2 골든셋 작성으로 진행해도 됩니다. 이번 재리뷰에서 새로 발견된 차단 문제는 없으며, 아래 P3 두 항목은 후속 유지보수 개선 권고입니다.

## 이전 리뷰 반영 확인

| 이전 발견 사항 | 상태 | 확인 결과 |
|---|---|---|
| assertion과 abstention 동시 허용 | 해결 | model validator에서 상호 배타성 강제, 충돌 테스트 추가, Requirement 동기화 |
| all-zero `relevance_grades` 허용 | 해결 | non-empty mapping에 최소 하나의 grade 1~3 요구, 단일·다중 zero 테스트 추가 |
| 오류 종류와 맞지 않는 CLI 다음 조치 | 해결 | `DatasetError.kind`로 content/io 안내 분리, stderr/stdout 테스트 추가 |
| CLI 테스트의 느슨한 종료 코드 단언 | 해결 | no-argv와 invalid subcommand에서 정확히 code 2 검증 |

## 발견 사항

### P3 — `DatasetError.kind`가 임의 문자열이라 안내 분기 오타를 정적으로 막지 못함

위치: `evaluation/dataset.py`의 `DatasetError`

현재 `kind: str = "content"`이고 CLI는 `kind == "io"`만 특별 처리합니다. 내부 코드만 생성하므로 현재 동작에는 문제가 없지만, 향후 `kind="i/o"` 같은 오타가 들어가면 오류 없이 content 안내로 fallback됩니다.

권고:

- `Literal["content", "io"]` 또는 `DatasetErrorKind` enum을 사용합니다.
- 알 수 없는 kind를 허용하지 않거나 생성 시 즉시 검증합니다.

이 항목은 Phase 2 진행을 막지 않습니다.

### P3 — Pydantic model은 생성 후 mutation으로 무결성 규칙을 우회할 수 있음

위치: `evaluation/schema.py`의 `GoldenCase`, `AnswerAssertion`

생성 시 validator는 충분히 엄격하지만 기본 Pydantic 설정에서는 생성 후 list mutation을 다시 검증하지 않습니다. 예를 들어 유효한 사례를 만든 뒤 `answer_assertions.append(...)` 또는 `relevance_grades[...] = 0`으로 변경하면 model validator가 재실행되지 않습니다.

현재 Phase 1~2 흐름은 JSONL에서 객체를 로드한 뒤 읽기 전용으로 사용하므로 실제 blocker는 아닙니다. 다만 이후 evaluator나 테스트가 model을 직접 수정하기 시작하면 불변식이 깨질 수 있습니다.

권고:

- GoldenCase를 평가 파이프라인에서 불변 객체처럼 취급한다는 원칙을 문서화합니다.
- 변경이 필요하면 `GoldenCase.model_validate({**case.model_dump(), ...변경값})`처럼 전체 데이터를 다시 검증해 새 객체를 만듭니다. Pydantic의 `model_copy(update=...)`는 update 값을 자동 재검증하지 않으므로 이 용도로 사용하면 안 됩니다.
- 실제 mutation 요구가 생길 때 `validate_assignment=True`와 tuple/immutable collection 전환을 검토합니다. `validate_assignment`만으로는 list 내부 mutation까지 잡지 못한다는 점에 유의해야 합니다.

## 잘 구현된 부분

- schema와 dataset import가 corpus, vectorstore, Ollama에 의존하지 않습니다.
- 필수 필드, extra field, strict bool, source 빈 값·중복·Unicode 정규화를 일관되게 검증합니다.
- positive grade 관계, all-zero grade 금지, assertion/abstention 상호 배타성이 model-level에서 보장됩니다.
- Answer eligibility가 공개 함수 하나로 통일돼 이후 evaluator와 공유할 수 있습니다.
- JSON 파싱·schema·중복 ID·파일·인코딩 오류가 위치와 종류를 보존한 `DatasetError`로 변환됩니다.
- 구성 validator가 실제 Answer 평가 대상에서 intent 수량을 집계하고 여러 오류를 한 번에 보고합니다.
- CLI는 JSON report를 stdout에, 진단과 다음 조치를 stderr에 분리합니다.
- 테스트가 정상·실패·경계 조건을 폭넓게 다루며 외부 서비스 없이 빠르게 실행됩니다.
- Requirement와 상세 설계가 구현된 불변식에 맞게 함께 갱신됐습니다.

## 검증 결과

실행일: 2026-08-02

```text
pytest -q test_evaluation_schema.py test_evaluation_dataset.py
81 passed in 0.06s

pytest -q
102 passed, 1 skipped, 1 warning in 3.02s

npm test
1 test file, 9 tests passed

python -m evaluation.dataset --help
exit 0, help 출력 정상

python -c "import evaluation.schema, evaluation.dataset"
exit 0

git diff --check
통과
```

전체 Python 테스트의 warning은 공유 conda 환경의 `torchvision` image extension 로드 경고이며 Phase 1 구현 실패는 아닙니다.

## 결론

Phase 1의 요구사항과 승인 조건을 충족했습니다. 이전 리뷰에서 발견된 데이터 무결성 및 CLI 진단 문제도 회귀 테스트와 함께 해결됐으므로 **Phase 1을 승인하며 Phase 2 진행을 권고합니다.**

남은 P3 항목은 API 사용 범위가 확장될 때 고려할 방어적 개선이며 현재 마일스톤 진행을 막지 않습니다.
