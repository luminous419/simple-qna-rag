# M4.1 상세 설계 독립 리뷰 — Iteration 4

검토일: 2026-08-08  
검토자: Codex (독립 상세 설계 최종 기본 회차 리뷰)  
검토 대상: `milestone_dev_orchestration_guide.md`, M4.1 `Requirement.md`,
`Plan.md`, `Design.md`, `Field_Spec_Inventory.md`, `Traceability.md`,
`Design_Review_Iteration_1.md`~`Design_Review_Iteration_3.md`, 현행 제품 코드·테스트·evaluation API  
구현·설계 원문 변경: 없음

## 1. 판정

**FAIL — 구현 단계 진입 불가**

- 점수: **9.2 / 10.0**
- 발견사항: **CRITICAL 0 / MAJOR 1 / MINOR 2 / TRIVIAL 1**
- 9.7 Gate: **미통과** — CRITICAL/MAJOR 0, MINOR 최소화, 9.7 이상 중
  MAJOR 1건과 점수 조건을 충족하지 못했다.
- 기본 iteration: **4회 완료로 종료**
- 조건부 연장 자격: **충족** — CRITICAL 0, 점수 9.0 이상, MAJOR 1건(2건 이하),
  Iteration 3의 8.8점·5건에서 9.2점·4건으로 실질 개선됐고, 잔여 문제는 현재
  문서/API 정합화로 구체적이며 해결 가능하다. 따라서 가이드에 따라 최대 2회의
  추가 회차 자격은 있으나, 이번 4차 Gate 자체가 통과한 것은 아니다.

Iteration 3의 facade 타입, metrics 합계, app state 표현, FieldSpec 중복 수정은
확인됐다. 그러나 M3 회귀 wrapper가 존재하지 않는 1-인자 fingerprint API를
호출하므로 핵심 자동 Gate를 실행할 수 없다. 이 결함은 M3-02 폐쇄 주장의 근거인
baseline/vectorstore 불변성과 REQ-006.2/.4를 동시에 막는다.

## 2. 독립 재검증

- 현행 `evaluation.baseline.run_baseline(dataset_path, output_dir, *, ...)`는
  `overall_success`와 `gate_evaluation`을 반환하고 JSON/Markdown을 한 번의
  `write_report(..., render_markdown=...)` 호출로 생성한다. 설계의 최종 성공 조건
  `overall_success is True and overall_pass is True` 자체는 올바르다.
- 현행 `evaluation.fingerprint.collect_fingerprint`의 실제 signature는
  `collect_fingerprint(data_dir, vectorstore_path, dataset_path)`이다. 반환값도
  dataset/corpus/vectorstore/git/python을 포함한 복합 dict이며, vectorstore 경로
  하나만 받는 overload는 없다.
- canonical vectorstore 2-file hash의 직접 API는
  `evaluation.reporting.build_vectorstore_fingerprint(vectorstore_path)`이며,
  `index.faiss`와 `index.pkl` SHA-256을 반환한다.
- `evaluation.compare.M3_GATES`는 14개이고, `evaluate_gates()`는 모든 item의
  `pass is True`일 때만 `overall_pass=True`로 계산한다.
- 현행 `config.resolve_runtime_path`는
  `(env_name, default_path, legacy_path, *, environ=None) -> Path`이고 단위 테스트가
  이 signature와 legacy fallback을 직접 사용한다. 설계 §4.3의 2-인자 호출과 다르다.
- 실제 public uppercase data symbol은 내부 `_TRUE_VALUES`를 제외하면 41개이고,
  `resolve_runtime_path`를 합치면 호환 감사 대상 42개라는 산정은 맞다.
- metrics 여섯 family의 created 제외 이론 상한 `6+22+44+16+5+8=101`은 맞다.
  다만 logging 설계가 별도로 증가시키는 `logging_dropped_fields_total`의 저장소와
  registry 소유권은 family=6/101 assertion에 포함되지 않았다.

## 3. CRITICAL

없음.

## 4. MAJOR

### M4-01 — M3 regression wrapper의 fingerprint 호출은 현행 API로 실행되지 않는다 (M3-02 재개방)

**근거**

- Design §10.1/§10.3은
  `from evaluation.fingerprint import collect_fingerprint` 후
  `collect_fingerprint(VECTORSTORE_PATH)`를 pre/post 두 번 호출한다.
- 실제 symbol은 `collect_fingerprint(data_dir, vectorstore_path, dataset_path)`로
  필수 positional 인자가 3개다. 설계대로 구현하면 baseline 실행 전에 즉시
  `TypeError`가 발생한다.
- 이 함수는 vectorstore 전용 fingerprint가 아니라 dataset/corpus/git metadata까지
  수집한다. 단순히 인자 세 개를 채우면 corpus와 dataset도 불변 판정에 암묵적으로
  추가되어 Design §10.3의 “canonical 2-file만, 범위 확장 없음” 계약과 달라진다.
- vectorstore 두 파일만 비교하려면 현행
  `evaluation.reporting.build_vectorstore_fingerprint(VECTORSTORE_PATH)`가 정확한
  API다. 또는 반환 shape와 비교 대상을 명시한 별도 adapter가 필요하다.

**영향**

REQ-006.2의 vectorstore 불변 증거와 REQ-006.4의 자동 회귀 wrapper가 실행 불가다.
따라서 Traceability의 M3-02 폐쇄와 M2-05 최종 폐쇄를 인정할 수 없고, 통합 Gate의
exit 0/1/2 테스트도 현재 pseudocode로 작성할 수 없다.

**필수 수정**

§10.1/§10.3의 import·호출·반환 shape를 실제 vectorstore 전용 API로 교체하고,
pre/post 두 dict의 정확한 비교 필드를 명시한다. 테스트는 실제 symbol을 monkeypatch
또는 호출해 잘못된 arity를 잡고, 14 gate true/`overall_success=False`, 2-file
mutation, 환경 미충족의 exit 1/1/2를 각각 검증해야 한다. Traceability의 M3-02는
그 수정 전까지 재개방 상태로 바꿔야 한다.

## 5. MINOR

### m4-01 — `resolve_runtime_path`의 보존 대상 signature와 새 parser 역할이 충돌한다

Design §4.3은 `resolve_runtime_path(raw, project_root)`를 path parser처럼 호출하지만,
현행 public helper와 테스트는 `(env_name, default_path, legacy_path, *, environ)` 계약을
사용한다. §4.4와 Inventory는 이 함수를 42번째 호환 심볼로 세면서 name/type/value만
검사하고 callable signature와 legacy fallback은 검사하지 않는다. 기존 helper를
호환 wrapper로 유지하고 새 2-인자 parser를 별도 private symbol로 만들거나, 호환 범위에서
helper를 제외한다는 명시적 결정을 Requirement/Traceability에 동기화해야 한다.

### m4-02 — dropped-field counter가 metrics registry·상한·테스트에서 누락됐다

Design §6.2는 금지 key를 drop할 때 `logging_dropped_fields_total`을 증가시킨다고
요구하지만 §7.2는 여섯 family만 정의하고 §7.3/§7.5는 family=6, sample=101을
assert한다. 이 이름이 Prometheus Counter라면 실제 sample은 최소 1개 늘고 registry
소유권도 정해야 한다. process-local integer라면 명칭·관찰 API와 테스트 주입 seam을
명시해야 한다. 150 상한을 깨지는 않지만 현재 두 섹션을 동시에 그대로 구현할 수 없다.

## 6. TRIVIAL

### t4-01 — Design 머리말의 iteration 상태와 리뷰 링크가 한 회차 뒤처져 있다

Design은 여전히 “Iteration 2 리뷰 폐쇄 대상”이라고 표시하고 리뷰 링크도 Iteration 2까지만
나열한다. 실행 계약 내용은 Iteration 3 지적을 반영하므로 상태와 감사 링크를 최신화해야 한다.

## 7. Iteration 3 지적 폐쇄 재판정

| 리뷰 ID | 판정 | 근거 |
|---|---|---|
| M3-01 | **폐쇄** | `facade_type`/`facade_adapter`, 5개 path의 `str` 투영, `PROJECT_ROOT`의 `Path` 유지와 42-symbol runtime 호환 테스트가 연결됐다. |
| M3-02 | **재개방** | 최종 성공 조건은 수정됐지만 pre/post vectorstore 호출이 실제 API와 불일치해 실행 불가(M4-01). |
| m3-01 | **폐쇄** | 상한식이 `6+22+44+16+5+8=101`로 정정돼 readiness 중복이 없다. |
| m3-02 | **폐쇄** | `app.state`는 core lifecycle 4개만 표현하고 전체 attribute 수를 고정하지 않는다. |
| t3-01 | **폐쇄** | `FieldSpec.name` 선언은 1회뿐이다. |

## 8. Requirement 실행 가능성 및 범위 판정

| Requirement | 판정 | 이유 |
|---|---|---|
| REQ-001 | 실행 가능 | lock 도구/version, canonical body, Linux CI, snapshot schema가 symbol/test로 연결된다. clean install은 구현 CI 증거가 필요하다. |
| REQ-002 | 조건부 실행 가능 | Settings/facade 경로는 구체적이나 public helper signature 보존 범위가 m4-01로 남는다. |
| REQ-003 | 조건부 실행 가능 | logging schema와 출력 표면은 구체적이나 dropped counter의 관찰 구현이 m4-02로 미정이다. |
| REQ-004 | 조건부 실행 가능 | 여섯 핵심 family와 101 상한은 실행 가능하나 dropped counter 포함 여부를 확정해야 정확한 registry assertion이 된다. |
| REQ-005 | 실행 가능 | bootstrap/lifespan/readiness 상태표와 TestClient/subprocess seam이 실제 FastAPI 구조에 대응한다. |
| REQ-006 | **실행 불가** | M4-01 때문에 baseline/vectorstore 불변 및 통합 회귀 wrapper가 현행 evaluation API로 실행되지 않는다. |

새 M4.2 동시성/timeout이나 M4.3 index lifecycle/container 구현을 끌어온 범위 침범은
발견하지 않았다. M3 gate 실행과 canonical 2-file read-only fingerprint는 M4.1의
회귀 보존 범위다. 반대로 3-인자 `collect_fingerprint` 전체를 사용해 corpus/dataset/git
비교까지 암묵적으로 확장하는 것은 현재 §10.3 범위를 침범하므로 피해야 한다.

## 9. 9.7 Gate 및 조건부 연장 최소 폐쇄 조건

이번 기본 4회차는 **9.7 Gate FAIL**로 종료한다. 조건부 연장 자격은 충족하므로 다음
회차를 진행한다면 아래를 모두 닫은 뒤 독립 재검증해야 한다.

1. M4-01의 실제 fingerprint symbol/signature/shape를 정정하고 M3-02 및 REQ-006.2/.4
   테스트 연결을 다시 증명한다.
2. m4-01의 public helper signature 보존 여부와 테스트 범위를 한 가지 계약으로 고정한다.
3. m4-02 counter의 저장소·registry·sample 상한을 일관되게 고정한다.
4. t4-01 상태/링크를 최신화하고 Markdown link check 및 `git diff --check`를 통과한다.

연장 회차에서도 동일 근본 API 불일치가 다시 재발하거나, 새 MAJOR가 증가하거나,
두 회 연속 점수가 개선되지 않으면 가이드에 따라 남은 횟수와 무관하게 중단해야 한다.

## 10. 문서 정적 검증

- `python scripts/check_markdown_links.py`: PASS (실패 링크 0)
- `git diff --check`: PASS
- 구현·설계 원문·baseline·vectorstore 변경: 없음
