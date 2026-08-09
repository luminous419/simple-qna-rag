# M4.1 상세 설계 독립 Gate 리뷰 — 설계 재개 사이클 1

검토일: 2026-08-08

검토자: Codex (독립 Gate 검토자)

검토 대상: `milestone_dev_orchestration_guide.md`, M4.1 `Requirement.md`,
`Plan.md`, `Stop_Report.md`, `Design_Review_Iteration_5.md`, Claude 갱신본
`Design.md`, `Field_Spec_Inventory.md`, `Traceability.md`, 실제
`src/simple_qna_rag/config.py`와 `evaluation.reporting`/`evaluation.baseline` API

구현·설계 원문 변경: 없음. 본 리뷰 파일만 추가했다.

## 1. Gate 판정

**FAIL — 구현 단계 진입 불가**

- 점수: **9.2 / 10.0**
- 발견사항: **CRITICAL 0 / MAJOR 2 / MINOR 0 / TRIVIAL 0**
- 9.7 Gate: **미통과** — 점수 미달이며 MAJOR 2건이 남았다.
- 구현 진입 승인: **불승인**

`Path(VECTORSTORE_PATH)` 정규화와 canonical 2-file 범위는 실행 가능하게
정정됐다. 그러나 새 integration case는 요구된 실제 API 호출을 관찰하지 않아
대체 구현도 통과할 수 있고, `MODEL_VALIDATORS` 5열을 `--check`가 재생성할 수 있는
단일 입력 모델도 설계되지 않았다. 따라서 `M5-01`과 `m5-01`의 폐쇄 주장을 아직
수용할 수 없다.

## 2. 독립 검증 결과

- 현행 `config.VECTORSTORE_PATH`는 실제로 `str`이며
  `resolve_runtime_path(env_name, default_path, legacy_path, *, environ=None) -> Path`
  공개 helper 계약도 확인했다.
- 실제 `evaluation.reporting.build_vectorstore_fingerprint(vectorstore_path: Path)`는
  `vectorstore_path / "index.faiss"`, `vectorstore_path / "index.pkl"`을 읽어 정확히
  `index_faiss_sha256`, `index_pkl_sha256` 두 key를 반환한다.
- 임시 canonical 두 파일을 만든 뒤 `Path(str_path)`로 정규화해 실제 API를 호출한
  결과 TypeError 없이 두 key와 각 파일 SHA-256이 정확히 반환됐다. 따라서 Design
  §10.1의 helper 구현 자체는 실행 가능하다.
- Design §10.1은 경로 source를 module-import된 legacy facade
  `VECTORSTORE_PATH: str` 하나로 고정하고 helper 경계에서 한 번만 `Path(...)`로
  바꾼다. 별도 `Settings` 생성 경로가 없어 source/type 계약도 일관적이다.
- `Field_Spec_Inventory.md`의 현재 `MODEL_VALIDATORS` header, delimiter, 데이터
  3행은 모두 정확히 5열이다.
- 실제 저장소에는 아직 `scripts/generate_field_spec.py`가 없다. 구현 전 설계
  단계인 점은 허용되지만, 설계된 `MODEL_VALIDATORS` 자료형은 callable tuple뿐이라
  5열의 관련 필드/default 표현/판정을 결정론적으로 생성할 자료를 담지 않는다.
- 실제 `run_baseline()`은 payload에 `overall_success`를 제공하고, Design은 이를
  `gate_evaluation.overall_pass`와 함께 요구한다. 실제 fingerprint와 report API
  signature 및 canonical 2-file 범위는 최신 §10과 맞는다.

## 3. CRITICAL

없음.

## 4. MAJOR

### R1-MAJ-01 — integration case가 `build_vectorstore_fingerprint` 실제 호출을 증명하지 않는다

**근거**

Design §10.4 항목 6은 module의 `VECTORSTORE_PATH`를 임시 `str` 경로로 바꾸고
`_vectorstore_fingerprint()`의 반환 key와 SHA-256 값만 검사한다. 그러나
`build_vectorstore_fingerprint` symbol에 spy/wrapper를 설치하거나 호출 횟수와
인자 타입/값을 assert하지 않는다. 구현자가 helper 안에서 두 파일을 직접 읽어 같은
dict를 만들거나 다른 함수로 동일 결과를 만들더라도 이 테스트는 통과한다.

이는 재개 조건의 핵심인 “실제 `build_vectorstore_fingerprint` 호출”을 검증하는
것이 아니라 동등한 출력만 검증한다. Iteration 4/5의 실패 원인이 호출부와 실제 API의
연결이었으므로 결과 기반 간접 검증만으로는 동일 계열 회귀를 닫기에 부족하다.

**필수 수정**

integration test에서 module이 참조하는 `build_vectorstore_fingerprint`를 실제 함수를
호출하는 recording wrapper로 monkeypatch하고, `_vectorstore_fingerprint()` 호출 후
다음을 함께 assert한다.

1. recording wrapper가 정확히 1회 호출됨
2. 전달 인자가 `Path`이고 `Path(str(tmp_path))`와 같음
3. wrapper가 실제 `evaluation.reporting.build_vectorstore_fingerprint`를 호출함
4. 반환 key가 정확한 두 key뿐이고 값이 각 canonical 파일 SHA-256과 같음

### R1-MAJ-02 — 5열 `MODEL_VALIDATORS` 표의 generator `--check` 입력 계약이 없다

**근거**

Design §4.1의 `MODEL_VALIDATORS`는
`tuple[Callable[[Settings], Settings], ...]`이며 세 함수만 보유한다. 이 구조에는
5열 표가 요구하는 표시용 제약 문자열, 관련 필드 번호/이름, default 비교 표현,
default 판정 metadata가 없다. 같은 절은 generator가 `FIELD_SPECS`를 순회한다고만
정의하며, `MODEL_VALIDATORS` 표를 생성하는 알고리즘이나 canonical serialization을
정의하지 않는다.

반면 `Field_Spec_Inventory.md`와 `Traceability.md`는 구현 단계의
`scripts/generate_field_spec.py --check`가 이 5열 표를 동일하게 생성한다고 이미
폐쇄 판정한다. 현 계약으로는 generator가 함수 이름/소스 코드를 취약하게 해석하거나
별도 수기 mapping을 가져야 하므로, 표와 실행 validator가 drift 없이 같은 source에서
생성된다는 보장이 없다. 단순히 현재 Markdown 행을 5열로 고친 것만으로는 Stop Report의
재개 조건 3을 완전히 충족하지 못한다.

**필수 수정**

`ModelValidatorSpec` 같은 선언형 schema에 최소 `callable`, `constraint`,
`related_fields`, canonical default rendering을 두고 `MODEL_VALIDATORS`의 단일
원본으로 삼거나, 동등하게 결정론적인 generator 입력/출력 계약을 명시한다.
`--check`가 41-field 표와 3-row/5-column validator 표 모두를 생성하고 checked-in
artifact와 diff 0을 검사한다는 테스트를 Traceability에 연결해야 한다.

## 5. MINOR

없음.

## 6. 요구사항 추적성과 구현 가능성

| Requirement | 판정 | 근거 |
|---|---|---|
| REQ-001 | 실행 가능 | lock, snapshot, Linux CI 경계와 테스트가 연결됨. clean install 결과는 구현 CI 증거로 이월됨 |
| REQ-002 | **부분 실행 불가** | typed Settings/facade/path helper는 구체적이나, REQ-002.6의 생성·검사 증거가 R1-MAJ-02로 완결되지 않음 |
| REQ-003 | 실행 가능 | positive logging schema, 금지 payload, output surface, failure swallowing이 symbol/test에 연결됨 |
| REQ-004 | 실행 가능 | 7 family, bounded labels, created 제외 102 sample, registry 수명주기 계약이 일관됨 |
| REQ-005 | 실행 가능 | import-free bootstrap, lifespan, live/ready 상태표와 deprecated alias가 연결됨 |
| REQ-006 | **부분 실행 불가** | fingerprint helper 자체는 실행 가능하지만 R1-MAJ-01 때문에 요구된 실제 API 호출 증거와 폐쇄 추적성이 부족함 |

M4.2 concurrency/timeout이나 M4.3 index lifecycle/container 구현을 끌어온 범위
침범은 발견하지 않았다. 나머지 Requirement → symbol → test → evidence 연결은
구현 가능한 수준이다.

## 7. 이전 blocker 재판정

| 기존 ID | 판정 | 독립 재검증 |
|---|---|---|
| M5-01 | **부분 폐쇄, Gate상 미폐쇄(R1-MAJ-01)** | `str` 단일 source와 `Path` 정규화 helper는 실제 실행 가능하나 test가 지정 API 호출 자체를 관찰하지 않음 |
| m5-01 | **재개방(R1-MAJ-02)** | 현재 Markdown은 5열이나 generator가 동일 5열을 재생성할 입력/알고리즘 계약이 없음 |

## 8. 재개 Gate 조건

- R1-MAJ-01의 recording integration test 계약을 설계에 반영한다.
- R1-MAJ-02의 선언형 validator inventory와 generator `--check` 계약을 설계·필드
  inventory·Traceability에 같은 source로 반영한다.
- 다음 독립 리뷰에서 **9.7 이상, CRITICAL/MAJOR 0, MINOR 최소화**를 다시 확인한다.
- 구현 및 M4.2/M4.3 진입은 그 전까지 승인하지 않는다.

## 9. 정적 검증

본 리뷰 파일 작성 후 아래 명령을 실행해 결과를 기록한다.

- `python scripts/check_markdown_links.py`: PASS
- `git diff --check`: PASS

