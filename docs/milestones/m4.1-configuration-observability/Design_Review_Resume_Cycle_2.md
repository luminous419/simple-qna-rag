# M4.1 상세 설계 독립 Gate 리뷰 — 설계 재개 사이클 2

검토일: 2026-08-08

검토자: Codex (독립 Gate 검토자)

검토 대상: `milestone_dev_orchestration_guide.md`, 최신 M4.1 `Design.md`,
`Field_Spec_Inventory.md`, `Traceability.md`,
`Design_Review_Resume_Cycle_1.md`, 실제 `src/simple_qna_rag/config.py`와
`evaluation.reporting`/`evaluation.baseline`/`evaluation.compare` API

구현·설계 원문 변경: 없음. 본 리뷰 파일만 추가했다.

## 1. Gate 판정

**FAIL — 구현 단계 진입 불가**

- 점수: **9.4 / 10.0**
- 발견사항: **CRITICAL 0 / MAJOR 1 / MINOR 1 / TRIVIAL 0**
- 9.7 Gate: **미통과** — 점수 미달이며 MAJOR 1건이 남았다.
- 구현 진입 승인: **불승인**

`R1-MAJ-02`는 runtime과 generator가 같은 `ModelValidatorSpec.callable`을
사용하고, 41-field 및 3-row/5-column 산출물을 같은 선언형 입력에서 생성·검사할
수 있도록 실행 가능한 수준으로 정정됐다. 그러나 `R1-MAJ-01`의 필수 test code는
Python `unittest.mock.Mock`에 존재하지 않는 공개 `.wraps` 속성을 assert하므로
그대로 구현하면 항상 실패한다. 핵심 폐쇄 증거가 실행 불가능한 이상 Gate를 통과시킬
수 없다.

## 2. 독립 검증 결과

- 현행 `config.VECTORSTORE_PATH`는 `str`이고, 공개
  `resolve_runtime_path(env_name, default_path, legacy_path, *, environ=None) -> Path`
  계약은 최신 설계의 facade/type 설명과 일치한다.
- 실제 `evaluation.reporting.build_vectorstore_fingerprint(vectorstore_path: Path)`는
  단일 인자를 받아 canonical `index.faiss`/`index.pkl`만 읽고 정확히
  `index_faiss_sha256`, `index_pkl_sha256` 두 key를 반환한다.
- 실제 `evaluation.baseline.run_baseline(dataset_path, output_dir, *, ...) -> dict`,
  `evaluation.compare.evaluate_gates(payload) -> dict`,
  `evaluation.reporting.write_report(payload, output_dir, name,
  render_markdown=None) -> tuple[Path, Path]` 시그니처는 Design §10.1과 맞는다.
- Design §10.1의 `_vectorstore_fingerprint()`는 legacy facade `str`을 호출 경계에서
  `Path`로 한 번 정규화하고 실제 fingerprint API symbol을 호출한다. 이 production
  wrapper symbol 자체는 구현 가능하다.
- Design §10.4 항목 6의 `mock.Mock(wraps=real_build_vectorstore_fingerprint)`는
  실제 함수를 delegate하며 정확히 1회·인자·two-key SHA를 관찰할 수 있다. 다만
  이어지는 `recording_wrapper.wraps` 접근은 Python 3.11에서 `AttributeError`를
  발생시킨다. `Mock`이 보관하는 내부 속성은 `_mock_wraps`이며 공개 `wraps`가 아니다.
- `Field_Spec_Inventory.md`에는 필드 41행과 validator 데이터 3행이 있고 validator
  표는 정확히 5열이다.
- `ModelValidatorSpec`은 `callable`, `constraint`, `related_fields`,
  `default_rendering`을 보유한다. runtime의 `_validator_namespace()`와 generator의
  판정 계산이 동일 `mv.callable` 객체를 사용하고, default 값과 필드 번호는
  `FIELD_SPECS`에서 계산한다. 함수 소스 파싱이나 별도 수기 validator mapping 없이
  두 산출물을 생성하는 계약이 구체적이다.
- 아직 `scripts/generate_field_spec.py`, `tests/unit/test_settings_inventory.py`,
  `tests/integration/test_m3_regression_gate.py`는 존재하지 않는다. 구현 전 설계 Gate인
  점에서 그 자체는 결함이 아니며, 이번 판정은 명시된 test가 구현 시 실행 가능한지에
  관한 것이다.

## 3. CRITICAL

없음.

## 4. MAJOR

### R2-MAJ-01 — recording wrapper의 “실제 API 호출” assertion이 실행 불가능하다

**근거**

Design §10.4 항목 6과 Traceability의 `R1-MAJ-01` 폐쇄 행은 다음 assertion을
필수 증거로 고정한다.

```python
recording_wrapper = mock.Mock(wraps=real_build_vectorstore_fingerprint)
assert recording_wrapper.wraps is real_build_vectorstore_fingerprint
```

그러나 `unittest.mock.Mock` 생성자의 `wraps=`는 delegation 설정 인자이지 생성된
Mock의 공개 `.wraps` 속성이 아니다. 실제 Python 3.11에서 두 번째 줄은 false를
반환하는 것이 아니라 `AttributeError: 'function' object has no attribute 'wraps'`를
발생시킨다. 따라서 설계가 요구하는 integration case는 정확히 1회, `Path` 인자,
실제 함수 delegation, 정확한 two-key/SHA-256 계산을 모두 수행한 뒤에도 해당 줄에서
실패한다. `R1-MAJ-01`의 폐쇄 test가 실행 가능하다는 주장을 수용할 수 없다.

**필수 수정**

공개 mock 동작으로 실제 delegation을 증명하도록 계약을 고친다. 가장 단순한 방법은
실제 함수 주위에 명시적 recording function을 두어 그 함수가
`real_build_vectorstore_fingerprint(path)`를 호출하게 하고, 별도 call list 또는
spy로 정확히 1회와 인자를 검증하는 것이다. `Mock(wraps=...)`를 유지한다면 실제
delegation은 반환 SHA와 call 기록으로 증명하고 존재하지 않는 `.wraps` assertion을
제거해야 한다. 내부 구현 세부인 `._mock_wraps`에 test 계약을 결합하는 방식은
권장하지 않는다.

수정 후에도 아래 네 증거는 한 test에서 모두 유지해야 한다.

1. recording wrapper 정확히 1회 호출
2. 유일한 positional 인자가 `Path`이며 `Path(str(tmp_path))`와 동일
3. 명시적 wrapper body가 실제
   `evaluation.reporting.build_vectorstore_fingerprint`를 호출
4. 결과 key가 정확히 두 개이고 두 값이 canonical 파일의 SHA-256과 동일

## 5. MINOR

### R2-MIN-01 — Traceability가 재개 사이클 1을 폐쇄 증거로 잘못 서술한다

Traceability 머리말은 이번 재개 사이클 증거를
`Design_Review_Resume_Cycle_1.md`의 `R1-MAJ-01`/`R1-MAJ-02` “폐쇄”라고 표현한다.
그러나 해당 리뷰는 두 항목을 새 MAJOR로 열고 Gate를 실패시킨 문서다. 폐쇄 대상의
발견 출처와 폐쇄 증거를 구분해, 사이클 1은 발견 근거로 두고 향후 PASS한 독립 리뷰를
폐쇄 증거로 가리켜야 한다.

## 6. R1 항목 재판정

| 기존 ID | 판정 | 독립 재검증 |
|---|---|---|
| R1-MAJ-01 | **미폐쇄 — R2-MAJ-01로 계승** | wrapper delegation 자체는 맞지만 필수 `.wraps` assertion이 `AttributeError`를 내므로 test 전체가 실행 불가 |
| R1-MAJ-02 | **폐쇄** | `ModelValidatorSpec`이 runtime 등록과 generator 판정의 동일 callable 원본이며, `FIELD_SPECS`와 함께 41-field 및 3-row/5-column diff-0 test를 구현할 입력·알고리즘·출력·결정론 계약을 제공 |

## 7. 요구사항 추적성과 구현 가능성

| Requirement | 판정 | 근거 |
|---|---|---|
| REQ-001 | 실행 가능 | lock, snapshot, Linux CI 제약과 test/evidence 연결이 유지됨 |
| REQ-002 | 실행 가능 | `FIELD_SPECS`/`ModelValidatorSpec`, frozen dynamic Settings, facade 호환, 41-field 및 3-row/5-column generator 계약이 구현 가능 |
| REQ-003 | 실행 가능 | positive logging schema, 금지 payload, output surface, failure swallowing이 symbol/test에 연결됨 |
| REQ-004 | 실행 가능 | 7 family, bounded labels, 102 sample 상한과 registry 수명주기 계약이 일관됨 |
| REQ-005 | 실행 가능 | import-free bootstrap, lifespan, live/ready 상태표와 deprecated alias가 연결됨 |
| REQ-006 | **부분 실행 불가** | 실제 evaluation API와 wrapper production symbol은 맞지만 R1-MAJ-01의 필수 integration test 계약이 실행 불가 |

M4.2 concurrency/timeout과 M4.3 index lifecycle/container 범위를 구현으로 끌어온
침범은 발견하지 않았다. R2-MAJ-01과 R2-MIN-01 외 Requirement → symbol → test →
evidence 연결은 구현 가능한 수준이다.

## 8. 구현 진입 재개 조건

- R2-MAJ-01의 존재하지 않는 `.wraps` assertion을 실행 가능한 명시적 recording
  wrapper 계약으로 교체한다.
- Design §10.4와 Traceability의 R1-MAJ-01 폐쇄 설명을 같은 계약으로 동기화한다.
- R2-MIN-01의 폐쇄 증거 링크 표현을 바로잡는다.
- 다음 독립 리뷰에서 **9.7 이상, CRITICAL/MAJOR 0, MINOR 최소화**를 확인한다.
- 위 조건 전에는 M4.1 제품 구현 및 M4.2/M4.3 진입을 승인하지 않는다.

## 9. 정적 검증

본 리뷰 파일 작성 후 아래 명령을 실행해 결과를 기록한다.

- `python scripts/check_markdown_links.py`: PASS (68개 파일, 292개 링크, 실패 0)
- `git diff --check`: PASS
