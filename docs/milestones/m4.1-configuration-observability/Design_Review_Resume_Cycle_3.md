# M4.1 상세 설계 독립 Gate 리뷰 — 설계 재개 사이클 3

검토일: 2026-08-08

검토자: Codex (독립 Gate 검토자)

검토 대상: 최신 [Design.md](Design.md), [Traceability.md](Traceability.md),
[Field_Spec_Inventory.md](Field_Spec_Inventory.md),
[Design_Review_Resume_Cycle_1.md](Design_Review_Resume_Cycle_1.md),
[Design_Review_Resume_Cycle_2.md](Design_Review_Resume_Cycle_2.md), 실제 Python
3.11 `unittest.mock`과 `evaluation.reporting`/`evaluation.baseline`/
`evaluation.compare` API

구현·설계 원문 변경: 없음. 본 리뷰 파일만 추가했다.

## 1. Gate 판정

**PASS — M4.1 구현 단계 진입 승인**

- 점수: **9.8 / 10.0**
- 발견사항: **CRITICAL 0 / MAJOR 0 / MINOR 0 / TRIVIAL 0**
- 9.7 Gate: **통과**
- 구현 진입 승인: **승인**

`R2-MAJ-01`은 Python 3.11의 공개 Mock API만 사용하는 실행 가능한 계약으로
정정됐고 실제 최소 spike가 통과했다. `R2-MIN-01`도 발견, 부분 폐쇄·재개방,
최종 폐쇄의 주체와 시점을 구분하는 감사 연혁으로 정정됐다. 남은 항목은 구현과
Linux CI에서 증거를 생성할 정상 이월 사항이며 설계 Gate blocker가 아니다.

## 2. 독립 실행 검증

Python 3.11.8에서 임시 디렉터리에 canonical `index.faiss`/`index.pkl`을 만들고,
설계 §10.4 항목 6과 같은 최소 spike를 실행했다.

```python
real_fn = build_vectorstore_fingerprint
spy = mock.Mock(wraps=real_fn)
module.build_vectorstore_fingerprint = spy
module.VECTORSTORE_PATH = str(tmp_path)

result = module._vectorstore_fingerprint()

spy.assert_called_once_with(Path(str(tmp_path)))
call_arg = spy.call_args.args[0]
assert isinstance(call_arg, Path)
assert call_arg == Path(str(tmp_path))
assert result == real_fn(tmp_path)
```

결과는 **PASS**였다. `assert_called_once_with`와 `call_args`는 공개 recording
API로 정확히 1회 및 `Path` 인자를 관찰했고, `Mock(wraps=real_fn)`은 실제 함수를
호출해 정확한 two-key SHA-256 dict를 반환했다. 이전 계약의 `spy.wraps` 접근은
같은 환경에서 여전히 `AttributeError`를 발생시켜 Cycle 2 지적도 재현됐다.
따라서 이번 수정은 내부 `._mock_wraps`에 기대지 않으면서 회귀 원인을 직접
제거한다.

실제 API 시그니처와 동작도 재확인했다.

- `build_vectorstore_fingerprint(vectorstore_path: Path) -> dict`: canonical 두
  파일만 읽고 `index_faiss_sha256`, `index_pkl_sha256`을 반환한다.
- `run_baseline(dataset_path, output_dir, *, ...) -> dict`: keyword-only 옵션과
  `overall_success` 계약이 Design §10.1과 일치한다.
- `evaluate_gates(payload: dict) -> dict`: `pass is None`을 포함하면
  `overall_pass=False`가 되는 구현이다.
- `write_report(payload, output_dir, name, render_markdown=None) ->
  tuple[Path, Path]`: JSON/Markdown을 한 호출로 쓰는 계약이 일치한다.
- 현행 `config.VECTORSTORE_PATH`는 `str`이고 공개 `resolve_runtime_path(...,
  *, environ=None) -> Path` 시그니처가 유지돼, facade 경계의 단일 `Path()`
  정규화 설계가 실제 코드와 연결된다.

## 3. 발견사항

### CRITICAL

없음.

### MAJOR

없음.

### MINOR

없음.

### TRIVIAL

없음.

## 4. 이전 재개 항목 최종 판정

| 기존 ID | 최종 판정 | 독립 재검증 |
|---|---|---|
| R1-MAJ-01 | **최종 폐쇄(R2-MAJ-01로 계승 후 Cycle 3 폐쇄)** | Cycle 1이 실제 API 호출 관찰 누락을 발견했고, Cycle 2가 `.wraps` 공개 속성 오인을 재개방했으며, Cycle 3의 공개 Mock API 계약이 Python 3.11 spike에서 통과 |
| R1-MAJ-02 | **폐쇄 유지** | `ModelValidatorSpec.callable`을 runtime과 generator가 공유하고, 41-field 및 3-row/5-column 표의 입력·알고리즘·결정론·diff-0 test가 한 선언형 source에 연결됨 |
| R2-MAJ-01 | **폐쇄** | 존재하지 않는 `.wraps` assertion이 제거됐고 `wraps=`/`assert_called_once_with`/`call_args` 및 실제 함수 독립 결과 비교만 사용 |
| R2-MIN-01 | **폐쇄** | Traceability 머리말과 §2가 Cycle 1을 발견 출처, Cycle 2를 R1-MAJ-02 폐쇄 및 R1-MAJ-01 재개방 근거, Cycle 3을 R2-MAJ-01 최종 폐쇄 주체로 구분 |

Traceability의 `R1-MAJ-01` 행은 **부분 폐쇄 → 재개방 → 최종 폐쇄**를 한 행에
보존하고, 후신 `R2-MAJ-01` 행은 실제 최종 수정만 기술한다. 발견 문서와 폐쇄
증거가 더 이상 혼동되지 않으며 과거 FAIL 리뷰 원문도 변경되지 않았다.

## 5. 전체 요구사항 실행 가능성

| Requirement | 판정 | 근거 |
|---|---|---|
| REQ-001 | 실행 가능 | lock, canonical snapshot, Linux x86_64/Python 3.11/Node 22 CI 검증 경계가 symbol·test·evidence에 연결됨 |
| REQ-002 | 실행 가능 | 41-field 단일 원본, frozen dynamic Settings, facade 호환, 3-row/5-column validator generator 계약이 구체적이고 결정론적임 |
| REQ-003 | 실행 가능 | positive logging schema, 금지 payload, 전체 output surface, failure swallowing과 drop counter가 테스트 가능함 |
| REQ-004 | 실행 가능 | 7 family, label allowlist, created-series 공개 API, fresh-registry 102 sample 상한 계약이 일관됨 |
| REQ-005 | 실행 가능 | import-free bootstrap, lifespan, live/ready 상태표, deprecated alias가 acceptance test에 연결됨 |
| REQ-006 | 실행 가능 | M3 API 재사용, baseline/canonical two-file 불변성, exit 0/1/2, JSON/Markdown 판정 parity 및 공개 recording wrapper 증거가 연결됨 |

`Field_Spec_Inventory.md`는 필드 41행과 validator 3행/5열을 정확히 유지한다.
실제 generator, generated artifact, settings tests가 아직 없는 것은 구현 전 설계
Gate의 정상 상태이며 Traceability §3에 구현 증거 생성 항목으로 명시돼 있다.
Linux clean install, live evaluator opt-in, histogram 실측도 동일하게 구현/CI 단계
검증 대상으로 투명하게 이월됐다. M4.2 concurrency/timeout이나 M4.3 index
lifecycle/container를 M4.1 구현 범위로 끌어온 침범은 발견하지 않았다.

## 6. 구현 진입 조건 및 승인 범위

M4.1 구현은 승인한다. 구현자는 Design의 선언형 source와 공개 API 계약을 그대로
구현하고, Traceability §3의 이월 증거를 코드·테스트·Linux CI에서 폐쇄해야 한다.
이 승인은 M4.2/M4.3 구현의 조기 착수를 승인하는 것이 아니며, 두 milestone은
Design §12의 seam 범위만 유지한다.

## 7. 정적 검증

- `python scripts/check_markdown_links.py`: PASS (69개 파일, 302개 링크, 실패 0)
- `git diff --check`: PASS
