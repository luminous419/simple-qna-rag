# M4.1 상세 설계 독립 리뷰 — Iteration 5

검토일: 2026-08-08

검토자: Codex (독립 상세 설계 조건부 연장 1회차 리뷰)

검토 대상: `milestone_dev_orchestration_guide.md`, M4.1 최신 `Requirement.md`,
`Plan.md`, `Design.md`, `Field_Spec_Inventory.md`, `Traceability.md`,
`Design_Review_Iteration_1.md`~`Design_Review_Iteration_4.md`, 현행 제품 코드·테스트·evaluation API

구현·설계 원문 변경: 없음

## 1. 판정

**FAIL — 구현 단계 진입 불가, 조건부 연장 즉시 중단**

- 점수: **9.3 / 10.0**
- 발견사항: **CRITICAL 0 / MAJOR 1 / MINOR 1 / TRIVIAL 0**
- 9.7 Gate: **미통과** — MAJOR 1건과 점수 조건을 충족하지 못했다.
- 구현 진입 승인: **불승인**
- 조건부 연장 상태: **중단** — Iteration 4의 M4-01과 동일한 근본 문제인
  “현행 fingerprint API 호출 경로가 실행되지 않음”이 두 회 연속 재발했다.
  가이드의 조건부 연장 즉시 중단 조건 `(1) 동일한 근본 문제가 2회 연속 재발`에
  해당하므로 남은 1회의 연장을 진행하지 않는다.

Iteration 4의 `m4-01`, `m4-02`, `t4-01`은 폐쇄됐다. 그러나 `M4-01`은 함수
이름과 arity만 고쳤을 뿐 실제 전달 인자의 타입 계약을 연결하지 못해 재개방 상태다.
따라서 REQ-006.2/.4 회귀 wrapper는 여전히 문서 그대로 구현할 수 없다.

## 2. 독립 재검증

- 실제 `evaluation.reporting.build_vectorstore_fingerprint` signature는
  `(vectorstore_path: Path) -> dict`이고, `index.faiss`/`index.pkl`을 읽어
  `index_faiss_sha256`/`index_pkl_sha256` 두 key를 반환한다.
- 실제 `evaluation.fingerprint.collect_fingerprint` signature는
  `(data_dir, vectorstore_path, dataset_path) -> dict`다. 이를 wrapper에서 제거한
  방향과 canonical 2-file 범위는 맞다.
- 실제 `evaluation.baseline.run_baseline`은 `overall_success`와
  `gate_evaluation`을 함께 반환하며, 실제 `write_report(payload, output_dir, name,
  render_markdown=None)`를 한 번 호출한다. 설계의 최종 성공 조건은 맞다.
- 실제 `evaluation.compare.M3_GATES`는 14개이고 `evaluate_gates()`는 모든 item이
  `pass is True`일 때만 `overall_pass=True`다.
- 실제 `config.resolve_runtime_path` signature는
  `(env_name, default_path, legacy_path, *, environ=None) -> Path`이며, 최신 설계는
  이를 public wrapper로 보존하고 FieldSpec parser를 `_parse_runtime_path`로 분리했다.
- 실제 `config.VECTORSTORE_PATH`는 legacy facade 호환 때문에 **`str`**이다.
  이 값을 설계 §10.1 그대로 `build_vectorstore_fingerprint(VECTORSTORE_PATH)`에
  전달해 재현하면 `TypeError: unsupported operand type(s) for /: 'str' and 'str'`가 난다.
- 설치된 Prometheus client에서 `disable_created_metrics()` 후 labels 없는 Counter를
  1회 증가시키면 sample은 정확히 1개다. 따라서 7-family/102-sample 계산은 성립한다.

## 3. CRITICAL

없음.

## 4. MAJOR

### M5-01 — 회귀 wrapper가 facade의 `str` 경로를 `Path` 전용 API에 전달한다 (M4-01 재개방)

**근거**

- Design §10.1 pseudocode는
  `build_vectorstore_fingerprint(VECTORSTORE_PATH)`를 호출하지만
  `VECTORSTORE_PATH`의 정의/import/source를 별도로 고정하지 않는다.
- 같은 Design §4.2/§4.4와 Field Spec Inventory #4는 한 release 호환을 위해
  `config.VECTORSTORE_PATH`를 `facade_type=str, facade_adapter=str`로 노출하도록
  명시한다. 현행 runtime 값도 실제로 `str`이다.
- 실제 `build_vectorstore_fingerprint`는 내부에서
  `vectorstore_path / "index.faiss"`를 수행하므로 `str`을 받으면 즉시
  `TypeError`가 난다. 독립 runtime 호출로 그대로 재현됐다.
- §10.4의 “실제 arity 고정” 테스트는 signature만 검사하므로 이 인자 타입/호출부
  연결 실패를 잡지 못한다. two-file mutation fixture도 fingerprint 함수가 정상
  호출된 뒤에야 의미가 있다.

**영향**

REQ-006.2의 vectorstore 불변 검사와 REQ-006.4의 M3 자동 회귀 wrapper가 baseline
실행 전에 실패한다. 따라서 Traceability의 M4-01 폐쇄 및 REQ-006.2/.4 실행 가능
주장을 인정할 수 없고 구현 진입 Gate를 닫을 수 없다.

**재개 시 필수 수정**

wrapper가 사용할 경로의 단일 source와 타입을 명시한다. 예를 들어 typed Settings의
`Path` 값을 주입하거나, legacy facade를 써야 한다면 호출 경계에서
`Path(VECTORSTORE_PATH)`로 정규화한다. 테스트는 signature introspection에 그치지 말고
현행 facade 값과 임시 canonical 2-file 디렉터리를 사용해 실제 호출이 성공하고 정확한
두 key를 반환하는 integration case를 포함해야 한다.

## 5. MINOR

### m5-01 — Field Spec의 MODEL_VALIDATORS 표가 열 수와 행 데이터가 불일치한다

`Field_Spec_Inventory.md`의 `MODEL_VALIDATORS` 표 header는 5열(`#`, 제약, 관련 필드,
default 값, 판정)인데 세 데이터 행은 끝에 `Path`/`None` 또는 `str`/`str` 두 cell이
추가돼 7열이다. validator 증거와 facade metadata가 섞인 형태라 generated preview의
감사 가능성을 낮춘다. 구현 재개 시 불필요한 두 cell을 제거하고 generator `--check`가
동일한 5열을 생성하도록 고정해야 한다.

## 6. Iteration 4 폐쇄 재판정

| 리뷰 ID | 판정 | 독립 재검증 결과 |
|---|---|---|
| M4-01 | **재개방(M5-01)** | 함수 symbol/arity/2-file shape는 정정됐으나 실제 facade `str` 인자를 연결하면 동일 호출 경로가 `TypeError`로 실행 불가 |
| m4-01 | **폐쇄** | public 4-인자 helper와 private 2-인자 parser가 분리되고 signature test가 명시됨 |
| m4-02 | **폐쇄** | labels 없는 7번째 Counter, registry DI, 상한 102, scrape test가 일관되며 실제 sample 1개도 재현됨 |
| t4-01 | **폐쇄** | Design 머리말이 Iteration 5 실행 계약 및 Iteration 4 폐쇄 대상으로 갱신되고 리뷰 1~4 링크가 존재함 |

## 7. 전체 설계 실행 가능성과 범위 판정

| Requirement | 판정 | 이유 |
|---|---|---|
| REQ-001 | 실행 가능 | lock/tool/snapshot/CI 계약이 symbol과 테스트에 연결됨. Linux clean install은 구현 CI 증거로 남음 |
| REQ-002 | 실행 가능 | FieldSpec/from_sources/facade/public helper 계약이 구현 가능한 수준으로 정합화됨. 표 형식 오류는 m5-01 |
| REQ-003 | 실행 가능 | positive logging schema, output-surface disposition, request-id, failure swallowing과 Counter가 연결됨 |
| REQ-004 | 실행 가능 | 7-family, bounded labels, created 제외 102 sample, registry 수명주기가 일관됨 |
| REQ-005 | 실행 가능 | import-free bootstrap, lifespan, live/ready 상태표와 실패 매트릭스가 연결됨 |
| REQ-006 | **실행 불가** | M5-01 때문에 wrapper의 pre/post vectorstore fingerprint 호출이 시작 단계에서 실패함 |

M4.2 concurrency/timeout 또는 M4.3 index lifecycle/container 구현을 끌어온 범위
침범은 발견하지 않았다. 실패 원인은 범위 확대가 아니라 M4.1 내부의 경로 타입 연결
누락이다.

## 8. 9.7 Gate, 연장 중단 및 재개 조건

- **9.7 Gate FAIL**: CRITICAL 0이나 MAJOR 1, MINOR 1, 9.3/10이므로 통과하지 못했다.
- **연장 중단 조건 충족**: Iteration 4 M4-01과 Iteration 5 M5-01은 모두 실제
  `build_vectorstore_fingerprint` 호출 경로가 실행되지 않는 동일 근본 문제다.
  두 회 연속 재발했으므로 조건부 연장을 즉시 중단한다.
- **구현 진입 승인 없음**: 이 문서는 구현 진입을 승인하지 않는다.
- **재개 조건**: 별도 사용자 결정으로 새 리뷰 사이클을 승인하고, (1) 경로 source/type
  정규화, (2) facade 값을 포함한 실제 fingerprint 호출 integration test, (3)
  MODEL_VALIDATORS 5열 정합화, (4) Markdown link check와 `git diff --check` PASS를
  모두 제시한 경우에만 새 Gate를 열 수 있다. 단순히 남은 6회차를 자동 진행해서는 안 된다.

## 9. 정적 검증

아래 검사는 본 리뷰 파일 작성 후 실행한 결과다.

- `python scripts/check_markdown_links.py`: PASS (검사 65파일, 링크 274개, 실패 0)
- `git diff --check`: PASS
- 구현·설계 원문·baseline·vectorstore 변경: 없음
