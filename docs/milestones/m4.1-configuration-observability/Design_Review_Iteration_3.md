# M4.1 상세 설계 독립 리뷰 — Iteration 3

검토일: 2026-08-08  
검토자: Codex (독립 상세 설계 리뷰)  
검토 대상: `milestone_dev_orchestration_guide.md`, M4.1 `Requirement.md`,
`Plan.md`, `Design.md`, `Field_Spec_Inventory.md`, `Traceability.md`,
`Design_Review_Iteration_1.md`, `Design_Review_Iteration_2.md`, 현행 코드·evaluation API  
구현 변경: 없음

## 1. 판정

**FAIL — 구현 단계 진입 불가**

- 점수: **8.8 / 10.0**
- 발견사항: **CRITICAL 0 / MAJOR 2 / MINOR 2 / TRIVIAL 1**
- 9.7 Gate: **미통과** (`CRITICAL/MAJOR 0`, MINOR 최소화, 9.7 이상 조건 중
  MAJOR 2건 및 점수 조건 불충족)
- Iteration 2 폐쇄 재판정: **M2-01~M2-04, m2-01~m2-03 폐쇄 / M2-05 재개방**

Iteration 3는 bootstrap 생존성, 실행 가능한 FieldSpec 형태, base env/CLI 병합,
fallback metric, deprecation 종료점, output-surface audit, metrics 수명주기를 구체화했다.
그러나 공개 facade의 path 타입 호환 계약이 실제 코드와 모순되고, M3 회귀 wrapper가
기존 baseline API의 최종 성공 판정을 버린다. 두 문제 모두 REQ-006 호환·회귀 Gate를
직접 깨므로 설계 개정과 재리뷰 전에는 구현으로 진행할 수 없다.

## 2. 독립 재검증

- 현행 `config.py`를 runtime import해 확인한 타입은 `PROJECT_ROOT: Path`,
  `STATIC_DIR/TEMPLATES_DIR/VECTORSTORE_PATH/DATA_DIR/INTENT_MODEL_PATH: str`이다.
  Field Spec Inventory는 이 여섯 필드를 모두 `Path`로 선언한다.
- `evaluation.baseline.run_baseline()`은 `gate_evaluation`과 별도로
  `overall_success`를 반환하며, stage 성공과 retrieval/answer fingerprint invariant를
  함께 반영한다. 현행 baseline CLI도 이 값을 최종 exit 판정에 사용한다.
- `evaluation.compare.evaluate_gates()`는 실제로 14개 item을 만들고 `pass is True`가
  아닌 item이 하나라도 있으면 `overall_pass=False`로 계산한다.
- `evaluation.reporting.write_report()`는 문서 설명대로 한 호출에서 JSON/Markdown을
  함께 쓰고, `format` 인자는 없다. 이 부분의 M2-05 API signature 정정은 확인됐다.
- `evaluation.reporting.build_vectorstore_fingerprint()`와
  `evaluation.fingerprint.collect_fingerprint()`가 계산하는 vectorstore 값은 디렉터리
  전체 파일 목록 hash가 아니라 `index.faiss`와 `index.pkl` 두 SHA-256이다.
- Prometheus §7.2의 여섯 family를 가능한 label 조합으로 재계산하면 created 제외
  sample 상한은 101이다. 다만 §7.3 표시 수식에는 readiness 5가 두 번 적혀 있어
  표시된 항들을 그대로 합치면 106이 된다.

## 3. CRITICAL

없음.

## 4. MAJOR

### M3-01 — Field Spec의 path 타입과 “기존 facade 타입 불변” 계약이 동시에 성립하지 않는다

**근거**

- Field Spec Inventory #2~#6은 `STATIC_DIR`, `TEMPLATES_DIR`, `VECTORSTORE_PATH`,
  `DATA_DIR`, `INTENT_MODEL_PATH` annotation을 모두 `Path`로 고정한다.
- Design §4.4는 각 facade 상수를 `get_settings().<field>`로 직접 대입하면서 기존 공개
  심볼의 “값·타입·이름 불변”을 약속한다.
- 실제 `src/simple_qna_rag/config.py`에서 위 다섯 상수는 모두 `str`이다. 특히 기존
  테스트도 templates/static 값을 문자열로 관찰한다. 따라서 설계대로 직접 대입하면
  공개 타입이 `str -> Path`로 바뀌고, 타입을 보존하려고 `str(...)`로 변환하면
  §4.4의 생성 규칙과 Inventory의 facade 1:1 주장이 더 이상 사실이 아니다.
- 같은 경로 군에서 `PROJECT_ROOT`만 실제로 `Path`라서 일괄 변환으로 해결할 수도 없다.

**영향**

REQ-002.3과 REQ-006.1의 한 release 공개 import 호환을 깨며, string API를 기대하는
외부 consumer에서 비교·직렬화·문자열 연산이 달라진다. Traceability의 M2-02 폐쇄 근거도
41행의 타입이 “실제 config.py를 그대로 반영했다”는 잘못된 전제에 의존한다.

**필수 수정**

Settings 내부 정규 타입과 legacy facade 출력 타입을 구분하는 machine-readable
`facade_adapter`/`facade_type`을 FieldSpec에 추가하고, 위 다섯 필드는 facade에서
`str`로 투영하도록 명시한다. 또는 Settings 자체도 기존 타입을 유지한다. runtime
`type()`과 값 equality를 42개 공개 심볼 전수에 대해 이전 `config.py` snapshot과 비교하는
호환 테스트를 추가하고 Inventory/Traceability의 annotation 의미를 명확히 해야 한다.

### M3-02 — M3 wrapper가 `run_baseline()`의 최종 성공 판정을 무시해 fingerprint 실패를 PASS 처리한다 (M2-05 재개방)

**근거**

- Design §10.1 wrapper는 `payload["gate_evaluation"]["overall_pass"]`만 보고 0/1을
  반환한다.
- 실제 `run_baseline()`은 별도 `overall_success`를 계산하며, 모든 stage 성공뿐 아니라
  retrieval/answer의 corpus 및 vectorstore fingerprint 일치까지 포함한다. 기존 baseline
  CLI의 권위 있는 성공 판정도 `overall_success`다.
- 모든 evaluator가 성공하고 14 gate가 모두 통과했지만 retrieval과 answer 사이
  fingerprint가 달라진 경우 `gate_evaluation.overall_pass=True`,
  `overall_success=False`가 가능하다. 제안 wrapper는 이 경우 exit 0을 반환한다.
- §10.2/§10.3은 baseline/vectorstore pre/post 검사를 서술하지만 §10.1의 실제 wrapper
  pseudocode에는 검사 symbol이나 호출이 없고, §10.4 테스트 이름만으로 연결한다.
  더구나 §10.3의 “디렉터리 파일 목록+각 파일 sha256” 설명은 실제 재사용 API가
  `index.faiss`/`index.pkl`만 hash하는 계약과 다르다.

**영향**

REQ-006.2/.4와 “M3 14 gate와 baseline bytes 보존” acceptance에서 회귀 실행 중
vectorstore 변경을 성공으로 승인할 수 있다. 동일 판정 모델 재사용을 선언하면서 그
모델의 최종 성공 필드를 버리므로 M2-05는 폐쇄되지 않았다.

**필수 수정**

wrapper exit 0 조건을 최소 `payload["overall_success"] is True and
payload["gate_evaluation"]["overall_pass"] is True`로 고정한다. baseline 두 파일과
vectorstore의 pre/post snapshot을 wrapper의 구체 symbol 및 `try/finally` 검증 경로에
연결하고, fingerprint mismatch인데 14 gate는 모두 true인 fixture가 exit 1임을
테스트한다. vectorstore 보호 범위는 실제 M3 canonical 두 파일로 정정하거나, 디렉터리
전수를 요구한다면 별도 canonicalizer를 명시해야 한다.

## 5. MINOR

### m3-01 — metrics 이론 상한 수식에 readiness 항이 중복돼 합계와 모순된다

Design §7.3은 `rag_readiness=5`를 두 번 더한 뒤 합계를 101이라고 적었다. 실제 여섯
family 기준 올바른 식은 `6+22+44+16+5+8=101`이다. 구현 schema 자체는 올바르므로
M2-04의 fallback 추가 방향은 폐쇄로 인정하되, 감사 가능한 수식에서 중복 행을 제거하고
Traceability의 “상한식 101” 근거를 동기화해야 한다.

### m3-02 — bootstrap/app state의 “정확히 4개 속성” 주장이 같은 설계와 충돌한다

Design §3.3은 `app.state`가 정확히 네 속성이라고 하지만 §3.2는
`bootstrap_error`와 성공 시 `templates`를, §7.4는 `metrics_registry`를 추가한다.
상태 전이에 필요한 네 settings/engine 속성이라고 범위를 좁혀 표현하고, 테스트가
전체 attribute 개수를 고정하지 않도록 해야 한다. M2-01의 health 생존 계약 자체는
route-first 등록과 sanitized mount failure로 폐쇄됐다.

## 6. TRIVIAL

### t3-01 — `FieldSpec` 예시에서 `name` 필드가 두 번 선언돼 있다

Design §4.1 dataclass 코드 블록의 동일한 `name: str` 행 하나를 제거한다. Python에서
최종 annotation 하나로 축약될 수 있어 본질적 실행 blocker는 아니지만, “10-column”
정의의 감사 가독성을 해친다.

## 7. Iteration 2 항목별 폐쇄 재판정

| ID | 판정 | 독립 재검증 결과 |
|---|---|---|
| M2-01 | 폐쇄 | package-fixed bootstrap, health-first route 등록, missing/file matrix가 mount 실패 생존 경로를 닫음 |
| M2-02 | 폐쇄(신규 M3-01 별도) | `derive`/factory/annotation/validator/create_model 및 41행 전수는 제시됨. 다만 facade 호환 투영은 새 결함 |
| M2-03 | 폐쇄 | `from_sources(base_environ, cli_overrides)`의 병합 우선순위와 혼합 subprocess case가 명확함 |
| M2-04 | 폐쇄(정오표 필요) | fallback family와 가능한 조합 상한이 추가됨. 수식 중복은 m3-01 |
| M2-05 | **재개방** | 실제 signature는 맞췄으나 권위 있는 `overall_success`와 fingerprint 실패를 wrapper가 무시(M3-02) |
| m2-01 | 폐쇄 | RFC 8594 Sunset 날짜와 제거 버전 0.3.0이 Requirement/Roadmap 계약으로 확정됨 |
| m2-02 | 폐쇄 | print/logging/stdio/uvicorn 4종 정적 audit와 동적 output capture가 결합됨 |
| m2-03 | 폐쇄 | process 1회 설정과 registry별 factory가 별도 symbol로 분리됨 |

## 8. 새 모순과 범위 판정

- **새 모순:** Settings의 `Path` annotation과 legacy facade `str` 타입 불변,
  `gate_evaluation=True`만 보는 wrapper와 기존 `overall_success=False`, metrics 상한식의
  중복 readiness 항, app state “정확히 4개”와 추가 state 속성이 확인됐다.
- **M4.2 범위:** queue saturation/orphan은 enum seam과 향후 cardinality 재계산 의무만
  남겨 실제 동시성/timeout 구현 침범이 없다.
- **M4.3 범위:** index lifecycle/container 구현 침범은 없다. M3 회귀 fingerprint는
  M4.1 보존 범위지만 “디렉터리 전수 hash”로 새 계약을 확장하지 말고 기존 canonical
  두 파일과 맞추거나 확장 필요성을 별도 승인해야 한다.
- **제품 범위:** 30개 신규 env alias는 REQ-002의 설정 단일 원본 범위 안이지만, legacy
  facade 타입 변경은 관측 기반 refactor가 아니라 공개 API 변경이므로 허용되지 않는다.

## 9. 9.7 Gate 도달을 위한 최소 폐쇄 조건

1. FieldSpec/Inventory에 Settings 내부 타입과 legacy facade 타입 변환을 구분하고 42개
   공개 symbol의 runtime 타입·값 호환 테스트를 추가한다.
2. M3 wrapper가 `overall_success`와 14-gate를 함께 판정하고 pre/post immutable checks를
   실행 가능한 symbol로 포함하도록 개정한다.
3. fingerprint 보호 범위를 실제 `evaluation.fingerprint` API와 일치시킨다.
4. metrics 상한식 중복, app state 속성 표현, FieldSpec 중복 행을 정정한다.
5. 위 수정 후 CRITICAL/MAJOR 0, MINOR 최소화, **9.7/10 이상**을 다시 독립 판정한다.
