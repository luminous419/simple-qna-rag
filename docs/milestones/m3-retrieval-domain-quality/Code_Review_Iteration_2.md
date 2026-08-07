# M3 Code Review — Iteration 2

- 검토일: 2026-08-07
- 범위: `HEAD` 대비 현재 작업 트리의 모든 M3 변경, M3 규범 문서, Phase 4 ADR,
  `evaluation/baselines/m3_initial.{json,md}`, `m3-final-rerun` 전체 artifact
- 점수: **8.8 / 10**
- Gate: **STOP / REJECT**
- Findings: **CRITICAL 0, MAJOR 2, MINOR 1, TRIVIAL 0**
- 합격 조건: score >= 9.7, CRITICAL 0, MAJOR 0

## 종합 판정

Iteration 1의 M1(통합 gate 불일치), M2(Routing schema/metadata), m1(추적 문서
노후화)은 해소됐다. 최종 rerun의 baseline은 14/14 항목을 모두 `pass=true`,
`overall_pass=true`로 기록하며, 참조 Retrieval/Routing/Answer artifact의 지표와
일치한다. Routing child는 schema 1.1.0, router prompt SHA-256, routing policy 및
candidate metadata를 포함하고, Traceability와 승인 baseline도 현재 숫자와 경로를
반영한다.

그러나 Iteration 1 M3의 핵심 조건인 **사람의 blind review**는 아직 충족되지
않았다. 또한 default-template 채택 후 Answer evaluator가 intent accuracy를 제외해야
한다는 상세 설계와 달리 최종 artifact가 오해 가능한 `0.0`을 공식 기록한다. 따라서
현재 작업 트리는 M3 완료/승인 baseline으로 승격할 수 없다.

## Iteration 1 findings 재검증

| 기존 finding | 판정 | 근거 |
|---|---|---|
| M1 — authoritative aggregate gate 실패 | **해소** | `m3-final-rerun/baseline_20260807T142338609328Z.json`의 14개 항목이 모두 `pass=true`, `overall_pass=true`; Retrieval latency 2205.49/2290.58ms가 child와 일치 |
| M2 — baseline Routing schema/metadata 누락 | **해소** | 최종 Routing child가 `schema_version=1.1.0`, non-null `router_prompt_sha256`, `routing_policy`, candidate block을 기록; baseline이 공용 `build_routing_payload()` 사용 |
| M3 — 사람 blind-review gate 미충족 | **미해소 (MAJOR)** | ADR이 여전히 “에이전트 대리 블라인드 채점”이라고 명시하며 실제 사람의 사례별 blind worksheet 검토 증거가 없음 |
| m1 — 추적 문서/수치 노후화 | **해소** | Traceability, Roadmap, 승인 baseline이 rerun 경로와 14/14 결과 및 최신 640/1 테스트 결과를 반영 |

## CRITICAL

없음.

## MAJOR

### M1 — Phase 4의 필수 사람 blind review가 사용자 승인으로 대체됐다

근거:

- Plan Phase 4 수용 기준은 “같은 question/context를 사용한 29개 paired 결과와
  **완성된 사람 검토**”를 명시한다.
- Requirement §4.2는 paired blind review와 사례별 두 검토 축 기록을 요구한다.
- `evaluation/reports/m3/m3-p4-intent-ab/ADR.md`는 실제 채점자를 “에이전트
  대리”라고 명시하고, 이를 사람 검토의 자동 대체라고 설명한다.
- 이후 사용자가 집계 결과와 default 채택을 승인한 기록은 존재하지만, 이는 variant
  정체를 가린 상태에서 29개 사례를 사람이 직접 평가했다는 증거가 아니다.

영향: 사용자 승인 gate와 사람 blind-review evidence gate는 서로 다른 단계다.
현재 기록은 결론 승인만 증명하며, 평가 편향을 통제하도록 설계된 핵심 증거를
충족하지 않는다. Traceability와 baseline의 “paired blind 평가 완료” 및 M3 완료
표시는 실제 증거보다 강하다.

필수 조치: 사람이 비식별 worksheet 29건을 직접 검토하고 사례별 두 축과 선호를
완성한 뒤 기존 key로 재집계한다. reviewer 역할과 승인 기록을 보존하고, 결과에
따라 ADR/default/baseline/추적 문서를 정정한다.

### M2 — default 모드 Answer report가 비활성 intent를 공식 정확도 0%로 기록한다

근거:

- Design §8.6은 `ANSWER_TEMPLATE_MODE="default"`에서 Answer report의
  `intent.accuracy`를 `null`로 두고
  `intent_excluded_reason="ANSWER_TEMPLATE_MODE=default (classifier 비활성)"`를
  기록하도록 명시한다.
- 최종 Answer JSON과 baseline의 Answer stage는 각각
  `intent={evaluated_count:29, excluded_count:0, correct_count:0, accuracy:0.0}`을
  기록한다. 제외 사유 필드도 없다.
- `evaluation/answers.py`는 응답의 호환값 `other`를 실제 classifier 출력처럼
  29건 모두 채점하며, template mode에 따른 제외 분기가 없다.
- Traceability는 공개 계약 보존을 완료로 표시하고, 승인 baseline은 default 모드를
  채택했으므로 이 잘못된 0%는 공식 M3 artifact에 포함된다.

영향: 비활성 classifier를 실패한 classifier로 표시해 schema 의미가 실제 production
경로와 어긋난다. downstream 비교/대시보드가 0% 회귀로 해석할 수 있고, 명시된
M3-REQ-009 호환성·metadata 계약을 위반한다.

필수 조치: evaluator가 template mode를 구조화 metadata로 기록하고 default 모드에서는
intent를 전부 제외해 `accuracy=null`, `evaluated_count=0`, `excluded_count=29` 및
명시적 제외 사유를 출력하도록 한다. intent/default 양쪽 테스트를 추가하고 공식
Answer 및 통합 baseline을 재실행한다.

## MINOR

### m1 — Iteration 1 M1이 요구한 저장/재로드 gate 동등성 회귀 테스트가 없다

`test_evaluation_baseline.py`는 14개 gate의 존재와 wiring을 확인하지만, 최종 baseline이
참조하는 세 child JSON을 디스크에서 다시 읽어 `evaluate_gates()` 결과가 저장된
`gate_evaluation`과 동일함을 검증하지 않는다. 현재 rerun artifact는 수동 대조상
일치하므로 즉시 기능 결함은 아니지만, Iteration 1에서 발생했던 in-memory/저장 artifact
불일치를 재발 방지하지 못한다. exact child payload serialize/reload parity 통합 테스트를
추가해야 한다.

## 긍정 관찰

- 최종 Retrieval은 warm-up 3건, 동일 process/engine, 성공 42건, fallback 0,
  query embedding 42회, candidate embedding 0회, stored-vector lookup 2100회를 기록한다.
- 최종 Routing은 3회 `[76,75,75]`, document `[61,60,60]`, web `[15,15,15]`이며
  변동 사례를 별도 기록한다.
- Answer evaluator v2는 official profile, reviewed variants SHA 및 rules fingerprint를
  포함하고 assertion/abstention 회귀가 없다.
- 상세 live report는 Git ignored 경로에 있어 질문·모델 답변·로컬 실행 metadata의
  우발적 커밋 위험을 줄인다. 변경 diff에서 credential/private-key 노출은 발견하지 못했다.
- M2 golden dataset 및 baseline 2개는 `HEAD`와 byte-level diff가 없고 SHA-256은 각각
  `61b768...d1017a`, `e1edf2...a6d5`, `844e3c...d3f8`이다.

## 직접 수행한 검증

| 검사 | 결과 |
|---|---|
| `pytest -q` | **640 passed, 1 skipped**, warning 1 |
| `npm test` | **9 passed** |
| `python -m evaluation.dataset validate evaluation/datasets/golden.jsonl` | 통과, 76 valid cases |
| `python scripts/check_markdown_links.py` | 43 files(tracked 26 + untracked 17), 103 links, 실패 0 |
| `git diff --check HEAD` | 통과 |
| M2 immutable diff (`golden.jsonl`, `m2_initial.json`, `m2_initial.md`) | `HEAD` 대비 변경 없음 |
| M2 immutable SHA-256 | 승인 hash와 일치 |
| `m3-final-rerun` child ↔ aggregate 수동 대조 | 14/14 gate 값·fingerprint·참조 경로 일치 |

## Gate 권고

**STOP / REJECT.** 현재 점수 8.8/10, CRITICAL 0, MAJOR 2이므로 합격 조건을
충족하지 않는다. 사람 blind review를 실제 완료하고 default-mode intent metadata를
설계 계약대로 수정·재실행한 뒤, 저장 child artifact 재로드 동등성 테스트까지 추가해
독립 재검토해야 한다.
