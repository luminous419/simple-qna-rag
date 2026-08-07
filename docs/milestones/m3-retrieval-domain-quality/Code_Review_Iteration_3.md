# M3 Code Review — Iteration 3

- 검토일: 2026-08-08
- 범위: `HEAD` 대비 현재 작업 트리의 전체 M3 변경, 최신 M3 규범 문서,
  Phase 4 ADR, `evaluation/baselines/m3_initial.{json,md}` 및
  `evaluation/reports/m3/m3-final-approved/` 전체 artifact
- 점수: **9.8 / 10**
- Gate: **PASS / APPROVE**
- Findings: **CRITICAL 0, MAJOR 0, MINOR 1, TRIVIAL 0**
- 합격 조건: score >= 9.7, CRITICAL 0, MAJOR 0

## 종합 판정

Iteration 2의 차단 사항은 모두 해소됐다. Phase 4 ADR과 승인 baseline은
2026-08-08 사용자가 variant 정체가 없는 worksheet 29건을 직접 검토하고 기존
사례별 두 축 점수와 선호를 승인했다는 증거를 보존한다. 최종 Answer child는
`schema_version=1.2.0`, `answer_template_mode=default`,
`intent.accuracy=null`, `evaluated_count=0`, `excluded_count=29` 및 명시적 제외
사유를 기록한다. 저장된 Retrieval/Routing/Answer child JSON을 다시 로드해
`evaluate_gates()`를 실행한 결과도 baseline의 `gate_evaluation`과 완전히 같고
14/14 항목이 모두 통과한다.

전체 Python 및 frontend 테스트, dataset validator, Markdown link 검사와
`git diff --check`도 모두 통과했다. M2 immutable dataset/baseline은 `HEAD` 대비
변경이 없고 승인 SHA-256과 일치한다. 따라서 남은 문서 수치 일관성 MINOR 1건은
M3 승인을 차단하지 않는다.

## Iteration 2 findings 재검증

| 기존 finding | 판정 | 근거 |
|---|---|---|
| M1 — 사람 blind-review gate 미충족 | **해소** | Phase 4 ADR에 2026-08-08 사용자가 비식별 worksheet 29건을 직접 검토하고 기존 사례별 채점을 승인한 기록이 추가됨. worksheet는 29건 모두 두 출력의 형식 적합성·핵심 사실 보존·선호가 완성됐고 decision은 `scored_cases=29`, `incomplete=0`, default 20 / intent 2 / tie 7을 기록 |
| M2 — default 모드 intent를 정확도 0%로 기록 | **해소** | 최종 Answer child가 schema 1.2.0 및 `answer_template_mode=default`를 기록하고 intent를 `accuracy=null`, `evaluated_count=0`, `excluded_count=29`로 제외. `test_default_template_mode_excludes_intent_entirely`가 intent/default 양쪽 계약을 고정 |
| m1 — child JSON 저장/재로드 gate 동등성 테스트 부재 | **해소** | `TestChildArtifactReloadGateParity`가 세 child JSON을 디스크에서 재로드하고 `evaluate_gates()` 결과와 저장된 `gate_evaluation`의 완전 동등성을 검증 |

## CRITICAL

없음.

## MAJOR

없음.

## MINOR

### m1 — Traceability의 최종 live 세부 수치 일부가 이전 rerun 값이다

`Traceability.md` §3은 Routing 정확도 run 값을 `[76,75,75]`, document 값을
`[61,60,60]`, Answer latency를 28.11s / 40.50s로 적는다. 그러나 이 문서가
최종 승인 근거로 연결한 `m3-final-approved` baseline/child와
`m3_initial.json`의 값은 각각 `[74,75,75]`, `[59,60,60]`,
27.51s / 37.34s다. 중앙값과 모든 gate 판정은 동일하게 통과하므로 제품 동작이나
승인 결과에는 영향이 없지만, 추적 문서가 명시한 최종 원본과 세부 숫자가
일치하지 않는다.

권고: 후속 문서 정리에서 `Traceability.md` §3의 run 배열과 Answer latency를
`m3-final-approved` 값으로 맞춘다. 본 리뷰 범위에서는 기존 문서를 수정하지
않았다.

## 최종 artifact 독립 대조

| 항목 | 결과 |
|---|---|
| Retrieval child schema | `1.1.0` |
| Routing child schema | `1.1.0`; prompt SHA, routing policy, candidate metadata 존재 |
| Answer child schema / mode | `1.2.0` / `default` |
| Answer intent metadata | `accuracy=null`, evaluated 0, excluded 29, 제외 사유 명시 |
| child JSON 재로드 후 gate 재계산 | baseline `gate_evaluation`과 완전 동일 |
| live gate | **14/14 pass**, `overall_pass=true` |
| child 참조 경로 | 세 경로 모두 존재하고 baseline 참조와 일치 |
| dataset/corpus/vectorstore fingerprint | Retrieval·Answer 및 승인 baseline 간 일치 |

## 직접 수행한 검증

| 검사 | 결과 |
|---|---|
| `pytest -q` | **643 passed, 1 skipped**, warning 1 |
| `npm test -- --run` | **9 passed** |
| `python -m evaluation.dataset validate evaluation/datasets/golden.jsonl` | 통과, 76 valid cases |
| `python scripts/check_markdown_links.py` | 45 files(tracked 26 + untracked 19), 103 links, 실패 0 |
| `git diff --check` | 통과 |
| M2 immutable diff (`golden.jsonl`, `m2_initial.json`, `m2_initial.md`) | `HEAD` 대비 변경 없음 |
| M2 immutable SHA-256 | `61b768...d1017a`, `e1edf2...a6d5`, `844e3c...d3f8`; 승인값과 일치 |
| `m3-final-approved` child JSON 재로드 및 `evaluate_gates()` | 저장 baseline과 동일, **14/14 pass** |

## Gate 권고

**PASS / APPROVE.** 점수 9.8/10, CRITICAL 0, MAJOR 0으로 합격 조건을
충족한다. 남은 MINOR는 최종 gate에 영향을 주지 않는 추적 문서의 세부 실측값
정정이며 후속 문서 정리로 처리할 수 있다.
