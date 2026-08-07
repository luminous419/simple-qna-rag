# M3 Routing Simplification 독립 설계 리뷰 — Cycle 2

검토일: 2026-08-07  
검토 범위: `Requirement.md`, `Plan.md`, `Design.md`, `Routing_Simplification_Review_1.md`, `evaluation/datasets/golden.jsonl`

## 1. Gate 판정

**PASS — 9.8/10**

- CRITICAL: 0
- MAJOR: 0
- MINOR: 0
- 통과 기준: 9.7 이상, CRITICAL 0, MAJOR 0

Cycle 1 리뷰의 두 MAJOR가 모두 해소됐다. Requirement의 Unicode 경계 예시는 두 command grammar와 일치하고, Design의 현재형 구현·인수 테스트 계약은 새 S1~S12 행렬과 WEB 8 / DOCUMENT 12 / NONE 56 exact set만 사용한다. 이전 R1~R36 및 `has_particle`/`TOPIC_HEAD`/`REQUEST_TAIL`/WEB 10·DOCUMENT 12·NONE 54 내용은 비규범 감사 기록으로 명시적으로 격리됐다.

따라서 라우팅 단순화 설계는 구현 착수 가능한 상태이며 Gate를 통과한다.

## 2. Cycle 1 발견사항 재검증

### M1 — 규범 §7.2와 후속 구현·테스트 계약 충돌

**해소됨.**

- `Design.md` §7.4는 단순화 Cycle 1의 실제 classifier 규범 행렬을 S1~S12로 새로 정의했다.
- S7 `웹검색으로 최신 환율 알려줘`, S8 `웹검색에서 사용하는 API 구조 알려줘`, S9 `구글에서 사용하는 검색 기술 알려줘`는 모두 NONE 경로의 기존 LLM 반환·예외 계약을 유지한다.
- §7.4는 “S1~S12와 §7.2의 8/12/56 exact set만 현재 구현·인수 테스트에 사용”한다고 명시한다.
- 이전 R1~R36은 별도 제목 아래 “비규범 감사 기록”, “테스트로 구현하지 않는다”고 명시됐다.
- §11.1의 `routing_signals.py` 산출물은 두 command grammar, 검색 행위 술어 집합, 인용·부정 전처리, DOCUMENT 신호로 정정됐다. 폐기된 §7.2.1 어휘 상수나 `PROHIBITION_WINDOW`를 현재 산출물로 요구하지 않는다.
- 규범 §12.2는 신호 stub 12칸, S1~S12, NONE 폴백 4경로, WEB 실패 재시도만 요구한다. R1~R36 및 `TOPIC_HEAD`, `REQUEST_TAIL`, `has_particle` 테스트를 구현하지 말라고 명시한다.

### M2 — Requirement Unicode 경계 양성 예시의 grammar 충돌

**해소됨.**

Requirement M3-REQ-004의 Unicode 양성 예시는 `질문:웹에서 검색해줘`와 `(구글링해서 알려줘)`로 정정됐다. 전자는 채널+조사+검색 행위 명령, 후자는 직접 검색 명령이므로 같은 절의 두 grammar와 일치한다. 일반 응답 술어만 있는 `웹검색으로 최신 환율 알려줘`는 Design §7.2와 S7에서 일관되게 NONE/LLM이다.

## 3. 현재 규범 정합성

| 영역 | 현재 계약 | 판정 |
|---|---|---|
| Requirement | 두 WEB command grammar, 일반 응답 술어·관형절은 NONE/LLM | 일치 |
| Plan Phase 3 | 이전 fast path·주제어·거리 예외 미구현, precision-first | 일치 |
| Design §7.2 | 유일 구현 계약, WEB 8 / DOCUMENT 12 / NONE 56 | 일치 |
| Design §7.4 | 신호 stub 12칸 + 실제 classifier S1~S12 | 일치 |
| Design §11.1 | 최소 routing 모듈 산출물 | 일치 |
| Design §12.1~12.2 | 8/12/56 exact set과 S1~S12만 규범 테스트 | 일치 |
| 추적표·부록 A | 8/12/56 및 관형절 최소쌍 | 일치 |
| 이전 Iteration 1~6 | §7.2-L, R1~R36, §12.2-L로 비규범 표시 | 감사 기록으로 허용 |

문서 전체 검색에서 `has_particle`, `TOPIC_HEAD`, `REQUEST_TAIL`, WEB 10 / DOCUMENT 12 / NONE 54는 이전 Iteration 1~6 감사 기록과 그 해소 이력에만 남아 있다. 현재 규범 §7.2, S1~S12, §11.1, §12.1~12.2 및 추적표는 이를 구현하도록 요구하지 않는다.

## 4. 골든셋 독립 검증

| 검증 항목 | 결과 |
|---|---|
| 행 수 | 76 |
| SHA-256 | `61b768acd8d33522ef76e3baadd4bf19b44cc25daa79ad7ea255fb0a09d1017a` |
| WEB exact set | 8건 |
| DOCUMENT exact set | 12건 |
| NONE complement | 56건 |
| 집합 교집합·누락 | 없음 |
| 결정론 WEB expected-route 오탐 | 0 |
| 결정론 DOCUMENT expected-route 오탐 | 0 |

WEB exact set은 `ws-000`, `ws-002`, `ws-005`, `ws-007`, `ws-009`, `rr-ws-python-version-001`, `rr-ws-bitcoin-price-001`, `rr-ws-samsung-stock-001`이다. DOCUMENT exact set은 `bd-000`, `ua-000`~`ua-006`, `dq-agent-arch-001`, `dq-agent-vs-model-001`, `dq-kb-price-outlook-001`, `dq-kb-gangnam-001`이다. 나머지 56건은 NONE이며 세 집합은 76건을 정확히 분할한다.

## 5. 구현 가능성 및 잔여 위험

두 WEB grammar, 검색 행위 술어 집합 하나, 인용·부정 전처리, DOCUMENT 신호만으로 분류기를 구현할 수 있다. S1~S12는 양성, 부정, 인용, 일반 응답 술어, 관형절 최소쌍, 검색 주제, 복합어 경계, rollback을 모델 없이 검증한다. `_decide_tool()`의 신호 우선 구조와 NONE fallback 보존도 구체적이다.

결정론 WEB recall을 의도적으로 낮춘 만큼 `ws-003`과 `ws-008` 등은 LLM 품질에 의존한다. 이는 결함이 아니라 명시된 precision-first trade-off이며, Phase 3의 76건×3회 live gate에서 web recall 15/15와 accuracy 69/76 이상을 확인해야 한다. 구현 중 recall을 올리기 위해 조사·주제어·거리 예외를 다시 추가해서는 안 된다.

## 6. 결론

라우팅 단순화 Cycle 2는 1차 리뷰의 명세 충돌을 새 알고리즘 확장 없이 정리했다. `SOURCE_PARTICLE` 관형절 bypass는 제거된 상태를 유지하고, 현재형 구현·테스트 지시는 하나의 8/12/56 계약으로 수렴한다.

최종 판정은 **PASS**다. Phase 3 구현을 진행할 수 있다.
