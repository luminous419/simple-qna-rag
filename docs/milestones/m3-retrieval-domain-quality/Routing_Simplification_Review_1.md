# M3 Routing Simplification 독립 설계 리뷰 — Cycle 1

검토일: 2026-08-07  
검토 범위: `Requirement.md`, `Plan.md`, `Design.md` 규범 §7.2 및 그 구현·테스트 참조, `Design_Review_Iteration_6.md`, `evaluation/datasets/golden.jsonl`

## 1. Gate 판정

**REVISE — 8.8/10**

- CRITICAL: 0
- MAJOR: 2
- MINOR: 0
- 통과 기준: 9.7 이상, CRITICAL 0, MAJOR 0

단순화 방향 자체는 타당하다. 규범 §7.2는 WEB을 두 command grammar로 제한하고 `SOURCE_PARTICLE` fast path, `TOPIC_HEAD`, 조사 뒤 관형절 cue, 거리 상수를 모두 제거했다. 따라서 Iteration 6에서 지적된 `웹검색에서 사용하는 API 구조 알려줘` 및 `구글에서 사용하는 검색 기술 알려줘`의 관형절 우회는 새 규범에서는 구조적으로 사라진다. 결정론 규칙이 놓친 표현을 NONE/LLM에 위임하는 precision-first 원칙도 구현 가능하고 이전안보다 작다.

그러나 같은 문서의 구현·통합 테스트 계약과 요구사항의 경계 예시가 폐기된 Iteration 6 결과를 여전히 요구한다. 구현자가 규범 §7.2와 후속 테스트 계약을 동시에 만족할 수 없으므로 현재 상태로 Phase 3 구현 착수는 승인할 수 없다.

## 2. 발견사항

### MAJOR

#### M1. 규범 §7.2와 필수 통합 테스트 계약이 서로 반대 결과를 요구한다

- 위치: `Design.md` §7.2(특히 WEB 8 / DOCUMENT 12 / NONE 56 및 boundary table), §7.4 실제 classifier 행렬 R1~R36, §12.2 `test_agent_routing_policy.py`, §11.1 `routing_signals.py` 산출물 표
- 근거:
  - 새 규범은 `웹검색으로 최신 환율 알려줘`와 `웹검색으로 이번 학기 수업방식 알려줘`를 일반 응답 술어이므로 NONE/LLM으로 보낸다.
  - 하지만 §7.4의 R28·R30·R34는 같은 형태를 `has_particle=True` fast path로 WEB에 고정한다. R1~R3도 `웹 기준으로 답해줘`, `인터넷에서 알려줘`, `온라인 자료를 보여줘`를 WEB으로 요구하여 새 두 번째 grammar의 “검색 행위 술어만 허용” 계약과 충돌한다.
  - §12.2는 이 폐기된 R1~R36 전체를 여전히 필수 통합 테스트로 지정하며 `TOPIC_HEAD`, `REQUEST_TAIL`, `SOURCE_PARTICLE` 기반 설명을 구현 요구처럼 남긴다.
  - §11.1은 신규 모듈 산출물에 “어휘 상수(§7.2.1)”와 `PROHIBITION_WINDOW`를 요구하지만 현재 유일 규범 §7.2에는 §7.2.1이 없고, 해당 참조는 비규범 감사 기록 §7.2-L의 세부를 가리킨다.
- 영향: 규범대로 구현하면 필수 통합 테스트 명세가 실패하고, 통합 테스트대로 구현하면 단순화 목적과 관형절 우회 제거가 다시 깨진다. “§7.2-L은 구현 입력이 아니다”라는 선언만으로는 후속 규범성 테스트 지시를 무효화할 수 없다.
- 수정 요구:
  1. §7.4의 실제 classifier 행렬을 새 두 grammar 기준으로 재작성한다. 최소한 R1~R3, R9, R17, R28, R30, R34의 기대값과 근거를 다시 정하고, 관형절 최소쌍 두 건을 명시적으로 NONE으로 고정한다.
  2. §12.2를 새 행렬만 참조하도록 바꾸고 `TOPIC_HEAD`, `REQUEST_TAIL`, `has_particle` 및 폐기된 순위 3 설명을 제거한다.
  3. §11.1 산출물 표를 실제 최소 구현 요소(두 grammar, 검색 행위 술어 집합, 인용/부정 전처리, DOCUMENT 신호)로 맞춘다.
  4. 비규범 §7.2-L 내부의 옛 수치·fixture는 감사 기록으로 남겨도 되지만, 규범 절이나 구현·테스트 절에서 그 하위 절을 참조하지 않도록 한다.

#### M2. Requirement의 Unicode 경계 양성 예시가 같은 Requirement의 command grammar와 충돌한다

- 위치: `Requirement.md` M3-REQ-004의 두 grammar 및 일반 응답 술어 제외 문단, Unicode 경계 문단
- 근거: Requirement는 `웹검색으로 알려줘`처럼 `SOURCE_PARTICLE + 알려`만 있는 표현을 NONE/LLM으로 명시한다. 그런데 바로 뒤 Unicode 경계 설명은 `질문:웹검색으로 최신 환율 알려줘`를 “진짜 검색 명령”이자 명시 WEB 요청으로 규정한다. 콜론은 왼쪽 경계만 바꿀 뿐, 마지막 일반 응답 술어를 검색 행위 술어로 바꾸지 않는다. `Design.md` 규범 §7.2도 콜론 경계 양성으로 `질문:웹에서 검색해줘`를 사용하고 `웹검색으로 최신 환율 알려줘`는 NONE으로 둔다.
- 영향: 요구사항만 읽어도 동일 문형의 판정이 WEB과 NONE으로 갈려 구현 및 인수 테스트 기준이 불명확하다.
- 수정 요구: Unicode 경계 양성 예시를 규범 §7.2와 동일한 `질문:웹에서 검색해줘`로 교체한다. `(구글링해서 알려줘)`는 직접 검색 명령 grammar에 해당하므로 유지할 수 있다.

## 3. 독립 검증 결과

| 검증 항목 | 결과 |
|---|---|
| `golden.jsonl` 행 수 | 76 |
| SHA-256 | `61b768acd8d33522ef76e3baadd4bf19b44cc25daa79ad7ea255fb0a09d1017a` |
| 규범 WEB exact set | 8건, 모두 `expected_route=web_search`, 오탐 0 |
| 규범 DOCUMENT exact set | 12건, 모두 `expected_route=document_qa`, 오탐 0 |
| NONE complement | 56건 |
| 세 집합의 교집합/누락 | 없음, 76건 완전 분할 |
| Iteration 6 관형절 우회 | 새 규범에서 제거됨 |

WEB exact set은 `ws-000`, `ws-002`, `ws-005`, `ws-007`, `ws-009`, `rr-ws-python-version-001`, `rr-ws-bitcoin-price-001`, `rr-ws-samsung-stock-001`이다. DOCUMENT exact set은 `bd-000`, `ua-000`~`ua-006`, `dq-agent-arch-001`, `dq-agent-vs-model-001`, `dq-kb-price-outlook-001`, `dq-kb-gangnam-001`이며 나머지 56건은 NONE이다. 데이터셋 원문과 대조한 결과 두 결정론 집합의 expected route 오탐은 없다.

## 4. 구현 가능성 평가

새 핵심 알고리즘은 구현 가능하다. 두 WEB grammar를 각각 하나의 정규식/판정 함수로 제한하고, 검색 행위 술어 집합 하나와 인용·부정 전처리만 두는 구조는 독립 단위 테스트가 가능하다. DOCUMENT 우선 신호, `_decide_tool()`의 LLM 이전 판정, NONE 경로의 기존 fallback 보존도 기존 설계의 seam을 그대로 활용할 수 있다.

남은 위험은 알고리즘 난도가 아니라 문서의 실행 지시 불일치다. M1과 M2를 정리하면 구현 복잡도 gate와 precision-first 목표를 함께 만족할 가능성이 높다. 다음 리뷰에서는 문서 전체 검색으로 `has_particle`, `TOPIC_HEAD`, `REQUEST_TAIL`, `WEB 10 / DOCUMENT 12 / NONE 54`가 규범 구현·테스트 지시에서 제거됐는지 확인해야 한다. 비규범 감사 기록 안의 등장은 허용하되, 그 기록을 참조하는 현재형 산출물·테스트 계약은 없어야 한다.

## 5. 결론

Cycle 1은 이전 여섯 차례의 근본 문제를 더 많은 예외로 봉합하지 않고 command grammar 자체를 좁혔다는 점에서 올바른 재설계다. 특히 `SOURCE_PARTICLE` 관형절 bypass는 해소됐고 골든 8/12/56 집합도 데이터셋과 정확히 일치한다.

다만 두 MAJOR는 구현 전에 반드시 고쳐야 하는 명세 충돌이다. 수정 범위는 새 알고리즘 변경이 아니라 Requirement 예시 하나와 Design의 후속 구현·테스트 참조 정합화이므로, 동일 단순화안을 유지한 채 Cycle 2에서 신속히 재검토할 수 있다.
