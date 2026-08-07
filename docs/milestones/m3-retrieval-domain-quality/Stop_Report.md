# M3 작업 중단 보고서

- 기록일: 2026-08-06
- 상태: **설계 Gate 미통과로 중단**
- 중단 단계: 상세 설계 최종 독립 리뷰(Iteration 4)
- 최종 판정: **STOP, 9.4/10**
- 발견사항: **CRITICAL 0, MAJOR 2, MINOR 1, TRIVIAL 0**

> 2026-08-06 재개: iteration 규칙을 기본 4회와 조건부 최대 2회로 개정했다. 현재 결과는 CRITICAL 0, 9.4/10, MAJOR 2이며 개선 추세와 해결 가능한 범위를 갖춰 조건부 연장 기준을 충족하므로 이 중단 상태를 해제하고 Iteration 5부터 재개한다. 이 문서는 최초 중단 판단의 기록으로 보존한다.

> 2026-08-07 최종 중단: 사용자 지시에 따라 Iteration 6까지 구조적 재설계를 수행했으나 최종 독립 리뷰가 **STOP 9.5/10, CRITICAL 0, MAJOR 1, MINOR 0**으로 판정됐다. `SOURCE_PARTICLE`이 있는 표현이 full-clause 주제 스캔을 우회해 관형절 주제 질문을 결정론적 WEB으로 강제할 수 있다. 총 6회 상한에 도달했으므로 설계 Gate를 닫고 구현에는 착수하지 않는다.

> 2026-08-07 재개: 사용자가 결정론적 라우팅 설계를 단순화하는 별도 재설계 사이클을 승인했다. 기존 Iteration 1~6과 중단 판단은 보존하며, 새 `Routing Simplification` 사이클을 최대 6회로 시작한다. 이 사이클은 명백한 검색 실행 명령만 WEB override하고 모호한 표현은 LLM으로 넘기는 precision-first 원칙을 따른다.

## 중단 사유

`m3_orchestration_guide.md`는 문서와 코드의 품질 Gate를 CRITICAL/MAJOR 0, MINOR 최소화, 9.7/10 이상으로 정하고 iteration을 최대 4회로 제한한다. 상세 설계는 네 번째 최종 리뷰에서도 MAJOR 2건과 MINOR 1건이 남아 기준을 충족하지 못했다. 따라서 제품 코드 구현, live 평가, commit/push/PR/merge에는 착수하지 않고 M3 작업을 중단한다.

## 완료된 작업

- M3 요구사항, 개발 계획, 상세 설계 작성
- M2 승인 baseline과 golden dataset을 근거로 정량 Gate 정정
- 상세 설계 독립 리뷰 4회와 Claude Code 설계 개선 3회
- evaluator v2, MMR, routing, Intent 실험, 조건부 BM25, 테스트·rollback·artifact 계약 구체화
- Markdown 링크 검사 범위, 동일 프로세스 warm-up, null-safe MMR fallback 등 이전 CRITICAL/MAJOR 해소
- Claude Code 작업은 최종 개선을 포함해 Sonnet 5로 수행했으며 Opus는 사용하지 않음

## 미해결 항목

### MAJOR 1 — `WEB_FUSED`가 정밀도 경계를 우회

`WEB_FUSED`는 whole-token, 거리, 요청 종결 조건을 건너뛰므로 `웹검색 방법 알려줘`, `웹 검색 기술을 보여줘`, `구글링 기능 알려줘`, `web search API 구조 알려줘` 같은 주제 질문을 명시적인 외부 검색 요청으로 오판할 수 있다. 강한 검색 명령과 검색 기술에 관한 질문을 분리하는 계약과 회귀 fixture가 필요하다.

### MAJOR 2 — 출처·수단 조사 규칙의 Requirement/Design 불일치

Requirement는 출처·수단 조사 결합을 독립 충분조건으로 기술하지만 Design은 모든 경우에 요청 종결과 거리 제한을 요구한다. 조사 결합을 독립 충분조건으로 구현할지, 정밀도를 위해 추가 조건을 요구하도록 Requirement를 수정할지 결정하고 양성·음성 경계 fixture를 추가해야 한다.

### MINOR 1 — R18 테스트 설명 오류

`웹 개발 방법 알려줘`가 NONE인 이유는 `REQUEST_TAIL` 실패가 아니라 허용 거리 초과다. 테스트 설명을 실제 판정 원인과 일치시켜야 한다.

## 재개 조건

M3를 재개하려면 별도 승인 아래 새로운 설계 보완 사이클을 열고 다음 조건을 모두 만족해야 한다.

1. `WEB_FUSED`의 명령/주제 경계를 명시하고 양성·음성 fixture로 고정한다.
2. 출처·수단 조사 규칙을 Requirement와 Design에서 하나의 계약으로 통일한다.
3. R18 테스트 설명을 정정한다.
4. 독립 리뷰에서 CRITICAL/MAJOR 0, MINOR 최소화, 9.7/10 이상을 확인한다.
5. 설계 Gate 통과 전에는 제품 코드 구현이나 live 평가를 시작하지 않는다.

## 관련 문서

- `Requirement.md`
- `Plan.md`
- `Design.md`
- `Design_Review.md`
- `Design_Review_Iteration_2.md`
- `Design_Review_Iteration_3.md`
- `Design_Review_Iteration_4.md`
