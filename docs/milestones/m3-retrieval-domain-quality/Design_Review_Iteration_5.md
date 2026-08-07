# M3 Retrieval & Domain Quality 상세 설계 조건부 연장 독립 Gate 리뷰 — Iteration 5

- 검토일: 2026-08-06
- 대상: 최신 `Requirement.md`, `Plan.md`, `Design.md`, `Design_Review_Iteration_4.md`, `Stop_Report.md`, 개정 `m3_orchestration_guide.md`, 실제 `evaluation/datasets/golden.jsonl`, 승인 M2 baseline
- 결론: **STOP**
- 점수: **9.5 / 10**
- 발견사항: **CRITICAL 0, MAJOR 1, MINOR 0, TRIVIAL 0**

## 1. Gate 판정

Iteration 4의 M2(조사 결합 계약 불일치)는 Requirement와 Design 모두 조사 결합을 `REQUEST_TAIL` 및 거리 조건의 면제가 아닌 보조 구문 근거로 명시해 해소됐다. m1도 R18은 거리 초과, R19~R20은 `REQUEST_TAIL` 미충족이라고 정확히 분리돼 해소됐다. M1의 지정 음성 4개와 양성 2개 또한 새 규칙 및 실제 classifier 행렬 R21~R29에서 기대대로 고정됐다.

그러나 `WEB_FUSED`의 정밀도 경계는 완전히 닫히지 않았다. `WEB_FUSED_TOPIC_GAP=2`는 m+2까지만 주제어를 억제하고 m+3부터 의미와 무관하게 WEB을 허용하므로, 주제어 앞에 두 어절만 끼우면 같은 검색 기술·기능·API 질문이 다시 결정론적 WEB이 된다. 더구나 왼쪽 경계를 문장 시작 또는 공백으로만 정의해 문장부호 직후의 실제 강한 검색 명령은 반대로 놓친다. 이는 Iteration 4 M1의 동일 근본 문제인 “융합형 표면 문법이 명령/주제 의미를 충분히 구분하지 못함”이 연속 iteration에서 재발한 것이다.

Gate 기준 CRITICAL/MAJOR 0 및 9.7 이상을 충족하지 못하며, 조건부 연장 중 동일 근본 문제가 2회 연속 재발하면 즉시 중단한다는 `m3_orchestration_guide.md` 규칙도 발동한다. 따라서 **Iteration 6을 허용하지 않고 STOP**한다.

## 2. 발견사항

### MAJOR

#### M1. `WEB_FUSED`의 위치 기반 양쪽 경계가 m+3 주제 우회와 문장부호 뒤 강한 명령 과소탐지를 동시에 만든다

- severity: **MAJOR**
- 위치: `Requirement.md` M3-REQ-004(147행), `Design.md` §7.2.1 `WEB_FUSED`/`WEB_FUSED_TOPIC_GAP`(1088, 1103~1106행), §7.2.3 순위 1(1118행), §7.2.4 경계 fixture(1157, 1191~1194행), §7.4 R21~R24/R30(1310~1319행), §12.1 테스트 계약(1642, 1654행)
- 근거: m+2 음성 `web search API 구조 알려줘`는 억제되지만, 같은 의미의 `웹검색 관련 API 구조 알려줘`는 `WEB_FUSED`의 m=0 뒤 `구조`가 m+3이어서 억제되지 않고 마지막 `알려줘`가 `REQUEST_TAIL=True`이므로 순위 1 WEB이 된다. `웹검색 관련 핵심 기능 알려줘`, `구글링 사용 관련 기술 알려줘`도 같은 우회다. 반면 m+3 양성 R30 `웹검색으로 이번 학기 수업방식 알려줘`는 검색 대상의 내용에 우연히 `방식`이 포함된 실제 검색 명령이어서, “m+3은 항상 허용”의 근거가 아니라 주제 head의 의미 역할을 구별해야 함을 보여준다. 또한 왼쪽 경계를 시작/공백으로만 한정하면 `질문:웹검색으로 최신 환율 알려줘` 또는 `(구글링해서 알려줘)`의 융합형은 매치되지 않는다. 전자는 `웹검색으로`가 `CHANNEL` whole-token도 아니므로 NONE이고, 후자도 괄호 때문에 융합형과 채널 양쪽을 놓쳐 강한 명령 보존 계약을 위반한다.
- 영향: Iteration 4에서 막으려던 검색 기술·기능·API 주제 오탐은 주제어를 m+3으로 한 칸 미는 표현만으로 재현되고 LLM 교정 기회 없이 WEB override가 된다. 동시에 자연스러운 레이블·괄호 문장부호 뒤의 강한 검색 명령은 결정론적 보호를 잃어 web recall이 모델 가용성에 다시 의존한다. 지정 4음성·2양성과 현재 골든 76건은 이 두 경계를 포함하지 않아 일반화 증거가 아니다.
- 수정안: 고정 거리만으로 주제/검색 대상을 구분하지 말고, `WEB_FUSED` 뒤 구간에서 `TOPIC_HEAD`가 검색 행위 자체의 head인지 검색 대상 명사 내부인지 판별하는 제한된 문법을 정의하라. 최소 paired fixture로 `web search API 구조 알려줘`(m+2, NONE), `웹검색 관련 API 구조 알려줘`(m+3, NONE), `웹검색으로 이번 학기 수업방식 알려줘`(m+3, WEB)를 함께 고정해야 한다. 왼쪽 경계는 Unicode 문자·숫자 내부 임베딩만 금지하고 공백뿐 아니라 허용 문장부호 경계를 인정하며, `freewebsearch`는 NONE, `질문:웹검색으로 최신 환율 알려줘`와 `(구글링해서 알려줘)`는 WEB인 회귀 fixture를 추가하라. 다만 같은 근본 문제가 두 번째 연속 iteration에 재발했으므로 이 수정은 M3 Iteration 6 승인 사유가 아니라 향후 별도 재개 조건으로 남긴다.

## 3. 필수 count 및 기대값 검증

| 검증 대상 | 독립 확인 | 판정 |
|---|---:|---|
| 골든 dataset | 총 76; category 51/15/3/7; expected route document 61/web 15 | 일치 |
| 승인 M2 routing | correct 59/76; mismatch 17; document TP 44/61; web TP 15/15; 오류 17건 전부 document→web | 일치 |
| §7.2 dry-run 계약 | WEB 10 / DOCUMENT 12 / NONE 54; 세 ID 집합 exact equality; WEB·DOCUMENT 오탐 0 | count·합계·기대 집합 일치 |
| 양성 fixture | 11개 입력, 모두 WEB | 일치 |
| 부정·충돌·단독 언급 fixture | 표의 입력 16개(맨몸 요청 3개를 개별 입력으로 계산) | 일치 |
| 채널 주제어 방지 fixture | 13개 입력(NONE 12, DOCUMENT 1) | 일치 |
| 골든과 중복 제외 fixture 총계 | 11 + 16 + 13 - `bd-000` 중복 1 = **39개** | 일치 |
| 실제 classifier 행렬 | R1~R30 = **30행** | count·표 기대값 일치 |

골든 exact set은 Design §7.2.4와 실제 dataset을 대조했다. WEB 10은 `ws-000`, `ws-002`, `ws-003`, `ws-005`, `ws-007`, `ws-008`, `ws-009`, `rr-ws-python-version-001`, `rr-ws-bitcoin-price-001`, `rr-ws-samsung-stock-001`; DOCUMENT 12는 `bd-000`, `ua-000`~`ua-006`, `dq-agent-arch-001`, `dq-agent-vs-model-001`, `dq-kb-price-outlook-001`, `dq-kb-gangnam-001`; NONE은 나머지 54건이다.

## 4. 지정 경계와 이전 finding 상태

| 검증 항목 | 결과 |
|---|---|
| 지정 음성 4개 | `웹검색 방법 알려줘`, `웹 검색 기술을 보여줘`, `구글링 기능 알려줘`, `web search API 구조 알려줘` 모두 NONE으로 고정됨 |
| 지정 양성 2개 | `웹검색으로 최신 환율 알려줘`, `구글링해서 알려줘` 모두 WEB으로 고정됨 |
| m+2 경계 | `web search API 구조 알려줘`가 NONE으로 억제됨 |
| m+3 경계 | R30 양성은 WEB이나, `웹검색 관련 API 구조 알려줘`도 WEB이 되는 새 우회가 존재하여 불합격 |
| 왼쪽 경계 | `freewebsearch` 임베딩은 차단하지만 문장부호 뒤 강한 명령까지 차단하여 불합격 |
| 조사 결합 | Requirement 145행과 Design 1086·1101·1120·1127행 모두 `REQUEST_TAIL` + 거리 조건에 종속됨; Iteration 4 M2 해소 |
| R18/R19~R20 설명 | R18=거리 초과, R19~R20=`REQUEST_TAIL` 거짓으로 §7.4·§12.1 모두 일치; Iteration 4 m1 해소 |

R25~R27은 조사 결합이 독립 충분조건이 아님을 실제 classifier 경로에서 검증하고, R28~R29는 지정 강한 명령을 보존한다. R30은 m+3에서 억제 창이 넓지 않음을 확인하지만, 같은 위치의 주제 질문 음성 대조군이 없어 위치와 의미 역할을 혼동한다. 따라서 39 fixture와 R1~R30의 내부 기대값은 일관되지만 필요한 결정 경계를 완전하게 규정하지 못한다.

## 5. 추적성·구현 가능성

Requirement/Plan/Design의 M3-REQ-001~010 및 NFR 추적표, 69/76·54/61·15/15 gate, Phase 순서, rollback, 제품/평가 단일 정책 경계는 전반적으로 구현 가능하다. 조사 결합 계약과 R18 설명은 상하위 문서 및 테스트 설명까지 동기화됐다. 신규 라우팅 모듈·테스트·CLI 명령의 파일 경로와 책임도 구체적이다.

다만 M1은 M3-REQ-004의 핵심인 검색 기술·기능·API 주제 억제와 강한 검색 명령 보존을 동시에 깨므로 단순 fixture 누락 이상의 의미 계약 결함이다. 현재 거리 휴리스틱을 그대로 구현하면 결정론적으로 재현되는 오분류가 생기므로 구현 착수 가능 판정을 내릴 수 없다.

## 6. 조건부 연장 추세와 최종 결론

| Iteration | 점수 | CRITICAL | MAJOR | MINOR | 추세 |
|---:|---:|---:|---:|---:|---|
| 4 | 9.4 | 0 | 2 | 1 | 조건부 연장 진입 가능 |
| 5 | 9.5 | 0 | 1 | 0 | 수치상 개선, 그러나 Iteration 4 M1의 동일 근본 문제 연속 재발 |

- **Gate: STOP**
- **Score: 9.5 / 10**
- **Counts: CRITICAL 0, MAJOR 1, MINOR 0, TRIVIAL 0**
- 이전 MAJOR 2/MINOR 1 중: M2와 m1은 완전 해소, M1은 지정 사례만 해소됐고 근본 문제는 m+3/왼쪽 경계에서 재발
- 새 MAJOR 증가 여부: 총 MAJOR는 2→1로 감소했으나, 잔여 M1은 동일 근본 문제의 새 경계 재현
- 점수 개선 여부: 9.4→9.5로 개선됐지만 9.7 미만
- Iteration 6 제안: **불허**. 조건부 연장 즉시 중단 조건인 동일 근본 문제 2회 연속 재발에 해당한다.

원문 Requirement/Plan/Design, 제품 코드, dataset, baseline은 수정하지 않았고 이 리뷰 문서만 작성했다. commit/push는 수행하지 않았다.
