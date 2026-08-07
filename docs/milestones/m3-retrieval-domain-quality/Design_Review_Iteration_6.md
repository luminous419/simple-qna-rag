# M3 Retrieval & Domain Quality 상세 설계 최종 독립 Gate 리뷰 — Iteration 6

- 검토일: 2026-08-07
- 대상: 최신 `Requirement.md`, `Plan.md`, `Design.md`, Iteration 1~5 리뷰와 `Stop_Report.md`, 실제 `evaluation/datasets/golden.jsonl`, 승인 M2 baseline
- 결론: **STOP**
- 점수: **9.5 / 10**
- 발견사항: **CRITICAL 0, MAJOR 1, MINOR 0, TRIVIAL 0**

## 1. 최종 Gate 판정

Iteration 5 M1의 두 직접 반례는 해소됐다. `CHANNEL_REQUEST_MAX_WORD_GAP`과 `WEB_FUSED_TOPIC_GAP`은 제거됐고, bare `CHANNEL`과 bare `WEB_FUSED` 모두 마지막 명령형 서술어 앞까지 full-clause `TOPIC_HEAD` scan을 거친다. 따라서 `웹검색 관련 API 구조 알려줘` 계열의 m+3 우회는 더 이상 존재하지 않는다. `WEB_FUSED`와 `CHANNEL`은 동일 command-intent 함수를 공유하며, 차이는 DOCUMENT와 충돌할 때의 명시 강도 순위뿐이다. Unicode `(?<!\w)` 왼쪽 경계도 문장부호·괄호 뒤를 허용하면서 한글·ASCII 영숫자·underscore 직후의 내부 임베딩을 거부한다.

그러나 새 구조는 `SOURCE_PARTICLE`가 붙으면 절의 문법적 역할을 보지 않고 full-clause scan을 전부 생략한다. 이 때문에 같은 검색 기술·기능 주제 질문에 `에서`를 붙이는 것만으로 결정론적 WEB override를 다시 만들 수 있다. 고정 거리 우회를 닫았지만, 명령/주제 의미를 표면 형태 하나로 확정하는 동일 근본 문제가 `SOURCE_PARTICLE` 우회로 이동했다. Gate 기준 CRITICAL/MAJOR 0과 9.7 이상을 충족하지 못하므로 최종 판정은 **STOP**이다.

## 2. 발견사항

### MAJOR

#### M1. `SOURCE_PARTICLE`의 무조건 scan 생략이 관형절 속 검색 주제를 즉시 WEB으로 승격한다

- severity: **MAJOR**
- 위치: `Requirement.md` M3-REQ-004 145~156행, 특히 148·156행; `Design.md` §7.2.1 `SOURCE_PARTICLE` 1088행, 주제절 억제 규칙 1102~1112행, §7.2.3 1125~1133행, §7.2.4 fixture 1143~1223행, §12.1 routing test 계약 1674행 및 추적표 1904행
- 근거: 음성 `웹검색 관련 API 구조 알려줘`는 bare match라 전체 절의 `구조`를 찾아 NONE이다. 하지만 최소 변형 `웹검색에서 사용하는 API 구조 알려줘`는 `WEB_FUSED+에서`로 `has_particle=True`가 되어 `사용하는 API 구조`를 전혀 스캔하지 않고 WEB이 된다. 이 문장의 `에서`는 요청된 정보의 출처를 지시하는 것이 아니라 관형절 `웹검색에서 사용하는` 안에서 API의 사용 영역을 표시하며, head는 여전히 `API 구조`다. `구글에서 사용하는 검색 기술 알려줘`도 `CHANNEL+에서`만으로 같은 오탐을 만든다. 반대로 양성 `웹검색으로 이번 학기 수업방식 알려줘`는 `으로`가 실제 수단이고 WEB이어야 한다. 현재 55 fixture와 R1~R36에는 `SOURCE_PARTICLE + 관형형 서술어 + TOPIC_HEAD + REQUEST_TAIL` 음성 대조군이 없다.
- 영향: 사용자가 조사 하나와 관형형 `사용하는/쓰는/제공하는`을 넣으면 검색 기술·기능·API 주제 질문이 LLM 교정 기회 없이 WEB으로 강제된다. 이는 M3-REQ-004의 “채널 언급만으로 명시가 성립하지 않음”, “검색 자체를 설명·서술 대상으로 삼는 주제 질문은 3순위로 넘김” 계약을 깨고, Iteration 4~5의 동일 근본 정밀도 결함을 새 표면 형태로 재현한다.
- 수정안: `has_particle`를 곧바로 command로 확정하지 말고, 조사 뒤가 목적어 절인지 관형절인지 제한된 형태로 구분하라. 최소한 `SOURCE_PARTICLE` 뒤 구간에 관형형 cue(`사용하는`, `쓰는`, `제공하는` 등)가 있고 그 관형절의 head가 `TOPIC_HEAD`이면 억제하되, 실제 수단 양성은 유지해야 한다. 다음 최소쌍을 동일 unit/property 표와 실제 classifier 행렬에 추가하라: `웹검색으로 이번 학기 수업방식 알려줘` → WEB, `웹검색에서 사용하는 API 구조 알려줘` → NONE, `구글에서 최신 환율 알려줘` → WEB, `구글에서 사용하는 검색 기술 알려줘` → NONE. 임의 동사 목록 확대가 아니라 관형형/명령형 역할을 하나의 공통 판정 함수에서 다루어야 한다.

## 3. 독립 재현 결과

| 검증 대상 | 독립 결과 | 판정 |
|---|---:|---|
| 골든 dataset 원문 | 76행; category document 51 / web 15 / boundary 3 / unanswerable 7; expected route document 61 / web 15 | 일치 |
| dataset SHA-256 | `61b768acd8d33522ef76e3baadd4bf19b44cc25daa79ad7ea255fb0a09d1017a` | 승인 기준과 일치 |
| 승인 M2 routing | correct 59/76, mismatch 17; document TP 44/61, web TP 15/15; 17건 전부 document→web | 일치 |
| §7.2 순수 규칙 dry-run | **WEB 10 / DOCUMENT 12 / NONE 54**, 세 ID 집합 exact equality, WEB·DOCUMENT 오탐 0 | 일치 |
| 문서화 fixture | 양성 13 + 부정·충돌·단독 17 + 채널 주제 방지 17 + 왼쪽 경계/property 실행칸 8 = **55** | 선언 기대값 일치; M1 최소쌍은 부재 |
| 실제 classifier 행렬 | R1~R36 = **36행**, 기대 route/LLM 예외·no-tool 계약 일치 | 일치; M1 미포함 |

골든 WEB exact set은 `ws-000`, `ws-002`, `ws-003`, `ws-005`, `ws-007`, `ws-008`, `ws-009`, `rr-ws-python-version-001`, `rr-ws-bitcoin-price-001`, `rr-ws-samsung-stock-001`이다. DOCUMENT exact set은 `bd-000`, `ua-000`~`ua-006`, `dq-agent-arch-001`, `dq-agent-vs-model-001`, `dq-kb-price-outlook-001`, `dq-kb-gangnam-001`이며 나머지 54건은 NONE이다.

Unicode 경계는 독립 정규식 실행에서 문자열 시작·공백·쉼표·콜론·`(`/`[` 뒤를 허용했고, `무료웹검색사이트`, `freewebsearch`, `_웹검색...`, `3웹검색...` 내부 매치를 거부했다. `googleapis`도 검색/서치/search 접미사가 없어 `WEB_FUSED`가 아니며 `CHANNEL` whole-token에도 실패한다.

## 4. 추적성·복잡도·구현 가능성

Requirement/Plan/Design의 M3-REQ-001~010, NFR, 추적표, Phase 3 명령은 76건 3회 평가, `69/76`, `54/61`, 각 run `15/15`, 모델 없는 unit/integration 행렬을 같은 계약으로 가리킨다. 제품과 evaluator가 `classify_explicit_signal()`을 공유하고 rollback flag에서 M2 LLM 경로로 돌아가는 설계도 구현 가능하다. 두 거리 상수 제거와 단일 함수 통합은 이전안보다 복잡도를 실제로 줄였다.

다만 `SOURCE_PARTICLE`를 의미 확정자로 간주해 scan을 생략하는 한 구현은 간단해도 정책은 완전하지 않다. M1은 fixture 한 건 추가만으로 끝나는 누락이 아니라 `has_particle` branch의 판정 계약을 바꿔야 하는 구조 결함이며, 현 설계 그대로의 구현 착수 승인은 불가하다.

## 5. 최종 결론

- **Gate: STOP**
- **Score: 9.5 / 10**
- **Counts: CRITICAL 0, MAJOR 1, MINOR 0, TRIVIAL 0**
- Iteration 5 직접 반례: m+3 우회와 문장부호 왼쪽 경계는 해소
- 잔여 근본 문제: `SOURCE_PARTICLE` 무조건 scan 생략으로 동일 명령/주제 의미 경계가 새 형태로 재발
- 총 6회 상한: 추가 Iteration 제안 없음. 구현 착수 전 M1을 별도 재개 조건으로 처리해야 한다.

원문 Requirement/Plan/Design, 제품 코드, dataset, baseline은 수정하지 않았고 이 리뷰 문서만 작성했다. commit/push는 수행하지 않았다.
