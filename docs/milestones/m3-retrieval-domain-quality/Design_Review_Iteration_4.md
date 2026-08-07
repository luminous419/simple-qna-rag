# M3 Retrieval & Domain Quality 상세 설계 최종 독립 Gate 리뷰 — Iteration 4

- 검토일: 2026-08-06
- 대상: 최신 `Requirement.md`, `Plan.md`, `Design.md`, Iteration 1~3 리뷰, 현재 제품·평가 코드, `evaluation/datasets/golden.jsonl`, 승인 M2 baseline
- 결론: **STOP**
- 점수: **9.4 / 10**
- 발견사항: **CRITICAL 0, MAJOR 2, MINOR 1, TRIVIAL 0**

## 1. 최종 Gate 판정

Iteration 3의 MAJOR였던 `CHANNEL` 전역 substring 오탐은 `CHANNEL` 어절 전체 일치, `last-i <= 2`, `REQUEST_TAIL`의 AND 조건으로 직접 보완됐다. 이에 따라 필수 반례 `websocket 설정을 알려줘`, `Google AI 에이전트 구조를 보여줘`, `인터넷 회사 찾아가는 길`, `온라인 게임 확인 방법`, `웹 개발 방법 알려줘`는 모두 결정론적 신호가 `NONE`이며 WEB override가 아니다. 실제 classifier를 쓰는 R13, R14, R18~R20도 이 계약을 고정한다.

그러나 같은 정밀도 계약을 `WEB_FUSED`가 우회한다. 또한 Requirement가 독립 조건으로 쓴 출처·수단 조사 결합과 Design의 실제 판정식이 일치하지 않아 명시적 웹 사용 요청을 과소 탐지한다. 최대 설계 iteration의 최종 Gate 기준은 CRITICAL/MAJOR 0 및 9.7 이상인데 MAJOR 2건이 남았으므로 추가 설계 iteration을 승인하지 않고 **STOP**한다. 구현에 착수하기 전 아래 두 계약을 별도 승인·보완해야 한다.

## 2. 발견사항

### MAJOR

#### M1. `WEB_FUSED`가 whole-token·거리·`REQUEST_TAIL`을 모두 우회해 동일한 주제어 오탐을 재생산한다

- severity: **MAJOR**
- 위치: `Design.md` §7.2.1 `WEB_FUSED`(1088행), §7.2.3 순위 1~2(1112~1113행), §7.2.4 fixture(1138~1182행), §7.4 R1~R20(1276~1300행), `Requirement.md` M3-REQ-004(145행)
- 근거: `WEB_FUSED`는 `(웹|인터넷|온라인|구글|web)\s?(검색|서치|search)` 또는 `구글링`의 부분 문자열이면 즉시 순위 2 WEB이다. 순위 1의 주제어 예외는 `DOCUMENT` 증거까지 동시에 있어야 한다. 따라서 `웹검색 방법 알려줘`, `웹 검색 기술을 보여줘`, `구글링 기능 알려줘`, `web search API 구조 알려줘`는 외부 검색 실행 요청이 아니라 검색 기술·기능·API에 관한 주제 질문인데도 거리와 `REQUEST_TAIL`, 채널 whole-token 판정을 모두 건너뛰어 WEB이 된다. 이는 Requirement가 “기술·회사·서비스명의 일부 또는 역사·개념 등 주제 서술 대상”을 명시 웹 사용 요청에서 제외한 계약과 모순된다. 현재 31개 fixture에는 이 경로의 음성 반례가 없다.
- 영향: Iteration 3에서 제거하려던 document→web 과다 라우팅이 융합 토큰 표현만 바꾸면 다시 발생하며, 순위 2는 LLM 교정 기회도 없앤다. 골든 76건의 WEB 오탐 0은 이 주제 표현을 포함하지 않아 일반화 증거가 아니다.
- 수정안: `WEB_FUSED`를 “강한 검색 **명령**”으로 재정의해 명령형 종결 또는 별도 요청 근거를 요구하고, 주제 head가 뒤따르는 경우 DOCUMENT가 없어도 NONE으로 보내라. 최소한 위 네 문장을 음성 fixture와 실제 classifier 행렬에 추가하고, `웹검색으로 최신 환율 알려줘`·`구글링해서 알려줘` 양성은 그대로 WEB인지 경계 테스트하라.

#### M2. 출처·수단 조사 규칙이 Requirement에서는 독립 충분조건이지만 Design에서는 끝 요청+거리의 필수조건에 종속된다

- severity: **MAJOR**
- 위치: `Requirement.md` M3-REQ-004(145행), `Design.md` §7.2.1 `CHANNEL_SOURCE_PARTICLE` 및 `REQUEST_TAIL`/gap(1085~1101행), §7.2.3 순위 4(1115행), §7.2.4 양성 fixture(1138~1150행)
- 근거: Requirement는 채널 지시어가 “(a) 출처·수단 조사와 직접 결합하거나, (b) 요청 동사가 문장 끝에서 국소적으로 결합”할 때 명시 요청이라고 OR로 규정한다. 반면 Design은 조사 결합 사례에도 “별도의 거리 면제 규칙은 두지 않는다”고 명시하고, 모든 순위 4에 `REQUEST_TAIL`과 `last-i <= 2`를 동시에 요구한다. 따라서 `인터넷에서 최신 소식 부탁해`, `웹에서 확인할 수 있을까`, `구글로 좀 찾아줄 수 있어?`는 출처·수단을 명시한 자연스러운 요청이지만 마지막 어절에 현재 `REQUEST` 어간이 없거나 채널과 멀어 NONE이 된다. 조사 결합 양성 fixture도 모두 우연히 거리가 1이고 마지막 어절이 `REQUEST`라서 이 불일치를 검출하지 못한다.
- 영향: 상위 요구사항과 구현 판정식의 추적성이 깨지고, 명시적 웹 요청의 recall이 표현 방식에 과도하게 의존한다. 특히 조사 결합을 독립 근거로 승인한 이해와 실제 제품 동작이 달라진다.
- 수정안: 둘 중 하나를 명시적으로 선택해 동기화하라. (A) Requirement의 (a)를 실제 독립 충분조건으로 구현하되 단독 명사구 오탐 방지를 위한 문장성/요청성 조건을 별도로 정의하거나, (B) 안전한 정밀도를 위해 조사 결합도 `REQUEST_TAIL`+gap을 요구한다고 Requirement를 고친다. 선택한 계약에 조사 결합이 있으나 tail/gap이 없는 양성·음성 경계 fixture를 추가해야 한다.

### MINOR

#### m1. 테스트 요약이 R18의 실패 원인을 `REQUEST_TAIL` 미충족으로 잘못 기술한다

- severity: **MINOR**
- 위치: `Design.md` §12.1 `test_agent_routing_policy.py` 설명(1630행), §7.2.1 설명(1101행), R18(1293행)
- 근거: `웹 개발 방법 알려줘`의 마지막 어절 `알려줘`는 `REQUEST_TAIL=True`이고, NONE인 이유는 `j-i=3 > 2`이다. §7.2.1과 R18은 이를 올바르게 쓰지만 §12.1은 R18~R20 전부를 “`REQUEST_TAIL` 미충족”으로 묶는다.
- 영향: 구현자가 R18 assert의 원인을 잘못 고정하거나, gap 경계 테스트가 누락될 수 있다.
- 수정안: §12.1을 “R18은 gap 초과, R19~R20은 `REQUEST_TAIL` 거짓”으로 정정한다.

## 3. count 및 기대값 검증

| 검증 대상 | 독립 확인 | 판정 |
|---|---:|---|
| 골든 dataset | 총 76; category 51/15/3/7; expected route document 61/web 15 | 일치 |
| 승인 M2 routing | correct 59/76; document TP 44/61; web TP 15/15; document→web 오류 17 | 일치 |
| §7.2 dry-run 계약 | WEB 10 / DOCUMENT 12 / NONE 54, 각 ID 집합 exact equality와 WEB·DOCUMENT 오탐 0 | count와 합계 일치 |
| 양성 fixture | 표의 입력 9개, 모두 WEB | 일치 |
| 음성/충돌 fixture | 첫 표 13개 입력(맨몸 요청 3개를 개별 입력으로 계산) + 주제어 방지 표 9개 = 22개 | 일치 |
| fixture 총계 | 양성 9 + 음성/충돌 22 = **31** | 일치 |
| 실제 classifier 행렬 | R1~R20 = **20행** | count 일치 |

필수 5문장의 세부 판정도 일치한다. `websocket...`은 whole-token 실패, `Google AI...`와 `웹 개발...`은 gap 초과, `인터넷 회사...`와 `온라인 게임...`은 `REQUEST_TAIL=False`다. R19는 신호가 NONE인 뒤 stub LLM이 web을 선택하므로 최종 반환은 web이지만, 이는 결정론적 WEB 오탐이 아니라 3순위 모델 판단을 보존한다는 의도된 테스트다.

## 4. 이전 finding 및 추적성

| 이전 finding | 최종 상태 |
|---|---|
| Iteration 1의 metric 분모/게이트, warm-up, null-safe MMR, evaluator fail-closed, baseline·명령 문제 | 해소 유지 |
| Iteration 2의 WEB 양성 recall 부족 | 대표 9개 표현에 한해 해소; M2의 Requirement 불일치로 일반 계약은 미해소 |
| Iteration 2의 tracked-only Markdown 검사 | 해소 유지 |
| Iteration 3 M1의 `websocket`/Google AI 등 전역 substring 오탐 | 지정 반례는 해소; `WEB_FUSED` 우회로 인해 근본 정밀도 계약은 미해소 |
| Iteration 3 t1의 Requirement/Design 충돌 문구 | 우선순위·주제 제외 문구는 Requirement에 반영되어 해소 |

Requirement/Plan/Design의 전체 추적표, 69/76·54/61·15/15 count gate, Phase 순서, rollback 및 제품/평가 단일 정책 경계는 구현 가능한 수준이다. 다만 M1·M2는 핵심 M3-REQ-004의 의미와 직접 충돌하므로 문서 추적표에 행이 존재한다는 사실만으로 충족으로 볼 수 없다.

## 5. 테스트 명령 검증

설계의 정적·Phase 3 명령은 현재 repository layout 및 module entry point와 맞는다. 구현 후 최소 검증 명령은 다음과 같다.

```bash
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
pytest -q tests/unit/test_query_router.py tests/unit/test_routing_signals.py \
  tests/integration/test_agent_routing.py tests/integration/test_agent_routing_policy.py \
  tests/integration/test_evaluation_routing.py
python -m evaluation.routing --help
python -m evaluation.routing --dataset evaluation/datasets/golden.jsonl --mode live \
  --runs 3 --output evaluation/reports/m3/m3-p3-routing
pytest -q
npm test
python scripts/check_markdown_links.py
git diff --check
```

신규 파일·모듈은 아직 설계 산출물이므로 현재 실행 가능하다는 뜻이 아니라, Phase 구현 후 정확히 생성돼야 하는 명령 계약이다. live 명령은 로컬 모델/서비스가 필요한 opt-in 검증이며 CI의 모델 없는 fixture 행렬과 구분돼 있다.

## 6. 최종 결론

- **Gate: STOP**
- **Score: 9.4 / 10**
- **Counts: CRITICAL 0, MAJOR 2, MINOR 1, TRIVIAL 0**
- 통과 기준 대비: CRITICAL/MAJOR 0 불충족, MINOR 최소화 미완, 9.7 미만
- 잔여 조치: 구현 착수 전 M1의 융합형 주제 우회와 M2의 조사 충분조건을 별도 설계 보완·승인하고 m1 문구를 정정한다. 최대 iteration의 최종 리뷰이므로 이 문서는 추가 iteration을 제안하지 않으며 현재 설계 Gate를 닫는다.

원문 Requirement/Plan/Design, 제품 코드, dataset, baseline은 수정하지 않았고 이 리뷰 파일만 작성했다. commit/push는 수행하지 않았다.
