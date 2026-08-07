# M3 Retrieval & Domain Quality 상세 설계 독립 재리뷰 — Iteration 2

- 검토일: 2026-08-06
- 대상: `Requirement.md`, `Plan.md`, `Design.md`, `Design_Review.md`, 현재 제품·평가 코드, `golden.jsonl`, 승인 M2 baseline
- 결론: **REVISE**
- 점수: **9.3 / 10**
- 발견사항: **CRITICAL 0, MAJOR 2, MINOR 0, TRIVIAL 2**

## 1. Gate 재판정

Iteration 1의 CRITICAL 1, MAJOR 4, MINOR 4, TRIVIAL 2에 대한 핵심 수정은 모두 설계에 들어갔다. 특히 routing metric은 현재 evaluator 및 승인 baseline과 연속적인 61/44/54 계약으로 통일됐고, 동일 process warm-up, LLM 전 explicit signal 흐름, `trace=None` MMR 폴백, evaluator v2 fail-closed, RSS 측정, Phase 0 로그, candidate 정규식, loopback 제한, 모듈 축소가 구체화됐다.

그러나 explicit WEB 판별 규칙이 요구사항이 약속하는 명시적 웹 요청 전체를 포괄하지 않고, Markdown link checker의 기본 파일 열거가 Gate 실행 시점의 신규 미추적 Markdown을 검사하지 않는다. 둘 다 요구사항 핵심 계약 또는 필수 회귀 Gate를 실제로 빠져나갈 수 있게 하므로 MAJOR다. 따라서 Gate 조건인 CRITICAL/MAJOR 0과 9.7점 이상을 충족하지 못해 **REVISE**다.

## 2. 발견사항

### MAJOR

#### M1. explicit WEB 신호 규칙이 “명시적 웹 요청”의 정상 표현을 NONE으로 놓친다

- severity: **MAJOR**
- 위치: `Requirement.md` M3-REQ-004, `Design.md` §7.2, §7.4, §12.1 `test_routing_signals.py`
- 근거: 요구사항은 사용자가 웹/인터넷/실시간 검색을 명시하면 LLM 전에 web route를 확정한다고 규정한다. 그러나 설계의 WEB 규칙은 합성어 `웹검색|인터넷검색|웹서치` 또는 채널 토큰과 제한된 검색 동사(`검색`, `찾`, `알아봐`, `확인해`)의 동시 출현만 인정한다. 따라서 `웹 기준으로 답해줘`, `인터넷에서 알려줘`, `온라인 자료를 보여줘`, `구글로 조회해줘`처럼 채널을 명시했지만 허용 동사가 없는 정상 요청은 NONE이 되어 LLM 예외/no-tool 시 기존 fallback에 의존한다. 골든셋 76건의 “오탐 0” 검사는 precision만 보호하고 이러한 false negative를 검출하지 않는다.
- 영향: M3-REQ-004의 가장 강한 사용자 의도가 모델 가용성과 무관해야 한다는 계약이 일부 표현에서 다시 깨진다. §7.4의 12칸 테스트도 classifier를 WEB로 stub하거나 이미 WEB로 분류된 입력만 쓰면 이 결함을 잡지 못한다.
- 수정안: “명시적 채널 + 요청 표현”의 최소 positive grammar를 확정한다. 예를 들어 채널 토큰과 `검색|찾|알아봐|확인|조회|보여|알려|답해` 계열의 결합을 인정하되, 단독 채널 언급의 인용/부정 반례도 함께 둔다. 골든셋 오탐 0 외에 위 양성 paraphrase와 `웹 검색은 하지 말고 문서로 답해` 같은 충돌·부정 입력을 독립 fixture로 추가하고, 실제 classifier → `_decide_tool()` → exception/no-tool까지 연결해 검증한다.

#### M2. link checker의 기본 파일 집합이 신규 미추적 Markdown을 제외해 Phase Gate가 fail-open 된다

- severity: **MAJOR**
- 위치: `Requirement.md` M3-NFR-005, `Plan.md` §3 및 Phase 0, `Design.md` §4.5 “대상 파일”, §12.4
- 근거: 설계는 기본 입력을 `git ls-files -z -- '*.md' '*.markdown'`로 한정한다. 이 명령은 Git 추적 파일만 반환한다. 현재 작업 상태에서도 `docs/milestones/m3-retrieval-domain-quality/` 전체가 미추적(`??`)이므로, 공통 명령 `python scripts/check_markdown_links.py`는 바로 이 마일스톤의 신규 Markdown을 검사하지 않는다. Phase 0 이후 새로 만드는 review/result 문서도 commit 전 Gate에서 같은 방식으로 누락될 수 있다.
- 영향: 깨진 상대 path/anchor를 가진 신규 산출물이 검사 성공으로 보고될 수 있어 M3-NFR-005와 모든 Phase 공통 Gate가 재현 가능한 회귀 방지 장치가 되지 못한다. 이는 link checker 자체의 anchor 구현 정확성과 무관하게 입력 집합 단계에서 발생하는 fail-open이다.
- 수정안: 기본 입력을 tracked와 untracked-nonignored의 합집합으로 만든다. 예: `git ls-files -z --cached --others --exclude-standard -- '*.md' '*.markdown'`를 사용하고 중복 제거·stable sort한다. 또는 repo walk를 기본으로 하고 `.gitignore` 판정을 적용한다. 단위/통합 테스트에 “Git repo 안의 신규 미추적 Markdown에 깨진 링크가 있으면 exit 1”을 추가하고, 실제 현재 worktree에서 신규 M3 문서가 검사 파일 수에 포함되는지 Phase 0 로그에 남긴다.

### TRIVIAL

#### t1. Phase 0의 “코드 변경 없음” 표현이 link checker 구현 산출물과 모순된다

- severity: **TRIVIAL**
- 위치: `Design.md` §13.1 흐름도의 `Phase 0 (기준 고정, 코드 변경 없음)`, `Plan.md` Phase 0 작업 7·산출물
- 근거: Plan과 Design §4.5는 Phase 0에서 `scripts/check_markdown_links.py`와 단위 테스트를 신규 구현하도록 명시한다.
- 영향: 실행 담당자가 Phase 0을 관측 전용으로 해석할 여지가 있지만, 산출물 목록이 명확해 구현 자체를 막지는 않는다.
- 수정안: `코드 변경 없음`을 `제품 코드 변경 없음(평가/회귀 도구만 추가)`으로 바꾼다.

#### t2. 동일 항목이 Design 본문에 중복돼 검토 기준의 단일성을 흐린다

- severity: **TRIVIAL**
- 위치: `Design.md` §4.4 절차의 1번 항목, §6.3 검증표의 V5 항목
- 근거: warm-up 절차 1번 문장과 V5 semantic mapping 검사가 각각 연속해서 두 번 적혀 있다.
- 영향: 의미 변화는 없지만 이후 수정 시 한 사본만 바뀌는 문서 drift 위험이 있다.
- 수정안: 각 중복 행을 하나씩 삭제한다.

## 3. Iteration 1 finding 재검증

| 1차 finding | Iteration 2 판정 | 현재 근거 |
|---|---|---|
| C1 — 61/44/54 metric 모순 | **해소** | Requirement §4.1, Plan Phase 0/3, Design §5.8/§7.5/§14.3이 모두 `expected_route` 분모 61, M2 44, M3 54를 사용한다. `golden.jsonl` 직접 집계도 document route 61/web 15이며 baseline confusion matrix TP는 44다. 기존 evaluator는 `case.expected_route.value`를 사용하므로 연속성도 유지된다. |
| M1 — 별도 process warm-up | **해소** | Requirement M3-NFR-002와 Design §4.4가 evaluator 내부 동일 process·동일 engine, warm-up 표본 폐기 후 전체 42/29건 재측정, 구조화 metadata와 gate `pass=null`을 명시한다. 통합 테스트도 같은 engine과 measured count를 확인한다. |
| M2 — explicit signal이 LLM 뒤 | **핵심 순서 해소, 신규 범위 결함 M1** | Design §7.4는 DOCUMENT 즉시 반환, WEB LLM 정제 후 exception/no-tool/빈 query에서 결정론적 추출, NONE에서 기존 계약 보존을 명시한다. 다만 §7.2 classifier의 positive 범위가 불충분하다. |
| M3 — `trace=None` 폴백 실패 | **해소** | `_bump()`/`_note()` null-safe helper, `_candidate_vectors()` 의사코드, trace 유무 × miss/dimension/non-finite 6칸 행렬, 제품 `query()`와 초기화 강등 테스트가 명시됐다. |
| M4 — Markdown link 검사 부재 | **도구·anchor/path 설계는 해소, 신규 입력 결함 M2** | 표준 라이브러리 스크립트, path/anchor/코드블록/외부 URL/exit code/명령이 구체화됐다. 다만 tracked-only 입력은 신규 문서를 누락한다. |
| m1 — routing float gate | **해소** | `CountGate`와 `Fraction`, correct/denominator 구조가 유일 판정 출처로 정의됐다. |
| m2 — BM25 `tracemalloc` | **해소** | tokenizer별 새 subprocess 3회, OS별 단위 정규화한 `ru_maxrss` peak median이 주 판정값이며 `tracemalloc`은 진단값이다. |
| m3 — evaluator variants fail-open | **해소** | 공식 `v2`는 파일 부재/schema 오류/기대 SHA 불일치 시 exit 2, 무변형은 `v2-no-variants`와 `official=false`로 분리됐다. |
| m4 — Phase 0 로그 부재 | **해소** | `evaluation/reports/m3/.../logs/` 경로, SHA-256, UTC, exit code 계약이 정의됐다. |
| t1 — candidate ID 불일치 | **해소** | 단일 정규식과 phase letter 의미, 금지 표기, 예시가 일치한다. |
| t2 — loopback 미검증 | **해소** | `urllib.parse`/`ipaddress` 기반 loopback 함수, 비-loopback 자동 억제, report reason과 양·음성 테스트가 정의됐다. |

## 4. 필수 확인 항목별 결론

1. **61/44/54 metric 및 evaluator 연속성:** 완결. dataset 76건은 category 51/15/3/7, `expected_route` 61/15이며 승인 baseline TP 44와 현 evaluator 정의가 일치한다.
2. **동일 process warm-up:** 구현 가능한 API·metadata·gate·테스트로 완결. warm-up 실패는 측정을 버리지는 않되 latency 판정을 `null`로 만드는 계약도 명확하다.
3. **LLM 전 explicit signal 및 exception/no-tool:** 실행 순서와 query 계약은 완결됐으나 WEB signal taxonomy의 recall 누락 때문에 M1이 남는다.
4. **`trace=None` MMR 폴백:** null-safe 의사코드와 요구된 6칸 행렬이 완결됐다. 초기화 강등도 별도 제품 테스트가 있다.
5. **Markdown link checker:** anchor/path/exit code/표준 라이브러리 실행 방식은 충분히 구체적이나 tracked-only 기본 입력 때문에 Gate 실행 가능성이 M2에서 차단된다.
6. **기타 1차 권고:** evaluator v2 fail-closed, BM25 RSS, Phase 0 logs, candidate regex, loopback, 모듈 축소 모두 반영됐다. Phase 1 gate를 `compare.py`에 유지하고 `fingerprint.py`를 thin CLI로 제한한 축소도 타당하다.
7. **신규 모순·과설계·누락:** MAJOR 2와 편집상 TRIVIAL 2 외에 새 CRITICAL/과설계는 확인하지 못했다.

## 5. 최종 결론

현재 설계는 Iteration 1 대비 크게 개선됐고 핵심 기술 계약 대부분은 구현 착수 가능한 수준이다. 그러나 필수 routing 정책과 필수 Markdown 회귀 Gate가 정상 입력을 놓치는 두 MAJOR가 있어 승인 Gate를 통과할 수 없다.

- **Gate: REVISE**
- **Score: 9.3 / 10**
- **Counts: CRITICAL 0, MAJOR 2, MINOR 0, TRIVIAL 2**
- 다음 iteration 최소 조건: M1/M2 수정 및 관련 테스트 계약 추가, CRITICAL/MAJOR 0, 점수 9.7 이상

## 6. 검증 기록

- `evaluation/datasets/golden.jsonl` 직접 집계: 76건; category `document_qa=51`, `web_search=15`, `boundary=3`, `unanswerable=7`; `expected_route=document_qa=61`, `web_search=15`.
- `evaluation/baselines/m2_initial.json`: routing `correct_count=59`, document confusion TP `44`, web TP `15`; retrieval/answer 승인 수치와 문서 표를 대조했다.
- 현재 코드: `evaluation/routing.py`가 `expected_route`를 기준 label로 사용하고, `RAGEngine.query()`는 retrieval trace를 제공하지 않으며, evaluator들은 엔진을 지연 획득한다는 전제를 확인했다.
- 현재 Git 상태: M3 문서 디렉터리가 미추적 상태이므로 `git ls-files` 단독 link checker가 이를 제외한다는 M2를 실제 worktree 상태로 확인했다.
- 문서 수정은 하지 않았고 이 리뷰 파일만 추가했다. commit/push는 수행하지 않았다.
