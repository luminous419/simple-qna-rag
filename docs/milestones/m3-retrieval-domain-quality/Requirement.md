# M3 Retrieval & Domain Quality 요구사항

상태: **착수 제안** — 범위·수용 기준 승인 필요
기준일: 2026-08-05 (라우팅 단순화 사이클 1회차: 2026-08-07, §5 M3-REQ-004 결정론적 WEB 신호 정의 재작성)
기준선: [사용자 승인 M2 최초 기준선](../../../evaluation/baselines/m2_initial.md)

## 1. 목적

M3는 승인된 M2 평가 자산을 사용해 로컬 문서 검색의 응답 시간을 줄이고, 문서 질문을 불필요하게 웹으로 보내는 오류를 교정하며, Answer 측정의 알려진 false negative를 제거한다. 동시에 현재 Intent Classifier가 답변 품질에 실제로 기여하는지를 대조 실험으로 판단해 유지·개선 또는 제거를 결정한다.

M3는 후보 기술을 모두 도입하는 마일스톤이 아니다. 같은 corpus와 골든셋에서 사용자 가치가 측정되고 비용·위험이 통제되는 변경만 제품 경로로 승격한다.

## 2. 기준 상태와 문제 정의

비교 기준은 `evaluation/baselines/m2_initial.{json,md}`의 승인된 실행이다. 기준선은 Git commit `95a4fd17d6658e548658f4e922750ae114625851`, dataset SHA-256 `61b768acd8d33522ef76e3baadd4bf19b44cc25daa79ad7ea255fb0a09d1017a`, corpus manifest SHA-256 `5c0d648d032bc231c93cf1c545e5b3f8d26337da56677a4a0c1a66f24c82374a`, `index.faiss` SHA-256 `c52fb288f3bc780d681ab68d62dd5b4c545843ee9dc53ef5528bd50db4d69820`, `index.pkl` SHA-256 `3f7217a2825f141bbea4c75dbf2b15738c3bcd50e3c0900e25a6a4a7ed91bb00`를 사용했다.

| 영역 | M2 승인 수치 | 알려진 문제 |
|---|---:|---|
| Retrieval 품질, 42건 | Recall@1/3/5/10 `94.05/94.05/95.24/97.62%`, MRR@10 `98.21%`, nDCG@10 `95.43%` | 이미 높으므로 최적화 중 회귀 방지가 우선 |
| Retrieval latency | 평균 `16.84초`, median `16.85초`, p95 `22.61초` | MMR 평균 `14.35초`; 후보 본문을 매 질문 재임베딩 |
| Routing, 76건 | accuracy `77.63% (59/76)`, document route recall `72.13% (44/61)`, web search recall `100% (15/15)` | 17개 오류가 모두 `document_qa → web_search` |
| Answer, 29건 | assertion `75% (24/32)`, abstention `89.66% (26/29)`, source any-hit `100%`, source recall `95.45%` | 사람 검토상 assertion 8개와 abstention 3개 실패가 모두 evaluator false negative |
| Intent, 29건 | accuracy `51.72% (15/29)` | 도메인 불일치; 별도 분류 계층의 효용이 입증되지 않음 |
| Answer latency | 평균 `55.48초`, median `54.16초`, p95 `74.88초` | Retrieval 지연이 포함되지만 생성 변동도 큼 |

M2 기준선은 소급 수정하지 않는다. evaluator 규칙 변경은 schema/evaluator version을 올리고, 기존 규칙 결과와 새 규칙 결과를 구분해 보고한다.

## 3. 범위 결정

### 3.1 필수 범위

1. **MMR 계산 경로 최적화**: FAISS에 저장된 후보 벡터 또는 요청 간 안전한 캐시를 재사용해 후보 본문의 반복 임베딩을 제거한다. 저장된 벡터와 문서의 대응 관계를 검증할 수 없으면 해당 방식은 채택하지 않는다.
2. **문서 우선 라우팅 교정**: 17개 실패 사례를 신호·표현 유형으로 분류하고, 명시적 웹 요청과 실제 최신성이 필요한 질문의 web recall을 지키면서 문서 질문 recall을 높인다.
3. **Answer evaluator v2**: Unicode/대소문자 외에 공백, 숫자·단위, underscore와 승인된 동의 표현을 일관되게 처리하고 abstention 표현을 확장한다. 제품 답변을 평가기에 맞춰 바꾸지 않는다.
4. **Intent 효용 결정**: 동일 질문·동일 검색 context에서 현재 intent별 템플릿과 단순 기본 템플릿을 짝지어 비교한다. 결과에 따라 classifier를 유지·개선하거나 제거한다.
5. **M3 비교·승인 기록**: 동일 조건 비교, 실패 사례, 자원·latency, 사람 검토와 승인 결과를 보존한다.

### 3.2 조건부 범위

한국어 BM25 tokenizer는 표준 라이브러리 또는 이미 설치된 경량 후보와 현재 공백 tokenizer를 오프라인 A/B 비교한다. 다음을 모두 만족할 때만 제품 경로에 적용한다.

- Retrieval 42건에서 Recall@10 또는 nDCG@10이 **최소 1.00%p** 개선된다.
- M3 Retrieval 품질 floor를 모두 만족한다.
- 새 필수 native runtime, 외부 서비스 또는 배포 크기 급증이 없다.
- 초기화 시간과 인덱스 메모리 증가가 각각 **20% 이하**다. 메모리는 동일 process의 **RSS peak**를 주 판정값으로 하고 `tracemalloc`은 Python allocation 진단값으로만 병기한다(native/모델 메모리를 놓치지 않기 위함).

충족하지 않으면 실험 결과만 기록하고 현행 tokenizer를 유지한다. 형태소 분석기 도입 자체는 완료 조건이 아니다.

### 3.3 제외 범위

- heading/문단 청킹 변경, 문서 metadata schema 재설계, 전체 재색인
- 웹 원문 수집·LLM 종합·도메인 신뢰도 정책
- query rewriting, multi-query, 답변 faithfulness LLM judge
- embedding/reranker/LLM 모델 교체
- ANN/vector database, semantic cache, 증분·원자적 인덱싱
- 운영 로그·동시성·dependency lock·vectorstore provenance
- 새 대규모 학습 dataset 또는 classifier 아키텍처 연구

청킹·모델·웹 콘텐츠 변경은 corpus/vectorstore 비교 조건, 보안 및 비용을 동시에 바꾸므로 M3의 원인 귀속을 흐린다. 운영·확장 항목은 각각 M4/M5의 책임을 유지한다.

## 4. 정량 목표와 회귀 허용치

모든 비율은 반올림 표시값이 아니라 report의 원시 count/float로 판정한다. 후보 실험은 동일 dataset·corpus·vectorstore fingerprint와 모델 설정을 사용한다. 의도적으로 dataset 또는 artifact를 바꾼 실행은 동일 조건 비교가 아니며 별도 승인 없이는 M3 gate에 사용할 수 없다.

### 4.1 승격 필수 gate

| 지표 | M2 기준 | M3 합격 기준 | 근거 |
|---|---:|---:|---|
| Retrieval 평균 latency | 16.84초 | **8.42초 이하** (50% 이상 감소) | MMR 병목 제거의 사용자 가치 |
| Retrieval p95 latency | 22.61초 | **13.57초 이하** (40% 이상 감소) | tail latency도 함께 제한 |
| MMR 평균 latency | 14.35초 | **2.87초 이하** (80% 이상 감소) | 반복 임베딩 제거 여부 확인 |
| Recall@10 | 97.62% | **95.24% 이상** | 42건 기준 최대 1개 사례 수준의 저하 허용 |
| Recall@5 | 95.24% | **92.86% 이상** | 최종 후보 전단 품질 보호 |
| MRR@10 | 98.21% | **96.00% 이상** | 상위 순위 관련성 보호 |
| nDCG@10 | 95.43% | **93.00% 이상** | graded relevance 보호 |
| Routing accuracy | 77.63% (59/76) | **90.79% 이상 (69/76)** | 오류를 17건에서 최대 7건으로 축소 |
| Document route recall (분모 61) | 72.13% (44/61) | **88.52% 이상 (54/61)** | 과다 웹 라우팅의 직접 목표 |
| Web search recall (분모 15) | 100% (15/15) | **100% (15/15)** | 최신·명시적 웹 요청 회귀 금지 |
| Source any-hit | 100% | **100%** | 근거 제공 회귀 금지 |
| Source mean recall | 95.45% | **93.00% 이상** | 검색 순위 변경의 답변 영향 제한 |

#### Routing recall의 단일 metric 계약

Routing recall은 evaluator가 이미 사용하는 정의 하나만 쓴다. 다른 분모를 섞어 표기하지 않는다.

- **분모는 `expected_route` 기준이다.** `expected_route == document_qa`인 사례는 **61건**(category `document_qa` 51 + `boundary` 3 + `unanswerable` 7), `expected_route == web_search`인 사례는 **15건**이다.
- `document_route_recall = confusion_matrix[document_qa][document_qa] / 61`, `web_search_recall = confusion_matrix[web_search][web_search] / 15`. 이는 `evaluation/routing.py`가 `expected_route`를 actual label로 넘겨 `precision_recall_f1()`을 계산하는 현재 구현과 동일하며, M2 승인값 `72.13% = 44/61`을 그대로 재현한다.
- **category `document_qa` 51건은 recall 분모가 아니다.** 51을 분모로 쓰면 M2 승인값 자체가 재현되지 않으므로 요구사항·계획·설계·gate·추적표는 어디에서도 51을 recall 분모로 사용하지 않는다. category 단위 분석이 필요하면 `document_qa_category_recall`이라는 **다른 이름의 보조 지표**를 새로 정의하고 M2 분자를 다시 산출해 병기해야 하며, M3 gate에는 쓰지 않는다.
- M3 최소 기준 `54/61 (88.52%)`은 accuracy `69/76`(오류 ≤ 7건)과 `web_search_recall = 15/15`(모든 오류가 `document_qa → web_search`)이 동시에 성립할 때 강제되는 값이므로 세 gate의 count가 서로 모순되지 않는다.
- 판정은 표시 백분율이 아니라 count 비교 또는 `Fraction`으로 수행한다: `accuracy >= Fraction(69, 76)`, `document_route_recall >= Fraction(54, 61)`, 각 run의 `web_search_recall == Fraction(15, 15)`.

#### Latency 측정 조건

Latency는 동일 host에서 **동일 process·동일 engine 인스턴스로 warm-up을 수행한 뒤** 전체 42건을 실행한 report로 판정한다. 절대 기준과 감소율을 모두 만족해야 한다. warm-up 계약은 M3-NFR-002에 정의한다. 환경 변화가 확인되면 M2 commit과 후보 commit을 같은 세션에서 각 1회 재실행한 paired 비교를 우선하며, 재실행 기준값과 원래 승인 기준선을 모두 보고한다.

Routing은 비결정성을 고려해 후보 설정으로 전체 76건을 **3회** 실행한다. 각 실행에서 web search recall 15/15를 만족하고, 3회 중 중앙값이 accuracy 및 document route recall 기준을 만족해야 한다. 세 실행 간 route가 달라진 사례를 별도 표시한다.

### 4.2 Answer evaluator와 Intent 판정 gate

- evaluator v2 fixture는 M2 사람이 확인한 8개 assertion 변형과 3개 올바른 abstention을 모두 true positive로 판정해야 한다.
- 기존 자동 true positive 전체가 v2에서 false negative로 바뀌면 안 된다.
- 숫자·부정·단위의 의미를 지우는 과도한 정규화는 금지한다. 동의어는 dataset의 `any_of` 또는 검토된 명시적 규칙으로만 추가한다.
- v1 점수는 재현 가능해야 하며 v2 점수와 한 열에 혼합하지 않는다. v2 결과에는 evaluator/schema version과 규칙 fingerprint를 기록한다.
- 공식 v2 판정은 검토 변형 표에 대해 **fail-closed**여야 한다. 변형 파일이 없거나 schema가 깨졌거나 fingerprint가 기대값과 다르면 다른 규칙으로 조용히 계속 실행하지 않고 실행을 실패시키거나 "판정 불가"로 처리한다. 변형 없이 정규화만 쓰는 실험은 별도 profile로 명시하고 공식 gate에 사용하지 않는다.

Intent 결정은 29개 Answer 대상에 대해 검색 결과를 고정한 paired blind review로 수행한다. 최소 두 검토 축(질문 형식 적합성, 핵심 사실 보존)을 사례별 0/1로 기록한다.

- intent별 템플릿이 기본 템플릿보다 총 선호 사례에서 **최소 20%p** 앞서고, assertion/abstention/source gate에 회귀가 없으면 classifier를 유지할 수 있다.
- 유지하는 경우 `yesno`·`uncertain` 실패를 보강하고 intent accuracy **75.86% 이상 (22/29)**을 달성해야 한다.
- 위 효용 기준을 충족하지 못하면 classifier와 학습 artifact를 즉시 삭제해야 한다는 뜻은 아니다. production 답변 경로를 기본 템플릿으로 단순화하고, 호환성·artifact 정리는 후속 승인 범위로 분리한다.
- 동률 또는 검토자 불일치는 사용자 승인 gate에서 보수적으로 “입증되지 않음”으로 처리한다.

Answer End-to-End latency는 생성 변동이 크므로 승격 차단 기준을 평균·p95 각각 M2 대비 **10% 초과 악화 금지**(평균 `61.03초`, p95 `82.37초` 이하)로 둔다. Retrieval 최적화 효과는 별도 Retrieval gate로 판정한다.

## 5. 기능 요구사항

### M3-REQ-001 — 비교 가능한 실험 경계

각 후보 실행은 candidate ID, Git SHA와 dirty 상태, Python·모델·retrieval 설정, dataset/corpus/vectorstore fingerprint, evaluator version을 기록해야 한다. M2 metadata 한계를 숨기지 않고 dependency snapshot 부재를 report limitation에 유지한다.

### M3-REQ-002 — 안전한 MMR 벡터 재사용

MMR은 동일 후보 문서의 embedding을 질문마다 다시 계산하지 않아야 한다. 구현은 다음 계약을 만족해야 한다.

- query embedding은 질문당 한 번 이하로 계산한다.
- 후보 문서와 재사용 벡터의 대응이 안정적이고 검증 가능해야 한다.
- 대응 실패, dimension 불일치 또는 non-finite vector는 조용히 잘못된 결과를 내지 않고 명확히 실패하거나 검증된 기존 경로로 폴백한다.
- 공용 cache를 사용하면 key에 문서 내용 또는 안정 ID와 embedding 설정을 포함하고 크기 제한·동시 접근 안전성을 정의한다.
- 계측 활성/비활성 여부가 문서 순서를 바꾸지 않는다. **계측이 비활성인 제품 호출 경로(trace 미제공)에서도 위 폴백이 동일하게 동작해야 하며, 계측 객체 부재가 새로운 예외를 만들어서는 안 된다.**

### M3-REQ-003 — 검색 품질·성능 비교

현행과 후보를 같은 42건에 실행해 per-case 순위 변화와 BM25/Dense/RRF/MMR/reranker/total latency를 생성한다. 집계만으로 숨지 않도록 floor를 깬 사례와 source 순위 변화가 report에 있어야 한다.

### M3-REQ-004 — 라우팅 오류 taxonomy와 정책

17개 M2 오류를 최소한 `명시적 로컬 문서 신호`, `연도/최신 표현`, `정책·시장·기업 주제`, `복합/모호 표현`으로 중복 태깅한다. 정책은 다음 우선순위를 외부에서 검증 가능하게 해야 한다.

1. 사용자가 웹/인터넷/실시간 검색을 명시하면 web search
2. 사용자가 제공 문서·로컬 자료·문서 내 근거를 명시하면 document QA
3. 나머지는 골든 실패 taxonomy와 실제 최신성 필요 여부로 판정

1·2순위는 **LLM 호출 이전에** 결정론적으로 판정해야 하며, LLM이 예외를 던지거나 도구를 선택하지 않아도 그 우선순위가 유지되어야 한다. 즉 명시 신호가 있는 질문의 최종 route는 모델 가용성과 무관해야 한다. 이때 tool query 계약(web은 검색어, document는 원본 질문)도 함께 보존해야 하며, web 경로의 검색어는 LLM이 정제한 값을 우선 쓰되 LLM 실패 시 검증된 결정론적 추출 결과로 대체한다.

**[라우팅 단순화 사이클 1회차, 2026-08-07]** 기존 설계(Iteration 1~6)는 조사·주제어·거리 예외를 조합했고, 최종 리뷰에서 `SOURCE_PARTICLE`가 관형절의 주제 질문을 WEB으로 승격하는 MAJOR가 남아 중단됐다. 새 사이클은 의미를 추측하는 예외를 제거하고 **검색 행위 자체가 명령형 술어로 드러난 경우만** 결정론적 WEB으로 인정한다. false positive 비용이 더 크므로 결정론적 WEB은 precision 100%를 계약으로 하며, 놓친 명시 표현과 애매한 경계는 실패가 아니라 3순위 LLM 위임이다.

1순위의 "명시"는 채널 지시어(웹/인터넷/온라인/구글/포털/검색엔진, "web"/"google" 포함)나 채널과 검색이 한 어구로 붙은 **융합형** 표현(예: "웹검색", "웹 검색", "웹서치", "구글링")이 문장 어디에든 등장하는 것만으로 성립하지 않는다. 다음 **두 문형 중 하나**에 해당할 때만 명시적 웹 **사용 요청**으로 인정한다.

1. **직접 검색 명령**: 융합형 검색 표현에 명령형 어미가 바로 결합한다. 예: `웹검색해줘`, `구글링해줘`, `구글링해서 알려줘`. 중간 명사구를 허용하지 않는다.
2. **채널 지정 검색 명령**: 채널·융합 표현에 `에서/으로/로`가 직접 결합하고, 문장의 마지막 술어가 `검색/찾/조회/확인/알아보` 계열의 **검색 행위 명령**이다. 예: `최신 환율을 인터넷에서 찾아줘`, `온라인에서 검색해줘`. `알려/답해/보여` 같은 일반 응답 동사는 검색 행위 명령으로 보지 않는다.

두 문형 중 어느 쪽도 성립하지 않으면 3순위 LLM이 판단한다. 여기에는 검색 기술·기능·방법·API·구조 질문, 인용·부정, 관형절, 일반 응답 동사만 있는 출처 표현(`웹검색으로 알려줘`, `인터넷에서 답해줘`), 조사 없는 채널 언급, 최신성 표현만 있는 질문이 포함된다. 특히 `웹검색에서 사용하는 API 구조 알려줘`와 `구글에서 사용하는 검색 기술 알려줘`는 조사 뒤 관형절을 분석하는 예외를 추가하지 않고도 마지막 술어가 일반 응답 동사이므로 NONE이다.

**SOURCE_PARTICLE, 융합형 표현, 연도·최신성 표현은 각각 충분조건이 아니다.** `SOURCE_PARTICLE + 일반 응답 동사`도 충분조건이 아니다. `TOPIC_HEAD`, 조사 뒤 관형절, 어절 거리 같은 보정 목록은 구현 계약에 두지 않는다.

부정·금지 표현("하지 말고", "없이", "말고" 등)이 문장 안에 있으면, 그 문장이 위 두 문형을 만족하더라도 명시 WEB 요청으로 인정하지 않는다. 인용·인라인 코드 구간(`"..."`, `` `...` `` 등) 안의 토큰은 이 판정에서 제외한다. 채널 지시어·융합형 표현의 왼쪽 경계는 문자열 시작이나 공백 뒤로만 한정하지 않는다. 쉼표·콜론·괄호·따옴표 등 정상적인 문장부호 뒤도 왼쪽 경계로 인정하되, 영숫자·한글이 곧바로 이어지는 복합어 내부(예: "freewebsearch", "websocket", "googleapis")는 경계로 인정하지 않는 Unicode 문자 경계로 판정한다. 즉 "질문:웹에서 검색해줘"나 "(구글링해서 알려줘)"처럼 문장부호·괄호 바로 뒤에 오는 진짜 검색 명령은 명시 요청으로 인정해야 한다.

동일 질문에서 웹 명령 문형과 명시적 문서 범위 지시가 동시에 관측되면(예: "웹검색으로 이 문서 내용을 확인해줘"), 웹 명령 문형(1순위)이 문서 범위 지시(2순위)보다 강한 신호로 우선한다 — 이 우선순위는 기존 설계와 동일하게 유지한다(§ 앞머리 우선순위 목록).

명시 신호가 없는(3순위) 질문에서는 라우터가 선택한 tool query 계약과 Agent 장애 시 기존 keyword fallback, 웹 검색 실패 시 원본 질문의 document QA 재시도를 그대로 보존한다.

### M3-REQ-005 — Routing 반복 평가

live evaluator는 동일 설정의 3회 결과를 묶어 run별 지표, 중앙값, 사례별 변동 횟수를 보고할 수 있어야 한다. 구현 방식은 단일 CLI의 `--runs` 옵션 또는 동일 명령의 세 report를 비교하는 별도 도구 중 선택할 수 있다.

### M3-REQ-006 — Answer evaluator v2

정규화와 abstention 판정은 순수 함수로 분리하고 단위 테스트 가능해야 한다. report는 v1/v2 규칙을 명시하며 과거 baseline 파일을 덮어쓰지 않는다. v2가 의미 기반 correctness나 faithfulness를 보증한다고 표현해서는 안 된다.

### M3-REQ-007 — Intent 대조 실험과 결정 기록

두 variant는 같은 질문과 같은 retrieved context를 사용해야 하며 출력 순서를 무작위화하거나 A/B 정체를 가린 worksheet를 제공해야 한다. 결정 기록은 유지/개선 또는 단순화 결론, 정량 결과, 반대 근거, 선택된 production 경로를 포함해야 한다.

### M3-REQ-008 — 조건부 BM25 실험

tokenizer 실험은 tokenizer 인터페이스와 fixture를 통해 현행 공백 분리와 후보를 비교한다. 제품 반영 여부는 §3.2 gate로 결정하며, 미채택 실험 코드가 production 의존성이나 기본 경로에 남아서는 안 된다.

### M3-REQ-009 — 실패 안전성과 호환성

기존 public CLI/API 응답의 `answer`, `sources`, `success`, `search_type`, `intent` 계약을 승인 없이 깨지 않는다. vectorstore 원본을 덮어쓰거나 재생성하지 않는다. live opt-in 없이 Ollama·네트워크·모델 다운로드를 요구하는 테스트를 CI에 추가하지 않는다.

### M3-REQ-010 — 승인 baseline과 추적성

최종 후보의 통합 live report, 사람 검토 worksheet, 요구사항별 결과표와 알려진 한계를 사용자에게 제시한다. 사용자 승인 전에는 M3 baseline을 `evaluation/baselines/`에 고정하거나 Roadmap 상태를 완료로 바꾸지 않는다.

## 6. 비기능 요구사항

### M3-NFR-001 — 재현성

결정론적 테스트는 고정 fixture와 stable ordering을 사용한다. 보고서 JSON은 기계 판독 가능하고 사람이 보는 Markdown과 동일한 집계를 가져야 한다.

### M3-NFR-002 — 성능 측정 건전성

`time.perf_counter()`를 유지하고 모델 초기 로딩과 warm-up 여부를 명시한다. 병렬 실행으로 서로 자원을 다투는 latency 결과를 공식 비교에 사용하지 않는다.

warm-up은 다음 계약을 만족해야 한다.

- warm-up과 공식 측정은 **동일 process·동일 engine 인스턴스**에서 연속 수행한다. 별도 프로세스 실행은 모델 객체와 메모리가 종료와 함께 사라지므로 warm-up으로 인정하지 않는다.
- warm-up 실행분의 결과·latency는 공식 집계(지표·latency 통계)에서 **완전히 버린다.** warm-up은 공식 표본을 잠식하지 않으므로, warm-up 후 측정되는 사례 수는 승인 구성(Retrieval 42건, Answer 29건)과 정확히 같아야 한다.
- report에는 warm-up 요청 수, 실제 실행 수, 성공/실패 수, 측정 제외 여부를 **구조화된 metadata 필드**로 기록한다. 사람이 작성한 자유 서술(notes)만으로 warm-up을 주장할 수 없다.
- warm-up이 0건이거나 실패하면 그 실행의 latency gate는 "판정 불가"로 표시한다.

### M3-NFR-003 — 보안·프라이버시

상세 질문·답변·context가 포함된 timestamp report와 worksheet는 `evaluation/reports/` 아래에 두고 Git에 커밋하지 않는다. 웹 원문 수집이나 외부 전송을 새로 추가하지 않는다.

corpus 파일명 같은 로컬 자산 정보를 프롬프트에 넣는 기능은 LLM endpoint가 **loopback**일 때만 활성화한다. endpoint가 loopback이 아니면 해당 기능을 자동 비활성화하고 report에 억제 사유를 남긴다.

### M3-NFR-004 — 유지보수성

최적화는 production retrieval 로직을 evaluator에 복제하지 않는다. 새 전략 경계는 fake로 테스트할 수 있어야 하며 optional dependency가 없을 때 import 자체가 실패하지 않아야 한다.

### M3-NFR-005 — 회귀 방지

Python·frontend 전체 테스트, dataset validation, Markdown local link 검사와 `git diff --check`가 통과해야 한다. live 평가는 명시적 opt-in과 사용자 승인 절차를 따른다.

Markdown local link 검사는 **새 외부 dependency 없이** 저장소 안의 표준 라이브러리 스크립트로 수행하며, 모든 Phase가 같은 명령을 쓴다.

```bash
python scripts/check_markdown_links.py
```

검사 범위와 실패 조건은 상세 설계에서 확정한다(깨진 상대 경로와 존재하지 않는 heading anchor는 실패, 외부 URL은 검사 대상 아님).

## 7. 완료 및 승인 조건

M3 완료에는 다음이 모두 필요하다.

1. M3-REQ-001~010과 NFR-001~005의 추적표가 작성되고 미충족 항목이 없다.
2. §4의 Retrieval, Routing, Answer 회귀 gate를 모두 만족한다.
3. Intent 유지/단순화 결정과 조건부 BM25 채택/기각 결정에 근거가 있다.
4. mock/단위/통합/프런트엔드 테스트가 성공하고 live report가 기술 오류 없이 완료된다.
5. 상세 worksheet의 source relevance, assertion/abstention 및 A/B 결과를 사람이 검토한다.
6. CRITICAL/MAJOR 이슈가 0이고 MINOR가 승인 가능할 만큼 최소화된다.
7. 사용자가 최종 report와 잔여 위험을 승인한다.

승인 전 상태는 “진행 중”이며, 목표 일부 달성만으로 완료 처리하지 않는다.

## 8. 전제와 열린 결정

- 기준 환경은 Python 3.11, Ollama `gpt-oss:20b`, BGE-M3 embedding/reranker와 현재 18개 corpus/vectorstore다.
- M2 기준선에는 dependency snapshot과 vectorstore 생성 provenance가 없다. 이는 M3 비교의 알려진 한계이며 M4 범위를 선점하지 않는다.
- 라우팅의 3회 live 실행 및 Intent blind review 비용을 사용자가 승인해야 한다.
- 사용자 승인 시 §4 수용 기준을 변경할 수 있으나, 구현 결과를 본 뒤 유리하게 사후 완화하면 변경 사유와 원래 기준을 함께 기록해야 한다.
- **정정 기록(2026-08-05, 설계 리뷰 iteration 1)**: Document QA recall 기준의 초기 표기 `86.27% 이상 (44/51)`은 서로 다른 모집단(category 51, `expected_route` 61)을 섞은 값이어서 단일 metric으로 성립하지 않았다. §4.1의 단일 metric 계약(분모 61, M2 `44/61`, M3 `54/61`)으로 정정했다. 이는 목표를 완화한 것이 아니라 M2 승인값과 재현 가능하게 정합시킨 강화(86.27% → 88.52%)이며, gate·계획·설계·추적표가 모두 같은 정의를 쓴다. 이 정정에 대한 사용자 승인이 필요하다.
- **라우팅 단순화 사이클 기록(2026-08-07, 라우팅 단순화 사이클 1회차)**: 기존 설계 Iteration 1~6은 감사 기록으로 `Design_Review.md`~`Design_Review_Iteration_6.md`, `Stop_Report.md`에 보존한다. 별도 단순화 사이클은 최대 6회이며 §5 M3-REQ-004의 1순위만 재작성한다. 2·3순위, DOCUMENT, tool query, fallback 계약은 변경하지 않는다. 규범 기대값은 WEB 8 / DOCUMENT 12 / NONE 56, 결정론적 WEB·DOCUMENT 오탐 0이며 상세 grammar와 경계 표는 `Design.md` §7.2에 있다.
