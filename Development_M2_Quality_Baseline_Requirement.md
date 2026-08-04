# M2 Quality Baseline & CI 요구사항

## 1. 문서 목적

이 문서는 `M2 — Quality Baseline & Continuous Integration` 구현의 필수 요구사항과 수용 기준을 정의합니다. 구현 에이전트와 리뷰어는 이 문서를 범위와 완료 여부를 판단하는 기준으로 사용합니다.

구현 순서와 작업 분해는 [Development_M2_Quality_Baseline_Plan.md](Development_M2_Quality_Baseline_Plan.md), 상위 목표는 [Roadmap.md](Roadmap.md), 현재 기술 부채는 [Problem.md](Problem.md)를 참조합니다.

## 2. 규범 용어

- **MUST / 필수**: M2 완료를 위해 반드시 충족해야 합니다.
- **SHOULD / 권장**: 특별한 사유가 없으면 충족해야 합니다. 제외 시 근거를 기록합니다.
- **MAY / 선택**: 범위와 일정이 허용할 때 구현할 수 있습니다.

## 3. 배경

현재 프로젝트는 문서 등록, Hybrid Retrieval, MMR, reranker, Intent 분류, Ollama 답변 생성, Agent 라우팅과 장애 폴백을 제공합니다. Python 테스트 21개와 프런트엔드 테스트 9개가 있지만 대부분 mock 또는 DOM 단위 테스트이며 다음이 없습니다.

- 실제 문서를 기준으로 한 골든 평가셋
- Retrieval 품질 지표
- 답변 핵심 사실과 출처 일치 평가
- 단계별 latency 기준선
- Pull Request 자동 테스트
- 기준선 결과와 실행 환경의 재현 가능한 기록

M2는 검색 알고리즘 자체를 개선하는 마일스톤이 아니라, 이후 변경을 객관적으로 비교할 수 있는 측정 기반을 만드는 마일스톤입니다.

## 4. 목표

- 대표 질문 60개 이상의 버전 관리 가능한 골든 평가셋을 구축합니다.
- 실제 production retrieval 경로를 사용해 Recall@K, MRR, nDCG를 측정합니다.
- Agent 라우팅 정확도와 오류 유형을 측정합니다.
- 문서 답변의 핵심 사실 포함 여부, abstention, 출처 일치를 기본 평가합니다.
- Retrieval 및 End-to-End latency 기준선을 기록합니다.
- 외부 서비스 없이 실행 가능한 Python/Node 검증을 CI에 연결합니다.
- 결과에 Git revision, 설정, 모델, 데이터셋 버전을 포함해 비교 가능하게 만듭니다.

## 5. 범위 제외

다음은 M2에서 구현하지 않습니다.

- 한국어 BM25 tokenizer 교체
- MMR 임베딩 캐싱 또는 검색 알고리즘 최적화
- chunk 전략과 검색 하이퍼파라미터 변경
- Intent Classifier 재학습 또는 제거
- 웹 검색 원문 수집과 LLM 요약
- RAGAS 또는 LLM judge를 필수 평가기로 도입
- 대화형 RAG, multi-hop, semantic cache
- Docker 또는 프로덕션 배포 자동화
- 운영용 Prometheus/Grafana 관측성

평가 기반 구축에 필요한 최소한의 production 코드 계측은 허용하지만, 기존 검색 결과를 의도적으로 변경해서는 안 됩니다.

## 6. 전제와 실행 환경

- Python 3.11을 기준 환경으로 사용합니다.
- Node.js 20 이상을 프런트엔드 검증 환경으로 사용합니다.
- 로컬 live 평가는 Ollama `gpt-oss:20b`, 현재 `data/`, 현재 `vectorstore/`를 사용합니다.
- `data/`와 `vectorstore/`는 Git에 커밋하지 않습니다.
- CI는 Ollama, DuckDuckGo, Hugging Face 모델 다운로드, 로컬 vectorstore에 의존하지 않아야 합니다.
- 골든 평가셋에는 원문 문서 전체나 민감한 내용을 복제하지 않고 식별자와 평가에 필요한 최소 사실만 기록합니다.

## 7. 요구사항

### M2-REQ-001 — 평가 패키지 구조

구현은 최소한 다음 구조를 제공해야 합니다. 파일명은 합리적인 사유가 있으면 조정할 수 있지만 역할 분리는 유지해야 합니다.

```text
evaluation/
├── __init__.py
├── schema.py
├── dataset.py
├── metrics.py
├── retrieval.py
├── routing.py
├── answers.py
├── baseline.py
├── reporting.py
└── datasets/
    └── golden.jsonl
```

추가로 다음을 제공해야 합니다.

```text
tests/evaluation/ 또는 test_evaluation_*.py
.github/workflows/ci.yml
```

평가 로직은 production 모듈을 호출해야 하며 검색 파이프라인을 별도로 복제해서는 안 됩니다.

### M2-REQ-002 — 골든 평가셋 규모와 구성

`evaluation/datasets/golden.jsonl`은 UTF-8 JSON Lines 형식이어야 하며 고유한 평가 사례를 최소 60개 포함해야 합니다.

최소 구성:

- 문서 QA 사례 40개 이상
- 웹 검색 사례 10개 이상
- 문서/웹 경계 또는 답변 불가 사례 10개 이상
- 모든 60개 사례에 기대 라우팅 결과 포함
- 문서 QA 중 30개 이상에 Retrieval 정답 문서 포함
- 문서 QA 중 20개 이상에 답변 핵심 사실 규칙 포함
- 답변 불가 사례 5개 이상
- 한국어 질문 80% 이상
- explanation, comparison, procedure, yesno 유형을 각각 5개 이상 포함 — 전체 60개 사례가 아니라 **Answer 평가 대상 사례**(M2-REQ-008 기준: `answer_assertions`가 하나 이상 있거나 `expect_abstention=true`) 중에서 셉니다. Phase 6의 intent 정확도는 실제 `RAGEngine.query()` 결과에 대해서만 측정되므로, web_search나 Retrieval 전용 document_qa 사례에만 각 유형을 배치해서는 이 조건을 충족할 수 없습니다.

한 사례가 여러 구성 조건을 동시에 충족할 수 있습니다.

### M2-REQ-003 — 골든 사례 스키마

각 JSON 객체는 다음 필드를 지원해야 합니다.

```json
{
  "id": "rag-001",
  "question": "RAG에서 MMR은 어떤 역할을 하나요?",
  "category": "document_qa",
  "expected_route": "document_qa",
  "expected_intent": "explanation",
  "relevant_sources": ["retrieval-augmented-generation-rag.pdf"],
  "relevance_grades": {
    "retrieval-augmented-generation-rag.pdf": 3,
    "retriever.pdf": 2
  },
  "answer_assertions": [
    {"any_of": ["다양성", "중복을 줄"]},
    {"any_of": ["관련성", "relevance"]}
  ],
  "expect_abstention": false,
  "tags": ["korean", "explanation", "retrieval"],
  "notes": "사람 검토용 설명"
}
```

필수 필드:

- `id`: 전체 파일에서 유일한 안정적 식별자
- `question`: 비어 있지 않은 문자열
- `category`: `document_qa`, `web_search`, `boundary`, `unanswerable` 중 하나
- `expected_route`: `document_qa`, `web_search` 중 하나
- `tags`: 문자열 배열

조건부 필드:

- Retrieval 평가 사례는 `relevant_sources`가 비어 있지 않아야 합니다.
- nDCG 평가 사례는 `relevance_grades`를 가져야 하며 등급은 0~3 정수여야 합니다.
- `relevance_grades`에서 양수 등급(1~3)을 받은 source는 정규화 후 `relevant_sources`에도 포함되어야 합니다 — 그렇지 않으면 nDCG 정답은 있는데 Retrieval 평가(`relevant_sources` 기준 대상 판정) 대상에서는 제외되는 모순이 생깁니다. grade 0(비관련) source는 `relevant_sources` 밖에 있어도 됩니다.
- `relevance_grades`가 비어 있지 않다면 최소 1개는 양수(1~3) 등급이어야 합니다 — 전부 0이면 nDCG의 IDCG가 항상 0이 돼 어떤 검색 결과를 넣어도 지표가 무의미해집니다.
- 답변 평가 사례는 `answer_assertions` 또는 `expect_abstention=true` 중 하나를 가져야 하며, **둘을 동시에 설정할 수 없습니다** — assertion은 "모델이 답하면 이 핵심 사실을 포함해야 한다"는 기대이고 abstention은 "모델이 답을 거절해야 한다"는 기대라서, 하나의 답변이 두 기대를 동시에 만족할 수 없습니다(M2_Phase1_code_review_result.md P1).
- `expected_intent`는 현재 지원 라벨 중 하나여야 합니다.

문서 파일명의 Unicode 정규화 차이로 평가가 깨지지 않도록 source 비교 전에 Unicode NFC 정규화, 경로 구분자 통일, basename 추출, 대소문자 정규화를 수행해야 합니다. 평가셋의 source ID 생성 규칙은 코드와 문서에 명시해야 합니다.

### M2-REQ-004 — 데이터셋 검증 명령

다음 명령 또는 동등하게 명확한 단일 명령을 제공해야 합니다.

```bash
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
```

검증기는 다음 오류에서 0이 아닌 종료 코드를 반환해야 합니다.

- JSON 파싱 실패
- 필수 필드 누락 또는 잘못된 enum/type
- 중복 ID
- 최소 사례 수 또는 구성 비율 미달
- 빈 질문
- 잘못된 relevance grade
- 답변 평가 사례의 assertion/abstention 조건 누락

CI는 이 검증 명령을 실행해야 합니다.

### M2-REQ-005 — Retrieval 평가

Retrieval 평가는 현재 `RAGEngine`의 실제 검색 경로를 사용해야 합니다. production 검색 로직을 평가 코드에 복사하면 안 됩니다.

필수 지표:

- Recall@1, Recall@3, Recall@5, Recall@10
- MRR@10
- nDCG@10: `relevance_grades`가 있는 사례 대상
- 평가 사례 수와 제외된 사례 수
- 평균, median, p95 retrieval latency

문서 일치 여부는 M2-REQ-003의 정규화된 source ID로 판단합니다. 각 사례별 검색 순위, source ID, 성공 여부와 latency를 결과에 보존해야 합니다.

실행 명령:

```bash
python -m evaluation.retrieval \
  --dataset evaluation/datasets/golden.jsonl \
  --output evaluation/reports/retrieval
```

명령은 vectorstore 또는 필요한 모델이 없을 때 원인을 설명하는 오류 메시지와 0이 아닌 종료 코드를 반환해야 합니다.

### M2-REQ-006 — Retrieval 단계 계측

평가 실행은 가능한 범위에서 다음 단계의 latency와 후보 수를 기록해야 합니다.

- BM25
- Dense Retrieval
- RRF
- MMR
- Re-ranker
- 전체 Retrieval

기존 `RAGEngine._retrieve_documents()`의 동작과 반환 결과는 보존해야 합니다. 계측 구현은 다음 중 하나를 사용할 수 있습니다.

- 호환되는 trace 반환 메서드 추가
- 선택적 observer/callback
- 평가 모드에서만 활성화되는 계측 객체

시간 측정은 `time.perf_counter()`를 사용해야 합니다. 계측 비활성 상태에서 기존 호출자가 변경될 필요가 없어야 합니다.

### M2-REQ-007 — Routing 평가

Routing 평가는 두 모드를 제공해야 합니다.

1. **offline**: 외부 LLM 없이 고정 응답/mock을 사용해 파싱, 집계, 오류 분류와 리포팅을 검증
2. **live**: 실제 `_decide_tool()`을 사용해 골든셋 전체 또는 선택된 subset 평가

live 필수 지표:

- 전체 정확도
- `document_qa` precision/recall/F1
- `web_search` precision/recall/F1
- confusion matrix
- 무도구 선택 및 예외 건수
- 평균, median, p95 routing latency
- 실패 사례의 ID, 질문, 기대값, 실제값

실행 명령:

```bash
python -m evaluation.routing \
  --dataset evaluation/datasets/golden.jsonl \
  --mode live \
  --output evaluation/reports/routing
```

live 모드는 `RUN_LIVE_LLM_TESTS=1` 또는 동등한 명시적 opt-in 없이는 실행되지 않아야 합니다. 기존 16개 `ROUTING_CASES`는 골든 평가셋으로 이전하거나 골든 평가셋에서 파생해 중복된 정답 소스를 만들지 않아야 합니다.

### M2-REQ-008 — 기본 Answer 평가

Answer 평가 대상은 `category`가 아니라 필드 존재 여부로 결정합니다 — `answer_assertions`가 하나 이상 있거나 `expect_abstention=true`인 모든 사례가 대상이며, `category=unanswerable`인 abstention 사례도 포함됩니다. `category=document_qa`이지만 두 필드가 모두 없는 사례(Retrieval 평가 전용)는 대상에서 제외됩니다. 실제 `RAGEngine.query()` 결과를 사용해야 합니다.

필수 자동 지표:

- answer assertion coverage: 각 `answer_assertions[].any_of` 중 하나 이상 포함 여부
- 전체 assertion 통과율
- abstention 정확도
- 반환 source와 `relevant_sources`의 일치율
- intent 정확도: `expected_intent`가 있는 사례
- 성공/오류 건수
- 평균, median, p95 End-to-End latency

문자열 비교는 Unicode NFC와 대소문자 정규화를 적용해야 합니다. assertion coverage는 답변의 진실성을 완전히 대체하지 않는 보조 지표임을 리포트에 명시해야 합니다.

사람 검토를 위해 각 사례의 질문, 답변, 반환 출처, 기대 핵심 사실, 자동 점수와 다음 빈 평가 필드를 포함한 Markdown 또는 CSV worksheet를 생성해야 합니다.

- faithfulness: 1~5
- relevance: 1~5
- completeness: 1~5
- citation correctness: 1~5
- reviewer note

실행 명령:

```bash
python -m evaluation.answers \
  --dataset evaluation/datasets/golden.jsonl \
  --output evaluation/reports/answers
```

LLM judge는 M2 필수가 아닙니다. 추가하더라도 기본 결정론적 평가와 분리하고 모델·프롬프트 버전을 결과에 기록해야 합니다.

### M2-REQ-009 — 통합 baseline 실행

다음과 동등한 단일 진입점을 제공해야 합니다.

```bash
RUN_LIVE_LLM_TESTS=1 python -m evaluation.baseline \
  --dataset evaluation/datasets/golden.jsonl \
  --output evaluation/reports
```

통합 실행은 dataset validation, retrieval, live routing, answer 평가 순서로 실행해야 합니다. 일부 단계 실패 시 전체 명령은 0이 아닌 종료 코드를 반환하되, 이미 생성된 단계 결과와 실패 원인을 보존해야 합니다.

선택적으로 `--skip-routing`, `--skip-answers`, `--limit`, `--tag`를 제공할 수 있습니다. 옵션을 제공한다면 README 또는 평가 문서에 사용법을 기록해야 합니다.

### M2-REQ-010 — 리포트 형식과 재현성 메타데이터

각 평가 명령은 다음 두 형식을 생성해야 합니다.

- 기계 판독용 JSON
- 사람 판독용 Markdown

각 리포트에는 최소한 다음 메타데이터가 포함되어야 합니다.

- schema version
- 실행 UTC timestamp
- Git commit SHA와 dirty 여부
- dataset 경로와 SHA-256
- Python 버전
- 평가 명령과 옵션
- embedding, reranker, Ollama 모델 이름
- 주요 retrieval 설정
- 사례 수, 성공/실패/제외 수
- 집계 지표
- 사례별 결과 또는 별도 상세 결과 파일 링크
- corpus manifest: `data/`의 각 파일에 대한 정규화 source ID, 크기, SHA-256을 담은 **배열 자체가 리포트에 포함**되어야 합니다(집계 SHA-256만으로는 어떤 파일이 달라졌는지 리포트만으로 확인할 수 없습니다) — 이 배열 전체를 정렬해 직렬화한 값의 SHA-256도 함께 기록합니다
- vectorstore fingerprint: `index.faiss`/`index.pkl`의 SHA-256

`data/`와 `vectorstore/`는 Git에서 제외되므로(§6) 이 두 필드 없이는 서로 다른 corpus/인덱스에서 생성된 리포트를 같은 조건으로 착각해 비교할 위험이 있습니다. 두 필드는 파일을 읽어 해시만 계산하면 되므로 모델이나 FAISS 로드 없이 저비용으로 산출할 수 있어야 합니다.

이 두 필드는 실제로 `data/`/`vectorstore/`를 사용하는 평가(Retrieval, Answer, 통합 baseline)의 리포트에서는 필수이며 값이 없으면 안 됩니다. Routing 평가는 corpus나 vectorstore를 전혀 사용하지 않으므로 이를 무조건 요구하면 독립 실행되는 Routing 평가가 불필요하게 로컬 데이터에 의존하게 됩니다 — Routing 리포트는 동일한 필드를 `null`로 채우고 그 이유를 별도 필드에 기록하는 것으로 충분하며, `data/`나 `vectorstore/`의 존재를 Routing 평가의 실행 조건으로 만들어서는 안 됩니다.

통합 baseline 리포트는 이 필드들을 각 단계 결과 안에만 두지 않고 **최상위(top-level)에도** non-null로 포함해야 합니다 — 리포트를 읽는 사람이나 프로그램이 단계별 하위 구조를 해석하지 않고도 실행 환경을 식별할 수 있어야 합니다. `evaluation.routing`을 통합 baseline이 아니라 단독으로 실행한 리포트에는 이 top-level 요구가 적용되지 않고, 앞 문단의 Routing 규칙(`null` + 사유)을 따릅니다.

timestamp가 포함된 실행 결과 전체는 기본적으로 Git에서 제외해야 합니다. 대신 검토된 최초 기준선 요약 하나를 고정 경로에 커밋해야 합니다.

권장 경로:

```text
evaluation/baselines/m2_initial.json
evaluation/baselines/m2_initial.md
evaluation/reports/                 # gitignored
```

기준선 파일은 실제 측정 결과여야 하며 임의의 목표값으로 채워서는 안 됩니다.

### M2-REQ-011 — 메트릭 정확성 테스트

Recall, MRR, nDCG, precision/recall/F1, percentile 계산은 작은 고정 입력과 손으로 계산 가능한 기대값을 사용한 단위 테스트를 가져야 합니다.

필수 경계 테스트:

- 빈 검색 결과
- relevant source가 여러 개인 경우
- 중복 source: Recall/MRR/nDCG는 모두 동일하게 정규화되고 최초 등장 순서로 중복 제거된 source ID
  목록을 입력으로 사용해야 하며(청크 단위 원본 리스트를 지표마다 다르게 자르면 안 됨), "k"는 세
  지표 모두에서 "top-k 고유 source"를 의미해야 합니다
- relevance grade가 없는 경우
- 모든 예측이 한 route인 경우
- 빈 latency 목록
- Unicode NFD/NFC 파일명 비교
- Windows와 POSIX 경로 비교

메트릭 단위 테스트는 모델, vectorstore, Ollama, 네트워크를 사용하면 안 됩니다.

### M2-REQ-012 — 기존 동작 회귀 방지

M2 구현 후 다음이 계속 통과해야 합니다.

```bash
pytest -q
npm test
git diff --check
```

기존 Web UI, `/rag`, `/health`, CLI 응답 스키마와 Agent 폴백 의미를 변경하면 안 됩니다. 평가를 위한 production 코드 변경에는 기존 동작이 유지됨을 검증하는 테스트를 추가해야 합니다.

### M2-REQ-013 — CI

GitHub Actions workflow는 Pull Request와 기본 브랜치 push에서 실행되어야 합니다.

필수 job:

1. **python-tests**
   - Python 3.11
   - Python 의존성 설치
   - `pytest -q`
   - dataset validation
   - 평가 메트릭 단위 테스트
2. **frontend-tests**
   - Node.js 20 이상
   - `npm ci`
   - `npm test`
   - vendor 동기화 후 `static/vendor/`에 diff가 없는지 확인

CI는 Ollama, DDGS 네트워크 호출, Hugging Face 대형 모델 다운로드, 로컬 vectorstore를 요구하면 안 됩니다. live 평가를 일반 PR의 필수 check로 만들면 안 됩니다.

workflow와 dependency action은 가능한 한 명시적 major version으로 고정해야 합니다. CI 실패 시 어느 검증이 실패했는지 job/step 이름으로 식별 가능해야 합니다.

### M2-REQ-014 — 문서화

README의 테스트 섹션에 다음을 추가해야 합니다.

- dataset validation
- 개별 평가 실행
- 통합 baseline 실행
- live 실행 전제조건
- 리포트 위치
- CI와 로컬 실행의 차이

평가 패키지에는 데이터셋 작성법, source ID 규칙, 메트릭 정의, 결과 해석의 한계를 설명하는 문서가 있어야 합니다. 구현 완료 시 [Roadmap.md](Roadmap.md)의 M2 상태와 [Problem.md](Problem.md)의 관련 항목을 갱신해야 합니다.

### M2-REQ-015 — 보안과 데이터 취급

- 평가 리포트에 문서 전체 내용, API token, 환경변수 값, 사용자 개인정보를 기록하면 안 됩니다.
- 질문이나 답변에 민감 정보가 포함될 수 있음을 고려해 상세 리포트는 기본적으로 Git에서 제외해야 합니다.
- 외부에서 받은 FAISS 인덱스를 평가 편의를 위해 로드하면 안 됩니다.
- live 웹 검색은 M2 baseline의 필수 입력으로 사용하지 않습니다. Routing은 도구 선택까지만 평가합니다.

### M2-REQ-016 — 오류 처리와 종료 코드

모든 CLI는 성공 시 0, 검증 실패·필수 artifact 누락·평가 실행 실패 시 0이 아닌 종료 코드를 반환해야 합니다. 오류 메시지는 최소한 다음 정보를 포함해야 합니다.

- 실패 단계
- 관련 사례 ID 또는 파일 경로
- 사용자가 취할 수 있는 다음 조치

개별 사례 실패가 전체 프로세스를 즉시 중단해야 하는지, 실패로 기록하고 계속할지는 명령별로 일관되게 문서화해야 합니다. 기본 정책은 dataset/schema 오류는 즉시 중단하고, 모델 추론 중 개별 사례 오류는 기록 후 나머지를 계속 평가하는 것입니다.

## 8. 비기능 요구사항

### M2-NFR-001 — 결정론

모델 호출이 없는 dataset/metric/report 테스트는 반복 실행 시 동일한 결과를 내야 합니다. 정렬 순서와 JSON 직렬화 형식을 안정적으로 유지해야 합니다.

### M2-NFR-002 — 비교 가능성

동일한 dataset과 설정으로 생성한 두 리포트는 집계 지표와 메타데이터를 프로그램으로 비교할 수 있어야 합니다. 가능하면 baseline 비교 명령 또는 함수도 제공합니다.

### M2-NFR-003 — 기존 코드와의 결합도

평가 모듈 import가 RAG 모델을 즉시 로드하면 안 됩니다. 모델과 vectorstore는 실제 live 명령을 실행할 때만 지연 초기화해야 합니다.

### M2-NFR-004 — 실행 시간

외부 의존성이 없는 전체 CI는 일반적인 GitHub-hosted runner에서 10분 이내를 목표로 합니다. live baseline은 이 제한의 적용 대상이 아니며 실제 소요 시간을 리포트에 기록합니다.

### M2-NFR-005 — 유지보수성

schema, metric, runner, reporting 역할을 분리하고 공개 함수에는 타입 힌트와 간결한 docstring을 제공합니다. 평가 코드에서 production private 속성에 직접 접근해야 한다면 그 이유와 안정성 경계를 문서화합니다.

## 9. M2 완료 수용 기준

M2는 다음 조건을 모두 충족할 때 완료로 간주합니다.

1. 최소 60개 사례의 골든 평가셋이 schema와 구성 검증을 통과합니다.
2. Retrieval, Routing, Answer 평가 명령과 통합 baseline 명령이 제공됩니다.
3. 필수 metric과 경계 조건의 단위 테스트가 통과합니다.
4. 실제 로컬 환경에서 최초 baseline JSON/Markdown이 생성되고 고정 경로에 저장됩니다.
5. 리포트에 요구된 실행 환경과 설정 메타데이터가 포함됩니다(corpus manifest·vectorstore fingerprint 포함, M2-REQ-010).
6. GitHub Actions에서 Python, dataset, frontend 검증이 외부 모델 없이 통과합니다.
7. 기존 Python/프런트엔드 테스트가 모두 통과합니다.
8. README, Roadmap, Problem 문서가 구현 상태와 일치합니다.
9. Retrieval 알고리즘이나 기존 API 의미가 의도치 않게 변경되지 않았습니다.
10. 리뷰에서 요구사항별 증거를 확인할 수 있습니다.

M2는 기준선 구축 마일스톤이므로 Recall이나 답변 점수가 특정 목표를 넘는 것을 완료 조건으로 삼지 않습니다. 측정된 값이 낮더라도 정확하게 기록되고 재현 가능하면 M2 목적을 충족합니다. 개선 목표와 회귀 허용치는 최초 기준선을 검토한 뒤 M3 계획에서 확정합니다.

## 10. 요구사항 추적표

구현 Pull Request 설명에는 다음 표를 복사해 증거를 채워야 합니다.

| 요구사항 | 상태 | 구현/테스트/리포트 증거 |
|---|---|---|
| M2-REQ-001~004 | 완료 | `evaluation/schema.py`, `evaluation/dataset.py`, `evaluation/datasets/golden.jsonl`, `test_evaluation_schema.py`, `test_evaluation_dataset.py`; dataset 76건 validation 통과 |
| M2-REQ-005~006 | 완료 | `rag_engine.py` 선택적 trace, `evaluation/retrieval.py`, `test_evaluation_retrieval.py`; [최초 기준선 Retrieval 결과](evaluation/baselines/m2_initial.md#retrieval-기준선) |
| M2-REQ-007 | 완료 | `evaluation/routing.py`, `test_evaluation_routing.py`, `test_agent_routing.py`; offline/live 분리 및 [Routing 기준선](evaluation/baselines/m2_initial.md#routing-기준선) |
| M2-REQ-008~010 | 완료 | `evaluation/answers.py`, `evaluation/baseline.py`, `evaluation/reporting.py`, 관련 테스트; 사용자 승인 [JSON](evaluation/baselines/m2_initial.json)/[Markdown](evaluation/baselines/m2_initial.md) |
| M2-REQ-011~012 | 완료 | `test_evaluation_metrics.py`, evaluator/reporting/baseline 회귀 테스트; 최종 `pytest -q` 349 passed, 1 skipped |
| M2-REQ-013 | 완료 | `.github/workflows/ci.yml`; PR #10 `python-tests`·`frontend-tests` 성공, 외부 live 서비스 미사용 |
| M2-REQ-014~016 | 완료 | [README](README.md#테스트-방법), [평가 가이드](evaluation/README.md), evaluator CLI 오류·종료 코드 테스트, [Roadmap](Roadmap.md), [Problem](Problem.md) |
| M2-NFR-001~005 | 완료 | 결정론적 metric/report 테스트, fingerprint 비교, lazy import/`--help` 테스트, CI 10분 이내, schema·metric·runner·reporting 역할 분리 |
