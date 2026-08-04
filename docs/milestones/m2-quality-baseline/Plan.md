# M2 Quality Baseline & CI 개발 계획

상태: **완료** — Phase 0~9 구현, CI 검증, 최초 live baseline 사용자 승인 및 문서화 완료

## 1. 목적

이 계획은 [Requirement.md](Requirement.md)의 요구사항을 구현하기 위한 작업 순서, 산출물, 검증 방법을 정의합니다.

M2의 목표는 현재 RAG 품질을 높이는 것이 아니라 현재 품질과 성능을 재현 가능하게 측정하는 것입니다. 검색 알고리즘 최적화와 모델 변경은 M2 기준선이 완성된 후 M3에서 진행합니다.

## 2. 확정 범위

- 골든 평가셋: 고유 사례 최소 60개
- 평가 영역: Retrieval, Agent Routing, 기본 Answer Quality
- 성능 영역: Retrieval 단계별 및 End-to-End latency
- 자동화: 외부 모델 없이 실행 가능한 GitHub Actions CI
- live 평가: 로컬 Ollama/vectorstore를 사용하는 명시적 opt-in
- 결과물: JSON, Markdown, 사람 검토 worksheet, 최초 M2 baseline

## 3. 주요 결정

### 단일 골든셋

Routing, Retrieval, Answer 평가가 동일한 질문 ID와 정답을 공유하도록 `golden.jsonl` 하나를 정답 원천으로 사용합니다. 각 evaluator는 필요한 필드가 있는 사례만 선택합니다.

### production 경로 재사용

평가를 위해 검색 알고리즘을 복제하지 않습니다. `RAGEngine`에 비파괴적인 trace/observer 경계를 추가하고 evaluator는 이를 호출합니다.

### CI와 live 평가 분리

일반 PR CI는 mock, schema, metric, frontend 테스트만 실행합니다. 모델 다운로드, vectorstore, Ollama가 필요한 baseline은 로컬 또는 별도 실행 환경에서 수행합니다.

### 최초 baseline은 품질 gate가 아님

M2에서는 현재 값을 정확히 기록합니다. 특정 Recall이나 답변 점수를 임의로 합격 기준으로 삼지 않습니다. M3에서 최초 baseline을 검토해 개선 목표와 회귀 허용치를 정합니다.

### 기본 Answer 평가는 혼합 방식

자동 평가는 핵심 사실 문자열 그룹, abstention, source 일치, intent와 latency를 측정합니다. Faithfulness 등 의미 기반 품질은 사람이 채울 worksheet로 보완합니다. LLM judge는 필수 범위가 아닙니다.

## 4. 작업 단계

### Phase 0 — 착수 전 기준 상태 고정

목표: 구현 전 현재 테스트와 실행 환경을 기록합니다.

작업:

1. 현재 브랜치와 dirty 상태 기록
2. `pytest -q`, `npm test`, `git diff --check` 실행
3. 현재 Python, Node, Ollama 모델 버전 기록
4. `data/` 문서 목록과 `vectorstore/` 존재 여부 확인
5. 기존 `test_agent_routing.py`의 16개 사례와 중복 정답 관리 방식 확인

산출물:

- 개발 PR 또는 작업 로그의 시작 상태

완료 조건:

- 기존 테스트 실패가 있으면 M2 변경과 구분해 기록함

관련 요구사항: M2-REQ-012

### Phase 1 — 평가 schema와 dataset validation

목표: 모델 없이도 검증 가능한 안정적인 데이터 계약을 만듭니다.

작업:

1. `evaluation/schema.py`에 case model과 enum 정의
2. source ID 정규화 함수 구현
3. JSONL loader와 명확한 validation error 구현
4. 데이터셋 규모·구성 규칙 validator 구현
5. dataset CLI와 종료 코드 구현
6. schema, 중복 ID, Unicode/경로 정규화 단위 테스트 작성

설계 주의:

- 새 대형 의존성이 꼭 필요하지 않으면 표준 라이브러리와 현재 의존성을 우선 사용
- validation은 모든 오류에 사례 ID 또는 line number 포함
- import 시 모델을 로드하지 않음

산출물:

- `evaluation/schema.py`
- `evaluation/dataset.py`
- schema/validation 테스트

검증:

```bash
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
pytest -q
```

관련 요구사항: M2-REQ-001~004, M2-REQ-011, M2-NFR-001, M2-NFR-003, M2-NFR-005

### Phase 2 — 골든 평가셋 작성

목표: 현재 로컬 문서와 실제 사용 패턴을 대표하는 최소 60개 평가 사례를 만듭니다.

작업:

1. `data/` 문서별 정규화 source ID 목록 생성
2. 기술 문서, 경제/정책 문서 등 현재 corpus 분포 파악
3. 문서 QA 40개 이상 작성
4. 웹 검색 10개 이상 작성
5. 경계/답변 불가 10개 이상 작성
6. Retrieval 관련 문서와 graded relevance 검토
7. 답변 핵심 사실 assertion과 abstention 기대값 검토
8. Intent와 tag 분포 검사
9. 데이터셋 validation 실행

작성 원칙:

- 질문과 정답은 사람이 원문을 확인해 작성
- 하나의 표현만 강제하지 않도록 `answer_assertions.any_of`에 동의 표현 사용
- 현재 문서로 답할 수 없는 사례는 corpus를 확인한 뒤 명시
- 원문 전체, 개인정보, 저작권상 불필요한 장문을 평가셋에 복사하지 않음
- 파일명 Unicode 정규화 차이를 source ID로 흡수

산출물:

- `evaluation/datasets/golden.jsonl`
- 평가셋 작성 가이드

완료 조건:

- M2-REQ-002의 규모와 분포를 validator가 자동 확인
- 최소 2회 사람 검토: source relevance 검토와 answer assertion 검토
- Claude Code는 골든셋 초안을 만든 뒤 사용자에게 검토를 요청하고, 사용자 승인 전에는 Phase 2를 완료 처리하지 않음

관련 요구사항: M2-REQ-002~004, M2-REQ-015

### Phase 3 — 공통 metric과 reporting 기반

목표: evaluator가 공유할 정확하고 결정론적인 계산·리포트 계층을 만듭니다.

작업:

1. Recall@K, MRR@10, nDCG@10 구현
2. binary precision/recall/F1과 confusion matrix 구현
3. mean, median, p95 latency 구현
4. answer assertion coverage helper 구현
5. Git/data/config/environment metadata 수집 구현
6. stable JSON과 Markdown reporter 구현
7. timestamped reports gitignore 설정
8. 손계산 fixture 기반 metric 테스트 작성

산출물:

- `evaluation/metrics.py`
- `evaluation/reporting.py`
- metric/reporting 테스트
- `evaluation/reports/` ignore 규칙

검증:

```bash
pytest -q
```

관련 요구사항: M2-REQ-010~011, M2-NFR-001~002, M2-NFR-005

### Phase 4 — Retrieval trace와 evaluator

목표: production retrieval 결과와 단계별 시간을 기준선으로 측정합니다.

작업:

1. 기존 `_retrieve_documents()` 동작 특성 테스트로 고정
2. `time.perf_counter()` 기반 선택적 retrieval trace 설계
3. BM25, Dense, RRF, MMR, reranker 후보 수와 latency 기록
4. 기존 호출자를 깨지 않는 호환 API 구현
5. Retrieval evaluator와 CLI 구현
6. 사례별 순위/latency와 집계 리포트 생성
7. 모델/vectorstore 누락 오류 처리
8. fake retriever를 사용한 evaluator 단위 테스트 작성

리뷰 포인트:

- 계측 전후 검색 문서 순서가 동일한가
- 평가 코드가 production 검색 로직을 복제하지 않았는가
- 계측을 끈 일반 요청에 불필요한 직렬화 비용이 없는가

산출물:

- production retrieval trace 경계
- `evaluation/retrieval.py`
- Retrieval evaluator 테스트

검증:

```bash
python -m evaluation.retrieval --help
pytest -q
```

관련 요구사항: M2-REQ-005~006, M2-REQ-012, M2-REQ-016

### Phase 5 — Routing evaluator와 기존 사례 통합

목표: Agent 도구 선택 품질과 오류 유형을 골든셋 기준으로 측정합니다.

작업:

1. offline evaluator와 고정 예측 fixture 구현
2. live evaluator에서 `_decide_tool()` 지연 호출
3. route별 precision/recall/F1, confusion matrix, latency 구현
4. no-tool/exception을 별도 오류로 기록
5. `RUN_LIVE_LLM_TESTS=1` opt-in 강제
6. 기존 `ROUTING_CASES`를 골든셋에서 파생하도록 변경
7. 기존 80% live 회귀 기준의 위치와 의미 유지

산출물:

- `evaluation/routing.py`
- routing evaluator 테스트
- 단일 정답 원천을 사용하는 live regression test

검증:

```bash
python -m evaluation.routing --help
pytest -q
RUN_LIVE_LLM_TESTS=1 python -m evaluation.routing \
  --dataset evaluation/datasets/golden.jsonl \
  --mode live \
  --output evaluation/reports/routing
```

관련 요구사항: M2-REQ-007, M2-REQ-011~012, M2-REQ-016

### Phase 6 — Answer evaluator와 사람 검토 worksheet

목표: 현재 문서 답변의 결정론적 기본 점수와 사람이 검토할 자료를 만듭니다.

작업:

1. live RAG query runner 구현
2. assertion coverage, abstention, source, intent 평가 구현
3. End-to-End latency 기록
4. 개별 오류 후 계속 진행하고 오류 상세 기록
5. JSON/Markdown 결과 생성
6. faithfulness/relevance/completeness/citation worksheet 생성
7. mock RAG 결과 기반 evaluator 단위 테스트 작성

주의:

- assertion coverage를 semantic correctness로 표현하지 않음
- 오류 답변 문자열과 abstention을 구분
- 상세 결과에 민감한 원문 context를 저장하지 않음

산출물:

- `evaluation/answers.py`
- Answer evaluator 테스트
- 사람 검토 worksheet

검증:

```bash
python -m evaluation.answers --help
pytest -q
```

관련 요구사항: M2-REQ-008, M2-REQ-010, M2-REQ-015~016

### Phase 7 — 통합 baseline과 최초 측정

목표: 하나의 명령으로 전체 평가를 실행하고 최초 기준선을 고정합니다.

작업:

1. 단계 orchestration과 실패 보존 정책 구현
2. `--limit`, `--tag`, skip 옵션 필요성 검토 및 구현
3. 실제 vectorstore와 Ollama로 전체 baseline 실행
4. 실패 사례와 사람 검토 worksheet 검토
5. 실행 환경과 설정 metadata 확인
6. 검토된 baseline JSON/Markdown을 고정 경로에 저장
7. M3에서 사용할 개선 후보와 수치 기록. M2 중에는 최적화하지 않음

승인 게이트:

- Claude Code는 최초 live baseline과 주요 실패 사례를 요약해 사용자에게 제시해야 함
- 사용자가 baseline을 검토·승인하기 전에는 고정 baseline을 최종 확정하거나 M2를 완료 처리하지 않음

산출물:

- `evaluation/baseline.py`
- `evaluation/baselines/m2_initial.json`
- `evaluation/baselines/m2_initial.md`
- timestamped 상세 report

검증:

```bash
RUN_LIVE_LLM_TESTS=1 python -m evaluation.baseline \
  --dataset evaluation/datasets/golden.jsonl \
  --output evaluation/reports
```

관련 요구사항: M2-REQ-009~010, M2-NFR-002

### Phase 8 — CI

목표: 외부 서비스 없는 회귀 검증을 모든 Pull Request에서 자동 실행합니다.

작업:

1. Python 3.11 job 작성
2. 기존 테스트와 dataset validation 실행
3. Node 20+ job 작성
4. `npm ci`, `npm test` 실행
5. vendor 동기화 후 diff 검사
6. 캐시 사용 시 key에 lock/requirements hash 포함
7. job/step 이름과 실패 메시지 정리
8. live 평가가 PR 필수 job에 포함되지 않았는지 확인

산출물:

- `.github/workflows/ci.yml`

검증:

- Pull Request에서 두 필수 job 성공
- Ollama와 vectorstore가 없는 runner에서 성공

관련 요구사항: M2-REQ-004, M2-REQ-011~013, M2-NFR-004

### Phase 9 — 문서 및 마일스톤 종료

목표: 사용자와 다음 개발자가 평가를 재현하고 M3 계획에 활용할 수 있게 합니다.

작업:

1. README에 평가 명령과 전제조건 추가
2. 평가셋 작성 가이드와 metric 정의 작성
3. baseline 결과 해석과 한계 기록
4. Roadmap M2를 완료로 갱신
5. Problem의 품질 기준선/CI 항목 제거 또는 남은 범위로 수정
6. 요구사항 추적표에 구현·테스트·리포트 증거 기록
7. 전체 검증 실행

최종 검증:

```bash
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
pytest -q
npm ci
npm test
git diff --check
```

live 검증:

```bash
RUN_LIVE_LLM_TESTS=1 python -m evaluation.baseline \
  --dataset evaluation/datasets/golden.jsonl \
  --output evaluation/reports
```

관련 요구사항: M2-REQ-009~016

## 5. 권장 구현 순서와 커밋 단위

가능하면 다음처럼 독립 검토 가능한 커밋으로 나눕니다.

1. `evaluation schema 및 dataset validator 추가`
2. `M2 golden dataset 추가`
3. `평가 metric 및 report 생성기 추가`
4. `RAG retrieval trace와 retrieval evaluator 추가`
5. `routing evaluator 및 기존 사례 통합`
6. `answer evaluator와 review worksheet 추가`
7. `통합 baseline과 최초 결과 추가`
8. `Python 및 frontend CI 추가`
9. `M2 사용법과 완료 상태 문서화`

각 커밋은 관련 단위 테스트를 포함해야 하며, production 동작 변경과 대규모 데이터 변경을 가능하면 분리합니다.

## 6. 위험과 대응

### 골든셋 품질이 평가 품질을 제한

대응:

- 원문 기반 사람 검토
- source relevance와 answer assertion을 분리 검토
- 사례별 notes와 tags로 불확실한 정답 표시

### Unicode 파일명 불일치

대응:

- NFC, basename, 경로 구분자, 대소문자 정규화 테스트
- source ID 규칙을 단일 함수로 관리

### live 평가의 비용과 변동성

대응:

- 명시적 opt-in
- 모델과 설정 metadata 기록
- CI와 분리
- 개별 실패를 보존하고 전체 분포로 해석

### 평가를 위해 production 로직이 갈라질 위험

대응:

- production retrieval 경로 재사용
- trace는 선택적 계측으로 구현
- 계측 전후 검색 결과 동일성 테스트

### 문자열 기반 Answer 평가의 한계

대응:

- `any_of` 동의 표현
- 지표 이름을 assertion coverage로 제한
- 사람 검토 worksheet 병행
- LLM judge는 후속 선택 사항으로 분리

### CI에서 대형 모델을 다운로드할 위험

대응:

- import-time lazy loading 검증
- evaluator unit test는 fake/mock 사용
- live 명령만 모델 초기화

## 7. 완료 정의

다음이 모두 충족되어야 M2를 완료 처리합니다.

- 요구사항 문서의 M2 완료 수용 기준 10개 충족
- 골든셋 validation 통과
- 최초 live baseline 생성 및 검토 완료
- 외부 의존성 없는 CI 통과
- 기존 테스트와 신규 evaluator 테스트 통과
- 요구사항 추적표에 증거 링크 작성
- 문서가 실제 명령과 결과에 맞게 갱신됨

M2 완료 후 바로 검색 코드를 수정하지 않고 baseline 리뷰를 통해 M3의 우선순위와 수치 목표를 먼저 합의합니다.

## 8. Claude Code 작업 지침

Claude Code에 이 계획을 전달할 때 다음 원칙을 함께 적용합니다.

1. 먼저 요구사항 문서 전체를 읽고 ID별 체크리스트를 만듭니다.
2. 현재 코드와 테스트를 실행해 Phase 0 상태를 기록합니다.
3. Phase 순서를 따르되 독립적으로 검증 가능한 작은 변경으로 진행합니다.
4. 기존 RAG 검색 결과나 API 의미를 바꾸는 최적화를 M2에 섞지 않습니다.
5. `data/`, `vectorstore/`, timestamped 상세 report를 커밋하지 않습니다.
6. 불명확한 정답을 임의로 만들지 말고 원문 확인 또는 사용자 결정을 요청합니다.
7. 각 Phase 종료 시 관련 요구사항, 변경 파일, 실행한 테스트와 남은 위험을 보고합니다.
8. 최종 보고에는 요구사항 추적표와 실제 baseline 결과를 포함합니다.
9. Phase 2 골든셋 정답과 Phase 7 최초 baseline은 사용자 승인 게이트를 통과해야 하며, 에이전트가 스스로 승인한 것으로 간주하지 않습니다.
