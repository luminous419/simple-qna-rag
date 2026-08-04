# M2 Phase 4·5·6 병렬 개발 상세 계획 및 규칙

## 1. 문서 목적

이 문서는 Claude Code가 M2 Phase 4, Phase 5, Phase 6을 병렬로 구현할 때 따라야 할 실행 지침이다.

상위 기준 문서는 다음과 같다.

1. `Development_M2_Quality_Baseline_Requirement.md`
2. `Development_M2_Quality_Baseline_Development_Plan.md`
3. `Development_M2_Quality_Baseline_Design.md`
4. `M2_Phase3_code_review_result.md`

기준이 충돌하면 **Requirement → Development Plan → Design → 이 문서** 순서로 우선한다. 구현 중 계약의 모순이나 공통 모듈 변경 필요성이 발견되면 임의로 넓게 수정하지 말고 통합 담당자에게 보고한다.

이번 작업 범위는 Phase 4·5·6 구현과 병합 검증까지다. **Phase 7 통합 baseline, 실제 최초 baseline 확정, 사용자 승인, Phase 8 CI, Phase 9 최종 문서화는 수행하지 않는다.**

## 2. 실행 전략

Phase 4·5·6은 별도 작업 브랜치 또는 Git worktree에서 병렬 구현한다.

```text
현재 Phase 3 승인 상태
        │
        ├── Phase 4: Retrieval trace/evaluator
        ├── Phase 5: Routing evaluator/case migration
        └── Phase 6: Answer evaluator/worksheet
                    │
             각 Phase 자체 검증
                    │
       Phase 4 → Phase 5 → Phase 6 순서로 병합
                    │
              전체 통합 검증
                    │
            Phase 7 시작 전 코드 리뷰
```

병렬 작업을 하나의 공유 working tree에서 동시에 편집하지 않는다. 별도 worktree를 사용할 수 없다면 Phase별 구현을 순차 적용하되, 아래 파일 소유권과 검증 게이트는 그대로 유지한다.

### 2.1 병렬 작업 시작 전 기준점 확인

세 작업은 동일한 Phase 3 승인 코드에서 출발해야 한다. 시작 전에 다음을 확인한다.

```bash
git status --short
git rev-parse --abbrev-ref HEAD
git log -1 --oneline
```

- 승인된 Phase 1~3 구현과 데이터셋·문서가 기준 commit에 포함돼 있어야 한다.
- untracked 또는 uncommitted Phase 2~3 산출물이 남아 있으면 새 worktree가 이를 자동으로 상속하지 못한다.
- 이런 상태에서는 Claude가 임의로 commit, stash, reset 또는 변경 폐기를 하지 않는다.
- 먼저 사용자에게 기준 commit 생성 여부와 포함 범위를 확인한다.
- 기준 commit이 준비된 후 세 Phase 브랜치를 같은 commit에서 만든다.
- 각 Phase 작업 시작 시 기준 commit SHA를 완료 보고에 기록한다.

사용자가 commit을 원하지 않으면 병렬 worktree 방식을 사용하지 않고 현재 working tree에서 Phase 4 → 5 → 6을 순차 구현한다.

## 3. 공통 개발 규칙

### 3.1 공통 모듈은 동결한다

다음 파일은 Phase 3까지 승인된 공통 계약이므로 병렬 작업자가 수정하지 않는다.

- `evaluation/schema.py`
- `evaluation/dataset.py`
- `evaluation/metrics.py`
- `evaluation/reporting.py`
- `evaluation/__init__.py`
- `config.py`
- `.gitignore`
- `requirements.txt`
- `Development_M2_Quality_Baseline_Requirement.md`
- `Development_M2_Quality_Baseline_Development_Plan.md`
- `Development_M2_Quality_Baseline_Design.md`
- `evaluation/README.md`

공통 모듈에 반드시 필요한 결함이 발견되면 다음 형식으로 보고하고 해당 Phase 변경과 분리한다.

```text
공통 변경 요청
- 대상 파일/함수:
- 필요한 이유:
- 기존 계약에 미치는 영향:
- 대안:
- 추가해야 할 회귀 테스트:
```

통합 담당자가 승인하기 전에는 공통 모듈을 변경하지 않는다.

### 3.2 승인된 API를 재사용한다

각 evaluator는 다음 API를 복제하거나 변형하지 않고 직접 import해 사용한다.

- `evaluation.dataset.load_jsonl`
- `evaluation.schema.normalize_source_id`
- `evaluation.schema.is_answer_eval_eligible`
- `evaluation.metrics.dedupe_preserve_order`
- `evaluation.metrics.normalize_relevance_grades`
- `evaluation.metrics.recall_at_k`
- `evaluation.metrics.mrr_at_k`
- `evaluation.metrics.ndcg_at_k`
- `evaluation.metrics.precision_recall_f1`
- `evaluation.metrics.percentile`
- `evaluation.metrics.mean_median`
- `evaluation.metrics.assertion_coverage`
- `evaluation.reporting.build_metadata`
- `evaluation.reporting.write_report`
- `evaluation.reporting.build_reproducibility_metadata`
- `evaluation.reporting.build_not_applicable_reproducibility_metadata`

source 정규화, 중복 제거, 지표 계산, metadata 생성을 evaluator 안에 다시 구현하지 않는다.

### 3.3 지연 로딩과 테스트 격리

- evaluator 모듈 import만으로 RAG 모델, embedding, reranker, Ollama, vectorstore를 로드하면 안 된다.
- `get_rag_engine()`과 `agent._decide_tool`은 실제 live 실행 시점에만 import하거나 호출한다.
- 일반 단위 테스트는 실제 `data/`, `vectorstore/`, Ollama, Hugging Face 다운로드, DDGS 및 네트워크를 사용하지 않는다.
- fake engine, fake retriever, monkeypatch 또는 callable 주입을 사용한다.
- live 실행은 `RUN_LIVE_LLM_TESTS=1`이 명시된 경우만 허용한다.
- 로컬 모델이 없다는 이유로 오프라인 테스트를 skip하면 안 된다.

### 3.4 오류 처리

- dataset/schema 오류는 즉시 중단한다.
- 평가 중 개별 사례의 모델 호출 오류는 사례 ID와 원인을 기록하고 나머지 사례를 계속 평가한다.
- CLI 성공은 exit 0, 잘못된 옵션·필수 artifact 누락·실행 실패는 non-zero를 반환한다.
- CLI 오류에는 실패 단계, 파일 경로 또는 사례 ID, 사용자가 취할 다음 조치가 포함돼야 한다.
- vectorstore 누락 안내에는 `document_register.py` 실행 방법을 포함한다.
- 예외를 성공 결과 또는 점수 0으로 조용히 바꾸지 않는다.

### 3.5 리포트와 보안

- 각 evaluator는 timestamp가 붙은 JSON과 Markdown을 `evaluation/reports/` 아래에 생성한다.
- `write_report()`를 사용하고 자체 파일명 충돌 처리기를 만들지 않는다.
- Retrieval과 Answer 리포트의 `corpus_manifest`, `corpus_manifest_sha256`, `vectorstore_fingerprint`는 non-null이어야 한다.
- Routing 리포트는 위 세 필드를 null로 두고 `reproducibility_note`에 corpus/vectorstore를 사용하지 않는 이유를 기록한다.
- 리포트와 worksheet에 검색 원문 chunk 전체를 저장하지 않는다.
- API token, 환경변수 값, 개인정보를 기록하지 않는다.
- Answer worksheet에는 모델 답변, source 파일명·페이지, assertion 자동 점수만 저장한다. `sources[].content`는 저장하지 않는다.

### 3.6 작업 범위 통제

- 검색 알고리즘, prompt, 모델 설정, 라우팅 정책을 개선하거나 변경하지 않는다.
- baseline 점수를 높이기 위한 production 동작 변경은 금지한다.
- 기존 공개 API의 반환 형식과 호출 방식을 깨뜨리지 않는다.
- unrelated refactor, formatting-only 대량 변경, 문서 전면 개편을 하지 않는다.
- 기존 사용자 변경을 되돌리거나 덮어쓰지 않는다.

## 4. 파일 소유권

| 영역 | 단독 소유 파일 | 읽기 전용 의존 파일 |
|---|---|---|
| Phase 4 | `rag_engine.py`, `evaluation/retrieval.py`, `test_evaluation_retrieval.py` | schema, dataset, metrics, reporting, config |
| Phase 5 | `evaluation/routing.py`, `test_evaluation_routing.py`, `test_agent_routing.py`, `evaluation/datasets/golden.jsonl` | schema, dataset, metrics, reporting, agent |
| Phase 6 | `evaluation/answers.py`, `test_evaluation_answers.py` | schema, dataset, metrics, reporting, rag_engine, prompt_templates |
| 통합 담당자 | 충돌 해결 및 최종 검증만 수행 | 모든 Phase 산출물 |

Phase 작업자가 다른 Phase의 단독 소유 파일을 수정하지 않는다.

특히 다음 충돌 지점을 지킨다.

- `rag_engine.py`는 Phase 4만 수정한다.
- `golden.jsonl`과 `test_agent_routing.py`는 Phase 5만 수정한다.
- Phase 6은 병렬 작업 중 Phase 4의 `rag_engine.py` 변경을 요구하거나 가져오지 않는다. 기존 `query()` 반환 계약만 사용한다.
- 공통 문서와 `evaluation/README.md` 갱신은 Phase 9로 미룬다.

## 5. Phase 4 — Retrieval trace와 evaluator

### 5.1 목표

production 검색 로직을 복사하지 않고 `RAGEngine._retrieve_documents()`에 완전 opt-in 계측을 추가하고, 동일 검색 경로를 사용하는 Retrieval evaluator를 구현한다.

### 5.2 구현 파일

- 수정: `rag_engine.py`
- 신규: `evaluation/retrieval.py`
- 신규: `test_evaluation_retrieval.py`

필요하면 Phase 4 전용 characterization test 파일을 하나 더 만들 수 있지만, 동일 테스트를 여러 파일로 불필요하게 분산하지 않는다.

### 5.3 Retrieval trace 계약

`rag_engine.py`에 다음 타입을 추가한다.

```python
@dataclass
class RetrievalStageTrace:
    name: str
    latency_ms: float
    candidate_count: int

@dataclass
class RetrievalTrace:
    stages: list[RetrievalStageTrace] = field(default_factory=list)
```

stage 이름은 다음 값만 사용한다.

- `bm25`
- `dense`
- `rrf`
- `mmr`
- `reranker`
- `total`

latency는 `time.perf_counter()`로 측정하고 millisecond로 저장한다. `candidate_count`는 해당 stage가 반환한 문서 수다.

`_retrieve_documents()` 시그니처는 다음과 같이 확장한다.

```python
def _retrieve_documents(
    self,
    question: str,
    trace: RetrievalTrace | None = None,
):
```

필수 불변식:

- 기존 `self._retrieve_documents(question)` 호출은 수정 없이 동작한다.
- `trace=None`이면 `RetrievalStageTrace` 객체나 stage list를 생성하지 않는다.
- 계측 유무에 따라 검색 문서의 내용, 객체, 개수와 순서가 달라지면 안 된다.
- 검색 로직은 단일 메서드에 하나만 존재해야 하며 traced/untraced 복제 구현을 만들지 않는다.
- `total` stage는 trace 사용 시 마지막에 정확히 한 번 기록한다.
- 활성화되지 않은 stage를 리포트에 가짜 0ms 값으로 추가하지 않는다.

현재 네 검색 분기를 그대로 보존한다.

1. Hybrid → RRF → 선택적 MMR → 선택적 Reranker
2. MMR-only → 선택적 Reranker
3. Reranker-only
4. Plain similarity

### 5.4 Retrieval evaluator 계약

`evaluation/retrieval.py`는 다음 공개 함수를 제공한다.

```python
def evaluate_retrieval(
    dataset_path: Path,
    output_dir: Path,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    limit: int | None = None,
    tag: str | None = None,
) -> dict:
    ...

def main(argv: list[str] | None = None) -> int:
    ...
```

CLI:

```bash
python -m evaluation.retrieval \
  --dataset evaluation/datasets/golden.jsonl \
  --output evaluation/reports/retrieval \
  [--limit N] [--tag TAG]
```

처리 순서:

1. dataset을 로드·검증한다.
2. tag와 limit을 결정론적인 원본 순서로 적용한다.
3. `relevant_sources`가 있는 사례를 Retrieval 대상으로 삼는다.
4. `get_rag_engine()`은 함수 내부에서 지연 호출한다.
5. 사례마다 새 `RetrievalTrace()`를 만들고 production `_retrieve_documents()`를 호출한다.
6. `docs[].metadata["source"]`를 `normalize_source_id()`로 정규화한다.
7. 정규화된 source 목록에 `dedupe_preserve_order()`를 정확히 한 번 적용한다.
8. relevant source도 같은 정규화 규칙을 적용한다.
9. `relevance_grades`는 `normalize_relevance_grades()`를 거친 뒤 nDCG에 전달한다.
10. 동일한 dedupe된 `ranked_ids`를 Recall/MRR/nDCG 모두에 전달한다.
11. 사례별 순위, metric, trace, latency, 성공/실패를 보존한다.
12. 집계 결과에 metadata와 reproducibility metadata를 결합하고 JSON/Markdown을 쓴다.

필수 집계:

- Recall@1, @3, @5, @10
- MRR@10
- nDCG@10
- 평균, median, p95 retrieval latency
- total/success/failure/excluded 사례 수
- Recall/MRR 제외 수
- nDCG 제외 수
- stage별 latency와 후보 수 요약

제외 규칙:

- `relevant_sources`가 비어 있으면 Recall/MRR 대상에서 제외한다.
- `relevance_grades`가 비어 있으면 nDCG 대상에서 제외한다.
- 두 제외 수는 별도로 기록한다.
- 개별 검색 실패는 failure로 기록하고 전체 실행을 계속한다.

재현성 필드는 실제 `DATA_DIR`/`VECTORSTORE_PATH`를 대상으로 `build_reproducibility_metadata()`를 호출해 채운다. `vectorstore_document_count`는 engine이 이미 초기화된 경우에만 opportunistic하게 기록하고 이를 위해 별도 모델을 로드하지 않는다.

### 5.5 Phase 4 필수 테스트

- 네 검색 분기 각각에서 `trace=None`과 `trace=RetrievalTrace()` 결과가 순서까지 동일하다.
- trace가 없을 때 stage 객체가 생성되지 않는다.
- 각 분기에서 활성 stage 이름, 순서, candidate count가 정확하다.
- `total`이 마지막에 한 번만 존재한다.
- 중복 source가 `A,A,A,B`로 반환돼도 모든 metric이 `[A,B]` 순위를 사용한다.
- relevant source 다수, 빈 검색 결과, grade 없는 사례와 실패 사례를 검증한다.
- tag/limit 필터가 결정론적으로 동작한다.
- evaluator 테스트는 fake engine과 실제와 같은 metadata 구조를 가진 fake documents를 사용한다.
- 리포트의 corpus manifest와 vectorstore fingerprint가 non-null이다. 단위 테스트에서는 reporting helper를 mock할 수 있다.
- `--help`는 모델이나 vectorstore를 로드하지 않고 exit 0이다.
- 필수 artifact 누락 CLI는 원인과 다음 조치를 출력하고 non-zero다.

### 5.6 Phase 4 완료 명령

```bash
python -m evaluation.retrieval --help
pytest -q test_evaluation_retrieval.py
pytest -q
git diff --check
```

실제 live Retrieval 실행은 선택 사항이며 CI 완료 조건이 아니다. 실행하지 않았다면 완료 보고서에 명시한다.

권장 커밋: `RAG retrieval trace와 retrieval evaluator 추가`

## 6. Phase 5 — Routing evaluator와 기존 회귀 사례 통합

### 6.1 목표

offline/live 공통 Routing evaluator를 만들고 기존 `ROUTING_CASES` 16건을 골든셋으로 이관해 정답 원천을 하나로 통합한다.

### 6.2 구현 파일

- 신규: `evaluation/routing.py`
- 신규: `test_evaluation_routing.py`
- 수정: `test_agent_routing.py`
- 수정: `evaluation/datasets/golden.jsonl`

`agent.py`의 라우팅 정책과 `_decide_tool()` 구현은 수정하지 않는다.

### 6.3 Routing evaluator 계약

```python
def evaluate_routing(
    cases: list[GoldenCase],
    decide_tool: Callable[[str], tuple[str | None, str | None]],
    *,
    measure_latency: bool = True,
) -> dict:
    ...

def main(argv: list[str] | None = None) -> int:
    ...
```

CLI:

```bash
python -m evaluation.routing \
  --dataset evaluation/datasets/golden.jsonl \
  --output evaluation/reports/routing \
  --mode {offline,live} \
  [--limit N] [--tag TAG]
```

필수 동작:

- offline/live가 동일한 `evaluate_routing()` 집계 코어를 사용한다.
- live만 `agent._decide_tool()`을 지연 import해 사용한다.
- live는 `RUN_LIVE_LLM_TESTS=1` 없이 실행하지 않고 사용법과 opt-in 방법을 안내하며 non-zero로 종료한다.
- offline은 data, vectorstore, Ollama와 네트워크가 없어도 동작한다.
- `decide_tool()` 예외와 `None`/알 수 없는 route를 별도 실패 유형으로 기록한다.
- 개별 사례 실패 후 다음 사례를 계속 평가한다.
- 질문, 사례 ID, 기대 route, 실제 route, 오류를 실패 상세에 보존한다.

필수 집계:

- 전체 accuracy
- `document_qa` precision/recall/F1
- `web_search` precision/recall/F1
- confusion matrix
- no-tool 선택 건수
- exception 건수
- 평균, median, p95 routing latency
- total/success/failure/excluded 사례 수
- 실패 사례 목록

Routing은 corpus/vectorstore를 사용하지 않는다.

```python
build_not_applicable_reproducibility_metadata(
    "routing은 corpus/vectorstore를 사용하지 않음"
)
```

을 사용하며 data/vectorstore 존재 여부를 확인하지 않는다.

### 6.4 기존 16개 사례 이관 규칙

`test_agent_routing.py`의 질문과 기대 route 16쌍을 **문구와 의미를 바꾸지 않고** `golden.jsonl`로 이관한다.

- 모든 이관 사례에 `routing_regression` tag를 추가한다.
- `routing_regression` 부분집합은 정확히 16건이어야 한다.
- 이미 골든셋에 질문이 완전히 같은 사례가 있으면 새 사례를 중복 추가하지 않고 해당 기존 사례에 tag를 추가한다.
- 비슷하지만 문구가 다른 사례는 기존 16개 질문 보존을 위해 별도 사례로 추가한다.
- ID는 기존 ID와 충돌하지 않는 안정적인 값을 사용한다.
- 기대 route에 따라 category를 `document_qa` 또는 `web_search`로 정한다.
- routing 전용 사례에 근거 없는 answer assertion이나 relevant source를 만들지 않는다.
- dataset 최소 구성과 validator를 계속 통과해야 한다.

이관 후 `test_agent_routing.py`의 하드코딩 `ROUTING_CASES`를 삭제한다. 다음 방식으로 골든셋을 읽는다.

```python
def _load_routing_regression_cases() -> list[GoldenCase]:
    cases = load_jsonl(Path("evaluation/datasets/golden.jsonl"))
    selected = [case for case in cases if "routing_regression" in case.tags]
    assert len(selected) == 16
    return selected
```

live 정확도 최소 기준 `MIN_ACCURACY = 0.8`은 유지한다.

### 6.5 Phase 5 필수 테스트

- offline perfect prediction과 일부 오분류의 accuracy/PR/F1/confusion matrix가 정확하다.
- 모든 예측이 한 route인 경우를 검증한다.
- no-tool과 exception이 구분되고 나머지 사례 평가가 계속된다.
- 빈 입력 또는 필터 결과 0건을 성공 accuracy로 처리하지 않는다. 명시적 오류 또는 제외 사유가 있어야 한다.
- latency 비활성 시 측정값 없음이 0ms와 구분된다.
- Routing metadata의 corpus/vectorstore 필드는 null이고 사유가 non-empty다.
- offline 테스트는 `data/`, `vectorstore/`, Ollama와 네트워크 없이 통과한다.
- live opt-in이 없으면 모델 import/호출 전에 non-zero로 종료한다.
- `routing_regression` 태그가 정확히 16건이며 원래 질문·기대 route 쌍과 1:1로 일치한다.
- `test_agent_routing.py`에 중복 정답 리스트가 남아 있지 않다.
- `--help`는 agent/모델을 로드하지 않고 exit 0이다.

### 6.6 Phase 5 완료 명령

```bash
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
python -m evaluation.routing --help
pytest -q test_evaluation_routing.py
pytest -q
git diff --check
```

로컬 Ollama가 준비된 경우에만 다음을 선택 실행한다.

```bash
RUN_LIVE_LLM_TESTS=1 python -m evaluation.routing \
  --dataset evaluation/datasets/golden.jsonl \
  --mode live \
  --tag routing_regression \
  --output evaluation/reports/routing

RUN_LIVE_LLM_TESTS=1 pytest -q test_agent_routing.py -v
```

live 실행을 하지 않았거나 모델 환경 문제로 실패했다면 오프라인 구현 실패로 숨기지 말고 완료 보고에 구분해 기록한다.

권장 커밋: `routing evaluator 및 기존 사례 통합`

## 7. Phase 6 — Answer evaluator와 사람 검토 worksheet

### 7.1 목표

실제 `RAGEngine.query()` 경로를 사용하는 Answer evaluator를 만들고 assertion, abstention, source, intent 및 latency를 자동 평가하며 사람이 검토할 안전한 Markdown worksheet를 생성한다.

### 7.2 구현 파일

- 신규: `evaluation/answers.py`
- 신규: `test_evaluation_answers.py`

`rag_engine.py`, `prompt_templates.py`, 골든셋과 공통 평가 모듈은 수정하지 않는다.

### 7.3 Answer 대상 및 호출 계약

대상 판정은 category가 아니라 다음 공개 함수만 사용한다.

```python
from evaluation.schema import is_answer_eval_eligible
```

- assertion이 하나 이상인 사례를 포함한다.
- `expect_abstention=True`인 unanswerable 사례를 포함한다.
- 둘 다 없는 retrieval 전용 document QA 사례는 제외한다.
- evaluator 안에 eligibility 함수를 다시 정의하지 않는다.

`get_rag_engine()`은 `evaluate_answers()` 내부에서 지연 호출한다. 실제 반환 형식은 다음과 같다.

```python
{
    "answer": str,
    "sources": [
        {
            "index": int,
            "source": str,
            "page": int | None,
            "content": str,
        }
    ],
    "success": bool,
    "intent": str,
}
```

evaluator는 `sources[].content`를 결과나 worksheet에 복사하지 않는다.

### 7.4 Answer evaluator 계약

`evaluation/answers.py`는 최소한 다음을 제공한다.

```python
ABSTENTION_PHRASES = (
    "제공된 문서에서 관련 정보를 찾을 수 없습니다",
    "제공된 문서만으로는 확실한 답변이 어렵습니다",
)

def evaluate_answers(
    dataset_path: Path,
    output_dir: Path,
    limit: int | None = None,
    tag: str | None = None,
) -> dict:
    ...

def _detect_abstention(answer: str) -> bool:
    ...

def _extract_returned_source_ids(sources: list[dict]) -> tuple[list[str], int]:
    ...

def _source_match(
    returned_sources: list[str],
    relevant_sources: list[str],
) -> dict | None:
    ...

def _abstention_confusion(flags: list[tuple[bool, bool]]) -> dict:
    ...

def _fence_for(text: str) -> str:
    ...

def write_review_worksheet(results: list[dict], output_path: Path) -> Path:
    ...

def main(argv: list[str] | None = None) -> int:
    ...
```

CLI:

```bash
python -m evaluation.answers \
  --dataset evaluation/datasets/golden.jsonl \
  --output evaluation/reports/answers \
  [--limit N] [--tag TAG]
```

### 7.5 자동 평가 규칙

#### Assertion

- `assertion_coverage()`를 직접 재사용한다.
- 사례별 passed/total과 전체 assertion 통과율을 기록한다.
- assertion coverage가 진실성을 완전히 보장하지 않는다는 한계를 리포트에 명시한다.

#### Abstention

- 두 공식 prompt 문구를 모두 NFC 정규화 후 인식한다.
- 대소문자 변경이 의미 있는 한국어 문구는 원문 포함 여부로 판정한다.
- Answer 평가 대상 전체에서 TP/TN/FP/FN을 계산한다.
- accuracy는 `(TP + TN) / N`이다.
- 대상이 0건이면 accuracy는 `None`이고 제외 사유를 기록한다.

#### Source

- `sources`는 문자열 리스트가 아니라 dict 리스트로 처리한다.
- `source`가 없거나 non-string/empty인 항목은 건너뛰고 skipped count를 기록한다.
- 레거시 문자열 entry를 임의로 지원하지 않는다.
- 양쪽 source는 `normalize_source_id()` 후 `dedupe_preserve_order()`를 사용한다.
- relevant source가 없으면 source 평가는 `None`이고 제외 수에 포함한다.
- `source_any_hit = bool(returned ∩ relevant)`
- `source_recall = len(returned ∩ relevant) / len(relevant)`

#### Intent

- `expected_intent`가 있는 성공 사례를 대상으로 actual `result["intent"]`와 비교한다.
- 정확도, 대상 수, 제외 수를 기록한다.

#### Latency와 실패

- 사례별 End-to-End latency를 `time.perf_counter()`로 측정한다.
- 평균, median, p95를 기록한다.
- query가 `success=False`를 반환하거나 예외가 발생하면 사례 실패로 기록하고 계속한다.
- 실패 답변 문자열을 정상 assertion/abstention 결과로 채점하지 않는다.

#### Reproducibility

- `build_reproducibility_metadata(DATA_DIR, VECTORSTORE_PATH)`를 사용한다.
- corpus manifest와 vectorstore fingerprint는 non-null이어야 한다.
- 단위 테스트에서는 engine과 reporting helper를 함께 mock한다.

### 7.6 Worksheet 계약

Markdown 표 한 셀에 전체 답변을 넣지 않는다. 각 사례를 별도 section으로 렌더링한다.

각 section에는 다음 항목이 있어야 한다.

- 사례 ID
- 질문
- 자동 점수
- 반환 source 파일명과 페이지
- 기대 assertion 요약
- 모델 답변 원문
- Faithfulness 1~5 빈칸
- Relevance 1~5 빈칸
- Completeness 1~5 빈칸
- Citation correctness 1~5 빈칸
- Reviewer note 빈칸

답변 fence는 답변에 포함된 가장 긴 연속 backtick보다 하나 길고 최소 3개여야 한다.

```python
fence_length = max(longest_backtick_run + 1, 3)
```

질문·source·assertion을 Markdown에 넣을 때도 구조를 깨뜨릴 수 있는 줄바꿈과 특수문자를 안전하게 렌더링한다. 검색 chunk 본문은 포함하지 않는다.

### 7.7 Phase 6 필수 테스트

- evaluator가 공통 `is_answer_eval_eligible()`을 사용하며 category로 대상을 제한하지 않는다.
- assertion 사례와 unanswerable abstention 사례가 모두 대상에 포함된다.
- retrieval 전용 사례는 제외된다.
- 두 공식 abstention 문구가 각각 인식된다.
- TP/TN/FP/FN이 모두 올바르게 집계되고 빈 flags는 accuracy `None`과 사유를 반환한다.
- 실제와 같은 source dict, 누락 key, non-string 및 empty source를 검증한다.
- source 부분/완전/불일치와 relevant source 없음 제외를 검증한다.
- intent 일치/불일치/제외를 검증한다.
- query 실패가 assertion 또는 abstention 정상 결과에 포함되지 않고 다음 사례가 계속된다.
- 답변에 3개, 4개 이상의 backtick, pipe와 여러 줄이 포함돼도 worksheet 구조가 유지된다.
- worksheet에 `sources[].content`가 노출되지 않는다.
- 리포트 재현성 필드가 non-null이다.
- `--help`는 모델이나 vectorstore를 로드하지 않고 exit 0이다.
- 필수 artifact 누락은 원인과 다음 조치를 포함하고 non-zero다.

### 7.8 Phase 6 완료 명령

```bash
python -m evaluation.answers --help
pytest -q test_evaluation_answers.py
pytest -q
git diff --check
```

실제 Answer 실행은 Ollama와 local vectorstore가 필요하므로 선택 사항이다. 실행하지 않았다면 완료 보고에 명시한다.

권장 커밋: `answer evaluator와 review worksheet 추가`

## 8. 병합 규칙

### 8.1 병합 전 각 작업자의 보고 형식

각 Phase 작업자는 다음을 보고한다.

```text
Phase N 완료 보고
- 변경 파일:
- 구현한 계약:
- 실행한 테스트와 결과:
- 실행하지 않은 live 검증과 이유:
- 공통 모듈 변경 여부: 없음/요청 번호
- 알려진 한계 또는 후속 항목:
- 커밋 SHA:
```

테스트가 실패한 상태, 공통 모듈을 승인 없이 변경한 상태 또는 live 실패를 숨긴 상태로 완료 처리하지 않는다.

### 8.2 병합 순서

1. Phase 4
2. Phase 5
3. Phase 6

각 병합 직후 해당 Phase 전용 테스트와 `pytest -q`를 실행한다. 이전 Phase 테스트가 깨지면 다음 Phase를 병합하지 않는다.

### 8.3 충돌 해결 원칙

- 한 Phase의 구현을 삭제해 다른 Phase에 맞추지 않는다.
- 공통 API 계약을 바꾸는 방식으로 충돌을 우회하지 않는다.
- Phase 5의 골든셋 추가로 Phase 6 대상 수가 달라질 수 있으므로 Phase 6 테스트는 고정 fixture를 사용하고 실제 골든셋 건수를 하드코딩하지 않는다.
- 통합 후 골든셋 validator가 통과해야 한다.
- 리포트 schema field 이름이 Phase별로 불필요하게 달라지지 않았는지 확인한다.

## 9. Phase 4·5·6 통합 완료 게이트

세 Phase 병합 후 통합 담당자가 다음을 모두 실행한다.

```bash
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl

python -m evaluation.retrieval --help
python -m evaluation.routing --help
python -m evaluation.answers --help

pytest -q test_evaluation_retrieval.py
pytest -q test_evaluation_routing.py
pytest -q test_evaluation_answers.py
pytest -q

npm test -- --run
git diff --check
```

필수 확인:

- 모든 `--help` 명령이 모델과 vectorstore를 초기화하지 않는다.
- 기본 pytest에서 live LLM 테스트는 명확한 이유로 skip되고 오프라인 테스트는 모두 통과한다.
- `routing_regression` 태그가 정확히 16건이다.
- `ROUTING_CASES` 하드코딩 정답 목록이 제거됐다.
- Retrieval과 Answer 리포트 계약의 reproducibility 필드는 non-null이다.
- Routing 리포트 계약의 reproducibility 필드는 null이고 사유가 있다.
- production `_retrieve_documents()`의 trace 비활성 결과가 기존과 동일하다.
- 상세 리포트와 worksheet에 검색 원문 chunk 전체가 저장되지 않는다.
- 기존 프런트엔드와 Python 기능에 회귀가 없다.

## 10. Live 검증 정책

Phase 4·5·6의 구현 완료와 오프라인 병합 검증은 live 모델 없이 가능해야 한다.

다음 live 실행은 환경이 준비된 경우에만 수행한다.

- 실제 Retrieval evaluator
- 실제 Routing evaluator와 기존 16건 회귀
- 실제 Answer evaluator와 worksheet 생성

live 실행 결과가 낮은 점수라는 이유만으로 구현 실패로 간주하지 않는다. M2는 품질 목표를 달성하는 단계가 아니라 현재 품질을 정확히 측정하는 baseline 단계다.

다만 다음은 구현 실패다.

- evaluator가 실행되지 않음
- 필수 artifact 누락 오류가 이해하기 어렵거나 exit 0임
- metric 계산 또는 대상 제외가 계약과 다름
- metadata/fingerprint 누락
- 개별 실패 때문에 결과 보존 없이 전체가 즉시 중단됨
- live opt-in 없이 외부 모델을 호출함

실제 최초 baseline을 고정 경로에 저장하거나 승인하는 작업은 Phase 7에서만 수행한다.

## 11. Claude Code에 전달할 최종 지시

아래 내용을 작업 요청으로 사용한다.

> `M2_Phase_4_5_6_dev_detail_plan_and_rule.md`를 전체 읽고, 상위 Requirement와 Development Plan의 Phase 4·5·6 항목도 확인한 뒤 작업하라. Phase 4, 5, 6을 별도 worktree/브랜치 또는 명확히 분리된 작업 단위로 병렬 구현하라. 파일 소유권과 공통 모듈 동결 규칙을 지키고, 공통 계약 변경이 필요하면 임의 수정하지 말고 요청으로 보고하라. 각 Phase의 오프라인 테스트 게이트를 통과시킨 뒤 Phase 4 → 5 → 6 순서로 병합하고 통합 완료 게이트를 실행하라. live 모델 검증은 명시적 opt-in과 로컬 환경이 준비된 경우에만 수행하고, 실행하지 않았다면 그대로 보고하라. Phase 7 이후 작업, 실제 baseline 승인, 문서 전면 개편은 수행하지 마라. 마지막에는 Phase별 변경 파일, 테스트 결과, 미실행 live 검증, 알려진 한계와 커밋 SHA를 구분해 보고하라.

## 12. 완료 정의

다음 조건을 모두 만족해야 이번 작업을 완료로 간주한다.

1. Phase 4·5·6 산출물이 모두 구현됐다.
2. 파일 소유권 위반과 승인되지 않은 공통 모듈 변경이 없다.
3. Phase별 필수 오프라인 테스트가 통과한다.
4. 골든셋 validator와 전체 Python·프런트엔드 회귀 테스트가 통과한다.
5. 세 CLI의 `--help`가 외부 모델 없이 동작한다.
6. live 검증 실행 여부와 결과가 오프라인 결과와 구분돼 보고됐다.
7. Phase 7에서 사용할 evaluator API와 리포트가 준비됐다.
8. 실제 baseline을 임의로 승인하거나 고정하지 않았다.
