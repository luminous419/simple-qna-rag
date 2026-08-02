# M2 Quality Baseline — 상세 개발 계획 (Development Plan)

## 0. 문서 관계

- 상위 목표: [Roadmap.md](Roadmap.md)
- 요구사항(수용 기준의 원천): [Development_M2_Quality_Baseline_Requirement.md](Development_M2_Quality_Baseline_Requirement.md) — 이하 "요구사항 문서"
- 작업 단계/결정 사항(원천): [Development_M2_Quality_Baseline_Plan.md](Development_M2_Quality_Baseline_Plan.md) — 이하 "상위 계획"
- 이 문서: 상위 계획의 Phase 0~9를 **현재 코드베이스의 실제 클래스/함수/설정에 근거해** 실행 가능한 수준으로 구체화한다. 상위 계획에 이미 있는 결정(단일 골든셋, production 경로 재사용, CI/live 분리, baseline은 gate가 아님, 혼합 Answer 평가)은 반복하지 않고 그대로 따른다.
- 이 문서는 계획서이며, 이 시점까지 `evaluation/` 코드는 아직 작성되지 않았다. 모든 함수 시그니처는 설계 제안이고, 실제 구현 중 세부 조정이 있으면 이 문서를 갱신한다.
- **개정 이력**:
  - 1차: [Problem.md](Problem.md) "M2 상세 개발 계획 1차 검토 결과"(codex 리뷰, P1 4건 + P2 3건) 반영 — (1) Answer/Retrieval evaluator eligibility를 category가 아닌 필드 존재로 재정의(§3.2, Phase 6), (2) baseline 리포트에 corpus/vectorstore fingerprint 추가(§3.6), (3) Retrieval 계측을 `trace=None` 기본값의 완전 opt-in으로 재설계(§3.4), (4) Phase 0에 `import web_server`/`pip check` 깨끗한 venv 검증 추가(§1), (5) `pydantic>=2,<3` 명시 및 `Field(default_factory=...)` 사용(§3.2), (6) nDCG gain 공식·중복 처리·집계 방식 고정(§3.5), (7) 답변 검토 worksheet를 표 대신 사례별 Markdown 섹션으로 변경(Phase 6).
  - 2차: [Problem.md](Problem.md) "M2 상세 개발 계획 2차 검토 결과"(P1 2건 + P2 3건) 반영 — (1) 이 계획과 어긋나 있던 [요구사항 문서](Development_M2_Quality_Baseline_Requirement.md)의 M2-REQ-008/010/011을 함께 수정해 두 문서를 동기화(계획 문서 단독 수정으로는 "구현자와 리뷰어가 다른 기준을 적용" 위험이 남는다는 지적 반영), (2) Recall/MRR/nDCG가 서로 다른 "k" 의미(chunk 단위 vs dedup된 source 단위)를 쓰던 문제를 `dedupe_preserve_order()` 단일 함수로 통일(§3.5, Phase 4), (3) abstention 판정 문구를 explanation류 1개에서 실제 프롬프트 템플릿 2종(explanation류 + yesno류) 모두로 확장(Phase 6), (4) Answer source 일치율 공식(`source_any_hit`/`source_recall`)과 `relevant_sources` 없는 사례의 제외 처리를 명시(Phase 6), (5) worksheet의 답변 fence 길이를 고정 3틱에서 답변 내 최장 backtick 연속+1로 동적 계산하도록 변경(Phase 6).
  - 3차: [Problem.md](Problem.md) "M2 상세 개발 계획 3차 검토 결과"(P1 1건 + P2 2건) 반영 — (1) corpus/vectorstore fingerprint를 요구하는 범위를 evaluator별로 명확히 구분(Retrieval·Answer·baseline은 필수, corpus/vectorstore를 쓰지 않는 Routing은 `null`+사유)하고 요구사항 문서 M2-REQ-010에도 동일하게 반영, `build_reproducibility_metadata()`/`build_not_applicable_reproducibility_metadata()` 두 헬퍼로 분리(§3.6, Phase 4/5/6/7), (2) Answer evaluator가 `RAGEngine.query()`의 실제 반환 형식(source 문자열이 아니라 dict 리스트)을 `_extract_returned_source_ids()`로 명시적으로 변환하도록 추가(Phase 6), (3) abstention 정확도를 TP/TN/FP/FN 기반 `_abstention_confusion()`으로 명확히 정의(Phase 6).
  - 4차: [Problem.md](Problem.md) "M2 상세 개발 계획 4차 검토 결과"(P1 1건 + P2 1건) 반영 — (1) `build_corpus_manifest()`/`build_reproducibility_metadata()`가 집계 SHA-256뿐 아니라 파일별 `source_id`/`size_bytes`/`sha256` 배열(`corpus_manifest`) 자체를 반환·리포트에 포함하도록 수정하고 canonical 직렬화 규칙(정렬 기준, `json.dumps` 옵션)을 명시, 요구사항 문서 M2-REQ-010에도 "배열 자체가 리포트에 포함되어야 함"을 반영(§3.6), (2) 통합 baseline 최종 리포트의 top-level에 Retrieval 단계 fingerprint를 승격하고 Answer 단계가 독립적으로 재계산한 값과 불일치하면 실패 처리하는 invariant를 정의, `--skip-answers`에서도 top-level이 유지됨과 Routing 단독 실행은 이 규칙 대상이 아님을 명시(§3.6, Phase 7), 요구사항 문서에도 top-level 요구를 반영.
  - 5차: [Problem.md](Problem.md) "M2 상세 개발 계획 5차 검토 결과"(P1 1건 + P2 1건) 반영 — (1) 이전 버전에서 제목만 "실행 완료"였고 실제로는 미실행이었던 Phase 0 clean-venv 게이트를 **이번 개정에서 실제로 실행**했다: 공유 conda 환경과 무관한 `/usr/local/bin/python3.11`로 새 venv를 만들어 `pip install -r requirements.txt`, `pip check`, `import web_server`, `TestClient` import, `pytest -q`를 모두 통과시켰고 결과를 §1에 표로 기록 — 공유 환경의 `import web_server` 실패는 저장소 문제가 아니라 순수 공유 환경 오염이었음을 확정하고 `requirements.txt` 선행 수정이 불필요함을 결론지었다, (2) `normalize_source_id()`가 basename만 남기면서 생기는 정규화 충돌(다른 경로의 동일 파일명, 대소문자/NFC-NFD만 다른 이름)을 `build_corpus_manifest()`의 `CorpusManifestError`(corpus 실제 파일 충돌)와 `dataset.py`의 사례 내부 `relevant_sources` 중복 검사(골든셋 저작 실수)로 이원화해 방어(§3.3). 이 리뷰가 다루지 않은 나머지 절은 4차 개정 그대로다.
  - 6차: Phase 1 구현 설계를 담은 [Development_M2_Quality_Baseline_Design.md](Development_M2_Quality_Baseline_Design.md)의 2차 [design_review.md](design_review.md) 검토(P1 2건 + P2 2건 + P3 1건) 중, Design.md 자체가 아니라 이 상위 계획의 Phase 3/4 설계에 속하는 항목 2건을 반영 — (1) `evaluation/metrics.py`에 `normalize_relevance_grades()` 헬퍼를 추가하고 `evaluate_retrieval()`의 nDCG 호출 시퀀스에 정규화 단계를 명시해, grade key 정규화를 evaluator가 빠뜨리면 nDCG가 조용히 0으로 계산되는 위험을 추적 가능하게 함(§3.5), (2) 요구사항 문서 M2-REQ-002/003을 Design.md와 동기화 — intent 최소 수량이 Answer 평가 대상 사례 기준임을 명시하고, `relevance_grades`의 양수 등급 source가 `relevant_sources`에도 포함돼야 한다는 조건부 필드 규칙을 추가했으며, §5.2 골든셋 배분안에도 이 두 제약을 반영(document_qa 42개 중 intent는 assertion 보유 22개 안에서 확보, 양수 등급 source는 반드시 relevant_sources에 포함). 나머지 항목(schema.py의 model_validator, extra=forbid 등)은 Design.md 자체 개정 사항이며 이 문서는 변경하지 않았다.
  - 7차: Design.md의 3차 [design_review.md](design_review.md) 검토(P1 1건 + P2 2건 + P3 1건) 중, 이 상위 계획에 속하는 항목 1건을 반영 — Phase 1 `dataset.py`와 Phase 6 `answers.py`에 각각 독립적으로 정의돼 있던 `_is_answer_eval_eligible()`을 `evaluation/schema.py`의 공개 함수 `is_answer_eval_eligible()` 하나로 통일하고, Phase 6 §코드 스니펫과 테스트 목록(§Phase 6)을 이 함수를 import해서 쓰는 형태로 갱신했다. 나머지 항목(3차 P1은 Design.md의 스캐폴딩 재검증 결과 이미 해당 없음으로 확인됨, 나머지 P2/P3는 Design.md 자체 문서화 사항)은 이 문서를 변경하지 않았다.

## 1. Phase 0 — 착수 전 상태 기록 (clean-venv 게이트 포함 실행 완료)

**Problem.md 5차 리뷰 P1 반영**: 이전 버전은 절 제목에 "실행 완료"라고 적어두고도 본문의 clean-venv 체크리스트는 미완료 상태로 남겨 제목과 내용이 충돌했다. 이번 개정에서 아래 clean-venv 검증을 실제로 수행해 결과를 채웠으므로 제목의 "실행 완료"가 이제 본문과 일치한다.

이 계획을 작성하며 실제로 실행/기록한 결과다. 구현 착수 시점에 다시 한 번 확인한다.

| 항목 | 값 |
|---|---|
| 브랜치 | `fix/security-review-followups` (PR #3, master로 병합 대기) |
| 작업 트리 | dirty — `Problems.md`/구 improvement_*.md 삭제, `README.md` 재작성, `frontend_tests/render.test.js` 수정, `Roadmap.md`/`Problem.md`/`Development_M2_*.md` 신규(untracked). **M2 착수 전 PR #3을 먼저 머지하고 이 문서 재구조화 변경을 별도로 커밋해 브랜치를 깨끗한 상태로 만들 것을 권장** (§8 참고) |
| `pytest -q` | `21 passed, 1 skipped, 1 warning` (스킵 = 라이브 Ollama 라우팅 테스트) |
| `npm test` | `9 passed` (Vitest) |
| `git diff --check` | 통과 (공백 오류 없음) |
| Python | 3.11.8 |
| Node | v22.17.0 / npm 10.9.2 |
| `vectorstore/` | 존재함 (`index.faiss`, `index.pkl`) |
| `data/` | 18개 파일 (PDF 15, TXT/기타 3) — §5 참고 |
| Ollama 모델 | `gpt-oss:20b` 로컬 설치 확인됨 (13GB, `config.OLLAMA_MODEL`과 일치) — live baseline 실행 가능 |
| pytest 설정 파일 | 없음 (`pytest.ini`/`pyproject.toml`/`conftest.py` 부재) — 모든 테스트는 저장소 루트의 평평한 `test_*.py`로 자동 수집됨 |
| 기존 라이브 라우팅 사례 | `test_agent_routing.py`의 `ROUTING_CASES` 16개, `MIN_ACCURACY=0.8`, `RUN_LIVE_LLM_TESTS=1` opt-in. 정답은 이 리스트에만 존재(별도 정답 소스 없음) → Phase 5에서 골든셋으로 흡수 필요 |

`pytest -q`가 21 passed를 보여도 **애플리케이션이 실제로 뜨는지는 검증하지 않는다** — 어떤 `test_*.py`도 `web_server.py`를 import하지 않기 때문이다. 공유 conda 환경(`common`)에서는 다음 세 명령이 이 문제를 드러냈다.

```bash
python -c "import web_server"                        # ImportError
python -c "from fastapi.testclient import TestClient" # ImportError (동일 원인)
python -m pip check                                    # 9건의 버전 불일치
```

- `import web_server` / `TestClient` import 둘 다 **동일한 원인으로 실패**: 공유 환경에 설치된 `email-validator==1.3.1`이 그 환경의 `fastapi`/`pydantic`이 요구하는 `>=2.0`을 만족하지 못해 `pydantic.networks` 모듈 로드 시점에 `ImportError`가 난다.
- `pip check`가 보고하는 9건 중 `torchvision`/`torch`, `langchain-classic`, `langgraph-prebuilt`, `google-api-core`/`protobuf`, `opentelemetry-*`, `langchain-postgres`/`sqlalchemy`, `streamlit`/`protobuf` 항목은 이 저장소의 `requirements.txt`에 없는 패키지(`langgraph-prebuilt`, `google-api-core`, `streamlit`, `langchain-postgres`, `opentelemetry-*` 등)가 관여하고 있어 공유 conda 환경에 다른 프로젝트들이 설치한 패키지와의 충돌로 추정됐다.

**clean-venv 검증 결과(Problem.md 5차 리뷰 P1 반영 — 실제로 실행 완료)**:

`/usr/local/bin/python3.11`(conda `common` 환경과 무관한 별도 Python 설치, `pip`/`setuptools`만 있는 상태)로 새 venv를 만들고 `pip install -r requirements.txt`부터 새로 실행했다.

| 명령 | 결과 |
|---|---|
| `pip install -r requirements.txt` | 성공. `pydantic-2.13.4`가 fastapi(`0.141.1`)의 전이 의존성으로 자동 설치됨(§3.2에서 계획한 `pydantic>=2,<3` 명시 필요성과 일치) |
| `python -m pip check` | `No broken requirements found.` — 공유 환경에서 보였던 9건은 전부 이 저장소와 무관한 다른 프로젝트의 패키지였음이 확인됨 |
| `python -c "import web_server"` | 성공(exit 0) — 공유 환경의 `email-validator` 충돌은 저장소 문제가 아니라 순수 공유 환경 오염이었음이 확정됨. **`requirements.txt`에 `email-validator` 관련 수정은 필요 없음** |
| `python -c "from fastapi.testclient import TestClient"` | 성공(exit 0, `httpx`/`starlette.testclient` 관련 `DeprecationWarning` 하나만 출력, 오류 아님) |
| `pytest -q` | `21 passed, 1 skipped` — 공유 환경과 동일 |
| `npm test` | venv와 무관(Node 별도 실행 환경) — 이미 §1 위쪽 표에 기록된 `9 passed` 그대로 유효 |

결론: **이 저장소의 `requirements.txt`는 그 자체로 문제가 없다.** 공유 conda 환경에서만 관찰됐던 실패는 그 환경에 설치된 다른 프로젝트들의 패키지가 만든 오염이었다. `email-validator` 등 의존성 수정을 Phase 1보다 먼저 처리할 필요는 없다. 다만 이 결론이 다시 조용히 깨지지 않도록, Phase 8의 CI `python-tests` job에 `python -m pip check`와 `python -c "import web_server"`를 스모크 스텝으로 유지한다(이미 Phase 8 워크플로에 포함되어 있음, §4).

Phase 0 착수 게이트 체크리스트(완료):

- [x] 새 Python 3.11 venv 생성 후 `pip install -r requirements.txt` — 성공
- [x] 위 venv에서 `python -c "import web_server"`, `python -c "from fastapi.testclient import TestClient"`, `python -m pip check` 재실행 — 모두 통과
- [x] 결과가 공유 환경과 다름(clean venv에서 통과) — 원인을 위 표에 기록함, `requirements.txt` 선행 수정 불필요로 결론
- [x] `pytest -q`, `npm test`, `git diff --check` 재확인 — clean venv에서도 `pytest -q` 동일 결과, `npm test`/`git diff --check`는 위쪽 표 값 그대로 유효

이 게이트가 통과했으므로 "기존 테스트에 실패가 없다"는 전제를 Web UI/API 계층을 포함해 저장소 전체에 적용한다. 이번 마일스톤 동안 발생하는 모든 실패는 M2 변경에 기인한 것으로 간주한다.

## 2. 요구사항 → Phase/파일 추적 매트릭스

| 요구사항 | 구현 Phase | 주요 파일 |
|---|---|---|
| M2-REQ-001 (패키지 구조) | 1 | `evaluation/__init__.py` 등 전체 |
| M2-REQ-002 (규모/구성) | 1, 2 | `evaluation/dataset.py`, `evaluation/datasets/golden.jsonl` |
| M2-REQ-003 (스키마) | 1 | `evaluation/schema.py` |
| M2-REQ-004 (validate CLI) | 1 | `evaluation/dataset.py` |
| M2-REQ-005 (Retrieval 지표) | 3, 4 | `evaluation/metrics.py`, `evaluation/retrieval.py` |
| M2-REQ-006 (단계 계측) | 4 | `rag_engine.py`, `evaluation/retrieval.py` |
| M2-REQ-007 (Routing 평가) | 5 | `evaluation/routing.py`, `test_agent_routing.py` |
| M2-REQ-008 (Answer 평가) | 6 | `evaluation/answers.py` |
| M2-REQ-009 (통합 baseline) | 7 | `evaluation/baseline.py` |
| M2-REQ-010 (리포트/메타데이터) | 3, 7 | `evaluation/reporting.py`, `evaluation/baselines/` |
| M2-REQ-011 (metric 테스트) | 3 (+ 4/5/6 evaluator 테스트) | `test_evaluation_metrics.py` 등 |
| M2-REQ-012 (회귀 방지) | 0, 4 (전 Phase 공통) | 기존 테스트 스위트 전체 |
| M2-REQ-013 (CI) | 8 | `.github/workflows/ci.yml` |
| M2-REQ-014 (문서화) | 9 | `README.md`, `evaluation/README.md`, `Roadmap.md`, `Problem.md` |
| M2-REQ-015 (보안/데이터 취급) | 2, 6, 7 (전 Phase 공통) | golden.jsonl 작성 원칙, reporting.py |
| M2-REQ-016 (오류 처리/종료 코드) | 1, 4, 5, 6, 7 | 각 CLI `main()` |
| M2-NFR-001 (결정론) | 3 | `evaluation/reporting.py` |
| M2-NFR-002 (비교 가능성) | 3, 7 | `evaluation/reporting.py`, `evaluation/baseline.py` |
| M2-NFR-003 (지연 로딩) | 1 (설계 원칙, 전 Phase 적용) | 모든 `evaluation/*.py` |
| M2-NFR-004 (실행 시간) | 8 | `.github/workflows/ci.yml` |
| M2-NFR-005 (유지보수성) | 1~7 (전 Phase 공통) | 타입힌트/docstring 컨벤션 |

## 3. 설계 확정 사항

### 3.1 패키지/테스트 배치

```text
evaluation/
├── __init__.py              # 빈 파일. import 시 부수효과 없음(NFR-003)
├── schema.py
├── dataset.py
├── metrics.py
├── reporting.py
├── retrieval.py
├── routing.py
├── answers.py
├── baseline.py
├── datasets/
│   └── golden.jsonl
├── baselines/
│   ├── m2_initial.json
│   └── m2_initial.md
└── reports/                 # gitignore 대상, 타임스탬프 상세 리포트
```

테스트는 저장소 관례(평평한 `test_*.py`, `tests/` 디렉토리 없음, `conftest.py` 없음)를 그대로 따른다. `evaluation/` 자체는 패키지지만 테스트는 패키지 밖 루트에 둔다.

```text
test_evaluation_schema.py
test_evaluation_dataset.py
test_evaluation_metrics.py
test_evaluation_reporting.py
test_evaluation_retrieval.py
test_evaluation_routing.py
test_evaluation_answers.py
```

새 런타임 의존성은 추가하지 않는다. `pydantic`(fastapi 종속으로 이미 설치됨), `argparse`/`unicodedata`/`hashlib`/`statistics`/`time`/`json`(표준 라이브러리)만 사용한다.

### 3.2 데이터 모델 (`evaluation/schema.py`)

이 설계는 **Pydantic 2**의 `field_validator`/`model_validator` API를 전제한다. `requirements.txt`는 현재 Pydantic 버전을 직접 고정하지 않고 fastapi의 전이 의존성에 맡기고 있으므로, Phase 1에서 `requirements.txt`에 `pydantic>=2,<3`을 명시적으로 추가한다(별도 최소 변경, `email-validator` 건과 별개). Pydantic 1이 설치되면 이 스키마는 import 시점에 깨진다.

```python
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator

class Category(str, Enum):
    DOCUMENT_QA = "document_qa"
    WEB_SEARCH = "web_search"
    BOUNDARY = "boundary"
    UNANSWERABLE = "unanswerable"

class Route(str, Enum):
    DOCUMENT_QA = "document_qa"
    WEB_SEARCH = "web_search"

class Intent(str, Enum):
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    PROCEDURE = "procedure"
    YESNO = "yesno"
    OTHER = "other"
    UNCERTAIN = "uncertain"          # intent_classifier의 라벨과 동일 (라벨 6개)

class AnswerAssertion(BaseModel):
    any_of: list[str]

    @field_validator("any_of")
    @classmethod
    def non_empty(cls, v):
        if not v or not all(s.strip() for s in v):
            raise ValueError("any_of는 비어 있지 않은 문자열을 하나 이상 포함해야 함")
        return v

class GoldenCase(BaseModel):
    id: str
    question: str
    category: Category
    expected_route: Route
    expected_intent: Optional[Intent] = None
    relevant_sources: list[str] = Field(default_factory=list)
    relevance_grades: dict[str, int] = Field(default_factory=dict)
    answer_assertions: list[AnswerAssertion] = Field(default_factory=list)
    expect_abstention: bool = False
    tags: list[str] = Field(default_factory=list)
    notes: Optional[str] = None

    @field_validator("id")
    @classmethod
    def id_not_blank(cls, v): ...   # strip 후 비어있지 않아야 함

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, v): ...

    @field_validator("relevance_grades")
    @classmethod
    def grades_in_range(cls, v):
        # 0~3 정수만 허용
        ...
```

**중요한 설계 정정(Problem.md P1 리뷰 반영)**: `GoldenCase` 자체에는 "answer_assertions 또는 expect_abstention 중 하나 필요" 같은 **cross-field 필수 검증을 두지 않는다.** 초안에 있던 `model_validator`가 이 조건을 개별 사례 단위로(게다가 사실상 모든 `document_qa` 사례에) 강제했는데, 이는 요구사항과 어긋난다 — 요구사항은 문서 QA 40개 이상 중 **20개 이상만** answer assertion을 가지면 되고(M2-REQ-002), 나머지 문서 QA 사례는 Retrieval 평가 전용(assertion도 abstention도 없이 `relevant_sources`만 가짐)이어도 정상이다. `expect_abstention=true` 사례는 별도로 `category=unanswerable`에 배치되며(§5.2), `document_qa`의 20개 카운트와는 무관하다.

대신 **평가 대상 여부(eligibility)는 field 존재 여부로만 판단하고 category와 분리한다**:

- Routing 평가: 모든 사례(전부 `expected_route`를 가짐)
- Retrieval 평가: `relevant_sources`가 비어 있지 않은 사례 (category 무관)
- Answer 평가: `answer_assertions`가 비어 있지 않거나 `expect_abstention=True`인 사례 (category 무관 — `unanswerable`의 abstention 사례도 포함됨)

"document_qa 20개 이상 answer_assertions", "document_qa 30개 이상 relevant_sources", "전체 5개 이상 expect_abstention" 같은 **집계 규칙**은 개별 스키마가 아니라 `dataset.py`의 구성 validator(`validate_composition`, Phase 1)에서만 검사한다 — 사례 하나만 봐서는 이 사례가 그 집계 안에서 어떤 역할을 하는지 알 수 없기 때문이다.

### 3.3 source ID 정규화 규칙

`data/` 실제 파일명 예시(공백, 한글, "복사본" 접미사, 괄호 포함):

```text
Retrieval-Augmented Generation (RAG) 복사본.pdf
2025_한국정부_부동산정책_정리.txt
2025 KB 부동산 보고서 복사본.pdf
```

`DirectoryLoader` + `PyPDFLoader`/`TextLoader`는 `doc.metadata["source"]`에 로드 시점의 파일 경로(예: `data/Retrieval-Augmented Generation (RAG) 복사본.pdf`, OS/실행 위치에 따라 구분자나 선행 `./` 차이가 날 수 있음)를 넣는다. 골든셋의 `relevant_sources`는 **basename만** 기록한다(전체 경로 아님).

`evaluation/schema.py`에 다음 함수를 두고 `retrieval.py`/`answers.py`가 모두 이 함수만 사용해 비교한다.

```python
import os
import unicodedata

def normalize_source_id(raw: str) -> str:
    """source 비교용 정규화: NFC → 경로 구분자 통일 → basename → 소문자."""
    value = unicodedata.normalize("NFC", raw)
    value = value.replace("\\", "/")
    value = os.path.basename(value)
    return value.casefold()
```

`os.path.basename`은 POSIX/Windows 어느 쪽에서 생성된 경로든 앞서 `\\`→`/` 치환을 해두면 정상 동작한다. `casefold()`는 `lower()`보다 유니코드 대소문자 비교에 더 안전해 채택한다.

**source ID 유일성(Problem.md 5차 리뷰 P2 반영)**: `normalize_source_id()`는 경로를 버리고 basename만 남기므로, 서로 다른 하위 디렉터리에 같은 파일명이 있거나 대소문자·Unicode 표현만 다른 두 파일이 있으면 서로 다른 문서가 같은 source ID로 조용히 합쳐진다. 현재 `data/`는 평평한 구조라(하위 디렉터리 없음, §1 Phase 0에서 확인한 18개 파일 전부 `data/` 바로 아래) 실제로 발생하지 않지만, 검증 없이 넘어가면 향후 하위 디렉터리가 추가되는 순간 조용한 데이터 오염으로 이어진다. 이를 두 지점에서 방어한다 — 두 지점으로 나누는 이유는 각자 접근 가능한 정보와 실행 전제(특히 CI에서 `data/`가 존재하지 않는다는 것, M2-REQ-013)가 다르기 때문이다.

1. **`build_corpus_manifest()`(§3.6, `data/`를 실제로 스캔하는 유일한 지점)**: 정규화된 `source_id`가 중복되는 실제 파일이 있으면 `CorpusManifestError`를 발생시키고, 충돌한 실제 경로 목록을 함께 담는다. `data/`가 없는 CI 환경(`evaluation.dataset validate`만 실행)은 이 검사를 거치지 않는다.

   ```python
   class CorpusManifestError(Exception):
       def __init__(self, message: str, collisions: dict[str, list[Path]]): ...

   def build_corpus_manifest(data_dir: Path) -> dict:
       """... 정규화된 source_id가 같은 서로 다른 실제 파일이 있으면 CorpusManifestError
       (충돌 source_id -> 실제 경로 목록)를 발생시킨다."""
   ```

2. **`evaluation/dataset.py`의 golden case 검증(Phase 1)**: `data/`를 스캔하지 않고도 확인 가능한 좁은 범위로, 한 `GoldenCase`의 `relevant_sources` 리스트 안에서 정규화 후 중복되는 항목이 있으면 오류로 처리한다(예: 같은 파일을 대소문자만 다르게 두 번 적은 저작 실수). 이 검사는 `data/`를 읽지 않으므로 CI(`python -m evaluation.dataset validate`)에서도 그대로 동작하고 Phase 1의 지연 로딩 원칙(NFR-003)을 위반하지 않는다.

두 검사는 서로 다른 것을 잡는다 — (1)은 corpus 자체의 물리적 충돌, (2)는 골든셋 저작 실수. 어느 쪽도 상대를 대체하지 않는다.

경계 테스트(§3.5 목록에 추가): 같은 basename의 서로 다른 경로(`a/x.pdf` vs `b/x.pdf`), 대소문자만 다른 이름(`X.pdf` vs `x.pdf`), NFC/NFD 표현만 다른 이름이 각각 `build_corpus_manifest()`에서 `CorpusManifestError`로 검출되는지, 그리고 `relevant_sources` 내부 중복이 `dataset.py`에서 검출되는지.

**향후 확장 여지**: 하위 디렉터리에 동일 basename을 실제로 지원해야 하는 상황이 오면(현재 M2 범위 밖), `normalize_source_id()`를 basename 대신 `data/` 기준 정규화 상대 경로로 바꾸고 `schema_version`을 올려 전환한다 — 이 시점의 모든 골든셋 `relevant_sources` 값도 상대 경로로 다시 작성해야 하므로 마이그레이션으로 취급한다.

### 3.4 RAGEngine 계측 설계 — production 코드 최소 변경, 계측은 완전히 opt-in

**설계 정정(Problem.md P1 리뷰 반영)**: 초안은 항상 trace 객체를 만드는 `_retrieve_documents_traced()`를 실제 구현으로 삼고 기존 `_retrieve_documents()`가 이를 감싸는 wrapper였다. 이 경우 계측을 끌 방법이 없어 **모든 일반 Web/API 요청에서도 항상 trace 객체와 stage 목록이 생성**됐고, 이는 M2-REQ-006 "계측 비활성 상태에서 기존 호출자가 변경될 필요가 없어야 한다"의 취지(비활성 시 비용 없음)와 맞지 않았다.

수정된 설계: `trace` 인자를 `None`으로 기본값을 주는 **단일 메서드**로 유지한다. 로직은 여전히 하나뿐이므로(분기 위험 없음), `trace is None`이면 `RetrievalStageTrace`를 단 하나도 생성하지 않는다.

```python
# rag_engine.py 에 추가
from dataclasses import dataclass, field
import time

@dataclass
class RetrievalStageTrace:
    name: str                 # "bm25" | "dense" | "rrf" | "mmr" | "reranker" | "total"
    latency_ms: float
    candidate_count: int

@dataclass
class RetrievalTrace:
    stages: list[RetrievalStageTrace] = field(default_factory=list)


class RAGEngine:
    ...
    def _retrieve_documents(self, question: str, trace: "RetrievalTrace | None" = None):
        """
        기존 호출부(RAGEngine.query() 등)는 `self._retrieve_documents(question)`을
        그대로 호출하면 되고 동작·반환값이 기존과 100% 동일하다(M2-REQ-012).
        trace=RetrievalTrace()를 넘긴 evaluator 호출만 단계별 latency를 얻는다.
        """
        t_total0 = time.perf_counter() if trace is not None else None

        def stage(name, fn):
            if trace is None:
                return fn()
            t0 = time.perf_counter()
            result = fn()
            trace.stages.append(
                RetrievalStageTrace(name, (time.perf_counter() - t0) * 1000, len(result))
            )
            return result

        if USE_HYBRID_SEARCH:
            bm25_docs = stage("bm25", lambda: self.bm25_retriever.invoke(question, top_k=BM25_TOP_K))
            dense_docs = stage(
                "dense",
                lambda: self.dense_retriever.invoke(question) if self.dense_retriever else [],
            )
            docs = stage(
                "rrf",
                lambda: self._reciprocal_rank_fusion(bm25_docs, dense_docs, top_k=RRF_TOP_K, k=RRF_CONSTANT),
            )
            if USE_MMR:
                docs = stage("mmr", lambda: self._apply_mmr(question, docs, top_k=MMR_K, lambda_mult=MMR_LAMBDA))
            if USE_RERANKER:
                docs = stage("reranker", lambda: self._rerank_documents(question, docs, top_k=RERANKER_TOP_K))
        elif USE_MMR:
            docs = stage("dense", lambda: self.dense_retriever.invoke(question))
            if USE_RERANKER:
                docs = stage("reranker", lambda: self._rerank_documents(question, docs, top_k=RERANKER_TOP_K))
        elif USE_RERANKER:
            docs = stage("dense", lambda: self.dense_retriever.invoke(question))
            docs = stage("reranker", lambda: self._rerank_documents(question, docs, top_k=RERANKER_TOP_K))
        else:
            docs = stage("dense", lambda: self.dense_retriever.invoke(question))

        if trace is not None:
            trace.stages.append(
                RetrievalStageTrace("total", (time.perf_counter() - t_total0) * 1000, len(docs))
            )
        return docs
```

이 네 개 분기(Hybrid[+MMR][+Reranker] / MMR-only[+Reranker] / Reranker-only / plain similarity)는 현재 `_retrieve_documents()`에 실제로 존재하는 분기 구조 그대로다 — 새로 추가하는 게 아니라 각 호출을 `stage()` 헬퍼로 감싸기만 한다.

이렇게 하면:

- 기존 호출자는 코드 변경 없이 그대로 동작.
- `trace=None`일 때 생성되는 객체가 전혀 없다(클로저 호출 오버헤드 정도만 남음 — dict/list 할당이나 직렬화는 없음). 이 오버헤드가 실측상 문제가 되면(임베딩/LLM 호출에 비하면 무시 가능할 것으로 예상하지만) Phase 4 완료 보고에 실측치를 남기고 필요 시 재검토한다.
- 로직이 하나뿐이므로 "계측 전후 검색 문서 순서가 동일한가"가 구조적으로 성립.

**단위 테스트(Phase 4, 필수 — 선택 아님)**: config 조합 4가지(Hybrid+MMR+Reranker, MMR-only, Reranker-only, plain similarity)에 대해 `bm25_retriever`/`dense_retriever`/`vectorstore`를 결정론적 가짜 문서를 반환하는 더미로 monkeypatch하고, 각 조합에서 `trace=None`과 `trace=RetrievalTrace()` 두 호출이 **동일한 문서 리스트(순서 포함)**를 반환하는지 characterization test로 고정한다. 모델/벡터스토어가 필요 없으므로 CI에서도 실행 가능하다(`test_evaluation_retrieval.py` 또는 별도 `test_rag_engine_retrieval.py`에 배치).

### 3.5 Metric 정의 (`evaluation/metrics.py`)

**source 순위 단위 통일(Problem.md 2차 리뷰 P1 반영)**: 검색 결과는 chunk 목록이고 골든 정답은 source(파일) 단위다. 초안은 `recall_at_k`/`mrr_at_k`는 원본 chunk 리스트를 그대로 `[:k]`로 자르고, `ndcg_at_k`(내부 `_dcg`)만 중복 source를 먼저 제거한 뒤 순위를 다시 매겼다 — 그 결과 같은 검색 결과에 대해 지표마다 "k"의 의미가 달랐다(예: `A,A,A,B` 반환 시 Recall@3은 B를 놓친 것으로, nDCG는 B를 2위로 처리). 이를 막기 위해 **중복 제거를 metric 함수 밖, 호출부에서 단 한 번만 수행**하고 세 지표 모두 그 결과를 그대로 받는다.

```python
def dedupe_preserve_order(ids: list[str]) -> list[str]:
    """최초 등장 순서를 유지한 채 중복 제거. Recall/MRR/nDCG가 모두 이 함수의 출력을
    입력으로 받아야 하며, 그래야 세 지표에서 "k"가 동일하게 "top-k 고유 source"를 의미한다."""
    seen = set()
    result = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            result.append(i)
    return result
```

| 함수 | 시그니처 | 비고 |
|---|---|---|
| `dedupe_preserve_order` | `(ids: list[str]) -> list[str]` | 위 정의. `recall_at_k`/`mrr_at_k`/`ndcg_at_k`는 **이미 이 함수를 거친 리스트**를 받는다고 전제하고, 내부에서 별도로 중복을 제거하지 않는다(제거 로직이 두 곳에 있으면 다시 갈라질 위험이 있으므로) |
| `recall_at_k` | `(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float` | `len({ranked_ids[:k]} ∩ relevant)/len(relevant)`. `relevant`가 비면 `ValueError` (평가 대상 아님을 호출부에서 걸러야 함) |
| `mrr_at_k` | `(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float` | 첫 관련 문서의 `1/rank`, k 밖이면 0 |
| `ndcg_at_k` | `(ranked_ids: list[str], relevance_grades: dict[str, int], k: int) -> float` | §nDCG 공식 참고 |
| `precision_recall_f1` | `(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict` | 라벨별 precision/recall/F1 + `confusion_matrix: dict[str, dict[str, int]]` |
| `percentile` | `(values: list[float], p: float) -> float` | nearest-rank 방식, 결정론적(외부 라이브러리 percentile 보간 방식 차이를 피함). `values`가 비면 `None` 반환(M2-REQ-011 "빈 latency 목록" 케이스) |
| `mean_median` | `(values: list[float]) -> tuple[float, float]` | 빈 리스트 시 `(None, None)` |
| `assertion_coverage` | `(answer: str, assertions: list[AnswerAssertion]) -> tuple[int, int]` | `(통과한 assertion 수, 전체 assertion 수)`. 비교 전 NFC 정규화 + `casefold()` |
| `normalize_relevance_grades` | `(grades: dict[str, int]) -> dict[str, int]` | `{normalize_source_id(k): v for k, v in grades.items()}`. key 정규화 후 서로 다른 raw key가 충돌하면 `ValueError`(Phase 1 schema가 골든셋 저작 시점에는 이미 이 충돌을 막아주지만, 이 함수는 임의의 dict를 받을 수 있으므로 방어적으로 재검사한다) |

**grade key 정규화 책임(Design.md의 design_review.md 2차 P2 반영)**: Design.md의 `GoldenCase`는 `relevance_grades`의 key를 원본 표기 그대로 저장하고 자동으로 정규화하지 않는다 — 대신 이 계약을 지키는 책임은 evaluator 쪽에 있다. `ndcg_at_k(ranked_ids, relevance_grades, k)`의 `relevance_grades.get(doc_id, 0)` 조회가 정확히 맞으려면 `doc_id`(이미 `normalize_source_id()`를 거친 값)와 `relevance_grades`의 key가 같은 정규화 형태여야 한다. 이를 빠뜨리면 schema/타입 검증은 모두 통과하면서 nDCG 점수만 조용히 0으로 계산되는 위험이 있으므로, 반드시 `normalize_relevance_grades()`를 거친 mapping만 `ndcg_at_k`에 전달한다는 precondition을 함수 docstring에 명시하고 `test_evaluation_metrics.py`에 "정규화하지 않은 key로 호출하면 관련 문서가 grade 0으로 처리돼 nDCG가 부정확해진다"를 보이는 회귀 테스트를 추가한다(Phase 4).

`evaluation/retrieval.py`의 `evaluate_retrieval()`(Phase 4)이 사례마다 정확히 다음 순서로 호출한다: `docs = engine._retrieve_documents(question, trace=trace)` → `raw_ids = [normalize_source_id(d.metadata.get("source","")) for d in docs]` → `ranked_ids = dedupe_preserve_order(raw_ids)` → `relevant_ids = set(dedupe_preserve_order([normalize_source_id(s) for s in case.relevant_sources]))` → `grades = normalize_relevance_grades(case.relevance_grades)` → `recall_at_k`/`mrr_at_k`는 `(ranked_ids, relevant_ids, k)`를, `ndcg_at_k`는 `(ranked_ids, grades, k)`를 받는다. `answers.py`의 source 일치율 계산(§Phase 6)도 동일한 `normalize_source_id` + `dedupe_preserve_order` 조합을 재사용한다.

경계 테스트(M2-REQ-011 명시 항목)는 각각 최소 1개 이상 케이스로 커버:
빈 검색 결과, relevant source 다수, **중복 source가 top-k 경계에 걸치는 경우(`A,A,A,B`류 입력에서 recall/MRR/nDCG가 동일한 dedup 순위를 기준으로 일관된 값을 내는지)**, relevance grade 없는 문서, 모든 예측이 한 route, 빈 latency 목록, NFC/NFD 동일 파일명 동일 취급, Windows(`\\`)/POSIX(`/`) 경로 동일 취급.

#### nDCG 공식과 집계 방식 (Problem.md 1차 리뷰 P2 반영 — 명시적으로 고정)

구현자에 따라 값이 달라지지 않도록 다음을 그대로 따른다. `ranked_ids`는 위에서 이미 `dedupe_preserve_order()`를 거쳤으므로 `_dcg`는 더 이상 자체적으로 중복을 제거하지 않는다(2차 리뷰 반영 — 제거 로직을 한 곳으로 통일).

```python
import math

def _dcg(ranked_ids: list[str], relevance_grades: dict[str, int], k: int) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        grade = relevance_grades.get(doc_id, 0)
        gain = (2 ** grade) - 1
        dcg += gain / math.log2(rank + 1)
    return dcg

def ndcg_at_k(ranked_ids, relevance_grades, k):
    dcg = _dcg(ranked_ids, relevance_grades, k)
    ideal_order = sorted(relevance_grades, key=relevance_grades.get, reverse=True)
    idcg = _dcg(ideal_order, relevance_grades, k)
    return dcg / idcg if idcg > 0 else 0.0
```

- gain은 `2**grade - 1` (표준 graded gain). grade 0/1/2/3 → gain 0/1/3/7.
- 중복 source ID 처리는 `dedupe_preserve_order()`(호출부, 위 참고)에서 최초 등장만 남기는 것으로 이미 끝났으므로 여기서는 추가 처리가 없다.
- 사례별 nDCG를 계산한 뒤, **`relevance_grades`가 있는 사례만 대상으로 단순 평균(macro average)**한다.
- `relevance_grades`가 없는(=nDCG 대상이 아닌) 사례는 집계에서 제외하고, 리포트에 `ndcg_excluded_count`로 별도 보고한다(recall/MRR 대상 사례 수와 다를 수 있음에 유의 — recall/MRR은 `relevant_sources`만 있으면 계산되지만 nDCG는 `relevance_grades`가 있어야 함).

### 3.6 리포트 규칙 (`evaluation/reporting.py`)

```python
def build_metadata(dataset_path: Path, command: list[str], extra: dict) -> dict: ...
def write_report(payload: dict, output_dir: Path, name: str) -> tuple[Path, Path]:
    """output_dir/{name}_{utc_timestamp}.json, .md 생성 후 경로 반환. JSON은 sort_keys=True, ensure_ascii=False, indent=2."""

def build_corpus_manifest(data_dir: Path) -> dict:
    """
    data_dir 하위 모든 파일의 정규화 source id(schema.normalize_source_id)/크기(byte)/SHA-256을
    entries 배열로 나열한다. Problem.md 4차 리뷰 P1 반영 — entries는 source_id 오름차순으로
    정렬한 뒤 canonical JSON(json.dumps(entries, sort_keys=True, ensure_ascii=False,
    separators=(",", ":")))으로 직렬화해 SHA-256을 계산한 manifest_sha256을 함께 반환한다.
    이 canonical 직렬화 형식(정렬 기준, separators, ensure_ascii)이 바뀌면 이전 실행과
    manifest_sha256을 비교할 수 없게 되므로, 바뀔 때는 reporting.py의 schema_version을
    함께 올려야 한다. 파일을 읽어 해시만 계산하므로 모델/FAISS 로드가 필요 없다(저비용,
    Phase 4/6/7의 매 실행마다 불러도 부담 없음).

    Returns: {"entries": [{"source_id":.., "size_bytes":.., "sha256":..}, ...], "manifest_sha256": ..}
    """

def build_vectorstore_fingerprint(vectorstore_path: Path) -> dict:
    """
    index.faiss/index.pkl 두 파일의 SHA-256만 계산한다. FAISS를 역직렬화하지 않으므로
    임베딩 모델 로드 비용이 없고, allow_dangerous_deserialization과도 무관하다(바이트만 읽음).
    """

def build_reproducibility_metadata(data_dir: Path, vectorstore_path: Path) -> dict:
    """
    Problem.md 3차 리뷰 P1 반영 — corpus/vectorstore를 실제로 사용하는 평가(Retrieval,
    Answer, 통합 baseline)가 공통으로 호출한다. build_corpus_manifest()/
    build_vectorstore_fingerprint()를 감싸 다음을 반환한다:

        {
            "corpus_manifest": [...],          # build_corpus_manifest()["entries"] 그대로
            "corpus_manifest_sha256": "...",
            "vectorstore_fingerprint": {"index_faiss_sha256": "...", "index_pkl_sha256": "..."},
            "reproducibility_note": None,
        }

    Problem.md 4차 리뷰 P1 반영 — 이전 버전은 corpus_manifest_sha256(집계 해시)만 반환해서
    "어떤 파일이 바뀌었는지" 리포트만으로 확인할 수 없었다. corpus_manifest 배열 전체를
    포함해 M2-REQ-010의 "파일별 정규화 source ID, 크기, SHA-256" 요구를 리포트 자체로
    충족한다(별도 manifest 파일을 만들지 않음 — data/가 소규모라 배열이 작고, 상대 경로
    참조가 리포트 이동 시 깨지는 문제를 피할 수 있음).

    data_dir/vectorstore_path가 없으면 FileNotFoundError를 그대로 던진다 — 호출부(각
    evaluator의 main())가 기존 오류 처리 정책(사람이 읽을 오류 + exit(2))으로 잡는다.
    """

def build_not_applicable_reproducibility_metadata(reason: str) -> dict:
    """
    Routing처럼 corpus/vectorstore를 쓰지 않는 evaluator가 호출한다. 위와 동일한 키 집합을
    갖되 corpus_manifest/corpus_manifest_sha256/vectorstore_fingerprint는 None,
    reproducibility_note에 reason을 채운다 — 모든 리포트가 evaluator 종류와 무관하게 같은
    schema를 갖도록 한다.
    """
    return {
        "corpus_manifest": None,
        "corpus_manifest_sha256": None,
        "vectorstore_fingerprint": None,
        "reproducibility_note": reason,
    }
```

메타데이터 필드(M2-REQ-010 최소 목록 + Problem.md 1차/3차/4차 리뷰로 추가된 재현성 필드):

- 기존: `schema_version`, `generated_at_utc`, `git_commit`, `git_dirty`, `dataset_path`, `dataset_sha256`, `python_version`, `command`, `embedding_model`(`config.EMBEDDING_MODEL_NAME`), `reranker_model`(`config.RERANKER_MODEL`, `USE_RERANKER` 반영), `ollama_model`(`config.OLLAMA_MODEL`), `retrieval_config`(`USE_HYBRID_SEARCH`, `USE_MMR`, `USE_RERANKER`, `MMR_K`, `MMR_LAMBDA`, `RRF_TOP_K`, `BM25_TOP_K`, `DENSE_TOP_K`, `RERANKER_TOP_K`, `RETRIEVAL_K` 중 활성 파이프라인에 해당하는 값), `case_counts`(`total`/`success`/`failure`/`excluded`)
- 추가: `corpus_manifest`(파일별 `source_id`/`size_bytes`/`sha256` 배열, 4차 리뷰 반영), `corpus_manifest_sha256`(배열 전체의 집계 해시), `vectorstore_fingerprint`(`index_faiss_sha256`, `index_pkl_sha256`), `reproducibility_note`, `vectorstore_document_count`(선택 — Phase 4/6/7에서 `RAGEngine`이 이미 초기화된 시점에 `len(engine.vectorstore.docstore._dict)`로 opportunistic하게 채움. 별도로 이 값만을 위해 모델을 로드하지 않는다)

**evaluator별 적용 범위(Problem.md 3차 리뷰 P1 반영, 4차 리뷰 P2로 baseline 항목 구체화 — 요구사항 문서 M2-REQ-010에도 동일하게 반영됨)**:

| Evaluator | 호출 | 필드 값 |
|---|---|---|
| Retrieval (Phase 4) | `build_reproducibility_metadata()` | 필수, non-null. `data/`/`vectorstore/` 없으면 기존과 동일하게 실행 자체가 실패 |
| Answer (Phase 6) | `build_reproducibility_metadata()` | 필수, non-null. 이유는 동일(`RAGEngine.query()`가 실제로 corpus/vectorstore를 사용) |
| Routing (Phase 5) | `build_not_applicable_reproducibility_metadata("routing은 corpus/vectorstore를 사용하지 않음")` | `null` + 사유. `data/`/`vectorstore/`가 없어도 Routing 평가는 정상 실행되어야 한다 |
| 통합 baseline (Phase 7) | Retrieval 단계에서 계산한 `build_reproducibility_metadata()` 결과 하나를 **top-level**과 Retrieval 단계 결과에 동일하게 기록하고, Answer 단계 실행 시 별도로 다시 계산한 값과 비교해 **다르면 baseline 전체를 실패 처리**(무결성 invariant). Routing 단계 결과는 그대로 `null`/`not_applicable` 유지 | top-level `corpus_manifest`/`corpus_manifest_sha256`/`vectorstore_fingerprint`는 항상 non-null — 소비자가 단계별 내부를 해석하지 않고 top-level만 봐도 실행 환경을 식별할 수 있어야 함(4차 리뷰 P2) |

**통합 baseline 계약 상세(Problem.md 4차 리뷰 P2 반영)**:

- `run_baseline()`의 최종 JSON/Markdown **top-level**에 `corpus_manifest`/`corpus_manifest_sha256`/`vectorstore_fingerprint`가 non-null로 존재해야 한다(Retrieval 단계 값을 그대로 승격). 이 값을 얻기 위해 소비자가 `stages.retrieval.corpus_manifest_sha256` 같은 하위 경로를 알 필요가 없다.
- `--skip-answers`를 쓰더라도 top-level fingerprint는 Retrieval 단계에서 이미 계산됐으므로 그대로 유지된다.
- Answer 단계가 실행되면 자체적으로도 `build_reproducibility_metadata()`를 호출한다(Phase 6, 독립 실행 시에도 필요하므로). 통합 baseline에서는 이 값과 Retrieval 단계 값의 `corpus_manifest_sha256`/`vectorstore_fingerprint`가 **완전히 일치해야 하며**, 불일치하면(동일 실행 중 `data/`/`vectorstore/`가 바뀌었다는 뜻이므로) `run_baseline()`은 이미 생성된 단계 결과를 보존한 채 비0 종료 코드로 끝난다.
- `evaluation.routing`을 baseline이 아니라 **단독** 실행하면 이 top-level 계약이 아니라 Phase 5의 Routing 리포트 규칙(`null`/`not_applicable`)을 그대로 따른다 — 통합 baseline의 top-level 규칙은 baseline 명령에서 생성된 리포트에만 적용된다.

**한계(리포트에 명시)**: `embedding_model`/`chunk_size`/`chunk_overlap`은 **현재 `config.py` 값**이며, `vectorstore/`가 실제로 그 값으로 생성됐다는 보장은 코드로 확인할 수 없다(인덱스 생성 시점의 설정을 별도로 기록하는 장치가 현재 `document_register.py`에 없음). `corpus_manifest_sha256`/`vectorstore_fingerprint`는 "이번 실행과 다음 실행이 같은 `data/`/`vectorstore/`를 봤는지"를 비교하는 용도로만 신뢰하고, "인덱스가 현재 config와 정합한지"는 보장하지 않는다(인덱스 생성 provenance 부재, Problem.md 4차 리뷰에서도 M2 범위 밖으로 확인됨). 이 갭을 완전히 없애려면 `document_register.py`가 생성 시점 설정을 인덱스와 함께 기록해야 하며, 이는 M2 범위를 벗어나므로(M2는 검색 알고리즘/인덱싱 로직을 바꾸지 않음) Problem.md에 P2 후속 항목으로 남긴다(§7 참고). **Phase 7에서 생성하는 `evaluation/baselines/m2_initial.md`에도 이 한계를 사람이 읽는 문장으로 명시한다** — 리포트 메타데이터 필드로만 남기지 않고, 최초 baseline을 해석하는 사람이 반드시 보게 한다(4차 리뷰 실행 권고 4 반영).

`test_evaluation_reporting.py`(Phase 3)에 report schema 테스트를 추가한다: Retrieval/Answer 스타일 리포트는 `corpus_manifest`/`corpus_manifest_sha256`/`vectorstore_fingerprint`가 모두 non-null이어야 함, Routing 스타일 리포트는 세 필드가 모두 null이고 `reproducibility_note`가 채워져야 함, `build_reproducibility_metadata()`가 `data_dir`/`vectorstore_path` 부재 시 `FileNotFoundError`를 던지는지, `data/` 파일 추가·삭제·내용 변경 시 `corpus_manifest_sha256`이 바뀌고 `corpus_manifest` 배열에서 어떤 `source_id`가 바뀌었는지 확인 가능한지(4차 리뷰 P1). `build_corpus_manifest()`가 정규화 후 같은 `source_id`로 충돌하는 임시 디렉터리(같은 basename의 다른 하위 경로, 대소문자만 다른 이름, NFC/NFD만 다른 이름)를 넣었을 때 `CorpusManifestError`와 충돌 경로 목록을 반환하는지도 여기서 검증한다(5차 리뷰 P2, §3.3). Phase 7 테스트에는 통합 baseline top-level 값이 Retrieval 단계 값과 동일한지, Retrieval/Answer 단계 값이 인위적으로 달라지도록 mock했을 때 `run_baseline()`이 실패 처리하는지를 추가한다(4차 리뷰 P2).

`evaluation/reports/`는 `.gitignore`에 추가한다. `evaluation/baselines/`는 커밋 대상이며 Phase 7 승인 게이트를 통과한 파일만 들어간다.

## 4. Phase별 상세 작업

### Phase 1 — 스키마와 dataset validator

파일: `evaluation/__init__.py`, `evaluation/schema.py`, `evaluation/dataset.py`, `test_evaluation_schema.py`, `test_evaluation_dataset.py`, `requirements.txt`(`pydantic>=2,<3` 추가, §3.2)

`evaluation/dataset.py`:

```python
def load_jsonl(path: Path) -> list[GoldenCase]:
    """줄 단위 파싱. 실패 시 DatasetError(line_number, raw_line, cause) 발생."""

class DatasetError(Exception):
    def __init__(self, message: str, *, case_id: str | None = None, line_number: int | None = None): ...

@dataclass
class ValidationReport:
    total: int
    by_category: dict[str, int]
    document_qa_with_relevant_sources: int   # M2-REQ-002 "문서 QA 중 30개 이상"
    document_qa_with_answer_assertions: int  # M2-REQ-002 "문서 QA 중 20개 이상"
    total_with_expect_abstention: int        # M2-REQ-002 "답변 불가 사례 5개 이상" — category 무관 집계
    korean_ratio: float
    intent_counts: dict[str, int]
    errors: list[str]

def validate_composition(cases: list[GoldenCase]) -> ValidationReport:
    """
    M2-REQ-002의 9개 구성 규칙을 모두 검사. §3.2에서 정정한 대로, 개별 GoldenCase는
    category와 answer_assertions/expect_abstention 조합을 강제하지 않으므로 이 함수가
    유일하게 "document_qa 중 20개 이상 answer_assertions" 같은 category-scoped 집계
    규칙을 검사하는 자리다. 위반 시 errors에 사람이 읽을 수 있는 메시지 누적.
    """

def main(argv: list[str] | None = None) -> int:
    """`validate <path>` 서브커맨드. 오류 시 case id/line number 포함 메시지 stderr 출력 후 1 반환."""

if __name__ == "__main__":
    raise SystemExit(main())
```

한국어 질문 판정: `re.search(r"[가-힣]", question)` (한글 음절 포함 여부).

테스트 목록(`test_evaluation_dataset.py`): 정상 60개 통과, 중복 id, 필수 필드 누락, 잘못된 enum, 빈 질문, 사례 수 미달, 카테고리별 최소치 미달, relevance grade 범위 밖(4 이상/음수), **`document_qa` 중 `answer_assertions` 보유가 20개 미만인 구성(전체 사례 수는 충분해도 실패해야 함)**, **`answer_assertions`도 `expect_abstention`도 없는 순수 retrieval 전용 `document_qa` 사례가 있어도 그 자체는 통과해야 함(§3.2 정정 사항의 회귀 방지 테스트)**, intent 유형 5개 미만, 한국어 비율 80% 미만, Unicode NFC/NFD 파일명이 같은 source로 정규화되는지.

검증:

```bash
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl   # Phase 2 완료 전까지는 실패해도 정상(파일 없음)
pytest -q
```

커밋: `evaluation schema 및 dataset validator 추가`

### Phase 2 — 골든 평가셋 작성 (§5에 상세, 완료)

산출물: `evaluation/datasets/golden.jsonl`(62개 사례), `evaluation/README.md`(작성 가이드 포함, 문서 위치는 Phase 9에서 최종화하되 초안은 여기서 시작).

**사용자 승인 게이트**: 초안 작성 후 정답(카테고리 분포, source 매핑, assertion 문구, abstention 사례)에 대한 사람 검토 2회(§5)를 사용자에게 명시적으로 요청한다. 승인 전에는 Phase 2를 완료 처리하지 않는다(상위 계획 §4 Phase 2 완료 조건, §8 게이트 참고). **완료**: M2_Phase2_code_review_result.md 리뷰에서 발견된 P1 3건(확장자 없는 경제 PDF가 `document_register.py`의 `**/*.pdf`/`**/*.txt` 글롭에 걸리지 않아 인덱싱 누락, 비교·절차형 질문의 독립적 사실이 하나의 `any_of`에 섞여 부분 답변도 통과, `evaluation/README.md` 부재)를 모두 수정한 뒤(파일명에 `.pdf` 확장자 추가 + vectorstore 재생성으로 18개 source 전부 검색 후보에 포함됨을 실제 유사도 검색으로 확인, 8개 사례의 복합 assertion을 독립 `AnswerAssertion` 객체로 분리, README 작성) source relevance·answer assertion 검토를 재요청해 승인받았다.

### Phase 3 — metric과 reporting (완료)

파일: `evaluation/metrics.py`, `evaluation/reporting.py`, `test_evaluation_metrics.py`, `test_evaluation_reporting.py`, `.gitignore`에 `evaluation/reports/` 추가

검증:

```bash
pytest -q
```

**완료**: M2_Phase3_code_review_result.md 리뷰에서 P1 1건("write_report()가 초 단위 timestamp만 써서 같은 evaluator 리포트를 1초 안에 두 번 쓰면 기존 결과를 조용히 덮어씀") + P2 2건(percentile()이 범위 밖 p를 조용히 clamp, corpus manifest 파일 추가/삭제 회귀 테스트 누락) + P3 1건(recall_at_k/mrr_at_k/ndcg_at_k가 k<1을 검증하지 않아 음수 slice로 잘못된 값을 냄)이 발견됐다. write_report()는 마이크로초 정밀도 + 배타적 생성("x" 모드) + 충돌 시 suffix 재시도로 수정했고, percentile()과 세 metric 함수는 잘못된 입력에 ValueError를 던지도록 통일했다. 회귀 테스트 18건을 추가해 재검증했다.

커밋: `평가 metric 및 report 생성기 추가`

### Phase 4 — Retrieval trace와 evaluator

파일: `rag_engine.py`(§3.4 변경), `evaluation/retrieval.py`, `test_evaluation_retrieval.py`(§3.4의 4-분기 characterization test 포함 — 선택 아님, 필수)

`evaluation/retrieval.py`:

```python
def evaluate_retrieval(
    dataset_path: Path,
    output_dir: Path,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    limit: int | None = None,
    tag: str | None = None,
) -> dict:
    """
    get_rag_engine()을 이 함수 내부에서만 호출(지연 로딩, NFR-003). 사례마다
    engine._retrieve_documents(question, trace=RetrievalTrace())를 호출해 docs를 얻고,
    normalize_source_id()로 정규화한 뒤 dedupe_preserve_order()로 중복 제거한 단일
    ranked_ids를 만들어 recall_at_k/mrr_at_k/ndcg_at_k 세 곳에 동일하게 전달한다(§3.5 —
    지표마다 다른 "k" 의미를 갖지 않도록). relevant_sources가 빈 사례는 Recall/MRR에서,
    relevance_grades가 빈 사례는 nDCG에서 각각 제외하고 제외 수를 보고한다(두 제외 카운트는
    서로 다를 수 있음). reporting.build_reproducibility_metadata(DATA_DIR, VECTORSTORE_PATH)를
    호출해 리포트 메타데이터에 포함한다(§3.6 — Retrieval은 corpus/vectorstore를 필수로 사용하므로
    non-null이어야 함).
    """

def main(argv=None) -> int:
    """--dataset --output [--limit] [--tag] 옵션. vectorstore/모델 부재 시
    FileNotFoundError를 잡아 사람이 읽을 오류(원인 + document_register.py 안내) 후 exit(2)."""
```

evaluator 단위 테스트는 `RAGEngine` 대신 `fake retriever`(고정 문서 리스트를 반환하는 더미 객체)를 주입해 모델/벡터스토어 없이 recall/MRR/nDCG 계산 로직과 CLI 파싱만 검증한다(M2-REQ-011 "메트릭 단위 테스트는 모델/vectorstore/Ollama/네트워크 사용 금지"와 동일 원칙을 evaluator 테스트에도 적용). 별도로 `rag_engine.py`의 `_retrieve_documents()` 4-분기 characterization test(§3.4)도 이 Phase에서 함께 작성한다.

검증:

```bash
python -m evaluation.retrieval --help
pytest -q
```

로컬 전용(선택, CI 아님):

```bash
python -m evaluation.retrieval --dataset evaluation/datasets/golden.jsonl --output evaluation/reports/retrieval
```

커밋: `RAG retrieval trace와 retrieval evaluator 추가`

### Phase 5 — Routing evaluator, 기존 사례 통합

파일: `evaluation/routing.py`, `test_evaluation_routing.py`, `test_agent_routing.py`(수정)

```python
def evaluate_routing(
    cases: list[GoldenCase],
    decide_tool: Callable[[str], tuple[str | None, str | None]],
    *,
    measure_latency: bool = True,
) -> dict:
    """offline/live 공통 코어. decide_tool을 주입받아 오프라인은 fixture 함수,
    라이브는 agent._decide_tool을 그대로 넘긴다. corpus/vectorstore를 전혀 사용하지
    않으므로 reporting.build_not_applicable_reproducibility_metadata(
    "routing은 corpus/vectorstore를 사용하지 않음")를 호출한다(Problem.md 3차 리뷰 P1
    반영) — data/나 vectorstore/가 없어도 이 함수는 정상 동작해야 한다."""

def main(argv=None) -> int:
    """--dataset --output --mode {offline,live} [--limit] [--tag].
    --mode live인데 RUN_LIVE_LLM_TESTS=1이 아니면 사용법 안내 후 exit(2)."""
```

**기존 `ROUTING_CASES` 통합 방법**: `test_agent_routing.py`의 16개 사례를 의미 그대로(명시적 웹검색 키워드/암묵적 실시간성/설명/비교/절차/예-아니오/엣지케이스) golden.jsonl에 `tags`로 `"routing_regression"`을 추가해 편입한다. 라이브 회귀 테스트는 전체 60여 개가 아니라 이 태그가 붙은 부분집합만 돌려 기존과 비슷한 실행 시간(16개, 약 1분)을 유지한다.

```python
# test_agent_routing.py 변경 후
from evaluation.dataset import load_jsonl

def _load_routing_regression_cases():
    cases = load_jsonl(Path("evaluation/datasets/golden.jsonl"))
    return [c for c in cases if "routing_regression" in c.tags]

def test_routing_regression_accuracy():
    cases = _load_routing_regression_cases()
    result = evaluate_routing(cases, _decide_tool)
    ...
    assert result["accuracy"] >= MIN_ACCURACY  # 0.8 유지
```

정답의 유일한 원천이 golden.jsonl이 되므로(요구사항 문서 "중복된 정답 소스를 만들지 않아야 함" 충족), `ROUTING_CASES` 하드코딩 리스트는 삭제한다.

검증:

```bash
python -m evaluation.routing --help
pytest -q
RUN_LIVE_LLM_TESTS=1 python -m evaluation.routing --dataset evaluation/datasets/golden.jsonl --mode live --output evaluation/reports/routing
RUN_LIVE_LLM_TESTS=1 pytest test_agent_routing.py -v   # 여전히 opt-in, golden.jsonl 기반으로 동작 확인
```

커밋: `routing evaluator 및 기존 사례 통합`

### Phase 6 — Answer evaluator와 사람 검토 worksheet

파일: `evaluation/answers.py`, `test_evaluation_answers.py`

```python
# 실제 프롬프트 템플릿(prompt_templates.py)에 존재하는 두 공식 거절 문구를 모두 인식한다.
# Problem.md 2차 리뷰 P2 반영 — 초안은 explanation/comparison/procedure/other/uncertain 템플릿의
# 문구만 인식해서, yesno 템플릿이 정상적으로 답변을 거절해도 abstention 오탐(불일치)이 났다.
ABSTENTION_PHRASES = (
    "제공된 문서에서 관련 정보를 찾을 수 없습니다",  # EXPLANATION/COMPARISON/PROCEDURE/DEFAULT_TEMPLATE
    "제공된 문서만으로는 확실한 답변이 어렵습니다",   # YESNO_TEMPLATE (prompt_templates.py:129)
)
# 두 템플릿 세트가 이후 바뀌면 이 목록도 함께 갱신해야 한다(coupling risk, evaluation/README.md에 명시).

from evaluation.schema import is_answer_eval_eligible
# Design.md의 design_review.md 3차 P2 반영 — 이전에는 이 파일에 별도로
# _is_answer_eval_eligible()을 정의했으나, Phase 1 dataset.py의 구성 검증
# (intent 최소 수량 집계)과 정의가 갈라질 위험이 있어 evaluation/schema.py의
# 공개 함수 하나로 통일했다. 이 파일과 dataset.py 둘 다 여기서 import한다.

def evaluate_answers(dataset_path, output_dir, limit=None, tag=None) -> dict:
    """is_answer_eval_eligible()을 만족하는 사례만 대상(category 필터 없음 — Problem.md
    1차 리뷰 반영: 이전 초안은 category == document_qa로 제한해 unanswerable 카테고리의
    abstention 사례가 abstention 정확도 계산에서 누락됐었다).
    get_rag_engine().query()를 사례별로 호출, 개별 실패는 기록 후 계속. source_ids, skipped =
    _extract_returned_source_ids(result["sources"])로 변환한 뒤 _source_match()에 넘긴다
    (Problem.md 3차 리뷰 P2 반영). 사례별 {expected_abstention, predicted_abstention}을
    모아 두었다가 마지막에 _abstention_confusion()으로 집계한다(3차 리뷰 P2 반영).
    reporting.build_reproducibility_metadata(DATA_DIR, VECTORSTORE_PATH)를 호출해 리포트
    메타데이터에 포함한다(3차 리뷰 P1 반영 — Answer도 실제 corpus/vectorstore를 사용하므로
    Retrieval과 동일하게 non-null 필수)."""

def _detect_abstention(answer: str) -> bool:
    normalized = unicodedata.normalize("NFC", answer)
    return any(unicodedata.normalize("NFC", phrase) in normalized for phrase in ABSTENTION_PHRASES)

def _extract_returned_source_ids(sources: list[dict]) -> tuple[list[str], int]:
    """
    Problem.md 3차 리뷰 P2 반영 — RAGEngine.query()/rag_tool.func()가 실제로 반환하는
    sources는 문자열 리스트가 아니라 {"index":.., "source":.., "page":.., "content":..}
    형태의 dict 리스트다(rag_engine.py:456-465 참고). _source_match()는 문자열 리스트를
    받도록 이미 순수하게 설계돼 있으므로(단위 테스트 용이성 유지), 이 함수가 그 사이의
    변환을 전담한다. "source" 키가 없거나 값이 문자열이 아닌 항목은 건너뛰고 skipped
    카운트에 반영한다(레거시로 문자열을 바로 섞어 보내는 것은 지원하지 않음 — 항상
    dict 리스트를 기대한다).

    Returns: (source_ids, skipped_count)
    """
    ids, skipped = [], 0
    for entry in sources:
        source = entry.get("source") if isinstance(entry, dict) else None
        if isinstance(source, str) and source:
            ids.append(source)
        else:
            skipped += 1
    return ids, skipped

def _source_match(returned_sources: list[str], relevant_sources: list[str]) -> dict | None:
    """
    Problem.md 2차 리뷰 P2 반영 — "source 일치율"의 정의를 명시적으로 고정한다. 인자는
    이미 _extract_returned_source_ids()를 거친 순수 문자열 리스트여야 한다(위 함수가 dict
    → str 변환을 전담하므로 이 함수는 여전히 list[str]만 받는 순수/테스트하기 쉬운
    형태로 유지한다). relevant_sources가 비어 있으면(예: 순수 abstention 사례) 이 사례는
    source 평가에서 제외하고 None을 반환한다 — 호출부가 이를 source_evaluation_excluded
    카운트에 반영한다. 양쪽 다 normalize_source_id() + dedupe_preserve_order()를 거친 뒤
    비교한다(Retrieval 지표와 동일한 정규화/중복 제거 함수 재사용, §3.5).
    """
    if not relevant_sources:
        return None
    returned = set(dedupe_preserve_order([normalize_source_id(s) for s in returned_sources]))
    relevant = set(dedupe_preserve_order([normalize_source_id(s) for s in relevant_sources]))
    return {
        "source_any_hit": bool(returned & relevant),
        "source_recall": len(returned & relevant) / len(relevant),
    }

def _abstention_confusion(flags: list[tuple[bool, bool]]) -> dict:
    """
    Problem.md 3차 리뷰 P2 반영 — "abstention 정확도"를 Answer 평가 대상 전체(사례가
    expect_abstention=True인지 여부와 무관하게 is_answer_eval_eligible()을 만족하는
    모든 사례)에 대한 이진 분류로 정의한다. flags는 사례마다
    (expected_abstention, predicted_abstention) 튜플이다.

        TP: expected=True,  predicted=True   (정상적으로 거절)
        TN: expected=False, predicted=False  (정상적으로 답변)
        FP: expected=False, predicted=True   (답변 가능한데 잘못 거절)
        FN: expected=True,  predicted=False  (거절해야 하는데 답변을 시도)

    accuracy = (TP+TN)/N. N=0이면 accuracy=None이고 별도 필드
    abstention_accuracy_excluded_reason에 "Answer 평가 대상 사례 없음"을 기록한다.
    """
    tp = sum(1 for e, p in flags if e and p)
    tn = sum(1 for e, p in flags if not e and not p)
    fp = sum(1 for e, p in flags if not e and p)
    fn = sum(1 for e, p in flags if e and not p)
    n = len(flags)
    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": (tp + tn) / n if n > 0 else None,
        "abstention_accuracy_excluded_reason": None if n > 0 else "Answer 평가 대상 사례 없음",
    }

def _fence_for(text: str) -> str:
    """
    Problem.md 2차 리뷰 P2 반영 — 답변 자체에 3중 backtick(또는 그보다 긴 연속 backtick)
    코드블록이 포함되면 고정된 3틱 fence로는 worksheet 구조가 깨질 수 있다. 답변에 등장하는
    가장 긴 연속 backtick 길이보다 1 긴 fence를 동적으로 생성한다(표준 CommonMark 규칙).
    """
    longest_run = 0
    current = 0
    for ch in text:
        if ch == "`":
            current += 1
            longest_run = max(longest_run, current)
        else:
            current = 0
    return "`" * max(longest_run + 1, 3)

def write_review_worksheet(results: list[dict], output_path: Path) -> Path:
    """
    Problem.md 1차 리뷰 P2 반영: 답변 전체를 Markdown 표의 한 셀에 넣지 않는다(줄바꿈, `|`,
    표/코드블록이 섞인 LLM 답변이 표 구조를 깨뜨릴 수 있음). 대신 사례별 섹션 형식을 쓴다.
    모델 답변은 _fence_for(answer)로 계산한 동적 길이 fence로 감싼다(2차 리뷰 P2 반영).

    각 사례마다:
        ## {id}
        **질문**: {question}
        **자동 점수**: assertion {passed}/{total} · abstention 일치 {bool} ·
                       source_any_hit {bool} · source_recall {ratio 또는 "N/A(제외)"}
        **반환 출처**: {source list, 한 줄 bullet}
        **기대 핵심 사실**: {answer_assertions 요약, 한 줄 bullet}
        **모델 답변**:
        {fence}
        {raw answer 그대로}
        {fence}
        **Faithfulness (1-5)**: _(빈칸)_
        **Relevance (1-5)**: _(빈칸)_
        **Completeness (1-5)**: _(빈칸)_
        **Citation correctness (1-5)**: _(빈칸)_
        **Reviewer note**: _(빈칸)_
        ---
    """
```

주의(M2-REQ-015): worksheet/JSON 어디에도 검색된 원문 chunk 전체 본문을 저장하지 않는다 — 저장하는 것은 모델이 생성한 답변 텍스트, 반환된 source 파일명/페이지, assertion 자동 채점 결과뿐이다.

단위 테스트는 `get_rag_engine`을 mock 결과로 대체해 assertion coverage/abstention/source match/intent 정확도 계산 로직만 검증한다. mock의 `sources`는 **실제 `RAGEngine.query()`와 동일한 dict 리스트 형태**를 사용한다(3차 리뷰 P2 반영 — 이전 mock 예시가 문자열 리스트를 흉내 내 실제 반환 schema와 달랐다):

```python
{
    "answer": "...",
    "sources": [{"index": 1, "source": "example.pdf", "page": 3, "content": "..."}],
    "success": True,
    "intent": "explanation",
}
```

추가로 다음을 명시적으로 검증한다:

- `is_answer_eval_eligible()`의 4가지 eligibility 조합(assertion만/abstention만/둘 다/둘 다 없음)은 `evaluation/schema.py`의 `TestIsAnswerEvalEligible`(Phase 1, Design.md §6.1)에서 이미 검증한다 — Phase 6에서는 로직을 다시 테스트하지 않고, `evaluate_answers()`가 실제로 `evaluation.schema.is_answer_eval_eligible`을 **import해서** 쓰고 있는지(중복 재정의가 아닌지)만 통합 테스트로 확인한다(design_review.md Design.md 3차 P2 반영).
- `_detect_abstention()`: 두 공식 문구(explanation류, yesno류) 각각을 포함한 답변이 모두 abstention으로 인식되는지(2차 리뷰 버그의 회귀 방지).
- `_extract_returned_source_ids()`: `source` 키가 없는 dict, `source` 값이 문자열이 아닌 dict가 각각 건너뛰어지고 `skipped` 카운트에 반영되는지(3차 리뷰 P2 반영).
- `_source_match()`: `relevant_sources`가 빈 사례에서 `None`을 반환하는지(제외 처리), 부분 일치/완전 일치/불일치 각각에서 `source_recall` 값이 정의대로 나오는지.
- `_abstention_confusion()`: TP/TN/FP/FN 각 케이스가 최소 1개씩 올바르게 집계되는지, `flags`가 빈 리스트일 때 `accuracy=None`과 사유 문자열이 채워지는지(3차 리뷰 P2 반영).
- `_fence_for()`: 답변에 3중 backtick, 4중 이상 backtick이 섞인 입력에서도 반환된 fence로 감쌌을 때 worksheet 전체가 유효한 Markdown으로 파싱되는지.
- `evaluate_answers()`가 반환하는 리포트의 `corpus_manifest_sha256`/`vectorstore_fingerprint`가 non-null인지(3차 리뷰 P1 반영 — `get_rag_engine`과 함께 `reporting.build_reproducibility_metadata`도 mock).

검증:

```bash
python -m evaluation.answers --help
pytest -q
```

커밋: `answer evaluator와 review worksheet 추가`

### Phase 7 — 통합 baseline과 최초 측정

파일: `evaluation/baseline.py`

```python
def run_baseline(
    dataset_path: Path,
    output_dir: Path,
    *,
    skip_routing: bool = False,
    skip_answers: bool = False,
    limit: int | None = None,
    tag: str | None = None,
) -> dict:
    """순서: dataset validate → retrieval → (skip 아니면) live routing →
    (skip 아니면) answers. 각 단계 결과를 계속 누적하고, 실패한 단계가 있어도
    이미 끝난 단계 결과는 보존한 채 마지막에 비0 종료 코드로 반환.

    retrieval 단계에서 계산한 reporting.build_reproducibility_metadata() 결과를 최종
    리포트의 top-level corpus_manifest/corpus_manifest_sha256/vectorstore_fingerprint로
    승격한다(Problem.md 4차 리뷰 P2). answers 단계는 (Phase 6에서 독립 실행도 지원해야
    하므로) 자체적으로 다시 build_reproducibility_metadata()를 호출하며, run_baseline()은
    이 값과 retrieval 단계 값의 corpus_manifest_sha256/vectorstore_fingerprint가 일치하는지
    검사한다 — 불일치하면(동일 실행 도중 data/나 vectorstore/가 바뀌었다는 뜻) 이미 만든
    단계 결과는 보존한 채 비0 종료 코드로 끝난다. routing 단계 결과는 그대로
    null/not_applicable 유지(§3.6 evaluator별 적용 범위 표)."""
```

실행(사용자 환경, 실제 Ollama/vectorstore 필요):

```bash
RUN_LIVE_LLM_TESTS=1 python -m evaluation.baseline \
  --dataset evaluation/datasets/golden.jsonl \
  --output evaluation/reports
```

**승인 게이트**: 실행 결과(전체 요약 수치 + 주요 실패 사례 목록)를 사용자에게 제시하고, 승인 후에만 `evaluation/baselines/m2_initial.json`/`.md`로 고정한다. 승인 전 임의로 "최초 baseline"이라고 확정하지 않는다.

커밋: `통합 baseline과 최초 결과 추가` (baseline JSON/MD는 승인 이후 별도 커밋 또는 같은 커밋에 포함 — 사용자 승인 시점에 결정)

### Phase 8 — CI

파일: `.github/workflows/ci.yml`

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [master]

jobs:
  python-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"
          cache-dependency-path: requirements.txt
      - run: pip install -r requirements.txt
      - run: python -m pip check
      - run: python -c "import web_server"   # Phase 0에서 발견된 email-validator 문제의 회귀 감지 (§1)
      - run: pytest -q
      - run: python -m evaluation.dataset validate evaluation/datasets/golden.jsonl

  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"
      - run: npm ci
      - run: npm test
      - run: npm run sync-vendor
      - run: git diff --exit-code static/vendor/
```

- `pytest -q`가 `test_evaluation_metrics.py` 등 metric 단위 테스트를 이미 포함하므로 별도 job 불필요.
- live routing/answer/baseline은 어떤 job에도 없음 — Ollama/vectorstore 불필요 확인은 로컬에서 `unset OLLAMA_BASE_URL` 등으로 별도 확인하거나, CI 환경 자체가 Ollama 미설치이므로 자연히 검증됨.
- `python-tests`가 `requirements.txt` 전체(`torch`, `sentence-transformers`, `faiss-cpu` 포함)를 설치해야 하는 이유: 기존 mock 기반 테스트(`test_agent.py` 등)도 `agent.py → rag_engine.py`를 import 시점에 실제로 로드하므로, 무거운 라이브러리 설치 자체는 피할 수 없다(모델 **가중치 다운로드**는 발생하지 않음 — NFR-003 준수). NFR-004의 "10분 이내"는 이 설치 시간까지 포함한 목표이며, `cache: pip`로 반복 실행 시간을 줄인다. 실측 후 10분을 넘기면 §7 위험에 기록하고 캐시 전략을 재검토한다.

검증: PR을 열어 두 job이 실제로 성공하는지 확인.

커밋: `Python 및 frontend CI 추가`

### Phase 9 — 문서 및 마일스톤 종료

- `README.md` "테스트 방법" 섹션에 dataset validate, 개별 evaluator 실행, 통합 baseline, live 전제조건, 리포트 위치, CI/로컬 차이 추가.
- `evaluation/README.md` 신설: 골든셋 작성 가이드(§5 내용 정리), source ID 규칙(§3.3), metric 정의(§3.5), 결과 해석의 한계(assertion coverage ≠ 진실성, LLM judge 미포함 등).
- `Roadmap.md`의 M2 상태를 "계획 완료 / 착수 대기" → "완료"로 갱신.
- `Problem.md`의 "P1 — 품질 기준선과 CI 부재" 항목 제거(해결됨) 또는 남은 범위(있다면)로 축소.
- 요구사항 문서 §10 추적표에 실제 구현/테스트/리포트 링크 채움.
- 최종 검증 전체 실행(§본 문서 상위 계획 Phase 9 명령 그대로) + live 검증 1회.

커밋: `M2 사용법과 완료 상태 문서화`

## 5. 골든셋 작성 상세 지침 (Phase 2 확장)

### 5.1 현재 corpus 개요 (`data/`, 18개 파일)

| 그룹 | 파일 | 언어 | 비고 |
|---|---|---|---|
| RAG/LangChain 기술 문서 (시리즈) | Retrieval-Augmented Generation (RAG), 검색기(Retriever), 리랭커(Reranker), 벡터스토어(Vector Store) 저장, 임베딩(Embedding), 체인(Chain) 생성, 출력파서(Output Parser), 텍스트 분할(Text Splitter), 프롬프트(Prompt), 도큐먼트 로드(Document Loader), LangGraph-개요 | 한국어(용어 일부 영어) | 이 프로젝트의 핵심 도메인. `document_qa` 대부분을 여기서 뽑음 |
| AI/LLM 개념 | LLM (Large Language Model), google-ai-agents-whitepaper, SPRI_AI_Brief_2023년12월호_F | 한국어+영어 혼재 | google whitepaper는 영어 원문 가능성 높음 — 한국어 80% 룰에 영향 없도록 질문 자체는 한국어로 작성 |
| 경제/정책 (도메인 밖) | 2025 KB 부동산 보고서, 2025년 한국 경제 전망, 2025_한국정부_부동산정책_정리, 2025_APEC_정상회담_주요_협의사항 | 한국어 | boundary/unanswerable, 그리고 intent 다양성(yesno/procedure 등) 확보용으로 유용. 대형 PDF(6MB, 1MB)라 표/스캔 품질 편차 가능 — 원문 확인 필수 |

### 5.2 60개 사례 배분안 (요구사항 문서 최소치 정확히 충족 + 여유분)

| category | 목표 수 | 근거 |
|---|---|---|
| `document_qa` | 42 | REQ 최소 40 + 여유 2 |
| `web_search` | 10 | REQ 최소 10 |
| `boundary` / `unanswerable` | 10 | REQ 최소 10 (`boundary` 3~4 + `unanswerable` 6~7 권장 — abstention 5개 이상은 주로 `unanswerable`에서 확보) |
| 합계 | 62 | REQ 최소 60 |

`document_qa` 42개 내부 배분(중복 허용, 한 사례가 여러 기준 동시 충족 가능). §3.2 정정에 따라 이 42개 전부가 assertion/abstention을 가질 필요는 없다 — **22개는 assertion을 갖고 나머지 약 20개는 `relevant_sources`만 있는 순수 Retrieval 평가 전용 사례로 남아도 정상**이다:

- Retrieval 정답 포함(`relevant_sources`): 32개 이상 (REQ 최소 30)
- 답변 핵심 사실 assertion 포함: 22개 이상 (REQ 최소 20)
- **intent 유형(explanation/comparison/procedure/yesno) 각 5개 이상: 반드시 assertion을 가진 22개 중에서 확보한다(Design.md의 design_review.md 2차 P1, 요구사항 문서 M2-REQ-002 동기화 반영).** Phase 6의 intent 정확도는 Answer 평가 대상(assertion 보유 또는 abstention) 사례에서만 측정되므로, 나머지 20개 Retrieval 전용 사례에 intent를 붙여도 `dataset.py`의 구성 검증(§3.1 Phase 1, `is_answer_eval_eligible()` 기준 집계)을 통과시킬 수 없다. 22개 안에서 4개 유형이 각 5개 이상 자연스럽게 나오도록 질문을 배분한다(예: explanation="RAG에서 MMR이 뭐야?", comparison="BM25와 Dense Retrieval의 차이는?", procedure="문서를 벡터스토어에 등록하는 절차는?", yesno="LangGraph는 LangChain과 별도 설치가 필요한가요?")
- `relevance_grades`(nDCG용, 0~3): document_qa 중 최소 15개에는 주 출처(3) + 관련 출처(1~2) 등급을 함께 부여해 nDCG가 실제로 의미 있게 계산되도록 함(REQ에는 최소 수량이 명시되어 있지 않지만, 표본이 너무 적으면 nDCG 지표가 사실상 무의미하므로 계획 단계에서 목표치를 정함). **양수 등급(1~3)을 받은 모든 source는 정규화 후 `relevant_sources`에도 포함해야 한다**(Design.md `GoldenCase.positive_grades_are_in_relevant_sources`가 저작 시점에 강제) — grade 0(비관련/방해) source만 `relevant_sources` 밖에 자유롭게 추가할 수 있다.

`web_search` 10개: "오늘/최신/지금" 등 실시간성 키워드 위주로 작성하고, `answer_assertions`/`relevant_sources`는 부여하지 않는다(M2-REQ-015 — 웹 검색은 라우팅 평가까지만).

`boundary`/`unanswerable` 10개: 기존 `test_agent_routing.py`의 "이 문서에서 관련 내용을 찾아줘" 같은 엣지 케이스(boundary)와, corpus에 없는 사실을 묻는 질문(unanswerable, `expect_abstention=true`)을 혼합.

### 5.3 작성 절차 (Plan.md §작성 원칙 그대로 적용)

1. `evaluation/dataset.py`의 `normalize_source_id()` 기준으로 `data/` 18개 파일의 정규화 ID 목록을 먼저 뽑아 둔다(다음 커맨드로 초안 확인 가능: `python -c "import os,unicodedata;[print(unicodedata.normalize('NFC', os.path.basename(f))) for f in os.listdir('data')]"`).
2. Claude Code가 각 사례에 대해 **원문 문서를 실제로 읽고** 질문/정답/assertion 초안을 작성한다(추측 금지).
3. 초안을 `evaluation/datasets/golden.jsonl`에 작성 후 `python -m evaluation.dataset validate`로 구성 규칙 통과를 자동 확인한다.
4. 사람 검토 2회를 사용자에게 요청한다:
   - **1차: source relevance 검토** — 각 `relevant_sources`/`relevance_grades`가 실제로 맞는지
   - **2차: answer assertion 검토** — 각 `answer_assertions.any_of`가 실제 문서 내용과 일치하고, 표현이 지나치게 좁지 않은지(동의어 포함 여부)
5. 승인 전에는 Phase 2를 완료 처리하지 않는다.

## 6. CI 상세는 §4 Phase 8 참고. 추가 위험

- `requirements.txt` 전체 설치로 인한 실행 시간이 NFR-004 목표(10분)를 넘길 가능성 — 1차 구현 후 실측하고, 초과 시 `actions/cache`로 pip 캐시를 더 적극적으로 활용하거나 job을 분리(예: 무거운 의존성 설치와 dataset validate만 별도 lightweight job으로 분리)하는 것을 검토한다. 이번 계획에서는 우선 단일 job으로 구현하고 실측치를 Phase 8 완료 보고에 기록한다.
- `gpt-oss:20b`는 로컬에 확인됨 — live baseline 실행 자체는 막히지 않는다. 다만 `ollama list`에 다른 대형 모델(120b 등)도 있어 동시 실행 시 메모리 경합 가능성이 있으므로, live baseline 실행 전 다른 Ollama 세션이 없는지 확인한다.

## 7. 위험과 대응

상위 계획(§6)의 5개 위험(골든셋 품질, Unicode 파일명, live 평가 비용/변동성, production 로직 분기, 문자열 기반 Answer 평가 한계)을 그대로 따르며, 이 문서에서 각각의 **구체적 구현 대응**을 §3.3(source ID), §3.4(단일 로직 + `trace=None` 기본값), §3.6/Phase 6(assertion coverage 정의와 worksheet 병행)에 이미 반영했다. 추가로:

- **CI 실행 시간 초과**: §6에 기록.
- **golden.jsonl과 기존 `ROUTING_CASES`의 의미 손실**: Phase 5에서 태그(`routing_regression`)로 원래 16개의 의도를 보존하며 이관하므로, 이관 시 원래 질문/기대값을 1:1로 대조하는 체크리스트를 Phase 5 PR 설명에 포함한다.
- **공유 개발 환경의 의존성 오염이 저장소 자체 문제로 오인될 위험**(Problem.md P1 리뷰로 발견): 현재 공유 conda 환경에서 `import web_server`가 `email-validator` 버전 문제로 실패하고 `pip check`가 9건을 보고한다(§1에 실측 기록). 이 중 저장소 `requirements.txt`와 직접 관련된 것은 `email-validator`뿐일 가능성이 높지만, **깨끗한 venv로 확인하기 전까지는 단정하지 않는다.** Phase 0 체크리스트(§1)와 CI의 `import web_server`/`pip check` 스텝(Phase 8)으로 이중으로 감시한다.
- **인덱스 재현성 갭**: `corpus_manifest_sha256`/`vectorstore_fingerprint`(§3.6)는 "같은 파일을 봤는지"는 보장하지만 "현재 config로 생성됐는지"는 보장하지 못한다. `document_register.py`가 생성 시점 설정을 기록하지 않는 한 완전히 닫히지 않는 갭이며, M2 범위를 벗어나므로 Phase 9에서 Problem.md에 P2 후속 항목으로 남긴다.

## 8. 승인 게이트 체크리스트

- [x] (착수 전, 권장) PR #3 머지 후 이 문서 작성에 사용한 재구조화 변경(README/Roadmap/Problem 등)을 정리해 M2 브랜치를 깨끗하게 시작 — PR #4/#5로 반영·병합 완료
- [x] Phase 0: 깨끗한 venv에서 `import web_server`/`TestClient`/`pip check` 실행 완료 — 모두 통과, 저장소 자체 문제 아님으로 결론(§1). `requirements.txt` 선행 수정 불필요
- [x] Phase 2: 골든셋 source relevance 검토 승인 (사용자) — M2_Phase2_code_review_result.md P1(확장자 없는 경제 PDF 인덱싱 누락) 반영 후 재검토·승인
- [x] Phase 2: 골든셋 answer assertion 검토 승인 (사용자) — M2_Phase2_code_review_result.md P1(복합 assertion 분리) 반영 후 재검토·승인
- [ ] Phase 7: 최초 live baseline 결과 및 주요 실패 사례 검토·승인 (사용자)

## 9. 완료 정의 체크리스트 (요구사항 문서 §9 그대로, 진행 시 체크)

- [x] 골든 평가셋(최소 60개) schema/구성 검증 통과 — Phase 2 완료(62개 사례, 사람 검토 2회 승인)
- [ ] Retrieval/Routing/Answer 평가 명령 + 통합 baseline 명령 제공
- [ ] 필수 metric·경계 조건 단위 테스트 통과
- [ ] 실제 로컬 환경 최초 baseline JSON/Markdown 생성 및 고정 경로 저장(승인 후)
- [ ] 리포트에 요구된 실행 환경/설정 메타데이터 포함
- [ ] GitHub Actions에서 Python/dataset/frontend 검증이 외부 모델 없이 통과
- [ ] 기존 Python/프런트엔드 테스트 전부 통과
- [ ] README/Roadmap/Problem 문서가 구현 상태와 일치
- [ ] Retrieval 알고리즘/기존 API 의미 의도치 않게 변경되지 않음
- [ ] 리뷰에서 요구사항별 증거 확인 가능(§2 매트릭스 + PR 설명의 추적표)

## 10. 요구사항 추적표 (초기 상태 — 착수 시점)

| 요구사항 | 상태 | 구현/테스트/리포트 증거 |
|---|---|---|
| M2-REQ-001~004 | 계획됨 | Phase 1, §3.1~3.3 |
| M2-REQ-005~006 | 계획됨 | Phase 4, §3.4~3.5 |
| M2-REQ-007 | 계획됨 | Phase 5 |
| M2-REQ-008~010 | 계획됨 | Phase 6, 7, §3.6 |
| M2-REQ-011~012 | 계획됨 | Phase 0(기준), 3, 4 |
| M2-REQ-013 | 계획됨 | Phase 8 |
| M2-REQ-014~016 | 계획됨 | Phase 9, 각 Phase의 CLI 오류 처리 |
| M2-NFR-001~005 | 계획됨 | §3.1, 3.4, 3.6, Phase 8 |

구현이 진행되며 각 셀을 "완료"로 갱신하고 실제 커밋/테스트 파일/리포트 경로로 채운다.
