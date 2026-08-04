# M2.5 Repository Restructuring 상세 이전 계획

상태: **완료** — Phase 0~5 구현, PR #12 GitHub Actions와 사용자 최종 승인 완료

## 1. 목적

M2.5는 루트에 혼재한 제품 코드, 테스트, 문서, 학습 자산과 런타임 데이터를 역할별 경계로 재배치하고, Python 패키지·경로·실행 명령을 일관된 계약으로 정리하는 중간 마일스톤입니다.

이 작업은 기능 개발이나 품질 최적화가 아닙니다. M2에서 승인한 동작과 기준선을 보존하면서 M3 개발이 예측 가능한 구조에서 진행되도록 만드는 것이 목적입니다.

## 2. 핵심 원칙

1. **동작 보존**: 검색, 라우팅, 답변, Web UI와 평가 결과의 의미를 바꾸지 않습니다.
2. **역할 기반 배치**: 파일 형식이 아니라 제품 코드, 평가, 테스트, 문서, 학습, 런타임 자산이라는 책임으로 분류합니다.
3. **명시적 경로**: 현재 작업 디렉터리에 의존하는 `./data`, `./vectorstore` 같은 경로를 제거합니다.
4. **표준 Python 패키지**: 제품 코드는 `src/simple_qna_rag` 아래에 두고 절대 import를 사용합니다.
5. **안전한 데이터 이전**: Git에서 제외된 문서·인덱스를 자동 삭제하거나 덮어쓰지 않습니다.
6. **추적 가능한 이동**: 가능한 한 `git mv`에 해당하는 순수 이동과 내용 변경을 분리합니다.
7. **단계별 검증**: 각 Phase는 독립적으로 테스트 가능해야 하며 실패한 상태에서 다음 Phase로 넘어가지 않습니다.
8. **M2 기준선 불변**: 승인된 baseline의 수치, dataset 내용과 corpus/vectorstore fingerprint를 소급 수정하지 않습니다.

## 3. 현재 구조의 문제

- 저장소 루트에 제품 Python 모듈, CLI, 테스트와 완료된 M2 문서가 함께 있습니다.
- 테스트 13개가 제품 모듈과 같은 루트에 있어 범위와 종류가 드러나지 않습니다.
- `config.py`, 템플릿 로딩, 평가 코드와 테스트가 저장소 루트 또는 현재 작업 디렉터리를 암묵적으로 전제합니다.
- Git 관리 대상인 평가셋·승인 baseline과 Git 제외 대상인 원본 문서·vectorstore·상세 report의 구분이 디렉터리 최상위에서 명확하지 않습니다.
- intent 학습 코드, 학습 데이터와 학습된 모델 artifact의 경계가 없습니다.
- 완료된 마일스톤 계획과 리뷰 문서가 현재 운영 문서처럼 루트에 노출됩니다.
- Python 프로젝트 메타데이터와 설치 가능한 package 계약이 없습니다.

## 4. 목표 디렉터리 구조

```text
simple-qna-rag/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── package.json
├── package-lock.json
├── vitest.config.js
│
├── src/
│   └── simple_qna_rag/
│       ├── __init__.py
│       ├── config.py
│       ├── agent.py
│       ├── rag_engine.py
│       ├── query_router.py
│       ├── intent_classifier.py
│       ├── prompt_templates.py
│       ├── tools.py
│       ├── web_search.py
│       ├── cli/
│       │   ├── query.py
│       │   └── index_documents.py
│       └── web/
│           └── server.py
│
├── evaluation/
│   ├── README.md
│   ├── *.py
│   ├── datasets/
│   ├── baselines/
│   └── reports/                 # Git 제외
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── frontend/
│
├── web/
│   ├── templates/
│   └── static/
│       └── vendor/
│
├── training/
│   └── intent_classifier/
│       ├── datasets/
│       ├── generate.py
│       └── train.py
│
├── models/
│   └── intent_classifier/       # 버전 관리되는 모델 설정·가중치
│
├── runtime/                     # 전체 Git 제외
│   ├── documents/
│   └── vectorstore/
│
├── docs/
│   ├── Roadmap.md
│   ├── Problem.md
│   ├── architecture/
│   │   └── Repository_Structure.md
│   ├── milestones/
│   │   ├── m2-quality-baseline/
│   │   └── m2.5-repository-restructuring/
│   └── reviews/
│       └── m2-quality-baseline/
│
├── scripts/
│   └── sync-vendor.js
└── .github/workflows/ci.yml
```

### 구조 결정

- `README.md`, `LICENSE`, packaging과 Node 설정만 루트에 유지합니다.
- 제품 Python 코드는 install 가능한 `simple_qna_rag` package로 만듭니다.
- `evaluation/`은 제품 런타임과 수명주기가 다른 독립 품질 도구이므로 최상위 subsystem으로 유지합니다.
- Web template/static은 Python 코드와 분리하되 package 설정으로 배포 포함 여부를 명시합니다.
- 버전 관리되는 intent 모델과 데이터는 각각 `models/`, `training/`에 둡니다.
- 사용자 원본 문서와 재생성 가능한 vectorstore만 `runtime/`에 두고 전체를 Git에서 제외합니다.
- timestamped 평가 report는 계속 `evaluation/reports/`에 두어 승인 baseline과 가까운 위치를 유지합니다.

## 5. 파일 이전 매핑

### 5.1 제품 코드

| 현재 | 목표 |
|---|---|
| `agent.py` | `src/simple_qna_rag/agent.py` |
| `config.py` | `src/simple_qna_rag/config.py` |
| `rag_engine.py` | `src/simple_qna_rag/rag_engine.py` |
| `query_router.py` | `src/simple_qna_rag/query_router.py` |
| `intent_classifier.py` | `src/simple_qna_rag/intent_classifier.py` |
| `prompt_templates.py` | `src/simple_qna_rag/prompt_templates.py` |
| `tools.py` | `src/simple_qna_rag/tools.py` |
| `web_search.py` | `src/simple_qna_rag/web_search.py` |
| `web_server.py` | `src/simple_qna_rag/web/server.py` |
| `document_query_cli.py` | `src/simple_qna_rag/cli/query.py` |
| `document_register.py` | `src/simple_qna_rag/cli/index_documents.py` |

### 5.2 테스트와 Web 자산

| 현재 | 목표 |
|---|---|
| 루트 `test_*.py` | `tests/unit/` 또는 외부 경계가 있는 경우 `tests/integration/` |
| `frontend_tests/` | `tests/frontend/` |
| `templates/` | `web/templates/` |
| `static/` | `web/static/` |

테스트 분류는 실제 dependency 기준으로 결정합니다. mock/fake만 쓰는 테스트는 `unit`, 여러 제품 모듈 또는 FastAPI 경계를 함께 검증하면 `integration`으로 분류합니다. live Ollama·네트워크 테스트는 기본 pytest 실행에서 계속 opt-in이어야 합니다.

### 5.3 학습과 모델 자산

| 현재 | 목표 |
|---|---|
| `generate_intent_dataset.py` | `training/intent_classifier/generate.py` |
| `train_intent_classifier.py` | `training/intent_classifier/train.py` |
| `train_intent_classifier.sh` | `training/intent_classifier/train.sh` |
| `intent_dataset/` | `training/intent_classifier/datasets/` |
| `intent-bge-m3-softmax/` | `models/intent_classifier/` |

### 5.4 런타임 데이터

| 현재 | 목표 | 정책 |
|---|---|---|
| `data/` | `runtime/documents/` | 사용자 원본, Git 제외, 보존 우선 |
| `vectorstore/` | `runtime/vectorstore/` | 재생성 가능하지만 자동 삭제 금지 |

런타임 자산은 Git이 추적하지 않으므로 코드 PR의 rename으로 처리할 수 없습니다. 이전 스크립트 또는 문서화된 명령은 대상이 없을 때만 이동하고, 양쪽에 파일이 있으면 중단해야 합니다. 기존 경로는 한 릴리스 동안 경고와 함께 fallback으로 읽을 수 있지만 새 인덱스는 새 경로에만 생성합니다.

### 5.5 문서

| 현재 | 목표 |
|---|---|
| `Roadmap.md` | `docs/Roadmap.md` |
| `Problem.md` | `docs/Problem.md` |
| M2 Requirement/Plan/Development Plan/Design | `docs/milestones/m2-quality-baseline/` |
| M2 상세 개발 지시서 | `docs/milestones/m2-quality-baseline/implementation-guides/` |
| `design_review.md`, M2 code review 문서 | `docs/reviews/m2-quality-baseline/` |
| 본 M2.5 계획 | `docs/milestones/m2.5-repository-restructuring/Plan.md` |

문서 이동 후에는 README, Roadmap, Problem, 평가 README와 각 문서의 상대 링크를 전부 갱신하고 local Markdown link checker를 실행합니다.

## 6. 경로와 실행 계약

### 6.1 설정 경로

- 저장소 루트가 아닌 package 또는 명시적 설정을 기준으로 기본 경로를 계산합니다.
- `SIMPLE_QNA_RAG_DOCUMENTS_DIR`, `SIMPLE_QNA_RAG_VECTORSTORE_DIR`, `SIMPLE_QNA_RAG_MODEL_DIR` 환경변수로 재정의할 수 있게 합니다.
- CLI 인자는 환경변수보다 우선하고, 환경변수는 기본값보다 우선합니다: `CLI > environment > project default`.
- 모든 경로는 시작 시 `Path.resolve()`로 정규화하고 오류 메시지에 실제 해석된 경로를 표시합니다.
- import만으로 디렉터리 생성, 모델 로드, network 접근 또는 vectorstore 접근이 발생해서는 안 됩니다.

### 6.2 Python import와 packaging

- `pyproject.toml`에 `src` layout, Python 3.11, package metadata와 pytest 설정을 정의합니다.
- 제품 코드 import는 `from simple_qna_rag...`로 통일합니다.
- 저장소 루트가 우연히 `sys.path`에 들어가서만 성공하는 import를 금지합니다.
- editable install(`python -m pip install -e .`)과 일반 install 모두 smoke test를 통과해야 합니다.
- 임의의 `sys.path` 조작은 사용하지 않습니다.

### 6.3 공식 실행 명령

최종 명령은 다음 형태를 목표로 합니다.

```bash
simple-qna-rag-web
simple-qna-rag-query
simple-qna-rag-index
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
python -m evaluation.baseline --dataset evaluation/datasets/golden.jsonl --output evaluation/reports
```

CLI entry point 이름과 인자는 `pyproject.toml`에서 고정하고 README와 `--help` 테스트에서 검증합니다. 기존 Python 파일 직접 실행 명령은 migration 안내를 제공한 뒤 제거합니다.

### 6.4 Web 자산

- FastAPI template/static 경로는 현재 작업 디렉터리가 아니라 설정된 project asset 경로에서 계산합니다.
- 저장소 checkout 실행과 설치 package 실행 중 어떤 배포 방식을 지원할지 `pyproject.toml`과 README에 명시합니다.
- HTML, JavaScript와 vendor 파일 내용은 이 마일스톤에서 변경하지 않습니다.

### 6.5 M2 평가 호환성

- `evaluation/datasets/golden.jsonl` 내용과 승인된 `evaluation/baselines/m2_initial.*`은 이동하지 않고 수정하지 않습니다.
- 평가 코드가 제품 package를 새 import 경로로 호출하도록 변경합니다.
- 현재 baseline 내부에 기록된 과거 경로와 commit SHA는 역사적 사실이므로 수정하지 않습니다.
- 구조 변경 후 live baseline 전체 재실행은 필수가 아닙니다. 대신 offline 전체 테스트, dataset hash, baseline hash와 corpus/vectorstore fingerprint 불변을 확인합니다.
- 제품 동작에 영향을 주는 불가피한 변경이 발견되면 M2.5 범위를 중단하고 별도 기능 변경으로 분리합니다.

## 7. 비범위

- Retrieval 알고리즘, chunking, MMR, reranker 또는 prompt 개선
- Agent routing과 Intent Classifier 품질 개선
- dependency upgrade 또는 광범위한 lock 정책 도입
- vectorstore 재생성 방식, provenance 또는 atomic swap 구현
- Docker, 배포 자동화와 운영 관측성 구현
- M2 승인 baseline 수치 재계산 또는 evaluator 규칙 변경
- 문서 내용의 대규모 재작성이나 완료된 리뷰 이력 삭제

현재 구조 때문에 최소한으로 필요한 dependency metadata 변경은 허용하지만 version upgrade는 별도 작업으로 분리합니다.

## 8. 단계별 실행 계획

### Phase 0 — 기준 상태와 이동 계약 고정

상태: **완료** — 결과는 [Phase_0_Baseline.md](Phase_0_Baseline.md) 참조

작업:

1. 시작 commit, branch와 working tree 상태 기록
2. tracked/untracked 파일을 역할별로 inventory화
3. 현재 공식 명령과 exit code 기록
4. dataset, baseline, corpus와 vectorstore fingerprint 기록
5. 전체 offline 검증 실행
6. 이 문서의 목표 구조와 호환 정책 사용자 승인

완료 조건:

- 이동 전 테스트 결과와 fingerprint가 기록되어 있습니다.
- 로컬 runtime 자산의 실제 경로와 충돌 여부를 읽기 전용으로 확인했습니다.
- 구현 중 변경하지 않을 M2 artifact 목록이 확정됐습니다.

### Phase 1 — 문서와 테스트 정리

상태: **완료** — 결과는 [Phase_1_Result.md](Phase_1_Result.md) 참조

작업:

1. `docs/` 구조 생성 및 M2 문서 이동
2. `tests/unit`, `tests/integration`, `tests/frontend`로 테스트 이동
3. Markdown 링크, pytest/vitest 경로와 CI 설정 수정
4. repository structure 문서와 문서 index 작성

완료 조건:

- 루트에는 마일스톤 상세 문서와 `test_*.py`가 남지 않습니다. 제품 Python 모듈 제거는 Phase 2 완료 조건입니다.
- 모든 Markdown local link가 존재합니다.
- 테스트 수와 skip 정책이 이동 전과 같습니다.
- 제품 코드와 런타임 경로는 아직 변경하지 않아 동작 위험이 분리됩니다.

### Phase 2 — 제품 코드 패키지화

상태: **완료** — 결과는 [Phase_2_Result.md](Phase_2_Result.md) 참조

작업:

1. `pyproject.toml`과 `src/simple_qna_rag` 생성
2. 제품 모듈을 순수 이동한 뒤 절대 import로 변경
3. CLI와 Web server entry point 정의
4. evaluation과 테스트 import 수정
5. import side effect 및 다른 작업 디렉터리 실행 테스트 추가

완료 조건:

- editable install과 clean environment install 방식이 문서화됩니다.
- `python -c "import simple_qna_rag"`와 Web server import가 성공합니다.
- 저장소 루트 밖을 current directory로 두어도 설정·template/static 경로가 올바릅니다.
- 기존 제품/평가/프런트엔드 테스트가 모두 통과합니다.

### Phase 3 — Web·학습·모델 자산 정리

상태: **완료** — 결과는 [Phase_3_Result.md](Phase_3_Result.md) 참조

작업:

1. template/static을 `web/`으로 이동하고 asset resolution 수정
2. intent 학습 코드와 dataset을 `training/`으로 이동
3. 버전 관리되는 intent 모델 artifact를 `models/`로 이동
4. script, README, `.gitignore`, CI 경로 수정

완료 조건:

- Web UI smoke test와 frontend test가 새 경로를 사용합니다.
- intent classifier가 새 기본 모델 경로에서 로드됩니다.
- 학습 명령의 `--help`와 dry/import 수준 검증이 성공합니다.
- vendor sync 후 `web/static/vendor/`에 예상 밖 diff가 없습니다.

### Phase 4 — 런타임 경로 전환과 호환 migration

상태: **완료** — 결과는 [Phase_4_Result.md](Phase_4_Result.md) 참조

작업:

1. 기본 경로를 `runtime/documents`, `runtime/vectorstore`로 전환
2. 환경변수와 CLI override 구현
3. 기존 `data`, `vectorstore` 탐지 및 migration 안내 구현
4. 충돌·빈 디렉터리·권한 오류 테스트
5. 실제 로컬 자산은 사용자 확인 가능한 방식으로 이전

완료 조건:

- 기존 runtime 파일이 삭제·덮어쓰기 되지 않았습니다.
- 새 경로만 있는 환경, 기존 경로만 있는 환경, 양쪽이 충돌하는 환경의 동작이 테스트됩니다.
- index와 query가 동일한 resolved 경로를 사용합니다.
- M2 corpus/vectorstore fingerprint가 이동 전과 동일합니다.

### Phase 5 — 문서화와 최종 회귀 검증

상태: **완료** — 결과는 [Phase_5_Final_Result.md](Phase_5_Final_Result.md) 참조

작업:

1. README의 설치, 테스트, 실행, 색인과 migration 명령 갱신
2. Roadmap에서 M2.5 상태 갱신
3. Problem에서 구조 정리로 해결된 항목과 새로 발견한 문제 정리
4. CI 전체 실행 및 새 checkout 관점의 명령 검증
5. 변경 전후 파일 mapping과 예외 사항 기록

완료 조건:

- §9의 전체 완료 조건을 모두 충족합니다.
- 문서와 실제 CLI `--help`, CI 명령이 일치합니다.
- 구조 변경과 무관한 코드 diff가 없거나 별도 근거가 기록됩니다.
- 사용자 리뷰 전에는 M2.5를 완료로 표시하지 않습니다.

## 9. 전체 완료 조건

### 9.1 구조

- [x] 루트에 제품 Python 모듈, `test_*.py`, 마일스톤 상세 문서가 남아 있지 않습니다.
- [x] 제품 코드는 `src/simple_qna_rag`, 테스트는 `tests`, 문서는 `docs`에 있습니다.
- [x] 학습 코드·학습 데이터·학습 모델·사용자 runtime 데이터가 서로 분리되어 있습니다.
- [x] `docs/architecture/Repository_Structure.md`가 각 디렉터리의 책임과 허용 파일을 설명합니다.

### 9.2 기능과 품질 보존

- [x] Python 전체 테스트가 이동 전과 같거나 더 많은 수로 통과합니다.
- [x] 프런트엔드 테스트가 모두 통과하고 vendor diff가 없습니다.
- [x] golden dataset validation이 76건으로 통과합니다.
- [x] 승인된 `m2_initial.json`과 `m2_initial.md` 내용 hash가 이동 전과 같습니다.
- [x] corpus manifest와 vectorstore fingerprint가 물리적 경로 이동 전후 동일합니다.
- [x] Clean CI Web server import와 세 CLI smoke가 성공했습니다. 실제 LLM query·재색인은 비파괴 원칙에 따라 사용자 승인으로 면제했습니다.
- [x] live test의 opt-in 정책과 기본 offline CI 정책이 유지됩니다.

### 9.3 실행과 경로

- [x] 제품 모듈이 package 절대 import만 사용하며 `sys.path` 우회가 없습니다.
- [x] 공식 Web/query/index 명령과 `--help`가 동작합니다.
- [x] 지원하는 모든 명령은 저장소 루트가 아닌 current directory에서도 올바른 경로를 사용합니다.
- [x] runtime 경로는 환경변수 또는 CLI로 재정의할 수 있습니다.
- [x] 기존 runtime 경로 migration이 데이터 손실 없이 동작하고 충돌 시 중단합니다.
- [x] import 시 모델, vectorstore, network 또는 파일 생성 side effect가 없습니다.

### 9.4 문서와 자동화

- [x] README, Roadmap, Problem과 평가 가이드의 모든 local link가 유효합니다.
- [x] README 명령을 그대로 실행할 수 있습니다.
- [x] GitHub Actions가 새 package와 테스트 경로에서 모두 성공합니다. (PR #12)
- [x] `.gitignore`가 `runtime/`, 평가 report, 환경·IDE·cache 파일을 올바르게 제외합니다.
- [x] `git diff --check`가 성공하고 의도하지 않은 generated file이 없습니다.

### 9.5 변경 통제

- [x] M2.5 비범위인 검색·라우팅·답변 품질 변경이 포함되지 않았습니다.
- [x] 순수 이동과 논리 변경을 리뷰에서 구분할 수 있습니다.
- [x] 이동 전후 mapping, 실행 결과, 알려진 예외와 rollback 방법이 최종 리뷰 문서에 기록됩니다.
- [x] 사용자 최종 승인 후 Roadmap의 M2.5를 완료로 변경했습니다. (2026-08-05)

## 10. 검증 명령 기준

구현 과정에서 실제 package/entry point 이름이 확정되면 명령을 갱신하되 다음 범위를 축소하면 안 됩니다.

```bash
python -m pip check
python -c "import simple_qna_rag"
python -c "from simple_qna_rag.web.server import app"
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
pytest -q
npm ci
npm test
npm run sync-vendor
git diff --exit-code -- web/static/vendor/
git diff --check
```

추가 자동 검증:

- Markdown local link 검사
- `evaluation/datasets/golden.jsonl` SHA-256 비교
- `evaluation/baselines/m2_initial.json/.md` SHA-256 비교
- runtime 이동 전후 corpus manifest와 `index.faiss`/`index.pkl` SHA-256 비교
- 저장소 루트 외부 current directory에서 Web/query/index `--help`와 import 실행

공유 개발환경의 `pip check`는 M2 종료 시점부터 알려진 외부 package 충돌이 있으므로, 결과를 숨기지 않고 기록하되 M2.5 변경으로 새 충돌이 추가됐는지를 별도로 판정합니다. 최종 CI는 clean install에서 성공해야 합니다.

## 11. 롤백과 안전 규칙

- tracked 파일 이동은 Phase별 commit으로 되돌릴 수 있게 분리합니다.
- runtime migration 전 source/target의 존재, 파일 수, 크기와 hash를 확인합니다.
- source와 target이 모두 존재하면 자동 병합하거나 덮어쓰지 않고 실패합니다.
- vectorstore 재생성으로 migration 실패를 덮지 않습니다.
- runtime 이동 완료와 hash 확인 전 기존 경로를 삭제하지 않습니다.
- 구조 변경 도중 기능 회귀가 생기면 마지막 통과 Phase로 돌아가고 M3 작업을 섞지 않습니다.

## 12. 구현 및 리뷰 단위 권고

다음 단위를 각각 독립 commit 또는 PR로 유지하는 것을 권장합니다.

1. 문서·테스트 이동과 링크/CI 수정
2. `src` package 전환과 import/entry point 수정
3. Web·학습·모델 자산 이동
4. runtime 경로 설정과 migration
5. 최종 문서와 검증 결과

Phase 2와 Phase 4는 경로와 import 영향이 크므로 병렬 개발하지 않습니다. Phase 1 완료 후 Phase 2, Phase 3, Phase 4를 순차 진행하고 Phase 5에서 통합 검증합니다.

## 13. 승인 게이트

1. **계획 승인**: 목표 구조, 환경변수명, 공식 CLI와 migration 호환 기간 승인
2. **Phase 1 승인**: 문서·테스트 분류와 링크/CI 검토
3. **Phase 2 승인**: package/API/import 호환성 검토
4. **Phase 4 실행 승인**: 로컬 runtime 자산의 실제 이동 전 source/target inventory 확인
5. **최종 승인**: 전체 완료 조건, CI와 fingerprint 비교 결과 확인

Claude Code 또는 다른 구현 에이전트는 사용자 승인 없이 runtime 파일을 삭제·덮어쓰거나 M2 baseline을 수정해서는 안 됩니다.
