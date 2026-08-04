# M2.5 Phase 2 결과

측정일: 2026-08-05 (Asia/Seoul)

상태: **완료** — 제품 Python package와 CLI entry point 전환 및 회귀 검증 통과

관련 문서:

- [M2.5 상세 계획](Plan.md)
- [Phase 0 기준 상태](Phase_0_Baseline.md)
- [Phase 1 결과](Phase_1_Result.md)
- [Repository Structure](../../architecture/Repository_Structure.md)

## 1. 작업 범위

- 루트 제품 모듈을 `src/simple_qna_rag` package로 이동
- 모든 제품 내부 import를 `simple_qna_rag.*` 절대 import로 통일
- evaluation과 테스트가 새 package를 사용하도록 import 변경
- `pyproject.toml`에 src layout, dependency metadata와 CLI entry point 정의
- Web/query/index 공식 console script와 안전한 `--help` 구현
- config의 기본 asset/runtime 경로를 package 위치에서 계산
- CI가 editable package를 설치하고 새 Web module을 import하도록 변경
- README와 evaluator 오류 안내를 새 명령으로 갱신

Web·학습·모델 자산의 물리적 이동과 `runtime/` 전환은 각각 Phase 3·4 범위이므로 수행하지 않았습니다.

## 2. 제품 package 결과

```text
src/simple_qna_rag/
├── __init__.py
├── agent.py
├── config.py
├── intent_classifier.py
├── prompt_templates.py
├── query_router.py
├── rag_engine.py
├── tools.py
├── web_search.py
├── cli/
│   ├── query.py
│   ├── index_documents.py
│   └── web.py
└── web/
    └── server.py
```

루트에는 제품 Python 모듈이 남아 있지 않습니다. `generate_intent_dataset.py`, `train_intent_classifier.py`는 제품 런타임이 아니라 학습 자산이므로 Phase 3에서 이동합니다.

## 3. Packaging 계약

`pyproject.toml`에 다음 계약을 정의했습니다.

- package 이름: `simple-qna-rag`
- package 버전: `0.2.5`
- Python: `>=3.11,<3.12`
- build backend: setuptools
- dependency 원천: 기존 `requirements.txt`
- package source: `src/`
- pytest 대상: `tests/`

설치 명령:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

일반 wheel 생성과 별도 임시 venv wheel 설치도 성공했습니다. Phase 3 전 wheel에는 Python package만 포함되며 template/static/model 자산을 사용하는 완전한 wheel 배포는 아직 지원하지 않습니다. 현재 지원 대상은 repository checkout의 editable install입니다.

## 4. 공식 CLI

| 명령 | 역할 |
|---|---|
| `simple-qna-rag-web` | FastAPI Web server 실행 |
| `simple-qna-rag-query` | 대화형 문서 질의 |
| `simple-qna-rag-index` | PDF/TXT index 생성 |

세 명령은 저장소 밖 current directory에서 `--help`와 종료 코드 0을 확인했습니다. Web CLI는 도움말 경로에서 FastAPI/RAG module을 import하지 않도록 lightweight wrapper에서 실제 server import를 지연합니다.

## 5. 경로 계약

`simple_qna_rag.config.PROJECT_ROOT`는 editable package 파일 위치에서 repository root를 계산합니다. 다음 기존 위치를 절대경로로 제공합니다.

- `DATA_DIR` → `<repo>/data`
- `VECTORSTORE_PATH` → `<repo>/vectorstore`
- `INTENT_MODEL_PATH` → `<repo>/intent-bge-m3-softmax`
- `TEMPLATES_DIR` → `<repo>/templates`
- `STATIC_DIR` → `<repo>/static`

저장소 밖 CWD의 subprocess 테스트에서 여섯 경로를 모두 검증했습니다. 환경변수·CLI override와 `runtime/` 기본 경로는 Phase 4에서 구현합니다.

## 6. Import와 side effect

- 제품 내부의 legacy root import를 제거했습니다.
- 임의의 `sys.path` 조작은 없습니다.
- `import simple_qna_rag`는 모델, vectorstore 또는 network에 접근하지 않습니다.
- evaluation의 live Agent/RAG import 지연 계약을 새 module 경로로 유지했습니다.
- Web server는 template/static의 절대경로를 사용하므로 CWD에 의존하지 않습니다.

공유 Conda 환경의 `from simple_qna_rag.web.server import app`는 Phase 0에서 기록한 구버전 `email-validator` 문제 때문에 실패합니다. 이는 구조 변경 전부터 존재한 환경 결함이며, 새 CI smoke test는 clean dependency install에서 검증해야 합니다.

## 7. 검증 결과

| 검증 | 결과 |
|---|---|
| Editable install | 성공, `simple-qna-rag 0.2.5` |
| Wheel build | 성공, Python package 15개 module 포함 |
| 별도 임시 venv wheel install | 성공 |
| 외부 CWD 세 CLI `--help` | 성공 |
| 외부 CWD config 경로 | 성공 |
| Golden dataset validation | 성공, 76건 |
| Python tests | `353 passed, 1 skipped, 1 warning` |
| Frontend tests | `9 passed` |
| vendor sync/diff | 성공, tracked diff 없음 |
| `git diff --check` | 성공 |

Python warning과 Node engine 경고는 Phase 0에 기록된 기존 환경 항목과 동일합니다.

## 8. 보호 대상 비교

| 보호 대상 | Phase 0 SHA-256 | Phase 2 판정 |
|---|---|---|
| `evaluation/datasets/golden.jsonl` | `61b768acd8d33522ef76e3baadd4bf19b44cc25daa79ad7ea255fb0a09d1017a` | 동일 |
| `evaluation/baselines/m2_initial.json` | `e1edf23e7bc7c7584ee0e166de7ace211e02fc1d064fccbbac1d081a35faa6d5` | 동일 |
| `evaluation/baselines/m2_initial.md` | `844e3c9cdc75ab244ca39fabe7693502e67a258892f25aea9d177f20025ad3f8` | 동일 |
| corpus manifest | `5c0d648d032bc231c93cf1c545e5b3f8d26337da56677a4a0c1a66f24c82374a` | 동일 |
| `index.faiss` | `c52fb288f3bc780d681ab68d62dd5b4c545843ee9dc53ef5528bd50db4d69820` | 동일 |
| `index.pkl` | `3f7217a2825f141bbea4c75dbf2b15738c3bcd50e3c0900e25a6a4a7ed91bb00` | 동일 |

## 9. 완료 조건 판정

| 조건 | 판정 |
|---|---|
| 제품 코드 `src/simple_qna_rag` 이동 | 충족 |
| package 절대 import와 `sys.path` 우회 제거 | 충족 |
| editable install과 wheel build/install | 충족 |
| Web/query/index entry point와 도움말 | 충족 |
| 외부 CWD 기본 경로 해석 | 충족 |
| evaluation·테스트 package import 전환 | 충족 |
| Phase 0 테스트 및 M2 보호 대상 보존 | 충족 |
| clean CI Web import | PR/merge 전 GitHub Actions 확인 필요 |

Phase 2의 로컬 구현과 검증은 완료됐습니다. 다음 Phase에서는 `templates/static`, intent 학습 코드·dataset과 버전 관리되는 intent 모델 artifact를 목표 디렉터리로 이동합니다.
