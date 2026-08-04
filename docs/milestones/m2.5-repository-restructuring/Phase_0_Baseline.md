# M2.5 Phase 0 기준 상태

측정일: 2026-08-05 (Asia/Seoul)

상태: **Phase 0 완료** — 구조 변경 전 기준 상태 고정

관련 계획: [Plan.md](Plan.md)

## 1. Git 시작 상태

| 항목 | 값 |
|---|---|
| Branch | `master` |
| HEAD | `522001770aeca9d720a2442b72e400b65dd9b343` |
| HEAD 설명 | `Merge pull request #11 from luminous419/docs/m2-phase9-close-out` |
| Working tree | M2.5 계획 작업으로 dirty |
| 수정 파일 | `Roadmap.md` |
| 신규 파일 | `Development_M2_5_Repository_Restructuring_Plan.md`, 본 Phase 0 문서 |

Phase 0 측정 전에 존재한 제품 코드나 runtime 변경은 없습니다. 이후 비교에서는 위 M2.5 문서 변경과 구현 변경을 구분해야 합니다.

## 2. 도구 버전

| 도구 | 버전 | 판정 |
|---|---|---|
| Python | `3.11.8` | 프로젝트 권장 major/minor와 일치 |
| Node.js | `22.17.0` | README 권장 `22.22.2+`보다 낮음 |
| npm | `10.9.2` | 테스트 실행 가능 |

Node 22.17.0에서도 현재 테스트는 통과하지만 `npm ci`가 `jsdom`과 `undici` engine 경고를 냅니다. M2.5 최종 clean 검증은 Node 22.22.2 이상 또는 GitHub Actions의 호환되는 최신 Node 22 환경에서 수행해야 합니다.

## 3. 저장소 inventory

### 3.1 tracked 파일의 최상위 분포

| 위치 | 파일 수 | 현재 역할 |
|---|---:|---|
| 저장소 루트 | 48 | 제품 코드, 테스트, 문서, 설정이 혼재 |
| `.github/` | 1 | CI |
| `evaluation/` | 13 | 평가 코드·dataset·승인 baseline |
| `frontend_tests/` | 1 | 프런트엔드 테스트 |
| `intent-bge-m3-softmax/` | 2 | intent 모델 artifact |
| `intent_dataset/` | 2 | intent 학습 dataset |
| `scripts/` | 1 | vendor 동기화 |
| `static/` | 6 | Web JavaScript와 vendor |
| `templates/` | 1 | Web template |

루트 48개에는 다음 범주가 함께 있습니다.

- 제품 Python 모듈 12개: Agent, RAG, routing, intent, Web/CLI와 설정
- Python 테스트 13개
- M2 계획·설계·리뷰와 현재 운영 문서 15개
- 학습 Python/Shell 진입점 3개
- build/dependency/저장소 설정 5개

M2.5 구현 중 실제 파일 수가 바뀌면 순수 이동, package metadata 추가, 테스트 추가와 삭제를 각각 구분해 보고해야 합니다.

### 3.2 Git 제외 runtime 자산

| 현재 경로 | 파일 수 | 전체 크기 | 목표 경로 | 충돌 |
|---|---:|---:|---|---|
| `data/` | 18 | 19,468,839 bytes | `runtime/documents/` | 없음 — `runtime/` 미존재 |
| `vectorstore/` | 2 | 2,078,626 bytes | `runtime/vectorstore/` | 없음 — `runtime/` 미존재 |
| `evaluation/reports/` | timestamped 결과 | 측정 제외 | 현 위치 유지 | 해당 없음 |

`data/`, `vectorstore/`, `evaluation/reports/`는 `.gitignore`에 의해 제외됩니다. Phase 0에서는 읽기와 hash 계산만 수행했으며 이동, 삭제, 재색인 또는 덮어쓰기를 하지 않았습니다.

## 4. M2 보호 대상 fingerprint

아래 값을 M2.5 완료 시 동일하게 비교합니다.

| 보호 대상 | SHA-256 |
|---|---|
| `evaluation/datasets/golden.jsonl` | `61b768acd8d33522ef76e3baadd4bf19b44cc25daa79ad7ea255fb0a09d1017a` |
| `evaluation/baselines/m2_initial.json` | `e1edf23e7bc7c7584ee0e166de7ace211e02fc1d064fccbbac1d081a35faa6d5` |
| `evaluation/baselines/m2_initial.md` | `844e3c9cdc75ab244ca39fabe7693502e67a258892f25aea9d177f20025ad3f8` |
| corpus manifest (18개) | `5c0d648d032bc231c93cf1c545e5b3f8d26337da56677a4a0c1a66f24c82374a` |
| `vectorstore/index.faiss` | `c52fb288f3bc780d681ab68d62dd5b4c545843ee9dc53ef5528bd50db4d69820` |
| `vectorstore/index.pkl` | `3f7217a2825f141bbea4c75dbf2b15738c3bcd50e3c0900e25a6a4a7ed91bb00` |

Golden dataset, corpus와 vectorstore 값은 승인된 [M2 최초 baseline](../../../evaluation/baselines/m2_initial.md)에 기록된 값과 일치합니다. 승인 baseline 파일 자체의 두 hash는 M2.5에서 소급 편집이 없음을 확인하기 위한 추가 보호값입니다.

## 5. 기존 실행 계약

### 5.1 정상 확인된 명령

| 명령 | 현재 결과 |
|---|---|
| `python -m evaluation.dataset validate evaluation/datasets/golden.jsonl` | 성공, 76건 |
| `python -m evaluation.retrieval --help` | 성공 |
| `python -m evaluation.routing --help` | 성공 |
| `python -m evaluation.answers --help` | 성공 |
| `python -m evaluation.baseline --help` | 성공 |
| `pytest -q` | 성공 |
| `npm ci` | 성공, Node engine 경고 있음 |
| `npm test` | 성공 |
| `npm run sync-vendor` | 성공 |
| `git diff --exit-code -- static/vendor/` | 성공 |
| `git diff --check` | 성공 |

### 5.2 현재 환경에서 실패한 명령

`python -c "import web_server"`는 다음 기존 환경 문제로 실패했습니다.

```text
ImportError: email-validator version >= 2.0 required
```

`python -m pip check`도 공유 Conda 환경에 설치된 프로젝트 외 package를 포함해 다음 계열의 충돌을 보고했습니다.

- Torchvision과 Torch
- LangChain Classic/LangGraph와 LangChain Core/Text Splitters
- Google API/Streamlit과 Protobuf
- OpenTelemetry instrumentation과 semantic conventions
- LangChain Postgres와 SQLAlchemy

이는 M2.5 코드 변경 전의 기준 결함입니다. PR #11 이전 clean GitHub Actions에서는 Web server import와 Python CI가 성공했으므로 저장소의 clean install 결과와 현재 공유 환경의 상태를 분리해 판단합니다. M2.5는 새 충돌을 추가하면 안 되며 최종 clean CI에서는 Web server import가 성공해야 합니다.

## 6. Offline 검증 결과

### Golden dataset

- 전체 76건, validation 성공
- Document QA 51, Web search 15, Unanswerable 7, Boundary 3
- Answer 평가 대상 29건
- Korean ratio 1.0

### Python

```text
349 passed, 1 skipped, 1 warning in 6.45s
```

warning은 공유 환경의 `torchvision` image extension 로드 실패이며 현재 테스트 대상 기능에는 사용되지 않습니다.

### Frontend

```text
Test Files  1 passed (1)
Tests       9 passed (9)
```

vendor 파일 4개를 잠금 dependency에서 다시 동기화한 뒤 tracked diff가 없음을 확인했습니다.

## 7. Phase 0 완료 조건 판정

| 완료 조건 | 판정 | 근거 |
|---|---|---|
| 이동 전 테스트 결과와 fingerprint 기록 | 충족 | §4, §6 |
| runtime source/target과 충돌 여부 확인 | 충족 | 기존 20개 파일 확인, 목표 `runtime/` 미존재 |
| 변경 금지 M2 artifact 확정 | 충족 | Golden dataset, 승인 baseline 2개, corpus/vectorstore fingerprint |
| 구조 변경 전 기존 결함 분리 | 충족 | Web import, pip dependency, Node engine 경고 기록 |
| runtime 비파괴 확인 | 충족 | 읽기/hash만 수행, 이동·삭제·재생성 없음 |

Phase 0는 완료됐습니다. 다음 게이트는 이 기준 상태를 검토한 뒤 Phase 1 문서·테스트 이동을 시작하는 것입니다.

## 8. Phase 1 전달사항

1. 제품 코드, `data/`, `vectorstore/`와 평가 dataset/baseline 내용은 변경하지 않습니다.
2. 문서와 테스트 이동만 수행하고 링크, pytest, Vitest와 CI 경로를 함께 갱신합니다.
3. 이동 뒤 Python `349 passed, 1 skipped`, frontend 9건을 최소 기준으로 비교합니다.
4. `m2_initial.*`, `golden.jsonl` hash를 다시 확인합니다.
5. 공유 환경의 Web import 실패를 Phase 1 회귀로 오판하지 않되 clean CI import는 반드시 확인합니다.
6. Phase 1 완료 전 runtime migration이나 `src` package 이동을 시작하지 않습니다.
