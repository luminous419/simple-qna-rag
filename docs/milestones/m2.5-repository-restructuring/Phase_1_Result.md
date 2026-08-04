# M2.5 Phase 1 결과

측정일: 2026-08-05 (Asia/Seoul)

상태: **완료** — 문서·테스트 이전 및 회귀 검증 통과

관련 문서:

- [M2.5 상세 계획](Plan.md)
- [Phase 0 기준 상태](Phase_0_Baseline.md)
- [Repository Structure](../../architecture/Repository_Structure.md)

## 1. 작업 범위

Phase 1에서는 다음 항목만 변경했습니다.

- 루트의 현재 운영 문서를 `docs/`로 이동
- 완료된 M2 계획·설계·상세 지시를 milestone 디렉터리로 이동
- 완료된 M2 설계·코드 리뷰를 review 디렉터리로 이동
- Python 테스트를 `tests/unit`, `tests/integration`으로 분류
- 프런트엔드 테스트를 `tests/frontend`로 이동
- pytest와 Vitest discovery 및 테스트 내부 repository 경로 수정
- README, 평가 가이드와 문서 local link 갱신
- 문서 index와 repository structure 규칙 추가

제품 Python 모듈, Web runtime 자산, 학습·모델 자산, `data/`, `vectorstore/`, 평가 dataset/baseline 내용은 변경하지 않았습니다.

## 2. 문서 이동 결과

```text
docs/
├── README.md
├── Roadmap.md
├── Problem.md
├── architecture/Repository_Structure.md
├── milestones/
│   ├── m2-quality-baseline/
│   │   ├── Requirement.md
│   │   ├── Plan.md
│   │   ├── Development_Plan.md
│   │   ├── Design.md
│   │   └── implementation-guides/
│   └── m2.5-repository-restructuring/
│       ├── Plan.md
│       ├── Phase_0_Baseline.md
│       └── Phase_1_Result.md
└── reviews/m2-quality-baseline/
```

루트의 마일스톤 상세 문서와 review 문서는 모두 제거됐습니다. `README.md`만 프로젝트 진입 문서로 루트에 유지합니다.

완료된 M2 문서 본문의 당시 파일명·명령은 역사적 기록일 수 있어 일괄 개작하지 않았습니다. 현재 사용자가 따라야 하는 링크와 명령은 루트 README, 평가 README와 문서 index를 기준으로 갱신했습니다.

## 3. 테스트 이동 결과

### Unit

- Evaluation schema, dataset, metrics, reporting
- Query router
- Web search

### Integration

- Agent orchestration과 routing regression
- Retrieval, Routing, Answer evaluator
- 통합 baseline

### Frontend

- Markdown rendering과 XSS 회귀

`pytest.ini`에 `testpaths = tests`, `pythonpath = .`를 추가했습니다. 이는 제품 코드가 아직 루트에 있는 Phase 1의 임시 import 계약입니다. Phase 2에서 `src` package와 `pyproject.toml`을 도입할 때 package 기반 설정으로 교체합니다.

## 4. 설정 변경

| 파일 | 변경 |
|---|---|
| `pytest.ini` | 중첩 테스트 discovery와 현재 루트 모듈 import 보존 |
| `vitest.config.js` | `tests/frontend/**/*.test.js` 탐색 |
| `package.json` | 프런트엔드 테스트 위치 설명 갱신 |
| `tests/frontend/render.test.js` | 새 위치에서 `static/render.js` 참조 |
| Python path-sensitive 테스트 3개 | 저장소 루트와 golden dataset 경로 재계산 |

GitHub Actions의 명령 자체는 `pytest -q`, `npm test`를 사용하므로 workflow 변경 없이 새 설정을 따릅니다.

## 5. 검증 결과

| 검증 | 결과 |
|---|---|
| Golden dataset validation | 성공, 76건 |
| Python tests | `349 passed, 1 skipped, 1 warning` |
| Frontend tests | `9 passed` |
| vendor sync | 성공, tracked diff 없음 |
| Markdown local links | 성공, 누락 0 |
| `git diff --check` | 성공 |

Python warning과 공유환경 dependency 문제는 [Phase 0 기준 상태](Phase_0_Baseline.md)에 기록된 기존 항목과 동일합니다.

## 6. 보호 대상 비교

| 보호 대상 | Phase 0 SHA-256 | Phase 1 판정 |
|---|---|---|
| `evaluation/datasets/golden.jsonl` | `61b768acd8d33522ef76e3baadd4bf19b44cc25daa79ad7ea255fb0a09d1017a` | 동일 |
| `evaluation/baselines/m2_initial.json` | `e1edf23e7bc7c7584ee0e166de7ace211e02fc1d064fccbbac1d081a35faa6d5` | 동일 |
| `evaluation/baselines/m2_initial.md` | `844e3c9cdc75ab244ca39fabe7693502e67a258892f25aea9d177f20025ad3f8` | 동일 |
| corpus manifest | `5c0d648d032bc231c93cf1c545e5b3f8d26337da56677a4a0c1a66f24c82374a` | 동일 |
| `index.faiss` | `c52fb288f3bc780d681ab68d62dd5b4c545843ee9dc53ef5528bd50db4d69820` | 동일 |
| `index.pkl` | `3f7217a2825f141bbea4c75dbf2b15738c3bcd50e3c0900e25a6a4a7ed91bb00` | 동일 |

## 7. 완료 조건 판정

| 조건 | 판정 |
|---|---|
| 루트의 마일스톤 상세 문서와 `test_*.py` 제거 | 충족 |
| 문서의 milestone/review 분리 | 충족 |
| unit/integration/frontend 테스트 분리 | 충족 |
| Markdown local link 유효 | 충족 |
| 테스트 수와 skip 정책 유지 | 충족 |
| 제품 코드와 runtime 경로 미변경 | 충족 |
| M2 보호 대상 불변 | 충족 |

Phase 1은 완료됐습니다. 다음 Phase에서는 제품 Python 모듈을 `src/simple_qna_rag`로 이동하고 package metadata와 CLI entry point를 정의합니다.
