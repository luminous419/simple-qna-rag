# M2.5 Phase 5 최종 결과

측정일: 2026-08-05 (Asia/Seoul)

상태: **로컬 구현·검증 완료, GitHub Actions와 사용자 최종 승인 대기**

## 최종 구조

```text
simple-qna-rag/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── package.json
├── package-lock.json
├── vitest.config.js
├── src/simple_qna_rag/          # 제품 package와 CLI
├── evaluation/                  # evaluator, dataset, baseline, reports
├── tests/                       # unit, integration, frontend
├── web/                         # templates, static, vendor
├── training/intent_classifier/  # 학습 코드와 dataset
├── models/intent_classifier/    # 버전 관리 모델 artifact
├── runtime/                     # Git 제외 documents/vectorstore
├── docs/                        # Roadmap, Problem, architecture, milestone/review
├── scripts/
└── .github/workflows/ci.yml
```

루트의 제품 Python 모듈, `test_*.py`, 마일스톤 상세 문서와 IDE `.iml` 파일을 제거했습니다.

## Phase별 결과

| Phase | 결과 |
|---|---|
| 0 | 시작 commit, inventory, 테스트와 M2 보호 fingerprint 고정 |
| 1 | 문서와 테스트를 `docs/`, `tests/`로 분류·이동 |
| 2 | `src/simple_qna_rag` package, `pyproject.toml`, CLI entry point 도입 |
| 3 | Web·학습·모델 자산을 `web/`, `training/`, `models/`로 이동 |
| 4 | 문서·vectorstore를 `runtime/`으로 비파괴 이동하고 override/legacy 계약 구현 |
| 5 | 현재 문서, 완료조건, 전체 회귀와 fingerprint 최종 검증 |

## 공식 명령

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps

simple-qna-rag-index
simple-qna-rag-query
simple-qna-rag-web

python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
pytest -q
npm ci
npm test
```

제품 CLI는 저장소 밖 current directory에서도 동작합니다. 평가 명령은 repository checkout의 최상위 `evaluation/` subsystem을 사용하므로 저장소 루트에서 실행합니다.

## 최종 자동 검증

| 검증 | 결과 |
|---|---|
| Golden dataset | 76건, valid |
| Python | `358 passed, 1 skipped, 1 warning` |
| Frontend | `9 passed` |
| Editable install | 성공 |
| Wheel build/별도 venv install | 성공, Python package와 CLI 범위 |
| 외부 CWD CLI `--help` | Web/query/index 성공 |
| 외부 CWD config 경로 | 성공 |
| 환경변수 override | 성공 |
| legacy-only fallback | 경고와 함께 성공 |
| new/legacy 충돌 | 자동 병합 없이 실패 확인 |
| vendor sync/diff | 성공 |
| Markdown local link | 누락 0 |
| legacy root import·`sys.path` 우회 | 없음 |
| `git diff --check` | 성공 |

공유 Conda 환경의 Torchvision warning, `pip check` 충돌과 `email-validator`로 인한 Web server import 실패는 Phase 0 이전부터 존재한 환경 결함입니다. clean install GitHub Actions에서 새 package Web import가 성공하는지는 commit/PR 이후 확인해야 합니다.

## M2 보호 대상

| 대상 | SHA-256/값 | 결과 |
|---|---|---|
| Golden dataset | `61b768acd8d33522ef76e3baadd4bf19b44cc25daa79ad7ea255fb0a09d1017a` | 동일 |
| `m2_initial.json` | `e1edf23e7bc7c7584ee0e166de7ace211e02fc1d064fccbbac1d081a35faa6d5` | 동일 |
| `m2_initial.md` | `844e3c9cdc75ab244ca39fabe7693502e67a258892f25aea9d177f20025ad3f8` | 동일 |
| corpus file count | 18 | 동일 |
| corpus manifest | `5c0d648d032bc231c93cf1c545e5b3f8d26337da56677a4a0c1a66f24c82374a` | 동일 |
| `index.faiss` | `c52fb288f3bc780d681ab68d62dd5b4c545843ee9dc53ef5528bd50db4d69820` | 동일 |
| `index.pkl` | `3f7217a2825f141bbea4c75dbf2b15738c3bcd50e3c0900e25a6a4a7ed91bb00` | 동일 |

M2 기준선 문서 안의 과거 `data/`, `vectorstore/` 경로와 commit SHA는 당시 실행의 역사적 사실이므로 수정하지 않았습니다.

## 변경 통제

검색 알고리즘, routing, prompt, evaluator metric과 Golden dataset은 변경하지 않았습니다. 논리 변경은 repository 구조를 지원하는 다음 항목으로 제한됩니다.

- package 절대 import
- repository 기준 asset 경로
- 공식 CLI와 argparse 도움말
- runtime 환경변수·CLI override
- legacy fallback 및 충돌 중단
- CI package 설치와 새 import 경로

## Rollback

- tracked 파일은 Git rename/내용 변경으로 복원할 수 있습니다.
- runtime rollback이 필요하면 `runtime/documents`와 `runtime/vectorstore`의 현재 hash를 다시 확인한 뒤 기존 `data`, `vectorstore`로 각각 이동합니다.
- 기존 경로와 새 경로가 동시에 존재하면 config가 중단하므로 먼저 어느 쪽을 보존할지 명시적으로 결정해야 합니다.
- vectorstore를 재생성해 rollback 오류를 덮어쓰면 안 됩니다.

## 남은 승인 게이트

1. 변경을 commit/PR한 뒤 GitHub Actions Python/Frontend job 성공 확인
2. clean CI의 `from simple_qna_rag.web.server import app` 성공 확인
3. 필요 시 실제 Web/API 및 query smoke test 수행
4. 사용자 최종 승인 후 Roadmap에서 M2.5를 `완료`로 변경

commit과 push는 이 작업에서 수행하지 않았습니다.
