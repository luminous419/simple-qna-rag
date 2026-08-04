# Repository Structure

이 문서는 파일을 어느 디렉터리에 배치할지 결정하는 기준을 정의합니다. M2.5 이전이 진행되는 동안에는 현재 구조와 목표 구조가 일시적으로 함께 존재할 수 있습니다.

## 디렉터리 책임

| 디렉터리 | 책임 | 허용되는 파일 |
|---|---|---|
| 저장소 루트 | 프로젝트 진입과 build 설정 | `README.md`, `LICENSE`, dependency·test·tool 설정 |
| `src/simple_qna_rag/` | 설치·실행되는 제품 코드 | RAG, Agent, routing, Web server, CLI package |
| `evaluation/` | 품질 평가 subsystem | evaluator, golden dataset, 승인 baseline, Git 제외 report |
| `tests/unit/` | 빠르고 격리된 단위 검증 | 순수 함수·단일 모듈·fake 기반 테스트 |
| `tests/integration/` | 여러 제품 경계의 조합 검증 | Agent orchestration, evaluator/제품 연계, Web/API 테스트 |
| `tests/frontend/` | 브라우저 렌더링과 보안 회귀 | Vitest/jsdom 테스트 |
| `web/` | 제품 Web 자산 | template, JavaScript, CSS, vendored frontend library |
| `training/` | 학습 과정과 입력 | dataset 생성, 학습 코드, 학습 dataset |
| `models/` | 버전 관리되는 모델 artifact | 설정과 프로젝트가 배포하는 가중치 |
| `runtime/` | 로컬 실행 자산 | 사용자 문서와 vectorstore; 전체 Git 제외 |
| `docs/` | 프로젝트 지식과 의사결정 | Roadmap, Problem, architecture, milestone, review |
| `scripts/` | 개발·build 자동화 | vendor sync 등 제품 런타임이 아닌 script |

## 테스트 분류

- 파일·모델·네트워크를 fake 또는 임시 경로로 대체하고 단일 책임을 검증하면 `unit`입니다.
- 여러 제품 모듈의 호출 순서, fallback 또는 evaluator와 제품 코드의 계약을 검증하면 `integration`입니다.
- 실제 Ollama나 네트워크가 필요한 테스트는 위치와 관계없이 명시적 opt-in이어야 합니다.
- 프런트엔드 JavaScript 테스트는 `tests/frontend`에 둡니다.

## 문서 분류

```text
docs/
├── README.md
├── Roadmap.md
├── Problem.md
├── architecture/
├── milestones/<milestone>/
└── reviews/<milestone>/
```

- `milestones`에는 요구사항, 계획, 상세 설계와 구현 지시를 둡니다.
- `reviews`에는 특정 시점의 설계·코드 평가를 둡니다.
- 역사적 문서의 본문에 기록된 당시 파일명과 명령은 이력 보존을 위해 유지할 수 있습니다. 현재 사용자가 실행할 명령은 루트 README와 subsystem README에서만 최신 상태로 관리합니다.

## 데이터 분류

- `evaluation/datasets`: Git으로 관리하는 품질 검증 입력
- `evaluation/baselines`: 사용자가 승인한 장기 비교 결과
- `evaluation/reports`: 실행마다 생성되는 상세 결과, Git 제외
- `training/.../datasets`: 모델 학습 입력
- `models`: 버전 관리되는 모델 artifact
- `runtime/documents`: 사용자 원본 문서, Git 제외
- `runtime/vectorstore`: 원본 문서로 생성한 검색 index, Git 제외

서로 다른 범주의 데이터를 편의상 같은 디렉터리에 넣지 않습니다.

## 경로 규칙

- 제품 코드는 current working directory를 저장소 루트라고 가정하지 않습니다.
- package resource 또는 명시적 설정에서 경로를 계산합니다.
- runtime 경로는 `CLI > environment > default` 우선순위를 따릅니다.
- import만으로 디렉터리 생성, 모델 로드, vectorstore 접근이나 network 호출을 해서는 안 됩니다.
- 사용자 runtime 파일은 충돌 시 자동 병합·덮어쓰기하지 않습니다.

## 현재 상태

M2.5 Phase 0~5의 로컬 구현을 완료했습니다. 제품 코드, 테스트, 문서, Web, 학습·모델과 runtime 자산이 이 문서의 책임 기준에 따라 배치되어 있습니다. 실제 GitHub Actions 성공과 사용자 최종 승인 후 M2.5를 완료 처리합니다.
