# M2 Phase 7·8 병렬 개발 상세 계획 및 규칙

## 1. 문서 목적

이 문서는 Claude Code가 M2 Phase 7과 Phase 8을 병렬로 구현할 때 따라야 할 실행 지침이다.

상위 기준 문서는 다음과 같다.

1. `Development_M2_Quality_Baseline_Requirement.md`
2. `Development_M2_Quality_Baseline_Development_Plan.md`
3. `Development_M2_Quality_Baseline_Design.md`
4. `M2_Phase_4_5_6_code_review_result.md`

기준이 충돌하면 **Requirement → Development Plan → Design → 이 문서** 순서로 우선한다. 구현 중 계약의 모순이나 공통 모듈 변경 필요성이 발견되면 임의로 범위를 넓히지 말고 통합 담당자와 사용자에게 보고한다.

이번 작업 범위는 다음과 같다.

- Phase 7 통합 baseline runner 구현 및 오프라인 검증
- 사용자의 로컬 환경이 준비된 경우 최초 live baseline 실행과 결과 제시
- **사용자 승인 이후에만** 최초 baseline 고정 파일 생성
- Phase 8 GitHub Actions CI 구현 및 실제 PR 검증
- 두 Phase 병합 후 전체 오프라인 통합 검증

다음은 이번 범위가 아니다.

- Phase 9 문서 전면 정리와 M2 완료 처리
- `Roadmap.md`의 M2 상태를 완료로 변경
- `Problem.md`의 품질 기준선·CI 문제를 최종 제거
- 요구사항 추적표 최종 확정
- 검색 품질, prompt, 모델, 라우팅 정책 또는 production 알고리즘 개선
- 측정 결과를 높이기 위한 골든셋·정답·설정 변경

## 2. 병렬 실행 전략

Phase 7과 Phase 8은 Phase 4·5·6 승인 코드를 동일한 기준점으로 삼아 별도 브랜치 또는 Git worktree에서 병렬 구현한다.

```text
Phase 4·5·6 승인 기준 commit
             │
             ├── Phase 7: 통합 runner·오프라인 테스트
             │              │
             │              └── live 실행 → 사용자 승인 → baseline 고정
             │
             └── Phase 8: GitHub Actions CI → PR에서 실제 성공 확인
                            │
                    두 작업 병합 및 전체 검증
                            │
                     Phase 7·8 코드 리뷰
                            │
                         Phase 9
```

두 작업을 하나의 공유 working tree에서 동시에 편집하지 않는다. 별도 worktree를 사용할 수 없다면 **Phase 7 구현 → Phase 8 구현** 순서로 적용하되, 아래 파일 소유권과 검증 게이트는 그대로 지킨다.

Phase 7의 live 실행과 승인 대기는 Phase 8 구현을 막지 않는다. 반대로 Phase 8 CI가 완료되지 않았더라도 Phase 7의 오프라인 runner 테스트와 live 결과 검토는 진행할 수 있다.

### 2.1 시작 전 기준점 확인

두 작업은 같은 승인 commit에서 시작해야 한다. 시작 전에 다음을 확인한다.

```bash
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git log -1 --oneline
```

- Phase 1~6의 승인된 코드와 테스트가 기준 commit에 포함돼 있어야 한다.
- untracked 또는 uncommitted 산출물은 새 worktree에 자동으로 전달되지 않는다.
- dirty working tree가 있다면 Claude가 임의로 commit, stash, reset, clean 또는 변경 폐기를 하지 않는다.
- 병렬 브랜치를 만들 기준 commit이 불명확하면 사용자에게 먼저 보고한다.
- 각 작업의 완료 보고에는 기준 commit SHA와 최종 commit SHA를 기록한다.
- 브랜치·worktree 생성과 commit은 사용자가 허용한 경우에만 수행한다. 허용되지 않았다면 현재 working tree에서 순차 작업한다.

### 2.2 권장 작업 단위

```text
브랜치 A: m2-phase7-baseline
브랜치 B: m2-phase8-ci
통합 브랜치: 현재 M2 작업 브랜치
```

브랜치명은 저장소 관례에 맞춰 조정할 수 있다. 중요한 조건은 두 브랜치가 같은 기준 commit에서 출발하고 서로의 단독 소유 파일을 수정하지 않는 것이다.

## 3. 공통 개발 규칙

### 3.1 파일 소유권

| 영역 | 단독 소유 파일 | 읽기 전용 의존 파일 |
|---|---|---|
| Phase 7 | `evaluation/baseline.py`, `test_evaluation_baseline.py` | evaluation의 schema, dataset, metrics, reporting, retrieval, routing, answers |
| Phase 7 승인 후 | `evaluation/baselines/m2_initial.json`, `evaluation/baselines/m2_initial.md` | 승인된 timestamp report |
| Phase 8 | `.github/workflows/ci.yml` | `requirements.txt`, `package.json`, lockfile, 전체 테스트, dataset validator |
| 통합 담당자 | 충돌 해결과 최종 검증만 수행 | 모든 Phase 산출물 |

Phase 7 테스트 파일명은 현재 저장소의 `test_evaluation_*.py` 관례를 따른다. 기존 테스트 파일 변경이 반드시 필요하면 해당 이유와 추가되는 계약을 보고하고, 가능하면 Phase 7 전용 테스트 파일에 작성한다.

Phase 8은 CI를 통과시키기 위해 production 코드나 테스트 기대값을 임의로 수정하지 않는다. CI에서 기존 결함이 발견되면 workflow 변경으로 숨기지 말고 별도 문제로 보고한다.

### 3.2 공통 모듈은 기본적으로 동결한다

다음 파일은 Phase 1~6에서 승인된 계약이므로 병렬 작업자가 원칙적으로 수정하지 않는다.

- `evaluation/schema.py`
- `evaluation/dataset.py`
- `evaluation/metrics.py`
- `evaluation/reporting.py`
- `evaluation/retrieval.py`
- `evaluation/routing.py`
- `evaluation/answers.py`
- `rag_engine.py`
- `agent.py`
- `config.py`
- `requirements.txt`
- `package.json`
- package lockfile
- `.gitignore`
- M2 Requirement, Plan, Design 문서
- `README.md`, `Roadmap.md`, `Problem.md`, `evaluation/README.md`

Phase 7에서 evaluator orchestration을 위해 기존 공개 함수를 호출하는 것은 허용하지만, evaluator 내부 구현을 복제하거나 기존 반환 계약을 바꾸지 않는다.

공통 변경이 반드시 필요하면 다음 형식으로 보고하고 승인 전에는 수정하지 않는다.

```text
공통 변경 요청
- 요청 Phase:
- 대상 파일/함수:
- 필요한 이유:
- 기존 계약과 병렬 작업에 미치는 영향:
- 변경하지 않는 대안:
- 추가할 회귀 테스트:
```

### 3.3 범위와 품질 원칙

- M2는 측정 기반을 만드는 마일스톤이며 점수 최적화 단계가 아니다.
- 낮은 Recall, routing accuracy 또는 assertion coverage를 이유로 production 동작이나 골든 정답을 바꾸지 않는다.
- 검색 알고리즘, prompt, 모델 설정, 라우팅 정책, 인덱싱 로직을 변경하지 않는다.
- existing public API, Web UI, `/rag`, `/health`, CLI 응답 의미와 Agent fallback을 변경하지 않는다.
- unrelated refactor, 대량 formatting, 문서 전면 개편을 하지 않는다.
- 기존 사용자 변경을 덮어쓰거나 되돌리지 않는다.
- import만으로 모델, vectorstore, Ollama 또는 네트워크를 초기화하면 안 된다.

### 3.4 오류 처리와 종료 코드

- dataset/schema 검증 실패는 후속 평가를 시작하지 않고 즉시 실패 처리한다.
- evaluator 내부 개별 사례 실패는 evaluator가 정한 기존 정책대로 기록하고 나머지 사례를 계속 처리한다.
- 통합 runner의 한 단계가 실패하더라도 이미 완료된 단계 결과와 실패 원인을 보존한다.
- 전체 통합 명령은 하나 이상의 필수 단계 실패 또는 fingerprint invariant 실패 시 non-zero를 반환한다.
- `--skip-*`으로 명시적으로 제외된 단계는 실패가 아니라 `skipped` 상태와 사유로 기록한다.
- CLI 성공은 exit 0, 잘못된 옵션·필수 artifact 누락·실행 실패는 non-zero다.
- 오류에는 실패 단계, 관련 경로나 사례 ID, 가능한 다음 조치를 포함한다.
- 예외를 점수 0이나 성공 결과로 조용히 바꾸지 않는다.

### 3.5 보안과 결과 파일

- timestamped 상세 결과는 `evaluation/reports/` 아래에 생성하며 Git에 커밋하지 않는다.
- 문서 chunk 원문 전체, API token, 환경변수 값, 개인정보를 리포트에 기록하지 않는다.
- 질문과 모델 답변이 포함된 live 상세 report도 기본적으로 커밋하지 않는다.
- 외부에서 받은 FAISS 인덱스를 사용하지 않는다.
- `evaluation/baselines/`에는 사용자에게 검토받은 최초 baseline 요약만 커밋할 수 있다.
- 최초 baseline Markdown에도 assertion coverage가 진실성을 보장하지 않는다는 한계와 인덱스 생성 provenance 부재를 사람이 읽을 수 있는 문장으로 기록한다.

## 4. Phase 7 — 통합 baseline과 최초 측정

### 4.1 목표

하나의 명령으로 dataset validation, Retrieval, live Routing, Answer 평가를 정해진 순서로 실행하고, 단계별 결과와 재현성 metadata를 하나의 통합 결과로 보존한다.

통합 runner 구현 완료와 최초 baseline 확정은 서로 다른 게이트다.

```text
runner 구현·오프라인 테스트 완료
              ≠
최초 live baseline 사용자 승인·고정 완료
```

### 4.2 구현 파일

- 신규: `evaluation/baseline.py`
- 신규: `test_evaluation_baseline.py`
- 사용자 승인 후에만 신규:
  - `evaluation/baselines/m2_initial.json`
  - `evaluation/baselines/m2_initial.md`

승인 전에는 `m2_initial.*` 파일을 placeholder, 예시 값 또는 일부 실행 결과로 만들지 않는다.

### 4.3 공개 API와 실행 순서

최소한 다음과 동등한 API를 제공한다.

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
    """validate → retrieval → routing → answers 순서로 실행한다."""
```

CLI 진입점은 다음과 동등해야 한다.

```bash
RUN_LIVE_LLM_TESTS=1 python -m evaluation.baseline \
  --dataset evaluation/datasets/golden.jsonl \
  --output evaluation/reports
```

실행 순서는 반드시 다음과 같다.

1. dataset 로드 및 composition validation
2. Retrieval 평가
3. `--skip-routing`이 아니면 live Routing 평가
4. `--skip-answers`가 아니면 Answer 평가
5. 단계 상태, 집계 결과, 오류, top-level metadata를 통합
6. 통합 JSON/Markdown report 기록
7. 모든 필수 단계와 invariant가 성공한 경우에만 exit 0

dataset validation이 실패하면 evaluator를 호출하지 않는다. 그 외 단계 실패는 가능한 한 다음 단계 실행을 계속하되, 후속 단계가 안전하게 실행될 수 없는 명확한 의존성 실패라면 `blocked` 또는 `not_run` 상태와 이유를 기록한다.

### 4.4 옵션 계약

- `--skip-routing`: Routing만 명시적으로 제외한다.
- `--skip-answers`: Answer만 명시적으로 제외한다.
- Retrieval은 top-level 재현성 metadata의 기준 단계이므로 skip 옵션을 제공하지 않는다.
- `--limit`: 제공한다면 **양의 정수만** 허용한다. `0`과 음수는 argparse 단계에서 non-zero다.
- Python API의 `limit`도 `None` 또는 양의 정수만 허용한다.
- `--tag`: 모든 관련 evaluator에 동일한 필터 의미로 전달한다.
- 필터 후 평가 사례가 없으면 조용한 성공으로 처리하지 말고 기존 evaluator 계약과 일치하는 명시적 결과 또는 오류를 반환한다.
- skip된 단계는 통합 report에 상태와 사유를 남긴다. key 자체를 생략하지 않는다.

Phase 7에서 evaluator의 필터링 로직을 다시 구현하지 말고 승인된 공개 API를 호출한다. 공개 API의 차이 때문에 orchestration adapter가 필요하면 baseline 모듈 안에서 최소 범위로 처리한다.

### 4.5 단계 상태와 실패 보존

통합 결과는 최소한 각 단계에 다음 의미를 구분할 수 있어야 한다.

- `success`: 정상 완료
- `failed`: 실행했으나 실패
- `skipped`: 사용자가 옵션으로 제외
- `not_run` 또는 동등 상태: 선행 실패 때문에 안전하게 실행하지 못함

각 실패 단계에는 최소한 다음을 보존한다.

- 단계명
- 오류 유형
- 사람이 읽을 수 있는 오류 메시지
- 관련 파일 경로 또는 가능한 경우 사례 ID
- 다음 조치

한 단계가 report를 먼저 만들고 이후 통합 과정에서 실패했다면 그 report 경로도 보존한다. 이미 성공한 단계 결과를 삭제하거나 최종 결과에서 누락하지 않는다.

`run_baseline()` 내부에서 `sys.exit()`을 호출하지 않는다. library API는 결과 또는 명시적 예외 계약을 사용하고, CLI `main()`이 최종 exit code를 결정하도록 분리한다.

### 4.6 통합 리포트 계약

통합 실행은 timestamp가 붙은 JSON과 Markdown을 생성한다. `evaluation.reporting.write_report()`와 기존 metadata helper를 재사용하고 자체 충돌 회피 로직을 만들지 않는다.

통합 report에는 최소한 다음이 있어야 한다.

- schema version과 생성 UTC
- 실행 명령과 옵션
- dataset 경로와 SHA-256
- Git commit과 dirty 여부
- Python 및 모델·retrieval 설정
- 전체 성공 여부와 최종 종료 의미
- 각 단계의 상태, 집계 지표, 사례 수, 실패 원인과 report 참조
- 총 실행 시간과 가능한 단계별 시간
- top-level `corpus_manifest`
- top-level `corpus_manifest_sha256`
- top-level `vectorstore_fingerprint`
- 재현성 한계 설명

Markdown은 JSON을 단순 dump한 문서가 아니라 사람이 다음 내용을 빠르게 검토할 수 있어야 한다.

- 전체 실행 성공 여부
- Retrieval 핵심 지표와 latency
- Routing accuracy, PR/F1, confusion 및 실패 수
- Answer assertion, abstention, source, intent, latency와 실패 수
- 실패한 단계와 주요 실패 사례
- 실행 환경 fingerprint와 알려진 해석 한계

동적 Markdown table cell은 승인된 공통 escape helper를 사용한다.

### 4.7 fingerprint 불변식

Retrieval 단계가 산출한 재현성 metadata를 통합 report의 top-level 기준으로 사용한다.

필수 조건:

1. top-level `corpus_manifest`, `corpus_manifest_sha256`, `vectorstore_fingerprint`는 항상 Retrieval 단계 값과 동일하고 non-null이다.
2. `--skip-answers`에서도 top-level fingerprint는 유지된다.
3. Answer를 실행하면 Answer가 독립 계산한 `corpus_manifest_sha256`과 `vectorstore_fingerprint`를 Retrieval 값과 비교한다.
4. 둘 중 하나라도 다르면 같은 실행 중 corpus 또는 vectorstore가 변한 것으로 보고 전체 baseline을 실패 처리한다.
5. 불일치가 발생해도 Retrieval, Routing, Answer의 이미 생성된 결과와 두 fingerprint 값을 모두 보존한다.
6. Routing 단계의 corpus/vectorstore metadata는 기존 계약대로 `null`과 `not_applicable` 사유를 유지한다. Routing 값을 top-level로 승격하지 않는다.

dictionary 비교는 key 순서가 아니라 의미상 전체 값이 같은지 확인해야 한다. 비교 대상과 직렬화 규칙은 `reporting.py`의 기존 계약을 그대로 사용한다.

### 4.8 live opt-in

통합 baseline은 실제 모델을 호출하므로 `RUN_LIVE_LLM_TESTS=1` 같은 명시적 opt-in 없이는 실행하지 않는다.

- opt-in이 없으면 모델을 초기화하기 전에 명확한 오류와 non-zero를 반환한다.
- `python -m evaluation.baseline --help`는 모델, vectorstore, Ollama를 초기화하지 않고 exit 0이어야 한다.
- Routing을 skip하더라도 Retrieval과 Answer가 live artifact를 사용한다는 점을 숨기지 않는다.
- Ollama, `data/`, `vectorstore/`가 없으면 어떤 준비가 필요한지 안내한다.
- vectorstore 안내에는 프로젝트의 문서 등록 절차 또는 `document_register.py`를 언급한다.

### 4.9 Phase 7 오프라인 테스트

모든 테스트는 fake evaluator, monkeypatch, 임시 dataset/output을 사용하며 실제 모델, `data/`, `vectorstore/`, Ollama, 네트워크를 사용하지 않는다.

최소 테스트 항목:

1. 정상 실행 순서가 validate → retrieval → routing → answers인지 확인
2. 정상 결과에 네 단계 상태와 집계 결과가 보존되는지 확인
3. dataset validation 실패 시 evaluator가 하나도 호출되지 않고 non-zero 의미가 되는지 확인
4. Retrieval 실패 후 가능한 단계 처리 정책과 기존 결과 보존 확인
5. Routing 실패 후 Answer가 계속 실행되고 전체는 실패인지 확인
6. Answer 실패 시 앞 단계 결과가 유지되고 전체는 실패인지 확인
7. `--skip-routing`, `--skip-answers`, 둘 모두의 상태와 사유 확인
8. `limit=0/-1` API 거부와 CLI exit 2 확인
9. `tag`와 양의 `limit`이 evaluator에 정확히 전달되는지 확인
10. top-level fingerprint가 Retrieval 값과 같은지 확인
11. `--skip-answers`에서도 top-level fingerprint가 유지되는지 확인
12. Retrieval/Answer fingerprint가 일치하면 성공하는지 확인
13. corpus manifest SHA 불일치와 vectorstore fingerprint 불일치를 각각 실패 처리하는지 확인
14. fingerprint 불일치 시 양쪽 값과 단계 결과가 보존되는지 확인
15. Routing의 null/not-applicable metadata가 top-level을 오염시키지 않는지 확인
16. JSON과 evaluator 전용 Markdown에 요구된 핵심 내용이 있는지 확인
17. 질문·오류에 pipe, backslash, 줄바꿈이 있어도 Markdown table이 깨지지 않는지 확인
18. report filename 충돌 시 기존 파일을 덮어쓰지 않는지 확인
19. `--help`와 module import가 live dependency를 초기화하지 않는지 확인
20. 개별 단계 실패 시 최종 CLI가 non-zero, 전체 성공 시 exit 0인지 확인

테스트는 결과 파일이 존재한다는 사실만 확인하지 말고, 상태·지표·fingerprint·실패 보존 invariant의 실제 값을 검증한다.

### 4.10 Phase 7 검증 명령

오프라인 구현 게이트:

```bash
python -m evaluation.baseline --help
pytest -q test_evaluation_baseline.py
pytest -q
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
git diff --check
```

live 실행 게이트는 사용자의 로컬 artifact가 준비된 경우에만 수행한다.

```bash
RUN_LIVE_LLM_TESTS=1 python -m evaluation.baseline \
  --dataset evaluation/datasets/golden.jsonl \
  --output evaluation/reports
```

실행하지 못했다면 성공했다고 추정하지 말고, 미실행 사유와 필요한 준비를 완료 보고에 적는다.

### 4.11 최초 baseline 사용자 승인 게이트

live 실행이 성공해도 자동으로 `m2_initial`을 확정하지 않는다. 먼저 사용자에게 다음을 제시한다.

- 실행 commit과 dirty 여부
- dataset SHA-256
- corpus manifest SHA-256과 vectorstore fingerprint
- 사용 모델과 주요 retrieval 설정
- Retrieval 핵심 지표·latency
- Routing 핵심 지표·latency와 실패 유형
- Answer 핵심 지표·latency
- 전체 및 단계별 성공/실패/제외 수
- 주요 실패 사례 목록과 worksheet 위치
- 측정 한계와 알려진 위험
- timestamped JSON/Markdown 경로

사용자가 명시적으로 승인한 뒤에만 timestamped 결과를 검토된 요약으로 변환하여 다음 경로에 저장한다.

```text
evaluation/baselines/m2_initial.json
evaluation/baselines/m2_initial.md
```

고정 파일 조건:

- 실제 승인된 실행 값과 fingerprint를 그대로 사용한다.
- 임의 목표값, 보정값 또는 재실행하지 않은 추정치를 넣지 않는다.
- 상세 질문·답변·민감 가능 내용은 필요한 최소 수준으로 축약한다.
- 원본 timestamped report 경로 또는 식별 정보를 기록한다.
- 인덱스가 현재 config로 생성됐음을 fingerprint만으로 보장할 수 없다는 provenance 한계를 명시한다.
- assertion coverage는 진실성·faithfulness 전체를 보장하지 않는다고 명시한다.

사용자가 결과 수정이나 재실행을 요청하면 승인 전 결과를 확정하지 말고 요청된 조건으로 다시 실행한다.

## 5. Phase 8 — GitHub Actions CI

### 5.1 목표

Pull Request와 기본 브랜치 push에서 Python·dataset·frontend 회귀 검증을 외부 서비스 없이 자동 실행한다.

### 5.2 구현 파일

- 신규 또는 수정: `.github/workflows/ci.yml`

Phase 8 작업자는 CI를 통과시키기 위한 production/test/dependency 변경을 하지 않는다. workflow 밖의 변경이 필요하면 원인과 제안을 별도 보고한다.

### 5.3 trigger와 최소 권한

workflow는 최소한 다음 이벤트에서 실행한다.

```yaml
on:
  pull_request:
  push:
    branches: [master]
```

실제 기본 브랜치가 `master`가 아니라면 저장소 설정을 확인해 정확한 이름을 사용한다. 추정으로 다른 브랜치를 추가하지 않는다.

workflow는 checkout과 테스트에 필요한 최소 read 권한만 사용한다. secret을 요구하거나 pull request 코드에 write 권한을 부여하지 않는다.

### 5.4 python-tests job

필수 조건:

- runner: GitHub-hosted Ubuntu
- Python 3.11
- `actions/checkout`과 `actions/setup-python`의 명시적 major version
- `requirements.txt` 기반 pip cache
- 저장소 의존성 전체 설치
- `python -m pip check`
- `python -c "import web_server"`
- `pytest -q`
- dataset validation

권장 구조:

```yaml
python-tests:
  runs-on: ubuntu-latest
  steps:
    - name: Checkout
      uses: actions/checkout@v4
    - name: Set up Python 3.11
      uses: actions/setup-python@v5
      with:
        python-version: "3.11"
        cache: pip
        cache-dependency-path: requirements.txt
    - name: Install Python dependencies
      run: python -m pip install -r requirements.txt
    - name: Check Python dependencies
      run: python -m pip check
    - name: Smoke-test web server import
      run: python -c "import web_server"
    - name: Run Python tests
      run: pytest -q
    - name: Validate golden dataset
      run: python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
```

step 이름만 보고도 실패 지점을 식별할 수 있어야 한다. metric 테스트는 전체 `pytest -q`에 포함되므로 같은 테스트를 별도 중복 실행하지 않는다.

### 5.5 frontend-tests job

필수 조건:

- runner: GitHub-hosted Ubuntu
- Node.js 20 이상, 계획 기준은 Node 20
- npm cache와 현재 lockfile 사용
- `npm ci`
- `npm test`
- vendor 동기화
- 동기화 후 `static/vendor/` diff가 없는지 확인

권장 구조:

```yaml
frontend-tests:
  runs-on: ubuntu-latest
  steps:
    - name: Checkout
      uses: actions/checkout@v4
    - name: Set up Node.js 20
      uses: actions/setup-node@v4
      with:
        node-version: "20"
        cache: npm
    - name: Install frontend dependencies
      run: npm ci
    - name: Run frontend tests
      run: npm test
    - name: Sync vendored frontend assets
      run: npm run sync-vendor
    - name: Verify vendored assets are current
      run: git diff --exit-code -- static/vendor/
```

실제 script 이름은 `package.json`을 확인해 사용한다. 존재하지 않는 command를 계획 예시만 보고 만들거나 우회하지 않는다.

### 5.6 CI 격리 조건

일반 PR CI에서는 다음을 실행하거나 요구하면 안 된다.

- Ollama 설치·호출
- `RUN_LIVE_LLM_TESTS=1`
- live Routing, Answer 또는 통합 baseline
- DDGS 및 기타 외부 검색 호출
- Hugging Face 모델 가중치 다운로드
- 로컬 `data/` 또는 `vectorstore/`
- API key나 repository secret

Python 의존성 패키지 설치는 허용되지만 테스트 import나 collection 과정에서 모델 가중치를 내려받아서는 안 된다. 이런 현상이 발생하면 테스트 skip으로 숨기지 말고 지연 import/초기화 계약 위반으로 보고한다.

### 5.7 CI 실행 시간과 cache

- 외부 의존성이 없는 전체 CI는 일반 GitHub-hosted runner에서 10분 이내를 목표로 한다.
- Python cache key는 `requirements.txt`, npm cache는 lockfile 변화에 의해 무효화돼야 한다.
- cache correctness를 위해 dependency hash를 무시하는 수동 고정 key를 만들지 않는다.
- 첫 실행과 cache hit 실행 시간을 각각 완료 보고에 기록한다.
- 10분을 넘으면 검증을 삭제하거나 무거운 테스트를 skip하지 말고 병목 step과 시간을 보고한다.
- 필요하면 후속으로 job 분리나 cache 전략 개선을 제안하되, 요구 검증을 제거하지 않는다.

### 5.8 Phase 8 로컬·실환경 검증

로컬에서 가능한 사전 검증:

```bash
python -m pip check
python -c "import web_server"
pytest -q
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
npm ci
npm test
npm run sync-vendor
git diff --exit-code -- static/vendor/
git diff --check
```

공유 개발 환경의 `pip check`가 다른 프로젝트 패키지 때문에 실패할 수 있다. 이 경우 repository 요구사항 문제라고 단정하거나 `requirements.txt`를 변경하지 말고, 깨끗한 Python 3.11 환경 또는 GitHub runner 결과와 구분해 보고한다.

Phase 8 최종 게이트:

- 실제 Pull Request에서 `python-tests` 성공
- 실제 Pull Request에서 `frontend-tests` 성공
- 각 job과 주요 step의 실행 시간 기록
- Ollama, vectorstore, live opt-in, secret 없이 성공

workflow YAML 작성과 로컬 명령 성공만으로 Phase 8을 최종 완료라고 판단하지 않는다. GitHub 실행 권한이나 PR 생성 권한이 없다면 **구현 완료, 원격 검증 대기**로 보고한다.

## 6. 병합 전략

### 6.1 독립 검증 후 병합

각 브랜치는 자체 게이트를 통과한 뒤 병합 후보가 된다.

Phase 7 후보 조건:

- baseline module import와 `--help`가 side effect 없이 성공
- Phase 7 전용 오프라인 테스트 통과
- 전체 Python 테스트와 dataset validation 통과
- live 미실행 여부와 이유가 명확함
- 사용자 승인 전 baseline 고정 파일이 없음

Phase 8 후보 조건:

- workflow가 요구 trigger와 두 job을 포함
- live 외부 의존성이 없음
- 로컬에서 재현 가능한 명령 통과 또는 환경 오염과 저장소 실패가 구분됨
- 가능하면 PR에서 두 job 성공, 불가능하면 원격 검증 대기 상태가 명확함

### 6.2 권장 병합 순서

파일 소유권이 분리돼 있어 기술적으로 어느 순서든 가능하지만 다음 순서를 권장한다.

1. Phase 7 runner와 오프라인 테스트 병합
2. Phase 8 workflow 병합
3. 통합 브랜치로 PR을 열어 CI 실제 검증
4. 준비된 로컬 환경에서 Phase 7 live baseline 실행
5. 사용자에게 결과 제시
6. 사용자 승인 후 `m2_initial.json/.md`를 별도 변경으로 추가
7. Phase 7·8 최종 코드 리뷰
8. Phase 9 시작

live baseline 실행은 CI 병합 전에도 가능하지만, 고정 baseline은 승인받은 실행 commit과 실제 기록된 commit 관계가 명확해야 한다. 승인 후 코드가 달라졌다면 fingerprint와 Git metadata를 확인하고 필요하면 다시 실행한다.

### 6.3 충돌 처리

- 한 브랜치가 다른 Phase의 단독 소유 파일을 수정했다면 자동으로 선택하지 말고 변경 이유를 검토한다.
- 공통 파일의 서로 다른 변경은 기계적으로 합치지 않는다.
- workflow가 Phase 7 live 명령을 일반 job에 추가했다면 제거하고 요구사항에 맞게 되돌린다.
- Phase 7이 CI 전용 조건 분기나 GitHub 환경 감지를 포함하면 역할 침범 여부를 검토한다.
- 충돌 해결 후 각 Phase 테스트뿐 아니라 전체 검증을 다시 실행한다.

## 7. 통합 완료 게이트

두 Phase 병합 후 최소한 다음을 실행한다.

```bash
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
python -m evaluation.retrieval --help
python -m evaluation.routing --help
python -m evaluation.answers --help
python -m evaluation.baseline --help
pytest -q
npm test
npm run sync-vendor
git diff --exit-code -- static/vendor/
git diff --check
```

추가 확인:

- evaluator와 baseline import가 모델을 초기화하지 않는다.
- Phase 7 통합 report의 top-level fingerprint invariant가 테스트로 증명된다.
- 한 단계 실패 시 이전 결과 보존과 non-zero 종료가 테스트로 증명된다.
- CI에는 live 명령, vectorstore, network secret 의존성이 없다.
- PR에서 Python·frontend job의 실제 상태와 소요 시간을 확인한다.
- live baseline을 실행했다면 timestamped report는 Git에서 제외돼 있다.
- 사용자 승인 전에는 `evaluation/baselines/m2_initial.*`이 없다.

실제 live 실행은 오프라인 통합 게이트와 별개로 기록한다. live 실행 실패를 숨기기 위해 오프라인 성공을 전체 Phase 7 완료라고 표현하지 않는다.

## 8. 중단 및 보고가 필요한 상황

다음 상황에서는 임의 해결보다 사용자 또는 통합 담당자에게 보고한다.

- Phase 1~6 승인 기준 commit을 확정할 수 없음
- dirty 변경이 소유 파일과 겹침
- evaluator 공개 API만으로 Phase 7 orchestration이 불가능함
- 공통 reporting 또는 evaluator 계약 변경이 필요함
- live 실행 중 corpus/vectorstore fingerprint가 변함
- local `data/`와 `vectorstore/`의 출처나 신뢰성을 확인할 수 없음
- baseline 결과에 token, 개인정보, 문서 원문 등 민감 정보가 포함됨
- 최초 baseline을 고정하기 위한 사용자 승인이 없음
- CI를 통과시키려면 production 동작, 테스트 기대값 또는 dependency를 바꿔야 함
- GitHub-hosted runner에서 모델 가중치나 외부 서비스 접근이 발생함
- CI가 10분 목표를 크게 초과하며 요구 검증을 유지한 채 해결하기 어려움
- 기본 브랜치나 required check 설정을 저장소에서 확인할 수 없음

## 9. 완료 보고 형식

각 작업자는 다음 형식으로 보고한다.

```text
Phase:
기준 commit SHA:
최종 commit SHA 또는 uncommitted 상태:

변경 파일:
- ...

구현한 계약:
- ...

실행한 검증:
- 명령: 결과

미실행 검증:
- 항목: 이유와 실행 조건

알려진 한계/위험:
- ...

공통 변경 요청 또는 범위 이탈:
- 없음 / 상세

다음 게이트:
- ...
```

Phase 7은 추가로 다음을 보고한다.

- 단계별 상태와 실패 보존 테스트 결과
- top-level fingerprint 및 Retrieval/Answer 일치 검증 결과
- live opt-in 실행 여부
- live 실행 시 전체 요약 지표, 주요 실패 사례, report 경로
- 사용자 승인 여부
- `m2_initial.*` 생성 여부와 근거

Phase 8은 추가로 다음을 보고한다.

- workflow trigger와 job 목록
- Python/Node/action 버전
- PR check URL 또는 식별 정보
- 첫 실행/cache hit 실행 시간
- Ollama·vectorstore·secret 없이 실행됐는지
- 원격 검증 완료 또는 대기 여부

## 10. Claude Code에 전달할 실행 지시문

다음 지시문을 이 문서와 함께 전달한다.

> `M2_Phase_7_8_dev_detail_plan_and_rule.md`를 처음부터 끝까지 읽고, 상위 Requirement, Development Plan, Design 및 `M2_Phase_4_5_6_code_review_result.md`의 관련 계약을 확인한 뒤 작업하라. Phase 7과 Phase 8을 동일한 Phase 4·5·6 승인 commit에서 별도 worktree/브랜치 또는 명확히 분리된 작업 단위로 병렬 구현하라. Phase 7은 `evaluation/baseline.py`와 전용 테스트만 소유하고, Phase 8은 `.github/workflows/ci.yml`만 소유한다. 공통 evaluator, reporting, production 코드와 dependency 파일은 임의 수정하지 말고 변경이 필요하면 지정 형식으로 요청하라. Phase 7은 validate → retrieval → live routing → answers 순서, 단계 실패 결과 보존, non-zero 종료, skip 상태, positive limit, top-level fingerprint와 Retrieval/Answer 일치 invariant를 오프라인 테스트로 증명하라. live 실행은 명시적 opt-in과 신뢰할 수 있는 로컬 artifact가 있을 때만 수행하고, 최초 baseline은 실행 결과와 주요 실패 사례를 사용자에게 먼저 제시하여 명시적 승인을 받은 뒤에만 `evaluation/baselines/m2_initial.json/.md`로 고정하라. Phase 8은 PR과 기본 브랜치 push에서 Python 3.11 및 Node 20 job을 실행하고, pip check, web_server import, pytest, dataset validation, npm ci/test, vendor diff를 포함하되 Ollama, 네트워크, 모델 가중치, vectorstore, live evaluator 및 secret에 의존하지 않게 하라. workflow 작성만으로 완료라 하지 말고 실제 PR job 성공 여부와 시간을 보고하라. 두 작업을 병합한 뒤 전체 오프라인 검증을 다시 수행하라. Phase 9 문서화, M2 완료 처리, 품질 최적화와 unrelated refactor는 수행하지 마라. 마지막에는 Phase별 변경 파일, 기준/최종 SHA, 테스트와 PR 결과, live 미실행 항목, 사용자 승인 상태, 알려진 한계와 다음 게이트를 분리해 보고하라.

## 11. 최종 완료 체크리스트

### Phase 7 구현

- [ ] `evaluation/baseline.py`와 전용 테스트가 추가됐다.
- [ ] validate → retrieval → routing → answers 순서가 보장된다.
- [ ] 단계 실패 후 이전 결과와 실패 원인이 보존된다.
- [ ] 전체 실패 시 CLI가 non-zero를 반환한다.
- [ ] skip 상태와 사유가 report에 남는다.
- [ ] API/CLI limit은 양수만 허용한다.
- [ ] top-level fingerprint가 Retrieval과 동일하고 non-null이다.
- [ ] Answer fingerprint 불일치를 실패 처리한다.
- [ ] Routing null metadata가 top-level을 오염시키지 않는다.
- [ ] import와 `--help`가 live dependency를 초기화하지 않는다.
- [ ] JSON과 사람이 읽을 수 있는 Markdown을 생성한다.
- [ ] 오프라인 테스트가 실제 모델·data·vectorstore·네트워크를 사용하지 않는다.

### Phase 7 live 및 승인

- [ ] live 실행 전 opt-in과 artifact를 확인했다.
- [ ] timestamped report의 지표와 주요 실패 사례를 사용자에게 제시했다.
- [ ] Git/dataset/corpus/vectorstore/model metadata를 함께 제시했다.
- [ ] 사용자가 최초 baseline을 명시적으로 승인했다.
- [ ] 승인 후에만 `m2_initial.json/.md`를 생성했다.
- [ ] baseline에 측정 한계와 인덱스 provenance 한계가 기록됐다.

### Phase 8

- [ ] PR과 기본 브랜치 push trigger가 있다.
- [ ] Python 3.11 `python-tests` job이 있다.
- [ ] pip check, web server import, pytest, dataset validation을 실행한다.
- [ ] Node 20 `frontend-tests` job이 있다.
- [ ] npm ci, npm test, vendor sync/diff를 실행한다.
- [ ] dependency cache가 requirements/lockfile 변화에 맞춰 무효화된다.
- [ ] job/step 이름으로 실패 지점을 알 수 있다.
- [ ] live evaluator, Ollama, DDGS, 모델 가중치, vectorstore, secret을 요구하지 않는다.
- [ ] 실제 PR에서 두 job 성공 여부와 시간을 확인했다.

### 통합

- [ ] 두 작업이 동일한 승인 commit에서 출발했다.
- [ ] 파일 소유권 위반과 미승인 공통 변경이 없다.
- [ ] 전체 Python·frontend·dataset·vendor·diff 검증이 통과한다.
- [ ] live 미실행 또는 원격 CI 미검증 사항을 완료로 과장하지 않았다.
- [ ] Phase 9와 품질 최적화 범위를 침범하지 않았다.
- [ ] Phase 9가 사용할 runner, CI, 승인 baseline 상태가 명확하다.
