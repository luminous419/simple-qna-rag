# M4.1 Configuration & Observability Foundation 개발 계획

상태: **착수 — 요구사항·계획 정의**  
요구사항: [Requirement.md](Requirement.md)

## 1. 실행 원칙

M4.1은 dependency → settings → logging/metrics → health → 통합 증거 순으로
직렬 진행한다. concurrency, index, container 설계를 끌어들이지 않는다.

- Codex: 요구사항·계획, 상세 설계/코드 리뷰, Gate 판정
- Claude Code Sonnet 5: 상세 설계, 구현/테스트, 리뷰 반영, 승인 후 Git 작업
- 프로젝트 리더: iteration과 단계 Gate 관리

## 2. Phase

### Phase 0 — 기준선과 inventory 고정

- M3 baseline bytes/fingerprint와 현재 전체 테스트를 재확인한다.
- `config.py` 및 모든 consumer를 symbol/AST로 조사해 settings field spec을 만든다.
- Python/Node dependency snapshot과 기존 CI 환경을 기록한다.

완료 증거: baseline report, settings inventory, 변경 전 테스트 결과.

### Phase 1 — dependency lock과 typed settings

- Linux Python 3.11 hash lock과 갱신 도구/version을 고정한다.
- immutable Settings, validation, cache/reset seam, redacted config check를 구현한다.
- `config.py` facade와 기존 환경변수/CLI 호환을 유지한다.
- clean install과 settings 결정론적 테스트를 추가한다.

### Phase 2 — structured logging과 metrics

- 단일 event/field/error/label schema를 Design에서 확정한다.
- request context와 safe logging/metric wrapper를 제품 seam에 적용한다.
- M3 `RetrievalTrace`를 변경하지 않고 제품 observation을 별도 sink로 투영한다.
- 실제 collector sample cardinality와 payload/secret 금지 테스트를 실행한다.

### Phase 3 — 기본 health와 통합 Gate

- lifespan 기반 settings/engine 상태와 live/ready/deprecated health를 구현한다.
- M4.2가 saturation reason을 추가할 수 있는 stable interface를 제공한다.
- 전체 회귀, M3 14 gate, 문서/traceability/report를 검증한다.

## 3. 설계 전 executable checks

상세 설계는 다음 read-only/작은 spike 결과를 먼저 기록한다.

1. 선택한 lock 도구로 Linux profile lock이 두 번 동일하게 생성되는지 확인한다.
2. prometheus-client의 created-series 제어 public API와 실제 sample 명명 규칙을 확인한다.
3. FastAPI lifespan/TestClient에서 engine failure와 health 상태표 seam을 확인한다.
4. 현재 settings consumer inventory가 누락 0임을 AST와 runtime import 양쪽에서 확인한다.

## 4. 리뷰와 iteration

각 문서/코드 iteration은 최신 orchestration guide의 기본 4회와 조건부 최대 6회
규칙을 따른다. 발견사항을 단순 설명 추가로 덮지 않고 모순된 계약을 교체한다.

Gate는 다음 순서로 닫는다.

1. Requirement ID → Design symbol → test → evidence 추적성 확인
2. Phase 테스트와 전체 회귀 실행
3. Codex 독립 리뷰
4. Claude 수정 및 재검증
5. 9.7 이상, CRITICAL/MAJOR 0일 때 다음 Phase 진행

## 5. 최종 검증 명령군

상세 설계에서 lock 명령과 test path를 확정하되 최소 다음을 유지한다.

```bash
python -m evaluation.dataset validate evaluation/datasets/golden.jsonl
pytest -q
npm ci
npm test
npm run sync-vendor
git diff --exit-code -- web/static/vendor
python scripts/check_markdown_links.py
git diff --check
```

## 6. 완료와 후속

M4.1 Traceability 전부 PASS와 리뷰 Gate 통과 후 Roadmap을 M4.2로 이동한다.
그 전에는 M4.2 concurrency/timeout 구현이나 M4.3 index/container 변경을 시작하지
않는다.

