# M4 Production Readiness 복구 결정

결정일: 2026-08-08  
상태: **승인 — 분할 실행**  
근거: [중단 보고서](Stop_Report.md), [Iteration 4 리뷰](Design_Review_Iteration_4.md)

## 1. 결정

기존 M4 요구사항·계획·설계와 네 차례 리뷰는 감사 기록으로 동결한다. 기존 설계의
Iteration 5를 이어가지 않고 M4 목표를 다음 세 하위 마일스톤으로 나누어 순차
완료한다.

1. **M4.1 Configuration & Observability Foundation**
2. **M4.2 Safe Serving Boundary**
3. **M4.3 Artifact & Deployment Safety**

각 하위 마일스톤은 독립 Requirement/Plan/Design/Traceability와 리뷰 iteration을
갖는다. 앞 단계의 승인된 API와 증거만 다음 단계의 입력으로 사용한다.

## 2. Architecture 결정

### 2.1 Timeout 정책

M4.2는 로컬 단일 프로세스 서비스의 현실적인 경계를 채택한다.

- HTTP request deadline과 빠른 timeout 응답은 보장한다.
- timeout/cancel 뒤 실행 중인 동기 worker는 실제 외부 호출이 끝날 때까지 slot을
  계속 점유한다. 실행 중인 thread를 강제 종료하거나 slot을 조기 반환하지 않는다.
- orphan worker 수와 포화 상태를 metric/readiness에 반영한다.
- 모든 slot이 일정 시간 이상 orphan이면 readiness 503과 신규 요청 거절을 유지하고,
  runbook의 supervisor restart 절차로 회복한다.
- Ollama connect/read timeout은 request deadline 이하로 제한하지만, trickle response의
  강제 worker 종료는 M4 필수 조건에서 제외한다.

따라서 기존 M4-REQ-007.4의 “외부 호출 overall worker deadline”은 M4.2에서
“bounded HTTP response + honest slot ownership + bounded upstream operation timeout +
supervisor recovery”로 대체한다. killable subprocess 격리는 실측상 supervisor
방식으로 회복할 수 없을 때 M5 후보로 둔다.

### 2.2 Gate 실행 위치

최종 M4.3 Gate는 하나의 GitHub Actions workflow/run에서 실행한다. 로컬 결과와
선행 CI artifact를 별도 run ID로 재조립하지 않는다. live 실행이 필요한 경우 같은
workflow의 명시적 protected job에서 수행하고, 필수 secret/runner가 없으면 M4.3을
완료로 선언하지 않는다.

### 2.3 Index 승인 경계

legacy import는 repository에 커밋된 M3 baseline의 정확한 index hash만 승인한다.
임의 CLI hash pair는 승인 증거로 사용하지 않는다. 새 index는 staging, manifest/hash
검증, atomic activation을 거친다.

### 2.4 Settings 단일 원본

field count를 문서에 중복 기재하지 않는다. machine-readable field specification 또는
`Settings` metadata가 field/type/default/env alias/validator/consumer의 단일 원본이며,
테스트와 호환 facade는 이 원본에서 count/mapping을 계산한다.

### 2.5 설계 방식

race, filesystem trust, container scanner, evidence assembler처럼 문서만으로 증명하기
어려운 항목은 executable prototype과 실패 테스트를 먼저 만든 뒤 검증된 symbol과
상태 전이만 Design에 반영한다. prototype은 해당 하위 마일스톤 구현에 포함하며
설계 Gate를 우회하는 제품 구현으로 간주하지 않는다.

## 3. 하위 마일스톤 경계

| 마일스톤 | 포함 | 명시적 제외/후속 |
|---|---|---|
| M4.1 | dependency lock, typed settings, structured logging, bounded metrics, 기본 live/ready, 회귀 보존 | concurrency, request timeout, index lifecycle, container |
| M4.2 | blocking offload, bounded concurrency/queue, timeout/cancel/orphan, drain, body/host/CORS/error, load/fault tests | index lifecycle, production container, 최종 14-gate assemble |
| M4.3 | versioned index, provenance, atomic activation/rollback, OCI container/runbook, 단일-workflow 전체 Gate와 M4 baseline | distributed queue/vector DB/Kubernetes |

## 4. 진행 Gate

- M4.1 → M4.2: M4.1 Requirement 전부 PASS, 전체 M3 회귀, 리뷰 9.7 이상,
  CRITICAL/MAJOR 0.
- M4.2 → M4.3: bounded load/health/drain/orphan recovery 자동 테스트 PASS와 같은
  리뷰 Gate.
- M4 완료: M4.1~M4.3 추적표 전부 PASS, 단일 workflow의 전체 필수 Gate PASS,
  M4 baseline 생성 및 Roadmap 갱신.

