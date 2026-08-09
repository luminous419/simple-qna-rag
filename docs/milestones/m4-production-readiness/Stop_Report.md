# M4 Production Readiness 중단 보고서

중단일: 2026-08-08  
중단 단계: 상세 설계 Gate  
최종 판정: **FAIL — 구현 단계 진입 불가**

근거 문서:

- [요구사항](Requirement.md)
- [개발 계획](Plan.md)
- [상세 설계](Design.md)
- [Iteration 1 리뷰](Design_Review_Iteration_1.md)
- [Iteration 2 리뷰](Design_Review_Iteration_2.md)
- [Iteration 3 리뷰](Design_Review_Iteration_3.md)
- [Iteration 4 리뷰](Design_Review_Iteration_4.md)
- [오케스트레이션 지침](../../../milestone_dev_orchestration_guide.md)

## 1. 중단 결정

상세 설계 리뷰를 기본 허용 횟수인 4회 수행했지만 품질 Gate를 통과하지 못했다.
Iteration 4 결과는 4.2/10, CRITICAL 0건, MAJOR 6건, MINOR 3건이다.

조건부 Iteration 5~6 연장은 다음 필수 조건을 충족해야 하지만, 점수 9.0 이상과
MAJOR 2건 이하를 만족하지 못했다. 따라서 지침에 따라 설계 사이클을 중단하며,
코드 구현·통합 테스트·GitHub PR/merge는 수행하지 않는다.

## 2. Iteration 결과

| Iteration | 점수 | CRITICAL | MAJOR | MINOR | 판정 |
|---:|---:|---:|---:|---:|---|
| 1 | 6.8 | 0 | 7 | 3 | FAIL |
| 2 | 5.9 | 0 | 6 | 4 | FAIL |
| 3 | 4.6 | 0 | 7 | 2 | FAIL |
| 4 | 4.2 | 0 | 6 | 3 | FAIL, 연장 자격 없음 |

문서 분량과 구체성은 증가했지만, 실제 라이브러리 API와 race/trust/evidence
경계를 대조할수록 구현 불가능하거나 자기모순인 계약이 계속 발견됐다. 점수와
MAJOR 수가 실질적으로 개선되지 않았으므로 단순한 추가 문서 반복은 적절하지 않다.

## 3. 잔여 MAJOR

1. **QueryExecutor 소유권과 slot 회수**: callback 등록 실패 및 loop shutdown
   경계에서 이미 실행 중인 worker의 slot 소유권과 finalizer 완료가 보장되지 않는다.
2. **Ollama overall worker deadline**: request-scoped client의 per-operation timeout은
   trickle response의 전체 실행 시간을 제한하지 못해 M4-REQ-007.4를 충족하지 않는다.
3. **Legacy index trust root**: approved root 최초 open이 root symlink를 거부하지 않아
   모든 path component가 operator-owned라는 provenance를 증명하지 못한다.
4. **Container build/evidence**: Dockerfile frontend 기능 버전과 명령이 맞지 않고,
   container result/evidence artifact를 함께 생성·업로드하는 계약이 불완전하다.
5. **14-gate 실행 DAG**: CI producer와 local candidate의 run binding, live evidence,
   immutable assemble 및 self-test evidence 수가 서로 모순된다.
6. **Typed settings inventory**: 설계의 Settings 선언은 53개인데 acceptance test는
   30개를 요구하며, 필드별 default/env/validator 계약도 완결되지 않았다.

MINOR는 M3 RetrievalTrace의 `total` 보존, contained-open fd 정리, image scanner
WORKDIR prefix canonicalization 3건이다.

## 4. 재개 조건

현재 설계를 그대로 Iteration 5로 확장하지 않는다. 다음 조건을 먼저 충족한 새 설계
사이클로만 재개한다.

1. Ollama router/answer를 종료 가능한 process boundary로 격리할지, 아니면
   M4-REQ-007.4의 overall worker deadline을 완화할지 architecture 결정을 내린다.
2. QueryExecutor, evidence assembler, container verifier를 문서 의사코드가 아니라
   작은 executable prototype으로 먼저 구현해 race/failure 테스트를 통과시킨다.
3. Settings inventory를 machine-readable schema 한 곳에서 생성하고 field count,
   facade export, validation test가 같은 원본을 사용하도록 확정한다.
4. Container evidence와 최종 Gate를 같은 GitHub workflow/run에서 닫을지,
   서명된 producer attestation을 local candidate에 재결합할지 하나로 고정한다.
5. 새 리뷰 사이클과 iteration 한도를 사용자가 승인한다.

## 5. 보존 상태

- 제품 코드와 M3 승인 baseline은 변경하지 않았다.
- M4 요구사항·계획·설계·추적표와 네 차례 리뷰는 다음 재설계를 위한 감사 기록으로
  보존한다.
- M4 baseline, 구현 결과, 통합/인수 증거, PR 및 merge는 생성되지 않았다.

## 6. 재개 기록

2026-08-08 사용자가 권고안을 승인했다. 기존 설계 Iteration 5를 재개하지 않고
[M4 복구 결정](Recovery_Decision.md)에 따라 M4.1~M4.3으로 분할한 새 사이클을
시작한다. 이 문서의 중단 판정과 네 차례 리뷰는 감사 기록으로 유지한다.
