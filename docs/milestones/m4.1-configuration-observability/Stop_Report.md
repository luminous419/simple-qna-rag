# M4.1 설계 Gate 중단 보고서

중단일: 2026-08-08  
상태: **중단 — 구현 진입 불승인**

## 1. 결론

M4.1 상세 설계는 기본 4회와 조건부 연장 1회를 수행했으나 최종
[Iteration 5 리뷰](Design_Review_Iteration_5.md)에서 **9.3/10, CRITICAL 0,
MAJOR 1, MINOR 1**로 9.7 Gate를 통과하지 못했다.

Iteration 4와 5에서 M3 regression wrapper의 fingerprint 호출 경로가 실제 API로
실행되지 않는 동일 근본 문제가 두 회 연속 재발했다. 이에
`milestone_dev_orchestration_guide.md`의 조건부 연장 즉시 중단 규칙을 적용해 남은
Iteration 6을 실행하지 않고 구현 진입을 중단한다.

## 2. 최종 blocker

legacy facade의 `VECTORSTORE_PATH`는 `str`이지만
`evaluation.reporting.build_vectorstore_fingerprint()`는 `Path`를 요구한다. 현 설계의
wrapper는 이 경계에서 정규화하지 않아 baseline 실행 전 `TypeError`가 재현된다.

따라서 다음 계약은 아직 증명되지 않았다.

- M4.1-REQ-006.2: M3 baseline/runtime vectorstore 불변
- M4.1-REQ-006.4: M3 자동 회귀와 JSON/Markdown 단일 판정 모델
- M4.1 구현 단계 진입 Gate

## 3. 폐쇄된 항목

- dependency lock/tool/snapshot/CI 계약
- typed Settings, env/CLI overlay, legacy facade 투영 계약
- payload-safe logging과 전체 output-surface audit
- 7-family/102-sample bounded metrics 계약
- bootstrap/lifespan live/ready 상태표
- M3 wrapper의 `overall_success` + 14-gate 결합 판정 방향

## 4. 재개 조건

자동으로 Iteration 6을 이어가지 않는다. 별도 재개 결정 후 새 리뷰 사이클에서 다음을
먼저 제시해야 한다.

1. wrapper가 typed Settings의 `Path`를 직접 주입하거나 호출 경계에서
   `Path(VECTORSTORE_PATH)`로 정규화하는 단일 source/type 계약
2. legacy facade `str`과 임시 canonical `index.faiss`/`index.pkl`을 사용해 실제
   fingerprint 호출이 성공하는 integration spike/test
3. `Field_Spec_Inventory.md`의 `MODEL_VALIDATORS` 표를 정확한 5열로 정정
4. Markdown 링크 검사와 `git diff --check` 통과

재개 리뷰에서 9.7 이상, CRITICAL/MAJOR 0, MINOR 최소화를 확인하기 전에는 제품
구현·M4.2·M4.3을 시작하지 않는다.
