# M4.1 post-merge 운영 acceptance 예외 결정

결정일: 2026-08-10  
결정자: 사용자  
대상 merge SHA: `fd14eecfd6036a8192c04037213e39593c5c9d27`  
상태: **M4.1 미완료 보존 / 위험 수용 후 M4.2 착수 허용**

## 결정

M4.1 구현과 pre-merge Code Quality Gate 결과는 유지한다. 그러나 필수
`m3-live-regression-gate`의 완주 증거와 `m4-regression-report` receipt가 없으므로
M4.1을 완료로 표시하지 않는다.

사용자는 native Linux X64 실행 환경을 현재 확보할 수 없는 제약을 확인하고, 이
운영 acceptance 미완료 위험을 명시적으로 수용해 M4.2 개발 cycle의 착수를
예외적으로 허용했다. 이는 M4.1 Gate 면제나 PASS 판정이 아니며,
`Plan.md`의 정상 선행조건에 대한 한정된 사용자 예외다.

## 확보된 증거

- PR #15가 `master`의 대상 SHA로 병합됐다.
- hosted `frontend-tests`와 `python-tests`는 성공했다.
- Linux X64 Docker 에뮬레이션에서 lock 설치, 프로젝트 설치, vectorstore 및
  Ollama preflight가 성공했다.
- live 실행의 부분 보고서에서 retrieval 42건의 Recall@10은 약 0.9762,
  MRR@10은 약 0.9821, NDCG@10은 약 0.9543이었다.
- routing 76건은 성공 76, 실패 0, accuracy 1.0이었다.
- 실행 전후 `index.faiss`와 `index.pkl`의 SHA-256은 변하지 않았다.

## 미확보 증거와 영향

- GitHub Actions run `31305429161`의 live job은 Mac의 Linux X64 에뮬레이션
  성능으로 45분 제한을 초과해 `cancelled`로 끝났다.
- final-answer 평가와 14개 Gate aggregate가 생성되지 않았다.
- 따라서 post-merge Operational Acceptance Gate, 전체 14/14 결과 및 정상
  `m4-regression-report` receipt는 미통과 상태다.
- M4.2 변경은 진행할 수 있지만, M4.1 완료나 M4 전체 release readiness의 근거로
  이 예외를 사용할 수 없다.

## 후속 통제

1. 이 항목을 기술부채이자 M4 release 차단 조건으로 유지한다.
2. M4.2는 M4.1의 settings, logging, metrics 및 health interface를 사용할 수 있지만
   M4.1 live 품질 보존을 검증했다고 가정해서는 안 된다.
3. native Linux X64 또는 동등하게 신뢰할 수 있는 실행 환경이 확보되면 대상 SHA
   또는 그 후속 release candidate에서 live 14-gate와 receipt 검증을 재실행한다.
4. M4.3 완료 또는 M4 release 판정 전에는 이 부채를 반드시 해소하거나 사용자의
   별도 release-risk 승인을 받아야 한다.

