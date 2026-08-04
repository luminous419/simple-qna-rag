# M2.5 Phase 3 결과

측정일: 2026-08-05 (Asia/Seoul)

상태: **완료** — Web·학습·모델 자산 이전 및 회귀 검증 통과

## 이전 결과

| 이전 대상 | 새 위치 |
|---|---|
| `templates/`, `static/` | `web/templates/`, `web/static/` |
| intent dataset 생성·학습 코드 | `training/intent_classifier/` |
| intent 학습 dataset | `training/intent_classifier/datasets/` |
| intent model 설정·가중치 | `models/intent_classifier/` |

config, FastAPI asset 경로, frontend import, vendor sync, CI vendor diff와 학습 기본 입출력 경로를 모두 새 위치로 변경했습니다. 학습 Shell script의 사용자별 절대 venv 경로도 제거했습니다.

## 검증 결과

- Python: `353 passed, 1 skipped`
- Frontend: `9 passed`
- `npm ci`와 `web/static/vendor/` 동기화·diff 성공
- Web template/static와 intent model 기본 경로 존재 확인
- 학습 generate/train module import 성공, 학습이나 dataset 재생성은 실행하지 않음
- 제품 코드, M2 평가 dataset/baseline과 runtime 데이터 내용 변경 없음

Phase 3 이후에도 일반 wheel에는 repository 외부의 Web/model 자산이 포함되지 않습니다. 완전한 wheel 배포는 M2.5의 목표가 아니며 repository checkout의 editable install을 공식 개발·실행 방식으로 유지합니다.
