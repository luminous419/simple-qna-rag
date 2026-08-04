# M2 Phase 3 코드 재리뷰 결과

리뷰 대상:

- `evaluation/metrics.py`
- `evaluation/reporting.py`
- `test_evaluation_metrics.py`
- `test_evaluation_reporting.py`
- `.gitignore`
- M2 요구사항 및 상세 개발 계획의 Phase 3 계약

## 종합 평가

현재 평가: **승인(9.7/10)**

이전 리뷰에서 발견한 P1 1건, P2 2건, P3 1건이 모두 구현과 회귀 테스트에 반영됐습니다. 같은 이름의 리포트를 빠르게 반복 생성하거나 기존 파일과 경로가 충돌해도 이전 결과가 보존됩니다. percentile과 top-k metric은 잘못된 입력을 명시적으로 거부하며, corpus manifest의 파일 추가·삭제·내용 변경도 테스트로 고정됐습니다.

Phase 3 전용 테스트 85건, 전체 Python 테스트와 프런트엔드 테스트가 모두 통과했습니다. 이번 재리뷰에서 Phase 4 진행을 막는 새 문제는 발견되지 않았습니다.

## 이전 리뷰 반영 확인

| 이전 발견 사항 | 상태 | 확인 결과 |
|---|---|---|
| P1 — 같은 이름의 리포트가 1초 안에 생성되면 덮어씀 | 해결 | microsecond timestamp, 배타적 파일 생성, suffix 재시도 적용; 연속 20회 및 강제 충돌 테스트 통과 |
| P2 — `percentile()`이 p 범위를 검증하지 않음 | 해결 | p가 0~100 밖이면 빈 values에서도 `ValueError`; 경계·오류 테스트 추가 |
| P2 — manifest 파일 추가·삭제 테스트 누락 | 해결 | 추가, 삭제, 내용 변경 source 식별 테스트 추가 |
| P3 — Recall/MRR/nDCG가 `k <= 0`을 허용 | 해결 | 공통 `_require_positive_k()` 적용 및 세 함수별 회귀 테스트 추가 |

## 발견 사항

### P3 — Markdown 생성 중 일반 I/O 실패 시 예약된 JSON이 남을 수 있음

위치: `evaluation/reporting.py:160-177`

현재 구현은 JSON을 배타적으로 생성해 stem을 예약하고 Markdown을 생성합니다. Markdown 경로가 이미 존재하는 `FileExistsError`는 JSON을 삭제하고 다음 suffix로 재시도하므로 충돌 상황은 안전하게 처리됩니다.

다만 Markdown 쓰기 중 디스크 용량 부족, 권한 변경 등 `FileExistsError` 이외의 예외가 발생하면 이미 생성된 JSON이 남은 채 함수가 실패할 수 있습니다. 호출자는 예외를 받으므로 성공 리포트로 오인할 가능성은 낮고, 정상적인 평가 실행을 막는 결함도 아닙니다.

후속 권고:

- Markdown 생성 구간에서 일반 예외가 발생해도 이 호출이 생성한 JSON을 정리한 뒤 예외를 다시 발생시킵니다.
- 더 강한 pair 원자성이 필요해지면 임시 디렉터리 또는 임시 파일에 두 결과를 쓴 뒤 최종 경로로 이동하는 방식을 검토합니다.

이 항목은 Phase 4 진행을 막지 않습니다.

## 상세 검토 결과

### Metric 정확성

- `dedupe_preserve_order()`는 최초 등장 순서를 유지하며 중복 source를 제거합니다.
- Recall/MRR/nDCG는 내부에서 서로 다른 중복 제거를 수행하지 않고 동일한 dedupe 결과를 받는 계약입니다.
- Recall@K는 복수 relevant source 중 top-k에서 찾은 비율을 정확히 계산합니다.
- MRR@K는 첫 relevant source의 역순위를 계산하고 k 밖의 결과는 제외합니다.
- nDCG는 `2**grade - 1` gain과 `log2(rank + 1)` discount를 사용합니다.
- relevance grade가 없는 검색 결과는 grade 0으로 처리하고 IDCG가 0이면 0.0을 반환합니다.
- source grade key는 `normalize_source_id()` 기준으로 정규화되며 충돌 시 실패합니다.
- precision/recall/F1과 confusion matrix는 라벨별 TP/FP/FN을 일관되게 사용합니다.
- nearest-rank percentile, 평균 및 중앙값은 빈 latency와 정상 입력을 구분합니다.
- assertion coverage는 Unicode NFC 및 `casefold()` 정규화를 적용합니다.
- Recall/MRR/nDCG의 `k < 1`과 percentile의 범위 밖 p는 명시적인 `ValueError`입니다.

### Reporting 및 재현성

- 공통 메타데이터에 schema version, UTC 생성 시각, Git commit/dirty 상태, dataset hash, Python 및 모델·검색 설정이 포함됩니다.
- 활성 파이프라인에 해당하는 검색 설정만 기록됩니다.
- corpus manifest는 파일별 정규화 source ID, 크기, SHA-256 배열 전체를 포함합니다.
- manifest 배열은 source ID로 정렬되고 canonical JSON 직렬화로 집계 SHA-256을 생성합니다.
- 파일 추가·삭제·내용 변경 시 manifest hash가 바뀌며 entries 비교로 변경 source를 식별할 수 있습니다.
- 같은 basename, 대소문자 차이 및 NFC/NFD 차이로 정규화 source ID가 충돌하면 `CorpusManifestError`가 발생합니다.
- vectorstore fingerprint는 `index.faiss`와 `index.pkl`을 역직렬화하지 않고 바이트 hash만 계산합니다.
- Retrieval/Answer용 재현성 필드는 non-null이고 Routing용 helper는 동일한 key를 null과 사유로 채웁니다.
- JSON은 정렬된 key, UTF-8, 고정 indent로 생성됩니다.
- 연속 실행은 microsecond timestamp로 구분되고 동일 경로 충돌은 배타적 생성과 suffix 재시도로 처리됩니다.
- `evaluation/reports/`는 Git에서 제외되지만 승인된 `evaluation/baselines/` 경로는 제외되지 않습니다.

### 테스트 품질

- metric 및 reporting 테스트는 모델, Ollama, 운영 vectorstore와 네트워크를 사용하지 않습니다.
- 작은 고정 입력과 손으로 확인 가능한 기대값으로 각 계산식을 검증합니다.
- 빈 검색 결과, 복수 relevant source, top-k 경계 중복 source, grade 누락, 단일 route 예측, 빈 latency를 포함합니다.
- NFC/NFD 및 Windows/POSIX 경로 정규화 충돌을 검증합니다.
- 연속 20회 리포트 생성 결과를 모두 보존하는지 확인합니다.
- 고정된 동일 timestamp와 선점된 파일을 사용해 suffix 재시도 경로도 결정론적으로 검증합니다.

## 검증 결과

실행일: 2026-08-03

```text
pytest -q test_evaluation_metrics.py test_evaluation_reporting.py
85 passed in 0.25s

pytest -q
187 passed, 1 skipped, 1 warning

npm test -- --run
1 test file, 9 tests passed

git diff --check
통과
```

전체 Python 테스트의 warning은 공유 conda 환경의 `torchvision` image extension 로드 경고이며 Phase 3 구현 실패는 아닙니다. Live LLM 테스트는 기본 조건에 따라 skip됐으며 Phase 3 단위 테스트는 외부 실행 환경에 의존하지 않았습니다.

## 결론

Phase 3의 metric과 reporting 구현은 요구사항 및 상세 개발 계획의 계산·재현성 계약을 충족합니다. 이전 리뷰에서 확인된 결과 유실 위험과 입력·테스트 경계도 모두 해결됐습니다.

따라서 **M2 Phase 3를 승인하며 Phase 4 진행을 권고합니다.** 남은 P3 항목은 비정상 I/O 실패 시의 정리 강화를 위한 방어적 개선이며 현재 마일스톤 진행을 막지 않습니다.
