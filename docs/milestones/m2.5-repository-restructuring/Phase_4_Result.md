# M2.5 Phase 4 결과

측정일: 2026-08-05 (Asia/Seoul)

상태: **완료** — runtime 경로 전환, 비파괴 이동과 호환 계약 검증 통과

## 실제 이동

| 기존 | 새 위치 | 결과 |
|---|---|---|
| `data/` | `runtime/documents/` | 18개 파일 보존 |
| `vectorstore/` | `runtime/vectorstore/` | `index.faiss`, `index.pkl` 보존 |

target이 없는 것을 확인한 뒤 디렉터리를 이동했습니다. 자동 병합, 덮어쓰기, 삭제 또는 재색인은 수행하지 않았습니다.

## 경로 계약

기본 경로:

- 문서: `<repo>/runtime/documents`
- vectorstore: `<repo>/runtime/vectorstore`
- intent model: `<repo>/models/intent_classifier`

환경변수:

- `SIMPLE_QNA_RAG_DOCUMENTS_DIR`
- `SIMPLE_QNA_RAG_VECTORSTORE_DIR`
- `SIMPLE_QNA_RAG_MODEL_DIR`

우선순위는 `CLI > environment > repository default`입니다. 새 기본 경로가 없고 기존 경로만 있으면 `FutureWarning`과 함께 한시적으로 기존 경로를 사용합니다. 새 경로와 기존 경로가 동시에 존재하면 자동 병합하지 않고 `RuntimeError`로 중단합니다.

## CLI override

- `simple-qna-rag-index --documents-dir ... --vectorstore-dir ...`
- `simple-qna-rag-query --vectorstore-dir ... --model-dir ...`
- `simple-qna-rag-web --documents-dir ... --vectorstore-dir ... --model-dir ...`

세 명령의 도움말과 환경변수 override를 subprocess에서 확인했습니다.

## Fingerprint 비교

| 보호 대상 | Phase 0 | Phase 4 |
|---|---|---|
| corpus 파일 수 | 18 | 18 |
| corpus manifest | `5c0d648d032bc231c93cf1c545e5b3f8d26337da56677a4a0c1a66f24c82374a` | 동일 |
| `index.faiss` | `c52fb288f3bc780d681ab68d62dd5b4c545843ee9dc53ef5528bd50db4d69820` | 동일 |
| `index.pkl` | `3f7217a2825f141bbea4c75dbf2b15738c3bcd50e3c0900e25a6a4a7ed91bb00` | 동일 |

## 검증

- 새 기본 경로, 환경변수 우선, legacy fallback, 양쪽 경로 충돌 테스트 통과
- runtime/ 전체 Git 제외 확인
- Golden dataset validation 76건 통과
- Python 전체 테스트 통과
- Frontend 전체 테스트 통과
- M2 승인 dataset/baseline 파일 내용 변경 없음

Phase 4는 완료됐습니다. Phase 5에서 전체 문서, 테스트, 링크, packaging, CI 설정과 M2.5 전체 완료 조건을 최종 검증합니다.
