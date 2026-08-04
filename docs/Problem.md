# 알려진 문제

이 문서는 현재 해결되지 않은 문제만 관리합니다. 해결된 문제의 상세 이력은 Git 커밋과 Pull Request에서 확인합니다. 마일스톤 목표와 우선순위는 [Roadmap](Roadmap.md), 현재 품질 수치는 [M2 최초 기준선](../evaluation/baselines/m2_initial.md)을 참조하십시오.

## 우선순위 정의

- **P0**: 데이터 손실, 심각한 보안 사고 또는 서비스 전체 중단. 즉시 해결
- **P1**: 핵심 품질이나 운영 안정성을 크게 제한하는 문제. 다음 관련 마일스톤에서 우선 해결
- **P2**: 제한된 상황의 품질·성능·재현성·유지보수 문제. 계획에 따라 해결
- **P3**: 실제 규모나 요구가 생길 때 검토할 개선 사항

## P1 — MMR 재임베딩으로 인한 Retrieval·Answer 지연

M2 기준선에서 Retrieval 평균 latency는 16.84초, p95는 22.61초였습니다. 이 중 MMR 평균이 14.35초로 대부분을 차지합니다. Answer End-to-End 평균은 55.48초, p95는 74.88초입니다.

현재 MMR은 매 질문마다 후보 문서 본문을 다시 임베딩하며 FAISS 인덱스 생성 시 계산한 벡터를 재사용하지 않습니다.

영향:

- 단일 질문 응답이 느립니다.
- 전체 평가와 튜닝 반복 시간이 길어집니다.
- 후보 문서와 동시 요청이 늘면 CPU·메모리 사용량이 커집니다.

해결 방향:

- FAISS 또는 별도 cache에서 후보 문서 임베딩 재사용
- MMR 입력 후보 수와 계산 방식 실험
- 변경 전후 선택 문서 순서, Recall/MRR/nDCG와 latency를 M2 기준선으로 비교
- 품질 회귀 없이 latency가 줄었는지 확인한 뒤 적용

관련 마일스톤: `M3 — Retrieval & Domain Quality`

## P1 — Document QA를 Web search로 과다 라우팅

M2 live Routing 정확도는 77.63%였습니다. 오분류 17건이 모두 `document_qa → web_search`였고 Document QA recall은 72.13%, Web search precision은 46.88%였습니다.

주로 최신 연도, 정책, 경제, 부동산 또는 기업 동향을 언급하지만 실제 답은 로컬 문서에 있는 질문을 Web search로 보냅니다.

영향:

- 로컬 문서 근거가 있는데도 외부 검색 결과를 보여 줄 수 있습니다.
- 로컬 우선·출처 검증 원칙이 약해집니다.
- 네트워크 의존성과 응답 품질 편차가 불필요하게 증가합니다.

해결 방향:

- 17개 실패 사례를 정책·표현별로 분류
- “최신 표현”과 “로컬 문서 명시” 신호의 우선순위 재설계
- 라우팅 prompt, 경계 규칙 또는 2단계 판정 실험
- Web search recall 100%를 가능한 한 유지하면서 Document QA recall 개선
- 같은 76개 dataset으로 confusion matrix와 latency 비교

관련 마일스톤: `M3 — Retrieval & Domain Quality`

## P1 — Intent Classifier의 도메인 불일치와 낮은 정확도

M2 Answer 대상 29건에서 intent 정확도는 51.72%였습니다. 특히 `yesno`가 comparison/explanation/other로, `uncertain`이 comparison/explanation/other로 자주 분류됐습니다.

현재 학습 데이터는 IT 헬프데스크와 HR 비중이 크고 실제 RAG/AI 문서 질문과 다릅니다. `uncertain` 표현도 충분하지 않습니다. Agent routing과 Intent Classifier가 별도 판단을 수행해 구조 복잡성도 높습니다.

영향:

- 질문에 맞지 않는 답변 형식이 선택될 수 있습니다.
- unanswerable 질문도 일반 비교·설명 형식으로 처리될 수 있습니다.
- 두 분류 계층의 책임과 오류 분석이 복잡합니다.

해결 방향:

- M2 실패 사례를 포함한 실제 도메인 데이터 보강
- `yesno`와 `uncertain`의 confidence calibration
- Intent Classifier가 답변 품질에 주는 실제 효용 측정
- 효과가 작으면 규칙 또는 LLM prompt 기반 형식 선택으로 단순화

관련 마일스톤: `M3 — Retrieval & Domain Quality`

## P1 — 운영 관측성과 동시 요청 대응 부족

주요 단계가 `print()` 로그와 동기 호출로 구현되어 있습니다. FastAPI의 async endpoint 안에서 Ollama, 웹 검색, embedding과 reranker 작업을 동기 실행합니다.

영향:

- 운영 중 단계별 latency와 오류율 집계가 어렵습니다.
- 긴 요청이 이벤트 루프를 점유해 동시 사용자 처리량이 낮을 수 있습니다.
- 프로세스별 대형 모델과 인덱스 로딩 때문에 단순 worker 확장이 비쌉니다.

해결 방향:

- 구조화된 로그와 단계별 latency/error metric
- blocking 작업의 thread offload 또는 비동기 API 검토
- 부하 테스트로 단일 프로세스 처리량과 메모리 기준 확립
- readiness와 liveness 분리

관련 마일스톤: `M4 — Production Readiness`

## P2 — 규칙 기반 Answer 평가의 false negative

M2 자동 assertion 통과율은 75%였지만, 사람 검토 결과 누락된 8개 assertion은 답변에 의미상 모두 포함돼 있었습니다. 띄어쓰기, 영문 약어, underscore와 동의 표현 차이가 원인이었습니다.

자동 abstention false negative 3건도 실제로는 모두 “문서에 정보가 없다”고 올바르게 거절했지만 detector의 제한된 공식 문구와 일치하지 않았습니다.

영향:

- 실제 답변 품질보다 자동 점수가 낮게 측정될 수 있습니다.
- 모델·prompt 변경 효과와 evaluator 표현 민감도가 섞입니다.

해결 방향:

- 숫자·단위·공백·underscore canonicalization
- assertion 동의 표현 보강 원칙 정립
- abstention 표현 패턴 확장
- 자동 규칙 변경 전후에 기존 true positive가 깨지지 않는 회귀 테스트
- M2 기준선 자체는 소급 변경하지 않고 evaluator schema/version을 올려 비교 단절을 명시

관련 마일스톤: `M3 — Retrieval & Domain Quality`

## P2 — Baseline dependency snapshot 부재

고정 baseline에는 Python 버전, 모델, Retrieval 설정, dataset/corpus/vectorstore fingerprint가 있지만 실제 설치된 LangChain, FAISS, Transformers, Torch 등 dependency 버전 snapshot은 없습니다.

`requirements.txt`의 다수 항목이 하한 또는 넓은 범위이므로 같은 Git과 artifact fingerprint에서도 설치 시점에 따라 실행 결과가 달라질 수 있습니다.

2026-08-05 공유 개발환경의 `pip check`에서도 Torch/Torchvision, LangChain, Protobuf, OpenTelemetry, SQLAlchemy 조합의 버전 불일치가 보고됐습니다. 현재 테스트 349건은 통과하지만 새 환경에서 동일 조합이 재현된다는 보장은 없습니다. 프런트엔드도 Node.js 22.17.0에서는 `jsdom`과 `undici`의 engine 경고가 발생해 22.22.2 이상이 필요합니다.

해결 방향:

- 평가 경로 핵심 package의 실제 버전을 `importlib.metadata`로 수집
- 정렬된 `dependency_versions`와 canonical SHA-256 기록
- `requirements.txt` SHA-256도 함께 기록
- 장기적으로 lock 또는 constraints 정책 도입

관련 마일스톤: `M4 — Production Readiness`

## P2 — Vectorstore 생성 provenance 부재

Corpus manifest와 `index.faiss`/`index.pkl` fingerprint는 동일 파일을 사용했는지는 확인하지만, vectorstore가 현재 embedding model, chunk size와 overlap 설정으로 생성됐는지는 증명하지 못합니다.

해결 방향:

- 인덱스 생성 시 schema version, 생성 시각, corpus hash, embedding model, chunk 설정을 sidecar manifest로 저장
- 로드 및 평가 전에 sidecar와 현재 설정을 검증
- 불일치 시 명확한 오류와 재색인 안내

관련 마일스톤: `M4 — Production Readiness`

## P2 — Corpus source ID 충돌 오류의 CLI 처리 불일치

`build_corpus_manifest()`는 정규화된 source ID가 충돌하면 `CorpusManifestError`를 발생시킵니다. 그러나 독립 Retrieval/Answer CLI는 이 예외를 명시적으로 처리하지 않아 traceback이 노출될 수 있습니다. 통합 baseline은 evaluator 예외를 단계 실패로 보존하지만 독립 CLI와 메시지·종료 코드 계약이 일관되지 않습니다.

해결 방향:

- Retrieval, Answer, baseline에서 `CorpusManifestError`를 명시적으로 처리
- 충돌 source ID와 실제 경로, 다음 조치를 출력
- 독립 CLI는 동일한 종료 코드 정책 사용
- 세 실행 경로의 traceback 없는 오류 회귀 테스트

관련 마일스톤: `M4 — Production Readiness`

## P2 — 한국어 BM25 토크나이징 한계

문서와 질문을 단순 공백 기준으로 나누기 때문에 조사, 어미와 복합명사를 충분히 처리하지 못합니다.

해결 방향:

- 형태소 분석기 또는 한국어 tokenizer 후보를 M2 기준선으로 비교
- Dense·Hybrid 대비 실제 추가 이득 측정
- 품질, 인덱싱 시간과 배포 복잡성을 함께 평가

관련 마일스톤: `M3 — Retrieval & Domain Quality`

## P2 — 웹 검색 답변 품질 편차

웹 검색은 DuckDuckGo의 제목과 요약을 최대 3개 나열합니다. 결과의 신뢰도와 관련성에 편차가 있고 여러 출처를 종합하거나 문장 단위 근거를 검증하지 않습니다.

해결 방향:

- 결과 중복 제거, 도메인 신뢰 정책과 관련성 필터
- 웹 콘텐츠를 LLM에 전달한다면 prompt injection 방어와 인용 검증을 함께 설계
- 원문 수집·요약 전 개인정보, 저작권과 robots 정책 검토

관련 마일스톤: `M3 — Retrieval & Domain Quality`

## P2 — 문서 인덱스의 파괴적 전체 재생성

`simple-qna-rag-index`는 기존 `runtime/vectorstore/`를 삭제하고 전체 인덱스를 다시 생성합니다. 증분 인덱싱, 버전 관리, 원자적 교체와 자동 복구가 없습니다.

영향:

- 생성 중 실패하면 기존 정상 인덱스를 잃을 수 있습니다.
- 일부 문서 변경에도 전체 embedding을 다시 계산합니다.

해결 방향:

- 임시 경로에서 새 인덱스를 완성한 뒤 원자적으로 교체
- provenance sidecar와 checksum 검증
- 필요하면 문서 hash 기반 증분 인덱싱

관련 마일스톤: `M4 — Production Readiness`

## P2 — 신뢰되지 않은 FAISS 인덱스 로딩 위험

FAISS 로딩에 `allow_dangerous_deserialization=True`가 사용됩니다. 직접 생성한 로컬 인덱스만 사용하면 허용 가능하지만 외부 인덱스를 로드하면 임의 코드 실행 위험이 있습니다.

해결 방향:

- 외부 인덱스를 직접 로드하지 않는 운영 정책
- artifact 출처와 checksum 검증
- 가능한 경우 pickle 기반 docstore를 피하는 형식 검토

관련 마일스톤: `M4 — Production Readiness`

## P3 — 대규모 문서 집합의 확장성

현재 FAISS `IndexFlatIP`와 전체 메모리 로딩은 소규모 문서 집합에 적합합니다. 실제 문서 수와 부하 지표가 임계치를 넘기 전에는 변경하지 않습니다.

검토 조건:

- 평가·부하 테스트에서 합의한 latency 또는 메모리 기준 초과
- 운영 문서 증가율로 현재 구조의 한계가 예측됨

후보 방향:

- IVF/HNSW 등 ANN 인덱스
- 양자화 또는 외부 vector database
- cache와 인덱스 분할

관련 마일스톤: `M5 — Scale & Advanced Capabilities`
