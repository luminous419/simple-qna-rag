# 알려진 문제

이 문서는 현재 해결되지 않은 문제만 관리합니다. 해결된 문제의 상세 이력은 Git 커밋과 Pull Request에서 확인합니다. 로드맵 수준의 목표와 우선순위는 [Roadmap.md](Roadmap.md)를 참조하십시오.

## 우선순위 정의

- **P0**: 데이터 손실, 심각한 보안 사고 또는 서비스 전체 중단. 즉시 해결
- **P1**: 핵심 품질이나 운영 안정성을 막는 문제. 다음 마일스톤에서 해결
- **P2**: 제한된 상황의 품질·성능·유지보수 문제. 계획에 따라 해결
- **P3**: 규모나 요구가 생길 때 검토할 개선 사항

## M2 상세 개발 계획 6차 검토 결과

업데이트된 [Development_M2_Quality_Baseline_Development_Plan.md](Development_M2_Quality_Baseline_Development_Plan.md)는 5차 검토의 두 항목을 모두 반영했습니다. clean-venv 게이트를 실제 실행해 저장소 의존성 설치와 Web/API import가 정상임을 확인했고, corpus의 정규화 source ID 충돌과 골든셋 내부 중복을 각각 검출하는 오류·테스트 계획도 추가했습니다.

현재 실행 가능성 평가: **매우 높은 실행 가능성(9.4/10)**

Phase 1 착수는 가능합니다. 다만 M2의 핵심 목적인 baseline 비교 가능성을 위해 아래 P1 메타데이터 계약을 Phase 3 이전에 확정해야 하며, 새 오류의 CLI 처리와 문서 정합성도 해당 Phase에서 함께 보완해야 합니다.

### P1 — 실제 설치된 Python 의존성 버전이 baseline 메타데이터에 없음

`requirements.txt`는 LangChain, Transformers, FAISS, NumPy, FastAPI 등 다수 패키지에 하한 또는 넓은 범위만 지정합니다. 실제 clean venv에서도 현재 resolver가 선택한 `fastapi 0.141.1`, `pydantic 2.13.4`가 설치됐지만, 이후 같은 Git commit에서 다시 설치하면 더 새로운 버전이 선택될 수 있습니다.

현재 리포트 메타데이터는 Python 버전과 모델·retrieval 설정은 기록하지만 실제 설치된 라이브러리 버전이나 dependency snapshot은 기록하지 않습니다. LangChain, FAISS, sentence-transformers 등의 버전 차이는 검색 결과, serialization 또는 호출 동작을 바꿀 수 있으므로 동일한 Git/data/vectorstore fingerprint를 가진 두 baseline도 실제 실행 환경이 다를 수 있습니다.

해결 방향:

- 리포트에 최소한 평가 경로에 직접 영향을 주는 패키지 버전을 기록합니다: `langchain`, `langchain-core`, `langchain-community`, `langchain-huggingface`, `langchain-ollama`, `faiss-cpu`, `numpy`, `sentence-transformers`, `transformers`, `torch`, `pydantic`.
- `importlib.metadata.version()`으로 수집한 정렬된 `dependency_versions` 객체와 canonical SHA-256을 top-level metadata에 추가합니다.
- `requirements.txt` 자체의 SHA-256도 기록해 선언된 의존성과 실제 resolve 결과를 함께 비교할 수 있게 합니다.
- 패키지가 없거나 버전을 읽지 못할 때 필수 패키지는 명확한 오류로 실패시키고, 선택 패키지는 `not_installed`로 기록하는 정책을 정합니다.
- 패키지 순서와 무관하게 동일한 hash가 생성되고 버전 하나가 바뀌면 hash가 변경되는 단위 테스트를 추가합니다.

관련 위치:

- `Development_M2_Quality_Baseline_Development_Plan.md` §3.6 리포트 메타데이터
- `Development_M2_Quality_Baseline_Requirement.md` M2-REQ-010
- `requirements.txt`

### P2 — `CorpusManifestError`의 evaluator CLI 처리 계약 누락

source ID 충돌 시 `build_corpus_manifest()`가 `CorpusManifestError`와 충돌 경로 목록을 반환하도록 한 것은 적절합니다. 그러나 `build_reproducibility_metadata()`와 evaluator CLI의 오류 처리 설명은 여전히 `FileNotFoundError`만 명시합니다.

그대로 구현하면 충돌을 올바르게 탐지하고도 Retrieval, Answer 또는 baseline CLI에서 traceback이 노출되거나 evaluator마다 서로 다른 종료 코드를 반환해 M2-REQ-016의 예측 가능한 오류 처리 조건을 충족하지 못할 수 있습니다.

해결 방향:

- Retrieval, Answer, baseline `main()`이 `CorpusManifestError`를 공통으로 잡아 오류 종류, 충돌 source ID, 실제 경로 목록을 stderr에 출력하고 `exit(2)`로 종료하도록 합니다.
- 부분 단계 결과가 이미 생성된 통합 baseline에서는 실패 원인을 보존하고 최종 비0 종료 정책을 따릅니다.
- 세 CLI 각각에 충돌 manifest mock을 주입해 traceback 없이 동일한 종료 코드와 메시지 구조를 반환하는 테스트를 추가합니다.

관련 위치:

- `Development_M2_Quality_Baseline_Development_Plan.md` §3.3, §3.6, Phase 4·6·7
- `Development_M2_Quality_Baseline_Requirement.md` M2-REQ-016

### P3 — Phase 0 완료 결과와 위험 설명의 오래된 문구가 충돌

§1에서는 clean venv 검증으로 공유 환경 오염임을 확정했지만 §7은 여전히 “깨끗한 venv로 확인하기 전까지는 단정하지 않는다”고 서술합니다. 또한 “이번 마일스톤 동안 발생하는 모든 실패는 M2 변경에 기인한다”는 결론은 외부 서비스 상태, 로컬 모델, 파일 변경 또는 비결정적 실행 가능성을 배제하므로 지나치게 강합니다.

해결 방향:

- §7의 공유 환경 위험 문구를 clean-venv 검증 완료 상태로 갱신합니다.
- 이후 실패는 우선 M2 회귀로 조사하되, Git revision·dependency fingerprint·data/vectorstore fingerprint·외부 모델 상태를 확인한 뒤 원인을 판정한다고 수정합니다.

### 수용된 P2 후속 문제 — 인덱스 생성 provenance 부재

vectorstore가 현재 embedding/chunk 설정으로 만들어졌음을 증명하지 못하는 문제는 M2 범위 밖으로 명확히 기록됐고 최초 baseline에도 표시하도록 계획됐습니다. M2 blocker는 아니며 후속 마일스톤에서 sidecar manifest로 해결합니다.

### 실행 권고

1. 현재 상태로 Phase 1~2를 진행할 수 있습니다.
2. Phase 3의 reporting schema를 구현하기 전에 dependency snapshot 필드를 Requirement와 상세 계획에 추가합니다.
3. Phase 3~7에서 `CorpusManifestError`의 공통 CLI 처리와 테스트를 포함합니다.
4. §7의 오래된 Phase 0 설명을 현재 검증 결과와 맞춥니다.

일정은 한 명 기준 약 2~3주로 여전히 타당합니다.

## M2 상세 개발 계획 5차 검토 결과 — 계획에 반영 완료

아래 두 항목은 5차 검토에서 발견됐으며 업데이트된 상세 계획에 모두 반영됐습니다. 인덱스 생성 provenance는 범위 밖 후속 문제로 6차 결과에 이어서 관리합니다. 현재 조치 대상은 위의 6차 검토 결과를 따릅니다.

현재 실행 가능성 평가: **매우 높은 실행 가능성(9.5/10)**

설계상 구현을 막는 문제는 없습니다. 아래 P1 실행 게이트를 실제로 통과한 뒤 Phase 1에 착수할 수 있으며, P2 항목은 Phase 1의 validator 구현 때 함께 보완하는 것이 적절합니다.

### P1 — Phase 0이 “실행 완료”로 표시됐지만 필수 clean-venv 검증은 미완료

Phase 0 표에는 기존 `pytest`, `npm test`, 공유 conda 환경의 import 실패 결과가 기록돼 있지만, 계획이 저장소 자체의 실행 가능성을 판정하는 기준으로 정한 깨끗한 Python 3.11 venv 검증은 아직 체크되지 않았습니다. 따라서 절 제목의 “실행 완료”와 본문의 미완료 체크리스트가 서로 충돌합니다.

특히 현재 공유 환경에서는 `import web_server`와 `TestClient` import가 실패하므로, clean venv 결과 없이 Phase 1을 시작하면 이것이 저장소 의존성 문제인지 공유 환경 오염인지 확정할 수 없습니다.

해결 방향:

- Phase 0 제목을 검증 완료 전까지 “부분 완료” 또는 “상태 기록 완료, 착수 게이트 미완료”로 변경합니다.
- 깨끗한 Python 3.11 venv에서 `pip install -r requirements.txt`, `pip check`, `import web_server`, `TestClient` import, `pytest -q`, `npm test`를 실행합니다.
- clean venv에서도 실패하면 `email-validator>=2.0` 등 필요한 의존성 수정을 Phase 1보다 먼저 별도 커밋으로 처리합니다.
- 결과와 실행 일시를 Phase 0 표에 기록하고 체크리스트를 완료한 뒤에만 Phase 1을 시작합니다.

관련 위치:

- `Development_M2_Quality_Baseline_Development_Plan.md` §1 Phase 0

### P2 — 정규화된 source ID의 유일성 검증 누락

source ID는 경로를 제거한 basename을 NFC 정규화하고 `casefold()`한 값입니다. 이 방식은 현재의 평평한 `data/` 구조에는 적합하지만, 서로 다른 하위 디렉터리에 같은 파일명이 있거나 대소문자·Unicode 표현만 다른 두 파일이 존재하면 서로 다른 문서가 같은 source ID로 합쳐집니다.

이 경우 골든셋의 `relevant_sources`가 어느 파일을 뜻하는지 모호해지고, corpus manifest도 같은 `source_id`를 가진 여러 항목을 생성합니다. `source_id` 하나만을 정렬 키로 사용하므로 충돌 항목의 순서까지 파일 순회 순서에 의존할 수 있어 manifest hash의 결정론도 약해집니다.

해결 방향:

- dataset/corpus validator가 정규화된 source ID 중복을 검사하고 충돌한 실제 경로 목록과 함께 실패하도록 합니다.
- `build_corpus_manifest()`도 중복 source ID를 발견하면 명확한 오류를 반환하도록 동일 규칙을 재사용합니다.
- 같은 basename의 서로 다른 경로, 대소문자만 다른 이름, NFC/NFD가 다른 이름을 경계 테스트에 추가합니다.
- 향후 하위 디렉터리의 동일 basename을 지원해야 한다면 basename 대신 `data/` 기준 정규화 상대 경로로 schema version을 올려 전환합니다.

관련 위치:

- `Development_M2_Quality_Baseline_Development_Plan.md` §3.3 `normalize_source_id()`
- `Development_M2_Quality_Baseline_Development_Plan.md` §3.6 `build_corpus_manifest()`

### 수용된 P2 후속 문제 — 인덱스 생성 provenance 부재

현재 계획은 같은 corpus/vectorstore 파일을 사용했는지는 증명하지만, 해당 vectorstore가 현재 embedding model과 chunk 설정으로 생성됐는지는 증명하지 못합니다. 이 한계는 계획과 최초 baseline에 명시하도록 반영됐으므로 M2 착수의 blocker는 아닙니다.

후속 마일스톤에서 `document_register.py`가 인덱스 생성 시점의 corpus hash, embedding 모델, chunk size/overlap, 생성 시각, schema version을 sidecar manifest로 저장하도록 개선해야 합니다.

### 실행 권고

1. Phase 0 clean-venv 게이트를 실제로 실행하고 결과를 문서화합니다.
2. 통과하면 Phase 1을 시작하면서 source ID 유일성 validator와 테스트를 추가합니다.
3. 이후 Phase 2의 골든셋 사용자 검토 게이트부터는 현재 상세 계획대로 진행합니다.

한 명 기준 약 2~3주의 일정과 Phase 0~9의 작업 순서는 여전히 현실적입니다.

## M2 상세 개발 계획 4차 검토 결과 — 계획에 반영 완료

아래 두 계획 결함은 4차 검토에서 발견됐으며 업데이트된 상세 계획과 Requirement에 모두 반영됐습니다. 인덱스 생성 provenance는 범위 밖 후속 문제로 5차 결과에 이어서 관리합니다. 현재 조치 대상은 위의 5차 검토 결과를 따릅니다.

현재 실행 가능성 평가: **매우 높은 실행 가능성(9.3/10)**

아래 리포트 계약 두 항목을 구현 전에 명확히 하면 Phase 0부터 실행할 수 있습니다. 두 항목 모두 국소적인 schema 및 메타데이터 보완으로 해결 가능하며 전체 Phase 구성이나 예상 기간을 변경할 정도의 문제는 아닙니다.

### P1 — 리포트에 corpus manifest 전체 내용이 포함되는지 불명확

`M2-REQ-010`은 `data/`의 각 파일에 대한 정규화 source ID, 크기, SHA-256과 전체 목록의 SHA-256을 리포트에 요구합니다. 상세 계획의 `build_corpus_manifest()`도 이 전체 manifest를 생성합니다.

그러나 `build_reproducibility_metadata()`의 반환 예시와 메타데이터 필드 목록에는 `corpus_manifest_sha256`만 있고, 파일별 manifest 자체나 이를 가리키는 경로가 없습니다. 구현자가 예시 그대로 작성하면 전체 hash는 비교할 수 있지만 어떤 파일이 달라졌는지 확인하거나 Requirement의 파일별 메타데이터 조건을 충족할 수 없습니다.

해결 방향:

- 리포트에 `corpus_manifest` 배열과 `corpus_manifest_sha256`을 함께 포함하거나, 별도 manifest JSON을 생성하고 리포트에 상대 경로와 SHA-256을 기록합니다.
- 별도 파일 방식을 사용한다면 JSON/Markdown 리포트와 함께 이동해도 참조가 유지되는 상대 경로 규칙을 정합니다.
- manifest 항목의 정렬 기준과 전체 hash 계산에 사용한 canonical JSON 형식을 schema version에 포함합니다.
- 파일 추가·삭제·내용 변경 시 전체 hash가 바뀌고, 파일별 ID/크기/hash가 리포트 또는 연결된 manifest에서 확인되는 테스트를 추가합니다.

관련 위치:

- `Development_M2_Quality_Baseline_Requirement.md:306-323`
- `Development_M2_Quality_Baseline_Development_Plan.md` §3.6 `build_corpus_manifest()` 및 메타데이터 필드 목록

### P2 — 통합 baseline의 top-level fingerprint 계약이 모호함

상세 계획은 통합 baseline이 Retrieval에서 계산한 재현성 메타데이터를 Answer에 재사용하고 Routing 단계의 값은 `null/not_applicable`로 유지하도록 했습니다. 이 방향은 타당하지만, 최종 통합 JSON/Markdown의 **top-level** `corpus_manifest`와 `vectorstore_fingerprint`가 non-null인지, 아니면 단계별 결과에만 존재하는지가 명확하지 않습니다.

Requirement는 통합 baseline도 실제 corpus/vectorstore를 사용하는 평가로 분류하므로 최종 리포트 소비자가 단계 내부를 해석하지 않아도 실행 환경을 식별할 수 있어야 합니다.

해결 방향:

- 통합 baseline top-level에는 Retrieval에서 계산한 non-null corpus/vectorstore 메타데이터를 기록합니다.
- 단계별 metadata에서도 Retrieval/Answer는 동일한 fingerprint, Routing은 `null`과 사유를 유지합니다.
- Retrieval과 Answer 단계의 fingerprint가 다르면 통합 실행을 실패 처리하도록 invariant를 정의합니다.
- `--skip-answers`에서도 top-level fingerprint가 유지되고, Routing만 단독 실행하는 경우에는 통합 baseline이 아니라 Routing report 규칙을 따른다는 테스트를 추가합니다.

관련 위치:

- `Development_M2_Quality_Baseline_Requirement.md:318-323`
- `Development_M2_Quality_Baseline_Development_Plan.md` §3.6 evaluator별 적용 범위 및 Phase 7

### 범위 밖 후속 문제 — 인덱스 생성 provenance 부재

현재 fingerprint는 두 실행이 같은 corpus와 vectorstore 파일을 사용했는지는 판별하지만, 그 vectorstore가 현재 embedding model, chunk size, overlap 설정으로 생성됐는지는 증명하지 못합니다. 상세 계획도 이를 M2 범위 밖으로 명시했습니다.

M2 착수를 막지는 않지만 최초 baseline을 해석할 때 반드시 한계로 표시해야 하며, 후속 마일스톤에서는 `document_register.py`가 인덱스 생성 시점의 corpus hash, embedding 모델, chunk 설정, 생성 시각과 schema version을 sidecar manifest로 저장하도록 개선해야 합니다.

### 실행 권고

1. 위 두 report schema 계약을 계획과 Requirement에 명시적으로 반영합니다.
2. Phase 0의 깨끗한 가상환경 검증을 수행합니다.
3. Phase 1~2를 진행하되 골든셋 두 차례 사용자 승인 게이트를 유지합니다.
4. 최초 baseline에는 인덱스 생성 provenance를 검증할 수 없다는 한계를 함께 기록합니다.

데이터 큐레이션과 승인 시간을 포함한 한 명 기준 약 2~3주 예상은 여전히 타당합니다.

## M2 상세 개발 계획 3차 검토 결과 — 계획에 반영 완료

아래 세 항목은 3차 검토에서 발견됐으며 업데이트된 상세 계획과 Requirement에 모두 반영됐습니다. 현재 조치 대상은 위의 4차 검토 결과를 따릅니다.

현재 실행 가능성 평가: **높은 실행 가능성(9/10)**

아래 세부 정의를 구현 전에 확정하고 Phase 0 검증을 통과한다는 조건으로 실행 가능합니다.

### P1 — 재현성 fingerprint의 평가 명령별 적용 범위 불일치

공식 Requirement의 `M2-REQ-010`은 **각 평가 명령의 리포트**에 corpus manifest와 vectorstore fingerprint를 요구합니다. 반면 상세 계획은 Retrieval 및 통합 baseline에서 fingerprint를 생성·전달한다고 명시하지만, 독립 실행되는 Answer 및 Routing evaluator의 처리 방식은 명확하지 않습니다.

Routing 평가는 corpus와 vectorstore를 사용하지 않으므로 이를 무조건 요구하면 독립적인 routing 평가가 불필요하게 로컬 데이터에 의존합니다. 반대로 Answer 평가는 실제 RAG 결과를 측정하므로 두 fingerprint가 반드시 필요합니다.

해결 방향:

- Retrieval, Answer, 통합 baseline 리포트에는 corpus manifest와 vectorstore fingerprint를 필수로 기록합니다.
- Routing 리포트에는 동일한 schema 필드를 `null` 또는 `not_applicable`로 기록하고 사유를 명시하며, `data/`와 `vectorstore/`의 존재를 실행 조건으로 만들지 않습니다.
- 독립 실행되는 `evaluation.answers`도 공통 fingerprint 수집 함수를 호출하도록 상세 계획에 명시합니다.
- fingerprint 필드의 required/nullable 규칙과 누락 시 종료 정책을 report schema 테스트에 추가합니다.

관련 위치:

- `Development_M2_Quality_Baseline_Requirement.md:293-321`
- `Development_M2_Quality_Baseline_Development_Plan.md` Phase 4, Phase 6, Phase 7

### P2 — Answer evaluator의 반환 source 자료형 변환 미정의

상세 계획의 `_source_match()`는 `returned_sources: list[str]`를 입력으로 받고 각 원소를 `normalize_source_id()`에 전달합니다. 그러나 현재 `RAGEngine.query()`의 `sources`는 source 이름뿐 아니라 page, content 등의 필드를 포함한 객체 목록이며, 계획의 mock 예시도 객체 목록을 사용합니다.

호출부에서 객체를 source 문자열로 변환하지 않고 그대로 전달하면 정규화 과정에서 타입 오류가 나거나 잘못된 비교가 수행될 수 있습니다.

해결 방향:

- `result["sources"]`에서 각 객체의 `source` 필드를 추출하는 공통 helper를 정의합니다.
- 문자열 형식의 legacy/mock 입력을 허용할지 명시하고, 허용하지 않는다면 validator에서 즉시 명확한 오류를 반환합니다.
- `source`가 없거나 문자열이 아닌 항목의 처리 정책을 정하고 실패·제외 건수에 반영합니다.
- 실제 `RAGEngine.query()` 응답과 같은 객체 목록을 사용하는 단위 테스트를 추가합니다.

관련 위치:

- `Development_M2_Quality_Baseline_Development_Plan.md` Phase 6 `_source_match()` 및 Answer evaluator mock
- `rag_engine.py`의 `RAGEngine.query()` 반환 schema

### P2 — abstention 정확도의 집계 정의 미완성

두 공식 abstention 문구를 탐지하는 규칙은 추가됐지만, 요구사항의 "abstention 정확도"를 어떤 사례 집합과 공식으로 집계하는지는 아직 명시되지 않았습니다. 개별 사례의 `abstention 일치 bool`만으로는 정상 답변을 잘못 거절한 false positive와 답변 불가 사례에서 거절하지 않은 false negative를 리포트에서 구분하기 어렵습니다.

해결 방향:

- `expected_abstention = case.expect_abstention`, `predicted_abstention = _detect_abstention(answer)`로 정의합니다.
- Answer 평가 대상 전체에서 TP, TN, FP, FN과 `accuracy = (TP + TN) / N`을 집계합니다.
- 분모가 0인 경우의 값을 `null`로 처리하고 제외 사유와 건수를 기록합니다.
- 최소한 true positive, true negative, false positive, false negative 사례를 각각 테스트합니다.

관련 위치:

- `Development_M2_Quality_Baseline_Requirement.md:249-257`
- `Development_M2_Quality_Baseline_Development_Plan.md` Phase 6

### 실행 권고

위 세 항목은 아키텍처를 다시 설계해야 하는 문제는 아니며, 구현 전 계약과 테스트를 보완하면 됩니다. 보완 후 Phase 0부터 계획대로 실행할 수 있습니다. 특히 깨끗한 가상환경에서 Python 의존성 설치, `pip check`, 핵심 모듈 import, `pytest -q`, `npm test`를 먼저 통과시켜야 하며 실패 시 M2 기능 구현보다 환경 문제를 우선 해결해야 합니다.

데이터 큐레이션과 사용자 승인 시간을 포함한 예상 기간은 상세 계획대로 한 명 기준 약 2~3주가 타당합니다.

## M2 상세 개발 계획 2차 검토 결과 — 계획에 반영 완료

아래 다섯 항목은 2차 검토에서 발견됐으며 업데이트된 상세 계획과 Requirement에 모두 반영됐습니다. 현재 조치 대상은 위의 3차 검토 결과를 따릅니다.

현재 실행 가능성 평가: **높은 실행 가능성(8.5/10)**

### P1 — 상세 계획과 공식 요구사항 원본 불일치

상세 계획은 Answer 평가 eligibility를 category가 아니라 필드 존재 여부로 올바르게 수정했습니다. `answer_assertions`가 존재하거나 `expect_abstention=true`이면 `unanswerable`을 포함해 Answer 평가 대상이 됩니다.

하지만 공식 요구사항의 `M2-REQ-008`은 여전히 Answer 평가를 문서 QA 사례로 제한합니다. 상세 계획은 요구사항 문서를 수용 기준의 원천으로 선언하므로 구현자와 리뷰어가 서로 다른 기준을 적용할 수 있습니다.

상세 계획에 추가된 corpus/vectorstore fingerprint도 `M2-REQ-010`의 필수 메타데이터에 아직 반영되지 않았습니다.

해결 방향:

- `M2-REQ-008`에서 category 제한을 제거하고 필드 기반 eligibility를 명시합니다.
- `M2-REQ-010`에 corpus manifest와 vectorstore fingerprint를 필수화합니다.
- `M2-REQ-011`에 중복 source 처리 규칙을 추가합니다.
- M2 완료 수용 기준에 fingerprint 포함 여부를 추가합니다.

관련 위치:

- `Development_M2_Quality_Baseline_Requirement.md:245-277`
- `Development_M2_Quality_Baseline_Requirement.md:293-324`
- `Development_M2_Quality_Baseline_Development_Plan.md:181-189`
- `Development_M2_Quality_Baseline_Development_Plan.md:350-379`

### P1 — Retrieval 지표 사이의 source 순위 단위 불일치

현재 검색 결과는 chunk 목록이고 골든 정답은 source 파일 단위입니다. 같은 PDF의 여러 chunk가 검색 결과에 반복될 수 있습니다.

상세 계획의 Recall은 top-k chunk를 먼저 자른 뒤 집합으로 변환하지만 nDCG는 중복 source를 먼저 제거한 뒤 고유 source 기준으로 순위를 다시 계산합니다. 따라서 같은 검색 결과에 대해 지표마다 `k`의 의미가 달라집니다.

예를 들어 `A.pdf, A.pdf, A.pdf, B.pdf`가 반환되면 Recall@3은 B를 찾지 못한 것으로 처리하지만 nDCG는 B를 2위로 처리합니다.

해결 방향:

- 평가 단위를 source 또는 chunk 중 하나로 명시적으로 확정합니다.
- 현재 골든셋이 source 파일 단위이므로, 반환된 source ID를 최초 등장 순서로 중복 제거한 공통 목록을 만듭니다.
- Recall, MRR, nDCG가 모두 같은 정규화 목록을 사용하도록 합니다.
- 중복 source가 top-k 경계에 걸리는 사례를 단위 테스트에 추가합니다.

관련 위치:

- `Development_M2_Quality_Baseline_Development_Plan.md:302-348`

### P2 — abstention 판정 문구가 하나로 고정됨

상세 계획은 `제공된 문서에서 관련 정보를 찾을 수 없습니다`만 abstention으로 인식합니다. 하지만 yes/no 프롬프트는 `제공된 문서만으로는 확실한 답변이 어렵습니다`라는 다른 공식 문구를 사용합니다.

Intent Classifier가 yes/no 템플릿을 선택하면 올바르게 답변을 거절해도 자동 평가에서 실패할 수 있습니다.

해결 방향:

- 모든 공식 abstention 문구를 중앙 목록으로 관리하거나
- 골든 사례별로 `accepted_abstention_phrases`를 지원합니다.
- 각 Intent 템플릿의 공식 문구를 포함한 테스트를 추가합니다.

관련 위치:

- `Development_M2_Quality_Baseline_Development_Plan.md:549-568`
- `prompt_templates.py:22`
- `prompt_templates.py:129`

### P2 — Answer source 일치율 공식 미정의

상세 계획에는 `source 일치 ratio`가 있지만 expected source recall, returned source precision, F1 또는 하나 이상의 hit 중 무엇을 뜻하는지 정의하지 않았습니다. `relevant_sources`가 없는 abstention 사례의 0 denominator 처리도 정해지지 않았습니다.

해결 방향:

- 최소한 다음 지표를 명시합니다.
  - `source_any_hit`
  - `source_recall = |unique(returned) ∩ relevant| / |relevant|`
  - `source_evaluation_excluded`
- `relevant_sources`가 없는 사례는 source 평가에서 제외하고 제외 수를 보고합니다.
- source ID는 Retrieval 지표와 동일한 정규화 및 중복 제거 함수를 사용합니다.

관련 위치:

- `Development_M2_Quality_Baseline_Development_Plan.md:570-594`
- `Development_M2_Quality_Baseline_Requirement.md:249-257`

### P2 — 고정 Markdown fence가 모델 답변으로 깨질 수 있음

worksheet를 Markdown 표 대신 사례별 section으로 변경한 것은 적절하지만, 답변을 고정된 triple-backtick fence로 감싸면 답변 자체에 triple backtick 코드 블록이 포함될 때 외부 worksheet 구조가 깨질 수 있습니다.

해결 방향:

- 답변에 포함된 가장 긴 backtick 연속 길이보다 하나 긴 동적 fence를 생성하거나
- HTML escape 후 `<pre><code>`로 렌더링하거나
- 원문 답변은 JSON/CSV에 저장하고 Markdown에는 요약만 표시합니다.
- triple backtick뿐 아니라 더 긴 fence가 포함된 입력도 테스트합니다.

관련 위치:

- `Development_M2_Quality_Baseline_Development_Plan.md:570-594`

### 실행 권고

구현 전에 다음 순서로 문서를 보완합니다.

1. 상세 계획의 변경 사항을 공식 Requirement에 동기화
2. 모든 Retrieval 지표의 source deduplication 규칙 통일
3. abstention 허용 문구 확정
4. Answer source match 공식 확정
5. worksheet의 동적 fence 또는 안전한 대체 형식 확정

위 다섯 항목을 반영한 후 Phase 0부터 실행할 수 있습니다. Phase 0의 깨끗한 가상환경 검증이 실패하면 M2 구현보다 의존성 문제를 먼저 해결해야 합니다. 데이터 큐레이션과 사용자 승인 시간을 포함한 예상 기간은 한 명 기준 약 2~3주입니다.

## M2 상세 개발 계획 1차 검토 결과 — 계획에 반영 완료

[Development_M2_Quality_Baseline_Development_Plan.md](Development_M2_Quality_Baseline_Development_Plan.md)의 초안에서 발견했던 항목입니다. 업데이트된 상세 계획에 모두 반영됐으며 아래 내용은 1차 검토 이력으로만 보존합니다. 현재 조치 대상은 위의 2차 검토 결과를 따릅니다.

현재 실행 가능성 평가: **조건부 실행 가능(7/10)**

### P1 — 평가 category와 evaluator eligibility 충돌

상세 계획은 모든 `document_qa` 사례에 `answer_assertions` 또는 `expect_abstention=true`를 강제하지만, 요구사항은 문서 QA 40개 중 Answer 평가 사례를 최소 20개만 요구합니다.

또한 Answer evaluator가 `category == document_qa`만 처리하도록 설계되어 있어 `unanswerable` category에 배치한 abstention 사례가 실제 Answer 평가에서 제외됩니다.

영향:

- Retrieval 전용 문서 QA 사례까지 불필요한 Answer 정답 작성이 강제됩니다.
- 최소 5개로 계획한 답변 불가 사례가 abstention 정확도에 포함되지 않을 수 있습니다.
- 골든셋 validator와 evaluator가 서로 다른 사례 집합을 해석할 수 있습니다.

해결 방향:

- category와 평가 대상을 분리합니다.
- Routing은 모든 사례를 평가합니다.
- Retrieval은 `relevant_sources`가 존재하는 사례를 평가합니다.
- Answer는 category와 무관하게 `answer_assertions`가 존재하거나 `expect_abstention=true`인 사례를 평가합니다.
- 모든 `document_qa`에 Answer 조건을 강제하지 않고 전체 구성에서 최소 20개인지 검사합니다.

관련 위치:

- `Development_M2_Quality_Baseline_Development_Plan.md:157-166`
- `Development_M2_Quality_Baseline_Development_Plan.md:438-453`
- `Development_M2_Quality_Baseline_Requirement.md:93-109`

### P1 — baseline에 corpus와 vectorstore fingerprint 누락

상세 계획의 리포트 메타데이터는 골든셋 SHA-256, Git revision, 모델과 검색 설정을 기록하지만 실제 평가 대상인 `data/`와 `vectorstore/`의 상태를 식별하지 않습니다.

두 디렉터리는 Git에서 제외되므로 문서나 인덱스가 변경되어도 같은 평가 환경으로 기록될 수 있습니다.

영향:

- 서로 다른 corpus에서 생성된 baseline을 잘못 비교할 수 있습니다.
- 최초 baseline을 동일 조건으로 재현하기 어렵습니다.
- 인덱스가 현재 문서 및 설정으로 생성됐는지 확인할 수 없습니다.

해결 방향:

- 정규화된 source ID, 파일 크기, SHA-256을 포함한 corpus manifest를 생성합니다.
- corpus manifest 전체 SHA-256을 리포트에 기록합니다.
- `index.faiss`와 `index.pkl`의 SHA-256을 기록합니다.
- 인덱스 문서·청크 수와 생성 당시 embedding 모델, chunk size/overlap을 기록합니다.
- Requirement의 `M2-REQ-010`에도 이 메타데이터를 필수 항목으로 추가합니다.

관련 위치:

- `Development_M2_Quality_Baseline_Development_Plan.md:262-272`
- `Development_M2_Quality_Baseline_Requirement.md:293-324`

### P1 — Retrieval 계측을 비활성화할 수 없음

상세 계획은 기존 `_retrieve_documents()`가 항상 `_retrieve_documents_traced()`를 호출하도록 설계합니다. 일반 Web/API 요청에서도 단계별 타이머 측정과 trace 객체 생성이 수행되며 실제로 계측을 끄는 경로가 없습니다.

이는 계측 비활성 상태에서 기존 호출자와 동작을 보존하라는 `M2-REQ-006`의 의도와 맞지 않습니다.

해결 방향:

- 선택적 observer/callback 또는 `trace=None` 인자를 사용합니다.
- 일반 production 요청에서는 계측 객체를 생성하지 않습니다.
- Hybrid, MMR-only, reranker-only, similarity 분기를 모두 포함한 결정론적 characterization test를 필수로 추가합니다.
- 계측 활성·비활성 시 최종 문서 순서가 동일함을 검증합니다.

관련 위치:

- `Development_M2_Quality_Baseline_Development_Plan.md:205-245`
- `Development_M2_Quality_Baseline_Requirement.md:198-215`

### P1 — Phase 0이 애플리케이션 실행 가능성을 확인하지 못함

상세 계획은 기존 테스트 실패가 없다고 기록하지만 현재 환경에서 FastAPI import가 실패합니다. 설치된 `email-validator==1.3.1`과 현재 Pydantic/FastAPI 조합이 호환되지 않으며, `pip check`에서도 Torch, LangChain, protobuf 등 여러 환경 충돌이 확인됩니다.

기존 `pytest`는 `web_server.py`를 import하지 않기 때문에 이 문제를 발견하지 못합니다.

해결 방향:

- Phase 0에 다음 검증을 추가합니다.

```bash
python -c "import web_server"
python -c "from fastapi.testclient import TestClient"
python -m pip check
```

- 공유 Conda 환경과 분리된 새 Python 3.11 가상환경에서 `requirements.txt` 설치와 smoke test를 수행합니다.
- 프로젝트 의존성에서 재현되는 충돌과 현재 공유 환경에만 존재하는 충돌을 구분해 기록합니다.

관련 위치:

- `Development_M2_Quality_Baseline_Development_Plan.md:11-29`
- `requirements.txt`

### P2 — Pydantic 2 API 버전 전제 미명시

스키마 예시는 `field_validator`와 `model_validator`를 사용하므로 Pydantic 2가 필요합니다. 하지만 `requirements.txt`는 Pydantic 버전을 직접 제한하지 않고 FastAPI의 전이 의존성에 맡깁니다.

해결 방향:

- `pydantic>=2,<3`을 직접 명시하거나
- 평가 schema를 dataclass와 수동 validator로 구현합니다.
- Pydantic을 유지하면 list/dict 기본값은 `Field(default_factory=...)`를 사용합니다.

관련 위치:

- `Development_M2_Quality_Baseline_Development_Plan.md:93-166`
- `requirements.txt`

### P2 — nDCG 공식과 집계 방식 미확정

상세 계획은 rank discount를 정의하지만 graded relevance의 gain 공식과 사례 집계 방식을 완전히 고정하지 않았습니다. 구현자에 따라 서로 다른 baseline 값이 나올 수 있습니다.

해결 방향:

- graded gain을 다음처럼 명시합니다.

```python
gain = 2**grade - 1
dcg += gain / log2(rank + 1)
```

- 중복 source는 최초 순위만 인정합니다.
- 사례별 nDCG를 계산한 뒤 대상 사례의 macro average를 사용합니다.
- `relevance_grades`가 없는 사례는 집계에서 제외하고 제외 수를 보고합니다.

관련 위치:

- `Development_M2_Quality_Baseline_Development_Plan.md:247-260`

### P2 — Markdown worksheet가 LLM 답변으로 깨질 수 있음

상세 계획은 질문과 전체 답변을 Markdown 표의 셀에 넣도록 설계합니다. 답변에 줄바꿈, Markdown 표, `|`, 코드 블록이 포함되면 worksheet 구조가 깨질 수 있습니다.

해결 방향:

- 사례별 Markdown section 형식을 사용하거나
- escaping을 보장하는 CSV writer를 사용합니다.
- Markdown을 사용한다면 답변을 표 셀에 넣지 말고 별도 fenced/section 영역에 배치합니다.

관련 위치:

- `Development_M2_Quality_Baseline_Development_Plan.md:445-451`

### 실행 권고

다음 순서로 상세 계획을 보완한 뒤 구현을 시작합니다.

1. evaluator eligibility 규칙 수정
2. corpus/vectorstore fingerprint 요구사항 추가
3. optional Retrieval tracing 설계로 변경
4. 깨끗한 가상환경과 Web import smoke test 추가
5. Pydantic 버전 전략 확정
6. nDCG 공식과 집계 방식 확정
7. 사람 검토 worksheet 형식 변경

위 항목을 반영하면 상세 계획은 Claude Code가 단계적으로 실행하기에 충분히 구체적입니다. 예상 기간은 데이터 큐레이션과 두 번의 사용자 승인 게이트를 포함해 한 명 기준 약 2~3주가 현실적입니다.

## P1 — 품질 기준선과 CI 부재

현재 Python 테스트 21개와 프런트엔드 테스트 9개가 있지만 대부분 mock 또는 DOM 단위 테스트입니다. 실제 문서에 대한 검색 적중률, 답변 충실성, 전체 지연 시간을 정량적으로 비교하는 골든 평가셋과 자동 CI가 없습니다.

영향:

- 검색 파라미터나 모델 변경이 실제 개선인지 판단하기 어렵습니다.
- 기본 테스트 실행에서는 실제 Ollama 라우팅 테스트가 제외됩니다.
- 저장소에 CI 구성이 없어 테스트 실행이 개발자 절차에 의존합니다.

해결 방향:

- 대표 질문, 관련 문서, 기대 답변 특성을 포함한 평가셋 구축
- Recall@K, MRR, nDCG와 답변 충실성·관련성 측정
- Python 및 프런트엔드 테스트를 실행하는 CI 추가
- 라이브 Ollama 테스트는 수동 또는 별도 환경의 주기 작업으로 분리

관련 마일스톤: `M2 — Quality Baseline & CI`

## P1 — Intent Classifier의 도메인 불일치

현재 분류기는 9,000개 학습 데이터와 1,000개 검증 데이터로 재학습됐지만, 데이터의 중심 도메인이 IT 헬프데스크와 HR이며 실제 RAG/AI 문서 질문과 다릅니다. `uncertain` 라벨은 모델 출력에는 존재하지만 현재 데이터에는 학습 예시가 없습니다.

영향:

- 도메인 밖 질문에서 낮은 신뢰도 또는 잘못된 답변 형식이 선택될 수 있습니다.
- Agent와 Intent Classifier가 각각 LLM 라우팅과 답변 형식 결정을 수행해 시스템 복잡도가 증가합니다.

해결 방향:

- 실제 서비스 질문으로 의도 분류 효용과 오류율 측정
- 유지할 경우 RAG/AI 도메인 데이터와 `uncertain` 예시 추가 및 confidence calibration
- 효과가 작으면 규칙 또는 LLM 기반 형식 선택으로 단순화

관련 마일스톤: `M3 — Retrieval & Domain Quality`

## P1 — 운영 관측성과 동시 요청 대응 부족

현재 주요 단계가 `print()` 로그와 동기 호출로 구현되어 있습니다. FastAPI의 `async` endpoint 안에서 Ollama, 웹 검색, 임베딩, reranker 작업을 동기 실행합니다.

영향:

- 단계별 지연 시간과 실패율을 집계하기 어렵습니다.
- 긴 요청이 이벤트 루프를 점유해 동시 사용자 처리량이 낮아질 수 있습니다.
- 프로세스별로 대형 모델과 인덱스를 로드하므로 단순 worker 확장이 비쌉니다.

해결 방향:

- 구조화된 로그와 단계별 latency/error 메트릭 도입
- blocking 작업의 thread offload 또는 비동기 API 검토
- 부하 테스트를 통해 단일 프로세스 처리량과 메모리 기준 확립
- readiness와 liveness 상태 분리

관련 마일스톤: `M4 — Production Readiness`

## P2 — 한국어 BM25 토크나이징 한계

문서와 질문을 단순 공백 기준으로 나누기 때문에 조사, 어미, 복합명사를 충분히 처리하지 못합니다.

영향:

- 한국어 키워드 검색의 recall과 순위 품질이 낮아질 수 있습니다.

해결 방향:

- 평가 기준선을 먼저 만든 후 형태소 분석기 또는 한국어용 tokenizer 후보 비교
- 품질, 인덱싱 시간, 배포 복잡성을 함께 측정해 선택

관련 마일스톤: `M3 — Retrieval & Domain Quality`

## P2 — MMR 문서 임베딩 재계산

MMR 단계가 매 질문마다 후보 문서 본문을 다시 임베딩합니다. FAISS 생성 시 계산한 벡터를 재사용하지 않습니다.

영향:

- 후보 문서 수가 늘수록 응답 지연과 CPU 사용량이 증가합니다.

해결 방향:

- FAISS 인덱스 또는 별도 캐시에서 후보 문서 임베딩 재사용
- 변경 전후 latency와 선택 결과를 평가셋으로 비교

관련 마일스톤: `M3 — Retrieval & Domain Quality`

## P2 — 웹 검색 답변 품질 편차

현재 웹 검색은 DuckDuckGo의 제목과 요약을 최대 3개 나열합니다. 검색 결과의 신뢰도와 관련성에 편차가 있으며, 여러 출처를 종합한 답변 생성이나 문장 단위 근거 검증은 수행하지 않습니다.

영향:

- 최신 정보 질문에 스팸성 또는 부정확한 결과가 노출될 수 있습니다.
- 사용자가 검색 결과를 직접 비교해야 합니다.

해결 방향:

- 결과 중복 제거, 도메인 신뢰 정책, 관련성 필터 도입
- 웹 콘텐츠를 LLM에 전달할 경우 prompt injection 방어와 인용 검증을 함께 설계
- 원문 수집·요약 전에 개인정보, 저작권, robots 정책 검토

관련 마일스톤: `M3 — Retrieval & Domain Quality`

## P2 — 문서 인덱스의 파괴적 전체 재생성

`document_register.py`는 새 인덱스를 만들기 전에 기존 `vectorstore/`를 삭제합니다. 증분 인덱싱, 인덱스 버전, 원자적 교체 또는 자동 복구 기능이 없습니다.

영향:

- 생성 중 실패하면 기존 정상 인덱스를 잃을 수 있습니다.
- 문서 일부 변경에도 전체 임베딩을 다시 계산해야 합니다.

해결 방향:

- 임시 경로에서 새 인덱스를 완성한 뒤 원자적으로 교체
- 인덱스 메타데이터에 문서·모델·청크 설정 버전 기록
- 이후 필요하면 문서 해시 기반 증분 인덱싱 도입

관련 마일스톤: `M4 — Production Readiness`

## P2 — 설정과 Python 의존성의 재현성 부족

애플리케이션 설정은 `config.py` 상수에 고정되어 있고 Python 의존성은 대부분 하한 또는 범위만 지정합니다. 환경별 모델, 경로, 웹 검색 정책을 코드 수정 없이 바꾸기 어렵고 시간이 지나면 설치 결과가 달라질 수 있습니다.

해결 방향:

- 환경변수 기반 typed settings와 예제 환경 파일 제공
- 재현 가능한 Python lock 파일 또는 constraints 정책 도입
- 모델 이름, 학습 데이터, 인덱스 설정 버전을 함께 기록

관련 마일스톤: `M4 — Production Readiness`

## P2 — 신뢰되지 않은 FAISS 인덱스 로딩 위험

FAISS 로딩에 `allow_dangerous_deserialization=True`가 사용됩니다. 현재처럼 직접 생성한 로컬 인덱스만 사용하면 허용 가능한 선택이지만, 외부에서 받은 인덱스를 로드하면 임의 코드 실행 위험이 있습니다.

해결 방향:

- 외부 인덱스를 절대 직접 로드하지 않는 운영 정책 명시
- 인덱스 artifact의 출처와 checksum 검증
- 가능한 경우 pickle 기반 docstore를 피하는 저장 형식 검토

관련 마일스톤: `M4 — Production Readiness`

## P3 — 대규모 문서 집합의 확장성

현재 FAISS `IndexFlatIP`와 전체 메모리 로딩은 소규모 문서 집합에 적합합니다. 실제 문서 수와 latency가 임계치를 넘기 전에는 변경하지 않습니다.

검토 조건:

- 평가·부하 테스트에서 합의한 latency 또는 메모리 기준 초과
- 운영 문서 증가율로 현재 구조의 한계가 예측됨

후보 방향:

- IVF/HNSW 등 ANN 인덱스
- 양자화 또는 외부 벡터 데이터베이스
- 캐시와 인덱스 분할

관련 마일스톤: `M5 — Scale & Advanced Capabilities`
