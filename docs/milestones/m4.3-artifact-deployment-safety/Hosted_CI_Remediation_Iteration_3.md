# M4.3 Artifact & Deployment Safety — Hosted CI Remediation Iteration 3

작성자: Claude Code Sonnet 5 (hosted-CI remediation worker)
대상: PR #18, hosted run
[31609022196](https://github.com/luminous419/simple-qna-rag/actions/runs/31609022196)
기준 revision: `1e7fbac`(`agent/m4-3-artifact-deployment-safety`, PR #18
HEAD) — 이 문서가 기술하는 변경은 같은 브랜치에 새 커밋으로 추가되며 PR을
새로 열지 않는다. **merge는 수행하지 않는다** — fresh Codex 리뷰가 필요하다.

## 0. 범위

[Code_Review_Iteration_4.md](Code_Review_Iteration_4.md)(판정 PASS
9.7/10, `MINOR 2`)가 지적한 `CR-I4-MIN-01`(약한 hardlink e2e oracle)/
`CR-I4-MIN-02`(문서 whitespace 결함)와, 그 리뷰 시점에는 아직 도달하지
못했던 hosted run 31609022196의 `container` job 실제 실패
(`scan_image_layers.py`가 `forbidden_count=153`으로 재발)를 고쳤다.
스캐너 수정 이후 실제 이미지로 `container_smoke.py`를 처음부터 끝까지
로컬 실행해 검증하는 과정에서, 이 remediation의 원래 범위 밖이지만
동일한 hosted `container` job을 막는 3개의 추가 결함(설정 바인딩
불일치, FAISS embeddings 배선, 테스트 픽스처 docstore 타입, 실패한
엔진의 artifact reason 유실)을 발견해 함께 고쳤다 — 상세는 §3 참조.

이 remediation이 건드리지 않은 것: `M4.1_BLOCKED=true`, protected M3
live `NOT_RUN`, `overall_release_ready=false` 산출 경로,
`m3-live-regression-gate` 블록, Native Linux/Ollama/DDGS, self-hosted
runner/environment 승인 경계, 위 결함들과 무관한 어떤 파일도 수정하지
않았다.

## 1. 진단 — hosted run 31609022196의 실제 위반 내용

Iteration 2가 도입한 "symlink/hardlink는 경로만으로 절대 허용목록에
들지 않는다"는 보수적 fail-closed 정책(CR-I3-MAJ-02)은 코드 리뷰
관점에서는 안전했지만, 실제 Debian 베이스 이미지의 CA 신뢰 저장소가
거의 전부 symlink로 구성돼 있다는 사실과 충돌했다. `docker run`으로
빌드된 `production` target 이미지의 실제 바이트를 직접 조사해 다음을
확인했다:

- `/etc/ssl/certs/*.pem` 300개 중 299개가 symlink다 — 2단계 체인
  (`002c0b4f.0 -> GlobalSign_Root_R46.pem`(같은 디렉터리, 상대
  경로) `-> /usr/share/ca-certificates/mozilla/GlobalSign_Root_R46.crt`
  (절대 경로, 진짜 regular 파일))이 표준 `openssl rehash` 레이아웃이다.
- `/usr/lib/ssl/cert.pem`은 `/etc/ssl/certs/ca-certificates.crt`(진짜
  regular 파일)를 가리키는 단일 symlink다.
- `usr/local/lib/python3.11/site-packages/certifi/cacert.pem`과
  `pip/_vendor/certifi/cacert.pem`은 REGULAR 파일이지만, 실제 certifi
  업스트림 포맷은 각 `BEGIN CERTIFICATE` 블록 앞에 `# Issuer:`/
  `# Subject:`/`# Label:`/`# Serial:`/`# MD5 Fingerprint:`/
  `# SHA1 Fingerprint:`/`# SHA256 Fingerprint:` 7줄의 고정 코멘트가
  붙는다(설치된 `certifi==2026.7.22` 패키지로 직접 확인, 145개 블록
  전부 정확히 7줄) — Iteration 2의 `_STRICT_PEM_BUNDLE_RE`는 코멘트를
  전혀 허용하지 않아 이 실제 파일을 거부했다.

hosted 로그의 153개 violation 전부가 위 두 클래스(신뢰 경로의 진짜
symlink 300여 개 중 credential로 잡힌 것들, certifi 코멘트 포맷)에
해당했다 — 시크릿이나 실제 정책 위반은 하나도 없었다.

## 2. 수정 — `scripts/scan_image_layers.py`

### 2.1 OCI layer-state-aware bounded link resolution (symlink/hardlink)

`classify_member()`에 `is_link`/`link_target_verified` 파라미터를
추가했다 — 이 함수 자신은 여전히 링크를 절대 직접 resolve하지 않고,
호출자(`scan()`)가 이미 독립적으로 검증한 결과만 받는다(신뢰 경로
+콘텐츠 검증의 기존 이중 게이트 원칙 유지). `scan()`은 이제 레이어를
순서대로 처리하며 whiteout-aware OCI union 파일시스템 상태
(`_MergedEntry`/`_update_merged_state`)를 누적 구축한다 — opaque
whiteout(`.wh..wh..opq`)은 그 시점까지 상속된 상태에서 해당 디렉터리
하위를 전부 마스킹하고, exact whiteout(`.wh.<name>`)은 해당 경로만
제거하며, 같은 레이어의 실제 write는 같은 레이어의 whiteout보다 항상
우선한다.

신뢰 경로의 symlink/hardlink는 `_resolve_trusted_link_content()`가
resolve한다 — `_MAX_LINK_HOPS=40`으로 bounded, visited-path set으로
cycle 차단, 매 hop마다 정규화한 target이 이미지 루트를 벗어나면
(`..`로 시작) 즉시 거부, 최종 도달한 멤버가 merged state에 없으면
(dangling, whiteout으로 마스킹된 경우 포함) 거부한다. 체인의 끝이
**genuine regular 멤버**이고 그 멤버 **자신의 정규화 경로도 신뢰
경로**일 때만(허용목록을 신뢰 경로 밖으로 넓히지 않기 위한 defense in
depth) 그 바이트를 읽어 기존 `is_verified_ca_bundle()`로 독립
검증한다. hardlink의 linkname은 심볼릭 링크와 다르게 아카이브 내
멤버의 정확한 경로를 가리키므로 별도 정규화 경로로 처리했다. 대상이
현재 레이어에 없으면 이전 레이어(들)까지 outer tar를 재오픈해 조회한다
— "OCI layer-state-aware"의 실질적 의미로, 서로 다른 레이어에 나뉘어
쓰인 symlink와 그 target도 resolve된다.

이 변경은 이전 remediation의 "모든 non-regular 멤버를 credential로
분류"라는 보수적 정책을 유지하지 않지만, fail-closed 성질은 정확히
동일한 수준으로 보존한다 — dangling/cycle/traversal/절대 경로 탈출/
non-regular 최종 멤버/신뢰 경로 밖 최종 멤버/whiteout 마스킹된 경로는
전부 여전히 즉시 거부되고, 오직 "신뢰 경로에 있고, 신뢰 경로의 진짜
regular CA 콘텐츠에 bounded/cycle-safe하게 도달하는" 링크만 예외를
받는다.

### 2.2 certifi 코멘트 포맷을 인식하는 좁은 문법

`is_verified_ca_bundle()`의 grammar에 `_CERTIFI_COMMENT_FIELD`를
추가했다 — 정확히 7개의 알려진 필드 접두사(`# Issuer:`, `# Subject:`,
`# Label:`, `# Serial:`, `# MD5 Fingerprint:`, `# SHA1 Fingerprint:`,
`# SHA256 Fingerprint:`)만 인식하고, 각 값은 개행 없이 512바이트로
제한된다(실측 최장 코멘트 줄 159바이트 대비 넉넉한 여유). 이 코멘트는
`BEGIN CERTIFICATE` 블록 바로 앞에서만 허용되며, 인식되지 않는 코멘트
(`# note: ...`, `# SECRET=...` 등)나 블록 뒤에 붙은 코멘트는 여전히
전체 `fullmatch`를 깨뜨려 거부된다 — CR-I3-MAJ-01이 닫은 "시크릿을
인증서 뒤에 붙이면 통과" 취약점을 코멘트라는 새 경로로 재도입하지
않는다. 실제 설치된 `certifi==2026.7.22`의 `cacert.pem` 전체(145개
블록)가 이 grammar로 fullmatch됨을 별도 스크립트로 직접 검증했다.

## 3. 원래 범위 밖에서 발견/수정한 3개 결함 — 실제 이미지로 `container_smoke.py` e2e 검증 중 발견

`run_m43_acceptance.py`의 `container_static_and_connectivity` node는
`container_smoke.py`의 argv 계약만 단위 테스트하고 실제 docker
컨테이너를 띄우지 않으며, 지난 세 번의 hosted run은 모두 `container_smoke.py`가
실행되기 전 단계(lock drift, 스캐너 오탐, 이번 오탐)에서 먼저 실패해
이 코드 경로에 도달한 적이 없었다. §2 수정 이후 실제 `production`
이미지로 `container_smoke.py`를 처음 끝까지 로컬 실행했더니
`status=FAIL`이 나왔다 — 이 remediation의 원래 트리거(스캐너)와는
무관하지만, 고치지 않으면 스캐너를 고쳐도 hosted `container` job이
다음 스텝에서 계속 실패하므로 범위에 포함해 진단·수정했다.

### 3.1 `_settings_binding_snapshot()`이 deterministic_test 모드에서도 실제 `EMBEDDING_MODEL_NAME`을 보고

`EMBEDDING_PROVIDER="deterministic_test"`일 때 `_build_embeddings()`는
`EMBEDDING_MODEL_NAME`을 전혀 읽지 않는데도(§ 5.2-a, 테스트 시임을
직접 사용), `_settings_binding_snapshot()`은 항상 실제
`EMBEDDING_MODEL_NAME`(예: `BAAI/bge-m3`)을 반환했다.
`container_smoke.py`의 fixture builder는 매니페스트에
`"deterministic-test-fixture"`라는 고정 문자열을 새겨 넣으므로, 로드
시점 스냅샷과 항상 불일치해 `IndexTrustError("settings_mismatch")`로
매 컨테이너가 즉시 거부됐다(hosted CI에서 이 경로가 한 번도 끝까지
실행된 적이 없어 발견되지 못한 결함, `5b91840`부터 존재).
`rag_engine.py`에 `DETERMINISTIC_TEST_EMBEDDING_MODEL_NAME` 상수를
단일 진실 공급원으로 추가하고 `_settings_binding_snapshot()`과
`container_smoke.py`의 fixture builder 양쪽이 이를 import해서
쓰도록 해, 두 값이 다시 독립적으로 drift할 수 없게 만들었다.

### 3.2 trust-verified FAISS 재구성이 `embeddings.embed_query`(raw callable)를 넘겨 `.embeddings`가 `None`이 됨

`index/verification.py::_construct_faiss_from_verified_bytes()`가
`FAISS(embeddings.embed_query, ...)`를 호출했다 — LangChain FAISS는
`embedding_function`이 실제 `Embeddings` 인스턴스일 때만
`.embeddings` 프로퍼티를 채우고, bound method 같은 raw callable을
받으면 `.embeddings`가 `None`이 된다(레거시 호환 코드 경로, deprecation
경고 동반). `vector_index.py::StoredVectorIndex.build()`가
`vectorstore.embeddings.embed_query(...)`로 MMR canary 검증을
수행하므로 `AttributeError: 'NoneType' object has no attribute
'embed_query'`로 엔진 초기화가 매번 실패했다(`MMR_VECTOR_SOURCE`
기본값이 `"stored"`이므로 이 경로가 항상 실행됨). `embeddings.embed_query`
대신 `embeddings` 객체 자체를 넘기도록 한 줄을 고쳤다 — 레거시 경로
(`_load_vectorstore_legacy`가 `FAISS.load_local(path, embeddings,
...)`로 이미 올바르게 호출하던 패턴)와 동일하게 맞춘 것뿐이다.

### 3.3 `DeterministicTestEmbeddings`가 duck-typed일 뿐 실제 `Embeddings` 서브클래스가 아니었음

3.2를 고친 뒤에도 `.embeddings`가 여전히 `None`이었다 — 테스트 시임의
`DeterministicTestEmbeddings`가 `embed_documents`/`embed_query`만
구현하고 `langchain_core.embeddings.Embeddings`를 상속하지 않아서,
FAISS의 `isinstance(embedding_function, Embeddings)` 체크가 항상
`False`였다(실제 `HuggingFaceEmbeddings`는 이 ABC를 상속하므로 프로덕션
경로에서는 이 결함이 절대 나타나지 않는다 — deterministic_test 전용
결함). `tests/support/simple_qna_rag_test_seam/deterministic_embeddings.py`가
`Embeddings`를 상속하도록 한 줄 고쳤다 — 이 파일은 `src/`가 아니라
production 이미지에 절대 COPY되지 않는 테스트 인프라이므로 이 변경은
production 코드 경로에 어떤 영향도 주지 않는다.

### 3.4 `container_smoke.py` fixture의 docstore가 `Document` 대신 raw string을 저장

3.2/3.3을 고친 뒤 `StoredVectorIndex.build()`가 새로운 지점에서
실패했다 — `document.page_content`인데 `document`가 `str`이었다.
`container_smoke.py::_build_fixture_index()`가
`InMemoryDocstore({str(i): t for i, t in enumerate(texts)})`로 raw
텍스트 문자열을 직접 저장했지만, 실제 프로덕션 인덱스 빌드 경로
(`cli/index_lifecycle.py`/`cli/index_documents.py`)는 항상
text-splitter가 만든 `Document` 객체를 저장한다. 픽스처가
`Document(page_content=t)`로 감싸도록 고쳤다 — 실제 프로덕션 데이터
형태와 픽스처를 일치시키는 수정이다.

### 3.5 실패한 엔진의 artifact reason이 `app.state.engine=None`으로 유실되어 negative-control이 항상 일반 `engine_init_failed`로만 보고됨

`get_rag_engine()`은 `RAGEngine.initialize()`가 `False`를 반환하면
(내부에서 `IndexTrustError`/`TestEmbeddingSeamUnavailable`을 잡아
`self._artifact_error_reason`에 저장한 뒤) 그 정보를 전혀 담지 않은
평범한 `RuntimeError("RAG 엔진 초기화 실패")`만 던졌다. `server.py`의
`_make_lifespan()`은 이 예외를 잡아 `app.state.engine = None`으로
설정한 뒤 `getattr(app.state.engine, "_artifact_error_reason", None)`을
읽었는데 — `app.state.engine`이 이미 `None`이므로 이 값은 구조적으로
항상 `None`이었다. 결과적으로 `production_test_seam_sealed`가
검증하는 negative-control(테스트 시임 마운트 없이 실행 -> 503
`artifact_test_embedding_seam_unavailable` 기대)이 실제로는 503은
맞지만 reason이 항상 일반 `engine_init_failed`로만 나와 절대
`True`가 될 수 없었다 — 이 M4.1/M4.3 test-seam sealing 계약이
end-to-end로는 한 번도 실제로 검증된 적이 없었던 dead-wiring
결함이었다.

`rag_engine.py`에 `EngineArtifactError(RuntimeError)`(`.reason` 보유)를
추가하고, `get_rag_engine()`이 `_artifact_error_reason`이 설정된
경우 이 예외로 던지도록 고쳤다(또한 실패한 인스턴스를 모듈 전역에
캐시하지 않도록 해 다음 호출이 재시도 없이 조용히 깨진 엔진을
반환하던 부수 결함도 같이 닫았다). `server.py`의 `_make_lifespan()`은
이제 `app.state.engine`이 아니라 **캐치한 예외 자체**의
`getattr(exc, "reason", None)`에서 artifact reason을 읽는다 — 어떤
`engine_factory`든 `.reason`을 가진 예외를 던지면 이 배선이 동작한다.

## 4. 재검증 결과

| Check | 결과 |
|---|---|
| `pytest -q tests/unit/test_scan_image_layers.py`(39→51) | **51 passed** |
| `pytest -q`(전체 unit+integration, macOS 로컬) | **1220 passed, 1 skipped**(1206 → 1220, §2/§3의 신규 테스트 14개) |
| linux/amd64 `docker build --target production`으로 만든 실제 이미지에 `scan_image_layers.py` 실행 | **`forbidden_count: 0`**(hosted run 31609022196의 153에서) |
| 같은 실제 이미지에 `container_smoke.py` 처음부터 끝까지 로컬 실행 | **`status: PASS`**(모든 필드 true — §3 수정 전에는 `status: FAIL`) |
| linux/amd64 `python:3.11-slim`, `uv==0.8.15`, `compile_lock.sh --verify` | **PASS**, 103 packages, drift 없음(lock 파일 변경 없음) |
| `run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303` | **status=PASS**, 17개 node 전부 `success_count=10/10` |
| `run_m43_acceptance.py --profile deterministic --repeat 3 --seed 4303 --inject-evidence-mismatch`(negative control) | `negative_control.result=REJECTED_AS_EXPECTED` |
| `check_markdown_links.py` | 검사 파일 119개, 링크 547개, 실패 0 |
| `generate_field_spec.py --check`, `logging_callsite_audit.py --check` | 둘 다 exit 0 |
| `git diff --exit-code -- evaluation/baselines/m3_initial.*` | exit 0 |
| `npm test` | 9 passed |
| `npm run sync-vendor` 이후 `git diff --exit-code -- web/static/vendor/` | exit 0 |
| `git diff --check`(이 remediation의 전체 변경, `Code_Review_Iteration_3.md` 포함) | 실패 0(CR-I4-MIN-02 종료) |

CR-I4-MIN-01은 `test_scan_flags_hardlink_bypass_end_to_end`가 이제
`etc/ssl/certs/innocent.pem`(하드링크 자체)과
`app/secrets/key.pem`(대상) 두 멤버의 정확한 violation record를
개별 assert하도록 강화해 닫았다 — 하드링크 자신이 실수로 예외 처리돼도
더 이상 무관한 `app/secrets/key.pem`만으로 테스트가 통과할 수 없다.

Native Linux/Ollama/DDGS, protected M3 live, M4.1 live, self-hosted
runner/environment 승인 경계는 이 remediation에서도 실행·변경하지
않았다. `M4.1_BLOCKED=true`, protected M3 live `NOT_RUN`,
`overall_release_ready=false` 산출 경로는 변경된 파일에 포함되지
않는다.

## 5. 변경 파일

- `scripts/scan_image_layers.py` — §2 (링크 resolution, certifi grammar)
- `tests/unit/test_scan_image_layers.py` — §2 신규/강화 테스트
- `src/simple_qna_rag/rag_engine.py` — §3.1, §3.5
- `src/simple_qna_rag/index/verification.py` — §3.2
- `tests/support/simple_qna_rag_test_seam/deterministic_embeddings.py` — §3.3
- `scripts/container_smoke.py` — §3.1, §3.4
- `src/simple_qna_rag/web/server.py` — §3.5
- `tests/unit/test_container_smoke_bare_script.py`(신규) — bare-script `ModuleNotFoundError` 회귀를 환경 독립적인 실제 subprocess로 재현(Code_Review_Iteration_4.md가 지적한 pytest-import 환경 의존성 caveat 해소)
- `docs/milestones/m4.3-artifact-deployment-safety/Code_Review_Iteration_3.md` — CR-I4-MIN-02(trailing whitespace, EOF 여분 빈 줄) 제거

## 6. 다음 단계

이 커밋은 기존 PR #18에 push된다. **merge하지 않는다** — fresh Codex
리뷰가 필요하며, 이 문서와 `Implementation_Report.md`/`Traceability.md`의
갱신된 절을 참조 자료로 남긴다. hosted CI가 다시 통과하는지는 push
이후 별도로 확인해야 한다.
