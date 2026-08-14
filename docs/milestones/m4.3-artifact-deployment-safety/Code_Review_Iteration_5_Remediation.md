# M4.3 Artifact & Deployment Safety — Code Review Iteration 5 Remediation

작성자: Claude Code Sonnet 5 (remediation worker)
대상: PR #18, hosted run 31615907683 시점 리뷰
기준 revision: `3041979` (`agent/m4-3-artifact-deployment-safety`, PR #18
HEAD) — 이 문서가 기술하는 변경은 같은 브랜치에 새 커밋으로 추가되며 PR을
새로 열지 않는다. **merge는 수행하지 않는다** — fresh Codex 리뷰가 필요하다.
대상 리뷰: [Code_Review_Iteration_5.md](Code_Review_Iteration_5.md)
(판정 **FAIL — 9.0/10**, `CRITICAL 0 / MAJOR 2 / MINOR 0 / TRIVIAL 0`)

## 0. 범위

이 문서는 Code Review Iteration 5가 지적한 2개 finding
(`CR-I5-MAJ-01`, `CR-I5-MAJ-02`)을 finding별로 코드 변경/신규 테스트/실제
실행 결과에 1:1로 매핑한다. 이전 세션이 작성한
[Implementation_Report.md](Implementation_Report.md)의 §1~§12는
remediation 이전 시점의 기록으로 그대로 보존했고, 이 remediation의 요약은
그 문서의 §13에 추가했다. [Traceability.md](Traceability.md)의
M4.3-REQ-005/NFR-003 행에 이 remediation을 반영했다.

이 remediation이 건드리지 않은 것: OCI 레이어 해석
(`_resolve_trusted_link_content`/`_update_merged_state`/whiteout 처리)
로직 자체, `M4.1_BLOCKED=true`, protected M3 live `NOT_RUN`,
`overall_release_ready=false` 산출 경로, `m3-live-regression-gate` 블록,
Native Linux/Ollama/DDGS, self-hosted runner/environment 승인 경계. 두
finding은 모두 `scripts/scan_image_layers.py`의 certifi 코멘트 grammar와
`src/simple_qna_rag/rag_engine.py`/`src/simple_qna_rag/web/server.py`의
엔진 실패 상태 배선에만 있었다.

## 1. CR-I5-MAJ-01 — certifi 코멘트가 임의 텍스트를 밀반입할 수 있음

### 원본 결함

Iteration 3이 추가한 `_CERTIFI_COMMENT_FIELD`/`_COMMENT_LINE`는
`(?:_COMMENT_LINE)*`(0개 이상, 순서 무관, 중복 허용)와
`[^\r\n]{0,512}`(개행만 아니면 어떤 바이트든 512자까지 허용) 조합이었다.
결과적으로 실제 강제되는 것은 "일곱 개 접두사 중 하나로 시작하는 줄이
0개 이상"뿐이었고, 정확히 7개/정확한 순서/중복 없음/필드별 문법은 전혀
강제되지 않았다. 리뷰는 다음 5줄(정상 인증서 뒤에 붙임)이 전부 `True`를
반환함을 재현했다:

```text
# Issuer: API_TOKEN=supersecret
# Label: ../../etc/shadow
# Serial: arbitrary free text
# Issuer: x
# Issuer: y
```

### 수정

`scripts/scan_image_layers.py`의 grammar를 반복 가능한 `(?:...)*` 대신
**정확히 한 번씩, 정확한 순서로** 일곱 필드 이름을 리터럴로 나열하는 단일
고정 시퀀스(`_CERTIFI_STANZA`)로 재작성했다:

```text
# Issuer: <값>\n
# Subject: <값>\n
# Label: <값>\n
# Serial: <값>\n
# MD5 Fingerprint: <값>\n
# SHA1 Fingerprint: <값>\n
# SHA256 Fingerprint: <값>\n
```

각 필드 이름이 시퀀스 안에 정확히 한 번만 리터럴로 등장하므로, 누락된
필드/중복된 필드/순서가 바뀐 필드/추가된 8번째 필드는 모두 파서가 다음에
기대하는 리터럴 필드명(또는 `-----BEGIN CERTIFICATE-----`)과 실제 다음
줄이 일치하지 않아 구조적으로 거부된다 — "0개 이상 허용"이 아니라
"정확히 이 순서의 7개, 그 이상도 이하도 아님"이 된 것이 핵심 수정이다.

각 필드는 실제 설치된 certifi 패키지 전체(top-level + pip-vendored, 두
독립 버전)에서 실제 관측된 값의 문자 알파벳/최대 길이로부터 유도한
필드별 bounded grammar를 추가로 받는다:

| 필드 | grammar | 근거 |
|---|---|---|
| `Serial` | `[0-9]{1,48}` (10진수만) | 실측 최대 48자리, RFC 5280 시리얼은 최대 20 옥텟(~47자리) |
| `MD5/SHA1/SHA256 Fingerprint` | 콜론 구분 소문자 hex, 알고리즘별 정확한 바이트 길이(16/20/32바이트) | 실측 길이 47/59/95바이트 정확히 고정 |
| `Issuer`/`Subject` | `[A-Za-z0-9 ()/,._=\-]{1,256}`(밑줄 포함) | RFC 4514류 DN 렌더링에서 실측된 전체 알파벳(§1.1 참조) |
| `Label` | `"[A-Za-z0-9 ().\-]{0,126}"`(따옴표로 감싼 문자열, 콤마/슬래시/등호/밑줄 제외) | 실측 alphabet, 최대 61바이트 |

콜론, `@`, `$`, 따옴표(Label 바깥), 중괄호, 제어 문자 등 key=value/
token/JSON/secret에 전형적인 문자는 모든 필드의 알파벳 바깥에 있으므로
여전히 매치를 깨뜨린다 — 리뷰의 정확한 재현 입력(`API_TOKEN=supersecret`
등)에 대해 `is_verified_ca_bundle()`이 `False`를 반환함을 직접
확인했다(§4 재검증 결과 참조). 전체 X.509 검증(`ssl.SSLContext.
load_verify_locations`)과 전체-입력 `fullmatch`(부분 소비 불가) 계약은
그대로 유지된다.

### §1.1 — Issuer/Subject 알파벳에 밑줄을 포함시킨 이유(실제 이미지 재현)

최초 구현은 밑줄을 제외했다(단일 certifi 버전의 관측 문자 집합 기준).
그러나 이 remediation의 §3 재검증 과정에서 실제 `docker buildx build
--platform linux/amd64 --target production`으로 만든 이미지에
`scan_image_layers.py`를 실행하자 `forbidden_count: 1`이
나왔다(`usr/local/lib/python3.11/site-packages/pip/_vendor/certifi/
cacert.pem`) — **실제 정상 파일에 대한 새 오탐**이었다. 원인을 `docker
cp`로 이미지 밖으로 꺼내 직접 조사한 결과, base 이미지에 번들된 pip의
벤더 certifi 사본(로컬 개발 venv의 pip보다 오래된 버전)의 "Entrust.net
Certification Authority (2048)" 항목이 `OU=www.entrust.net/CPS_2048
incorp. by ref. ...`처럼 **밑줄을 포함한 정당한 DN**을 렌더링했다.
top-level `certifi/cacert.pem`(밑줄 없음)과 pip-vendored 사본(밑줄
있음) 두 파일의 전체 문자 알파벳을 다시 비교해 이 차이가 유일한
차이임을 확인한 뒤, `_ISSUER_SUBJECT_VALUE`에만 밑줄을 추가했다(Label/
Serial/Fingerprint 문법은 무변경). `Label`은 밑줄을 계속 제외한다(실측
alphabet에 없음). 재빌드한 이미지로 재확인한 결과 `forbidden_count: 0`
으로 복귀했다(§4).

밑줄을 허용해도 리뷰가 지적한 `API_TOKEN=supersecret` 같은 스머글링이
다시 열리지 않는다 — 그 예시가 거부되는 것은 애초에 밑줄이 아니라
**구조(정확히 7필드/정확한 순서)와 Serial/Fingerprint 필드의 엄격한
숫자/hex 문법** 때문이다. 밑줄 하나를 추가로 허용하는 것은 콜론/`@`/
`$`/따옴표/중괄호/제어문자를 여전히 배제하는 좁은 alphabet 안에서의
조정일 뿐이며, 공격자가 값을 삽입하려면 여전히 7개 필드 전부가 정확한
위치에서 각자의 문법(특히 3개의 정확한 바이트 길이 hex fingerprint와
숫자 전용 Serial)을 통과해야 한다.

### 신규 테스트 (`tests/unit/test_scan_image_layers.py`, 51→71)

| 테스트 | 오라클 |
|---|---|
| `test_is_verified_ca_bundle_accepts_full_installed_certifi_bundle` | 실제 설치된 `certifi` 패키지(`requirements.lock`의 locked 의존성) 전체를 `certifi.where()`로 읽어 그대로 검증 — 145개 이상 블록 전체가 fullmatch되는 **실제 upstream positive oracle** |
| `test_is_verified_ca_bundle_accepts_real_pip_vendored_entry_with_underscore_in_dn` | §1.1에서 실제 이미지 조사로 발견한 정확한 real 항목(pip-vendored Entrust.net 2048, `docker cp`로 추출)을 그대로 verify — 밑줄 포함 정당 DN 회귀 오라클 |
| `test_is_verified_ca_bundle_rejects_stanza_missing_a_field`(parametrize×7) | 7개 필드 각각을 하나씩 제거 → 전부 거부 |
| `test_is_verified_ca_bundle_rejects_duplicate_field` | `# Issuer:`를 중복 삽입 → 거부 |
| `test_is_verified_ca_bundle_rejects_reordered_fields` | `Label`/`Serial` 순서 교환 → 거부 |
| `test_is_verified_ca_bundle_rejects_extra_field` | 인식된 접두사를 가진 8번째 줄 삽입 → 거부 |
| `test_is_verified_ca_bundle_rejects_token_path_key_value_material_per_field`(parametrize×7) | 7개 필드 전부에 대해 token/path/key-value/private-material 페이로드(콜론 기반 `token: secret`, `../../etc/shadow`, `arbitrary free text`, JSON, 등) → 전부 거부 |
| `test_is_verified_ca_bundle_rejects_reordered_recognized_prefix_example` | 리뷰 원문의 정확한 재현 입력(밑줄 없이 콜론/자유 텍스트만 사용)을 그대로 검증 → 거부 |

## 2. CR-I5-MAJ-02 — 실패한 엔진 상태와 오래된 artifact reason이 재시도에서 살아남음

### 원본 결함

`RAGEngine.__new__`는 생성된 모든 인스턴스를 `cls._instance`에
저장하는데, `get_rag_engine()`은 실패 시 모듈 전역 `_rag_engine`만
`None`으로 유지하고 클래스 레벨 싱글톤은 그대로 뒀다. `RAGEngine()`을
다시 호출하면 `__new__`가 이 실패한 객체를 그대로 반환하고,
`RAGEngine._initialized`가 이미 `True`이므로 `__init__`도 재실행되지
않는다(`_artifact_error_reason`을 포함한 모든 필드가 이전 실패 값
그대로). `initialize()`도 시도 시작 시점에 `_artifact_error_reason`을
리셋하지 않았다. 결과적으로 "artifact 실패 → 재시도 → 무관한 일반
실패" 순서에서 두 번째(일반) 실패가 첫 번째의 오래된 artifact reason으로
잘못 보고될 수 있었다.

별도로, `EngineArtifactError.__init__`은 어떤 문자열이든 그대로
저장했고, `server.py::_make_lifespan`은 `engine_factory` 호출을 감싼
`except Exception as exc:`에서 `getattr(exc, "reason", None)`로 무조건
artifact reason을 읽었다 — `EngineArtifactError`가 아닌, 우연히
`.reason` 속성을 가진 어떤 예외라도 artifact 실패로 재분류되어 공개
`/health/ready` 응답 필드(`artifact_{reason}`)에 그 값이 노출될 수
있었다.

### 수정

`src/simple_qna_rag/rag_engine.py`:

1. **`EngineArtifactError.__init__`**이 `reason`을
   `index_verification.REASONS`(기존 `IndexTrustError.reason`이 이미
   지켜야 했던 공개 대상 allowlist, Design.md §5.2)의 멤버인지 검증하고
   아니면 `ValueError`를 던지도록 했다 — 이 예외 타입 자체가 허용목록
   밖의 reason으로는 아예 생성될 수 없다.
2. **`RAGEngine.initialize()`**가 시도 시작 시점에
   `self._artifact_error_reason = None`을 무조건 리셋한다 — 같은
   객체에 대해 `initialize()`가 직접 재호출되는 미래의 어떤 경로에서도
   이전 시도의 reason이 살아남지 않는다.
3. **`get_rag_engine()`**이 `initialize()` 실패 시 `RAGEngine._instance
   = None`과 `RAGEngine._initialized = False`를 모두 리셋한다 — 다음
   `RAGEngine()` 호출은 완전히 새 객체를 생성하고 `__init__`도
   새로 실행되므로(신선한 identity), 실패한 객체가 재사용될 수 없다.
   또한 reason이 `index_verification.REASONS`의 멤버일 때만
   `EngineArtifactError`를 던지고, 그 외(허용목록 밖의 값)는 일반
   `RuntimeError("RAG 엔진 초기화 실패")`로 강등한다 — 허용목록 밖의
   reason은 절대 공개 artifact reason으로 노출되지 않는다.

`src/simple_qna_rag/web/server.py`:

4. `_make_lifespan()`의 `engine_factory(candidate)` 호출을 감싼
   예외 처리를 `except EngineArtifactError as exc:`(disclosed reason,
   `exc.reason`을 그대로 사용)와 `except Exception as exc:`(일반
   실패, reason 미설정) 두 개로 분리했다 — `.reason` 속성의 존재
   여부가 아니라 **타입**으로 분류하므로, 무관한 예외가 우연히
   `.reason`을 가져도 더 이상 재분류되지 않는다.

### 신규 테스트

`tests/unit/test_rag_engine_singleton.py`(신규, 9개):

| 테스트 | 오라클 |
|---|---|
| `test_artifact_failure_discards_both_singleton_layers` | artifact 실패 후 `RAGEngine._instance`/`_initialized`/모듈 `_rag_engine` 모두 리셋됨을 직접 확인 |
| `test_artifact_then_retry_ordinary_failure_yields_ordinary_engine_init_failed` | 핵심 회귀 오라클 — artifact 실패 → 재시도 일반 실패가 평범한 `RuntimeError`("RAG 엔진 초기화 실패")이지 `EngineArtifactError`가 아님을 확인 |
| `test_artifact_then_retry_success_uses_fresh_identity` | `id(self)`를 캡처해 재시도 성공이 **다른 객체**임을 직접 증명(신선한 identity) |
| `test_ordinary_failure_then_retry_success_is_unaffected` | 일반 실패→성공 재시도 계약이 이 수정으로 바뀌지 않았음을 확인 |
| `test_path_like_reason_never_becomes_artifact_error` | reason이 `"../../etc/passwd"`(path-like)면 일반 실패로 강등 |
| `test_arbitrary_unallowlisted_reason_never_becomes_artifact_error` | 임의의 허용목록 밖 문자열도 동일하게 강등 |
| `test_engine_artifact_error_rejects_reason_outside_allowlist` | `EngineArtifactError` 생성자 자체가 path-like/`=`포함/빈 문자열/오탈자 reason을 `ValueError`로 거부 |
| `test_engine_artifact_error_accepts_every_allowlisted_reason` | `index_verification.REASONS`의 모든 멤버가 정상 생성됨을 확인 |
| `test_initialize_resets_stale_artifact_reason_before_each_attempt` | `get_rag_engine()`을 거치지 않고 `initialize()`를 직접 두 번째 호출해도(같은 객체) 이전 reason이 리셋됨을 방어적으로 확인 |

`tests/integration/test_health_endpoints.py`(8→10):

| 테스트 | 오라클 |
|---|---|
| `test_health_ready_engine_artifact_error_discloses_allowlisted_reason` | `EngineArtifactError`를 던지면 `/health/ready`가 `artifact_{reason}`을 정확히 보고 |
| `test_health_ready_arbitrary_reason_bearing_exception_is_not_reclassified` | `.reason`을 가진 무관한 예외(`EngineArtifactError`가 아님)를 던지면 `engine_init_failed`로 남고 그 reason이 노출되지 않음을 확인 — lifespan이 타입으로 분류함을 end-to-end로 증명 |

### 회귀 오라클 검증 (fix 적용 전 코드로 되돌려 확인)

두 finding의 신규 테스트가 실제로 유의미한지 확인하기 위해, `git
stash`로 `rag_engine.py`/`server.py`의 수정만 되돌린 뒤 같은 테스트를
재실행했다:

```
git stash push -- src/simple_qna_rag/rag_engine.py src/simple_qna_rag/web/server.py
venv/bin/python -m pytest -q tests/unit/test_rag_engine_singleton.py tests/integration/test_health_endpoints.py
# 9 failed, 10 passed — 실패 목록: 8개 신규 singleton 테스트 +
# test_health_ready_arbitrary_reason_bearing_exception_is_not_reclassified
git stash pop
```

원본 코드에서 정확히 이 회귀 테스트들만 실패하고 나머지는 그대로
통과함을 확인한 뒤 fix를 복원했다.

## 3. 실제 linux/amd64 이미지 대상 재검증

`docker buildx build --platform linux/amd64 --target production -f
deploy/Dockerfile --load .`로 만든 실제 이미지(대부분 캐시 hit, 신규
레이어는 `src/`/COPY만)에:

- `scan_image_layers.py --image <이미지>` → **`forbidden_count: 0`**
  (§1.1의 밑줄 회귀를 잡아 고친 뒤의 최종 결과 — 고치기 전 1회
  `forbidden_count: 1`을 실제로 관측했다)
- `container_smoke.py --image <이미지>` → **`status: PASS`**, 모든
  필드 `true`(`host_gateway_reachable`, `mock_query_ok`,
  `root_page_ok`, `static_asset_ok`, `production_test_seam_sealed`) —
  `production_test_seam_sealed`는 CR-I5-MAJ-02가 고친
  `EngineArtifactError`/negative-control 배선을 실제 컨테이너로
  end-to-end 실행한다.

두 실행 모두 완료 후 이미지를 `docker rmi`로 정리했다.

## 4. 재검증 결과

| 명령 | 결과 |
|---|---|
| `pytest -q tests/unit/test_scan_image_layers.py` | 71 passed(51→71, +20) |
| `pytest -q tests/unit/test_rag_engine_singleton.py`(신규) | 9 passed |
| `pytest -q tests/integration/test_health_endpoints.py` | 10 passed(8→10, +2) |
| `pytest -q`(전체 unit+integration) | **1251 passed, 1 skipped**(1220→1251, +31) |
| `docker buildx build --platform linux/amd64 --target production` 실제 이미지에 `scan_image_layers.py` | **`forbidden_count: 0`**(§3) |
| 같은 이미지에 `container_smoke.py` | **`status: PASS`**, 전 필드 true(§3) |
| linux/amd64 `python:3.11-slim`, `uv==0.8.15`, `compile_lock.sh --verify` | **PASS**, 103 packages, drift 없음 |
| `run_m43_acceptance.py --profile deterministic --repeat 10 --seed 4303` | status=PASS, 전체 node `success_count=10/10` |
| `run_m43_acceptance.py --profile deterministic --repeat 3 --seed 4303 --inject-evidence-mismatch`(negative control) | `negative_control.result=REJECTED_AS_EXPECTED` |
| `check_markdown_links.py` | 파일 121개, 링크 558개, 실패 0 |
| `generate_field_spec.py --check`, `logging_callsite_audit.py --check` | 둘 다 exit 0 |
| `git diff --exit-code -- evaluation/baselines/m3_initial.*` | exit 0 |
| `npm test` | 9 passed |
| `npm run sync-vendor` 이후 `git diff --exit-code -- web/static/vendor/` | exit 0 |
| `git diff --check` | 실패 0 |
| adversarial certifi probes (`Code_Review_Iteration_5.md` 원문 재현) | **거부 확인**(reproduced bypass now returns `False`) |

Native Linux/Ollama/DDGS, protected M3 live, M4.1 live, self-hosted
runner/environment 승인 경계는 이 remediation에서도 실행·변경하지
않았다. `M4.1_BLOCKED=true`, protected M3 live `NOT_RUN`,
`overall_release_ready=false` 산출 경로는 변경된 파일에 포함되지
않는다.

## 5. 변경 파일

- `scripts/scan_image_layers.py` — §1 (exact-stanza certifi grammar,
  필드별 bounded grammar, 밑줄 alphabet 조정)
- `tests/unit/test_scan_image_layers.py` — §1 신규/강화 테스트(51→71)
- `src/simple_qna_rag/rag_engine.py` — §2 (`EngineArtifactError`
  allowlist, `initialize()` reason 리셋, `get_rag_engine()` 싱글톤
  discard)
- `src/simple_qna_rag/web/server.py` — §2 (`EngineArtifactError` 타입
  기반 분류)
- `tests/unit/test_rag_engine_singleton.py`(신규) — §2 singleton
  identity/retry/allowlist 테스트 9개
- `tests/integration/test_health_endpoints.py` — §2 lifespan
  분류 e2e 테스트 2개(8→10)
- `docs/milestones/m4.3-artifact-deployment-safety/Code_Review_Iteration_5.md`
  — closure note 추가(§6)
- `docs/milestones/m4.3-artifact-deployment-safety/Implementation_Report.md`,
  `Traceability.md` — 이 remediation 요약 반영

## 6. 다음 단계

이 커밋은 기존 PR #18에 push된다. **merge하지 않는다** — fresh Codex
리뷰가 필요하며, 이 문서와 `Implementation_Report.md`/`Traceability.md`의
갱신된 절을 참조 자료로 남긴다. hosted CI가 다시 통과하는지는 push
이후 별도로 확인해야 한다.
