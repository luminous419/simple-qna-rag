# M4.3 Artifact & Deployment Safety — Hosted CI Remediation Iteration 6

작성자: Claude Sonnet 5 (hosted CI remediation worker)
대상: PR #18, 진단용 hosted run
[31804369490](https://github.com/luminous419/simple-qna-rag/actions/runs/31804369490)
(기준 commit `2418119`, Iteration 5 diagnostic push — `container` FAILURE,
`python-tests` FAILURE, `frontend-tests` SUCCESS,
`m3-live-regression-gate` SKIPPED, `m43-deterministic`/`m4-assemble`은
상류 실패에 종속). 기준 revision: `2418119`
(`agent/m4-3-artifact-deployment-safety`, PR #18 HEAD) — 이 문서가 기술하는
변경은 같은 브랜치에 새 커밋으로 추가될 예정이며 PR을 새로 열지 않는다.
**merge/commit/push는 수행하지 않는다** — fresh Codex 리뷰가 필요하다.

## 0. 범위와 결론 요약

이 iteration은 서로 독립적인 두 개의 ordinary hosted 블로커를 다룬다.

- **블로커 A (`container` job)**: `/health/ready`가 60초 폴링 예산 내내
  결정론적으로 `503 engine_init_failed`를 반환한다(가설 A/B 중 가설
  B — Hosted_Container_Remediation_Iteration_5.md가 이미 확정한 대로
  순수 타이밍 문제가 아니다). 코드 추적으로 실제 결함을 찾았다:
  `index/verification.py`의 contained-open 경로(`open_contained_root`/
  `ContainedDir.open_subdir`/`open_member`)가 `EACCES`(permission denied)를
  전혀 처리하지 않아 raw `OSError`로 전파시키고, `RAGEngine.initialize()`의
  범용 `except Exception: return False`가 이를 통째로 삼켜
  **진단 불가능한 일반 `engine_init_failed`**로 뭉갠다 — 실제 원인이
  `IndexTrustError`/`TestEmbeddingSeamUnavailable`이었다면 받았을
  `artifact_*` 처리를 전혀 받지 못한다. 이 gap을 닫고(§1), 그와 별개로
  `lifecycle.py::_publish()`가 `versions/<id>/` 자신은 0o555로 명시적으로
  chmod하면서 그 부모 `versions/` 디렉터리는 한 번도 명시적으로 chmod하지
  않고 순수히 caller의 ambient umask에 맡겨온 것을 발견해 같은 방식으로
  0o755를 명시했다(§2) — 빌드 호스트(hosted runner)와 나중에 읽는 UID(컨테이너
  10001)가 다를 때 umask가 022보다 엄격하면 정확히 이 경로에서 `EACCES`가
  난다. 안전한 예외-타입 관측성(§3)과 hosted 로그 tail이 60초짜리
  health-check 스팸에 밀려 정작 `startup` 이벤트 자체를 놓치는 진짜 결함도
  함께 닫았다(§4, 이 iteration 자체의 실제 hosted job 로그로 재확인).
- **블로커 B (`python-tests` job)**: `compile_lock.sh --verify`가
  Iteration 5가 lock/`compile_lock.sh`를 전혀 건드리지 않았음에도 실패했다.
  linux/amd64 컨테이너(hosted와 동일 아키텍처)로 재현한 결과, 원인은
  `requirements.txt`의 변경이 아니라 **unpinned transitive
  dependency**(`langsmith`)가 committed lock의 `0.10.18` 이후 상류에
  새 버전(`0.11.0`)을 발행했고, `compile_lock.sh`가 매번 빈 `mktemp`
  파일로 컴파일해 uv에게 "이미 잠긴 버전"이라는 선호 정보를 전혀 주지
  않았기 때문임을 확인했다(§5). Task 지시대로 `langsmith`만 손으로
  pin하거나 lock을 최신으로 재생성하는 "latest 따라가기"가 아니라,
  **모든** unpinned transitive 패키지에 구조적으로 적용되는 수정을
  적용했다: `compile_once()`가 uv를 부르기 *전에* 현재 committed
  `requirements.lock`으로 `-o` 대상을 미리 채운다 — `uv pip compile`은
  이미 존재하는 output 파일의 버전을 "선호"로 취급해 `requirements.txt`가
  실제로 바뀐 패키지만 재해석하고 나머지는 그대로 고정한다(§6, uv 0.8.15
  `--help`의 `-U/--upgrade` 설명이 이 기본 동작의 근거).

이 iteration이 건드리지 않은 것(§8): `_poll_ready`의 `max_seconds`(60초
그대로), `evaluate_readiness()`의 분기 순서/우선순위, negative control의
판정 임계값(`expect_status=503`/
`expect_reason="artifact_test_embedding_seam_unavailable"`),
`compute_all_ok`/`_ALL_OK_KEYS`, `M4.1_BLOCKED=true`, protected M3 live
`NOT_RUN`, `overall_release_ready=false` 산출 경로,
`m3-live-regression-gate` 블록, `.github/workflows/ci.yml`,
`scripts/scan_image_layers.py`/`assemble_m4_evidence.py`/
`check_m4_baseline.py`, `requirements.lock`/`requirements.txt`의 실제
내용(§6의 수정은 오직 lock **생성 절차**만 바꾼다 — 로컬에서 재생성해도
바이트 단위로 committed lock과 동일함을 §7에서 확인했다), CPU torch
extra-index 시맨틱(`--extra-index-url`/`--index-strategy
unsafe-best-match`), `--generate-hashes` 계약, uv 0.8.15 고정, Native
Linux/Ollama/DDGS/live/self-hosted 승인 경계, 어떤 approval도.

## 1. 블로커 A 근본 원인 #1 — EACCES가 disclosed 사유 없이 삼켜짐

### 1.1 코드 추적

`RAGEngine.initialize()`(`rag_engine.py:248`)의 예외 처리:

```python
except IndexTrustError as exc:
    self._artifact_error_reason = exc.reason
    return False
except TestEmbeddingSeamUnavailable as exc:
    self._artifact_error_reason = exc.reason
    return False
except Exception:
    # M4.1 REPLACE — 예외 원문/traceback은 로그에 남기지 않는다.
    return False
```

`_load_vectorstore()` → `index_verification.load_verified_faiss()` →
`verify_version()` → `ContainedDir.open_subdir("versions")` →
`.open_subdir(version_id)` → `.open_member(...)`가 이 경로의 유일한
파일시스템 접근이다. 수정 전 `index/verification.py`:

```python
def open_subdir(self, name: str) -> "ContainedDir":
    ...
    try:
        fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise TrustBoundaryError("version_dir_symlink") from None
        if exc.errno == errno.ENOENT:
            raise TrustBoundaryError("version_dir_missing") from None
        if exc.errno == errno.ENOTDIR:
            raise TrustBoundaryError("version_dir_not_directory") from None
        raise                                   # <-- EACCES는 여기로 떨어진다
    return ContainedDir(fd)
```

`open_member`/`open_contained_root`도 동일하게 ELOOP/ENOENT(+ENOTDIR)만
`TrustBoundaryError`로 번역하고 그 외(EACCES 포함)는 raw `OSError`로
재전파한다. `_load_vectorstore()`는 오직
`except index_verification.TrustBoundaryError`만 잡으므로, EACCES가
나면 raw `PermissionError`가 그대로 `initialize()`의 범용
`except Exception: return False`까지 새어나가 `_artifact_error_reason`을
전혀 세팅하지 못한 채 실패한다. 그 결과 `evaluate_readiness()`는
`engine_error is not None`만 보고 무조건 `engine_init_failed`를
반환한다(`observability/health.py:29`) — 실제 원인이 무엇이든 동일한
문자열이 되어 hosted 로그에서 구분이 불가능하다.

**hosted 실패와의 부합**: bind-mount 권한 처리는 native Linux 커널과
macOS Docker Desktop의 gRPC-FUSE/VirtioFS 공유 계층이 서로 다르다는 것이
Hosted_Container_Remediation_Iteration_5.md §2.3이 이미 남긴 미확정
가설이었다. `EACCES`는 정확히 이 계층 차이가 발현될 자리이며, 발생하면
빠르고(`os.open` 실패는 즉시 반환) 결정론적(같은 UID/같은 마운트 조건이면
매번 동일)이다 — run 31804369490의 실제 로그(§4에서 재확인)가 보여준
"컨테이너는 살아있고, `/health/ready`가 60초 내내 서브밀리초 지연으로
매번 동일한 사유의 503을 반환"하는 지문과 정확히 일치한다.

### 1.2 수정 (`src/simple_qna_rag/index/verification.py`)

세 진입점 모두에 `EACCES` 분기를 추가하고, 새 reason 3개를 공개
allowlist(`REASONS`)에 추가했다:

```python
if exc.errno == errno.EACCES:
    raise TrustBoundaryError("root_permission_denied") from None       # open_contained_root
    raise TrustBoundaryError("version_dir_permission_denied") from None # ContainedDir.open_subdir
    raise TrustBoundaryError("member_permission_denied") from None      # ContainedDir.open_member
```

`REASONS`에 신규 추가하지 않으면 이 수정은 실제로는 무의미하다 —
`get_rag_engine()`(`rag_engine.py:847`)이 `EngineArtifactError`로
감싸기 전에 `reason in index_verification.REASONS`를 직접 확인하고,
`EngineArtifactError.__init__` 자신도 동일 allowlist를 재확인하기
때문이다(CR-I5-MAJ-02). allowlist에 없으면 그대로 `RuntimeError("RAG
엔진 초기화 실패")`로 강등되어 다시 `engine_init_failed`가 된다 — 이
디자인은 건드리지 않았고(§8), 새 reason 3개를 그 allowlist에 정확히
추가하는 것으로 체인 전체가 disclosed 경로를 taste한다.

판정 로직/우선순위/negative control은 전혀 바뀌지 않는다 — 이 수정은
오직 "지금까지 raw OSError였던 것"을 "이미 존재하는 `artifact_{reason}`
공개 채널"로 라우팅할 뿐이다.

## 2. 블로커 A 근본 원인 #2 — `versions/` 부모 디렉터리가 ambient umask에 방치됨

`index/lifecycle.py::_publish()`는 매 publish마다 `dest`(`versions/<id>/`)
자신을 `0o555`로, 그 안의 세 파일을 `0o444`로 명시적으로 chmod한다. 하지만
`versions/`(그 부모)는 `_assert_same_filesystem()`의
`versions_dir.mkdir(parents=True, exist_ok=True)`로만 생성되고 **한 번도
명시적으로 chmod된 적이 없다** — 순수히 인덱스를 빌드하는 프로세스의
ambient umask에 의존한다. 빌드 호스트의 umask가 `0o022`보다 엄격하면(예:
`0o027`/`0o077`), `versions/`는 group/other에 traverse(x) 권한이 없는
상태로 생성되고, 나중에 **다른 UID**(예: 컨테이너의 `10001`, 빌드한
호스트 프로세스의 UID와 다름)가 `INDEX_ROOT`를 읽기 전용으로
bind-mount해서 `versions/<id>/manifest.json`을 열려고 하면 정확히
`open_subdir("versions")` 단계에서 `EACCES`가 난다 — §1이 닫은 바로 그
지점이다.

### 2.1 수정 (`src/simple_qna_rag/index/lifecycle.py::_publish`)

```python
os.chmod(dest, 0o555)
os.chmod(index_root / "versions", 0o755)   # 신규 — dest와 동일한 명시적 계약
_fsync_dir(index_root / "versions")
```

`versions/`의 유일한 내용물은 이미 독립적으로 `0o555`/`0o444`로 보호되는
버전 디렉터리들뿐이므로(비밀 정보 없음, 불투명한 16진수 버전 ID의 나열일
뿐), world read+traverse를 명시하는 것은 기존에 이미 확립된 보안 posture
(디렉터리는 world-readable+traversable-only, 쓰기 금지)를 한 단계 위로
그대로 연장할 뿐 — 아무것도 loosening하지 않는다. idempotent이므로 반복
publish에도 안전하다.

`container_smoke.py::_build_fixture_index()`가 fixture 빌드 후 이미
`os.chmod(index_root, 0o755)`로 최상위 디렉터리만 명시적으로 고정해온
것과 정확히 같은 패턴이며, 그 한 단계 빠졌던 `versions/` 자신을 이
iteration이 닫는다.

## 3. 안전한 예외-타입 관측성

M4.1 REPLACE(Design.md §6.1)는 `str(exc)`/traceback을 로그에 남기는
것을 금지한다(경로/URL/시크릿을 담을 수 있음). 그래서
`RAGEngine.initialize()`의 범용 `except Exception:` 분기는 지금까지
예외에 대한 어떤 신호도 남기지 않았다 — hosted에서 다음에 다시 실패해도
`engine_init_failed`라는 문자열 외에는 아무것도 알 수 없었다
(Hosted_Container_Remediation_Iteration_5.md/평가 보고서가 명시적으로
지적한 gap).

`type(exc).__name__`(예: `PermissionError`, `ConnectionError`,
`TimeoutError`)은 호출자가 제어할 수 없는 고정된 Python 식별자이며
경로/URL/시크릿을 담을 수 없다 — `str(exc)`와 근본적으로 다른 안전
등급이다. `web/server.py::_make_lifespan()`의 범용
`except Exception as exc:` 분기에서만 이를 캡처해 컨테이너 자신의
구조화 `startup` 로그에만(공개 HTTP 응답 바디에는 아님 — `/health/ready`는
여전히 `{"status", "reason"}` 두 필드만 반환, `test_health_ready_*`
기존 테스트로 확인됨) 추가했다:

```python
except Exception as exc:  # fail-soft engine diagnostic
    app.state.engine = None
    app.state.engine_error = str(exc)
    app.state.engine_error_type = type(exc).__name__
...
log_event("startup", metrics_registry=registry, reason=reason,
          engine_error_type=app.state.engine_error_type)  # None이면 키 자체를 생략
```

`observability/logging.py`의 payload-safe 스키마(`_EVENT_KEYS`,
`_FIELD_VALIDATORS`)에 `startup` 이벤트의 선택 필드로
`engine_error_type`을 추가하고, `^[A-Za-z_][A-Za-z0-9_.]{0,63}$` 형태만
허용하는 전용 validator로 바인딩했다 — 이 경계에서 벗어난 값(우연히 긴
문자열/URL 등)은 로그 계층 자체가 안전한 기본값으로 클램프한다
(로그가 예외적으로 오염되어도 이 grammar가 두 번째 방어선이 된다).
`EngineArtifactError`/`SettingsError` 분기는 건드리지 않았다 — 이미
disclosed reason이 있으므로 추가 정보가 필요 없다.

## 4. `container_log_tail`이 60초 health-check 스팸에 밀려 startup 이벤트를 놓침

이 iteration을 시작하며 이번 hosted run 31804369490의 실제
`container_smoke.json`/job 콘솔 로그를 다시 읽었다(`gh api
repos/luminous419/simple-qna-rag/actions/jobs/94779612040/logs
--allow-escape-sequences`). Iteration 5가 새로 추가한
`container_log_tail`(마지막 16000바이트)이 실제로 어떤 내용을
포함했는지 확인한 결과 — `/health/ready`에 대한 초당 1회
request_start/request_end 로그 쌍이 정확히 26개 연속으로 나타났을 뿐,
`{"event": "startup", "reason": "engine_init_failed", ...}` 줄은
tail 안에 전혀 없었다. 60초짜리 전체 폴링 예산이 초당 로그 쌍을
만들어내는데, `docker logs --tail 200`의 라인 수 제한보다 먼저
16000바이트 제한이 걸려 가장 최근 ~26개의 반복적인 health-check
줄만 남고 그보다 앞서 한 번 찍힌 `startup` 줄(진단에 결정적인 유일한
줄)은 밀려났다 — Iteration 5가 만든 관측 메커니즘 자체의 진짜 결함이다.

### 4.1 수정 (`scripts/container_smoke.py::_capture_container_logs`)

```python
if len(combined) <= max_bytes:
    return combined
lines = combined.splitlines(keepends=True)
startup_blob = "".join(ln for ln in lines if '"event": "startup"' in ln)
remaining = max_bytes - len(startup_blob)
tail_blob = combined[-remaining:] if remaining > 0 else ""
return startup_blob + tail_blob
```

`"event": "startup"` 줄(들)을 먼저 무조건 보존하고, 남는 바이트
예산만큼만 기존과 동일하게 최신 tail을 채운다. `max_bytes` 상한(16000)과
"성공 경로에서는 아예 채우지 않는다"는 기존 계약은 그대로다 — 실패 시
채워지는 내용의 **선별 방식**만 바뀌었다. §3의 `engine_error_type`이
바로 이 `startup` 줄에 실리므로, 이 수정이 없었다면 §3의 관측성 개선
자체가 다음 hosted 실패에서도 다시 잘려나갈 뻔했다.

## 5. 블로커 B 근본 원인 — `compile_lock.sh`가 매번 빈 파일에서 재해석함

`bash scripts/compile_lock.sh --verify`를 로컬(macOS ARM, 참고용)과
hosted와 동일한 `linux/amd64`(`docker run --platform linux/amd64
python:3.11.15-slim`, `pip install uv==0.8.15`) 양쪽에서 재현했다.
linux/amd64 결과가 hosted job 로그와 정확히 일치한다(둘 다 "Resolved
103 packages"). committed `requirements.lock`과 신선하게 재컴파일한
결과를 패키지 라인 단위로 diff:

```
diff <(grep -E '^[a-zA-Z0-9_.-]+==' requirements.lock | sort) \
     <(grep -E '^[a-zA-Z0-9_.-]+==' requirements.lock.linux | sort)
43c43
< langsmith==0.10.18 \
---
> langsmith==0.11.0 \
```

유일한 차이는 `langsmith` 하나뿐이다. `requirements.txt`는 `langsmith`를
직접 명시하지 않는다 — `langchain`/`langchain-core` 등이 끌어오는
**unpinned transitive dependency**다. committed lock이 마지막으로
생성된 시점 이후 PyPI에 `langsmith 0.11.0`이 새로 발행되었고,
`compile_lock.sh`의 `compile_once()`는 매번 `mktemp`로 만든 빈 파일에
`uv pip compile`을 실행해왔다 — uv에게 "이 패키지는 이미 X로 잠겨있다"는
선호 정보를 줄 방법이 구조적으로 없었으므로, 매 실행마다 index에서
그 순간 "최신"으로 만족되는 버전을 새로 골랐다. `requirements.lock`/
`requirements.txt`/`compile_lock.sh`를 전혀 건드리지 않은 Iteration 5가
그럼에도 이 job을 실패시킨 이유가 바로 이것 — 실패는 이 push가 아니라
PyPI 위 langsmith의 새 릴리스 타이밍에 달려 있었다.

## 6. 수정 (`scripts/compile_lock.sh::compile_once`)

```bash
compile_once() {
  local out="$1"
  if [ -f "$LOCK_FILE" ]; then
    cp "$LOCK_FILE" "$out"
  fi
  uv pip compile "$REQUIREMENTS_FILE" \
    --extra-index-url "$EXTRA_INDEX_URL" \
    --index-strategy unsafe-best-match \
    --generate-hashes --no-annotate \
    -o "$out" >/dev/null
  validate_and_normalize_header "$out"
}
```

`uv pip compile --help`의 `-U`/`--upgrade` 설명("Allow package upgrades,
**ignoring pinned versions in any existing output file**")이 이
기본(비-`-U`) 동작의 근거다 — 이미 존재하는 `-o` 대상 파일의 버전을
uv가 선호로 취급한다는 뜻이며, `--upgrade`/`--upgrade-package`를 주지
않는 한 이 iteration 이전의 `compile_once()`는 그 선호를 줄 방법이 아예
없었다(`mktemp`가 항상 빈 파일을 만듦). `requirements.txt`가 실제로
바뀌어 어떤 패키지의 제약이 달라지면 uv는 그 부분만 다시 해석하고,
바뀌지 않은 나머지 unpinned transitive 패키지는 committed lock에 이미
있던 버전 그대로 유지된다 — "langsmith만 손으로 pin"하는 국소 패치가
아니라, 앞으로 어떤 이름의 unpinned transitive 패키지가 드리프트하든
동일하게 막는 구조적 수정이다. 의도적으로 특정 패키지를 올리고 싶으면
committed `requirements.lock`에서 그 줄을 지우거나(또는 파일 전체를
지우고 처음부터 재생성) 다시 실행하면 된다 — 새 CLI 플래그를 추가하지
않았다(최소 변경 원칙).

`--extra-index-url`/`--index-strategy unsafe-best-match`/
`--generate-hashes`/`--no-annotate`, 헤더 정규화(`validate_and_
normalize_header`), 두 번 컴파일해 자기-재현성을 확인하는 기존 로직,
`--verify` 모드의 committed-lock 바디 비교는 전혀 바뀌지 않았다 — 오직
`compile_once()`가 uv를 부르기 *전에* 무엇을 `-o` 대상에 미리 써두는지만
바뀌었다.

## 7. 재검증 결과

- **로컬(macOS ARM64, 참고용)**: `bash scripts/compile_lock.sh --verify`는
  여전히 실패한다 — 이는 사전에 알려진, 이 수정과 무관한 별개의
  플랫폼 마커 차이 때문이다(macOS ARM 해석은 애초에 102개 패키지만
  선택하고 `greenlet`이 빠진다 — hosted/committed는 103개). 이 iteration
  이전에도 로컬 macOS는 hosted lock을 검증할 수 없었다(Hosted_Container_
  Remediation_Iteration_5.md §2.3과 동일하게 로컬-hosted 플랫폼 차이가
  존재).
- **linux/amd64(hosted와 동일 아키텍처, `python:3.11.15-slim` +
  `uv==0.8.15`)**: 수정 전 `bash scripts/compile_lock.sh --verify`는
  `committed requirements.lock has drifted from requirements.txt`로
  실패(hosted job 로그와 정확히 동일한 메시지, 동일한 "Resolved 103
  packages" 두 번). 수정 후 동일 컨테이너·동일 커맨드:
  **`compile_lock.sh: --verify PASS (reproducible, no drift)`**.
- 수정된 `compile_once()`로 재컴파일한 결과와 committed
  `requirements.lock`을 헤더(경로 의존적인 `-o` 인자 제외) 이후 전체
  바이트 단위로 diff: **완전히 동일**(해시 포함) — `langsmith==0.10.18`이
  그대로 유지됨을 직접 확인했다.
- `git diff --exit-code -- requirements.lock requirements.txt`: exit 0
  — 이 iteration은 lock의 **내용**을 전혀 바꾸지 않았다(생성 절차만).
- `venv/bin/python -m pytest -q`(전체 unit+integration, macOS 로컬):
  **1320 passed, 1 skipped**(신규 테스트 20건 포함 — Iteration 5 기준
  1306 passed 대비 순수 추가, 회귀 없음. 세부:
  `test_index_verification.py` 신규 permission-denied 매트릭스 1건,
  `test_index_lifecycle.py` 신규 `versions/` chmod 1건,
  `test_observability_logging.py` 신규 `engine_error_type` 스키마 4건,
  `tests/integration/test_health_endpoints.py` 신규 안전 관측성 1건,
  `test_container_smoke_readiness_diagnostics.py` 신규 tail-보존 2건,
  `test_compile_lock_header_contract.py` 신규 seeding 2건. 실행 중
  `docs/generated/logging_callsite_disposition.json`의 라인번호만
  드리프트한 것을 1회 잡아 재생성으로 닫았다 — 최종 실행은 1320/1320
  전부 그린).
- `docker build --platform linux/amd64 --target production -f
  deploy/Dockerfile -t simple-qna-rag:iter6-repro .`: **성공**.
- `venv/bin/python scripts/container_smoke.py --image
  simple-qna-rag:iter6-repro`(실제 docker, 보안 플래그·negative
  control 전부 포함, macOS `linux/amd64` 에뮬레이션): **`status:
  PASS`**, 6개 boolean 전부 true, `readiness_sequence.ready_last_reason:
  "ok"`, `ready_poll_elapsed_seconds: 7.04` — 회귀 없음, §1~§4의 변경이
  정상 경로를 전혀 방해하지 않음을 확인.
- `python scripts/generate_field_spec.py --check`: exit 0.
- `python scripts/logging_callsite_audit.py --check`: exit 0(신규
  `engine_error_type` validator/주석 추가로 인한 순수 라인번호 드리프트만
  있었고, `docs/generated/logging_callsite_disposition.json`을
  재생성해 반영 — 콜사이트 분류 자체는 바뀌지 않음).
- `python scripts/check_markdown_links.py`: 검사 파일 137개(이 문서
  포함), 링크 596개, 실패 0개.
- `git diff --exit-code -- .github/workflows/ci.yml
  scripts/scan_image_layers.py scripts/assemble_m4_evidence.py
  scripts/check_m4_baseline.py evaluation/baselines/m3_initial.*`
  (protected 경계): exit 0.
- Native Linux/Ollama/DDGS, protected M3/M4.1 live gate, self-hosted
  runner/environment 승인 경계는 이 remediation에서도 실행·변경하지
  않았다. `M4.1_BLOCKED=true`, protected M3 live `NOT_RUN`,
  `overall_release_ready=false` 산출 경로는 변경된 파일에 포함되지
  않는다.

## 8. 이 remediation이 건드리지 않은 것

`_poll_ready`의 `max_seconds`(60초 그대로, 다시 늘리지 않았다),
`evaluate_readiness()`의 분기 순서/우선순위, negative control의 판정
임계값(`expect_status=503`/
`expect_reason="artifact_test_embedding_seam_unavailable"`),
`compute_all_ok`/`_ALL_OK_KEYS`(5개 boolean 그대로),
`get_rag_engine()`의 allowlist-검증 로직 자체(REASONS에 3개 항목을
추가했을 뿐 검증 메커니즘은 그대로), `M4.1_BLOCKED=true`, protected M3
live `NOT_RUN`, `overall_release_ready=false` 산출 경로,
`m3-live-regression-gate` 블록, `.github/workflows/ci.yml`,
`scripts/scan_image_layers.py`/`assemble_m4_evidence.py`/
`check_m4_baseline.py`, `requirements.lock`/`requirements.txt`의 실제
패키지 목록/버전/해시(생성 **절차**만 바뀌었고 §7에서 바이트 단위
동일함을 확인), CPU torch extra-index 시맨틱, `--generate-hashes` 계약,
uv 0.8.15 고정, Native Linux/Ollama/DDGS/live/self-hosted 승인 경계,
위 결함과 무관한 어떤 파일도 수정하지 않았다.

## 9. 남은 hosted 검증 필요

이 커밋은 기존 PR #18에 push될 예정이다. **merge/commit/push는 수행하지
않는다** — fresh Codex 리뷰가 필요하다. §1~§2의 EACCES/umask 이론은
코드 추적과 로컬 unit 테스트(§7의 permission-denied 매트릭스)로 강하게
뒷받침되고 §4에서 실제 hosted job 로그로 메커니즘 자체(관측성 gap)를
재확인했지만, hosted 환경에서 `EACCES`가 실제로 발생하고 있었는지 자체는
100% 확증되지 않았다 — 다음 hosted 재실행이 결정적 증거다. 만약 §1~§2가
실제 원인이었다면 `container` job은 이제 `status: PASS`로 통과해야
한다. 만약 여전히 실패한다면, §3~§4가 추가한
`readiness_sequence.ready_last_reason`(이제 `artifact_root_permission_
denied`/`artifact_version_dir_permission_denied`/
`artifact_member_permission_denied` 중 하나로 바뀌어 있어야 정상이며,
여전히 plain `engine_init_failed`라면 EACCES가 원인이 아니었다는 뜻)과
`container_smoke.json`/job 콘솔의 `container_log_tail`의 `"event":
"startup"` 줄에 실린 `engine_error_type`(예: `ConnectionError`이면
mock Ollama 호스트-게이트웨이 경로, `MemoryError`/`RuntimeError`이면
2-vCPU 리소스 제약 등 §0에서 배제하지 못한 다른 가설 쪽)이 다음
iteration의 범위를 정확히 좁혀준다 — 다시 추측하지 않아도 된다.
`python-tests`(§6, linux/amd64 컨테이너로 이미 검증 완료 —
`compile_lock.sh --verify PASS`)는 hosted에서도 통과할 것으로 예상되며,
`container`/`m4-assemble`(하류 종속) 포함 전체 job이 이 push 이후
재확인이 필요하다.
