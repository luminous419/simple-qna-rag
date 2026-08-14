# M4.3 Artifact & Deployment Safety — Hosted CI Remediation Iteration 2

작성자: Claude Code Sonnet 5 (hosted-CI remediation worker)
대상: PR #18, hosted run
[31606183756](https://github.com/luminous419/simple-qna-rag/actions/runs/31606183756)
기준 revision: `765bde3`(`agent/m4-3-artifact-deployment-safety`, PR #18
HEAD) — 이 문서가 기술하는 변경은 같은 브랜치에 새 커밋으로 추가되며 PR을
새로 열지 않는다. **merge는 수행하지 않는다** — fresh Codex 리뷰가 필요하다.

## 0. 범위

독립 리뷰 [Code_Review_Iteration_3.md](Code_Review_Iteration_3.md)(판정
FAIL 8.8/10, `MAJOR 2`)가 지적한 `CR-I3-MAJ-01`/`CR-I3-MAJ-02` 2개 finding과,
hosted run 31606183756이 `container` job에서 실제로 낸 실패
(`container_smoke.py`의 `ModuleNotFoundError: No module named 'tests'`)를
고쳤다. `python-tests` job이 재차 드러낸 `requirements.lock` drift
(`charset-normalizer` 3.4.9→3.5.0, 상류 릴리스로 인한 것으로 Hosted CI
Remediation Iteration 1과 동일한 성격)도 같은 절차로 재컴파일했다.

이 remediation이 건드리지 않은 것: `M4.1_BLOCKED=true`, protected M3 live
`NOT_RUN`, `overall_release_ready=false` 산출 경로, `m3-live-regression-gate`
블록, Native Linux/Ollama/DDGS, self-hosted runner/environment 승인 경계,
위 세 결함과 무관한 어떤 파일도 수정하지 않았다.

## 1. CR-I3-MAJ-01 — `is_verified_ca_bundle`이 인증서 뒤에 임의 바이트를 허용

### 1.1 원인

기존 구현은 `_PEM_LABEL_RE.findall()`로 `BEGIN ...` 라벨만 찾아 전부
`CERTIFICATE`인지 확인한 뒤, 문자열 전체를
`SSLContext.load_verify_locations(cadata=...)`에 위임했다. OpenSSL의
PEM 파서는 인식하지 못하는 텍스트를 앞뒤로 무시하고 유효한 인증서
블록만 뽑아 쓰므로, "순수 CERTIFICATE 블록만" 계약이 실제로는 지켜지지
않았다. 리뷰가 재현한 세 probe(`REAL_CA_CERT + "API_TOKEN=..."`,
`REAL_CA_CERT + "not-a-pem secret payload"`, `REAL_CA_CERT +
"-----END PRIVATE KEY-----"`) 모두 `True`를 반환했다 — 신뢰 저장소
경로에 놓인 유효한 CA 뒤에 시크릿을 이어붙이면 허용목록을 그대로
통과했다.

### 1.2 수정

`is_verified_ca_bundle()`을 라벨 스캔에서 전체-입력 소비(full-consumption)
정규식 `fullmatch`로 재작성했다:

```python
_WS = r"[ \t\r\n]*"
_B64_LINE = r"[A-Za-z0-9+/=]+"
_CERT_BLOCK = (
    r"-----BEGIN CERTIFICATE-----\r?\n"
    rf"(?:{_B64_LINE}\r?\n)+"
    r"-----END CERTIFICATE-----"
)
_STRICT_PEM_BUNDLE_RE = re.compile(rf"^{_WS}(?:{_CERT_BLOCK}{_WS})+$")
```

`fullmatch`는 입력 문자열의 처음부터 끝까지 앵커링되므로, 하나 이상의
완전한 `BEGIN CERTIFICATE`/`END CERTIFICATE` 블록과 그 사이의 순수
공백(스페이스/탭/CR/LF)만으로 전체 바이트가 소비돼야 한다. 앞뒤로
붙은 시크릿·키=값 텍스트, 다른 PEM 라벨(개인키/CSR 등), 짝이 맞지 않는
BEGIN/END, base64 알파벳 밖의 바이트 — 이 중 하나라도 있으면 최소 한
바이트가 매치에서 빠져 전체 `fullmatch`가 실패한다. 정규식을 통과한
뒤에도 기존과 동일하게 ASCII 디코드와
`ssl.SSLContext.load_verify_locations(cadata=...)` 구조적 검증을
그대로 거치므로, 문법은 맞지만 실제 X.509로 파싱되지 않는 base64(예:
`test_malformed_cert_under_trust_store_path_is_still_credential` fixture)는
여전히 거부된다.

### 1.3 신규 테스트

`tests/unit/test_scan_image_layers.py`에 `is_verified_ca_bundle()` 레벨
7개(붙인/앞에 붙인/사이에 낀 시크릿, 라벨 없는 `END PRIVATE KEY` 꼬리,
짝 없는 BEGIN, 라벨 불일치 END, non-ASCII 바이트)와 `classify_member()`
엔드투엔드 1개(신뢰 경로의 정규 파일에 진짜 CA + 시크릿을 이어붙인
콘텐츠가 `credential`로 분류되는지)를 추가했다.

## 2. CR-I3-MAJ-02 — symlink/hardlink가 경로만으로 허용목록을 우회

### 2.1 원인

`scan()`이 `TarInfo.issym()`/`islnk()` 둘 다를 `is_symlink=True`로
넘겼고, `classify_member()`는 신뢰 경로의 symlink/hardlink를 콘텐츠
확인 없이 즉시 clean 처리했다. `member.linkname`은 전혀 읽지 않았다 —
정규화도, 허용목록 대상 target 검증도, 참조된 멤버의 실제 바이트
검증도 없었다. 결과: 레이어에 실제 시크릿(`app/secrets/key.pem`)과
그것을 가리키는 신뢰 경로 하드링크(`etc/ssl/certs/innocent.pem`)를
같이 넣으면, 하드링크 자체가 경로만으로 예외 처리됐다. 리뷰가 재현한
`classify_member("etc/ssl/certs/innocent.pem", is_symlink=True)`는
`None`을 반환했다.

### 2.2 수정

`classify_member()`의 `is_symlink: bool` 파라미터를 `is_regular_file:
bool`로 교체하고, CA 콘텐츠 허용목록 분기를
`is_regular_file and _is_trusted_ca_path(norm) and read_content is not None`
조건으로 좁혔다 — symlink/hardlink/device/FIFO/디렉터리는
`is_regular_file=False`이므로 이 분기에 절대 도달하지 않고, 항상
일반 `FORBIDDEN_PATTERNS` 루프로 떨어진다. `scan()`은 이제
`member.isfile()`(tarfile의 실제 멤버 타입 검사)을 그대로
`is_regular_file`로 전달한다 — path-only 판단이 아니라 멤버 타입을
먼저 검증한 뒤에만 콘텐츠 읽기/분류를 수행한다.

이 변경은 리뷰가 제시한 두 선택지 중 "보수적으로 모든 링크를
credential로 분류" 쪽을 택했다 — target을 레이어 내에서 bounded/
cycle-safe하게 resolve해 재검증하는 대안보다 attack surface가 작고,
실제 컨테이너 CA 신뢰 저장소는 symlink를 정규 파일 옆에 두므로(예:
`usr/lib/ssl/cert.pem -> /etc/ssl/certs/ca-certificates.crt`) 정규
파일 쪽이 스캔되면 링크 자체가 credential로 분류돼도 실제 CA 신뢰가
깨지지 않는다. 부작용으로, 기존
`test_symlinked_openssl_default_bundle_is_allowed_without_content`
테스트의 기대값(symlink는 허용)이 뒤집혔으므로
`test_symlink_at_trusted_ca_path_is_still_credential`로 이름과 assertion을
갱신했다(이제 `("credential", ".pem")`을 기대).

부수 결함도 같이 닫았다: `FORBIDDEN_PATTERNS`에 `.crt` 항목이 없어서,
`_is_trusted_ca_path()`가 `.crt` 확장자를 허용목록 후보로 인식함에도
불구하고 신뢰 경로 밖의 `.crt` 파일이나 신뢰 경로의 비정규 `.crt`
멤버는 어떤 패턴에도 걸리지 않고 조용히 통과했다 — `.pem`과 동일하게
`.crt`도 `("credential", ".crt")`로 잡히도록 패턴을 추가했다. 이
항목이 없었다면 MAJ-02의 fail-closed 수정이 `.pem` 확장자에만
성립하고 `.crt`에는 성립하지 않았을 것이다.

### 2.3 신규 테스트

symlink/hardlink가 신뢰 경로에서 여전히 credential인지(2), 그 target이
traversal 경로를 시도해도 목적지를 따라가거나 resolve하지 않는지(2,
symlink+hardlink), character device/FIFO/디렉터리가 신뢰 경로에서
비정규 멤버로 거부되는지(3), 신뢰 경로 밖 `.crt`가 잡히는지(1),
`scan()` 진입점 전체를 통해(합성 OCI tar를 만들어 `export_image`를
monkeypatch) 하드링크 우회가 실제로 violation을 내는지(1) — 총 9개.

## 3. Hosted run 31606183756 `container` job 실패 — `container_smoke.py` import 오류

### 3.1 원인

`Scan OCI layers` 스텝은 통과(`violations: []`)했지만 다음 스텝
`Container security/mock smoke`가 즉시 실패했다:

```
File ".../scripts/container_smoke.py", line 148, in run_smoke
    from tests.support.mock_ollama import start_mock_ollama_server
ModuleNotFoundError: No module named 'tests'
```

`ci.yml`의 `container` job은 `python scripts/container_smoke.py`를 바로
실행한다(`pytest`를 거치지 않는다). 스크립트 파일로 직접 실행되는
Python 프로세스는 스크립트 자신이 있는 디렉터리(`scripts/`)만
`sys.path[0]`에 넣고, 현재 작업 디렉터리(저장소 루트)는 넣지 않는다.
`pyproject.toml`의 `[tool.pytest.ini_options] pythonpath = ["."]`는
`pytest` 실행에만 적용되는 ini 옵션이라 이 경로에는 관여하지 않는다.
따라서 `tests/` 패키지(최상위 `__init__.py` 없이 namespace 패키지로
존재)를 import할 방법이 전혀 없었다 — 이 버그는 M4.3 feature 커밋
(`5b91840`)부터 존재했지만, 이전 두 hosted run은 그보다 앞선
스텝(lock drift, 스캐너 false positive)에서 먼저 실패해 이 코드
경로까지 도달하지 못했었다.

### 3.2 수정

`run_smoke()` 진입부에서, 이미 모듈 최상단에 정의돼 있던
`REPO_ROOT = Path(__file__).resolve().parents[1]`을 명시적으로
`sys.path`에 삽입한 뒤 `tests.support.mock_ollama`를 import하도록
한 줄을 추가했다:

```python
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tests.support.mock_ollama import start_mock_ollama_server
```

같은 파일의 `_build_fixture_index()`가 이미 동일한 패턴(`tests/support`를
`sys.path`에 넣어 `simple_qna_rag_test_seam`을 import)을 쓰고 있어
스타일을 맞췄다. 정책/검증 로직은 전혀 건드리지 않았다 — 순수
import 배선 버그다. 로컬에서 `python scripts/container_smoke.py
--image nonexistent:image`로 재현한 결과 `ModuleNotFoundError`는
사라지고, 존재하지 않는 이미지이므로 `docker_run_failed`로만
실패한다(정상 동작).

## 4. `requirements.lock` drift(재발)

`bash scripts/compile_lock.sh --verify`가 linux/amd64 컨테이너 안에서
다시 drift를 보고했다 — `charset-normalizer` 3.4.9→3.5.0(hash 목록
포함) 단 1개 패키지, 패키지 총수는 103개로 불변. Hosted CI Remediation
Iteration 1과 완전히 동일한 성격(상류 PyPI 릴리스로 인한 재해석
드리프트)이므로 같은 절차(`docker run --rm --platform linux/amd64
python:3.11-slim`에서 `uv==0.8.15`로 `compile_lock.sh` 재실행 후
`--verify`)로 재컴파일했다. `test_lock_package_count_is_103` 등
하드코딩된 패키지 수 assertion은 그대로 유효해 테스트 변경은
필요 없었다.

## 5. 재검증 결과

- `pytest -q tests/unit/test_scan_image_layers.py`: **39 passed**(기존
  20개 + 신규 19개 — MAJ-01 8개, MAJ-02 9개, `.crt` 회귀 1개,
  duplicate/whiteout-history 회귀 2개, 기존 symlink 테스트 갱신 1개는
  이름 변경으로 카운트 유지).
- `pytest -q`(전체 unit+integration, macOS 로컬): **1206 passed, 1
  skipped**(회귀 없음).
- linux/amd64 `python:3.11-slim`, `uv==0.8.15`,
  `bash scripts/compile_lock.sh --verify`: **PASS**(재컴파일 후,
  103 packages, drift 없음).
- `python scripts/run_m43_acceptance.py --profile deterministic --repeat
  10 --seed 4303`: **status=PASS**, `layer_scanner`/
  `container_static_and_connectivity` 포함 17개 node 전부
  `success_count=10/10`.
- `python scripts/run_m43_acceptance.py --profile deterministic --repeat
  3 --seed 4303 --inject-evidence-mismatch`(negative control): exit 1
  (기대하는 성공), `negative_control.result=REJECTED_AS_EXPECTED`.
- `python -m evaluation.dataset validate evaluation/datasets/golden.jsonl`:
  **valid=true**.
- `python scripts/generate_field_spec.py --check`,
  `python scripts/logging_callsite_audit.py --check`: 둘 다 **exit 0**.
- `python scripts/check_markdown_links.py`: 검사 파일 117개, 링크 536개,
  실패 0개.
- `git diff --exit-code -- evaluation/baselines/m3_initial.*`: **exit 0**.
- `npm test`: **9 passed**. `npm run sync-vendor` 이후
  `git diff --exit-code -- web/static/vendor/`: **exit 0**.
- `git diff --check`: 실패 0.
- Native Linux/Ollama/DDGS, protected M3/M4.1 live gate, self-hosted
  runner/environment 승인 경계는 이 remediation에서도 실행·변경하지
  않았다. `M4.1_BLOCKED=true`, protected M3 live `NOT_RUN`,
  `overall_release_ready=false` 산출 경로는 변경된 파일에 포함되지
  않는다(`git diff`로 확인 — 이 remediation이 건드린 파일은
  `requirements.lock`, `scripts/scan_image_layers.py`,
  `scripts/container_smoke.py`, `tests/unit/test_scan_image_layers.py`
  4개뿐).

## 6. 이 remediation이 건드리지 않은 것

`M4.1_BLOCKED=true`, protected M3 live `NOT_RUN`,
`overall_release_ready=false` 산출 경로, `m3-live-regression-gate` 블록,
`scripts/assemble_m4_evidence.py`/`scripts/check_m4_baseline.py`의
baseline 상태 분리 로직, index lifecycle/container smoke의 나머지 계약
(정적 자산 검증, negative production-seam 검증 등), Hosted CI
Remediation Iteration 1이 도입한 CA 허용목록의 경로/접미사 목록 자체 —
이번 세 결함과 무관한 어떤 파일도 수정하지 않았다.

## 7. 다음 단계

이 커밋은 기존 PR #18에 push된다. **merge하지 않는다** — fresh Codex
리뷰가 필요하며, 이 문서와 `Implementation_Report.md` §11 /
`Traceability.md`의 갱신된 행을 참조 자료로 남긴다. hosted CI가 다시
통과하는지는 push 이후 별도로 확인해야 한다.
