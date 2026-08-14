# M4.3 Hosted CI Remediation Iteration 4 — 독립 코드 리뷰

리뷰어: Codex (fresh independent review)  
대상 기준: `b072fc9f4d88046693b4e06fcb482aeeea2b0046` 이후 working-tree diff  
대상 run: [GitHub Actions 31730391051](https://github.com/luminous419/simple-qna-rag/actions/runs/31730391051)  
리뷰 일자: 2026-08-14

## 1. 판정

**FAIL — 9.5/10**

| 등급 | 건수 |
|---|---:|
| CRITICAL | 0 |
| MAJOR | 1 |
| MINOR | 0 |
| TRIVIAL | 0 |

통과 기준인 **CRITICAL 0 / MAJOR 0 / 9.7 이상**을 만족하지
못한다. `requirements.lock`의 실질 body, 상류 버전/해시,
extra-index 해석, 그리고 positive readiness 60초 변경 자체는 타당하다.
다만 이번 remediation이 추가한 lock header canonicalization은 uv 출력
형식이 예상과 다를 때 실패하지 않고, `--verify`도 그 실패를
감지하지 못한다.

## 2. 필수 수정 사항

### CR-HCI4-MAJ-01 — header 정규화가 불일치를 조용히 통과시켜 canonical lock 계약을 보장하지 못함

**위치:** `scripts/compile_lock.sh:23-35`, `scripts/compile_lock.sh:63-77`

`compile_once()`의 새 `awk`는 2번째 줄에서 `-o [^ ]+$`가 맞을 때만
치환한다. `sub()`이 0건을 치환해도 `awk`는 정상 종료하므로,
uv가 header 줄을 옮기거나, output 인자를 `--output-file` 형태로 표시하거나,
`-o` 뒤에 다른 인자를 기록하거나, header 생성 정책을 바꾸면
절대 `mktemp` 경로가 그대로 남은 파일이 성공 산출물로 취급된다.

더 큰 문제는 `--verify`가 `canonical_body()`로 모든 선행 `#` 줄을
제거한 후 body만 비교한다는 점이다. 따라서 새로 생성한 두 파일의
header 정규화가 모두 실패해도 body가 같으면 reproducibility PASS가 나고,
커밋된 `requirements.lock`의 header가 임의 값이어도 no-drift PASS가 난다.
즉 문서 §1.3의 “최종 파일의 header를 호스트/플랫폼에 무관하게
결정론적으로 만든다”는 신규 계약이 실제 gate로 고정되지 않았다.

**필수 조치:**

1. 치환 전 header를 exact/bounded grammar로 검증하고 예상한 단 한 건이
   치환되지 않으면 nonzero로 종료해야 한다.
2. 치환 후 2번째 줄이 정확한 canonical command/header인지 다시
   확인해야 한다.
3. `--verify`가 body 동등성뿐 아니라 커밋된 header의 canonical
   형태도 검증하게 해야 한다.
4. fake `uv`를 사용한 결정론적 shell test로 (가) 현재 header PASS,
   (나) header 위치 변경, (다) `-o`가 말미가 아닌 경우, (라) `-o`
   누락, (마) 커밋 header 변조가 모두 fail-closed됨을 고정해야 한다.

이는 단순 주석 cosmetic 문제가 아니라, 이번 diff가 명시적으로
추가한 canonical artifact 보장이 관련 형식 drift에서 fail-open하는
문제이므로 MAJOR로 판정한다.

## 3. 검증한 정상 범위

### 3.1 hosted 실패 진단

run `31730391051`의 exact head SHA는 `b072fc9f4d88046693b4e06fcb482aeeea2b0046`이다.
`python-tests`는 hash-locked install과 `pip check`를 통과한 뒤 lock verify에서
실패했고, `container`는 이미지 build/layer scan을 통과한 뒤 smoke에서
실패했다. `frontend-tests`/`m43-deterministic`는 성공,
`m3-live-regression-gate`는 SKIPPED, `m4-assemble`는 필수 producer 실패에
따라 실패했다. remediation 보고서의 실패 분류와 일치한다.

### 3.2 lock body, 상류 버전/해시, index 의미론

- `b072fc9`와의 body diff는 package 세트를 바꾸지 않고
  `filelock 3.32.2 -> 3.32.3`, `uvicorn 0.52.2 -> 0.52.3`와 각 두
  distribution hash만 바꾸었다.
- [PyPI filelock 3.32.3 metadata](https://pypi.org/pypi/filelock/3.32.3/json)의
  wheel/sdist SHA-256은 lock의
  `7f0ca4...a1f09`/`0ffa18...3487f`와 정확히 일치한다.
- [PyPI uvicorn 0.52.3 metadata](https://pypi.org/pypi/uvicorn/0.52.3/json)의
  wheel/sdist SHA-256은 lock의
  `116af2...2c7c`/`18857b...8b58`와 정확히 일치한다.
- `--extra-index-url https://download.pytorch.org/whl/cpu` +
  `--index-strategy unsafe-best-match`는 기존 계약에서 변경되지 않았다.
  uv 공식 문서상 `unsafe-best-match`는 복수 index의 candidate를
  합쳐 best version을 고르는 pip-style 동작이며 dependency-confusion 위험을
  수반한다. 이 리스크는 신규 변경이 아니고, CPU torch 해석을 위해
  이미 승인된 lock 경계다. 참고:
  [uv multiple-index semantics](https://docs.astral.sh/uv/pip/compatibility/#packages-that-exist-on-multiple-indexes).
- 보고된 linux/amd64 Python 3.11 + `uv==0.8.15` 2회 compile/no-drift
  증거와 current body diff는 hosted Python 3.11 x64 대상에 적합하다.
  다음 hosted run의 native x64 `--verify` PASS가 최종 receipt다.

### 3.3 positive readiness 60초

`scripts/container_smoke.py:189`는 positive `expect_status=200` 호출의
예산만 10초에서 60초로 늘린다. 200 판정, 후속 `/rag` 200,
root/static asset, host-gateway, graceful stop, image identity는 그대로며
`compute_all_ok()` 의 boolean 의미도 바뀌지 않았다. 60초 내에 200이 없으면
여전히 `ready_ok=false`/exit 1이므로 진짜 기동 실패를 성공으로
마스킹하지 않는다.

negative seam 호출(`scripts/container_smoke.py:234-236`)은 기본 10초,
`expect_status=503`, exact reason
`artifact_test_embedding_seam_unavailable`를 그대로 유지한다. 따라서
production test seam 봉인과 fail-closed 부정 경로는 약화되지 않았다.
run 31730391051에서 negative seam은 이미 true였고 positive ready/mock query만
false였던 비대칭도 이 변경 범위와 일치한다. 다만 60초가
hosted에서 실제로 해소하는지는 다음 run으로 확인해야 하며,
현재 증거만으로 “안정적으로 초과한다”를 보장할 수는 없다.

### 3.4 보호 경계

exact diff는 `requirements.lock`, `scripts/compile_lock.sh`,
`scripts/container_smoke.py`와 remediation 보고서뿐이다.
`.github/workflows/ci.yml`, M3 baseline, assembler/checker, protected
`m3-live-regression-gate`, self-hosted runner/environment 승인 설정,
`M4.1_BLOCKED=true`, `operational_status=BLOCKED`,
`overall_release_ready=false` 산출 경로에는 diff가 없다.

## 4. 독립 재현 검증

| 검증 | 결과 |
|---|---|
| `pytest -q tests/unit/test_container_smoke_contract.py tests/unit/test_container_smoke_bare_script.py` | **17 passed** |
| `bash -n scripts/compile_lock.sh` | PASS |
| `python -m compileall -q scripts/container_smoke.py` | PASS |
| 대상 diff `git diff --check` | PASS |
| protected workflow/M3 baseline/assembler/checker `git diff --exit-code b072fc9 -- ...` | PASS |
| PyPI release metadata 대조 | filelock/uvicorn 버전·hash 정확히 일치 |

현재 17개 container smoke 테스트는 기존 argv/static/receipt 계약을
회귀 확인하지만, positive 호출이 `max_seconds=60`을 전달하는지와
header 정규화의 성공/실패 경계를 고정하는 테스트는 없다.
60초 상수 test 누락은 코드 분기가 단순하고 실제 호출이 명확하므로
별도 finding으로 올리지 않았으나, header fail-closed test는
CR-HCI4-MAJ-01의 필수 폐쇄 증거다.

## 5. 다음 Gate

CR-HCI4-MAJ-01을 코드와 결정론적 test로 폐쇄한 뒤 fresh Codex
재리뷰를 받아야 한다. 그 리뷰가 CRITICAL 0 / MAJOR 0 / 9.7
이상을 만족하면 기존 PR에 push해 hosted `python-tests`, `container`,
`m4-assemble`을 재실행한다. 이 리뷰는 merge, protected live Gate 실행,
또는 전체 M4 release-ready 판정을 승인하지 않는다.
