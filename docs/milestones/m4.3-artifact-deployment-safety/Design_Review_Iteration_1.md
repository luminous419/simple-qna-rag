# M4.3 Artifact & Deployment Safety 설계 리뷰 — Iteration 1

리뷰 일자: 2026-08-12 (Asia/Seoul)  
리뷰어: Fresh Codex (독립 설계 reviewer)  
대상: Claude Code Sonnet 5의 `Design.md`, M4.3 Requirement/Plan/Traceability,
`milestone_dev_orchestration_guide.md`, 현재 코드/tests/`.github/workflows/ci.yml`,
M4.1/M4.2 승인·예외 산출물  
판정 범위: pre-merge Design Quality Gate

## 판정

**FAIL — 5.8/10**

CRITICAL 1건, MAJOR 7건, MINOR 1건이 남아 있어 Gate 기준인 CRITICAL/MAJOR 0,
score 9.7/10 이상을 충족하지 못한다. 아직 구조적으로 생성될 수 없는 merge-SHA hosted
receipt, protected environment receipt, Native Linux/Ollama/live/self-hosted 실행 증거의 부재는
이 pre-merge FAIL 사유로 사용하지 않았다. FAIL은 모두 현재 설계대로 구현하면 재현되는
신뢰 경계 결함, 실행 불가능 계약 또는 fail-open checker에 근거한다.

## Findings

### DR-I1-CRIT-01 — 검증 뒤 경로 재오픈으로 공격자 pickle이 실행될 수 있다

- **Severity:** CRITICAL
- **Gate:** Pre-merge
- **근거:** `Design.md:401-406`은 검증 바이트를 재사용해 파일을 두 번 열지 않는다고
  주장하지만, `load_verified_faiss`는 `Design.md:420-429`에서 `verify_version()`이 fd를
  닫은 뒤 `FAISS.load_local(str(version_dir), ..., allow_dangerous_deserialization=True)`에
  경로를 넘겨 `index.faiss`와 `index.pkl`을 다시 연다. 따라서 hash 검증과 pickle
  역직렬화 사이에 파일/디렉터리 교체 창이 생긴다. 더구나 `contained_open`은
  `Design.md:311-325`에서 조상들을 path 기반으로 사전 검사하고 `O_NOFOLLOW`를 최종
  요소에만 적용하므로, 검사 뒤 조상 디렉터리 교체도 막지 못한다. 이는
  M4.3-REQ-001.4/001.5 및 NFR-003의 핵심 RCE 경계를 위반한다.
- **수정안:** root부터 `openat`/`dir_fd` 체인으로 각 디렉터리를
  `O_DIRECTORY|O_NOFOLLOW`로 열고 fd identity를 고정한다. 검증된 `index.faiss`와
  `index.pkl` bytes를 private 임시 디렉터리 또는 Linux `/proc/self/fd` 기반의 동일 fd
  identity로 materialize하여 FAISS loader가 정확히 검증한 bytes만 읽게 하거나,
  LangChain의 deserialize API를 좁게 감싸 검증 bytes에서 직접 복원한다. 공격 스레드가
  verify 직후 member와 ancestor를 교체하는 실제 race test에서 malicious pickle call이
  0회임을 증명해야 한다.

### DR-I1-MAJ-02 — symlink/dangling `current`가 안전하지 않은 legacy pickle 경로로 downgrade된다

- **Severity:** MAJOR
- **Gate:** Pre-merge
- **근거:** `resolve_current`는 `Design.md:446-450`에서 먼저
  `pointer_path.exists()`를 호출한다. dangling symlink는 `False`이므로
  `CurrentPointerMissing`으로 분류되고, `_load_vectorstore`는 `Design.md:765-785`에서
  manifest/hash 검증 없는 legacy `FAISS.load_local(..., allow_dangerous_deserialization=True)`로
  폴백한다. 이는 “current가 없음”과 “current가 존재하지만 신뢰 경계를 위반함”을
  구분하지 못하는 fail-open downgrade다.
- **수정안:** `exists()` 선검사를 제거하고 contained fd-open의 `ENOENT`만 genuine absence로
  분류한다. `lstat`상 symlink/non-regular/dangling entry는 항상 trust failure로 처리하고
  legacy fallback은 `lstat(current) == ENOENT`인 경우에만 허용한다. dangling/root/current
  symlink 각각이 legacy loader 0-call임을 spy로 검증한다.

### DR-I1-MAJ-03 — legacy import의 승인 근거가 CLI가 지정한 임의 파일이다

- **Severity:** MAJOR
- **Gate:** Pre-merge
- **근거:** 설계는 `Design.md:881-884`에서 임의 expected hash를 금지하고 커밋된 baseline만
  신뢰한다고 주장하지만, CLI는 `Design.md:866-868`에
  `--baseline-json PATH`를 공개한다. 호출자는 자신의 index hashes를 담은 임의 JSON을
  넘겨 승인된 legacy artifact로 만들 수 있다. 이는 Requirement/Traceability의
  “committed M3 hash import” 계약과 정면 충돌한다.
- **수정안:** production CLI에서 `--baseline-json`을 제거하고 패키지/checkout의 고정
  `evaluation/baselines/m3_initial.json` 및 고정 baseline ID를 사용한다. 테스트 주입은
  Python 함수 인자나 명시적 test-only seam으로 분리하되 production entry point에서
  접근할 수 없게 한다. 최소한 파일 bytes의 별도 pinned SHA와 git tracked/blob identity를
  검증하고 dirty/untracked baseline을 거부한다.

### DR-I1-MAJ-04 — retention 삭제가 공격자 TOCTOU에 안전하지 않고 staging retention도 미정의다

- **Severity:** MAJOR
- **Gate:** Pre-merge
- **근거:** cleanup은 `Design.md:718-724`에서 `realpath`와 `is_symlink/is_dir`를 검사한 뒤
  path 기반 `shutil.rmtree(target)`를 호출한다. advisory lifecycle lock은 비협조적 로컬
  actor의 rename/symlink 교체를 막지 않으므로 검사와 삭제 사이 root 밖 tree 삭제가
  가능하다. 또한 `.staging` 실패 잔여물을 보존한다고 한 `Design.md:535-537`과 달리
  `cleanup()`의 후보는 `versions/`만 열거하며(`Design.md:701-708`), staging의 TTL,
  owner, operation liveness, dry-run/apply 삭제 정책이 없다. disk-full 실패가 반복되면
  무제한 잔여물로 다음 publish까지 막는다.
- **수정안:** 삭제도 root/versions dirfd에 고정하고 `openat`/`fstatat` 기반 no-follow
  walk와 fd-relative unlink/rmdir을 사용하거나, immutable version directory의 inode를
  고정한 안전한 삭제 primitive를 구현한다. `.staging`은 별도 dry-run-first 정책으로
  최소 age, owner/mode, UUID name, non-symlink, active lock/operation 보호, bounded retention을
  명시하고 race/fault tests에 포함한다.

### DR-I1-MAJ-05 — durability primitive와 activation history가 crash-safe 계약을 완성하지 못한다

- **Severity:** MAJOR
- **Gate:** Pre-merge
- **근거:** `_write_fsync`는 `Design.md:541-544`에서 단일 `os.write`만 호출하므로 short
  write를 처리하지 않는다. lock도 `Design.md:659-663`의 `touch` 후 path 재-open이라
  symlink 교체 및 root 신뢰 문제가 남는다. activation은 pointer parent fsync 뒤
  `_append_history`를 실행하며(`Design.md:621-630`), history append/fsync 실패 시 pointer는
  이미 바뀌었지만 previous 기록과 receipt가 없다. retention과 `--to-previous`는 history에
  의존하므로(`Design.md:638-646`, `704-716`) rollback/retention 대칭성이 깨진다.
- **수정안:** 완전 쓰기 loop를 사용하고 file fd fsync뿐 아니라 file creation을 담은 parent
  directory fsync를 각 상태 전이에 명시한다. lock은 trusted root dirfd에서
  `O_CREAT|O_NOFOLLOW`로 열고 regular-file/owner/mode를 검증한다. pointer 전환과 동일한
  crash-recovery journal에 pre/post version을 먼저 durable 기록한 후 replace/fsync/commit
  marker 순서를 정의하고, 재시작 시 incomplete transition을 결정론적으로 reconcile한다.
  history/receipt fsync fault를 포함해 current/previous/protected retention의 상태 대수를
  검증한다.

### DR-I1-MAJ-06 — 제안한 production image는 정적 자산이 없어 readiness smoke가 성립하지 않는다

- **Severity:** MAJOR
- **Gate:** Pre-merge
- **근거:** 현재 settings는 `src/simple_qna_rag/settings.py:252-270`에서
  `PROJECT_ROOT/web/static`과 `PROJECT_ROOT/web/templates`를 runtime 경로로 사용한다.
  그러나 production stage는 `Design.md:982-988`에서 `src/`, metadata, 빈 runtime만
  COPY하고 `web/`을 복사하지 않는다. M4.2 readiness는 static mount 실패를 503으로
  우선 처리하므로, `Design.md:1152-1156`의 ready 200/mock query 계약은 이 이미지로
  달성할 수 없다.
- **수정안:** production COPY allowlist에 필요한 `web/static` 및 실제 사용하는 template
  tree만 명시적으로 추가하고 forbidden-content layer scan 대상에 포함한다. production
  image 자체에서 `/health/live`, `/health/ready`, `/`, static asset 응답까지 smoke한다.

### DR-I1-MAJ-07 — Linux hosted container에서 mock Ollama 주소가 연결되지 않는다

- **Severity:** MAJOR
- **Gate:** Pre-merge
- **근거:** `container_smoke.py` 절차는 `Design.md:1138-1151`에서 host thread의 mock server를
  띄우고 컨테이너에 `http://host.docker.internal:<port>`를 전달하지만, 제시한
  `docker run` argv에는 Linux Engine에서 그 이름을 만드는
  `--add-host=host.docker.internal:host-gateway` 또는 동등한 network 설정이 없다.
  따라서 목표 hosted 환경인 `ubuntu-latest`에서 mock query가 실행 불가능하다.
- **수정안:** Linux용 `--add-host ...:host-gateway`를 정확한 argv 계약에 넣거나 mock을
  별도 container/network로 실행해 service name으로 연결한다. argv unit test와 실제
  ubuntu Docker smoke를 모두 요구하되, 그 실제 post-merge receipt 부재 자체는 현재
  pre-merge FAIL 근거로 삼지 않는다.

### DR-I1-MAJ-08 — assembler가 상세 evidence를 검증하지 않아 합성 PASS가 가능하다

- **Severity:** MAJOR
- **Gate:** Pre-merge
- **근거:** assembler는 `Design.md:1420-1438`에서 producer receipt 하나만 읽고 identity를
  확인하며, `Design.md:1475-1482`도 상세 `layer_scan.json`, `container_smoke.json`,
  `m43.json`, negative receipt는 identity 검증 대상이 아니라고 명시한다. baseline에
  상세 파일명/hash를 “인용”한다고 하지만 제시된 알고리즘에는 required detailed file,
  schema/status, receipt-to-file digest manifest 검사가 없다. `upload-artifact`는 기본적으로
  일부 path가 없어도 경고로 성공할 수 있어, 최소 receipt만 있으면 container scan/smoke
  결과가 없거나 SKIPPED/FAIL이어도 `needs=success`와 producer `OK`가 되어 deterministic
  PASS를 합성할 수 있다.
- **수정안:** 각 producer receipt를 canonical envelope로 만들어 required payload 목록,
  payload SHA-256, schema, semantic PASS/expected-negative-FAIL을 포함한다. assembler가 fresh
  directory의 exact allowlist/exact-one 파일 집합, 모든 payload hash와 typed outcome을 같은
  parser로 검증한 뒤에만 producer `OK`를 만들고, upload에는
  `if-no-files-found: error`를 명시한다.

### DR-I1-MIN-09 — baseline checker의 스키마/상태 대수가 전체 필드 누락을 명시적으로 거부하지 않는다

- **Severity:** MINOR
- **Gate:** Pre-merge
- **근거:** `Design.md:1553-1574`의 checker는 존재하는 `gates.items()`만 enum 검사하고
  required top-level/gate key 집합, producer 집합, schema/version, deterministic constituent
  algebra를 검증하지 않는다. 예를 들어 `python_tests` 등 deterministic gate가 누락돼도
  별도 issue가 없고, `deterministic_status="PASS"`를 producer 상태와 재계산하지 않는다.
  정상 assembler만 신뢰하면 현재 예시는 맞지만, checker 자체가 malformed/synthetic
  candidate를 완전히 fail-closed하지 못한다. Traceability `M4.3-REQ-008`의 enum omission
  rejection과도 불일치한다.
- **수정안:** strict exact-key JSON schema를 먼저 적용하고 required gate/producer 집합과
  enum을 전수 검사한다. deterministic status를 네 required gate와 verified producer에서
  재계산하고 `M4.1_BLOCKED`, operational status, live gates, overall-ready를 하나의 total
  state function으로 검증한다. 모든 key별 omission, extra key, wrong type, `null`, truthy
  string negative cases를 추가한다. 상세 evidence가 없는 상태에서 checker가 PASS할 수
  있으므로 DR-I1-MAJ-08과 함께 닫기 전까지 실질적으로는 MAJOR 경로의 일부다.

## 확인된 강점과 보존 조건

- `Design.md:19-24`, `1510-1546`과 Traceability `49-61, 111-126`은
  `M4.1_BLOCKED=true`, protected M3 live `NOT_RUN`, `overall_release_ready=false`를 명시하며,
  M4.1 예외 문서 `Operational_Acceptance_Exception.md:10-17, 30-48` 및 M4.2 최종 검증
  `Final_Verification_Report.md:18-20, 81-86`과 일치한다. 이 분리는 유지해야 한다.
- 현재 `.github/workflows/ci.yml:96-175`의 protected live block은 self-hosted
  `[self-hosted, ollama-m3]`, protected environment, trusted trigger, timeout/concurrency,
  Ollama/vectorstore preflight를 보존한다. M4.3 구현은 이 block의 semantic snapshot이
  0-diff임을 검사해야 하며, hosted M4.3 DAG의 `needs`에 이 live job을 넣어서는 안 된다.
- `Design.md:1634-1807`의 watchdog exact terminal/run-bound argv, `--peek`, failure propagation,
  dry-run 테스트 범위는 현재 working-tree delta `scripts/orchestration_watchdog.py:34-56`의
  의도와 부합한다. 다만 “6종”이라고 쓰고 실제 7개 테스트를 열거하는 문서 표현은 구현
  전 정리하는 편이 좋다. 이 미커밋 fix와 테스트/provenance를 한 commit scope로 보존해야
  한다.
- Native Linux/Ollama/DDGS/M3 live/self-hosted 실행은 이 deterministic pre-merge review의
  대상에서 의도적으로 제외했다. 해당 post-merge 증거가 없다는 이유만으로 추가 finding을
  만들지 않았다.

## 재검토 진입 조건

CRITICAL 1건과 MAJOR 7건을 설계에서 먼저 해소하고, 수정된 symbol/상태 머신/정확한 argv와
negative test가 Requirement 및 Traceability에 다시 연결되어야 한다. 다음 독립 iteration은
특히 (1) 검증 bytes와 pickle load identity가 동일함, (2) root-to-leaf fd-relative open/delete,
(3) pinned legacy approval source, (4) crash-recoverable pointer/history algebra, (5) 실제
production image의 static/mock connectivity, (6) detailed evidence와 baseline strict schema를
우선 재검증해야 한다.
