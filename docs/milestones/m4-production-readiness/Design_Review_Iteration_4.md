# M4 Production Readiness 상세 설계 리뷰 — Iteration 4

검토일: 2026-08-08  
검토 대상: [Requirement](Requirement.md), [Plan](Plan.md), 최신
[Design](Design.md), [Traceability](Traceability.md),
[Iteration 1 리뷰](Design_Review_Iteration_1.md),
[Iteration 2 리뷰](Design_Review_Iteration_2.md),
[Iteration 3 리뷰](Design_Review_Iteration_3.md), 현재 제품 코드·CI·packaging 구조  
프로세스 기준: [milestone 개발 orchestration guide](../../../milestone_dev_orchestration_guide.md)  
검토 방식: Iteration 3의 M3-01~M3-07 및 m3-01~m3-02를 우선 폐쇄 감사하고,
설계 의사코드를 실제 코드/API와 대조해 race, trust boundary, clean build,
14-gate evidence 조립을 실패시키는 방향으로 독립 검증

## 1. 결론

**Gate: FAIL**  
**점수: 4.2 / 10.0**  
**발견사항: CRITICAL 0 / MAJOR 6 / MINOR 3 / TRIVIAL 0**

Iteration 4는 이전 회차보다 실제 라이브러리 seam, openat, production stage,
gate DAG를 구체화했지만 구현 Gate를 열 수 있는 상태는 아니다. executable spike의
핵심 관찰은 재현됐으나 그 관찰로부터 도출한 전체 계약은 아직 성립하지 않는다.

가장 직접적인 차단점은 다음과 같다. `QueryExecutor`는 callback 등록 실패 시 이미
실행 중인 future보다 slot을 먼저 반환하며, 완료 callback이 만든 finalize task가
실행되기 전 loop가 멈추는 경우를 의도적으로 보장 밖에 둔다. Ollama trickle worker는
설계 스스로 overall deadline을 보장하지 못한다고 인정한다. Settings는 실제 53개
필드를 선언하면서 30개라고 테스트하도록 정의했다. legacy 승인 root는 최초 open에서
`O_NOFOLLOW`를 쓰지 않아 root 자체 symlink를 허용한다. container artifact는 참조하는
result JSON을 업로드하지 않고 local run과 일치할 `run_id`도 생산할 수 없다. 마지막으로
gate self-test는 `live_mode="skip"`으로 live evidence를 만들지 않으면서 13개 evidence가
생긴다고 요구한다.

guide의 품질 기준(9.7 이상, CRITICAL/MAJOR 0)에 미달한다. 또한 기본 4회차 종료 시점의
조건부 연장 자격 중 `CRITICAL=0`, 잔여 문제가 구체적이라는 조건은 만족하지만,
**점수 9.0 이상 및 MAJOR 2건 이하 조건을 만족하지 않으므로 연장 자격이 없다.**
guide를 그대로 적용하면 이 설계 사이클은 여기서 중단해야 한다.

## 2. Iteration 3 발견사항 폐쇄 감사

| 이전 발견 | 판정 | 근거와 잔여 사항 |
|---|---|---|
| M3-01 executor ownership token/cancellation | **미해결** | grant-cancel 경로는 추가됐지만 callback 등록 실패 후 실행 worker의 소유권과 callback task 실행 보장이 닫히지 않았다. M4-01. |
| M3-02 request-scoped LLM/DDGS deadline | **부분 해결** | lock된 API spike와 DDGS subprocess는 타당하다. 그러나 Ollama의 overall worker deadline은 설계가 명시적으로 포기한다. M4-02. |
| M3-03 exact RetrievalTrace/ObservationSink | **대체로 해결** | 3필드 `RetrievalStageTrace`와 `safe_sink_call()` 통일은 실제 코드와 맞는다. 다만 `total` trace 보존 설명과 오류 시 stage pair 계약이 불완전하다. m4-01. |
| M3-04 committed hash/openat provenance | **미해결** | 승인 hash는 실제 M3 baseline 값과 일치한다. 그러나 최초 approved root open이 symlink를 따라가므로 “모든 component” 계약이 성립하지 않는다. M4-03. |
| M3-05 production Docker/PEP 517/CI evidence | **미해결** | `requirements.txt` COPY와 마지막 production stage는 개선됐다. `COPY --exclude` frontend 버전과 container evidence artifact 구성은 실행 불가다. M4-04. |
| M3-06 fresh 14-gate DAG/assemble/self-test | **미해결** | local root와 container inbox 분리는 개선됐다. run binding 및 artifact 조립과 self-test count가 모순된다. M4-05. |
| M3-07 settings inventory/Prometheus API | **미해결** | Prometheus non-env API 주장은 확인됐지만 Settings 필드 수 계약이 실제 선언과 다르다. M4-06. |
| m3-01 smoke cleanup/bounded poll | **해결** | `trap`과 30회 bounded poll이 실제 CI 블록에 포함됐다(Design.md:2645-2666). |
| m3-02 module::symbol 근거 | **해결** | `ChatOllama._set_clients`, `OllamaLLM._set_clients`, `DDGS._search_sync` 등 구체 symbol이 표기됐다(Design.md:1716-1837). |

## 3. executable spike 재검증

현재 프로젝트 venv에서 다음을 read-only introspection으로 재실행했다.

| 항목 | 실측 결과 | 설계 주장 |
|---|---|---|
| package version | `langchain-ollama==0.3.10`, `ollama==0.6.0`, `ddgs==9.14.4` | **일치** |
| `ChatOllama.model_copy(update=...)` | 복사 전후 `_client`가 동일 객체, 실제 timeout은 원래 5초 유지 | **일치** |
| `OllamaLLM.model_copy(update=...)` | 동일하게 client 재생성 안 됨 | **일치** |
| `validate_model_on_init` 기본값 | 두 클래스 모두 `False` | **일치** |
| `ChatOllama.bind_tools([])` | `RunnableBinding`, `.bound is original_llm`, 네트워크 호출 없음 | **일치** |
| DDGS API/구현 | `.text()`에 per-call timeout 없음; `_search_sync()`가 `ThreadPoolExecutor` context manager를 사용해 exit 시 `shutdown(wait=True)` | **일치** |
| Prometheus public API | 현재 system 환경에서 top-level 및 `prometheus_client.metrics`에 `disable_created_metrics()` 존재. M4 lock 대상 0.26.0은 아직 설치 전이므로 lock 설치 gate에서 재확인 필요 | **주요 방향 일치**, 구현 전 lock 증거 필요 |

즉 spike 자체를 허위라고 볼 근거는 없다. 아래 발견사항은 spike 결과가 아니라 그
결과를 이용해 작성한 소유권·deadline·artifact 조립 설계가 완결되지 않은 문제다.

## 4. MAJOR 발견사항

### M4-01 — executor ownership token이 “이미 제출됐지만 callback 미등록” future를 소유하지 못한다

근거:

- `loop.run_in_executor()`가 future를 반환한 뒤 `future.add_done_callback()`이
  동기 예외를 던지면 `callback_registered=False`인 `finally`가 곧바로
  `_finalize(ticket)`를 호출한다(Design.md:1153-1177). 그러나 worker future는 이미
  실행 중일 수 있다. slot을 먼저 반환하면 다음 ticket이 admit되어 실제 worker 수가
  `concurrency_limit`를 초과하며, “실행 중 작업이 slot을 정직하게 점유”해야 하는
  M4-REQ-006.3을 위반한다.
- 설계는 이 경우를 “future의 완료 콜백은 절대 호출되지 않으므로 caller가 유일한
  소유자”라고 설명하지만, callback 부재와 worker 부재는 같은 명제가 아니다. caller는
  future가 끝날 때까지 기다리거나 별도의 guaranteed completion watcher로 소유권을
  넘겨야 한다.
- `_schedule_finalize()`는 `ensure_future()` 성공만 확인하고 새 task가 실제 한 번
  실행되기 전에 loop가 정지/종료되는 경우를 보장 밖으로 둔다
  (Design.md:1237-1270, 1431-1434). 그 테스트도 release나 `_running==0`을 assert하지
  않고 “ensure_future가 정상 호출됨”만 확인한다. 이는 Iteration 3이 요구한
  `callback_queued_then_loop_shutdown` 완료성 검증을 폐쇄하지 않는다.
- 취소된 caller가 `await asyncio.shield(self._finalize(...))`를 수행하면 내부 task는
  계속되지만 caller await 자체는 즉시 `CancelledError`로 끝날 수 있다. 따라서 state
  table의 “같은 run 호출 안에서 즉시 -1” 주장은 shield의 실제 계약보다 강하다
  (Design.md:1122-1139, 1389).
- queue timeout과 grant가 경쟁해 이미 deadline이 지난 경우에도 새 worker를 제출해
  즉시 executing timeout으로 버린다(Design.md:1111-1121, 1387). 만료된 요청이
  expensive model 작업을 새로 시작할 수 있어 overload 회복을 해친다.

영향: M4-REQ-006.2/.3, M4-NFR-002. callback 등록 실패나 shutdown 경계에서
concurrency 상한·drain 완료·slot 회수가 증명되지 않는다.

수정안: future 생성 직후 caller가 future ownership을 유지하고, callback 등록 성공
시에만 callback으로 이전한다. 등록 실패 시 future가 끝날 때 finalize하는 별도 watcher를
loop shutdown 정책과 함께 보장하거나 제출 자체를 취소 가능한 방식으로 되돌려야 한다.
finalizer completion task set을 executor가 강한 참조로 보유하고 shutdown drain이 그
task들의 완료까지 기다리게 한다. 이미 만료된 grant는 worker를 제출하지 않고 slot을
반환한다. 테스트는 실제 `_running==0`, next admit, worker 수 상한까지 단언해야 한다.

### M4-02 — request-scoped Ollama client는 per-operation timeout일 뿐 overall worker deadline이 아니다

근거:

- 설계는 `ChatOllama.invoke()`/`OllamaLLM`이 내부 streaming aggregate를 사용하고,
  httpx read timeout보다 짧은 간격의 trickle이면 worker가 deadline을 무기한 넘길 수
  있음을 정확히 분석한다(Design.md:1789-1807). 그런데 해결하지 않고 orphan metric과
  runbook 문서화로 남긴다.
- M4-REQ-007.4는 외부 호출의 **connect/read/overall timeout**을 모두 요구한다.
  QueryExecutor의 caller timeout은 외부 동기 호출 자체나 worker의 overall timeout이
  아니며, 모든 slot이 trickle worker로 점유되면 이후 요청이 계속 거절된다.
- `test_stream_trickle_bounded_by_budget`은 chunk 단위 `budget.expired()` 검사를
  기대하지만(Design.md:1917), 바로 앞의 API 분석은 LangChain Ollama 공개 API에 그런
  hook이 없다고 결론낸다. 테스트 기대와 선택한 구현 seam이 동시에 성립할 수 없다.
- DDGS에는 killable subprocess를 도입해 실제 bounded worker 반환을 만들었으나,
  router/answer에는 신뢰도가 높다는 정책 판단만으로 같은 필수 요구를 면제했다.

영향: M4-REQ-007.4, M4-NFR-002. 설계 자체가 필수 overall timeout을 충족하지 못함을
인정하므로 구현으로 해소될 여지가 없는 명세 위반이다.

수정안: router/answer도 deadline-aware transport/response iterator 또는 종료 가능한
process boundary로 격리한다. 이것이 비용상 허용되지 않는다면 Requirement의 overall
timeout을 변경해야 하므로 사용자 결정이 필요한 설계 변경이다. 어느 쪽이든 현재의
“caller만 bounded + worker orphan 허용”을 필수 gate 통과로 간주해서는 안 된다.

### M4-03 — legacy approved root 자체는 `O_NOFOLLOW` 없이 열려 symlink provenance를 허용한다

근거:

- 설계는 모든 component를 `O_NOFOLLOW`로 연다고 주장하지만 최초 root는
  `os.open(approved_legacy_import_root, os.O_RDONLY | os.O_DIRECTORY)`로 열며
  `O_NOFOLLOW`가 없다(Design.md:2311-2321).
- 따라서 `runtime/vectorstore` 자체가 공격자 디렉터리로 향하는 symlink여도 최초
  open은 성공하고 이후 모든 file open은 그 공격자 inode 아래에서 안전하게 완결된다.
  파일 내용이 승인 hash와 같다면 import가 통과한다. 이는 “operator-owned approved
  root” provenance가 아니라 “승인된 bytes가 있는 임의 symlink target”이다.
- 테스트 표는 바로 이 root 자체 symlink가 `ELOOP`로 실패한다고 기대하므로
  (Design.md:2400-2403) 의사코드와 acceptance test가 정면으로 모순된다.
- `os.fstat(fd, dir_fd=...)` 표기도 실제 Python API와 맞지 않는다. `os.fstat()`은
  열린 fd 하나만 받는다(Design.md:2342-2346). 파일 쪽 설명은 올바르지만 root/중간
  경로 구현자가 그대로 옮기면 `TypeError`가 난다.

영향: M4-REQ-008.2와 M4-NFR-004. M3-04의 핵심 trust boundary가 완전히 닫히지 않았다.

수정안: approved root도 trusted parent fd에서 basename을
`O_DIRECTORY|O_NOFOLLOW`로 openat 하거나, OS가 제공하면 `openat2()`의
`RESOLVE_BENEATH|RESOLVE_NO_SYMLINKS`를 사용한다. root가 filesystem anchor라 parent
fd를 신뢰할 수 없다면 배포 시 owner/mode가 검증된 상위 root부터 chain을 시작한다.
모든 검증은 `os.fstat(fd)`로 통일하고 root-symlink 실제 filesystem 테스트를 수행한다.

### M4-04 — container build/evidence job이 clean production 증거를 생산하지 못한다

근거:

- Dockerfile은 `# syntax=docker/dockerfile:1.7`을 고정하면서
  `COPY --exclude=...`를 사용한다(Design.md:2481-2499). `COPY --exclude`는 더 최신
  Dockerfile frontend 기능이므로 이 syntax pin으로는 parser 단계에서 실패한다.
  source exclusion을 build context/별도 package manifest/stage별 명시 COPY로 구현하거나
  지원 frontend 버전을 고정해야 한다.
- CI의 evidence 작성 명령은 `write_evidence()`의 필수 계약인 `command`, timestamps,
  `profile`, `run_id`, `result_artifact_path`, `candidate_root`를 제공하지 않는다
  (Design.md:2679-2691 vs 2875-2879). 별도 CLI가 이 값을 만드는 규칙도 정의되지 않았다.
- upload 단계는 `container.json` 한 파일만 업로드한다(Design.md:2692-2701). 그러나
  `container.json`은 별도 result artifact를 가리키고 assemble은 그 artifact까지
  copy+rehash해야 한다(Design.md:3096-3103). inbox에 result artifact가 없으므로 정상
  assemble은 항상 실패한다.
- `if: always()`라 앞선 image build가 실패하면 `docker inspect qna-rag:ci` 자체가
  실패해 evidence 작성 step도 중단된다. 실패 evidence조차 남긴다는 `always()`의
  목적을 달성하지 못한다.

영향: M4-REQ-009.1/.4와 M4-REQ-010.2. clean CI에서 production image 및 검증 가능한
container evidence를 만들 수 없다.

수정안: 실제 지원 syntax로 Dockerfile을 고정하고 clean builder spike를 CI에 먼저
추가한다. container result JSON과 evidence JSON을 같은 artifact root에 만들고 둘 모두
업로드한다. write CLI의 완전한 인자/schema를 설계하며, image 미생성 시 digest를 nullable
failure field로 기록할 수 있게 해 실패 evidence도 항상 생성한다.

### M4-05 — container/local run binding과 13-gate self-test가 동시에 성립하지 않는다

근거:

- 모든 evidence의 `run_id`는 최종 local `m4_gate --run-id`와 같아야 한다
  (Design.md:3015-3019). 그러나 container evidence는 선행 GitHub CI run에서 생성되고,
  이후 local Phase 7이 새 UUID를 만든다(Design.md:3196 이후). CI producer가 미래 local
  UUID를 알 경로가 없고 assemble은 copy+rehash만 하며 binding을 재발행하지 않는다.
- container artifact 이름에는 GitHub `run_id`를 쓰면서 evidence 공통 `run_id`는 local
  `EVIDENCE_RUN_ID`를 요구한다. `ci_run_id`와 evidence `run_id`라는 두 namespace를
  분리했다고 설명하지만 producer/consumer 변환 계약이 없다.
- `GATE_DAG`는 container 제외 13개이며 그중 하나가 `live_smoke`다
  (Design.md:3130-3147). self-test는 `live_mode="skip"`으로 실행하고도 evidence 13개가
  모두 생성된다고 요구한다(Design.md:3181-3184). skip 계약상 live evidence는 만들지
  않아야 하므로 가능한 수는 12개다.
- “`--live-mode {run,import,skip}`”는 import source를 별도 인자로 정의하지 않았고,
  의사 시그니처도 `live_mode: str`만 받는다(Design.md:3108-3120, 3149-3152). 그런데
  self-test는 `--live-mode import <dir>`를 하나의 모드처럼 사용한다. 실행 가능한 CLI
  grammar가 확정되지 않았다.
- static regression에서 orchestration self-test를 제외하고 “일반 CI unit test 단계”에서
  별도 실행한다고 했지만 현재 제시된 CI 변경 블록에는 그 별도 step이 없다
  (Design.md:2957-2967). 14-gate final runner 밖에서 필수 runner self-test가 실제로
  실행됐다는 evidence도 없다.

영향: M4-REQ-010.2/.4. 정상 실행으로 14개 PASS를 만드는 경로가 존재하지 않는다.

수정안: container의 immutable producer binding(`ci_run_id`)과 candidate assembly
binding(`run_id`)을 명시적으로 분리하고, assemble이 원본 attestation/hash를 검증한 뒤
새 candidate-bound wrapper/result를 원자적으로 발행하게 한다. 또는 Phase 7 전체를 같은
GitHub run 안에서 실행해 하나의 run ID를 공유한다. self-test는 skip이면 12개를 기대하고
live import/run 전용 테스트에서 13개를 기대하도록 나누며, import source 전용 CLI 인자를
정의한다. self-test 자체의 CI/evidence root도 gate 목록에 연결한다.

### M4-06 — “30-field complete inventory” gate는 실제 53-field dataclass와 즉시 충돌한다

근거:

- Design.md:226-294의 `Settings` annotation을 기계적으로 세면 **53개**다. 문서는 같은
  dataclass가 완전 인벤토리라고 선언하면서 `dataclasses.fields(Settings)`가 **30개**인지
  assert한다고 요구한다(Design.md:370-407). 제안 테스트는 구현 직후 반드시 실패한다.
- §4.3 주석은 “정확한 전체 목록/type/default/검증은 §4.3b 인벤토리 표가 단일 원본”이라고
  하지만 §4.3b에는 별도 필드 표가 없고 다시 dataclass가 단일 원본이라고 한다
  (Design.md:258-264, 370-384). 구현자가 default/env 이름/validator를 복원할 기계적
  mapping이 없다.
- 53개 중 검증 규칙은 소수 운영 필드와 upstream 합만 다룬다. retrieval K 관계,
  `0<=mmr_lambda<=1`, chunk overlap/size 관계, timeout/region, tuple parsing 등의
  유효 범위가 확정되지 않아 “typed/validated settings” 요구를 충족하지 못한다.
- 기존 호환 facade가 53개 필드를 언제 평가하고, CLI가 import 후 `os.environ`을 변경하는
  현재 패턴과 전역 cached settings가 어떻게 공존하는지도 정의되지 않았다. import 시
  config 상수가 먼저 materialize되면 CLI precedence는 깨진다.
- Prometheus spike 방향은 타당하다. `disable_created_metrics()` non-env 호출은 direct
  environ gate와 충돌하지 않는다. 이 부분은 M3-07의 폐쇄로 인정한다.

영향: M4-REQ-001과 settings gate. Phase 1 acceptance가 자기모순이라 구현 및 CI 통과가
불가능하다.

수정안: 필드 count를 생성된 schema/명시적 `ENV_NAME -> field/type/default/validator`
mapping에서 도출하고 magic number를 제거하거나 정확한 53으로 고정한다. 모든 numeric 및
cross-field invariant를 표로 확정한다. CLI override는 `Settings.load(overrides=...)`처럼
환경변수 mutation 없이 명시 전달하고, facade materialization 시점을 테스트로 고정한다.

## 5. MINOR 발견사항

### m4-01 — `RetrievalTrace("total")` 보존 계약이 문서와 테스트에서 빠져 있다

현재 trace는 6개 substage 외에 마지막 `RetrievalStageTrace(name="total", ...)`도
추가한다(`rag_engine.py:569-572`). 개정 helper는 6개 substage를 정확히 보존하지만,
상위 retrieval latency가 “`RetrievalTrace(name="total")`을 재사용”한다고 잘못된 타입
이름으로만 서술되고(Design.md:634-638), trace의 total append와 정확 값 보존 테스트가
없다. M3 평가 schema의 “exact preservation”을 주장하려면 기존 total entry까지 같은
순서/값 규칙으로 유지하는 테스트를 추가해야 한다.

### m4-02 — `_open_contained()`는 성공 시 중간 directory fd를 누수한다

`opened`에 모든 fd를 넣지만 성공 시 최종 fd만 반환하고 중간 fd를 닫지 않는다
(Design.md:2841-2856). 반복 gate write/read마다 path depth만큼 descriptor가 누적된다.
최종 fd를 제외한 중간 fd는 성공/실패 모두 `finally`에서 닫고, 빈 path도 거부해야 한다.

### m4-03 — scanner canonicalization 설명의 두 비교 형태가 반대로 기술됐다

실제 image member가 `app/runtime/documents/x`일 때 저장소 상대 forbidden prefix와
비교하려면 member에서 `workdir_prefix`를 **제거한 형태**도 만들어야 한다. 설계는
“정규화된 이름”과 “workdir prefix + 정규화된 이름”을 비교한다고 써서 이미 `app/`이
붙은 member에 다시 `app/`을 붙이는 형태가 된다(Design.md:2723-2734). unit positive
control이 요구하는 `app/runtime/documents/secret.txt`가 `runtime/documents/`에 걸리려면
prefix strip 규칙을 명시해야 한다.

## 6. 신규 모순 교차 점검

| 점검 항목 | 판정 | 근거 |
|---|---|---|
| ownership token race | **FAIL** | callback 등록 실패/loop shutdown 경계 미소유(M4-01) |
| request-scoped ChatOllama/OllamaLLM | **PASS(생성 seam)** | 0.3.10 introspection으로 재생성·binding 확인 |
| Ollama overall worker deadline | **FAIL** | trickle path를 스스로 보장 밖으로 둠(M4-02) |
| DDGS subprocess worker deadline | **PASS(설계 seam)** | 9.14.4 내부 wait=True 문제 및 spawn/terminate/kill 경계 확인 |
| exact RetrievalTrace | **부분 PASS** | 3필드/6 substage는 맞으나 total 회귀 검증 누락(m4-01) |
| openat provenance | **FAIL** | approved root 최초 open이 symlink follow(M4-03) |
| final production Docker stage | **PASS** | 마지막 stage가 production이고 prod 명령도 target 명시 |
| PEP 517 inputs | **PASS** | `requirements.txt`, pyproject, README, LICENSE, src COPY 명시 |
| Docker frontend/CI evidence | **FAIL** | syntax-feature 및 업로드/binding 불일치(M4-04/M4-05) |
| 14-gate roots/assemble/self-test | **FAIL** | producer/local run binding과 skip=13 모순(M4-05) |
| Settings complete inventory | **FAIL** | 선언 53 vs acceptance 30(M4-06) |
| Prometheus API | **PASS(설계 방향)** | public non-env disable API 확인; lock install 후 재증명 필요 |

## 7. 점수와 Gate 판정

| 평가 축 | 배점 | 점수 | 판정 |
|---|---:|---:|---|
| 요구사항 정합성 | 2.0 | 0.7 | overall timeout/settings gate 위반 |
| concurrency/lifecycle 정확성 | 2.0 | 0.7 | future ownership/finalizer 완료성 미증명 |
| 보안/provenance | 1.5 | 0.8 | root symlink 경계 미폐쇄 |
| container/재현성 | 1.5 | 0.7 | build syntax와 evidence artifact 불완전 |
| 자동 gate/추적성 | 1.5 | 0.4 | 14-gate 정상 PASS 경로 없음 |
| 관측성/API 구체성 | 1.5 | 0.9 | API spike와 sink는 개선, trace/settings 잔여 모순 |
| **합계** | **10.0** | **4.2** | **FAIL** |

**최종 Gate: FAIL. 구현 단계 진입 불가.**

## 8. 조건부 연장 자격 판정(기본 4회 종료)

| guide 조건 | 결과 | 근거 |
|---|---|---|
| CRITICAL 0 | 충족 | 0건 |
| 점수 9.0 이상 | **불충족** | 4.2 |
| MAJOR 2건 이하 | **불충족** | 6건 |
| 이전보다 실질 개선 | 일부 충족 | spike, sink, Docker final stage, openat 방향은 개선됐으나 핵심 Gate는 여전히 실패 |
| 잔여 문제가 구체적·해결 가능 | 일부 충족 | 대부분 구체적이나 Ollama overall deadline은 architecture/요구사항 선택 필요 |

**조건부 Iteration 5 연장 자격: 없음.** guide의 기본 4회 제한을 적용하면 현재
설계 사이클은 중단해야 한다. 재개하려면 orchestration 규칙 자체의 새 승인 또는
요구사항/architecture 결정을 먼저 받아야 한다. 특히 Ollama overall worker deadline을
process boundary로 구현할지, 필수 요구에서 완화할지 결정이 필요하다.

## 9. 검증 기록

- `langchain-ollama==0.3.10`, `ollama==0.6.0`, `ddgs==9.14.4` local venv
  introspection: 완료.
- M3 승인 baseline의 두 SHA-256과 Design.md 상수: 일치.
- Design.md Settings annotation 기계 계수: 53.
- 현재 Docker client/buildx는 존재하지만 daemon이 실행 중이지 않아 실제 image build는
  수행하지 못했다. 이 리뷰는 구현 전 설계 리뷰이며, Docker syntax/CI wiring은 정적
  계약 대조로 판정했다.
- 문서 링크 검사 및 `git diff --check`: 아래 최종 실행 결과를 기준으로 확인.

