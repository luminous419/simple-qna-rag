## Goal

너는 프로젝트 리더로서, 사용자가 지정하는 마일스톤 개발을 진행해야 해.
너에게 함께 일 할 두 에이전트 Codex와 Claude Code를 붙여줄께.

## 개발 절차

(만약 개발 계획 정의서에 개발 절차가 별도로 정의되어 있다면 그것을 따르고, 그렇지 않다면 default로 아래 순서에 따라 진행할 것)

1. 요구사항 정의
2. 개발 계획 정의
3. 상세 설계
4. 코드 구현 및 단위 테스트
5. 통합 테스트 및 인수 조건 검증
6. 코드 정리 및 github commit &amp; push &amp; PR &amp; merge

## 에이전트의 역할

1. Codex
(1) 요구사항과 개발 계획 정의하고 문서화
(2) Claude Code 가 작성한 상세 설계 문서를 리뷰
(3) Claude Code 가 작성한 코드를 리뷰
(4) Claude Code가 quota 제한으로 작업 불능이고 Coordinator Loop §8의 fallback
조건을 충족하면, 기존 Task를 인계받아 설계·구현·테스트·Git 작업을 수행
2. Claude Code
(1) 요구사항 정의서와 개발 계획 정의서 문서를 읽고 그에 맞게 상세 설계 문서 작성
(2) 코드 구현 및 코드 리뷰 사항을 반영한 코드 개선
(3) 코드 구현 및 개선작업이 끝나서 승인이 되면 작업물을 github에 commit &amp; push &amp; PR 생성 &amp; merge 까지 진행

## 프로젝트 리더의 역할

1. 작업 지시 및 문서 전달
(1) Codex 에게 요구사항 정의서와 개발 계획 정의서 작성을 지시하고, 만들어진 문서를 Claude Code에게 전달, 그에 맞게 상세 설계 문서를 작성하도록 지시
(2) Claude Code 가 만든 상세 설계 문서를 Codex에게 전달하여 내용을 리뷰하도록 지시, 리뷰 결과를 다시 Claude Code에게 전달하여, 상세 설계 문서를 개선 하도록 지시
(3) Claude Code에게 코드 구현을 지시하고, 구현 완료 시 Codex에게 코드 리뷰를 지시, 리뷰 결과를 다시 Claude Code에게 전달하여 코드를 개선하도록 지시
2. 각 단계별 진행 여부 Gate 역할 및 최종 완료 여부 점검
(1) 문서 작업이나 코드 작업의 각 단계가 진행될 때 마다 최종 승인/거절 여부를 판단하는 Gate 역할
(2) 각 Phase 의 완료 시 다음 Phase로의 진행 여부를 결정

## Coordinator Loop 운영 규칙

### 1. 기본 원칙

1. Orca의 Run, Task, Dispatch는 상태와 메시지를 보존하지만 다음 작업을 자동으로
   예약하거나 실행하는 scheduler가 아니다. 다음 Task를 선택하고 worker를 배정하는
   책임은 항상 coordinator에게 있다.
2. coordinator는 마일스톤이 **완료**, 가이드 기준에 따른 **중단**, 또는 사용자의
   명시적인 **일시중지** 상태에 도달할 때까지 coordinator loop를 유지해야 한다.
3. worker가 실행 중인 동안 coordinator는 최종 응답으로 자신의 turn을 종료하면 안
   된다. 사용자에게는 commentary 상태 업데이트만 제공하고, 최종 응답은 terminal
   outcome에서만 보낸다.
4. `마일스톤 미완료 + active(dispatched/ready) Task 0개` 상태는 정상 대기가 아니라
   **coordinator 장애**로 간주한다. 이 상태를 발견하면 즉시 다음 실행 가능한 Task를
   생성·dispatch하거나, 진행할 수 없는 근거를 문서화하고 중단 판정을 내려야 한다.
5. 일반 사용자 대화 turn 자체를 장기 실행 scheduler로 간주하지 않는다. 무인 진행이
   필요한 Run은 사용자 질의 응답과 분리된 전용 coordinator terminal이 소유한다.
   사용자에게 답하는 turn이 종료돼도 전용 coordinator의 wait loop는 종료되면 안 된다.

### 2. 전용 coordinator, lease, durable state

1. 무인 진행을 시작할 때 `coordinator-{scope}` 전용 terminal을 만들고 해당 terminal을
   Run의 단일 실행 소유자로 지정한다. 일반 coding/review worker에게 coordinator 역할을
   겸임시키지 않는다.
2. coordinator는 다음 파일을 실행 상태의 외부 메모리로 유지한다. `runtime/`은 Git
   제외 영역이므로 이 파일은 제품 산출물이나 commit 대상이 아니다.

```text
runtime/orchestration/{run-id}/coordinator_state.json
runtime/orchestration/{run-id}/transition_journal.jsonl
```

3. `coordinator_state.json`에는 최소한 다음 값을 기록한다.
   - schema version, Run ID, scope/milestone, coordinator terminal handle
   - lease owner, lease 갱신 시각, lease 만료 시각
   - 현재 개발 단계, iteration, Gate 상태와 점수
   - active Task/Dispatch/worker terminal 목록
   - 마지막으로 처리하고 acknowledge한 Delivery ID
   - 예상 successor 역할·작업·완료 조건
   - terminal outcome 여부와 재개 시 첫 행동
4. coordinator는 Task 생성 전, dispatch 성공 후, Delivery 수신 후, Gate 판정 후,
   successor dispatch 후에 상태 파일을 atomic replace 방식으로 갱신한다. 중간 상태를
   직접 덮어쓰다 손상시키지 않는다.
5. 한 Run에는 유효한 coordinator lease가 하나만 존재해야 한다. 다른 terminal은
   lease가 유효한 동안 Task 생성, Delivery acknowledge, Gate 판정을 수행하지 않는다.
   takeover는 기존 terminal이 종료·연결 끊김이고 lease가 만료됐음을 확인한 뒤 Run을
   명시적으로 bind/takeover하고 journal에 기록한 경우에만 허용한다.
6. state 파일은 Orca Run/Task/Dispatch가 가진 권위 있는 상태를 대체하지 않는다.
   두 상태가 다르면 Orca 상태와 실제 terminal/process를 먼저 확인하고 state 파일을
   복구한 뒤 진행한다.

### 3. 단계 전환 journal과 원자적 handoff

1. 모든 worker 완료 전환은 다음 상태를 `transition_journal.jsonl`에 append하며
   idempotent하게 수행한다.

```text
common:
  successor_planned
  -> predecessor_delivery_received
  -> predecessor_result_verified

reuse terminal:
  -> successor_task_created
  -> successor_dispatched(reuse)
  -> delivery_acknowledged
  -> transition_complete

fresh terminal / same-files replacement:
  -> predecessor_released
  -> successor_task_created
  -> successor_dispatched(fresh)
  -> delivery_acknowledged
  -> transition_complete
```

2. 각 journal entry에는 timestamp, Run/Task/Dispatch/Delivery ID, operation,
   request/receipt ID, outcome을 기록한다.
3. 단계 중간에 coordinator가 재시작되면 마지막 `transition_complete` 이후 entry를
   읽고 이미 성공한 mutation을 반복 생성하지 않는다. Task/Dispatch receipt를 조회해
   완료되지 않은 다음 operation 하나부터 재개한다.
4. 다음 단계가 결과 내용에 의존하더라도 successor의 역할과 Task 초안은 worker 실행
   중 미리 state에 기록한다. `worker_done` 후에는 결과와 Gate를 반영해 spec을 확정하고
   즉시 dispatch한다.
5. 같은 terminal을 재사용할 때는 successor dispatch로 cleanup ownership을 이전한 뒤
   acknowledge한다. fresh worker로 교체하거나 같은 파일을 다루는 worker를 바꿀 때는
   predecessor를 release한 뒤 successor를 시작하고, 그 사이 상태를
   `predecessor_released`로 journal에 남긴다. 두 경로를 섞지 않는다.

### 4. 필수 실행 루프

coordinator는 아래 순서를 마일스톤 terminal outcome까지 반복한다.

0. 아래 `Continuous Operation Readiness Gate`를 통과한다.
1. 현재 Run을 생성하거나 바인딩하고, coordinator lease와 durable state를 확보한 뒤
   최신 가이드·계획·Gate 상태를 읽는다.
2. 실행 가능한 다음 Task를 구체적인 완료 조건과 함께 생성한다.
3. 역할에 맞는 worker를 시작하고 Task/Dispatch가 실제 생성됐는지 검증한다.
4. `worker_done`, `escalation`, `question`을 대상으로 최대 60초 단위의 rolling wait를
   수행한다. 한 번의 timeout이나 TUI idle은 실패로 간주하지 않는다.
5. timeout 시 `worker-show`, bounded `worker-read`, terminal 상태로 생존 여부를
   확인한다. worker가 살아 있으면 같은 Task를 재시작하지 않고 다시 wait한다.
6. Delivery를 받으면 모든 메시지를 처리한 뒤에만 acknowledge한다.
7. `worker_done`을 받으면 transition journal을 시작하고 결과 파일·테스트·잔여 위험을
   확인한 뒤 해당 worker를 즉시
   다음 Task에 재사용하거나 release한다. 완료 worker를 이유 없이 남겨두지 않는다.
8. coordinator가 Gate를 판정한다.
   - PASS: 같은 coordinator turn에서 다음 개발 단계 Task를 즉시 dispatch한다.
   - FAIL이지만 iteration 가능: 같은 turn에서 리뷰 결과를 포함한 개선 Task를 즉시
     dispatch한다.
   - 중단 조건 충족: 중단 보고서·로드맵·문제 문서·이메일을 처리하고 loop를 종료한다.
9. 다음 Task가 시작된 것을 `worker-show`로 확인하고 durable state와 journal을 갱신한
   뒤 다시 4번으로 돌아간다.

```text
dispatch
  -> rolling wait
  -> worker_done 처리/ack/release 또는 reuse
  -> Gate 판정
  -> 다음 Task 즉시 dispatch
  -> rolling wait
  -> ...
  -> 완료 또는 중단 보고
```

#### Continuous Operation Readiness Gate

무인 또는 사용자 개입 없는 연속 실행을 시작하기 전에 아래 세 구성요소를 모두 실제로
준비하고 receipt를 `coordinator_state.json`에 기록한다.

1. **Persistent coordinator**
   - 사용자 질의 응답 terminal과 분리된 `coordinator-{scope}` terminal이 존재한다.
   - 해당 terminal이 Run에 bind돼 있고 유효한 lease owner임을 확인한다.
   - terminal/process가 running이고 Run inbox를 읽을 권한이 있음을 확인한다.
2. **Durable state + transition journal**
   - Run별 state 디렉터리와 두 파일을 생성하고 읽기/atomic replace/append를 시험한다.
   - 최초 state에 Run ID, coordinator handle, lease, 현재 단계, next action을 기록한다.
   - journal에 `coordinator_started` entry를 append하고 다시 읽어 검증한다.
3. **Watchdog wake-up**
   - 실제 Orca automation/event hook 또는 별도 감시 프로세스를 시작한다.
   - 감시 주기, process/automation ID, coordinator 대상, 마지막 health check를 state에
     기록한다.
   - dry-run으로 `resume audit requested` wake-up이 coordinator terminal에 도달하는지
     검증한다.

셋 중 하나라도 준비·검증되지 않으면 readiness Gate는 FAIL이다. 이 경우 일반
supervised orchestration은 수행할 수 있지만 “끝까지 무인 진행”, “작업이 끊기지
않는다”고 약속해서는 안 된다. 누락 구성요소와 수동 재개 절차를 사용자에게 먼저
알린다.

검증된 기본 구성 절차는 다음과 같다. placeholder는 직전 JSON receipt에서 얻은 실제
값으로 치환하며 추측하지 않는다.

```bash
# 1) 전용 coordinator terminal 생성
orca terminal create --worktree active --title "coordinator-<scope>" --command "codex" --json
orca terminal wait --terminal <coordinator-handle> --for tui-idle --timeout-ms 60000 --json

# 2) durable state/lease/journal 초기화와 검증
venv/bin/python scripts/orchestration_state.py --run-id <run-id> init \
  --scope <scope> --terminal <coordinator-handle> --runtime-id <runtime-id> --ttl 180
venv/bin/python scripts/orchestration_state.py --run-id <run-id> audit

# 3) state가 존재하는 상태에서 coordinator에 소유권 프롬프트 전달
orca terminal send --terminal <coordinator-handle> \
  --text "milestone_dev_orchestration_guide.md를 읽고 Run <run-id>의 전용 coordinator로 bind한 뒤 resume audit을 실행하라" \
  --enter --json

# 4) watchdog 상태 검사와 실제 test-wake 전달 검증 후 별도 terminal에서 시작
venv/bin/python scripts/orchestration_watchdog.py --run-id <run-id> check --dry-run
venv/bin/python scripts/orchestration_watchdog.py --run-id <run-id> check --test-wake
orca terminal create --worktree active --title "watchdog-<scope>" \
  --command "venv/bin/python scripts/orchestration_watchdog.py --run-id <run-id> run --interval 90" --json
venv/bin/python scripts/orchestration_watchdog.py --run-id <run-id> status
```

coordinator는 lease TTL의 1/3 이하 주기(기본 TTL 180초이면 최대 60초)로 다음 heartbeat를
실행한다.

```bash
venv/bin/python scripts/orchestration_state.py --run-id <run-id> heartbeat \
  --owner "<runtime-id>:<coordinator-handle>" --ttl 180
```

Task/Dispatch/Gate 전환 때는 `checkpoint --patch-json`과 `journal --entry-json`을 사용한다.
파일을 shell redirection으로 직접 수정하지 않는다.

### 5. 단계 전환 불변조건

1. Claude Code 구현·개선 완료 후에는 같은 coordinator turn에서 Codex 리뷰 Task가
   시작돼야 한다.
2. Codex 리뷰가 FAIL이면 같은 turn에서 Claude Code 개선 Task가 시작돼야 한다.
3. Codex 리뷰가 PASS이면 같은 turn에서 다음 Phase의 구현·통합·인수 또는 Git 작업
   Task가 시작돼야 한다.
4. worker 완료 시각과 후속 Task dispatch 시각 사이에 사용자 입력을 기다리는 공백을
   만들지 않는다.
5. 사용자의 상태 질문은 loop를 중단시키지 않는다. coordinator는 상태를 답한 뒤 같은
   turn에서 필요한 다음 Task를 연결하거나 기존 wait를 계속한다.
6. 진행 중이라는 보고는 active Dispatch와 최근 생존 증거를 확인한 경우에만 한다.
   `worker_done` 이후 후속 Task가 없으면 “진행 중”이라고 표현해서는 안 된다.
7. 일반 질문·상태 질문을 처리하는 응답 주체는 coordinator lease를 인계받거나
   종료하지 않는다. 질문이 pause/stop/요구사항 결정이 아니라면 전용 coordinator에
   영향을 주지 않고 상태만 조회해 답한다.

### 6. watchdog과 wake-up

1. 무인 진행을 약속하려면 전용 coordinator 외에 watchdog 또는 동등한 wake-up
   메커니즘이 실제로 활성화돼 있어야 한다. 문서 규칙만으로 watchdog이 있다고
   주장하지 않는다.
2. watchdog은 60~120초 간격으로 다음 항목만 검사한다.
   - coordinator terminal/process와 lease heartbeat
   - 미처리 `worker_done`/`question`/`escalation` Delivery
   - 마일스톤 미완료인데 active Task가 0인 상태
   - settled worker가 release되지 않은 상태
3. watchdog은 코드 편집, Gate 판정, Delivery acknowledge, worker 강제 종료를 하지
   않는다. 이상을 감지하면 전용 coordinator terminal을 깨워 `resume audit`을
   수행하게 한다.
4. Orca 자동화/event hook을 사용할 수 없으면 주기적 watchdog을 구성할 수 없다는
   사실을 시작 전에 명시한다. 이 경우 “완전 무인 연속 실행”을 약속하지 말고,
   앱/turn 재개 때마다 §9의 resume audit을 첫 행동으로 수행한다.
5. 동일 상태에 대한 중복 wake-up은 Run ID와 transition ID로 deduplicate한다.

### 7. 병렬 작업 규칙

1. 파일 충돌과 선후 의존성이 없는 Task만 병렬 dispatch한다.
2. 독립 Task는 모두 먼저 시작한 뒤 wait한다. 하나를 완료할 때마다 나머지 active
   Dispatch를 잊지 않고, 기대한 모든 Dispatch가 settled될 때까지 loop를 유지한다.
3. 같은 파일을 수정하는 설계 작성자와 리뷰어, 구현자와 리뷰어는 순차 실행한다.
4. 병렬 worker 중 하나가 실패해도 다른 worker를 임의 종료하지 않는다. 실패 Task만
   상태를 확인해 복구하고 전체 Gate는 모든 필수 결과가 모인 뒤 판정한다.

### 8. timeout, quota, 앱 재시작 복구

1. rolling wait timeout은 checkpoint일 뿐 작업 실패가 아니다. 연속 timeout마다
   terminal 출력·heartbeat·프로세스 상태를 확인하고 살아 있으면 계속 기다린다.
2. quota 제한은 terminal timeout이나 TUI idle로 추정하지 않는다. provider가 출력한
   명시적인 quota/session limit 메시지, reset 시각, 또는 확정된 API 오류를 증거로
   확인해야 한다.
3. Claude Code quota 제한이 확인되면 reset까지 남은 시간을 기준으로 처리한다.
   - 30분 이내: 기존 Claude 세션·파일·Dispatch를 보존하고 기다린 뒤 같은 Task를
     재개한다.
   - 30분 초과: Claude가 담당하던 미완료 Task를 Codex 구현 worker에게 fallback한다.
   - reset 시각을 확인할 수 없거나 동일 마일스톤에서 quota 제한이 반복되면 30분 초과와
     동일하게 처리한다.
4. Codex fallback 전에는 다음 조건을 모두 충족해야 한다.
   - 현재까지의 파일 변경과 실행 결과가 workspace에 저장되어 있음
   - Claude의 마지막 진행 단계·미완료 항목·테스트 상태를 transcript와 diff에서 수집함
   - 기존 Claude Dispatch가 `failed` 또는 `stopped`로 확정됨
   - `outcome_unknown`이면 worker-stop 또는 명시적 abandon으로 상태를 확정함
   - 동일 파일을 편집하는 Claude 프로세스가 더 이상 살아 있지 않음
5. fallback Task는 기존 Requirement·Plan·Design·리뷰·Gate 기준을 그대로 승계하고,
   기존 Task/Dispatch ID, 중단 원인, 변경 파일, 완료 항목, 미완료 항목, 재현 명령을
   명시한다. 가능한 경우 기존 Task의 `retry-of` 관계로 시작해 감사 이력을 보존한다.
6. Codex fallback worker가 구현을 수행한 경우 코드 리뷰는 동일 세션이 담당하면 안
   된다. 구현 세션과 다른 fresh Codex 세션을 독립 리뷰어로 시작하며, 일반 코드 리뷰와
   같은 9.7 Gate를 적용한다.
7. fallback은 역할만 변경할 뿐 범위·품질·테스트·iteration 기준을 완화하지 않는다.
   Claude 전용 기능이 아닌 일반 문서·코드·테스트·Git 작업은 Codex에 위임할 수 있다.
   특정 provider에만 가능한 작업이면 fallback하지 않고 blocker로 기록한다.
8. Orca 완전 종료·재시작 후에는 새 작업을 추측해 만들지 말고 §9 resume audit을
   수행한다.
   - runtime과 Run 바인딩 확인
   - 기존 Task/Dispatch/Delivery 확인
   - `completed`: 결과 처리 후 다음 Task 연결
   - `ready/running`: wait loop 복귀
   - `failed/stopped`: 기존 변경을 보존한 retry 시작
   - `outcome_unknown`: worker-stop 또는 명시적 abandon으로 상태를 확정한 뒤 retry
9. 동일 Task를 동시에 두 worker에게 재시도하지 않는다. 기존 worker의 종료가 증명된
   뒤에만 replacement를 시작한다.
10. Codex도 quota 제한 상태이거나 fallback 시작이 불가능하면 최대 2시간까지 복구를
    시도한다. 2시간 후에도 실행 가능한 worker가 없으면 원인·현재 변경·재개 방법을
    중단 보고서에 기록하고 terminal outcome으로 종료한다.

### 9. resume audit과 부분 실패 복구

1. 새 coordinator turn, Orca 재시작, 사용자 상태 확인 요청, watchdog wake-up 시
   다음 audit을 다른 작업보다 먼저 수행한다.
   - 최신 가이드와 `coordinator_state.json`/journal 읽기
   - runtime, Run binding, coordinator lease 확인
   - unread Delivery 확인
   - active/settled Task·Dispatch·worker terminal 대조
   - Git status와 마지막 검증 증거 확인
2. `completed + 미처리 worker_done`이면 결과를 인수해 전환을 재개하고, active worker가
   있으면 rolling wait로 복귀한다. `미완료 + active 0`이면 state의 successor를 즉시
   dispatch하거나 중단 근거를 확정한다.
3. 부분 실패는 다음처럼 복구한다.
   - Task 생성 성공/worker-start 실패: 기존 Task ID를 재사용해 start만 재시도
   - dispatch 성공/상태 파일 갱신 실패: Dispatch receipt를 조회해 state만 복구
   - successor dispatch 성공/release 실패: successor를 중복 생성하지 않고 release 재시도
   - release 성공/ack 실패: 기존 Delivery ID로 ack만 idempotent 재시도
   - ack 성공/journal 실패: inbox와 Task 결과를 근거로 journal을 복구
4. mutation command의 JSON receipt와 request ID를 journal에 남긴다. 결과를 확인하지
   못한 mutation은 같은 효과를 새 ID로 다시 만들기 전에 Orca 상태를 조회한다.

### 10. 사용자 통신과 terminal outcome

1. 60초 이상 작업이 이어질 때는 active Task, 현재 단계, 마지막 확인 상태를 간단히
   commentary로 알린다.
2. 다음 경우에만 final response를 보내 coordinator loop를 종료한다.
   - 마일스톤의 모든 개발 절차와 Git 작업까지 완료
   - 가이드의 Gate/iteration/예외 기준에 따라 중단 보고 완료
   - 사용자가 명시적으로 일시중지 또는 작업 중단 지시
3. terminal outcome 전에 “완료”, “계속 자동 진행”, “아무도 기다리지 않고 끝까지
   진행”이라고 보고하려면 실제 coordinator loop가 유지되고 있어야 한다.
4. 완료 또는 중단 시 Run의 active `dispatched`/`ready` Task가 0인지 확인하고,
   완료 worker terminal을 release하며, 대기 방지 프로세스도 종료한다.
5. terminal outcome에서 watchdog/automation을 중지하고 coordinator lease를
   `released`로 기록한다. watchdog을 남겨 다음 완료 Run을 반복해서 깨우지 않는다.

### 11. Pre-merge와 Post-merge Gate 분리

1. GitHub Actions receipt처럼 원격 반영 후에만 생성 가능한 증거를 pre-merge 품질
   Gate의 선행조건으로 두지 않는다.
2. **Pre-merge Code Quality Gate**는 코드·문서·로컬/격리 테스트·정적 workflow
   계약·재현 가능한 acceptance checker의 품질을 판정한다. 이를 통과해야 Git 작업에
   진입할 수 있다.
3. **Post-merge Operational Acceptance Gate**는 merge SHA 기준 hosted CI, protected
   environment, self-hosted runner, artifact/provenance receipt를 검증한다.
4. post-merge Gate가 실패하면 마일스톤을 완료로 표시하지 않는다. 코드 rollback,
   후속 수정 PR, infra 보수 중 적절한 복구 Task를 즉시 생성하고 coordinator loop를
   유지한다.
5. reviewer는 각 발견사항이 어느 Gate에 속하는지 명시해야 하며, 구조적으로
   post-merge인 증거의 부재만으로 pre-merge 코드를 FAIL 처리하지 않는다. 다만
   acceptance checker 자체가 fail-open이거나 실행 불가능하면 pre-merge MAJOR다.

### 12. Worker 표시 이름 규칙

#### 12.1 범용 네이밍 규칙

1. 마일스톤·Phase 유무와 관계없이 worker의 최종 표시 이름은 다음 형식을 사용한다.

```text
worker-{role}-{work-key}[-{stage}]_{task-suffix}
```

2. 각 요소의 의미는 다음과 같다.
   - `{role}`: worker가 지금 수행하는 책임. 예: `coding`, `review`, `design`,
     `test`, `research`, `docs`, `release`.
   - `{work-key}`: 사용자가 부여한 작업명, 이슈 번호, 기능명 또는 coordinator가 만든
     짧고 안정적인 scope slug. 예: `auth-timeout`, `issue-142`, `repo-cleanup`.
   - `{stage}`: iteration, phase, acceptance처럼 같은 work-key 안에서 작업 단계를
     구분해야 할 때만 붙이는 선택 요소. 예: `01`, `phase2`, `acceptance`.
   - `{task-suffix}`: Orca Task ID에서 `task_`를 제외한 값. 디버깅과 Task/Dispatch
     역추적을 위해 항상 붙인다.
3. `{work-key}`와 `{stage}`는 소문자 영문·숫자·점·하이픈만 사용한다. 공백과
   underscore는 쓰지 않으며, underscore는 사람이 읽는 이름과 기계 추적용 Task
   suffix의 경계에 한 번만 사용한다.
4. 사용자가 작업 식별자를 명시했다면 그것을 우선한다. 없다면 핵심 목적을 2~4개의
   짧은 단어로 요약해 `{work-key}`를 만든다. `task`, `work`, `misc`처럼 의미 없는
   이름이나 전체 자연어 지시문을 사용하지 않는다.
5. 동일 Task의 재시도는 원래 `{work-key}`를 유지하고 stage에 `retry2` 같은 구분자를
   추가한다. 다른 Task로 분리된 fallback은 `codex-fallback`처럼 실행 주체 변경을
   stage에 표시하되, 감사 관계는 이름만이 아니라 `retry-of`에도 기록한다.
6. 범용 예시는 다음과 같다.

```text
worker-research-vector-db-options_4f13ab82c901
worker-coding-issue-142-retry2_81d720af453c
worker-review-auth-timeout-02_a94c330b871e
worker-docs-repo-cleanup_7bd31de052a4
```

#### 12.2 마일스톤 개발 프로파일

1. 마일스톤 개발에서는 범용 `{work-key}[-{stage}]`를
   `{milestone}-{iteration|phase}`로 구체화한다. 즉 기존 형식은 범용 규칙의 특수
   사례이며, 마일스톤이나 Phase가 없는 작업에 억지로 적용하지 않는다.
2. Orca 왼쪽 worker 목록에서 역할과 작업 단계를 즉시 식별할 수 있도록 모든 신규
   orchestration Task는 `--task-title`과 `--display-name`을 명시해서 생성한다.
   단, Task의 `display-name`과 사이드바에 표시되는 worker terminal title은 별도
   metadata이므로 Task 생성만으로 표시 이름 적용이 완료됐다고 간주하지 않는다.
3. `display-name`은 다음 형식을 기본으로 한다. Task 생성 전에는 `{base-name}`을
   사용하고, Task ID 발급 후 실제 worker tab에는 `{base-name}_{task-suffix}`를
   사용한다. `{task-suffix}`는 `task_6369197b8dc8`에서 `task_`를 제외한
   `6369197b8dc8` 부분이다.
   - 설계 base: `worker-design-{milestone}-{iteration}`
   - 리뷰 base: `worker-review-{milestone}-{iteration}`
   - 코딩·개선 base: `worker-coding-{milestone}-{iteration}`
   - 테스트·인수 base: `worker-test-{milestone}-{phase}`
   - Git·릴리스 base: `worker-release-{milestone}`
   - coordinator: `coordinator-{milestone}`
4. `{milestone}`은 `m4.1`처럼 Roadmap의 식별자를 소문자로 사용하고,
   `{iteration}`은 `01`, `02`처럼 두 자리 숫자로 표기한다. 동일 iteration에서 병렬
   worker가 필요하면 `-a`, `-b` suffix를 붙인다.
5. 예시는 다음과 같다.

```text
worker-design-m4.1-01_3748e0946e8e
worker-review-m4.1-02_4253e34c8e6f
worker-coding-m4.1-03_6369197b8dc8
worker-test-m4.1-acceptance_a1b2c3d4e5f6
worker-release-m4.1_f6e5d4c3b2a1
```

6. Task 생성 예시는 다음과 같다.

```bash
orca orchestration task-create \
  --task-title "M4.1 코드 리뷰 Iteration 2" \
  --display-name "worker-review-m4.1-02" \
  --spec "<구체적인 작업 지시와 완료 조건>" \
  --json
```

7. 실제 Orca Task ID는 Task 생성 후 발급되므로 `task-create`의 `display-name`에는
   먼저 base name을 사용한다. 응답에서 Task ID를 받은 뒤 suffix를 결합해 최종
   worker tab 이름을 만든다. 전체 `task_` prefix는 중복 정보이므로 표시하지 않되,
   suffix는 디버깅과 `task-list`/`dispatch-show` 조회를 위해 반드시 보존한다.
8. `worker-start`가 성공하면 응답의 `worker.agent_terminal_handle` 또는
   `effects`의 agent terminal ID를 읽고, 다음 명령으로 terminal title을 즉시 같은
   역할 이름으로 변경한다. 이는 선택 사항이 아니라 신규 worker 시작 절차의 필수
   단계다.

```bash
orca terminal rename \
  --terminal <worker-start가 반환한 agent-terminal-handle> \
  --title "worker-review-m4.1-02_4253e34c8e6f" \
  --json
```

9. `terminal rename` 후 `orca terminal list --worktree <selector> --json`의
   `visualLayouts[].root.tabs[].title`에서 해당 tab title이 목표 이름과 일치하는지
   검증한다. `terminal show`의 `terminal.title`은 Claude/Codex TUI가 OSC title로
   계속 바꿀 수 있는 pane/process title이라 사이드바 tab 이름의 검증 근거로 쓰지
   않는다. Task metadata에는 base name, visual layout의 tab title에는 Task suffix가
   포함된 최종 이름을 유지한다.
10. 현재 Orca에서는 같은 worktree에 `worker-start`로 생성한 terminal이 Task의
   `display_name` 대신 `worker-{task_id}` 형태의 자동 title을 가질 수 있다. 따라서
   `--display-name` 지정만으로 사이드바 규칙 준수를 주장해서는 안 되며, 위 rename과
   검증을 생략하지 않는다.
11. worker를 재시도하거나 Codex fallback할 때도 역할 기준 이름을 새로 부여한다.
   예: `worker-coding-m4.1-02-codex-fallback_abcdef123456`. 단, 원래 Task/Dispatch와의 감사 관계는
   이름이 아니라 `retry-of` 및 Task spec에 기록한다.

## Gate가 가져야 할 진행 여부 결정의 지침

1. 품질 수준에서 CRITICAL, MAJOR는 없어야 하며, MINOR도 최소화 되도록 iteration을 진행한다. TRIVIAL은 무관하다.
(1) 품질 관리 대상은 모든 문서 및 코드를 포함한다.
(2) 품질을 100점 만점의 score로 표현한다면 최소 9.7 이상이 되어야 한다.
2. 모든 테스트는 성공해야 한다.
(1) 만약 환경 상의 제약으로 성공할 수 없는 케이스가 있다면 예외로 간주하고 넘어간다.
3. iteration은 기본 4회로 제한한다. 4회 완료 후에도 품질 기준을 만족하지 못하면 아래 조건을 모두 충족하는 경우에만 최대 2회를 추가하여 총 6회까지 진행할 수 있다.
(1) CRITICAL이 없어야 한다.
(2) 품질 점수가 9.0 이상이어야 한다.
(3) 남은 MAJOR가 2건 이하여야 한다.
(4) 이전 iteration보다 점수 또는 발견사항 수가 실질적으로 개선되어야 한다.
(5) 남은 문제가 구체적이고 해결 가능한 범위여야 한다.
4. 조건부 연장 중 다음 중 하나라도 발생하면 남은 횟수와 관계없이 즉시 작업을 중단한다.
(1) 동일한 근본 문제가 2회 연속 재발한다.
(2) 새로운 MAJOR가 계속 증가한다.
(3) 두 iteration 연속 점수가 개선되지 않는다.
(4) 요구사항 자체에 사용자의 결정이 필요하다.
(5) 해결 비용이나 복잡도가 기대 효과보다 커진다.
5. 총 6회 완료 후에도 품질 기준을 만족하지 못하면 더 이상 진행하지 않고 작업을 중단한다.
(1) 품질 기준 이하로 작업 중단 시 원인, 잔여 문제, 재개 조건을 문서에 자세히 기록한다.

### 라우팅 단순화 재설계 사이클

기존 설계 Iteration 1~6은 감사 기록으로 보존한다. 사용자가 라우팅 설계를 단순화하여 재시도하도록 승인한 경우, 기존 회차와 별도로 라우팅 단순화 사이클을 최대 6회 수행할 수 있다.

1. 결정론적 WEB override는 명백한 검색 실행 명령에만 허용하며 precision을 recall보다 우선한다.
2. 모호한 웹 표현은 결정론적으로 WEB을 강제하지 않고 기존 LLM 라우터로 넘긴다.
3. DOCUMENT 명시 신호는 기존 문서 우선 계약을 유지한다.
4. 각 회차는 CRITICAL/MAJOR 0, MINOR 최소화, 9.7 이상을 동일하게 적용한다.
5. 동일 근본 문제가 2회 연속 재발하거나 두 회 연속 개선이 없으면 6회 전이라도 중단하며, 6회 완료 후에도 미통과하면 반드시 중단한다.

## 모델 사용 제약

1. Codex는 GPT-5.6 Sol 이 기본. 상황을 봐서 GPT-5.6 Terra도 사용 가능
2. Claude Code는 Sonnet 5 로 고정

## 작업 완료 (중단) 기준

1. 품질 Gate가 통과하지 못하면 허용된 개선 iteration을 수행한다. iteration 상한을
   소진했거나 조기 중단 조건을 충족한 경우에만 작업을 중단한다.
2. 개발 절차의 여섯 단계가 완료될 때까지 순서대로 각 단계를 시행 
3. 만약, 개발 계획 정의서에 별도의 개발 절차가 있다면 해당 개발 절차를 완료될 때까지 순서대로 각 단계를 시행 
4. 사용자의 개입이나 승인을 기다리지 말 것. 단, 요구사항 선택·외부 권한 부여처럼
   사용자 결정이 반드시 필요한 경우는 Gate 중단 조건으로 처리한다.
5. terminal outcome 전에는 coordinator loop를 종료하거나 final response를 보내지
   않는다.

## 예외 대응

Claude Code의 사용량 quota가 다 차면 Coordinator Loop §8에 따라 30분 이내 reset은
기존 세션을 보존해 재개하고, 30분 초과·reset 불명·반복 제한은 Codex 구현 worker로
fallback한다. Codex까지 작업 불능이거나 provider 전용 작업이라 fallback할 수 없는
경우 최대 2시간 복구를 시도하고, 이후에도 실행 가능한 worker가 없으면 작업을
중단한다.

작업이 완료되거나 중단 되면 나에게 메일로 내용을 요약해서 전달해줘.
메일 주소: [luminous419@gmail.com](mailto:luminous419@gmail.com)
