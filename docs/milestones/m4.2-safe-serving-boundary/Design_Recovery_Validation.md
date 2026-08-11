# M4.2 Design Recovery Validation

대상: [Requirement](Requirement.md) · [Plan](Plan.md) · [Design](Design.md) ·
[Design Review Iteration 4](Design_Review_Iteration_4.md)
· [Design Recovery Review Iteration 1](Design_Recovery_Review_Iteration_1.md)
· [Design Recovery Review Iteration 2](Design_Recovery_Review_Iteration_2.md)

최초 실행일은 2026-08-10, Iteration 2/3 추가 실행일은 2026-08-11이며 작업 디렉터리는
repository root다. 이 문서는 M42-DR4-001/002 및 M42-RR1-001/002 복구 결정을 위해 수행한
bounded executable characterization을
기록한다. 아래 결과는 임시 inline prototype evidence이며 repository product code/test의 구현
또는 PASS receipt가 아니다. 실험 중 product code, test, config, lockfile은 변경하지 않았다.

## 1. 검토 범위와 설치 환경

다음을 편집 전에 읽고 대조했다.

- `Requirement.md`, `Plan.md`, 기존 `Design.md`, `Design_Review_Iteration_1.md`부터
  `Design_Review_Iteration_4.md`까지, repository root의
  `milestone_dev_orchestration_guide.md`
- 현재 `src/simple_qna_rag/observability/request_context.py`,
  `src/simple_qna_rag/settings.py`, `src/simple_qna_rag/web/server.py`
- 설치된 `starlette.middleware.base.BaseHTTPMiddleware` 0.50.0 source

환경/source 확인 command:

```bash
python - <<'PY'
import inspect, platform, starlette
from starlette.middleware import base
print(platform.python_version())
print(platform.platform())
print(starlette.__version__)
print(inspect.getsourcefile(base.BaseHTTPMiddleware))
print(inspect.getsource(base.BaseHTTPMiddleware))
PY
```

관측값:

```text
Python 3.11.8
macOS-26.5.2-arm64-arm-64bit
Starlette 0.50.0
/Users/luminous/program/anaconda/anaconda3/envs/common/lib/python3.11/site-packages/starlette/middleware/base.py
```

source에서 `call_next()`는 downstream 첫 frame을 memory receive stream에서 기다리고,
stream EOF이면서 downstream exception이 없으면
`RuntimeError("No response returned.")`를 발생시킨다.

## 2. 단일 bounded characterization command

세 prototype은 아래 한 command에서 각각 `asyncio.wait_for(..., 2.0)`으로 제한했다.

```bash
python - <<'PY'
import asyncio, json, platform, starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

scope = {"type":"http", "asgi":{"version":"3.0"}, "http_version":"1.1",
         "method":"POST", "scheme":"http", "path":"/rag", "raw_path":b"/rag",
         "query_string":b"", "root_path":"", "headers":[],
         "client":("127.0.0.1",1), "server":("test",80), "state":{}}
async def receive_disconnect(): return {"type":"http.disconnect"}
async def downstream_no_response(scope, receive, send): return None

class PassThrough(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next): return await call_next(request)

async def basehttp_case():
    sent=[]
    try: await PassThrough(downstream_no_response)(scope, receive_disconnect, sent.append)
    except Exception as exc:
        print(json.dumps({"basehttp_no_response":{"exception_type":type(exc).__name__,
              "message":str(exc),"frames":sent}}, sort_keys=True))

class PureRequestContext:
    def __init__(self, app, record): self.app,self.record=app,record
    async def __call__(self, scope, receive, send):
        starts=0; status=None; self.record.append(("start",scope["path"]))
        async def observed_send(message):
            nonlocal starts,status
            if message["type"]=="http.response.start": starts+=1; status=message["status"]
            await send(message)
        outcome="internal"; equivalent=500
        try:
            await self.app(scope, receive, observed_send)
            outcome,equivalent=("client_disconnected",499) if starts==0 else ("response",status)
        finally:
            self.record.append(("end",outcome,equivalent))
            self.record.append(("metric",outcome,equivalent))

async def pure_asgi_case():
    events=[]; frames=[]
    await PureRequestContext(downstream_no_response,events)(scope,receive_disconnect,frames.append)
    assert frames==[] and events.count(("end","client_disconnected",499))==1
    assert events.count(("metric","client_disconnected",499))==1
    print(json.dumps({"pure_asgi_no_response":{"exception":None,"frames":frames,
          "events":events,"end_count":1,"metric_count":1}}, sort_keys=True))

class Guard:
    def __init__(self): self.lock=asyncio.Lock(); self.owner=None; self.next=0
    async def acquire(self):
        async with self.lock:
            if self.owner is not None: raise RuntimeError("lifespan_already_active")
            self.next+=1; self.owner=self.next; return self.owner
    async def release(self, token):
        async with self.lock:
            if self.owner!=token: raise RuntimeError("lifespan_not_owner")
            self.owner=None

async def guard_case():
    g=Guard(); mutations={"settings_cache":0,"engine":0,"executor":0}
    first=await g.acquire()
    try: await g.acquire()
    except RuntimeError as exc: reject=str(exc)
    assert mutations=={"settings_cache":0,"engine":0,"executor":0}
    await g.release(first); failure_owner=g.owner
    second=await g.acquire(); await g.release(second); shutdown_owner=g.owner
    third=await g.acquire(); await g.release(third)
    print(json.dumps({"single_active_lifespan_guard":{"first_acquire":first,
          "concurrent_reject":reject,"mutations_before_reject":mutations,
          "failure_release_owner":failure_owner,"reacquire_after_failure":second,
          "shutdown_release_owner":shutdown_owner,"reacquire_after_shutdown":third,
          "final_owner":g.owner}}, sort_keys=True))

async def main():
    print(json.dumps({"environment":{"python":platform.python_version(),
          "starlette":starlette.__version__,"platform":platform.platform()}}, sort_keys=True))
    await asyncio.wait_for(basehttp_case(),2.0)
    await asyncio.wait_for(pure_asgi_case(),2.0)
    await asyncio.wait_for(guard_case(),2.0)
asyncio.run(main())
PY
```

## 3. Exact results

Command exit status는 0이었고 stdout은 다음과 같았다.

```json
{"environment": {"platform": "macOS-26.5.2-arm64-arm-64bit", "python": "3.11.8", "starlette": "0.50.0"}}
{"basehttp_no_response": {"exception_type": "RuntimeError", "frames": [], "message": "No response returned."}}
{"pure_asgi_no_response": {"end_count": 1, "events": [["start", "/rag"], ["end", "client_disconnected", 499], ["metric", "client_disconnected", 499]], "exception": null, "frames": [], "metric_count": 1}}
{"single_active_lifespan_guard": {"concurrent_reject": "lifespan_already_active", "failure_release_owner": null, "final_owner": null, "first_acquire": 1, "mutations_before_reject": {"engine": 0, "executor": 0, "settings_cache": 0}, "reacquire_after_failure": 2, "reacquire_after_shutdown": 3, "shutdown_release_owner": null}}
```

## 4. Evidence와 구현 acceptance의 경계

| 대상 | prototype이 증명한 것 | 구현 뒤 project test가 추가로 증명할 것 |
|---|---|---|
| M42-DR4-001 | 설치 `BaseHTTPMiddleware` frame-0 실패 재현; 최소 pure wrapper의 frame-0 정상 반환, `client_disconnected`/499-equivalent end·metric 각 1 | 실제 `create_app()` 전체 stack, request ID/start/end/duration/counter exactly once, queued/running 100 races, result/tie/outer cancel, pending task 0 |
| M42-DR4-002 | first acquire, concurrent deterministic reject, reject 전 global mutation 0, failure/shutdown release, 두 reacquire | 실제 settings/engine/executor factories, startup의 각 failure/cancellation, 정상 drain/grace expiry/drain error/shutdown cancellation, guard release exactly once |

`499`는 내부 분류와 log/metric status-equivalent다. disconnect 뒤 response frame은 0이어야 하며
HTTP 499 response를 보내지 않는다. single-active guard는 process-local app lifespan contract이고
다중 process deployment를 금지하지 않는다.

## 5. 복구 결론과 closure

- M42-DR4-001은 request-context 하나만 pure ASGI middleware로 교체하는 설계로 CLOSED다.
  route가 소유한 frame-0 disconnect terminal을 outer middleware가 예외/500으로 바꾸지 않으며,
  기존 request ID/log/metric owner는 exactly-once로 보존된다.
- M42-DR4-002는 process마다 active lifespan 정확히 하나만 허용하는 설계로 CLOSED다. concurrent
  second는 모든 global mutation 전에 실패하고 sole owner는 모든 startup/shutdown path에서 guard를
  놓는다. previous-cache generation/lease/rollback 복잡성은 제거한다.
- 이 closure는 design recovery closure다. Phase 2 이후 [Design §10](Design.md)의 actual project
  tests와 deterministic acceptance runner가 통과하기 전에는 구현 완료나 M4.2 PASS가 아니다.
- M4.1 Operational Acceptance는 계속 `M4.1_BLOCKED`이며 이 복구 evidence와 합성하지 않는다.

## 6. Iteration 2 immutable identity/teardown prototype — PROTOTYPE-ONLY

아래 command는 M42-RR1-001/002의 알고리즘 순서만 확인하는 bounded inline
**PROTOTYPE-ONLY** evidence다. repository product code/test/config를 import하거나 변경하지 않으며
project acceptance PASS를 뜻하지 않는다.

```bash
python - <<'PY'
import asyncio, json
class ProcessGuard:
 def __init__(self): self.owner=None; self.committed=None; self.cache=None; self.n=0
 def acquire(self):
  if self.owner is not None: raise RuntimeError('lifespan_already_active')
  self.n+=1; self.owner=self.n; return self.n
 def commit(self,s,c):
  if self.committed is None: self.committed=self.cache=s; c['cache_writes']+=1; return 'first'
  if self.committed is not s: raise RuntimeError('process_settings_identity_mismatch')
  return 'same'
 def release(self,x): assert self.owner==x; self.owner=None
def identity_case():
 g=ProcessGuard(); s=object(); d=object(); c={'loaders':0,'cache_writes':0,'engine':0,'executor':0,'app_state':0}
 a=g.acquire(); c['loaders']+=1; first=g.commit(s,c); c['engine']+=1; c['executor']+=1; c['app_state']+=1; g.release(a)
 b=g.acquire(); c['loaders']+=1; same=g.commit(s,c); c['engine']+=1; c['executor']+=1; c['app_state']+=1; g.release(b)
 before=c.copy(); x=g.acquire(); c['loaders']+=1
 try: g.commit(d,c)
 except RuntimeError as exc: rejected=str(exc)
 finally: g.release(x)
 assert g.cache is g.committed is s and c['cache_writes']==1
 assert (c['engine'],c['executor'],c['app_state'])==(before['engine'],before['executor'],before['app_state'])
 return {'first':first,'same':same,'different':rejected,'counts':c,'cache_is_committed':g.cache is g.committed,'released':g.owner is None}
class FakeExecutor:
 def __init__(self,fail=(),running=0,wait_result=True): self.fail=set(fail); self.running=running; self.wait_result=wait_result; self.trace=[]
 def begin_drain(self):
  self.trace.append('begin')
  if 'begin' in self.fail: raise RuntimeError('begin')
 async def wait_drained(self,timeout):
  self.trace.append('wait'); await asyncio.sleep(0)
  if 'wait' in self.fail: raise RuntimeError('wait')
  return self.wait_result
 def shutdown(self,*,wait,cancel_futures):
  assert (wait,cancel_futures)==(False,True); self.trace.append('shutdown')
  if 'shutdown' in self.fail: raise RuntimeError('shutdown')
 def snapshot(self): return {'running':self.running}
async def teardown(ex,cancel_wait=False):
 errors=[]; began=False
 try: ex.begin_drain(); began=True
 except BaseException as e: errors.append(type(e).__name__+':begin')
 if began:
  try:
   if cancel_wait: raise asyncio.CancelledError()
   await ex.wait_drained(1)
  except BaseException as e: errors.append(type(e).__name__+':wait')
 try: ex.shutdown(wait=False,cancel_futures=True)
 except BaseException as e: errors.append(type(e).__name__+':shutdown')
 ex.trace+=['STOPPED','release']
 return {'trace':ex.trace,'errors':errors,'released':True,'residual':ex.snapshot()}
async def main():
 rows={}
 for name,kw in {'normal_zero':dict(running=0),'normal_running':dict(running=1),'grace_expiry':dict(running=1,wait_result=False),'begin_error':dict(fail=('begin',)),'wait_error':dict(fail=('wait',)),'shutdown_error':dict(fail=('shutdown',)),'all_errors':dict(fail=('begin','wait','shutdown'))}.items(): rows[name]=await asyncio.wait_for(teardown(FakeExecutor(**kw)),1)
 rows['cancel_at_wait']=await asyncio.wait_for(teardown(FakeExecutor(running=1),cancel_wait=True),1)
 for row in rows.values(): assert row['trace'].count('begin')==1 and row['trace'].count('shutdown')==1 and row['trace'][-2:]==['STOPPED','release']
 print(json.dumps({'immutable_identity':identity_case()},sort_keys=True)); print(json.dumps({'teardown_matrix':rows},sort_keys=True))
asyncio.run(main())
PY
```

Exit status `0`; exact stdout:

```json
{"immutable_identity": {"cache_is_committed": true, "counts": {"app_state": 2, "cache_writes": 1, "engine": 2, "executor": 2, "loaders": 3}, "different": "process_settings_identity_mismatch", "first": "first", "released": true, "same": "same"}}
{"teardown_matrix": {"all_errors": {"errors": ["RuntimeError:begin", "RuntimeError:shutdown"], "released": true, "residual": {"running": 0}, "trace": ["begin", "shutdown", "STOPPED", "release"]}, "begin_error": {"errors": ["RuntimeError:begin"], "released": true, "residual": {"running": 0}, "trace": ["begin", "shutdown", "STOPPED", "release"]}, "cancel_at_wait": {"errors": ["CancelledError:wait"], "released": true, "residual": {"running": 1}, "trace": ["begin", "shutdown", "STOPPED", "release"]}, "grace_expiry": {"errors": [], "released": true, "residual": {"running": 1}, "trace": ["begin", "wait", "shutdown", "STOPPED", "release"]}, "normal_running": {"errors": [], "released": true, "residual": {"running": 1}, "trace": ["begin", "wait", "shutdown", "STOPPED", "release"]}, "normal_zero": {"errors": [], "released": true, "residual": {"running": 0}, "trace": ["begin", "wait", "shutdown", "STOPPED", "release"]}, "shutdown_error": {"errors": ["RuntimeError:shutdown"], "released": true, "residual": {"running": 0}, "trace": ["begin", "wait", "shutdown", "STOPPED", "release"]}, "wait_error": {"errors": ["RuntimeError:wait"], "released": true, "residual": {"running": 0}, "trace": ["begin", "wait", "shutdown", "STOPPED", "release"]}}}
```

이 toy matrix는 different identity의 cache/engine/executor/app-state mutation 0, begin failure 뒤에도
shutdown/STOPPED/release, wait error/cancellation 뒤에도 mandatory shutdown, zero/running residual과
grace expiry 순서를 확인한다. actual shield 재취소와 primary/secondary propagation은 §4.3.2의
project fake matrix가 구현 뒤 증명해야 한다.

## 7. Iteration 2 request terminal proof prototype — PROTOTYPE-ONLY

```bash
python - <<'PY'
import asyncio, json
async def classify(*,proven_disconnect,frames):
 if frames: return 'response'
 if proven_disconnect: return 'client_disconnected'
 raise RuntimeError('downstream_no_response')
async def main():
 out={'proven_disconnect':await classify(proven_disconnect=True,frames=[])}
 try: await classify(proven_disconnect=False,frames=[])
 except RuntimeError as exc: out['unproven_no_response']={'outcome':'internal','exception':str(exc)}
 print(json.dumps({'request_context_terminal_proof':out},sort_keys=True))
asyncio.run(asyncio.wait_for(main(),1.0))
PY
```

Exit status `0`; exact stdout:

```json
{"request_context_terminal_proof": {"proven_disconnect": "client_disconnected", "unproven_no_response": {"exception": "downstream_no_response", "outcome": "internal"}}}
```

기존 §2 prototype의 `starts==0` 분류는 최소 characterization일 뿐 closure evidence가 아니다.
실제 설계/acceptance는 route marker 또는 observed `http.disconnect`가 있어야만 499-equivalent를
허용하고, 증거 없는 no-response는 internal programming error로 구별한다.

## 8. Iteration 2 closure와 evidence boundary

| 대상 | prototype-only 관측 | 구현 뒤 필수 project acceptance |
|---|---|---|
| M42-RR1-001 | first commit, same-object reacquire, distinct-object fixed reject; cache write 1, reject 뒤 engine/executor/app-state delta 0 | module ASGI/CLI 공통 primitive, initial invalid, config facade/engine/executor `is`, exact loader counts, fresh subprocess identity isolation |
| M42-RR1-002 | begin/wait/shutdown error와 wait cancellation 뒤 mandatory shutdown→STOPPED→release; zero/running/grace residual | cancellation at every await/shield boundary, combined errors, original primary/cancel propagation, secondary aggregation/log receipt, reacquire only after STOPPED+release |
| request terminal proof | proven frame-0 disconnect와 unproven frame-0 programming error 분리 | actual app route marker/receive observation, logs/metrics/request ID exactly once, erroneous downstream no-response negative fixture |

따라서 M42-RR1-001/002는 [Design §4.3](Design.md)의 approved simplified design contract에서는
CLOSED지만 product 구현 완료/PASS는 아니다. M4.1은 계속 별도 `M4.1_BLOCKED`다.

## 9. Iteration 3 closure boundary

Recovery Review Iteration 2의 M42-RR2-001/002는 [Design §4.3](Design.md)의 실행 순서를
수정함으로써 design scope에서 CLOSED다. 다음 prototype은 그 순서의 bounded characterization일
뿐이며 product module, actual app/global cache/facade/engine/executor 또는 project test를 실행한
PASS receipt가 아니다. 특히 teardown task 생성 실패, shield 재취소와 ordered
primary/secondary/`ExceptionGroup`은 구현 뒤 project fake matrix가 계속 증명해야 한다.

## 10. Iteration 3 startup/cleanup ordering prototype — PROTOTYPE-ONLY

아래 command는 invalid loader, different identity, candidate 대입 전/loader 중 cancellation,
executor constructor failure와 executor-none teardown을 1초로 제한해 실행한다. invalid loader는
REQ-009.2가 허용한 `settings_invalid` transaction만 세고, identity mismatch는 app/cache/config/
engine/executor mutation delta가 모두 0인지 별도로 검사한다.

```bash
python - <<'PY'
import asyncio, json

class SettingsError(Exception): pass
class Guard:
 def __init__(self): self.owner=None; self.committed=None; self.cache=None; self.seq=0
 def acquire(self,trace):
  assert self.owner is None; self.seq+=1; self.owner=self.seq; trace.append('acquire'); return self.seq
 def commit(self,candidate,counts,trace):
  trace.append('commit_or_verify')
  if self.committed is None:
   self.committed=self.cache=candidate; counts['cache']+=1; trace.append('first')
  elif self.committed is not candidate: raise RuntimeError('process_settings_identity_mismatch')
  else: trace.append('same')
 def release(self,lease,trace): assert self.owner==lease; self.owner=None; trace.append('release')
class Loader:
 def __init__(self,result=None,error=None): self.result=result; self.error=error
 def __call__(self,trace):
  trace.append('loader_start')
  if self.error: raise self.error
  trace.append('loader_return'); return self.result
async def teardown(executor,guard,lease,trace):
 if executor is not None:
  trace.append('begin'); trace.append('wait'); trace.append('shutdown')
 trace.append('STOPPED'); guard.release(lease,trace)
async def run_case(guard,loader,*,constructor_error=False):
 trace=[]; counts={k:0 for k in ('app','cache','config','engine','executor')}
 lease=guard.acquire(trace); candidate=None; executor=None; grace=0.0; trace.append('locals')
 primary=None
 try:
  try: candidate=loader(trace)
  except SettingsError:
   counts['app']+=1; trace.append('settings_invalid_transaction')
  else:
   guard.commit(candidate,counts,trace)
   counts['app']+=1; counts['config']+=1; grace=3.0; trace.append('app_config_grace')
   counts['engine']+=1; trace.append('engine_construct')
   if constructor_error: raise RuntimeError('executor_constructor_failed')
   counts['executor']+=1; executor=object(); trace.append('executor_construct')
 except BaseException as exc:
  primary=type(exc).__name__+':'+str(exc)
 finally:
  await teardown(executor,guard,lease,trace)
 return {'counts':counts,'grace':grace,'primary':primary,'trace':trace}
async def main():
 committed=object(); g=Guard(); g.committed=g.cache=committed
 rows={}
 rows['loader_failure']=await run_case(g,Loader(error=SettingsError('invalid')))
 rows['different_identity']=await run_case(g,Loader(result=object()))
 rows['cancel_before_candidate']=await run_case(g,Loader(error=asyncio.CancelledError()))
 rows['cancel_during_loader']=await run_case(g,Loader(error=asyncio.CancelledError('during')))
 rows['constructor_failure']=await run_case(g,Loader(result=committed),constructor_error=True)
 for name,row in rows.items():
  assert row['trace'][-2:]==['STOPPED','release'] and g.owner is None
  if name!='constructor_failure': assert not {'begin','wait','shutdown'} & set(row['trace'])
 assert rows['different_identity']['counts']=={k:0 for k in ('app','cache','config','engine','executor')}
 assert rows['different_identity']['trace']==['acquire','locals','loader_start','loader_return','commit_or_verify','STOPPED','release']
 assert rows['loader_failure']['trace']==['acquire','locals','loader_start','settings_invalid_transaction','STOPPED','release']
 assert rows['constructor_failure']['counts']=={'app':1,'cache':0,'config':1,'engine':1,'executor':0}
 print(json.dumps({'iteration3_startup_cleanup':rows},sort_keys=True))
asyncio.run(asyncio.wait_for(main(),1.0))
PY
```

Exit status `0`; exact stdout:

```json
{"iteration3_startup_cleanup": {"cancel_before_candidate": {"counts": {"app": 0, "cache": 0, "config": 0, "engine": 0, "executor": 0}, "grace": 0.0, "primary": "CancelledError:", "trace": ["acquire", "locals", "loader_start", "STOPPED", "release"]}, "cancel_during_loader": {"counts": {"app": 0, "cache": 0, "config": 0, "engine": 0, "executor": 0}, "grace": 0.0, "primary": "CancelledError:during", "trace": ["acquire", "locals", "loader_start", "STOPPED", "release"]}, "constructor_failure": {"counts": {"app": 1, "cache": 0, "config": 1, "engine": 1, "executor": 0}, "grace": 3.0, "primary": "RuntimeError:executor_constructor_failed", "trace": ["acquire", "locals", "loader_start", "loader_return", "commit_or_verify", "same", "app_config_grace", "engine_construct", "STOPPED", "release"]}, "different_identity": {"counts": {"app": 0, "cache": 0, "config": 0, "engine": 0, "executor": 0}, "grace": 0.0, "primary": "RuntimeError:process_settings_identity_mismatch", "trace": ["acquire", "locals", "loader_start", "loader_return", "commit_or_verify", "STOPPED", "release"]}, "loader_failure": {"counts": {"app": 1, "cache": 0, "config": 0, "engine": 0, "executor": 0}, "grace": 0.0, "primary": null, "trace": ["acquire", "locals", "loader_start", "settings_invalid_transaction", "STOPPED", "release"]}}}
```

관측 trace는 모든 case에서 acquire 직후 locals가 먼저이고, candidate가 생기기 전 app/global
mutation이 없으며, different identity의 commit/verify 다음에는 STOPPED→release만 있음을 보인다.
loader failure만 explicit settings-invalid transaction을 publish한다. cancellation 두 위치와 constructor
failure는 initialized executor/grace로 teardown에 들어가 begin/wait/shutdown 0, STOPPED→release를
보존한다. executor가 실제 존재하는 mandatory shutdown 및 cleanup error aggregation은 기존 §6
prototype과 구현 후 acceptance matrix의 범위다.

| Finding | Iteration 3 prototype-only evidence | 구현 뒤 필수 project acceptance |
|---|---|---|
| M42-RR2-001 | loader local candidate→commit/verify 순서; different identity app/cache/config/engine/executor delta 0; invalid loader만 별도 diagnostic | actual module ASGI/CLI 공통 primitive, facade identity, atomic health transaction과 fresh-subprocess mutation spies |
| M42-RR2-002 | pre-candidate cancellation, loader failure, constructor failure에서 initialized zero/configured grace와 executor-none STOPPED→release | teardown task creation/evaluation failure, shield cancellation boundaries, executor-present mandatory shutdown once, ordered primary/secondary/`ExceptionGroup` |

## 11. Iteration 4 attempt-owner/canonical-tail prototype — PROTOTYPE-ONLY

아래 bounded command는 final base iteration의 세 attempt class를 분리한다. identity mismatch spy는
`app.__dict__`, health/log/metric sinks, process cache/config와 engine/executor factory counts의
full-attempt before/after를 비교한다. invalid loader spy는 atomic `settings_invalid` transaction 정확히
1개와 generic stopped observer 0, exact-owner release 1을 검사한다. started owner spy는 모든 fallible
observer/snapshot/error aggregation이 non-throwing atomic `STOPPED`와 exact-owner release보다 앞서며
그 둘이 final two durable external actions인지 검사한다. release 뒤 diagnostic sink failure는
non-durable/best-effort이고 즉시 reacquire를 막지 않는다.

```bash
python - <<'PY'
import json
class Guard:
 def __init__(self): self.owner=None; self.seq=0
 def acquire(self): self.seq+=1; self.owner=self.seq; return self.seq
 def release(self, token):
  code='ok' if self.owner==token else 'owner_mismatch'
  if code=='ok': self.owner=None
  return code
class World:
 def __init__(self):
  self.app={}; self.health=[]; self.logs=[]; self.metrics=[]
  self.cache={'settings':'committed'}; self.config={'settings':'committed'}
  self.factories={'engine':0,'executor':0}; self.durable=[]
 def snap(self):
  return {'app':dict(self.app),'health':list(self.health),'logs':list(self.logs),
   'metrics':list(self.metrics),'cache':dict(self.cache),'config':dict(self.config),
   'factories':dict(self.factories)}
def mismatch(w,g):
 before=w.snap(); lease=g.acquire(); g.release(lease); after=w.snap()
 return {'delta':{k:int(before[k]!=after[k]) for k in before},
  'stopped':w.durable.count('STOPPED'),'release':int(g.owner is None)}
def invalid(w,g):
 lease=g.acquire(); w.health.append(('atomic','settings_invalid')); code=g.release(lease)
 return {'settings_invalid_transactions':w.health.count(('atomic','settings_invalid')),
  'generic_stopped':0,'release':int(code=='ok' and g.owner is None)}
def started(w,g):
 lease=g.acquire()
 for action in ('observer','snapshot','aggregate'): w.durable.append(action)
 w.durable.append('STOPPED'); code=g.release(lease); w.durable.append('release')
 nondurable=[]
 try: nondurable.append('release_diag:'+code); raise RuntimeError('sink down')
 except RuntimeError: pass
 reacquired=g.acquire(); g.release(reacquired)
 return {'durable':w.durable,'tail':w.durable[-2:],
  'nondurable':nondurable,'reacquired':g.owner is None}
g=Guard()
rows={'identity_mismatch':mismatch(World(),g),'invalid_loader':invalid(World(),g),
 'started':started(World(),g)}
assert all(v==0 for v in rows['identity_mismatch']['delta'].values())
assert rows['identity_mismatch']['stopped']==0 and rows['identity_mismatch']['release']==1
assert rows['invalid_loader']=={'settings_invalid_transactions':1,'generic_stopped':0,'release':1}
assert rows['started']['tail']==['STOPPED','release']
assert rows['started']['durable'][:-2]==['observer','snapshot','aggregate']
assert rows['started']['reacquired']
print(json.dumps({'iteration4_attempt_owners':rows},sort_keys=True))
PY
```

Exit status `0`; exact stdout:

```json
{"iteration4_attempt_owners": {"identity_mismatch": {"delta": {"app": 0, "cache": 0, "config": 0, "factories": 0, "health": 0, "logs": 0, "metrics": 0}, "release": 1, "stopped": 0}, "invalid_loader": {"generic_stopped": 0, "release": 1, "settings_invalid_transactions": 1}, "started": {"durable": ["observer", "snapshot", "aggregate", "STOPPED", "release"], "nondurable": ["release_diag:ok"], "reacquired": true, "tail": ["STOPPED", "release"]}}}
```

| Finding | Iteration 4 prototype-only evidence | 구현 뒤 필수 project acceptance |
|---|---|---|
| M42-RR3-001 | identity mismatch observer/factory full delta 0, STOPPED 0, release 1; invalid loader transaction 1/generic stopped 0/release 1 | actual app `__dict__`, real sinks/cache/config/factory spies, atomic transaction and exact token release |
| M42-RR3-002 | fallible actions precede exact durable `STOPPED→release` tail; post-release diagnostic failure does not prevent reacquire | started/partially-started cancellation/error matrix, atomic STOPPED snapshot, release diagnostic sink failures and concurrent reacquire |

이는 bounded prototype evidence이며 product code/test PASS 주장이 아니다. 이전 §10 prototype의 모든-case
STOPPED trace는 RR2 당시 characterization이고, RR3 closure가 identity mismatch와 invalid loader에 대해
이를 명시적으로 supersede한다. approved product scope는 변하지 않는다.

## 12. 문서 검증 commands

문서 편집 완료 뒤 다음을 실행한다.

```bash
python scripts/check_markdown_links.py
git diff --check
```
