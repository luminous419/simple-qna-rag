# M4.1 Configuration & Observability Foundation — 실행 계약 (설계 재개 사이클 3)

상태: **설계 재개 사이클 3 —
[Design_Review_Resume_Cycle_2.md](Design_Review_Resume_Cycle_2.md) MAJOR
`R2-MAJ-01`/MINOR `R2-MIN-01` 폐쇄 대상**. Iteration 1~5, 설계 재개 사이클 1의
판정·폐쇄 기록과 설계 재개 사이클 2의 독립 재검증(§1, §2)은 감사 이력으로
그대로 보존하며 원문을 재작성하지 않는다 — 이번 사이클은 §1 폐쇄 맵에
`R2-MAJ-01`/`R2-MIN-01` 행을 추가하고, R2-MAJ-01이 지적한 §10.4 항목 6
recording wrapper integration test의 존재하지 않는 공개 `Mock.wraps`
attribute assertion을 실행 가능한 계약으로 교체하며, R2-MIN-01이 지적한
Traceability의 설계 재개 사이클 1 폐쇄 증거 서술을 정정한다. 다른 섹션은
설계 재개 사이클 2 상태에서 변경하지 않는다.
요구사항: [Requirement.md](Requirement.md) · 계획: [Plan.md](Plan.md) · 추적: [Traceability.md](Traceability.md)
필드 전수 증거: [Field_Spec_Inventory.md](Field_Spec_Inventory.md)
리뷰: [Design_Review_Iteration_1.md](Design_Review_Iteration_1.md),
[Design_Review_Iteration_2.md](Design_Review_Iteration_2.md),
[Design_Review_Iteration_3.md](Design_Review_Iteration_3.md),
[Design_Review_Iteration_4.md](Design_Review_Iteration_4.md),
[Design_Review_Iteration_5.md](Design_Review_Iteration_5.md),
[Design_Review_Resume_Cycle_1.md](Design_Review_Resume_Cycle_1.md),
[Design_Review_Resume_Cycle_2.md](Design_Review_Resume_Cycle_2.md)
중단 기록: [Stop_Report.md](Stop_Report.md)
상위 결정: [M4 Recovery Decision](../m4-production-readiness/Recovery_Decision.md)

## 0. 범위와 원칙

M4.1만 상세화한다(dependency lock, typed settings, structured logging, bounded
metrics, 기본 live/ready). M4.2/M4.3은 §12 seam만 명시한다. 제품 코드는 이
문서에서 수정하지 않는다. Iteration 1의 executable 증거(§1)는 재검증 없이
그대로 승계하고, 그 증거로 확정할 수 없었던 계약만 이번 개정에서 확정한다.
Requirement REQ-002.4의 web 예외 문구(모델·index 초기화 전 exit 2 원칙에서
`simple-qna-rag-web`을 제외하고 `/health/ready` 503으로 표현)는 §3의 근거이며
이 개정에서 손대지 않는다.

## 1. 리뷰 폐쇄 맵

| 리뷰 ID | 요지 | 폐쇄 섹션 |
|---|---|---|
| M1-01 | exit-2와 settings_invalid health 양립 불가 | §3 |
| M1-02 | FieldSpec/from_env 알고리즘 없음 | §4 |
| M1-03 | 3 CLI 경로가 현실과 불일치 | §5 |
| M1-04 | payload logging 185건 미정리 | §6 |
| M1-05 | metrics dependency/schema 미확정 | §7 |
| M1-06 | lock 도구/canonical/artifact 미확정 | §8 |
| M1-07 | logging/health acceptance matrix 누락 | §9 |
| M1-08 | M3 14-gate·baseline·vectorstore 미설계 | §10 |
| m1-01 | Phase 번호 불일치 | §11.1 |
| m1-02 | `/health` deprecation 계약 부족 | §11.2 |
| m1-03 | `disable_created_metrics()` 소유권 불명 | §11.3 |
| t1-01 | §4.2 잘못된 cross-reference | §11.4(§7로 정정) |
| t1-02 | `patttern` 오탈자 | §11.4 |
| M2-01 | bootstrap이 invalid static/template에서 health 생존 미보장 | §3.2, §3.4, §3.6 |
| M2-02 | `FieldSpec`/Settings 생성 계약이 실행 불가 | §4.1, [Field_Spec_Inventory.md](Field_Spec_Inventory.md) |
| M2-03 | CLI override가 base env를 전부 대체 | §4.3, §5.1 |
| M2-04 | fallback metric 누락, sample 상한식이 관측치와 혼동 | §7.2, §7.3 |
| M2-05 | M3 gate wrapper가 실제 evaluation API와 불일치 | §10.1 |
| m2-01 | `/health` Sunset 값이 placeholder | §11.2 |
| m2-02 | logging audit가 `print`만 포괄 | §6.1 |
| m2-03 | `configure_metrics()`의 "1회"와 재사용 계약 충돌 | §7.4 |
| M5-01 | 회귀 wrapper가 facade `str` 경로를 `Path` 전용 API에 그대로 전달(M4-01 재개방) | §10.1 |
| m5-01 | `MODEL_VALIDATORS` 표 header 5열/데이터 행 7열 불일치 | [Field_Spec_Inventory.md](Field_Spec_Inventory.md) |
| R1-MAJ-01 | integration case가 `build_vectorstore_fingerprint` 실제 호출을 증명하지 않고 결과 동등성만 검증 | §10.4 항목 6 |
| R1-MAJ-02 | `MODEL_VALIDATORS`에 5열 표를 재생성할 입력/알고리즘 계약이 없음 | §4.1.1 |
| R2-MAJ-01 | recording wrapper integration case가 Python 3.11 `unittest.mock.Mock`에 없는 공개 `.wraps` 속성을 assert해 실행 불가능 | §10.4 항목 6 |
| R2-MIN-01 | Traceability 머리말이 설계 재개 사이클 1 리뷰를 폐쇄 증거로 잘못 서술 | [Traceability.md](Traceability.md) 머리말 |

## 2. Iteration 1 executable 증거(승계, 재검증 안 함)

| 항목 | 결과 |
|---|---|
| lock 재현성(uv) | `--extra-index-url .../whl/cpu --index-strategy unsafe-best-match` 필요, 102 패키지/2,499줄, 2회 실행 body 동일(header 제외) |
| lock 재현 환경 | macOS 실행, 설치 자체는 Linux CI에서만 검증(§8) |
| prometheus_client 0.23.1 | `disable_created_metrics()`가 유일한 created-series 공개 API, label allowlist는 클라이언트 자체 미강제 |
| cardinality spike | custom 8-bucket/route2/status3/stage4/error4/reason4 스키마로 1,000건 시뮬레이션 시 실 sample 67개, 기본 11-bucket/stage6이면 146개 |
| lifespan/TestClient | 예외 미포착 시 `with TestClient(app)` 진입 자체가 실패; `app.state`는 `with` 블록 밖에서는 설정되지 않음 |
| settings consumer inventory | 공개 심볼 42개(필드 41 + `resolve_runtime_path`), AST+runtime `hasattr`+subprocess 문자열 grep 3중 확인, 참조 누락 0 |
| CLI import 순서 | `cli/web.py`/`cli/query.py`는 env 설정 후 지연 import로 config를 로드(첫 import가 materialize 시점) |
| print inventory(정정) | `src/simple_qna_rag` 전체 **185건, 9개 파일**(Design §1 원 71건/3파일 수치는 폐기) |

## 3. 프로세스 경계와 상태 계약 (M1-01)

### 3.1 원칙

Requirement REQ-002.4 예외에 따라 프로세스를 두 그룹으로 분리한다.

- **exit-2 그룹**: `--check-config`, `simple-qna-rag-query`, `simple-qna-rag-index`.
  `Settings.from_env()` 실패 시 `SettingsError`를 최외곽 `main()`에서 잡아
  `sys.exit(2)`. 모델/index 초기화 이전.
- **health-표현 그룹**: `simple-qna-rag-web`(`--check-config` 미사용). 전체
  `Settings` 검증 실패를 exit이 아니라 `/health/ready` 503으로 표현한다.

### 3.2 web bootstrap 상태(package-fixed, import-free, M2-01)

`Bootstrap`은 **env를 전혀 읽지 않는다**(리뷰가 지적한 "env 입력인지 고정
파생값인지 불명확" 문제 제거) — `load_bootstrap()`은 인자 없이
`PROJECT_ROOT`/`STATIC_DIR`/`TEMPLATES_DIR`을 `config.py`/`Settings`와
**독립적으로** `Path(__file__)` 기준 상수로 계산한다(REQ-002.4의 "PROJECT_ROOT
기반 고정 상수"를 문자 그대로 구현). `bootstrap.py`는 `config.py`/`settings.py`를
import하지 않으므로 §4의 Settings 로딩이 실패해도 영향받지 않는다.

```python
# src/simple_qna_rag/web/bootstrap.py (신규, config.py/settings.py 미import)
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]  # -> repo root
STATIC_DIR: Final[Path] = _REPO_ROOT / "web" / "static"
TEMPLATES_DIR: Final[Path] = _REPO_ROOT / "web" / "templates"

@dataclass(frozen=True)
class Bootstrap:
    static_dir: Path = STATIC_DIR
    templates_dir: Path = TEMPLATES_DIR

def load_bootstrap() -> Bootstrap:
    return Bootstrap()  # 인자 없음 — package/repo 고정
```

`Settings.STATIC_DIR`/`TEMPLATES_DIR`(§4.2)은 동일 계산식의 별도 facade
값이며 코드 경로는 공유하지 않는다 — 의도된 중복이고 §3.6이
`bootstrap.STATIC_DIR == Settings().STATIC_DIR`로 drift를 검출한다.
`create_app`은 health route를 **가장 먼저** 등록한 뒤 mount를 시도하므로
mount 실패가 route 등록을 막지 못한다:

```python
def create_app(
    bootstrap: Bootstrap = load_bootstrap(),
    settings_loader: Callable[[], Settings] = get_settings,
    engine_factory: Callable[[Settings], RagEngine] = RagEngine.from_settings,
) -> FastAPI:
    app = FastAPI(lifespan=_make_lifespan(settings_loader, engine_factory))
    _register_health_routes(app)                 # 최우선 — mount 성패와 무관
    app.state.bootstrap_error = _mount_static_and_templates(app, bootstrap)
    _register_api_routes(app)
    return app

def _mount_static_and_templates(app: FastAPI, bootstrap: Bootstrap) -> str | None:
    try:
        if not bootstrap.static_dir.is_dir():
            raise NotADirectoryError
        app.mount("/static", StaticFiles(directory=bootstrap.static_dir), name="static")
        if not bootstrap.templates_dir.is_dir():
            raise NotADirectoryError
        app.state.templates = Jinja2Templates(directory=bootstrap.templates_dir)
    except (FileNotFoundError, NotADirectoryError):
        return "static_mount_failed"   # sanitized — 경로 문자열 미노출
    return None
```

module import 시점에 전체 `Settings`를 생성하지 않고 mount 실패도 예외를
전파하지 않는다 — M1-01의 핵심 결함(invalid settings/누락 경로가 module
import/app 생성을 중단시켜 health route 자체가 없어지는 문제)을 제거한다.

### 3.3 lifespan 2단계 상태 전이

```python
def _make_lifespan(settings_loader, engine_factory):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.settings = settings_loader()
            app.state.settings_error = None
        except SettingsError as exc:
            app.state.settings = None
            app.state.settings_error = str(exc)
        if app.state.settings is not None:
            try:
                app.state.engine = engine_factory(app.state.settings)
                app.state.engine_error = None
            except Exception as exc:  # noqa: BLE001 — health로 표현, 재발생 금지
                app.state.engine = None
                app.state.engine_error = str(exc)
        else:
            app.state.engine, app.state.engine_error = None, None
        yield
    return lifespan
```

`lifespan`은 core lifecycle 속성 4개(`settings`,`settings_error`,`engine`,
`engine_error`)를 설정한다 — `bootstrap_error`(§3.2), `templates`(§3.2 mount
성공 시), `metrics_registry`(§7.4)는 다른 시점에 추가되는 별도 속성이며
테스트는 `app.state`의 전체 attribute 개수를 고정하지 않는다. 어느 예외도
`raise`로 전파하지 않는다(기존 `web/server.py` 39-49행의 `raise` 제거가
이 계약의 핵심).

### 3.4 health 상태표(REQ-005.1/.2, M1-01/M2-01 재정합)

`observability/health.py::evaluate_readiness(bootstrap_error, settings_error,
engine_error) -> (status_code, reason)`. `bootstrap_error`가 최우선이다 —
static/template mount 실패는 배포 결함이며 settings/engine 상태와 무관하게
드러나야 한다:

| `bootstrap_error` | `settings_error` | `engine_error` | `/health/live` | `/health/ready` | `reason` |
|---|---|---|---|---|---|
| not None | — | — | 200 | 503 | `static_mount_failed` |
| None | not None | — | 200 | 503 | `settings_invalid` |
| None | None | not None | 200 | 503 | `engine_init_failed` |
| None | None | None | 200 | 200 | `ok` |

`/health/live`는 `app.state` 접근 없이 상수 200만 반환한다(REQ-005.1 — event
loop 응답성만 증명). `reason` 값 4종(`ok`,`settings_invalid`,
`engine_init_failed`,`static_mount_failed`)은 §7.2 `rag_readiness` 라벨
allowlist와 1:1로 동기화한다(`other`를 더해 5종, §7.3).

### 3.5 CLI exit-2 그룹 계약

```python
def main(argv=None) -> int:
    try:
        settings = Settings.from_env(_overrides_from_args(parse_args(argv)))
    except SettingsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    ...
```

query/index CLI의 기존 "초기화 예외 -> exit 1" 매핑은 `SettingsError` 발생
지점(설정 로드) 이후 별도 예외 클래스이므로 그대로 유지한다 — exit 2는 오직
`SettingsError`, exit 1은 그 이후 초기화/처리 예외.

### 3.6 subprocess/TestClient 증거 매트릭스(M2-01 필수 수정)

**subprocess**는 저장소 기본 경로로 `create_app()`이 import/생성 단계에서
죽지 않는지 확인하는 smoke test, **TestClient**는 `Bootstrap`을 생성자
인자로 직접 주입해(env 불필요) 실패 조합을 커버한다.

| # | 계층 | static_dir | templates_dir | 기대 결과 |
|---|---|---|---|---|
| 1 | subprocess: `python -c "...create_app(load_bootstrap())"` | 기본값(존재) | 기본값(존재) | exit 0 |
| 2 | TestClient: `Bootstrap(static_dir=tmp/"missing", templates_dir=valid)` | 없음 | 존재 | `/health/live`=200, `/health/ready`=503 `static_mount_failed` |
| 3 | TestClient: static_dir이 일반 파일 | 파일 | 존재 | 동일 |
| 4 | TestClient: `Bootstrap(static_dir=valid, templates_dir=tmp/"missing")` | 존재 | 없음 | 동일 |
| 5 | TestClient: templates_dir이 일반 파일 | 존재 | 파일 | 동일 |

`tests/integration/test_web_bootstrap_matrix.py`: 1행은 exit code만, 2~5행은
`/health/ready` body의 `reason`까지 assert. 같은 파일이
`bootstrap.STATIC_DIR == Settings().STATIC_DIR`(§3.2 이중 계산 drift)도
확인한다.

## 4. Settings 단일 원본: FieldSpec/from_sources (M1-02, M2-02, M2-03)

### 4.1 FieldSpec 스키마(실행 가능한 10-column, M2-02 필수 수정)

```python
@dataclass(frozen=True)
class FieldSpec:
    name: str                                       # Settings 필드명 == config.py 상수명
    annotation: Any                                  # 실제 pydantic annotation(Literal/int/str/Path/float)
    default: object = PydanticUndefined               # default_factory와 상호 배타
    default_factory: Callable[[], object] | None = None
    env_alias: str | None = None                     # None이면 파생 필드/root 상수(직접 env 없음)
    parser: Callable[[str], object] | None = None     # env_alias 있을 때만
    derive: Callable[[Mapping[str, object]], object] | None = None  # derived_from 비어있지 않을 때만
    derived_from: tuple[str, ...] = ()                # derive가 읽는 선행 필드명(위상정렬 키)
    validators: tuple[Callable[[object], object], ...] = ()  # per-field, field_validator body
    consumers: tuple[str, ...] = ()                   # module path, §1.4 3중 확인 산출
    facade_type: type | None = None                   # None이면 facade == annotation(예: PROJECT_ROOT)
    facade_adapter: Callable[[object], object] | None = None  # None이면 항등, 있으면 facade 투영 함수(예: str)

@dataclass(frozen=True)
class ModelValidatorSpec:                              # R1-MAJ-02 필수 수정(재개 사이클 2) — 선언형 단일 원본
    callable: Callable[["Settings"], "Settings"]        # model_validator(mode="after") body. Settings 인스턴스와
                                                          # §4.1.1의 duck-typed namespace 양쪽에서 동일하게 호출된다.
    constraint: str                                     # 5열 표 "제약" 컬럼의 유일한 source, 예: "CHUNK_OVERLAP < CHUNK_SIZE"
    related_fields: tuple[str, ...]                     # constraint 좌→우 순서의 FIELD_SPECS.name.
                                                          # §4.1.1이 FIELD_SPECS 순번(#N)으로 투영해 "관련 필드" 컬럼을 만든다.
    default_rendering: Callable[[Mapping[str, object]], str]  # {field_name: default_value} -> "200 < 1000".
                                                          # constraint와 동일한 연산자·순서를 재사용해야 한다(§4.5 회귀 대상).

MODEL_VALIDATORS: tuple[ModelValidatorSpec, ...] = (
    ModelValidatorSpec(
        callable=_check_chunk_overlap_lt_chunk_size,
        constraint="CHUNK_OVERLAP < CHUNK_SIZE",
        related_fields=("CHUNK_OVERLAP", "CHUNK_SIZE"),
        default_rendering=lambda d: f"{d['CHUNK_OVERLAP']} < {d['CHUNK_SIZE']}",
    ),
    ModelValidatorSpec(
        callable=_check_mmr_k_le_fetch_k,
        constraint="MMR_K <= MMR_FETCH_K",
        related_fields=("MMR_K", "MMR_FETCH_K"),
        default_rendering=lambda d: f"{d['MMR_K']} <= {d['MMR_FETCH_K']}",
    ),
    ModelValidatorSpec(
        callable=_check_reranker_le_rrf,
        constraint="RERANKER_TOP_K <= RRF_TOP_K",
        related_fields=("RERANKER_TOP_K", "RRF_TOP_K"),
        default_rendering=lambda d: f"{d['RERANKER_TOP_K']} <= {d['RRF_TOP_K']}",
    ),
)  # cross-field 제약 — FieldSpec 밖 별도 목록. §4.1.1 generator와 runtime validator 실행의 유일한 원본(단일 source).
```

한 필드는 세 경로 중 정확히 하나로 값을 얻는다(불변식, §4.5 테스트로 검증):
(a) `env_alias is not None`→env/CLI를 `parser`로, 없으면
`default`/`default_factory`, (b) `env_alias is None and derived_from`→
`derive(values)`, (c) 그 외→순수 `default_factory()`(예: `PROJECT_ROOT`).

`FIELD_SPECS: tuple[FieldSpec, ...]`가 유일한 원본이다. **41행을 손으로
중복 나열하지 않는다**(REQ-002.6, M4 MAJOR M4-06 "30 vs 53" 재발 방지 원칙을
스키마에도 적용). `scripts/generate_field_spec.py`가 `FIELD_SPECS`를 순회해
`docs/generated/settings_field_spec.md`(CI diff 0 검사)를 만든다. 이 설계의
증거로 동일 41행을 [Field_Spec_Inventory.md](Field_Spec_Inventory.md)에
미리 생성한다(annotation/default/env_alias/validators/consumers +
"default가 validators를 통과하는가" 컬럼 포함).

**Settings 생성 symbol(frozen, `create_model` 기반)** — 정적
`class Settings(BaseModel)` 정의는 없다, FIELD_SPECS가 유일한 원본:

```python
def _field_definitions(specs) -> dict[str, tuple[Any, FieldInfo]]:
    return {
        s.name: (s.annotation, FieldInfo(default_factory=s.default_factory)
                  if s.default_factory is not None else FieldInfo(default=s.default))
        for s in specs
    }

def _validator_namespace(specs) -> dict[str, classmethod]:
    ns: dict[str, classmethod] = {}
    for s in specs:
        for i, fn in enumerate(s.validators):
            ns[f"_check_{s.name}_{i}"] = field_validator(s.name)(fn)
    for i, mv in enumerate(MODEL_VALIDATORS):
        ns[f"_check_model_{i}"] = model_validator(mode="after")(mv.callable)
    return ns

Settings = create_model(
    "Settings",
    __config__=ConfigDict(frozen=True, extra="forbid"),
    __validators__=_validator_namespace(FIELD_SPECS),
    **_field_definitions(FIELD_SPECS),
)
```

### 4.1.1 `ModelValidatorSpec` 선언형 단일 원본과 generator `--check` 계약 (R1-MAJ-02 필수 수정, 재개 사이클 2)

Iteration 5~설계 재개 사이클 1까지의 `MODEL_VALIDATORS`는
`tuple[Callable[["Settings"], "Settings"], ...]`뿐이라 5열 표(`#`,제약,관련
필드,default 값,판정)를 만들 표시용 metadata를 담지 못했다(리뷰 지적). §4.1의
`ModelValidatorSpec`이 그 metadata를 담는 선언형 단일 원본이며, runtime
validator 등록(`_validator_namespace`가 `mv.callable`을 `model_validator`에
그대로 넘김, §4.1)과 아래 generator 둘 다 이 자료구조 하나만 읽는다 — 함수
이름·소스코드 파싱이나 별도 수기 mapping은 없다.

**41-field 표 생성**: `render_field_specs_table(FIELD_SPECS) -> str`이 §4.1
10-column을 `FIELD_SPECS` 순서 그대로 순회해 Markdown 표를 만든다(기존
Field_Spec_Inventory.md 41-행 표와 동형, 열 순서 고정, REQ-002.6).

**3-row/5-column validator 표 생성**: `render_model_validators_table(MODEL_VALIDATORS,
FIELD_SPECS) -> str`이 `MODEL_VALIDATORS`를 순회해 만든다:

```python
def _default_field_values(specs: tuple[FieldSpec, ...]) -> dict[str, object]:
    # from_sources()의 "raw 없음"(unset) 분기와 동일한 계산 — §4.1
    # default_pass 컬럼(Field_Spec_Inventory.md)과 동일 source.
    values: dict[str, object] = {}
    for spec in _topo_sorted(specs):
        values[spec.name] = (
            spec.derive(values) if spec.derive
            else spec.default_factory() if spec.default_factory is not None
            else spec.default
        )
    return values

def render_model_validators_table(validators, specs) -> str:
    index_of = {s.name: i + 1 for i, s in enumerate(specs)}   # FIELD_SPECS #번호, 삽입 순서 고정
    defaults = _default_field_values(specs)
    namespace = SimpleNamespace(**defaults)                    # duck-typed Settings 대역
    rows = []
    for n, mv in enumerate(validators, start=1):
        related = ", ".join(f"#{index_of[name]}" for name in mv.related_fields)
        default_value = mv.default_rendering(defaults)
        try:
            mv.callable(namespace)                               # 실제 validator 함수를 그대로 재사용
            verdict = "PASS"
        except ValueError:
            verdict = "FAIL"
        rows.append((n, mv.constraint, related, default_value, verdict))
    return _to_markdown_table(header=("#", "제약", "관련 필드", "default 값", "판정"), rows=rows)
```

`mv.callable(namespace)`는 `Settings` model_validator body가 속성 접근만
하므로(예: `self.CHUNK_OVERLAP >= self.CHUNK_SIZE: raise ValueError(...)`)
`SimpleNamespace`에도 동일하게 동작한다 — runtime에서 pydantic이 호출하는
함수와 generator가 판정에 쓰는 함수가 동일 객체(`mv.callable`)이므로 두 표
사이 drift가 구조적으로 불가능하다(별도 boolean predicate 필드를 두지 않은
이유이자 R1-MAJ-02 "동일 source" 요구의 근거).

**`--check` 계약**: `scripts/generate_field_spec.py`는 위 두 표를 이어붙여
`docs/generated/settings_field_spec.md`를 생성한다. 인자 없이 실행하면 파일을
덮어쓰고, `--check`는 새로 렌더링한 내용과 checked-in 파일을 비교해 다르면
unified diff를 stdout에 출력하고 exit 1, 같으면 exit 0(§13 CI 게이트). 렌더링은
결정론적이다 — 유일한 입력이 `FIELD_SPECS`/`MODEL_VALIDATORS`(둘 다 튜플,
삽입 순서 고정)뿐이고 `index_of`도 리스트 순서 기반 `dict`이므로 dict/set
순회 비결정성에 의존하지 않는다. 같은 입력으로 2회 연속 실행한 렌더링 바이트가
동일함을 §4.5가 테스트로 고정한다.

### 4.2 확정 카테고리와 대표 예시(전수는 [Field_Spec_Inventory.md](Field_Spec_Inventory.md))

| 카테고리 | 대표 필드 | env_alias | validator | facade_type/facade_adapter |
|---|---|---|---|---|
| bootstrap root(1, **env 없음**) | `PROJECT_ROOT` | `None`(§3.2와 동일 원칙) | 없음 | `None`/`None`(facade==`Path`, 무변환) |
| bootstrap derived path(2, **env 없음**) | `STATIC_DIR`,`TEMPLATES_DIR` | `None` | 없음, `derived_from=("PROJECT_ROOT",)` | `str`/`str`(§4.4 투영, PROJECT_ROOT만 예외) |
| model-level 제약 있는 필드(6) | `CHUNK_SIZE`/`CHUNK_OVERLAP`, `MMR_K`/`MMR_FETCH_K`, `RERANKER_TOP_K`/`RRF_TOP_K` | `SIMPLE_QNA_RAG_<NAME>` | `CHUNK_OVERLAP<CHUNK_SIZE`, `MMR_K<=MMR_FETCH_K`, `RERANKER_TOP_K<=RRF_TOP_K` | `None`/`None` |
| M3 rollback flag(4, 확정) | `MMR_VECTOR_SOURCE`, `ROUTING_SIGNAL_OVERRIDE`, `ROUTING_CORPUS_TOPIC_HINT`, `ANSWER_TEMPLATE_MODE` | 기존 env 이름 유지 | 값·이름 불변(REQ-006.1), §5.3 | `None`/`None` |
| 기존 env-backed path(3) | `VECTORSTORE_PATH`,`DATA_DIR`,`INTENT_MODEL_PATH` | 기존 이름 유지(`_VECTORSTORE_DIR`/`_DOCUMENTS_DIR`/`_MODEL_DIR`) | 없음 | `str`/`str`(Settings 정규 타입은 `Path`, facade만 `str` 투영) |
| 나머지(retrieval/모델/웹검색/enum 등 33개) | `RETRIEVAL_K`,`USE_*`,`OLLAMA_*`,`WEB_SEARCH_*`,`BM25_TOKENIZER` 등 | `SIMPLE_QNA_RAG_<NAME>`(신규 도입) | `>0`/`Literal[...]`/`0<=x<=1` 또는 없음 | `None`/`None` |

env_alias는 REQ-002.3이 강제하는 기존 8개(path 3 + M3 flag 4 +
`BM25_TOKENIZER`)를 정확히 유지하고, bootstrap path 3개(`PROJECT_ROOT`/
`STATIC_DIR`/`TEMPLATES_DIR`, env_alias 없음)를 제외한 나머지 30개엔
`SIMPLE_QNA_RAG_<NAME>` 규칙으로 신규 부여한다(3+8+30=41 —
Field_Spec_Inventory.md "요약" 재검산. REQ-002.2 단일 원본의 확장, default는
config.py 리터럴과 동일). `resolve_runtime_path`는 41+1=42번째 심볼로
`config.py`에 기존 시그니처를 보존한 공개 호환 wrapper로 잔류한다(§4.3/§4.4,
m4-01) — FieldSpec 경로용 pure parser는 별도 private `_parse_runtime_path`다.

**Settings 정규 타입과 legacy facade 타입 분리(M3-01 필수 수정)**: `STATIC_DIR`/
`TEMPLATES_DIR`/`VECTORSTORE_PATH`/`DATA_DIR`/`INTENT_MODEL_PATH`는 Settings
내부 annotation이 `Path`(§3.6 drift 비교용)이지만 기존 `config.py` 다섯
상수는 `str`이었다(§2) — 위 표의 `facade_type=str,facade_adapter=str`가 이
다섯 필드에서만 투영이 필요함을 명시한다. `PROJECT_ROOT`는 기존에도 `Path`라
`facade_type=None`(무변환)이며, 같은 path 군에서 한 필드만 예외였던 리뷰
지적(§2)을 필드별 `facade_type`으로 해소한다.

### 4.3 `from_sources` 알고리즘(결정론, base<-env<-CLI 단일 병합 경로, M2-03)

`from_env(overrides)`가 `overrides`로 `os.environ`을 완전히 대체하던
결함(M1-03 재개방 원인) 제거 — base environment와 CLI override를 **별도
인자**로 받는 `from_sources`를 유일한 생성 경로로 고정한다.

```python
ENV_PREFIX = "SIMPLE_QNA_RAG_"

@classmethod
def from_sources(cls, base_environ: Mapping[str, str] | None = None,
                  cli_overrides: Mapping[str, str] | None = None) -> "Settings":
    # base(기본 os.environ) <- cli_overrides. cli_overrides는 base 위에 merge되며 base를 비우지 않는다.
    base = dict(os.environ if base_environ is None else base_environ)
    merged = {**base, **(cli_overrides or {})}      # CLI가 이긴다, base는 유지
    values: dict[str, object] = {}
    for spec in _topo_sorted(FIELD_SPECS):          # derived_from 위상정렬
        if spec.env_alias is None:
            values[spec.name] = spec.derive(values) if spec.derive else spec.default_factory()
            continue
        raw = merged.get(spec.env_alias)
        if raw is None or raw == "":                # unset==빈 문자열, default로
            values[spec.name] = spec.default_factory() if spec.default_factory else spec.default
            continue
        try:
            values[spec.name] = spec.parser(raw)
        except Exception as exc:
            raise SettingsError(f"{spec.env_alias}: invalid value {raw!r}", exit_code=2) from exc
    known = {s.env_alias for s in FIELD_SPECS if s.env_alias}
    unknown = sorted(k for k in merged if k.startswith(ENV_PREFIX) and k not in known)
    if unknown:
        raise SettingsError(f"unknown keys: {unknown}", exit_code=2)
    try:
        return cls.model_validate(values)            # §4.1 MODEL_VALIDATORS 여기서 실행
    except pydantic.ValidationError as exc:
        raise SettingsError(str(exc), exit_code=2) from exc

@classmethod
def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
    return cls.from_sources(base_environ=environ)  # REQ-002.3 하위호환 별칭
```

- **bool parser**: `{"1","true","yes","on"}`→`True`, `{"0","false","no","off"}`→
  `False`(대소문자 무시), 그 외 `SettingsError`(기존 `_env_bool`의 "조용히
  False"보다 엄격).
- **path parser**: `~` 확장 후 상대경로는 `PROJECT_ROOT` 기준 resolve —
  신규 private `_parse_runtime_path(raw, project_root)`가 담당한다(m4-01).
  공개 `resolve_runtime_path(env_name, default_path, legacy_path, *,
  environ)`는 시그니처와 legacy fallback 동작을 그대로 보존하는 별도 호환
  wrapper이며 `from_sources`의 field parser로 호출되지 않는다.
- **unknown key**: `merged` 기준 `ENV_PREFIX` 미등록 키는 exit 2(오타 방지,
  `cli_overrides` 오타도 포함).
- **캐시**: 기존 §4.4의 `get_settings`/`reset_settings_cache` 유지.

### 4.4 `config.py` facade(M3-01 필수 수정 — facade_adapter 투영)

기존 공개 심볼 42개(값·타입·이름 불변)를 유지한다. 각 상수는 모듈 최상단에서
`_facade_value(get_settings(), spec)`로 대입해 첫 import 시점이 materialize
시점이 되도록 해 §2의 CLI 지연 평가 순서를 보존한다:

```python
def _facade_value(settings: Settings, spec: FieldSpec) -> object:
    raw = getattr(settings, spec.name)
    return spec.facade_adapter(raw) if spec.facade_adapter is not None else raw
```

`facade_adapter is None`인 36개 필드(`PROJECT_ROOT` 포함)는 Settings 값을
그대로 대입하고, `facade_type=str`인 5개(§4.2)는 `str(Path)`로 투영해 기존
`str` 타입을 보존한다. `_env_bool`/`_env_enum`은 env를 읽지 않는 순수
파서로 남기고(REQ-002.2 위반 소지 제거), 기존 import 이름만 유지한다.
`resolve_runtime_path`는 `(env_name, default_path, legacy_path, *,
environ)` 시그니처와 legacy fallback 동작(§4.3 원문)을 그대로 보존한 채
`config.py`에 잔류하는 별도 공개 호환 심볼이다 — `FIELD_SPECS`의 path
parser 역할은 신규 private `_parse_runtime_path(raw, project_root)`가
대신하며(m4-01), 두 함수는 서로 호출하지 않는다.

### 4.5 테스트

- `tests/unit/test_settings.py`: valid/boundary/conflict fixture(validator별
  최소 1건), unknown-key exit 2, empty-string==unset, bool truth table 6종,
  **`Settings.from_sources()`(인자 없음) 성공** — 모든 default가 모든
  field/model validator를 통과함을 실행으로 증명(Field_Spec_Inventory.md
  "default pass" 컬럼과 동형).
- `tests/unit/test_settings_inventory.py`: `len(FIELD_SPECS)==41`,
  `{s.name}==set(Settings.model_fields)`, §4.1 "세 경로 중 하나" 불변식,
  **`scripts/generate_field_spec.py --check`가 생성하는 41-field 표와
  3-row/5-column `MODEL_VALIDATORS` 표(§4.1.1) 모두 checked-in
  `docs/generated/settings_field_spec.md`와 diff 0**(R1-MAJ-02 필수 수정,
  재개 사이클 2), 동일 입력으로 2회 연속 실행한 렌더링 바이트가 동일함(결정론)도
  함께 확인.
- `tests/unit/test_settings_from_sources.py`(신규, M2-03): `base_environ`
  단독 키 보존, `cli_overrides`가 동일 키를 덮어씀, M3 flag 4개가 무관한
  override 존재 시에도 유지(§5.3 연결).
- `tests/integration/test_cli_entrypoints.py`(확장): 3 CLI override
  subprocess로 facade 값 반영 확인(§5).
- `tests/unit/test_settings_facade_compat.py`(신규, M3-01): 42개 공개 심볼
  전수(41 필드+`resolve_runtime_path`)를 이전 `config.py` snapshot과 비교 —
  41개 필드는 name→`type()`/값(path 5개는 `type() is str`, `PROJECT_ROOT`는
  `type() is Path`), `resolve_runtime_path`는 값 비교 대신
  `inspect.signature()` 일치로 검증한다(m4-01, 아래 bullet과 동일 근거) —
  §4.2 facade_type 표와 동형 검증.
- `tests/unit/test_config.py`(기존 유지, m4-01): `resolve_runtime_path`의
  `(env_name, default_path, legacy_path, *, environ)` signature를
  `inspect.signature()`로 고정 검증하고, 기존 4개 legacy fallback 시나리오
  (§2 승계)가 무수정 통과함을 재확인 — 42번째 호환 심볼은 name/type/값이
  아니라 signature+동작으로 검증한다(Field_Spec_Inventory.md와 동기화).

## 5. 세 CLI 흐름과 exit 매트릭스 (M1-03)

### 5.1 통일 흐름

세 CLI 모두: `parse_args -> build cli_overrides mapping(SIMPLE_QNA_RAG_* 키만) ->
Settings.from_sources(base_environ=os.environ, cli_overrides=cli_overrides) ->
consumer factory(settings)`. **`os.environ`에 쓰지 않는다** — override는
별도 mapping으로 유지되고 `from_sources`가 병합한다(이중 계약 제거, 후자로
고정). `cli/index_documents.py`의 module top-level `config` import 후 전역을
직접 변경하는 경로를 제거하고, 다른 두 CLI처럼 "override 적용 직후 지연
import"로 통일한다.

### 5.2 exit 매트릭스

| CLI | valid | invalid enum/range/conflict | unknown env key | override 적용 |
|---|---|---|---|---|
| `simple-qna-rag-query` | 0 | **2**(SettingsError) | 2 | 있음, subprocess 검증 |
| `simple-qna-rag-index` | 0 | **2** | 2 | 있음, subprocess 검증 |
| `simple-qna-rag-web`(기본) | 0(serve) | 0(serve, `/health/ready` 503 `settings_invalid`) | 0(동일) | 있음 |
| `simple-qna-rag-web --check-config` | 0(stdout JSON) | **2**(stderr, 모델/엔진 로드 없음) | 2 | 있음 |

query/index의 초기화 이후 실패는 기존과 동일하게 exit 1을 유지한다(exit 2는
`SettingsError` 전용).

### 5.3 테스트

- `tests/integration/test_cli_entrypoints.py`: 위 4행 × {valid, invalid,
  unknown-key, override} subprocess matrix, invalid 케이스에 "engine/index
  constructor not called" assertion 추가. **신규 행**: 기존 env(override
  대상 아닌 값)와 CLI override를 동시에 준 subprocess에서 override 대상만
  바뀌고 비-overridden 필드·M3 flag 4개는 보존됨을 assert한다.
- `tests/integration/test_check_config_cli.py`: `--check-config` stdout JSON
  schema, exit 0/2, secret/절대경로 미노출.

## 6. Payload-safe logging (M1-04)

### 6.1 output-surface disposition(185건/9파일 print + 전체 출력 표면, m2-02 필수 수정)

`print(...)`만 세던 감사 범위를 4종으로 확대한다.
`scripts/logging_callsite_audit.py`가 `ast`로 스캔해
`docs/generated/logging_callsite_disposition.json`을 생성한다:

| 종류 | 탐지 대상 |
|---|---|
| `print` | `print(...)` 호출(기존 185건/9파일) |
| `logging` | `logging.<method>(...)`, `getLogger(...)` 반환값(alias 추적)의 `.<method>(...)` |
| `stdio_write` | `sys.stdout.write(...)`, `sys.stderr.write(...)` |
| `uvicorn_logger` | `uvicorn.run(access_log=, log_config=)` 인자, `logging.getLogger("uvicorn.*")` 참조 |

**안정 identity**(리뷰 지적 정정): key는
`f"{module_path}::{enclosing_qualname}#{ordinal}"`(같은 enclosing 함수/모듈 내
같은 종류의 등장 순번, 1-index)이다 — 앞부분 편집으로 line 번호가 밀려도
drift하지 않는다. `line`은 진단용 보조 필드로만 남는다.

| 분류 | 조건 | 처리 |
|---|---|---|
| `REPLACE` | `src/simple_qna_rag/{web,agent.py,rag_engine.py,observability}` 요청 처리 경로 | `log_event(...)`로 치환, 원문 제거 |
| `KEEP_CLI` | `cli/*.py` 사용자 대면 stdout, uvicorn access log(로컬용) | 유지 |
| `REMOVE` | debug 잔재, 중복 출력 | 삭제 |

Iteration 1이 지목한 2건은 `REPLACE` 확정: `web/server.py:96`(질문 원문) →
`log_event("request_end", ...)`(question 미포함). `agent.py:228`(검색어 원문) →
`log_event("routing", ...)`(검색어 미포함).

`tests/unit/test_logging_callsite_disposition.py`: `print` 합계 185 일치, 4종
전체 `UNCLASSIFIED` 0, `REPLACE` 대상에 `print(`/미치환 `logging.*`/
`sys.std{out,err}.write(` 잔존 0. `tests/integration/
test_output_surface_capture.py`(신규): §9.1 5경로를 `capsys`+`caplog`로 캡처해
**실제 실행 중** stdout/stderr/logging record에 금지 payload 0건임을 동적으로
확인 — 정적 grep이 놓치는 f-string 조합의 보완 증거.

### 6.2 positive schema(허용 key만, 그 외 거부하지 않고 무시 X — 예외)

```python
ALLOWED_EVENTS = Literal[
    "request_start", "request_end", "routing", "web_search",
    "retrieval", "generation", "startup", "readiness",
]
```

| event | 필수 key | 타입 |
|---|---|---|
| 공통 | `timestamp`,`level`,`event`,`service`,`version`,`request_id` | str/str/enum/str/str/str\|None |
| `request_start` | `route`,`method` | enum(`rag`,`health`)/str |
| `request_end` | + `status_code`,`duration_ms` | int/float |
| `routing` | `decision`,`confidence` | enum/float(0..1, 소수 3자리 clamp) |
| `web_search`/`retrieval`/`generation` | `stage`,`duration_ms`,`error_code` | enum/float/enum\|None |
| `startup`/`readiness` | `reason` | enum\|None, `request_id=None` 허용(§9.2) |

금지 key(허용 목록 밖 전부 거부): `question|answer|context|sources_content|
prompt|exception_text|` 및 절대경로 정규식 `^(/Users/|/home/|C:\\)`. 런타임
`log_event(..., metrics_registry: CollectorRegistry | None = None)`는 금지
key를 만나면 예외 없이 drop 후 `metrics_registry.logging_dropped_fields_total`
(Counter, labels 없음, §7.2 family 7, m4-02)을 증가시킨다(REQ-003.4).
`metrics_registry`는 `app.state.metrics_registry`(§7.4
`build_metrics_registry()` 반환값)를 그대로 주입받으며, `None`이면(registry가
아직 준비되지 않은 극초기 경로) drop만 수행하고 counter는 증가시키지 않는다.
개발/CI 전용 `log_event_strict`만 `ValueError`를 던진다.

### 6.3 request-id

- 서버 생성 기본값: UUID4.
- 클라이언트 `X-Request-Id` 재사용 조건: 정규식 `^[A-Za-z0-9_-]{1,64}$` 통과
  시에만 재사용, 불통과 시 서버가 새 UUID4로 대체(요청 거부하지 않음).
- `request_context.py::REQUEST_ID: ContextVar[str | None]`, 미들웨어가
  `try/finally`로 설정/리셋.

### 6.4 테스트

- `tests/unit/test_observability_logging.py`: 이벤트별 positive schema,
  금지 key drop(예외 없음) + 주입한 fresh `metrics_registry`의
  `logging_dropped_fields_total` 증가(m4-02), `log_event_strict` 예외 케이스,
  handler 예외 삼킴(mock 강제 실패).
- `tests/unit/test_request_id.py`: 유효/무효 헤더 6종 매트릭스.

## 7. Bounded metrics (M1-05)

### 7.1 dependency

`requirements.txt`에 `prometheus-client==0.23.1`(§2 확인 버전)을 direct
dependency로 추가하고 §8 lock 재생성 대상에 포함한다.

### 7.2 family 표(정확한 name/type/labels/allowed values/buckets, M2-04 fallback 추가)

REQ-004.1은 request/response status, stage duration/error,
**route/fallback**, readiness reason을 요구한다 — Iteration 2가 지적한 대로
`rag_fallback_total`이 누락돼 있었다. `agent.py`의 web-search 폴백과
`rag_engine.py`의 MMR stored→embed 강등(config.py 주석 "검증 실패 시에도
엔진은 자동으로 embed 강등한다")을 계측 대상으로 고정한다.

| family | type | labels | allowed values | buckets |
|---|---|---|---|---|
| `rag_requests_total` | Counter | `route`,`status` | route∈{`rag`,`health`}; status∈{`2xx`,`4xx`,`5xx`} | — |
| `rag_request_duration_seconds` | Histogram | `route` | route∈{`rag`,`health`} | `[0.05,0.1,0.25,0.5,1,2,5,10]`(8, +Inf 자동 추가) |
| `rag_stage_duration_seconds` | Histogram | `stage` | stage∈{`routing`,`web_search`,`retrieval`,`generation`} | 위와 동일(8) |
| `rag_stage_errors_total` | Counter | `stage`,`error_code` | stage 위와 동일; error_code∈{`timeout`,`upstream`,`validation`,`internal`} | — |
| `rag_readiness` | Gauge | `reason` | reason∈{`ok`,`settings_invalid`,`engine_init_failed`,`static_mount_failed`,`other`}(§3.4와 동기화, 5종) | — |
| `rag_fallback_total`(신규) | Counter | `kind`,`reason` | kind∈{`web_search`,`mmr_vector_source`}; reason∈{`low_confidence`,`empty_retrieval`,`validation_failed`,`other`} | — |
| `logging_dropped_fields_total`(신규, m4-02) | Counter | 없음 | — | — |

7번째 family `logging_dropped_fields_total`은 §6.2 `log_event`가 금지 key를
drop할 때 증가시키는 REQ-003.4 근거다 — 별도 process-local 변수가 아니라
`build_metrics_registry()`(§7.4)가 등록하는 정식 Prometheus Counter이며
labels가 없어 항상 sample 정확히 1개를 노출한다.

label allowlist 밖 값은 `_clamp_label(value, allowed, default="other")`가
`"other"`로 치환한다(prometheus_client는 라벨을 강제하지 않으므로 wrapper
책임, §2 확인 사실).

### 7.3 이론 상한식(전체 조합 기준)과 재계산된 fresh-registry 실측(M2-04)

이전 "67"은 **관측된 조합**이지 가능한 label 조합 전부의 안전 상한이
아니었다(리뷰 지적). 두 수치를 분리한다.

**이론 상한**(created 제외 — `disable_created_metrics()`가 생성 전 호출됨, §7.4):

```
max_samples = |route|*|status|                       # rag_requests_total = 2*3 = 6
            + |route|*(len(buckets)+1+2)              # duration: 8경계+자동 +Inf+_count+_sum = 2*11 = 22
            + |stage|*(len(buckets)+1+2)               # stage_duration = 4*11 = 44
            + |stage|*|error_code|                     # stage_errors 전체 조합 = 4*4 = 16
            + |reason|                                 # readiness = 5
            + |kind|*|fallback_reason|                 # fallback 전체 조합 = 2*4 = 8
            + 1                                         # logging_dropped_fields_total, labels 없음(m4-02) = 1
                                                          # 합계 = 102
```

이전 설계의 `(buckets+2)`는 Histogram이 마지막 경계에 `+Inf`를 자동
추가한다는 사실을 놓쳐 조합당 샘플 수를 1개 과소산정했다(10 vs 실제 11) —
이 개정에서 정정한다.

**fresh-registry 실측**: 새 `CollectorRegistry()`에 §7.2 7-family(신규
`logging_dropped_fields_total` 포함, m4-02)를 등록하고 1,000 unique 합성
요청(stage_errors 8%·fallback 5% 발생률로 전체 조합을 최소 1회 이상 강제
커버) + 최소 1건의 금지 key drop 이벤트를 주입한 sample 합계는 **102**로
이론 상한과 정확히 일치한다(Histogram은 라벨 1회 관측만으로 전체 bucket
행이 노출되고, Counter/Gauge도 1,000건 안에서 전체 조합이 강제 커버되므로
상한이 곧 실측치가 되며, labels 없는 Counter는 관측 1회만으로 sample 1개가
항상 노출된다). 상한 150 대비 여유 48.
`tests/unit/test_observability_metrics.py`가 위 시뮬레이션 후 family=7,
sample 합계=102, `_created` 0건을 assert한다.

### 7.4 process 설정과 registry factory 분리(m2-03 필수 수정)

"정확히 한 번"(process 전역 설정)과 "재호출 시 재사용"(registry별 collector
생성)의 계약 충돌(리뷰 지적)을 두 symbol로 분리한다.

```python
_process_metrics_configured = False

def configure_process_metrics() -> None:
    """process-global, 1회만 실제 동작. Counter/Histogram/Gauge 생성 전에
    disable_created_metrics()를 호출한다. registry와 무관."""
    global _process_metrics_configured
    if _process_metrics_configured:
        return
    disable_created_metrics()
    _process_metrics_configured = True

def build_metrics_registry(registry: CollectorRegistry | None = None) -> CollectorRegistry:
    """registry별 factory — 호출마다 §7.2 7-family를 등록한 CollectorRegistry를
    반환한다(fresh registry로 테스트 격리). configure_process_metrics()를
    스스로 호출하지만 실제 disable_created_metrics()는 최초 1회만 동작.
    `logging_dropped_fields_total`(labels 없음, m4-02)은 다른 6-family처럼
    reg에 등록됨과 동시에 `reg.logging_dropped_fields_total` 속성으로도
    노출된다 — registry 자체가 유일한 DI 지점(app.state.metrics_registry)이므로
    log_event(§6.2)는 별도 counter 객체를 전달받지 않고 이 속성을 통해
    `.inc()`한다."""
    configure_process_metrics()
    reg = registry if registry is not None else CollectorRegistry()
    ...  # §7.2 6-family + logging_dropped_fields_total(Counter, labels 없음)을 reg에 등록
    reg.logging_dropped_fields_total = ...  # 위에서 생성한 Counter 인스턴스 재노출(m4-02)
    return reg
```

app bootstrap(lifespan)은 `app.state.metrics_registry =
build_metrics_registry()`를 호출한다(전역 REGISTRY 미사용). `/metrics`
라우트는 `app.state.metrics_registry`만 참조한다(모듈 전역 직접 참조 금지).

### 7.5 테스트

`tests/unit/test_observability_metrics.py`: §7.3 상한/실측(102) 회귀,
allowlist 밖 라벨 clamp, `_created` 샘플 0건, `configure_process_metrics()`
반복 호출이 `disable_created_metrics()`를 1회만 부르는지(mock call count),
`build_metrics_registry()`를 N번 호출해 N개의 독립 registry가 서로 샘플을
공유하지 않는지(fresh registry 격리), `logging_dropped_fields_total`이
labels 없이 family 7로 등록되고 `reg.logging_dropped_fields_total.inc()`가
`/metrics` scrape 출력에 반영되는지(m4-02).

## 8. Dependency lock 재현성 (M1-06, Iteration 1에서 폐쇄 — 변경 없음)

**8.1 도구 pin**: `uv==0.8.15`(§2 확인 버전)를 CI `Install uv` step에 명시
고정.

**8.2 canonical 비교**: `scripts/compile_lock.sh`가 §2의 정확한 명령
(`--extra-index-url .../whl/cpu --index-strategy unsafe-best-match
--generate-hashes --no-annotate`)을 temp 파일명으로 2회 실행하고, 선두
`^#` 연속 라인(명령/타임스탬프 헤더)을 제거한 본문의 `sha256`이 다르면 CI
실패. 커밋된 `requirements.lock`도 동일 canonicalizer로 drift 검출.

**8.3 snapshot artifact**: `scripts/dependency_snapshot.py`가 canonical
JSON(`schema_version`,`profile`,`uv_version`,`python_version`,`node_version`,
`lock_sha256_canonical`,`package_lock_sha256`,`package_count` — `generated_at`
제외, hash 안정성)을 stdout과 `dependency_snapshot.json`에 쓴다. CI에
`actions/upload-artifact@v4`(`retention-days: 90`) 추가, cache key를
`hashFiles('requirements.lock')`로 변경.

**8.4 테스트**: `tests/unit/test_dependency_lock.py`(`--require-hashes` 형식,
nvidia-\* 0건, `prometheus-client` 존재), `tests/unit/test_dependency_snapshot.py`
(스키마 필드 완전성, canonical JSON key 정렬).

## 9. Logging/health acceptance matrix (M1-07)

### 9.1 request start/end 소유자

`RequestContextMiddleware`(ASGI, `observability/request_context.py`)가
start/end 이벤트의 단일 소유자다. 아래 5경로 각각 정확히 1쌍(start+end)을
방출한다:

| 경로 | status | 비고 |
|---|---|---|
| 정상 `/rag` | 200 | |
| 미존재 라우트 | 404 | FastAPI 기본 핸들러 경유해도 미들웨어가 감쌈 |
| `/rag` validation 실패 | 422 | pydantic body validation |
| engine 미준비(`app.state.engine is None`) | 503 | `/rag` 핸들러 진입 시 즉시 |
| 핸들러 내부 예외 | 500 | `end` 이벤트에 `error_code=internal` 포함 후 재발생(FastAPI 표준 500 응답 유지) |

`tests/integration/test_request_logging_matrix.py`가 injected sink(list
capture)로 5행 각각 start/end 쌍 존재, 금지 payload 0을 assert한다.

### 9.2 startup/readiness process event

`request_id`가 없는 프로세스 레벨 이벤트(`startup`,`readiness`)는 §6.2 공통
스키마를 쓰되 `request_id=None`을 명시적으로 허용하는 별도 분기로
표현한다 — "요청 범위 request ID"는 request-scope 이벤트에만 필수이고
process 이벤트는 예외다.

## 10. M3 14-gate·baseline·vectorstore (M1-08)

### 10.1 gate 재사용(M2-05 필수 수정 — 실제 API로 재설계)

실제 시그니처: `evaluation.compare.evaluate_gates(payload: dict) -> dict`
(`payload={"retrieval":dict|None,"routing":dict|None,"answers":dict|None}`,
반환 `{"spec_version","overall_pass","items"}`, `item["pass"]`는
`bool | None` — `None`이 판정 불가이며 별도 `UNKNOWN` 문자열은 없다)과
`evaluation.reporting.write_report(payload, output_dir, name,
render_markdown=None) -> tuple[Path, Path]`(JSON+Markdown을 **한 번의 호출**로
함께 씀, `format` 인자 없음). 이 두 함수를 직접 조립하지 않고, M3가 이미
조립을 구현한 `evaluation.baseline.run_baseline(dataset_path, output_dir, *,
skip_routing=False, skip_answers=False, ...) -> dict`를 재사용한다 —
retrieval/routing/answers를 순서대로 실행하고 `evaluate_gates()` 결과를
`payload["gate_evaluation"]`에 넣은 뒤 `write_report(..., "baseline",
render_markdown=_render_baseline_markdown)`을 내부에서 정확히 한 번
호출하므로, M4.1은 payload 조립이나 렌더러를 새로 만들지 않는다(REQ-006.4
"같은 판정 모델"을 코드 재사용으로 만족).

**M5-01 필수 수정 — 경로 source/type 단일화(재개 사이클 1)**: wrapper가 쓰는
vectorstore 경로의 source는 **legacy facade `simple_qna_rag.config.VECTORSTORE_PATH`
하나**로 고정한다(§4.2 — 값은 `str`). 이 스크립트를 위해 별도로 typed
`Settings()`/`get_settings()`를 새로 생성하지 않는다 — 스크립트는 `config.py`
facade만 참조하는 기존 CLI 관례(§5.1)를 따르는 얇은 wrapper이고, `Settings`
캐시(`get_settings`)에 대한 추가 결합을 만들 이유가 없다. `build_vectorstore_fingerprint`가
요구하는 `Path`로의 변환은 **호출 경계 정확히 한 곳**, 아래 `_vectorstore_fingerprint()`
헬퍼 내부에서만 `Path(VECTORSTORE_PATH)`로 수행한다 — facade 필드 자체의 공개
타입(`str`, REQ-002.3)은 바꾸지 않고, 정규화된 `Path` 값이 헬퍼 밖으로
새어나가지도 않는다. Iteration 4/5가 지적한 "함수 이름/arity만 고치고 실제
인자 타입 연결을 빠뜨리는" 재발을 막기 위해 pre/post 두 호출 지점이 이 헬퍼
하나만 부르도록 고정한다:

```python
# scripts/run_m4_regression_gate.py (신규, 얇은 wrapper — 조립 로직 없음)
from simple_qna_rag.config import VECTORSTORE_PATH  # legacy facade, str(§4.2) — 유일한 source
from evaluation.baseline import run_baseline
from evaluation.reporting import build_vectorstore_fingerprint  # index.faiss/index.pkl 2-file SHA-256 dict(index_faiss_sha256,index_pkl_sha256), 1-인자(vectorstore_path: Path)

BASELINE_FILES = (
    Path("evaluation/baselines/m3_initial.json"),
    Path("evaluation/baselines/m3_initial.md"),
)

def _hash_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _vectorstore_fingerprint() -> dict:
    # M5-01: str(legacy facade) -> Path 정규화는 이 함수 안에서 정확히 한 번만 일어난다.
    return build_vectorstore_fingerprint(Path(VECTORSTORE_PATH))

def main(argv: list[str] | None = None) -> int:
    pre_baseline = {p: _hash_bytes(p) for p in BASELINE_FILES}
    pre_vectorstore = _vectorstore_fingerprint()
    immutable_ok = False
    try:
        payload = run_baseline(
            dataset_path=Path("evaluation/datasets/golden.jsonl"),  # 공식 dataset, ci.yml validate 대상과 동일
            output_dir=Path("evaluation/reports/m4_regression"),     # M3 baseline과 분리 — REQ-006.2 보존
        )
    except RuntimeError as exc:  # RUN_LIVE_LLM_TESTS=1 opt-in 미충족(run_baseline 기존 계약)
        print(str(exc), file=sys.stderr)
        return 2  # gate FAIL(1)과 환경 미충족(2)을 구분
    finally:
        post_baseline = {p: _hash_bytes(p) for p in BASELINE_FILES}
        post_vectorstore = _vectorstore_fingerprint()
        immutable_ok = pre_baseline == post_baseline and pre_vectorstore == post_vectorstore
    if not immutable_ok:
        print("baseline/vectorstore mutated during regression run", file=sys.stderr)
        return 1
    overall_success = payload["overall_success"]
    overall_pass = payload["gate_evaluation"]["overall_pass"]
    return 0 if (overall_success is True and overall_pass is True) else 1
```

`pre_vectorstore`/`post_vectorstore`는 각각
`{"index_faiss_sha256": str, "index_pkl_sha256": str}`이고 `==` 비교는 두
key의 exact equality다(M4-01). 3-인자
`evaluation.fingerprint.collect_fingerprint(data_dir, vectorstore_path,
dataset_path)`는 corpus/dataset/git metadata까지 암묵적으로 섞어 §10.3의
"canonical 2-file만, 범위 확장 없음" 계약을 벗어나므로 이 wrapper에서
완전히 제거하고 사용하지 않는다.

exit 0은 `overall_success is True and gate_evaluation.overall_pass is True`
둘 다일 때만이다(M3-02) — `run_baseline()`의 권위 있는 `overall_success`
(stage 성공 + fingerprint 일치)를 무시하지 않는다. `overall_pass`는 `pass is
None`인 gate가 있으면 False로 계산되어 판정 불가를 PASS로 승격하지 않는다.
14 gate 전부 True이나 `overall_success=False`인 fixture는 exit 1, exit
2는 `RUN_LIVE_LLM_TESTS` 미설정 등 환경 제약(가이드 예외 규칙, gate 판정과
별개).

### 10.2 baseline bytes 보존(M3-02 — §10.1 symbol에 연결)

§10.1의 `try/finally`가 `BASELINE_FILES`(`m3_initial.json`,`m3_initial.md`)
각각의 sha256을 `_hash_bytes()`로 pre/post 비교해 `immutable_ok`에 반영한다.
`git diff --exit-code -- evaluation/baselines/m3_initial.*`를 CI에 이중
확인으로 추가한다(Plan §5 기존 vendor diff 패턴 재사용).

### 10.3 vectorstore 불변성(M3-02 — canonical 2-file 정정)

`evaluation.reporting.build_vectorstore_fingerprint(vectorstore_path) -> dict`
(실제 API, `index.faiss`/`index.pkl`만 읽고 FAISS를 역직렬화하지 않음)가
디렉터리 전체 파일 목록이 아니라 canonical 두 파일의 SHA-256을
`{"index_faiss_sha256": str, "index_pkl_sha256": str}`로 반환한다(M4-01 —
3-인자 `evaluation.fingerprint.collect_fingerprint`는 이 wrapper에서 사용하지
않는다). §10.1의 `try/finally`가 pre/post 두 dict를 `==` exact equality로
비교해 불일치 시 `immutable_ok=False`로 gate를 즉시 실패시키고 mutation
없이 종료한다(재시도 없음). 디렉터리 전수 hash가 필요하면 별도
canonicalizer를 새로 설계해야 하며 이 문서는 확장하지 않는다.

### 10.4 테스트

`tests/integration/test_m3_regression_gate.py`(M4-01 재설계):

1. **실제 arity 고정**: `inspect.signature(build_vectorstore_fingerprint)`가
   `(vectorstore_path)` 단일 positional 인자임을 검증 — 3-인자
   `collect_fingerprint`가 다시 호출부에 섞여 들어가는 회귀를 막는다.
2. **exit 0/1/2 matrix**: valid dataset+14 gate 전부 PASS+
   `overall_success=True`→0, `RUN_LIVE_LLM_TESTS` 미설정(`RuntimeError`)→2,
   그 외 gate 실패/불변성 위반→1.
3. **신규 fixture(M3-02 유지)**: 14 gate `pass=True`이지만
   `payload["overall_success"]=False`(fingerprint mismatch)인 경우 exit 1임을
   mock `run_baseline()`으로 확인 — `overall_pass`만 보던 이전 계약의 회귀
   방지.
4. **신규 fixture(M4-01, two-file mutation)**: `index.faiss`/`index.pkl` 중
   한 파일만 변경돼도(양쪽 다 변경되는 경우 포함) `pre_vectorstore !=
   post_vectorstore`가 되어 `run_baseline()` 결과와 무관하게 exit 1.
5. `gate_evaluation.items`가 14개, `pass: None` 포함 payload가
   `evaluate_gates()`에서 `overall_pass=False`로 떨어짐(회귀), JSON/Markdown
   parity, baseline/vectorstore pre-post hash 동일성(dict exact equality).
6. **legacy facade 실제 호출 integration case(R1-MAJ-01/R2-MAJ-01 필수
   수정, 재개 사이클 3)**: signature introspection(#1)과 결과-동등성만 보는
   검증(설계 재개 사이클 1 버전) 둘 다, 헬퍼가 두 파일을 직접 읽어 같은
   dict를 만드는 대체 구현도 통과시킬 수 있다는 점이 Iteration 4/5와 설계
   재개 사이클 1에서 반복된 재발 원인이었다. 설계 재개 사이클 2 버전은 그
   문제를 스크립트 모듈이 참조하는 `build_vectorstore_fingerprint` 심볼을
   실제 함수를 감싸 호출을 기록하는 recording wrapper로 바꿔치기해 고치려
   했으나, 그 뒤 delegation을 재확인하는 `recording_wrapper.wraps is
   real_build_vectorstore_fingerprint` assertion이 Python 3.11
   `unittest.mock.Mock`에 없는 공개 `.wraps` 속성을 참조해 `AttributeError`로
   항상 실패했다(`Mock`은 이 값을 비공개 `_mock_wraps`에만 보관한다,
   R2-MAJ-01). 이번 버전은 그 assertion을 제거하고, 공개 Mock API와 반환값
   비교만으로 같은 4항목을 증명한다.

   ```python
   from unittest import mock
   from evaluation.reporting import (
       build_vectorstore_fingerprint as real_build_vectorstore_fingerprint,
   )

   def test_vectorstore_fingerprint_invokes_real_api(monkeypatch, tmp_path):
       (tmp_path / "index.faiss").write_bytes(b"faiss-bytes")
       (tmp_path / "index.pkl").write_bytes(b"pkl-bytes")
       real_fn = real_build_vectorstore_fingerprint
       spy = mock.Mock(wraps=real_fn)
       monkeypatch.setattr(
           run_m4_regression_gate, "build_vectorstore_fingerprint", spy
       )
       monkeypatch.setattr(run_m4_regression_gate, "VECTORSTORE_PATH", str(tmp_path))

       result = run_m4_regression_gate._vectorstore_fingerprint()

       spy.assert_called_once_with(Path(str(tmp_path)))
       call_arg = spy.call_args.args[0]
       assert isinstance(call_arg, Path)
       assert call_arg == Path(str(tmp_path))
       expected = real_fn(tmp_path)
       assert result == expected
       assert set(result) == {"index_faiss_sha256", "index_pkl_sha256"}
       assert result["index_faiss_sha256"] == hashlib.sha256(
           (tmp_path / "index.faiss").read_bytes()
       ).hexdigest()
       assert result["index_pkl_sha256"] == hashlib.sha256(
           (tmp_path / "index.pkl").read_bytes()
       ).hexdigest()
   ```

   `real_fn`(= 실제 `evaluation.reporting.build_vectorstore_fingerprint`)을
   먼저 보관한 뒤 `spy = mock.Mock(wraps=real_fn)`으로 감싸는 순서는
   유지한다 — `wraps=`는 `Mock` 생성자의 공개 인자이며, 그 자체는 "호출되면
   `real_fn`을 실행하고 그 반환값을 돌려준다"는 문서화된 공개 동작을
   설정하는 유효한 계약이다. 재개 사이클 2에서 깨졌던 부분은 그 뒤에 생성된
   Mock 인스턴스에서 delegation을 **공개 속성으로 재확인**하려 한 시도뿐이었다
   — 이번 버전은 그 시도를 하지 않는다.
   `monkeypatch.setattr(run_m4_regression_gate, "build_vectorstore_fingerprint",
   spy)`는 스크립트 모듈이 `from evaluation.reporting import
   build_vectorstore_fingerprint`로 바인딩한 바로 그 심볼을 바꿔치기하므로
   `_vectorstore_fingerprint()` 내부 호출은 반드시 `spy`를 거친다(원본
   `config.py`/실제 runtime vectorstore는 건드리지 않음, REQ-006.2 보존).
   위 assert들이 필수 수정 4항목을 각각 봉쇄한다:
   `spy.assert_called_once_with(Path(str(tmp_path)))`가 "정확히 1회 호출,
   유일한 인자가 `Path(str(tmp_path))`"를 한 번에 고정하고,
   `spy.call_args.args[0]`에 대한 `isinstance(call_arg, Path)`/`call_arg ==
   Path(str(tmp_path))`가 인자 타입과 값을 다시 한번 명시적으로 확인한다.
   "wrapper가 실제 `evaluation.reporting.build_vectorstore_fingerprint`를
   호출함"은 Mock의 내부 속성이 아니라 **동작의 결과**로 증명한다:
   `real_fn`을 독립적으로 한 번 더 호출한 `expected = real_fn(tmp_path)`와
   `spy`를 통해 나온 `result`가 `==`로 일치함을 확인한다 — `real_fn`은
   canonical 파일을 읽어 SHA-256을 계산하기만 하는 결정론적 함수이므로,
   `spy.assert_called_once_with(...)`(정확한 인자로 정확히 1회 호출됨)와
   이 등가성을 함께 보면 "`spy` 호출이 실제로 `real_fn`을 실행해 그 반환값을
   그대로 전달했다"는 것과 동치다. `set(result)`와 두 `hexdigest()` 비교가
   "정확한 two-key와 SHA-256"을 각각 증명한다. 이 케이스가 통과해야 #2~#5의
   mock 기반 pre/post 비교 fixture가 "실제로 호출되는 함수"를 비교하고
   있음이 증명된다.

## 11. MINOR/TRIVIAL 폐쇄

### 11.1 m1-01 — Phase 번호 정합화

Plan.md의 Phase 0~3(기준선/lock+settings/logging+metrics/health+통합)에
맞춰 이 문서의 §3~§10을 Phase 1(§4,§5,§8 — settings/CLI/lock),
Phase 2(§6,§7 — logging/metrics), Phase 3(§3,§9,§10 — health/통합 gate)로
귀속한다. Traceability의 evidence phase 컬럼도 동일 라벨을 쓴다.

### 11.2 m1-02/m2-01 — `/health` deprecation 계약(Sunset 확정)

`/health`(deprecated alias)는 응답 body를 `/health/ready` 판정에서 파생하되
기존 `status`/`rag_engine_initialized` key와 값 semantics(엔진 없어도 이전
버전은 `healthy`였음)를 한 release 보존한다. 응답 헤더: `Deprecation: true`,
**`Sunset: Fri, 06 Nov 2026 00:00:00 GMT`**(RFC 8594 HTTP-date, 설계 확정일
2026-08-08 + 90일). 제거 대상은 **패키지 버전 0.3.0**(M4.2 릴리스, 현재
`pyproject.toml` 0.2.5의 다음 minor)이며 Roadmap.md §"분할 실행"에 동일 값을
기록한다(Requirement REQ-005.3과 동기화). `tests/integration/
test_health_endpoints.py`가 두 헤더 값과 기존 body shape을 함께 검증한다.

### 11.3 m1-03/m2-03 — created-series 소유권

§7.4에서 `configure_process_metrics()`(process-global, 1회)와
`build_metrics_registry()`(registry별 factory)로 분리 확정했다 — 이 항목은
별도 재설명 없이 §7.4를 유일한 근거로 삼는다.

### 11.4 t1-01/t1-02

t1-01: 이전 문서의 "§4.2"(metrics wrapper) 오참조는 본 개정에서 metrics가
§7로 재배치되며 해소됐다(§7.4). t1-02: `patttern` 오탈자는 §11.3
"pattern"으로 정정 완료(본 문서에 재발 없음).

## 12. M4.2/M4.3 확장 seam(명시만)

- `evaluate_readiness()`(§3.4)는 `reason` 문자열 반환 순수 함수 — M4.2는
  `queue_saturated`/`orphan_worker`를 반환값에 추가만 하면 된다.
- `metrics.py`의 label allowlist(§7.2)는 상수 frozenset — M4.2 신규
  stage/error 값 추가 시 cardinality를 §7.3 공식으로 재계산해야 하며 이
  재계산 의무를 M4.2 설계 진입 조건으로 남긴다.
- `FIELD_SPECS`(§4.1)는 M4.3 index/container 필드를 동일 스키마로 추가
  가능하나 이 문서는 그 필드를 설계하지 않는다.

## 13. 회귀·호환성 보존

`/rag` 응답 schema, 3개 CLI entry point, M3 rollback 환경변수 값·이름 불변.
M3 baseline·runtime vectorstore는 읽기 전용(§10.2/10.3). 최종 검증 명령군은
Plan §5에 다음을 추가한다:

```bash
pytest tests/unit/test_settings.py tests/unit/test_settings_inventory.py \
  tests/unit/test_settings_facade_compat.py \
  tests/unit/test_observability_logging.py tests/unit/test_observability_metrics.py \
  tests/unit/test_logging_callsite_disposition.py tests/unit/test_dependency_lock.py \
  tests/unit/test_dependency_snapshot.py tests/unit/test_request_id.py \
  tests/integration/test_health_endpoints.py tests/integration/test_check_config_cli.py \
  tests/integration/test_cli_entrypoints.py tests/integration/test_request_logging_matrix.py \
  tests/integration/test_m3_regression_gate.py
python scripts/generate_field_spec.py --check
python scripts/logging_callsite_audit.py --check
bash scripts/compile_lock.sh --verify
git diff --exit-code -- evaluation/baselines/m3_initial.*
```
