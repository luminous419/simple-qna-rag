"""M4.1 §7.5 — bounded metrics cardinality/label-safety tests."""

import itertools
from unittest import mock

from prometheus_client import CollectorRegistry

from simple_qna_rag.observability import metrics as M


def test_clamp_functions_pass_through_allowed_values():
    assert M.clamp_route("rag") == "rag"
    assert M.clamp_status("2xx") == "2xx"
    assert M.clamp_stage("routing") == "routing"
    assert M.clamp_error_code("timeout") == "timeout"
    assert M.clamp_readiness_reason("ok") == "ok"
    assert M.clamp_fallback_kind("web_search") == "web_search"
    assert M.clamp_fallback_reason("low_confidence") == "low_confidence"


def test_clamp_functions_replace_out_of_allowlist_values():
    assert M.clamp_route("bogus-route-value") not in ("bogus-route-value",)
    assert M.clamp_status("999") == "5xx"
    assert M.clamp_stage("bogus") == "generation"
    assert M.clamp_error_code("bogus") == "internal"
    assert M.clamp_readiness_reason("bogus") == "other"
    assert M.clamp_fallback_kind("bogus") == "web_search"
    assert M.clamp_fallback_reason("bogus") == "other"


def test_build_metrics_registry_registers_seven_families():
    reg = M.build_metrics_registry()
    for name in (
        "rag_requests_total",
        "rag_request_duration_seconds",
        "rag_stage_duration_seconds",
        "rag_stage_errors_total",
        "rag_readiness",
        "rag_fallback_total",
        "logging_dropped_fields_total",
    ):
        assert hasattr(reg, name), name


def test_fresh_registries_do_not_share_samples():
    reg_a = M.build_metrics_registry()
    reg_b = M.build_metrics_registry()
    reg_a.rag_requests_total.labels(route="rag", status="2xx").inc()

    from prometheus_client import generate_latest

    text_b = generate_latest(reg_b).decode("utf-8")
    assert 'route="rag"' not in text_b or "rag_requests_total" not in text_b.split(
        'route="rag"'
    )[0]
    # Simplest robust check: registry B's counter sample must be 0/absent.
    sample_b = reg_b.get_sample_value(
        "rag_requests_total", {"route": "rag", "status": "2xx"}
    )
    assert sample_b in (None, 0.0)


def test_configure_process_metrics_calls_disable_created_metrics_once():
    M._process_metrics_configured = False
    with mock.patch.object(M, "disable_created_metrics") as mocked:
        M.configure_process_metrics()
        M.configure_process_metrics()
        M.configure_process_metrics()
        assert mocked.call_count == 1
    M._process_metrics_configured = True  # restore for the rest of the suite


def test_no_created_series_samples_exposed():
    reg = M.build_metrics_registry()
    reg.rag_requests_total.labels(route="rag", status="2xx").inc()
    reg.rag_fallback_total.labels(kind="web_search", reason="low_confidence").inc()
    reg.logging_dropped_fields_total.inc()

    from prometheus_client import generate_latest

    text = generate_latest(reg).decode("utf-8")
    assert "_created" not in text


def test_logging_dropped_fields_total_has_no_labels_and_reflects_inc():
    reg = M.build_metrics_registry()
    reg.logging_dropped_fields_total.inc()
    reg.logging_dropped_fields_total.inc()
    value = reg.get_sample_value("logging_dropped_fields_total")
    assert value == 2.0


def _simulate_1000_requests(reg: CollectorRegistry) -> None:
    routes = tuple(M.ROUTES)
    statuses = tuple(M.STATUSES)
    stages = tuple(M.STAGES)
    error_codes = tuple(M.ERROR_CODES)
    reasons = tuple(M.READINESS_REASONS)
    kinds = tuple(M.FALLBACK_KINDS)
    fb_reasons = tuple(M.FALLBACK_REASONS)

    combo_cycle = itertools.cycle(itertools.product(stages, error_codes))
    fallback_cycle = itertools.cycle(itertools.product(kinds, fb_reasons))
    reason_cycle = itertools.cycle(reasons)

    for i in range(1000):
        route = routes[i % len(routes)]
        status = statuses[i % len(statuses)]
        reg.rag_requests_total.labels(route=route, status=status).inc()
        reg.rag_request_duration_seconds.labels(route=route).observe(0.1)

        stage = stages[i % len(stages)]
        reg.rag_stage_duration_seconds.labels(stage=stage).observe(0.2)

        if i % 12 == 0:  # ~8% error rate, forces full stage x error_code coverage
            stage_e, error_code = next(combo_cycle)
            reg.rag_stage_errors_total.labels(stage=stage_e, error_code=error_code).inc()

        if i % 20 == 0:  # ~5% fallback rate, forces full kind x reason coverage
            kind, fb_reason = next(fallback_cycle)
            reg.rag_fallback_total.labels(kind=kind, reason=fb_reason).inc()

        if i % 250 == 0:
            reg.rag_readiness.labels(reason=next(reason_cycle)).set(1)

    # force full coverage of the two 4x4 combination spaces deterministically
    for stage, error_code in itertools.product(stages, error_codes):
        reg.rag_stage_errors_total.labels(stage=stage, error_code=error_code).inc()
    for kind, fb_reason in itertools.product(kinds, fb_reasons):
        reg.rag_fallback_total.labels(kind=kind, reason=fb_reason).inc()
    for reason in reasons:
        reg.rag_readiness.labels(reason=reason).set(1)

    reg.logging_dropped_fields_total.inc()


def test_1000_unique_requests_sample_count_is_bounded_and_equals_102():
    reg = M.build_metrics_registry()
    _simulate_1000_requests(reg)

    from prometheus_client import generate_latest
    from prometheus_client.parser import text_string_to_metric_families

    text = generate_latest(reg).decode("utf-8")
    families = list(text_string_to_metric_families(text))
    sample_count = sum(len(f.samples) for f in families)

    assert sample_count == 102
    assert sample_count <= 150
