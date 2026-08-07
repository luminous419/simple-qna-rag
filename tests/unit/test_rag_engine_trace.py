"""Tests for the null-safe instrumentation helpers in rag_engine.py
(M3-REQ-002, Design.md §6.4)."""

from __future__ import annotations

from simple_qna_rag.rag_engine import RetrievalTrace, _bump, _note


def test_bump_with_none_trace_is_a_no_op():
    # Must not raise, and there is nothing to inspect afterwards — the
    # contract is simply "no exception, no side effect."
    _bump(None, "some_counter")
    _bump(None, "some_counter", delta=5)


def test_note_with_none_trace_is_a_no_op():
    _note(None, "some message")


def test_bump_increments_counter_from_zero():
    trace = RetrievalTrace()
    assert trace.counters == {}
    _bump(trace, "hits")
    assert trace.counters["hits"] == 1
    _bump(trace, "hits")
    assert trace.counters["hits"] == 2


def test_bump_accepts_custom_delta():
    trace = RetrievalTrace()
    _bump(trace, "candidate_embedding_calls", 50)
    assert trace.counters["candidate_embedding_calls"] == 50
    _bump(trace, "candidate_embedding_calls", 3)
    assert trace.counters["candidate_embedding_calls"] == 53


def test_note_appends_messages_in_order():
    trace = RetrievalTrace()
    _note(trace, "first")
    _note(trace, "second")
    assert trace.notes == ["first", "second"]


def test_retrieval_trace_counters_and_notes_default_empty():
    trace = RetrievalTrace()
    assert trace.counters == {}
    assert trace.notes == []
    assert trace.stages == []


def test_independent_trace_instances_do_not_share_state():
    a = RetrievalTrace()
    b = RetrievalTrace()
    _bump(a, "x")
    _note(a, "note-a")
    assert b.counters == {}
    assert b.notes == []
