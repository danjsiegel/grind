from __future__ import annotations

import pytest

from grind.engine.state_machine import (
    TERMINAL_STATES,
    finalized_actionable_finding_set,
    is_terminal,
    is_valid_transition,
)
from grind.models import (
    DecidedBy,
    FindingCategory,
    FindingConfidence,
    FindingSeverity,
    FindingStatus,
    RunState,
)
from grind.models.disposition import Disposition
from grind.models.finding import Finding


# ── helpers ──────────────────────────────────────────────────────────────────

RUN_ID = "run_20260516_120000_abcdef01"


def _finding(
    fid: str,
    severity: FindingSeverity,
    status: FindingStatus = FindingStatus.OPEN,
) -> Finding:
    return Finding(
        finding_id=fid,
        run_id=RUN_ID,
        stage_id="stage-001",
        stable_id="a" * 16,
        title="test finding",
        severity=severity,
        confidence=FindingConfidence.PROVEN,
        category=FindingCategory.CORRECTNESS,
        rationale="rationale",
        exact_fix_action="fix it",
        status=status,
    )


def _disposition(
    finding_id: str,
    decided_by: DecidedBy,
    iteration: int = 1,
) -> Disposition:
    return Disposition(
        disposition_id=f"d-{finding_id}",
        finding_id=finding_id,
        stage_id="stage-002",
        iteration=iteration,
        decided_by=decided_by,
        decision=FindingStatus.OPEN,
        justification="confirmed at src/foo.py:42",
    )


# ── Terminal states ───────────────────────────────────────────────────────────

def test_terminal_states_contains_expected() -> None:
    assert RunState.COMPLETED in TERMINAL_STATES
    assert RunState.FAILED in TERMINAL_STATES
    assert RunState.ABORTED in TERMINAL_STATES
    assert len(TERMINAL_STATES) == 3


def test_non_terminal_states_not_in_terminal() -> None:
    non_terminals = {s for s in RunState} - TERMINAL_STATES
    assert RunState.PLANNING in non_terminals
    assert RunState.ACTING in non_terminals
    assert RunState.AWAITING_OPERATOR in non_terminals


def test_is_terminal_true_for_terminal_states() -> None:
    for state in (RunState.COMPLETED, RunState.FAILED, RunState.ABORTED):
        assert is_terminal(state)


def test_is_terminal_false_for_non_terminal_states() -> None:
    for state in RunState:
        if state not in TERMINAL_STATES:
            assert not is_terminal(state)


# ── Terminal-state immutability ───────────────────────────────────────────────

@pytest.mark.parametrize("terminal", [RunState.COMPLETED, RunState.FAILED, RunState.ABORTED])
@pytest.mark.parametrize("target", list(RunState))
def test_terminal_state_cannot_transition(terminal: RunState, target: RunState) -> None:
    assert not is_valid_transition(terminal, target)


# ── Valid explicit transitions ────────────────────────────────────────────────

@pytest.mark.parametrize("from_state, to_state", [
    (RunState.CREATED, RunState.PLANNING),
    (RunState.PLANNING, RunState.PLAN_REVIEW),
    (RunState.PLAN_REVIEW, RunState.PLAN_READY),
    (RunState.PLAN_REVIEW, RunState.PLANNING),          # plan rejected → replanning
    (RunState.PLAN_READY, RunState.DOING),
    (RunState.DOING, RunState.AWAITING_VALIDATION),
    (RunState.AWAITING_VALIDATION, RunState.VALIDATING),
    (RunState.VALIDATING, RunState.SEMANTIC_AUDITING),
    (RunState.SEMANTIC_AUDITING, RunState.CHECKING),
    (RunState.SEMANTIC_AUDITING, RunState.CHECK_FAILED), # no findings but still gated
    (RunState.CHECKING, RunState.ADJUDICATING),
    (RunState.ADJUDICATING, RunState.CHECK_PASSED),
    (RunState.ADJUDICATING, RunState.CHECK_FAILED),
    (RunState.CHECK_PASSED, RunState.COMPLETED),
    (RunState.CHECK_FAILED, RunState.ACTING),
    (RunState.ACTING, RunState.AWAITING_VALIDATION),
    (RunState.ACTING, RunState.COMPLETED),
])
def test_valid_explicit_transition(from_state: RunState, to_state: RunState) -> None:
    assert is_valid_transition(from_state, to_state)


# ── Universal transitions (any nonterminal) ───────────────────────────────────

@pytest.mark.parametrize("from_state", [
    RunState.CREATED,
    RunState.PLANNING,
    RunState.DOING,
    RunState.CHECKING,
    RunState.ACTING,
    # awaiting_operator → awaiting_operator is the self-loop case, explicitly excluded
])
def test_any_nonterminal_can_enter_awaiting_operator(from_state: RunState) -> None:
    assert is_valid_transition(from_state, RunState.AWAITING_OPERATOR)


@pytest.mark.parametrize("from_state", [
    RunState.CREATED,
    RunState.PLANNING,
    RunState.DOING,
    RunState.CHECKING,
    RunState.ACTING,
    RunState.AWAITING_OPERATOR,
])
def test_any_nonterminal_can_fail(from_state: RunState) -> None:
    assert is_valid_transition(from_state, RunState.FAILED)


@pytest.mark.parametrize("from_state", [
    RunState.CREATED,
    RunState.PLANNING,
    RunState.DOING,
    RunState.CHECKING,
    RunState.ACTING,
    RunState.AWAITING_OPERATOR,
])
def test_any_nonterminal_can_abort(from_state: RunState) -> None:
    assert is_valid_transition(from_state, RunState.ABORTED)


# ── awaiting_operator resume ──────────────────────────────────────────────────

@pytest.mark.parametrize("resume_target", [
    RunState.PLANNING,
    RunState.DOING,
    RunState.CHECKING,
    RunState.ACTING,
    RunState.VALIDATING,
    RunState.ADJUDICATING,
    RunState.CHECK_FAILED,
])
def test_awaiting_operator_can_resume_to_any_nonterminal(resume_target: RunState) -> None:
    assert is_valid_transition(RunState.AWAITING_OPERATOR, resume_target)


def test_awaiting_operator_cannot_resume_to_itself() -> None:
    assert not is_valid_transition(RunState.AWAITING_OPERATOR, RunState.AWAITING_OPERATOR)


def test_awaiting_operator_cannot_resume_to_completed() -> None:
    # completed is not reachable from awaiting_operator — it is not a universal
    # target and awaiting_operator has no explicit transition to completed.
    assert not is_valid_transition(RunState.AWAITING_OPERATOR, RunState.COMPLETED)


# ── Invalid transitions ───────────────────────────────────────────────────────

def test_cannot_skip_planning_to_doing() -> None:
    assert not is_valid_transition(RunState.PLANNING, RunState.DOING)


def test_cannot_go_from_acting_to_planning() -> None:
    assert not is_valid_transition(RunState.ACTING, RunState.PLANNING)


def test_cannot_go_from_checking_to_doing() -> None:
    assert not is_valid_transition(RunState.CHECKING, RunState.DOING)


def test_cannot_go_from_created_directly_to_completed() -> None:
    assert not is_valid_transition(RunState.CREATED, RunState.COMPLETED)


def test_cannot_go_from_plan_ready_to_validating() -> None:
    assert not is_valid_transition(RunState.PLAN_READY, RunState.AWAITING_VALIDATION)


# ── finalized_actionable_finding_set gating ───────────────────────────────────

def test_raw_finding_without_disposition_excluded() -> None:
    findings = [_finding("f1", FindingSeverity.HIGH)]
    result = finalized_actionable_finding_set(findings, [], iteration=1)
    assert result == []


def test_adjudicator_disposition_includes_blocking_finding() -> None:
    findings = [_finding("f1", FindingSeverity.HIGH)]
    dispositions = [_disposition("f1", DecidedBy.ADJUDICATOR)]
    result = finalized_actionable_finding_set(findings, dispositions, iteration=1)
    assert len(result) == 1
    assert result[0].finding_id == "f1"


def test_engine_disposition_includes_blocking_finding() -> None:
    findings = [_finding("f1", FindingSeverity.CRITICAL)]
    dispositions = [_disposition("f1", DecidedBy.ENGINE)]
    result = finalized_actionable_finding_set(findings, dispositions, iteration=1)
    assert len(result) == 1


def test_operator_disposition_excluded_from_actionable_set() -> None:
    findings = [_finding("f1", FindingSeverity.HIGH)]
    dispositions = [_disposition("f1", DecidedBy.OPERATOR)]
    result = finalized_actionable_finding_set(findings, dispositions, iteration=1)
    assert result == []


def test_medium_severity_finding_excluded() -> None:
    findings = [_finding("f1", FindingSeverity.MEDIUM)]
    dispositions = [_disposition("f1", DecidedBy.ADJUDICATOR)]
    result = finalized_actionable_finding_set(findings, dispositions, iteration=1)
    assert result == []


def test_low_severity_finding_excluded() -> None:
    findings = [_finding("f1", FindingSeverity.LOW)]
    dispositions = [_disposition("f1", DecidedBy.ADJUDICATOR)]
    result = finalized_actionable_finding_set(findings, dispositions, iteration=1)
    assert result == []


def test_fixed_finding_excluded_despite_valid_disposition() -> None:
    findings = [_finding("f1", FindingSeverity.HIGH, FindingStatus.FIXED)]
    dispositions = [_disposition("f1", DecidedBy.ADJUDICATOR)]
    result = finalized_actionable_finding_set(findings, dispositions, iteration=1)
    assert result == []


def test_disposition_from_different_iteration_excluded() -> None:
    findings = [_finding("f1", FindingSeverity.HIGH)]
    dispositions = [_disposition("f1", DecidedBy.ADJUDICATOR, iteration=2)]
    result = finalized_actionable_finding_set(findings, dispositions, iteration=1)
    assert result == []


def test_mixed_severities_only_blocking_returned() -> None:
    findings = [
        _finding("f1", FindingSeverity.CRITICAL),
        _finding("f2", FindingSeverity.HIGH),
        _finding("f3", FindingSeverity.MEDIUM),
        _finding("f4", FindingSeverity.LOW),
        _finding("f5", FindingSeverity.INFO),
    ]
    dispositions = [
        _disposition("f1", DecidedBy.ADJUDICATOR),
        _disposition("f2", DecidedBy.ENGINE),
        _disposition("f3", DecidedBy.ADJUDICATOR),
        _disposition("f4", DecidedBy.ADJUDICATOR),
        _disposition("f5", DecidedBy.ENGINE),
    ]
    result = finalized_actionable_finding_set(findings, dispositions, iteration=1)
    ids = {f.finding_id for f in result}
    assert ids == {"f1", "f2"}
