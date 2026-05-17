from __future__ import annotations

from grind.models.disposition import Disposition
from grind.models.enums import DecidedBy, FindingSeverity, FindingStatus, RunState
from grind.models.finding import Finding

# Terminal states are immutable with respect to state progression (§7.3)
TERMINAL_STATES: frozenset[RunState] = frozenset({
    RunState.COMPLETED,
    RunState.FAILED,
    RunState.ABORTED,
})

# Directed transitions not covered by the three universal rules.
# Universal rules (any nonterminal state):
#   → awaiting_operator  (hold entry)
#   → failed             (engine/model failure)
#   → aborted            (operator abort)
# Universal rule (awaiting_operator only):
#   → any nonterminal non-hold state (resume from hold)
_EXPLICIT_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.CREATED: frozenset({RunState.PLANNING}),
    RunState.PLANNING: frozenset({RunState.PLAN_REVIEW}),
    RunState.PLAN_REVIEW: frozenset({RunState.PLAN_READY, RunState.PLANNING}),
    RunState.PLAN_READY: frozenset({RunState.DOING}),
    RunState.DOING: frozenset({RunState.AWAITING_VALIDATION}),
    RunState.AWAITING_VALIDATION: frozenset({RunState.VALIDATING}),
    RunState.VALIDATING: frozenset({RunState.SEMANTIC_AUDITING}),
    RunState.SEMANTIC_AUDITING: frozenset({RunState.CHECKING, RunState.CHECK_FAILED}),
    RunState.CHECKING: frozenset({RunState.ADJUDICATING}),
    RunState.ADJUDICATING: frozenset({RunState.CHECK_PASSED, RunState.CHECK_FAILED}),
    RunState.CHECK_PASSED: frozenset({RunState.COMPLETED}),
    RunState.CHECK_FAILED: frozenset({RunState.ACTING}),
    RunState.ACTING: frozenset({RunState.AWAITING_VALIDATION, RunState.COMPLETED}),
}

_UNIVERSAL_TARGETS: frozenset[RunState] = frozenset({
    RunState.AWAITING_OPERATOR,
    RunState.FAILED,
    RunState.ABORTED,
})


def is_terminal(state: RunState) -> bool:
    """Return True if state is a terminal run state."""
    return state in TERMINAL_STATES


def is_valid_transition(from_state: RunState, to_state: RunState) -> bool:
    """Return True if from_state → to_state is a legal state machine transition (§7.2).

    Applies the three universal rules before consulting the explicit transition table.

    Universal rules (all nonterminal states, including awaiting_operator):
      any nonterminal → awaiting_operator   (hold entry)
      any nonterminal → failed              (engine/model failure)
      any nonterminal → aborted             (operator abort)

    awaiting_operator additionally allows:
      awaiting_operator → any nonterminal except itself   (resume from hold)

    Self-loop on awaiting_operator is excluded: a run cannot enter hold while
    already in hold — only the hold-entry trigger mechanisms can raise a new hold,
    and those are modelled as resume → re-trigger, not a direct self-loop.
    """
    # Terminal states are immutable — no outbound transitions
    if from_state in TERMINAL_STATES:
        return False
    # Self-loop on awaiting_operator is not a valid transition
    if from_state == RunState.AWAITING_OPERATOR and to_state == RunState.AWAITING_OPERATOR:
        return False
    # Universal: any nonterminal → awaiting_operator / failed / aborted
    if to_state in _UNIVERSAL_TARGETS:
        return True
    # Resume: awaiting_operator → any nonterminal non-self non-terminal state
    if from_state == RunState.AWAITING_OPERATOR:
        return to_state not in TERMINAL_STATES
    # Explicit directed transitions
    return to_state in _EXPLICIT_TRANSITIONS.get(from_state, frozenset())


# Severity levels that make a finding blocking (§8.5)
_BLOCKING_SEVERITIES: frozenset[FindingSeverity] = frozenset({
    FindingSeverity.CRITICAL,
    FindingSeverity.HIGH,
})


def finalized_actionable_finding_set(
    findings: list[Finding],
    dispositions: list[Disposition],
    iteration: int,
) -> list[Finding]:
    """Return the authoritative actionable finding set for the given iteration (§6.5, §7.3).

    Only findings whose severity is critical or high, whose status is open, and which have
    a same-iteration disposition with decided_by = adjudicator or engine and a non-empty
    justification are included.

    Raw checker findings with no valid disposition are excluded — they cannot drive acting.
    """
    valid_finding_ids: set[str] = {
        d.finding_id
        for d in dispositions
        if (
            d.iteration == iteration
            and d.decided_by in (DecidedBy.ADJUDICATOR, DecidedBy.ENGINE)
            and d.justification  # must reference concrete evidence
        )
    }
    return [
        f
        for f in findings
        if (
            f.severity in _BLOCKING_SEVERITIES
            and f.status == FindingStatus.OPEN
            and f.finding_id in valid_finding_ids
        )
    ]
