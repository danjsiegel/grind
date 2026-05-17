from __future__ import annotations

from enum import StrEnum


class RunState(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    PLAN_REVIEW = "plan_review"
    PLAN_READY = "plan_ready"
    DOING = "doing"
    AWAITING_VALIDATION = "awaiting_validation"
    VALIDATING = "validating"
    SEMANTIC_AUDITING = "semantic_auditing"
    CHECKING = "checking"
    ADJUDICATING = "adjudicating"
    CHECK_PASSED = "check_passed"
    CHECK_FAILED = "check_failed"
    ACTING = "acting"
    AWAITING_OPERATOR = "awaiting_operator"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class OperatorStatus(StrEnum):
    NONE = "none"
    HOLD = "hold"
    APPROVED = "approved"
    REJECTED = "rejected"


class OperatorActionType(StrEnum):
    RESUME = "resume"
    APPROVE = "approve"
    REJECT = "reject"
    ABORT = "abort"
    RESTORE_CHECKPOINT = "restore_checkpoint"
    PATCH_POLICY = "patch_policy"


class TaskSourceKind(StrEnum):
    INLINE = "inline"
    FILE = "file"
    SPEC_SECTION = "spec_section"


class TaskStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ModelRole(StrEnum):
    PLANNER = "planner"
    IMPLEMENTER = "implementer"
    CHECKER = "checker"
    ADJUDICATOR = "adjudicator"


class FindingSeverity(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class FindingConfidence(StrEnum):
    PROVEN = "proven"
    LIKELY = "likely"
    SPECULATIVE = "speculative"


class FindingCategory(StrEnum):
    CORRECTNESS = "correctness"
    MISSING_IMPLEMENTATION = "missing_implementation"
    MISSING_VALIDATION = "missing_validation"
    SCOPE_VIOLATION = "scope_violation"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    SECURITY = "security"
    TEST_COVERAGE = "test_coverage"
    REGRESSION = "regression"
    INVARIANT_VIOLATION = "invariant_violation"
    SYSTEM_ERROR = "system_error"
    PROCESS_ARTIFACT = "process_artifact"


class FindingStatus(StrEnum):
    OPEN = "open"
    FIXED = "fixed"
    REJECTED = "rejected"
    DEFERRED = "deferred"
    DUPLICATE = "duplicate"
    PROCESS_ARTIFACT = "process_artifact"


class EvidenceType(StrEnum):
    DIFF = "diff"
    VALIDATION_OUTPUT = "validation_output"
    FILE_REF = "file_ref"
    MODEL_OUTPUT = "model_output"
    OPERATOR_NOTE = "operator_note"
    TEST_RESULT = "test_result"
    SYMBOL_REF = "symbol_ref"


class DecidedBy(StrEnum):
    ADJUDICATOR = "adjudicator"
    OPERATOR = "operator"
    ENGINE = "engine"


class InvariantKind(StrEnum):
    SIGNATURE = "signature"
    RETURN_TYPE = "return_type"
    STATE_RELATION = "state_relation"
    DEPENDENCY = "dependency"
    BEHAVIORAL = "behavioral"


class InvariantSourceKind(StrEnum):
    SPEC = "spec"
    POLICY = "policy"
    OPERATOR = "operator"
    SEMANTIC_AUDIT = "semantic_audit"
    VALIDATION = "validation"


class EnforcementMode(StrEnum):
    ADVISORY = "advisory"
    BLOCKING = "blocking"


class InvariantStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETIRED = "retired"


class CheckpointKind(StrEnum):
    TASK_BASELINE = "task_baseline"
    PRE_ACT = "pre_act"


class CaptureMode(StrEnum):
    GIT_PATCH_BUNDLE = "git_patch_bundle"
    SAFE_PATH_SNAPSHOT = "safe_path_snapshot"


class CheckpointStatus(StrEnum):
    AVAILABLE = "available"
    RESTORED = "restored"
    SUPERSEDED = "superseded"
    FAILED = "failed"


class HoldType(StrEnum):
    SCOPE_EXPANSION = "scope_expansion"
    VALIDATION_BLOCKED = "validation_blocked"
    RISKY_COMMAND = "risky_command"
    MAX_ITERATIONS = "max_iterations"
    CRITICAL_DISAGREEMENT = "critical_disagreement"
    BUDGET_EXCEEDED = "budget_exceeded"
    LEASE_CONFLICT = "lease_conflict"
    PLAN_REVIEW = "plan_review"
    PLAN_REJECTED_TWICE = "plan_rejected_twice"
    CONTEXT_OVERFLOW = "context_overflow"
    STORE_UNAVAILABLE = "store_unavailable"
    INVARIANT_BASELINE_CONFLICT = "invariant_baseline_conflict"
    DIMINISHING_RETURNS = "diminishing_returns"
    CHECKPOINT_UNAVAILABLE = "checkpoint_unavailable"
