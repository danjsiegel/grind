from __future__ import annotations

from collections.abc import Sequence
import shlex


SHELL_METACHARACTERS = {"|", ">", "<", "&&", "||", ";"}
# Characters that enable command substitution even without shell metacharacters
_SUBSTITUTION_CHARS = {"`", "$("}
FALLBACK_FORBIDDEN_COMMANDS = [
    "rm -rf",
    "git push --force",
    "git reset --hard",
]
# Suffixes that make an otherwise-forbidden prefix safe.  e.g. "--force-with-lease"
# should not be blocked just because it starts with "--force".
_FORBIDDEN_COMMAND_SAFE_SUFFIXES: dict[str, tuple[str, ...]] = {
    "git push --force": ("-with-lease", "-with-lease="),
}
ELEVATED_PREFIXES = {
    "sudo",
    "chmod",
    "chown",
    "apt",
    "apt-get",
    "brew",
    "yum",
}


class ValidationCommandError(ValueError):
    pass


def normalize_shell_free_command(command: str) -> list[str]:
    argv = shlex.split(command)
    if not argv:
        raise ValidationCommandError("validation command cannot be empty")
    if any(token in SHELL_METACHARACTERS for token in argv):
        raise ValidationCommandError("shell metacharacters are not allowed on the default validation path")
    # Block command substitution via backtick or $( even without a shell metacharacter token
    joined = " ".join(argv)
    if any(sub in joined for sub in _SUBSTITUTION_CHARS):
        raise ValidationCommandError("shell metacharacters are not allowed on the default validation path")
    return argv


def classify_command(argv: Sequence[str], *, forbidden_commands: Sequence[str] | None = None) -> str:
    if not argv:
        raise ValidationCommandError("validation command cannot be empty")
    rendered = " ".join(argv)
    patterns = list(forbidden_commands or []) + FALLBACK_FORBIDDEN_COMMANDS
    for pattern in patterns:
        if rendered.startswith(pattern):
            # Check whether the rest of the string is a known safe suffix that
            # disambiguates the command (e.g. "--force-with-lease").
            remainder = rendered[len(pattern):]
            safe_suffixes = _FORBIDDEN_COMMAND_SAFE_SUFFIXES.get(pattern, ())
            if remainder == "" or not any(remainder.startswith(s) for s in safe_suffixes):
                return "risky"
    if argv[0] in ELEVATED_PREFIXES:
        return "elevated"
    return "safe"