from __future__ import annotations

from collections.abc import Sequence
import shlex


SHELL_METACHARACTERS = {"|", ">", "<", "&&", "||", ";"}
FALLBACK_FORBIDDEN_COMMANDS = [
    "rm -rf",
    "git push --force",
    "git reset --hard",
]
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
    return argv


def classify_command(argv: Sequence[str], *, forbidden_commands: Sequence[str] | None = None) -> str:
    if not argv:
        raise ValidationCommandError("validation command cannot be empty")
    rendered = " ".join(argv)
    patterns = list(forbidden_commands or []) + FALLBACK_FORBIDDEN_COMMANDS
    if any(rendered.startswith(pattern) for pattern in patterns):
        return "risky"
    if argv[0] in ELEVATED_PREFIXES:
        return "elevated"
    return "safe"