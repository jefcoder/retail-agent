"""Sandbox-side problem outcome status enum.

Emitted in the per-problem envelope written to output.jsonl for consumers
of sandbox runs (test runner, dashboards, downstream tooling).

Values must stay compatible with the canonical status literals enforced in
``tests/test_sandbox_envelope.py``.
"""

from enum import Enum


class SandboxProblemStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
