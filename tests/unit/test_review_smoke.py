"""Keep nested MCP failures actionable in the release smoke artifact."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import cast

import pytest


@pytest.mark.parametrize(
    ("create_command", "error_type", "detail"),
    [
        (False, "FileNotFoundError", "failing-scs"),
        (True, "ExceptionGroup", "Connection closed"),
    ],
)
def test_failed_review_smoke_records_the_cause(
    tmp_path: Path, create_command: bool, error_type: str, detail: str
) -> None:
    script = Path(__file__).parents[2] / "scripts/smoke-review.py"
    command = tmp_path / "failing-scs"
    output = tmp_path / "report.json"

    if create_command:
        command.write_text("#!/bin/sh\nexit 1\n")
        command.chmod(0o700)

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--command",
            str(command),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0

    decoded = cast(object, json.loads(output.read_text()))
    assert isinstance(decoded, dict)
    report = cast(dict[str, object], decoded)
    assert report["passed"] is False
    assert report["error"] == error_type

    details = report.get("errors")
    assert isinstance(details, list)
    assert any(
        isinstance(message, str) and detail in message
        for message in cast(list[object], details)
    )
