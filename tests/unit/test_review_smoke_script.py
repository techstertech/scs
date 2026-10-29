"""The release smoke command must fail closed when its MCP process fails."""

import json
from pathlib import Path
import subprocess
import sys


def test_failed_mcp_process_cannot_leave_a_success_report(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text(json.dumps({"passed": True}))
    script = Path(__file__).parents[2] / "scripts" / "smoke-review.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--command",
            "/usr/bin/false",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode != 0
    assert json.loads(output.read_text())["passed"] is False
