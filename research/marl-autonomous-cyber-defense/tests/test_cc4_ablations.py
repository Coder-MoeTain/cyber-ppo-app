"""Smoke tests for the ablation analysis entry point."""
from pathlib import Path
import subprocess
import sys


def test_ablation_analysis_cli_help_loads():
    root = Path(__file__).parents[1]
    script = root / "scripts" / "analyze_cc4_ablations.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--full-glob" in result.stdout
    assert "--no-risk-glob" in result.stdout
    assert "--risk-rule-glob" in result.stdout
