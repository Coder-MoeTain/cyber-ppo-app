"""CLI smoke tests for the optional published H-MARL comparator."""
from pathlib import Path
import subprocess
import sys


def test_hmarl_baseline_runner_help_does_not_require_ray_import():
    root = Path(__file__).parents[1]
    script = root / "scripts" / "run_hmarl_baseline.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--hmarl-root" in result.stdout
    assert "--variant" in result.stdout


def test_paired_analysis_exposes_optional_hmarl_comparison():
    root = Path(__file__).parents[1]
    script = root / "scripts" / "analyze_cc4_results.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--hmarl-file" in result.stdout
