from pathlib import Path


def test_remote_benchmark_script_downloads_both_e004_archives():
    script = Path("scripts/benchmark-remote.ps1").read_text(encoding="utf-8")

    assert "[switch]$E004" in script
    assert '"benchmark-e004"' in script
    assert '"E004_BASELINE_ARCHIVE="' in script
    assert '"E004_GUIDED_ARCHIVE="' in script
    assert "[switch]$E005" in script
    assert '"benchmark-e005"' in script
    assert '"E005_BASELINE_ARCHIVE="' in script
    assert '"E005_SRPG_ARCHIVE="' in script
    assert "if ($E004 -and $E005)" in script
    assert "foreach ($archivePrefix in $archivePrefixes)" in script
    assert "Start-Process $reports[-1]" in script
