# PowerShell: run from the repository root after copying bundle tools/tests.
$ErrorActionPreference = 'Stop'
python tests/test_safe_ttt_core.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python tests/test_normalized_step_and_router.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python tests/test_source_gradient_audit.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
# The following tests also validate against the TDE source layout; they can
# fail closed if your current repository differs from the earlier snapshot.
python tests/test_actual_tde_head_fixed_targets.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
python tests/test_full_snapshot_yolox_audit.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host 'ALL SAFE TTT CPU CHECKS PASSED'
