$CKPT = "YOLOX_outputs/tde_yolox_s/latest_ckpt.pth"
$LOG = "eval_benchmark_results.txt"

"" > $LOG
Write-Host "Starting Complete TDE-YOLOX Zero-Shot Evaluation Suite..." -ForegroundColor Cyan

$EXPS = @(
    "exps/default/tde_yolox_s.py",
    "exps/default/test_acdc_fog.py",
    "exps/default/test_acdc_night.py",
    "exps/default/test_acdc_rain.py",
    "exps/default/test_acdc_snow.py",
    "exps/default/test_dawn.py",
    "exps/default/test_udacity_large.py"
)

foreach ($EXP in $EXPS) {
    Write-Host "`n================================================" -ForegroundColor Yellow
    Write-Host "Running Evaluation: $EXP" -ForegroundColor Yellow
    Write-Host "================================================" -ForegroundColor Yellow
    
    "=== EVALUATION: $EXP ===" >> $LOG
    python tools/eval.py -f $EXP -c $CKPT -b 16 --fuse --fp16 >> $LOG 2>&1
}

Write-Host "`nAll Evaluations Completed! Results saved to $LOG" -ForegroundColor Green