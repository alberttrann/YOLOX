$CKPT = "YOLOX_outputs/tde_yolox_s/latest_ckpt.pth"
$LOG = "dawn_weather_breakdown.txt"

"" > $LOG
Write-Host "Evaluating DAWN Multi-Weather Breakdown..." -ForegroundColor Cyan

$WEATHERS = @("dawn_adverse_coco", "weather_splits/dawn_fog", "weather_splits/dawn_rain", "weather_splits/dawn_snow", "weather_splits/dawn_sand")

foreach ($W in $WEATHERS) {
    $JSON_PATH = "D:/DAWN/$W.json"
    Write-Host "`n---> Evaluating: $W" -ForegroundColor Yellow
    "=====================================" >> $LOG
    "EVALUATION: $W" >> $LOG
    "=====================================" >> $LOG

    python tools/eval.py -f exps/default/test_dawn.py -c $CKPT -b 16 --fuse --fp16 val_ann $JSON_PATH test_ann $JSON_PATH >> $LOG 2>&1
}

Write-Host "`nAll DAWN evaluations completed! Results saved to $LOG" -ForegroundColor Green