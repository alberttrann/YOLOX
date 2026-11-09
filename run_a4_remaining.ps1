# tdevNext.3 A4 remaining verification suite.
# Place in D:\YOLOX-3rd; run stages explicitly, on one GPU at a time.
# Requires already installed tdevnext3_a4_rigorous overlay.
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('Check','Matched2','Matched80','ResumeA4','P0','P0Bootstrap','P1','P2CampaignManifest','P2CampaignRun','P2Profile','P2Export')]
    [string] $Stage,
    [ValidateSet(0,1,2)] [int] $Seed = 0,
    [string[]] $Domains = @('clean','fog','rain','snow','night'),
    [string[]] $Models = @('A0','A1','A2','A3','A4','A4R'),
    [ValidateRange(10,10000)] [int] $Iterations = 200,
    [switch] $Force
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repo
$diag = 'YOLOX_outputs/a4_diagnostics'
$inputModels = [ordered]@{
    A0  = @{ exp='exps/default/tde_yolox_s_fourier_off.py';                 ck='YOLOX_outputs/tdevnext3_a0_off/epoch_2_ckpt.pth';                 branches=@('on') }
    A1  = @{ exp='exps/default/tde_yolox_s_fourier_spatial.py';             ck='YOLOX_outputs/tdevnext3_a1_spatial/epoch_2_ckpt.pth';             branches=@('on') }
    A2  = @{ exp='exps/default/tde_yolox_s_fourier_spectral.py';            ck='YOLOX_outputs/tdevnext3_a2_spectral/epoch_2_ckpt.pth';            branches=@('on','fourier-off') }
    A3  = @{ exp='exps/default/tde_yolox_s_fourier_spectral_memory.py';     ck='YOLOX_outputs/tdevnext3_a3_spectral_memory/epoch_2_ckpt.pth';     branches=@('on','memory-off') }
    A4  = @{ exp='exps/default/tde_yolox_s_fourier_a4.py';                  ck='YOLOX_outputs/tdevnext3_a4_fourier_grounded/epoch_2_ckpt.pth';    branches=@('on','fourier-off','memory-off','both-off') }
    A4R = @{ exp='exps/default/tde_yolox_s_fourier_a4_control.py';          ck='YOLOX_outputs/tdevnext3_a4_orthogonal_control/epoch_2_ckpt.pth'; branches=@('on') }
}
$dataExps = [ordered]@{
    clean = 'exps/default/tde_yolox_s_fourier_off.py'
    fog   = 'exps/default/test_acdc_fog.py'
    rain  = 'exps/default/test_acdc_rain.py'
    snow  = 'exps/default/test_acdc_snow.py'
    night = 'exps/default/test_acdc_night.py'
}
$matched = @(
    @{key='A0'; exp='exps/default/tde_yolox_s_fourier_off.py';               name='tdevnext3_matched_a0_seed'},
    @{key='A1'; exp='exps/default/tde_yolox_s_fourier_spatial.py';           name='tdevnext3_matched_a1_seed'},
    @{key='A2'; exp='exps/default/tde_yolox_s_fourier_spectral.py';          name='tdevnext3_matched_a2_seed'},
    @{key='A3'; exp='exps/default/tde_yolox_s_fourier_spectral_memory.py';   name='tdevnext3_matched_a3_seed'}
)
function RunPython([string[]] $Argv) {
    Write-Host "RUN: python $($Argv -join ' ')" -ForegroundColor Cyan
    & python @Argv
    if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit=$LASTEXITCODE): $($Argv -join ' ')" }
}
function NeedFile([string] $Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing file: $Path" }
}
function CheckOutputs([string] $left, [string] $right) {
    foreach ($p in @($left,$right)) {
        NeedFile "$p/metadata.json"
        NeedFile "$p/detections.json"
    }
}
function Pair([string] $Domain, [string] $Name, [string] $Left, [string] $Right, [bool] $different) {
    CheckOutputs $Left $Right
    $target = "$diag/bootstrap/$Domain/$Name.json"
    if ((Test-Path $target) -and -not $Force) { Write-Host "SKIP $target"; return }
    $argv = @('tools/a4_paired_bootstrap.py','--left',$Left,'--right',$Right,
              '--iterations',"$Iterations",'--seed','137','--output',$target)
    if ($different) { $argv += '--allow-different-checkpoints' }
    RunPython -Argv $argv
}
foreach ($d in $Domains) { if (-not $dataExps.Contains($d)) { throw "Unknown domain $d" } }
foreach ($m in $Models) { if (-not $inputModels.Contains($m)) { throw "Unknown model $m" } }

switch ($Stage) {
    'Check' {
        foreach ($m in @('A4','A4R')) {
            $cfg=$inputModels[$m]
            NeedFile $cfg.ck
            RunPython -Argv @('tools/a4_verify_checkpoint.py','--model-exp',$cfg.exp,'--checkpoint',$cfg.ck)
        }
        foreach ($file in @('tools/a4_eval_counterfactual.py','tools/a4_paired_bootstrap.py','tools/train_fourier_a4_controls.py')) { NeedFile $file }
        Write-Host 'CHECK PASS: A4 and A4-R epoch-2 checkpoints and source provenance' -ForegroundColor Green
    }
    'Matched2' {
        foreach ($m in $matched) {
            $name="$($m.name)$Seed"; $ck="YOLOX_outputs/$name/epoch_2_ckpt.pth"
            if ((Test-Path $ck) -and -not $Force) { Write-Host "SKIP $ck"; continue }
            $argv=@('tools/train_fourier_a4_controls.py','-f',$m.exp,'-expn',$name,
                    '-d','1','-b','8','--fp16','-c','weights/yolox_s.pth','--stop-after-epoch','2')
            if ($Seed -ne 0) { $argv+=@('seed',"$Seed") }
            RunPython -Argv $argv
        }
    }
    'Matched80' {
        foreach ($m in $matched) {
            $name="$($m.name)$Seed"; $last="YOLOX_outputs/$name/latest_ckpt.pth"
            $argv=@('tools/train_fourier_a4_controls.py','-f',$m.exp,'-expn',$name,'-d','1','-b','8','--fp16')
            if (Test-Path $last) { $argv+=@('--resume','-c',$last) }
            else { $argv+=@('-c','weights/yolox_s.pth') }
            if ($Seed -ne 0) { $argv+=@('seed',"$Seed") }
            RunPython -Argv $argv
        }
    }
    'ResumeA4' {
        if ($Seed -eq 0) {
            foreach ($m in @('A4','A4R')) {
                $cfg=$inputModels[$m];$last=(Split-Path -Parent $cfg.ck)+'/latest_ckpt.pth'
                NeedFile $last
                RunPython -Argv @('tools/train_fourier_a4.py','-f',$cfg.exp,'-d','1','-b','8','--fp16','--resume','-c',$last)
            }
        } else {
            foreach ($m in @('A4','A4R')) {
                $cfg=$inputModels[$m]
                $expn= if ($m -eq 'A4') {"tdevnext3_a4_fourier_grounded_seed$Seed"} else {"tdevnext3_a4_orthogonal_control_seed$Seed"}
                $last="YOLOX_outputs/$expn/latest_ckpt.pth"
                $argv=@('tools/train_fourier_a4.py','-f',$cfg.exp,'-expn',$expn,'-d','1','-b','8','--fp16')
                if (Test-Path $last) {$argv+=@('--resume','-c',$last)}
                else {$argv+=@('-c','weights/yolox_s.pth')}
                $argv+=@('seed',"$Seed")
                RunPython -Argv $argv
            }
        }
    }
    'P0' {
        foreach ($m in $Models) {
            $cfg=$inputModels[$m];NeedFile $cfg.ck
            foreach ($d in $Domains) {
                $out="$diag/p0/$m/$d"
                $complete=$true
                foreach ($b in $cfg.branches) {
                    if (-not ((Test-Path "$out/$b/metadata.json") -and (Test-Path "$out/$b/detections.json"))) {$complete=$false}
                }
                if ($complete -and -not $Force) {Write-Host "SKIP $m $d (complete)";continue}
                RunPython -Argv (@('tools/a4_eval_counterfactual.py','--model-exp',$cfg.exp,
                    '--data-exp',$dataExps[$d],'--checkpoint',$cfg.ck,
                    '--weights','ema','--branches') + $cfg.branches + @('--batch-size','8','--conf','0.01',
                    '--nms','0.60','--device','cuda','--fp16','--output-dir',$out))
            }
        }
    }
    'P0Bootstrap' {
        foreach ($d in $Domains) {
            Pair $d 'A2_fourier_on_minus_off' "$diag/p0/A2/$d/on" "$diag/p0/A2/$d/fourier-off" $false
            Pair $d 'A3_engram_on_minus_off' "$diag/p0/A3/$d/on" "$diag/p0/A3/$d/memory-off" $false
            Pair $d 'A4_fourier_on_minus_off' "$diag/p0/A4/$d/on" "$diag/p0/A4/$d/fourier-off" $false
            Pair $d 'A4_engram_on_minus_off' "$diag/p0/A4/$d/on" "$diag/p0/A4/$d/memory-off" $false
            Pair $d 'A4_both_on_minus_both_off' "$diag/p0/A4/$d/on" "$diag/p0/A4/$d/both-off" $false
            Pair $d 'A2_minus_A1' "$diag/p0/A2/$d/on" "$diag/p0/A1/$d/on" $true
            Pair $d 'A4_minus_A4R' "$diag/p0/A4/$d/on" "$diag/p0/A4R/$d/on" $true
        }
    }
    'P1' {
        foreach ($m in @('A4','A4R')) {
            $cfg=$inputModels[$m];NeedFile $cfg.ck
            RunPython -Argv @('tools/a4_feature_probe.py','--model-exp',$cfg.exp,'--checkpoint',$cfg.ck,
                        '--weights','ema','--images','24','--output',"$diag/p1/${m}_features.json")
            RunPython -Argv @('tools/a4_gradient_coverage.py','--model-exp',$cfg.exp,'--checkpoint',$cfg.ck,
                        '--batch-size','2','--batches','4','--device','cuda','--output',"$diag/p1/${m}_gradients.json")
        }
        foreach ($d in $Domains) {
            foreach ($c in @(
                @{name='A2';off='fourier-off'}, @{name='A3';off='memory-off'},
                @{name='A4';off='fourier-off'}, @{name='A4';off='memory-off'})) {
                $m=$c.name;$branch=$c.off
                $left="$diag/p0/$m/$d/on";$right="$diag/p0/$m/$d/$branch"
                CheckOutputs $left $right
                RunPython -Argv @('tools/a4_match_differences.py','--left',$left,'--right',$right,
                            '--output',"$diag/p1/${m}_${d}_${branch}_predictions.json")
            }
        }
    }
    'P2CampaignManifest' {
        $mods=@()
        foreach ($m in $Models) {
            $cfg=$inputModels[$m];NeedFile $cfg.ck
            $mods+=@{name=$m;model_exp=$cfg.exp;checkpoint=$cfg.ck}
        }
        $ds=@()
        foreach ($d in $Domains) {$ds+=@{name=$d;data_exp=$dataExps[$d];split='val'}}
        $manifest=@{batch_size=8;conf=0.01;nms=0.60;models=$mods;domains=$ds}
        New-Item -ItemType Directory -Force -Path "$diag/p2" | Out-Null
        $json = $manifest | ConvertTo-Json -Depth 10
        $path = Join-Path $repo "$diag/p2/campaign_epoch2.json"
        [System.IO.File]::WriteAllText($path, $json, [System.Text.UTF8Encoding]::new($false))
        Write-Host "Created $diag/p2/campaign_epoch2.json"
        Write-Host 'NOTE: a4_campaign.py does not propagate --fp16. Campaign evaluates all models consistently in FP32.' -ForegroundColor Yellow
    }
    'P2CampaignRun' {
        NeedFile "$diag/p2/campaign_epoch2.json"
        RunPython -Argv @('tools/a4_campaign.py','--manifest',"$diag/p2/campaign_epoch2.json",
                    '--output-root',"$diag/p2/campaign_results",'--execute')
    }
    'P2Profile' {
        foreach ($m in @('A4','A4R')) {
            $cfg=$inputModels[$m];NeedFile $cfg.ck
            RunPython -Argv @('tools/a4_profile_matrix.py','--model-exp',$cfg.exp,'--checkpoint',$cfg.ck,
                '--weights','ema','--fp16','--sizes','576x1024','768x768','1088x1920',
                '--batch-size','1','--warmup','10','--repeats','40','--output',"$diag/p2/${m}_profile.json")
        }
    }
    'P2Export' {
        foreach ($m in @('A4','A4R')) {
            $cfg=$inputModels[$m];NeedFile $cfg.ck
            RunPython -Argv @('tools/a4_verify_checkpoint.py','--model-exp',$cfg.exp,'--checkpoint',$cfg.ck)
            RunPython -Argv @('tools/export_full_fourier.py','--model-exp',$cfg.exp,'--checkpoint',$cfg.ck,
                '--weights','ema','--output',"artifacts/${m}_dynamic.onnx",'--height','576','--width','1024','--dynamic')
        }
    }
}
