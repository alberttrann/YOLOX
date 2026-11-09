# tdevNext.3: read-only local evidence snapshot

P0 branches present: **55/55**. Completed bootstrap files: **31**.

## Training loader

`{'path': 'D:\\YOLOX-3rd\\exps\\default\\tde_yolox_s.py', 'status': 'inspected', 'mosaic_detection_constructed': False, 'batch_sampler_constructed': True, 'get_dataset_called': True, 'assessment': 'POSSIBLE_NO_MOSAIC_WRAPPER', 'note': 'Static detection only. Verify actual runtime dataset wrapper if custom factories or monkeypatches exist.'}`

## Already computed point-estimate contrasts

|Domain|Left|Right|ΔAP (pp)|ΔAR100 (pp)|Δsmall AP (pp)|Δlarge AP (pp)|
|---|---|---|---:|---:|---:|---:|
|clean|A2/on|A2/fourier-off|+0.000|+0.000|+0.000|+0.000|
|clean|A3/on|A3/memory-off|+0.000|+0.000|+0.000|+0.000|
|clean|A4/on|A4/fourier-off|+0.000|+0.000|+0.000|+0.000|
|clean|A4/on|A4/memory-off|+0.000|+0.000|+0.000|+0.000|
|clean|A4/on|A4/both-off|+0.000|+0.000|+0.000|+0.000|
|clean|A2/on|A1/on|-0.200|-0.100|-0.100|+0.100|
|clean|A4/on|A4R/on|+0.100|+0.200|+0.000|+0.500|
|fog|A2/on|A2/fourier-off|+0.000|+0.000|+0.000|+0.100|
|fog|A3/on|A3/memory-off|+0.000|+0.000|+0.000|-0.100|
|fog|A4/on|A4/fourier-off|+0.000|+0.000|+0.000|+0.000|
|fog|A4/on|A4/memory-off|+0.000|+0.000|+0.100|+0.000|
|fog|A4/on|A4/both-off|+0.000|+0.000|+0.000|+0.000|
|fog|A2/on|A1/on|+5.700|+5.000|+2.600|+12.900|
|fog|A4/on|A4R/on|+0.300|+2.300|-0.200|-4.800|
|rain|A2/on|A2/fourier-off|+0.000|+0.000|+0.000|+0.000|
|rain|A3/on|A3/memory-off|+0.000|+0.000|+0.000|-0.200|
|rain|A4/on|A4/fourier-off|+0.000|+0.000|+0.000|+0.000|
|rain|A4/on|A4/memory-off|+0.000|+0.000|+0.000|+0.000|
|rain|A4/on|A4/both-off|+0.000|+0.000|+0.000|+0.000|
|rain|A2/on|A1/on|-0.400|-0.500|+0.200|-2.000|
|rain|A4/on|A4R/on|+0.000|+1.600|-2.900|+11.200|
|snow|A2/on|A2/fourier-off|+0.000|+0.000|+0.000|+0.000|
|snow|A3/on|A3/memory-off|+0.200|+0.000|+0.000|+0.100|
|snow|A4/on|A4/fourier-off|+0.000|+0.000|+0.000|-0.100|
|snow|A4/on|A4/memory-off|+0.100|+0.000|+0.000|+0.200|
|snow|A4/on|A4/both-off|+0.100|+0.000|+0.000|+0.100|
|snow|A2/on|A1/on|-0.200|+0.200|+2.000|-1.200|
|snow|A4/on|A4R/on|+0.400|+1.500|+2.700|+2.000|
|night|A2/on|A2/fourier-off|+0.000|+0.000|+0.000|+0.000|
|night|A3/on|A3/memory-off|-0.100|-0.300|+0.000|+0.000|
|night|A4/on|A4/fourier-off|+0.000|+0.000|+0.000|+0.000|
|night|A4/on|A4/memory-off|+0.000|+0.000|+0.000|+0.000|
|night|A4/on|A4/both-off|+0.000|+0.000|+0.000|+0.000|
|night|A2/on|A1/on|-0.700|-0.700|-0.300|-2.800|
|night|A4/on|A4R/on|-0.600|+1.100|-0.600|-1.900|

## Missing evidence

Uncaptured P0 branches: 0.

## Warnings


All numbers above were **read from existing files**; no inference, scoring or bootstrap was run.
