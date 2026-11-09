# A4.1 read-only paired point metric summary

No retraining, COCO scoring or bootstrap. Values come from native evaluator metadata.

|Domain|Comparison|ΔAP (pp)|ΔAR100 (pp)|Δsmall AP (pp)|Δsmall AR100 (pp)|Δlarge AP (pp)|
|---|---|---:|---:|---:|---:|---:|
|clean|dct − anchor|-0.200|+0.000|+0.000|+0.100|-1.100|
|clean|dct − orth|+0.000|+0.200|+0.100|+0.300|-0.300|
|clean|orth − anchor|-0.200|-0.200|-0.100|-0.200|-0.800|
|clean|dct/on − dct/fourier-off|+0.000|+0.000|+0.000|+0.000|+0.100|
|clean|dct/on − dct/memory-off|+0.000|+0.000|+0.100|+0.000|+0.000|
|clean|dct/on − dct/spi-off|+0.000|+0.000|+0.000|+0.000|+0.000|
|clean|dct/on − dct/context-off|+0.000|+0.000|+0.000|+0.000|+0.100|
|clean|dct/on − dct/dense-off|+0.000|+0.000|+0.000|+0.000|+0.000|
|clean|dct/on − dct/all-new-off|+0.000|+0.000|+0.000|+0.000|+0.000|
|fog|dct − anchor|-0.200|-1.800|-3.800|-4.900|-1.000|
|fog|dct − orth|+1.800|-0.100|-2.200|-3.900|+4.300|
|fog|orth − anchor|-2.000|-1.700|-1.600|-1.000|-5.300|
|fog|dct/on − dct/fourier-off|+0.000|+0.000|+0.100|-0.100|+0.100|
|fog|dct/on − dct/memory-off|-0.100|+0.200|-0.100|-0.200|+0.100|
|fog|dct/on − dct/spi-off|+0.000|+0.000|+0.000|+0.000|+0.100|
|fog|dct/on − dct/context-off|+0.000|+0.000|+0.700|+0.000|+0.100|
|fog|dct/on − dct/dense-off|+0.000|+0.000|+0.000|+0.000|+0.100|
|fog|dct/on − dct/all-new-off|-0.200|+0.300|+0.500|-0.200|+0.100|
|rain|dct − anchor|+1.400|+1.200|+1.600|+0.600|+5.100|
|rain|dct − orth|+0.300|-0.300|+1.000|-1.900|+8.700|
|rain|orth − anchor|+1.100|+1.500|+0.600|+2.500|-3.600|
|rain|dct/on − dct/fourier-off|+0.000|+0.000|+0.000|+0.000|+0.000|
|rain|dct/on − dct/memory-off|+0.200|+0.100|+0.000|+0.000|+0.300|
|rain|dct/on − dct/spi-off|+0.000|+0.000|+0.000|+0.000|+0.100|
|rain|dct/on − dct/context-off|+0.100|+0.200|+0.000|+0.000|+0.400|
|rain|dct/on − dct/dense-off|+0.100|+0.000|+0.000|+0.000|+0.400|
|rain|dct/on − dct/all-new-off|+0.200|+0.200|+0.000|+0.000|+0.200|
|snow|dct − anchor|+1.500|+1.500|+3.400|+4.700|+7.400|
|snow|dct − orth|-0.200|-1.400|+1.200|+1.000|+1.400|
|snow|orth − anchor|+1.700|+2.900|+2.200|+3.700|+6.000|
|snow|dct/on − dct/fourier-off|+0.000|+0.000|-0.100|+0.000|+0.000|
|snow|dct/on − dct/memory-off|+0.100|+0.000|+0.100|+0.000|+0.200|
|snow|dct/on − dct/spi-off|+0.000|-0.100|+0.000|+0.000|+0.000|
|snow|dct/on − dct/context-off|-0.100|-0.200|+0.000|+0.000|+0.000|
|snow|dct/on − dct/dense-off|+0.000|+0.000|-0.100|+0.000|+0.000|
|snow|dct/on − dct/all-new-off|-0.100|-0.300|-0.100|+0.000|+0.200|
|night|dct − anchor|-0.500|+2.000|+0.400|+0.100|+2.700|
|night|dct − orth|+1.400|+1.100|-0.100|-0.400|+4.300|
|night|orth − anchor|-1.900|+0.900|+0.500|+0.500|-1.600|
|night|dct/on − dct/fourier-off|+0.000|+0.000|+0.000|+0.000|-0.100|
|night|dct/on − dct/memory-off|+0.000|+0.100|+0.000|-0.100|+0.500|
|night|dct/on − dct/spi-off|+0.000|+0.000|+0.000|+0.000|+0.000|
|night|dct/on − dct/context-off|+0.000|+0.000|+0.000|+0.000|-0.100|
|night|dct/on − dct/dense-off|+0.000|+0.000|+0.000|+0.000|+0.000|
|night|dct/on − dct/all-new-off|+0.000|+0.000|-0.100|-0.100|+0.500|

## Integrity / missing reports

No detected metadata/protocol issues in completed comparisons.

## Interpretation

A4.1 DCT vs orth isolates the fixed filter basis more closely than A4.1 vs Mosaic anchor.
A4.1 vs anchor changes multiple model components but uses a matched real-Mosaic training policy.
One seed and two epochs cannot establish long-run superiority or statistically reliable gains.
All-scale AP AND AR and per-class behavior must be inspected before promotion.
