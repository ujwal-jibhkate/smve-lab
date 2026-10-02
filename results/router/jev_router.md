# Stage 4b: Jev as the router

Jev `jev-1.13.0`; median latency per call 131 ms (query only) / 131 ms (query + top 3), measured from the laptop. Cost for all 2,838 queries × 2 calls: $0.234. Results on held-out queries only; zero-shot Jev routers are never trained. 'gain kept': 0 = random routing, 1 = oracle. Latency at a budget includes Jev's overhead (A: max(0, Jev − base), B: full Jev call).


## A. pooled 5-fold cross-validation


### SciFact (300 queries; escalation helps on 19%) · never 0.7032 · always 0.7305 · oracle best 0.7642

| router | AUC [95% CI] | gain kept | nDCG@10 @20% | latency @20% | ECE |
|---|---|---|---|---|---|
| random | – | +0.000 | 0.7087 | 331 ms | – |
| Jev B: lower candidate better | 0.791 [0.735, 0.845] | +0.234 | 0.7192 | 462 ms | 0.102 |
| Jev B: top-1 not direct | 0.732 [0.666, 0.792] | +0.274 | 0.7239 | 462 ms | 0.318 |
| Jev A: needs reasoning | 0.557 [0.473, 0.631] | +0.181 | 0.7123 | 457 ms | 0.647 |
| Jev A: ambiguous | 0.555 [0.472, 0.647] | -0.024 | 0.7093 | 457 ms | 0.167 |
| first-stage logistic | 0.674 [0.597, 0.748] | +0.143 | 0.7206 | 331 ms | 0.038 |
| first-stage boosted | 0.768 [0.702, 0.829] | +0.246 | 0.7239 | 331 ms | 0.044 |
| Jev A logistic | 0.625 [0.543, 0.704] | +0.107 | 0.7129 | 457 ms | 0.174 |
| Jev B logistic | 0.757 [0.683, 0.826] | +0.188 | 0.7157 | 462 ms | 0.119 |
| first-stage + Jev B logistic | 0.750 [0.685, 0.810] | +0.223 | 0.7171 | 462 ms | 0.038 |
| first-stage + Jev B boosted | 0.785 [0.727, 0.841] | +0.198 | 0.7234 | 462 ms | 0.032 |

### NFCorpus (323 queries; escalation helps on 32%) · never 0.3315 · always 0.3421 · oracle best 0.3636

| router | AUC [95% CI] | gain kept | nDCG@10 @20% | latency @20% | ECE |
|---|---|---|---|---|---|
| random | – | +0.000 | 0.3336 | 331 ms | – |
| Jev B: lower candidate better | 0.578 [0.504, 0.646] | +0.064 | 0.3391 | 463 ms | 0.155 |
| Jev B: top-1 not direct | 0.456 [0.388, 0.522] | +0.117 | 0.3369 | 463 ms | 0.344 |
| Jev A: needs reasoning | 0.656 [0.592, 0.717] | +0.104 | 0.3344 | 457 ms | 0.103 |
| Jev A: ambiguous | 0.466 [0.402, 0.535] | -0.031 | 0.3352 | 457 ms | 0.427 |
| first-stage logistic | 0.585 [0.521, 0.648] | -0.033 | 0.3312 | 331 ms | 0.126 |
| first-stage boosted | 0.540 [0.471, 0.604] | -0.122 | 0.3325 | 331 ms | 0.110 |
| Jev A logistic | 0.604 [0.546, 0.665] | +0.133 | 0.3360 | 457 ms | 0.063 |
| Jev B logistic | 0.602 [0.537, 0.664] | +0.090 | 0.3363 | 463 ms | 0.064 |
| first-stage + Jev B logistic | 0.546 [0.473, 0.612] | -0.026 | 0.3322 | 463 ms | 0.066 |
| first-stage + Jev B boosted | 0.555 [0.488, 0.618] | -0.096 | 0.3317 | 463 ms | 0.071 |

### ArguAna (1406 queries; escalation helps on 53%) · never 0.4980 · always 0.6975 · oracle best 0.7393

| router | AUC [95% CI] | gain kept | nDCG@10 @20% | latency @20% | ECE |
|---|---|---|---|---|---|
| random | – | +0.000 | 0.5379 | 345 ms | – |
| Jev B: lower candidate better | 0.505 [0.474, 0.535] | +0.016 | 0.5363 | 476 ms | 0.217 |
| Jev B: top-1 not direct | 0.550 [0.520, 0.581] | +0.122 | 0.5560 | 476 ms | 0.176 |
| Jev A: needs reasoning | 0.510 [0.482, 0.538] | +0.008 | 0.5380 | 467 ms | 0.271 |
| Jev A: ambiguous | 0.469 [0.440, 0.499] | -0.054 | 0.5297 | 467 ms | 0.164 |
| first-stage logistic | 0.593 [0.564, 0.624] | +0.141 | 0.5507 | 345 ms | 0.044 |
| first-stage boosted | 0.626 [0.597, 0.656] | +0.171 | 0.5550 | 345 ms | 0.033 |
| Jev A logistic | 0.460 [0.432, 0.491] | -0.091 | 0.5248 | 467 ms | 0.159 |
| Jev B logistic | 0.494 [0.465, 0.525] | -0.010 | 0.5379 | 476 ms | 0.122 |
| first-stage + Jev B logistic | 0.606 [0.575, 0.634] | +0.157 | 0.5540 | 476 ms | 0.041 |
| first-stage + Jev B boosted | 0.642 [0.613, 0.671] | +0.199 | 0.5515 | 476 ms | 0.024 |

## B. trained on SciFact train only (transfer)


### SciFact (300 queries; escalation helps on 19%) · never 0.7032 · always 0.7305 · oracle best 0.7642

| router | AUC [95% CI] | gain kept | nDCG@10 @20% | latency @20% | ECE |
|---|---|---|---|---|---|
| random | – | +0.000 | 0.7087 | 331 ms | – |
| Jev B: lower candidate better | 0.791 [0.730, 0.843] | +0.234 | 0.7192 | 462 ms | 0.102 |
| Jev B: top-1 not direct | 0.732 [0.668, 0.795] | +0.274 | 0.7239 | 462 ms | 0.318 |
| Jev A: needs reasoning | 0.557 [0.471, 0.642] | +0.181 | 0.7123 | 457 ms | 0.647 |
| Jev A: ambiguous | 0.555 [0.475, 0.637] | -0.024 | 0.7093 | 457 ms | 0.167 |
| first-stage logistic | 0.737 [0.669, 0.794] | +0.175 | 0.7200 | 331 ms | 0.039 |
| first-stage boosted | 0.743 [0.682, 0.801] | +0.195 | 0.7151 | 331 ms | 0.064 |
| Jev A logistic | 0.474 [0.391, 0.561] | -0.158 | 0.7013 | 457 ms | 0.027 |
| Jev B logistic | 0.787 [0.734, 0.837] | +0.243 | 0.7160 | 462 ms | 0.099 |
| first-stage + Jev B logistic | 0.794 [0.738, 0.847] | +0.248 | 0.7218 | 462 ms | 0.049 |
| first-stage + Jev B boosted | 0.821 [0.775, 0.867] | +0.260 | 0.7252 | 462 ms | 0.036 |

### NFCorpus (323 queries; escalation helps on 32%) · never 0.3315 · always 0.3421 · oracle best 0.3636

| router | AUC [95% CI] | gain kept | nDCG@10 @20% | latency @20% | ECE |
|---|---|---|---|---|---|
| random | – | +0.000 | 0.3336 | 331 ms | – |
| Jev B: lower candidate better | 0.578 [0.514, 0.644] | +0.064 | 0.3391 | 463 ms | 0.155 |
| Jev B: top-1 not direct | 0.456 [0.390, 0.520] | +0.117 | 0.3369 | 463 ms | 0.344 |
| Jev A: needs reasoning | 0.656 [0.594, 0.718] | +0.104 | 0.3344 | 457 ms | 0.103 |
| Jev A: ambiguous | 0.466 [0.396, 0.534] | -0.031 | 0.3352 | 457 ms | 0.427 |
| first-stage logistic | 0.536 [0.469, 0.602] | -0.073 | 0.3305 | 331 ms | 0.131 |
| first-stage boosted | 0.545 [0.482, 0.612] | -0.076 | 0.3325 | 331 ms | 0.142 |
| Jev A logistic | 0.375 [0.310, 0.446] | -0.041 | 0.3365 | 457 ms | 0.435 |
| Jev B logistic | 0.510 [0.439, 0.583] | +0.099 | 0.3368 | 463 ms | 0.157 |
| first-stage + Jev B logistic | 0.521 [0.453, 0.590] | -0.042 | 0.3340 | 463 ms | 0.184 |
| first-stage + Jev B boosted | 0.591 [0.529, 0.658] | -0.001 | 0.3331 | 463 ms | 0.123 |

### ArguAna (1406 queries; escalation helps on 53%) · never 0.4980 · always 0.6975 · oracle best 0.7393

| router | AUC [95% CI] | gain kept | nDCG@10 @20% | latency @20% | ECE |
|---|---|---|---|---|---|
| random | – | +0.000 | 0.5379 | 345 ms | – |
| Jev B: lower candidate better | 0.505 [0.471, 0.533] | +0.016 | 0.5363 | 476 ms | 0.217 |
| Jev B: top-1 not direct | 0.550 [0.520, 0.583] | +0.122 | 0.5560 | 476 ms | 0.176 |
| Jev A: needs reasoning | 0.510 [0.481, 0.542] | +0.008 | 0.5380 | 467 ms | 0.271 |
| Jev A: ambiguous | 0.469 [0.438, 0.498] | -0.054 | 0.5297 | 467 ms | 0.164 |
| first-stage logistic | 0.520 [0.490, 0.552] | +0.076 | 0.5449 | 345 ms | 0.333 |
| first-stage boosted | 0.502 [0.473, 0.534] | +0.036 | 0.5402 | 345 ms | 0.319 |
| Jev A logistic | 0.476 [0.446, 0.507] | -0.043 | 0.5368 | 467 ms | 0.331 |
| Jev B logistic | 0.517 [0.487, 0.549] | +0.044 | 0.5413 | 476 ms | 0.296 |
| first-stage + Jev B logistic | 0.515 [0.486, 0.545] | +0.066 | 0.5440 | 476 ms | 0.391 |
| first-stage + Jev B boosted | 0.508 [0.476, 0.540] | +0.032 | 0.5338 | 476 ms | 0.281 |

## Paired comparisons (bootstrap over the same queries, 5,000 resamples)

| test set | router A | router B | protocol | ΔAUC (A − B) [95% CI] | p |
|---|---|---|---|---|---|
| SciFact | Jev B: lower candidate better | first-stage boosted | pooled CV | +0.023 [-0.051, +0.099] | 0.536 |
| SciFact | first-stage + Jev B boosted | first-stage boosted | SciFact-trained | +0.078 [+0.036, +0.121] | 0.001 |
| NFCorpus | Jev A: needs reasoning | first-stage logistic | pooled CV | +0.070 [-0.011, +0.151] | 0.084 |
| ArguAna | Jev B: lower candidate better | first-stage boosted | pooled CV | -0.121 [-0.160, -0.081] | 0.000 |
