# Stage 4a: baseline routers (when to escalate to the cross-encoder)

Base = dense+lexical -> MaxSim@10; escalate = cross-encoder over the top 20. Results on held-out queries only. gain kept: 0 = random routing, 1 = oracle.


## A. pooled 5-fold cross-validation


### SciFact (300 queries; escalation helps on 19%)

never 0.7032 at 5 ms · always 0.7305 at 1,635 ms · oracle best 0.7642

| router | AUC [95% CI] | gain kept | nDCG@10 @10% | @20% | @30% | ECE |
|---|---|---|---|---|---|---|
| oracle | – | +1.000 | 0.7498 | 0.7642 | 0.7642 | nan |
| random | – | +0.000 | 0.7060 | 0.7087 | 0.7114 | nan |
| heuristic: small top-2 gap | 0.658 [0.597, 0.721] | +0.079 | 0.7086 | 0.7124 | 0.7068 | nan |
| heuristic: dense/lexical disagree | 0.420 [0.335, 0.500] | -0.037 | 0.7065 | 0.7157 | 0.7156 | nan |
| logistic | 0.674 [0.597, 0.749] | +0.143 | 0.7082 | 0.7206 | 0.7243 | 0.038 |
| boosted | 0.768 [0.708, 0.823] | +0.246 | 0.7080 | 0.7239 | 0.7367 | 0.044 |

### NFCorpus (323 queries; escalation helps on 32%)

never 0.3315 at 5 ms · always 0.3421 at 1,637 ms · oracle best 0.3636

| router | AUC [95% CI] | gain kept | nDCG@10 @10% | @20% | @30% | ECE |
|---|---|---|---|---|---|---|
| oracle | – | +1.000 | 0.3513 | 0.3597 | 0.3634 | nan |
| random | – | +0.000 | 0.3325 | 0.3336 | 0.3347 | nan |
| heuristic: small top-2 gap | 0.483 [0.413, 0.544] | +0.003 | 0.3329 | 0.3344 | 0.3377 | nan |
| heuristic: dense/lexical disagree | 0.440 [0.374, 0.503] | +0.149 | 0.3339 | 0.3367 | 0.3383 | nan |
| logistic | 0.585 [0.519, 0.644] | -0.033 | 0.3301 | 0.3312 | 0.3323 | 0.126 |
| boosted | 0.540 [0.469, 0.612] | -0.122 | 0.3332 | 0.3325 | 0.3305 | 0.110 |

### ArguAna (1406 queries; escalation helps on 53%)

never 0.4980 at 9 ms · always 0.6975 at 1,689 ms · oracle best 0.7393

| router | AUC [95% CI] | gain kept | nDCG@10 @10% | @20% | @30% | ECE |
|---|---|---|---|---|---|---|
| oracle | – | +1.000 | 0.5760 | 0.6366 | 0.6821 | nan |
| random | – | +0.000 | 0.5181 | 0.5379 | 0.5579 | nan |
| heuristic: small top-2 gap | 0.520 [0.489, 0.550] | +0.052 | 0.5157 | 0.5393 | 0.5606 | nan |
| heuristic: dense/lexical disagree | 0.431 [0.402, 0.461] | -0.079 | 0.5075 | 0.5266 | 0.5447 | nan |
| logistic | 0.593 [0.562, 0.622] | +0.141 | 0.5267 | 0.5507 | 0.5732 | 0.044 |
| boosted | 0.626 [0.599, 0.655] | +0.171 | 0.5264 | 0.5550 | 0.5797 | 0.033 |

## B. trained on SciFact train only (transfer)


### SciFact (300 queries; escalation helps on 19%)

never 0.7032 at 5 ms · always 0.7305 at 1,635 ms · oracle best 0.7642

| router | AUC [95% CI] | gain kept | nDCG@10 @10% | @20% | @30% | ECE |
|---|---|---|---|---|---|---|
| oracle | – | +1.000 | 0.7498 | 0.7642 | 0.7642 | nan |
| random | – | +0.000 | 0.7060 | 0.7087 | 0.7114 | nan |
| heuristic: small top-2 gap | 0.658 [0.591, 0.725] | +0.079 | 0.7086 | 0.7124 | 0.7068 | nan |
| heuristic: dense/lexical disagree | 0.420 [0.340, 0.504] | -0.037 | 0.7065 | 0.7157 | 0.7156 | nan |
| logistic | 0.737 [0.673, 0.798] | +0.175 | 0.7158 | 0.7200 | 0.7237 | 0.039 |
| boosted | 0.743 [0.685, 0.802] | +0.195 | 0.7063 | 0.7151 | 0.7257 | 0.064 |

### NFCorpus (323 queries; escalation helps on 32%)

never 0.3315 at 5 ms · always 0.3421 at 1,637 ms · oracle best 0.3636

| router | AUC [95% CI] | gain kept | nDCG@10 @10% | @20% | @30% | ECE |
|---|---|---|---|---|---|---|
| oracle | – | +1.000 | 0.3513 | 0.3597 | 0.3634 | nan |
| random | – | +0.000 | 0.3325 | 0.3336 | 0.3347 | nan |
| heuristic: small top-2 gap | 0.483 [0.415, 0.557] | +0.003 | 0.3329 | 0.3344 | 0.3377 | nan |
| heuristic: dense/lexical disagree | 0.440 [0.376, 0.506] | +0.149 | 0.3339 | 0.3367 | 0.3383 | nan |
| logistic | 0.536 [0.470, 0.599] | -0.073 | 0.3318 | 0.3305 | 0.3303 | 0.131 |
| boosted | 0.545 [0.477, 0.609] | -0.076 | 0.3334 | 0.3325 | 0.3328 | 0.142 |

### ArguAna (1406 queries; escalation helps on 53%)

never 0.4980 at 9 ms · always 0.6975 at 1,689 ms · oracle best 0.7393

| router | AUC [95% CI] | gain kept | nDCG@10 @10% | @20% | @30% | ECE |
|---|---|---|---|---|---|---|
| oracle | – | +1.000 | 0.5760 | 0.6366 | 0.6821 | nan |
| random | – | +0.000 | 0.5181 | 0.5379 | 0.5579 | nan |
| heuristic: small top-2 gap | 0.520 [0.491, 0.548] | +0.052 | 0.5157 | 0.5393 | 0.5606 | nan |
| heuristic: dense/lexical disagree | 0.431 [0.401, 0.462] | -0.079 | 0.5075 | 0.5266 | 0.5447 | nan |
| logistic | 0.520 [0.489, 0.548] | +0.076 | 0.5187 | 0.5449 | 0.5706 | 0.333 |
| boosted | 0.502 [0.472, 0.531] | +0.036 | 0.5161 | 0.5402 | 0.5628 | 0.319 |

Average latency at an escalation budget = base + budget × cross-encoder time, e.g. SciFact: 10%: 168 ms, 20%: 331 ms, 30%: 494 ms.

## Logistic-regression coefficients (standardised features, all data)

| feature | coefficient |
|---|---|
| lex_gap1_10 | +0.674 |
| hyb_margin12 | -0.635 |
| hyb_gap1_10 | -0.634 |
| hyb_top1 | +0.618 |
| lex_margin12 | +0.365 |
| dense_top1 | +0.265 |
| dense_gap1_10 | -0.263 |
| dense_lex_jaccard10 | +0.227 |
| hyb_std10 | +0.203 |
| lex_top1 | -0.189 |
| lex_top1_in_dense10 | -0.144 |
| q_lex_terms | -0.138 |
| dense_margin12 | -0.137 |
| q_tokens | +0.050 |
| hyb_top1_is_dense_top1 | +0.043 |
| dense_top1_in_lex10 | +0.004 |
