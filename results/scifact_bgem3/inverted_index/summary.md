# Query latency with a hand-built inverted index (SciFact, 300 queries, CPU)

Times are medians per query in ms; `old_latency_ms` = the earlier general-purpose code path. Scores identical to the earlier evaluation (checked).

| method | index_mb | query_nnz | postings_read_mean | encode_ms | score_ms | topk_ms | total_ms | total_p95_ms | old_latency_ms | ndcg@10 | recall@100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SMVE w4096_k16 | 21.83 | 161.30 | 255,826.87 | 1.49 | 1.64 | 0.0503 | 3.44 | 4.30 | 8.47 | 0.3948 | 0.7424 |
| SMVE w16384_k32_center | 67.42 | 377.11 | 363,305.50 | 9.43 | 2.46 | 0.0639 | 11.95 | 14.34 | 36.86 | 0.5185 | 0.8323 |
| SMVE w65536_k32_center | 93.32 | 446.78 | 216,480.64 | 17.79 | 1.55 | 0.0615 | 19.39 | 22.94 | 86.65 | 0.5471 | 0.8573 |
| SMVE w8192_k8_r4 | 64.03 | 392.09 | 372,791.31 | 11.10 | 2.51 | 0.0599 | 13.74 | 16.88 | 40.40 | 0.5040 | 0.8483 |
| BM25 | 3.77 | 8.58 | 3,750.62 | 0.0430 | 0.0361 | 0.0466 | 0.1290 | 0.1760 | 1.04 | 0.6896 | 0.9249 |
| BGE-M3 lexical | 8.03 | 20.17 | 17,803.68 | 0.0083 | 0.1130 | 0.0480 | 0.1697 | 0.2442 | 2.28 | 0.6376 | 0.8980 |
| BGE-M3 dense (brute force) | 21.23 | nan | nan | 0.0003 | 0.4309 | 0.0528 | 0.4854 | 0.5261 | 0.6730 | 0.6542 | 0.8970 |
| MUVERA r40_k4_p32 (brute force) | 424.59 | nan | nan | 1.22 | 8.27 | 0.0846 | 9.61 | 10.89 | 9.90 | 0.5530 | 0.8580 |
