# First stages on SciFact: SMVE (112 settings) vs MUVERA (54 settings)

All on CPU. MaxSim rerank = exact MaxSim over the first stage's top 100.

## References

| method | nDCG@10 | Recall@100 | nDCG@10 after MaxSim@100 | index | latency |
|---|---|---|---|---|---|
| Exhaustive MaxSim | 0.6991 | 0.9203 | 0.6991 | 3,811.5 MB | 535.4 ms |
| BM25 | 0.6896 | 0.9249 | 0.7086 | 3.6 MB | 1.0 ms |
| BGE-M3 dense | 0.6542 | 0.8970 | 0.6966 | 21.2 MB | 0.7 ms |
| BGE-M3 lexical | 0.6376 | 0.8980 | 0.7048 | 6.0 MB | 2.3 ms |
| BGE-M3 dense + lexical | 0.6842 | 0.9237 | 0.6977 | 27.3 MB | 2.9 ms |

## Equal budget: index size

| budget | metric | SMVE best (setting) | MUVERA best (setting) |
|---|---|---|---|
| ≤ 25 MB | recall@100 | 0.7424 (w4096_k16) | – |
| ≤ 25 MB | rerank_ndcg@10 | 0.6397 (w2048_k8_r2) | – |
| ≤ 25 MB | ndcg@10 | 0.4402 (w65536_k8) | – |
| ≤ 50 MB | recall@100 | 0.8150 (w32768_k16) | 0.5754 (r10_k4_p8_center) |
| ≤ 50 MB | rerank_ndcg@10 | 0.6688 (w8192_k32) | 0.5131 (r10_k4_p8_center) |
| ≤ 50 MB | ndcg@10 | 0.5229 (w32768_k16_center) | 0.2369 (r10_k4_p8) |
| ≤ 100 MB | recall@100 | 0.8573 (w65536_k32_center) | 0.7048 (r10_k4_p16) |
| ≤ 100 MB | rerank_ndcg@10 | 0.6923 (w8192_k8_r4) | 0.6011 (r20_k4_p8) |
| ≤ 100 MB | ndcg@10 | 0.5569 (w8192_k8_r4_center) | 0.3441 (r10_k4_p16) |
| ≤ 200 MB | recall@100 | 0.8713 (w8192_k8_r8_center) | 0.8218 (r20_k4_p16) |
| ≤ 200 MB | rerank_ndcg@10 | 0.6971 (w65536_k64_center) | 0.6709 (r20_k4_p16) |
| ≤ 200 MB | ndcg@10 | 0.5979 (w131072_k64_center) | 0.4795 (r20_k4_p16) |
| ≤ 400 MB | recall@100 | 0.8713 (w8192_k16_r8_center) | 0.8327 (r40_k4_p16_center) |
| ≤ 400 MB | rerank_ndcg@10 | 0.6971 (w65536_k64_center) | 0.6872 (r20_k4_p32_center) |
| ≤ 400 MB | ndcg@10 | 0.5979 (w131072_k64_center) | 0.4985 (r20_k4_p32) |

## Equal budget: single-query latency

| budget | metric | SMVE best (setting) | MUVERA best (setting) |
|---|---|---|---|
| ≤ 10 ms | recall@100 | 0.7424 (w4096_k16) | 0.8580 (r40_k4_p32) |
| ≤ 10 ms | rerank_ndcg@10 | 0.6397 (w2048_k8_r2) | 0.6888 (r40_k4_p32) |
| ≤ 10 ms | ndcg@10 | 0.4164 (w8192_k8_center) | 0.5530 (r40_k4_p32) |
| ≤ 30 ms | recall@100 | 0.8150 (w32768_k16) | 0.8693 (r40_k5_p32) |
| ≤ 30 ms | rerank_ndcg@10 | 0.6755 (w4096_k64) | 0.6915 (r40_k5_p32) |
| ≤ 30 ms | ndcg@10 | 0.5229 (w32768_k16_center) | 0.5530 (r40_k4_p32) |
| ≤ 100 ms | recall@100 | 0.8573 (w65536_k32_center) | 0.8693 (r40_k5_p32) |
| ≤ 100 ms | rerank_ndcg@10 | 0.6923 (w8192_k8_r4) | 0.6915 (r40_k5_p32) |
| ≤ 100 ms | ndcg@10 | 0.5569 (w8192_k8_r4_center) | 0.5559 (r40_k6_p32) |
| ≤ 250 ms | recall@100 | 0.8713 (w8192_k16_r8_center) | 0.8693 (r40_k5_p32) |
| ≤ 250 ms | rerank_ndcg@10 | 0.6971 (w65536_k64_center) | 0.6915 (r40_k5_p32) |
| ≤ 250 ms | ndcg@10 | 0.5979 (w131072_k64_center) | 0.5559 (r40_k6_p32) |

## Top 5 settings per family (by nDCG@10 after MaxSim rerank)

| family | setting | nDCG@10 | Recall@100 | nDCG@10 after MaxSim@100 | index | latency |
|---|---|---|---|---|---|---|
| SMVE | w65536_k64_center | 0.5719 | 0.8647 | 0.6971 | 159.6 MB | 141.5 ms |
| SMVE | w8192_k8_r8_center | 0.5611 | 0.8713 | 0.6970 | 162.4 MB | 144.2 ms |
| SMVE | w8192_k16_r8_center | 0.5681 | 0.8713 | 0.6959 | 270.2 MB | 223.5 ms |
| SMVE | w8192_k16_r8 | 0.5654 | 0.8547 | 0.6929 | 212.1 MB | 176.8 ms |
| SMVE | w8192_k8_r4 | 0.5040 | 0.8483 | 0.6923 | 63.8 MB | 40.4 ms |
| MUVERA | r40_k5_p32 | 0.5438 | 0.8693 | 0.6915 | 849.2 MB | 17.2 ms |
| MUVERA | r40_k5_p32_center | 0.5289 | 0.8647 | 0.6915 | 849.2 MB | 16.7 ms |
| MUVERA | r40_k4_p32_center | 0.5283 | 0.8540 | 0.6909 | 424.6 MB | 10.0 ms |
| MUVERA | r40_k6_p32_center | 0.5440 | 0.8530 | 0.6898 | 1,698.4 MB | 32.0 ms |
| MUVERA | r40_k4_p32 | 0.5530 | 0.8580 | 0.6888 | 424.6 MB | 9.9 ms |
