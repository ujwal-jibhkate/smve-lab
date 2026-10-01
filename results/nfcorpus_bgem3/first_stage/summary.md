# First stages on NFCorpus: SMVE (18 settings) vs MUVERA (8 settings)

All on CPU. MaxSim rerank = exact MaxSim over the first stage's top 100.

## References

| method | nDCG@10 | Recall@100 | nDCG@10 after MaxSim@100 | index | latency |
|---|---|---|---|---|---|
| Exhaustive MaxSim | 0.3441 | 0.2914 | 0.3441 | 2,802.7 MB | 342.5 ms |
| BM25 | 0.3279 | 0.2572 | 0.3427 | 2.8 MB | 0.7 ms |
| BGE-M3 dense | 0.3165 | 0.2825 | 0.3393 | 14.9 MB | 0.2 ms |
| BGE-M3 lexical | 0.2857 | 0.2335 | 0.3321 | 4.6 MB | 1.6 ms |
| BGE-M3 dense + lexical | 0.3287 | 0.2886 | 0.3418 | 19.5 MB | 2.1 ms |

## Equal budget: index size

| budget | metric | SMVE best (setting) | MUVERA best (setting) |
|---|---|---|---|
| ≤ 25 MB | recall@100 | 0.1901 (w16384_k16) | – |
| ≤ 25 MB | rerank_ndcg@10 | 0.2946 (w4096_k32) | – |
| ≤ 25 MB | ndcg@10 | 0.1651 (w16384_k16) | – |
| ≤ 50 MB | recall@100 | 0.2158 (w65536_k32) | – |
| ≤ 50 MB | rerank_ndcg@10 | 0.3209 (w65536_k32) | – |
| ≤ 50 MB | ndcg@10 | 0.2146 (w65536_k32) | – |
| ≤ 100 MB | recall@100 | 0.2331 (w65536_k64) | 0.2054 (r20_k4_p16) |
| ≤ 100 MB | rerank_ndcg@10 | 0.3262 (w65536_k64) | 0.3171 (r20_k4_p16) |
| ≤ 100 MB | ndcg@10 | 0.2489 (w65536_k64) | 0.2229 (r20_k4_p16) |
| ≤ 200 MB | recall@100 | 0.2331 (w65536_k64) | 0.2339 (r40_k4_p16) |
| ≤ 200 MB | rerank_ndcg@10 | 0.3262 (w65536_k64) | 0.3266 (r40_k4_p16) |
| ≤ 200 MB | ndcg@10 | 0.2489 (w65536_k64) | 0.2504 (r40_k4_p16) |
| ≤ 400 MB | recall@100 | 0.2331 (w65536_k64) | 0.2382 (r40_k4_p32) |
| ≤ 400 MB | rerank_ndcg@10 | 0.3262 (w65536_k64) | 0.3305 (r20_k5_p32) |
| ≤ 400 MB | ndcg@10 | 0.2489 (w65536_k64) | 0.2595 (r40_k4_p32) |

## Equal budget: single-query latency

| budget | metric | SMVE best (setting) | MUVERA best (setting) |
|---|---|---|---|
| ≤ 10 ms | recall@100 | 0.1838 (w4096_k32) | 0.2382 (r40_k4_p32) |
| ≤ 10 ms | rerank_ndcg@10 | 0.2946 (w4096_k32) | 0.3305 (r20_k5_p32) |
| ≤ 10 ms | ndcg@10 | 0.1620 (w4096_k32) | 0.2595 (r40_k4_p32) |
| ≤ 30 ms | recall@100 | 0.2173 (w16384_k64) | 0.2394 (r40_k5_p32) |
| ≤ 30 ms | rerank_ndcg@10 | 0.3172 (w16384_k64) | 0.3305 (r20_k5_p32) |
| ≤ 30 ms | ndcg@10 | 0.2181 (w16384_k64) | 0.2617 (r40_k5_p32) |
| ≤ 100 ms | recall@100 | 0.2331 (w65536_k64) | 0.2394 (r40_k5_p32) |
| ≤ 100 ms | rerank_ndcg@10 | 0.3262 (w65536_k64) | 0.3305 (r20_k5_p32) |
| ≤ 100 ms | ndcg@10 | 0.2489 (w65536_k64) | 0.2617 (r40_k5_p32) |
| ≤ 250 ms | recall@100 | 0.2331 (w65536_k64) | 0.2394 (r40_k5_p32) |
| ≤ 250 ms | rerank_ndcg@10 | 0.3262 (w65536_k64) | 0.3305 (r20_k5_p32) |
| ≤ 250 ms | ndcg@10 | 0.2489 (w65536_k64) | 0.2617 (r40_k5_p32) |

## Top 5 settings per family (by nDCG@10 after MaxSim rerank)

| family | setting | nDCG@10 | Recall@100 | nDCG@10 after MaxSim@100 | index | latency |
|---|---|---|---|---|---|---|
| MUVERA | r20_k5_p32 | 0.2350 | 0.2338 | 0.3305 | 297.6 MB | 7.1 ms |
| MUVERA | r40_k4_p16 | 0.2504 | 0.2339 | 0.3266 | 148.8 MB | 6.1 ms |
| SMVE | w65536_k64 | 0.2489 | 0.2331 | 0.3262 | 79.1 MB | 71.2 ms |
| MUVERA | r40_k4_p32 | 0.2595 | 0.2382 | 0.3261 | 297.6 MB | 7.0 ms |
| SMVE | w65536_k64_center | 0.2339 | 0.2325 | 0.3256 | 107.6 MB | 99.6 ms |
| MUVERA | r40_k5_p32 | 0.2617 | 0.2394 | 0.3253 | 595.2 MB | 11.8 ms |
| MUVERA | r40_k5_p16 | 0.2465 | 0.2282 | 0.3229 | 297.6 MB | 6.8 ms |
| SMVE | w65536_k32 | 0.2146 | 0.2158 | 0.3209 | 46.1 MB | 44.2 ms |
| SMVE | w65536_k32_center | 0.2225 | 0.2106 | 0.3180 | 62.7 MB | 53.3 ms |
| SMVE | w16384_k64 | 0.2181 | 0.2173 | 0.3172 | 55.5 MB | 27.8 ms |
