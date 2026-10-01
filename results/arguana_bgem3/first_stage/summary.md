# First stages on ArguAna: SMVE (18 settings) vs MUVERA (8 settings)

All on CPU. MaxSim rerank = exact MaxSim over the first stage's top 100.

## References

| method | nDCG@10 | Recall@100 | nDCG@10 after MaxSim@100 | index | latency |
|---|---|---|---|---|---|
| Exhaustive MaxSim | 0.4841 | 0.9666 | 0.4841 | 3,992.8 MB | 827.3 ms |
| BM25 | 0.4919 | 0.9730 | 0.4884 | 4.6 MB | 1.5 ms |
| BGE-M3 dense | 0.5266 | 0.9815 | 0.4844 | 35.5 MB | 0.9 ms |
| BGE-M3 lexical | 0.3405 | 0.9303 | 0.4877 | 7.6 MB | 2.6 ms |
| BGE-M3 dense + lexical | 0.4986 | 0.9879 | 0.4849 | 43.1 MB | 3.9 ms |

## Equal budget: index size

| budget | metric | SMVE best (setting) | MUVERA best (setting) |
|---|---|---|---|
| ≤ 25 MB | recall@100 | – | – |
| ≤ 25 MB | rerank_ndcg@10 | – | – |
| ≤ 25 MB | ndcg@10 | – | – |
| ≤ 50 MB | recall@100 | 0.8784 (w65536_k16) | – |
| ≤ 50 MB | rerank_ndcg@10 | 0.4779 (w65536_k16) | – |
| ≤ 50 MB | ndcg@10 | 0.3319 (w65536_k16) | – |
| ≤ 100 MB | recall@100 | 0.9154 (w65536_k16_center) | – |
| ≤ 100 MB | rerank_ndcg@10 | 0.4830 (w16384_k32_center) | – |
| ≤ 100 MB | ndcg@10 | 0.3751 (w65536_k16_center) | – |
| ≤ 200 MB | recall@100 | 0.9239 (w65536_k32_center) | 0.9168 (r20_k4_p16) |
| ≤ 200 MB | rerank_ndcg@10 | 0.4838 (w65536_k32_center) | 0.4816 (r20_k4_p16) |
| ≤ 200 MB | ndcg@10 | 0.3783 (w65536_k32_center) | 0.3844 (r20_k4_p16) |
| ≤ 400 MB | recall@100 | 0.9239 (w65536_k32_center) | 0.9346 (r20_k5_p16) |
| ≤ 400 MB | rerank_ndcg@10 | 0.4838 (w65536_k32_center) | 0.4836 (r40_k4_p16) |
| ≤ 400 MB | ndcg@10 | 0.3783 (w65536_k32_center) | 0.3915 (r20_k5_p16) |

## Equal budget: single-query latency

| budget | metric | SMVE best (setting) | MUVERA best (setting) |
|---|---|---|---|
| ≤ 10 ms | recall@100 | – | 0.9168 (r20_k4_p16) |
| ≤ 10 ms | rerank_ndcg@10 | – | 0.4816 (r20_k4_p16) |
| ≤ 10 ms | ndcg@10 | – | 0.3844 (r20_k4_p16) |
| ≤ 30 ms | recall@100 | 0.8762 (w4096_k16_center) | 0.9538 (r40_k5_p32) |
| ≤ 30 ms | rerank_ndcg@10 | 0.4765 (w4096_k16_center) | 0.4840 (r40_k5_p32) |
| ≤ 30 ms | ndcg@10 | 0.3194 (w4096_k16_center) | 0.4262 (r40_k4_p32) |
| ≤ 100 ms | recall@100 | 0.9154 (w65536_k16_center) | 0.9538 (r40_k5_p32) |
| ≤ 100 ms | rerank_ndcg@10 | 0.4830 (w16384_k32_center) | 0.4840 (r40_k5_p32) |
| ≤ 100 ms | ndcg@10 | 0.3751 (w65536_k16_center) | 0.4262 (r40_k4_p32) |
| ≤ 250 ms | recall@100 | 0.9239 (w65536_k32_center) | 0.9538 (r40_k5_p32) |
| ≤ 250 ms | rerank_ndcg@10 | 0.4838 (w65536_k32_center) | 0.4840 (r40_k5_p32) |
| ≤ 250 ms | ndcg@10 | 0.3783 (w65536_k32_center) | 0.4262 (r40_k4_p32) |

## Top 5 settings per family (by nDCG@10 after MaxSim rerank)

| family | setting | nDCG@10 | Recall@100 | nDCG@10 after MaxSim@100 | index | latency |
|---|---|---|---|---|---|---|
| MUVERA | r40_k5_p32 | 0.4089 | 0.9538 | 0.4840 | 1,421.1 MB | 28.2 ms |
| SMVE | w65536_k32_center | 0.3783 | 0.9239 | 0.4838 | 118.1 MB | 127.9 ms |
| MUVERA | r40_k4_p32 | 0.4262 | 0.9502 | 0.4838 | 710.6 MB | 16.0 ms |
| MUVERA | r40_k4_p16 | 0.3890 | 0.9324 | 0.4836 | 355.3 MB | 13.8 ms |
| SMVE | w16384_k32_center | 0.3542 | 0.9097 | 0.4830 | 87.1 MB | 51.6 ms |
| MUVERA | r40_k5_p16 | 0.3877 | 0.9367 | 0.4829 | 710.6 MB | 16.6 ms |
| SMVE | w65536_k64_center | 0.3753 | 0.9239 | 0.4829 | 204.5 MB | 200.7 ms |
| MUVERA | r20_k5_p16 | 0.3915 | 0.9346 | 0.4825 | 355.3 MB | 13.7 ms |
| SMVE | w16384_k64_center | 0.3490 | 0.9047 | 0.4822 | 143.4 MB | 111.3 ms |
| SMVE | w65536_k64 | 0.3455 | 0.8905 | 0.4818 | 147.4 MB | 158.1 ms |
