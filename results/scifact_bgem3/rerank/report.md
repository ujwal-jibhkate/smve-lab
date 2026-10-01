# Reranking: cross-encoder vs MaxSim (SciFact, 300 test queries)

Cross-encoder: `BAAI/bge-reranker-v2-m3`, 568M params, fp16 on mps. MaxSim rerank: BGE-M3 ColBERT vectors on CPU. Latency = one query on its own (median of 20), first stage + rerank. GFLOPs = per query.

## All pipelines

| config | ndcg@10 | mrr@10 | recall@10 | recall@100 | latency | GFLOPs | storage | rerank_device |
|---|---|---|---|---|---|---|---|---|
| SMVE | 0.5471 | 0.5131 | 0.6860 | 0.8573 | 87 ms | 3.54 | 92.8 MB | - |
| SMVE + MaxSim @10 | 0.6151 | 0.6031 | 0.6860 | 0.8573 | 106 ms | 3.74 | 3.90 GB | CPU |
| SMVE + MaxSim @20 | 0.6472 | 0.6328 | 0.7276 | 0.8573 | 106 ms | 3.95 | 3.90 GB | CPU |
| SMVE + MaxSim @50 | 0.6735 | 0.6533 | 0.7699 | 0.8573 | 135 ms | 4.58 | 3.90 GB | CPU |
| SMVE + MaxSim @100 | 0.6897 | 0.6676 | 0.7932 | 0.8573 | 169 ms | 5.63 | 3.90 GB | CPU |
| SMVE + cross-encoder @10 | 0.6316 | 0.6241 | 0.6860 | 0.8573 | 923 ms | 2,655.24 | 1.24 GB | mps fp16 |
| SMVE + cross-encoder @20 | 0.6687 | 0.6531 | 0.7427 | 0.8573 | 1,729 ms | 5,335.85 | 1.24 GB | mps fp16 |
| SMVE + cross-encoder @50 | 0.6919 | 0.6677 | 0.7839 | 0.8573 | 3,831 ms | 13,354.47 | 1.24 GB | mps fp16 |
| SMVE + cross-encoder @100 | 0.7099 | 0.6838 | 0.8062 | 0.8573 | 7,577 ms | 26,693.40 | 1.24 GB | mps fp16 |
| Dense + lexical | 0.6842 | 0.6512 | 0.8174 | 0.9237 | 2.9 ms | 0.01 | 27.3 MB | - |
| Dense + lexical + MaxSim @10 | 0.7032 | 0.6770 | 0.8174 | 0.9237 | 8.0 ms | 0.20 | 3.84 GB | CPU |
| Dense + lexical + MaxSim @20 | 0.7016 | 0.6755 | 0.8172 | 0.9237 | 12 ms | 0.39 | 3.84 GB | CPU |
| Dense + lexical + MaxSim @50 | 0.6976 | 0.6734 | 0.8072 | 0.9237 | 26 ms | 0.97 | 3.84 GB | CPU |
| Dense + lexical + MaxSim @100 | 0.6977 | 0.6735 | 0.8072 | 0.9237 | 109 ms | 1.94 | 3.84 GB | CPU |
| Dense + lexical + cross-encoder @10 | 0.7238 | 0.7005 | 0.8174 | 0.9237 | 840 ms | 2,456.55 | 1.17 GB | mps fp16 |
| Dense + lexical + cross-encoder @20 | 0.7305 | 0.7011 | 0.8436 | 0.9237 | 1,645 ms | 4,928.81 | 1.17 GB | mps fp16 |
| Dense + lexical + cross-encoder @50 | 0.7264 | 0.6975 | 0.8329 | 0.9237 | 3,642 ms | 12,372.58 | 1.17 GB | mps fp16 |
| Dense + lexical + cross-encoder @100 | 0.7260 | 0.6965 | 0.8356 | 0.9237 | 7,122 ms | 24,820.31 | 1.17 GB | mps fp16 |

## References

| method | nDCG@10 | latency | GFLOPs | storage |
|---|---|---|---|---|
| Exhaustive MaxSim | 0.6991 | 535.4 ms | 100.39 | 3.81 GB |
| BGE-M3 dense | 0.6542 | 0.7 ms | 0.01 | 21.2 MB |

## Depth-100 pipelines vs exhaustive MaxSim (nDCG@10, paired)

| pipeline | Δ [95% CI] | better / tied / worse | Wilcoxon p |
|---|---|---|---|
| SMVE + MaxSim @100 | -0.0094 [-0.0200, -0.0011] | 20 / 270 / 10 | 0.85 |
| SMVE + cross-encoder @100 | +0.0108 [-0.0150, +0.0365] | 55 / 204 / 41 | 0.42 |
| Dense + lexical + MaxSim @100 | -0.0014 [-0.0043, +0.0000] | 18 / 278 / 4 | 0.13 |
| Dense + lexical + cross-encoder @100 | +0.0268 [-0.0011, +0.0549] | 59 / 199 / 42 | 0.081 |
