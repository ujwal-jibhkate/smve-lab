# SMVE vs. MaxSim on SciFact (300 test queries)

## Quality (mean over queries; relative change vs. baseline in brackets)

|  | MaxSim | SMVE w=65,536 k=32 | SMVE w=65,536 k=32 centered | Hybrid: SMVE top-100 + MaxSim rerank |
|---|---|---|---|---|
| ndcg@1 | 0.6033 | 0.4300 (-28.7%) | 0.4300 (-28.7%) | 0.6000 (-0.6%) |
| ndcg@10 | 0.6991 | 0.5442 (-22.2%) | 0.5471 (-21.7%) | 0.6898 (-1.3%) |
| ndcg@100 | 0.7231 | 0.5741 (-20.6%) | 0.5844 (-19.2%) | 0.7049 (-2.5%) |
| recall@10 | 0.8106 | 0.6880 (-15.1%) | 0.6860 (-15.4%) | 0.7932 (-2.1%) |
| recall@100 | 0.9203 | 0.8250 (-10.4%) | 0.8573 (-6.8%) | 0.8573 (-6.8%) |
| precision@1 | 0.6033 | 0.4300 (-28.7%) | 0.4300 (-28.7%) | 0.6000 (-0.6%) |
| precision@10 | 0.0913 | 0.0767 (-16.1%) | 0.0773 (-15.3%) | 0.0893 (-2.2%) |
| mrr@10 | 0.6743 | 0.5089 (-24.5%) | 0.5131 (-23.9%) | 0.6676 (-1.0%) |
| map@10 | 0.6568 | 0.4939 (-24.8%) | 0.4969 (-24.4%) | 0.6499 (-1.1%) |

## Cost

Index build = one-off offline work per corpus. Online = encoding the queries + scoring them. Single-query latency = median time for one query on its own. All on CPU.

|  | index build (offline) | online, 300 queries | online per query (batched) | single-query latency | index size |
|---|---|---|---|---|---|
| MaxSim | 0.00 s | 27.64 s | 92.1 ms | 535.4 ms | 3.81 GB |
| SMVE w=65,536 k=32 | 186.71 s | 0.96 s | 3.2 ms | 64.7 ms | 72.8 MB |
| SMVE w=65,536 k=32 centered | 196.60 s | 0.96 s | 3.2 ms | 86.7 ms | 92.8 MB |
| Hybrid: SMVE top-100 + MaxSim rerank | 196.60 s | 5.28 s | 17.6 ms | 101.0 ms | 3.90 GB |

## Paired comparison against MaxSim

|  | SMVE w=65,536 k=32 | SMVE w=65,536 k=32 centered | Hybrid: SMVE top-100 + MaxSim rerank |
|---|---|---|---|
| ΔnDCG@10 [95% CI] | -0.1549 [-0.1894, -0.1215] | -0.1520 [-0.1860, -0.1193] | -0.0094 [-0.0200, -0.0011] |
| nDCG@10 better/tied/worse | 21 / 169 / 110 | 16 / 178 / 106 | 20 / 270 / 10 |
| nDCG@10 Wilcoxon p | 6.2e-16 | 2.3e-16 | 0.85 |
| ΔRecall@100 [95% CI] | -0.0953 [-0.1310, -0.0623] | -0.0630 [-0.0947, -0.0350] | -0.0630 [-0.0947, -0.0350] |
| top-10 overlap with MaxSim | 0.451 | 0.482 | 0.891 |
| MaxSim top-10 found in top-100 | 0.867 | 0.891 | 0.891 |
| median rank of MaxSim's #1 | 1 | 1 | 1 |
| online speed-up | 28.8× | 28.7× | 5.2× |
| single-query speed-up | 8.3× | 6.2× | 5.3× |
| index smaller by | 52.3× | 41.1× | 1.0× |
| break-even (queries) | 2,100 | 2,211 | 2,638 |

## Seed variance (different random anchors)

|  | seeds | nDCG@10 mean | nDCG@10 std | R@100 mean | R@100 std |
|---|---|---|---|---|---|
| (65536, 32, False) | 3 | 0.5436 | 0.0048 | 0.8240 | 0.0079 |
| (65536, 32, True) | 3 | 0.5563 | 0.0080 | 0.8589 | 0.0027 |
