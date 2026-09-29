# Retrieval methods vs. MaxSim on SciFact (300 test queries)

## Quality (mean over queries; relative change vs. baseline in brackets)

|  | MaxSim | BGE-M3 dense | BGE-M3 lexical | BGE-M3 dense + 0.3·lexical | SMVE w=65,536 k=32 centered | BGE-M3 dense + 0.3·lexical + MaxSim rerank@100 |
|---|---|---|---|---|---|---|
| ndcg@1 | 0.6033 | 0.5300 (-12.2%) | 0.5067 (-16.0%) | 0.5633 (-6.6%) | 0.4300 (-28.7%) | 0.6033 (+0.0%) |
| ndcg@10 | 0.6991 | 0.6542 (-6.4%) | 0.6376 (-8.8%) | 0.6842 (-2.1%) | 0.5471 (-21.7%) | 0.6977 (-0.2%) |
| ndcg@100 | 0.7231 | 0.6801 (-6.0%) | 0.6666 (-7.8%) | 0.7080 (-2.1%) | 0.5844 (-19.2%) | 0.7233 (+0.0%) |
| recall@10 | 0.8106 | 0.7851 (-3.1%) | 0.7654 (-5.6%) | 0.8174 (+0.8%) | 0.6860 (-15.4%) | 0.8072 (-0.4%) |
| recall@100 | 0.9203 | 0.8970 (-2.5%) | 0.8980 (-2.4%) | 0.9237 (+0.4%) | 0.8573 (-6.8%) | 0.9237 (+0.4%) |
| precision@1 | 0.6033 | 0.5300 (-12.2%) | 0.5067 (-16.0%) | 0.5633 (-6.6%) | 0.4300 (-28.7%) | 0.6033 (+0.0%) |
| precision@10 | 0.0913 | 0.0880 (-3.6%) | 0.0853 (-6.6%) | 0.0920 (+0.7%) | 0.0773 (-15.3%) | 0.0910 (-0.4%) |
| mrr@10 | 0.6743 | 0.6217 (-7.8%) | 0.6051 (-10.3%) | 0.6512 (-3.4%) | 0.5131 (-23.9%) | 0.6735 (-0.1%) |
| map@10 | 0.6568 | 0.6060 (-7.7%) | 0.5900 (-10.2%) | 0.6355 (-3.2%) | 0.4969 (-24.4%) | 0.6560 (-0.1%) |

## Cost

Index build = one-off offline work per corpus. Online = encoding the queries + scoring them. Single-query latency = median time for one query on its own. All on CPU.

|  | index build (offline) | online, 300 queries | online per query (batched) | single-query latency | index size |
|---|---|---|---|---|---|
| MaxSim | 0.00 s | 27.64 s | 92.1 ms | 535.4 ms | 3.81 GB |
| BGE-M3 dense | 0.00 s | 0.00 s | 0.0 ms | 0.7 ms | 21.2 MB |
| BGE-M3 lexical | 0.00 s | 0.03 s | 0.1 ms | 2.3 ms | 6.0 MB |
| BGE-M3 dense + 0.3·lexical | 0.00 s | 0.06 s | 0.2 ms | 2.9 ms | 27.3 MB |
| SMVE w=65,536 k=32 centered | 196.60 s | 0.96 s | 3.2 ms | 86.7 ms | 92.8 MB |
| BGE-M3 dense + 0.3·lexical + MaxSim rerank@100 | 0.00 s | 4.15 s | 13.8 ms | 16.6 ms | 3.84 GB |

## Paired comparison against MaxSim

|  | BGE-M3 dense | BGE-M3 lexical | BGE-M3 dense + 0.3·lexical | SMVE w=65,536 k=32 centered | BGE-M3 dense + 0.3·lexical + MaxSim rerank@100 |
|---|---|---|---|---|---|
| ΔnDCG@10 [95% CI] | -0.0449 [-0.0693, -0.0209] | -0.0616 [-0.0951, -0.0291] | -0.0149 [-0.0351, +0.0054] | -0.1520 [-0.1860, -0.1193] | -0.0014 [-0.0043, +0.0000] |
| nDCG@10 better/tied/worse | 34 / 211 / 55 | 42 / 179 / 79 | 34 / 221 / 45 | 16 / 178 / 106 | 18 / 278 / 4 |
| nDCG@10 Wilcoxon p | 0.00019 | 6.3e-05 | 0.13 | 2.3e-16 | 0.13 |
| ΔRecall@100 [95% CI] | -0.0233 [-0.0480, -0.0007] | -0.0223 [-0.0513, +0.0057] | +0.0033 [-0.0140, +0.0207] | -0.0630 [-0.0947, -0.0350] | +0.0033 [-0.0140, +0.0207] |
| top-10 overlap with MaxSim | 0.598 | 0.411 | 0.633 | 0.482 | 0.979 |
| MaxSim top-10 found in top-100 | 0.965 | 0.745 | 0.979 | 0.891 | 0.979 |
| median rank of MaxSim's #1 | 1 | 1 | 1 | 1 | 1 |
| online speed-up | 7469.7× | 1016.1× | 485.7× | 28.7× | 6.7× |
| single-query speed-up | 795.5× | 234.9× | 185.1× | 6.2× | 32.3× |
| index smaller by | 179.5× | 630.3× | 139.7× | 41.1× | 1.0× |
| break-even (queries) | 0 | 0 | 0 | 2,211 | 0 |

## Seed variance (different random anchors)

|  | seeds | nDCG@10 mean | nDCG@10 std | R@100 mean | R@100 std |
|---|---|---|---|---|---|
| (65536, 32, False) | 3 | 0.5436 | 0.0048 | 0.8240 | 0.0079 |
| (65536, 32, True) | 3 | 0.5563 | 0.0080 | 0.8589 | 0.0027 |
