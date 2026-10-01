# All methods across datasets (SciFact, NFCorpus, ArguAna)

SMVE / MUVERA 'fixed setting' = chosen on SciFact beforehand; 'best' = picked on each dataset's test set (optimistic). '→MaxSim' = exact MaxSim rerank of the first stage's top 100.

## nDCG@10 (first stage alone)

| method | SciFact | NFCorpus | ArguAna |
|---|---|---|---|
| Exhaustive MaxSim | 0.6991 | 0.3441 | 0.4841 |
| BM25 | 0.6896 | 0.3279 | 0.4919 |
| BGE-M3 dense | 0.6542 | 0.3165 | 0.5266 |
| BGE-M3 lexical | 0.6376 | 0.2857 | 0.3405 |
| BGE-M3 dense + lexical | 0.6842 | 0.3287 | 0.4986 |
| SMVE (fixed setting) | 0.5471 | 0.2225 | 0.3783 |
| SMVE (best on this dataset, optimistic) | 0.5719 | 0.2489 | 0.3783 |
| MUVERA (fixed setting) | 0.5530 | 0.2595 | 0.4262 |
| MUVERA (best on this dataset, optimistic) | 0.5438 | 0.2350 | 0.4089 |

## Recall@100

| method | SciFact | NFCorpus | ArguAna |
|---|---|---|---|
| Exhaustive MaxSim | 0.9203 | 0.2914 | 0.9666 |
| BM25 | 0.9249 | 0.2572 | 0.9730 |
| BGE-M3 dense | 0.8970 | 0.2825 | 0.9815 |
| BGE-M3 lexical | 0.8980 | 0.2335 | 0.9303 |
| BGE-M3 dense + lexical | 0.9237 | 0.2886 | 0.9879 |
| SMVE (fixed setting) | 0.8573 | 0.2106 | 0.9239 |
| SMVE (best on this dataset, optimistic) | 0.8647 | 0.2331 | 0.9239 |
| MUVERA (fixed setting) | 0.8580 | 0.2382 | 0.9502 |
| MUVERA (best on this dataset, optimistic) | 0.8693 | 0.2338 | 0.9538 |

## nDCG@10 after MaxSim rerank of the top 100

| method | SciFact | NFCorpus | ArguAna |
|---|---|---|---|
| Exhaustive MaxSim | 0.6991 | 0.3441 | 0.4841 |
| BM25 | 0.7086 | 0.3427 | 0.4884 |
| BGE-M3 dense | 0.6966 | 0.3393 | 0.4844 |
| BGE-M3 lexical | 0.7048 | 0.3321 | 0.4877 |
| BGE-M3 dense + lexical | 0.6977 | 0.3418 | 0.4849 |
| SMVE (fixed setting) | 0.6898 | 0.3180 | 0.4838 |
| SMVE (best on this dataset, optimistic) | 0.6971 | 0.3262 | 0.4838 |
| MUVERA (fixed setting) | 0.6888 | 0.3261 | 0.4838 |
| MUVERA (best on this dataset, optimistic) | 0.6915 | 0.3305 | 0.4840 |

## Headroom (Q2)

| dataset | MaxSim | headroom vs dense | headroom vs dense+lexical | headroom vs BM25 | SMVE − dense (alone) | MUVERA − dense (alone) | SMVE→MaxSim − dense+lex→MaxSim | MUVERA→MaxSim − dense+lex→MaxSim | best first stage →MaxSim |
|---|---|---|---|---|---|---|---|---|---|
| SciFact | 0.6991 | +0.0449 | +0.0149 | +0.0096 | -0.1071 | -0.1012 | -0.0079 | -0.0089 | BM25 |
| NFCorpus | 0.3441 | +0.0277 | +0.0154 | +0.0162 | -0.0939 | -0.0570 | -0.0238 | -0.0156 | Exhaustive MaxSim |
| ArguAna | 0.4841 | -0.0425 | -0.0145 | -0.0078 | -0.1483 | -0.1004 | -0.0011 | -0.0011 | BM25 |
