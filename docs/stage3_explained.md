# Stage 3 explained simply: the alternatives, real latency, and more datasets

Stage 3 asked: **is SMVE a good way to search, compared with the alternatives, at realistic cost,
on more than one dataset?** It had three parts:

- **3a** BM25 and MUVERA, compared with SMVE at the *same budget*
- **3b** a real inverted index, to measure honest query time
- **3c** two more datasets (NFCorpus, ArguAna), to see if the conclusions hold

Like the repetitions note, everything here is explained from scratch, with small formulas and the
intuition behind each one. Numbers are from SciFact unless said otherwise.

---

## 0. The mental model: search has two jobs

| Job | What it does | Must be | Examples |
|---|---|---|---|
| **1. First stage** | look at *all* documents, return ~100 candidates | cheap per document | BM25, dense, SMVE, MUVERA |
| **2. Reranker** | carefully re-order those ~100 | accurate | MaxSim, cross-encoder |

A reranker can only **re-order** what the first stage gives it. If a relevant document isn't in the
candidate list, nothing later can save it. So a first stage is judged mostly by:

```
Recall@100 = (relevant docs found in the top 100) / (all relevant docs)
```

and the final quality by **nDCG@10** (how good the top 10 is, rewarding relevant docs near the top).

We pay three kinds of cost: **storage** (index size), **latency** (time for one query) and **compute**
(arithmetic per query). A method is only "better" if it is better **at the same cost**. That is the idea
of an *equal budget* comparison (section 3).

---

## 1. BM25: the classic word-matching baseline

### The formula

For a query `q` and document `d`:

```
BM25(q, d) = Σ  IDF(t) ·        tf(t,d) · (k1 + 1)
            t∈q          ─────────────────────────────────────────────
                          tf(t,d) + k1 · (1 − b + b · len(d) / avglen)

IDF(t) = ln( 1 + (N − df(t) + 0.5) / (df(t) + 0.5) )
```

| Symbol | Meaning |
|---|---|
| `tf(t,d)` | how many times word t appears in document d |
| `df(t)` | in how many documents word t appears |
| `N` | number of documents |
| `len(d)`, `avglen` | document length, and the average length |
| `k1 = 1.2`, `b = 0.75` | standard settings |

### The intuition, piece by piece

1. **IDF: rare words matter more.** "the" is in every document, so it tells you nothing (IDF ≈ 0).
   "tamoxifen" is in 5 of 5,000 documents: matching it is strong evidence (IDF ≈ 7).
2. **tf with saturation: repeats help, but less and less.** The fraction `tf·(k1+1) / (tf + k1)`
   goes 1.0 → 1.4 → 1.6 → … → never above 2.2 as tf grows. The 5th "protein" adds less than the 1st.
   Without this, a document could win just by repeating a word.
3. **Length normalisation: long documents shouldn't win by being long.** A long document contains
   more words by chance, so `b` makes each occurrence count a bit less when `len(d) > avglen`.

### Why it's fast

Everything except the query depends only on the corpus, so we precompute one weight per (document, word).
A query is then just "add up the weights of the query's words", which is a sparse dot product:

```
score(q, d) = Σ  count(t in q) · weight(t, d)
             t
```

That is exactly the same machinery as SMVE, which is why both can use an inverted index (section 4).

Our hand-written BM25 matches the `bm25s` library on all three datasets (0.6896 vs 0.6896 on SciFact).

**Surprise:** on SciFact, BM25 (0.690 nDCG@10, 1 ms, 4 MB) nearly ties exhaustive MaxSim
(0.699, 535 ms, 3.8 GB). Scientific claims use precise vocabulary, so exact word matching works well.

---

## 2. MUVERA: SMVE's closest relative

MUVERA (Google, 2024) has the same goal as SMVE: turn a *bag* of token vectors into *one* vector whose
dot product approximates MaxSim. The difference is **how tokens are grouped**.

| | SMVE | MUVERA |
|---|---|---|
| grouping | **soft**: each token joins its top-k of w random anchors | **hard**: each token lands in exactly 1 of B buckets |
| query pooling | sum per anchor | sum per bucket |
| doc pooling | mean per anchor | mean per bucket |
| final vector | **sparse** (mostly zeros) | **dense** (every number stored) |

### Step 1: SimHash buckets (random hyperplanes)

Pick `k_sim` random directions `g_1 … g_k`. For a token `x`, write down which **side** of each
hyperplane it falls on:

```
bits(x) = [ sign(x·g_1), sign(x·g_2), …, sign(x·g_k) ]     e.g. [+, −, −, +, +]
bucket(x) = those bits read as a binary number              e.g. 10011₂ = 19
```

With k_sim = 5 there are B = 2⁵ = 32 buckets.

**Why similar tokens share buckets.** For one random hyperplane, two vectors at angle θ end up on the same
side with probability

```
P(same side) = 1 − θ/π
```

(The hyperplane separates them only if it happens to cut through the angle between them, and that's a
θ/π share of all directions.) For all k_sim bits to agree:

```
P(same bucket) = (1 − θ/π)^k_sim
```

| cosine | θ | P(same bucket), k_sim = 5 |
|---|---|---|
| 0.9 | 26° | 0.46 |
| 0.7 | 46° | 0.23 |
| 0.3 | 73° | 0.08 |

Similar tokens usually share a bucket; unrelated ones rarely do. More bits mean finer buckets, but also more
"near-misses" for similar tokens. That's the same coverage problem as SMVE's failure mode A.

### Step 2: shrink each token (random projection ψ)

A 1024-number block per bucket would be huge, so each token is first squeezed to `d_proj` numbers with
a random ±1 matrix:

```
ψ(x) = (1/√d_proj) · S x        S has random ±1 entries
```

On average this keeps dot products intact: `E[ψ(x)·ψ(y)] = x·y`. It's the same "random projections roughly
preserve geometry" idea (Johnson–Lindenstrauss) that SMVE's anchors rely on.

### Step 3: pool per bucket, repeat, concatenate

```
query block for bucket b = Σ ψ(q_i)   over query tokens in bucket b
doc   block for bucket b = mean ψ(d_j) over doc tokens in bucket b
```

**Fill empty buckets:** if no document token landed in bucket b, use the document token whose bits are
*closest* to b (fewest differing bits). Otherwise a query token in b would find nothing at all,
which is exactly SMVE's "S = 0" failure.

Do it all R times with fresh random hyperplanes and concatenate:

```
dimensions = R · 2^k_sim · d_proj        paper setting: 20 · 32 · 16 = 10,240
```

### Why the dot product approximates MaxSim

`query · doc = Σ over buckets of (sum of the query's tokens in b) · (mean of the doc's tokens in b)`.
Each query token gets compared with the *average* of the document tokens near it, the ones in its bucket.
MaxSim compares it with the *best* one. So MUVERA, like SMVE, replaces "max" with "average of the neighbours".

---

## 3. Comparing fairly: equal budget and the Pareto frontier

Both SMVE and MUVERA have knobs (w, k, R, k_sim, d_proj…). Turn them up and quality improves, but so
does cost. Comparing "SMVE at its best" with "MUVERA at its default" would be meaningless.

**Equal budget:** pick a cost limit, find each method's *best* setting under that limit, compare.

**Pareto frontier:** a setting is on the frontier if no other setting is both cheaper *and* better.
The frontier is the best quality you can buy at each price. The plots draw it as a line through the dots.

### Results on SciFact (112 SMVE vs 54 MUVERA settings)

| budget | SMVE Recall@100 | MUVERA Recall@100 |
|---|---|---|
| index ≤ 100 MB | **0.857** | 0.705 |
| query ≤ 20 ms (stage 3b timings) | 0.857 | 0.869 |

**Storage: SMVE wins clearly.** A sparse vector stores only its non-zeros. SMVE at w = 65,536 stores about
2,200 numbers per document. MUVERA stores every one of its thousands to tens of thousands of numbers (10,240 at the paper's setting), most of them small.
That's TopK's core argument: an expressive vector *can be big, as long as it's sparse*.

**Latency: roughly a tie** once SMVE runs on a real inverted index (section 4). MUVERA is still faster below about 10 ms.

### The big lesson: a first stage should *differ* from its reranker

| first stage → MaxSim rerank of top 100 | nDCG@10 (SciFact) |
|---|---|
| **BM25 → MaxSim** | **0.709** |
| exhaustive MaxSim (no first stage) | 0.699 |
| best SMVE → MaxSim | 0.697 |
| best MUVERA → MaxSim | 0.692 |

How can "BM25, then MaxSim" beat "MaxSim on everything"? Think of each method as making **mistakes**:

- MaxSim sometimes ranks a document highly because many of its tokens look vaguely similar
  (a "false friend") even though it doesn't share the key terms.
- BM25 never retrieves a document that shares no words with the query.

When BM25 picks the candidates, many of MaxSim's false friends never reach MaxSim. The two methods'
mistakes are **different**, so combining them filters errors out.

SMVE and MUVERA are *approximations of MaxSim*. Their mistakes are MaxSim's mistakes (plus a few
more), so reranking them with MaxSim can at best recover MaxSim itself (0.697 ≈ 0.699). They add speed,
not a second opinion.

---

## 4. Stage 3b: how an inverted index works, and what makes SMVE slow

### The index

A document matrix stored row by row answers "which words are in document d?". Search needs the opposite:
"which documents contain word t?". So we flip it:

```
word t  →  posting list: [doc ids that contain t], [their weights]
```

That is an **inverted index** (like the index at the back of a book). In code it's simply the
column-by-column (CSC) layout of the same sparse matrix.

### Scoring a query (term at a time)

```
scores = [0, 0, …, 0]                       one per document
for each query term t (weight q_t):
    for each (doc, w) in posting_list(t):
        scores[doc] += q_t · w
return the 100 highest scores
```

Documents sharing no term with the query are never touched. The work is

```
postings read = Σ  df(t)          (sum of posting-list lengths of the query's terms)
               t∈q
```

and the right panel of `inverted_index/plots/latency.png` shows scoring time is almost exactly proportional to it.

### Results (one query at a time, CPU)

| | query terms | postings read | total time | before (scipy code path) |
|---|---|---|---|---|
| BM25 | 9 | 3.8 K | 0.13 ms | 1.0 ms |
| BGE-M3 lexical | 20 | 18 K | 0.17 ms | 2.3 ms |
| **SMVE w=65,536 k=32** | **447** | **216 K** | **19 ms** | 87 ms |

### Finding 1: SMVE queries are very long

Each query token keeps k = 32 anchors, so a 26-token query has up to 26 · 32 = 832 non-zeros (447 after
tokens share anchors). Compare BM25's 9 words. A rough estimate of the work:

```
average posting length ≈ (non-zeros per doc × number of docs) / w = 2,238 × 5,183 / 65,536 ≈ 177
postings read          ≈ 447 × 177 ≈ 79 K
```

We measured **216 K**, about 2.7× more. The reason: queries tend to pick **popular** anchors (ones many
documents also use), and popular anchors have longer posting lists. The same skew makes common words
expensive in text search.

**Why this matters at scale.** Postings grow with the number of documents. At MS MARCO's 8.8 M documents,
an exhaustive search would read roughly 216 K × 8.8 M / 5,183 ≈ **370 million** entries per query.
Real engines avoid reading everything with **dynamic pruning** (WAND / MaxScore): each list knows
its largest weight, and if the remaining lists can't push a document into the top 100, they're skipped.
That works best when a *few* terms dominate the score. With ~450 similar-sized terms, there is little to skip.
**This is SMVE's main open scaling question.**

### Finding 2: SMVE's real bottleneck is encoding the query

| w | anchor matrix B | projection X·B | top-k |
|---|---|---|---|
| 4,096 | 17 MB | 0.8 ms | 0.1 ms |
| 65,536 | **268 MB** | **17.7 ms** | 0.6 ms |

To encode a query, every token is multiplied by every anchor: `X (26 × 1024) · B (1024 × 65,536)`.
That is 2 · 26 · 1024 · 65,536 ≈ 3.5 billion multiply-adds, and it must **read the whole 268 MB matrix B**
each time. With only 26 rows, the CPU spends its time *fetching* B from memory, not computing.
Doubling w doubles both. Fixes: store B in half precision (half the bytes), do it on the GPU, or use a
*structured* random projection (e.g. fast Hadamard transforms) that never stores B at all.

---

## 5. Stage 3c: more datasets, and the idea of headroom

### The datasets

| | SciFact | NFCorpus | ArguAna |
|---|---|---|---|
| task | find abstracts supporting a scientific claim | find medical papers for a health question | find the **counter-argument** to an argument |
| documents | 5,183 | 3,633 | 8,674 |
| test queries | 300 (short) | 323 (short) | 1,406 (**long**, ~200 tokens) |
| relevant docs per query | ~1 | ~38 (graded 1–2) | 1 |

Two quirks we handled:

- **Graded relevance (NFCorpus).** nDCG uses the grade as the gain: `DCG = Σ grade / log2(rank + 1)`,
  so a grade-2 document at the top is worth twice a grade-1 one.
- **Self-matches (ArguAna).** Each query is itself an argument in the corpus, so it would trivially match
  itself at rank 1. BEIR's standard evaluation removes a document whose id equals the query's id; so do we.

### Headroom: the most an approximation could ever add

```
headroom = nDCG@10(exhaustive MaxSim) − nDCG@10(cheap single vector)
```

SMVE and MUVERA *approximate* MaxSim, so at best they reach MaxSim. Over the cheap dense vector they
can add **at most the headroom**. If headroom is small, there's little to win, however good the approximation.

| | SciFact | NFCorpus | ArguAna |
|---|---|---|---|
| exhaustive MaxSim | 0.699 | 0.344 | 0.484 |
| BGE-M3 dense | 0.654 | 0.317 | **0.527** |
| **headroom** | **+0.045** | **+0.028** | **−0.042** |
| SMVE (setting fixed on SciFact) | 0.547 | 0.223 | 0.378 |
| MUVERA (setting fixed on SciFact) | 0.553 | 0.260 | 0.426 |

### Why MaxSim *loses* on ArguAna

```
MaxSim(q, d) = Σ   max  q_i · d_j
              i∈q  j∈d
```

It's a **sum over every query token**. ArguAna queries have ~200 tokens, most of them generic ("people",
"should", "because"…). Each finds a decent match in almost any argument, so those 200 terms add a large,
similar amount to every document and drown the few tokens that matter. A dense vector summarises the
argument *as a whole*, which is what "find the counter-argument" needs. Long, discursive queries are
where token-by-token matching struggles.

### Not cheating: settings fixed in advance

If we picked the best SMVE setting on each dataset's own test queries, we'd be tuning on the answers.
So SMVE and MUVERA are reported at **one setting each, chosen on SciFact beforehand**. The
"best on this dataset" rows exist only to show the optimistic ceiling.

---

## 6. Putting it together: what Stage 3 says about SMVE

We wrote the decision rule *before* looking:

> SMVE has merit if (a) it beats MUVERA on first-stage recall at equal budget, or
> (b) on datasets with large headroom, SMVE + MaxSim beats dense + lexical + MaxSim at similar latency.

- **(a) Partly.** On SciFact SMVE wins at equal **storage** (~5× smaller index at matched recall) and ties at equal
  **latency**. With the setting fixed on SciFact, MUVERA does better on NFCorpus and ArguAna.
- **(b) Not testable here.** With BGE-M3, none of the three datasets has large headroom. Where there is
  some, SMVE + MaxSim is slightly behind dense + lexical + MaxSim.

**Honest summary:** with BGE-M3 on these datasets, SMVE's strengths are **engineering**. It has a small
sparse index, it fits existing inverted-index search engines, and it needs no training or clustering. It is
not better retrieval quality. The setting where it could shine is one with real headroom:
a ColBERT-only model (no free dense vector), or data where token-level matching matters.

**Lessons that hold regardless:**

1. Judge methods at **equal cost**, never at default settings.
2. A first stage should **complement** its reranker (BM25 → MaxSim), not imitate it.
3. Measure **real** latency with the right data structure. Our first SMVE timings were 4.5× too slow.
4. Check **headroom** before investing in approximating an expensive method.

---

## 7. Glossary

| term | meaning |
|---|---|
| first stage / reranker | cheap search over everything / careful re-ordering of the top ~100 |
| Recall@100 | share of relevant documents that made the top 100 |
| equal budget | compare each method's best setting under the same cost limit |
| Pareto frontier | settings not beaten by any setting that is both cheaper and better |
| IDF | inverse document frequency: rare words weigh more |
| SimHash | bucket = signs of a vector's dot products with random hyperplanes |
| FDE | MUVERA's "fixed dimensional encoding", its single dense vector |
| inverted index | word → list of documents containing it (CSC layout) |
| posting list | that list for one word / anchor; its length is the word's df |
| postings read | total posting entries a query touches; drives search time |
| WAND / MaxScore | pruning tricks that skip posting entries that can't reach the top-k |
| headroom | MaxSim's advantage over the cheap method; the most an approximation can add |
| self-match | a query that is itself a corpus document (ArguAna); removed before scoring |
