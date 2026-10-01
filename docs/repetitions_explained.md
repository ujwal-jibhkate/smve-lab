# The repetitions experiment, explained simply

This note explains **what** we are testing, **why**, and the **intuition** behind every choice.
No prior knowledge assumed beyond "a vector is a list of numbers" and "a dot product
measures how similar two vectors are".

---

## 1. Quick recap: what SMVE does

Every token (word piece) of a text is a vector `x` with `d` numbers (d = 1024 for BGE-M3).
All token vectors have length 1, so the dot product of two tokens is their **cosine similarity**:

```
cos(x, y) = x · y        (1 = same direction, 0 = unrelated)
```

MaxSim uses these cosines directly. SMVE replaces them with something cheaper:

1. **Anchors.** Pick `w` random directions `b_1 … b_w` (random unit vectors). Think of them
   as `w` random "landmarks" scattered around the space.
2. **Project.** For a token `x`, compute how close it is to every landmark: `x · b_a` for a = 1…w.
3. **Keep the top k.** Remember only the `k` landmarks the token is closest to. Forget the rest.
4. **Pool.** Add up (query) or average (document) these kept values over all tokens of the text.

So each token is summarised as: *"I am near landmarks 17, 402 and 3051, with these strengths."*

---

## 2. How SMVE compares two tokens

Take one query token `x` and one document token `y`. Let `T(x)` be the set of x's top-k
landmarks and `T(y)` the set of y's. SMVE's similarity for this pair is

```
S(x, y) =   Σ  (x · b_a) · (y · b_a)
          a ∈ T(x) ∩ T(y)
```

In words: **look only at landmarks that BOTH tokens picked, multiply their strengths, add up.**

Why this tracks the true cosine: if `x` and `y` point in similar directions, the landmarks
closest to `x` are also close to `y`, so they share many landmarks with big values, and S is large.
If they are unrelated, they share few or no landmarks, and S is small or zero.

(For the curious: for one random unit anchor `b`, the average of `(x·b)(y·b)` is exactly
`(x·y)/d`. So products of projections really do carry the cosine, on average. Top-k keeps the
largest, most informative terms.)

**Analogy.** Two people each pick their k favourite restaurants from a random list of w.
If their tastes are similar (high cosine) they'll probably share some favourites. If tastes differ,
they probably won't. Counting shared favourites is a cheap way to guess how similar their tastes are.

---

## 3. What can go wrong: two failure modes

Because the landmarks are **random**, the guess is sometimes unlucky.

### Failure mode A: "no overlap", S = 0 exactly

Two genuinely similar tokens can end up with **no landmark in common**, just because of where
the random landmarks happened to fall. Then `S = 0`, as if the tokens were unrelated.

- This is a **bias**: it only ever pushes the score *down*, never up.
- It hurts most for medium-similar pairs (cosine 0.4–0.7). Near-identical pairs almost always overlap.

### Failure mode B: "noisy overlap"

Even when they do overlap, *how many* landmarks they share is random. Run it again with different
landmarks and you'd get a different S. The score **wobbles** around its typical value.

- This is **noise** (variance), not bias.

**Restaurant analogy:** A = two friends with similar taste who happen to share zero favourites.
B = they share favourites, but whether it's 2 or 5 is partly luck.

---

## 4. The idea: repetitions

TopK's blog suggests: *do SMVE R times with different random landmarks, then glue the results together.*

```
landmark sets:   B_1, B_2, …, B_R        (each has w random landmarks)
per token:       top-k inside EACH set   →  R·k kept landmarks in total
final vector:    [ SMVE with B_1 | SMVE with B_2 | … | SMVE with B_R ]   (width R·w)
score:           S = S_1 + S_2 + … + S_R
```

### Why it should help failure mode A

If one try has probability `q` of "no overlap", and the tries are independent, then

```
P(no overlap in ALL R tries) = q^R
```

| q (one try) | R = 1 | R = 2 | R = 4 | R = 8 |
|---|---|---|---|---|
| 0.36 | 0.36 | 0.13 | 0.017 | 0.0003 |

One lucky try out of R is enough, so the chance of a total miss shrinks fast.
(In reality the tries aren't perfectly independent, so the real drop is a bit slower. We measured
0.36 falling to about 0.19 at R = 2 and 0.00 at R = 8 for real token pairs at cosine 0.6.)

### Why it should help failure mode B

The total score averages R independent noisy estimates. Averaging reduces noise:

```
if one estimate has spread σ,   the average of R has spread  σ / √R
```

R = 4 halves the noise; R = 16 quarters it. Same reason polls ask 1000 people instead of 10.

---

## 5. The catch: repetitions are not free

Each token now keeps **R·k** landmarks instead of **k**. That means R times more non-zero numbers:
R× storage, R× work. Of course more budget helps; any method gets better with more budget.

So the real question is:

> **Is repetitions a *smarter way to spend* a budget, or just *more* budget?**

To answer that fairly we compare two options **with exactly the same budget**:

| Option | Landmarks | Kept per token |
|---|---|---|
| Repetitions | R blocks of w (total R·w) | k from **each** block (R·k total) |
| One wide matrix | one block of R·w | the top R·k from **anywhere** |

Both use R·w landmarks and keep R·k numbers per token. The **only** difference is *where the
top-k is taken*: per block, or across everything.

**Key insight.** R blocks of w random landmarks, stacked side by side, are *statistically identical*
to one big set of R·w random landmarks. Randomness has no "blocks". So the only thing repetitions
change is the rule "take k from each block", and we'd expect that to matter very little.

**Analogy.** "Pick your 8 favourite restaurants from each of 4 random lists" vs "pick your 32
favourites from all 4 lists combined". With random lists, you end up with nearly the same picks.

That's the **prediction: at equal budget, repetitions ≈ one wide matrix (a tie).**

(In code this is why the implementation is tiny: make one matrix with R·w columns, and take the
top-k inside each block of w columns. See `topk_blocks` in `src/smve_lab/smve.py`.)

---

## 6. Step 1a: testing the mechanism on token pairs (done)

Retrieval scores like nDCG mix many effects together. To see failure modes A and B **directly**,
we test SMVE on single pairs of tokens where we *know* the true cosine.

### How we make a pair with an exact cosine `c`

```
x = random unit vector
u = random unit vector, made perpendicular to x
y = c·x + √(1 − c²)·u
```

Check: `x·y = c·(x·x) + √(1−c²)·(x·u) = c·1 + 0 = c`, and y has length 1. So the pair's true cosine is exactly c.
We do this for c = 0.2, 0.3, …, 0.9, with 1000 pairs each.

### What we measure

| Measurement | Formula / meaning | Which failure mode |
|---|---|---|
| **Zero fraction** | share of pairs with S = 0 exactly | A (coverage) |
| **Coefficient of variation** | `std(S) / mean(S)` over the non-zero pairs, at one cosine level. "How big is the wobble compared to the typical value" | B (noise) |
| **Spearman correlation** | do pairs with higher true cosine get higher S? 1 = same ranking, 0 = no relation. We use ranks because ranking is what retrieval cares about | overall quality |

### Why these settings

- **Synthetic pairs at d = 128 and d = 1024:** d = 128 is the size of ColBERTv2 (what TopK used), d = 1024 is BGE-M3.
  This tests whether dimension changes the story.
- **Real BGE-M3 token pairs:** real tokens aren't spread evenly in space (see section 8), so we check the effect survives on real data.
- **Real pairs, centered:** to see what `--center on` does to single pairs.
- **Focus on cosine ≥ 0.7:** MaxSim takes each query token's *best* match in a document, so losing a strong
  match is what hurts rankings. Losing a weak match (cosine 0.2) barely matters.
- **Budgets ×1, ×2, ×4, ×8:** each doubling of budget, done both ways (repetitions vs one wide matrix).

### What we found

1. **The tie is real.** At every budget, in every setting, repetitions and one wide matrix give the same
   numbers (Spearman within 0.00–0.007). More budget helps a lot, but how you split it doesn't matter.

   | Real BGE-M3 pairs | Spearman | zero fraction at cosine ≥ 0.7 |
   |---|---|---|
   | budget ×1 | 0.73 | 5.5% |
   | budget ×8, repetitions | 0.948 | 0.0% |
   | budget ×8, one wide matrix | 0.949 | 0.0% |

2. **Dimension doesn't matter much.** d = 128 and d = 1024 look almost the same.
3. **Failure mode A is big at small budgets.** With w = 4096, k = 8: 36% of real pairs at cosine 0.6 score exactly zero.
4. **Surprise: centering makes single pairs worse** (see section 8).

---

## 7. Step 1b: the retrieval sweep (running now)

Token pairs are only part of the story. Real SMVE also **pools** many tokens into one vector per
document, and real ranking involves thousands of documents. So we also measure actual retrieval
quality (nDCG@10, Recall@100) on SciFact, for many settings:

| Knob | Values | Why |
|---|---|---|
| `w` (landmarks) | 1K … 65K, plus 131K for some | find where adding landmarks stops helping |
| `k` (kept per token) | 4, 8, 16, 32, 64 | k controls the non-zeros (cost) and failure mode A |
| centering | off / on | see where centering helps or hurts after pooling |
| `R` (repetitions) | 1; and 2, 4, 8 for a few (w, k) | check whether the tie from 1a also holds for retrieval |
| seeds | a few extra for key settings | how much results change with different random landmarks |

**Fairness rule:** when comparing repetitions, the x-axis is **non-zeros per document** (the real budget),
not `w`. Otherwise repetitions would look better simply because they use more budget.

Plots (in `results/scifact_bgem3/smve/sweep/plots/`): quality vs `w` (one line per `k`, with
MaxSim / dense / hybrid as reference lines), heatmaps over (w, k), a centering-effect heatmap,
repetitions vs one wide matrix, and quality vs cost. All numbers: `sweep/summary.md`.

### What we found (120 runs)

**How much do results change by luck?** Re-running the same setting with a different random seed moves
nDCG@10 by about **±0.01** (std 0.005–0.02). So *any difference smaller than ~0.02 is not meaningful.*

1. **The tie holds for real retrieval too.** Across 30 equal-budget pairs, repetitions minus one wide matrix =
   **−0.0007 nDCG@10 on average**, largest difference 0.011, all inside the ±1 seed band
   (`reps_equal_budget.png`: every point sits on the y = x line). Conclusion: *repetitions are just another
   way to spend budget; they are not a trick that makes SMVE better.*
2. **Budget (k) matters most, then w.** With k = 4, SMVE stays around nDCG@10 ≈ 0.3 no matter how large w
   gets. With k = 64 it keeps climbing all the way to w = 128K. In failure-mode terms: small k means too few
   landmarks per token, so failure mode A (S = 0) dominates, and more landmarks can't fix that.
3. **Best standalone SMVE: nDCG@10 = 0.598** (w = 128K, k = 64, centered). That is still below
   BGE-M3 dense (0.654), dense + lexical (0.684) and MaxSim (0.699), while costing ~204 ms per query and a
   185 MB index (dense: 0.7 ms, 21 MB). The k = 64 curve hasn't flattened yet, so bigger settings might
   get a bit closer, but the cost keeps rising too.
4. **As a first stage (Recall@100), SMVE also trails:** best ≈ 0.87, vs 0.897 for dense and 0.924 for dense + lexical.
   On this dataset, dense + lexical is the better candidate generator for MaxSim reranking.
5. **Centering helps in most settings** (mostly blue in `centering_effect.png`): up to **+0.07 nDCG@10** at small
   k / small w, close to zero at large w and k. This matches section 8: it hurts single pairs but helps once
   tokens are pooled into documents, and the help matters most when the budget is small.

**One-line summary:** SMVE's quality is set by its budget (mostly k); repetitions don't change that; centering
is a small, fairly reliable gain; and on SciFact with BGE-M3, standalone SMVE doesn't reach the simpler dense vectors.

---

## 8. Why centering hurts single pairs but can help retrieval

BGE-M3 token vectors all lean towards one shared direction `μ` (the average token).
Its length is |μ| = 0.54, so |μ|² ≈ 0.29: **every pair of tokens gets about 0.29 of "free" cosine**
just from this shared lean.

Centering removes it: `x' = normalize(x − μ)`. If a typical token has `x·μ ≈ |μ|²`, then

```
(x − μ)·(y − μ) = x·y − x·μ − y·μ + |μ|²  ≈  c − 0.29
|x − μ|²        = 1 − 2·x·μ + |μ|²         ≈  0.71

so the centered cosine  c' ≈ (c − 0.29) / 0.71
```

A pair with original cosine 0.7 becomes about **0.58** after centering: it *looks* less similar,
shares fewer landmarks, and hits "S = 0" more often (5.5% → 44% in our test). That's why
centering hurts at the pair level.

On SciFact retrieval, though, centering helped a little. So its benefit must come from the **pooling
step**, which the pair test doesn't include. Our hypothesis: a few landmarks pointing along μ get
picked by almost *every* token, and a document's average on those landmarks becomes an
uninformative blur shared by all documents. Centering stops that. The sweep's centering heatmap
will show where each effect wins.

---

## 9. Glossary

| Term | Meaning |
|---|---|
| anchor / landmark | a random direction `b_a` used to describe tokens |
| `w` | number of anchors per block |
| `k` | anchors kept per token (per block) |
| `R` | repetitions: number of independent anchor blocks |
| budget | non-zeros per token (R·k) and anchors (R·w); what storage and speed depend on |
| cosine | similarity of two unit vectors, their dot product |
| bias | an error that always pushes the same way (here: down) |
| variance / noise | random wobble around the typical value |
| Spearman correlation | agreement between two rankings (1 = identical order) |
| coefficient of variation | std / mean: noise size relative to the typical value |
| nDCG@10, Recall@100 | retrieval quality: rank quality of the top 10, and share of relevant docs found in the top 100 |
