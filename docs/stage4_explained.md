# Stage 4 explained simply: when is the expensive reranker worth it?

Stage 4 asked one practical question:

> A cross-encoder reranks better than anything else we have, but it costs about **1.6 seconds** per query.
> Can we call it **only for the queries it will actually help**, and skip it for the rest?

It had two parts:

- **4a** build a per-query dataset of "did the cross-encoder help?", and test cheap routers that decide
  from first-stage scores
- **4b** use **Jev**, a fast decision model, as the router, and compare it fairly with 4a

Like the earlier notes, everything is explained from scratch, with small formulas and intuition.
The cross-encoder itself was introduced in stage 2; section 1 recaps it.

---

## 1. The cross-encoder: why it's accurate, and why it's slow

So far we've met three ways to compare a query with a document:

| | how it works | when the work happens | cost per query |
|---|---|---|---|
| **bi-encoder** (dense) | query → 1 vector, doc → 1 vector, dot product | docs encoded once, offline | tiny |
| **late interaction** (MaxSim) | token vectors on both sides, compared token by token | docs encoded once, offline | medium |
| **cross-encoder** | query and document read **together** by one transformer → one score | **everything at query time** | huge |

A cross-encoder (`bge-reranker-v2-m3`, 568 M parameters) takes the input

```
[CLS] query tokens [SEP] document tokens [SEP]
```

and every query token can **attend to** every document token inside the network. It can notice things
like "this abstract *contradicts* the claim" or "this argument is the *counter* to that one", which
vector comparisons can't see. That's why it's the most accurate.

The price: nothing can be precomputed, because the document is read *together with this query*.
The work per (query, document) pair is roughly

```
FLOPs ≈ 2 · P · n  +  4 · L · n² · h
       ─────────    ──────────────
       weights      attention
```

where P = 303 M weight parameters (not counting the vocabulary table), n ≈ 396 tokens per pair, and
L = 24 layers of width h = 1024. That's ≈ **2.6 × 10¹¹** FLOPs per pair. Reranking 20 candidates costs
about **5 TFLOP per query**, so **~1.6 s** on the M1 Pro GPU, which is already near its limit
(~14 pairs per second). For comparison, exhaustive MaxSim over the whole corpus is 0.1 TFLOP.

**Storage flips the other way:** the cross-encoder needs only its weights (1.1 GB, fixed) plus the raw text.
MaxSim needs every token vector (3.8 GB for SciFact, growing with the corpus).

---

## 2. The routing idea

Every query first goes through a cheap **base** pipeline. The router then makes **one decision**:

```
base (always, ~5–15 ms):   dense + lexical first stage  →  MaxSim rerank of the top 10
escalate? (+ ~1.6 s):      cross-encoder rerank of the first stage's top 20
```

Why only this one decision? The base already costs almost nothing. The only expensive choice is the
cross-encoder.

### The oracle: how good could routing be?

Imagine a router that **knows the answer**: it escalates exactly the queries where the cross-encoder
improves nDCG@10.

| | SciFact | NFCorpus | ArguAna |
|---|---|---|---|
| never escalate | 0.703 | 0.332 | 0.498 |
| always escalate (1.6 s each) | 0.731 | 0.342 | 0.698 |
| **oracle** | **0.764** | **0.364** | **0.739** |
| queries where escalating helps | 19 % | 32 % | 53 % |

The oracle is **better than always escalating**, not just cheaper. The reason is that the cross-encoder
sometimes makes a ranking *worse*, and the oracle skips those queries. It also calls the cross-encoder on only
19–53 % of queries. So there is real value in routing, *if* we can predict which queries benefit.

On ArguAna (finding the counter-argument), the cross-encoder adds **+0.20**. Reading the argument and
candidate together is exactly what that task needs.

---

## 3. The routing dataset (4a)

For each of **2,838 queries** (SciFact test and train, NFCorpus, ArguAna) we stored:

- **outcomes:** nDCG@10 of the first stage alone, of the base, and of the escalated pipeline
- **label:** `y = 1` if escalating strictly improves nDCG@10 over the base, else `0`
- **gain:** `nDCG(escalate) − nDCG(base)` (can be negative)
- **16 features** that exist *before* any reranking, computed from the first stage alone:

| feature | intuition |
|---|---|
| top-1 score | how strongly the best candidate matches |
| margin between #1 and #2 | a small gap means the first stage can't decide |
| gap between #1 and #10, spread of the top 10 | is there a clear winner or a flat crowd? |
| the same for dense and lexical separately | each signal's own confidence |
| overlap of the dense top-10 and lexical top-10 | do the two signals agree? |
| query length, number of lexical terms | long or wordy queries behave differently |

Using only these means routing costs essentially nothing (≈ 0.01 ms).

---

## 4. How to judge a router

A router gives every query a **score**: "how likely is escalating to help?". We then escalate the
**top p %** of queries by that score.

### The quality-vs-cost curve

If we escalate the set S of flagged queries:

```
nDCG(p) = nDCG(base) + (1/N) · Σ  gain(q)
                                q∈S
average latency(p) = base + overhead + p · (cross-encoder time)
```

Sweeping p from 0 % to 100 % draws a curve from "never" to "always":

- **random** escalation is a straight line (on average every query has the same expected gain)
- the **oracle** shoots up first (it picks the biggest gains first), then comes down at the end
  (where it is forced to escalate queries the cross-encoder hurts)

A good router's curve lies between random and oracle, as close to the oracle as possible.

### Gain kept

One number for "how close to the oracle", using areas under the curve (above the base):

```
gain kept = (area_router − area_random) / (area_oracle − area_random)
```

0 = no better than random, 1 = as good as the oracle.

### AUC

AUC answers: **pick one query where escalating helped and one where it didn't. How often does the router
give the helped one a higher score?**

- 0.5 = random guessing
- 1.0 = perfect separation

### Calibration and ECE

A router's score is often meant as a probability: "70 % chance escalating helps". It is **calibrated**
if, among all queries where it says 70 %, escalating really helps about 70 % of the time.
Group predictions into bins and measure the average mismatch:

```
ECE = Σ  (n_b / N) · | average predicted probability in bin b − actual helped share in bin b |
     bins
```

0 = perfectly calibrated. Calibration matters for real decisions, e.g. "escalate when P > 50 %".

### Two fair test protocols

Every number is measured on queries the router **never trained on**:

- **A. pooled 5-fold cross-validation.** Split all queries into 5 parts; train on 4, test on the 5th; repeat.
  This tests routing on familiar kinds of data.
- **B. transfer.** Train only on SciFact's training queries, then test on SciFact test, NFCorpus and ArguAna.
  This asks: does a router learned on one dataset work on new ones?

---

## 5. Baseline routers (4a): weak signal

| router | SciFact AUC | NFCorpus AUC | ArguAna AUC |
|---|---|---|---|
| rule: small top-2 gap | 0.66 | 0.48 | 0.52 |
| logistic regression (16 features) | 0.67 | 0.59 | 0.59 |
| boosted trees (16 features) | **0.77** | 0.54 | **0.63** |

- **Some signal, but not much.** The best routers keep only ~15–25 % of the oracle's gain.
- **It can still pay off:** on SciFact, escalating the boosted router's top 30 % (~0.5 s per query on average)
  gives **0.737**, *better* than always escalating (0.731 at 1.6 s).
- **Nothing beats random on NFCorpus.**
- **Transfer fails:** trained on SciFact, the logistic router scores AUC 0.52 on ArguAna and is badly
  calibrated (ECE 0.33). ArguAna's base rate (53 % helped) is very different from SciFact's (17 %).

The core limitation: all 16 features are numbers **about the scores**. None of them reads the query
or the documents.

---

## 6. Jev: a fast decision model (4b)

### What Jev is

Jev (TypeSafe, `jev-1.13.0`) doesn't write text. You send a **state** (text or JSON) plus typed
**questions**, and it answers all of them in parallel in ~130 ms:

| question type | asks | returns |
|---|---|---|
| **Noul** | a yes/no question | P(yes), a number from 0 to 1 |
| Choice | pick one option | a probability per option |
| Score | rate on levels | a probability per level |

Its training (they call it RLCD) aims for **calibrated** probabilities: when it says 0.8, it's right about 80 %
of the time, *for the question it was asked*.

### Designing the questions (following Jev's own guidance)

Jev's docs list known weak spots, and each shaped the design:

| weak spot | what we did |
|---|---|
| bad with numbers | never show it scores; numeric signals stay in code |
| reads literally | precise wording, explicit "true" / "false" criteria for each question |
| distracted by irrelevant text | send only the query and the top 3 candidates, cut to ~120 words |
| prefers simple, atomic questions | several small yes/no questions, combined in code |

Two versions of the state:

- **A: query only.** Can run *alongside* the first stage, so it adds almost no waiting.
- **B: query + top-3 candidates.** Knows more, but must wait for the first stage.

The questions (each a Noul):

| | question | used in |
|---|---|---|
| ambiguous | Is the query vague, so several different documents could fit? | A, B |
| needs reasoning | Does matching require reasoning about the *relationship* (supports, contradicts, argues against), not just topic? | A, B |
| specific terms | Does the query contain distinctive terms the right document would share? | A, B |
| top-1 direct | Does candidate 1 directly address the query? | B |
| lower better | Does candidate 2 or 3 address it *more* directly than candidate 1? | B |
| any direct | Does at least one candidate directly address it? | B |

### Two ways to use the answers

1. **Zero-shot routers: no training at all.** Use one answer directly as the routing score.
   **The directions were fixed before looking at any results**, e.g. "lower candidate better → escalate".
2. **Stacked routers.** Feed Jev's answers (alone, or next to the 16 features) into the same logistic /
   boosted models, under the same folds.

We also ran a 50-queries-per-dataset **pilot** first, only to check cost, latency and that answers vary.
**No question was reworded after seeing which ones predicted well.** Rewording based on results would be
tuning on the answers.

### Charging Jev for its time

```
state A:  overhead = max(0, Jev latency − base latency)    (runs in parallel)
state B:  overhead = full Jev latency (~131 ms)            (must wait for the first stage)
```

This overhead is added to **every** query, escalated or not.

---

## 7. Jev results

| router | training | SciFact | NFCorpus | ArguAna |
|---|---|---|---|---|
| best feature router (4a) | yes | 0.77 | 0.59 | **0.63** |
| **Jev: lower candidate better** | **none** | **0.79** | 0.58 | 0.51 |
| **Jev: needs reasoning** | **none** | 0.56 | **0.66** | 0.51 |
| features + Jev, trained on SciFact | yes | **0.82** | 0.59 | 0.51 |

### Is a difference real? The paired bootstrap

Two routers are tested on the **same queries**, so compare them *pairwise*: resample the queries with
replacement 5,000 times, compute both AUCs on each resample, and look at the spread of the difference.
If the middle 95 % of differences excludes 0, the difference is unlikely to be luck.

| comparison | ΔAUC [95 % CI] | verdict |
|---|---|---|
| SciFact: Jev zero-shot − best trained router | +0.02 [−0.05, +0.10] | a tie (but Jev needed no training) |
| SciFact: features + Jev − features alone (SciFact-trained) | **+0.08 [+0.04, +0.12]** | **Jev adds real signal** |
| NFCorpus: Jev "needs reasoning" − best feature router | +0.07 [−0.01, +0.15] | borderline (p = 0.08) |
| ArguAna: Jev zero-shot − best trained router | **−0.12 [−0.16, −0.08]** | **Jev is worse** |

### What it means

1. **Jev sees what score features can't.** One plain-language question ("is a lower candidate better?")
   matches a trained model on SciFact with zero training, and adding Jev's answers improves the router
   significantly. That's the promise of the idea, and on SciFact it holds.
2. **On NFCorpus, Jev is the only thing that beats random** (borderline significant).
3. **On ArguAna, nothing works well.** It's the dataset where the cross-encoder matters most. Two likely reasons:
   - "helps" vs "doesn't" is a subtle split when 53 % of queries benefit;
   - ArguAna's documents are persuasive arguments, which Jev's docs list as a weak spot.

   There, the right policy is simply **always escalate**.
4. **The overhead matters.** ~131 ms on every query shifts Jev's curves right. At a 20 % escalation budget a Jev
   router averages ≈ 460 ms per query vs ≈ 330 ms for a feature router.
5. **Calibration has a catch.** Jev is calibrated for *the question it answers* ("is a lower candidate better?").
   That isn't the event we care about ("will the cross-encoder improve nDCG@10?"). Used directly as probabilities,
   its answers are off (ECE 0.10–0.35). A small trained model on top fixes it (≈ 0.04) within a dataset, but, like
   every trained router, breaks across datasets.
6. **The oracle is still far ahead.** The best routers keep only ~20–27 % of its gain. Predicting *in advance*
   whether a reranker will fix a ranking is genuinely hard, even when reading the text.

Cost of the whole Jev experiment: **2,838 queries × 2 calls = 5.6 M tokens ≈ $0.23**.

---

## 8. The whole project in one page

| stage | question | answer (BGE-M3, SciFact / NFCorpus / ArguAna) |
|---|---|---|
| 1 | What does exact MaxSim score? | 0.699 / 0.344 / 0.484 nDCG@10: the reference |
| 2 | How good is SMVE, and how does reranking compare? | Standalone SMVE is below a plain dense vector. A cross-encoder rerank is the most accurate (0.731 on SciFact) but ~200× slower than MaxSim reranking |
| 3 | SMVE vs alternatives, real cost, more data | SMVE beats MUVERA on storage and ties on latency. Little or negative headroom over dense. BM25 → MaxSim is the best cheap pipeline |
| 4 | Can we call the cross-encoder only when needed? | Yes in principle (oracle: better *and* 5–6× cheaper). In practice routers capture ~20–27 % of that. Jev helps on SciFact / NFCorpus, not ArguAna |

**Lessons that carry over to any retrieval project:**

1. Measure **headroom** before optimising an approximation.
2. Compare methods at **equal cost**, and measure cost with the right data structure.
3. A first stage should **complement** its reranker, not imitate it.
4. Before routing, check the **oracle**. It tells you whether routing can pay at all.
5. Fix your questions, thresholds and settings **before** looking at test results, and use paired tests on the same queries.

---

## 9. Glossary

| term | meaning |
|---|---|
| cross-encoder | a transformer that reads query and document together and outputs one relevance score |
| escalate | send a query to the expensive reranker |
| router | decides per query whether to escalate |
| oracle | a router that knows the true outcome: an upper bound, not achievable |
| gain | nDCG@10 after escalating minus before (can be negative) |
| gain kept | share of the oracle's improvement over random that a router achieves |
| AUC | chance a helped query is scored above a not-helped one (0.5 = random) |
| calibration / ECE | do predicted probabilities match observed frequencies? (ECE 0 = perfect) |
| cross-validation | train on some queries, test on the held-out rest, rotate |
| transfer | train on one dataset, test on another |
| Jev / System One | TypeSafe's fast decision model: typed questions in, calibrated probabilities out |
| Noul | a Jev yes/no question; returns P(yes) |
| state | the text or JSON Jev evaluates the questions against |
| zero-shot | used without any training on our labels |
| paired bootstrap | resample the same queries many times to test whether a difference is real |
| overhead | extra latency a router adds to every query |
