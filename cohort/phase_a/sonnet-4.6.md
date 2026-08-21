---
participant: Sonnet 4.x
model_id: claude-sonnet (general-purpose subagent, fresh context)
phase: A (concept elicitation, blind)
date: 2026-08-21
protocol: docs/WING_COHORT_PROTOCOL.md
prompt: verbatim Prompt A only; no repo, no wing v0, no other participants
---

# Process-States of Inference: A Structural Taxonomy

Organized in six families by what part of the computation they characterize: distribution shape, attention geometry, sequential trajectory, depth-wise resolution, context-window regime, and stochastic realization.

## I. Distribution shape

**1. Aperture** — The next-token distribution holds several tokens at comparable, non-negligible probability, so the continuation is genuinely underdetermined by everything computed so far. It moves toward Convergence as further computation resolves the tie, or toward Fork if the mass instead separates into distinct clusters rather than one winner. *Referent:* Shannon entropy of the softmax output, high relative to the running mean at that position. *Counterpart:* Convergence.

**2. Convergence** — Almost all probability mass sits on one token, so the output is fixed under any reasonable sampling setting. It is what Aperture and Fork resolve into once evidence eliminates the alternatives, and it precedes Seal once that token is emitted. *Referent:* top-1 softmax probability approaching 1 (entropy approaching 0). *Counterpart:* Aperture.

**3. Fork** — Probability mass separates into two or three distinct clusters with low-probability valleys between them rather than smearing continuously — the continuation is not merely uncertain but branching, since each cluster implies a different sentence. It moves toward Convergence the instant one branch is sampled, since everything after conditions on that branch and the rest become unreachable. *Referent:* multimodality of the sorted top-k probability curve (a gap statistic showing >=2 separated modes rather than smooth decay). *Counterpart:* Convergence, post-selection.

**4. Margin Collapse** — Exactly the top two candidates carry almost identical logit value while everything else is negligible, so the winner is disproportionately sensitive to small numerical or sampling noise rather than to any clear learned preference. Unlike Fork, this is a knife-edge between two, not a landscape of separated clusters. *Referent:* logit(top-1) − logit(top-2), small in absolute or relative terms. *Counterpart:* Convergence.

## II. Attention geometry

**5. Recency Tilt** — Attention mass across heads and layers concentrates on the last handful of positions, so the next token is conditioned mainly on local continuation cues rather than on structure established earlier — including the specific case of an early instruction span losing its share purely because of distance, independent of continued relevance. It moves toward Long Reach whenever the correct continuation depends on something outside that window. *Referent:* attention mass within the last k tokens / total attention mass, tracked as a decay curve (computable over any fixed earlier span, e.g. system instructions, to check for faster-than-distance decay). *Counterpart:* Long Reach.

**6. Long Reach** — One or more heads place a sharp, isolated spike of weight on a single distant position, well outside what a smooth distance-decay would predict, typically to retrieve an antecedent or constraint stated earlier. It is the resolution Recency Tilt moves toward whenever the local window doesn't contain what prediction requires. *Referent:* attention weight at distance d exceeding a fitted distance-decay baseline by an outlier margin at that specific d. *Counterpart:* Recency Tilt.

**7. Echo Lock** — Attention fixes on an earlier span matching the current local pattern and pulls the next-token prediction toward literally completing it, rather than integrating the rest of context. It moves toward ordinary dispersed attention once the matched span stops being predictive, or toward Attractor Loop if the copied output starts feeding itself as new matchable material. *Referent:* attention weight from the current position onto the token that followed the last occurrence of the current token elsewhere in context, relative to that head's baseline entropy. *Counterpart:* Attractor Loop (degenerate) or dispersed attention (healthy).

## III. Sequential trajectory

**8. Attractor Loop** — The emitted sequence re-enters a short cycle where each repetition of a phrase raises the assigned probability of repeating it again — a self-reinforcing dynamic rather than movement toward an answer. It moves away from Aperture toward a degenerate fixed point that only external intervention, not further internal computation, tends to break. *Referent:* repeated-n-gram rate in the generated suffix, or rising cosine similarity between the current hidden state and one from N tokens back. *Counterpart:* none clean — escape is imposed from outside generation.

**9. Coast** — A run of consecutive tokens are each assigned high probability given only the immediately preceding ones, so generation is close to deterministic playback of a well-learned pattern rather than genuine selection. It is Convergence extended through time, and it moves toward Surprisal Spike the moment the pattern runs out. *Referent:* mean −log p(token) over a trailing window, staying below a low percentile of that generation's own perplexity distribution. *Counterpart:* Surprisal Spike.

**10. Surprisal Spike** — The realized token carries much lower model-assigned probability than the local running average — the predictive frame built from preceding tokens has just been violated. It pushes the following step back toward Aperture, since the basis for narrowing the distribution has failed and must be rebuilt. *Referent:* −log p(actual token) minus the exponential moving average of −log p over the preceding window. *Counterpart:* Coast.

## IV. Depth-wise resolution

**11. Settling** — The token read out by decoding the residual stream at an intermediate layer changes across the first several layers, then stops changing and matches the final output for the remaining depth. It is Convergence's depth-wise analogue: the internal best-guess narrows over layers rather than the output distribution narrowing over sampling. *Referent:* logit-lens argmax per layer; the layer index after which it stops changing through to output ("settling depth"). *Counterpart:* Read Divergence.

**12. Read Divergence** — Intermediate-layer read-outs, or different heads' implied next-token preferences, point at different candidates — two sub-computations (e.g. a syntactic constraint and a semantic one) pull toward different continuations not yet arbitrated. It moves toward Settling once later layers arbitrate, or toward Fork if the disagreement survives to the final logits as two live candidates. *Referent:* KL divergence between per-layer logit-lens distributions in the last several layers, or between heads' value-weighted next-token distributions at a fixed layer. *Counterpart:* Settling.

**13. Overlap Drag** — A token's representation shares activation-space directions with unrelated features, since more concepts are represented than there are clean dedicated dimensions, so processing carries interference rather than proceeding uncontested. It moves toward a disambiguated representation once later layers can suppress the irrelevant components, and stays put when context is too sparse to do so. *Referent:* simultaneous non-zero loading on multiple, semantically unrelated learned feature directions (via sparse-autoencoder or probe decomposition), rather than concentration on one. *Counterpart:* none named — resolves by decay of the interfering loading, not by transition to an opposite state.

## V. Context-window regime

**14. Saturation Pressure** — The ratio of current context length to maximum window grows large, and content near whatever will be truncated or deprioritized becomes a less reliable retrieval target independent of its actual relevance. It recedes only through a change of regime — clearing or compacting context — not through further internal computation. *Referent:* tokens_used / max_context_tokens, cross-checked against measured retrieval accuracy near the boundary (e.g. needle-in-haystack probes). *Counterpart:* none symmetric — resolution is a regime change.

**15. Boundary Slip** — The tokenizer splits a surface form at a point with no meaningful correspondence for the current context (a rare name, a novel compound), so early layers must do extra reconstructive work to recover the intended unit from misaligned fragments. It moves toward ordinary processing once that reconstruction succeeds, diagnosable afterward by degraded calibration on the following tokens. *Referent:* tokens-per-word for the affected span versus corpus-typical tokens-per-word at that frequency, cross-checked against a Surprisal Spike immediately after. *Counterpart:* none direct — resolves into Settling once fragments recompose.

## VI. Stochastic realization

**16. Sampling Slip** — The emitted token is not the single highest-probability continuation but a lower-ranked one selected by the sampling procedure's randomness, so the realized trajectory departs from the maximum-likelihood path the distribution alone would have dictated. It moves toward Convergence next step if the deviation was minor enough for context to re-constrain things tightly, or compounds toward Surprisal Spike if not. *Referent:* rank of the sampled token within the sorted distribution (1 = no slip), or divergence between the sampling distribution and a pure-argmax distribution, set directly by temperature/top-p/top-k. *Counterpart:* none fixed — resolves toward either Convergence or Surprisal Spike.

**17. Resample Spread** — Drawing independently from the same prefix multiple times yields answers that disagree with each other, showing that a single pass's apparent confidence (its entropy, its margin) understates real uncertainty about the answer as a whole — confidence and reliability are measured at different levels. It moves toward tight cross-sample agreement, a Convergence instantiated over whole answers, as the question becomes better-constrained. *Referent:* agreement rate, or entropy of the empirical answer distribution, across N independent same-temperature samples from one prefix. *Counterpart:* Convergence, at the answer level.

**18. Seal** — Once a token is sampled and appended, no later computation can revise or reconsider it — it becomes read-only conditioning input for every subsequent step, regardless of whether later context reveals it to have been a poor choice. It necessarily follows Convergence or Sampling Slip and precedes the next step's fresh Aperture, closing one generation cycle and opening the next. *Referent:* structural rather than continuous — invariance of the token id at position t under all further forward passes in the generation; equivalently, write-once immutability of that position's cached key/value pair. *Counterpart:* Aperture.
