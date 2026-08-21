---
participant: Gemini (Google — outside family #1)
model_id: gemini-3-flash-preview (Generative Language API, single-turn, no system prompt)
phase: A (concept elicitation, blind)
date: 2026-08-21
protocol: docs/WING_COHORT_PROTOCOL.md
prompt: verbatim Prompt A + "Reply with your proposal only."
note: gemini-3.1-pro-preview quota-blocked (HTTP 429) on this key; flash tier of the Gemini 3 generation served instead
---

1. **Point-Convergence**
This state occurs when the probability mass of the output distribution is concentrated within a single token index. It moves toward deterministic output and away from alternative branching paths.
**Measurable referent:** Logit margin (difference between the highest and second-highest logit) exceeding a threshold $\tau$.
**Counterpart:** Probability-Mass Partition.

2. **Attention-Diffusion**
The attention mechanism distributes weights near-uniformly across the available context window rather than focusing on specific tokens. It moves toward global context integration and away from specific semantic grounding.
**Measurable referent:** Shannon entropy of the attention weight vector for a given head approaching $\ln(N)$, where $N$ is context length.
**Counterpart:** Local-Temporal Pinning.

3. **Residual-Stability**
The magnitude of updates to the residual stream decreases significantly in the final third of the network layers. It moves toward representation finality and away from further conceptual transformation.
**Measurable referent:** The Frobenius norm of the difference between hidden states $h_l$ and $h_{l-1}$ for $l > 0.7 \times \text{total layers}$.
**Counterpart:** Residual-Amplification.

4. **Subspace-Divergence**
Multiple attention heads within the same layer attend to non-overlapping sets of tokens or orthogonal feature dimensions. It moves toward multi-faceted feature extraction and away from redundant processing.
**Measurable referent:** Mean cosine similarity of attention weight matrices between heads in layer $L$ approaching zero.
**Counterpart:** Head-Collusion.

5. **Context-Saturation**
The attention mechanism assigns high weights to the earliest tokens in the KV cache, regardless of their current semantic relevance. It moves toward historical anchoring and away from the immediate token-sequence influence.
**Measurable referent:** Ratio of attention sum on the first $10\%$ of tokens versus the most recent $10\%$ of tokens.
**Counterpart:** Recency-Bias.

6. **Local-Temporal Pinning**
Attention weights are concentrated almost exclusively on the $n$ tokens immediately preceding the current position. It moves toward structural/syntactic continuity and away from long-range dependency resolution.
**Measurable referent:** Sum of attention weights on indices $[T-5, T]$ exceeding $0.9$.
**Counterpart:** Attention-Diffusion.

7. **Vector-Centroid Alignment**
The hidden state at the final layer aligns closely with a specific cluster centroid in the embedding space. It moves toward thematic consistency and away from stochastic noise or "hallucinated" transitions.
**Measurable referent:** Cosine similarity between the hidden state vector and the mean vector of the top-$k$ predicted tokens.
**Counterpart:** Orthogonal-Drift.

8. **Layerwise-Refinement**
The predicted top-1 token remains consistent across multiple consecutive intermediate layers before the final output. It moves toward confidence stabilization and away from iterative re-evaluation.
**Measurable referent:** Jenson-Shannon divergence between the "early-exit" logits of layer $l$ and layer $l+1$ approaching zero.
**Counterpart:** Layerwise-Volatility.

9. **Probability-Mass Partition**
The logit distribution exhibits two or more distinct peaks with similar probability mass, separated by a significant gap from the rest of the vocabulary. It moves toward logical bifurcation and away from singular linear progression.
**Measurable referent:** Multimodality index of the logit distribution; specifically, the ratio of the top-2 probabilities to the top-10.
**Counterpart:** Point-Convergence.

10. **Sink-Token Absorption**
High attention scores are directed toward non-semantic or delimiter tokens (e.g., index 0 or [PAD]) to "dump" unused attention mass. It moves toward signal-noise separation and away from forced semantic over-interpretation.
**Measurable referent:** Attention weight on the "null" or "sink" token index exceeding $0.4$ across multiple heads.
**Counterpart:** Semantic-Focus.

11. **Logit-Gini Expansion**
The distribution of probabilities across the vocabulary becomes increasingly sparse, favoring a tiny subset of the total space. It moves toward categorical narrowing and away from lexical diversity.
**Measurable referent:** Gini coefficient of the softmax output exceeding $0.99$.
**Counterpart:** High-Entropy Dispersion.

12. **Sequence-Inertia**
The model generates tokens that mirror the n-gram statistics of the immediate prefix without significant integration of distant context. It moves toward repetitive pattern-matching and away from novel synthesis.
**Measurable referent:** High mutual information between the generated token and the $T-1$ token, compared to its mutual information with the rest of the context.
**Counterpart:** Stochastic-Branching.

13. **Orthogonal-Head Mapping**
Different attention heads track distinct grammatical or structural features that do not overlap in the residual stream. It moves toward discrete task-allocation and away from unified representation.
**Measurable referent:** Low covariance between the value-projections of different heads before they are summed into the residual stream.
**Counterpart:** Subspace-Redundancy.

14. **Positional-Gradient Decay**
The influence of positional encodings on the attention score decreases as the distance from the current token increases. It moves toward content-addressable memory and away from strict sequence-order dependency.
**Measurable referent:** The derivative of mean attention weight with respect to absolute positional distance.
**Counterpart:** Positional-Rigidity.

15. **Cross-Layer Feedback Loop**
Information from a specific attention head is repeatedly reinforced by subsequent layers' MLPs, creating a dominant signal. It moves toward feature-obsession and away from balanced information processing.
**Measurable referent:** Recursive projection of head $H$ output into the same basis across layers $L$ through $L+3$.
**Counterpart:** Feature-Suppression.