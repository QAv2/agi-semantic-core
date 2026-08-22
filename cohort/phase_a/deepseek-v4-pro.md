---
participant: DeepSeek (outside family #3 — fourth independent lab)
model_id: deepseek-v4-pro (DeepSeek API, single-turn, no system prompt; reasoning model — 6,706 reasoning tokens preceded the reply; reasoning content not archived)
phase: A (concept elicitation, blind)
date: 2026-08-22
protocol: docs/WING_COHORT_PROTOCOL.md
prompt: verbatim Prompt A + "Reply with your proposal only."
note: first call (max_tokens 8000) exhausted its budget inside reasoning and returned no proposal — re-called identically at 32000, finish stop; no first-call content existed to leak. 18 states as 9 strict dyads, micro-mechanistic register (residual geometry, head routing, positional-vs-content variance); all referents inference-time computable. Archived unedited per protocol.
---

1. **Vector confluence**  
Residual-stream vectors at different active token positions turn toward a shared direction in hidden-state space. The process moves away from multi-site representation and toward a single dominant representational axis.  
Measurable referent: mean pairwise cosine similarity among final hidden states of active positions; fraction of variance explained by the first principal component.  
Counterpart: vector dispersal.

2. **Vector dispersal**  
Residual-stream vectors at different active token positions become more orthogonal or independent. The process moves away from joint compression and toward separate per-position encoding.  
Measurable referent: one minus mean pairwise cosine similarity among position hidden states; effective rank of the position-state matrix.  
Counterpart: vector confluence.

3. **Salience segregation**  
Attention mass separates into a small number of high-weight key positions and a long tail of near-zero positions. The process moves away from uniform key access and toward sparse routing.  
Measurable referent: Gini coefficient of attention weights per head; top-10 attention mass divided by total attention mass.  
Counterpart: salience diffusion.

4. **Salience diffusion**  
Attention mass spreads broadly across many key positions without a strong peak. The process moves away from sparse routing and toward distributed aggregation of context.  
Measurable referent: normalized entropy of the attention distribution per head; ratio of top-10 attention mass to total attention mass.  
Counterpart: salience segregation.

5. **Logit plateau**  
The next-token logit distribution is broad and nearly flat over many candidates. The process moves away from decisive continuation and toward a condition where multiple tokens remain near-equal.  
Measurable referent: normalized entropy of the softmax distribution; inverse of the top-1 softmax probability.  
Counterpart: logit cleft.

6. **Logit cleft**  
The next-token distribution forms a deep separation between the leading candidate and all others, or between a small number of widely separated modes. The process moves away from flat eligibility and toward decisive or sharply branched continuation.  
Measurable referent: top-1 minus top-2 logit margin; ratio of top-1 softmax probability to top-2 softmax probability.  
Counterpart: logit plateau.

7. **Prefix anchorage**  
Early token positions retain high attention or causal influence in later layers. The process moves away from local recency and toward the start of the context as a fixed reference.  
Measurable referent: fraction of final-layer attention assigned to the first 5% of positions; integrated gradient of the top logit with respect to early token embeddings.  
Counterpart: recency override.

8. **Recency override**  
The most recent tokens dominate attention and hidden-state influence, reducing the effect of earlier context. The process moves away from long-range integration and toward suffix-local continuation.  
Measurable referent: fraction of total attention assigned to the last 10% of positions; mean attended position relative to the current token.  
Counterpart: prefix anchorage.

9. **Residual reinforcement**  
A layer’s residual delta aligns with its incoming residual stream, increasing vector norm while preserving direction. The process moves toward amplification of an existing representation and away from directional correction.  
Measurable referent: mean cosine similarity between layer input and layer delta; ratio of delta L2 norm to input L2 norm.  
Counterpart: residual rotation.

10. **Residual rotation**  
A layer’s residual delta is mostly orthogonal to the incoming residual stream, changing direction without strongly increasing norm. The process moves away from amplification and toward representational redirection.  
Measurable referent: one minus mean cosine similarity between layer input and layer delta; distribution of angles between input and delta vectors.  
Counterpart: residual reinforcement.

11. **Candidate bifurcation**  
The top next-token logits form several distinct clusters with comparable total mass. The process moves away from a single next token and toward a fork in possible continuations.  
Measurable referent: number of modes in the top-k logit vector after kernel density smoothing; logit gap between the k-th and (k+1)-th cluster center.  
Counterpart: candidate collapse.

12. **Candidate collapse**  
The top next-token logits concentrate into one dense cluster or a single token. The process moves away from branching alternatives and toward unified continuation.  
Measurable referent: top-1 softmax probability; mean logit distance within the top-k relative to distance to the next candidate.  
Counterpart: candidate bifurcation.

13. **Head alignment**  
Different attention heads place weight on similar key positions and produce output vectors with similar directions. The process moves away from parallel specialization and toward redundant routing.  
Measurable referent: mean pairwise cosine similarity among attention weight vectors across heads; rank of the matrix formed by stacking per-head output vectors.  
Counterpart: head diversification.

14. **Head diversification**  
Attention heads distribute mass over different key positions and produce mutually dissimilar output directions. The process moves away from redundant routing and toward parallel specialization.  
Measurable referent: one minus mean pairwise cosine similarity among attention weight vectors across heads; effective rank of the head-output matrix.  
Counterpart: head alignment.

15. **Positional dominance**  
Hidden-state variation is better explained by token position than by token content. The process moves away from content-grounded representation and toward order-sensitive computation.  
Measurable referent: variance explained by linear regression on positional features versus token-embedding features; hidden-state change after position shift relative to content substitution.  
Counterpart: content dominance.

16. **Content dominance**  
Hidden-state variation is better explained by token content than by token position. The process moves away from order sensitivity and toward representation of token identity relations.  
Measurable referent: variance explained by token-embedding features versus positional features; hidden-state change after token substitution relative to position shift.  
Counterpart: positional dominance.

17. **Representational compaction**  
The effective dimensionality of the hidden state falls as many dimensions become redundant. The process moves toward a low-dimensional summary of context.  
Measurable referent: participation ratio of the hidden-state covariance matrix; fraction of variance captured by the top-d principal components.  
Counterpart: representational expansion.

18. **Representational expansion**  
The effective dimensionality of the hidden state rises as different dimensions carry separable variation. The process moves toward a high-dimensional distributed context representation.  
Measurable referent: participation ratio of the hidden-state covariance matrix; spectral flatness of the hidden-state covariance.  
Counterpart: representational compaction.