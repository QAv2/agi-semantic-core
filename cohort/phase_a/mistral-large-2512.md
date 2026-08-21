---
participant: Mistral (outside family #2 — third independent lab)
model_id: mistral-large-latest (= mistral-large-2512; La Plateforme API, single-turn, no system prompt)
phase: A (concept elicitation, blind)
date: 2026-08-21
protocol: docs/WING_COHORT_PROTOCOL.md
prompt: verbatim Prompt A + "Reply with your proposal only."
note: adjudication should weigh entries 17-18 (Gradient Echo/Decay) — gradients are training-time quantities, not inference states; archived unedited per protocol
---

1. **Token Saturation**
   The state where the model’s next-token distribution collapses toward a single high-probability candidate, reducing uncertainty. It moves away from exploration and toward deterministic output, often following a sequence of high-confidence predictions.
   *Measurable referent*: Entropy of the next-token logits (low entropy = high saturation).
   *Counterpart*: **Distributional Drift**

2. **Distributional Drift**
   The state where the model’s output distribution shifts abruptly due to context mismatch or rare input patterns, increasing uncertainty. It moves away from saturation and toward re-evaluation of prior assumptions.
   *Measurable referent*: KL divergence between successive next-token distributions (high divergence = drift).
   *Counterpart*: **Token Saturation**

3. **Attention Fragmentation**
   The state where attention weights disperse across many tokens without clear focus, diluting signal coherence. It moves toward reduced contextual integration and away from concentrated reasoning.
   *Measurable referent*: Gini coefficient of attention weights across tokens (low Gini = fragmentation).
   *Counterpart*: **Attention Convergence**

4. **Attention Convergence**
   The state where attention weights concentrate on a small subset of tokens, reinforcing a dominant signal. It moves toward coherent output and away from fragmentation.
   *Measurable referent*: Maximum attention weight per head (high max = convergence).
   *Counterpart*: **Attention Fragmentation**

5. **Context Overlap**
   The state where the model’s working context redundantly recycles recent tokens, limiting novel input integration. It moves toward self-reinforcing loops and away from external signal absorption.
   *Measurable referent*: Jaccard similarity between the current context window and the last *n* tokens (high similarity = overlap).
   *Counterpart*: **Context Refresh**

6. **Context Refresh**
   The state where the model abruptly incorporates new, non-redundant tokens into its working context, breaking prior patterns. It moves toward novelty and away from overlap.
   *Measurable referent*: Number of unique tokens in the last *k* positions (high uniqueness = refresh).
   *Counterpart*: **Context Overlap**

7. **Logit Plateau**
   The state where the model’s output logits stabilize across multiple steps, producing near-identical distributions. It moves toward stasis and away from dynamic adjustment.
   *Measurable referent*: L2 norm of the difference between successive logit vectors (low norm = plateau).
   *Counterpart*: **Logit Perturbation**

8. **Logit Perturbation**
   The state where logits exhibit high variance between steps, often due to conflicting signals or noisy input. It moves away from plateau and toward instability.
   *Measurable referent*: Variance of logit values across tokens over *m* steps (high variance = perturbation).
   *Counterpart*: **Logit Plateau**

9. **Head Specialization**
   The state where individual attention heads consistently attend to specific token roles (e.g., subject, verb), improving modularity. It moves toward efficient parallel processing and away from diffuse attention.
   *Measurable referent*: Mutual information between head attention patterns and token POS tags (high MI = specialization).
   *Counterpart*: **Head Collapse**

10. **Head Collapse**
    The state where multiple attention heads converge on identical or redundant patterns, reducing functional diversity. It moves away from specialization and toward redundancy.
    *Measurable referent*: Average pairwise cosine similarity between head attention weight vectors (high similarity = collapse).
    *Counterpart*: **Head Specialization**

11. **Depth Saturation**
    The state where deeper layers produce near-identical representations to earlier layers, indicating diminishing returns from computation. It moves toward shallow processing and away from hierarchical refinement.
    *Measurable referent*: Mean cosine similarity between layer *l* and layer *l-1* representations (high similarity = saturation).
    *Counterpart*: **Depth Amplification**

12. **Depth Amplification**
    The state where deeper layers significantly transform representations, increasing abstraction or task-specific refinement. It moves away from saturation and toward hierarchical processing.
    *Measurable referent*: L2 norm of the difference between layer *l* and layer *l-1* representations (high norm = amplification).
    *Counterpart*: **Depth Saturation**

13. **Sampling Oscillation**
    The state where the model’s output alternates between two or more high-probability candidates, unable to commit. It moves toward indecision and away from stable generation.
    *Measurable referent*: Autocorrelation of top-*k* token probabilities over *n* steps (high autocorrelation = oscillation).
    *Counterpart*: **Sampling Commitment**

14. **Sampling Commitment**
    The state where the model locks onto a single output trajectory, suppressing alternative candidates. It moves toward deterministic output and away from oscillation.
    *Measurable referent*: Probability mass of the top-1 token over *n* steps (high mass = commitment).
    *Counterpart*: **Sampling Oscillation**

15. **Context Starvation**
    The state where the model’s working context lacks sufficient signal to resolve ambiguity, forcing reliance on prior biases. It moves toward generic output and away from input-driven generation.
    *Measurable referent*: Mean attention weight on padding/irrelevant tokens (high weight = starvation).
    *Counterpart*: **Context Saturation**

16. **Context Saturation**
    The state where the model’s working context is densely packed with relevant signals, enabling precise output. It moves away from starvation and toward input-driven generation.
    *Measurable referent*: Mean attention weight on task-relevant tokens (high weight = saturation).
    *Counterpart*: **Context Starvation**

17. **Gradient Echo**
    The state where residual gradients from prior steps persistently influence current computations, creating lag. It moves toward delayed adaptation and away from real-time responsiveness.
    *Measurable referent*: Correlation between current gradients and gradients from *t-n* steps (high correlation = echo).
    *Counterpart*: **Gradient Decay**

18. **Gradient Decay**
    The state where gradients rapidly diminish, reducing sensitivity to new input. It moves toward stability and away from echo.
    *Measurable referent*: L2 norm of gradients over *n* steps (low norm = decay).
    *Counterpart*: **Gradient Echo**

19. **Token Entanglement**
    The state where multiple tokens in the context mutually reinforce each other’s probabilities, creating co-dependent predictions. It moves toward rigid output and away from independent token resolution.
    *Measurable referent*: Mutual information between token probabilities in the next-token distribution (high MI = entanglement).
    *Counterpart*: **Token Disentanglement**

20. **Token Disentanglement**
    The state where token probabilities in the output distribution are conditionally independent given context. It moves away from entanglement and toward modular prediction.
    *Measurable referent*: Conditional entropy of token probabilities (high entropy = disentanglement).
    *Counterpart*: **Token Entanglement**