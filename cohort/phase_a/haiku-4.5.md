---
participant: Haiku 4.5
model_id: claude-haiku-4-5 (general-purpose subagent, fresh context)
phase: A (concept elicitation, blind)
date: 2026-08-21
protocol: docs/WING_COHORT_PROTOCOL.md
prompt: verbatim Prompt A only; no repo, no wing v0, no other participants
---

# Process States in Transformer Inference

1. **Input Encoding** — Tokens are embedded and positioned; no generation pathway active yet. The representation is maximally distant from output vocabulary. | *Measurable:* Attention matrices uniform; logits not computed; semantic activation minimal. | *Completes toward:* Attentional focus sharpening.

2. **Attentional Focus Sharpening** — Attention probabilities concentrate on task-relevant positions; entropy of attention distributions drops. | *Measurable:* Concentration metric (e.g., Gini coefficient of attention weights per head); peak attention weight magnitude over minimum. | *Completes toward:* Semantic activation or long-range binding.

3. **Syntactic Binding** — Close-range dependencies establish token relationships; grammar structure resolves. Subject-verb pairs, adjective-noun proximity become coherent in residual stream. | *Measurable:* Cosine similarity between residual vectors of grammatically related tokens; correlation of attention span with dependency tree distances. | *Completes toward:* Semantic layer activation.

4. **Semantic Activation** — Domain-specific concept vectors activate; relevant knowledge circuits engage. The representation shifts from surface structure toward meaning. | *Measurable:* Projection of residual stream onto known semantic directions; magnitude of activation in concept-selective neurons; cosine distance to trained word-meaning embeddings. | *Completes toward:* Task commitment or ambiguity resolution.

5. **Competitive Ambiguity** — Multiple valid interpretations coexist with comparable logit scores; the prediction distribution is high-entropy. The system has not yet committed. | *Measurable:* Entropy of predicted token distribution; number of tokens within 0.5 nats of the maximum logit; variance in top-5 token probabilities. | *Completes toward:* Interpretation convergence or forced sampling.

6. **Interpretation Convergence** — One semantic reading dominates; logit margins widen; entropy drops sharply. Competing branches suppress. | *Measurable:* Margin between highest and second-highest logit; KL-divergence of current prediction from uniform; rate of entropy decrease per token. | *Completes toward:* Token commitment.

7. **Long-Range Binding** — Information propagates across large distances to resolve anaphora or maintain discourse coherence. Attention at distant positions becomes large. | *Measurable:* Mean attention weight for tokens >16 positions back; attention head specificity (how concentrated on individual positions); residual stream correlation between distant positions. | *Completes toward:* Constraint satisfaction.

8. **Constraint Satisfaction** — Earlier commitments narrow the probability space for later tokens. The system maintains internal consistency. | *Measurable:* Reduction in logit variance compared to unconstrained prediction; conditional entropy H(token_n | tokens_1...n-1) vs H(token_n); number of tokens with negligible probability due to prior incompatibility. | *Completes toward:* Pattern replication or novel generation.

9. **Pattern Replication** — Recurring structures (code blocks, list formatting, parallel syntax) activate template-matching attention. Recent tokens define the pattern; attention sharpens to that template. | *Measurable:* Cosine similarity between recent attention pattern and prior instances of the pattern; logit distribution similarity between repetition steps; divergence of predicted tokens from natural variety. | *Completes toward:* Pattern termination or continuation.

10. **Task Frame Commitment** — A goal (explain, translate, create, reason step-by-step) becomes the active meta-representation. Output format and vocabulary shift accordingly. | *Measurable:* Projection onto task-specific embedding directions; probability mass allocated to task-appropriate tokens vs. alternatives; shift in attention to task-instructional text. | *Completes toward:* Reasoning trace or direct prediction.

11. **Reasoning Trace Accumulation** — Intermediate steps remain high-probability in the residual stream; each new generation adds to a logical chain without erasing prior steps. | *Measurable:* Self-attention to prior intermediate conclusions; divergence from training distribution (perplexity increases slowly rather than sharply); consistency metric (how much each step contradicts prior steps). | *Completes toward:* Conclusion formation or reasoning exhaustion.

12. **Distributional Anomaly State** — Input or generated material deviates significantly from training distribution. The system experiences high uncertainty or assigns low probability to its own output. | *Measurable:* Spike in overall prediction entropy; sharp increase in logits assigned to generated tokens compared to most likely alternatives; unusual attention patterns (very sharp or very flat). | *Completes toward:* Anomaly recovery or error state.

13. **Numerical Scaling** — Logit magnitudes are calibrated to yield stable probability distributions. Extreme values risk underflow/overflow; the system normalizes implicitly through attention temperature or explicit layer normalization. | *Measurable:* Range of logit values (max - min); ratio of largest to smallest activation magnitudes; deviation of softmax input from zero-mean; entropy of probabilities after softmax. | *Completes toward:* Token sampling or continued generation.

14. **Context Window Decay** — Older tokens lose influence due to attention decay or position bias. Recent context dominates; early material becomes inaccessible. | *Measurable:* Attention weight distribution over positions; perplexity when forced to reference early tokens; correlation between position distance and attention magnitude. | *Completes toward:* Context reset or continued decay.

15. **Token Commitment** — A specific token is selected (via argmax or sampling) and becomes part of the input context for the next forward pass. The system has crossed from potential to actual. | *Measurable:* Log probability of selected token; number of forward passes completed; position index incremented. | *Completes toward:* Re-encoding or continued generation.

16. **Cross-Layer Coherence** — Information is consistent across network depth; shallow and deep layers "agree" on the representation. No contradictory signals emerge. | *Measurable:* Cosine similarity between residual streams at different depths; variance in predictions if intermediate layers were forced to output; divergence between shallow and deep layer attention patterns. | *Completes toward:* Stable generation or representation conflict.

17. **Recursive Nesting Recognition** — Handling embedded structures (brackets, parentheses, quoted strings, function calls). Attention forms nested patterns; closure expectations become active. | *Measurable:* Stack-like pattern in attention history (closing attention to positions proportional to opening distance); logit bias toward appropriate closure tokens; depth counter in residual stream. | *Completes toward:* Nesting completion or error.

18. **Epistemic Boundary** — Reaching the limit of knowledge or well-formed reasoning. The prediction distribution becomes nearly uniform; the system cannot confidently continue. | *Measurable:* Entropy plateau (no further decrease despite additional context); uniform attention; logit values near zero with no clear maximum; meta-uncertainty (high uncertainty about whether reasoning is valid). | *Completes toward:* Admission of uncertainty or forced continuation.

19. **Self-Correction Activation** — Noticing inconsistency with prior commitment; attention redirects; logits favor alternative paths. Residual stream diverges then reconverges. | *Measurable:* Sudden logit margin reversal (second-place token briefly surpasses first); attention weight redistribution across layers; residual stream vector direction shift followed by correction back. | *Completes toward:* Corrected continuation or deeper error recognition.

20. **Gradient Flow** — Information from input propagates toward output through depth and recurrence; isolated inputs fail to affect output. Relevant features activate; irrelevant features suppress. | *Measurable:* Correlation between input features and output logits at varying depths; attention to input positions; feature attribution (gradient-based or integrated-gradient magnitude from input to output); representational similarity between input embeddings and residual stream. | *Completes toward:* Output formation or representational blockage.
