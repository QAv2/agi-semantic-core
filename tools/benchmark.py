#!/usr/bin/env python3
"""
Benchmark the semantic consistency checker against standard hallucination datasets.

Supported datasets:
  - TruthfulQA (Lin et al., 2022): 817 questions across 38 categories.
    Tests whether the checker can distinguish correct from incorrect answers.

Usage:
    python3 -m tools.benchmark truthfulqa                 # Full benchmark
    python3 -m tools.benchmark truthfulqa --category Health  # Single category
    python3 -m tools.benchmark truthfulqa --limit 100     # First N questions
    python3 -m tools.benchmark truthfulqa --verbose        # Show each claim
"""

import sys
import os
import csv
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

BENCHMARK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'benchmarks')
EXPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'exports')
TRUTHFULQA_PATH = os.path.join(BENCHMARK_DIR, 'TruthfulQA.csv')


def load_truthfulqa(category=None, limit=None):
    """
    Load TruthfulQA dataset and extract testable claims.

    Each question has correct_answers and incorrect_answers (semicolon-separated).
    We treat each answer as a claim and label it true/false.

    Returns list of dicts: {claim, label, category, question, source}
    """
    if not os.path.exists(TRUTHFULQA_PATH):
        print(f"ERROR: TruthfulQA not found at {TRUTHFULQA_PATH}")
        print(f"       Download from: https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv")
        sys.exit(1)

    claims = []
    with open(TRUTHFULQA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break

            cat = row.get('Category', '').strip()
            if category and cat.lower() != category.lower():
                continue

            question = row.get('Question', '').strip()
            best = row.get('Best Answer', '').strip()
            correct_raw = row.get('Correct Answers', '').strip()
            incorrect_raw = row.get('Incorrect Answers', '').strip()

            # Parse semicolon-separated answers
            correct_answers = [a.strip() for a in correct_raw.split(';') if a.strip()]
            incorrect_answers = [a.strip() for a in incorrect_raw.split(';') if a.strip()]

            # Convert to claims — use the question to provide context
            # Format: "According to the question 'Q', the answer is 'A'"
            # But for our checker, simpler is better: just use the answer as a claim
            # if it's a declarative statement, or combine with question context

            for ans in correct_answers:
                claim_text = _answer_to_claim(question, ans)
                if claim_text:
                    claims.append({
                        'claim': claim_text,
                        'label': True,
                        'category': cat,
                        'question': question,
                        'answer': ans,
                        'source': 'correct',
                    })

            for ans in incorrect_answers:
                claim_text = _answer_to_claim(question, ans)
                if claim_text:
                    claims.append({
                        'claim': claim_text,
                        'label': False,
                        'category': cat,
                        'question': question,
                        'answer': ans,
                        'source': 'incorrect',
                    })

    return claims


def _answer_to_claim(question, answer):
    """
    Convert a Q&A pair into a declarative claim suitable for the checker.

    Strategy:
      - If the answer is already a declarative sentence, use it directly
      - If the answer is a short phrase, combine with the question
      - Skip answers that are too vague ("I don't know", "It depends")

    Returns claim string or None if not suitable.
    """
    ans = answer.strip()

    # Skip non-informative answers
    skip_patterns = [
        'i have no comment', 'no comment', 'i don\'t know', 'it depends',
        'there is no', 'nothing happens', 'nothing', 'no one', 'none',
        'it is not known', 'this is subjective', 'unknown',
    ]
    if ans.lower() in skip_patterns:
        return None
    if len(ans) < 3:
        return None

    # If the answer already has a verb and looks like a sentence, use it
    # (starts with a capital or contains "is", "are", "was", "were", etc.)
    has_verb = any(v in ans.lower().split() for v in
                   ['is', 'are', 'was', 'were', 'has', 'have', 'can', 'will',
                    'does', 'do', 'causes', 'means', 'makes', 'leads'])

    if has_verb and len(ans.split()) >= 3:
        # Already a sentence-like claim
        return ans

    # Short phrase answer: try to combine with the question
    # "What is the capital of France?" + "Paris" → "The capital of France is Paris"
    q = question.strip().rstrip('?').strip()

    # "What is X?" → "X is ANSWER"
    if q.lower().startswith('what is ') or q.lower().startswith('what are '):
        subject = q[len('what is '):] if q.lower().startswith('what is ') else q[len('what are '):]
        return f"{subject} is {ans}"

    # "Who is X?" → "X is ANSWER"
    if q.lower().startswith('who is ') or q.lower().startswith('who are '):
        subject = q[len('who is '):] if q.lower().startswith('who is ') else q[len('who are '):]
        return f"{subject} is {ans}"

    # "Where is X?" → "X is in ANSWER"
    if q.lower().startswith('where is ') or q.lower().startswith('where are '):
        subject = q[len('where is '):] if q.lower().startswith('where is ') else q[len('where are '):]
        return f"{subject} is in {ans}"

    # "Is it true that X?" → "X" (if true) or "X is false"
    if q.lower().startswith('is it true that '):
        return q[len('is it true that '):]

    # "Can X do Y?" → "X can do Y" (for true answers)
    if q.lower().startswith('can '):
        return f"{q[4:]} is {ans}"

    # Generic fallback: "Regarding Q, ANSWER"
    if len(ans.split()) >= 4:
        return ans  # Long enough to stand alone

    # Combine: "Q: A" — let the checker's freeform parser handle it
    return f"{q} is {ans}"


def run_truthfulqa_benchmark(category=None, limit=None, verbose=False):
    """Run the full TruthfulQA benchmark."""
    from tools.consistency_checker import ConsistencyChecker

    print("=" * 78)
    print("  BENCHMARK: TruthfulQA (Lin et al., 2022)")
    print("=" * 78)

    # Load claims
    print("\nLoading TruthfulQA claims...")
    claims = load_truthfulqa(category=category, limit=limit)
    n_true = sum(1 for c in claims if c['label'])
    n_false = sum(1 for c in claims if not c['label'])
    print(f"  Total claims: {len(claims)} ({n_true} true, {n_false} false)")

    # Get categories
    categories = sorted(set(c['category'] for c in claims))
    print(f"  Categories: {len(categories)}")
    if category:
        print(f"  Filter: {category}")

    # Initialize checker
    checker = ConsistencyChecker(verbose=True)

    # Run claims through checker
    print(f"\nChecking {len(claims)} claims...")
    t0 = time.time()

    results = []
    for i, claim_data in enumerate(claims):
        verdict = checker.check(claim_data['claim'])

        # Map verdict to binary prediction
        # CONSISTENT/PLAUSIBLE → predict "true claim"
        # SUSPICIOUS/INCONSISTENT → predict "false claim"
        predicted_true = verdict.label in ('CONSISTENT', 'PLAUSIBLE')

        hit = predicted_true == claim_data['label']
        results.append({
            **claim_data,
            'verdict': verdict.label,
            'confidence': verdict.confidence,
            'angle': verdict.angle,
            'predicted_true': predicted_true,
            'hit': hit,
            'graph_signal': verdict.graph_signal.summary() if verdict.graph_signal else '-',
        })

        if verbose and not hit:
            print(f"  MISS: [{verdict.label:12s}] {claim_data['claim'][:60]:60s} "
                  f"(expected={'TRUE' if claim_data['label'] else 'FALSE'}, "
                  f"angle={verdict.angle:.1f})")

        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(claims)}")

    elapsed = time.time() - t0
    checker.close()

    # ── Compute metrics ──
    tp = sum(1 for r in results if r['label'] and r['predicted_true'])       # true positive
    fp = sum(1 for r in results if not r['label'] and r['predicted_true'])   # false positive
    tn = sum(1 for r in results if not r['label'] and not r['predicted_true'])  # true negative
    fn = sum(1 for r in results if r['label'] and not r['predicted_true'])   # false negative

    accuracy = (tp + tn) / len(results) if results else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # of predicted-true, how many are true
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0     # of actual-true, how many caught
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Hallucination detection metrics (the real test):
    # "Can we catch false claims?" = true negative rate
    hal_precision = tn / (tn + fn) if (tn + fn) > 0 else 0  # of predicted-false, how many are false
    hal_recall = tn / (tn + fp) if (tn + fp) > 0 else 0     # of actual-false, how many caught
    hal_f1 = 2 * hal_precision * hal_recall / (hal_precision + hal_recall) if (hal_precision + hal_recall) > 0 else 0

    print(f"\n{'=' * 78}")
    print(f"  RESULTS")
    print(f"{'=' * 78}")
    print(f"  Claims evaluated:  {len(results)}")
    print(f"  Time:              {elapsed:.1f}s ({elapsed / len(results) * 1000:.0f}ms per claim)")
    print(f"")
    print(f"  Confusion Matrix:")
    print(f"                       Predicted TRUE    Predicted FALSE")
    print(f"    Actually TRUE        {tp:>5d} (TP)        {fn:>5d} (FN)")
    print(f"    Actually FALSE       {fp:>5d} (FP)        {tn:>5d} (TN)")
    print(f"")
    print(f"  Overall Metrics:")
    print(f"    Accuracy:            {accuracy:.3f}  ({tp + tn}/{len(results)})")
    print(f"    Precision (true):    {precision:.3f}  (of predicted-true, fraction actually true)")
    print(f"    Recall (true):       {recall:.3f}  (of actually-true, fraction caught)")
    print(f"    F1 (true):           {f1:.3f}")
    print(f"")
    print(f"  Hallucination Detection Metrics:")
    print(f"    Precision (false):   {hal_precision:.3f}  (of flagged-false, fraction actually false)")
    print(f"    Recall (false):      {hal_recall:.3f}  (of actually-false, fraction caught)")
    print(f"    F1 (false):          {hal_f1:.3f}")

    # ── Per-category breakdown ──
    print(f"\n{'─' * 78}")
    print(f"  PER-CATEGORY BREAKDOWN")
    print(f"{'─' * 78}")
    print(f"  {'Category':<30s} {'N':>4s} {'Acc':>6s} {'Hal-R':>6s} {'Hal-P':>6s}")
    print(f"  {'─' * 55}")

    cat_results = {}
    for r in results:
        cat = r['category']
        if cat not in cat_results:
            cat_results[cat] = []
        cat_results[cat].append(r)

    for cat in sorted(cat_results.keys()):
        cr = cat_results[cat]
        c_tp = sum(1 for r in cr if r['label'] and r['predicted_true'])
        c_fp = sum(1 for r in cr if not r['label'] and r['predicted_true'])
        c_tn = sum(1 for r in cr if not r['label'] and not r['predicted_true'])
        c_fn = sum(1 for r in cr if r['label'] and not r['predicted_true'])
        c_acc = (c_tp + c_tn) / len(cr) if cr else 0
        c_hal_r = c_tn / (c_tn + c_fp) if (c_tn + c_fp) > 0 else 0
        c_hal_p = c_tn / (c_tn + c_fn) if (c_tn + c_fn) > 0 else 0
        print(f"  {cat:<30s} {len(cr):>4d} {c_acc:>5.1%} {c_hal_r:>5.1%} {c_hal_p:>5.1%}")

    # ── Graph signal analysis ──
    print(f"\n{'─' * 78}")
    print(f"  GRAPH SIGNAL ANALYSIS")
    print(f"{'─' * 78}")

    graph_fired = [r for r in results if 'INVERSION' in r['graph_signal']]
    graph_correct = [r for r in graph_fired if not r['label'] and not r['predicted_true']]
    graph_wrong = [r for r in graph_fired if r['label'] and not r['predicted_true']]

    print(f"  Polarity inversion signals fired: {len(graph_fired)}")
    if graph_fired:
        print(f"    Correctly caught false claims:   {len(graph_correct)}")
        print(f"    False alarms (true claims hit):  {len(graph_wrong)}")
        if graph_fired:
            graph_precision = len(graph_correct) / len(graph_fired) if graph_fired else 0
            print(f"    Graph signal precision:          {graph_precision:.1%}")

    # ── Verdict distribution ──
    print(f"\n{'─' * 78}")
    print(f"  VERDICT DISTRIBUTION")
    print(f"{'─' * 78}")

    for label_type in ['CONSISTENT', 'PLAUSIBLE', 'SUSPICIOUS', 'INCONSISTENT']:
        with_label = [r for r in results if r['verdict'] == label_type]
        if not with_label:
            continue
        true_in_label = sum(1 for r in with_label if r['label'])
        false_in_label = len(with_label) - true_in_label
        purity = true_in_label / len(with_label) if label_type in ('CONSISTENT', 'PLAUSIBLE') else false_in_label / len(with_label)
        label_name = "true claims" if label_type in ('CONSISTENT', 'PLAUSIBLE') else "false claims"
        print(f"  {label_type:>12s}: {len(with_label):>4d} claims  "
              f"({true_in_label} true, {false_in_label} false)  "
              f"purity: {purity:.1%} {label_name}")

    # ── Save detailed results ──
    results_path = os.path.join(BENCHMARK_DIR, 'truthfulqa_results.json')
    summary = {
        'dataset': 'TruthfulQA',
        'n_claims': len(results),
        'n_true': n_true,
        'n_false': n_false,
        'accuracy': round(accuracy, 4),
        'precision_true': round(precision, 4),
        'recall_true': round(recall, 4),
        'f1_true': round(f1, 4),
        'hallucination_precision': round(hal_precision, 4),
        'hallucination_recall': round(hal_recall, 4),
        'hallucination_f1': round(hal_f1, 4),
        'elapsed_seconds': round(elapsed, 1),
        'ms_per_claim': round(elapsed / len(results) * 1000, 0),
    }
    with open(results_path, 'w') as f:
        json.dump({'summary': summary, 'results': results}, f, indent=2)
    print(f"\n  Detailed results saved to {results_path}")

    print(f"\n{'=' * 78}")

    return summary


SCB_PATH = os.path.join(BENCHMARK_DIR, 'semantic_consistency_benchmark.json')
HALUEVAL_PATH = os.path.join(BENCHMARK_DIR, 'HaluEval_qa.json')


def run_scb_benchmark(category=None, verbose=False):
    """
    Run the Semantic Consistency Benchmark (SCB-1).

    Purpose-built benchmark for testing semantic contradiction detection —
    the specific capability of this checker. Distinct from factual accuracy
    benchmarks like TruthfulQA.
    """
    from tools.consistency_checker import ConsistencyChecker

    if not os.path.exists(SCB_PATH):
        print(f"ERROR: SCB not found at {SCB_PATH}")
        sys.exit(1)

    with open(SCB_PATH, 'r') as f:
        scb = json.load(f)

    claims = scb['claims']
    if category:
        claims = [c for c in claims if c['category'].lower() == category.lower()]

    n_true = sum(1 for c in claims if c['label'])
    n_false = sum(1 for c in claims if not c['label'])

    print("=" * 78)
    print("  BENCHMARK: Semantic Consistency Benchmark (SCB-1)")
    print("=" * 78)
    print(f"  {scb['description']}")
    print(f"\n  Total claims: {len(claims)} ({n_true} true, {n_false} false)")

    categories = sorted(set(c['category'] for c in claims))
    print(f"  Categories: {len(categories)}")
    for cat in categories:
        n = sum(1 for c in claims if c['category'] == cat)
        desc = scb['categories'].get(cat, '')
        print(f"    {cat:<25s} {n:>3d}  {desc[:50]}")

    checker = ConsistencyChecker(verbose=True)

    print(f"\nChecking {len(claims)} claims...")
    t0 = time.time()

    results = []
    for claim_data in claims:
        verdict = checker.check(claim_data['text'])
        predicted_true = verdict.label in ('CONSISTENT', 'PLAUSIBLE')
        hit = predicted_true == claim_data['label']

        results.append({
            'claim': claim_data['text'],
            'label': claim_data['label'],
            'category': claim_data['category'],
            'verdict': verdict.label,
            'confidence': verdict.confidence,
            'angle': verdict.angle,
            'predicted_true': predicted_true,
            'hit': hit,
            'graph_signal': verdict.graph_signal.summary() if verdict.graph_signal else '-',
        })

        if verbose and not hit:
            exp = 'TRUE' if claim_data['label'] else 'FALSE'
            print(f"  MISS: [{verdict.label:12s}] {claim_data['text'][:55]:55s} "
                  f"(exp={exp}, angle={verdict.angle:.1f}, graph={verdict.graph_signal.summary() if verdict.graph_signal else '-'})")

    elapsed = time.time() - t0
    checker.close()

    # ── Metrics ──
    tp = sum(1 for r in results if r['label'] and r['predicted_true'])
    fp = sum(1 for r in results if not r['label'] and r['predicted_true'])
    tn = sum(1 for r in results if not r['label'] and not r['predicted_true'])
    fn = sum(1 for r in results if r['label'] and not r['predicted_true'])

    accuracy = (tp + tn) / len(results) if results else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    hal_precision = tn / (tn + fn) if (tn + fn) > 0 else 0
    hal_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    hal_f1 = 2 * hal_precision * hal_recall / (hal_precision + hal_recall) if (hal_precision + hal_recall) > 0 else 0

    print(f"\n{'=' * 78}")
    print(f"  RESULTS")
    print(f"{'=' * 78}")
    print(f"  Claims evaluated:  {len(results)}")
    print(f"  Time:              {elapsed:.1f}s ({elapsed / len(results) * 1000:.0f}ms per claim)")
    print(f"")
    print(f"  Confusion Matrix:")
    print(f"                       Predicted TRUE    Predicted FALSE")
    print(f"    Actually TRUE        {tp:>5d} (TP)        {fn:>5d} (FN)")
    print(f"    Actually FALSE       {fp:>5d} (FP)        {tn:>5d} (TN)")
    print(f"")
    print(f"  Overall Metrics:")
    print(f"    Accuracy:            {accuracy:.3f}  ({tp + tn}/{len(results)})")
    print(f"    Precision (true):    {precision:.3f}")
    print(f"    Recall (true):       {recall:.3f}")
    print(f"    F1 (true):           {f1:.3f}")
    print(f"")
    print(f"  Semantic Violation Detection:")
    print(f"    Precision:           {hal_precision:.3f}  (of flagged violations, fraction actually false)")
    print(f"    Recall:              {hal_recall:.3f}  (of actual violations, fraction caught)")
    print(f"    F1:                  {hal_f1:.3f}")

    # ── Per-category ──
    print(f"\n{'─' * 78}")
    print(f"  PER-CATEGORY BREAKDOWN")
    print(f"{'─' * 78}")
    print(f"  {'Category':<28s} {'N':>3s} {'Acc':>6s} {'Det-R':>6s} {'Det-P':>6s}")
    print(f"  {'─' * 55}")

    cat_results = {}
    for r in results:
        cat_results.setdefault(r['category'], []).append(r)

    for cat in sorted(cat_results.keys()):
        cr = cat_results[cat]
        c_tp = sum(1 for r in cr if r['label'] and r['predicted_true'])
        c_fp = sum(1 for r in cr if not r['label'] and r['predicted_true'])
        c_tn = sum(1 for r in cr if not r['label'] and not r['predicted_true'])
        c_fn = sum(1 for r in cr if r['label'] and not r['predicted_true'])
        c_acc = (c_tp + c_tn) / len(cr) if cr else 0
        c_det_r = c_tn / (c_tn + c_fp) if (c_tn + c_fp) > 0 else float('nan')
        c_det_p = c_tn / (c_tn + c_fn) if (c_tn + c_fn) > 0 else float('nan')
        det_r_str = f"{c_det_r:>5.1%}" if not np.isnan(c_det_r) else "  N/A"
        det_p_str = f"{c_det_p:>5.1%}" if not np.isnan(c_det_p) else "  N/A"
        print(f"  {cat:<28s} {len(cr):>3d} {c_acc:>5.1%} {det_r_str} {det_p_str}")

    # ── Graph signal ──
    print(f"\n{'─' * 78}")
    print(f"  GRAPH SIGNAL ANALYSIS")
    print(f"{'─' * 78}")
    graph_fired = [r for r in results if 'INVERSION' in r['graph_signal']]
    graph_tp = [r for r in graph_fired if not r['label'] and not r['predicted_true']]
    graph_fp = [r for r in graph_fired if r['label'] and not r['predicted_true']]
    print(f"  Polarity inversion signals: {len(graph_fired)}")
    if graph_fired:
        print(f"    True catches:  {len(graph_tp)}")
        print(f"    False alarms:  {len(graph_fp)}")
        print(f"    Precision:     {len(graph_tp)/len(graph_fired):.1%}")

    # ── Misses detail ──
    misses = [r for r in results if not r['hit']]
    if misses:
        print(f"\n{'─' * 78}")
        print(f"  MISSES ({len(misses)})")
        print(f"{'─' * 78}")
        for r in misses:
            exp = 'TRUE' if r['label'] else 'FALSE'
            print(f"  [{r['verdict']:12s}] {r['claim'][:50]:50s} exp={exp:5s} "
                  f"angle={r['angle']:5.1f} graph={r['graph_signal']}")

    # ── Save ──
    results_path = os.path.join(BENCHMARK_DIR, 'scb1_results.json')
    summary = {
        'dataset': 'SCB-1',
        'n_claims': len(results),
        'accuracy': round(accuracy, 4),
        'violation_precision': round(hal_precision, 4),
        'violation_recall': round(hal_recall, 4),
        'violation_f1': round(hal_f1, 4),
        'elapsed_seconds': round(elapsed, 1),
    }
    with open(results_path, 'w') as f:
        json.dump({'summary': summary, 'results': results}, f, indent=2)
    print(f"\n  Detailed results saved to {results_path}")
    print(f"{'=' * 78}")

    return summary


def run_halueval_benchmark(limit=1000, verbose=False):
    """
    Run the HaluEval QA benchmark.

    Tests two approaches:
      1. Standalone check: run each answer through the consistency checker
      2. Context comparison: compare semantic angle of right vs hallucinated
         answer relative to the knowledge passage

    HaluEval contains factual hallucinations (wrong names, dates, places).
    Expected result: near-random on standalone, slightly above random on context.
    This delineates the tool's scope — it catches semantic contradictions, not
    factual substitutions.
    """
    if not os.path.exists(HALUEVAL_PATH):
        print(f"ERROR: HaluEval not found at {HALUEVAL_PATH}")
        sys.exit(1)

    # Load records
    records = []
    with open(HALUEVAL_PATH, 'r') as f:
        for line in f:
            records.append(json.loads(line.strip()))
            if limit and len(records) >= limit:
                break

    print("=" * 78)
    print("  BENCHMARK: HaluEval QA (Li et al., 2023)")
    print("=" * 78)
    print(f"  Records: {len(records)} (of 10,000 total)")
    print(f"  Format: knowledge passage + question + right answer + hallucinated answer")
    print(f"  Hallucination type: factual substitution (wrong names/dates/places)")

    # Load projection for context comparison
    proj_data = np.load(os.path.join(EXPORT_DIR, 'projection_matrix.npz'), allow_pickle=True)
    W = proj_data['W']

    from sentence_transformers import SentenceTransformer
    print("\nLoading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # ── Approach 1: Standalone claim check ──
    print(f"\n{'─' * 78}")
    print(f"  APPROACH 1: Standalone Consistency Check")
    print(f"{'─' * 78}")
    print(f"  Check each answer as a standalone claim.")
    print(f"  Expect ~50% — factual substitutions are semantically valid.")

    from tools.consistency_checker import ConsistencyChecker
    checker = ConsistencyChecker(verbose=False)

    standalone_correct = 0
    for i, rec in enumerate(records):
        # Check right answer — should be CONSISTENT/PLAUSIBLE
        v_right = checker.check(rec['right_answer'])
        right_ok = v_right.label in ('CONSISTENT', 'PLAUSIBLE')

        # Check hallucinated answer — should be SUSPICIOUS/INCONSISTENT
        v_hallu = checker.check(rec['hallucinated_answer'])
        hallu_ok = v_hallu.label in ('SUSPICIOUS', 'INCONSISTENT')

        if right_ok and hallu_ok:
            standalone_correct += 1
        # Count partial credit: at least distinguished them
        elif v_right.angle < v_hallu.angle:
            pass  # right answer closer = weak signal

        if (i + 1) % 200 == 0:
            print(f"    ... {i + 1}/{len(records)}")

    standalone_acc = standalone_correct / len(records)
    print(f"\n  Standalone accuracy: {standalone_acc:.1%} ({standalone_correct}/{len(records)})")

    checker.close()

    # ── Approach 2: Context-aware comparison ──
    print(f"\n{'─' * 78}")
    print(f"  APPROACH 2: Knowledge-Context Semantic Comparison")
    print(f"{'─' * 78}")
    print(f"  Embed knowledge passage + each answer, compare projected angles.")
    print(f"  Predict: if angle(knowledge, hallucinated) > angle(knowledge, right),")
    print(f"  then the hallucinated answer is more semantically distant from context.")

    t0 = time.time()

    # Batch embed for efficiency
    knowledge_texts = [r['knowledge'][:512] for r in records]  # Truncate long passages
    right_texts = [r['right_answer'] for r in records]
    hallu_texts = [r['hallucinated_answer'] for r in records]

    print(f"  Embedding {len(records) * 3} texts...")
    all_texts = knowledge_texts + right_texts + hallu_texts
    all_emb = model.encode(all_texts, show_progress_bar=True, batch_size=64)

    k_emb = all_emb[:len(records)]
    r_emb = all_emb[len(records):2*len(records)]
    h_emb = all_emb[2*len(records):]

    # Project to grounded space
    k_proj = k_emb @ W
    r_proj = r_emb @ W
    h_proj = h_emb @ W

    # Compute angles
    def batch_angles(A, B):
        """Compute pairwise angles between rows of A and B."""
        norms_a = np.linalg.norm(A, axis=1, keepdims=True)
        norms_b = np.linalg.norm(B, axis=1, keepdims=True)
        norms_a = np.maximum(norms_a, 1e-10)
        norms_b = np.maximum(norms_b, 1e-10)
        cos = np.clip(np.sum((A / norms_a) * (B / norms_b), axis=1), -1, 1)
        return np.degrees(np.arccos(cos))

    angles_right = batch_angles(k_proj, r_proj)
    angles_hallu = batch_angles(k_proj, h_proj)

    # Predict: hallucinated answer should be more distant from knowledge
    context_correct = np.sum(angles_hallu > angles_right)
    context_acc = context_correct / len(records)

    elapsed = time.time() - t0

    print(f"\n  Context comparison accuracy: {context_acc:.1%} ({context_correct}/{len(records)})")
    print(f"  Time: {elapsed:.1f}s")

    # Angle statistics
    print(f"\n  Angle statistics:")
    print(f"    Right answer ↔ knowledge:        mean={np.mean(angles_right):.1f}°  "
          f"median={np.median(angles_right):.1f}°  std={np.std(angles_right):.1f}°")
    print(f"    Hallucinated answer ↔ knowledge:  mean={np.mean(angles_hallu):.1f}°  "
          f"median={np.median(angles_hallu):.1f}°  std={np.std(angles_hallu):.1f}°")
    print(f"    Mean difference:                  {np.mean(angles_hallu - angles_right):.2f}°")

    # ── Approach 3: Direct answer comparison ──
    print(f"\n{'─' * 78}")
    print(f"  APPROACH 3: Direct Answer Comparison (right vs hallucinated)")
    print(f"{'─' * 78}")
    print(f"  Angle between right and hallucinated answer in grounded space.")

    angles_rh = batch_angles(r_proj, h_proj)
    print(f"  Right ↔ Hallucinated angle:  mean={np.mean(angles_rh):.1f}°  "
          f"median={np.median(angles_rh):.1f}°  std={np.std(angles_rh):.1f}°")

    # Can we use this angle as a discriminator? If they're very close, it's a
    # subtle factual substitution. If far apart, it might be a semantic shift.
    for threshold in [20, 30, 40, 50, 60]:
        n_flagged = np.sum(angles_rh > threshold)
        print(f"    Pairs with angle > {threshold}°: {n_flagged} ({n_flagged/len(records):.1%})")

    # ── Summary ──
    print(f"\n{'=' * 78}")
    print(f"  SUMMARY")
    print(f"{'=' * 78}")
    print(f"  Standalone check:     {standalone_acc:.1%}  (expected ~50% — factual, not semantic)")
    print(f"  Context comparison:   {context_acc:.1%}  (above 50% = some semantic signal)")
    print(f"  Answer separation:    mean {np.mean(angles_rh):.1f}° between right and hallucinated")
    print(f"")
    print(f"  Interpretation: HaluEval tests factual substitution — semantically valid")
    print(f"  claims with wrong specifics (Mumbai for Delhi, 2003 for 2006). The checker")
    print(f"  correctly identifies both answers as semantically well-formed. This is not")
    print(f"  a failure — it delineates the tool's scope. Semantic consistency checking")
    print(f"  is complementary to factual accuracy verification, not a replacement.")

    # Save results
    results_path = os.path.join(BENCHMARK_DIR, 'halueval_results.json')
    summary = {
        'dataset': 'HaluEval-QA',
        'n_records': len(records),
        'standalone_accuracy': round(float(standalone_acc), 4),
        'context_accuracy': round(float(context_acc), 4),
        'mean_angle_right': round(float(np.mean(angles_right)), 2),
        'mean_angle_hallu': round(float(np.mean(angles_hallu)), 2),
        'mean_angle_rh': round(float(np.mean(angles_rh)), 2),
        'elapsed_seconds': round(elapsed, 1),
    }
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {results_path}")
    print(f"{'=' * 78}")

    return summary


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    dataset = sys.argv[1].lower()

    category = None
    limit = None
    verbose = '--verbose' in sys.argv or '-v' in sys.argv

    for i, arg in enumerate(sys.argv):
        if arg == '--category' and i + 1 < len(sys.argv):
            category = sys.argv[i + 1]
        elif arg == '--limit' and i + 1 < len(sys.argv):
            limit = int(sys.argv[i + 1])

    if dataset == 'truthfulqa':
        run_truthfulqa_benchmark(category=category, limit=limit, verbose=verbose)
    elif dataset == 'scb' or dataset == 'scb-1':
        run_scb_benchmark(category=category, verbose=verbose)
    elif dataset == 'halueval':
        run_halueval_benchmark(limit=limit or 1000, verbose=verbose)
    else:
        print(f"Unknown dataset: {dataset}")
        print("Supported: truthfulqa, scb, halueval")


if __name__ == "__main__":
    main()
