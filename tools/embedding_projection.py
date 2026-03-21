#!/usr/bin/env python3
"""
Phase 6: Embedding Projection — Bridge LLM embeddings to grounded semantic space.

Extracts LLM embeddings for all concepts, learns a linear projection matrix
from embedding space (384D) to grounded semantic space (14D), and evaluates
whether angular constraints survive the projection.

Usage:
    python -m tools.embedding_projection extract          # Step 1: Get LLM embeddings
    python -m tools.embedding_projection project          # Step 2: Learn projection matrix
    python -m tools.embedding_projection evaluate         # Step 3: Evaluate angular preservation
    python -m tools.embedding_projection check "X is Y"   # Step 4: Semantic consistency check
    python -m tools.embedding_projection report           # Full pipeline + report
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from api.semantic_core import SemanticCore

EXPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'exports')
EMBEDDINGS_PATH = os.path.join(EXPORT_DIR, 'llm_embeddings.npz')
PROJECTION_PATH = os.path.join(EXPORT_DIR, 'projection_matrix.npz')


def ensure_export_dir():
    os.makedirs(EXPORT_DIR, exist_ok=True)


def get_embedding_model():
    """Load sentence-transformers model (cached after first download)."""
    from sentence_transformers import SentenceTransformer
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model


def extract_embeddings():
    """Step 1: Extract LLM embeddings for all concepts."""
    ensure_export_dir()
    sc = SemanticCore()

    # Build input texts: "NAME: description" for richer context
    rows = sc.conn.execute(
        "SELECT name, description FROM concepts ORDER BY id"
    ).fetchall()

    names = []
    texts = []
    for r in rows:
        name = r['name']
        desc = r['description'] or ''
        names.append(name)
        # Use name alone for clean embedding, description as context
        if desc:
            texts.append(f"{name}: {desc}")
        else:
            texts.append(name)

    sc.close()

    print(f"Extracting embeddings for {len(names)} concepts...")
    model = get_embedding_model()

    t0 = time.time()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    elapsed = time.time() - t0

    print(f"  Done in {elapsed:.1f}s")
    print(f"  Embedding shape: {embeddings.shape}")
    print(f"  Embedding dim: {embeddings.shape[1]}")

    np.savez(EMBEDDINGS_PATH,
             names=np.array(names),
             texts=np.array(texts),
             embeddings=embeddings)

    print(f"  Saved to {EMBEDDINGS_PATH} ({os.path.getsize(EMBEDDINGS_PATH) / 1024:.0f} KB)")
    return names, embeddings


def load_embeddings():
    """Load previously extracted embeddings."""
    if not os.path.exists(EMBEDDINGS_PATH):
        print("No embeddings found. Run 'extract' first.")
        sys.exit(1)
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    return list(data['names']), data['embeddings']


def learn_projection():
    """Step 2: Learn linear projection from LLM space to grounded space."""
    ensure_export_dir()

    names, embeddings = load_embeddings()
    sc = SemanticCore()

    # Build target matrix (grounded 14D vectors)
    targets = []
    valid_indices = []
    for i, name in enumerate(names):
        concept = sc.get(name)
        if concept is not None:
            targets.append(concept.full_vector())
            valid_indices.append(i)

    sc.close()

    X = embeddings[valid_indices]  # (N, 384)
    Y = np.array(targets)          # (N, 14)

    N = X.shape[0]
    D_in = X.shape[1]
    D_out = Y.shape[1]

    print(f"Learning projection: {D_in}D → {D_out}D ({N} concepts)")

    # --- Method 1: Ordinary Least Squares ---
    # W = (X^T X)^{-1} X^T Y
    # Use pseudoinverse for numerical stability
    W_ols, residuals_ols, rank, sv = np.linalg.lstsq(X, Y, rcond=None)

    Y_pred_ols = X @ W_ols
    mse_ols = np.mean((Y - Y_pred_ols) ** 2)
    r2_ols = 1 - np.sum((Y - Y_pred_ols) ** 2) / np.sum((Y - Y.mean(axis=0)) ** 2)

    print(f"\n  OLS Projection:")
    print(f"    MSE:  {mse_ols:.6f}")
    print(f"    R²:   {r2_ols:.4f}")
    print(f"    Rank: {rank}")

    # Per-dimension R²
    r2_per_dim = []
    dim_labels = ['x', 'y', 'z', 'e', 'f', 'g', 'h', 'fx', 'fy', 'fz', 'fe', 'ff', 'fg', 'fh']
    for d in range(D_out):
        ss_res = np.sum((Y[:, d] - Y_pred_ols[:, d]) ** 2)
        ss_tot = np.sum((Y[:, d] - Y[:, d].mean()) ** 2)
        r2_d = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        r2_per_dim.append(r2_d)
    print(f"    Per-dimension R²:")
    for d, (label, r2) in enumerate(zip(dim_labels, r2_per_dim)):
        bar = '█' * int(r2 * 30) + '░' * (30 - int(r2 * 30))
        print(f"      {label:>2}: {bar} {r2:.3f}")

    # --- Method 2: Ridge Regression (regularized) ---
    # W = (X^T X + λI)^{-1} X^T Y
    # Cross-validate λ
    best_lambda = None
    best_cv_mse = float('inf')
    lambdas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    # Simple 5-fold CV
    fold_size = N // 5
    for lam in lambdas:
        cv_mses = []
        for fold in range(5):
            val_start = fold * fold_size
            val_end = val_start + fold_size
            X_train = np.vstack([X[:val_start], X[val_end:]])
            Y_train = np.vstack([Y[:val_start], Y[val_end:]])
            X_val = X[val_start:val_end]
            Y_val = Y[val_start:val_end]

            W_fold = np.linalg.solve(
                X_train.T @ X_train + lam * np.eye(D_in),
                X_train.T @ Y_train
            )
            Y_val_pred = X_val @ W_fold
            cv_mses.append(np.mean((Y_val - Y_val_pred) ** 2))

        mean_cv_mse = np.mean(cv_mses)
        if mean_cv_mse < best_cv_mse:
            best_cv_mse = mean_cv_mse
            best_lambda = lam

    # Train final ridge model with best lambda
    W_ridge = np.linalg.solve(
        X.T @ X + best_lambda * np.eye(D_in),
        X.T @ Y
    )
    Y_pred_ridge = X @ W_ridge
    mse_ridge = np.mean((Y - Y_pred_ridge) ** 2)
    r2_ridge = 1 - np.sum((Y - Y_pred_ridge) ** 2) / np.sum((Y - Y.mean(axis=0)) ** 2)

    print(f"\n  Ridge Projection (λ={best_lambda}):")
    print(f"    MSE:  {mse_ridge:.6f}")
    print(f"    R²:   {r2_ridge:.4f}")
    print(f"    CV MSE: {best_cv_mse:.6f}")

    # Choose better method
    if r2_ridge >= r2_ols:
        W_best = W_ridge
        method = f"ridge (λ={best_lambda})"
        r2_best = r2_ridge
        Y_pred_best = Y_pred_ridge
    else:
        W_best = W_ols
        method = "ols"
        r2_best = r2_ols
        Y_pred_best = Y_pred_ols

    print(f"\n  Selected: {method} (R²={r2_best:.4f})")

    # Save projection matrix and metadata
    np.savez(PROJECTION_PATH,
             W=W_best,
             method=np.array(method),
             r2=np.array(r2_best),
             r2_per_dim=np.array(r2_per_dim),
             dim_labels=np.array(dim_labels),
             mse=np.array(mse_ridge if method.startswith('ridge') else mse_ols),
             names=np.array(names),
             Y_true=Y,
             Y_pred=Y_pred_best)

    print(f"  Saved to {PROJECTION_PATH} ({os.path.getsize(PROJECTION_PATH) / 1024:.0f} KB)")
    return W_best, r2_best, Y, Y_pred_best, names


def load_projection():
    """Load previously learned projection."""
    if not os.path.exists(PROJECTION_PATH):
        print("No projection found. Run 'project' first.")
        sys.exit(1)
    data = np.load(PROJECTION_PATH, allow_pickle=True)
    return data


def evaluate_projection():
    """Step 3: Evaluate whether angular constraints survive the projection."""
    proj_data = load_projection()
    W = proj_data['W']
    Y_true = proj_data['Y_true']
    Y_pred = proj_data['Y_pred']
    names = list(proj_data['names'])

    sc = SemanticCore()

    print("=" * 70)
    print("ANGULAR PRESERVATION ANALYSIS")
    print("=" * 70)

    # Get all relations with known angular constraints
    relations = sc.conn.execute("""
        SELECT c1.name as name1, c2.name as name2, r.rel_type, r.angle_4d, r.angle_8d
        FROM relations r
        JOIN concepts c1 ON r.concept1_id = c1.id
        JOIN concepts c2 ON r.concept2_id = c2.id
        ORDER BY r.rel_type
    """).fetchall()

    # Build name->index lookup
    name_to_idx = {n: i for i, n in enumerate(names)}

    results_by_type = {}
    for rel in relations:
        n1, n2, rtype = rel['name1'], rel['name2'], rel['rel_type']
        if n1 not in name_to_idx or n2 not in name_to_idx:
            continue

        i, j = name_to_idx[n1], name_to_idx[n2]

        # True angle (from grounded vectors)
        true_angle = rel['angle_8d'] if rel['angle_8d'] else rel['angle_4d']
        if true_angle is None:
            continue

        # Projected angle (from LLM embeddings projected to grounded space)
        v1 = Y_pred[i]
        v2 = Y_pred[j]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            continue
        cos_sim = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1, 1)
        proj_angle = np.degrees(np.arccos(cos_sim))

        # Also compute angle in raw LLM embedding space (before projection)
        emb_data = np.load(EMBEDDINGS_PATH)
        emb = emb_data['embeddings']
        e1, e2 = emb[i], emb[j]
        cos_emb = np.clip(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)), -1, 1)
        emb_angle = np.degrees(np.arccos(cos_emb))

        if rtype not in results_by_type:
            results_by_type[rtype] = []
        results_by_type[rtype].append({
            'pair': f"{n1} — {n2}",
            'true_angle': true_angle,
            'proj_angle': proj_angle,
            'emb_angle': emb_angle,
            'error': abs(proj_angle - true_angle),
        })

    # Report per relation type
    for rtype in ['synonym', 'affinity', 'complement', 'opposition']:
        if rtype not in results_by_type:
            continue
        entries = results_by_type[rtype]
        errors = [e['error'] for e in entries]

        print(f"\n{'─' * 70}")
        print(f"  {rtype.upper()} ({len(entries)} pairs)")
        print(f"{'─' * 70}")

        # Expected angular ranges
        if rtype == 'synonym':
            expected = "< 30°"
        elif rtype == 'affinity':
            expected = "5–60°"
        elif rtype == 'complement':
            expected = "60–120° (target ~90°)"
        elif rtype == 'opposition':
            expected = "~90° (label only)"

        print(f"  Expected range:      {expected}")
        print(f"  Mean angle error:    {np.mean(errors):.1f}°")
        print(f"  Median angle error:  {np.median(errors):.1f}°")
        print(f"  Max angle error:     {np.max(errors):.1f}°")
        print(f"  Std angle error:     {np.std(errors):.1f}°")

        # Correlation between true and projected angles
        true_angles = [e['true_angle'] for e in entries]
        proj_angles = [e['proj_angle'] for e in entries]
        if len(true_angles) > 2:
            corr = np.corrcoef(true_angles, proj_angles)[0, 1]
            print(f"  Angle correlation:   {corr:.4f}")

        # Show best and worst preserved pairs
        sorted_entries = sorted(entries, key=lambda e: e['error'])
        print(f"\n  Best preserved (lowest error):")
        for e in sorted_entries[:5]:
            print(f"    {e['pair']:40s}  true={e['true_angle']:5.1f}°  proj={e['proj_angle']:5.1f}°  err={e['error']:4.1f}°")
        print(f"\n  Worst preserved (highest error):")
        for e in sorted_entries[-5:]:
            print(f"    {e['pair']:40s}  true={e['true_angle']:5.1f}°  proj={e['proj_angle']:5.1f}°  err={e['error']:4.1f}°")

    # --- Key test: LOVE + TRUTH → VIRTUE composition ---
    print(f"\n{'=' * 70}")
    print("COMPOSITION TEST: LOVE + TRUTH → VIRTUE")
    print(f"{'=' * 70}")

    for test_name, (a_name, b_name, expected_name) in [
        ("LOVE+TRUTH→VIRTUE", ("LOVE", "TRUTH", "VIRTUE")),
        ("FIRE+WATER→STEAM", ("FIRE", "WATER", "STEAM")),
        ("DARK+LIGHT→SHADOW", ("DARKNESS", "LIGHT", "SHADOW")),
    ]:
        if a_name not in name_to_idx or b_name not in name_to_idx:
            continue

        ai, bi = name_to_idx[a_name], name_to_idx[b_name]
        composed = (Y_pred[ai] + Y_pred[bi]) / 2  # Simple midpoint

        # Find nearest concept to composed vector
        dists = []
        for k, n in enumerate(names):
            v = Y_pred[k]
            norm_v = np.linalg.norm(v)
            norm_c = np.linalg.norm(composed)
            if norm_v < 1e-10 or norm_c < 1e-10:
                continue
            cos = np.clip(np.dot(composed, v) / (norm_v * norm_c), -1, 1)
            angle = np.degrees(np.arccos(cos))
            dists.append((n, angle))
        dists.sort(key=lambda x: x[1])

        print(f"\n  {test_name}:")
        print(f"    Nearest to midpoint:")
        for n, a in dists[:10]:
            marker = " <<<" if n == expected_name else ""
            print(f"      {n:25s} {a:5.1f}°{marker}")

        if expected_name in name_to_idx:
            ei = name_to_idx[expected_name]
            rank = next((i for i, (n, _) in enumerate(dists) if n == expected_name), -1)
            print(f"    {expected_name} rank: {rank + 1}")

    # --- Hallucination detection test ---
    print(f"\n{'=' * 70}")
    print("SEMANTIC CONSISTENCY TEST (hallucination detection)")
    print(f"{'=' * 70}")

    test_claims = [
        # True claims (low violation score expected)
        ("WATER", "LIQUID", "true"),
        ("LOVE", "EMOTION", "true"),
        ("FIRE", "HEAT", "true"),
        ("MURDER", "CRIME", "true"),
        # False claims (high violation score expected)
        ("WATER", "DRY", "false"),
        ("LOVE", "HATRED", "false"),
        ("FIRE", "COLD", "false"),
        ("TRUTH", "DECEPTION", "false"),
    ]

    print(f"\n  Claim: 'A is B' → check if projected vectors agree")
    print(f"  {'Claim':30s} {'Angle':>7s} {'Expected':>10s} {'Verdict':>8s}")
    print(f"  {'─' * 60}")

    for a_name, b_name, expected in test_claims:
        if a_name not in name_to_idx or b_name not in name_to_idx:
            continue
        ai, bi = name_to_idx[a_name], name_to_idx[b_name]
        v1, v2 = Y_pred[ai], Y_pred[bi]
        cos = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)
        angle = np.degrees(np.arccos(cos))

        # "A is B" should have small angle (high similarity)
        # Large angle → violation → hallucination signal
        threshold = 45  # degrees — tunable
        verdict = "OK" if (angle < threshold) == (expected == "true") else "MISS"
        claim = f"{a_name} is {b_name}"
        print(f"  {claim:30s} {angle:6.1f}° {'should match' if expected == 'true' else 'should differ':>10s}  {verdict:>5s}")

    sc.close()


def check_consistency(claim_text: str):
    """Step 4: Check a natural language claim for semantic consistency."""
    proj_data = load_projection()
    W = proj_data['W']

    model = get_embedding_model()
    sc = SemanticCore()

    # Parse "X is Y" style claims
    parts = claim_text.lower().split(' is ')
    if len(parts) != 2:
        parts = claim_text.lower().split(' are ')
    if len(parts) != 2:
        print(f"  Expected 'X is Y' format, got: {claim_text}")
        return

    subject, predicate = parts[0].strip(), parts[1].strip()

    # Embed both terms
    embeddings = model.encode([subject, predicate])
    proj_subject = embeddings[0] @ W  # Project to grounded space
    proj_predicate = embeddings[1] @ W

    # Angle between projected vectors
    cos = np.clip(
        np.dot(proj_subject, proj_predicate) /
        (np.linalg.norm(proj_subject) * np.linalg.norm(proj_predicate)),
        -1, 1
    )
    angle = np.degrees(np.arccos(cos))

    # Find nearest dictionary concepts for each
    names_data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    all_names = list(names_data['names'])
    all_emb = names_data['embeddings']
    all_proj = all_emb @ W

    def find_nearest(vec, top_n=5):
        results = []
        for i, name in enumerate(all_names):
            v = all_proj[i]
            n1, n2 = np.linalg.norm(vec), np.linalg.norm(v)
            if n1 < 1e-10 or n2 < 1e-10:
                continue
            c = np.clip(np.dot(vec, v) / (n1 * n2), -1, 1)
            a = np.degrees(np.arccos(c))
            results.append((name, a))
        results.sort(key=lambda x: x[1])
        return results[:top_n]

    print(f"\n  Claim: \"{claim_text}\"")
    print(f"  Angle between projected terms: {angle:.1f}°")

    if angle < 30:
        print(f"  Verdict: CONSISTENT (strong semantic alignment)")
    elif angle < 60:
        print(f"  Verdict: PLAUSIBLE (moderate alignment)")
    elif angle < 90:
        print(f"  Verdict: SUSPICIOUS (weak alignment)")
    else:
        print(f"  Verdict: INCONSISTENT (semantic violation — hallucination signal)")

    print(f"\n  '{subject}' maps nearest to:")
    for name, a in find_nearest(proj_subject):
        print(f"    {name:25s} {a:5.1f}°")

    print(f"\n  '{predicate}' maps nearest to:")
    for name, a in find_nearest(proj_predicate):
        print(f"    {name:25s} {a:5.1f}°")

    # Check if any known relations exist between nearest concepts
    subj_nearest = find_nearest(proj_subject, 3)
    pred_nearest = find_nearest(proj_predicate, 3)
    print(f"\n  Known relations between nearest matches:")
    found_any = False
    for sn, _ in subj_nearest:
        for pn, _ in pred_nearest:
            rels = sc.relations_of(sn)
            for r in rels:
                if r['other'] == pn:
                    print(f"    {sn} —[{r['type']}]→ {pn} (angle: {r.get('angle_8d', r.get('angle_4d', '?'))}°)")
                    found_any = True
    if not found_any:
        print(f"    (none found)")

    sc.close()


def run_full_report():
    """Run the full pipeline and generate a report."""
    print("=" * 70)
    print("AGI SEMANTIC CORE — PHASE 6: EMBEDDING PROJECTION")
    print("=" * 70)

    # Step 1
    print(f"\n{'=' * 70}")
    print("STEP 1: EXTRACT LLM EMBEDDINGS")
    print(f"{'=' * 70}")
    extract_embeddings()

    # Step 2
    print(f"\n{'=' * 70}")
    print("STEP 2: LEARN PROJECTION MATRIX")
    print(f"{'=' * 70}")
    learn_projection()

    # Step 3
    print(f"\n{'=' * 70}")
    print("STEP 3: EVALUATE ANGULAR PRESERVATION")
    print(f"{'=' * 70}")
    evaluate_projection()

    print(f"\n{'=' * 70}")
    print("PHASE 6 COMPLETE")
    print(f"{'=' * 70}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()

    if cmd == "extract":
        extract_embeddings()
    elif cmd == "project":
        learn_projection()
    elif cmd == "evaluate":
        evaluate_projection()
    elif cmd == "check":
        if len(sys.argv) < 3:
            print("Usage: python -m tools.embedding_projection check \"X is Y\"")
            return
        check_consistency(sys.argv[2])
    elif cmd == "report":
        run_full_report()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
