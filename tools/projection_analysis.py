#!/usr/bin/env python3
"""
Phase 6 — Projection Failure Mode Analysis
============================================

Comprehensive analysis of WHERE the 384D→14D linear projection succeeds and fails.
Answers: What does the LLM already know? What does the grounded dictionary add?

Usage:
    python3 -m tools.projection_analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import defaultdict
from api.semantic_core import SemanticCore

EXPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'exports')
EMBEDDINGS_PATH = os.path.join(EXPORT_DIR, 'llm_embeddings.npz')
PROJECTION_PATH = os.path.join(EXPORT_DIR, 'projection_matrix.npz')

DIM_LABELS = ['x', 'y', 'z', 'e', 'f', 'g', 'h', 'fx', 'fy', 'fz', 'fe', 'ff', 'fg', 'fh']
DIM_NAMES = {
    'x': 'Core X (spatial/abstract axis)',
    'y': 'Core Y (polar/creative axis)',
    'z': 'Core Z (structural/dynamic axis)',
    'e': 'Domain: Spatial',
    'f': 'Domain: Temporal',
    'g': 'Domain: Relational',
    'h': 'Domain: Personal',
    'fx': 'Function X',
    'fy': 'Function Y',
    'fz': 'Function Z',
    'fe': 'Function Spatial',
    'ff': 'Function Temporal',
    'fg': 'Function Relational',
    'fh': 'Function Personal',
}


def load_data():
    """Load all projection data and build lookup structures."""
    proj = np.load(PROJECTION_PATH, allow_pickle=True)
    emb = np.load(EMBEDDINGS_PATH, allow_pickle=True)

    names = list(proj['names'])
    Y_true = proj['Y_true']
    Y_pred = proj['Y_pred']
    W = proj['W']
    embeddings = emb['embeddings']
    r2_per_dim = proj['r2_per_dim']

    name_to_idx = {n: i for i, n in enumerate(names)}

    return {
        'names': names,
        'Y_true': Y_true,
        'Y_pred': Y_pred,
        'W': W,
        'embeddings': embeddings,
        'r2_per_dim': r2_per_dim,
        'name_to_idx': name_to_idx,
    }


def per_concept_mse(Y_true, Y_pred):
    """Compute MSE for each concept (average across 14 dims)."""
    return np.mean((Y_true - Y_pred) ** 2, axis=1)


def per_concept_r2(Y_true, Y_pred):
    """Compute R^2 for each concept (across 14 dims, treating dims as 'samples')."""
    ss_res = np.sum((Y_true - Y_pred) ** 2, axis=1)
    mean_true = np.mean(Y_true, axis=1, keepdims=True)
    ss_tot = np.sum((Y_true - mean_true) ** 2, axis=1)
    r2 = np.where(ss_tot > 1e-10, 1 - ss_res / ss_tot, 0.0)
    return r2


def section_header(title, char='='):
    width = 78
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def sub_header(title, char='-'):
    width = 72
    print(f"\n  {char * width}")
    print(f"  {title}")
    print(f"  {char * width}")


def run_analysis():
    print("=" * 78)
    print("  AGI SEMANTIC CORE — PROJECTION FAILURE MODE ANALYSIS")
    print("  384D (all-MiniLM-L6-v2) -> 14D (grounded dual-octonion space)")
    print("=" * 78)

    data = load_data()
    names = data['names']
    Y_true = data['Y_true']
    Y_pred = data['Y_pred']
    W = data['W']
    embeddings = data['embeddings']
    r2_per_dim = data['r2_per_dim']
    name_to_idx = data['name_to_idx']

    N = len(names)
    mse_per = per_concept_mse(Y_true, Y_pred)

    # Global stats
    global_r2 = 1 - np.sum((Y_true - Y_pred) ** 2) / np.sum((Y_true - Y_true.mean(axis=0)) ** 2)
    global_mse = np.mean(mse_per)

    print(f"\n  Concepts:   {N}")
    print(f"  Global R^2: {global_r2:.4f}")
    print(f"  Global MSE: {global_mse:.6f}")
    print(f"  MSE range:  [{mse_per.min():.6f}, {mse_per.max():.6f}]")
    print(f"  MSE median: {np.median(mse_per):.6f}")
    print(f"  MSE stdev:  {np.std(mse_per):.6f}")

    # Load DB metadata for grouping
    sc = SemanticCore()
    concept_meta = {}
    rows = sc.conn.execute(
        "SELECT name, trigram, level, description FROM concepts ORDER BY id"
    ).fetchall()
    for r in rows:
        concept_meta[r['name']] = {
            'trigram': r['trigram'],
            'level': r['level'],
            'description': r['description'] or '',
        }

    # ===================================================================
    # A. TOP 50 WORST-RECONSTRUCTED CONCEPTS
    # ===================================================================
    section_header("A. TOP 50 WORST-RECONSTRUCTED CONCEPTS (highest per-concept MSE)")

    sorted_worst = np.argsort(mse_per)[::-1]
    print(f"\n  {'Rank':>4s}  {'Name':30s}  {'MSE':>10s}  {'Trigram':>7s}  {'Level':>14s}")
    print(f"  {'----':>4s}  {'-'*30}  {'-'*10}  {'-'*7}  {'-'*14}")

    worst_trigrams = defaultdict(int)
    worst_levels = defaultdict(int)
    for rank, idx in enumerate(sorted_worst[:50], 1):
        name = names[idx]
        meta = concept_meta.get(name, {})
        trigram = meta.get('trigram', '?')
        level = meta.get('level', '?')
        worst_trigrams[trigram] += 1
        worst_levels[level] += 1
        print(f"  {rank:4d}  {name:30s}  {mse_per[idx]:10.6f}  {trigram:>7s}  {str(level):>14s}")

    sub_header("Pattern analysis: worst 50")
    print(f"  Trigram breakdown: {dict(sorted(worst_trigrams.items(), key=lambda x: -x[1]))}")
    print(f"  Level breakdown:   {dict(sorted(worst_levels.items(), key=lambda x: -x[1]))}")

    # Check if worst concepts have unusually large true vectors
    worst_norms = [np.linalg.norm(Y_true[idx]) for idx in sorted_worst[:50]]
    all_norms = np.linalg.norm(Y_true, axis=1)
    print(f"\n  Mean ||true|| for worst 50: {np.mean(worst_norms):.4f}  (overall: {np.mean(all_norms):.4f})")
    print(f"  Mean ||true|| for best 50:  {np.mean([np.linalg.norm(Y_true[i]) for i in np.argsort(mse_per)[:50]]):.4f}")

    # ===================================================================
    # B. TOP 50 BEST-RECONSTRUCTED CONCEPTS
    # ===================================================================
    section_header("B. TOP 50 BEST-RECONSTRUCTED CONCEPTS (lowest per-concept MSE)")

    sorted_best = np.argsort(mse_per)
    print(f"\n  {'Rank':>4s}  {'Name':30s}  {'MSE':>10s}  {'Trigram':>7s}  {'Level':>14s}")
    print(f"  {'----':>4s}  {'-'*30}  {'-'*10}  {'-'*7}  {'-'*14}")

    best_trigrams = defaultdict(int)
    best_levels = defaultdict(int)
    for rank, idx in enumerate(sorted_best[:50], 1):
        name = names[idx]
        meta = concept_meta.get(name, {})
        trigram = meta.get('trigram', '?')
        level = meta.get('level', '?')
        best_trigrams[trigram] += 1
        best_levels[level] += 1
        print(f"  {rank:4d}  {name:30s}  {mse_per[idx]:10.6f}  {trigram:>7s}  {str(level):>14s}")

    sub_header("Pattern analysis: best 50")
    print(f"  Trigram breakdown: {dict(sorted(best_trigrams.items(), key=lambda x: -x[1]))}")
    print(f"  Level breakdown:   {dict(sorted(best_levels.items(), key=lambda x: -x[1]))}")

    # ===================================================================
    # C. PER-TRIGRAM ANALYSIS
    # ===================================================================
    section_header("C. PER-TRIGRAM ANALYSIS")

    trigram_groups = defaultdict(list)
    for i, name in enumerate(names):
        meta = concept_meta.get(name, {})
        tri = meta.get('trigram', 'UNKNOWN')
        trigram_groups[tri].append(i)

    trigram_stats = []
    for tri in ['QIAN', 'KUN', 'ZHEN', 'XUN', 'LI', 'KAN', 'DUI', 'GEN']:
        if tri not in trigram_groups:
            continue
        indices = trigram_groups[tri]
        t_mse = mse_per[indices]
        # Per-trigram R^2
        Y_t = Y_true[indices]
        Y_p = Y_pred[indices]
        ss_res = np.sum((Y_t - Y_p) ** 2)
        ss_tot = np.sum((Y_t - Y_t.mean(axis=0)) ** 2)
        t_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        trigram_stats.append((tri, len(indices), np.mean(t_mse), np.median(t_mse), np.std(t_mse), t_r2))

    trigram_stats.sort(key=lambda x: -x[5])  # sort by R^2 descending

    print(f"\n  {'Trigram':>7s}  {'Count':>5s}  {'Mean MSE':>10s}  {'Med MSE':>10s}  {'Std MSE':>10s}  {'R^2':>8s}")
    print(f"  {'-'*7}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
    for tri, cnt, mean_mse, med_mse, std_mse, r2 in trigram_stats:
        bar = '#' * int(r2 * 20) + '.' * (20 - int(r2 * 20))
        print(f"  {tri:>7s}  {cnt:5d}  {mean_mse:10.6f}  {med_mse:10.6f}  {std_mse:10.6f}  {r2:8.4f}  {bar}")

    best_tri = trigram_stats[0]
    worst_tri = trigram_stats[-1]
    print(f"\n  Best captured trigram:  {best_tri[0]} (R^2={best_tri[5]:.4f}, n={best_tri[1]})")
    print(f"  Worst captured trigram: {worst_tri[0]} (R^2={worst_tri[5]:.4f}, n={worst_tri[1]})")

    # ===================================================================
    # D. PER-LEVEL ANALYSIS
    # ===================================================================
    section_header("D. PER-LEVEL ANALYSIS")

    level_order = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    level_names_map = {0: 'UNITY', 1: 'DYAD', 2: 'TRIAD', 3: 'TETRAD',
                       4: 'QUALITY', 5: 'DERIVED', 6: 'VERB', 7: 'ABSTRACT', 8: 'INTERROGATIVE'}

    level_groups = defaultdict(list)
    for i, name in enumerate(names):
        meta = concept_meta.get(name, {})
        lv = meta.get('level', -1)
        level_groups[lv].append(i)

    level_stats = []
    for lv in level_order:
        if lv not in level_groups or len(level_groups[lv]) < 2:
            lv_name = level_names_map.get(lv, str(lv))
            if lv in level_groups:
                level_stats.append((lv_name, len(level_groups[lv]), np.mean(mse_per[level_groups[lv]]), 0, 0, float('nan')))
            continue
        indices = level_groups[lv]
        lv_name = level_names_map.get(lv, str(lv))
        l_mse = mse_per[indices]
        Y_l = Y_true[indices]
        Y_p = Y_pred[indices]
        ss_res = np.sum((Y_l - Y_p) ** 2)
        ss_tot = np.sum((Y_l - Y_l.mean(axis=0)) ** 2)
        l_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        level_stats.append((lv_name, len(indices), np.mean(l_mse), np.median(l_mse), np.std(l_mse), l_r2))

    print(f"\n  {'Level':>14s}  {'Count':>5s}  {'Mean MSE':>10s}  {'Med MSE':>10s}  {'Std MSE':>10s}  {'R^2':>8s}")
    print(f"  {'-'*14}  {'-'*5}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}")
    for lv_name, cnt, mean_mse, med_mse, std_mse, r2 in level_stats:
        r2_str = f"{r2:8.4f}" if not np.isnan(r2) else "   n/a  "
        bar = '#' * int(max(0, r2) * 20) + '.' * (20 - int(max(0, r2) * 20)) if not np.isnan(r2) else ''
        print(f"  {lv_name:>14s}  {cnt:5d}  {mean_mse:10.6f}  {med_mse:10.6f}  {std_mse:10.6f}  {r2_str}  {bar}")

    # ===================================================================
    # E. DIMENSION-SPECIFIC FAILURES
    # ===================================================================
    section_header("E. DIMENSION-SPECIFIC FAILURES")
    print("  For each of the 14 dimensions, the concepts with largest absolute residual.")
    print("  This reveals which grounded encodings the LLM systematically misses.\n")

    residuals = Y_true - Y_pred  # (N, 14)

    for d in range(14):
        dim_label = DIM_LABELS[d]
        dim_name = DIM_NAMES[dim_label]
        dim_r2 = r2_per_dim[d]
        abs_resid = np.abs(residuals[:, d])
        sorted_idx = np.argsort(abs_resid)[::-1]

        print(f"  {dim_label:>2s} ({dim_name}) — R^2 = {dim_r2:.3f}")
        print(f"  {'':4s}{'Name':30s}  {'True':>8s}  {'Pred':>8s}  {'Resid':>8s}  {'Dir':>4s}")
        for rank in range(10):
            idx = sorted_idx[rank]
            true_val = Y_true[idx, d]
            pred_val = Y_pred[idx, d]
            resid = residuals[idx, d]
            direction = '+' if resid > 0 else '-'
            print(f"  {rank+1:3d} {names[idx]:30s}  {true_val:+8.4f}  {pred_val:+8.4f}  {resid:+8.4f}  {direction:>4s}")

        # Summary: mean bias and mean absolute residual
        mean_bias = np.mean(residuals[:, d])
        mean_abs_resid = np.mean(abs_resid)
        print(f"      Mean bias: {mean_bias:+.6f}   Mean |residual|: {mean_abs_resid:.6f}")
        print()

    # ===================================================================
    # F. COMPLEMENT PAIR ANALYSIS
    # ===================================================================
    section_header("F. COMPLEMENT PAIR ANALYSIS (992 pairs)")

    complement_rels = sc.conn.execute("""
        SELECT c1.name as name1, c2.name as name2, r.angle_4d, r.angle_8d
        FROM relations r
        JOIN concepts c1 ON r.concept1_id = c1.id
        JOIN concepts c2 ON r.concept2_id = c2.id
        WHERE r.rel_type = 'complement'
    """).fetchall()

    true_angles = []
    proj_angles = []
    angle_errors = []
    pair_details = []

    for rel in complement_rels:
        n1, n2 = rel['name1'], rel['name2']
        if n1 not in name_to_idx or n2 not in name_to_idx:
            continue

        i, j = name_to_idx[n1], name_to_idx[n2]

        # True angle in grounded 14D space
        v1_true = Y_true[i]
        v2_true = Y_true[j]
        norm1 = np.linalg.norm(v1_true)
        norm2 = np.linalg.norm(v2_true)
        if norm1 < 1e-10 or norm2 < 1e-10:
            continue
        cos_true = np.clip(np.dot(v1_true, v2_true) / (norm1 * norm2), -1, 1)
        true_angle = np.degrees(np.arccos(cos_true))

        # Projected angle
        v1_pred = Y_pred[i]
        v2_pred = Y_pred[j]
        norm1p = np.linalg.norm(v1_pred)
        norm2p = np.linalg.norm(v2_pred)
        if norm1p < 1e-10 or norm2p < 1e-10:
            continue
        cos_pred = np.clip(np.dot(v1_pred, v2_pred) / (norm1p * norm2p), -1, 1)
        proj_angle = np.degrees(np.arccos(cos_pred))

        true_angles.append(true_angle)
        proj_angles.append(proj_angle)
        angle_errors.append(proj_angle - true_angle)  # signed: negative = compression
        pair_details.append((n1, n2, true_angle, proj_angle))

    true_angles = np.array(true_angles)
    proj_angles = np.array(proj_angles)
    angle_errors = np.array(angle_errors)

    print(f"\n  Complement pairs analyzed: {len(true_angles)}")
    print(f"\n  True angle stats:")
    print(f"    Mean:   {np.mean(true_angles):6.2f}")
    print(f"    Median: {np.median(true_angles):6.2f}")
    print(f"    Std:    {np.std(true_angles):6.2f}")
    print(f"    Range:  [{np.min(true_angles):.2f}, {np.max(true_angles):.2f}]")

    print(f"\n  Projected angle stats:")
    print(f"    Mean:   {np.mean(proj_angles):6.2f}")
    print(f"    Median: {np.median(proj_angles):6.2f}")
    print(f"    Std:    {np.std(proj_angles):6.2f}")
    print(f"    Range:  [{np.min(proj_angles):.2f}, {np.max(proj_angles):.2f}]")

    print(f"\n  Angle error (projected - true):  negative = compression, positive = expansion")
    print(f"    Mean error:     {np.mean(angle_errors):+6.2f}")
    print(f"    Median error:   {np.median(angle_errors):+6.2f}")
    print(f"    Std error:      {np.std(angle_errors):6.2f}")
    print(f"    Mean |error|:   {np.mean(np.abs(angle_errors)):6.2f}")

    n_compressed = np.sum(angle_errors < 0)
    n_expanded = np.sum(angle_errors > 0)
    n_exact = np.sum(angle_errors == 0)
    print(f"\n  Compression vs expansion:")
    print(f"    Compressed (proj < true): {n_compressed:4d} ({100*n_compressed/len(angle_errors):5.1f}%)")
    print(f"    Expanded   (proj > true): {n_expanded:4d} ({100*n_expanded/len(angle_errors):5.1f}%)")

    # Angle correlation
    corr = np.corrcoef(true_angles, proj_angles)[0, 1]
    print(f"\n  True vs projected angle correlation: {corr:.4f}")

    # Text histogram of angle errors
    sub_header("Angle error distribution (projected - true)")
    bins = np.arange(-60, 65, 5)
    hist, edges = np.histogram(angle_errors, bins=bins)
    max_count = max(hist) if max(hist) > 0 else 1
    for i in range(len(hist)):
        bar_len = int(50 * hist[i] / max_count)
        bar = '#' * bar_len
        label = f"  [{edges[i]:+5.0f}, {edges[i+1]:+5.0f})"
        print(f"  {label}  {hist[i]:4d}  {bar}")

    # Worst-preserved complement pairs
    sub_header("10 worst-preserved complement pairs (largest |angle error|)")
    sorted_pairs = sorted(pair_details, key=lambda x: abs(x[3] - x[2]), reverse=True)
    print(f"  {'Name1':25s}  {'Name2':25s}  {'True':>7s}  {'Proj':>7s}  {'Error':>7s}")
    print(f"  {'-'*25}  {'-'*25}  {'-'*7}  {'-'*7}  {'-'*7}")
    for n1, n2, ta, pa in sorted_pairs[:10]:
        print(f"  {n1:25s}  {n2:25s}  {ta:7.2f}  {pa:7.2f}  {pa-ta:+7.2f}")

    sub_header("10 best-preserved complement pairs (smallest |angle error|)")
    print(f"  {'Name1':25s}  {'Name2':25s}  {'True':>7s}  {'Proj':>7s}  {'Error':>7s}")
    print(f"  {'-'*25}  {'-'*25}  {'-'*7}  {'-'*7}  {'-'*7}")
    for n1, n2, ta, pa in sorted_pairs[-10:]:
        print(f"  {n1:25s}  {n2:25s}  {ta:7.2f}  {pa:7.2f}  {pa-ta:+7.2f}")

    # ===================================================================
    # G. DICTIONARY VALUE-ADD SCORE
    # ===================================================================
    section_header("G. DICTIONARY VALUE-ADD SCORE")
    print("  For each concept, how much does the grounded 14D encoding DISAGREE")
    print("  with the LLM's projection? High disagreement = the dictionary is adding")
    print("  unique semantic structure that the LLM lacks.")
    print()
    print("  Score = ||Y_true - Y_pred|| / ||Y_true||  (normalized reconstruction error)")
    print("  A score of 1.0 means the LLM is as wrong as guessing the zero vector.")
    print("  A score > 1.0 means the LLM is actively pointing the wrong direction.")

    # Compute normalized error: ||true - pred|| / ||true||
    true_norms = np.linalg.norm(Y_true, axis=1)
    error_norms = np.linalg.norm(Y_true - Y_pred, axis=1)
    # Avoid division by zero
    value_add = np.where(true_norms > 1e-10, error_norms / true_norms, 0.0)

    sorted_va = np.argsort(value_add)[::-1]

    print(f"\n  {'Rank':>4s}  {'Name':30s}  {'ValueAdd':>9s}  {'||err||':>9s}  {'||true||':>9s}  {'Trigram':>7s}  {'Level':>14s}")
    print(f"  {'----':>4s}  {'-'*30}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*14}")

    va_trigrams = defaultdict(int)
    va_levels = defaultdict(int)
    for rank, idx in enumerate(sorted_va[:50], 1):
        name = names[idx]
        meta = concept_meta.get(name, {})
        trigram = meta.get('trigram', '?')
        level = meta.get('level', '?')
        va_trigrams[trigram] += 1
        va_levels[level] += 1
        print(f"  {rank:4d}  {name:30s}  {value_add[idx]:9.4f}  {error_norms[idx]:9.4f}  {true_norms[idx]:9.4f}  {trigram:>7s}  {str(level):>14s}")

    sub_header("Pattern analysis: top 50 value-add")
    print(f"  Trigram breakdown: {dict(sorted(va_trigrams.items(), key=lambda x: -x[1]))}")
    print(f"  Level breakdown:   {dict(sorted(va_levels.items(), key=lambda x: -x[1]))}")

    # Per-dimension value-add: which dimensions contribute most to disagreement?
    sub_header("Per-dimension contribution to value-add (mean |residual| per dimension)")
    dim_mean_abs_resid = np.mean(np.abs(residuals), axis=0)
    sorted_dims = np.argsort(dim_mean_abs_resid)[::-1]
    for d in sorted_dims:
        bar = '#' * int(dim_mean_abs_resid[d] * 100) + '.' * max(0, 20 - int(dim_mean_abs_resid[d] * 100))
        print(f"  {DIM_LABELS[d]:>2s} ({DIM_NAMES[DIM_LABELS[d]]:30s}):  {dim_mean_abs_resid[d]:.6f}  R^2={r2_per_dim[d]:.3f}  {bar}")

    # Bottom 50 (lowest value-add = LLM already captures well)
    sub_header("Bottom 50: lowest value-add (LLM already knows these)")
    sorted_va_asc = np.argsort(value_add)
    print(f"\n  {'Rank':>4s}  {'Name':30s}  {'ValueAdd':>9s}  {'Trigram':>7s}  {'Level':>14s}")
    print(f"  {'----':>4s}  {'-'*30}  {'-'*9}  {'-'*7}  {'-'*14}")
    low_trigrams = defaultdict(int)
    low_levels = defaultdict(int)
    for rank, idx in enumerate(sorted_va_asc[:50], 1):
        name = names[idx]
        meta = concept_meta.get(name, {})
        trigram = meta.get('trigram', '?')
        level = meta.get('level', '?')
        low_trigrams[trigram] += 1
        low_levels[level] += 1
        print(f"  {rank:4d}  {name:30s}  {value_add[idx]:9.4f}  {trigram:>7s}  {str(level):>14s}")

    print(f"\n  Trigram breakdown: {dict(sorted(low_trigrams.items(), key=lambda x: -x[1]))}")
    print(f"  Level breakdown:   {dict(sorted(low_levels.items(), key=lambda x: -x[1]))}")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    section_header("SUMMARY: WHERE THE LLM FAILS vs SUCCEEDS")

    print(f"""
  OVERALL: R^2 = {global_r2:.4f} — the LLM captures ~{100*global_r2:.0f}% of variance
  in the grounded 14D space via linear projection.

  WHAT THE LLM CAPTURES WELL:
    - Best dimensions:  {', '.join(f"{DIM_LABELS[d]}(R^2={r2_per_dim[d]:.2f})" for d in np.argsort(r2_per_dim)[::-1][:4])}
    - Best trigram:     {best_tri[0]} (R^2={best_tri[5]:.4f})
    - Best levels:      {', '.join(lv for lv, _, _, _, _, r2 in sorted(level_stats, key=lambda x: -x[5] if not np.isnan(x[5]) else 1)[:3])}

  WHAT THE LLM MISSES:
    - Worst dimensions: {', '.join(f"{DIM_LABELS[d]}(R^2={r2_per_dim[d]:.2f})" for d in np.argsort(r2_per_dim)[:4])}
    - Worst trigram:    {worst_tri[0]} (R^2={worst_tri[5]:.4f})
    - Worst levels:     {', '.join(lv for lv, _, _, _, _, r2 in sorted(level_stats, key=lambda x: x[5] if not np.isnan(x[5]) else 999)[:3])}

  COMPLEMENT ANALYSIS:
    - True angle mean:      {np.mean(true_angles):.2f} degrees
    - Projected angle mean: {np.mean(proj_angles):.2f} degrees
    - Systematic bias:      {np.mean(angle_errors):+.2f} degrees  ({'COMPRESSION' if np.mean(angle_errors) < 0 else 'EXPANSION'})
    - Angle correlation:    {corr:.4f}

  VALUE-ADD: The dictionary's highest value comes from concepts where the LLM
  embedding DISAGREES with the grounded encoding. These are concepts where
  the dictionary provides structure the LLM cannot infer from distributional
  statistics alone.
    - Mean value-add score: {np.mean(value_add):.4f}
    - Top quartile (>75th pct): {np.percentile(value_add, 75):.4f}
    - Concepts with value-add > 1.0: {np.sum(value_add > 1.0)} (LLM actively wrong)
""")

    sc.close()
    print("  Analysis complete.")


if __name__ == "__main__":
    run_analysis()
