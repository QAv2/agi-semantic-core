#!/usr/bin/env python3
"""
Phase 6b: MLP Projection — Nonlinear bridge from LLM embeddings to grounded semantic space.

Trains a small 2-layer MLP (384 -> 128 -> 14) with ReLU to project LLM embeddings
into the 14D grounded semantic space, and compares against the linear OLS baseline.

Key question: Does nonlinearity recover complement geometry (~90 degree relationships)
that linear projection compresses?

Usage:
    python3 -m tools.mlp_projection
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from api.semantic_core import SemanticCore

EXPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'exports')
EMBEDDINGS_PATH = os.path.join(EXPORT_DIR, 'llm_embeddings.npz')
PROJECTION_PATH = os.path.join(EXPORT_DIR, 'projection_matrix.npz')
MLP_OUTPUT_PATH = os.path.join(EXPORT_DIR, 'mlp_projection.npz')

DIM_LABELS = ['x', 'y', 'z', 'e', 'f', 'g', 'h', 'fx', 'fy', 'fz', 'fe', 'ff', 'fg', 'fh']


class SemanticMLP(nn.Module):
    """2-layer MLP: 384 -> 128 -> 14 with ReLU."""

    def __init__(self, d_in=384, d_hidden=128, d_out=14, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


def compute_r2(y_true, y_pred):
    """Compute R-squared (overall and per-dimension)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
    r2_overall = 1 - ss_res / ss_tot

    r2_per_dim = []
    for d in range(y_true.shape[1]):
        ss_res_d = np.sum((y_true[:, d] - y_pred[:, d]) ** 2)
        ss_tot_d = np.sum((y_true[:, d] - y_true[:, d].mean()) ** 2)
        r2_d = 1 - ss_res_d / ss_tot_d if ss_tot_d > 0 else 0.0
        r2_per_dim.append(r2_d)

    return r2_overall, np.array(r2_per_dim)


def compute_angle(v1, v2):
    """Angle in degrees between two vectors."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return None
    cos_sim = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
    return np.degrees(np.arccos(cos_sim))


def load_data():
    """Load LLM embeddings and grounded vectors, return aligned matrices."""
    # Load LLM embeddings
    emb_data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    names = list(emb_data['names'])
    embeddings = emb_data['embeddings']

    # Build grounded target vectors from database
    sc = SemanticCore()
    targets = []
    valid_indices = []
    for i, name in enumerate(names):
        concept = sc.get(name)
        if concept is not None:
            targets.append(concept.full_vector())
            valid_indices.append(i)
    sc.close()

    X = embeddings[valid_indices].astype(np.float32)
    Y = np.array(targets, dtype=np.float32)
    valid_names = [names[i] for i in valid_indices]

    return X, Y, valid_names


def load_complement_pairs(names):
    """Load complement relation pairs from database."""
    sc = SemanticCore()
    relations = sc.conn.execute("""
        SELECT c1.name as name1, c2.name as name2, r.angle_4d, r.angle_8d
        FROM relations r
        JOIN concepts c1 ON r.concept1_id = c1.id
        JOIN concepts c2 ON r.concept2_id = c2.id
        WHERE r.rel_type = 'complement'
    """).fetchall()
    sc.close()

    name_to_idx = {n: i for i, n in enumerate(names)}
    pairs = []
    for rel in relations:
        n1, n2 = rel['name1'], rel['name2']
        if n1 in name_to_idx and n2 in name_to_idx:
            true_angle = rel['angle_8d'] if rel['angle_8d'] else rel['angle_4d']
            if true_angle is not None:
                pairs.append({
                    'name1': n1,
                    'name2': n2,
                    'idx1': name_to_idx[n1],
                    'idx2': name_to_idx[n2],
                    'true_angle': true_angle,
                })
    return pairs


def load_all_relations(names):
    """Load all relations with angular constraints."""
    sc = SemanticCore()
    relations = sc.conn.execute("""
        SELECT c1.name as name1, c2.name as name2, r.rel_type, r.angle_4d, r.angle_8d
        FROM relations r
        JOIN concepts c1 ON r.concept1_id = c1.id
        JOIN concepts c2 ON r.concept2_id = c2.id
        ORDER BY r.rel_type
    """).fetchall()
    sc.close()

    name_to_idx = {n: i for i, n in enumerate(names)}
    by_type = {}
    for rel in relations:
        n1, n2, rtype = rel['name1'], rel['name2'], rel['rel_type']
        if n1 not in name_to_idx or n2 not in name_to_idx:
            continue
        true_angle = rel['angle_8d'] if rel['angle_8d'] else rel['angle_4d']
        if true_angle is None:
            continue
        if rtype not in by_type:
            by_type[rtype] = []
        by_type[rtype].append({
            'name1': n1,
            'name2': n2,
            'idx1': name_to_idx[n1],
            'idx2': name_to_idx[n2],
            'true_angle': true_angle,
        })
    return by_type


def train_mlp(X_train, Y_train, X_test, Y_test, epochs=500, lr=1e-3, batch_size=128,
              weight_decay=1e-4, patience=50, verbose=True):
    """Train the MLP with early stopping."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SemanticMLP(
        d_in=X_train.shape[1],
        d_hidden=128,
        d_out=Y_train.shape[1],
        dropout=0.1,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    criterion = nn.MSELoss()

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_t, Y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_test_loss = float('inf')
    best_state = None
    epochs_no_improve = 0

    t0 = time.time()

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1
        train_loss = train_loss_sum / n_batches

        # Evaluate
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test_t)
            test_loss = criterion(test_pred, Y_test_t).item()

        scheduler.step(test_loss)

        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1:4d}: train_loss={train_loss:.6f}  test_loss={test_loss:.6f}  "
                  f"best={best_test_loss:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    elapsed = time.time() - t0

    # Load best model
    model.load_state_dict(best_state)
    model.eval()

    if verbose:
        print(f"    Training complete in {elapsed:.1f}s ({epoch+1} epochs)")

    # Generate final predictions
    with torch.no_grad():
        Y_train_pred = model(X_train_t).cpu().numpy()
        Y_test_pred = model(X_test_t).cpu().numpy()

    return model, Y_train_pred, Y_test_pred


def main():
    print("=" * 70)
    print("AGI SEMANTIC CORE -- PHASE 6b: MLP NONLINEAR PROJECTION")
    print("=" * 70)

    # ─── Load data ───
    print("\n[1] Loading data...")
    X, Y, names = load_data()
    print(f"    Concepts: {len(names)}")
    print(f"    Input dim:  {X.shape[1]} (LLM embedding)")
    print(f"    Output dim: {Y.shape[1]} (grounded semantic)")

    # ─── Load linear baseline ───
    print("\n[2] Loading linear baseline...")
    lin_data = np.load(PROJECTION_PATH, allow_pickle=True)
    lin_r2_per_dim = lin_data['r2_per_dim']
    lin_r2_overall = float(lin_data['r2'])
    lin_Y_pred = lin_data['Y_pred']  # Full dataset predictions
    print(f"    Linear OLS R2 (overall): {lin_r2_overall:.4f}")

    # ─── Train/test split (80/20, deterministic) ───
    print("\n[3] Preparing train/test split (80/20)...")
    np.random.seed(42)
    N = X.shape[0]
    indices = np.random.permutation(N)
    split = int(0.8 * N)
    train_idx = indices[:split]
    test_idx = indices[split:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    print(f"    Train: {len(train_idx)} concepts")
    print(f"    Test:  {len(test_idx)} concepts")

    # ─── Train MLP ───
    print("\n[4] Training MLP (384 -> 128 -> 14, ReLU, dropout=0.1)...")
    torch.manual_seed(42)
    model, Y_train_pred, Y_test_pred = train_mlp(
        X_train, Y_train, X_test, Y_test,
        epochs=1000, lr=1e-3, batch_size=128,
        weight_decay=1e-4, patience=80,
    )

    # ─── Overall R2 ───
    print("\n[5] R2 Evaluation")
    print("=" * 70)

    r2_train, r2_train_per_dim = compute_r2(Y_train, Y_train_pred)
    r2_test, r2_test_per_dim = compute_r2(Y_test, Y_test_pred)

    # Also compute linear baseline on same split for fair comparison
    W_lin = lin_data['W']
    lin_train_pred = X_train @ W_lin
    lin_test_pred = X_test @ W_lin
    lin_r2_train, lin_r2_train_per_dim = compute_r2(Y_train, lin_train_pred)
    lin_r2_test, lin_r2_test_per_dim = compute_r2(Y_test, lin_test_pred)

    print(f"\n  {'':20s} {'LINEAR OLS':>12s}  {'MLP':>12s}  {'Delta':>8s}")
    print(f"  {'':20s} {'----------':>12s}  {'---':>12s}  {'-----':>8s}")
    print(f"  {'R2 (train)':20s} {lin_r2_train:12.4f}  {r2_train:12.4f}  {r2_train - lin_r2_train:+8.4f}")
    print(f"  {'R2 (test)':20s} {lin_r2_test:12.4f}  {r2_test:12.4f}  {r2_test - lin_r2_test:+8.4f}")

    # ─── Per-dimension R2 comparison ───
    print(f"\n  Per-dimension R2 (TEST SET):")
    print(f"  {'Dim':>4s}  {'Linear':>8s}  {'MLP':>8s}  {'Delta':>8s}  {'Bar'}")
    print(f"  {'----':>4s}  {'------':>8s}  {'---':>8s}  {'-----':>8s}  {'---'}")

    for d, label in enumerate(DIM_LABELS):
        lin_val = lin_r2_test_per_dim[d]
        mlp_val = r2_test_per_dim[d]
        delta = mlp_val - lin_val
        bar_lin = int(max(0, lin_val) * 25)
        bar_mlp = int(max(0, mlp_val) * 25)
        indicator = "+" if delta > 0.01 else ("-" if delta < -0.01 else "=")
        print(f"  {label:>4s}  {lin_val:8.4f}  {mlp_val:8.4f}  {delta:+8.4f}  "
              f"L|{'#' * bar_lin}{'.' * (25 - bar_lin)}| "
              f"M|{'#' * bar_mlp}{'.' * (25 - bar_mlp)}| {indicator}")

    # ─── Generate full-dataset MLP predictions for angular analysis ───
    print("\n[6] Generating full-dataset predictions for angular analysis...")
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_all_t = torch.tensor(X, dtype=torch.float32).to(device)
        Y_pred_all = model(X_all_t).cpu().numpy()

    # ─── Angular preservation analysis ───
    print("\n[7] Angular Preservation Analysis")
    print("=" * 70)

    relations_by_type = load_all_relations(names)

    for rtype in ['synonym', 'affinity', 'complement', 'opposition']:
        if rtype not in relations_by_type:
            continue
        pairs = relations_by_type[rtype]

        lin_errors = []
        mlp_errors = []
        lin_angles = []
        mlp_angles = []
        true_angles = []

        for p in pairs:
            i, j = p['idx1'], p['idx2']
            ta = p['true_angle']
            true_angles.append(ta)

            # Linear projection angle
            la = compute_angle(lin_Y_pred[i], lin_Y_pred[j])
            if la is not None:
                lin_angles.append(la)
                lin_errors.append(abs(la - ta))
            else:
                lin_angles.append(np.nan)
                lin_errors.append(np.nan)

            # MLP projection angle
            ma = compute_angle(Y_pred_all[i], Y_pred_all[j])
            if ma is not None:
                mlp_angles.append(ma)
                mlp_errors.append(abs(ma - ta))
            else:
                mlp_angles.append(np.nan)
                mlp_errors.append(np.nan)

        lin_errors = np.array(lin_errors)
        mlp_errors = np.array(mlp_errors)
        lin_angles = np.array(lin_angles)
        mlp_angles = np.array(mlp_angles)
        true_angles = np.array(true_angles)

        valid = ~np.isnan(lin_errors) & ~np.isnan(mlp_errors)
        le = lin_errors[valid]
        me = mlp_errors[valid]
        la = lin_angles[valid]
        ma = mlp_angles[valid]
        ta = true_angles[valid]

        print(f"\n  {rtype.upper()} ({len(le)} pairs)")
        print(f"  {'':25s} {'Linear':>10s}  {'MLP':>10s}  {'Delta':>10s}")
        print(f"  {'Mean angle error':25s} {np.mean(le):10.1f}  {np.mean(me):10.1f}  {np.mean(me)-np.mean(le):+10.1f}")
        print(f"  {'Median angle error':25s} {np.median(le):10.1f}  {np.median(me):10.1f}  {np.median(me)-np.median(le):+10.1f}")
        print(f"  {'Std angle error':25s} {np.std(le):10.1f}  {np.std(me):10.1f}  {np.std(me)-np.std(le):+10.1f}")
        print(f"  {'Max angle error':25s} {np.max(le):10.1f}  {np.max(me):10.1f}  {np.max(me)-np.max(le):+10.1f}")

        if len(ta) > 2:
            corr_lin = np.corrcoef(ta, la)[0, 1]
            corr_mlp = np.corrcoef(ta, ma)[0, 1]
            print(f"  {'Angle correlation':25s} {corr_lin:10.4f}  {corr_mlp:10.4f}  {corr_mlp-corr_lin:+10.4f}")

    # ─── Complement geometry deep dive ───
    print(f"\n{'=' * 70}")
    print("COMPLEMENT GEOMETRY DEEP DIVE (the key test)")
    print(f"{'=' * 70}")

    comp_pairs = load_complement_pairs(names)
    name_to_idx = {n: i for i, n in enumerate(names)}

    # Angle distribution analysis
    lin_comp_angles = []
    mlp_comp_angles = []
    true_comp_angles = []
    pair_details = []

    for p in comp_pairs:
        i, j = p['idx1'], p['idx2']
        ta = p['true_angle']
        la = compute_angle(lin_Y_pred[i], lin_Y_pred[j])
        ma = compute_angle(Y_pred_all[i], Y_pred_all[j])
        if la is not None and ma is not None:
            true_comp_angles.append(ta)
            lin_comp_angles.append(la)
            mlp_comp_angles.append(ma)
            pair_details.append({
                'pair': f"{p['name1']} -- {p['name2']}",
                'true': ta,
                'linear': la,
                'mlp': ma,
                'lin_err': abs(la - ta),
                'mlp_err': abs(ma - ta),
            })

    true_comp_angles = np.array(true_comp_angles)
    lin_comp_angles = np.array(lin_comp_angles)
    mlp_comp_angles = np.array(mlp_comp_angles)

    # How many complements are within 15 degrees of their true angle?
    lin_within_15 = np.sum(np.abs(lin_comp_angles - true_comp_angles) < 15)
    mlp_within_15 = np.sum(np.abs(mlp_comp_angles - true_comp_angles) < 15)
    lin_within_30 = np.sum(np.abs(lin_comp_angles - true_comp_angles) < 30)
    mlp_within_30 = np.sum(np.abs(mlp_comp_angles - true_comp_angles) < 30)

    total = len(true_comp_angles)
    print(f"\n  Complement pairs analyzed: {total}")
    print(f"\n  True angle range: [{np.min(true_comp_angles):.1f}, {np.max(true_comp_angles):.1f}] degrees")
    print(f"  True angle mean:  {np.mean(true_comp_angles):.1f} degrees")
    print(f"\n  Angle preservation accuracy:")
    print(f"    {'Metric':30s} {'Linear':>10s}  {'MLP':>10s}")
    print(f"    {'------':30s} {'------':>10s}  {'---':>10s}")
    print(f"    {'Within 15 deg of true':30s} {lin_within_15:>7d}/{total}  {mlp_within_15:>7d}/{total}")
    print(f"    {'Within 30 deg of true':30s} {lin_within_30:>7d}/{total}  {mlp_within_30:>7d}/{total}")
    print(f"    {'Mean projected angle':30s} {np.mean(lin_comp_angles):10.1f}  {np.mean(mlp_comp_angles):10.1f}")
    print(f"    {'Std projected angle':30s} {np.std(lin_comp_angles):10.1f}  {np.std(mlp_comp_angles):10.1f}")

    # Are MLP angles more spread (less compressed toward center)?
    lin_spread = np.std(lin_comp_angles)
    mlp_spread = np.std(mlp_comp_angles)
    true_spread = np.std(true_comp_angles)
    print(f"\n  Angle spread (std) -- true: {true_spread:.1f}, linear: {lin_spread:.1f}, MLP: {mlp_spread:.1f}")
    print(f"  Compression ratio -- linear: {lin_spread/true_spread:.3f}, MLP: {mlp_spread/true_spread:.3f}")
    print(f"  (1.0 = no compression, <1.0 = angles squeezed toward mean)")

    # Show best and worst MLP-improved pairs
    pair_details.sort(key=lambda p: p['mlp_err'] - p['lin_err'])
    print(f"\n  Top 10 pairs where MLP MOST IMPROVES over linear:")
    print(f"    {'Pair':40s} {'True':>6s} {'Lin':>6s} {'MLP':>6s} {'LinErr':>7s} {'MLPErr':>7s} {'Improv':>7s}")
    for p in pair_details[:10]:
        print(f"    {p['pair']:40s} {p['true']:6.1f} {p['linear']:6.1f} {p['mlp']:6.1f} "
              f"{p['lin_err']:7.1f} {p['mlp_err']:7.1f} {p['lin_err']-p['mlp_err']:+7.1f}")

    print(f"\n  Top 10 pairs where MLP MOST DEGRADES vs linear:")
    for p in pair_details[-10:]:
        print(f"    {p['pair']:40s} {p['true']:6.1f} {p['linear']:6.1f} {p['mlp']:6.1f} "
              f"{p['lin_err']:7.1f} {p['mlp_err']:7.1f} {p['lin_err']-p['mlp_err']:+7.1f}")

    # ─── Canonical complement pairs (the "poster children") ───
    print(f"\n  Canonical complement pairs:")
    canonical = [
        ('HOT', 'COLD'), ('LIGHT', 'DARK'), ('UP', 'DOWN'), ('IN', 'OUT'),
        ('YANG', 'YIN'), ('LOVE', 'FEAR'), ('ORDER', 'CHAOS'),
        ('LIFE', 'DEATH'), ('CREATION', 'DESTRUCTION'), ('MIND', 'BODY'),
    ]
    print(f"    {'Pair':30s} {'True':>6s} {'Lin':>6s} {'MLP':>6s} {'LinErr':>7s} {'MLPErr':>7s}")
    for n1, n2 in canonical:
        if n1 in name_to_idx and n2 in name_to_idx:
            i, j = name_to_idx[n1], name_to_idx[n2]
            # Find true angle
            ta = None
            for p in comp_pairs:
                if (p['name1'] == n1 and p['name2'] == n2) or (p['name1'] == n2 and p['name2'] == n1):
                    ta = p['true_angle']
                    break
            if ta is None:
                continue
            la = compute_angle(lin_Y_pred[i], lin_Y_pred[j])
            ma = compute_angle(Y_pred_all[i], Y_pred_all[j])
            if la is not None and ma is not None:
                print(f"    {n1+' -- '+n2:30s} {ta:6.1f} {la:6.1f} {ma:6.1f} "
                      f"{abs(la-ta):7.1f} {abs(ma-ta):7.1f}")

    # ─── Save outputs ───
    print(f"\n{'=' * 70}")
    print("SAVING RESULTS")
    print(f"{'=' * 70}")

    # Save model state dict as numpy-compatible format
    model_state = {f"model_{k}": v.cpu().numpy() for k, v in model.state_dict().items()}

    np.savez(MLP_OUTPUT_PATH,
             # Predictions
             Y_pred_all=Y_pred_all,
             Y_true=Y,
             names=np.array(names),
             # Train/test split
             train_idx=train_idx,
             test_idx=test_idx,
             Y_train_pred=Y_train_pred,
             Y_test_pred=Y_test_pred,
             # Metrics
             r2_train=np.array(r2_train),
             r2_test=np.array(r2_test),
             r2_train_per_dim=r2_train_per_dim,
             r2_test_per_dim=r2_test_per_dim,
             # Architecture info
             architecture=np.array("384->128->14, ReLU, dropout=0.1"),
             # Model weights
             **model_state,
             )

    file_size = os.path.getsize(MLP_OUTPUT_PATH) / 1024
    print(f"  Saved to {MLP_OUTPUT_PATH} ({file_size:.0f} KB)")

    # ─── Summary ───
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Linear OLS R2 (test):  {lin_r2_test:.4f}")
    print(f"  MLP R2 (test):         {r2_test:.4f}")
    print(f"  Delta:                 {r2_test - lin_r2_test:+.4f}")
    print(f"")
    print(f"  Complement mean error: linear={np.mean(np.abs(lin_comp_angles - true_comp_angles)):.1f} deg, "
          f"MLP={np.mean(np.abs(mlp_comp_angles - true_comp_angles)):.1f} deg")
    print(f"  Complement within 15 deg: linear={lin_within_15}/{total}, MLP={mlp_within_15}/{total}")
    print(f"  Complement compression: linear={lin_spread/true_spread:.3f}, MLP={mlp_spread/true_spread:.3f}")

    if r2_test > lin_r2_test:
        gain_pct = (r2_test - lin_r2_test) / lin_r2_test * 100
        print(f"\n  MLP improves overall R2 by {gain_pct:.1f}% over linear baseline.")
    else:
        print(f"\n  MLP does not improve overall R2 over linear baseline.")

    comp_improvement = np.mean(np.abs(lin_comp_angles - true_comp_angles)) - np.mean(np.abs(mlp_comp_angles - true_comp_angles))
    if comp_improvement > 0:
        print(f"  MLP reduces complement angle error by {comp_improvement:.1f} degrees on average.")
    else:
        print(f"  MLP does not reduce complement angle error (delta: {comp_improvement:+.1f} deg).")


if __name__ == "__main__":
    main()
