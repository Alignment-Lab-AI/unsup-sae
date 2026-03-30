"""
Visualization and benchmarking plots for unsupsae.

Generates:
1. Method comparison bar chart (probe accuracy by method)
2. Signal contribution heatmap (per-feature signal breakdown)
3. Feature overlap matrix between methods
4. Score distributions: F-ratio strip plot + per-rank probe accuracy
5. Lift curve: cumulative probe accuracy as features are added
6. 3D signal space: features embedded in their signal coordinates
7. Gap analysis: what our method gets right/wrong vs oracle
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")


def _ensure_figdir():
    os.makedirs(FIGDIR, exist_ok=True)


def _savefig(fig, name):
    _ensure_figdir()
    fig.savefig(os.path.join(FIGDIR, name), dpi=150, bbox_inches='tight')
    print(f"  saved figures/{name}")


def plot_method_comparison(eval_results, save=True):
    """Bar chart comparing probe accuracy across methods."""
    methods = eval_results['methods']
    n_classes = eval_results['n_classes']
    chance = 1.0 / n_classes

    names, acc10, acc20 = [], [], []
    for key in ['ours', 'naive', 'variance', 'selectivity', 'oracle_coef', 'oracle_f']:
        m = methods[key]
        names.append(m['name'].replace('Baseline: ', '').replace('Oracle: ', ''))
        acc10.append(np.mean(m['probes'][:10]))
        acc20.append(np.mean(m['probes']))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w/2, acc10, w, label='Top-10 mean', color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x + w/2, acc20, w, label='Top-20 mean', color='#FF9800', alpha=0.85)
    ax.axhline(y=chance, color='gray', linestyle='--', linewidth=1, label=f'Chance ({chance:.2f})')

    # Show oracle lift percentages
    oracle_10 = np.mean(methods['oracle_f']['probes'][:10])
    lift_pct = (acc10[0] - chance) / (oracle_10 - chance) * 100
    ax.annotate(f'{lift_pct:.0f}% of\noracle lift', xy=(0, acc10[0]),
               xytext=(-0.6, acc10[0] + 0.015), fontsize=8, color='#2196F3',
               fontweight='bold', ha='center')

    ax.set_ylabel('Single-feature probe accuracy (5-fold CV)')
    ax.set_title('Unsupervised Feature Discovery vs Baselines & Oracles')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(chance - 0.03, max(acc10 + acc20) * 1.08)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=7)

    plt.tight_layout()
    if save:
        _savefig(fig, 'method_comparison.png')
    return fig


def plot_signal_heatmap(results, eval_results, save=True):
    """Heatmap of per-signal scores for top features (ours + oracle)."""
    all_signals = results.get('all_signals', {})
    if not all_signals:
        return None

    our_fids = eval_results['methods']['ours']['features'][:10]
    oracle_fids = eval_results['methods']['oracle_f']['features'][:10]

    all_fids = []
    seen = set()
    for fid in our_fids + oracle_fids:
        if fid not in seen and fid in all_signals:
            all_fids.append(fid)
            seen.add(fid)

    signal_names = ['stab', 'proc', 'msim_var', 'sel', 'pc', 'gi', 'freq_w', 'cvs']
    display_names = ['kNN\nstability', 'Procrustes', 'Manifold\nselect.', 'Activation\nselect.',
                     'PC\nalign.', 'Graph\ncentrality', 'Frequency', 'CV\nstability']

    matrix = np.zeros((len(all_fids), len(signal_names)))
    for i, fid in enumerate(all_fids):
        s = all_signals.get(fid, {})
        for j, sn in enumerate(signal_names):
            matrix[i, j] = s.get(sn, 0.0)

    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        mn, mx = col.min(), col.max()
        if mx - mn > 1e-10:
            matrix[:, j] = (col - mn) / (mx - mn)

    fig, ax = plt.subplots(figsize=(10, max(6, len(all_fids) * 0.35)))

    our_set = set(our_fids)
    oracle_set = set(oracle_fids)
    row_labels, row_colors = [], []
    for fid in all_fids:
        in_ours = fid in our_set
        in_oracle = fid in oracle_set
        if in_ours and in_oracle:
            row_labels.append(f'{fid} (both)')
            row_colors.append('#4CAF50')
        elif in_ours:
            row_labels.append(f'{fid} (ours)')
            row_colors.append('#2196F3')
        else:
            row_labels.append(f'{fid} (oracle)')
            row_colors.append('#F44336')

    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xticks(range(len(display_names)))
    ax.set_xticklabels(display_names, fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i, color in enumerate(row_colors):
        ax.get_yticklabels()[i].set_color(color)

    ax.set_title('Signal Breakdown: Our Top-10 vs Oracle Top-10\n'
                 '(blue=ours, red=oracle, green=both)', fontsize=10)
    plt.colorbar(im, ax=ax, label='Normalized signal value', shrink=0.8)

    all_f = eval_results['all_f_ratios']
    probes_dict = {}
    for key in ['ours', 'oracle_f']:
        for fid, p in zip(eval_results['methods'][key]['features'],
                         eval_results['methods'][key]['probes']):
            probes_dict[fid] = p

    for i, fid in enumerate(all_fids):
        f_val = all_f.get(fid, 0)
        p_val = probes_dict.get(fid, 0)
        ax.text(matrix.shape[1] + 0.3, i, f'F={f_val:.0f}  p={p_val:.2f}',
               va='center', fontsize=7)

    plt.tight_layout()
    if save:
        _savefig(fig, 'signal_heatmap.png')
    return fig


def plot_overlap_matrix(eval_results, save=True):
    """Overlap matrix between all methods' top-20 feature sets."""
    methods = eval_results['methods']
    keys = ['ours', 'naive', 'variance', 'selectivity', 'oracle_coef', 'oracle_f']
    names = [methods[k]['name'].replace('Baseline: ', '').replace('Oracle: ', '') for k in keys]
    sets = [set(methods[k]['features'][:20]) for k in keys]

    n = len(keys)
    overlap = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            overlap[i, j] = len(sets[i] & sets[j])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(overlap, cmap='Blues', interpolation='nearest')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{int(overlap[i,j])}', ha='center', va='center',
                   fontsize=10, color='white' if overlap[i,j] > 10 else 'black')
    ax.set_title('Top-20 Feature Overlap Between Methods')
    plt.colorbar(im, ax=ax, label='Shared features', shrink=0.8)
    plt.tight_layout()
    if save:
        _savefig(fig, 'overlap_matrix.png')
    return fig


def plot_score_distributions(eval_results, save=True):
    """Strip plot of F-ratios + smoothed per-rank probe accuracy."""
    methods = eval_results['methods']
    all_f = eval_results['all_f_ratios']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: strip/swarm plot of F-ratios by method
    ax = axes[0]
    method_keys = ['ours', 'selectivity', 'oracle_coef', 'oracle_f']
    method_labels = ['Ours', 'Selectivity', 'Oracle coef', 'Oracle F']
    colors = ['#2196F3', '#9C27B0', '#FF9800', '#F44336']

    # Background: all active features as gray violin/strip
    all_f_vals = np.array(list(all_f.values()))
    all_f_log = np.log1p(all_f_vals[all_f_vals > 0.1])
    parts = ax.violinplot([all_f_log], positions=[0], showmedians=True, widths=0.6)
    for pc in parts['bodies']:
        pc.set_facecolor('lightgray')
        pc.set_alpha(0.4)
    for key in ['cmins', 'cmaxes', 'cbars', 'cmedians']:
        if key in parts:
            parts[key].set_color('gray')

    # Each method's features as colored dots
    for i, (key, label, color) in enumerate(zip(method_keys, method_labels, colors)):
        fids = methods[key]['features'][:20]
        vals = np.log1p(np.array([all_f.get(f, 0) for f in fids]))
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals, c=color,
                  s=30, alpha=0.7, edgecolors='black', linewidths=0.3,
                  label=label, zorder=3)
        ax.plot([i + 1 - 0.2, i + 1 + 0.2], [np.median(vals)] * 2,
               color=color, linewidth=2, zorder=4)

    ax.set_xticks([0] + list(range(1, len(method_keys) + 1)))
    ax.set_xticklabels(['All\nactive'] + method_labels, fontsize=8)
    ax.set_ylabel('log(1 + F-ratio)')
    ax.set_title('F-ratio by Method (higher = more discriminative)')
    ax.legend(fontsize=7, loc='upper left')

    # Right: per-rank probe accuracy with rolling mean
    ax = axes[1]
    for key, color, label in zip(method_keys, colors, method_labels):
        probes = np.array(methods[key]['probes'])
        ranks = np.arange(1, len(probes) + 1)
        ax.scatter(ranks, probes, c=color, s=25, alpha=0.4, zorder=2)
        # Rolling mean (window=3)
        if len(probes) >= 3:
            kernel = np.ones(3) / 3
            smoothed = np.convolve(probes, kernel, mode='valid')
            ax.plot(ranks[1:-1], smoothed, color=color, linewidth=2, label=label,
                   alpha=0.85, zorder=3)
        else:
            ax.plot(ranks, probes, color=color, linewidth=2, label=label, alpha=0.85)

    chance = 1.0 / eval_results['n_classes']
    ax.axhline(y=chance, color='gray', linestyle='--', linewidth=1, label='Chance')
    ax.set_xlabel('Feature rank')
    ax.set_ylabel('Single-feature probe accuracy')
    ax.set_title('Per-Feature Probe Accuracy by Rank (smoothed)')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(chance - 0.02, None)

    plt.tight_layout()
    if save:
        _savefig(fig, 'score_distributions.png')
    return fig


def plot_lift_curve(eval_results, save=True):
    """Cumulative lift curve with shaded regions showing advantage over baselines."""
    methods = eval_results['methods']
    n_classes = eval_results['n_classes']
    chance = 1.0 / n_classes

    fig, ax = plt.subplots(figsize=(9, 5.5))

    curves = {}
    for key, color, label, ls in [
        ('ours', '#2196F3', 'Ours (unsupervised)', '-'),
        ('selectivity', '#9C27B0', 'Selectivity baseline', '--'),
        ('oracle_coef', '#FF9800', 'Oracle: classifier coef', '-.'),
        ('oracle_f', '#F44336', 'Oracle: F-ratio', ':'),
    ]:
        probes = np.array(methods[key]['probes'])
        cumulative = np.cumsum(probes) / np.arange(1, len(probes) + 1)
        ranks = np.arange(1, len(probes) + 1)
        ax.plot(ranks, cumulative, ls, color=color, label=label, linewidth=2.5, alpha=0.85)
        curves[key] = cumulative

    # Shade the lift we achieve over selectivity baseline
    if 'ours' in curves and 'selectivity' in curves:
        r = np.arange(1, min(len(curves['ours']), len(curves['selectivity'])) + 1)
        ax.fill_between(r, curves['selectivity'][:len(r)], curves['ours'][:len(r)],
                       alpha=0.15, color='#2196F3', label='Lift over selectivity')

    ax.axhline(y=chance, color='gray', linestyle='--', linewidth=1, label='Chance')

    # Annotate convergence point
    if 'ours' in curves and 'oracle_coef' in curves:
        ours_20 = curves['ours'][-1]
        oracle_20 = curves['oracle_coef'][-1]
        ax.annotate(f'Gap: {oracle_20 - ours_20:.3f}',
                   xy=(20, (ours_20 + oracle_20) / 2), fontsize=8, color='gray',
                   ha='right')

    ax.set_xlabel('Number of features (ranked by method)')
    ax.set_ylabel('Cumulative mean probe accuracy')
    ax.set_title('Feature Discovery Lift Curve')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(1, 20)
    ax.set_ylim(chance - 0.02, None)

    plt.tight_layout()
    if save:
        _savefig(fig, 'lift_curve.png')
    return fig


def plot_signal_space_3d(results, eval_results, save=True):
    """
    3D scatter of features in signal space (PCA of signal vectors).
    Shows where our picks live vs oracle picks vs all active features.
    """
    all_signals = results.get('all_signals', {})
    if not all_signals or len(all_signals) < 10:
        return None

    signal_names = ['stab', 'proc', 'msim_var', 'sel', 'pc', 'gi', 'freq_w', 'cvs']
    fids = list(all_signals.keys())
    X = np.array([[all_signals[f].get(s, 0) for s in signal_names] for f in fids])

    # Normalize each signal to [0,1]
    for j in range(X.shape[1]):
        mn, mx = X[:, j].min(), X[:, j].max()
        if mx - mn > 1e-10:
            X[:, j] = (X[:, j] - mn) / (mx - mn)

    # PCA to 3D
    X_c = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    coords = X_c @ Vt[:3].T  # (n_features, 3)
    explained = (S[:3] ** 2) / (S ** 2).sum()

    our_set = set(eval_results['methods']['ours']['features'][:20])
    oracle_set = set(eval_results['methods']['oracle_f']['features'][:20])
    all_f = eval_results['all_f_ratios']

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')

    # All features (gray, small)
    f_vals = np.array([all_f.get(f, 0) for f in fids])
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
              c='lightgray', s=5, alpha=0.2, zorder=1)

    # Our picks (blue)
    our_mask = np.array([f in our_set for f in fids])
    if our_mask.any():
        ax.scatter(coords[our_mask, 0], coords[our_mask, 1], coords[our_mask, 2],
                  c='#2196F3', s=60, alpha=0.8, edgecolors='black', linewidths=0.5,
                  label='Ours', zorder=3)
        for i, fid in enumerate(fids):
            if fid in our_set:
                ax.text(coords[i, 0], coords[i, 1], coords[i, 2],
                       f' {fid}', fontsize=6, color='#2196F3')

    # Oracle picks (red)
    oracle_mask = np.array([f in oracle_set and f not in our_set for f in fids])
    if oracle_mask.any():
        ax.scatter(coords[oracle_mask, 0], coords[oracle_mask, 1], coords[oracle_mask, 2],
                  c='#F44336', s=60, alpha=0.8, edgecolors='black', linewidths=0.5,
                  label='Oracle only', zorder=3)
        for i, fid in enumerate(fids):
            if fid in oracle_set and fid not in our_set:
                ax.text(coords[i, 0], coords[i, 1], coords[i, 2],
                       f' {fid}', fontsize=6, color='#F44336')

    # Both (green)
    both_mask = np.array([f in our_set and f in oracle_set for f in fids])
    if both_mask.any():
        ax.scatter(coords[both_mask, 0], coords[both_mask, 1], coords[both_mask, 2],
                  c='#4CAF50', s=80, alpha=0.9, edgecolors='black', linewidths=0.5,
                  label='Both', zorder=4)

    ax.set_xlabel(f'Signal PC1 ({explained[0]:.0%} var)')
    ax.set_ylabel(f'Signal PC2 ({explained[1]:.0%} var)')
    ax.set_zlabel(f'Signal PC3 ({explained[2]:.0%} var)')
    ax.set_title('Features in Signal Space (PCA of 8 scoring signals)\n'
                 'Blue=ours, Red=oracle-only, Green=both')
    ax.legend(fontsize=8, loc='upper left')

    # Also annotate the PCA loadings
    loading_text = 'PC loadings:\n'
    for k, pc in enumerate(Vt[:3]):
        top2 = np.argsort(np.abs(pc))[-2:][::-1]
        loading_text += f'  PC{k+1}: {signal_names[top2[0]]} ({pc[top2[0]]:+.2f}), {signal_names[top2[1]]} ({pc[top2[1]]:+.2f})\n'
    fig.text(0.02, 0.02, loading_text, fontsize=7, family='monospace',
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if save:
        _savefig(fig, 'signal_space_3d.png')
    return fig


def plot_gap_analysis(results, eval_results, save=True):
    """
    Gap analysis: for each oracle feature, show why our method ranked it
    where it did. Horizontal bar chart of signal contributions.
    """
    all_signals = results.get('all_signals', {})
    if not all_signals:
        return None

    oracle_fids = eval_results['methods']['oracle_f']['features'][:10]
    our_fids = eval_results['methods']['ours']['features'][:10]
    all_f = eval_results['all_f_ratios']

    # Show oracle features ordered by F-ratio
    fids = oracle_fids
    signal_names = ['stab', 'proc', 'msim_var', 'sel', 'pc', 'gi', 'freq_w', 'cvs']
    display = ['kNN stab', 'Procrustes', 'Manifold sel', 'Act. sel', 'PC align', 'Graph cent', 'Freq', 'CV stab']

    # Compute percentile rank of each oracle feature among all eligible features
    all_vals = {s: np.array([all_signals[f].get(s, 0) for f in all_signals]) for s in signal_names}

    fig, ax = plt.subplots(figsize=(12, 6))

    n_feats = len([f for f in fids if f in all_signals])
    bar_height = 0.08
    group_height = len(signal_names) * bar_height + 0.15
    colors = plt.cm.Set2(np.linspace(0, 1, len(signal_names)))

    our_set = set(our_fids)
    y_positions = []
    y_labels = []

    feat_i = 0
    for fid in fids:
        if fid not in all_signals:
            continue
        s = all_signals[fid]
        y_base = -feat_i * group_height
        y_positions.append(y_base)

        in_ours = fid in our_set
        label_color = '#4CAF50' if in_ours else '#F44336'
        f_val = all_f.get(fid, 0)
        probe = eval_results['methods']['oracle_f']['probes'][oracle_fids.index(fid)] if fid in oracle_fids else 0
        y_labels.append((y_base, fid, f_val, probe, label_color, in_ours))

        for j, sn in enumerate(signal_names):
            val = s.get(sn, 0)
            pctile = (all_vals[sn] < val).mean()  # percentile among all
            y = y_base - j * bar_height
            bar = ax.barh(y, pctile, height=bar_height * 0.85, color=colors[j],
                         alpha=0.8, edgecolor='white', linewidth=0.5)
            if feat_i == 0:
                ax.barh(y, 0, label=display[j])  # for legend

        feat_i += 1

    # Labels
    for y_base, fid, f_val, probe, color, in_ours in y_labels:
        tag = ' (found)' if in_ours else ' (missed)'
        ax.text(-0.02, y_base - len(signal_names) * bar_height / 2,
               f'{fid}{tag}\nF={f_val:.0f} p={probe:.2f}',
               ha='right', va='center', fontsize=7, color=color, fontweight='bold')

    ax.set_xlim(-0.02, 1.05)
    ax.set_xlabel('Percentile rank among all eligible features (higher = better)')
    ax.set_title('Gap Analysis: Oracle Top-10 Feature Signal Percentiles\n'
                 '(green = found by our method, red = missed)')
    ax.set_yticks([])
    ax.legend(fontsize=7, loc='lower right', ncol=2)
    ax.axvline(x=0.75, color='gray', linestyle=':', alpha=0.5, label='75th pctile')

    plt.tight_layout()
    if save:
        _savefig(fig, 'gap_analysis.png')
    return fig


def plot_compositional_accuracy(eval_results, save=True):
    """
    Bar chart: multi-feature classifier accuracy using our top-K vs oracle top-K.
    This tests whether selected features compositionally carry discriminative info.
    """
    comp = eval_results.get('compositional', {})
    if not comp:
        return None

    n_classes = eval_results['n_classes']
    chance = 1.0 / n_classes

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ks = [5, 10, 20]
    method_keys = ['ours', 'oracle_coef', 'oracle_f']
    labels = ['Ours (unsupervised)', 'Oracle: classifier coef', 'Oracle: F-ratio']
    colors = ['#2196F3', '#FF9800', '#F44336']

    x = np.arange(len(ks))
    n_methods = len(method_keys)
    w = 0.22

    for i, (mk, label, color) in enumerate(zip(method_keys, labels, colors)):
        vals = [comp.get((mk, k), 0) for k in ks]
        bars = ax.bar(x + (i - n_methods/2 + 0.5) * w, vals, w,
                     label=label, color=color, alpha=0.85, edgecolor='white')
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8)

    # All active features line
    all_acc = [v for (k, _), v in comp.items() if k == 'all_active']
    if all_acc:
        ax.axhline(y=all_acc[0], color='green', linestyle='-', linewidth=1.5,
                  alpha=0.7, label=f'All active ({all_acc[0]:.3f})')

    ax.axhline(y=chance, color='gray', linestyle='--', linewidth=1, label=f'Chance ({chance:.2f})')

    # Full model accuracy
    full_acc = eval_results.get('full_cv_acc', 0)
    if full_acc > 0:
        ax.axhline(y=full_acc, color='black', linestyle=':', linewidth=1,
                  alpha=0.5, label=f'Full model ({full_acc:.3f})')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Top-{k}' for k in ks], fontsize=11)
    ax.set_ylabel('Multi-feature classifier accuracy (5-fold CV)')
    ax.set_title('Compositional Accuracy: Can Selected Features Reconstruct the Classifier?')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_ylim(chance - 0.03, min(1.0, max(full_acc, max(comp.values())) * 1.08))

    plt.tight_layout()
    if save:
        _savefig(fig, 'compositional_accuracy.png')
    return fig


def make_all_figures(eval_results, results, graph, engine, feature_acts, labels):
    """Generate all visualization figures."""
    _ensure_figdir()
    print("\n[viz] generating figures...")

    plot_method_comparison(eval_results)
    plot_signal_heatmap(results, eval_results)
    plot_overlap_matrix(eval_results)
    plot_score_distributions(eval_results)
    plot_lift_curve(eval_results)
    plot_signal_space_3d(results, eval_results)
    plot_gap_analysis(results, eval_results)
    plot_compositional_accuracy(eval_results)

    print(f"[viz] all figures saved to {FIGDIR}/")
