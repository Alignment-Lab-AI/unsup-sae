"""
Main entry point: find good classifiers from SAE features, unsupervised.

Loads a pretrained model + SAE, extracts activations on AG News,
runs the unsupervised classifier discovery pipeline, and evaluates
against ground-truth labels (which are never used in the search).

Usage:
    python run.py [--n-samples 400] [--device cuda] [--k 30]
"""

import torch
import numpy as np
from collections import defaultdict

from engine import MetaconceptEngine
from graph import FeatureGraph


def load_model_and_sae(
    model_name="gpt2",
    sae_release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre",
    device="cuda",
):
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    print(f"[load] model={model_name}, sae={sae_release}/{sae_id}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    sae = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=device)
    print(f"[load] SAE dict size: {sae.cfg.d_sae}, model dim: {sae.cfg.d_in}")
    return model, sae, model.tokenizer


def get_feature_activations(
    model, sae, texts,
    hook_point="blocks.8.hook_resid_pre",
    device="cuda",
    max_seq_len=128,
):
    feature_acts_all = []
    residuals_all = []
    for text in texts:
        tokens = model.to_tokens(text, prepend_bos=True)[:, :max_seq_len]
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_point])
            resid = cache[hook_point][0]
            feat_acts = sae.encode(resid)
        feature_acts_all.append(feat_acts.cpu())
        residuals_all.append(resid.cpu())
    return feature_acts_all, residuals_all


def evaluate(good_features, feature_acts, labels, graph):
    """Evaluate discovered features against ground-truth labels."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    act_matrix = torch.stack([fa.mean(dim=0) for fa in feature_acts])
    labels_t = torch.tensor(labels)
    label_set = sorted(set(labels))
    n_classes = len(label_set)
    n_samples = len(labels)
    X_all = act_matrix.numpy()
    y_all = np.array(labels)

    # Vectorized F-ratio
    grand_mean = act_matrix.mean(dim=0)
    ss_b = torch.zeros(act_matrix.shape[1])
    ss_w = torch.zeros(act_matrix.shape[1])
    for c in label_set:
        mask = labels_t == c
        nc = mask.sum().item()
        if nc == 0:
            continue
        cv = act_matrix[mask]
        cm = cv.mean(dim=0)
        ss_b += nc * (cm - grand_mean) ** 2
        ss_w += ((cv - cm.unsqueeze(0)) ** 2).sum(dim=0)
    all_f_ratios = (ss_b / max(n_classes - 1, 1)) / (ss_w / max(n_samples - n_classes, 1) + 1e-10)

    # Full classifier
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    cv_scores = cross_val_score(clf, X_all, y_all, cv=5, scoring='accuracy')
    clf.fit(X_all, y_all)
    coef_importance = np.abs(clf.coef_).max(axis=0)

    # Per-feature probe accuracy
    def probe_accuracy(fid, n_cv=5):
        x = X_all[:, fid].reshape(-1, 1)
        probe = LogisticRegression(max_iter=500, random_state=42)
        return cross_val_score(probe, x, y_all, cv=n_cv, scoring='accuracy').mean()

    # Compute all baselines and oracles
    all_f_dict = {fid: float(all_f_ratios[fid]) for fid in graph.active_features}
    coef_dict = {fid: float(coef_importance[fid]) for fid in graph.active_features}
    oracle_by_f = sorted(all_f_dict.items(), key=lambda x: -x[1])
    oracle_by_coef = sorted(coef_dict.items(), key=lambda x: -x[1])

    methods = {}

    # Our method
    our_fids = [fid for fid, _ in good_features[:20]]
    methods['ours'] = {
        'name': 'Ours (unsupervised)',
        'features': our_fids,
        'f_ratios': [float(all_f_ratios[f]) for f in our_fids],
        'probes': [probe_accuracy(f) for f in our_fids],
        'coefs': [coef_dict.get(f, 0) for f in our_fids],
    }

    # Baselines
    all_mean = act_matrix.mean(dim=0)
    naive_top = all_mean.topk(20).indices.tolist()
    methods['naive'] = {
        'name': 'Baseline: mean activation',
        'features': naive_top,
        'f_ratios': [float(all_f_ratios[f]) for f in naive_top],
        'probes': [probe_accuracy(f) for f in naive_top],
    }

    act_var = act_matrix.var(dim=0)
    var_top = act_var.topk(20).indices.tolist()
    methods['variance'] = {
        'name': 'Baseline: activation variance',
        'features': var_top,
        'f_ratios': [float(all_f_ratios[f]) for f in var_top],
        'probes': [probe_accuracy(f) for f in var_top],
    }

    active_frac = (act_matrix > 0).float().mean(dim=0)
    sp = active_frac * (1 - active_frac)
    mwa = act_matrix.sum(dim=0) / (act_matrix > 0).float().sum(dim=0).clamp(min=1)
    sel_top = (sp * mwa).topk(20).indices.tolist()
    methods['selectivity'] = {
        'name': 'Baseline: selectivity',
        'features': sel_top,
        'f_ratios': [float(all_f_ratios[f]) for f in sel_top],
        'probes': [probe_accuracy(f) for f in sel_top],
    }

    # Oracles
    coef_fids = [f for f, _ in oracle_by_coef[:20]]
    methods['oracle_coef'] = {
        'name': 'Oracle: classifier coef',
        'features': coef_fids,
        'f_ratios': [float(all_f_ratios[f]) for f in coef_fids],
        'probes': [probe_accuracy(f) for f in coef_fids],
    }

    oracle_fids = [f for f, _ in oracle_by_f[:20]]
    methods['oracle_f'] = {
        'name': 'Oracle: F-ratio',
        'features': oracle_fids,
        'f_ratios': [float(all_f_ratios[f]) for f in oracle_fids],
        'probes': [probe_accuracy(f) for f in oracle_fids],
    }

    # Overlap
    our_set = set(our_fids)
    overlap = {
        'f_top20': len(our_set & set(oracle_fids)),
        'f_top50': len(our_set & set(f for f, _ in oracle_by_f[:50])),
        'coef_top20': len(our_set & set(coef_fids)),
        'coef_top50': len(our_set & set(f for f, _ in oracle_by_coef[:50])),
    }

    # ── Compositional evaluation: multi-feature classifiers ──
    from sklearn.linear_model import LogisticRegression as LR
    comp = {}
    for k in [5, 10, 20]:
        for method_key, fids in [('ours', our_fids), ('oracle_coef', coef_fids), ('oracle_f', oracle_fids)]:
            subset = fids[:k]
            X_sub = X_all[:, subset]
            clf_sub = LR(max_iter=1000, C=1.0, random_state=42)
            acc = cross_val_score(clf_sub, X_sub, y_all, cv=5, scoring='accuracy').mean()
            comp[(method_key, k)] = acc

    # Full model accuracy (all active features)
    active_fids = list(graph.active_features)
    X_active = X_all[:, active_fids]
    clf_active = LR(max_iter=1000, C=1.0, random_state=42)
    comp[('all_active', len(active_fids))] = cross_val_score(
        clf_active, X_active, y_all, cv=5, scoring='accuracy').mean()

    # Sparse oracle: L1-regularized logistic regression on all active features
    from sklearn.linear_model import LogisticRegression
    sparse_clf = LogisticRegression(max_iter=2000, C=0.1, penalty='l1',
                                     solver='saga', random_state=42)
    sparse_clf.fit(X_all[:, active_fids], y_all)
    sparse_coefs = np.abs(sparse_clf.coef_).max(axis=0)
    nonzero_mask = sparse_coefs > 1e-6
    sparse_fids = [active_fids[i] for i in np.argsort(-sparse_coefs) if sparse_coefs[i] > 1e-6]
    sparse_overlap_ours = len(set(our_fids) & set(sparse_fids[:50]))
    n_sparse_nonzero = int(nonzero_mask.sum())

    return {
        'methods': methods,
        'overlap': overlap,
        'compositional': comp,
        'sparse_oracle': {
            'n_nonzero': n_sparse_nonzero,
            'top50_fids': sparse_fids[:50],
            'overlap_ours_top20': sparse_overlap_ours,
        },
        'full_cv_acc': cv_scores.mean(),
        'full_cv_std': cv_scores.std(),
        'n_classes': n_classes,
        'all_f_ratios': all_f_dict,
        'coef_importance': coef_dict,
    }


def print_results(eval_results):
    methods = eval_results['methods']
    n_classes = eval_results['n_classes']
    full_acc = eval_results['full_cv_acc']

    print(f"\n{'='*70}")
    print(f"FULL COMPARISON")
    print(f"(chance = {1/n_classes:.4f}, full model = {full_acc:.4f})")
    print(f"{'='*70}")

    for key in ['ours', 'naive', 'variance', 'selectivity', 'oracle_coef', 'oracle_f']:
        m = methods[key]
        f10 = np.mean(m['f_ratios'][:10])
        a10 = np.mean(m['probes'][:10])
        f20 = np.mean(m['f_ratios'])
        a20 = np.mean(m['probes'])
        print(f"\n{m['name']}:")
        for i, (fid, f, p) in enumerate(zip(m['features'][:10], m['f_ratios'][:10], m['probes'][:10])):
            line = f"  feature {fid:6d}  F={f:7.1f}  probe={p:.4f}"
            if 'coefs' in m:
                line += f"  coef={m['coefs'][i]:.4f}"
            print(line)
        print(f"  top-10 mean: F={f10:.1f}  probe={a10:.4f}")
        print(f"  top-20 mean: F={f20:.1f}  probe={a20:.4f}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Full classifier accuracy: {full_acc:.4f}")
    print(f"  Chance: {1/n_classes:.4f}\n")
    print(f"  {'Method':<40} {'F-10':>6} {'Acc-10':>7} {'F-20':>6} {'Acc-20':>7}")
    print(f"  {'-'*40} {'-'*6} {'-'*7} {'-'*6} {'-'*7}")
    for key in ['ours', 'naive', 'variance', 'selectivity', 'oracle_coef', 'oracle_f']:
        m = methods[key]
        f10 = np.mean(m['f_ratios'][:10])
        a10 = np.mean(m['probes'][:10])
        f20 = np.mean(m['f_ratios'])
        a20 = np.mean(m['probes'])
        print(f"  {m['name']:<40} {f10:>6.1f} {a10:>7.4f} {f20:>6.1f} {a20:>7.4f}")

    o = eval_results['overlap']
    print(f"\n  Overlap (our top-20):")
    print(f"    vs oracle F-ratio top-20: {o['f_top20']}/20")
    print(f"    vs oracle F-ratio top-50: {o['f_top50']}/20")
    print(f"    vs classifier coef top-20: {o['coef_top20']}/20")
    print(f"    vs classifier coef top-50: {o['coef_top50']}/20")

    # Compositional evaluation
    if 'compositional' in eval_results:
        comp = eval_results['compositional']
        print(f"\n{'='*70}")
        print("COMPOSITIONAL EVALUATION (multi-feature classifier accuracy)")
        print(f"{'='*70}")
        all_active_acc = [v for (k, _), v in comp.items() if k == 'all_active']
        if all_active_acc:
            print(f"  All active features: {all_active_acc[0]:.4f}")
        print(f"\n  {'K':<6} {'Ours':>8} {'Oracle coef':>12} {'Oracle F':>10}")
        print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*10}")
        for k in [5, 10, 20]:
            ours_acc = comp.get(('ours', k), 0)
            ocoef_acc = comp.get(('oracle_coef', k), 0)
            of_acc = comp.get(('oracle_f', k), 0)
            print(f"  {k:<6} {ours_acc:>8.4f} {ocoef_acc:>12.4f} {of_acc:>10.4f}")

    # Sparse oracle
    if 'sparse_oracle' in eval_results:
        so = eval_results['sparse_oracle']
        print(f"\n  Sparse oracle (L1 logistic regression):")
        print(f"    Non-zero features: {so['n_nonzero']}")
        print(f"    Our top-20 overlap with sparse top-50: {so['overlap_ours_top20']}/20")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unsupervised SAE classifier discovery")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--sae-release", default="gpt2-small-res-jb")
    parser.add_argument("--sae-id", default="blocks.8.hook_resid_pre")
    parser.add_argument("--hook-point", default="blocks.8.hook_resid_pre")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-samples", type=int, default=400)
    parser.add_argument("--min-cofire", type=int, default=3)
    parser.add_argument("--save-figures", action="store_true", help="Save visualization figures")
    args = parser.parse_args()

    # Load
    model, sae, tokenizer = load_model_and_sae(
        args.model, args.sae_release, args.sae_id, args.device
    )
    decoder_dirs = sae.W_dec.data
    engine = MetaconceptEngine(decoder_dirs, k=args.k, device=args.device)

    # Data
    print("\n[data] loading AG News...")
    from datasets import load_dataset
    ds = load_dataset("ag_news", split="test")
    ds = ds.shuffle(seed=42).select(range(args.n_samples))
    texts = ds["text"]
    labels = ds["label"]

    print("[data] extracting activations...")
    feature_acts, _ = get_feature_activations(
        model, sae, texts, hook_point=args.hook_point, device=args.device
    )

    # Build graph
    graph = FeatureGraph(engine, feature_acts, min_cofire_count=args.min_cofire)

    # Perturbation ensemble
    print("\n[ensemble] running perturbation ensemble...")
    ensemble_scores = defaultdict(float)
    n_runs = 0
    rng = np.random.default_rng(42)

    for n_pc in [3, 4, 6, 8, 12]:
        for boot in range(3):
            if boot > 0:
                idx = rng.choice(len(feature_acts), size=len(feature_acts), replace=True)
                boot_acts = [feature_acts[i] for i in idx]
            else:
                boot_acts = feature_acts
            results = graph.find_good_classifiers(boot_acts, n_clusters=n_pc, top_n=50)
            for rank, (fid, score) in enumerate(results["good_classifiers"]):
                ensemble_scores[fid] += score * (50 - rank)
            n_runs += 1

    print(f"  {n_runs} runs completed")
    good = sorted(ensemble_scores.items(), key=lambda x: -x[1])[:20]

    # Evaluate
    print("\n[eval] evaluating against ground-truth labels...")
    eval_results = evaluate(good, feature_acts, labels, graph)
    print_results(eval_results)

    n_active = len(graph.active_features)
    print(f"\n  Cost: {len(engine._sim_cache)} manifold sims "
          f"(vs {n_active*(n_active-1)//2} all-pairs, "
          f"{1 - len(engine._sim_cache)/max(n_active*(n_active-1)//2, 1):.1%} savings)")

    # Visualizations
    if args.save_figures:
        from visualize import make_all_figures
        make_all_figures(eval_results, results, graph, engine, feature_acts, labels)

    print("\ndone.")
    return good, eval_results, results


if __name__ == "__main__":
    main()
