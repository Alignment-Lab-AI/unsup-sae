"""
Co-firing graph with manifold-weighted edges and multi-signal classifier scoring.

Features are nodes. Edges connect features that co-activate across inputs,
weighted by manifold similarity. This mirrors FST composition: features are
states, co-firing = arcs, and finding good classifiers = finding features
on high-weight composed paths (computed via PageRank).

Scoring combines geometry signals (kNN stability, Procrustes invariance,
manifold selectivity) with data signals (activation selectivity, PC alignment,
cross-split stability) in a two-stage pipeline:
  1. Geometry gate: filter features below 25th percentile geometry quality
  2. Data-driven ranking: rank passing features by data signals with
     dampened geometry multiplier
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import sparse
from collections import defaultdict, Counter
from sklearn.decomposition import TruncatedSVD


class FeatureGraph:
    """
    Weighted co-firing graph over SAE features.

    Construction:
    - For each input, identify top-k active features
    - Count co-firing pairs across inputs
    - Filter by minimum co-fire count (removes noise edges)
    - Weight surviving edges by manifold similarity

    Classifier scoring:
    - PageRank on weighted graph → global importance (graph centrality)
    - kNN stability under random projection → rotation invariance
    - Cross-subsample Procrustes alignment → structural consistency
    - Manifold similarity variance → selectivity in geometry space
    - PC alignment + cross-split correlation → data-driven discriminability
    - Two-stage combination: geometry gate → data-ranked
    """

    def __init__(
        self,
        engine,
        feature_acts_list: list[torch.Tensor],
        min_cofire_count: int = 3,
        top_features_per_sample: int = 50,
    ):
        self.engine = engine

        print("[graph] building co-firing edges...")
        cofire_counts = defaultdict(int)
        feature_counts = Counter()

        for acts in feature_acts_list:
            mean_acts = acts.mean(dim=0)
            n_active = min(top_features_per_sample, (mean_acts > 0).sum().item())
            topk_vals, topk_ids = mean_acts.topk(n_active)
            active = topk_ids[topk_vals > 0]
            for fid in active.tolist():
                feature_counts[fid] += 1
            if len(active) >= 2:
                pairs = torch.combinations(active, r=2)
                lo = torch.min(pairs, dim=1).values
                hi = torch.max(pairs, dim=1).values
                for a, b in zip(lo.tolist(), hi.tolist()):
                    cofire_counts[(a, b)] += 1

        self.edges = [(a, b, c) for (a, b), c in cofire_counts.items()
                      if c >= min_cofire_count]
        self.feature_counts = feature_counts
        self.active_features = sorted(feature_counts.keys())

        print(f"[graph] {len(self.active_features)} features, {len(self.edges)} edges")

        print(f"[graph] computing manifold weights...")
        self.weighted_edges = []
        for i, (a, b, count) in enumerate(self.edges):
            msim = engine.manifold_similarity(a, b)
            self.weighted_edges.append((a, b, count, msim))
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(self.edges)}")

        print(f"[graph] done. {len(engine._sim_cache)} cached sims")

    def find_good_classifiers(
        self,
        feature_acts_list: list[torch.Tensor],
        n_clusters: int = 4,
        alpha: float = 0.85,
        n_iters: int = 30,
        top_n: int = 20,
    ) -> dict:
        """
        Score features as classifiers without labels.

        Returns dict with:
        - good_classifiers: list of (feature_id, score) tuples
        - global_importance: PageRank scores
        - combined_scores: full score dict
        - stability, msim_variance, procrustes: per-signal dicts
        """
        n_samples = len(feature_acts_list)

        # ── PageRank on manifold-weighted co-firing graph ──
        global_importance = self._compute_pagerank(alpha, n_iters)

        # ── Signal 1: kNN stability under random projections (batched) ──
        stability_scores = self._compute_knn_stability(n_proj=5, k_check=15)

        # ── Signal 2: Manifold similarity variance ──
        msim_variance = self._compute_msim_variance()

        # ── Signal 3: Procrustes cross-subsample stability ──
        act_matrix = torch.stack([fa.mean(dim=0) for fa in feature_acts_list])
        procrustes_scores = self._compute_procrustes_stability(act_matrix, n_samples)

        # ── Signal 4: PC alignment ──
        X = act_matrix.numpy()
        X_centered = X - X.mean(axis=0, keepdims=True)
        n_comp = min(n_clusters * 3, n_samples - 1, X.shape[1])
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        svd.fit(X_centered)
        components = svd.components_

        # ── Signal 5: Cross-split PC correlation stability ──
        cv_scores = self._compute_cv_stability(X_centered, svd, n_samples)

        # ── Feature frequency and selectivity ──
        active_mask = (act_matrix > 0).float()
        freq_vec = active_mask.sum(dim=0)
        feature_freq = {fid: int(freq_vec[fid].item()) for fid in self.active_features}
        min_freq = max(3, n_samples * 0.02)
        active_frac = active_mask.mean(dim=0)
        selectivity = active_frac * (1 - active_frac)

        # ── Two-stage scoring ──
        disc_scores = {}
        combined_scores = {}
        all_signals = {}

        for fid in self.active_features:
            freq = feature_freq.get(fid, 0)
            if freq < min_freq:
                disc_scores[fid] = 0.0
                combined_scores[fid] = 0.0
                continue

            feat_var = np.var(X_centered[:, fid])
            pc_score = 0.0
            if feat_var > 1e-10:
                loadings = components[:, fid]
                pc_score = np.sum(loadings ** 2) / (feat_var + 1e-10)

            s = {
                'stab': stability_scores.get(fid, 0.0),
                'proc': procrustes_scores.get(fid, 0.0),
                'msim_var': msim_variance.get(fid, 0.0),
                'sel': selectivity[fid].item(),
                'pc': pc_score,
                'gi': global_importance.get(fid, 0.0),
                'freq_w': np.log1p(freq) / np.log1p(n_samples),
                'cvs': cv_scores.get(fid, 0.0),
            }
            disc_scores[fid] = pc_score * s['freq_w']
            all_signals[fid] = s

        # Stage 1: geometry gate (25th percentile)
        if all_signals:
            geom_scores = {
                fid: (s['stab'] + 0.1) * (s['proc'] + 0.1) * (s['msim_var'] + 0.05)
                for fid, s in all_signals.items()
            }
            geom_thresh = np.percentile(list(geom_scores.values()), 25)

            # Stage 2: data-driven ranking with dampened geometry multiplier
            for fid, s in all_signals.items():
                if geom_scores[fid] < geom_thresh:
                    combined_scores[fid] = 0.0
                    continue
                geom_mult = (geom_scores[fid] / (geom_thresh + 1e-10)) ** 0.3
                data = ((s['sel'] + 0.01) * (s['pc'] + 0.01) * s['freq_w']
                        * (s['cvs'] + 0.1) * (s['gi'] ** 0.25 + 0.1))
                combined_scores[fid] = data * geom_mult

        good_classifiers = sorted(combined_scores.items(), key=lambda x: -x[1])[:top_n]

        return {
            "good_classifiers": good_classifiers,
            "global_importance": global_importance,
            "disc_scores": disc_scores,
            "combined_scores": combined_scores,
            "stability": stability_scores,
            "msim_variance": msim_variance,
            "procrustes": procrustes_scores,
            "cv_stability": cv_scores,
            "all_signals": all_signals,
        }

    # ── Internal signal computations ──

    def _compute_pagerank(self, alpha, n_iters):
        feat_to_idx = {f: i for i, f in enumerate(self.active_features)}
        n = len(self.active_features)
        rows, cols, weights = [], [], []
        for a, b, count, msim in self.weighted_edges:
            if a not in feat_to_idx or b not in feat_to_idx:
                continue
            w = count * max(msim, 0.0)
            if w > 0:
                ia, ib = feat_to_idx[a], feat_to_idx[b]
                rows.extend([ia, ib])
                cols.extend([ib, ia])
                weights.extend([w, w])

        scores = np.zeros(n)
        if weights:
            A = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n))
            row_sums = np.array(A.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1.0
            T = sparse.diags(1.0 / row_sums) @ A
            scores = np.ones(n) / n
            for _ in range(n_iters):
                scores = alpha * (T.T @ scores) + (1 - alpha) / n

        return {f: scores[feat_to_idx[f]] for f in self.active_features}

    def _compute_knn_stability(self, n_proj=5, k_check=15):
        print("[graph] computing kNN stability...")
        D = self.engine.D
        active_idx_cpu = torch.tensor(self.active_features)
        active_idx = active_idx_cpu.to(self.engine.device)
        orig_nbrs = self.engine.neighbor_ids[active_idx_cpu][:, :k_check].to(self.engine.device)

        overlap_accum = torch.zeros(len(self.active_features), device=self.engine.device)
        for _ in range(n_proj):
            P = F.normalize(torch.randn(D, D // 2, device=self.engine.device), dim=0)
            projected = F.normalize(self.engine.dirs @ P, dim=1)
            proj_active = projected[active_idx]
            chunk = 512
            for ci in range(0, len(active_idx), chunk):
                ce = min(ci + chunk, len(active_idx))
                sims = proj_active[ci:ce] @ projected.T
                sims[torch.arange(ce - ci, device=self.engine.device), active_idx[ci:ce]] = -2.0
                proj_top = sims.topk(k_check, dim=1).indices
                orig_sorted = orig_nbrs[ci:ce].sort(dim=1).values
                proj_sorted = proj_top.sort(dim=1).values
                matches = (orig_sorted.unsqueeze(2) == proj_sorted.unsqueeze(1)).any(dim=2).sum(dim=1)
                overlap_accum[ci:ce] += matches.float() / k_check

        overlap_accum /= n_proj
        return {fid: overlap_accum[i].item() for i, fid in enumerate(self.active_features)}

    def _compute_msim_variance(self):
        adj_msims = defaultdict(list)
        for a, b, count, msim in self.weighted_edges:
            adj_msims[a].append(msim)
            adj_msims[b].append(msim)
        result = {}
        for fid in self.active_features:
            sims = adj_msims.get(fid, [])
            result[fid] = float(np.std(sims)) if len(sims) >= 3 else 0.0
        return result

    def _compute_procrustes_stability(self, act_matrix, n_samples, n_splits=5):
        print("[graph] computing Procrustes cross-subsample stability...")
        active_ids_list = list(self.active_features)
        n_act = len(active_ids_list)
        consistency = torch.zeros(n_act)

        for _ in range(n_splits):
            perm = torch.randperm(n_samples)
            half = n_samples // 2
            act_a = act_matrix[perm[:half]][:, active_ids_list].float()
            act_b = act_matrix[perm[half:2*half]][:, active_ids_list].float()

            n_comp_p = min(20, half - 1, n_act)
            Ua, Sa, Vha = torch.linalg.svd(act_a - act_a.mean(0, keepdim=True), full_matrices=False)
            Ub, Sb, Vhb = torch.linalg.svd(act_b - act_b.mean(0, keepdim=True), full_matrices=False)

            feat_a = (Vha[:n_comp_p] * Sa[:n_comp_p].unsqueeze(1)).T
            feat_b = (Vhb[:n_comp_p] * Sb[:n_comp_p].unsqueeze(1)).T

            M = feat_a.T @ feat_b
            Up, Sp, Vhp = torch.linalg.svd(M)
            aligned_a = feat_a @ (Up @ Vhp)

            cos_sim = F.cosine_similarity(aligned_a, feat_b, dim=1)
            consistency += cos_sim.clamp(min=0)

        consistency /= n_splits
        return {fid: consistency[i].item() for i, fid in enumerate(active_ids_list)}

    def _compute_cv_stability(self, X_centered, svd, n_samples, n_splits=5):
        pc_embedding = svd.transform(X_centered)
        cv_stability = np.zeros(len(self.active_features))
        rng = np.random.default_rng(42)
        active_arr = np.array(self.active_features)

        for _ in range(n_splits):
            perm = rng.permutation(n_samples)
            half = n_samples // 2
            Xa = X_centered[perm[:half]][:, active_arr]
            Xb = X_centered[perm[half:2*half]][:, active_arr]
            Pa = pc_embedding[perm[:half]]
            Pb = pc_embedding[perm[half:2*half]]

            corr_a = (Xa.T @ Pa) / (half - 1)
            corr_b = (Xb.T @ Pb) / (half - 1)
            norm_a = np.linalg.norm(corr_a, axis=1, keepdims=True) + 1e-10
            norm_b = np.linalg.norm(corr_b, axis=1, keepdims=True) + 1e-10
            cos = np.sum((corr_a / norm_a) * (corr_b / norm_b), axis=1)
            cv_stability += np.clip(cos, 0, 1)

        cv_stability /= n_splits
        return {fid: cv_stability[i] for i, fid in enumerate(self.active_features)}
