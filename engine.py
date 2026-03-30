"""
Metaconcept engine: manifold geometry on SAE decoder directions.

Precomputes k-NN structure over unit-normalized decoder directions and
provides manifold-aware similarity (not just cosine) and agglomerative
composition of feature groups.
"""

import torch
import torch.nn.functional as F
from itertools import combinations


class MetaconceptEngine:
    """
    Manifold geometry engine for SAE decoder directions.

    Given decoder directions (d_sae, d_model), precomputes k-NN and provides:
    - manifold_similarity(i, j): similarity respecting manifold curvature
    - compose(indices): agglomerative merge via manifold-aware averaging

    Memory-efficient: stores neighbor indices only, not direction copies.
    Similarities are lazily computed and cached.
    """

    def __init__(self, directions: torch.Tensor, k: int = 30, device: str = "cuda"):
        self.device = device
        self.k = k
        self.D = directions.shape[1]
        self.N = directions.shape[0]
        self.dirs = F.normalize(directions.float().to(device), dim=1)

        print(f"[engine] precomputing {k}-NN for {self.N} directions...")
        self.neighbor_ids = self._build_knn(k)
        print(f"[engine] done.")

        self._sim_cache = {}

    def _build_knn(self, k, chunk_size=2048):
        all_ids = []
        for start in range(0, self.N, chunk_size):
            end = min(start + chunk_size, self.N)
            chunk = self.dirs[start:end]
            sims = chunk @ self.dirs.T
            sims[torch.arange(end - start, device=self.device),
                 torch.arange(start, end, device=self.device)] = -2.0
            topk = sims.topk(k, dim=1)
            all_ids.append(topk.indices.cpu())
        return torch.cat(all_ids, dim=0)

    def get_neighborhood(self, idx: int) -> torch.Tensor:
        return self.dirs[self.neighbor_ids[idx]].to(self.device)

    def locate(self, d_j: torch.Tensor, neighborhood: torch.Tensor) -> torch.Tensor:
        sims = neighborhood @ d_j
        mask = sims > 0
        if mask.sum() == 0:
            return d_j.clone()
        return F.normalize(neighborhood[mask].mean(dim=0), dim=0)

    def merge_locate(self, d_j: torch.Tensor,
                     neighborhood_a: torch.Tensor,
                     neighborhood_b: torch.Tensor) -> torch.Tensor:
        agreeing = torch.cat([
            neighborhood_a[neighborhood_a @ d_j > 0],
            neighborhood_b[neighborhood_b @ d_j > 0],
        ], dim=0)
        if agreeing.shape[0] == 0:
            return d_j.clone()
        return F.normalize(agreeing.mean(dim=0), dim=0)

    def manifold_similarity(self, idx_i: int, idx_j: int) -> float:
        """
        Manifold-aware similarity between features i and j.

        Measures how much two features "pull toward each other" when
        projected through their combined neighborhoods — captures
        manifold curvature that cosine similarity misses.

        Cached and symmetric.
        """
        key = (min(idx_i, idx_j), max(idx_i, idx_j))
        if key in self._sim_cache:
            return self._sim_cache[key]

        d_i, d_j = self.dirs[idx_i], self.dirs[idx_j]
        C_i, C_j = self.get_neighborhood(idx_i), self.get_neighborhood(idx_j)

        d_j_shifted = self.merge_locate(d_j, C_i, C_j)
        d_i_shifted = self.locate(d_i, torch.cat([C_i, C_j], dim=0))

        delta_j = d_j - d_j_shifted
        delta_i = d_i_shifted - d_i
        num = (delta_j * delta_i).sum()
        denom = delta_j.norm() * delta_i.norm() + 1e-10
        val = (num / denom).item()

        self._sim_cache[key] = val
        return val

    def compose(self, feature_indices: list[int]) -> tuple[torch.Tensor, list[tuple]]:
        """
        Greedy agglomerative merge. Returns composed direction + merge log.
        Tracks member sets through merges for correct importance propagation.
        """
        if len(feature_indices) == 1:
            return self.dirs[feature_indices[0]].clone(), []

        active = {}
        for idx in feature_indices:
            active[idx] = {
                "dir": self.dirs[idx].clone(),
                "neighborhood": self.get_neighborhood(idx).clone(),
                "members": {idx},
            }

        merge_log = []
        next_id = self.N

        while len(active) > 1:
            keys = list(active.keys())
            best_score = -float('inf')
            best_pair = None

            for a, b in combinations(keys, 2):
                d_a, C_a = active[a]["dir"], active[a]["neighborhood"]
                d_b, C_b = active[b]["dir"], active[b]["neighborhood"]
                C_both = torch.cat([C_a, C_b], dim=0)
                d_b_shifted = self.merge_locate(d_b, C_a, C_b)
                d_a_shifted = self.locate(d_a, C_both)
                delta_b = d_b - d_b_shifted
                delta_a = d_a_shifted - d_a
                score = ((delta_b * delta_a).sum() / (delta_b.norm() * delta_a.norm() + 1e-10)).item()
                if score > best_score:
                    best_score = score
                    best_pair = (a, b)

            a, b = best_pair
            members_a, members_b = active[a]["members"], active[b]["members"]
            merge_log.append((members_a.copy(), members_b.copy(), best_score))

            d_new = self.merge_locate(active[a]["dir"], active[a]["neighborhood"], active[b]["neighborhood"])
            C_new = torch.cat([active[a]["neighborhood"], active[b]["neighborhood"]], dim=0)
            active[next_id] = {"dir": d_new, "neighborhood": C_new, "members": members_a | members_b}
            next_id += 1
            del active[a], active[b]

        final_key = list(active.keys())[0]
        return active[final_key]["dir"], merge_log
