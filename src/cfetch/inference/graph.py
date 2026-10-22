#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .ingress import stable_int_seed
from .common import (
    EPS,
    add_external_common_args,
    build_external_ingress_plan,
    clip_label,
    encode_clip_tables,
    load_vae_from_args,
    release_plan_images,
    summarize_tensor,
    summarize_values,
    write_ingress_plan_streaming,
    write_json,
    write_jsonl,
    write_row_tables,
)

from tqdm.auto import tqdm


@dataclass(frozen=True)
class GraphDiffusionConfig:
    latent_source: str = "vae"
    latent_transform: str = "mu_varnorm"
    graph_k: int = 128
    sigma: float = 0.0
    sigma_quantile: float = 0.50
    weight_power: float = 2.0
    symmetrize: bool = True
    restart: float = 0.30
    iterations: int = 80
    clamp_anchors: bool = True
    clip_prob_threshold: float = 0.70
    row_prob_threshold: float = 0.70
    query_clip_accept_fraction: float = 0.25
    min_anchor_mass: float = 0.002
    min_row_clips: int = 8
    unknown_label: str = "OutOfSample"
    chunk_size: int = 512
    batch_size: int = 8
    num_workers: int = 0
    progress: bool = True

    poolwise_graph: bool = True
    anchor_k_per_label: int = 8
    anchor_edge_boost: float = 1.0
    anchor_score_tau: float = 0.002
    anchor_score_reference: float = 0.002
    max_query_clips_per_graph: int = 4000
    include_neighbor_details: bool = False


@dataclass
class GraphClipVote:
    query_clip_id: str
    clip_len: int
    pool_id: str
    key_ids: tuple[int, ...]
    rowids: tuple[int, ...]
    subject_key_id: Optional[int]
    subject_rowid: Optional[int]
    subject_record_id: Optional[str]
    in_sample: bool
    call_state: str
    top_label: Optional[str]
    top_probability: float
    second_label: Optional[str]
    second_probability: float
    probability_margin: float
    anchor_mass: float
    anchored_score_exp: float
    anchored_score_hard: float
    entropy: float
    probabilities: dict[str, float]
    nearest_seed_distance: float
    nearest_seed_label: Optional[str]
    neighbors: list[dict[str, Any]]


@dataclass
class GraphRowVote:
    record_id: str
    key_id: int
    rowid: int
    pool_id: str
    clips_total: int
    clips_in_sample: int
    in_sample_fraction: float
    in_sample: bool
    call_state: str
    top_label: Optional[str]
    label_weights: dict[str, float]
    label_counts: dict[str, int]
    membership_score: float
    membership_margin: float
    top_probability: float
    second_label: Optional[str]
    second_probability: float
    probability_margin: float
    anchor_mass_mean: float
    anchor_mass_p25: float
    anchor_mass_p50: float
    anchor_mass_p75: float
    anchored_score_exp_mean: float
    anchored_score_exp_p50: float
    anchored_score_hard_mean: float
    anchored_score_hard_p50: float
    entropy_mean: float
    probability_p25: float
    probability_p50: float
    probability_p75: float
    decision_margin: float
    uncertainty_score: float
    probabilities_mean_by_label: dict[str, float]


def _rms_cdist_chunk(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    X = X.float().contiguous()
    Y = Y.float().contiguous()
    dim = max(1, int(X.shape[1]))
    x2 = X.square().sum(dim=1, keepdim=True)
    y2 = Y.square().sum(dim=1).view(1, -1)
    d2 = (x2 + y2 - 2.0 * (X @ Y.t())) / float(dim)
    return torch.sqrt(d2.clamp_min(0.0))


def _rms_distance_to_one(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Exact RMS-L2 from each row of X to one vector, without [N, dim] broadcast."""
    X = X.float().contiguous()
    y = y.float().contiguous().view(-1)
    dim = max(1, int(X.shape[1]))
    d2 = (X.square().sum(dim=1) + y.square().sum() - 2.0 * (X @ y)) / float(dim)
    return torch.sqrt(d2.clamp_min(0.0))


def _resolve_graph_knn_device() -> torch.device:
    req = os.environ.get("GRAPH_KNN_DEVICE", "auto").strip().lower()
    if req == "cpu":
        return torch.device("cpu")
    if req.startswith("cuda"):
        return torch.device(req if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def exact_knn_edges(Z: torch.Tensor, *, k: int, chunk_size: int, progress: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Z_cpu = Z.float().cpu().contiguous()
    n = int(Z_cpu.shape[0])
    if n <= 1:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32)
    kk = max(1, min(int(k), n - 1)); step = max(1, int(chunk_size))
    total_edges = n * kk
    rows_out = torch.empty((total_edges,), dtype=torch.long)
    cols_out = torch.empty((total_edges,), dtype=torch.long)
    dvals_out = torch.empty((total_edges,), dtype=torch.float32)
    work_device = _resolve_graph_knn_device()
    try:
        Z_work = Z_cpu.to(work_device, non_blocking=True).contiguous()
    except RuntimeError as e:
        print(f"[graph_diffusion] could not place embeddings on {work_device}; falling back to CPU: {e}", file=sys.stderr, flush=True)
        work_device = torch.device("cpu"); Z_work = Z_cpu
    starts = range(0, n, step)
    with torch.inference_mode():
        for start_i in tqdm(starts, desc=f"build exact kNN graph ({work_device})", unit="chunk", disable=not progress, total=math.ceil(n / step)):
            end_i = min(n, start_i + step)
            d = _rms_cdist_chunk(Z_work[start_i:end_i], Z_work)
            local = torch.arange(start_i, end_i, dtype=torch.long, device=d.device)
            d[torch.arange(end_i - start_i, dtype=torch.long, device=d.device), local] = float("inf")
            vals, idxs = torch.topk(d, k=kk, dim=1, largest=False, sorted=True)
            lo = start_i * kk; hi = end_i * kk
            rows_out[lo:hi] = torch.arange(start_i, end_i, dtype=torch.long).view(-1, 1).expand(end_i - start_i, kk).reshape(-1)
            cols_out[lo:hi] = idxs.detach().cpu().reshape(-1).long()
            dvals_out[lo:hi] = vals.detach().cpu().reshape(-1).float()
            del d, vals, idxs
    del Z_work
    if work_device.type == "cuda": torch.cuda.empty_cache()
    return rows_out, cols_out, dvals_out


def _auto_sigma(dists: torch.Tensor, sigma: float, q: float) -> float:
    if float(sigma) > 0:
        return float(sigma)
    vals = dists[torch.isfinite(dists) & (dists > 0)].float().cpu().numpy()
    if vals.size == 0:
        return 1.0
    return float(max(np.quantile(vals, min(1.0, max(0.0, float(q)))), EPS))


def make_transition(
    rows: torch.Tensor,
    cols: torch.Tensor,
    dists: torch.Tensor,
    n: int,
    cfg: GraphDiffusionConfig,
    edge_boosts: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    sigma = _auto_sigma(dists, float(cfg.sigma), float(cfg.sigma_quantile))
    w = torch.exp(-torch.pow((dists.float() / max(sigma, EPS)).clamp_min(0.0), float(cfg.weight_power))).clamp_min(EPS)
    if edge_boosts is not None and edge_boosts.numel() == w.numel():
        w = w * edge_boosts.float().clamp_min(EPS)
    rows = rows.long().cpu()
    cols = cols.long().cpu()
    if bool(cfg.symmetrize):
        rows, cols, w = torch.cat([rows, cols]), torch.cat([cols, rows]), torch.cat([w, w])
    row_sum = torch.zeros((n,), dtype=torch.float32)
    row_sum.index_add_(0, rows, w.float())
    w_norm = w.float() / row_sum[rows].clamp_min(EPS)
    return rows, cols, w_norm, sigma


def anchor_augmented_edges(
    A: torch.Tensor, Q: torch.Tensor, anchor_clips: Sequence[Any], *, n_anchor: int,
    k_per_label: int, chunk_size: int, progress: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kk = max(0, int(k_per_label))
    if kk <= 0 or A.numel() == 0 or Q.numel() == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32)
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(anchor_clips): by_label[clip_label(c)].append(int(i))
    label_specs = [(lab, idxs, min(kk, len(idxs))) for lab, idxs in sorted(by_label.items()) if idxs]
    total_edges = int(Q.shape[0]) * sum(k for _lab, _idxs, k in label_specs)
    rows_out = torch.empty((total_edges,), dtype=torch.long); cols_out = torch.empty((total_edges,), dtype=torch.long); dists_out = torch.empty((total_edges,), dtype=torch.float32)
    offset = 0; step = max(1, int(chunk_size))
    for lab, idxs, k_lab in label_specs:
        idx = torch.as_tensor(idxs, dtype=torch.long); A_lab = A[idx].float().contiguous()
        for start_i in tqdm(range(0, int(Q.shape[0]), step), desc=f"anchor edges {lab}", unit="chunk", total=math.ceil(int(Q.shape[0]) / step), disable=not progress, leave=False):
            end_i = min(int(Q.shape[0]), start_i + step)
            d = _rms_cdist_chunk(Q[start_i:end_i], A_lab); vals, rel = torch.topk(d, k=k_lab, dim=1, largest=False, sorted=True)
            count = (end_i - start_i) * k_lab; hi = offset + count
            rows_out[offset:hi] = (n_anchor + torch.arange(start_i, end_i, dtype=torch.long)).view(-1, 1).expand(end_i - start_i, k_lab).reshape(-1)
            cols_out[offset:hi] = idx[rel.reshape(-1).cpu()].long(); dists_out[offset:hi] = vals.detach().cpu().reshape(-1).float()
            offset = hi; del d, vals, rel
    return rows_out[:offset], cols_out[:offset], dists_out[:offset]


def _anchored_scores(top_probability: float, anchor_mass: float, cfg: GraphDiffusionConfig) -> tuple[float, float]:
    tau = float(cfg.anchor_score_tau) if float(cfg.anchor_score_tau) > 0 else max(float(cfg.min_anchor_mass), EPS)
    ref = float(cfg.anchor_score_reference) if float(cfg.anchor_score_reference) > 0 else max(float(cfg.min_anchor_mass), EPS)
    am = max(0.0, float(anchor_mass))
    p = max(0.0, min(1.0, float(top_probability)))
    exp_score = p * (1.0 - math.exp(-am / max(tau, EPS)))
    hard_score = p * min(1.0, am / max(ref, EPS))
    return float(max(0.0, min(1.0, exp_score))), float(max(0.0, min(1.0, hard_score)))


def diffuse_labels(rows: torch.Tensor, cols: torch.Tensor, weights: torch.Tensor, Y: torch.Tensor, n_anchor: int, cfg: GraphDiffusionConfig) -> torch.Tensor:
    F = Y.clone().float()
    restart = min(1.0, max(0.0, float(cfg.restart)))
    for _ in tqdm(range(max(1, int(cfg.iterations))), desc="diffuse labels", unit="iter", disable=not cfg.progress):
        propagated = torch.zeros_like(F)
        messages = F[cols] * weights.view(-1, 1)
        propagated.index_add_(0, rows, messages)
        F = restart * Y + (1.0 - restart) * propagated
        if bool(cfg.clamp_anchors) and n_anchor > 0:
            F[:n_anchor] = Y[:n_anchor]
    row_sums = F.sum(dim=1, keepdim=True).clamp_min(EPS)
    return F / row_sums


def _entropy(prob: np.ndarray) -> float:
    p = prob[prob > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p + EPS)) / max(math.log(len(prob)), EPS)) if len(prob) > 1 else 0.0


def _score_graph_single_scope(anchors: Any, queries: Any, cfg: GraphDiffusionConfig, *, graph_scope: str = "global") -> tuple[list[GraphClipVote], dict[str, Any]]:
    A = anchors.embeddings.float().cpu().contiguous()
    Q = queries.embeddings.float().cpu().contiguous()
    if A.numel() == 0 or Q.numel() == 0:
        return [], {"empty": True, "graph_scope": graph_scope}
    if A.shape[1] != Q.shape[1]:
        raise ValueError(f"Anchor/query embedding dims differ: {A.shape} vs {Q.shape}")
    labels = sorted({clip_label(c) for c in anchors.clips})
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    n_anchor = int(A.shape[0])
    n_query = int(Q.shape[0])
    Z = torch.cat([A, Q], dim=0).contiguous()

    rows, cols, dists = exact_knn_edges(Z, k=int(cfg.graph_k), chunk_size=int(cfg.chunk_size), progress=bool(cfg.progress))
    edge_boosts = torch.ones_like(dists, dtype=torch.float32)
    if int(cfg.anchor_k_per_label) > 0:
        ar, ac, ad = anchor_augmented_edges(A, Q, anchors.clips, n_anchor=n_anchor, k_per_label=int(cfg.anchor_k_per_label), chunk_size=int(cfg.chunk_size), progress=bool(cfg.progress))
        if ar.numel():
            rows = torch.cat([rows, ar])
            cols = torch.cat([cols, ac])
            dists = torch.cat([dists, ad])
            edge_boosts = torch.cat([edge_boosts, torch.full_like(ad, float(cfg.anchor_edge_boost), dtype=torch.float32)])

    rows, cols, weights, sigma = make_transition(rows, cols, dists, int(Z.shape[0]), cfg, edge_boosts=edge_boosts)
    Y = torch.zeros((int(Z.shape[0]), len(labels)), dtype=torch.float32)
    for i, clip in enumerate(anchors.clips):
        Y[i, label_to_idx[clip_label(clip)]] = 1.0
    F = diffuse_labels(rows, cols, weights, Y, n_anchor, cfg)

    anchor_mass = torch.zeros((int(Z.shape[0]),), dtype=torch.float32)
    direct_seed = cols < n_anchor
    if bool(direct_seed.any()):
        anchor_mass.index_add_(0, rows[direct_seed], weights[direct_seed])

    query_votes: list[GraphClipVote] = []
    for qi, clip in tqdm(enumerate(queries.clips), desc=f"emit graph clip votes {graph_scope}", unit="clip", total=len(queries.clips), disable=not cfg.progress, leave=False):
        node = n_anchor + int(qi)
        prob = F[node].detach().cpu().numpy().astype(np.float64)
        order = np.argsort(-prob)
        top_i = int(order[0]) if order.size else -1
        second_i = int(order[1]) if order.size > 1 else -1
        top_label = labels[top_i] if top_i >= 0 else None
        second_label = labels[second_i] if second_i >= 0 else None
        top_prob = float(prob[top_i]) if top_i >= 0 else 0.0
        second_prob = float(prob[second_i]) if second_i >= 0 else 0.0
        dseed = _rms_distance_to_one(A, Q[qi])
        best_idx = int(torch.argmin(dseed).item()) if dseed.numel() else -1
        best_dist = float(dseed[best_idx].item()) if best_idx >= 0 else float("inf")
        best_label = clip_label(anchors.clips[best_idx]) if best_idx >= 0 else None
        am = float(anchor_mass[node].item())
        anchored_exp, anchored_hard = _anchored_scores(top_prob, am, cfg)
        in_sample = bool(top_prob >= float(cfg.clip_prob_threshold) and am >= float(cfg.min_anchor_mass))
        if in_sample and second_prob >= top_prob * 0.95 and len(labels) > 1:
            state = "multi_hit_ambiguous"
        elif in_sample:
            state = "single_hit"
        else:
            state = "diffusion_low_confidence"
        neighbors: list[dict[str, Any]] = []
        if bool(cfg.include_neighbor_details):
            neigh_mask = rows == node
            neigh_idx = torch.nonzero(neigh_mask, as_tuple=False).flatten()[:10]
            for rank, ei in enumerate(neigh_idx.tolist(), start=1):
                nb = int(cols[int(ei)].item()); is_anchor = nb < n_anchor
                c = anchors.clips[nb] if is_anchor else queries.clips[nb - n_anchor]
                neighbors.append({"rank": rank, "node_kind": "anchor" if is_anchor else "query", "clip_id": getattr(c, "clip_id", ""), "label": clip_label(c) if is_anchor else None, "edge_weight": float(weights[int(ei)].item())})
        query_votes.append(GraphClipVote(
            query_clip_id=getattr(clip, "clip_id", ""),
            clip_len=int(getattr(clip, "clip_len", 0)),
            pool_id=getattr(clip, "pool_id", ""),
            key_ids=tuple(int(x) for x in getattr(clip, "key_ids", ())),
            rowids=tuple(int(x) for x in getattr(clip, "rowids", ())),
            subject_key_id=getattr(clip, "subject_key_id", None),
            subject_rowid=getattr(clip, "subject_rowid", None),
            subject_record_id=getattr(clip, "subject_record_id", None),
            in_sample=in_sample,
            call_state=state,
            top_label=top_label,
            top_probability=top_prob,
            second_label=second_label,
            second_probability=second_prob,
            probability_margin=float(top_prob - second_prob),
            anchor_mass=am,
            anchored_score_exp=anchored_exp,
            anchored_score_hard=anchored_hard,
            entropy=_entropy(prob),
            probabilities={lab: float(prob[i]) for lab, i in label_to_idx.items()},
            nearest_seed_distance=best_dist,
            nearest_seed_label=best_label,
            neighbors=neighbors,
        ))
    summary = {
        "graph_scope": graph_scope,
        "num_anchor_clips": n_anchor,
        "num_query_clips": n_query,
        "labels": labels,
        "graph_k": int(cfg.graph_k),
        "anchor_k_per_label": int(cfg.anchor_k_per_label),
        "anchor_edge_boost": float(cfg.anchor_edge_boost),
        "sigma": float(sigma),
        "num_directed_edges_after_sym": int(rows.numel()),
        "edge_distance_summary": summarize_tensor(dists),
        "query_anchor_mass_summary": summarize_values([v.anchor_mass for v in query_votes]),
        "query_anchored_score_exp_summary": summarize_values([v.anchored_score_exp for v in query_votes]),
        "query_anchored_score_hard_summary": summarize_values([v.anchored_score_hard for v in query_votes]),
    }
    return query_votes, summary


def _subdivide_query_pool(queries: Any, idxs: Sequence[int], max_size: int, seed: int, progress: bool) -> list[list[int]]:
    idxs = list(map(int, idxs)); cap = max(0, int(max_size))
    if cap <= 0 or len(idxs) <= cap: return [idxs]
    groups = int(math.ceil(len(idxs) / cap))
    from ..data.clips import kmeans_torch
    X = F.normalize(queries.embeddings[torch.as_tensor(idxs, dtype=torch.long)].float().cpu(), p=2, dim=1, eps=EPS)
    labels = kmeans_torch(X, groups, seed=int(seed), progress=bool(progress))
    out: list[list[int]] = []
    for g in range(groups):
        members = [idxs[i] for i, lab in enumerate(labels.tolist()) if int(lab) == g]
        for start_i in range(0, len(members), cap):
            if members[start_i:start_i + cap]: out.append(members[start_i:start_i + cap])
    return out


def score_graph_for_len(anchors: Any, queries: Any, cfg: GraphDiffusionConfig) -> tuple[list[GraphClipVote], dict[str, Any]]:
    if not bool(cfg.poolwise_graph):
        return _score_graph_single_scope(anchors, queries, cfg, graph_scope="global")
    pool_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, clip in enumerate(queries.clips): pool_to_indices[str(getattr(clip, "pool_id", "unassigned"))].append(int(i))
    scopes: list[tuple[str, list[int]]] = []
    for pool_id, idxs in sorted(pool_to_indices.items()):
        parts = _subdivide_query_pool(queries, idxs, int(cfg.max_query_clips_per_graph), seed=stable_int_seed(f"graph-subdivide|{pool_id}", 1234), progress=cfg.progress)
        scopes.extend((pool_id if len(parts) == 1 else f"{pool_id}__sub{j:03d}", part) for j, part in enumerate(parts))
    all_votes: list[GraphClipVote] = []; pool_summaries: dict[str, Any] = {}
    for scope_id, idxs in tqdm(scopes, desc="poolwise graph scopes", unit="scope", disable=not cfg.progress):
        idx = torch.as_tensor(idxs, dtype=torch.long)
        qsub = SimpleNamespace(clips=[queries.clips[i] for i in idxs], embeddings=queries.embeddings[idx].float().cpu().contiguous())
        votes, summary = _score_graph_single_scope(anchors, qsub, cfg, graph_scope=scope_id)
        all_votes.extend(votes); pool_summaries[scope_id] = summary
        del qsub
    summary = {
        "poolwise_graph": True, "num_original_pools": len(pool_to_indices), "num_graph_scopes": len(scopes),
        "max_query_clips_per_graph": int(cfg.max_query_clips_per_graph),
        "num_anchor_clips": int(anchors.embeddings.shape[0]), "num_query_clips": int(queries.embeddings.shape[0]),
        "labels": sorted({clip_label(c) for c in anchors.clips}), "graph_k": int(cfg.graph_k),
        "anchor_k_per_label": int(cfg.anchor_k_per_label), "anchor_edge_boost": float(cfg.anchor_edge_boost),
        "pool_summaries": pool_summaries,
        "query_anchor_mass_summary": summarize_values([v.anchor_mass for v in all_votes]),
        "query_anchored_score_exp_summary": summarize_values([v.anchored_score_exp for v in all_votes]),
        "query_anchored_score_hard_summary": summarize_values([v.anchored_score_hard for v in all_votes]),
        "clip_call_state_counts": dict(Counter(v.call_state for v in all_votes)),
    }
    return all_votes, summary


def aggregate_graph_rows(plan: Any, clip_votes: Sequence[GraphClipVote], cfg: GraphDiffusionConfig) -> list[GraphRowVote]:
    row_to_pool = {int(a.rowid): a.pool_id for a in plan.query_assignments}
    acc: dict[int, dict[str, Any]] = {}
    for vote in tqdm(clip_votes, desc="aggregate graph rows", unit="clip", disable=not cfg.progress, leave=False):
        if vote.subject_rowid is not None:
            row_iter = [(int(vote.subject_rowid), int(vote.subject_key_id if vote.subject_key_id is not None else vote.key_ids[0]))]
        else:
            row_iter = list(zip(vote.rowids, vote.key_ids))
        for rid, kid in row_iter:
            entry = acc.setdefault(int(rid), {
                "key_id": int(kid), "rowid": int(rid), "clips_total": 0, "clips_in_sample": 0,
                "label_weights": defaultdict(float), "label_counts": Counter(), "prob_sum": defaultdict(float),
                "prob_count": defaultdict(int), "top_probs": [], "anchor_masses": [], "anchored_exp": [], "anchored_hard": [], "entropies": [], "states": Counter(),
            })
            entry["clips_total"] += 1
            entry["states"][vote.call_state] += 1
            entry["top_probs"].append(float(vote.top_probability))
            entry["anchor_masses"].append(float(vote.anchor_mass))
            entry["anchored_exp"].append(float(vote.anchored_score_exp))
            entry["anchored_hard"].append(float(vote.anchored_score_hard))
            entry["entropies"].append(float(vote.entropy))
            for lab, p in vote.probabilities.items():
                entry["prob_sum"][lab] += float(p)
                entry["prob_count"][lab] += 1
            if vote.in_sample:
                entry["clips_in_sample"] += 1
                if vote.top_label is not None:
                    entry["label_weights"][vote.top_label] += float(vote.top_probability)
                    entry["label_counts"][vote.top_label] += 1
    for rec in plan.query_records:
        acc.setdefault(int(rec.rowid), {
            "key_id": int(rec.key_id), "rowid": int(rec.rowid), "clips_total": 0, "clips_in_sample": 0,
            "label_weights": defaultdict(float), "label_counts": Counter(), "prob_sum": defaultdict(float),
            "prob_count": defaultdict(int), "top_probs": [], "anchor_masses": [], "anchored_exp": [], "anchored_hard": [], "entropies": [], "states": Counter(),
        })
    rowid_to_record = {int(r.rowid): r for r in plan.query_records}
    out: list[GraphRowVote] = []
    for rowid, entry in sorted(acc.items(), key=lambda kv: kv[0]):
        total = int(entry["clips_total"])
        ins = int(entry["clips_in_sample"])
        frac = float(ins / total) if total else 0.0
        probs_by_label = {lab: float(entry["prob_sum"][lab] / max(1, entry["prob_count"][lab])) for lab in sorted(entry["prob_sum"])}
        ordered = sorted(probs_by_label.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        top_label, top_prob = ordered[0] if ordered else (None, 0.0)
        second_label, second_prob = ordered[1] if len(ordered) > 1 else (None, 0.0)
        top_probs = np.asarray(entry.get("top_probs", []), dtype=np.float64)
        anchor_m = np.asarray(entry.get("anchor_masses", []), dtype=np.float64)
        anchored_exp = np.asarray(entry.get("anchored_exp", []), dtype=np.float64)
        anchored_hard = np.asarray(entry.get("anchored_hard", []), dtype=np.float64)
        ent = np.asarray(entry.get("entropies", []), dtype=np.float64)
        anchor_p75 = float(np.quantile(anchor_m, 0.75)) if anchor_m.size else 0.0
        anchored_exp_mean = float(anchored_exp.mean()) if anchored_exp.size else 0.0
        row_in_sample = bool(
            total >= int(cfg.min_row_clips)
            and frac >= float(cfg.query_clip_accept_fraction)
            and float(top_prob) >= float(cfg.row_prob_threshold)
            and anchor_p75 >= float(cfg.min_anchor_mass)
        )
        if total <= 0 or total < int(cfg.min_row_clips):
            state = "unscored"
        elif not row_in_sample:
            state = "diffusion_low_confidence"
        elif entry["states"].get("multi_hit_ambiguous", 0) >= max(1, int(math.ceil(float(cfg.query_clip_accept_fraction) * total))):
            state = "multi_hit_ambiguous"
        else:
            state = "single_hit"
        decision_margin = min(
            frac - float(cfg.query_clip_accept_fraction),
            float(top_prob) - float(cfg.row_prob_threshold),
            anchor_p75 - float(cfg.min_anchor_mass),
        ) if total >= int(cfg.min_row_clips) else -float(cfg.query_clip_accept_fraction)
        denom = max(float(cfg.query_clip_accept_fraction), float(cfg.row_prob_threshold), 1.0 - float(cfg.row_prob_threshold), float(cfg.min_anchor_mass), EPS)
        uncertainty = float(max(0.0, 1.0 - min(1.0, abs(decision_margin) / denom)))
        rec = rowid_to_record.get(int(rowid))
        record_id = rec.record_id if rec is not None else f"query:row{rowid}:key{entry['key_id']}"
        out.append(GraphRowVote(
            record_id=record_id,
            key_id=int(entry["key_id"]),
            rowid=int(rowid),
            pool_id=row_to_pool.get(int(rowid), "unassigned"),
            clips_total=total,
            clips_in_sample=ins,
            in_sample_fraction=frac,
            in_sample=row_in_sample,
            call_state=state,
            top_label=top_label,
            label_weights={k: float(v) for k, v in dict(entry["label_weights"]).items()},
            label_counts={k: int(v) for k, v in dict(entry["label_counts"]).items()},
            membership_score=float(anchored_exp_mean),
            membership_margin=float(decision_margin),
            top_probability=float(top_prob),
            second_label=second_label,
            second_probability=float(second_prob),
            probability_margin=float(float(top_prob) - float(second_prob)),
            anchor_mass_mean=float(anchor_m.mean()) if anchor_m.size else 0.0,
            anchor_mass_p25=float(np.quantile(anchor_m, 0.25)) if anchor_m.size else 0.0,
            anchor_mass_p50=float(np.quantile(anchor_m, 0.50)) if anchor_m.size else 0.0,
            anchor_mass_p75=float(np.quantile(anchor_m, 0.75)) if anchor_m.size else 0.0,
            anchored_score_exp_mean=float(anchored_exp.mean()) if anchored_exp.size else 0.0,
            anchored_score_exp_p50=float(np.quantile(anchored_exp, 0.50)) if anchored_exp.size else 0.0,
            anchored_score_hard_mean=float(anchored_hard.mean()) if anchored_hard.size else 0.0,
            anchored_score_hard_p50=float(np.quantile(anchored_hard, 0.50)) if anchored_hard.size else 0.0,
            entropy_mean=float(ent.mean()) if ent.size else 0.0,
            probability_p25=float(np.quantile(top_probs, 0.25)) if top_probs.size else 0.0,
            probability_p50=float(np.quantile(top_probs, 0.50)) if top_probs.size else 0.0,
            probability_p75=float(np.quantile(top_probs, 0.75)) if top_probs.size else 0.0,
            decision_margin=float(decision_margin),
            uncertainty_score=uncertainty,
            probabilities_mean_by_label=probs_by_label,
        ))
    return out


def run_graph_diffusion(plan: Any, vae: Any | None, cfg: GraphDiffusionConfig, *, retain_clip_votes: bool = True) -> dict[str, Any]:
    tables = encode_clip_tables(plan, vae, latent_source=cfg.latent_source, latent_transform=cfg.latent_transform, batch_size=cfg.batch_size, num_workers=cfg.num_workers, progress=cfg.progress)
    released = release_plan_images(plan)
    all_clips: list[GraphClipVote] = []
    by_len: dict[str, Any] = {}
    for clip_len, (anchors, queries) in tables.items():
        votes, summary = score_graph_for_len(anchors, queries, cfg)
        by_len[str(clip_len)] = summary
        all_clips.extend(votes)
    del tables
    gc.collect()
    rows = aggregate_graph_rows(plan, all_clips, cfg)
    summary = {
        "method": "graph_diffusion_label_propagation",
        "config": asdict(cfg),
        "ingress": plan.to_summary_dict(),
        "num_query_clip_votes": len(all_clips),
        "num_query_clips_in_sample": int(sum(1 for v in all_clips if v.in_sample)),
        "num_query_rows": len(rows),
        "num_query_rows_in_sample": int(sum(1 for r in rows if r.in_sample)),
        "clip_call_state_counts": dict(Counter(v.call_state for v in all_clips)),
        "row_call_state_counts": dict(Counter(r.call_state for r in rows)),
        "row_membership_score_summary": summarize_values([r.membership_score for r in rows]),
        "row_top_probability_summary": summarize_values([r.top_probability for r in rows]),
        "row_anchor_mass_mean_summary": summarize_values([r.anchor_mass_mean for r in rows]),
        "row_anchor_mass_p75_summary": summarize_values([r.anchor_mass_p75 for r in rows]),
        "row_anchored_score_exp_summary": summarize_values([r.anchored_score_exp_mean for r in rows]),
        "graph_by_clip_len": by_len,
        "released_plan_image_bytes": released,
    }
    clip_votes_out = all_clips if retain_clip_votes else []
    if not retain_clip_votes:
        all_clips.clear()
        gc.collect()
    return {"config": cfg, "clip_votes": clip_votes_out, "row_votes": rows, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CFetch graph diffusion over frozen VAE clip representations.")
    add_external_common_args(p)
    p.add_argument("--graph-k", type=int, default=128)
    p.add_argument("--sigma", type=float, default=0.0, help="RMS-L2 kernel sigma; <=0 auto-estimates from kNN distances.")
    p.add_argument("--sigma-quantile", type=float, default=0.50)
    p.add_argument("--weight-power", type=float, default=2.0)
    p.add_argument("--no-symmetrize", dest="symmetrize", action="store_false", default=True)
    p.add_argument("--restart", type=float, default=0.30)
    p.add_argument("--iterations", type=int, default=80)
    p.add_argument("--no-clamp-anchors", dest="clamp_anchors", action="store_false", default=True)
    p.add_argument("--clip-prob-threshold", type=float, default=0.70)
    p.add_argument("--row-prob-threshold", type=float, default=0.70)
    p.add_argument("--query-clip-accept-fraction", type=float, default=0.25)
    p.add_argument("--min-anchor-mass", type=float, default=0.002)
    p.add_argument("--min-row-clips", type=int, default=8)
    p.add_argument("--unknown-label", default="OutOfSample")
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--poolwise-graph", dest="poolwise_graph", action="store_true", default=True, help="Run one graph per query pool with all anchors included.")
    p.add_argument("--global-graph", dest="poolwise_graph", action="store_false", help="Disable poolwise mode and build one global graph.")
    p.add_argument("--anchor-k-per-label", type=int, default=8, help="Forced query->anchor edges per seed label. 0 disables anchor augmentation.")
    p.add_argument("--anchor-edge-boost", type=float, default=1.0, help="Boost for forced anchor edges before row-normalization.")
    p.add_argument("--anchor-score-tau", type=float, default=0.002, help="Tau for exp anchored score top_p*(1-exp(-mass/tau)).")
    p.add_argument("--anchor-score-reference", type=float, default=0.002, help="Reference for hard anchored score top_p*min(1,mass/ref).")
    p.add_argument("--max-query-clips-per-graph", type=int, default=int(os.environ.get("MAX_QUERY_CLIPS_PER_GRAPH", "4000")), help="Oversized SWFT pools are subdivided in latent space before exact graph construction; 0 disables.")
    p.add_argument("--neighbor-details", action="store_true", help="Write graph neighbor details; disabled by default for memory safety.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    plan = build_external_ingress_plan(args)
    vae = load_vae_from_args(args)
    cfg = GraphDiffusionConfig(
        latent_source=str(args.latent_source),
        latent_transform=str(args.latent_transform),
        graph_k=int(args.graph_k),
        sigma=float(args.sigma),
        sigma_quantile=float(args.sigma_quantile),
        weight_power=float(args.weight_power),
        symmetrize=bool(args.symmetrize),
        restart=float(args.restart),
        iterations=int(args.iterations),
        clamp_anchors=bool(args.clamp_anchors),
        clip_prob_threshold=float(args.clip_prob_threshold),
        row_prob_threshold=float(args.row_prob_threshold),
        query_clip_accept_fraction=float(args.query_clip_accept_fraction),
        min_anchor_mass=float(args.min_anchor_mass),
        min_row_clips=min(int(args.min_row_clips), int(plan.cfg.query_mixups_per_row)),
        unknown_label=str(args.unknown_label),
        chunk_size=int(args.chunk_size),
        poolwise_graph=bool(args.poolwise_graph),
        anchor_k_per_label=int(args.anchor_k_per_label),
        anchor_edge_boost=float(args.anchor_edge_boost),
        anchor_score_tau=float(args.anchor_score_tau),
        anchor_score_reference=float(args.anchor_score_reference),
        max_query_clips_per_graph=int(args.max_query_clips_per_graph),
        include_neighbor_details=bool(args.neighbor_details),
        batch_size=int(args.vote_batch_size),
        num_workers=int(args.num_workers),
        progress=bool(args.progress),
    )
    result = run_graph_diffusion(plan, vae, cfg, retain_clip_votes=not bool(args.no_clip_vote_output))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "graph_diffusion_summary.json", result["summary"])
    if not bool(args.no_clip_vote_output):
        write_jsonl(out / "query_clip_graph_votes.jsonl", (asdict(v) for v in result["clip_votes"]))
    write_row_tables(plan, result["row_votes"], out, prefix="graph", unknown_label=cfg.unknown_label)
    write_ingress_plan_streaming(plan, out / "ingress_plan", mode=str(args.ingress_plan_output))
    print({"status": "complete", "method": result["summary"].get("method"), "rows": result["summary"].get("num_query_rows"), "rows_in_sample": result["summary"].get("num_query_rows_in_sample"), "out_dir": str(out)}, flush=True)


if __name__ == "__main__":
    main()
