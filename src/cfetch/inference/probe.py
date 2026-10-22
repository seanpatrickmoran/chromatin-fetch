#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .ingress import PARTITION_CALIBRATION, PARTITION_PROPER, anchor_partition
from .common import (
    EPS,
    add_external_common_args,
    build_external_ingress_plan,
    clip_label,
    encode_clip_tables,
    apply_latent_transform,
    load_vae_from_args,
    quantile,
    release_plan_images,
    summarize_values,
    write_ingress_plan_streaming,
    write_json,
    write_jsonl,
    write_row_tables,
)

from tqdm.auto import tqdm

@dataclass(frozen=True)
class ProbeStabilityConfig:
    latent_source: str = "vae"
    latent_transform: str = "mu_varnorm"
    alpha: float = 0.05
    calibration_fraction: float = 0.25
    split_seed: int = 1234
    min_train_per_class: int = 8
    min_calibration_per_class: int = 4
    max_train_per_class: int = 2048
    max_calibration_per_class: int = 512

    probe_epochs: int = 300
    probe_lr: float = 1.0e-2
    probe_weight_decay: float = 1.0e-3
    probe_batch_size: int = 256
    density_weight: float = 0.50
    min_probe_probability: float = 0.60
    max_distance_norm: float = 2.0
    clip_min_evidence: float = 0.0
    query_background_quantile: float = 0.80

    stability_probes: int = 1
    stability_noise_std: float = 0.01
    stability_seed: int = 4321
    score_batch_size: int = 4096

    query_clip_accept_fraction: float = 0.30
    row_min_evidence: float = 0.10
    min_row_clips: int = 8
    unknown_label: str = "OutOfSample"
    batch_size: int = 8
    num_workers: int = 0
    progress: bool = True


@dataclass
class ProbeCalibration:
    labels: list[str]
    mode: str
    feature_mean: list[float]
    feature_std: list[float]
    class_scores: dict[str, list[float]]
    class_thresholds: dict[str, float]
    class_distance_scales: dict[str, float]
    summary: dict[str, Any]
    model_state: Optional[dict[str, Any]] = None


@dataclass
class ProbeClipVote:
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
    top_p_value: float
    top_probability: float
    top_evidence: float
    second_label: Optional[str]
    second_p_value: float
    second_probability: float
    evidence_margin: float
    probability_margin: float
    distance_norm: float
    stability_mean: float
    stability_max: float
    p_values: dict[str, float]
    probabilities: dict[str, float]
    evidence_by_label: dict[str, float]
    nonconformity_scores: dict[str, float]
    rejection_reason: str


@dataclass
class ProbeRowVote:
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
    top_p_value: float
    top_probability: float
    top_evidence: float
    second_label: Optional[str]
    second_evidence: float
    evidence_margin: float
    probability_margin: float
    distance_norm_mean: float
    stability_mean: float
    stability_max: float
    evidence_p25: float
    evidence_p50: float
    evidence_p75: float
    decision_margin: float
    uncertainty_score: float
    dominant_rejection_reason: str
    rejection_reason_counts: dict[str, int]
    p_values_mean_by_label: dict[str, float]
    evidence_mean_by_label: dict[str, float]


def _as_flat_latents(mu: torch.Tensor, transform: str) -> torch.Tensor:
    z = mu.detach().float().reshape(mu.shape[0], -1)
    return apply_latent_transform(z, transform)


@torch.no_grad()
def encode_with_stability(plan: Any, vae: Any, *, split: str, clip_len: int, cfg: ProbeStabilityConfig) -> Any:
    from ..data.clips import planned_clip_collate

    if split == "anchor":
        ds = plan.make_anchor_clip_dataset(clip_len=int(clip_len))
    else:
        ds = plan.make_query_clip_dataset(clip_len=int(clip_len))
    if len(ds) == 0:
        return SimpleNamespace(clips=[], embeddings=torch.empty((0, 0)), features=torch.empty((0, 0)), stability=torch.empty((0, 2)))
    loader = torch.utils.data.DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=int(cfg.num_workers), collate_fn=planned_clip_collate, pin_memory=torch.cuda.is_available())
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(cfg.stability_seed) + (17 if split == "query" else 0) + int(clip_len))
    z_parts: list[torch.Tensor] = []
    stab_parts: list[torch.Tensor] = []
    for batch in tqdm(loader, desc=f"encode {split} stability k={clip_len}", unit="batch", total=len(loader), disable=not cfg.progress):
        obs = batch["obs"].float().contiguous()
        z0 = _as_flat_latents(vae.encode_matrices(obs, clamp=True), cfg.latent_transform)
        n_probes = max(0, int(cfg.stability_probes))
        if n_probes > 0:
            running_sum = torch.zeros((z0.shape[0],), dtype=torch.float32)
            running_max = torch.zeros((z0.shape[0],), dtype=torch.float32)

            for _probe_i in range(n_probes):
                noise = torch.randn(obs.shape, generator=gen, dtype=obs.dtype) * float(cfg.stability_noise_std)
                noisy = (obs + noise).clamp(0.0, 1.0)
                zp = _as_flat_latents(vae.encode_matrices(noisy, clamp=True), cfg.latent_transform)
                d = torch.sqrt((zp - z0).square().mean(dim=1).clamp_min(0.0)).cpu()
                running_sum += d
                running_max = torch.maximum(running_max, d)
                del noise, noisy, zp, d
            stab = torch.stack([running_sum / float(n_probes), running_max], dim=1).float()
        else:
            stab = torch.zeros((z0.shape[0], 2), dtype=torch.float32)
        z_parts.append(z0.cpu())
        stab_parts.append(stab.cpu())
    Z = torch.cat(z_parts, dim=0).contiguous()
    S = torch.cat(stab_parts, dim=0).contiguous()
    Ftr = torch.cat([Z, S], dim=1).contiguous()
    return SimpleNamespace(clips=list(ds.clips), embeddings=Z, features=Ftr, stability=S)


def encode_feature_tables(plan: Any, vae: Any | None, cfg: ProbeStabilityConfig) -> dict[int, tuple[Any, Any]]:
    if cfg.latent_source == "vae" and int(cfg.stability_probes) > 0:
        if vae is None:
            raise ValueError("stability probes require VAE latents")
        out: dict[int, tuple[Any, Any]] = {}
        clip_lens = sorted({int(c.clip_len) for c in list(plan.anchor_clips) + list(plan.query_clips)})
        for clip_len in clip_lens:
            out[clip_len] = (
                encode_with_stability(plan, vae, split="anchor", clip_len=clip_len, cfg=cfg),
                encode_with_stability(plan, vae, split="query", clip_len=clip_len, cfg=cfg),
            )
        return out
    base = encode_clip_tables(plan, vae, latent_source=cfg.latent_source, latent_transform=cfg.latent_transform, batch_size=cfg.batch_size, num_workers=cfg.num_workers, progress=cfg.progress)
    out = {}
    for clip_len, (a, q) in base.items():
        a.features = a.embeddings
        q.features = q.embeddings
        a.stability = torch.zeros((a.embeddings.shape[0], 2), dtype=torch.float32)
        q.stability = torch.zeros((q.embeddings.shape[0], 2), dtype=torch.float32)
        out[clip_len] = (a, q)
    return out


def split_by_label(labels: Sequence[str], cfg: ProbeStabilityConfig) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(int(cfg.split_seed))
    train: list[int] = []
    cal: list[int] = []
    by: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by[str(lab)].append(int(i))
    for lab, idxs0 in sorted(by.items()):
        idxs = np.asarray(idxs0, dtype=np.int64)
        rng.shuffle(idxs)
        n = int(idxs.size)
        if n <= 1:
            train.extend(idxs.tolist())
            continue
        desired_cal = max(int(cfg.min_calibration_per_class), int(round(float(cfg.calibration_fraction) * n)))
        max_cal = max(0, n - max(1, int(cfg.min_train_per_class)))
        if max_cal >= int(cfg.min_calibration_per_class):
            cal_n = min(desired_cal, max_cal)
            cal.extend(idxs[:cal_n].tolist())
            train.extend(idxs[cal_n:].tolist())
        else:
            train.extend(idxs.tolist())
            cal.extend(idxs.tolist())
    return train, cal


def standardize_fit(X: torch.Tensor, train_idx: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    Xt = X[list(train_idx)].float()
    mu = Xt.mean(dim=0)
    sd = Xt.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    return mu, sd


def standardize_apply(X: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    return (X.float() - mu.view(1, -1)) / sd.view(1, -1).clamp_min(1.0e-6)


def train_linear_probe(X: torch.Tensor, y: torch.Tensor, *, n_classes: int, cfg: ProbeStabilityConfig) -> torch.nn.Module:
    model = torch.nn.Linear(int(X.shape[1]), int(n_classes))
    counts = torch.bincount(y.long(), minlength=int(n_classes)).float().clamp_min(1.0)
    weights = counts.sum() / counts
    weights = weights / weights.mean().clamp_min(EPS)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.probe_lr), weight_decay=float(cfg.probe_weight_decay))
    batch_size = max(1, min(int(cfg.probe_batch_size), int(X.shape[0])))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(cfg.split_seed) + 911)
    for _epoch in tqdm(range(max(1, int(cfg.probe_epochs))), desc="train linear probe", unit="epoch", disable=not cfg.progress):
        perm = torch.randperm(int(X.shape[0]), generator=gen)
        for start in range(0, int(X.shape[0]), batch_size):
            idx = perm[start:start + batch_size]
            logits = model(X[idx])
            loss = F.cross_entropy(logits, y[idx].long(), weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model.eval()


def _rms_dist_to_centroid(X: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((X - centroid.view(1, -1)).square().mean(dim=1).clamp_min(0.0))


def _p_value_lower_better(score: float, calibration_scores: Sequence[float]) -> float:
    vals = np.asarray([float(x) for x in calibration_scores if math.isfinite(float(x))], dtype=np.float64)
    if vals.size == 0 or not math.isfinite(float(score)):
        return 0.0
    return float((1.0 + np.sum(vals >= float(score))) / (1.0 + vals.size))


def _diverse_cap_indices(X: torch.Tensor, indices: Sequence[int], cap: int, seed: int) -> list[int]:
    idxs = list(map(int, indices)); cap = max(0, int(cap))
    if cap <= 0 or len(idxs) <= cap: return idxs
    candidates = torch.as_tensor(idxs, dtype=torch.long)
    Z = F.normalize(X[candidates].float(), p=2, dim=1, eps=EPS)
    centroid = F.normalize(Z.mean(dim=0, keepdim=True), p=2, dim=1, eps=EPS)[0]
    first = int(torch.argmax(Z @ centroid).item())
    chosen = [first]; mask = torch.zeros((len(idxs),), dtype=torch.bool); mask[first] = True
    min_dist = (1.0 - Z @ Z[first]).clamp_min(0.0); min_dist[first] = -1.0
    while len(chosen) < cap:
        avail = torch.nonzero(~mask, as_tuple=False).flatten()
        if avail.numel() == 0: break
        vals = min_dist[avail]; best = float(vals.max().item())
        tied = avail[torch.nonzero(vals >= best - 1e-12, as_tuple=False).flatten()]
        rel = int(tied[int(seed + len(chosen)) % int(tied.numel())].item())
        chosen.append(rel); mask[rel] = True
        min_dist = torch.minimum(min_dist, (1.0 - Z @ Z[rel]).clamp_min(0.0)); min_dist[mask] = -1.0
    return [idxs[i] for i in chosen]


def _partitioned_probe_indices(anchors: Any, X: torch.Tensor, labels: Sequence[str], cfg: ProbeStabilityConfig) -> tuple[list[int], list[int]]:
    by_label_partition: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, clip in enumerate(anchors.clips):
        by_label_partition[(str(labels[i]), anchor_partition(clip))].append(int(i))
    train: list[int] = []; cal: list[int] = []
    for lab in sorted(set(labels)):
        proper = by_label_partition.get((lab, PARTITION_PROPER), [])
        calibration = by_label_partition.get((lab, PARTITION_CALIBRATION), [])
        if not proper or not calibration:
            # Compatibility fallback
            fallback_train, fallback_cal = split_by_label([labels[i] for i in range(len(labels))], cfg)
            return fallback_train, fallback_cal
        proper = _diverse_cap_indices(X, proper, int(cfg.max_train_per_class), int(cfg.split_seed) + 101)
        calibration = _diverse_cap_indices(X, calibration, int(cfg.max_calibration_per_class), int(cfg.split_seed) + 211)
        if len(proper) < int(cfg.min_train_per_class):
            raise ValueError(f"probe label {lab!r} has {len(proper)} proper clips after cap; need {cfg.min_train_per_class}")
        if len(calibration) < int(cfg.min_calibration_per_class):
            raise ValueError(f"probe label {lab!r} has {len(calibration)} calibration clips after cap; need {cfg.min_calibration_per_class}")
        train.extend(proper); cal.extend(calibration)
    return train, cal


def fit_probe_calibration(anchors: Any, cfg: ProbeStabilityConfig) -> tuple[ProbeCalibration, dict[str, Any]]:
    X = anchors.features.float().cpu().contiguous()
    labels = [clip_label(c) for c in anchors.clips]
    label_names = sorted(set(labels))
    label_to_idx = {lab: i for i, lab in enumerate(label_names)}
    train_idx, cal_idx = _partitioned_probe_indices(anchors, X, labels, cfg)
    if not train_idx:
        raise ValueError("No training seed clips available")
    mu, sd = standardize_fit(X, train_idx)
    Xs = standardize_apply(X, mu, sd)
    y_all = torch.tensor([label_to_idx[x] for x in labels], dtype=torch.long)
    y_train = y_all[train_idx]
    X_train = Xs[train_idx]

    centroids: dict[str, torch.Tensor] = {}
    distance_scales: dict[str, float] = {}
    for lab in label_names:
        idxs = [i for i in train_idx if labels[i] == lab]
        if idxs:
            C = Xs[idxs].mean(dim=0)
        else:
            C = X_train.mean(dim=0)
        centroids[lab] = C
        dtrain = _rms_dist_to_centroid(Xs[idxs] if idxs else X_train, C)
        distance_scales[lab] = float(max(np.quantile(dtrain.detach().cpu().numpy(), 0.95), EPS)) if dtrain.numel() else 1.0

    class_scores: dict[str, list[float]] = {lab: [] for lab in label_names}
    thresholds: dict[str, float] = {}
    model_state: Optional[dict[str, Any]] = None
    mode = "linear_probe" if len(label_names) >= 2 else "one_class_centroid"

    if len(label_names) >= 2:
        model = train_linear_probe(X_train, y_train, n_classes=len(label_names), cfg=cfg)
        with torch.no_grad():
            probs_all = F.softmax(model(Xs), dim=1)
        W = model.weight.detach().cpu().clone()
        b = model.bias.detach().cpu().clone()
        model_state = {"weight": W.tolist(), "bias": b.tolist()}
        for ci in cal_idx:
            lab = labels[int(ci)]
            c = label_to_idx[lab]
            prob = float(probs_all[int(ci), c].item())
            dist = float(_rms_dist_to_centroid(Xs[int(ci):int(ci)+1], centroids[lab]).item()) / max(distance_scales[lab], EPS)
            score = float((1.0 - prob) + float(cfg.density_weight) * dist)
            class_scores[lab].append(score)
    else:
        lab = label_names[0]
        for ci in cal_idx:
            dist = float(_rms_dist_to_centroid(Xs[int(ci):int(ci)+1], centroids[lab]).item()) / max(distance_scales[lab], EPS)
            class_scores[lab].append(dist)

    for lab in label_names:
        vals = class_scores.get(lab, [])
        thresholds[lab] = quantile(vals, 1.0 - float(cfg.alpha), default=float("inf"))

    summary = {
        "mode": mode,
        "labels": label_names,
        "num_train": len(train_idx),
        "num_calibration": len(cal_idx),
        "class_train_counts": dict(Counter(labels[i] for i in train_idx)),
        "class_calibration_counts": dict(Counter(labels[i] for i in cal_idx)),
        "class_score_summary": {lab: summarize_values(class_scores[lab]) for lab in label_names},
        "class_thresholds": thresholds,
        "class_distance_scales": distance_scales,
        "source_row_disjoint_partitions": True,
        "max_train_per_class": int(cfg.max_train_per_class),
        "max_calibration_per_class": int(cfg.max_calibration_per_class),
    }
    calibration = ProbeCalibration(
        labels=label_names,
        mode=mode,
        feature_mean=mu.detach().cpu().tolist(),
        feature_std=sd.detach().cpu().tolist(),
        class_scores=class_scores,
        class_thresholds=thresholds,
        class_distance_scales=distance_scales,
        summary=summary,
        model_state=model_state,
    )
    aux = {"mu": mu, "sd": sd, "centroids": centroids, "label_to_idx": label_to_idx}
    if len(label_names) >= 2:
        aux["model"] = model
    return calibration, aux


def _score_query_probe(Xq: torch.Tensor, S: torch.Tensor, calib: ProbeCalibration, aux: dict[str, Any], cfg: ProbeStabilityConfig) -> tuple[list[dict[str, Any]], float]:
    labels = calib.labels; centroids = aux["centroids"]
    votes: list[dict[str, Any]] = []; step = max(1, int(cfg.score_batch_size))
    for start_i in tqdm(range(0, int(Xq.shape[0]), step), desc="score probe query batches", unit="batch", disable=not cfg.progress):
        end_i = min(int(Xq.shape[0]), start_i + step)
        Xs = standardize_apply(Xq[start_i:end_i].float(), aux["mu"], aux["sd"])
        if calib.mode == "linear_probe":
            with torch.no_grad(): probs = F.softmax(aux["model"](Xs), dim=1).detach().cpu().numpy()
        else:
            probs = np.zeros((int(Xs.shape[0]), len(labels)), dtype=np.float64); lab = labels[0]
            dist = _rms_dist_to_centroid(Xs, centroids[lab]) / max(float(calib.class_distance_scales[lab]), EPS)
            probs[:, 0] = torch.exp(-dist.square()).detach().cpu().numpy()
        for local_i in range(int(Xs.shape[0])):
            global_i = start_i + local_i; pvals: dict[str, float] = {}; evs: dict[str, float] = {}; scores: dict[str, float] = {}; dnorms: dict[str, float] = {}
            for li, lab in enumerate(labels):
                dist_norm = float(_rms_dist_to_centroid(Xs[local_i:local_i+1], centroids[lab]).item()) / max(float(calib.class_distance_scales[lab]), EPS)
                prob = float(probs[local_i, li]); score = float((1.0 - prob) + float(cfg.density_weight) * dist_norm) if calib.mode == "linear_probe" else float(dist_norm)
                pval = _p_value_lower_better(score, calib.class_scores.get(lab, [])); density_decay = math.exp(-float(cfg.density_weight) * max(0.0, dist_norm - 1.0))
                pvals[lab] = pval; evs[lab] = float(max(0.0, min(1.0, pval * prob * density_decay))); scores[lab] = score; dnorms[lab] = dist_norm
            order = sorted(evs.items(), key=lambda kv: (kv[1], pvals.get(kv[0], 0.0), kv[0]), reverse=True)
            top_label, top_ev = order[0] if order else (None, 0.0); second_label, second_ev = order[1] if len(order) > 1 else (None, 0.0)
            top_idx = labels.index(top_label) if top_label in labels else -1; second_idx = labels.index(second_label) if second_label in labels else -1
            top_prob = float(probs[local_i, top_idx]) if top_idx >= 0 else 0.0; second_prob = float(probs[local_i, second_idx]) if second_idx >= 0 else 0.0
            top_p = float(pvals.get(str(top_label), 0.0)) if top_label is not None else 0.0; second_p = float(pvals.get(str(second_label), 0.0)) if second_label is not None else 0.0
            dist_norm = float(dnorms.get(str(top_label), float("inf"))) if top_label is not None else float("inf")
            reasons: list[str] = []
            if top_p < float(cfg.alpha): reasons.append("low_p_value")
            if top_prob < float(cfg.min_probe_probability): reasons.append("low_probability")
            if dist_norm > float(cfg.max_distance_norm): reasons.append("high_density_distance")
            if float(top_ev) < float(cfg.clip_min_evidence): reasons.append("low_clip_evidence")
            in_sample = not reasons
            state = "probe_low_confidence" if not in_sample else ("multi_hit_ambiguous" if second_p >= float(cfg.alpha) and second_ev >= 0.95 * float(top_ev) and len(labels) > 1 else "single_hit")
            votes.append({
                "in_sample": in_sample, "call_state": state, "top_label": top_label, "top_p_value": top_p, "top_probability": top_prob, "top_evidence": float(top_ev),
                "second_label": second_label, "second_p_value": second_p, "second_probability": second_prob, "evidence_margin": float(top_ev - float(second_ev)),
                "probability_margin": float(top_prob - second_prob), "distance_norm": dist_norm,
                "stability_mean": float(S[global_i, 0].item()) if S.numel() else 0.0, "stability_max": float(S[global_i, 1].item()) if S.numel() else 0.0,
                "p_values": pvals, "probabilities": {lab: float(probs[local_i, li]) for li, lab in enumerate(labels)}, "evidence_by_label": evs,
                "nonconformity_scores": scores, "rejection_reason": "+".join(reasons) if reasons else "passed",
            })
        del Xs, probs
    bg_thr = 0.0; q = float(cfg.query_background_quantile)
    if 0.0 < q < 1.0 and votes:
        bg_thr = float(np.quantile(np.fromiter((float(v["top_evidence"]) for v in votes), dtype=np.float64), q))
        for v in votes:
            if bool(v["in_sample"]) and float(v["top_evidence"]) < bg_thr:
                v["in_sample"] = False; v["call_state"] = "query_background_rejected"; v["rejection_reason"] = "low_query_tail_evidence"
    return votes, bg_thr


def score_probe_for_len(anchors: Any, queries: Any, cfg: ProbeStabilityConfig) -> tuple[list[ProbeClipVote], dict[str, Any]]:
    if anchors.features.numel() == 0 or queries.features.numel() == 0:
        return [], {"empty": True}
    calibration, aux = fit_probe_calibration(anchors, cfg)
    vote_dicts, bg_thr = _score_query_probe(queries.features.float().cpu(), queries.stability.float().cpu(), calibration, aux, cfg)
    out: list[ProbeClipVote] = []
    for clip, vd in zip(queries.clips, vote_dicts):
        out.append(ProbeClipVote(
            query_clip_id=getattr(clip, "clip_id", ""),
            clip_len=int(getattr(clip, "clip_len", 0)),
            pool_id=getattr(clip, "pool_id", ""),
            key_ids=tuple(int(x) for x in getattr(clip, "key_ids", ())),
            rowids=tuple(int(x) for x in getattr(clip, "rowids", ())),
            subject_key_id=getattr(clip, "subject_key_id", None),
            subject_rowid=getattr(clip, "subject_rowid", None),
            subject_record_id=getattr(clip, "subject_record_id", None),
            **vd,
        ))
    summary = dict(calibration.summary)
    summary["query_background_evidence_threshold"] = bg_thr
    summary["query_top_evidence_summary"] = summarize_values([v.top_evidence for v in out])
    summary["query_distance_norm_summary"] = summarize_values([v.distance_norm for v in out])
    summary["query_rejection_reason_counts"] = dict(Counter(v.rejection_reason for v in out))
    summary["stability_summary"] = {
        "anchor_mean": summarize_values(anchors.stability[:, 0].tolist()) if getattr(anchors, "stability", None) is not None and anchors.stability.numel() else {},
        "query_mean": summarize_values(queries.stability[:, 0].tolist()) if getattr(queries, "stability", None) is not None and queries.stability.numel() else {},
    }
    return out, summary


def aggregate_probe_rows(plan: Any, clip_votes: Sequence[ProbeClipVote], cfg: ProbeStabilityConfig) -> list[ProbeRowVote]:
    row_to_pool = {int(a.rowid): a.pool_id for a in plan.query_assignments}
    acc: dict[int, dict[str, Any]] = {}
    for vote in tqdm(clip_votes, desc="aggregate probe rows", unit="clip", disable=not cfg.progress, leave=False):
        if vote.subject_rowid is not None:
            row_iter = [(int(vote.subject_rowid), int(vote.subject_key_id if vote.subject_key_id is not None else vote.key_ids[0]))]
        else:
            row_iter = list(zip(vote.rowids, vote.key_ids))
        for rid, kid in row_iter:
            entry = acc.setdefault(int(rid), {
                "key_id": int(kid), "rowid": int(rid), "clips_total": 0, "clips_in_sample": 0,
                "label_weights": defaultdict(float), "label_counts": Counter(), "states": Counter(), "rejection_reasons": Counter(),
                "top_ps": [], "top_probs": [], "evidences": [], "distance_norms": [], "stability_means": [], "stability_maxes": [],
                "p_values_sum": defaultdict(float), "p_values_count": defaultdict(int), "evidence_sum": defaultdict(float), "evidence_count": defaultdict(int),
            })
            entry["clips_total"] += 1
            entry["states"][vote.call_state] += 1
            entry["rejection_reasons"][str(vote.rejection_reason)] += 1
            entry["top_ps"].append(float(vote.top_p_value))
            entry["top_probs"].append(float(vote.top_probability))
            entry["evidences"].append(float(vote.top_evidence))
            entry["distance_norms"].append(float(vote.distance_norm))
            entry["stability_means"].append(float(vote.stability_mean))
            entry["stability_maxes"].append(float(vote.stability_max))
            for lab, p in vote.p_values.items():
                entry["p_values_sum"][lab] += float(p)
                entry["p_values_count"][lab] += 1
            for lab, ev in vote.evidence_by_label.items():
                entry["evidence_sum"][lab] += float(ev)
                entry["evidence_count"][lab] += 1
            if vote.in_sample:
                entry["clips_in_sample"] += 1
                if vote.top_label is not None:
                    entry["label_weights"][vote.top_label] += float(vote.top_evidence)
                    entry["label_counts"][vote.top_label] += 1
    for rec in plan.query_records:
        acc.setdefault(int(rec.rowid), {
            "key_id": int(rec.key_id), "rowid": int(rec.rowid), "clips_total": 0, "clips_in_sample": 0,
            "label_weights": defaultdict(float), "label_counts": Counter(), "states": Counter(),
            "top_ps": [], "top_probs": [], "evidences": [], "distance_norms": [], "stability_means": [], "stability_maxes": [],
            "p_values_sum": defaultdict(float), "p_values_count": defaultdict(int), "evidence_sum": defaultdict(float), "evidence_count": defaultdict(int),
        })
    rowid_to_record = {int(r.rowid): r for r in plan.query_records}
    out: list[ProbeRowVote] = []
    for rowid, entry in sorted(acc.items(), key=lambda kv: kv[0]):
        total = int(entry["clips_total"])
        ins = int(entry["clips_in_sample"])
        frac = float(ins / total) if total else 0.0
        evidences = np.asarray(entry.get("evidences", []), dtype=np.float64)
        top_ps = np.asarray(entry.get("top_ps", []), dtype=np.float64)
        top_probs = np.asarray(entry.get("top_probs", []), dtype=np.float64)
        dist = np.asarray(entry.get("distance_norms", []), dtype=np.float64)
        stab_m = np.asarray(entry.get("stability_means", []), dtype=np.float64)
        stab_x = np.asarray(entry.get("stability_maxes", []), dtype=np.float64)
        ev_mean_by_label = {lab: float(entry["evidence_sum"][lab] / max(1, entry["evidence_count"][lab])) for lab in sorted(entry["evidence_sum"])}
        p_mean_by_label = {lab: float(entry["p_values_sum"][lab] / max(1, entry["p_values_count"][lab])) for lab in sorted(entry["p_values_sum"])}
        weights = dict(entry["label_weights"])
        top_label = max(weights.items(), key=lambda kv: (kv[1], kv[0]))[0] if weights else (max(ev_mean_by_label.items(), key=lambda kv: (kv[1], kv[0]))[0] if ev_mean_by_label else None)
        ordered_ev = sorted(ev_mean_by_label.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        second_label, second_ev = ordered_ev[1] if len(ordered_ev) > 1 else (None, 0.0)
        top_ev = float(ordered_ev[0][1]) if ordered_ev else 0.0
        membership_score = float(evidences.mean()) if evidences.size else 0.0
        row_in_sample = bool(total >= int(cfg.min_row_clips) and frac >= float(cfg.query_clip_accept_fraction) and membership_score >= float(cfg.row_min_evidence))
        if total <= 0 or total < int(cfg.min_row_clips):
            state = "unscored"
        elif not row_in_sample:
            state = "probe_low_confidence"
        elif entry["states"].get("multi_hit_ambiguous", 0) >= max(1, int(math.ceil(float(cfg.query_clip_accept_fraction) * total))):
            state = "multi_hit_ambiguous"
        else:
            state = "single_hit"
        decision_margin = min(frac - float(cfg.query_clip_accept_fraction), membership_score - float(cfg.row_min_evidence)) if total >= int(cfg.min_row_clips) else -float(cfg.query_clip_accept_fraction)
        denom = max(float(cfg.query_clip_accept_fraction), float(cfg.row_min_evidence), 1.0 - float(cfg.query_clip_accept_fraction), EPS)
        uncertainty = float(max(0.0, 1.0 - min(1.0, abs(decision_margin) / denom)))
        rejection_counts = dict(entry.get("rejection_reasons", Counter()))
        dominant_rejection_reason = max(rejection_counts.items(), key=lambda kv: (kv[1], kv[0]))[0] if rejection_counts else "unscored"
        rec = rowid_to_record.get(int(rowid))
        record_id = rec.record_id if rec is not None else f"query:row{rowid}:key{entry['key_id']}"
        out.append(ProbeRowVote(
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
            label_weights={k: float(v) for k, v in weights.items()},
            label_counts={k: int(v) for k, v in dict(entry["label_counts"]).items()},
            membership_score=membership_score,
            membership_margin=float(decision_margin),
            top_p_value=float(top_ps.mean()) if top_ps.size else 0.0,
            top_probability=float(top_probs.mean()) if top_probs.size else 0.0,
            top_evidence=top_ev,
            second_label=second_label,
            second_evidence=float(second_ev),
            evidence_margin=float(top_ev - float(second_ev)),
            probability_margin=0.0,
            distance_norm_mean=float(dist.mean()) if dist.size else 0.0,
            stability_mean=float(stab_m.mean()) if stab_m.size else 0.0,
            stability_max=float(stab_x.max()) if stab_x.size else 0.0,
            evidence_p25=float(np.quantile(evidences, 0.25)) if evidences.size else 0.0,
            evidence_p50=float(np.quantile(evidences, 0.50)) if evidences.size else 0.0,
            evidence_p75=float(np.quantile(evidences, 0.75)) if evidences.size else 0.0,
            decision_margin=float(decision_margin),
            uncertainty_score=uncertainty,
            dominant_rejection_reason=str(dominant_rejection_reason),
            rejection_reason_counts={k: int(v) for k, v in rejection_counts.items()},
            p_values_mean_by_label=p_mean_by_label,
            evidence_mean_by_label=ev_mean_by_label,
        ))
    return out


def run_probe_stability(plan: Any, vae: Any | None, cfg: ProbeStabilityConfig, *, retain_clip_votes: bool = True) -> dict[str, Any]:
    tables = encode_feature_tables(plan, vae, cfg)
    released = release_plan_images(plan)
    all_clips: list[ProbeClipVote] = []
    by_len: dict[str, Any] = {}
    for clip_len, (anchors, queries) in tables.items():
        votes, summary = score_probe_for_len(anchors, queries, cfg)
        by_len[str(clip_len)] = summary
        all_clips.extend(votes)
    del tables
    gc.collect()
    rows = aggregate_probe_rows(plan, all_clips, cfg)
    summary = {
        "method": "calibrated_probe_with_density_and_stability",
        "config": asdict(cfg),
        "ingress": plan.to_summary_dict(),
        "num_query_clip_votes": len(all_clips),
        "num_query_clips_in_sample": int(sum(1 for v in all_clips if v.in_sample)),
        "num_query_rows": len(rows),
        "num_query_rows_in_sample": int(sum(1 for r in rows if r.in_sample)),
        "clip_call_state_counts": dict(Counter(v.call_state for v in all_clips)),
        "clip_rejection_reason_counts": dict(Counter(v.rejection_reason for v in all_clips)),
        "row_call_state_counts": dict(Counter(r.call_state for r in rows)),
        "row_dominant_rejection_reason_counts": dict(Counter(r.dominant_rejection_reason for r in rows)),
        "row_membership_score_summary": summarize_values([r.membership_score for r in rows]),
        "probe_by_clip_len": by_len,
        "released_plan_image_bytes": released,
        "stability_aggregation": "sequential_online_mean_max",
    }
    clip_votes_out = all_clips if retain_clip_votes else []
    if not retain_clip_votes:
        all_clips.clear()
        gc.collect()
    return {"config": cfg, "clip_votes": clip_votes_out, "row_votes": rows, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CFetch calibrated frozen-representation probe with density and optional stability features.")
    add_external_common_args(p)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--calibration-fraction", type=float, default=0.25)
    p.add_argument("--split-seed", type=int, default=1234)
    p.add_argument("--min-train-per-class", type=int, default=8)
    p.add_argument("--min-calibration-per-class", type=int, default=4)
    p.add_argument("--probe-max-train-per-class", type=int, default=int(os.environ.get("PROBE_MAX_TRAIN_PER_CLASS", "2048")))
    p.add_argument("--probe-max-calibration-per-class", type=int, default=int(os.environ.get("PROBE_MAX_CALIBRATION_PER_CLASS", "512")))
    p.add_argument("--probe-score-batch-size", type=int, default=int(os.environ.get("PROBE_SCORE_BATCH_SIZE", "4096")))
    p.add_argument("--probe-epochs", type=int, default=300)
    p.add_argument("--probe-lr", type=float, default=1.0e-2)
    p.add_argument("--probe-weight-decay", type=float, default=1.0e-3)
    p.add_argument("--probe-batch-size", type=int, default=256)
    p.add_argument("--density-weight", type=float, default=0.50)
    p.add_argument("--min-probe-probability", type=float, default=0.60)
    p.add_argument("--max-distance-norm", type=float, default=2.0)
    p.add_argument("--clip-min-evidence", type=float, default=0.0)
    p.add_argument("--query-background-quantile", type=float, default=0.80, help="Optional query-side evidence quantile gate, e.g. 0.90 keeps only the high-evidence tail.")
    p.add_argument("--stability-probes", type=int, default=1, help="Number of finite-difference noise probes appended to the frozen latent features. 0 disables.")
    p.add_argument("--stability-noise-std", type=float, default=0.01)
    p.add_argument("--stability-seed", type=int, default=4321)
    p.add_argument("--query-clip-accept-fraction", type=float, default=0.30)
    p.add_argument("--row-min-evidence", type=float, default=0.10)
    p.add_argument("--min-row-clips", type=int, default=8)
    p.add_argument("--unknown-label", default="OutOfSample")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    plan = build_external_ingress_plan(args)
    vae = load_vae_from_args(args)
    cfg = ProbeStabilityConfig(
        latent_source=str(args.latent_source),
        latent_transform=str(args.latent_transform),
        alpha=float(args.alpha),
        calibration_fraction=float(args.calibration_fraction),
        split_seed=int(args.split_seed),
        min_train_per_class=int(args.min_train_per_class),
        min_calibration_per_class=int(args.min_calibration_per_class),
        max_train_per_class=int(args.probe_max_train_per_class),
        max_calibration_per_class=int(args.probe_max_calibration_per_class),
        probe_epochs=int(args.probe_epochs),
        probe_lr=float(args.probe_lr),
        probe_weight_decay=float(args.probe_weight_decay),
        probe_batch_size=int(args.probe_batch_size),
        density_weight=float(args.density_weight),
        min_probe_probability=float(args.min_probe_probability),
        max_distance_norm=float(args.max_distance_norm),
        clip_min_evidence=float(args.clip_min_evidence),
        query_background_quantile=float(args.query_background_quantile),
        stability_probes=int(args.stability_probes),
        stability_noise_std=float(args.stability_noise_std),
        stability_seed=int(args.stability_seed),
        score_batch_size=int(args.probe_score_batch_size),
        query_clip_accept_fraction=float(args.query_clip_accept_fraction),
        row_min_evidence=float(args.row_min_evidence),
        min_row_clips=min(int(args.min_row_clips), int(plan.cfg.query_mixups_per_row)),
        unknown_label=str(args.unknown_label),
        batch_size=int(args.vote_batch_size),
        num_workers=int(args.num_workers),
        progress=bool(args.progress),
    )
    result = run_probe_stability(plan, vae, cfg, retain_clip_votes=not bool(args.no_clip_vote_output))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "probe_stability_summary.json", result["summary"])
    if not bool(args.no_clip_vote_output):
        write_jsonl(out / "query_clip_probe_votes.jsonl", (asdict(v) for v in result["clip_votes"]))
    write_row_tables(plan, result["row_votes"], out, prefix="probe", unknown_label=cfg.unknown_label)
    write_ingress_plan_streaming(plan, out / "ingress_plan", mode=str(args.ingress_plan_output))
    print({"status": "complete", "method": result["summary"].get("method"), "rows": result["summary"].get("num_query_rows"), "rows_in_sample": result["summary"].get("num_query_rows_in_sample"), "out_dir": str(out)}, flush=True)


if __name__ == "__main__":
    main()
