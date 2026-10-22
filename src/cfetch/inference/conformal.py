#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
from tqdm.auto import tqdm

import numpy as np
import torch

from .ingress import (
    PARTITION_CALIBRATION,
    PARTITION_PROPER,
    anchor_partition,
)
from .common import (
    EPS,
    add_external_common_args,
    build_external_ingress_plan,
    clip_label,
    encode_clip_tables,
    load_vae_from_args,
    quantile,
    release_plan_images,
    summarize_values,
    write_ingress_plan_streaming,
    write_json,
    write_jsonl,
    write_row_tables,
)




@dataclass(frozen=True)
class DecayConformalConfig:
    latent_source: str = "vae"
    latent_transform: str = "mu_varnorm"
    alpha: float = 0.05
    k: int = 7
    calibration_fraction: float = 0.25
    min_proper_anchors: int = 8
    min_calibration_anchors: int = 20
    split_seed: int = 1234

    decay_power: float = 2.0
    decay_scale: float = 0.0
    boundary_tau_fraction: float = 0.12
    evidence_mode: str = "p_times_exp_boundary"
    clip_min_evidence: float = 0.015

    query_clip_accept_fraction: float = 0.25
    row_min_evidence: float = 0.06
    min_row_clips: int = 8
    subject_evidence_weight: float = 0.75
    unknown_label: str = "OutOfSample"
    batch_size: int = 8
    num_workers: int = 0
    include_neighbor_details: bool = False
    progress: bool = True


@dataclass
class ClassCalibration:
    label: str
    proper_indices: list[int]
    calibration_indices: list[int]
    calibration_scores: list[float]
    threshold: float
    scale: float
    mode: str


@dataclass
class DecayConformalClipVote:
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
    rejection_reason: Optional[str]
    hit_labels: list[str]
    top_label: Optional[str]
    top_p_value: float
    second_label: Optional[str]
    second_p_value: float
    p_margin: float
    membership_evidence: float
    top_evidence: float
    second_evidence: float
    evidence_margin: float
    top_score: float
    top_threshold: float
    distance_margin: float
    p_values: dict[str, float]
    evidence_by_label: dict[str, float]
    nonconformity_scores: dict[str, float]
    threshold_by_label: dict[str, float]
    distance_weight_by_label: dict[str, float]
    boundary_weight_by_label: dict[str, float]
    best_distance: Optional[float]
    neighbors: list[dict[str, Any]]


@dataclass
class DecayConformalRowVote:
    record_id: str
    key_id: int
    rowid: int
    pool_id: str

    clips_total: int
    clips_in_sample: int
    in_sample_fraction: float

    subject_clips_total: int
    subject_clips_in_sample: int
    subject_in_sample_fraction: float
    subject_evidence_mean: float
    participating_clips_total: int
    participating_clips_in_sample: int
    participating_in_sample_fraction: float
    participating_evidence_mean: float
    combined_evidence: float
    subject_evidence_weight: float

    in_sample: bool
    call_state: str
    rejection_reasons: list[str]
    dominant_rejection_reason: Optional[str]
    top_label: Optional[str]
    label_weights: dict[str, float]
    label_counts: dict[str, int]
    call_state_counts: dict[str, int]
    membership_score: float
    membership_margin: float
    top_p_value: float
    top_evidence: float
    second_label: Optional[str]
    second_evidence: float
    evidence_margin: float
    p_value_mean: float
    p_value_p50: float
    evidence_p25: float
    evidence_p50: float
    evidence_p75: float
    distance_margin_mean: float
    decision_margin: float
    uncertainty_score: float
    conformal_membership_support: float
    conformal_specificity_margin: float
    conformal_specificity_ratio: float
    conformal_preferred_label: Optional[str]
    conformal_preferred_p_value: float
    conformal_preferred_evidence: float
    p_values_mean_by_label: dict[str, float]
    evidence_mean_by_label: dict[str, float]


def _rms_l2(A: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if A.numel() == 0:
        return torch.empty((0,), dtype=torch.float32)
    return torch.sqrt((A - q.view(1, -1)).square().mean(dim=1).clamp_min(0.0))


def _kth_distance(q: torch.Tensor, A: torch.Tensor, k: int) -> tuple[float, int, list[tuple[int, float]]]:
    d = _rms_l2(A, q)
    if d.numel() == 0:
        return float("inf"), -1, []
    order = torch.argsort(d)
    j = min(max(1, int(k)) - 1, int(order.numel()) - 1)
    kth_idx = int(order[j].item())
    neigh = [(int(i.item()), float(d[int(i)].item())) for i in order[: min(10, int(order.numel()))]]
    return float(d[kth_idx].item()), kth_idx, neigh


def _conformal_p_value(score: float, calibration_scores: Sequence[float]) -> float:
    vals = np.asarray([float(x) for x in calibration_scores if math.isfinite(float(x))], dtype=np.float64)
    if vals.size == 0 or not math.isfinite(float(score)):
        return 0.0
    return float((1.0 + np.sum(vals >= float(score))) / (1.0 + vals.size))


def _calibration_threshold(scores: Sequence[float], alpha: float) -> float:
    vals = [float(x) for x in scores if math.isfinite(float(x))]
    return quantile(vals, 1.0 - float(alpha), default=float("inf")) if vals else float("inf")


def _calibration_scale(scores: Sequence[float], threshold: float) -> float:
    vals = np.asarray([float(x) for x in scores if math.isfinite(float(x)) and float(x) > 0], dtype=np.float64)
    if math.isfinite(threshold) and threshold > EPS:
        return float(threshold)
    return float(np.quantile(vals, 0.50)) if vals.size else 1.0


def _score_to_class(
    q: torch.Tensor,
    A: torch.Tensor,
    proper_indices: Sequence[int],
    k: int,
) -> tuple[float, Optional[float], list[tuple[int, float]]]:
    if not proper_indices:
        return float("inf"), None, []
    Aprop = A[list(proper_indices)].float()
    score, _rel_idx, neigh = _kth_distance(q.float(), Aprop, int(k))
    best = neigh[0][1] if neigh else None
    mapped = [(int(proper_indices[i]), float(d)) for i, d in neigh]
    return score, best, mapped


def fit_decay_conformal_calibration(anchors: Any, cfg: DecayConformalConfig) -> dict[str, Any]:
    """Fit calibration using source-row-disjoint proper/calibration anchor clips."""
    A = anchors.embeddings.float().cpu().contiguous()
    by_label_partition: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    rowids_by_label_partition: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    for i, clip in enumerate(anchors.clips):
        label = clip_label(clip)
        partition = anchor_partition(clip)
        by_label_partition[label][partition].append(int(i))
        rowids_by_label_partition[label][partition].update(int(x) for x in getattr(clip, "rowids", ()))

    calibrations: dict[str, ClassCalibration] = {}
    summaries: dict[str, Any] = {}
    errors: list[str] = []

    for label in sorted(by_label_partition):
        proper_idx = list(by_label_partition[label].get(PARTITION_PROPER, []))
        cal_idx = list(by_label_partition[label].get(PARTITION_CALIBRATION, []))
        proper_rows = rowids_by_label_partition[label].get(PARTITION_PROPER, set())
        cal_rows = rowids_by_label_partition[label].get(PARTITION_CALIBRATION, set())
        shared_rows = proper_rows & cal_rows

        if shared_rows:
            errors.append(f"{label}: proper/calibration share {len(shared_rows)} source rows")
        min_proper = max(int(cfg.min_proper_anchors), int(cfg.k))
        if len(proper_idx) < min_proper:
            errors.append(f"{label}: proper anchor clips={len(proper_idx)} < required {min_proper}")
        if len(cal_idx) < int(cfg.min_calibration_anchors):
            errors.append(f"{label}: calibration anchor clips={len(cal_idx)} < required {int(cfg.min_calibration_anchors)}")

        scores: list[float] = []
        if proper_idx and cal_idx:
            for ci in cal_idx:
                score, _best, _neigh = _score_to_class(A[int(ci)], A, proper_idx, int(cfg.k))
                if math.isfinite(score):
                    scores.append(float(score))
        if len(scores) < int(cfg.min_calibration_anchors):
            errors.append(f"{label}: usable calibration scores={len(scores)} < required {int(cfg.min_calibration_anchors)}")

        threshold = _calibration_threshold(scores, float(cfg.alpha))
        scale = float(cfg.decay_scale) if float(cfg.decay_scale) > 0 else _calibration_scale(scores, threshold)
        calibrations[label] = ClassCalibration(
            label=label,
            proper_indices=proper_idx,
            calibration_indices=cal_idx,
            calibration_scores=scores,
            threshold=threshold,
            scale=scale,
            mode="source_row_disjoint",
        )
        summaries[label] = {
            "mode": "source_row_disjoint",
            "num_proper": len(proper_idx),
            "num_calibration": len(cal_idx),
            "proper_source_rows": len(proper_rows),
            "calibration_source_rows": len(cal_rows),
            "cross_partition_shared_source_rows": len(shared_rows),
            "num_calibration_scores": len(scores),
            "minimum_resolvable_p_value": float(1.0 / (len(scores) + 1.0)) if scores else 1.0,
            "acceptance_threshold_approx": threshold,
            "decay_scale": scale,
            "score_summary": summarize_values(scores),
        }

    if errors:
        raise ValueError("Mach7 conformal calibration invariants failed:\n  - " + "\n  - ".join(errors))
    return {"calibrations": calibrations, "summary": summaries}


def _sigmoid(x: float) -> float:
    if x >= 40.0:
        return 1.0
    if x <= -40.0:
        return 0.0
    return float(1.0 / (1.0 + math.exp(-x)))


def decayed_evidence(
    score: float,
    p_value: float,
    calib: ClassCalibration,
    cfg: DecayConformalConfig,
) -> tuple[float, float, float, float]:
    if not math.isfinite(float(score)) or float(p_value) <= 0.0:
        return 0.0, float("-inf"), 0.0, 0.0
    scale = max(float(calib.scale), EPS)
    threshold = float(calib.threshold) if math.isfinite(float(calib.threshold)) else scale
    margin = float(threshold - float(score))
    distance_weight = math.exp(-float((max(score, 0.0) / scale) ** max(float(cfg.decay_power), EPS)))
    tau = max(float(cfg.boundary_tau_fraction) * max(abs(threshold), scale, EPS), EPS)
    boundary_weight = _sigmoid(margin / tau)
    mode = str(cfg.evidence_mode).lower()
    if mode in {"p", "p_value", "p_only"}:
        evidence = float(p_value)
    elif mode in {"p_times_exp", "exp"}:
        evidence = float(p_value) * distance_weight
    elif mode in {"p_times_boundary", "boundary"}:
        evidence = float(p_value) * boundary_weight
    elif mode in {"p_times_exp_boundary", "default"}:
        evidence = float(p_value) * distance_weight * boundary_weight
    else:
        raise ValueError("evidence_mode must be p, p_times_exp, p_times_boundary, or p_times_exp_boundary")
    return float(max(0.0, min(1.0, evidence))), margin, float(distance_weight), float(boundary_weight)


def score_decay_conformal_clip_tables(
    anchors: Any,
    queries: Any,
    cfg: DecayConformalConfig,
    calibration: dict[str, Any],
) -> list[DecayConformalClipVote]:
    if len(anchors.clips) == 0 or anchors.embeddings.shape[0] == 0:
        return []
    if len(queries.clips) == 0 or queries.embeddings.shape[0] == 0:
        return []

    A = anchors.embeddings.float().cpu().contiguous()
    Q = queries.embeddings.float().cpu().contiguous()
    calibrations: dict[str, ClassCalibration] = calibration["calibrations"]
    votes: list[DecayConformalClipVote] = []

    iterator = zip(queries.clips, Q)
    for clip, q in tqdm(
        iterator,
        desc="score Mach7 decayed conformal clips",
        unit="clip",
        total=len(queries.clips),
        disable=not cfg.progress,
    ):
        pvals: dict[str, float] = {}
        evs: dict[str, float] = {}
        scores: dict[str, float] = {}
        thresholds: dict[str, float] = {}
        distance_margins: dict[str, float] = {}
        distance_weights: dict[str, float] = {}
        boundary_weights: dict[str, float] = {}
        best_distances: list[float] = []
        neighbors: list[dict[str, Any]] = []

        for label, calib in sorted(calibrations.items()):
            score, best, neigh = _score_to_class(q, A, calib.proper_indices, int(cfg.k))
            pval = _conformal_p_value(score, calib.calibration_scores)
            evidence, margin, distance_weight, boundary_weight = decayed_evidence(score, pval, calib, cfg)
            pvals[label] = float(pval)
            evs[label] = float(evidence)
            scores[label] = float(score) if math.isfinite(score) else float("inf")
            thresholds[label] = float(calib.threshold)
            distance_margins[label] = float(margin)
            distance_weights[label] = float(distance_weight)
            boundary_weights[label] = float(boundary_weight)
            if best is not None:
                best_distances.append(float(best))
            if cfg.include_neighbor_details:
                for rank, (ai, dist) in enumerate(neigh[:5], start=1):
                    anchor = anchors.clips[int(ai)]
                    neighbors.append({
                        "label": label,
                        "rank": rank,
                        "anchor_clip_id": getattr(anchor, "clip_id", ""),
                        "anchor_label": clip_label(anchor),
                        "anchor_partition": anchor_partition(anchor),
                        "anchor_pool_id": getattr(anchor, "pool_id", ""),
                        "anchor_key_ids": list(getattr(anchor, "key_ids", ())),
                        "anchor_rowids": list(getattr(anchor, "rowids", ())),
                        "distance": float(dist),
                        "class_score": float(score) if math.isfinite(score) else float("inf"),
                        "class_threshold": float(calib.threshold),
                        "distance_weight": float(distance_weight),
                        "boundary_weight": float(boundary_weight),
                    })

        hit_labels = sorted(
            [label for label, p in pvals.items() if float(p) >= float(cfg.alpha)],
            key=lambda label: (evs.get(label, 0.0), pvals.get(label, 0.0), label),
            reverse=True,
        )
        global_rank = sorted(
            evs,
            key=lambda label: (evs.get(label, 0.0), pvals.get(label, 0.0), label),
            reverse=True,
        )
        preferred_label = hit_labels[0] if hit_labels else (global_rank[0] if global_rank else None)
        second_label = (
            hit_labels[1]
            if len(hit_labels) > 1
            else (next((label for label in global_rank if label != preferred_label), None))
        )
        membership_evidence = float(min(1.0, sum(max(0.0, evs.get(label, 0.0)) for label in hit_labels)))

        if not hit_labels:
            base_state = "no_hit_out_of_sample"
        elif len(hit_labels) == 1:
            base_state = "single_hit"
        else:
            base_state = "multi_hit_ambiguous"
        in_sample = bool(hit_labels and membership_evidence >= float(cfg.clip_min_evidence))
        if not hit_labels:
            call_state = base_state
            rejection_reason = "p_value"
        elif not in_sample:
            call_state = "weak_evidence_hit"
            rejection_reason = "evidence"
        else:
            call_state = base_state
            rejection_reason = None

        top_p = float(pvals.get(str(preferred_label), 0.0)) if preferred_label is not None else 0.0
        second_p = float(pvals.get(str(second_label), 0.0)) if second_label is not None else 0.0
        top_ev = float(evs.get(str(preferred_label), 0.0)) if preferred_label is not None else 0.0
        second_ev = float(evs.get(str(second_label), 0.0)) if second_label is not None else 0.0
        top_score = float(scores.get(str(preferred_label), float("inf"))) if preferred_label is not None else float("inf")
        top_threshold = float(thresholds.get(str(preferred_label), float("nan"))) if preferred_label is not None else float("nan")
        dist_margin = float(distance_margins.get(str(preferred_label), 0.0)) if preferred_label is not None else 0.0

        votes.append(DecayConformalClipVote(
            query_clip_id=getattr(clip, "clip_id", ""),
            clip_len=int(getattr(clip, "clip_len", 0)),
            pool_id=getattr(clip, "pool_id", ""),
            key_ids=tuple(int(x) for x in getattr(clip, "key_ids", ())),
            rowids=tuple(int(x) for x in getattr(clip, "rowids", ())),
            subject_key_id=getattr(clip, "subject_key_id", None),
            subject_rowid=getattr(clip, "subject_rowid", None),
            subject_record_id=getattr(clip, "subject_record_id", None),
            in_sample=in_sample,
            call_state=call_state,
            rejection_reason=rejection_reason,
            hit_labels=list(hit_labels),
            top_label=preferred_label,
            top_p_value=top_p,
            second_label=second_label,
            second_p_value=second_p,
            p_margin=float(top_p - second_p),
            membership_evidence=membership_evidence if in_sample else 0.0,
            top_evidence=top_ev,
            second_evidence=second_ev,
            evidence_margin=float(top_ev - second_ev),
            top_score=top_score,
            top_threshold=top_threshold,
            distance_margin=dist_margin,
            p_values=pvals,
            evidence_by_label=evs,
            nonconformity_scores=scores,
            threshold_by_label=thresholds,
            distance_weight_by_label=distance_weights,
            boundary_weight_by_label=boundary_weights,
            best_distance=min(best_distances) if best_distances else None,
            neighbors=neighbors,
        ))
    return votes


def _new_ledger() -> dict[str, Any]:
    return {
        "total": 0,
        "hits": 0,
        "membership_evidence": [],
        "top_ps": [],
        "top_evidences": [],
        "distance_margins": [],
        "label_weights": defaultdict(float),
        "label_counts": Counter(),
        "call_state_counts": Counter(),
        "p_values_sum": defaultdict(float),
        "p_values_count": defaultdict(int),
        "evidence_sum": defaultdict(float),
        "evidence_count": defaultdict(int),
    }


def _update_ledger(ledger: dict[str, Any], vote: DecayConformalClipVote) -> None:
    ledger["total"] += 1
    ledger["call_state_counts"][vote.call_state] += 1
    ledger["top_ps"].append(float(vote.top_p_value))
    ledger["top_evidences"].append(float(vote.top_evidence))
    ledger["distance_margins"].append(float(vote.distance_margin))
    ledger["membership_evidence"].append(float(vote.membership_evidence) if vote.in_sample else 0.0)
    for label, p in vote.p_values.items():
        ledger["p_values_sum"][label] += float(p)
        ledger["p_values_count"][label] += 1
    for label, ev in vote.evidence_by_label.items():
        ledger["evidence_sum"][label] += float(ev)
        ledger["evidence_count"][label] += 1
    if vote.in_sample:
        ledger["hits"] += 1
        # Ambiguous clips support membership but do not force a class label.
        if vote.call_state == "single_hit" and vote.top_label is not None:
            ledger["label_weights"][vote.top_label] += float(vote.top_evidence)
            ledger["label_counts"][vote.top_label] += 1


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _ledger_label_means(ledger: dict[str, Any], key: str, count_key: str) -> dict[str, float]:
    return {
        label: float(ledger[key][label] / max(1, ledger[count_key][label]))
        for label in sorted(ledger[key])
    }


def _blend(a: float, b: float, weight: float, *, a_available: bool, b_available: bool) -> float:
    if a_available and b_available:
        return float(weight * a + (1.0 - weight) * b)
    if a_available:
        return float(a)
    return float(b) if b_available else 0.0


def _blend_dicts(a: dict[str, float], b: dict[str, float], weight: float, *, a_available: bool, b_available: bool) -> dict[str, float]:
    labels = sorted(set(a) | set(b))
    return {
        label: _blend(a.get(label, 0.0), b.get(label, 0.0), weight, a_available=a_available, b_available=b_available)
        for label in labels
    }


def aggregate_decay_conformal_rows(
    plan: Any,
    clip_votes: Sequence[DecayConformalClipVote],
    cfg: DecayConformalConfig,
) -> list[DecayConformalRowVote]:
    """Aggregate controlled subject tests plus normalized all-participant evidence."""
    row_to_pool = {int(a.rowid): a.pool_id for a in plan.query_assignments}
    rowid_to_record = {int(r.rowid): r for r in plan.query_records}
    acc: dict[int, dict[str, Any]] = {
        int(rec.rowid): {
            "key_id": int(rec.key_id),
            "subject": _new_ledger(),
            "participant": _new_ledger(),
        }
        for rec in plan.query_records
    }

    for vote in tqdm(
        clip_votes,
        desc="aggregate Mach7 conformal rows",
        unit="clip",
        disable=not cfg.progress,
        leave=False,
    ):
        if vote.subject_rowid is not None:
            subject_rid = int(vote.subject_rowid)
            subject_kid = int(vote.subject_key_id if vote.subject_key_id is not None else vote.key_ids[0])
            entry = acc.setdefault(subject_rid, {"key_id": subject_kid, "subject": _new_ledger(), "participant": _new_ledger()})
            _update_ledger(entry["subject"], vote)

        for rid, kid in dict(zip(vote.rowids, vote.key_ids)).items():
            rid_i = int(rid)
            entry = acc.setdefault(rid_i, {"key_id": int(kid), "subject": _new_ledger(), "participant": _new_ledger()})
            _update_ledger(entry["participant"], vote)

    out: list[DecayConformalRowVote] = []
    w = min(1.0, max(0.0, float(cfg.subject_evidence_weight)))

    for rowid, entry in sorted(acc.items()):
        subject = entry["subject"]
        participant = entry["participant"]
        subject_total = int(subject["total"])
        subject_hits = int(subject["hits"])
        participant_total = int(participant["total"])
        participant_hits = int(participant["hits"])
        subject_frac = float(subject_hits / subject_total) if subject_total else 0.0
        participant_frac = float(participant_hits / participant_total) if participant_total else 0.0
        subject_ev = _mean(subject["membership_evidence"])
        participant_ev = _mean(participant["membership_evidence"])
        combined_ev = _blend(
            subject_ev,
            participant_ev,
            w,
            a_available=subject_total > 0,
            b_available=participant_total > 0,
        )

        rejection_reasons: list[str] = []
        if subject_total < int(cfg.min_row_clips):
            rejection_reasons.append("insufficient_subject_clips")
        if subject_frac < float(cfg.query_clip_accept_fraction):
            rejection_reasons.append("low_subject_accept_fraction")
        if combined_ev < float(cfg.row_min_evidence):
            rejection_reasons.append("low_combined_evidence")
        row_in_sample = not rejection_reasons
        dominant_reason = rejection_reasons[0] if rejection_reasons else None

        subject_p = _ledger_label_means(subject, "p_values_sum", "p_values_count")
        participant_p = _ledger_label_means(participant, "p_values_sum", "p_values_count")
        p_mean_by_label = _blend_dicts(
            subject_p,
            participant_p,
            w,
            a_available=subject_total > 0,
            b_available=participant_total > 0,
        )
        subject_label_ev = _ledger_label_means(subject, "evidence_sum", "evidence_count")
        participant_label_ev = _ledger_label_means(participant, "evidence_sum", "evidence_count")
        ev_mean_by_label = _blend_dicts(
            subject_label_ev,
            participant_label_ev,
            w,
            a_available=subject_total > 0,
            b_available=participant_total > 0,
        )

        subject_weights = dict(subject["label_weights"])
        participant_weights = dict(participant["label_weights"])
        labels = sorted(set(subject_weights) | set(participant_weights))
        label_weights = {
            label: _blend(
                float(subject_weights.get(label, 0.0)) / max(1, subject_total),
                float(participant_weights.get(label, 0.0)) / max(1, participant_total),
                w,
                a_available=subject_total > 0,
                b_available=participant_total > 0,
            )
            for label in labels
        }
        label_counts = dict(participant["label_counts"])

        ordered_ev = sorted(ev_mean_by_label.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
        preferred_label = ordered_ev[0][0] if ordered_ev else (max(label_weights, key=label_weights.get) if label_weights else None)
        top_ev = float(ordered_ev[0][1]) if ordered_ev else 0.0
        second_label, second_ev = ordered_ev[1] if len(ordered_ev) > 1 else (None, 0.0)
        specificity_margin = float(top_ev - float(second_ev))
        specificity_ratio = float(top_ev / max(float(second_ev), EPS)) if second_ev > 0.0 else (float("inf") if top_ev > 0.0 else 0.0)

        subject_states = dict(subject["call_state_counts"])
        if subject_total < int(cfg.min_row_clips):
            state = "unscored"
        elif not row_in_sample:
            state = "no_hit_out_of_sample"
        else:
            ambiguous_hits = int(subject_states.get("multi_hit_ambiguous", 0))
            single_hits = int(subject_states.get("single_hit", 0))
            state = "multi_hit_ambiguous" if ambiguous_hits > 0 and ambiguous_hits >= single_hits else "single_hit"

        fraction_margin = subject_frac - float(cfg.query_clip_accept_fraction)
        evidence_margin_to_gate = combined_ev - float(cfg.row_min_evidence)
        decision_margin = min(fraction_margin, evidence_margin_to_gate) if subject_total >= int(cfg.min_row_clips) else -1.0
        norm = max(float(cfg.query_clip_accept_fraction), float(cfg.row_min_evidence), 1.0 - float(cfg.query_clip_accept_fraction), EPS)
        uncertainty = float(max(0.0, 1.0 - min(1.0, abs(decision_margin) / norm)))

        subject_top_ps = np.asarray(subject["top_ps"], dtype=np.float64)
        subject_top_evs = np.asarray(subject["top_evidences"], dtype=np.float64)
        subject_dist = np.asarray(subject["distance_margins"], dtype=np.float64)
        rec = rowid_to_record.get(int(rowid))
        record_id = rec.record_id if rec is not None else f"query:row{rowid}:key{entry['key_id']}"

        out.append(DecayConformalRowVote(
            record_id=record_id,
            key_id=int(entry["key_id"]),
            rowid=int(rowid),
            pool_id=row_to_pool.get(int(rowid), "unassigned"),
            clips_total=subject_total,
            clips_in_sample=subject_hits,
            in_sample_fraction=subject_frac,
            subject_clips_total=subject_total,
            subject_clips_in_sample=subject_hits,
            subject_in_sample_fraction=subject_frac,
            subject_evidence_mean=subject_ev,
            participating_clips_total=participant_total,
            participating_clips_in_sample=participant_hits,
            participating_in_sample_fraction=participant_frac,
            participating_evidence_mean=participant_ev,
            combined_evidence=combined_ev,
            subject_evidence_weight=w,
            in_sample=row_in_sample,
            call_state=state,
            rejection_reasons=rejection_reasons,
            dominant_rejection_reason=dominant_reason,
            top_label=preferred_label,
            label_weights={k: float(v) for k, v in label_weights.items()},
            label_counts={k: int(v) for k, v in label_counts.items()},
            call_state_counts={k: int(v) for k, v in subject_states.items()},
            membership_score=combined_ev,
            membership_margin=float(decision_margin),
            top_p_value=float(subject_top_ps.mean()) if subject_top_ps.size else 0.0,
            top_evidence=top_ev,
            second_label=second_label,
            second_evidence=float(second_ev),
            evidence_margin=specificity_margin,
            p_value_mean=float(subject_top_ps.mean()) if subject_top_ps.size else 0.0,
            p_value_p50=float(np.quantile(subject_top_ps, 0.50)) if subject_top_ps.size else 0.0,
            evidence_p25=float(np.quantile(subject_top_evs, 0.25)) if subject_top_evs.size else 0.0,
            evidence_p50=float(np.quantile(subject_top_evs, 0.50)) if subject_top_evs.size else 0.0,
            evidence_p75=float(np.quantile(subject_top_evs, 0.75)) if subject_top_evs.size else 0.0,
            distance_margin_mean=float(subject_dist.mean()) if subject_dist.size else 0.0,
            decision_margin=float(decision_margin),
            uncertainty_score=uncertainty,
            conformal_membership_support=combined_ev,
            conformal_specificity_margin=specificity_margin,
            conformal_specificity_ratio=specificity_ratio,
            conformal_preferred_label=preferred_label,
            conformal_preferred_p_value=float(p_mean_by_label.get(str(preferred_label), 0.0)) if preferred_label is not None else 0.0,
            conformal_preferred_evidence=float(ev_mean_by_label.get(str(preferred_label), 0.0)) if preferred_label is not None else 0.0,
            p_values_mean_by_label=p_mean_by_label,
            evidence_mean_by_label=ev_mean_by_label,
        ))
    return out


def run_decay_conformal(plan: Any, vae: Any | None, cfg: DecayConformalConfig, *, retain_clip_votes: bool = True) -> dict[str, Any]:
    tables = encode_clip_tables(
        plan,
        vae,
        latent_source=cfg.latent_source,
        latent_transform=cfg.latent_transform,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        progress=cfg.progress,
    )
    released = release_plan_images(plan)
    all_clips: list[DecayConformalClipVote] = []
    calibration_by_len: dict[str, Any] = {}
    for clip_len, (anchors, queries) in tables.items():
        calibration = fit_decay_conformal_calibration(anchors, cfg)
        calibration_by_len[str(clip_len)] = calibration["summary"]
        all_clips.extend(score_decay_conformal_clip_tables(anchors, queries, cfg, calibration))

    del tables
    gc.collect()
    rows = aggregate_decay_conformal_rows(plan, all_clips, cfg)
    rejection_counter: Counter[str] = Counter()
    for row in rows:
        rejection_counter.update(row.rejection_reasons)

    ingress_summary = plan.to_summary_dict()
    ingress_diag = getattr(plan, "mach7_ingress_diagnostics", {})
    anchor_diag = ingress_diag.get("anchor_combination_diagnostics", {}) if isinstance(ingress_diag, dict) else {}
    ingress_summary["mach7"] = {
        "scale_profile": ingress_diag.get("scale_profile", {}),
        "seed_diversity_selection": {k: v for k, v in ingress_diag.get("seed_diversity_selection", {}).items() if k != "by_pool"},
        "source_row_split": ingress_diag.get("source_row_split", {}),
        "anchor_global_budgets": anchor_diag.get("global_budgets", {}),
        "cross_partition_shared_source_rows": anchor_diag.get("cross_partition_shared_source_rows", 0),
        "proper_anchor_clips": anchor_diag.get("proper_anchor_clips", 0),
        "calibration_anchor_clips": anchor_diag.get("calibration_anchor_clips", 0),
    }
    summary = {
        "suite": "mach7",
        "method": "mach7_decayed_conformal_knn",
        "config": asdict(cfg),
        "ingress": ingress_summary,
        "num_query_clip_votes": len(all_clips),
        "num_query_clips_in_sample": int(sum(1 for v in all_clips if v.in_sample)),
        "num_query_rows": len(rows),
        "num_query_rows_in_sample": int(sum(1 for r in rows if r.in_sample)),
        "clip_call_state_counts": dict(Counter(v.call_state for v in all_clips)),
        "clip_rejection_reason_counts": dict(Counter(str(v.rejection_reason or "accepted") for v in all_clips)),
        "row_call_state_counts": dict(Counter(r.call_state for r in rows)),
        "rejection_reason_counts": dict(rejection_counter),
        "row_membership_score_summary": summarize_values([r.membership_score for r in rows]),
        "row_subject_evidence_summary": summarize_values([r.subject_evidence_mean for r in rows]),
        "row_participating_evidence_summary": summarize_values([r.participating_evidence_mean for r in rows]),
        "row_conformal_specificity_margin_summary": summarize_values([r.conformal_specificity_margin for r in rows]),
        "row_conformal_specificity_ratio_summary": summarize_values([r.conformal_specificity_ratio for r in rows]),
        "row_preferred_label_counts": dict(Counter(str(r.conformal_preferred_label or "None") for r in rows)),
        "conformal_calibration_by_clip_len": calibration_by_len,
        "released_plan_image_bytes": released,
    }
    clip_votes_out = all_clips if retain_clip_votes else []
    if not retain_clip_votes:
        all_clips.clear()
        gc.collect()
    return {"config": cfg, "clip_votes": clip_votes_out, "row_votes": rows, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CFetch decayed class-conditional conformal kNN assignment."
    )
    add_external_common_args(p)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--conformal-k", type=int, default=7)
    p.add_argument("--calibration-fraction", type=float, default=0.25)
    p.add_argument("--min-proper-anchors", type=int, default=8)
    p.add_argument("--min-calibration-anchors", type=int, default=20)
    p.add_argument("--split-seed", type=int, default=1234)
    p.add_argument("--decay-power", type=float, default=2.0)
    p.add_argument("--decay-scale", type=float, default=0.0, help="<=0 uses the class calibration threshold/median.")
    p.add_argument("--boundary-tau-fraction", type=float, default=0.12)
    p.add_argument("--evidence-mode", default="p_times_exp_boundary", choices=["p", "p_times_exp", "p_times_boundary", "p_times_exp_boundary"])
    p.add_argument("--clip-min-evidence", type=float, default=0.015)
    p.add_argument("--query-clip-accept-fraction", type=float, default=0.25)
    p.add_argument("--row-min-evidence", type=float, default=0.06)
    p.add_argument("--min-row-clips", type=int, default=8)
    p.add_argument(
        "--subject-evidence-weight",
        type=float,
        default=float(os.environ.get("SUBJECT_EVIDENCE_WEIGHT", "0.75")),
        help="Blend controlled subject evidence with normalized evidence from all clips in which the row participated.",
    )
    p.add_argument("--unknown-label", default="OutOfSample")
    p.add_argument("--neighbor-details", action="store_true", help="Write per-clip nearest-anchor details; disabled by default for memory safety.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if not (0.0 <= float(args.subject_evidence_weight) <= 1.0):
        raise SystemExit("--subject-evidence-weight must be in [0,1]")
    plan = build_external_ingress_plan(args)
    vae = load_vae_from_args(args)
    cfg = DecayConformalConfig(
        latent_source=str(args.latent_source),
        latent_transform=str(args.latent_transform),
        alpha=float(args.alpha),
        k=int(args.conformal_k),
        calibration_fraction=float(args.calibration_fraction),
        min_proper_anchors=int(args.min_proper_anchors),
        min_calibration_anchors=int(args.min_calibration_anchors),
        split_seed=int(args.split_seed),
        decay_power=float(args.decay_power),
        decay_scale=float(args.decay_scale),
        boundary_tau_fraction=float(args.boundary_tau_fraction),
        evidence_mode=str(args.evidence_mode),
        clip_min_evidence=float(args.clip_min_evidence),
        query_clip_accept_fraction=float(args.query_clip_accept_fraction),
        row_min_evidence=float(args.row_min_evidence),
        min_row_clips=min(int(args.min_row_clips), int(plan.cfg.query_mixups_per_row)),
        subject_evidence_weight=float(args.subject_evidence_weight),
        unknown_label=str(args.unknown_label),
        batch_size=int(args.vote_batch_size),
        num_workers=int(args.num_workers),
        include_neighbor_details=bool(args.neighbor_details),
        progress=bool(args.progress),
    )
    result = run_decay_conformal(plan, vae, cfg, retain_clip_votes=not bool(args.no_clip_vote_output))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "decayed_conformal_summary.json", result["summary"])
    write_json(out / "mach7_ingress_diagnostics.json", getattr(plan, "mach7_ingress_diagnostics", {}))
    if not bool(args.no_clip_vote_output):
        write_jsonl(out / "query_clip_decayed_conformal_votes.jsonl", (asdict(v) for v in result["clip_votes"]))
    write_row_tables(plan, result["row_votes"], out, prefix="decayed_conformal", unknown_label=cfg.unknown_label)
    write_ingress_plan_streaming(plan, out / "ingress_plan", mode=str(args.ingress_plan_output))
    print({"status": "complete", "method": result["summary"].get("method"), "rows": result["summary"].get("num_query_rows"), "rows_in_sample": result["summary"].get("num_query_rows_in_sample"), "out_dir": str(out)}, flush=True)


if __name__ == "__main__":
    main()
