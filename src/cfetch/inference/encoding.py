from __future__ import annotations

import base64
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tqdm.auto import tqdm

from ..model.loading import VAEInference
from ..data.clips import (
    ImageRecord,
    PlannedClip,
    StrainerIngressPlan,
    clip_to_dict,
    planned_clip_collate,
    _record_tsv_base,
    _tsv_value,
    QUERY_COORD_VOTE_COLUMNS,
)

EPS = 1.0e-12


@dataclass(frozen=True)
class LatentVoteConfig:
    """
    Voting pass over constructed anchor/query mixup clips.

    The vote stage does not build rows or pools. It consumes a StrainerIngressPlan:
    anchor clips have labels from the seeded database; query clips are unlabeled
    candidates. Each query clip votes against seeded-anchor mixup clips of the
    same clip length.
    """

    latent_source: str = "vae"  # "vae" for model latents, "swft" for tests / dry-run planning.
    latent_transform: str = "mu_varnorm"  # raw mu was old behavior; mu_varnorm matches the L2 sweep geometry.
    l2_radius: float = 2.0
    seed_calibration: bool = True
    seed_calibration_quantile: float = 0.95
    seed_calibration_k: int = 5
    disable_l2: bool = False
    cosine_threshold: Optional[float] = None
    gate_mode: str = "any"  # any or all when both L2 and cosine are enabled.

    vote_top_k: int = 25
    min_anchor_votes: int = 3
    min_unique_anchor_rows: int = 4
    min_label_vote_fraction: float = 0.0
    min_vote_weight: float = 0.0

    query_clip_accept_fraction: float = 0.25
    min_row_clips: int = 1
    unknown_label: str = "OutOfSample"
    pool_accept_fraction: float = 0.25

    batch_size: int = 8
    num_workers: int = 0
    include_neighbor_details: bool = True
    progress: bool = True


@dataclass
class ClipVote:
    query_clip_id: str
    clip_len: int
    pool_id: str
    key_ids: tuple[int, ...]
    rowids: tuple[int, ...]
    subject_key_id: Optional[int]
    subject_rowid: Optional[int]
    subject_record_id: Optional[str]
    in_sample: bool
    top_label: Optional[str]
    selected_anchor_count: int
    total_anchor_pass_count: int
    top_label_count: int
    total_vote_weight: float
    top_label_weight: float
    top_label_fraction: float
    unique_anchor_rows_top_label: int
    best_l2: Optional[float]
    effective_l2_radius: Optional[float]
    best_cosine: Optional[float]
    neighbors: list[dict[str, Any]]


@dataclass
class RowVote:
    record_id: str
    key_id: int
    rowid: int
    pool_id: str
    clips_total: int
    clips_in_sample: int
    in_sample_fraction: float
    in_sample: bool
    top_label: Optional[str]
    label_weights: dict[str, float]
    label_counts: dict[str, int]
    membership_score: float = 0.0
    membership_margin: float = 0.0
    clips_scored: int = 0
    mean_best_l2: Optional[float] = None
    min_best_l2: Optional[float] = None
    mean_best_cosine: Optional[float] = None
    max_best_cosine: Optional[float] = None
    decision_margin: float = 0.0
    uncertainty_score: float = 1.0
    score_p25: float = 0.0
    score_p50: float = 0.0
    score_p75: float = 0.0
    margin_p25: float = 0.0
    margin_p50: float = 0.0
    margin_p75: float = 0.0
    best_l2_p25: Optional[float] = None
    best_l2_p50: Optional[float] = None
    best_l2_p75: Optional[float] = None
    best_cosine_p25: Optional[float] = None
    best_cosine_p50: Optional[float] = None
    best_cosine_p75: Optional[float] = None
    partner_sensitivity_score: float = 0.0
    partner_sensitivity_margin: float = 0.0


@dataclass
class PoolVote:
    pool_id: str
    clips_total: int
    clips_in_sample: int
    in_sample_fraction: float
    in_sample: bool
    rows_total: int
    rows_in_sample: int
    row_in_sample_fraction: float
    top_label: Optional[str]
    label_weights: dict[str, float]
    label_counts: dict[str, int]
    membership_score: float = 0.0
    membership_margin: float = 0.0
    clips_scored: int = 0
    mean_best_l2: Optional[float] = None
    min_best_l2: Optional[float] = None
    mean_best_cosine: Optional[float] = None
    max_best_cosine: Optional[float] = None


@dataclass
class VoteResult:
    config: LatentVoteConfig
    clip_votes: list[ClipVote]
    row_votes: list[RowVote]
    pool_votes: list[PoolVote]
    summary: dict[str, Any]
    plan: Optional[StrainerIngressPlan] = None

    def write(self, out_dir: str | Path, *, plan: Optional[StrainerIngressPlan] = None) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "vote_summary.json", self.summary)
        _write_jsonl(out / "query_clip_votes.jsonl", [asdict(v) for v in self.clip_votes])
        _write_jsonl(out / "query_row_votes.jsonl", [asdict(v) for v in self.row_votes])
        _write_jsonl(out / "query_pool_votes.jsonl", [asdict(v) for v in self.pool_votes])
        plan_to_use = plan if plan is not None else self.plan
        if plan_to_use is not None:
            write_query_row_votes_tsv(plan_to_use, self.row_votes, out / "query_row_votes.tsv")
            # Compact server-facing coordinate/vote table with exactly the public columns.
            # This table is final row-level clip-mixup evidence, not the rough assignment KNN.
            write_query_results_compact_from_row_votes(plan_to_use, self.row_votes, out / "query_results.tsv", self.config)
            write_query_clip_votes_tsv(plan_to_use, self.clip_votes, out / "query_clip_votes.tsv")
        write_query_pool_votes_tsv(self.pool_votes, out / "query_pool_votes.tsv")


@dataclass
class ClipEmbeddingTable:
    clips: list[PlannedClip]
    embeddings: torch.Tensor  # [N,D] float32 CPU


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (bytes, bytearray, memoryview)):
        b = bytes(obj)
        return {"encoding": "base64", "nbytes": len(b), "data": base64.b64encode(b).decode("ascii")}
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_json_safe(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")


def _apply_latent_transform(z: torch.Tensor, mode: str, *, eps: float = EPS) -> torch.Tensor:
    """Transform flattened VAE mu vectors before L2/cosine voting.

    The original voter used raw flattened mu.  The corrected suite defaults to
    per-sample mean-centering + variance normalization (mu_varnorm), matching the
    geometry used for the L2 ball sweeps.  Unit-normalized aliases are provided
    for cosine-style ablations.
    """
    if z.ndim != 2:
        raise ValueError(f"Expected flattened latents [B,D], got {tuple(z.shape)}")
    m = str(mode or "mu_raw").lower().replace("-", "_")
    aliases = {
        "raw": "mu_raw",
        "mu": "mu_raw",
        "none": "mu_raw",
        "center": "mu_center",
        "mc": "mu_center",
        "varnorm": "mu_varnorm",
        "vn": "mu_varnorm",
        "mc_varnorm": "mu_varnorm",
        "mu_mc_varnorm": "mu_varnorm",
        "cosine_mc_varnorm": "mu_varnorm_unit",
        "cos_mc_vn": "mu_varnorm_unit",
        "mc_vn_unit": "mu_varnorm_unit",
    }
    m = aliases.get(m, m)
    out = z.float()
    if m in {"mu_raw", "raw"}:
        return out.cpu().contiguous()
    if m in {"mu_center", "mu_center_unit", "mu_varnorm", "mu_varnorm_unit", "mu_unit"}:
        if m != "mu_unit":
            out = out - out.mean(dim=1, keepdim=True)
        if m in {"mu_varnorm", "mu_varnorm_unit"}:
            sd = torch.sqrt(out.square().mean(dim=1, keepdim=True).clamp_min(float(eps)))
            out = out / sd
        if m in {"mu_unit", "mu_center_unit", "mu_varnorm_unit"}:
            out = F.normalize(out, p=2, dim=1, eps=float(eps))
        return out.cpu().contiguous()
    raise ValueError(
        "latent_transform must be one of mu_raw, mu_center, mu_varnorm, "
        "mu_unit, mu_center_unit, mu_varnorm_unit"
    )


def _as_flat_latents(mu: torch.Tensor, latent_transform: str = "mu_raw") -> torch.Tensor:
    if mu.ndim < 2:
        raise ValueError(f"Expected batched latents, got {tuple(mu.shape)}")
    z = mu.detach().float().reshape(mu.shape[0], -1)
    return _apply_latent_transform(z, latent_transform)


@torch.no_grad()
def encode_plan_clips_with_vae(
    plan: StrainerIngressPlan,
    vae: VAEInference,
    *,
    split: str,
    clip_len: int,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    progress: bool = True,
    latent_transform: str = "mu_raw",
) -> ClipEmbeddingTable:
    if split not in {"anchor", "query"}:
        raise ValueError(f"split must be 'anchor' or 'query', got {split!r}")
    if split == "anchor":
        ds = plan.make_anchor_clip_dataset(clip_len=int(clip_len))
    else:
        ds = plan.make_query_clip_dataset(clip_len=int(clip_len))

    if len(ds) == 0:
        return ClipEmbeddingTable(clips=[], embeddings=torch.empty((0, 0), dtype=torch.float32))

    pin = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=pin,
        collate_fn=planned_clip_collate,
    )
    outs: list[torch.Tensor] = []
    clips: list[PlannedClip] = list(ds.clips)
    for batch in tqdm(loader, desc=f"encode {split} k={clip_len}", unit="batch", disable=not bool(progress), total=len(loader)):
        # batch["obs"] is public [B,T,1,H,W] in [0,1].
        mu = vae.encode_matrices(batch["obs"], clamp=True)
        outs.append(_as_flat_latents(mu, latent_transform=latent_transform))
    z = torch.cat(outs, dim=0).contiguous() if outs else torch.empty((0, 0), dtype=torch.float32)
    return ClipEmbeddingTable(clips=clips, embeddings=z)


def _record_feature_map(records: Sequence[ImageRecord], features: torch.Tensor) -> dict[str, torch.Tensor]:
    if len(records) != int(features.shape[0]):
        raise ValueError(f"records/features length mismatch: {len(records)} vs {tuple(features.shape)}")
    return {rec.record_id: features[i].detach().float().cpu() for i, rec in enumerate(records)}


def encode_plan_clips_with_swft_features(
    plan: StrainerIngressPlan,
    *,
    split: str,
    clip_len: int,
    progress: bool = True,
) -> ClipEmbeddingTable:
    """Dry-run embedding: mean SWFT record features for each planned clip."""
    if split == "anchor":
        clips = [c for c in plan.anchor_clips if int(c.clip_len) == int(clip_len)]
        fmap = _record_feature_map(plan.seed_records, plan.seed_features)
    elif split == "query":
        clips = [c for c in plan.query_clips if int(c.clip_len) == int(clip_len)]
        fmap = _record_feature_map(plan.query_records, plan.query_features)
    else:
        raise ValueError(f"split must be 'anchor' or 'query', got {split!r}")

    if not clips:
        return ClipEmbeddingTable(clips=[], embeddings=torch.empty((0, 0), dtype=torch.float32))
    rows: list[torch.Tensor] = []
    for clip in tqdm(clips, desc=f"SWFT clip embeddings {split} k={clip_len}", unit="clip", disable=not bool(progress)):
        rows.append(torch.stack([fmap[rid] for rid in clip.record_ids], dim=0).mean(dim=0))
    z = F.normalize(torch.stack(rows, dim=0).float(), p=2, dim=1, eps=EPS).cpu().contiguous()
    return ClipEmbeddingTable(clips=clips, embeddings=z)


def _enabled_metrics(cfg: LatentVoteConfig) -> tuple[bool, bool]:
    l2_on = not bool(cfg.disable_l2)
    cos_on = cfg.cosine_threshold is not None
    if not l2_on and not cos_on:
        raise ValueError("At least one vote gate must be enabled. Use L2 or --cosine-threshold.")
    return l2_on, cos_on


def _metric_pass_mask(l2: Optional[torch.Tensor], cos: Optional[torch.Tensor], cfg: LatentVoteConfig) -> torch.Tensor:
    l2_on, cos_on = _enabled_metrics(cfg)
    masks: list[torch.Tensor] = []
    if l2_on:
        if l2 is None:
            raise ValueError("L2 gate requested but l2 tensor is None")
        masks.append(l2 <= float(cfg.l2_radius))
    if cos_on:
        if cos is None:
            raise ValueError("cosine gate requested but cosine tensor is None")
        masks.append(cos >= float(cfg.cosine_threshold))
    if len(masks) == 1:
        return masks[0]
    mode = str(cfg.gate_mode).lower()
    if mode == "all":
        out = masks[0]
        for m in masks[1:]:
            out = out & m
        return out
    if mode == "any":
        out = masks[0]
        for m in masks[1:]:
            out = out | m
        return out
    raise ValueError(f"gate_mode must be 'any' or 'all', got {cfg.gate_mode!r}")


def _vote_strength(l2: Optional[torch.Tensor], cos: Optional[torch.Tensor], cfg: LatentVoteConfig) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    if not cfg.disable_l2 and l2 is not None:
        radius = max(float(cfg.l2_radius), EPS)
        parts.append((1.0 - (l2 / radius)).clamp_min(0.0))
    if cfg.cosine_threshold is not None and cos is not None:
        thr = float(cfg.cosine_threshold)
        den = max(1.0 - thr, EPS)
        parts.append(((cos - thr) / den).clamp_min(0.0))
    if not parts:
        raise ValueError("No vote strength metric is enabled")
    out = parts[0]
    for p in parts[1:]:
        out = out + p
    out = out / float(len(parts))
    return out.clamp_min(EPS)


def _clip_label(c: PlannedClip) -> str:
    return c.label or "unknown"


def _score_query_against_anchors(
    query_clip: PlannedClip,
    q: torch.Tensor,
    anchors: Sequence[PlannedClip],
    A: torch.Tensor,
    cfg: LatentVoteConfig,
) -> ClipVote:
    if A.ndim != 2 or q.ndim != 1:
        raise ValueError(f"Expected A [N,D], q [D]; got {tuple(A.shape)}, {tuple(q.shape)}")
    if A.shape[0] == 0:
        raise ValueError("Cannot vote with zero anchors")
    if A.shape[1] != q.numel():
        raise ValueError(f"Anchor/query embedding dim mismatch: {A.shape[1]} vs {q.numel()}")

    diff = A - q.view(1, -1)
    l2 = torch.sqrt(diff.square().mean(dim=1).clamp_min(0.0)) if not cfg.disable_l2 else None
    cos = F.cosine_similarity(A, q.view(1, -1).expand_as(A), dim=1, eps=EPS) if cfg.cosine_threshold is not None else None

    pass_mask = _metric_pass_mask(l2, cos, cfg)
    pass_idx = torch.nonzero(pass_mask, as_tuple=False).flatten()
    total_pass = int(pass_idx.numel())

    best_l2 = float(l2.min().item()) if l2 is not None and l2.numel() else None
    best_cos = float(cos.max().item()) if cos is not None and cos.numel() else None

    if total_pass == 0:
        return ClipVote(
            query_clip_id=query_clip.clip_id,
            clip_len=int(query_clip.clip_len),
            pool_id=query_clip.pool_id,
            key_ids=tuple(int(x) for x in query_clip.key_ids),
            rowids=tuple(int(x) for x in query_clip.rowids),
            subject_key_id=query_clip.subject_key_id,
            subject_rowid=query_clip.subject_rowid,
            subject_record_id=query_clip.subject_record_id,
            in_sample=False,
            top_label=None,
            selected_anchor_count=0,
            total_anchor_pass_count=0,
            top_label_count=0,
            total_vote_weight=0.0,
            top_label_weight=0.0,
            top_label_fraction=0.0,
            unique_anchor_rows_top_label=0,
            best_l2=best_l2,
            effective_l2_radius=(float(cfg.l2_radius) if not bool(cfg.disable_l2) else None),
            best_cosine=best_cos,
            neighbors=[],
        )

    strength_all = _vote_strength(l2, cos, cfg)
    cand_strength = strength_all[pass_idx]
    top_k = int(cfg.vote_top_k)
    if top_k > 0 and pass_idx.numel() > top_k:
        vals, rel = torch.topk(cand_strength, k=top_k, largest=True, sorted=True)
        chosen_idx = pass_idx[rel]
        chosen_strength = vals
    else:
        order = torch.argsort(cand_strength, descending=True)
        chosen_idx = pass_idx[order]
        chosen_strength = cand_strength[order]

    label_weight: dict[str, float] = defaultdict(float)
    label_count: Counter[str] = Counter()
    label_rows: dict[str, set[int]] = defaultdict(set)
    neighbors: list[dict[str, Any]] = []

    for rank, (ai_t, weight_t) in enumerate(zip(chosen_idx.tolist(), chosen_strength.tolist()), start=1):
        ai = int(ai_t)
        anchor = anchors[ai]
        label = _clip_label(anchor)
        weight = float(weight_t)
        label_weight[label] += weight
        label_count[label] += 1
        label_rows[label].update(int(x) for x in anchor.rowids)
        if cfg.include_neighbor_details:
            neighbors.append(
                {
                    "rank": rank,
                    "anchor_clip_id": anchor.clip_id,
                    "anchor_label": label,
                    "anchor_pool_id": anchor.pool_id,
                    "anchor_key_ids": list(anchor.key_ids),
                    "anchor_rowids": list(anchor.rowids),
                    "weight": weight,
                    "l2": float(l2[ai].item()) if l2 is not None else None,
                    "cosine": float(cos[ai].item()) if cos is not None else None,
                }
            )

    if label_weight:
        top_label, top_weight = max(label_weight.items(), key=lambda kv: (kv[1], label_count[kv[0]], kv[0]))
    else:
        top_label, top_weight = None, 0.0
    total_weight = float(sum(label_weight.values()))
    top_count = int(label_count.get(top_label or "", 0)) if top_label is not None else 0
    top_frac = float(top_weight / total_weight) if total_weight > 0 else 0.0
    unique_rows = len(label_rows.get(top_label or "", set())) if top_label is not None else 0

    in_sample = (
        int(chosen_idx.numel()) >= int(cfg.min_anchor_votes)
        and total_weight >= float(cfg.min_vote_weight)
        and top_frac >= float(cfg.min_label_vote_fraction)
        and unique_rows >= int(cfg.min_unique_anchor_rows)
    )

    return ClipVote(
        query_clip_id=query_clip.clip_id,
        clip_len=int(query_clip.clip_len),
        pool_id=query_clip.pool_id,
        key_ids=tuple(int(x) for x in query_clip.key_ids),
        rowids=tuple(int(x) for x in query_clip.rowids),
        subject_key_id=query_clip.subject_key_id,
        subject_rowid=query_clip.subject_rowid,
        subject_record_id=query_clip.subject_record_id,
        in_sample=bool(in_sample),
        top_label=top_label,
        selected_anchor_count=int(chosen_idx.numel()),
        total_anchor_pass_count=total_pass,
        top_label_count=top_count,
        total_vote_weight=total_weight,
        top_label_weight=float(top_weight),
        top_label_fraction=top_frac,
        unique_anchor_rows_top_label=int(unique_rows),
        best_l2=best_l2,
        effective_l2_radius=(float(cfg.l2_radius) if not bool(cfg.disable_l2) else None),
        best_cosine=best_cos,
        neighbors=neighbors,
    )


def score_clip_tables(
    anchors: ClipEmbeddingTable,
    queries: ClipEmbeddingTable,
    cfg: LatentVoteConfig,
) -> list[ClipVote]:
    if len(anchors.clips) == 0 or anchors.embeddings.shape[0] == 0:
        return []
    if len(queries.clips) == 0 or queries.embeddings.shape[0] == 0:
        return []
    A = anchors.embeddings.float().cpu().contiguous()
    Q = queries.embeddings.float().cpu().contiguous()
    if A.shape[1] != Q.shape[1]:
        raise ValueError(f"Anchor/query embedding dims differ: {A.shape} vs {Q.shape}")
    votes: list[ClipVote] = []
    iterator = zip(queries.clips, Q)
    for clip, q in tqdm(iterator, desc="score query clips", unit="clip", disable=not bool(cfg.progress), total=len(queries.clips)):
        votes.append(_score_query_against_anchors(clip, q, anchors.clips, A, cfg))
    return votes


def _clip_membership_score_and_margin(vote: ClipVote, cfg: LatentVoteConfig) -> tuple[float, float]:
    """Return a row-aggregatable one-class membership score for one query clip.

    Score is bounded below by 0 and is higher when the clip is well inside the
    enabled gate. Margin is signed, so negative values mean the best neighbor is
    outside the active threshold. This is what lets single-class seeding produce
    an OutOfSample decision instead of only a nearest-label vote.
    """
    scores: list[float] = []
    margins: list[float] = []
    if not bool(cfg.disable_l2) and vote.best_l2 is not None:
        radius = max(float(vote.effective_l2_radius if vote.effective_l2_radius is not None else cfg.l2_radius), EPS)
        margin = radius - float(vote.best_l2)
        margins.append(margin)
        scores.append(max(0.0, margin / radius))
    if cfg.cosine_threshold is not None and vote.best_cosine is not None:
        thr = float(cfg.cosine_threshold)
        den = max(1.0 - thr, EPS)
        margin = float(vote.best_cosine) - thr
        margins.append(margin)
        scores.append(max(0.0, margin / den))
    if not scores:
        return 0.0, 0.0
    return float(sum(scores) / len(scores)), float(sum(margins) / len(margins))


def aggregate_row_votes(
    plan: StrainerIngressPlan,
    clip_votes: Sequence[ClipVote],
    cfg: LatentVoteConfig,
) -> list[RowVote]:
    row_to_pool: dict[int, str] = {}
    for assignment in plan.query_assignments:
        row_to_pool[int(assignment.rowid)] = assignment.pool_id

    acc: dict[int, dict[str, Any]] = {}
    for vote in tqdm(clip_votes, desc="aggregate row votes", unit="clip", disable=not bool(cfg.progress), leave=False):
        clip_score, clip_margin = _clip_membership_score_and_margin(vote, cfg)
        if vote.subject_rowid is not None:
            row_iter = [(int(vote.subject_rowid), int(vote.subject_key_id if vote.subject_key_id is not None else vote.key_ids[0]))]
        else:
            row_iter = list(zip(vote.rowids, vote.key_ids))
        for rid, kid in row_iter:
            entry = acc.setdefault(
                int(rid),
                {
                    "key_id": int(kid),
                    "rowid": int(rid),
                    "clips_total": 0,
                    "clips_in_sample": 0,
                    "label_weights": defaultdict(float),
                    "label_counts": Counter(),
                    "score_sum": 0.0,
                    "margin_sum": 0.0,
                    "clips_scored": 0,
                    "l2_sum": 0.0,
                    "l2_count": 0,
                    "min_l2": None,
                    "cos_sum": 0.0,
                    "cos_count": 0,
                    "max_cos": None,
                    "scores": [],
                    "margins": [],
                    "l2_values": [],
                    "cos_values": [],
                },
            )
            entry["clips_total"] += 1
            entry["score_sum"] += float(clip_score)
            entry["margin_sum"] += float(clip_margin)
            entry["clips_scored"] += 1
            entry["scores"].append(float(clip_score))
            entry["margins"].append(float(clip_margin))
            if vote.best_l2 is not None:
                l2 = float(vote.best_l2)
                entry["l2_sum"] += l2
                entry["l2_count"] += 1
                entry["min_l2"] = l2 if entry["min_l2"] is None else min(float(entry["min_l2"]), l2)
                entry["l2_values"].append(l2)
            if vote.best_cosine is not None:
                c = float(vote.best_cosine)
                entry["cos_sum"] += c
                entry["cos_count"] += 1
                entry["max_cos"] = c if entry["max_cos"] is None else max(float(entry["max_cos"]), c)
                entry["cos_values"].append(c)
            if vote.in_sample:
                entry["clips_in_sample"] += 1
                if vote.top_label is not None:
                    entry["label_weights"][vote.top_label] += float(vote.top_label_weight)
                    entry["label_counts"][vote.top_label] += 1

    for rec in plan.query_records:
        acc.setdefault(
            int(rec.rowid),
            {
                "key_id": int(rec.key_id),
                "rowid": int(rec.rowid),
                "clips_total": 0,
                "clips_in_sample": 0,
                "label_weights": defaultdict(float),
                "label_counts": Counter(),
                "score_sum": 0.0,
                "margin_sum": 0.0,
                "clips_scored": 0,
                "l2_sum": 0.0,
                "l2_count": 0,
                "min_l2": None,
                "cos_sum": 0.0,
                "cos_count": 0,
                "max_cos": None,
            },
        )

    rowid_to_record = {int(r.rowid): r for r in plan.query_records}
    out: list[RowVote] = []
    for rowid, entry in sorted(acc.items(), key=lambda kv: kv[0]):
        weights = dict(entry["label_weights"])
        counts = dict(entry["label_counts"])
        top_label = max(weights.items(), key=lambda kv: kv[1])[0] if weights else None
        total = int(entry["clips_total"])
        ins = int(entry["clips_in_sample"])
        frac = float(ins / total) if total > 0 else 0.0
        scored = int(entry.get("clips_scored", 0))
        membership_score = float(entry["score_sum"] / scored) if scored > 0 else 0.0
        membership_margin = float(entry["margin_sum"] / scored) if scored > 0 else 0.0
        mean_l2 = float(entry["l2_sum"] / entry["l2_count"]) if int(entry.get("l2_count", 0)) > 0 else None
        mean_cos = float(entry["cos_sum"] / entry["cos_count"]) if int(entry.get("cos_count", 0)) > 0 else None
        def _q(vals: Sequence[float], q: float) -> Optional[float]:
            if not vals:
                return None
            return float(np.quantile(np.asarray(vals, dtype=np.float64), float(q)))

        score_p25 = _q(entry.get("scores", []), 0.25) or 0.0
        score_p50 = _q(entry.get("scores", []), 0.50) or 0.0
        score_p75 = _q(entry.get("scores", []), 0.75) or 0.0
        margin_p25 = _q(entry.get("margins", []), 0.25) or 0.0
        margin_p50 = _q(entry.get("margins", []), 0.50) or 0.0
        margin_p75 = _q(entry.get("margins", []), 0.75) or 0.0
        decision_margin = float(frac - float(cfg.query_clip_accept_fraction)) if total >= int(cfg.min_row_clips) else -float(cfg.query_clip_accept_fraction)
        denom = max(float(cfg.query_clip_accept_fraction), 1.0 - float(cfg.query_clip_accept_fraction), EPS)
        uncertainty_score = float(max(0.0, 1.0 - min(1.0, abs(decision_margin) / denom)))
        rec = rowid_to_record.get(int(rowid))
        record_id = rec.record_id if rec is not None else f"query:row{rowid}:key{entry['key_id']}"
        out.append(
            RowVote(
                record_id=record_id,
                key_id=int(entry["key_id"]),
                rowid=int(rowid),
                pool_id=row_to_pool.get(int(rowid), "unassigned"),
                clips_total=total,
                clips_in_sample=ins,
                in_sample_fraction=frac,
                in_sample=bool(total >= int(cfg.min_row_clips) and frac >= float(cfg.query_clip_accept_fraction)),
                top_label=top_label,
                label_weights={k: float(v) for k, v in weights.items()},
                label_counts={k: int(v) for k, v in counts.items()},
                membership_score=membership_score,
                membership_margin=membership_margin,
                clips_scored=scored,
                mean_best_l2=mean_l2,
                min_best_l2=float(entry["min_l2"]) if entry.get("min_l2") is not None else None,
                mean_best_cosine=mean_cos,
                max_best_cosine=float(entry["max_cos"]) if entry.get("max_cos") is not None else None,
                decision_margin=decision_margin,
                uncertainty_score=uncertainty_score,
                score_p25=score_p25,
                score_p50=score_p50,
                score_p75=score_p75,
                margin_p25=margin_p25,
                margin_p50=margin_p50,
                margin_p75=margin_p75,
                best_l2_p25=_q(entry.get("l2_values", []), 0.25),
                best_l2_p50=_q(entry.get("l2_values", []), 0.50),
                best_l2_p75=_q(entry.get("l2_values", []), 0.75),
                best_cosine_p25=_q(entry.get("cos_values", []), 0.25),
                best_cosine_p50=_q(entry.get("cos_values", []), 0.50),
                best_cosine_p75=_q(entry.get("cos_values", []), 0.75),
                partner_sensitivity_score=float(score_p75 - score_p25),
                partner_sensitivity_margin=float(margin_p75 - margin_p25),
            )
        )
    return out


def aggregate_pool_votes(
    clip_votes: Sequence[ClipVote],
    row_votes: Sequence[RowVote],
    cfg: LatentVoteConfig,
) -> list[PoolVote]:
    pool_acc: dict[str, dict[str, Any]] = {}
    for v in tqdm(clip_votes, desc="aggregate pool votes", unit="clip", disable=not bool(cfg.progress), leave=False):
        p = pool_acc.setdefault(
            v.pool_id,
            {
                "clips_total": 0,
                "clips_in_sample": 0,
                "label_weights": defaultdict(float),
                "label_counts": Counter(),
            },
        )
        p["clips_total"] += 1
        if v.in_sample:
            p["clips_in_sample"] += 1
            if v.top_label is not None:
                p["label_weights"][v.top_label] += float(v.top_label_weight)
                p["label_counts"][v.top_label] += 1

    rows_by_pool: dict[str, list[RowVote]] = defaultdict(list)
    for r in row_votes:
        rows_by_pool[r.pool_id].append(r)
        pool_acc.setdefault(
            r.pool_id,
            {
                "clips_total": 0,
                "clips_in_sample": 0,
                "label_weights": defaultdict(float),
                "label_counts": Counter(),
            },
        )

    out: list[PoolVote] = []
    for pool_id, p in sorted(pool_acc.items()):
        weights = dict(p["label_weights"])
        counts = dict(p["label_counts"])
        top_label = max(weights.items(), key=lambda kv: kv[1])[0] if weights else None
        clips_total = int(p["clips_total"])
        clips_in = int(p["clips_in_sample"])
        clip_frac = float(clips_in / clips_total) if clips_total > 0 else 0.0
        rows = rows_by_pool.get(pool_id, [])
        rows_total = len(rows)
        rows_in = sum(1 for r in rows if r.in_sample)
        row_frac = float(rows_in / rows_total) if rows_total > 0 else 0.0
        out.append(
            PoolVote(
                pool_id=pool_id,
                clips_total=clips_total,
                clips_in_sample=clips_in,
                in_sample_fraction=clip_frac,
                in_sample=bool(clips_total > 0 and clip_frac >= float(cfg.pool_accept_fraction)),
                rows_total=rows_total,
                rows_in_sample=rows_in,
                row_in_sample_fraction=row_frac,
                top_label=top_label,
                label_weights={k: float(v) for k, v in weights.items()},
                label_counts={k: int(v) for k, v in counts.items()},
            )
        )
    return out


def _query_record_maps(plan: StrainerIngressPlan) -> tuple[dict[str, ImageRecord], dict[int, ImageRecord]]:
    by_id = {r.record_id: r for r in plan.query_records}
    by_rowid = {int(r.rowid): r for r in plan.query_records}
    return by_id, by_rowid


def _assignment_maps(plan: StrainerIngressPlan) -> tuple[dict[str, Any], dict[int, Any]]:
    by_id = {a.record_id: a for a in plan.query_assignments}
    by_rowid = {int(a.rowid): a for a in plan.query_assignments}
    return by_id, by_rowid


def _dict_score_for_label(d: dict[str, float], label: Optional[str]) -> float:
    if label is None:
        return 0.0
    return float(d.get(label, 0.0))


def write_query_results_compact_from_row_votes(
    plan: StrainerIngressPlan,
    row_votes: Sequence[RowVote],
    path: str | Path,
    cfg: LatentVoteConfig,
) -> None:
    """Write the compact server-facing table from final row-level mixup evidence.

    The column names intentionally stay the same as the rough assignment table,
    but the values are final row evidence:
      - rough_label_vote: top label if row passed clip-mixup membership, else OutOfSample/Unscored
      - rough_label_vote_score: mean bounded membership score over clips containing the row
      - rough_label_vote_fraction: fraction of containing clips called in-sample
      - rough_label_vote_margin: signed mean active-gate margin; positive means inside threshold
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _by_id, by_rowid = _query_record_maps(plan)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=QUERY_COORD_VOTE_COLUMNS,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for rv in row_votes:
            rec = by_rowid.get(int(rv.rowid))
            base = _record_tsv_base(rec) if rec is not None else {}
            if int(rv.clips_total) <= 0:
                call = "Unscored"
            elif bool(rv.in_sample) and rv.top_label is not None:
                call = rv.top_label
            elif bool(rv.in_sample):
                call = "InSample"
            else:
                call = str(cfg.unknown_label)
            row = {
                "chrom1": base.get("chrom1", ""),
                "x1": base.get("x1", ""),
                "x2": base.get("x2", ""),
                "chrom2": base.get("chrom2", ""),
                "y1": base.get("y1", ""),
                "y2": base.get("y2", ""),
                "dimensions": base.get("dimensions", ""),
                "resolution": base.get("resolution", ""),
                "toolsource": base.get("toolsource", ""),
                "featuretype": base.get("featuretype", ""),
                "rough_label_vote": call,
                "rough_label_vote_score": float(rv.membership_score),
                "rough_label_vote_fraction": float(rv.in_sample_fraction),
                "rough_label_vote_margin": float(rv.membership_margin),
            }
            writer.writerow({k: _tsv_value(row.get(k)) for k in QUERY_COORD_VOTE_COLUMNS})


def write_query_row_votes_tsv(plan: StrainerIngressPlan, row_votes: Sequence[RowVote], path: str | Path) -> None:
    """Write one query image row per line with metadata plus final row-level vote outcome."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _by_id, by_rowid = _query_record_maps(plan)
    _assign_by_id, assign_by_rowid = _assignment_maps(plan)
    fieldnames = [
        "record_id", "key_id", "rowid", "source_db", "dataset", "coordinates",
        "chrom1", "x1", "x2", "chrom2", "y1", "y2", "dimensions", "inferred_side",
        "target_side", "resolution", "toolsource", "featuretype", "labels", "viewing_vmax", "true_max",
        "query_group", "pool_id", "rough_label_vote", "rough_label_vote_score", "rough_label_vote_fraction",
        "rough_label_vote_margin", "rough_second_label_vote", "rough_second_label_vote_score",
        "in_sample", "clips_total", "clips_in_sample", "in_sample_fraction",
        "membership_score", "membership_margin", "clips_scored", "mean_best_l2", "min_best_l2",
        "mean_best_cosine", "max_best_cosine",
        "decision_margin", "uncertainty_score",
        "score_p25", "score_p50", "score_p75", "margin_p25", "margin_p50", "margin_p75",
        "best_l2_p25", "best_l2_p50", "best_l2_p75", "best_cosine_p25", "best_cosine_p50", "best_cosine_p75",
        "partner_sensitivity_score", "partner_sensitivity_margin",
        "top_label", "top_label_weight", "top_label_fraction", "label_weights", "label_counts",
    ]
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for rv in row_votes:
            rec = by_rowid.get(int(rv.rowid))
            row = _record_tsv_base(rec) if rec is not None else {"record_id": rv.record_id, "key_id": rv.key_id, "rowid": rv.rowid}
            assignment = assign_by_rowid.get(int(rv.rowid))
            top_weight = _dict_score_for_label(rv.label_weights, rv.top_label)
            total_weight = float(sum(float(v) for v in rv.label_weights.values()))
            row.update({
                "query_group": getattr(assignment, "query_group", "") if assignment is not None else "",
                "pool_id": rv.pool_id,
                "rough_label_vote": getattr(assignment, "label_vote", None) if assignment is not None else None,
                "rough_label_vote_score": getattr(assignment, "label_vote_score", 0.0) if assignment is not None else 0.0,
                "rough_label_vote_fraction": getattr(assignment, "label_vote_fraction", 0.0) if assignment is not None else 0.0,
                "rough_label_vote_margin": getattr(assignment, "label_vote_margin", 0.0) if assignment is not None else 0.0,
                "rough_second_label_vote": getattr(assignment, "second_label_vote", None) if assignment is not None else None,
                "rough_second_label_vote_score": getattr(assignment, "second_label_vote_score", 0.0) if assignment is not None else 0.0,
                "in_sample": bool(rv.in_sample),
                "clips_total": int(rv.clips_total),
                "clips_in_sample": int(rv.clips_in_sample),
                "in_sample_fraction": float(rv.in_sample_fraction),
                "membership_score": float(rv.membership_score),
                "membership_margin": float(rv.membership_margin),
                "clips_scored": int(rv.clips_scored),
                "mean_best_l2": rv.mean_best_l2,
                "min_best_l2": rv.min_best_l2,
                "mean_best_cosine": rv.mean_best_cosine,
                "max_best_cosine": rv.max_best_cosine,
                "decision_margin": float(rv.decision_margin),
                "uncertainty_score": float(rv.uncertainty_score),
                "score_p25": float(rv.score_p25),
                "score_p50": float(rv.score_p50),
                "score_p75": float(rv.score_p75),
                "margin_p25": float(rv.margin_p25),
                "margin_p50": float(rv.margin_p50),
                "margin_p75": float(rv.margin_p75),
                "best_l2_p25": rv.best_l2_p25,
                "best_l2_p50": rv.best_l2_p50,
                "best_l2_p75": rv.best_l2_p75,
                "best_cosine_p25": rv.best_cosine_p25,
                "best_cosine_p50": rv.best_cosine_p50,
                "best_cosine_p75": rv.best_cosine_p75,
                "partner_sensitivity_score": float(rv.partner_sensitivity_score),
                "partner_sensitivity_margin": float(rv.partner_sensitivity_margin),
                "top_label": rv.top_label,
                "top_label_weight": top_weight,
                "top_label_fraction": float(top_weight / total_weight) if total_weight > 0 else 0.0,
                "label_weights": rv.label_weights,
                "label_counts": rv.label_counts,
            })
            writer.writerow({k: _tsv_value(row.get(k)) for k in fieldnames})


def write_query_clip_votes_tsv(plan: StrainerIngressPlan, clip_votes: Sequence[ClipVote], path: str | Path) -> None:
    """Write one query mixup clip per line with outcome and compact positional context."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _by_id, by_rowid = _query_record_maps(plan)
    assign_by_id, assign_by_rowid = _assignment_maps(plan)
    fieldnames = [
        "query_clip_id", "clip_len", "pool_id", "subject_key_id", "subject_rowid", "key_ids", "rowids", "coordinates", "toolsources", "datasets", "resolutions",
        "in_sample", "top_label", "selected_anchor_count", "total_anchor_pass_count", "top_label_count",
        "total_vote_weight", "top_label_weight", "top_label_fraction", "unique_anchor_rows_top_label",
        "best_l2", "effective_l2_radius", "best_cosine", "query_groups", "rough_label_votes",
    ]
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for cv in clip_votes:
            recs = [by_rowid.get(int(rid)) for rid in cv.rowids]
            metas = [r.meta for r in recs if r is not None]
            assignments = [assign_by_rowid.get(int(rid)) for rid in cv.rowids]
            row = {
                "query_clip_id": cv.query_clip_id,
                "clip_len": int(cv.clip_len),
                "pool_id": cv.pool_id,
                "subject_key_id": cv.subject_key_id,
                "subject_rowid": cv.subject_rowid,
                "key_ids": list(cv.key_ids),
                "rowids": list(cv.rowids),
                "coordinates": [m.get("coordinates") for m in metas],
                "toolsources": sorted({str(m.get("toolsource")) for m in metas if m.get("toolsource") is not None}),
                "datasets": sorted({str(m.get("dataset")) for m in metas if m.get("dataset") is not None}),
                "resolutions": sorted({str(m.get("resolution")) for m in metas if m.get("resolution") is not None}),
                "in_sample": bool(cv.in_sample),
                "top_label": cv.top_label,
                "selected_anchor_count": int(cv.selected_anchor_count),
                "total_anchor_pass_count": int(cv.total_anchor_pass_count),
                "top_label_count": int(cv.top_label_count),
                "total_vote_weight": float(cv.total_vote_weight),
                "top_label_weight": float(cv.top_label_weight),
                "top_label_fraction": float(cv.top_label_fraction),
                "unique_anchor_rows_top_label": int(cv.unique_anchor_rows_top_label),
                "best_l2": cv.best_l2,
                "effective_l2_radius": cv.effective_l2_radius,
                "best_cosine": cv.best_cosine,
                "query_groups": [getattr(a, "query_group", None) for a in assignments if a is not None],
                "rough_label_votes": [getattr(a, "label_vote", None) for a in assignments if a is not None],
            }
            writer.writerow({k: _tsv_value(row.get(k)) for k in fieldnames})

def write_query_pool_votes_tsv(pool_votes: Sequence[PoolVote], path: str | Path) -> None:
    """Write one query pool per line with final pool-level vote outcome."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pool_id", "in_sample", "clips_total", "clips_in_sample", "in_sample_fraction",
        "rows_total", "rows_in_sample", "row_in_sample_fraction",
        "top_label", "label_weights", "label_counts",
    ]
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for pv in pool_votes:
            row = {
                "pool_id": pv.pool_id,
                "in_sample": bool(pv.in_sample),
                "clips_total": int(pv.clips_total),
                "clips_in_sample": int(pv.clips_in_sample),
                "in_sample_fraction": float(pv.in_sample_fraction),
                "rows_total": int(pv.rows_total),
                "rows_in_sample": int(pv.rows_in_sample),
                "row_in_sample_fraction": float(pv.row_in_sample_fraction),
                "top_label": pv.top_label,
                "label_weights": pv.label_weights,
                "label_counts": pv.label_counts,
            }
            writer.writerow({k: _tsv_value(row.get(k)) for k in fieldnames})


def summarize_votes(
    plan: StrainerIngressPlan,
    cfg: LatentVoteConfig,
    clip_votes: Sequence[ClipVote],
    row_votes: Sequence[RowVote],
    pool_votes: Sequence[PoolVote],
) -> dict[str, Any]:
    by_len: dict[int, dict[str, int]] = defaultdict(lambda: {"total": 0, "in_sample": 0})
    label_counter: Counter[str] = Counter()
    for v in clip_votes:
        by_len[int(v.clip_len)]["total"] += 1
        if v.in_sample:
            by_len[int(v.clip_len)]["in_sample"] += 1
            if v.top_label is not None:
                label_counter[v.top_label] += 1
    row_scores = np.asarray([float(r.membership_score) for r in row_votes], dtype=np.float64) if row_votes else np.asarray([], dtype=np.float64)
    row_margins = np.asarray([float(r.membership_margin) for r in row_votes], dtype=np.float64) if row_votes else np.asarray([], dtype=np.float64)
    row_score_summary = {
        "mean": float(row_scores.mean()) if row_scores.size else float("nan"),
        "p50": float(np.quantile(row_scores, 0.50)) if row_scores.size else float("nan"),
        "p95": float(np.quantile(row_scores, 0.95)) if row_scores.size else float("nan"),
    }
    row_margin_summary = {
        "mean": float(row_margins.mean()) if row_margins.size else float("nan"),
        "p50": float(np.quantile(row_margins, 0.50)) if row_margins.size else float("nan"),
        "p95": float(np.quantile(row_margins, 0.95)) if row_margins.size else float("nan"),
    }
    return {
        "config": asdict(cfg),
        "ingress": plan.to_summary_dict(),
        "num_query_clip_votes": len(clip_votes),
        "num_query_clips_in_sample": int(sum(1 for v in clip_votes if v.in_sample)),
        "num_query_rows": len(row_votes),
        "num_query_rows_in_sample": int(sum(1 for r in row_votes if r.in_sample)),
        "num_query_pools": len(pool_votes),
        "num_query_pools_in_sample": int(sum(1 for p in pool_votes if p.in_sample)),
        "query_clip_votes_by_len": {str(k): v for k, v in sorted(by_len.items())},
        "accepted_clip_label_counts": dict(label_counter),
        "row_membership_score_summary": row_score_summary,
        "row_membership_margin_summary": row_margin_summary,
    }



def calibrate_l2_radius_from_anchors(
    anchors: ClipEmbeddingTable,
    cfg: LatentVoteConfig,
) -> dict[str, Any]:
    """Estimate a one-class L2 radius from seeded anchor mixup self-neighborhoods.

    For each anchor clip, take its kth nearest same-label seeded anchor clip
    (self excluded), then use a quantile of those distances. The user-supplied
    --l2-radius remains an upper cap, so calibration tightens permissive radii
    without exceeding an explicitly chosen bound.
    """
    if bool(cfg.disable_l2) or not bool(cfg.seed_calibration):
        return {
            "enabled": False,
            "effective_l2_radius": None if bool(cfg.disable_l2) else float(cfg.l2_radius),
            "user_l2_radius": None if bool(cfg.disable_l2) else float(cfg.l2_radius),
            "reason": "disabled" if not bool(cfg.seed_calibration) else "l2_disabled",
        }
    A = anchors.embeddings.float().cpu().contiguous()
    n = int(A.shape[0])
    if n < 2:
        return {
            "enabled": False,
            "effective_l2_radius": float(cfg.l2_radius),
            "user_l2_radius": float(cfg.l2_radius),
            "reason": "not_enough_anchors",
        }
    labels = [_clip_label(c) for c in anchors.clips]
    dists: list[float] = []

    for i in tqdm(range(n), desc="seed self-calibration", unit="anchor", disable=not bool(cfg.progress), leave=False):
        diff = A - A[i].view(1, -1)
        d = torch.sqrt(diff.square().mean(dim=1).clamp_min(0.0))
        mask = torch.ones((n,), dtype=torch.bool)
        mask[i] = False
        same = torch.tensor([labels[j] == labels[i] for j in range(n)], dtype=torch.bool)
        cand = d[mask & same]
        if cand.numel() == 0:
            cand = d[mask]
        if cand.numel() == 0:
            continue
        cand = torch.sort(cand).values
        kth = max(1, int(cfg.seed_calibration_k))
        j = min(kth - 1, cand.numel() - 1)
        dists.append(float(cand[j].item()))
    if not dists:
        return {
            "enabled": False,
            "effective_l2_radius": float(cfg.l2_radius),
            "user_l2_radius": float(cfg.l2_radius),
            "reason": "no_self_distances",
        }
    q = min(1.0, max(0.0, float(cfg.seed_calibration_quantile)))
    raw_radius = float(np.quantile(np.asarray(dists, dtype=np.float64), q))
    raw_radius = max(raw_radius, EPS)
    effective = min(float(cfg.l2_radius), raw_radius) if float(cfg.l2_radius) > 0 else raw_radius
    return {
        "enabled": True,
        "effective_l2_radius": float(effective),
        "user_l2_radius": float(cfg.l2_radius),
        "raw_calibrated_l2_radius": float(raw_radius),
        "quantile": q,
        "k": int(cfg.seed_calibration_k),
        "num_self_distances": len(dists),
        "self_distance_summary": {
            "min": float(np.min(dists)),
            "p50": float(np.quantile(dists, 0.50)),
            "p95": float(np.quantile(dists, 0.95)),
            "max": float(np.max(dists)),
        },
    }

def run_latent_vote(
    plan: StrainerIngressPlan,
    *,
    cfg: LatentVoteConfig,
    vae: Optional[VAEInference] = None,
) -> VoteResult:
    source = str(cfg.latent_source).lower()
    if source not in {"vae", "swft"}:
        raise ValueError("latent_source must be 'vae' or 'swft'")
    if source == "vae" and vae is None:
        raise ValueError("latent_source='vae' requires a VAEInference instance")

    all_votes: list[ClipVote] = []
    calibration_by_len: dict[str, Any] = {}
    clip_lens = sorted({int(c.clip_len) for c in plan.anchor_clips + plan.query_clips})
    for clip_len in tqdm(clip_lens, desc="vote clip lengths", unit="k", disable=not bool(cfg.progress)):
        if source == "vae":
            assert vae is not None
            anchors = encode_plan_clips_with_vae(
                plan,
                vae,
                split="anchor",
                clip_len=clip_len,
                batch_size=int(cfg.batch_size),
                num_workers=int(cfg.num_workers),
                progress=bool(cfg.progress),
                latent_transform=str(cfg.latent_transform),
            )
            queries = encode_plan_clips_with_vae(
                plan,
                vae,
                split="query",
                clip_len=clip_len,
                batch_size=int(cfg.batch_size),
                num_workers=int(cfg.num_workers),
                progress=bool(cfg.progress),
                latent_transform=str(cfg.latent_transform),
            )
        else:
            anchors = encode_plan_clips_with_swft_features(plan, split="anchor", clip_len=clip_len, progress=bool(cfg.progress))
            queries = encode_plan_clips_with_swft_features(plan, split="query", clip_len=clip_len, progress=bool(cfg.progress))

        calib = calibrate_l2_radius_from_anchors(anchors, cfg)
        calibration_by_len[str(clip_len)] = calib
        cfg_eff = cfg
        if (not bool(cfg.disable_l2)) and calib.get("effective_l2_radius") is not None:
            cfg_eff = replace(cfg, l2_radius=float(calib["effective_l2_radius"]))
        all_votes.extend(score_clip_tables(anchors, queries, cfg_eff))

    row_votes = aggregate_row_votes(plan, all_votes, cfg)
    pool_votes = aggregate_pool_votes(all_votes, row_votes, cfg)
    summary = summarize_votes(plan, cfg, all_votes, row_votes, pool_votes)
    summary["seed_calibration_by_clip_len"] = calibration_by_len
    return VoteResult(config=cfg, clip_votes=all_votes, row_votes=row_votes, pool_votes=pool_votes, summary=summary, plan=plan)


__all__ = [
    "ClipEmbeddingTable",
    "ClipVote",
    "LatentVoteConfig",
    "PoolVote",
    "RowVote",
    "VoteResult",
    "aggregate_pool_votes",
    "aggregate_row_votes",
    "encode_plan_clips_with_swft_features",
    "encode_plan_clips_with_vae",
    "_apply_latent_transform",
    "calibrate_l2_radius_from_anchors",
    "run_latent_vote",
    "score_clip_tables",
    "summarize_votes",
    "write_query_clip_votes_tsv",
    "write_query_results_compact_from_row_votes",
    "write_query_row_votes_tsv",
    "write_query_pool_votes_tsv",
]
