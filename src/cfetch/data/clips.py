from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .sqlite import (
    StrainerDataConfig,
    StrainerSQLiteDataset,
    load_db_paths,
)

@dataclass(frozen=True)
class SWFTFeatureConfig:
    """Apply HPK SWFFT settings.

    The training HPK command used --mode multiband --fft_backend swft
    --swft_win 32 --swft_hop 16 --swft_agg mean_std.  The feature path below
    keeps those defaults explicit and optionally quantizes the FFT input to u8
    after Vmax normalization to better match rustfftio/fftpk.py clustering.
    """

    target_side: int = 128
    fft_mode: str = "multiband"
    fft_backend: str = "swft"
    swft_win: int = 32
    swft_hop: int = 16
    swft_agg: str = "mean_std"
    rings: int = 8
    wedges: int = 12
    hann_window: bool = True
    log1p_power: bool = True
    eps: float = 1.0e-12
    input_scale: float = 255.0
    input_quantize_u8: bool = True
    l2_normalize: bool = True


@dataclass(frozen=True)
class StrainerAnchorQueryConfig:
    """
    Data-ingress plan for seeded-anchor and query databases.
    Each database row is one image. Clips are constructed later as unique
    unordered 4/5-image combinations from seeded or query pools.
    """

    seeded_database: str
    query_database: str
    clip_lens: tuple[int, ...] = (4,)
    mixups: int = 256
    num_query_groups: int = 7
    knn_k: int = 7
    query_grouping: str = "query_kmeans"  # "query_kmeans" or "anchor_knn"
    query_mixups_per_row: int = 4
    query_mixup_mode: str = "row_centered"  # "row_centered" or "group_random"
    query_local_k: int = 128
    anchor_mixups_per_row: int = 0
    mixup_sampler: str = "even_greedy"  # "coverage"/"random" legacy-ish or "even_greedy" balanced design.
    seed: int = 1234

    batch_size: int = 256
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    progress: bool = True

    seed_where: Optional[str] = None
    query_where: Optional[str] = None
    seed_limit: Optional[int] = None
    query_limit: Optional[int] = None
    read_verbose: bool = False

    label_column: str = "labels"
    fallback_label_column: str = "dataset"
    seed_group_by: tuple[str, ...] = ("labels", "dataset", "resolution", "toolsource")
    min_seed_group_size: int = 4
    min_query_group_size: int = 4

    feature: SWFTFeatureConfig = SWFTFeatureConfig()


@dataclass
class ImageRecord:
    record_id: str
    split: str
    index: int
    rowid: int
    key_id: int
    image: torch.Tensor  # [1,H,W] float32 [0,1]
    meta: dict[str, Any]


@dataclass
class PlannedClip:
    split: str
    clip_id: str
    clip_len: int
    key_ids: tuple[int, ...]
    rowids: tuple[int, ...]
    record_ids: tuple[str, ...]
    pool_id: str
    label: Optional[str] = None
    coarse_group: Optional[int] = None
    subject_record_id: Optional[str] = None
    subject_key_id: Optional[int] = None
    subject_rowid: Optional[int] = None


@dataclass
class QueryAssignment:
    record_id: str
    key_id: int
    rowid: int
    query_group: int
    pool_id: str
    nearest_anchor_clip_ids: list[str]
    nearest_anchor_labels: list[str]
    nearest_anchor_groups: list[int]
    nearest_scores: list[float]
    label_vote: Optional[str]
    label_vote_count: int = 0
    label_vote_score: float = 0.0
    label_vote_fraction: float = 0.0
    label_vote_margin: float = 0.0
    second_label_vote: Optional[str] = None
    second_label_vote_score: float = 0.0
    nearest_anchor_label_counts: dict[str, int] | None = None
    best_nearest_score: float = 0.0
    mean_nearest_score: float = 0.0
    nearest_score_kind: str = "cosine_distance"


@dataclass
class StrainerIngressPlan:
    cfg: StrainerAnchorQueryConfig
    seed_records: list[ImageRecord]
    query_records: list[ImageRecord]
    seed_features: torch.Tensor
    query_features: torch.Tensor
    anchor_clips: list[PlannedClip]
    anchor_features: torch.Tensor
    anchor_group_ids: list[int]
    anchor_group_labels: dict[int, dict[str, int]]
    query_assignments: list[QueryAssignment]
    query_pools: dict[str, list[str]]
    query_clips: list[PlannedClip]

    def make_query_clip_dataset(self, *, clip_len: Optional[int] = None) -> "PlannedClipDataset":
        clips = self.query_clips
        if clip_len is not None:
            clips = [c for c in clips if int(c.clip_len) == int(clip_len)]
        return PlannedClipDataset(self.query_records, clips)

    def make_anchor_clip_dataset(self, *, clip_len: Optional[int] = None) -> "PlannedClipDataset":
        clips = self.anchor_clips
        if clip_len is not None:
            clips = [c for c in clips if int(c.clip_len) == int(clip_len)]
        return PlannedClipDataset(self.seed_records, clips)

    def make_query_clip_dataloader(
        self,
        *,
        batch_size: int = 8,
        clip_len: Optional[int] = None,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: Optional[bool] = None,
    ) -> DataLoader:
        ds = self.make_query_clip_dataset(clip_len=clip_len)
        pin = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
        return DataLoader(
            ds,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            num_workers=int(num_workers),
            pin_memory=pin,
            collate_fn=planned_clip_collate,
        )

    def to_summary_dict(self) -> dict[str, Any]:
        anchor_label_counts = Counter(c.label or "unknown" for c in self.anchor_clips)
        single_anchor_label_warning = None
        if len(anchor_label_counts) <= 1 and len(self.anchor_clips) > 0:
            only_label = next(iter(anchor_label_counts.keys()))
            single_anchor_label_warning = (
                f"Only one seeded anchor label is present ({only_label!r}); query label_vote will therefore "
                "always resolve to that label. Use the in_sample gates/scores for membership, or seed negative/other classes."
            )
        return {
            "config": _json_safe_config(self.cfg),
            "num_seed_records": len(self.seed_records),
            "num_query_records": len(self.query_records),
            "num_anchor_clips": len(self.anchor_clips),
            "num_query_clips": len(self.query_clips),
            "num_query_pools": len(self.query_pools),
            "query_pool_sizes": {k: len(v) for k, v in sorted(self.query_pools.items())},
            "anchor_group_labels": self.anchor_group_labels,
            "anchor_label_counts": dict(anchor_label_counts),
            "single_anchor_label_warning": single_anchor_label_warning,
            "feature_dim": int(self.seed_features.shape[1]) if self.seed_features.ndim == 2 else None,
        }

    def write_plan(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        _write_json(out / "summary.json", self.to_summary_dict())
        _write_jsonl(out / "anchor_clips.jsonl", [clip_to_dict(c) for c in self.anchor_clips])
        _write_jsonl(out / "query_assignments.jsonl", [asdict(x) for x in self.query_assignments])
        write_query_assignments_tsv(self, out / "query_assignments.tsv")
        write_clips_tsv(self.anchor_clips, out / "anchor_clips.tsv")
        write_clips_tsv(self.query_clips, out / "query_clips.tsv")
        _write_jsonl(out / "query_pools.jsonl", [
            {"pool_id": pool_id, "record_ids": record_ids, "size": len(record_ids)}
            for pool_id, record_ids in sorted(self.query_pools.items())
        ])
        _write_jsonl(out / "query_clips.jsonl", [clip_to_dict(c) for c in self.query_clips])


class PlannedClipDataset(Dataset):
    """Dataset of constructed k-image clips. Records stay row-level; clips are derived."""

    def __init__(self, records: Sequence[ImageRecord], clips: Sequence[PlannedClip]):
        self.records_by_id = {r.record_id: r for r in records}
        self.clips = list(clips)

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        clip = self.clips[int(idx)]
        rows = [self.records_by_id[rid] for rid in clip.record_ids]
        # [T,1,H,W] in canonical sorted-key_id order.
        obs = torch.stack([r.image for r in rows], dim=0).contiguous()
        return {
            "obs": obs,
            "clip": clip_to_dict(clip),
            "rows": [r.meta for r in rows],
        }


def planned_clip_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    clip_lens = {int(x["obs"].shape[0]) for x in batch}
    if len(clip_lens) != 1:
        raise ValueError(
            "Mixed clip lengths in one batch. Pass clip_len=4 or clip_len=5 when creating the dataloader."
        )
    shapes = {tuple(x["obs"].shape) for x in batch}
    if len(shapes) != 1:
        raise ValueError(f"Mixed clip tensor shapes {sorted(shapes)}")
    return {
        "obs": torch.stack([x["obs"] for x in batch], dim=0).contiguous(),  # [B,T,1,H,W]
        "clip": [x["clip"] for x in batch],
        "rows": [x["rows"] for x in batch],
        "meta": {"kept": len(batch), "items": [x["clip"] for x in batch]},
    }


# -----------------------------------------------------------------------------
# SWFT multiband features, adapted from fftpk.py but without rustfftio/packing.
# -----------------------------------------------------------------------------


def _hann2d(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    wy = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
    wx = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
    return wy[:, None] * wx[None, :]


def _power_spectrum_swft(x: torch.Tensor, cfg: SWFTFeatureConfig) -> torch.Tensor:
    # x: [B,1,H,W] float, usually normalized image * 255.
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError(f"Expected [B,1,H,W], got {tuple(x.shape)}")
    _, _, h, w = x.shape
    win = max(2, min(int(cfg.swft_win), h, w))
    hop = max(1, min(int(cfg.swft_hop), win))

    patches = x.unfold(2, win, hop).unfold(3, win, hop)  # [B,1,ny,nx,win,win]
    if cfg.hann_window:
        patches = patches * _hann2d(win, win, x.device, x.dtype)[None, None, None, None, :, :]

    X = torch.fft.fft2(patches, dim=(-2, -1), norm="ortho")
    p = X.real.square() + X.imag.square()
    p = torch.fft.fftshift(p, dim=(-2, -1))
    if cfg.log1p_power:
        p = torch.log1p(p)
    p = p.clamp_min(0.0)

    agg = str(cfg.swft_agg).lower()
    if agg == "max":
        out = p.amax(dim=(2, 3))
    elif agg == "mean_std":
        mu = p.mean(dim=(2, 3))
        sd = p.std(dim=(2, 3), unbiased=False).clamp_min(0.0)
        out = torch.cat([mu, sd], dim=1)
    elif agg == "mean":
        out = p.mean(dim=(2, 3))
    else:
        raise ValueError(f"Unsupported swft_agg={cfg.swft_agg!r}")

    if out.shape[-2:] != (h, w):
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
    return out


def _polar_maps(h: int, w: int, device: torch.device, rings: int, wedges: int, eps: float):
    yy = torch.arange(h, device=device).float() - (h / 2.0)
    xx = torch.arange(w, device=device).float() - (w / 2.0)
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    r = torch.sqrt(X.square() + Y.square() + float(eps))
    r_norm = (r / r.max().clamp_min(float(eps))).clamp(0.0, 1.0)
    r_idx = (r_norm * (int(rings) - 1)).round().long().clamp(0, int(rings) - 1)
    theta = torch.atan2(Y, X)
    t_norm = (theta + math.pi) / (2.0 * math.pi)
    a_idx = (t_norm * int(wedges)).floor().long().clamp(0, int(wedges) - 1)
    return r_idx, a_idx


@torch.no_grad()
def _multiband_embed(power: torch.Tensor, cfg: SWFTFeatureConfig) -> torch.Tensor:
    # power: [B,C,H,W]; output [B,C*rings*wedges]
    b, c, h, w = power.shape
    rings, wedges = int(cfg.rings), int(cfg.wedges)
    r_idx, a_idx = _polar_maps(h, w, power.device, rings, wedges, float(cfg.eps))
    idx = (r_idx.reshape(1, h * w) * wedges + a_idx.reshape(1, h * w)).long().expand(b, h * w)
    dim = rings * wedges
    outs: list[torch.Tensor] = []
    for ch in range(c):
        p = power[:, ch : ch + 1].reshape(b, h * w)
        hist = torch.zeros((b, dim), device=power.device, dtype=power.dtype)
        hist.scatter_add_(1, idx, p)
        hist = hist / hist.sum(dim=1, keepdim=True).clamp_min(float(cfg.eps))
        outs.append(torch.sqrt(hist + float(cfg.eps)))
    out = torch.cat(outs, dim=1)
    if cfg.l2_normalize:
        out = F.normalize(out, p=2, dim=1, eps=float(cfg.eps))
    return out


@torch.no_grad()
def swft_multiband_features(images: torch.Tensor, cfg: SWFTFeatureConfig) -> torch.Tensor:
    """Return training-compatible SWFFT multiband features for [B,1,H,W] images.

    The caller has already Vmax-normalized, clamped to [0,1], and resized rows.
    For SWFFT grouping we emulate fftpk.py/rustfftio by scaling to 0..255 and,
    by default, truncating to uint8-valued floats before the FFT.
    """
    mode = str(getattr(cfg, "fft_mode", "multiband")).lower()
    backend = str(getattr(cfg, "fft_backend", "swft")).lower()
    if mode != "multiband" or backend != "swft":
        raise ValueError("This ingress feature path currently supports fft_mode='multiband' and fft_backend='swft'.")
    x = images.float().clamp(0.0, 1.0) * float(cfg.input_scale)
    if bool(getattr(cfg, "input_quantize_u8", True)):
        x = x.clamp(0.0, 255.0).floor()
    power = _power_spectrum_swft(x, cfg)
    return _multiband_embed(power, cfg)


# -----------------------------------------------------------------------------
# Ingress and planning.
# -----------------------------------------------------------------------------


def load_strainer_image_records(
    db_path: str | Path,
    *,
    split: str,
    target_side: int,
    read_verbose: bool = False,
    where: Optional[str] = None,
    limit: Optional[int] = None,
    extra_columns: Sequence[str] = (),
    progress: bool = True,
) -> list[ImageRecord]:
    ds = StrainerSQLiteDataset(
        StrainerDataConfig(
            db_paths=(str(db_path),),
            target_side=int(target_side),
            batch_size=1,
            shuffle=False,
            read_verbose=bool(read_verbose),
            extra_columns=tuple(extra_columns),
            where=where,
            limit=limit,
        )
    )
    records: list[ImageRecord] = []
    iterator = tqdm(range(len(ds)), desc=f"load {split} rows", unit="row", disable=not bool(progress))
    for i in iterator:
        item = ds[i]
        obs = item["obs"].contiguous()
        # StrainerSQLiteDataset exposes one row as [T,C,H,W] with T=1.
        # Clip ingress stores each database row as one image [C,H,W].
        if obs.ndim == 4 and int(obs.shape[0]) == 1:
            obs = obs[0].contiguous()
        if obs.ndim != 3:
            raise ValueError(f"Expected row image [C,H,W] after squeeze, got {tuple(obs.shape)}")
        meta = dict(item["meta"])
        key_id = int(meta.get("key_id") if meta.get("key_id") is not None else meta.get("rowid"))
        rowid = int(meta.get("rowid"))
        record_id = f"{split}:{Path(str(db_path)).name}:row{rowid}:key{key_id}"
        records.append(
            ImageRecord(
                record_id=record_id,
                split=split,
                index=i,
                rowid=rowid,
                key_id=key_id,
                image=obs,
                meta=meta,
            )
        )
    ds.close()
    return records


@torch.no_grad()
def compute_record_features(
    records: Sequence[ImageRecord],
    *,
    cfg: SWFTFeatureConfig,
    batch_size: int = 256,
    device: str | torch.device = "cpu",
    progress: bool = True,
    desc: str = "SWFT features",
) -> torch.Tensor:
    if not records:
        return torch.empty((0, int(cfg.rings) * int(cfg.wedges) * (2 if cfg.swft_agg == "mean_std" else 1)))
    dev = torch.device(device)
    outs: list[torch.Tensor] = []
    starts = range(0, len(records), int(batch_size))
    for start in tqdm(starts, desc=desc, unit="batch", disable=not bool(progress), total=math.ceil(len(records) / max(1, int(batch_size)))):
        chunk = records[start : start + int(batch_size)]
        images = torch.stack([r.image for r in chunk], dim=0).to(dev, non_blocking=True)
        feats = swft_multiband_features(images, cfg)
        outs.append(feats.detach().cpu())
    return torch.cat(outs, dim=0).contiguous()


def _meta_value(record: ImageRecord, name: str) -> Any:
    if name == "rowid":
        return record.rowid
    if name == "key_id":
        return record.key_id
    row = record.meta.get("row")
    if isinstance(row, dict) and name in row:
        return row.get(name)
    return record.meta.get(name)


def _clean_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    s = str(value).strip()
    return s if s else None


def record_label(record: ImageRecord, cfg: StrainerAnchorQueryConfig) -> str:
    label = _clean_label(_meta_value(record, cfg.label_column))
    if label is not None:
        return label
    fallback = _clean_label(_meta_value(record, cfg.fallback_label_column))
    return fallback if fallback is not None else "seeded"


def seed_pool_key(record: ImageRecord, cfg: StrainerAnchorQueryConfig) -> str:
    parts: list[str] = []
    for col in cfg.seed_group_by:
        if col == cfg.label_column:
            val = record_label(record, cfg)
        else:
            val = _meta_value(record, col)
        if isinstance(val, (list, tuple)):
            val = ",".join(map(str, val))
        parts.append(f"{col}={val}")
    return "seed|" + "|".join(parts)


def group_seed_records(records: Sequence[ImageRecord], cfg: StrainerAnchorQueryConfig) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        groups[seed_pool_key(rec, cfg)].append(i)
    return dict(groups)


def _comb_count(n: int, k: int) -> int:
    if n < k:
        return 0
    return math.comb(n, k)


def _wanted_mixup_count(n: int, k: int, max_mixups: int, row_coverage_target: int) -> int:
    total = _comb_count(int(n), int(k))
    if total <= 0:
        return 0
    wanted = total if int(max_mixups) <= 0 else min(int(max_mixups), total)
    coverage = max(0, int(row_coverage_target))
    if coverage > 0:
        coverage_wanted = int(math.ceil((int(n) * coverage) / max(int(k), 1)))
        wanted = min(total, max(wanted, coverage_wanted))
    return int(wanted)


def _combo_pair_iter(combo: Sequence[int]):
    c = tuple(sorted(int(x) for x in combo))
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            yield (c[i], c[j])


def _weighted_sample_no_replace(pool: Sequence[int], weights: Sequence[float], k: int, rng: random.Random) -> tuple[int, ...]:
    """Small deterministic weighted sample without replacement, pure Python."""
    items = [int(x) for x in pool]
    w = [max(float(v), 1.0e-12) for v in weights]
    out: list[int] = []
    for _ in range(min(int(k), len(items))):
        total = float(sum(w))
        r = rng.random() * total
        acc = 0.0
        pick = len(items) - 1
        for i, wi in enumerate(w):
            acc += wi
            if acc >= r:
                pick = i
                break
        out.append(items.pop(pick))
        w.pop(pick)
    return tuple(sorted(out))


def sample_unique_combinations(
    indices: Sequence[int],
    *,
    k: int,
    max_mixups: int,
    seed: int,
    row_coverage_target: int = 0,
    sampler: str = "coverage",
    candidate_pool: int = 256,
) -> list[tuple[int, ...]]:
    """Enumerate small groups; otherwise sample unique k-subsets deterministically.

    ``sampler='coverage'`` preserves the older behavior: a shuffled coverage pass
    followed by random unique k-sets.  ``sampler='even_greedy'`` uses a greedy
    low-reuse design that tries to equalize row reuse and pair reuse while still
    sampling within the SWFFT-defined seed/query pool supplied by the caller.
    """
    idxs = sorted(int(x) for x in indices)
    n = len(idxs)
    k = int(k)
    if n < k:
        return []
    total = _comb_count(n, k)
    wanted = _wanted_mixup_count(n, k, int(max_mixups), int(row_coverage_target))
    if wanted <= 0:
        return []
    if total <= wanted:
        return [tuple(c) for c in combinations(idxs, k)]

    mode = str(sampler or "coverage").lower().replace("-", "_")
    rng = random.Random(int(seed))

    if mode in {"random", "coverage", "legacy"}:
        seen: set[tuple[int, ...]] = set()
        coverage = max(0, int(row_coverage_target))
        rounds = max(1, coverage) if coverage > 0 else 1
        for _ in range(rounds):
            perm = idxs[:]
            rng.shuffle(perm)
            cursor = 0
            while len(seen) < wanted and cursor + k <= len(perm):
                seen.add(tuple(sorted(perm[cursor : cursor + k])))
                cursor += k
            if len(seen) >= wanted:
                break
        attempts = 0
        max_attempts = max(10_000, wanted * 200)
        while len(seen) < wanted and attempts < max_attempts:
            seen.add(tuple(sorted(rng.sample(idxs, k))))
            attempts += 1
        if len(seen) < wanted:
            for combo in combinations(idxs, k):
                seen.add(tuple(combo))
                if len(seen) >= wanted:
                    break
        return sorted(seen)

    if mode not in {"even_greedy", "greedy_even", "balanced", "even"}:
        raise ValueError("mixup_sampler must be one of random, coverage, or even_greedy")

    seen: set[tuple[int, ...]] = set()
    row_counts: Counter[int] = Counter()
    pair_counts: Counter[tuple[int, int]] = Counter()

    # Greedily choose among weighted random candidates.  Rows/pairs with lower
    # reuse get higher sampling weight and lower acceptance penalty.
    attempts = 0
    max_attempts = max(20_000, wanted * max(200, int(candidate_pool)))
    while len(seen) < wanted and attempts < max_attempts:
        best: tuple[tuple[float, ...], tuple[int, ...]] | None = None
        for _ in range(max(16, int(candidate_pool))):
            weights = [1.0 / (1.0 + float(row_counts[int(i)])) for i in idxs]
            combo = _weighted_sample_no_replace(idxs, weights, k, rng)
            if combo in seen:
                attempts += 1
                continue
            pairs = list(_combo_pair_iter(combo))
            penalty = (
                max(row_counts[int(i)] for i in combo),
                sum(row_counts[int(i)] for i in combo),
                max((pair_counts[p] for p in pairs), default=0),
                sum(pair_counts[p] for p in pairs),
                rng.random(),
            )
            if best is None or penalty < best[0]:
                best = (penalty, combo)
        if best is None:
            attempts += 1
            continue
        combo = best[1]
        if combo in seen:
            attempts += 1
            continue
        seen.add(combo)
        for i in combo:
            row_counts[int(i)] += 1
        for p in _combo_pair_iter(combo):
            pair_counts[p] += 1
        attempts += 1

    if len(seen) < wanted:
        # Deterministic low-reuse fallback.  This remains safe for near-exhaustive
        # requests because we already returned above when total <= wanted.
        for combo in combinations(idxs, k):
            combo = tuple(int(x) for x in combo)
            if combo in seen:
                continue
            seen.add(combo)
            if len(seen) >= wanted:
                break
    return sorted(seen)


def _stable_int_seed(text: str, seed: int) -> int:
    h = hashlib.blake2b(f"{seed}|{text}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False) & 0x7FFF_FFFF


def build_anchor_clips(
    records: Sequence[ImageRecord],
    features: torch.Tensor,
    cfg: StrainerAnchorQueryConfig,
) -> tuple[list[PlannedClip], torch.Tensor]:
    seed_groups = group_seed_records(records, cfg)
    clips: list[PlannedClip] = []
    combo_feats: list[torch.Tensor] = []

    seed_group_items = sorted(seed_groups.items())
    for pool_id, idxs in tqdm(seed_group_items, desc="anchor mixup pools", unit="pool", disable=not bool(cfg.progress), total=len(seed_group_items), leave=False, dynamic_ncols=True):
        if len(idxs) < int(cfg.min_seed_group_size):
            continue
        for clip_len in cfg.clip_lens:
            combos = sample_unique_combinations(
                idxs,
                k=int(clip_len),
                max_mixups=int(cfg.mixups),
                seed=_stable_int_seed(f"anchor|{pool_id}|k{clip_len}", cfg.seed),
                row_coverage_target=int(cfg.anchor_mixups_per_row),
                sampler=str(getattr(cfg, "mixup_sampler", "coverage")),
            )
            combo_iter = tqdm(
                combos,
                desc=f"anchor k={int(clip_len)} mixups",
                unit="mixup",
                disable=not bool(cfg.progress),
                leave=False,
                dynamic_ncols=True,
            )
            for combo in combo_iter:
                recs = [records[i] for i in combo]
                # Canonical order by key_id, not original sampled order.
                recs = sorted(recs, key=lambda r: (r.key_id, r.rowid))
                actual_idxs = tuple(r.index for r in recs)
                key_ids = tuple(int(r.key_id) for r in recs)
                rowids = tuple(int(r.rowid) for r in recs)
                record_ids = tuple(r.record_id for r in recs)
                labels = [record_label(r, cfg) for r in recs]
                label = Counter(labels).most_common(1)[0][0] if labels else None
                clip_id = f"anchor:{pool_id}:k{clip_len}:" + "-".join(map(str, key_ids))
                clips.append(
                    PlannedClip(
                        split="anchor",
                        clip_id=clip_id,
                        clip_len=int(clip_len),
                        key_ids=key_ids,
                        rowids=rowids,
                        record_ids=record_ids,
                        pool_id=pool_id,
                        label=label,
                    )
                )
                combo_feats.append(features[list(actual_idxs)].mean(dim=0))

    if combo_feats:
        feat = torch.stack(combo_feats, dim=0).contiguous()
        feat = F.normalize(feat.float(), p=2, dim=1, eps=float(cfg.feature.eps))
    else:
        feat = torch.empty((0, features.shape[1] if features.ndim == 2 else 0), dtype=torch.float32)
    return clips, feat


@torch.no_grad()
def kmeans_torch(x: torch.Tensor, k: int, *, iters: int = 40, seed: int = 1234, progress: bool = True) -> torch.Tensor:
    """Small pure-PyTorch k-means used only to coarsen anchor mixup features."""
    if x.ndim != 2:
        raise ValueError(f"Expected [N,D], got {tuple(x.shape)}")
    n = int(x.shape[0])
    if n == 0:
        return torch.empty((0,), dtype=torch.long)
    k = max(1, min(int(k), n))
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed))
    first = torch.randperm(n, generator=gen, device=x.device)[:k]
    centers = x[first].clone()
    labels = torch.zeros((n,), device=x.device, dtype=torch.long)
    for _ in tqdm(range(max(1, int(iters))), desc="k-means groups", unit="iter", disable=not bool(progress), leave=False):
        # cosine-ish distance because input is normalized.
        sim = x @ centers.t()
        labels = sim.argmax(dim=1)
        # Avoid index_add_ here: on a few CPU/PyTorch builds it can hang with
        # many OpenMP threads even for small tensors. The loop over k is tiny
        # for this use case (default k=7) and is deterministic.
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros((k,), device=x.device, dtype=torch.float32)
        for ci in range(k):
            mask = labels == ci
            cnt = int(mask.sum().item())
            counts[ci] = float(cnt)
            if cnt > 0:
                new_centers[ci] = x[mask].mean(dim=0)
        empty = counts <= 0
        new_centers = F.normalize(new_centers, p=2, dim=1, eps=1.0e-12)
        if empty.any():
            # Re-seed empty centers deterministically with random data points.
            repl = torch.randperm(n, generator=gen, device=x.device)[: int(empty.sum().item())]
            new_centers[empty] = x[repl]
        if torch.allclose(centers, new_centers, atol=1e-5, rtol=1e-4):
            centers = new_centers
            break
        centers = new_centers
    return labels.detach().cpu().long()


def assign_anchor_groups(
    anchor_clips: Sequence[PlannedClip],
    anchor_features: torch.Tensor,
    cfg: StrainerAnchorQueryConfig,
) -> tuple[list[int], dict[int, dict[str, int]], list[PlannedClip]]:
    if len(anchor_clips) == 0:
        raise ValueError("No anchor clips were created. Check seeded_database labels/groups and clip_lens/mixups.")
    labels = kmeans_torch(anchor_features.float(), int(cfg.num_query_groups), seed=int(cfg.seed) + 17, progress=bool(cfg.progress))
    group_ids = [int(x) for x in labels.tolist()]
    hist: dict[int, Counter] = defaultdict(Counter)
    out_clips: list[PlannedClip] = []
    for clip, gid in tqdm(zip(anchor_clips, group_ids), desc="label anchor groups", unit="clip", disable=not bool(cfg.progress), total=len(anchor_clips), leave=False):
        hist[gid][clip.label or "unknown"] += 1
        clip.coarse_group = gid
        out_clips.append(clip)
    return group_ids, {gid: dict(c) for gid, c in hist.items()}, out_clips


@torch.no_grad()
def assign_query_records_by_anchor_knn(
    query_records: Sequence[ImageRecord],
    query_features: torch.Tensor,
    anchor_clips: Sequence[PlannedClip],
    anchor_features: torch.Tensor,
    anchor_group_ids: Sequence[int],
    cfg: StrainerAnchorQueryConfig,
    *,
    query_group_ids: Optional[Sequence[int]] = None,
) -> tuple[list[QueryAssignment], dict[str, list[int]]]:
    """Assign each query row to a row-level pool and compute rough anchor distances.

    The grouping is row-level. By default the pool id is supplied by query-row
    SWFT k-means, then mixup clips are sampled *within* those row pools. The
    nearest seeded-anchor mixups are still recorded as diagnostics and as a
    rough one-class distance signal, but they are no longer the only way query
    groups are formed.
    """
    if len(query_records) == 0:
        return [], {}
    if len(anchor_clips) == 0:
        raise ValueError("Need at least one anchor clip before assigning query records")
    k = max(1, min(int(cfg.knn_k), len(anchor_clips)))
    q = F.normalize(query_features.float(), p=2, dim=1, eps=float(cfg.feature.eps))
    a = F.normalize(anchor_features.float(), p=2, dim=1, eps=float(cfg.feature.eps))
    sim = q @ a.t()
    vals, idxs = torch.topk(sim, k=k, dim=1, largest=True, sorted=True)

    provided_groups: Optional[list[int]] = None
    if query_group_ids is not None:
        provided_groups = [int(x) for x in query_group_ids]
        if len(provided_groups) != len(query_records):
            raise ValueError(f"query_group_ids length mismatch: {len(provided_groups)} vs {len(query_records)}")

    assignments: list[QueryAssignment] = []
    pools_by_group_idx: dict[str, list[int]] = defaultdict(list)
    for qi, rec in tqdm(enumerate(query_records), desc="assign query rows", unit="row", disable=not bool(cfg.progress), total=len(query_records), dynamic_ncols=True):
        nn_idx = [int(x) for x in idxs[qi].tolist()]
        sims = [float(x) for x in vals[qi].tolist()]
        distances = [float(max(0.0, 1.0 - s)) for s in sims]
        nn_groups = [int(anchor_group_ids[j]) for j in nn_idx]
        nn_labels = [anchor_clips[j].label or "unknown" for j in nn_idx]

        # Group assignment: either external/query SWFT k-means, or nearest-anchor
        # distance vote for backwards compatibility.
        if provided_groups is not None:
            query_group = int(provided_groups[qi])
        else:
            group_dists: dict[int, list[float]] = defaultdict(list)
            for g, d in zip(nn_groups, distances):
                group_dists[int(g)].append(float(d))
            query_group = min(group_dists.items(), key=lambda kv: (float(np.mean(kv[1])), kv[0]))[0]

        # Label diagnostic: distance-based, lower is better. With a single seeded
        # label this produces a useful distance score even though the label itself
        # cannot distinguish positive vs out-of-sample.
        label_dists: dict[str, list[float]] = defaultdict(list)
        label_count: Counter[str] = Counter()
        for lab, d in zip(nn_labels, distances):
            label_dists[lab].append(float(d))
            label_count[lab] += 1
        label_ranked = sorted(
            ((lab, float(np.mean(ds))) for lab, ds in label_dists.items()),
            key=lambda x: (x[1], -label_count[x[0]], x[0]),
        )
        if label_ranked:
            label_vote, label_vote_score = label_ranked[0]
        else:
            label_vote, label_vote_score = None, 0.0
        second_label_vote = label_ranked[1][0] if len(label_ranked) > 1 else None
        second_label_vote_score = float(label_ranked[1][1]) if len(label_ranked) > 1 else 0.0
        label_vote_count = int(label_count.get(label_vote or "", 0)) if label_vote is not None else 0
        label_vote_fraction = float(label_vote_count / k) if k > 0 else 0.0
        label_vote_margin = float(second_label_vote_score - float(label_vote_score)) if len(label_ranked) > 1 else 0.0

        pool_id = f"query_group_{int(query_group):03d}"
        pools_by_group_idx[pool_id].append(qi)
        assignments.append(
            QueryAssignment(
                record_id=rec.record_id,
                key_id=int(rec.key_id),
                rowid=int(rec.rowid),
                query_group=int(query_group),
                pool_id=pool_id,
                nearest_anchor_clip_ids=[anchor_clips[j].clip_id for j in nn_idx],
                nearest_anchor_labels=nn_labels,
                nearest_anchor_groups=nn_groups,
                nearest_scores=distances,
                label_vote=label_vote,
                label_vote_count=label_vote_count,
                label_vote_score=float(label_vote_score),
                label_vote_fraction=label_vote_fraction,
                label_vote_margin=label_vote_margin,
                second_label_vote=second_label_vote,
                second_label_vote_score=second_label_vote_score,
                nearest_anchor_label_counts={kk: int(v) for kk, v in label_count.items()},
                best_nearest_score=float(min(distances)) if distances else 0.0,
                mean_nearest_score=float(sum(distances) / len(distances)) if distances else 0.0,
                nearest_score_kind="cosine_distance",
            )
        )
    return assignments, dict(pools_by_group_idx)


@torch.no_grad()
def assign_query_records(
    query_records: Sequence[ImageRecord],
    query_features: torch.Tensor,
    anchor_clips: Sequence[PlannedClip],
    anchor_features: torch.Tensor,
    anchor_group_ids: Sequence[int],
    cfg: StrainerAnchorQueryConfig,
) -> tuple[list[QueryAssignment], dict[str, list[int]]]:
    mode = str(getattr(cfg, "query_grouping", "query_kmeans")).lower().replace("-", "_")
    if mode in {"query_kmeans", "swft_kmeans", "row_kmeans"}:
        labels = kmeans_torch(
            F.normalize(query_features.float(), p=2, dim=1, eps=float(cfg.feature.eps)),
            int(cfg.num_query_groups),
            seed=int(cfg.seed) + 113,
            progress=bool(cfg.progress),
        )
        return assign_query_records_by_anchor_knn(
            query_records,
            query_features,
            anchor_clips,
            anchor_features,
            anchor_group_ids,
            cfg,
            query_group_ids=[int(x) for x in labels.tolist()],
        )
    if mode in {"anchor_knn", "anchor", "knn"}:
        return assign_query_records_by_anchor_knn(
            query_records,
            query_features,
            anchor_clips,
            anchor_features,
            anchor_group_ids,
            cfg,
            query_group_ids=None,
        )
    raise ValueError("query_grouping must be 'query_kmeans' or 'anchor_knn'")



def _row_centered_combinations_for_pool(
    idxs: Sequence[int],
    *,
    features: torch.Tensor,
    k: int,
    mixups_per_row: int,
    local_k: int,
    seed: int,
    sampler: str = "coverage",
    progress: bool = True,
    desc: str = "row-centered mixups",
) -> list[tuple[int, tuple[int, ...]]]:
    """Return (subject_idx, combo_indices) clips centered on each row.

    Each query row is the subject of M clips.  Partners are drawn from the same
    SWFFT row-level pool and, when possible, from the subject's local SWFFT
    neighborhood.  ``even_greedy`` balances partner and pair reuse so repeated
    clips better deconvolve subject evidence from lucky partner effects.
    """
    pool = sorted(int(x) for x in idxs)
    k = int(k)
    if len(pool) < k:
        return []
    m = max(1, int(mixups_per_row))
    local_k = max(k - 1, int(local_k))
    feat = F.normalize(features.float(), p=2, dim=1, eps=1.0e-12) if features.numel() else features.float()
    out: list[tuple[int, tuple[int, ...]]] = []
    mode = str(sampler or "coverage").lower().replace("-", "_")
    partner_counts: Counter[int] = Counter()
    pair_counts: Counter[tuple[int, int]] = Counter()

    iterator = tqdm(
        pool,
        desc=desc,
        unit="row",
        disable=not bool(progress),
        leave=False,
        dynamic_ncols=True,
    )
    for subject in iterator:
        partners = [i for i in pool if i != subject]
        if len(partners) < (k - 1):
            continue

        if local_k > 0 and len(partners) > local_k and feat.ndim == 2 and feat.shape[0] > max(pool):
            p_tensor = torch.tensor(partners, dtype=torch.long)
            sims = feat[p_tensor] @ feat[int(subject)]
            top = torch.topk(sims, k=min(local_k, len(partners)), largest=True, sorted=True).indices.tolist()
            partners = [int(p_tensor[j].item()) for j in top]

        total = _comb_count(len(partners), k - 1)
        wanted = min(total, m) if total > 0 else 0
        if wanted <= 0:
            continue

        rng = random.Random(_stable_int_seed(f"rowcenter|{seed}|subject={subject}|k={k}", seed))
        seen: set[tuple[int, ...]] = set()
        if total <= wanted:
            for pcombo in combinations(partners, k - 1):
                seen.add(tuple(sorted((subject, *pcombo))))
        elif mode in {"even_greedy", "greedy_even", "balanced", "even"}:
            attempts = 0
            max_attempts = max(2000, wanted * 256)
            while len(seen) < wanted and attempts < max_attempts:
                best: tuple[tuple[float, ...], tuple[int, ...]] | None = None
                for _ in range(96):
                    weights = []
                    for pidx in partners:
                        pair = tuple(sorted((int(subject), int(pidx))))
                        weights.append(1.0 / (1.0 + partner_counts[int(pidx)] + pair_counts[pair]))
                    pcombo = _weighted_sample_no_replace(partners, weights, k - 1, rng)
                    combo = tuple(sorted((int(subject), *pcombo)))
                    if combo in seen:
                        attempts += 1
                        continue
                    pairs = list(_combo_pair_iter(combo))
                    penalty = (
                        max(partner_counts[int(i)] for i in pcombo),
                        sum(partner_counts[int(i)] for i in pcombo),
                        max((pair_counts[p] for p in pairs), default=0),
                        sum(pair_counts[p] for p in pairs),
                        rng.random(),
                    )
                    if best is None or penalty < best[0]:
                        best = (penalty, combo)
                if best is None:
                    attempts += 1
                    continue
                combo = best[1]
                seen.add(combo)
                for i in combo:
                    if int(i) != int(subject):
                        partner_counts[int(i)] += 1
                for p in _combo_pair_iter(combo):
                    pair_counts[p] += 1
                attempts += 1
            if len(seen) < wanted:
                for pcombo in combinations(partners, k - 1):
                    seen.add(tuple(sorted((subject, *pcombo))))
                    if len(seen) >= wanted:
                        break
        else:
            attempts = 0
            max_attempts = max(1000, wanted * 100)
            while len(seen) < wanted and attempts < max_attempts:
                pcombo = rng.sample(partners, k - 1)
                seen.add(tuple(sorted((subject, *pcombo))))
                attempts += 1
            if len(seen) < wanted:
                for pcombo in combinations(partners, k - 1):
                    seen.add(tuple(sorted((subject, *pcombo))))
                    if len(seen) >= wanted:
                        break
        out.extend((int(subject), combo) for combo in sorted(seen))
    return out


def build_query_clips(
    query_records: Sequence[ImageRecord],
    query_features: torch.Tensor,
    query_pools_by_idx: dict[str, list[int]],
    cfg: StrainerAnchorQueryConfig,
) -> tuple[dict[str, list[str]], list[PlannedClip]]:
    pools_by_record_id: dict[str, list[str]] = {}
    clips: list[PlannedClip] = []
    query_pool_items = sorted(query_pools_by_idx.items())
    mode = str(getattr(cfg, "query_mixup_mode", "row_centered")).lower().replace("-", "_")
    if mode not in {"row_centered", "subject_centered", "group_random", "random"}:
        raise ValueError("query_mixup_mode must be 'row_centered' or 'group_random'")

    for pool_id, idxs in tqdm(query_pool_items, desc="query mixup pools", unit="pool", disable=not bool(cfg.progress), total=len(query_pool_items), leave=False, dynamic_ncols=True):
        pools_by_record_id[pool_id] = [query_records[i].record_id for i in idxs]
        if len(idxs) < int(cfg.min_query_group_size):
            continue
        coarse_group = int(pool_id.rsplit("_", 1)[-1]) if "_" in pool_id else None
        for clip_len in cfg.clip_lens:
            clip_len = int(clip_len)
            if mode in {"row_centered", "subject_centered"}:
                subject_combos = _row_centered_combinations_for_pool(
                    idxs,
                    features=query_features,
                    k=clip_len,
                    mixups_per_row=max(1, int(cfg.query_mixups_per_row)),
                    local_k=int(cfg.query_local_k),
                    seed=_stable_int_seed(f"query|{pool_id}|k{clip_len}|rowcenter", cfg.seed),
                    sampler=str(getattr(cfg, "mixup_sampler", "coverage")),
                    progress=bool(cfg.progress),
                    desc=f"query k={clip_len} subject mixups",
                )
                combo_iter = tqdm(
                    subject_combos,
                    desc=f"query k={clip_len} emit clips",
                    unit="mixup",
                    disable=not bool(cfg.progress),
                    leave=False,
                    dynamic_ncols=True,
                )
                for mix_i, (subject_idx, combo) in enumerate(combo_iter):
                    subject_rec = query_records[int(subject_idx)]
                    recs = sorted([query_records[i] for i in combo], key=lambda r: (r.key_id, r.rowid))
                    key_ids = tuple(int(r.key_id) for r in recs)
                    rowids = tuple(int(r.rowid) for r in recs)
                    record_ids = tuple(r.record_id for r in recs)
                    clip_id = f"query:{pool_id}:k{clip_len}:subject{int(subject_rec.key_id)}:mix{mix_i}:" + "-".join(map(str, key_ids))
                    clips.append(
                        PlannedClip(
                            split="query",
                            clip_id=clip_id,
                            clip_len=clip_len,
                            key_ids=key_ids,
                            rowids=rowids,
                            record_ids=record_ids,
                            pool_id=pool_id,
                            label=None,
                            coarse_group=coarse_group,
                            subject_record_id=subject_rec.record_id,
                            subject_key_id=int(subject_rec.key_id),
                            subject_rowid=int(subject_rec.rowid),
                        )
                    )
            else:
                combos = sample_unique_combinations(
                    idxs,
                    k=clip_len,
                    max_mixups=int(cfg.mixups),
                    seed=_stable_int_seed(f"query|{pool_id}|k{clip_len}", cfg.seed),
                    row_coverage_target=int(cfg.query_mixups_per_row),
                    sampler=str(getattr(cfg, "mixup_sampler", "coverage")),
                )
                combo_iter = tqdm(
                    combos,
                    desc=f"query k={clip_len} mixups",
                    unit="mixup",
                    disable=not bool(cfg.progress),
                    leave=False,
                    dynamic_ncols=True,
                )
                for combo in combo_iter:
                    recs = sorted([query_records[i] for i in combo], key=lambda r: (r.key_id, r.rowid))
                    key_ids = tuple(int(r.key_id) for r in recs)
                    rowids = tuple(int(r.rowid) for r in recs)
                    record_ids = tuple(r.record_id for r in recs)
                    clip_id = f"query:{pool_id}:k{clip_len}:" + "-".join(map(str, key_ids))
                    clips.append(
                        PlannedClip(
                            split="query",
                            clip_id=clip_id,
                            clip_len=clip_len,
                            key_ids=key_ids,
                            rowids=rowids,
                            record_ids=record_ids,
                            pool_id=pool_id,
                            label=None,
                            coarse_group=coarse_group,
                        )
                    )
    return pools_by_record_id, clips

def build_strainer_anchor_query_ingress(
    seeded_database: str | Path,
    query_database: str | Path,
    **kwargs: Any,
) -> StrainerIngressPlan:
    cfg = StrainerAnchorQueryConfig(
        seeded_database=str(seeded_database),
        query_database=str(query_database),
        **kwargs,
    )
    needed_columns = tuple(dict.fromkeys((
        cfg.label_column,
        cfg.fallback_label_column,
        *cfg.seed_group_by,
        "name",
        "condition",
        "featuretype",
        "meta",
        "hic_path",
        "norm",
    )))
    seed_records = load_strainer_image_records(
        cfg.seeded_database,
        split="seed",
        target_side=int(cfg.feature.target_side),
        read_verbose=bool(cfg.read_verbose),
        where=cfg.seed_where,
        limit=cfg.seed_limit,
        extra_columns=needed_columns,
        progress=bool(cfg.progress),
    )
    query_records = load_strainer_image_records(
        cfg.query_database,
        split="query",
        target_side=int(cfg.feature.target_side),
        read_verbose=bool(cfg.read_verbose),
        where=cfg.query_where,
        limit=cfg.query_limit,
        extra_columns=needed_columns,
        progress=bool(cfg.progress),
    )
    if not seed_records:
        raise ValueError("seeded_database produced zero rows")
    if not query_records:
        raise ValueError("query_database produced zero rows")

    seed_features = compute_record_features(
        seed_records,
        cfg=cfg.feature,
        batch_size=int(cfg.batch_size),
        device=cfg.device,
        progress=bool(cfg.progress),
        desc="seed SWFT features",
    )
    query_features = compute_record_features(
        query_records,
        cfg=cfg.feature,
        batch_size=int(cfg.batch_size),
        device=cfg.device,
        progress=bool(cfg.progress),
        desc="query SWFT features",
    )
    anchor_clips, anchor_features = build_anchor_clips(seed_records, seed_features, cfg)
    anchor_group_ids, anchor_group_labels, anchor_clips = assign_anchor_groups(anchor_clips, anchor_features, cfg)
    query_assignments, query_pools_by_idx = assign_query_records(
        query_records,
        query_features,
        anchor_clips,
        anchor_features,
        anchor_group_ids,
        cfg,
    )
    query_pools, query_clips = build_query_clips(query_records, query_features, query_pools_by_idx, cfg)
    return StrainerIngressPlan(
        cfg=cfg,
        seed_records=seed_records,
        query_records=query_records,
        seed_features=seed_features,
        query_features=query_features,
        anchor_clips=anchor_clips,
        anchor_features=anchor_features,
        anchor_group_ids=anchor_group_ids,
        anchor_group_labels=anchor_group_labels,
        query_assignments=query_assignments,
        query_pools=query_pools,
        query_clips=query_clips,
    )



# -----------------------------------------------------------------------------
# Human-readable plan tables.
# -----------------------------------------------------------------------------


def _tsv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<BLOB:{len(bytes(value))} bytes>"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe_obj(value), ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _coord_fields(coords: Any) -> dict[str, Any]:
    if isinstance(coords, str):
        # Coordinates are normally already parsed by StrainerSQLiteDataset, but keep this robust.
        parts = [x.strip() for x in coords.split(",")]
    elif isinstance(coords, (list, tuple)):
        parts = list(coords)
    else:
        parts = []
    out = {
        "chrom1": "", "x1": "", "x2": "", "chrom2": "", "y1": "", "y2": "",
    }
    keys = ["chrom1", "x1", "x2", "chrom2", "y1", "y2"]
    for k, v in zip(keys, parts):
        out[k] = v
    return out


def _record_tsv_base(record: ImageRecord) -> dict[str, Any]:
    m = record.meta or {}
    coords = m.get("coordinates")
    row_verbose = m.get("row") if isinstance(m.get("row"), dict) else {}
    base = {
        "record_id": record.record_id,
        "key_id": int(record.key_id),
        "rowid": int(record.rowid),
        "source_db": m.get("source_db") or m.get("db_path"),
        "dataset": m.get("dataset"),
        "coordinates": coords,
        **_coord_fields(coords),
        "dimensions": m.get("dimensions"),
        "inferred_side": m.get("inferred_side"),
        "target_side": m.get("target_side"),
        "resolution": m.get("resolution"),
        "toolsource": m.get("toolsource"),
        "featuretype": m.get("featuretype") if "featuretype" in m else row_verbose.get("featuretype"),
        "labels": m.get("labels") if "labels" in m else row_verbose.get("labels"),
        "viewing_vmax": m.get("viewing_vmax"),
        "true_max": m.get("true_max"),
        "feature_source": m.get("feature_source"),
    }
    return base


QUERY_COORD_VOTE_COLUMNS = [
    "chrom1",
    "x1",
    "x2",
    "chrom2",
    "y1",
    "y2",
    "dimensions",
    "resolution",
    "toolsource",
    "featuretype",
    "rough_label_vote",
    "rough_label_vote_score",
    "rough_label_vote_fraction",
    "rough_label_vote_margin",
]


def _query_assignment_compact_row(record: Optional[ImageRecord], assignment: Any) -> dict[str, Any]:
    """Return the compact coordinate + rough-vote row used for public/server TSVs."""
    base = _record_tsv_base(record) if record is not None else {}
    return {
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
        "rough_label_vote": getattr(assignment, "label_vote", None),
        "rough_label_vote_score": float(getattr(assignment, "label_vote_score", 0.0)),
        "rough_label_vote_fraction": float(getattr(assignment, "label_vote_fraction", 0.0)),
        "rough_label_vote_margin": float(getattr(assignment, "label_vote_margin", 0.0)),
    }


def write_query_assignments_tsv(plan: StrainerIngressPlan, path: str | Path) -> None:
    """
    Write one query image row per line as the compact server-facing coordinate/vote TSV.

    The richer assignment diagnostics remain in query_assignments.jsonl.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    records = {r.record_id: r for r in plan.query_records}
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=QUERY_COORD_VOTE_COLUMNS,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for a in plan.query_assignments:
            rec = records.get(a.record_id)
            row = _query_assignment_compact_row(rec, a)
            writer.writerow({k: _tsv_value(row.get(k)) for k in QUERY_COORD_VOTE_COLUMNS})


def write_query_results_compact_tsv(plan: StrainerIngressPlan, path: str | Path) -> None:
    """Alias-style writer for the final public query_results.tsv table."""
    write_query_assignments_tsv(plan, path)


def write_clips_tsv(clips: Sequence[PlannedClip], path: str | Path) -> None:
    """Write planned anchor/query mixup clips as a compact TSV."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "clip_id", "clip_len", "pool_id", "label", "coarse_group", "subject_key_id", "subject_rowid", "key_ids", "rowids", "record_ids"]
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for c in clips:
            writer.writerow({
                "split": c.split,
                "clip_id": c.clip_id,
                "clip_len": int(c.clip_len),
                "pool_id": c.pool_id,
                "label": c.label,
                "coarse_group": c.coarse_group,
                "subject_key_id": c.subject_key_id,
                "subject_rowid": c.subject_rowid,
                "key_ids": ";".join(str(x) for x in c.key_ids),
                "rowids": ";".join(str(x) for x in c.rowids),
                "record_ids": ";".join(c.record_ids),
            })

# -----------------------------------------------------------------------------
# CLI helpers.
# -----------------------------------------------------------------------------


def parse_clip_lens(text: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(text, str):
        vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    else:
        vals = [int(x) for x in text]
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("clip_lens cannot be empty")
    for v in vals:
        if v < 2:
            raise ValueError(f"clip length must be >=2, got {v}")
    return tuple(vals)


def parse_group_by(text: str) -> tuple[str, ...]:
    vals = tuple(x.strip() for x in str(text).split(",") if x.strip())
    return vals or ("labels", "dataset", "resolution", "toolsource")


def add_anchor_query_ingress_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--seeded-database", required=True, help="SQLite DB containing seeded/annotated rows.")
    parser.add_argument("--query-database", required=True, help="SQLite DB containing the unmarked/query rows.")
    parser.add_argument("--out-dir", default="strainer_ingress_plan", help="Where to write JSONL plan files.")
    parser.add_argument("--target-side", type=int, default=128)
    parser.add_argument("--clip-lens", default="4", help="Comma-separated clip lengths, for example '4' or '4,5'.")
    parser.add_argument("--mixups", type=int, default=256, help="Max unique combinations per seed/query group per clip length. <=0 means exhaustive.")
    parser.add_argument("--num-query-groups", type=int, default=7, help="Number of coarse anchor-derived groups used to pool query rows.")
    parser.add_argument("--knn-k", type=int, default=7, help="Nearest anchor mixups used for rough distance diagnostics.")
    parser.add_argument("--query-grouping", default="query_kmeans", choices=["query_kmeans", "anchor_knn"], help="How row-level query pools are formed before mixup construction.")
    parser.add_argument("--query-mixups-per-row", type=int, default=4, help="Subject-centered query mixups per row.")
    parser.add_argument("--query-mixup-mode", default="row_centered", choices=["row_centered", "group_random"], help="row_centered scores each row from subject clips; group_random keeps old group-level random clips.")
    parser.add_argument("--query-local-k", type=int, default=128, help="Local SWFT neighbors used as partners for row-centered query mixups.")
    parser.add_argument("--anchor-mixups-per-row", type=int, default=0, help="Optional seeded-anchor mixup coverage target per row.")
    parser.add_argument("--mixup-sampler", default="even_greedy", choices=["random", "coverage", "even_greedy"], help="Combination design for anchor clips and query partners.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed-where", default=None)
    parser.add_argument("--query-where", default=None)
    parser.add_argument("--seed-limit", type=int, default=None)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--read-verbose", action="store_true")
    parser.add_argument("--label-column", default="labels")
    parser.add_argument("--fallback-label-column", default="dataset")
    parser.add_argument("--seed-group-by", default="labels,dataset,resolution,toolsource")
    parser.add_argument("--min-seed-group-size", type=int, default=4)
    parser.add_argument("--min-query-group-size", type=int, default=4)
    parser.add_argument("--fft-mode", default="multiband", choices=["multiband"], help="FFT feature mode matched to fftpk.py.")
    parser.add_argument("--fft-backend", default="swft", choices=["swft"], help="FFT backend matched to fftpk.py.")
    parser.add_argument("--swft-win", type=int, default=32)
    parser.add_argument("--swft-hop", type=int, default=16)
    parser.add_argument("--swft-agg", default="mean_std", choices=["mean", "max", "mean_std"])
    parser.add_argument("--rings", type=int, default=8)
    parser.add_argument("--wedges", type=int, default=12)
    parser.add_argument("--fft-input-quantize-u8", dest="fft_input_quantize_u8", action="store_true", default=True, help="Quantize SWFFT input to u8-valued floats after Vmax normalization; matches fftpk/rustfftio more closely.")
    parser.add_argument("--no-fft-input-quantize-u8", dest="fft_input_quantize_u8", action="store_false")
    return parser


def config_from_anchor_query_args(args: argparse.Namespace) -> StrainerAnchorQueryConfig:
    feature = SWFTFeatureConfig(
        target_side=int(args.target_side),
        fft_mode=str(getattr(args, "fft_mode", "multiband")),
        fft_backend=str(getattr(args, "fft_backend", "swft")),
        swft_win=int(args.swft_win),
        swft_hop=int(args.swft_hop),
        swft_agg=str(args.swft_agg),
        rings=int(args.rings),
        wedges=int(args.wedges),
        input_quantize_u8=bool(getattr(args, "fft_input_quantize_u8", True)),
    )
    return StrainerAnchorQueryConfig(
        seeded_database=str(args.seeded_database),
        query_database=str(args.query_database),
        clip_lens=parse_clip_lens(args.clip_lens),
        mixups=int(args.mixups),
        num_query_groups=int(args.num_query_groups),
        knn_k=int(args.knn_k),
        query_grouping=str(args.query_grouping),
        query_mixups_per_row=int(args.query_mixups_per_row),
        query_mixup_mode=str(args.query_mixup_mode),
        query_local_k=int(args.query_local_k),
        anchor_mixups_per_row=int(args.anchor_mixups_per_row),
        mixup_sampler=str(getattr(args, "mixup_sampler", "even_greedy")),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        device=str(args.device),
        seed_where=args.seed_where,
        query_where=args.query_where,
        seed_limit=args.seed_limit,
        query_limit=args.query_limit,
        read_verbose=bool(args.read_verbose),
        label_column=str(args.label_column),
        fallback_label_column=str(args.fallback_label_column),
        seed_group_by=parse_group_by(args.seed_group_by),
        min_seed_group_size=int(args.min_seed_group_size),
        min_query_group_size=int(args.min_query_group_size),
        feature=feature,
    )


def build_strainer_anchor_query_ingress_from_args(args: argparse.Namespace) -> StrainerIngressPlan:
    cfg = config_from_anchor_query_args(args)
    return build_strainer_anchor_query_ingress(
        cfg.seeded_database,
        cfg.query_database,
        clip_lens=cfg.clip_lens,
        mixups=cfg.mixups,
        num_query_groups=cfg.num_query_groups,
        knn_k=cfg.knn_k,
        query_grouping=cfg.query_grouping,
        query_mixups_per_row=cfg.query_mixups_per_row,
        query_mixup_mode=cfg.query_mixup_mode,
        query_local_k=cfg.query_local_k,
        anchor_mixups_per_row=cfg.anchor_mixups_per_row,
        mixup_sampler=cfg.mixup_sampler,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        device=cfg.device,
        seed_where=cfg.seed_where,
        query_where=cfg.query_where,
        seed_limit=cfg.seed_limit,
        query_limit=cfg.query_limit,
        read_verbose=cfg.read_verbose,
        label_column=cfg.label_column,
        fallback_label_column=cfg.fallback_label_column,
        seed_group_by=cfg.seed_group_by,
        min_seed_group_size=cfg.min_seed_group_size,
        min_query_group_size=cfg.min_query_group_size,
        feature=cfg.feature,
    )


def clip_to_dict(c: PlannedClip) -> dict[str, Any]:
    return {
        "split": c.split,
        "clip_id": c.clip_id,
        "clip_len": int(c.clip_len),
        "key_ids": list(c.key_ids),
        "rowids": list(c.rowids),
        "record_ids": list(c.record_ids),
        "pool_id": c.pool_id,
        "label": c.label,
        "coarse_group": c.coarse_group,
        "subject_record_id": c.subject_record_id,
        "subject_key_id": c.subject_key_id,
        "subject_rowid": c.subject_rowid,
    }


def _json_safe_config(cfg: StrainerAnchorQueryConfig) -> dict[str, Any]:
    d = asdict(cfg)
    return d


def _json_safe_obj(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (bytes, bytearray, memoryview)):
        b = bytes(x)
        return {"encoding": "base64", "nbytes": len(b), "data": __import__("base64").b64encode(b).decode("ascii")}
    if isinstance(x, dict):
        return {str(k): _json_safe_obj(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe_obj(v) for v in x]
    return x


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_json_safe_obj(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe_obj(row), ensure_ascii=False) + "\n")


__all__ = [
    "ImageRecord",
    "PlannedClip",
    "PlannedClipDataset",
    "QueryAssignment",
    "SWFTFeatureConfig",
    "StrainerAnchorQueryConfig",
    "StrainerIngressPlan",
    "add_anchor_query_ingress_args",
    "assign_query_records",
    "assign_query_records_by_anchor_knn",
    "build_anchor_clips",
    "build_query_clips",
    "build_strainer_anchor_query_ingress",
    "build_strainer_anchor_query_ingress_from_args",
    "compute_record_features",
    "config_from_anchor_query_args",
    "load_strainer_image_records",
    "planned_clip_collate",
    "sample_unique_combinations",
    "swft_multiband_features",
    "QUERY_COORD_VOTE_COLUMNS",
    "write_query_assignments_tsv",
    "write_query_results_compact_tsv",
    "write_clips_tsv",
]
