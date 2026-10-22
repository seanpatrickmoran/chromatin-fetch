from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


EPS = 1.0e-12


def resolve_device(requested: str | torch.device | None = "auto") -> str:
    if isinstance(requested, torch.device):
        requested = str(requested)
    req = str(requested or "auto").lower()
    if req == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if req.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False.")
    if req == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise SystemExit("MPS was requested but torch.backends.mps.is_available() is False.")
    return str(requested)


def parse_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for v in value:
            out.extend(parse_list(str(v)))
        return out
    text = str(value).strip()
    if not text:
        return []
    parts = re.split(r"[\s,]+", text)
    return [p for p in (x.strip() for x in parts) if p]


def parse_int_tuple(value: str | Sequence[int] | None, default: Sequence[int] = (4,)) -> tuple[int, ...]:
    if value is None:
        return tuple(int(x) for x in default)
    if isinstance(value, (list, tuple)) and not isinstance(value, str):
        return tuple(int(x) for x in value)
    parts = parse_list(str(value))
    return tuple(int(x) for x in parts) if parts else tuple(int(x) for x in default)


def parse_group_by(value: str | Sequence[str] | None) -> tuple[str, ...]:
    parts = parse_list(value if value is not None else "labels,dataset,resolution,toolsource")
    return tuple(parts) if parts else ("labels", "dataset", "resolution", "toolsource")


def apply_latent_transform(z: torch.Tensor, mode: str, *, eps: float = EPS) -> torch.Tensor:
    """Local mu_raw/mu_varnorm transform"""
    if z.ndim != 2:
        raise ValueError(f"Expected flattened latents [B,D], got {tuple(z.shape)}")
    m = str(mode or "mu_raw").lower().replace("-", "_")
    aliases = {
        "raw": "mu_raw", "mu": "mu_raw", "none": "mu_raw",
        "center": "mu_center", "mc": "mu_center",
        "varnorm": "mu_varnorm", "vn": "mu_varnorm", "mc_varnorm": "mu_varnorm",
        "mu_mc_varnorm": "mu_varnorm", "cosine_mc_varnorm": "mu_varnorm_unit",
        "cos_mc_vn": "mu_varnorm_unit", "mc_vn_unit": "mu_varnorm_unit",
    }
    m = aliases.get(m, m)
    out = z.float()
    if m in {"mu_raw", "raw"}:
        return out.cpu().contiguous()
    if m in {"mu_center", "mu_center_unit", "mu_varnorm", "mu_varnorm_unit", "mu_unit"}:
        if m != "mu_unit":
            out = out - out.mean(dim=1, keepdim=True)
        if m in {"mu_varnorm", "mu_varnorm_unit"}:
            out = out / torch.sqrt(out.square().mean(dim=1, keepdim=True).clamp_min(float(eps)))
        if m in {"mu_unit", "mu_center_unit", "mu_varnorm_unit"}:
            out = torch.nn.functional.normalize(out, p=2, dim=1, eps=float(eps))
        return out.cpu().contiguous()
    raise ValueError("latent_transform must be one of mu_raw, mu_center, mu_varnorm, mu_unit, mu_center_unit, mu_varnorm_unit")


class Mach7JSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, (bytes, bytearray, memoryview)):
            return f"<bytes:{len(bytes(obj))}>"
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, set):
            return sorted(obj)
        return super().default(obj)


def json_safe(obj: Any) -> Any:
    if is_dataclass(obj):
        return json_safe(asdict(obj))
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return f"<bytes:{len(bytes(obj))}>"
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else ("Infinity" if obj > 0 else "-Infinity" if obj < 0 else "NaN")
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, cls=Mach7JSONEncoder, indent=2, ensure_ascii=False, allow_nan=True)
        f.write("\n")


def write_jsonl(path: str | Path, rows: Iterable[Any], *, progress_every: int = 100_000) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    print(f"[mach7:write] start {p}", file=sys.stderr, flush=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, cls=Mach7JSONEncoder, ensure_ascii=False, allow_nan=True)
            f.write("\n")
            count += 1
            if int(progress_every) > 0 and count % int(progress_every) == 0:
                print(f"[mach7:write] {p.name}: {count:,} rows", file=sys.stderr, flush=True)
    print(f"[mach7:write] complete {p}: {count:,} rows", file=sys.stderr, flush=True)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
        return y if math.isfinite(y) else default
    except Exception:
        return default


def quantile(values: Sequence[float], q: float, default: float = float("nan")) -> float:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return default
    return float(np.quantile(vals, min(1.0, max(0.0, float(q)))))


def summarize_values(values: Sequence[float]) -> dict[str, Any]:
    vals = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if vals.size == 0:
        return {"n": 0, "mean": float("nan"), "p05": float("nan"), "p25": float("nan"), "p50": float("nan"), "p75": float("nan"), "p95": float("nan")}
    return {
        "n": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "p05": float(np.quantile(vals, 0.05)),
        "p25": float(np.quantile(vals, 0.25)),
        "p50": float(np.quantile(vals, 0.50)),
        "p75": float(np.quantile(vals, 0.75)),
        "p95": float(np.quantile(vals, 0.95)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def summarize_tensor(values: torch.Tensor) -> dict[str, Any]:
    x = values.detach().float().reshape(-1).cpu()
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {"n": 0, "mean": float("nan"), "p05": float("nan"), "p25": float("nan"), "p50": float("nan"), "p75": float("nan"), "p95": float("nan")}
    q = torch.quantile(x, torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], dtype=x.dtype))
    return {
        "n": int(x.numel()), "mean": float(x.mean().item()), "std": float(x.std(unbiased=False).item()),
        "p05": float(q[0].item()), "p25": float(q[1].item()), "p50": float(q[2].item()),
        "p75": float(q[3].item()), "p95": float(q[4].item()),
        "min": float(x.min().item()), "max": float(x.max().item()),
    }


def add_external_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--model", "--model-dir", dest="model_dir",
        default=os.environ.get("MODEL_DIR", ""),
        help="Local folder containing config.json and model.safetensors.",
    )
    parser.add_argument("--weights-name", default=None)
    parser.add_argument("--dtype", default=os.environ.get("DTYPE", "float32"), choices=["float32", "fp32", "bfloat16", "bf16", "float16", "fp16"])
    parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"))
    parser.add_argument("--latent-source", default="vae", choices=["vae", "swft"], help="Use VAE latents or SWFT clip features for dry runs.")
    parser.add_argument("--latent-transform", default="mu_varnorm")

    parser.add_argument(
        "--reference-database", "--seeded-database", dest="seeded_database",
        action="append", required=True,
        help="Reference-example DB path. Repeat for multiple reference classes.",
    )
    parser.add_argument(
        "--reference-labels", "--seed-labels", dest="seed_labels", default="",
        help="Labels corresponding to repeated reference databases, comma/space-separated.",
    )
    parser.add_argument("--query-database", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--clip-lens", default="4")
    parser.add_argument("--mixups", type=int, default=int(os.environ.get("MIXUPS", "256")))
    parser.add_argument("--num-query-groups", type=int, default=int(os.environ.get("NUM_QUERY_GROUPS", "24")))
    parser.add_argument("--knn-k", type=int, default=int(os.environ.get("KNN_K", "50")))
    parser.add_argument("--query-grouping", default=os.environ.get("QUERY_GROUPING", "query_kmeans"), choices=["query_kmeans", "anchor_knn"])
    parser.add_argument("--query-mixup-mode", default=os.environ.get("QUERY_MIXUP_MODE", "row_centered"), choices=["row_centered", "group_random"])
    parser.add_argument("--query-mixups-per-row", type=int, default=int(os.environ.get("QUERY_MIXUPS_PER_ROW", "16")))
    parser.add_argument("--query-local-k", type=int, default=int(os.environ.get("QUERY_LOCAL_K", "128")))
    parser.add_argument("--anchor-mixups-per-row", type=int, default=int(os.environ.get("ANCHOR_MIXUPS_PER_ROW", "0")))
    parser.add_argument("--mixup-sampler", default=os.environ.get("MIXUP_SAMPLER", "even_greedy"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "1234")))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "256")), help="SWFT ingress batch size.")
    parser.add_argument("--vote-batch-size", type=int, default=int(os.environ.get("VOTE_BATCH_SIZE", "8")), help="VAE clip-encoding batch size.")
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "0")))

    parser.add_argument("--seed-where", default=None)
    parser.add_argument("--query-where", default=None)
    parser.add_argument("--seed-limit", type=int, default=None)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument("--label-column", default="labels")
    parser.add_argument("--fallback-label-column", default="dataset")
    parser.add_argument("--seed-group-by", default="labels,dataset,resolution,toolsource")
    parser.add_argument("--min-seed-group-size", type=int, default=4)
    parser.add_argument("--min-query-group-size", type=int, default=4)
    parser.add_argument("--read-verbose", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false", default=True)

    parser.add_argument("--target-side", type=int, default=128)
    parser.add_argument("--swft-win", type=int, default=32)
    parser.add_argument("--swft-hop", type=int, default=16)
    parser.add_argument("--swft-agg", default="mean_std", choices=["mean", "max", "mean_std"])
    parser.add_argument("--rings", type=int, default=8)
    parser.add_argument("--wedges", type=int, default=12)

    parser.add_argument("--run-profile", default=os.environ.get("RUN_PROFILE", "custom"), choices=["custom", "sweep", "final"], help="sweep uses fewer subject mixups; final restores the full count.")
    parser.add_argument("--sweep-query-mixups-per-row", type=int, default=int(os.environ.get("SWEEP_QUERY_MIXUPS_PER_ROW", "4")))
    parser.add_argument("--final-query-mixups-per-row", type=int, default=int(os.environ.get("FINAL_QUERY_MIXUPS_PER_ROW", "16")))
    parser.add_argument("--max-seed-rows-per-pool", type=int, default=int(os.environ.get("MAX_SEED_ROWS_PER_POOL", "512")), help="0 disables the per-stratum source-row cap.")
    parser.add_argument("--max-seed-rows-per-label", type=int, default=int(os.environ.get("MAX_SEED_ROWS_PER_LABEL", "4096")), help="0 disables the global per-label source-row cap.")
    parser.add_argument("--max-proper-clips-per-label", type=int, default=int(os.environ.get("MAX_PROPER_CLIPS_PER_LABEL", "2048")))
    parser.add_argument("--max-calibration-clips-per-label", type=int, default=int(os.environ.get("MAX_CALIBRATION_CLIPS_PER_LABEL", "512")))
    parser.add_argument("--max-anchor-clips-per-label", type=int, default=int(os.environ.get("MAX_ANCHOR_CLIPS_PER_LABEL", "2048")), help="Graph/probe anchor cap per label and clip length.")
    parser.add_argument("--max-anchor-clips-per-pool", type=int, default=int(os.environ.get("MAX_ANCHOR_CLIPS_PER_POOL", "256")))
    parser.add_argument("--ingress-plan-output", default=os.environ.get("INGRESS_PLAN_OUTPUT", "compact"), choices=["none", "compact", "full"], help="compact streams TSVs and summary; full additionally streams JSONL.")
    parser.add_argument("--no-clip-vote-output", action="store_true", help="Skip the potentially large per-clip JSONL; row outputs and summaries are still written.")
    return parser


def effective_query_mixups(args: argparse.Namespace) -> int:
    profile = str(getattr(args, "run_profile", "custom")).lower()
    if profile == "sweep":
        return max(1, int(getattr(args, "sweep_query_mixups_per_row", 4)))
    if profile == "final":
        return max(1, int(getattr(args, "final_query_mixups_per_row", 16)))
    return max(1, int(getattr(args, "query_mixups_per_row", 16)))


def release_plan_images(plan: Any) -> dict[str, int]:
    released = {"seed_images": 0, "query_images": 0}
    for key, records in (("seed_images", getattr(plan, "seed_records", [])), ("query_images", getattr(plan, "query_records", []))):
        for rec in records:
            image = getattr(rec, "image", None)
            if torch.is_tensor(image):
                released[key] += int(image.numel() * image.element_size())
                rec.image = torch.empty((0,), dtype=torch.float32)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return released


def _row_to_dict(row: Any) -> dict[str, Any]:
    return asdict(row) if is_dataclass(row) else dict(row)


def write_ingress_plan_streaming(plan: Any, out_dir: str | Path, *, mode: str = "compact") -> None:
    mode = str(mode).lower()
    if mode == "none":
        return
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    write_json(out / "summary.json", plan.to_summary_dict())
    from ..data.clips import clip_to_dict, write_clips_tsv, write_query_assignments_tsv

    print("[mach7:write] ingress query_assignments.tsv", file=sys.stderr, flush=True)
    write_query_assignments_tsv(plan, out / "query_assignments.tsv")
    print("[mach7:write] ingress anchor_clips.tsv", file=sys.stderr, flush=True)
    write_clips_tsv(plan.anchor_clips, out / "anchor_clips.tsv")
    print("[mach7:write] ingress query_clips.tsv", file=sys.stderr, flush=True)
    write_clips_tsv(plan.query_clips, out / "query_clips.tsv")
    if mode == "full":
        write_jsonl(out / "anchor_clips.jsonl", (clip_to_dict(c) for c in plan.anchor_clips))
        write_jsonl(out / "query_clips.jsonl", (clip_to_dict(c) for c in plan.query_clips))
        write_jsonl(out / "query_assignments.jsonl", (asdict(a) if is_dataclass(a) else dict(a) for a in plan.query_assignments))
        write_jsonl(out / "query_pools.jsonl", ({"pool_id": p, "record_ids": ids, "size": len(ids)} for p, ids in sorted(plan.query_pools.items())))


def _set_record_label(record: Any, label_column: str, label: str) -> None:
    record.meta[label_column] = label
    row = record.meta.get("row")
    if isinstance(row, dict):
        row[label_column] = label


def build_external_ingress_plan(args: argparse.Namespace) -> Any:
    from ..data.clips import (
        PlannedClip,
        SWFTFeatureConfig,
        StrainerAnchorQueryConfig,
        StrainerIngressPlan,
        assign_anchor_groups,
        assign_query_records,
        compute_record_features,
        load_strainer_image_records,
    )
    from .ingress import (
        PARTITION_PROPER,
        build_partitioned_anchor_clips,
        build_query_clips_mach7,
        select_seed_rows_stratified_swft,
        split_seed_rows,
    )

    seed_dbs = parse_list(args.seeded_database)
    if not seed_dbs:
        raise ValueError("At least one --seeded-database is required")
    seed_labels = parse_list(getattr(args, "seed_labels", ""))
    if seed_labels and len(seed_labels) != len(seed_dbs):
        raise ValueError(f"--seed-labels has {len(seed_labels)} labels but {len(seed_dbs)} seed DBs were supplied")

    device = resolve_device(getattr(args, "device", "auto"))
    feature = SWFTFeatureConfig(
        target_side=int(args.target_side),
        swft_win=int(args.swft_win),
        swft_hop=int(args.swft_hop),
        swft_agg=str(args.swft_agg),
        rings=int(args.rings),
        wedges=int(args.wedges),
    )
    cfg = StrainerAnchorQueryConfig(
        seeded_database=";".join(seed_dbs),
        query_database=str(args.query_database),
        clip_lens=parse_int_tuple(args.clip_lens),
        mixups=int(args.mixups),
        num_query_groups=int(args.num_query_groups),
        knn_k=int(args.knn_k),
        query_grouping=str(args.query_grouping),
        query_mixups_per_row=effective_query_mixups(args),
        query_mixup_mode=str(args.query_mixup_mode),
        query_local_k=int(args.query_local_k),
        anchor_mixups_per_row=int(args.anchor_mixups_per_row),
        seed=int(args.seed),
        batch_size=int(args.batch_size),
        device=device,
        progress=bool(args.progress),
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

    seed_records: list[Any] = []
    for db_i, db_path in enumerate(seed_dbs):
        recs = load_strainer_image_records(
            db_path,
            split="seed",
            target_side=int(cfg.feature.target_side),
            read_verbose=bool(cfg.read_verbose),
            where=cfg.seed_where,
            limit=cfg.seed_limit,
            extra_columns=needed_columns,
            progress=bool(cfg.progress),
        )
        label_override = seed_labels[db_i] if seed_labels else None
        for rec in recs:
            rec.meta["external_seed_db"] = str(db_path)
            rec.meta["original_rowid"] = int(rec.rowid)
            rec.meta["original_key_id"] = int(rec.key_id)
            if label_override is not None:
                _set_record_label(rec, cfg.label_column, label_override)
            rec.index = len(seed_records)
            # Synthetic IDs prevent rowid/key_id collisions across seed DBs.
            rec.rowid = int(rec.index) + 1
            rec.key_id = int(rec.index) + 1
            seed_records.append(rec)

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

    seed_records, seed_features, seed_selection_diagnostics = select_seed_rows_stratified_swft(
        seed_records, seed_features, cfg,
        max_rows_per_pool=int(getattr(args, "max_seed_rows_per_pool", 0)),
        max_rows_per_label=int(getattr(args, "max_seed_rows_per_label", 0)),
        seed=int(getattr(args, "seed", 1234)),
    )
    is_conformal = hasattr(args, "calibration_fraction") and hasattr(args, "split_seed")
    if is_conformal:
        row_partition, split_diagnostics = split_seed_rows(
            seed_records,
            cfg,
            calibration_fraction=float(getattr(args, "calibration_fraction", 0.25)),
            split_seed=int(getattr(args, "split_seed", getattr(args, "seed", 1234))),
            conformal_k=int(getattr(args, "conformal_k", 7)),
            min_proper_anchors=int(getattr(args, "min_proper_anchors", 8)),
            min_calibration_anchors=int(getattr(args, "min_calibration_anchors", 20)),
        )
    else:
        row_partition = {i: PARTITION_PROPER for i in range(len(seed_records))}
        split_diagnostics = {"applied": False, "reason": "non_conformal_method"}

    sampler = str(args.mixup_sampler).lower().replace("-", "_")
    anchor_clips0, anchor_features0, anchor_diagnostics = build_partitioned_anchor_clips(
        seed_records,
        seed_features,
        cfg,
        planned_clip_cls=PlannedClip,
        row_partition=row_partition,
        sampler=sampler,
        max_proper_clips_per_label=int(getattr(args, "max_proper_clips_per_label", 0)) if is_conformal else 0,
        max_calibration_clips_per_label=int(getattr(args, "max_calibration_clips_per_label", 0)) if is_conformal else 0,
        max_anchor_clips_per_label=int(getattr(args, "max_anchor_clips_per_label", 0)) if not is_conformal else 0,
        max_anchor_clips_per_pool=int(getattr(args, "max_anchor_clips_per_pool", 0)),
    )

    proper_positions = [i for i, c in enumerate(anchor_clips0) if getattr(c, "mach7_partition", "") == PARTITION_PROPER]
    calibration_positions = [i for i, c in enumerate(anchor_clips0) if getattr(c, "mach7_partition", "") != PARTITION_PROPER]
    proper_clips = [anchor_clips0[i] for i in proper_positions]
    proper_features = anchor_features0[proper_positions]
    if not proper_clips:
        raise ValueError("Mach7 produced no proper anchor clips")
    proper_group_ids, anchor_group_labels, proper_clips = assign_anchor_groups(proper_clips, proper_features, cfg)

    calibration_clips = [anchor_clips0[i] for i in calibration_positions]
    calibration_features = anchor_features0[calibration_positions] if calibration_positions else torch.empty((0, anchor_features0.shape[1]), dtype=anchor_features0.dtype)
    anchor_clips = list(proper_clips) + calibration_clips
    anchor_features = torch.cat([proper_features, calibration_features], dim=0).contiguous()
    anchor_group_ids = list(proper_group_ids) + [-1] * len(calibration_clips)

    query_assignments, query_pools_by_idx = assign_query_records(
        query_records,
        query_features,
        proper_clips,
        proper_features,
        proper_group_ids,
        cfg,
    )
    query_pools, query_clips, query_diagnostics = build_query_clips_mach7(
        query_records,
        query_features,
        query_pools_by_idx,
        cfg,
        planned_clip_cls=PlannedClip,
        sampler=sampler,
    )

    plan = StrainerIngressPlan(
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
    plan.mach7_ingress_diagnostics = {
        "suite": "mach7",
        "effective_swft_config": asdict(feature),
        "seed_diversity_selection": seed_selection_diagnostics,
        "scale_profile": {
            "run_profile": str(getattr(args, "run_profile", "custom")),
            "effective_query_mixups_per_row": int(cfg.query_mixups_per_row),
            "max_seed_rows_per_pool": int(getattr(args, "max_seed_rows_per_pool", 0)),
            "max_seed_rows_per_label": int(getattr(args, "max_seed_rows_per_label", 0)),
            "max_proper_clips_per_label": int(getattr(args, "max_proper_clips_per_label", 0)),
            "max_calibration_clips_per_label": int(getattr(args, "max_calibration_clips_per_label", 0)),
            "max_anchor_clips_per_label": int(getattr(args, "max_anchor_clips_per_label", 0)),
            "max_anchor_clips_per_pool": int(getattr(args, "max_anchor_clips_per_pool", 0)),
        },
        "effective_combination_config": {
            "clip_lens": list(cfg.clip_lens),
            "mixups": int(cfg.mixups),
            "query_mixups_per_row": int(cfg.query_mixups_per_row),
            "query_local_k": int(cfg.query_local_k),
            "anchor_mixups_per_row": int(cfg.anchor_mixups_per_row),
            "mixup_sampler": sampler,
            "unordered_combinations": True,
            "canonical_vae_order": "key_id_then_rowid",
        },
        "source_row_split": split_diagnostics,
        "anchor_combination_diagnostics": anchor_diagnostics,
        "query_combination_diagnostics": query_diagnostics,
    }
    return plan


def load_vae_from_args(args: argparse.Namespace) -> Any | None:
    if str(getattr(args, "latent_source", "vae")).lower() != "vae":
        return None
    from ..model.loading import VAEInference

    device = resolve_device(getattr(args, "device", "auto"))
    model_value = str(getattr(args, "model_dir", "") or "").strip()
    if not model_value:
        raise ValueError("--model is required when --latent-source=vae")
    model_dir = Path(model_value).expanduser()
    return VAEInference.from_folder(
        model_dir,
        device=device,
        dtype=getattr(args, "dtype", "float32"),
        weights_name=getattr(args, "weights_name", None),
    )


def encode_clip_tables(plan: Any, vae: Any | None, *, latent_source: str, latent_transform: str, batch_size: int, num_workers: int = 0, progress: bool = True) -> dict[int, tuple[Any, Any]]:
    import inspect
    from .encoding import encode_plan_clips_with_swft_features, encode_plan_clips_with_vae

    source = str(latent_source).lower()
    out: dict[int, tuple[Any, Any]] = {}
    clip_lens = sorted({int(c.clip_len) for c in list(plan.anchor_clips) + list(plan.query_clips)})
    vae_sig = inspect.signature(encode_plan_clips_with_vae)
    supports_transform = "latent_transform" in vae_sig.parameters
    for clip_len in clip_lens:
        if source == "vae":
            if vae is None:
                raise ValueError("latent_source='vae' requires a VAEInference instance")
            common_kwargs = dict(split="anchor", clip_len=clip_len, batch_size=int(batch_size), num_workers=int(num_workers), progress=bool(progress))
            if supports_transform:
                anchors = encode_plan_clips_with_vae(plan, vae, **common_kwargs, latent_transform=str(latent_transform))
                common_kwargs["split"] = "query"
                queries = encode_plan_clips_with_vae(plan, vae, **common_kwargs, latent_transform=str(latent_transform))
            else:
                anchors = encode_plan_clips_with_vae(plan, vae, **common_kwargs)
                common_kwargs["split"] = "query"
                queries = encode_plan_clips_with_vae(plan, vae, **common_kwargs)
                anchors.embeddings = apply_latent_transform(anchors.embeddings, str(latent_transform))
                queries.embeddings = apply_latent_transform(queries.embeddings, str(latent_transform))
        elif source == "swft":
            anchors = encode_plan_clips_with_swft_features(plan, split="anchor", clip_len=clip_len, progress=bool(progress))
            queries = encode_plan_clips_with_swft_features(plan, split="query", clip_len=clip_len, progress=bool(progress))
        else:
            raise ValueError("latent_source must be 'vae' or 'swft'")
        out[int(clip_len)] = (anchors, queries)
    return out


def clip_label(clip: Any) -> str:
    return str(getattr(clip, "label", None) or "unknown")


def compact_call(row: dict[str, Any], unknown_label: str = "OutOfSample") -> str:
    if int(row.get("clips_total", 0) or 0) <= 0 or str(row.get("call_state", "")).lower() == "unscored":
        return "Unscored"
    if bool(row.get("in_sample", False)):
        if str(row.get("call_state", "")) == "multi_hit_ambiguous":
            return "Ambiguous"
        return str(row.get("top_label") or "InSample")
    return str(unknown_label)


def write_row_tables(plan: Any, row_votes: Sequence[Any], out_dir: str | Path, *, prefix: str, unknown_label: str = "OutOfSample") -> None:
    """Stream row JSONL/TSV outputs without duplicating all rows in memory."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / f"query_row_{prefix}_votes.jsonl", (_row_to_dict(r) for r in row_votes))
    first = _row_to_dict(row_votes[0]) if row_votes else {}
    fieldnames = list(first.keys())
    detailed_tsv = out / f"query_row_{prefix}_votes.tsv"
    print(f"[mach7:write] start {detailed_tsv}", file=sys.stderr, flush=True)
    with detailed_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for i, obj in enumerate(row_votes, start=1):
            row = _row_to_dict(obj)
            writer.writerow({k: _tsv_scalar(row.get(k)) for k in fieldnames})
            if i % 100_000 == 0:
                print(f"[mach7:write] {detailed_tsv.name}: {i:,} rows", file=sys.stderr, flush=True)
    print(f"[mach7:write] complete {detailed_tsv}: {len(row_votes):,} rows", file=sys.stderr, flush=True)

    from ..data.clips import QUERY_COORD_VOTE_COLUMNS, _record_tsv_base, _tsv_value
    by_rowid = {int(r.rowid): r for r in plan.query_records}
    compact_tsv = out / "query_results.tsv"
    print(f"[mach7:write] start {compact_tsv}", file=sys.stderr, flush=True)
    with compact_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(QUERY_COORD_VOTE_COLUMNS), delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for i, obj in enumerate(row_votes, start=1):
            row = _row_to_dict(obj)
            rec = by_rowid.get(int(row.get("rowid", -1)))
            base = _record_tsv_base(rec) if rec is not None else {}
            compact = {
                "chrom1": base.get("chrom1", ""), "x1": base.get("x1", ""), "x2": base.get("x2", ""),
                "chrom2": base.get("chrom2", ""), "y1": base.get("y1", ""), "y2": base.get("y2", ""),
                "dimensions": base.get("dimensions", ""), "resolution": base.get("resolution", ""),
                "toolsource": base.get("toolsource", ""), "featuretype": base.get("featuretype", ""),
                "rough_label_vote": compact_call(row, unknown_label),
                "rough_label_vote_score": safe_float(row.get("membership_score", row.get("top_p_value", row.get("top_probability", 0.0)))),
                "rough_label_vote_fraction": safe_float(row.get("in_sample_fraction", row.get("accepted_fraction", 0.0))),
                "rough_label_vote_margin": safe_float(row.get("decision_margin", row.get("membership_margin", 0.0))),
            }
            writer.writerow({k: _tsv_value(compact.get(k, "")) for k in QUERY_COORD_VOTE_COLUMNS})
            if i % 100_000 == 0:
                print(f"[mach7:write] {compact_tsv.name}: {i:,} rows", file=sys.stderr, flush=True)
    print(f"[mach7:write] complete {compact_tsv}: {len(row_votes):,} rows", file=sys.stderr, flush=True)


def _tsv_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True)
