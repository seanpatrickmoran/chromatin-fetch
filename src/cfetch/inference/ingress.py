from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from itertools import combinations
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


EPS = 1.0e-12
PARTITION_PROPER = "proper"
PARTITION_CALIBRATION = "calibration"


def stable_int_seed(text: str, seed: int) -> int:
    digest = hashlib.blake2b(f"{seed}|{text}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) & 0x7FFF_FFFF


def _comb_count(n: int, k: int) -> int:
    return math.comb(n, k) if n >= k >= 0 else 0


def _pair_keys(combo: Sequence[int]) -> list[tuple[int, int]]:
    vals = sorted(int(x) for x in combo)
    return [(vals[i], vals[j]) for i in range(len(vals)) for j in range(i + 1, len(vals))]


def _candidate_pool(
    indices: Sequence[int],
    *,
    k: int,
    wanted: int,
    seed: int,
    exhaustive_limit: int = 40_000,
    multiplier: int = 64,
) -> list[tuple[int, ...]]:
    idxs = sorted({int(x) for x in indices})
    total = _comb_count(len(idxs), int(k))
    if total <= 0:
        return []
    if total <= exhaustive_limit:
        return [tuple(c) for c in combinations(idxs, int(k))]

    target = min(total, max(int(wanted) * int(multiplier), 4096))
    rng = random.Random(int(seed))
    seen: set[tuple[int, ...]] = set()

    # Seed the candidate set with balanced, disjoint-ish windows.
    rounds = max(2, int(math.ceil(target * int(k) / max(1, len(idxs)))))
    for _ in range(rounds):
        perm = idxs[:]
        rng.shuffle(perm)
        for start in range(0, len(perm) - int(k) + 1, int(k)):
            seen.add(tuple(sorted(perm[start : start + int(k)])))
            if len(seen) >= target:
                return sorted(seen)

    attempts = 0
    max_attempts = max(100_000, target * 100)
    while len(seen) < target and attempts < max_attempts:
        seen.add(tuple(sorted(rng.sample(idxs, int(k)))))
        attempts += 1
    return sorted(seen)


def sample_unordered_combinations(
    indices: Sequence[int],
    *,
    k: int,
    max_mixups: int,
    seed: int,
    sampler: str,
    row_coverage_target: int = 0,
) -> tuple[list[tuple[int, ...]], dict[str, Any]]:
    """Select deterministic unique unordered combinations."""
    idxs = sorted({int(x) for x in indices})
    k = int(k)
    total = _comb_count(len(idxs), k)
    if total <= 0:
        return [], _combination_diagnostics([], idxs, k, sampler, total, 0)

    if int(max_mixups) <= 0:
        wanted = total
    else:
        wanted = min(int(max_mixups), total)
    coverage = max(0, int(row_coverage_target))
    if coverage > 0:
        wanted = min(total, max(wanted, int(math.ceil(len(idxs) * coverage / max(1, k)))))

    mode = str(sampler).lower().replace("-", "_")
    if mode not in {"even_greedy", "random_unique", "exhaustive"}:
        raise ValueError("mixup_sampler must be one of: even_greedy, random_unique, exhaustive")

    if mode == "exhaustive":
        selected = [tuple(c) for c in combinations(idxs, k)]
        return selected, _combination_diagnostics(selected, idxs, k, mode, total, wanted)

    if mode == "random_unique":
        candidates = _candidate_pool(idxs, k=k, wanted=wanted, seed=seed, multiplier=16)
        rng = random.Random(int(seed))
        rng.shuffle(candidates)
        selected = sorted(candidates[:wanted])
        return selected, _combination_diagnostics(selected, idxs, k, mode, total, wanted)

    candidates = _candidate_pool(idxs, k=k, wanted=wanted, seed=seed)
    row_counts: Counter[int] = Counter()
    pair_counts: Counter[tuple[int, int]] = Counter()
    selected: list[tuple[int, ...]] = []
    remaining = list(candidates)

    while remaining and len(selected) < wanted:
        def score(combo: tuple[int, ...]) -> tuple[Any, ...]:
            next_rows = [row_counts[int(i)] + 1 for i in combo]
            pairs = _pair_keys(combo)
            existing_pairs = [pair_counts[p] for p in pairs]
            return (
                max(next_rows),
                sum(v * v for v in next_rows),
                max(existing_pairs, default=0),
                sum(existing_pairs),
                stable_int_seed("combo:" + ",".join(map(str, combo)), seed),
            )

        best_pos = min(range(len(remaining)), key=lambda i: score(remaining[i]))
        combo = remaining.pop(best_pos)
        selected.append(combo)
        row_counts.update(combo)
        pair_counts.update(_pair_keys(combo))

    selected = sorted(selected)
    return selected, _combination_diagnostics(selected, idxs, k, mode, total, wanted)


def _combination_diagnostics(
    combos: Sequence[Sequence[int]],
    indices: Sequence[int],
    k: int,
    sampler: str,
    total_possible: int,
    requested: int,
) -> dict[str, Any]:
    row_counts: Counter[int] = Counter()
    pair_counts: Counter[tuple[int, int]] = Counter()
    for combo in combos:
        row_counts.update(int(x) for x in combo)
        pair_counts.update(_pair_keys(combo))
    counts = np.asarray([row_counts[int(i)] for i in indices], dtype=np.float64)
    pair_vals = np.asarray(list(pair_counts.values()), dtype=np.float64)
    return {
        "sampler": str(sampler),
        "clip_len": int(k),
        "rows_available": len(indices),
        "total_possible_combinations": int(total_possible),
        "requested_combinations": int(requested),
        "generated_combinations": len(combos),
        "row_participation": {
            "min": int(counts.min()) if counts.size else 0,
            "mean": float(counts.mean()) if counts.size else 0.0,
            "max": int(counts.max()) if counts.size else 0,
            "std": float(counts.std()) if counts.size else 0.0,
        },
        "unique_pairs": len(pair_counts),
        "pair_reuse": {
            "mean": float(pair_vals.mean()) if pair_vals.size else 0.0,
            "max": int(pair_vals.max()) if pair_vals.size else 0,
        },
    }


def _meta_value(record: Any, name: str) -> Any:
    if name == "rowid":
        return record.rowid
    if name == "key_id":
        return record.key_id
    row = record.meta.get("row") if isinstance(getattr(record, "meta", None), dict) else None
    if isinstance(row, dict) and name in row:
        return row.get(name)
    return record.meta.get(name)


def record_label(record: Any, cfg: Any) -> str:
    value = _meta_value(record, str(cfg.label_column))
    if value is None or not str(value).strip():
        value = _meta_value(record, str(cfg.fallback_label_column))
    return str(value).strip() if value is not None and str(value).strip() else "seeded"


def seed_pool_key(record: Any, cfg: Any) -> str:
    parts: list[str] = []
    for col in cfg.seed_group_by:
        value = record_label(record, cfg) if col == cfg.label_column else _meta_value(record, col)
        if isinstance(value, (list, tuple)):
            value = ",".join(map(str, value))
        parts.append(f"{col}={value}")
    return "seed|" + "|".join(parts)



def _minimum_rows_for_combinations(clip_len: int, required_combinations: int) -> int:
    n = max(0, int(clip_len))
    required = max(1, int(required_combinations))
    while _comb_count(n, int(clip_len)) < required:
        n += 1
    return n



def _swft_farthest_order(
    features: torch.Tensor,
    indices: Sequence[int],
    *,
    limit: int,
    seed: int,
    candidate_multiplier: int = 8,
) -> list[int]:
    """Deterministic greedy SWFT diversity order."""
    idxs = sorted({int(i) for i in indices})
    wanted = min(max(0, int(limit)), len(idxs))
    if wanted <= 0:
        return []
    if wanted >= len(idxs):
        return idxs

    max_candidates = max(wanted, wanted * max(2, int(candidate_multiplier)))
    if len(idxs) > max_candidates:
        rng = random.Random(int(seed))
        shuffled = idxs[:]
        rng.shuffle(shuffled)
        candidates = sorted(shuffled[:max_candidates])
    else:
        candidates = idxs

    idx_tensor = torch.as_tensor(candidates, dtype=torch.long)
    X = F.normalize(features[idx_tensor].float(), p=2, dim=1, eps=EPS)
    centroid = F.normalize(X.mean(dim=0, keepdim=True), p=2, dim=1, eps=EPS)[0]
    centroid_sim = X @ centroid
    tie = torch.tensor([stable_int_seed(f"swft-first|{i}", seed) for i in candidates], dtype=torch.int64)
    best_sim = float(centroid_sim.max().item())
    first_candidates = torch.nonzero(centroid_sim >= best_sim - 1.0e-12, as_tuple=False).flatten()
    first_rel = int(first_candidates[torch.argmin(tie[first_candidates])].item())

    selected_rel = [first_rel]
    selected_mask = torch.zeros((len(candidates),), dtype=torch.bool)
    selected_mask[first_rel] = True
    min_distance = (1.0 - (X @ X[first_rel])).clamp_min(0.0)
    min_distance[first_rel] = -1.0

    while len(selected_rel) < wanted:
        available = torch.nonzero(~selected_mask, as_tuple=False).flatten()
        if available.numel() == 0:
            break
        vals = min_distance[available]
        max_val = float(vals.max().item())
        tied = available[torch.nonzero(vals >= max_val - 1.0e-12, as_tuple=False).flatten()]
        rel = int(tied[torch.argmin(tie[tied])].item())
        selected_rel.append(rel)
        selected_mask[rel] = True
        d = (1.0 - (X @ X[rel])).clamp_min(0.0)
        min_distance = torch.minimum(min_distance, d)
        min_distance[selected_mask] = -1.0

    selected = [candidates[i] for i in selected_rel]
    if len(selected) < wanted:
        remaining = [i for i in idxs if i not in set(selected)]
        selected.extend(remaining[: wanted - len(selected)])
    return selected[:wanted]


def select_seed_rows_stratified_swft(
    records: Sequence[Any],
    features: torch.Tensor,
    cfg: Any,
    *,
    max_rows_per_pool: int = 0,
    max_rows_per_label: int = 0,
    seed: int = 1234,
) -> tuple[list[Any], torch.Tensor, dict[str, Any]]:
    """Apply stratified SWFT diversity selection before clip construction."""
    n = len(records)
    if n != int(features.shape[0]):
        raise ValueError(f"records/features mismatch: {n} vs {tuple(features.shape)}")
    pool_cap = max(0, int(max_rows_per_pool))
    label_cap = max(0, int(max_rows_per_label))
    if pool_cap <= 0 and label_cap <= 0:
        return list(records), features.float().cpu().contiguous(), {
            "applied": False, "input_rows": n, "selected_rows": n,
            "max_rows_per_pool": pool_cap, "max_rows_per_label": label_cap,
        }

    pools: dict[str, list[int]] = defaultdict(list)
    labels: dict[str, str] = {}
    for i, rec in enumerate(records):
        pool = seed_pool_key(rec, cfg)
        pools[pool].append(int(i))
        labels[pool] = record_label(rec, cfg)

    ordered_by_pool: dict[str, list[int]] = {}
    pool_summary: dict[str, Any] = {}
    for pool, idxs in sorted(pools.items()):
        keep_n = len(idxs) if pool_cap <= 0 else min(len(idxs), pool_cap)
        order = _swft_farthest_order(
            features, idxs, limit=keep_n,
            seed=stable_int_seed(f"seed-pool-diversity|{pool}", int(seed)),
        )
        ordered_by_pool[pool] = order
        pool_summary[pool] = {
            "label": labels[pool], "input_rows": len(idxs),
            "after_pool_cap": len(order),
        }

    pools_by_label: dict[str, list[str]] = defaultdict(list)
    for pool, lab in labels.items():
        if ordered_by_pool.get(pool):
            pools_by_label[lab].append(pool)

    final_indices: list[int] = []
    label_summary: dict[str, Any] = {}
    min_pool = max(1, int(getattr(cfg, "min_seed_group_size", 4)))
    for lab, lab_pools in sorted(pools_by_label.items()):
        candidates = [i for pool in sorted(lab_pools) for i in ordered_by_pool[pool]]
        if label_cap <= 0 or len(candidates) <= label_cap:
            chosen = candidates
        else:
            ranked_pools = sorted(lab_pools, key=lambda p: (-len(ordered_by_pool[p]), p))
            mandatory: list[int] = []
            for pool in ranked_pools:
                take = min(min_pool, len(ordered_by_pool[pool]))
                if len(mandatory) + take > label_cap:
                    continue
                mandatory.extend(ordered_by_pool[pool][:take])
            remaining_candidates = [i for i in candidates if i not in set(mandatory)]
            remaining_budget = label_cap - len(mandatory)
            diverse_fill = _swft_farthest_order(
                features, remaining_candidates, limit=remaining_budget,
                seed=stable_int_seed(f"seed-label-diversity|{lab}", int(seed)),
            ) if remaining_budget > 0 else []
            chosen = mandatory + diverse_fill
        final_indices.extend(chosen)
        label_summary[lab] = {
            "input_rows": sum(len(pools[p]) for p in lab_pools),
            "after_pool_caps": len(candidates),
            "selected_rows": len(chosen),
            "seed_pools": len(lab_pools),
        }

    selected_set = set(final_indices)
    final_indices = [i for i in range(n) if i in selected_set]
    selected_records = [records[i] for i in final_indices]
    for new_i, rec in enumerate(selected_records):
        rec.index = int(new_i)
    selected_features = features[torch.as_tensor(final_indices, dtype=torch.long)].float().cpu().contiguous()
    for pool in pool_summary:
        pool_summary[pool]["selected_rows"] = sum(1 for i in ordered_by_pool[pool] if i in selected_set)
    return selected_records, selected_features, {
        "applied": True,
        "selection": "stratified_swft_greedy_farthest",
        "input_rows": n,
        "selected_rows": len(selected_records),
        "discarded_rows": n - len(selected_records),
        "max_rows_per_pool": pool_cap,
        "max_rows_per_label": label_cap,
        "by_label": label_summary,
        "by_pool": pool_summary,
    }


def _allocate_clip_budgets(
    specs: Sequence[dict[str, Any]],
    *,
    global_cap: int,
    per_pool_cap: int,
    seed: int,
) -> dict[int, int]:
    requests: dict[int, int] = {}
    for pos, spec in enumerate(specs):
        req = int(spec["request"])
        if int(per_pool_cap) > 0:
            req = min(req, int(per_pool_cap))
        requests[pos] = max(0, req)
    cap = max(0, int(global_cap))
    if cap <= 0 or sum(requests.values()) <= cap:
        return requests

    active = [i for i, req in requests.items() if req > 0]
    budgets = {i: 0 for i in requests}

    active = sorted(active, key=lambda i: (-int(specs[i]["rows"]), str(specs[i]["pool"])))
    for i in active[:cap]:
        budgets[i] = 1
    remaining = cap - sum(budgets.values())
    if remaining <= 0:
        return budgets
    total_weight = sum(max(1.0, math.sqrt(float(requests[i]))) for i in active)
    fractions: list[tuple[float, int]] = []
    for i in active:
        room = requests[i] - budgets[i]
        if room <= 0:
            continue
        ideal = remaining * max(1.0, math.sqrt(float(requests[i]))) / max(total_weight, EPS)
        add = min(room, int(math.floor(ideal)))
        budgets[i] += add
        fractions.append((ideal - math.floor(ideal), i))
    remaining = cap - sum(budgets.values())
    order = sorted(fractions, key=lambda x: (-x[0], stable_int_seed(f"budget|{x[1]}", seed)))
    while remaining > 0:
        progressed = False
        for _frac, i in order:
            if budgets[i] < requests[i]:
                budgets[i] += 1
                remaining -= 1
                progressed = True
                if remaining <= 0:
                    break
        if not progressed:
            break
    return budgets


def split_seed_rows(
    records: Sequence[Any],
    cfg: Any,
    *,
    calibration_fraction: float,
    split_seed: int,
    conformal_k: int,
    min_proper_anchors: int,
    min_calibration_anchors: int,
) -> tuple[dict[int, str], dict[str, Any]]:
    """Partition source rows before any anchor combinations are constructed."""
    clip_min = max(int(x) for x in cfg.clip_lens)
    proper_required_clips = max(int(conformal_k), int(min_proper_anchors))
    calibration_required_clips = int(min_calibration_anchors)
    proper_min_rows = max(_minimum_rows_for_combinations(int(k), proper_required_clips) for k in cfg.clip_lens)
    calibration_min_rows = max(_minimum_rows_for_combinations(int(k), calibration_required_clips) for k in cfg.clip_lens)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        groups[seed_pool_key(rec, cfg)].append(int(i))

    groups_by_label: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
    for pool_id, idxs in sorted(groups.items()):
        if len(idxs) >= int(cfg.min_seed_group_size):
            label = record_label(records[idxs[0]], cfg)
            groups_by_label[label].append((pool_id, sorted(idxs)))

    assignment: dict[int, str] = {}
    label_summary: dict[str, Any] = {}
    frac = min(0.80, max(0.20, float(calibration_fraction)))

    for label, label_groups in sorted(groups_by_label.items()):
        proper_groups: list[str] = []
        calibration_groups: list[str] = []
        small_groups: list[tuple[str, list[int]]] = []
        proper_n = 0
        calibration_n = 0

        for pool_id, idxs in label_groups:
            n = len(idxs)
            if n >= proper_min_rows + calibration_min_rows:
                shuffled = idxs[:]
                random.Random(stable_int_seed(f"split|{label}|{pool_id}", split_seed)).shuffle(shuffled)
                cal_n = max(calibration_min_rows, int(round(frac * n)))
                cal_n = min(n - proper_min_rows, cal_n)
                cal = sorted(shuffled[:cal_n])
                proper = sorted(shuffled[cal_n:])
                for i in proper:
                    assignment[i] = PARTITION_PROPER
                for i in cal:
                    assignment[i] = PARTITION_CALIBRATION
                proper_n += len(proper)
                calibration_n += len(cal)
                proper_groups.append(pool_id)
                calibration_groups.append(pool_id)
            elif n >= clip_min:
                small_groups.append((pool_id, idxs))

        for pool_id, idxs in sorted(small_groups, key=lambda x: (-len(x[1]), x[0])):
            if proper_n < proper_min_rows:
                side = PARTITION_PROPER
            elif calibration_n < calibration_min_rows:
                side = PARTITION_CALIBRATION
            else:
                current_frac = calibration_n / max(1, proper_n + calibration_n)
                side = PARTITION_CALIBRATION if current_frac < frac else PARTITION_PROPER
            for i in idxs:
                assignment[i] = side
            if side == PARTITION_PROPER:
                proper_n += len(idxs)
                proper_groups.append(pool_id)
            else:
                calibration_n += len(idxs)
                calibration_groups.append(pool_id)

        if proper_n < proper_min_rows or calibration_n < calibration_min_rows:
            raise ValueError(
                f"Mach7 disjoint conformal split is impossible for label {label!r}: "
                f"proper_rows={proper_n} (need about {proper_min_rows}), "
                f"calibration_rows={calibration_n} (need about {calibration_min_rows}), clip_len={clip_min}. "
                "Add seed rows or seed pools; Mach7 will not reuse a source row across proper and calibration sets."
            )

        proper_rowids = {int(records[i].rowid) for i, side in assignment.items() if side == PARTITION_PROPER and record_label(records[i], cfg) == label}
        calibration_rowids = {int(records[i].rowid) for i, side in assignment.items() if side == PARTITION_CALIBRATION and record_label(records[i], cfg) == label}
        label_summary[label] = {
            "proper_source_rows": len(proper_rowids),
            "calibration_source_rows": len(calibration_rowids),
            "proper_seed_pools": sorted(set(proper_groups)),
            "calibration_seed_pools": sorted(set(calibration_groups)),
            "cross_partition_shared_source_rows": len(proper_rowids & calibration_rowids),
        }

    unassigned = [i for i in range(len(records)) if i not in assignment]
    return assignment, {
        "by_label": label_summary,
        "calibration_fraction_requested": float(calibration_fraction),
        "proper_required_anchor_clips": proper_required_clips,
        "calibration_required_anchor_clips": calibration_required_clips,
        "estimated_minimum_proper_rows_per_class": proper_min_rows,
        "estimated_minimum_calibration_rows_per_class": calibration_min_rows,
        "excluded_seed_rows_below_group_minimum": len(unassigned),
    }


def build_partitioned_anchor_clips(
    records: Sequence[Any],
    features: torch.Tensor,
    cfg: Any,
    *,
    planned_clip_cls: Any,
    row_partition: dict[int, str],
    sampler: str,
    max_proper_clips_per_label: int = 0,
    max_calibration_clips_per_label: int = 0,
    max_anchor_clips_per_label: int = 0,
    max_anchor_clips_per_pool: int = 0,
) -> tuple[list[Any], torch.Tensor, dict[str, Any]]:
    partition_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        if int(i) not in row_partition:
            continue
        partition_groups[(row_partition[int(i)], seed_pool_key(rec, cfg))].append(int(i))

    specs_by_budget: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for (partition, base_pool), idxs in sorted(partition_groups.items()):
        if len(idxs) < int(cfg.min_seed_group_size):
            continue
        label = record_label(records[idxs[0]], cfg)
        for clip_len0 in cfg.clip_lens:
            clip_len = int(clip_len0)
            total = _comb_count(len(idxs), clip_len)
            if total <= 0:
                continue
            request = total if int(cfg.mixups) <= 0 else min(int(cfg.mixups), total)
            if int(cfg.anchor_mixups_per_row) > 0:
                request = min(total, max(request, int(math.ceil(len(idxs) * int(cfg.anchor_mixups_per_row) / max(1, clip_len)))))
            specs_by_budget[(partition, label, clip_len)].append({
                "partition": partition, "label": label, "clip_len": clip_len,
                "pool": base_pool, "idxs": idxs, "rows": len(idxs),
                "total": total, "request": request,
            })

    clips: list[Any] = []
    combo_features: list[torch.Tensor] = []
    diagnostics: dict[str, Any] = {PARTITION_PROPER: {}, PARTITION_CALIBRATION: {}, "global_budgets": {}}

    budget_items = sorted(specs_by_budget.items())
    for (partition, label, clip_len), specs in tqdm(
        budget_items,
        desc="Mach7 anchor clip groups",
        unit="group",
        total=len(budget_items),
        disable=not bool(getattr(cfg, "progress", True)),
        leave=False,
        dynamic_ncols=True,
    ):
        if partition == PARTITION_CALIBRATION:
            global_cap = int(max_calibration_clips_per_label)
        elif int(max_proper_clips_per_label) > 0:
            global_cap = int(max_proper_clips_per_label)
        else:
            global_cap = int(max_anchor_clips_per_label)
        budgets = _allocate_clip_budgets(
            specs, global_cap=global_cap, per_pool_cap=int(max_anchor_clips_per_pool),
            seed=stable_int_seed(f"anchor-budget|{partition}|{label}|k{clip_len}", int(cfg.seed)),
        )
        diagnostics["global_budgets"][f"{partition}|{label}|k{clip_len}"] = {
            "requested": sum(int(s["request"]) for s in specs),
            "allocated": sum(budgets.values()),
            "global_cap": global_cap,
            "per_pool_cap": int(max_anchor_clips_per_pool),
            "seed_pools": len(specs),
        }
        for pos, spec in enumerate(specs):
            budget = int(budgets.get(pos, 0))
            base_pool = str(spec["pool"]); idxs = list(spec["idxs"])
            if budget <= 0:
                diagnostics[partition][f"{base_pool}|k{clip_len}"] = {
                    "rows_available": len(idxs), "generated_combinations": 0,
                    "reason": "global_clip_budget",
                }
                continue
            combos, diag = sample_unordered_combinations(
                idxs, k=clip_len, max_mixups=budget,
                seed=stable_int_seed(f"anchor|{partition}|{base_pool}|k{clip_len}", int(cfg.seed)),
                sampler=sampler, row_coverage_target=0,
            )
            diag["global_allocated_budget"] = budget
            diagnostics[partition][f"{base_pool}|k{clip_len}"] = diag
            pool_id = f"anchor_{partition}|{base_pool}"
            for combo in combos:
                recs = sorted([records[i] for i in combo], key=lambda r: (int(r.key_id), int(r.rowid)))
                actual_idxs = [int(r.index) for r in recs]
                key_ids = tuple(int(r.key_id) for r in recs)
                rowids = tuple(int(r.rowid) for r in recs)
                record_ids = tuple(str(r.record_id) for r in recs)
                labels = [record_label(r, cfg) for r in recs]
                clip_label_value = Counter(labels).most_common(1)[0][0]
                clip_id = f"anchor:{partition}:{base_pool}:k{clip_len}:" + "-".join(map(str, key_ids))
                clip = planned_clip_cls(
                    split="anchor", clip_id=clip_id, clip_len=clip_len,
                    key_ids=key_ids, rowids=rowids, record_ids=record_ids,
                    pool_id=pool_id, label=clip_label_value,
                )
                setattr(clip, "mach7_partition", partition)
                clips.append(clip)
                combo_features.append(features[actual_idxs].mean(dim=0))

    if not combo_features:
        raise ValueError("Mach7 produced zero anchor clips")
    anchor_features = F.normalize(torch.stack(combo_features).float(), p=2, dim=1, eps=EPS).contiguous()
    proper_rows = {rid for c in clips if getattr(c, "mach7_partition", "") == PARTITION_PROPER for rid in c.rowids}
    calibration_rows = {rid for c in clips if getattr(c, "mach7_partition", "") == PARTITION_CALIBRATION for rid in c.rowids}
    diagnostics["cross_partition_shared_source_rows"] = len(proper_rows & calibration_rows)
    diagnostics["proper_anchor_clips"] = sum(getattr(c, "mach7_partition", "") == PARTITION_PROPER for c in clips)
    diagnostics["calibration_anchor_clips"] = sum(getattr(c, "mach7_partition", "") == PARTITION_CALIBRATION for c in clips)
    if diagnostics["cross_partition_shared_source_rows"] != 0:
        raise AssertionError("Mach7 invariant failed: proper and calibration anchor clips share source rows")
    return clips, anchor_features, diagnostics


def _local_partner_lists(
    idxs: Sequence[int],
    features: torch.Tensor,
    *,
    local_k: int,
    progress: bool = True,
    desc: str = "Mach7 query local SWFT neighborhoods",
) -> dict[int, list[int]]:
    pool = sorted(int(x) for x in idxs)
    feat = F.normalize(features.float(), p=2, dim=1, eps=EPS)
    result: dict[int, list[int]] = {}
    for subject in tqdm(
        pool,
        desc=desc,
        unit="row",
        total=len(pool),
        disable=not bool(progress),
        leave=False,
        dynamic_ncols=True,
    ):
        partners = [i for i in pool if i != subject]
        if int(local_k) > 0 and len(partners) > int(local_k):
            p_tensor = torch.tensor(partners, dtype=torch.long)
            sims = feat[p_tensor] @ feat[int(subject)]
            order = torch.topk(sims, k=min(int(local_k), len(partners)), largest=True, sorted=True).indices.tolist()
            partners = [int(p_tensor[j].item()) for j in order]
        result[subject] = partners
    return result


def build_query_clips_mach7(
    query_records: Sequence[Any],
    query_features: torch.Tensor,
    query_pools_by_idx: dict[str, list[int]],
    cfg: Any,
    *,
    planned_clip_cls: Any,
    sampler: str,
) -> tuple[dict[str, list[str]], list[Any], dict[str, Any]]:
    mode = str(cfg.query_mixup_mode).lower().replace("-", "_")
    if mode != "row_centered":
        raise ValueError("Mach7 conformal ingress requires query_mixup_mode=row_centered")

    pools_by_record_id: dict[str, list[str]] = {}
    clips: list[Any] = []
    diagnostics: dict[str, Any] = {}

    query_pool_items = sorted(query_pools_by_idx.items())
    for pool_id, idxs0 in tqdm(
        query_pool_items,
        desc="Mach7 query mixup pools",
        unit="pool",
        total=len(query_pool_items),
        disable=not bool(getattr(cfg, "progress", True)),
        leave=False,
        dynamic_ncols=True,
    ):
        idxs = sorted(int(x) for x in idxs0)
        pools_by_record_id[pool_id] = [query_records[i].record_id for i in idxs]
        if len(idxs) < int(cfg.min_query_group_size):
            diagnostics[pool_id] = {"rows": len(idxs), "generated_clips": 0, "reason": "below_min_query_group_size"}
            continue

        local_partners = _local_partner_lists(
            idxs,
            query_features,
            local_k=int(cfg.query_local_k),
            progress=bool(getattr(cfg, "progress", True)),
            desc=f"Mach7 local SWFT neighbors {pool_id}",
        )
        participant_counts: Counter[int] = Counter()
        pair_counts: Counter[tuple[int, int]] = Counter()
        selected_by_subject: dict[int, set[tuple[int, ...]]] = defaultdict(set)
        subject_counts: Counter[int] = Counter()
        pool_clips_before = len(clips)

        for clip_len in cfg.clip_lens:
            clip_len = int(clip_len)
            partner_k = clip_len - 1
            rounds = max(1, int(cfg.query_mixups_per_row))
            subject_progress = tqdm(
                total=rounds * len(idxs),
                desc=f"Mach7 query k={clip_len} subject contexts {pool_id}",
                unit="context",
                disable=not bool(getattr(cfg, "progress", True)),
                leave=False,
                dynamic_ncols=True,
            )
            try:
                for round_i in range(rounds):
                    subject_order = idxs[:]
                    random.Random(stable_int_seed(f"query-order|{pool_id}|k{clip_len}|round{round_i}", int(cfg.seed))).shuffle(subject_order)
                    for subject in subject_order:
                        subject_progress.update(1)
                        partners = local_partners.get(subject, [])
                        total = _comb_count(len(partners), partner_k)
                        if total <= 0:
                            continue
                        if str(sampler).lower().replace("-", "_") == "exhaustive":
                            candidate_partners = [tuple(c) for c in combinations(partners, partner_k)]
                        else:
                            candidate_partners = _candidate_pool(
                                partners,
                                k=partner_k,
                                wanted=max(rounds, 64),
                                seed=stable_int_seed(f"query-candidates|{pool_id}|subject{subject}|k{clip_len}", int(cfg.seed)),
                                multiplier=16,
                            )
                        candidate_partners = [c for c in candidate_partners if c not in selected_by_subject[subject]]
                        if not candidate_partners:
                            continue

                        sampler_mode = str(sampler).lower().replace("-", "_")
                        if sampler_mode == "random_unique":
                            rng = random.Random(stable_int_seed(f"query-select|{pool_id}|subject{subject}|k{clip_len}|round{round_i}", int(cfg.seed)))
                            partners_selected = candidate_partners[rng.randrange(len(candidate_partners))]
                        else:
                            def score(partner_combo: tuple[int, ...]) -> tuple[Any, ...]:
                                full = tuple(sorted((subject, *partner_combo)))
                                next_counts = [participant_counts[i] + 1 for i in full]
                                existing_pairs = [pair_counts[p] for p in _pair_keys(full)]
                                subject_pair_reuse = sum(pair_counts[tuple(sorted((subject, p)))] for p in partner_combo)
                                return (
                                    max(next_counts),
                                    sum(v * v for v in next_counts),
                                    subject_pair_reuse,
                                    max(existing_pairs, default=0),
                                    sum(existing_pairs),
                                    stable_int_seed("query-combo:" + ",".join(map(str, full)), int(cfg.seed)),
                                )
                            partners_selected = min(candidate_partners, key=score)

                        selected_by_subject[subject].add(tuple(partners_selected))
                        full_combo = tuple(sorted((subject, *partners_selected)))
                        participant_counts.update(full_combo)
                        pair_counts.update(_pair_keys(full_combo))
                        subject_counts[subject] += 1

                        subject_rec = query_records[subject]
                        recs = sorted([query_records[i] for i in full_combo], key=lambda r: (int(r.key_id), int(r.rowid)))
                        key_ids = tuple(int(r.key_id) for r in recs)
                        rowids = tuple(int(r.rowid) for r in recs)
                        record_ids = tuple(str(r.record_id) for r in recs)
                        clip_id = (
                            f"query:{pool_id}:k{clip_len}:subject{int(subject_rec.key_id)}:"
                            f"mix{int(subject_counts[subject]) - 1}:" + "-".join(map(str, key_ids))
                        )
                        clips.append(planned_clip_cls(
                            split="query",
                            clip_id=clip_id,
                            clip_len=clip_len,
                            key_ids=key_ids,
                            rowids=rowids,
                            record_ids=record_ids,
                            pool_id=pool_id,
                            label=None,
                            coarse_group=int(pool_id.rsplit("_", 1)[-1]) if "_" in pool_id else None,
                            subject_record_id=subject_rec.record_id,
                            subject_key_id=int(subject_rec.key_id),
                            subject_rowid=int(subject_rec.rowid),
                        ))
            finally:
                subject_progress.close()

        subject_vals = np.asarray([subject_counts[i] for i in idxs], dtype=np.float64)
        participant_vals = np.asarray([participant_counts[i] for i in idxs], dtype=np.float64)
        pair_vals = np.asarray(list(pair_counts.values()), dtype=np.float64)
        diagnostics[pool_id] = {
            "rows": len(idxs),
            "generated_clips": len(clips) - pool_clips_before,
            "sampler": str(sampler),
            "subject_clip_count": {
                "min": int(subject_vals.min()) if subject_vals.size else 0,
                "mean": float(subject_vals.mean()) if subject_vals.size else 0.0,
                "max": int(subject_vals.max()) if subject_vals.size else 0,
                "std": float(subject_vals.std()) if subject_vals.size else 0.0,
            },
            "participant_clip_count": {
                "min": int(participant_vals.min()) if participant_vals.size else 0,
                "mean": float(participant_vals.mean()) if participant_vals.size else 0.0,
                "max": int(participant_vals.max()) if participant_vals.size else 0,
                "std": float(participant_vals.std()) if participant_vals.size else 0.0,
            },
            "unique_pairs": len(pair_counts),
            "pair_reuse": {
                "mean": float(pair_vals.mean()) if pair_vals.size else 0.0,
                "max": int(pair_vals.max()) if pair_vals.size else 0,
            },
        }

    return pools_by_record_id, clips, diagnostics


def anchor_partition(clip: Any) -> str:
    value = getattr(clip, "mach7_partition", None)
    if value in {PARTITION_PROPER, PARTITION_CALIBRATION}:
        return str(value)
    pool = str(getattr(clip, "pool_id", ""))
    if pool.startswith("anchor_proper|"):
        return PARTITION_PROPER
    if pool.startswith("anchor_calibration|"):
        return PARTITION_CALIBRATION
    return "unknown"
