"""Command-line entry point for cfetch reference-guided inference."""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


METHODS: dict[str, dict[str, str]] = {
    "conformal": {
        "module": "cfetch.inference.conformal",
        "directory": "conformal",
        "config": "conformal.json",
    },
    "graph_diffusion": {
        "module": "cfetch.inference.graph",
        "directory": "graph_diffusion",
        "config": "graph.json",
    },
    "probe": {
        "module": "cfetch.inference.probe",
        "directory": "probe",
        "config": "probe.json",
    },
}

METHOD_ALIASES = {
    "1": "conformal",
    "2": "graph_diffusion",
    "3": "probe",
    "conformal-decay": "conformal",
    "graph": "graph_diffusion",
    "graph-diffusion": "graph_diffusion",
    "probe-stability": "probe",
}

VOTE_COLUMNS = {
    "rough_label_vote",
    "rough_label_vote_score",
    "rough_label_vote_fraction",
    "rough_label_vote_margin",
}
COORDINATE_COLUMNS = ("chrom1", "x1", "x2", "chrom2", "y1", "y2")


def _print_top_help() -> None:
    print(
        "usage:\n"
        "  cfetch-infer [all] [--config-dir DIR] [--consensus-threshold {1,2,3}] OPTIONS\n"
        "  cfetch-infer single {1,2,3,conformal,graph_diffusion,probe} [--config FILE] OPTIONS\n\n"
        "The default mode runs all three CFetch assignment methods and writes a\n"
        "method-voted consensus. 'single 1', 'single 2', and 'single 3' run\n"
        "conformal, graph-diffusion, and probe assignment, respectively.\n\n"
    )


def _config_tokens(config: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    negative_flags = {
        "progress": "--no-progress",
        "symmetrize": "--no-symmetrize",
        "clamp_anchors": "--no-clamp-anchors",
    }
    for key, value in config.items():
        option = "--" + str(key).replace("_", "-")
        if isinstance(value, bool):
            if value:
                tokens.append(option)
            elif key in negative_flags:
                tokens.append(negative_flags[key])
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = ",".join(str(item) for item in value)
        tokens.extend([option, str(value)])
    return tokens


def _load_config(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise SystemExit(f"Configuration must be a JSON object: {config_path}")
    return _config_tokens(config)


def _resolve_config_dir(explicit: str | Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("CFETCH_CONFIG_DIR"):
        candidates.append(Path(os.environ["CFETCH_CONFIG_DIR"]))
    candidates.extend(
        [
            Path.cwd() / "configs" / "paper",
            Path(__file__).resolve().parents[2] / "configs" / "paper",
            Path(sys.prefix) / "share" / "cfetch" / "configs" / "paper",
        ]
    )
    for candidate in candidates:
        if all((candidate / spec["config"]).is_file() for spec in METHODS.values()):
            return candidate.resolve()
    rendered = ", ".join(str(path) for path in candidates)
    raise SystemExit(f"Could not locate the paper configurations. Checked: {rendered}")


def _resolve_method(value: str) -> str:
    method = METHOD_ALIASES.get(value, value)
    if method not in METHODS:
        valid = "1, 2, 3, conformal, graph_diffusion, or probe"
        raise SystemExit(f"Unknown single-method selector {value!r}; use {valid}")
    return method


def _run_method(method: str, arguments: Sequence[str]) -> None:
    module = importlib.import_module(METHODS[method]["module"])
    module.main(list(arguments))


def _coordinate_key(row: dict[str, str], columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(row[column]) for column in columns)


def _read_method_table(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"Missing TSV header: {path}")
        rows = [dict(row) for row in reader]
    missing = sorted(set(COORDINATE_COLUMNS + tuple(VOTE_COLUMNS)) - set(reader.fieldnames))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return list(reader.fieldnames), rows


def _score_percentiles(rows: Sequence[dict[str, str]]) -> list[float]:
    values: list[tuple[float, int]] = []
    for index, row in enumerate(rows):
        try:
            value = float(row["rough_label_vote_score"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid rough_label_vote_score at row {index + 2}") from error
        if not math.isfinite(value):
            raise ValueError(f"Non-finite rough_label_vote_score at row {index + 2}")
        values.append((value, index))

    percentiles = [0.0] * len(values)
    ordered = sorted(values)
    start = 0
    total = len(ordered)
    while start < total:
        stop = start + 1
        while stop < total and ordered[stop][0] == ordered[start][0]:
            stop += 1
        average_one_based_rank = ((start + 1) + stop) / 2.0
        percentile = average_one_based_rank / total if total else 0.0
        for _, original_index in ordered[start:stop]:
            percentiles[original_index] = percentile
        start = stop
    return percentiles


def _write_consensus(
    method_paths: dict[str, Path],
    out_dir: Path,
    *,
    threshold: int = 2,
) -> Path:
    if threshold not in {1, 2, 3}:
        raise ValueError("Consensus threshold must be 1, 2, or 3")

    headers: dict[str, list[str]] = {}
    key_columns: list[str] | None = None
    ordered_keys: list[tuple[str, ...]] | None = None
    indexed: dict[str, dict[tuple[str, ...], dict[str, str]]] = {}
    percentiles: dict[str, dict[tuple[str, ...], float]] = {}

    for method in METHODS:
        header, rows = _read_method_table(method_paths[method])
        headers[method] = header
        if key_columns is None:
            key_columns = list(COORDINATE_COLUMNS)
            if "resolution" in header:
                key_columns.append("resolution")
        keys = [_coordinate_key(row, key_columns) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate candidate coordinates in {method_paths[method]}")
        if ordered_keys is None:
            ordered_keys = keys
        elif set(keys) != set(ordered_keys):
            raise ValueError(f"Candidate universe differs for method {method}")
        indexed[method] = dict(zip(keys, rows))
        percentiles[method] = dict(zip(keys, _score_percentiles(rows)))

    assert key_columns is not None and ordered_keys is not None
    base_header = headers["conformal"]
    metadata_columns = [column for column in base_header if column not in VOTE_COLUMNS]
    method_columns = [
        f"{method}_{suffix}"
        for method in METHODS
        for suffix in ("label", "score", "score_percentile", "fraction", "margin")
    ]
    consensus_columns = [
        "rough_label_vote",
        "rough_label_vote_score",
        "rough_label_vote_fraction",
        "rough_label_vote_margin",
        "method_support",
        "consensus_threshold",
        "consensus_score",
    ]
    output_columns = [*metadata_columns, *consensus_columns, *method_columns]

    output_rows: list[dict[str, Any]] = []
    support_counts: Counter[tuple[str, int]] = Counter()
    for key in ordered_keys:
        source_rows = {method: indexed[method][key] for method in METHODS}
        labels = [source_rows[method]["rough_label_vote"].strip() or "Unscored" for method in METHODS]
        label_counts = Counter(labels)
        ranked = label_counts.most_common()
        top_label, top_support = ranked[0]
        second_support = ranked[1][1] if len(ranked) > 1 else 0
        tied = len(ranked) > 1 and ranked[1][1] == top_support
        consensus_label = top_label if top_support >= threshold and not tied else "NoConsensus"
        consensus_score = sum(percentiles[method][key] for method in METHODS) / len(METHODS)

        result: dict[str, Any] = {
            column: source_rows["conformal"].get(column, "") for column in metadata_columns
        }
        result.update(
            {
                "rough_label_vote": consensus_label,
                "rough_label_vote_score": f"{consensus_score:.12g}",
                "rough_label_vote_fraction": f"{top_support / len(METHODS):.12g}",
                "rough_label_vote_margin": f"{(top_support - second_support) / len(METHODS):.12g}",
                "method_support": top_support,
                "consensus_threshold": threshold,
                "consensus_score": f"{consensus_score:.12g}",
            }
        )
        for method, row in source_rows.items():
            result[f"{method}_label"] = row["rough_label_vote"]
            result[f"{method}_score"] = row["rough_label_vote_score"]
            result[f"{method}_score_percentile"] = f"{percentiles[method][key]:.12g}"
            result[f"{method}_fraction"] = row["rough_label_vote_fraction"]
            result[f"{method}_margin"] = row["rough_label_vote_margin"]
        output_rows.append(result)
        support_counts[(consensus_label, top_support)] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    consensus_path = out_dir / "query_results.tsv"
    with consensus_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    agreement_path = out_dir / "method_agreement.tsv"
    with agreement_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["consensus_label", "supporting_methods", "candidate_count", "candidate_fraction"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for (label, support), count in sorted(support_counts.items()):
            writer.writerow(
                {
                    "consensus_label": label,
                    "supporting_methods": support,
                    "candidate_count": count,
                    "candidate_fraction": count / len(output_rows) if output_rows else 0.0,
                }
            )

    summary = {
        "methods": list(METHODS),
        "consensus_threshold": threshold,
        "candidate_count": len(output_rows),
        "categorical_consensus": "A label is assigned when at least threshold methods agree.",
        "continuous_score": "Mean within-method percentile rank of rough_label_vote_score.",
        "method_outputs": {method: str(path) for method, path in method_paths.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return consensus_path


def _run_single(selector: str, arguments: list[str]) -> None:
    method = _resolve_method(selector)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config")
    parser.add_argument("--config-dir")
    parser.add_argument("--no-paper-config", action="store_true")
    options, method_args = parser.parse_known_args(arguments)

    if options.config and options.no_paper_config:
        raise SystemExit("Use either --config or --no-paper-config, not both")
    config_path: Path | None = None
    if options.config:
        config_path = Path(options.config)
    elif not options.no_paper_config:
        config_path = _resolve_config_dir(options.config_dir) / METHODS[method]["config"]
    _run_method(method, [*_load_config(config_path), *method_args])


def _run_all(arguments: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="cfetch-infer all",
        description="Run all three CFetch assignment methods and build a method-voted consensus.",
    )
    parser.add_argument("--config-dir")
    parser.add_argument("--consensus-threshold", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--out-dir", required=True)
    options, shared_args = parser.parse_known_args(arguments)

    config_dir = _resolve_config_dir(options.config_dir)
    output_root = Path(options.out_dir)
    method_paths: dict[str, Path] = {}
    for method, specification in METHODS.items():
        method_out = output_root / "method_outputs" / specification["directory"]
        method_args = [
            *_load_config(config_dir / specification["config"]),
            *shared_args,
            "--out-dir",
            str(method_out),
        ]
        _run_method(method, method_args)
        method_paths[method] = method_out / "query_results.tsv"

    consensus_path = _write_consensus(
        method_paths,
        output_root / "consensus",
        threshold=options.consensus_threshold,
    )
    print(
        {
            "status": "complete",
            "mode": "all",
            "consensus_threshold": options.consensus_threshold,
            "consensus": str(consensus_path),
        },
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] in {"-h", "--help"}:
        _print_top_help()
        return

    if raw[0] == "all":
        _run_all(raw[1:])
        return
    if raw[0] == "single":
        if len(raw) < 2:
            raise SystemExit("cfetch-infer single requires 1, 2, 3, conformal, graph_diffusion, or probe")
        _run_single(raw[1], raw[2:])
        return
    if raw[0] in METHODS or raw[0] in METHOD_ALIASES:
        _run_single(raw[0], raw[1:])
        return

    # No mode was supplied: the public default is the three-method ensemble.
    _run_all(raw)


if __name__ == "__main__":
    main()
