from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
from urllib.parse import quote

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset

TABLE = "imag_with_seqs"
BASE_COLUMNS = (
    "key_id", "numpyarr", "viewing_vmax", "true_max",
    "dataset", "coordinates", "dimensions", "resolution", "toolsource",
)


@dataclass(frozen=True)
class StrainerDataConfig:
    db_paths: tuple[str, ...]
    table: str = TABLE
    target_side: Optional[int] = 128
    batch_size: int = 1
    shuffle: bool = False
    num_workers: int = 0
    pin_memory: Optional[bool] = None
    normalize_by_viewing_vmax: bool = True
    clamp_01: bool = True
    read_verbose: bool = False
    extra_columns: tuple[str, ...] = ()
    where: Optional[str] = None
    limit: Optional[int] = None
    offset: int = 0


def _q(name: str) -> str:
    if not name or "\x00" in name:
        raise ValueError(f"Invalid SQLite identifier: {name!r}")
    return '"' + name.replace('"', '""') + '"'


def _connect(path: str | Path) -> sqlite3.Connection:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)
    conn = sqlite3.connect("file:" + quote(str(p), safe="/:\\") + "?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def get_table_columns(db_path: str | Path, table: str = TABLE) -> list[str]:
    with _connect(db_path) as conn:
        rows = conn.execute(f"PRAGMA table_info({_q(table)})").fetchall()
    cols = [str(r["name"]) for r in rows]
    if not cols:
        raise ValueError(f"No table {table!r} in {db_path!s}")
    return cols


def validate_database(
    db_path: str | Path,
    *,
    table: str = TABLE,
    decode_sample: bool = True,
) -> dict[str, Any]:
    """Validate the prepared CFetch SQLite interface and optionally decode one row."""
    columns = get_table_columns(db_path, table)
    missing = [column for column in BASE_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"{db_path}: missing required columns {missing}")

    with _connect(db_path) as conn:
        row_count = int(conn.execute(f"SELECT COUNT(*) FROM {_q(table)}").fetchone()[0])
        sample = None
        if decode_sample and row_count:
            sample = conn.execute(
                f"SELECT numpyarr, dimensions FROM {_q(table)} ORDER BY rowid LIMIT 1"
            ).fetchone()

    sample_shape = None
    if sample is not None:
        decoded = decode_numpyarr_blob(sample["numpyarr"], dimensions=_safe_int(sample["dimensions"]))
        sample_shape = [int(value) for value in decoded.shape]

    return {
        "database": str(Path(db_path)),
        "table": table,
        "row_count": row_count,
        "columns": columns,
        "sample_shape": sample_shape,
    }


def load_db_paths(db: str | Path | Sequence[str | Path] | None = None, jobs: str | Path | None = None) -> list[str]:
    out: list[str] = []
    if db is not None:
        if isinstance(db, (str, Path)):
            out.append(str(db))
        else:
            out.extend(str(x) for x in db if x)
    if jobs is not None:
        with open(jobs, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "db" not in obj:
                    raise ValueError(f"{jobs!s}:{line_no} missing 'db'")
                out.append(str(obj["db"]))
    deduped, seen = [], set()
    for x in out:
        if x not in seen:
            deduped.append(x)
            seen.add(x)
    if not deduped:
        raise ValueError("Provide at least one --db or --jobs file")
    return deduped


def parse_coordinates(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    s = str(value).strip()
    if not s:
        return None
    if s[0] in "[{":
        try:
            return json.loads(s)
        except Exception:
            pass
    out: list[Any] = []
    for i, part in enumerate(s.split(",")):
        part = part.strip()
        if i in (1, 2, 4, 5):
            try:
                out.append(int(part))
                continue
            except Exception:
                pass
        out.append(part)
    return out


def _json_or_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s or s[0] not in "[{":
        return value
    try:
        return json.loads(s)
    except Exception:
        return value


def _safe_int(value: Any) -> Optional[int]:
    try:
        return None if value is None else int(value)
    except Exception:
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def decode_numpyarr_blob(blob: bytes | memoryview | bytearray, dimensions: Optional[int] = None) -> np.ndarray:
    """Decode strainer.py `numpyarr` as raw little-endian FP32 square matrix."""
    if blob is None:
        raise ValueError("Missing numpyarr blob")
    if isinstance(blob, memoryview):
        blob = blob.tobytes()
    if not isinstance(blob, (bytes, bytearray)):
        raise TypeError(f"Expected bytes-like numpyarr, got {type(blob)!r}")
    if len(blob) % 4 != 0:
        raise ValueError(f"numpyarr byte length {len(blob)} is not divisible by 4")
    flat = np.frombuffer(blob, dtype="<f4")
    side = int(round(math.sqrt(int(flat.size))))
    if side * side != int(flat.size):
        candidates = [] if dimensions is None else [int(dimensions), int(dimensions) + 1, int(dimensions) - 1]
        for c in candidates:
            if c > 0 and c * c == int(flat.size):
                side = c
                break
        else:
            raise ValueError(f"numpyarr has {flat.size} float32 values; not a square image")
    arr = flat.reshape(side, side).astype(np.float32, copy=True)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_matrix_01(mat: np.ndarray, viewing_vmax: Optional[float], *, true_max: Optional[float] = None, clamp_01: bool = True) -> np.ndarray:
    x = np.asarray(mat, dtype=np.float32)
    denom = viewing_vmax if viewing_vmax and viewing_vmax > 0 else true_max
    if denom is None or denom <= 0 or not math.isfinite(float(denom)):
        denom = float(np.max(x)) if x.size else 1.0
    if denom <= 0 or not math.isfinite(float(denom)):
        denom = 1.0
    y = x / np.float32(denom)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if clamp_01:
        y = np.clip(y, 0.0, 1.0)
    return np.ascontiguousarray(y.astype(np.float32, copy=False))


normalize_with_viewing_vmax = normalize_matrix_01


def matrix01_to_tchw(matrix01: np.ndarray, *, target_side: Optional[int] = 128) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(matrix01)).float().unsqueeze(0).unsqueeze(0)
    if target_side is not None:
        side = int(target_side)
        if side <= 0:
            raise ValueError("target_side must be positive or None")
        if x.shape[-2:] != (side, side):
            x = F.interpolate(x, size=(side, side), mode="bilinear", align_corners=False)
    return x.contiguous()



class StrainerSQLiteDataset(Dataset):
    """Dataset for the `imag_with_seqs` table written by strainer.py."""
    def __init__(self, cfg: StrainerDataConfig | str | Path, **kwargs: Any):
        if isinstance(cfg, (str, Path)):
            cfg = StrainerDataConfig(db_paths=(str(cfg),), **kwargs)
        if len(cfg.db_paths) != 1:
            raise ValueError("StrainerSQLiteDataset takes one DB; use build_strainer_dataloader for many")
        self.cfg = cfg
        self.db_path = str(cfg.db_paths[0])
        self.table = cfg.table
        self._conn: Optional[sqlite3.Connection] = None
        self.schema_columns = get_table_columns(self.db_path, self.table)
        missing = [c for c in BASE_COLUMNS if c not in self.schema_columns]
        if missing:
            raise ValueError(f"{self.db_path}: missing columns {missing}")
        if cfg.read_verbose:
            selected_columns = list(self.schema_columns)
        else:
            selected_columns = []
            for c in (*BASE_COLUMNS, *tuple(cfg.extra_columns)):
                if c in self.schema_columns and c not in selected_columns:
                    selected_columns.append(c)
        self.read_columns = ["rowid", *selected_columns]
        self.rowids = self._load_rowids()

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_conn"] = None
        return d

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = _connect(self.db_path)
        return self._conn

    def _load_rowids(self) -> list[int]:
        where = f" WHERE ({self.cfg.where})" if self.cfg.where else ""
        q = f"SELECT rowid FROM {_q(self.table)}{where} ORDER BY rowid"
        params: list[Any] = []
        if self.cfg.limit is not None:
            q += " LIMIT ? OFFSET ?"
            params += [int(self.cfg.limit), int(self.cfg.offset)]
        elif self.cfg.offset:
            q += " LIMIT -1 OFFSET ?"
            params.append(int(self.cfg.offset))
        with _connect(self.db_path) as conn:
            return [int(r[0]) for r in conn.execute(q, params)]

    def __len__(self) -> int:
        return len(self.rowids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rowid = self.rowids[int(idx)]
        cols = ", ".join("rowid" if c == "rowid" else _q(c) for c in self.read_columns)
        row = self.conn.execute(f"SELECT {cols} FROM {_q(self.table)} WHERE rowid=?", (rowid,)).fetchone()
        rd = {k: row[k] for k in row.keys()}
        dim = _safe_int(rd.get("dimensions"))
        vmax = _safe_float(rd.get("viewing_vmax"))
        tmax = _safe_float(rd.get("true_max"))
        raw = decode_numpyarr_blob(rd["numpyarr"], dimensions=dim)
        matrix01 = normalize_matrix_01(raw, vmax, true_max=tmax, clamp_01=self.cfg.clamp_01) if self.cfg.normalize_by_viewing_vmax else raw.astype(np.float32, copy=False)
        if not self.cfg.normalize_by_viewing_vmax and self.cfg.clamp_01:
            matrix01 = np.clip(matrix01, 0.0, 1.0)
        obs = matrix01_to_tchw(matrix01, target_side=self.cfg.target_side)
        meta = {
            "rowid": int(rowid),
            "source_db": self.db_path,
            "db_path": self.db_path,
            "key_id": _safe_int(rd.get("key_id")),
            "dataset": rd.get("dataset"),
            "coordinates": parse_coordinates(rd.get("coordinates")),
            "coordinates_raw": rd.get("coordinates"),
            "dimensions": dim,
            "inferred_side": int(raw.shape[0]),
            "target_side": int(obs.shape[-1]),
            "resolution": _safe_int(rd.get("resolution")),
            "toolsource": rd.get("toolsource"),
            "viewing_vmax": vmax,
            "true_max": tmax,
        }
        for extra_name in tuple(getattr(self.cfg, "extra_columns", ())):
            if extra_name in rd and extra_name not in meta:
                meta[extra_name] = _json_or_text(rd[extra_name]) if extra_name == "meta" else rd[extra_name]
        if "meta" in rd:
            parsed_meta = _json_or_text(rd["meta"])
            if isinstance(parsed_meta, dict) and "feature_source" in parsed_meta:
                meta["feature_source"] = parsed_meta["feature_source"]
        if self.cfg.read_verbose:
            verbose = {}
            for k, v in rd.items():
                if k == "coordinates":
                    verbose[k] = parse_coordinates(v)
                elif k == "meta":
                    verbose[k] = _json_or_text(v)
                else:
                    verbose[k] = v
            meta["row"] = verbose
            meta["columns"] = list(self.schema_columns)
        return {"obs": obs, "meta": meta}


def strainer_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    shapes = {tuple(x["obs"].shape) for x in batch}
    if len(shapes) != 1:
        raise ValueError(f"Mixed obs shapes {sorted(shapes)}; set target_side or batch_size=1")
    metas = [x["meta"] for x in batch]
    return {"obs": torch.stack([x["obs"] for x in batch], 0).contiguous(), "meta": metas, "meta_dict": {"kept": len(metas), "items": metas}}


def build_strainer_dataloader(cfg: StrainerDataConfig) -> DataLoader:
    datasets = [
        StrainerSQLiteDataset(
            StrainerDataConfig(
                db_paths=(p,), table=cfg.table, target_side=cfg.target_side,
                batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory, normalize_by_viewing_vmax=cfg.normalize_by_viewing_vmax,
                clamp_01=cfg.clamp_01, read_verbose=cfg.read_verbose,
                extra_columns=cfg.extra_columns,
                where=cfg.where,
                limit=cfg.limit, offset=cfg.offset,
            )
        )
        for p in cfg.db_paths
    ]
    ds: Dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    pin = torch.cuda.is_available() if cfg.pin_memory is None else bool(cfg.pin_memory)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers, pin_memory=pin, collate_fn=strainer_collate)


def make_strainer_dataloader(db_path: str | Path | Sequence[str | Path], **kwargs: Any) -> DataLoader:
    paths = load_db_paths(db_path)
    return build_strainer_dataloader(StrainerDataConfig(db_paths=tuple(paths), **kwargs))


def add_strainer_dataloader_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--db", action="append", default=None)
    parser.add_argument("--jobs", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-side", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--where", type=str, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--read-verbose", action="store_true")
    parser.add_argument("--no-clamp-01", action="store_true")
    parser.add_argument("--no-vmax-normalize", action="store_true")
    return parser


add_strainer_data_args = add_strainer_dataloader_args


def config_from_args(args: argparse.Namespace) -> StrainerDataConfig:
    side = int(getattr(args, "target_side", 128))
    return StrainerDataConfig(
        db_paths=tuple(load_db_paths(getattr(args, "db", None), getattr(args, "jobs", None))),
        target_side=None if side <= 0 else side,
        batch_size=int(getattr(args, "batch_size", 1)),
        shuffle=bool(getattr(args, "shuffle", False)),
        num_workers=int(getattr(args, "num_workers", 0)),
        clamp_01=not bool(getattr(args, "no_clamp_01", False)),
        normalize_by_viewing_vmax=not bool(getattr(args, "no_vmax_normalize", False)),
        read_verbose=bool(getattr(args, "read_verbose", False)),
        extra_columns=(),
        where=getattr(args, "where", None),
        limit=getattr(args, "limit", None),
        offset=int(getattr(args, "offset", 0)),
    )


def make_strainer_dataloader_from_args(args: argparse.Namespace) -> DataLoader:
    return build_strainer_dataloader(config_from_args(args))


__all__ = [
    "StrainerDataConfig", "StrainerSQLiteDataset", "add_strainer_data_args",
    "add_strainer_dataloader_args", "build_strainer_dataloader", "config_from_args",
    "decode_numpyarr_blob", "get_table_columns", "load_db_paths", "make_strainer_dataloader",
    "make_strainer_dataloader_from_args", "matrix01_to_tchw", "normalize_matrix_01",
    "normalize_with_viewing_vmax", "parse_coordinates", "strainer_collate", "validate_database",
]
