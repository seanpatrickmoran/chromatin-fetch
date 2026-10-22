import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sample_coordinates import audit_outputs, bp, load_regions, prior_weights


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.yml = self.root / "regions.yaml"
        self.yml.write_text(
            "assembly: test\n"
            "chromosomes: [chr1, chr2]\n"
            "chromosome_sizes: {chr1: 100000, chr2: 100000}\n"
            "coordinates: [[10000, 90000], [0, 100000]]\n"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, extra=(), name="out", ok=True):
        out = self.root / name
        command = [
            sys.executable,
            str(ROOT / "sample_coordinates.py"),
            "--regions", str(self.yml),
            "--out", str(out),
            *extra,
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)

        return out, result

    def rows(self, out, res=5):
        with (out / f"candidates_{res}k.tsv").open() as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    def test_total_budget_and_reproducibility(self):
        args = [
            "--num-samples", "30",
            "--minrange", "5kb",
            "--maxrange", "60kb",
            "--decay", "gumbel",
            "--resolutions", "2kb", "5kb", "10kb",
        ]
        out, _ = self.run_tool(args)
        other, _ = self.run_tool(args, "other")
        report = json.loads((out / "summary.json").read_text())

        for res in [2, 5, 10]:
            self.assertEqual(len(self.rows(out, res)), 30)
            self.assertEqual(
                (out / f"candidates_{res}k.bedpe").read_bytes(),
                (other / f"candidates_{res}k.bedpe").read_bytes(),
            )
            self.assertEqual(
                report["dedupe_audits"][str(res * 1000)]["duplicate_pairs"],
                0,
            )

    def test_shortfall_and_mutual_exclusion(self):
        self.yml.write_text(
            "assembly: test\n"
            "chromosomes: [chr1]\n"
            "chromosome_sizes: {chr1: 20000}\n"
        )
        out, _ = self.run_tool([
            "--num-samples", "100",
            "--minrange", "5kb",
            "--maxrange", "5kb",
        ])

        self.assertEqual(len(self.rows(out)), 3)
        report = json.loads((out / "summary.json").read_text())
        self.assertEqual(report["sampling_budgets"]["5000"]["shortfall"], 97)

        self.run_tool(
            ["--num-samples", "10", "--samples-per-start", "1"],
            name="bad",
            ok=False,
        )
        self.run_tool(
            ["--num-samples", "10", "--tile", "force_full"],
            name="bad2",
            ok=False,
        )

    def test_draw_limit_and_postwrite_duplicate_detection(self):
        out, _ = self.run_tool([
            "--num-samples", "30",
            "--max-draws", "2",
        ])
        self.assertLessEqual(len(self.rows(out)), 2)

        for name in [
            "candidates_5k.bedpe",
            "candidates_5k.tsv",
            "patch_windows_5k.tsv",
        ]:
            path = out / name
            lines = path.read_text().splitlines()
            path.write_text("\n".join(lines + [lines[-1]]) + "\n")

        with self.assertRaisesRegex(ValueError, "dedupe"):
            audit_outputs(
                out / "candidates_5k.bedpe",
                out / "candidates_5k.tsv",
                out / "patch_windows_5k.tsv",
            )

    def test_unit_spellings(self):
        for spelling in [
            "50000", "50K", "50k", "50kb", "50KB", "50Kb",
            "50000bp", "50,000", "50_000",
        ]:
            self.assertEqual(bp(spelling), 50000)

        for spelling in ["5000000", "5M", "5m", "5mb", "5MB", "5Mb"]:
            self.assertEqual(bp(spelling), 5000000)

    def test_full_exact_and_multires(self):
        out, _ = self.run_tool([
            "--tile", "force_full",
            "--resolutions", "2k", "5k", "10k",
            "--minrange", "10kb",
            "--maxrange", "30kb",
        ])

        for res in [2, 5, 10]:
            r = res * 1000
            expected = {
                (chrom, x, y)
                for chrom, lo, hi in [
                    ("chr1", 10000, 90000),
                    ("chr2", 0, 100000),
                ]
                for x in range((lo + r - 1) // r * r, hi, r)
                for y in range(x + r, hi, r)
                if y + r <= hi and 10000 <= y - x <= 30000
            }
            observed = {
                (row["chrom1"], int(row["x1"]), int(row["y1"]))
                for row in self.rows(out, res)
            }
            self.assertEqual(observed, expected)

    def test_sampling_reproducible_unique(self):
        args = [
            "--minrange", "5k",
            "--maxrange", "60k",
            "--samples-per-start", "3",
            "--decay", "gumbel",
            "--decay-start", "20k",
            "--decay-scale", "10k",
        ]
        first, _ = self.run_tool(args)
        second, _ = self.run_tool(args, "out2")
        self.assertEqual(
            (first / "candidates_5k.bedpe").read_bytes(),
            (second / "candidates_5k.bedpe").read_bytes(),
        )

        rows = self.rows(first)
        self.assertEqual(len(rows), len({row["candidate_id"] for row in rows}))
        for row in rows:
            self.assertTrue(5000 <= int(row["distance_bp"]) <= 60000)

    def test_start_bounds_and_patch_edges(self):
        out, _ = self.run_tool([
            "--tile", "force_full",
            "--minrange", "5k",
            "--maxrange", "20k",
            "--start-min", "20k",
            "--start-max", "40k",
            "--patch-bins", "4",
            "--window-within-region",
        ])
        for row in self.rows(out):
            self.assertTrue(20000 <= int(row["x1"]) <= 40000)

        with (out / "patch_windows_5k.tsv").open() as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                lo, hi = (10000, 90000) if row["chrom"] == "chr1" else (0, 100000)
                self.assertTrue(
                    lo <= int(row["row_start"]) < int(row["row_end"]) <= hi
                )
                self.assertTrue(
                    lo <= int(row["col_start"]) < int(row["col_end"]) <= hi
                )
                self.assertEqual(int(row["row_end"]) - int(row["row_start"]), 20000)

    def test_yaml_and_empty_and_dry_run(self):
        out, _ = self.run_tool(["--dry-run"])
        self.assertFalse(out.exists())

        self.yml.write_text(
            "assembly: hg38\n"
            "chromosomes: [chr1, chr2]\n"
            "coordinates: [[0, 100000]]\n"
        )
        self.run_tool(ok=False)

        self.yml.write_text(
            "assembly: test\n"
            "chromosomes: [chr1]\n"
            "chromosome_sizes: {chr1: 10000}\n"
        )
        out, _ = self.run_tool([
            "--minrange", "100k",
            "--maxrange", "200k",
        ])
        self.assertEqual(len(self.rows(out)), 0)

    def test_whole_chromosome_and_invalid_range(self):
        self.yml.write_text('assembly: hg38\nchromosomes: ["1"]\n')
        _, regions = load_regions(self.yml)
        self.assertEqual(regions[0]["end"], 248956422)
        self.run_tool(["--minrange", "20k", "--maxrange", "10k"], ok=False)

    def test_tail_definition_and_span_shift(self):
        args = SimpleNamespace(
            decay_start=10000,
            decay_scale=10000,
            decay="none",
            beta=2,
        )
        offsets = np.arange(1, 10)
        stats = {"mean": 25000, "std": 15000}
        normal, params = prior_weights(offsets, 5000, stats, args, "outer_span")
        self.assertEqual(params["mean_start_separation_bp"], 20000)

        args.decay = "gumbel"
        gumbel, _ = prior_weights(offsets, 5000, stats, args, "outer_span")
        self.assertLess(gumbel[-1] / gumbel[0], normal[-1] / normal[0])
        self.assertEqual(bp("1.5mb"), 1500000)


if __name__ == "__main__":
    unittest.main()
