# Coordinate proposals for Cfetch patch extraction

Generate cis, upper-triangle candidate anchor pairs from chromosome regions. Outputs are coordinate proposals.

## Quick start

Dependencies: Python 3.10+, NumPy and PyYAML. No package installation of this tool is needed.

```bash
python sample_coordinates.py --regions test/regions.yaml --out candidates_5k --dry-run
python sample_coordinates.py --regions test/regions.yaml --out candidates_5k
```

Default: 5 kb, statistics-guided sampling, one unique offset per eligible first-anchor position, with first anchors spaced one resolution bin apart. The one-sample density is a configurable operational default, independent of the distance prior. Set `--samples-per-start 4` for up to four distinct offsets per position, or `--stride 20kb` for more widely spaced first anchors. Sampling exhausts available offsets if fewer than the requested number have representable positive probability.

```bash
# All supported resolutions; each gets its own files and resolution-specific prior.
python sample_coordinates.py --regions test/regions.yaml --out candidates_all \
  --resolutions 2kb 5kb 10kb

# A sharp soft tail, plus explicit hard distance bounds.
python sample_coordinates.py --regions test/regions.yaml --out candidates_tail \
  --minrange 50kb --maxrange 5mb \
  --decay gumbel --decay-start 1mb --decay-scale 200kb

# Alternative gradual tail weighting.
python sample_coordinates.py --regions test/regions.yaml --out candidates_powerlaw \
  --minrange 50kb --maxrange 5mb \
  --decay powerlaw --decay-start 500kb --beta 2

# Every eligible pair in the requested diagonal band; no stochastic sampling.
python sample_coordinates.py --regions test/regions.yaml --out candidates_tiled \
  --tile force_full --minrange 50kb --maxrange 2mb --dry-run

# Also restrict where the first anchor lies along the diagonal.
python sample_coordinates.py --regions test/regions.yaml --out candidates_local \
  --start-min 12mb --start-max 15mb --minrange 100kb --maxrange 1mb
```

Remove `--dry-run` to execute full tiling. Without explicit range bounds, full tiling includes all distinct upper-triangle bin pairs in the regions, subject to patch boundaries and stride. This can be a very large output; the dry run prints the exact count without writing files. No arbitrary maximum candidate cap is imposed. `--tile samplings` is accepted as an alias for `sampling`.

## Request a total instead of a per-start count

Use EITHER `--samples-per-start` OR `--num-samples`; specifying both is an error. Omitting both preserves the default of one sample per start. Full tiling does not accept either sampling budget.

```bash
python sample_coordinates.py --regions test/regions.yaml --out candidates_80000 \
  --num-samples 80000 \
  --minrange 50kb --maxrange 5mb \
  --decay gumbel --decay-start 1mb --decay-scale 200kb
```

`--num-samples 80000` requests **80,000 unique pairs per resolution**, shared across all named chromosomes, not 80,000 per chromosome. If all three resolutions are requested, the total target is 240,000 pairs. Chromosome counts are stochastic, not fixed quotas.

For each attempted draw, the sampler chooses uniformly among all eligible first-anchor positions across the requested regions, then draws a permitted distance using that position's conditional statistical/decay weights. Duplicate pairs are rejected and retried. This implements weighted sampling without replacement by rejecting previously selected pairs. As the sample fills, the distribution of remaining pairs changes; large budgets can pull in increasingly weakly weighted distances. It does not require every first-anchor position to be represented.

The target is hopeful: if the unique pool is exhausted or the attempt cap is reached, output is finalized with the available unique pairs and a recorded shortfall. The default cap is max(10,000, 50 times the smaller of the requested count and positive-weight capacity), separately per resolution. Override with `--max-draws 10000000`. A sharply decaying prior can cause repeated draws long before all technically positive-weight pairs are seen. The sampler does not flatten weights or ignore distance constraints to meet the target. Floating-point cumulative probabilities can make extraordinarily weak tails effectively unreachable.

`summary.json` records requested count, generated unique count, shortfall, capacity, attempted draws, duplicate draws rejected and stop reason under `sampling_budgets`. Dry-run reports the requested count and available-capacity upper bound; it cannot predict the realized count at the attempt cap.

## How many candidates?

A first-anchor position is one bin start x1 along the chromosome. An offset is the sampled separation d; the second anchor starts at y1 = x1 + d. At 5 kb, x1 advances by 5,000 bp by default. `--samples-per-start 1` chooses one second-anchor position for each eligible x1; `--samples-per-start 4` chooses up to four distinct second-anchor positions for that same x1. It does not mean one candidate per chromosome, and it does not tile all possible offsets.

In per-start mode, the count is the sum over first-anchor positions of min(samples-per-start, available positive-weight offsets). A rough estimate for long regions is covered length / stride times samples-per-start. Boundary losses, minrange and patch margins reduce that estimate. No extra multiplicative decay acceptance step is applied: decay changes which offsets are chosen, not the requested count at each start.

For one aligned 10 mb region at 5 kb, with minrange 50kb, maxrange 2mb, patch-bins 1 and default stride, the exact counts are:

| Samples per start | Candidate pairs |
| --- | ---: |
| 1 | 1,990 |
| 4 | 7,954 |
| 10 | 19,855 |
| force_full | 701,845 |

The first-anchor position nearest the end has only one legal offset, the preceding one has two, and so on. This accounts for counts slightly below simple multiplication. The default prior-dependent minrange can produce different totals. Use `--dry-run` with your actual YAML and options for planned counts before writing. Each resolution has a separate candidate set and count.

## YAML

```yaml
assembly: hg38
chromosomes: [chr1, chr2]
coordinates:
  - [10000000, 20000000]
  - [20000000, 30000000]
```

Coordinates are zero-based, half-open chromosome intervals, not BEDPE pairs. Both anchors must lie fully within the corresponding interval. If coordinates are omitted, every named chromosome uses its full length. If supplied, the list must have exactly one item per chromosome. Individual `null` entries select the whole chromosome. An empty list is not equivalent to omitting the list. The tool rejects duplicates, invalid ranges and chromosome aliases listed twice. Numeric names are accepted and preserved as strings in the output; `1` and `chr1` share size/prior lookup.

```yaml
assembly: hg38
chromosomes: [chr1, chr2, chrX]
# No coordinates: use full chromosomes.
```

Chromosome lengths for hg38/GRCh38 are bundled. For another assembly, provide a two-column chromosome-size file, or a `chromosome_sizes` mapping:

```yaml
assembly: custom_assembly
chromosomes: [contig1, contig2]
chromosome_sizes:
  contig1: 5000000
  contig2: 8000000
coordinates:
  - null
  - [1000000, 6000000]
```

`chrom_sizes: path/to/assembly.chrom.sizes` or `--chrom-sizes path` also works. Relative size paths are resolved relative to the YAML file. Assembly declares coordinate context; this tool performs no liftover. The bundled human-distance priors are not automatically recalibrated for other assemblies. Chromosomes without their own defaults use pooled moments, explicitly recorded in the plan. A size file is still required for unknown chromosomes.

## Hard bounds and patch geometry

- `--minrange` and `--maxrange` jointly retain **minrange <= y1 - x1 <= maxrange**. Outside that band is excluded in every mode. These are anchor-start separations, equivalent to center separation for equal-sized bins, not Euclidean perpendicular distance from the diagonal.
- `--start-min` and `--start-max` jointly constrain the first anchor's genomic start. They do not restrict the second anchor beyond its YAML region and distance bounds.
- Examples use `kb` and `mb` consistently. Plain bp values and case-insensitive suffixes are also accepted: `50000`, `50K`, `50k`, `50kb`, `50KB` are equivalent; `5000000`, `5M`, `5m`, `5mb`, `5MB` are equivalent. Decimal units such as `1.5mb` also work.
- Bin starts are aligned to the global resolution grid; intervals are rounded inward to retain complete bins. The main diagonal itself is excluded even if minrange is zero.
- In sampling mode, omitted minimum/maximum bounds default to the chromosome's bundled minimum and maximum start separations. Explicit CLI bounds replace the corresponding statistical bounds, allowing a search beyond the default range. Chromosome/region edges always apply.
- Full tiling ignores statistical range defaults: omitted bounds cover all possible distinct anchor pairs in the region. Explicit bounds still apply.
- `--stride` controls the first-anchor grid only. Full tiling with stride omitted enumerates every eligible bin pair.

`--patch-bins N` additionally defines an N by N extraction window around each proposed anchor pair. It defaults to **1 (the anchor bin itself)** because the downstream model's extraction width must be specified by the user. Set this to the actual patch width expected by your extractor, e.g. `--patch-bins 21` for a 21 by 21 window. The candidate bin is at array index `N//2`, including for even N. Complete windows must fit within chromosome bin boundaries; edge candidates are omitted, not zero-padded. `--window-within-region` additionally requires the whole window to fit in the YAML interval. These constraints also apply to full tiling.

Distance limits apply to the central anchor pair, not every pixel of its window. A window can cross the main diagonal if the center is closer than its span; increase minrange if that is unwanted.

## Statistical prior and decay formulas

Numerical per-chromosome defaults for 2, 5 and 10 kb and their version are stored in the bundled `src/sampling_defaults.json`. The script loads this file automatically relative to its own location, regardless of the working directory. Keep the `src/` directory alongside `sample_coordinates.py`; no extra CLI argument or original analysis output is required. The defaults define a parametric sampling prior, not an exact empirical distribution or a fitted chromatin contact-decay law. All stored distances use anchor-start separation (`y1 - x1`), matching the CLI bounds.

For each chromosome and resolution, the baseline discrete weight at eligible distance d is:

```
log p0(d) = -0.5 * ((d - prior_mean) / prior_std)^2 + constant
```

Weights are evaluated at legal bin separations and normalized over the available offsets. This does not guarantee that generated samples have the untruncated prior mean or standard deviation. Candidate first-anchor positions are tiled uniformly in per-start mode; chromosome allocation is not driven by the prior's pooling weights.

Optional tail weights multiply this baseline:

```
--decay none:
    w(d) = 1

--decay powerlaw:
    w(d) = max(1, d / d0)^(-beta)

--decay gumbel:
    w(d) = exp(1 - exp(max(0, (d - d0) / s)))
```

Default d0 is prior mean + standard deviation; default s is the standard deviation. These defaults are transparent heuristics, not fitted tail parameters. `--decay-start` and `--decay-scale` override them. The power-law exponent defaults to 2 and is a configurable sampling choice.

The Gumbel option uses a plateau followed by a scaled **minimum-Gumbel survival** tail. It is not a maximum-Gumbel distance draw or Gumbel-top-k sampler. It equals 1 through d0, then approximately 0.179, 0.00168 and 5.15e-9 at d0+s, d0+2s and d0+3s. This matches the requested strong suppression of extremes. Power-law decay alone is much more gradual. Taking log(log(d)) by itself does not specify a sampling distribution.

Weights are normalized separately over the offsets allowed at each first-anchor position. They are relative weights, not absolute inclusion probabilities or FDR values. Requested sample count, truncation and without-replacement sampling affect realized frequencies. In particular, setting minrange entirely beyond the knee conditions sampling on that tail: it will still draw from that permitted region. Extremely tiny weights may become numerically zero in floating-point arithmetic, and do not guarantee an eventual draw.

Gumbel formula reference: NIST Engineering Statistics Handbook, Extreme Value Type I Distribution, minimum-case CDF `1-exp(-exp(z))`: https://www.itl.nist.gov/div898/handbook/eda/section3/eda366g.htm . The plateau and scaling used here are an explicit sampling design choice.

## Outputs and reproducibility

For each resolution:

- `candidates_5k.bedpe`: six-column, headerless anchor coordinates.
- `candidates_5k.tsv`: coordinates, resolution, stable candidate ID and start separation.
- `patch_windows_5k.tsv`: genomic row/column extraction windows keyed by candidate ID.
- `plan.json`: arguments, input hash, sampling-defaults version/hash, effective bounds, counts and actual numerical priors.
- `summary.json`: completion marker plus realized counts and distance summaries.

Use the generated candidate coordinates with [rustrainer](https://github.com/seanpatrickmoran/rustrainer) to extract contact-map patches for downstream analysis. Configure the contact-map source and coordinate input according to the rustrainer documentation; the optional patch-window output specifies the requested extraction footprint.

In per-start mode, RNG streams are derived independently from seed, resolution and chromosome name, so adding a different chromosome or resolution does not perturb an existing stream. Total-budget mode uses a separate stream per resolution; changing the chromosome list changes the shared sampling pool and therefore the draw sequence. Same input and seed reproduce the output under the same numerical-library implementation. Pairs are unique within each resolution. After writing, an independent streaming read-back checks coordinate ordering and duplicates, row counts, BEDPE/TSV coordinate agreement and patch-window ID agreement. The audit is stored under `dedupe_audits` in `summary.json`. Files are finalized only after it passes; unexpected duplicate or unsorted rows stop completion. Streaming checks rely on sorted coordinates in contiguous chromosome blocks and verify that ordering explicitly. Duplicate draws in total-budget mode are removed before writing and counted separately. Outputs are streamed rather than materializing every full-tiling pair. The sampler never excludes existing Mustache coordinates automatically: unlike the negative generator, its purpose is to supply the entire search space.

Use a fresh output directory. Interrupted files carry `.partial`; there is no automatic resume or silent overwrite. Sampling is sparse, so failure to retrieve a feature cannot exclude its presence at unproposed coordinates.

## Package layout

- `sample_coordinates.py`: top-level CLI.
- `src/sampling_defaults.json`: numerical defaults and version.
- `src/hg38.chrom.sizes`: bundled chromosome lengths.
- `test/test_sampler.py`: unit and integration tests.
- `test/test_chr6.sh`: four chr6 example runs.
- `test/chr6_region.yaml`: hg38 chr6:0–7,100,000.
- `test/regions.yaml` and `test/whole_chromosomes.yaml`: additional YAML examples.

From the package directory:

```bash
bash test/test_chr6.sh --dry-run
bash test/test_chr6.sh
```

The shell script resolves the CLI and YAML relative to its own location and writes outputs under `test/outputs/` by default. Override with `OUTPUT_ROOT=/path/to/results bash test/test_chr6.sh`. `PYTHON=/path/to/python` selects the interpreter. Existing nonempty outputs are not overwritten.

## Tests

```bash
python test/test_sampler.py
```

Tests cover exact enumeration at 2/5/10 kb, hard bounds, first-anchor limits, even-sized window boundaries, reproducibility, duplicate prevention, empty regions, strict YAML list validation, dry-run behavior, chromosome sizes, distance convention, tail suppression, total-budget reproducibility, mutually exclusive budgets, pool-exhaustion shortfalls, attempt limits and detection of injected duplicate output rows.
