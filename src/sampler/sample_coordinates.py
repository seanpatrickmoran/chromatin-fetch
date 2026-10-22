"""Generate cis candidate BEDPE and extraction windows from YAML regions."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from itertools import zip_longest
import numpy as np
import yaml
HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "reference"
# Numerical defaults use anchor-start separations in base pairs.
with (SOURCE_DIR / "sampling_defaults.json").open() as defaults_file:
    defaults_payload = json.load(defaults_file)

SAMPLING_DEFAULTS_VERSION = defaults_payload["version"]
SAMPLING_DEFAULTS = defaults_payload["defaults"]


def bp(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not a coordinate')
    s = str(value).strip().lower().replace(',', '').replace('_', '')
    scale = 1
    for suffix, factor in [('mb', 1000000), ('kb', 1000), ('bp', 1), ('m', 1000000), ('k', 1000)]:
        if s.endswith(suffix):
            s, scale = (s[:-len(suffix)], factor)
            break
    x = float(s) * scale
    if not math.isfinite(x) or x < 0 or x != int(x):
        raise ValueError(f'Invalid nonnegative bp value: {value}')
    return int(x)


def canon(chrom):
    c = str(chrom)
    return 'chr' + (c[3:] if c.startswith('chr') else c)


def sizes_file(path):
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        c, n = line.split()[:2]
        c = canon(c)
        if c in out:
            raise ValueError(f'Duplicate chromosome in sizes: {c}')
        out[c] = bp(n)
    return out


def load_regions(path, sizes_override=None):
    doc = yaml.safe_load(path.read_text())
    if not isinstance(doc, dict) or not isinstance(doc.get('assembly'), str):
        raise ValueError('YAML requires assembly and chromosomes')
    names = doc.get('chromosomes')
    if not isinstance(names, list) or not names:
        raise ValueError('chromosomes must be a nonempty list')
    if any((not isinstance(c, (str, int)) or isinstance(c, bool) for c in names)):
        raise ValueError('Invalid chromosome name')
    names = [str(c) for c in names]
    if len(set(map(canon, names))) != len(names):
        raise ValueError('Chromosomes must be unique, including chr aliases')
    raw = doc.get('coordinates')
    if raw is None:
        raw = [None] * len(names)
    if not isinstance(raw, list) or len(raw) != len(names):
        raise ValueError(
            'coordinates must match the number of chromosomes exactly; omit it for whole chromosomes'
        )
    size_source = sizes_override or doc.get('chrom_sizes')
    if size_source:
        f = Path(size_source)
        if not f.is_absolute():
            f = path.parent / f
        sizes = sizes_file(f)
    elif isinstance(doc.get('chromosome_sizes'), dict):
        sizes = {canon(k): bp(v) for k, v in doc['chromosome_sizes'].items()}
    elif doc['assembly'].lower() in ('hg38', 'grch38'):
        sizes = sizes_file(SOURCE_DIR / 'hg38.chrom.sizes')
    else:
        raise ValueError(
            'For assemblies other than hg38/GRCh38, supply chrom_sizes or chromosome_sizes'
        )
    regions = []
    for c, bounds in zip(names, raw):
        if canon(c) not in sizes:
            raise ValueError(f'No chromosome size for {c}')
        size = sizes[canon(c)]
        if bounds is None:
            lo, hi = (0, size)
        elif isinstance(bounds, list) and len(bounds) == 2:
            lo, hi = map(bp, bounds)
        else:
            raise ValueError(f'{c}: expected null or [start, end]')
        if not 0 <= lo < hi <= size:
            raise ValueError(f'{c}: invalid interval [{lo}, {hi}) for size {size}')
        regions.append(dict(chrom=c, start=lo, end=hi, chromosome_size=size))
    return (doc, regions)


def pooled_stats(stats):
    n = sum((v['count'] for v in stats.values()))
    mu = sum((v['count'] * v['mean'] for v in stats.values())) / n
    var = sum((v['count'] * (v['std'] ** 2 + (v['mean'] - mu) ** 2) for v in stats.values())) / n
    return dict(
        count=n,
        mean=mu,
        std=math.sqrt(var),
        min=min((v['min'] for v in stats.values())),
        max=max((v['max'] for v in stats.values()))
    )


def prior_weights(offsets, resolution, stats, args, definition):
    shift = resolution if definition == 'outer_span' else 0
    mean = stats['mean'] - shift
    std = stats['std']
    if std <= 0:
        raise ValueError('Prior standard deviation must be positive')
    d = offsets.astype(float) * resolution
    logw = -0.5 * ((d - mean) / std) ** 2
    knee = args.decay_start if args.decay_start is not None else max(resolution, mean + std)
    scale = args.decay_scale if args.decay_scale is not None else std
    if args.decay == 'powerlaw':
        logw -= args.beta * np.log(np.maximum(1.0, d / knee))
    elif args.decay == 'gumbel':
        # Plateau followed by a scaled minimum-Gumbel survival tail.
        z = np.maximum(0.0, (d - knee) / scale)
        logw -= np.expm1(np.minimum(z, 700.0))
    logw -= logw.max()
    w = np.exp(logw)
    return (
        w,
        dict(mean_start_separation_bp=mean, std_bp=std, decay_start_bp=knee, decay_scale_bp=scale)
    )


def plan_region(region, resolution, stats, args, definition):
    r = dict(region)
    shift = resolution if definition == 'outer_span' else 0
    lower = args.minrange if args.minrange is not None else resolution if args.tile == 'force_full' else max(
        resolution,
        stats['min'] - shift
    )
    upper = args.maxrange if args.maxrange is not None else region['end'] - region[
        'start'
    ] if args.tile == 'force_full' else max(0, stats['max'] - shift)
    if lower > upper:
        raise ValueError(f"{r['chrom']} at {resolution}: effective minrange > maxrange")
    dmin = max(1, math.ceil(lower / resolution))
    dmax = math.floor(upper / resolution)
    # Place the candidate pixel at index patch_bins // 2, including even widths.
    left = args.patch_bins // 2
    right = args.patch_bins - left - 1
    lo = math.ceil(r['start'] / resolution)
    hi = r['end'] // resolution - 1
    lo = max(lo, left)
    hi = min(hi, r['chromosome_size'] // resolution - right - 1)
    if args.window_within_region:
        lo = max(lo, math.ceil(r['start'] / resolution) + left)
        hi = min(hi, r['end'] // resolution - right - 1)
    startlo = max(lo, math.ceil(args.start_min / resolution)) if args.start_min is not None else lo
    starthi = min(
        hi,
        math.floor(args.start_max / resolution)
    ) if args.start_max is not None else hi
    stride = args.stride or resolution
    if stride % resolution:
        raise ValueError('--stride must be a multiple of every requested resolution')
    starts = np.arange(startlo, starthi - dmin + 1, stride // resolution, dtype=np.int64)
    # start_max bounds the first anchor only; the second may extend farther.
    if args.start_max is not None:
        starts = np.arange(
            startlo,
            min(starthi, hi - dmin) + 1,
            stride // resolution,
            dtype=np.int64
        )
    maxoffset = min(dmax, hi - startlo)
    offsets = np.arange(dmin, maxoffset + 1, dtype=np.int64)
    if not len(offsets):
        starts = np.array([], dtype=np.int64)
    counts = np.maximum(0, np.minimum(dmax, hi - starts) - dmin + 1)
    full_count = int(counts.sum())
    weights = None
    params = {}
    if args.tile == 'sampling' and len(offsets):
        weights, params = prior_weights(offsets, resolution, stats, args, definition)
    numeric_counts = counts
    if weights is not None:
        available = np.cumsum(weights > 0)
        numeric_counts = available[np.minimum(counts, len(available)) - 1] if len(
            counts
        ) else counts
    estimated = full_count if args.tile == 'force_full' else None if args.num_samples is not None else int(
        np.minimum(numeric_counts, args.samples_per_start).sum()
    )
    r.update(
        resolution=resolution,
        lower_bound_bp=lower,
        upper_bound_bp=upper,
        dmin_bins=dmin,
        dmax_bins=dmax,
        start_positions=len(starts),
        eligible_pairs=full_count,
        positive_weight_pairs=int(numeric_counts.sum()),
        planned_candidates=estimated,
        prior=stats,
        prior_parameters=params
    )
    return (r, starts, offsets, weights, hi)


def generate(plan, resolution, args, rng):
    r, starts, offsets, weights, hi = plan
    cdf = np.cumsum(weights) if weights is not None else None
    for x in starts:
        count = max(0, min(r['dmax_bins'], hi - int(x)) - r['dmin_bins'] + 1)
        if not count:
            continue
        if args.tile == 'force_full':
            chosen = range(r['dmin_bins'], r['dmin_bins'] + count)
        elif args.samples_per_start == 1:
            total = cdf[count - 1]
            if total <= 0:
                continue
            index = min(int(np.searchsorted(cdf, rng.random() * total, side='right')), count - 1)
            chosen = [int(offsets[index])]
        else:
            w = weights[:count]
            valid = int(np.count_nonzero(w))
            k = min(valid, args.samples_per_start)
            if not k:
                continue
            indexes = rng.choice(count, size=k, replace=False, p=w / w.sum())
            chosen = sorted(offsets[indexes].tolist())
        for d in chosen:
            yield (r['chrom'], int(x) * resolution, (int(x) + int(d)) * resolution)


def sample_total(plans, resolution, args):
    """Uniform eligible first-anchor draws, prior-weighted offsets, reject repeats."""
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, resolution, 982451653]))
    pools = []
    widths = []
    seen = set()
    result = {pl[0]['chrom']: [] for pl in plans}
    for index, pl in enumerate(plans):
        r, starts, offsets, weights, hi = pl
        if weights is None or not len(starts):
            continue
        cdf = np.cumsum(weights)
        counts = np.minimum(r['dmax_bins'], hi - starts) - r['dmin_bins'] + 1
        active = np.flatnonzero(cdf[counts - 1] > 0)
        if len(active):
            pools.append((index, pl, cdf, counts, active))
            widths.append(len(active))
    cumulative = np.cumsum(widths)
    capacity = sum((pl[0]['positive_weight_pairs'] for pl in plans))
    target = min(args.num_samples, capacity)
    limit = args.max_draws or max(10000, 50 * target)
    draws = 0
    repeats = 0
    while len(seen) < target and draws < limit and len(pools):
        ticket = int(rng.integers(int(cumulative[-1])))
        which = int(np.searchsorted(cumulative, ticket, side='right'))
        previous = int(cumulative[which - 1]) if which else 0
        index, pl, cdf, counts, active = pools[which]
        r, starts, offsets, weights, hi = pl
        j = int(active[ticket - previous])
        count = int(counts[j])
        total = cdf[count - 1]
        k = min(int(np.searchsorted(cdf, rng.random() * total, side='right')), count - 1)
        x = int(starts[j]) * resolution
        y = x + int(offsets[k]) * resolution
        key = (index, x, y)
        draws += 1
        if key in seen:
            repeats += 1
            continue
        seen.add(key)
        result[r['chrom']].append((r['chrom'], x, y))
    for values in result.values():
        values.sort(key=lambda v: (v[1], v[2]))
    reason = 'target_reached' if len(
        seen
    ) == args.num_samples else 'available_support_exhausted' if len(
        seen
    ) == capacity else 'draw_limit_reached'
    audit = dict(
        requested=args.num_samples,
        generated_unique=len(seen),
        shortfall=args.num_samples - len(seen),
        positive_weight_capacity=capacity,
        draws=draws,
        duplicate_draws_rejected=repeats,
        max_draws=limit,
        status=reason
    )
    return (result, audit)


def audit_outputs(bed, tab, windows):
    """Independent streaming read-back; sorted per-chromosome coordinates make exact dedupe bounded-memory."""
    n = 0
    previous = None
    closed = set()
    current = None
    with bed.open() as bf, tab.open() as tf, windows.open() as wf:
        tr = csv.DictReader(tf, delimiter='\t')
        wr = csv.DictReader(wf, delimiter='\t')
        for line, t, w in zip_longest(bf, tr, wr):
            if line is None or t is None or w is None:
                raise ValueError('Output files have different row counts')
            fields = line.rstrip().split('\t')
            chrom = canon(fields[0])
            x, y = (int(fields[1]), int(fields[4]))
            key = (x, y)
            if chrom != current:
                if chrom in closed:
                    raise ValueError('Chromosome blocks repeat: cannot certify uniqueness')
                if current is not None:
                    closed.add(current)
                current = chrom
                previous = None
            if previous is not None and key <= previous:
                raise ValueError(
                    'Post-write dedupe check failed: duplicate or unsorted coordinate pair'
                )
            previous = key
            if fields != [t[k] for k in ['chrom1', 'x1', 'x2', 'chrom2', 'y1', 'y2']]:
                raise ValueError('BEDPE/TSV coordinate mismatch')
            if t['candidate_id'] != w['candidate_id']:
                raise ValueError('Candidate/window ID mismatch')
            n += 1
    return dict(rows_checked=n, unique_pairs=n, duplicate_pairs=0, status='passed')


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        epilog='Distance units: prefer 50kb / 5mb; also accepts 50000, 50K, 50KB, 5M and 5MB (case-insensitive).'
    )
    p.add_argument(
        '--regions',
        type=Path,
        required=True,
        help='YAML assembly, chromosomes, optional coordinate intervals'
    )
    p.add_argument('--out', type=Path, required=True)
    p.add_argument(
        '--resolutions',
        nargs='+',
        type=bp,
        default=[5000],
        help='2000 5000 10000 or 2kb 5kb 10kb'
    )
    p.add_argument('--tile', choices=['sampling', 'samplings', 'force_full'], default='sampling')
    p.add_argument('--minrange', type=bp, help='Inclusive minimum y1-x1 in bp')
    p.add_argument('--maxrange', type=bp, help='Inclusive maximum y1-x1 in bp')
    p.add_argument('--start-min', type=bp, help='Inclusive minimum first-anchor start')
    p.add_argument('--start-max', type=bp, help='Inclusive maximum first-anchor start')
    p.add_argument('--stride', type=bp, help='First-anchor stride; default one resolution bin')
    budget = p.add_mutually_exclusive_group()
    budget.add_argument(
        '--samples-per-start',
        type=int,
        help='Distinct offsets per first-anchor position; default 1 if no total requested'
    )
    budget.add_argument(
        '--num-samples',
        type=int,
        help='Target number of unique candidate pairs PER RESOLUTION across all listed chromosomes'
    )
    p.add_argument(
        '--max-draws',
        type=int,
        help='Total-budget attempt cap per resolution; default max(10000, 50 * feasible target)'
    )
    p.add_argument('--decay', choices=['none', 'powerlaw', 'gumbel'], default='none')
    p.add_argument(
        '--decay-start',
        type=bp,
        help='Soft-tail knee; default prior mean + std per chromosome'
    )
    p.add_argument(
        '--decay-scale',
        type=bp,
        help='Gumbel-tail scale; default prior standard deviation'
    )
    p.add_argument('--beta', type=float, default=2.0, help='Power-law exponent')
    p.add_argument(
        '--patch-bins',
        type=int,
        default=1,
        help='Extraction window width; default anchor bin only, set to your extractor width'
    )
    p.add_argument('--window-within-region', action='store_true')
    p.add_argument('--chrom-sizes', type=Path)
    p.add_argument('--seed', type=int, default=20261005)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.tile == 'samplings':
        a.tile = 'sampling'
    if a.tile == 'force_full' and (a.num_samples is not None or a.samples_per_start is not None):
        p.error('force_full cannot be combined with a sampling budget')
    if a.samples_per_start is None:
        a.samples_per_start = 1
    if a.num_samples is not None and a.num_samples < 1:
        p.error('num-samples must be positive')
    if a.max_draws is not None and (a.max_draws < 1 or a.num_samples is None):
        p.error('max-draws requires num-samples and must be positive')
    if any((r not in [2000, 5000, 10000] for r in a.resolutions)) or len(
        set(a.resolutions)
    ) != len(a.resolutions):
        p.error('Use unique resolutions drawn from 2k, 5k, 10k')
    if a.samples_per_start < 1 or a.patch_bins < 1 or a.seed < 0:
        p.error('samples-per-start and patch-bins must be positive; seed nonnegative')
    if a.stride == 0 or a.decay_scale == 0 or a.decay_start == 0 or (
        not math.isfinite(a.beta)
    ) or (a.beta <= 0):
        p.error('stride/decay scale/decay start/beta must be positive')
    if a.minrange is not None and a.maxrange is not None and (a.minrange > a.maxrange):
        p.error('minrange exceeds maxrange')
    if a.start_min is not None and a.start_max is not None and (a.start_min > a.start_max):
        p.error('start-min exceeds start-max')
    if a.tile == 'force_full' and a.decay != 'none':
        p.error('force_full does not use probability weights; omit --decay')
    doc, regions = load_regions(a.regions.resolve(), a.chrom_sizes)
    definition = 'start_separation'
    plans = []
    for res in a.resolutions:
        stats = SAMPLING_DEFAULTS[str(res)]
        fallback = pooled_stats(stats)
        for reg in regions:
            st = stats.get(canon(reg['chrom']), fallback)
            plan = plan_region(reg, res, st, a, definition)
            plan[0]['prior_scope'] = 'chromosome' if canon(
                reg['chrom']
            ) in stats else 'pooled_fallback'
            plans.append(plan)
            expected = plan[0]['planned_candidates']
            suffix = f'{
                expected:,
            } candidates' if expected is not None else 'shares total sampling budget'
            print(
                f"{reg['chrom']} {res // 1000}k: {plan[0]['start_positions']:,} starts, {
                    plan[0]['eligible_pairs']:,
                } eligible pairs -> {suffix}",
                flush=True
            )
        if a.num_samples is not None:
            capacity = sum(
                (pl[0]['positive_weight_pairs'] for pl in plans if pl[0]['resolution'] == res)
            )
            print(
                f'{res // 1000}k: requested {
                    a.num_samples:,
                } unique pairs; positive-weight capacity {capacity:,}; attainable upper bound {
                    min(capacity, a.num_samples):,
                }',
                flush=True
            )
    report = {
        'assembly': doc['assembly'],
        'mode': a.tile,
        'distance_definition': 'start_separation: y1-x1',
        'sampling_defaults_version': SAMPLING_DEFAULTS_VERSION,
        'seed': a.seed,
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        'sampling_defaults_sha256': hashlib.sha256(
            json.dumps(SAMPLING_DEFAULTS, sort_keys=True).encode()
        ).hexdigest(),
        'regions_sha256': hashlib.sha256(a.regions.read_bytes()).hexdigest(),
        'plans': [plan[0] for plan in plans]
    }
    if a.dry_run:
        print('Dry run; no files written.')
        return
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError('Output directory is not empty; use a fresh --out')
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / 'plan.json').write_text(json.dumps(report, indent=2) + '\n')
    report['sampling_budgets'] = {}
    report['dedupe_audits'] = {}
    for res in a.resolutions:
        draws = None
        if a.num_samples is not None:
            draws, budget = sample_total(
                [pl for pl in plans if pl[0]['resolution'] == res],
                res,
                a
            )
            report['sampling_budgets'][str(res)] = budget
            print(f'Target sampling {res // 1000}k: {budget}', flush=True)
        bed = a.out / f'candidates_{res // 1000}k.bedpe'
        tab = a.out / f'candidates_{res // 1000}k.tsv'
        windows = a.out / f'patch_windows_{res // 1000}k.tsv'
        with bed.with_suffix('.bedpe.partial').open('w') as bf, tab.with_suffix(
            '.tsv.partial'
        ).open('w') as tf, windows.with_suffix('.tsv.partial').open('w') as wf:
            tw = csv.writer(tf, delimiter='\t')
            ww = csv.writer(wf, delimiter='\t')
            tw.writerow(
                [
                    'chrom1',
                    'x1',
                    'x2',
                    'chrom2',
                    'y1',
                    'y2',
                    'resolution',
                    'candidate_id',
                    'distance_bp'
                ]
            )
            ww.writerow(
                [
                    'candidate_id',
                    'chrom',
                    'row_start',
                    'row_end',
                    'col_start',
                    'col_end',
                    'patch_bins',
                    'resolution'
                ]
            )
            total = 0
            for plan in [pl for pl in plans if pl[0]['resolution'] == res]:
                c = canon(plan[0]['chrom'])
                token = int.from_bytes(hashlib.sha256(c.encode()).digest()[:4], 'little')
                rng = np.random.default_rng(np.random.SeedSequence([a.seed, res, token]))
                n = 0
                dsum = 0
                dmin = None
                dmax = None
                iterator = draws[plan[0]['chrom']] if draws is not None else generate(
                    plan,
                    res,
                    a,
                    rng
                )
                for chrom, x, y in iterator:
                    ident = f'{chrom}:{x}:{y}:{res}'
                    row = [chrom, x, x + res, chrom, y, y + res]
                    bf.write('\t'.join(map(str, row)) + '\n')
                    tw.writerow(row + [res, ident, y - x])
                    left = a.patch_bins // 2
                    ww.writerow(
                        [
                            ident,
                            chrom,
                            x - left * res,
                            x + (a.patch_bins - left) * res,
                            y - left * res,
                            y + (a.patch_bins - left) * res,
                            a.patch_bins,
                            res
                        ]
                    )
                    n += 1
                    dsum += y - x
                    dmin = y - x if dmin is None else min(dmin, y - x)
                    dmax = y - x if dmax is None else max(dmax, y - x)
                if draws is None and n != plan[0]['planned_candidates']:
                    raise RuntimeError(f"Count mismatch: {n} vs {plan[0]['planned_candidates']}")
                plan[0]['actual_candidates'] = n
                plan[0]['sampled_mean_distance_bp'] = dsum / n if n else None
                plan[0]['sampled_min_distance_bp'] = dmin
                plan[0]['sampled_max_distance_bp'] = dmax
                total += n
            print(f'Wrote {total:,} candidates at {res // 1000}k', flush=True)
        audit = audit_outputs(*[f.with_suffix(f.suffix + '.partial') for f in [bed, tab, windows]])
        report['dedupe_audits'][str(res)] = audit
        print(f'Post-write dedupe {res // 1000}k: {audit}', flush=True)
        for f in [bed, tab, windows]:
            f.with_suffix(f.suffix + '.partial').replace(f)
    (a.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Completed: {a.out}')
if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError, yaml.YAMLError) as exc:
        sys.exit(f'ERROR: {exc}')
