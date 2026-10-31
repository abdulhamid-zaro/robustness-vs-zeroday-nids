#!/usr/bin/env python3
"""Uniform per-family subsample of the improved CSE-CIC-IDS2018 (Liu et al., CNS 2022), streamed from the zip.

The full release is about 10 GB compressed, too large for the 12 GB of RAM we have. Every row is kept with a
per-family probability; when a family exceeds its cap, its probability is halved and the rows already kept are
thinned with a fair coin, so every flow of a family ends up kept with the same final probability.
The labels and the family mapping are exactly those of the main script.

usage: python prep18.py ZIP OUT_DIR
"""
import sys, os, json, time, zipfile
import numpy as np
import pandas as pd

ZIP, OUT = sys.argv[1], sys.argv[2]
CAP = {'BENIGN': 1_500_000}
DEFCAP = 250_000
CHUNK = 500_000

def family(lbl):  # identical to the experiment script
    l = lbl.lower()
    if l == 'benign' or 'attempted' in l: return 'BENIGN'
    if l.startswith('dos'): return 'DoS'
    if l.startswith('ddos'): return 'DDoS'
    if 'portscan' in l: return 'PortScan'
    if l.startswith('web attack'): return 'WebAttack'
    if 'patator' in l or 'bruteforce' in l: return 'BruteForce'
    if l.startswith('botnet'): return 'Botnet'
    if l.startswith('infiltration'): return 'Infiltration'
    if 'heartbleed' in l: return 'Heartbleed'
    return 'Other'

rng = np.random.default_rng(2018)
rate, keep, label_counts, fam_counts = {}, {}, {}, {}
cols = None
t0 = time.time()
Z = zipfile.ZipFile(ZIP)
for name in sorted(n for n in Z.namelist() if n.endswith('.csv')):
    for ch in pd.read_csv(Z.open(name), chunksize=CHUNK, low_memory=False):
        if cols is None: cols = list(ch.columns)
        assert list(ch.columns) == cols, f'column mismatch in {name}'
        lab = ch['Label'].astype(str)
        for k, v in lab.value_counts().items(): label_counts[k] = label_counts.get(k, 0) + int(v)
        fam = lab.map(family).to_numpy()
        u = rng.random(len(ch))
        for f in np.unique(fam):
            sel = fam == f
            fam_counts[f] = fam_counts.get(f, 0) + int(sel.sum())
            r = rate.setdefault(f, 1.0)
            keep.setdefault(f, []).append(ch[sel & (u < r)])
            cap = CAP.get(f, DEFCAP)
            while sum(len(x) for x in keep[f]) > cap:
                r /= 2
                keep[f] = [x[rng.random(len(x)) < 0.5] for x in keep[f]]
            rate[f] = r
    print(f'{time.strftime("%H:%M:%S")} {name} done | kept', {f: sum(len(x) for x in v) for f, v in keep.items()},
          '| rates', rate, flush=True)

os.makedirs(OUT, exist_ok=True)
df = pd.concat([x for v in keep.values() for x in v], ignore_index=True)
df.to_csv(os.path.join(OUT, 'cse_cic_ids2018_sample.csv'), index=False)
meta = {'source': 'CSECICIDS2018_improved.zip (Liu et al., CNS 2022)', 'label_counts': label_counts,
        'family_counts_full': fam_counts, 'final_keep_rate': rate,
        'kept': {f: int(sum(len(x) for x in v)) for f, v in keep.items()}, 'rows_written': len(df),
        'caps': {'BENIGN': CAP['BENIGN'], 'default': DEFCAP}, 'seed': 2018, 'minutes': (time.time() - t0) / 60}
json.dump(meta, open(os.path.join(OUT, 'sample_meta.json'), 'w'), indent=1)
print('DONE', json.dumps(meta))
