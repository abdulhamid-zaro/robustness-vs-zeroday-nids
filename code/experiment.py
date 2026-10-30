#!/usr/bin/env python3
"""Controlled evaluation: constrained adversarial robustness vs. leave-one-attack-family-out (zero-day proxy) detection.

Data: CIC-IDS2017, improved release of Liu et al. (IEEE CNS 2022), building on Engelen et al. (IEEE SPW 2021).

Scenarios (binary detection: benign vs attack):
  in_dist        train/test on all attack families (reference)
  loao           leave-one-attack-family-out: the held-out family is a genuine unseen attack (zero-day proxy)
  noise          random sign noise of the same L-inf budget (control)
  fgsm_transfer  FGSM crafted on a surrogate MLP, transferred to each target model
  pgd_transfer   PGD crafted on a surrogate MLP, transferred to each target model
  pgd_whitebox   PGD crafted on the target itself (neural targets only)
  simba_blackbox score-based black-box coordinate search on the target itself (every model, trees included)
  mixup          convex hybrids of two different attack families

Perturbations touch only attacker-controllable features (forward-direction and timing features),
are bounded in standardized signed-log space, then projected back to the raw feature space:
clipped to the training range, integer features rounded, and min <= mean <= max repaired.

Every metric is computed from real data by this script. Nothing is hard-coded.

Additional modes:
  --split temporal   chronological 70/30 split of the de-duplicated flows instead of a seeded random split
  --openset          multi-class open-set baselines (1 - P(benign) and maximum-softmax rejection) under LOAO
  --milp             exact mixed-integer evasion search on the tree ensembles (Kantchelian et al. encoding)
  --supp / --supp2   supplementary analyses reported in the paper
"""
import argparse, json, os, time, hashlib, platform
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--data', default='/content/data')
p.add_argument('--out', default='/content/results')
p.add_argument('--seeds', default='0,1,2,3,4')
p.add_argument('--models', default='rf,xgb,cnn,ft')
p.add_argument('--eps', default='0.05,0.1,0.2,0.5')
p.add_argument('--benign_train', type=int, default=100000)
p.add_argument('--benign_test', type=int, default=40000)
p.add_argument('--fam_train', type=int, default=15000)
p.add_argument('--fam_test', type=int, default=5000)
p.add_argument('--epochs', type=int, default=10)
p.add_argument('--bb_per_fam', type=int, default=500)
p.add_argument('--fpr', type=float, default=0.01)
p.add_argument('--quick', action='store_true', help='tiny smoke test')
p.add_argument('--skip_loao', action='store_true')
p.add_argument('--repair', action='store_true', help='recompute derived flow features after perturbation (consistency repair)')
p.add_argument('--audit_only', action='store_true', help='only measure inter-feature consistency of perturbed samples')
p.add_argument('--supp', action='store_true', help='supplementary run: anomaly baselines, stronger SimBA, epsilon scale')
p.add_argument('--supp2', action='store_true', help='tree-aware attack on RF/XGB (split-threshold coordinate search)')
p.add_argument('--tree_n', type=int, default=200)
p.add_argument('--split', default='random', choices=['random', 'temporal'],
               help='random: per-family random 70/30 split (default). temporal: per family, the earliest 70%% of flows '
                    '(by flow start time) train, the latest 30%% test')
p.add_argument('--openset', action='store_true', help='LOAO with multi-class open-set baselines (max-softmax rejection) for RF/XGB')
p.add_argument('--milp', action='store_true', help='exact MILP evasion/certification for RF/XGB inside the eps-box (review round 3)')
p.add_argument('--milp_n', type=int, default=40, help='flows per family for the MILP run')
p.add_argument('--milp_eps', default='0.05,0.2')
p.add_argument('--milp_models', default='xgb,rf')
p.add_argument('--milp_time', type=float, default=60.0, help='solver time limit per flow (s)')
args = p.parse_args()
if args.quick:
    args.benign_train, args.benign_test, args.fam_train, args.fam_test, args.epochs = 3000, 1500, 400, 200, 2
    args.seeds = args.seeds.split(',')[0]
    args.bb_per_fam = 30

import torch, torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from scipy.stats import binomtest
import xgboost as xgb

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(args.out, exist_ok=True)
RES = os.path.join(args.out, 'results.csv')
LOG = os.path.join(args.out, 'run_log.txt')

def log(*a):
    s = time.strftime('%H:%M:%S ') + ' '.join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, 'a') as f:
        f.write(s + '\n')

# ------------------------------------------------------------------ data
DROP = ['id', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp', 'Attempted Category']
def family(lbl):
    # covers the label sets of both improved CIC-IDS2017 and improved CSE-CIC-IDS2018 (Liu et al., CNS 2022)
    l = lbl.lower()
    if l == 'benign' or 'attempted' in l: return 'BENIGN'   # authors' recommendation: Attempted -> benign
    if l.startswith('dos'): return 'DoS'
    if l.startswith('ddos'): return 'DDoS'
    if 'portscan' in l: return 'PortScan'                  # 2018: "Infiltration - NMAP Portscan"
    if l.startswith('web attack'): return 'WebAttack'
    if 'patator' in l or 'bruteforce' in l: return 'BruteForce'   # 2018: FTP-/SSH-BruteForce
    if l.startswith('botnet'): return 'Botnet'
    if l.startswith('infiltration'): return 'Infiltration'
    if 'heartbleed' in l: return 'Heartbleed'
    return 'Other'

def load():
    frames = []
    for f in sorted(os.listdir(args.data)):
        if not f.endswith('.csv'): continue
        df = pd.read_csv(os.path.join(args.data, f))
        # flow start time is kept outside the feature set (only used by --split temporal)
        ts = (pd.to_datetime(df['Timestamp'], errors='coerce', format='mixed') if 'Timestamp' in df.columns
              else pd.Series(pd.NaT, index=df.index))
        df = df.drop(columns=[c for c in DROP if c in df.columns])
        if getattr(ts.dt, 'tz', None) is not None: ts = ts.dt.tz_convert(None)
        df['ts'] = (ts - pd.Timestamp('1970-01-01')).dt.total_seconds()   # float seconds, NaN if unparsed
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df['family'] = df['Label'].astype(str).map(family)
    df = df.drop(columns=['Label'])
    feats = [c for c in df.columns if c not in ('family', 'ts')]
    df[feats] = df[feats].apply(pd.to_numeric, errors='coerce').astype('float64')
    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feats + ['family'])
    n0 = len(df)
    df = df.drop_duplicates(subset=feats + ['family'])   # same rows as before the ts column existed
    if args.split == 'temporal':
        log(f'timestamps parsed for {df.ts.notna().mean():.4%} of flows')
        df = df[df.ts.notna()]
    # flows whose identical feature vector carries different families are ambiguous -> drop
    amb = df.duplicated(subset=feats, keep=False)
    df = df[~amb]
    log(f'loaded {n0} rows, {n0 - len(df)} removed as duplicates/ambiguous, {len(df)} kept')
    nunique = df[feats].nunique()
    feats = [c for c in feats if nunique[c] > 1]
    return df, feats

# attacker-controllable features: forward direction + timing + flow-level size/rate aggregates
def modifiable(c):
    cl = c.lower()
    if 'bwd' in cl or 'flag' in cl or cl in ('protocol', 'icmp code', 'icmp type', 'down/up ratio',
                                             'fwd init win bytes', 'bwd init win bytes', 'fwd seg size min'):
        return False
    return ('fwd' in cl or 'flow' in cl or 'packet length' in cl or 'average packet size' in cl
            or cl.startswith('active') or cl.startswith('idle') or 'total tcp flow time' in cl
            or 'iat' in cl)

TRIPLETS = [('Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Max'),
            ('Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Max'),
            ('Flow IAT Min', 'Flow IAT Mean', 'Flow IAT Max'),
            ('Fwd IAT Min', 'Fwd IAT Mean', 'Fwd IAT Max'),
            ('Bwd IAT Min', 'Bwd IAT Mean', 'Bwd IAT Max'),
            ('Packet Length Min', 'Packet Length Mean', 'Packet Length Max'),
            ('Active Min', 'Active Mean', 'Active Max'),
            ('Idle Min', 'Idle Mean', 'Idle Max')]

# derived flow features (CICFlowMeter definitions; Flow Duration in microseconds)
RELATIONS = [
    ('Flow Bytes/s', ('Total Length of Fwd Packet', 'Total Length of Bwd Packet', 'Flow Duration'), lambda a, b, d: (a + b) / (d / 1e6)),
    ('Flow Packets/s', ('Total Fwd Packet', 'Total Bwd packets', 'Flow Duration'), lambda a, b, d: (a + b) / (d / 1e6)),
    ('Fwd Packets/s', ('Total Fwd Packet', 'Flow Duration'), lambda a, d: a / (d / 1e6)),
    ('Bwd Packets/s', ('Total Bwd packets', 'Flow Duration'), lambda a, d: a / (d / 1e6)),
    ('Fwd Packet Length Mean', ('Total Length of Fwd Packet', 'Total Fwd Packet'), lambda a, n: a / n),
    ('Bwd Packet Length Mean', ('Total Length of Bwd Packet', 'Total Bwd packets'), lambda a, n: a / n),
    ('Fwd Segment Size Avg', ('Total Length of Fwd Packet', 'Total Fwd Packet'), lambda a, n: a / n),
    ('Average Packet Size', ('Total Length of Fwd Packet', 'Total Length of Bwd Packet', 'Total Fwd Packet', 'Total Bwd packets'), lambda a, b, n, m: (a + b) / (n + m)),
]

def rel_value(X, idx, srcs, f):
    cols = [X[:, idx[c]] for c in srcs]
    with np.errstate(divide='ignore', invalid='ignore'):
        val = f(*cols)
    ok = np.isfinite(val) & np.all(np.stack([np.abs(c) > 0 for c in cols[-1:]]), axis=0)
    return val, ok

def rel_ok(X, idx, r):
    tgt, srcs, f = r
    val, ok = rel_value(X, idx, srcs, f)
    if ok.sum() < 100: return 0.0
    t = X[ok, idx[tgt]]
    return float((np.abs(t - val[ok]) / np.maximum(np.abs(val[ok]), 1e-9) <= 0.01).mean())

class Space:
    """signed-log + standardization fitted on training data; projection back to feasible raw space."""
    def __init__(self, Xtr, feats):
        self.feats = feats
        L = np.sign(Xtr) * np.log1p(np.abs(Xtr))
        self.mu, self.sd = L.mean(0), L.std(0) + 1e-6
        self.lo, self.hi = Xtr.min(0), Xtr.max(0)
        self.isint = np.all(np.abs(Xtr - np.round(Xtr)) < 1e-9, axis=0)
        self.mask = np.array([modifiable(c) for c in feats])
        idx = {c: i for i, c in enumerate(feats)}
        self.trip = [tuple(idx[c] for c in t) for t in TRIPLETS if all(c in idx for c in t)]
        self.nonneg = self.lo >= 0
        self.idx = idx
        # keep only the derived-feature relations that actually hold on clean training data
        self.rels = [r for r in RELATIONS if all(c in idx for c in r[1])]
        self.rels = [r for r in self.rels if rel_ok(Xtr, idx, r) >= 0.95]
    def to_z(self, X):
        return ((np.sign(X) * np.log1p(np.abs(X)) - self.mu) / self.sd).astype(np.float32)
    def to_raw(self, Z):
        L = Z.astype(np.float64) * self.sd + self.mu
        return np.sign(L) * np.expm1(np.abs(L))
    def project(self, X, repair=None):
        X = np.clip(X, self.lo, self.hi)
        X[:, self.isint] = np.round(X[:, self.isint])
        for a, b, c in self.trip:
            v = np.sort(X[:, [a, b, c]], axis=1)
            X[:, a], X[:, b], X[:, c] = v[:, 0], v[:, 1], v[:, 2]
        if args.repair if repair is None else repair:
            for tgt, srcs, f in self.rels:
                val, ok = rel_value(X, self.idx, srcs, f)
                X[ok, self.idx[tgt]] = val[ok]
        return X
    def violations(self, X):
        out = {}
        for tgt, srcs, f in self.rels:
            val, ok = rel_value(X, self.idx, srcs, f)
            t = X[ok, self.idx[tgt]]
            err = np.abs(t - val[ok]) / np.maximum(np.abs(val[ok]), 1e-9)
            out[tgt] = float((err > 0.01).mean()) if ok.any() else float('nan')
        return out

# ------------------------------------------------------------------ models
class MLP(nn.Module):
    def __init__(s, d):
        super().__init__(); s.n = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
    def forward(s, x): return s.n(x).squeeze(-1)

class CNN1D(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.c = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveMaxPool1d(8))
        s.f = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(s, x): return s.f(s.c(x.unsqueeze(1))).squeeze(-1)

class FTTransformer(nn.Module):
    """Feature-Tokenizer Transformer (Gorishniy et al., 2021), compact configuration."""
    def __init__(s, d, tok=32, layers=2, heads=4):
        super().__init__()
        s.w = nn.Parameter(torch.randn(d, tok) * 0.02); s.b = nn.Parameter(torch.zeros(d, tok))
        s.cls = nn.Parameter(torch.zeros(1, 1, tok))
        enc = nn.TransformerEncoderLayer(tok, heads, dim_feedforward=64, dropout=0.1, batch_first=True, norm_first=True)
        s.t = nn.TransformerEncoder(enc, layers)
        s.h = nn.Sequential(nn.LayerNorm(tok), nn.ReLU(), nn.Linear(tok, 1))
    def forward(s, x):
        t = x.unsqueeze(-1) * s.w + s.b
        t = torch.cat([s.cls.expand(x.shape[0], -1, -1), t], 1)
        return s.h(s.t(t)[:, 0]).squeeze(-1)

class AE(nn.Module):
    def __init__(s, d):
        super().__init__(); s.e = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 16), nn.ReLU()); s.d = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, d))
    def forward(s, x): return ((s.d(s.e(x)) - x) ** 2).mean(-1)

NETS = {'mlp': MLP, 'cnn': CNN1D, 'ft': FTTransformer}

def train_ae(Z, seed, epochs=20):
    torch.manual_seed(seed)
    m = AE(Z.shape[1]).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Zt = torch.tensor(Z, device=DEV); g = torch.Generator(device='cpu').manual_seed(seed)
    m.train()
    for _ in range(epochs):
        perm = torch.randperm(len(Zt), generator=g)
        for i in range(0, len(Zt), 512):
            b = perm[i:i + 512]; opt.zero_grad(); m(Zt[b]).mean().backward(); opt.step()
    m.eval(); return m

class Anomaly:
    """Benign-only detectors: every attack family is unseen by construction. Higher score = more anomalous."""
    def __init__(s, kind, sp, Xb, seed):
        from sklearn.ensemble import IsolationForest
        s.kind, s.sp = kind, sp
        Z = sp.to_z(Xb)
        if kind == 'iforest': s.m = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1).fit(Z)
        else: s.m = train_ae(Z, seed)
    def score(s, X):
        Z = s.sp.to_z(X)
        if s.kind == 'iforest': return -s.m.score_samples(Z)
        out = []
        with torch.no_grad():
            for i in range(0, len(Z), 4096): out.append(s.m(torch.tensor(Z[i:i + 4096], device=DEV)).cpu().numpy())
        return np.concatenate(out)

def train_net(kind, Z, y, seed):
    torch.manual_seed(seed)
    m = NETS[kind](Z.shape[1]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    pw = torch.tensor((y == 0).sum() / max(1, (y == 1).sum()), dtype=torch.float32, device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    Zt, yt = torch.tensor(Z, device=DEV), torch.tensor(y, dtype=torch.float32, device=DEV)
    g = torch.Generator(device='cpu').manual_seed(seed)
    m.train()
    for _ in range(args.epochs):
        perm = torch.randperm(len(Zt), generator=g)
        for i in range(0, len(Zt), 512):
            b = perm[i:i + 512]
            opt.zero_grad(); lossf(m(Zt[b]), yt[b]).backward(); opt.step()
    m.eval()
    return m

def net_score(m, Z):
    out = []
    with torch.no_grad():
        for i in range(0, len(Z), 4096):
            out.append(torch.sigmoid(m(torch.tensor(Z[i:i + 4096], device=DEV))).cpu().numpy())
    return np.concatenate(out)

class Model:
    def __init__(s, kind, sp, Xtr, ytr, seed):
        s.kind, s.sp = kind, sp
        if kind == 'rf':
            s.m = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=2, class_weight='balanced',
                                         n_jobs=-1, random_state=seed).fit(Xtr, ytr)
        elif kind == 'xgb':
            s.m = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                    tree_method='hist', device='cuda' if DEV == 'cuda' else 'cpu',
                                    scale_pos_weight=(ytr == 0).sum() / max(1, (ytr == 1).sum()), random_state=seed,
                                    eval_metric='logloss').fit(Xtr, ytr)
        else:
            s.m = train_net(kind, sp.to_z(Xtr), ytr, seed)
    def score(s, X):
        if s.kind in ('rf', 'xgb'): return s.m.predict_proba(X)[:, 1]
        return net_score(s.m, s.sp.to_z(X))

# ------------------------------------------------------------------ attacks (in z-space, on modifiable features only)
def craft(net, sp, X, eps, steps):
    Z0 = torch.tensor(sp.to_z(X), device=DEV)
    mask = torch.tensor(sp.mask, device=DEV, dtype=torch.float32)
    out = []
    for i in range(0, len(Z0), 2048):
        z0 = Z0[i:i + 2048]; z = z0.clone()
        a = eps if steps == 1 else eps / 4
        for _ in range(steps):
            z.requires_grad_(True)
            loss = net(z).sum()                     # attack logit; attacker wants it LOW
            gz, = torch.autograd.grad(loss, z)
            z = (z.detach() - a * gz.sign() * mask)
            z = z0 + torch.clamp(z - z0, -eps, eps)
        out.append(z.detach().cpu().numpy())
    return sp.project(sp.to_raw(np.concatenate(out)))

def simba(model, sp, X, eps, rng, passes=2):  # noqa
    """Score-based black-box attack (SimBA-style coordinate search) usable on any model, trees included.
    For each modifiable feature (random order), try +eps and -eps in z-space and keep the change if the
    attack score decreases. About 4 x (#modifiable features) queries per sample."""
    Z0 = sp.to_z(X); Z = Z0.copy()
    best = model.score(sp.project(sp.to_raw(Z)))
    idx = np.flatnonzero(sp.mask)
    for _ in range(passes):
        for j in rng.permutation(idx):
            for d in (eps, -eps):
                Zc = Z.copy(); Zc[:, j] = Z0[:, j] + d
                s = model.score(sp.project(sp.to_raw(Zc)))
                better = s < best
                Z[better, j] = Zc[better, j]; best = np.where(better, s, best)
    return sp.project(sp.to_raw(Z))

def noise(sp, X, eps, rng):
    Z = sp.to_z(X)
    Z = Z + eps * rng.choice([-1.0, 1.0], size=Z.shape).astype(np.float32) * sp.mask
    return sp.project(sp.to_raw(Z))

# ------------------------------------------------------------------ metrics
def thr_at_fpr(sb):
    return float(np.quantile(sb, 1 - args.fpr))

def mcnemar(a, b):
    # a, b: boolean detected flags on the same samples
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    return 1.0 if n01 + n10 == 0 else float(binomtest(n01, n01 + n10, 0.5).pvalue)

ROWS = []
def emit(**r):
    ROWS.append(r)
    pd.DataFrame(ROWS).to_csv(RES, index=False)

def sample(df, fam, n, rng):
    d = df[df.family == fam]
    return d.sample(n=min(n, len(d)), random_state=int(rng.integers(1e9)))

def split_pools(df, seed):
    """70/30 train/test pools per family: seeded random shuffle (default) or by flow start time (--split temporal)."""
    pool_tr, pool_te = [], []
    for f, d in df.groupby('family'):
        d = d.sort_values('ts', kind='stable') if args.split == 'temporal' else d.sample(frac=1.0, random_state=seed)
        k = int(0.7 * len(d)); pool_tr.append(d.iloc[:k]); pool_te.append(d.iloc[k:])
    return pd.concat(pool_tr), pd.concat(pool_te)

# ------------------------------------------------------------------ main
def main():
    t0 = time.time()
    log('device', DEV, '| torch', torch.__version__, '| xgb', xgb.__version__, '| python', platform.python_version(), '| args', vars(args))
    df, feats = load()
    log('features', len(feats), '| modifiable', sum(modifiable(c) for c in feats))
    log('family counts', df.family.value_counts().to_dict())
    fams = [f for f in ['DoS', 'DDoS', 'PortScan', 'BruteForce', 'Botnet', 'WebAttack'] if (df.family == f).sum() >= 50]
    json.dump({'features': feats, 'modifiable': [c for c in feats if modifiable(c)], 'families': fams},
              open(os.path.join(args.out, 'config.json'), 'w'), indent=1)

    for seed in [int(s) for s in args.seeds.split(',')]:
        rng = np.random.default_rng(seed)
        # split each family 70/30 into train/test pools (seeded)
        TR, TE = split_pools(df, seed)
        btr = sample(TR, 'BENIGN', args.benign_train, rng)
        bva, btr = btr.iloc[:len(btr) // 5], btr.iloc[len(btr) // 5:]        # benign validation for the threshold
        bte = sample(TE, 'BENIGN', args.benign_test, rng)
        atr = {f: sample(TR, f, args.fam_train, rng) for f in fams}
        ate = {f: sample(TE, f, args.fam_test, rng) for f in fams}
        X = lambda d: d[feats].to_numpy(np.float64)

        def fit_all(train_fams, tag):
            dtr = pd.concat([btr] + [atr[f] for f in train_fams])
            Xtr, ytr = X(dtr), (dtr.family != 'BENIGN').to_numpy(int)
            sp = Space(Xtr, feats)
            ms = {}
            for k in [m for m in args.models.split(',') if m]:
                t = time.time(); ms[k] = Model(k, sp, Xtr, ytr, seed)
                log(f'seed {seed} {tag} trained {k} in {time.time() - t:.0f}s')
            sur = train_net('mlp', sp.to_z(Xtr), ytr, seed + 1000)   # independent surrogate for transfer attacks
            return sp, ms, sur

        # ---------- in-distribution + perturbation scenarios
        sp, ms, sur = fit_all(fams, 'in_dist')
        Xb, Xa = X(bte), X(pd.concat([ate[f] for f in fams]))
        fam_of = np.concatenate([[f] * len(ate[f]) for f in fams])
        thr = {k: thr_at_fpr(m.score(X(bva))) for k, m in ms.items()}
        clean = {}
        for k, m in ms.items():
            sb, sa = m.score(Xb), m.score(Xa)
            y = np.r_[np.zeros(len(sb)), np.ones(len(sa))]; s = np.r_[sb, sa]
            clean[k] = sa >= thr[k]
            emit(seed=seed, model=k, scenario='in_dist', family='ALL', epsilon=0.0, n_eval=len(sa),
                 detection_rate=float(clean[k].mean()), fpr=float((sb >= thr[k]).mean()), auroc=float(roc_auc_score(y, s)),
                 f1=float(f1_score(y, s >= thr[k])), dr_clean=float(clean[k].mean()), mcnemar_p=np.nan)
            for f in fams:
                sel = fam_of == f
                emit(seed=seed, model=k, scenario='in_dist', family=f, epsilon=0.0, n_eval=int(sel.sum()),
                     detection_rate=float(clean[k][sel].mean()), fpr=float((sb >= thr[k]).mean()),
                     auroc=float(roc_auc_score(np.r_[np.zeros(len(sb)), np.ones(sel.sum())], np.r_[sb, sa[sel]])),
                     f1=np.nan, dr_clean=float(clean[k][sel].mean()), mcnemar_p=np.nan)

        log(f'seed {seed} in_dist detection@{args.fpr:.0%}FPR', {k: round(float(v.mean()), 4) for k, v in clean.items()})
        log(f'seed {seed} relations kept for audit/repair', [r[0] for r in sp.rels])
        AUD = os.path.join(args.out, 'audit.csv')
        def audit(name, Xv, eps):
            row = {'seed': seed, 'variant': name, 'epsilon': eps, 'repair': args.repair}
            row.update(sp.violations(Xv)); pd.DataFrame([row]).to_csv(AUD, mode='a', header=not os.path.exists(AUD), index=False)
        audit('clean', Xa.copy(), 0.0)
        if args.audit_only:
            for eps in [float(e) for e in args.eps.split(',')]:
                audit('noise', noise(sp, Xa, eps, rng), eps)
                audit('pgd_transfer', craft(sur, sp, Xa, eps, 10), eps)
            n = min(3000, len(Xa)); i = rng.integers(0, len(Xa), n); j = rng.integers(0, len(Xa), n)
            ok = fam_of[i] != fam_of[j]; i, j = i[ok], j[ok]
            lam = rng.uniform(0.3, 0.7, size=(len(i), 1))
            audit('mixup', sp.project(lam * Xa[i] + (1 - lam) * Xa[j]), 0.0)
            log(f'seed {seed} audit done'); continue
        for eps in [float(e) for e in args.eps.split(',')]:
            variants = {'noise': noise(sp, Xa, eps, rng),
                        'fgsm_transfer': craft(sur, sp, Xa, eps, 1),
                        'pgd_transfer': craft(sur, sp, Xa, eps, 10)}
            for k, m in ms.items():
                vs = dict(variants)
                if k in NETS: vs['pgd_whitebox'] = craft(m.m, sp, Xa, eps, 10)
                for sc, Xv in vs.items():
                    det = m.score(Xv) >= thr[k]
                    for f in ['ALL'] + fams:
                        sel = np.ones(len(det), bool) if f == 'ALL' else fam_of == f
                        emit(seed=seed, model=k, scenario=sc, family=f, epsilon=eps, n_eval=int(sel.sum()),
                             detection_rate=float(det[sel].mean()), fpr=np.nan, auroc=np.nan, f1=np.nan,
                             dr_clean=float(clean[k][sel].mean()), mcnemar_p=mcnemar(clean[k][sel], det[sel]))
            # black-box attack on every model, on a per-family subsample (query-heavy)
            sub = np.concatenate([rng.permutation(np.flatnonzero(fam_of == f))[:args.bb_per_fam] for f in fams])
            for k, m in ms.items():
                det = m.score(simba(m, sp, Xa[sub], eps, rng)) >= thr[k]
                cs, fs = clean[k][sub], fam_of[sub]
                for f in ['ALL'] + fams:
                    sel = np.ones(len(det), bool) if f == 'ALL' else fs == f
                    emit(seed=seed, model=k, scenario='simba_blackbox', family=f, epsilon=eps, n_eval=int(sel.sum()),
                         detection_rate=float(det[sel].mean()), fpr=np.nan, auroc=np.nan, f1=np.nan,
                         dr_clean=float(cs[sel].mean()), mcnemar_p=mcnemar(cs[sel], det[sel]))
            log(f'seed {seed} eps {eps} perturbation scenarios done')

        # ---------- mixup hybrids of two different families (raw space, then projected)
        n = min(3000, len(Xa)); i = rng.integers(0, len(Xa), n); j = rng.integers(0, len(Xa), n)
        ok = fam_of[i] != fam_of[j]; i, j = i[ok], j[ok]
        lam = rng.uniform(0.3, 0.7, size=(len(i), 1))
        Xm = sp.project(lam * Xa[i] + (1 - lam) * Xa[j])
        for k, m in ms.items():
            det = m.score(Xm) >= thr[k]
            emit(seed=seed, model=k, scenario='mixup', family='ALL', epsilon=0.0, n_eval=len(det),
                 detection_rate=float(det.mean()), fpr=np.nan, auroc=np.nan, f1=np.nan,
                 dr_clean=float((clean[k][i] & clean[k][j]).mean()), mcnemar_p=mcnemar(clean[k][i] & clean[k][j], det))

        # ---------- leave-one-attack-family-out: real unseen attacks
        if not args.skip_loao:
            for held in fams:
                spL, msL, _ = fit_all([f for f in fams if f != held], f'loao-{held}')
                Xh = X(ate[held])
                for k, m in msL.items():
                    th = thr_at_fpr(m.score(X(bva)))
                    sb, sh = m.score(Xb), m.score(Xh)
                    emit(seed=seed, model=k, scenario='loao', family=held, epsilon=0.0, n_eval=len(sh),
                         detection_rate=float((sh >= th).mean()), fpr=float((sb >= th).mean()),
                         auroc=float(roc_auc_score(np.r_[np.zeros(len(sb)), np.ones(len(sh))], np.r_[sb, sh])),
                         f1=np.nan, dr_clean=np.nan, mcnemar_p=np.nan)
        log(f'seed {seed} complete | elapsed {(time.time() - t0) / 60:.1f} min')

    h = hashlib.md5(open(RES, 'rb').read()).hexdigest()
    log('DONE rows', len(ROWS), 'md5', h)

def supp_main():
    t0 = time.time()
    log('SUPP device', DEV, '| args', vars(args))
    df, feats = load()
    fams = [f for f in ['DoS', 'DDoS', 'PortScan', 'BruteForce', 'Botnet', 'WebAttack'] if (df.family == f).sum() >= 50]
    SUP = os.path.join(args.out, 'results_supp.csv'); rows = []
    def put(**r):
        rows.append(r); pd.DataFrame(rows).to_csv(SUP, index=False)
    for seed in [int(s) for s in args.seeds.split(',')]:
        rng = np.random.default_rng(seed)
        TR, TE = split_pools(df, seed)
        btr = sample(TR, 'BENIGN', args.benign_train, rng)
        bva, btr = btr.iloc[:len(btr) // 5], btr.iloc[len(btr) // 5:]
        bte = sample(TE, 'BENIGN', args.benign_test, rng)
        atr = {f: sample(TR, f, args.fam_train, rng) for f in fams}
        ate = {f: sample(TE, f, args.fam_test, rng) for f in fams}
        X = lambda d: d[feats].to_numpy(np.float64)
        Xb, Xva = X(bte), X(bva)
        # (1) epsilon scale: multiplicative change of (1+|x|) implied by eps for each modifiable feature
        dtr = pd.concat([btr] + [atr[f] for f in fams]); sp = Space(X(dtr), feats)
        for e in [0.05, 0.1, 0.2, 0.5]:
            fac = np.exp(e * sp.sd[sp.mask])
            put(seed=seed, scenario='eps_scale', model='-', family='-', epsilon=e, n_eval=int(sp.mask.sum()), detection_rate=np.nan, auroc=np.nan,
                fpr=np.nan, value_median=float(np.median(fac)), value_q25=float(np.quantile(fac, .25)), value_q75=float(np.quantile(fac, .75)))
        # (2) anomaly detectors trained on benign only
        spb = Space(X(btr), feats)
        sub = {f: rng.permutation(len(ate[f]))[:args.bb_per_fam] for f in fams}
        for kind in ['iforest', 'ae']:
            t = time.time(); m = Anomaly(kind, spb, X(btr), seed)
            th = thr_at_fpr(m.score(Xva)); sb = m.score(Xb)
            for f in fams:
                Xf = X(ate[f]); sf = m.score(Xf)
                put(seed=seed, scenario='anomaly_unseen', model=kind, family=f, epsilon=0.0, n_eval=len(sf), detection_rate=float((sf >= th).mean()),
                    auroc=float(roc_auc_score(np.r_[np.zeros(len(sb)), np.ones(len(sf))], np.r_[sb, sf])), fpr=float((sb >= th).mean()))
            for e in [0.2, 0.5]:
                for f in fams:
                    Xs = X(ate[f])[sub[f]]
                    det0 = m.score(Xs) >= th; det = m.score(simba(m, spb, Xs, e, rng)) >= th
                    put(seed=seed, scenario='anomaly_simba', model=kind, family=f, epsilon=e, n_eval=len(det), detection_rate=float(det.mean()),
                        dr_clean=float(det0.mean()), auroc=np.nan, fpr=np.nan)
            log(f'seed {seed} anomaly {kind} done in {time.time() - t:.0f}s')
        # (3) stronger black-box search against the tree ensembles (5 passes instead of 2)
        Xtr_, ytr_ = X(dtr), (dtr.family != 'BENIGN').to_numpy(int)
        for kind in ['rf', 'xgb']:
            m = Model(kind, sp, Xtr_, ytr_, seed); th = thr_at_fpr(m.score(Xva))
            for f in fams:
                Xs = X(ate[f])[sub[f]][:200]
                det = m.score(simba(m, sp, Xs, 0.5, rng, passes=5)) >= th
                put(seed=seed, scenario='simba_strong', model=kind, family=f, epsilon=0.5, n_eval=len(det), detection_rate=float(det.mean()), auroc=np.nan, fpr=np.nan)
            log(f'seed {seed} strong simba {kind} done')
        log(f'seed {seed} supp complete | {(time.time() - t0) / 60:.1f} min')
    log('SUPP DONE rows', len(rows), 'md5', hashlib.md5(open(SUP, 'rb').read()).hexdigest())


# ------------------------------------------------------------------ tree-aware attack (review round 2)
def tree_thresholds(model, nfeat):
    """Split thresholds (raw feature space) used by a fitted RF or XGBoost model, per feature."""
    th = [[] for _ in range(nfeat)]
    if model.kind == 'rf':
        for est in model.m.estimators_:
            t = est.tree_
            for f, v in zip(t.feature, t.threshold):
                if f >= 0: th[f].append(v)
    else:
        df = model.m.get_booster().trees_to_dataframe()
        df = df[df.Feature != 'Leaf']
        for f, v in zip(df.Feature, df.Split):
            th[int(str(f).lstrip('f'))].append(v)
    return [np.unique(np.asarray(t, np.float64)) for t in th]

def tree_attack(model, sp, X, eps, rng, passes=3, K=24):
    """Score-based coordinate search that, unlike SimBA's +/-eps corners, tries every tree split threshold
    of a feature that falls inside the flow's eps-box (just below and just above it), plus both box ends.
    This is exact 1-D minimisation per coordinate for piecewise-constant tree ensembles (up to K sampled
    thresholds per flow and feature)."""
    TH = tree_thresholds(model, X.shape[1])
    Z0 = sp.to_z(X); Xc = sp.project(X.copy()); best = model.score(Xc)
    lo_raw = sp.to_raw(Z0 - eps); hi_raw = sp.to_raw(Z0 + eps)
    idx = np.flatnonzero(sp.mask)
    for _ in range(passes):
        for j in rng.permutation(idx):
            t = TH[j]
            cand_rows, cand_vals = [], []
            for i in range(len(X)):
                a, b = min(lo_raw[i, j], hi_raw[i, j]), max(lo_raw[i, j], hi_raw[i, j])
                inside = t[(t >= a) & (t <= b)]
                if len(inside) > K: inside = rng.choice(inside, K, replace=False)
                d = 1e-6 * (1 + np.abs(inside))
                vals = np.concatenate([[a, b], inside - d, inside + d])
                vals = vals[(vals >= a) & (vals <= b)]
                cand_rows.append(np.full(len(vals), i)); cand_vals.append(vals)
            r = np.concatenate(cand_rows); v = np.concatenate(cand_vals)
            Xt = Xc[r].copy(); Xt[:, j] = v
            Xt = sp.project(Xt)
            sc = model.score(Xt)
            order = np.lexsort((sc, r)); ro = r[order]
            first = order[np.r_[True, ro[1:] != ro[:-1]]]          # lowest-score candidate per flow
            fi = r[first]; better = sc[first] < best[fi]
            best[fi[better]] = sc[first[better]]; Xc[fi[better]] = Xt[first[better]]
    return Xc

def supp2_main():
    t0 = time.time()
    log('SUPP2 device', DEV, '| args', vars(args))
    df, feats = load()
    fams = [f for f in ['DoS', 'DDoS', 'PortScan', 'BruteForce', 'Botnet', 'WebAttack'] if (df.family == f).sum() >= 50]
    SUP = os.path.join(args.out, 'results_supp2.csv'); rows = []
    def put(**r):
        rows.append(r); pd.DataFrame(rows).to_csv(SUP, index=False)
    for seed in [int(s) for s in args.seeds.split(',')]:
        rng = np.random.default_rng(seed)
        TR, TE = split_pools(df, seed)
        btr = sample(TR, 'BENIGN', args.benign_train, rng)
        bva, btr = btr.iloc[:len(btr) // 5], btr.iloc[len(btr) // 5:]
        bte = sample(TE, 'BENIGN', args.benign_test, rng)
        atr = {f: sample(TR, f, args.fam_train, rng) for f in fams}
        ate = {f: sample(TE, f, args.fam_test, rng) for f in fams}
        X = lambda d: d[feats].to_numpy(np.float64)
        dtr = pd.concat([btr] + [atr[f] for f in fams]); sp = Space(X(dtr), feats)
        Xtr_, ytr_ = X(dtr), (dtr.family != 'BENIGN').to_numpy(int)
        sub = {f: rng.permutation(len(ate[f]))[:args.tree_n] for f in fams}
        for kind in ['rf', 'xgb']:
            t = time.time()
            m = Model(kind, sp, Xtr_, ytr_, seed); th = thr_at_fpr(m.score(X(bva)))
            for e in [0.2, 0.5]:
                for f in fams:
                    Xs = X(ate[f])[sub[f]]
                    det0 = m.score(sp.project(Xs.copy())) >= th
                    det_s = m.score(simba(m, sp, Xs, e, rng)) >= th
                    det_t = m.score(tree_attack(m, sp, Xs, e, rng)) >= th
                    put(seed=seed, model=kind, family=f, epsilon=e, n_eval=len(Xs), dr_clean=float(det0.mean()),
                        dr_simba=float(det_s.mean()), dr_tree=float(det_t.mean()))
            log(f'seed {seed} tree attack {kind} done in {time.time() - t:.0f}s')
        log(f'seed {seed} supp2 complete | {(time.time() - t0) / 60:.1f} min')
    log('SUPP2 DONE rows', len(rows), 'md5', hashlib.md5(open(SUP, 'rb').read()).hexdigest())

# ------------------------------------------------------------------ exact MILP evasion / certification (review round 3)
def tree_arrays(model):
    """Each tree as (left, right, feature, threshold, leaf_value) arrays; 'le' = go left if x <= t (sklearn),
    'lt' = go left if x < t (XGBoost). Leaf values add up to the RF probability or to the XGB margin."""
    trees = []
    if model.kind == 'rf':
        T = len(model.m.estimators_); c1 = list(model.m.classes_).index(1)
        for est in model.m.estimators_:
            t = est.tree_
            v = t.value[:, 0, :]; v = v / np.maximum(v.sum(1, keepdims=True), 1e-300)
            trees.append((t.children_left, t.children_right, t.feature, t.threshold.astype(np.float64), v[:, c1] / T))
        return trees, 'le'
    tdf = model.m.get_booster().trees_to_dataframe()
    for _, g in tdf.groupby('Tree', sort=True):
        ids = {nid: i for i, nid in enumerate(g.ID)}
        n = len(g); L = np.full(n, -1); R = np.full(n, -1); F = np.full(n, -2); TH = np.zeros(n); V = np.zeros(n)
        for i, (fe, sp_, y, no, gain) in enumerate(zip(g.Feature, g.Split, g.Yes, g.No, g.Gain)):
            if fe == 'Leaf': V[i] = gain
            else: F[i] = int(str(fe).lstrip('f')); TH[i] = np.float32(sp_); L[i] = ids[y]; R[i] = ids[no]
        root = ids[g.ID.iloc[0]]; assert root == 0
        trees.append((L, R, F, TH, V))
    return trees, 'lt'

def tree_sum(trees, rule, x):
    s = 0.0
    xf = x.astype(np.float32).astype(np.float64)
    for L, R, F, TH, V in trees:
        i = 0
        while L[i] != -1:
            go_left = xf[F[i]] <= TH[i] if rule == 'le' else xf[F[i]] < TH[i]
            i = L[i] if go_left else R[i]
        s += V[i]
    return s

def attack_box(sp, x, eps):
    """Raw-space interval per feature for an attacker who moves each modifiable feature by at most eps in z-space,
    before the min<=mean<=max sort of the projection (clip and integer rounding are monotone and applied here;
    the sort is modelled exactly inside the MILP). Non-modifiable features are fixed (up to float round-off)."""
    z0 = sp.to_z(x[None])[0]
    lo = sp.to_raw((z0 - eps * sp.mask)[None])[0]; hi = sp.to_raw((z0 + eps * sp.mask)[None])[0]
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    base = sp.project(sp.to_raw(z0[None]))[0]            # what an unmodified flow looks like after projection
    lo = np.clip(lo, sp.lo, sp.hi); hi = np.clip(hi, sp.lo, sp.hi)
    lo[sp.isint] = np.round(lo[sp.isint]); hi[sp.isint] = np.round(hi[sp.isint])
    fx = ~sp.mask
    lo[fx] = np.minimum(base[fx], x[fx]); hi[fx] = np.maximum(base[fx], x[fx])
    # widen by a relative 1e-6 so float32 round-off in the z-space attacks cannot step outside the box
    lo = lo - 1e-6 * np.maximum(1.0, np.abs(lo)); hi = hi + 1e-6 * np.maximum(1.0, np.abs(hi))
    return lo, hi, base

def _f32_below(t):   # largest float32 value <= t
    v = np.float32(t)
    return float(v) if float(v) <= t else float(np.nextafter(v, np.float32(-np.inf)))

def _f32_above(t):   # smallest float32 value > t
    v = np.float32(t)
    return float(np.nextafter(v, np.float32(np.inf))) if float(v) <= t else float(v)

def milp_attack(trees, rule, lo, hi, target, time_limit, sp, base, _debug=None):
    """Exact evasion search for a tree ensemble (Kantchelian et al., ICML 2016 encoding, restricted to the nodes
    reachable inside the attacker's box): a binary predicate per (feature, split) that can go either way, one
    indicator per reachable leaf, a real-valued variable per relevant feature linked to its predicates, and the
    constraint 'ensemble output < detection threshold'. The projection is modelled exactly: intervals are already
    clipped and rounded, and each min/mean/max triplet is the sorted version of three values drawn from the
    members' own intervals (permutation variables; integrality follows the source feature).
    infeasible => certified: no reachable perturbation evades. feasible => returns a concrete flow."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import coo_matrix
    hl, hh = lo.copy(), hi.copy()                        # hull: values a triplet position can take after the sort
    trips = [t for t in sp.trip if sp.mask[list(t)].all()]
    for a, b, c in trips:
        hl[[a, b, c]] = min(lo[a], lo[b], lo[c]); hh[[a, b, c]] = max(hi[a], hi[b], hi[c])
    lo32 = hl.astype(np.float32).astype(np.float64); hi32 = hh.astype(np.float32).astype(np.float64)
    preds = {}; const = 0.0; leaf_terms = []
    for L, R, F, TH, V in trees:
        leaves = []; stack = [(0, [])]
        while stack:
            i, cond = stack.pop()
            if L[i] == -1: leaves.append((V[i], cond)); continue
            f, t = F[i], TH[i]
            if rule == 'le': only_left, only_right = hi32[f] <= t, lo32[f] > t
            else: only_left, only_right = hi32[f] < t, lo32[f] >= t
            if only_left: stack.append((L[i], cond))
            elif only_right: stack.append((R[i], cond))
            else:
                k = preds.setdefault((f, t), len(preds))
                stack.append((L[i], cond + [(k, 1)])); stack.append((R[i], cond + [(k, 0)]))
        if len(leaves) == 1: const += leaves[0][0]
        else: leaf_terms.append(leaves)
    nP = len(preds); nL = sum(len(t) for t in leaf_terms)
    byf = {}
    for (f, t), k in preds.items(): byf.setdefault(f, []).append((t, k))
    if nL == 0:
        return ('feasible', base.copy()) if const < target else ('infeasible', None)
    tripf = {j for tr in trips for j in tr}
    xf = sorted(set(byf) | tripf)
    xi = {f: nP + nL + q for q, f in enumerate(xf)}
    n0 = nP + nL + len(xf)
    extra = []                                          # (lower, upper, integer) of auxiliary variables
    def var(l, u, integer):
        extra.append((l, u, integer)); return n0 + len(extra) - 1
    rows, cols, vals, lb, ub = [], [], [], [], []; r = 0
    def add(coefs, l, u):
        nonlocal r
        for cidx, v in coefs: rows.append(r); cols.append(cidx); vals.append(v)
        lb.append(l); ub.append(u); r += 1
    obj = {}; j = nP
    for leaves in leaf_terms:
        add([(j + q, 1.0) for q in range(len(leaves))], 1.0, 1.0)          # exactly one leaf per tree
        for v, cond in leaves:
            obj[j] = v
            for k, want in cond:                                            # leaf => its path conditions
                if want == 1: add([(j, 1.0), (k, -1.0)], -np.inf, 0.0)
                else: add([(j, 1.0), (k, 1.0)], -np.inf, 1.0)
            j += 1
    # evades the detector (row scaled to unit coefficients for numerical stability)
    sc_ = max(abs(v) for v in obj.values()) or 1.0
    add([(q, v / sc_) for q, v in obj.items() if v != 0], -np.inf, (target - const - 1e-9) / sc_)
    # feature values are modelled in normalised units u = (x - hl) / (hh - hl) in [0, 1]; integrality of integer
    # features is relaxed (this only enlarges the feasible set, so certificates stay valid; every evasion is
    # re-checked on the real model after the attacker's own projection, which rounds integer features)
    Lf = {f: hl[f] for f in xf}; Sf = {f: max(hh[f] - hl[f], 1e-12) for f in xf}
    nrm = lambda f, v: (v - Lf[f]) / Sf[f]
    for f, lst in byf.items():
        lst.sort()
        for (t1, k1), (t2, k2) in zip(lst[:-1], lst[1:]): add([(k1, 1.0), (k2, -1.0)], -np.inf, 0.0)
        for t, k in lst:                                                    # link predicate and value
            if sp.isint[f] and f not in tripf:   # integer feature: tighten to integers so rounding keeps the branch
                tt = float(np.float32(t)) if rule == 'lt' else float(t)
                up, dn = ((np.ceil(tt) - 1, np.ceil(tt)) if rule == 'lt' else (np.floor(tt), np.floor(tt) + 1))
            elif rule == 'le': up, dn = _f32_below(t), _f32_above(t)
            else: up, dn = float(np.nextafter(np.float32(t), np.float32(-np.inf))), float(np.float32(t))
            # clipping keeps the big-M of 2 valid; un < 0 (dnn > 1) correctly makes p=1 (p=0) infeasible
            un, dnn = min(max(nrm(f, up), -1.0), 1.0), min(max(nrm(f, dn), 0.0), 2.0)
            add([(xi[f], 1.0), (k, 2.0)], -np.inf, un + 2.0)                # p=1 => u <= up
            add([(xi[f], 1.0), (k, 2.0)], dnn, np.inf)                      # p=0 => u >= dn
    for a, b, c in trips:                                                   # exact sort of three boxed values
        src = (a, b, c)
        w = {(i, k): var(0, 1, True) for i in range(3) for k in range(3)}
        y = {(i, k): var(0.0, max(0.0, nrm(src[k], hi[src[k]])), False) for i in range(3) for k in range(3)}
        for i in range(3):
            add([(w[i, k], 1.0) for k in range(3)], 1.0, 1.0)
            add([(xi[src[i]], 1.0)] + [(y[i, k], -1.0) for k in range(3)], 0.0, 0.0)   # s_i = sum_k y_ik
            for k in range(3):
                add([(y[i, k], 1.0), (w[i, k], -nrm(src[k], lo[src[k]]))], 0.0, np.inf)   # y_ik >= lo_k w_ik
                add([(y[i, k], 1.0), (w[i, k], -nrm(src[k], hi[src[k]]))], -np.inf, 0.0)  # y_ik <= hi_k w_ik
        for k in range(3): add([(w[i, k], 1.0) for i in range(3)], 1.0, 1.0)
        add([(xi[a], 1.0), (xi[b], -1.0)], -np.inf, 0.0); add([(xi[b], 1.0), (xi[c], -1.0)], -np.inf, 0.0)
        Sx, Lx = Sf[a], Lf[a]
        if Sx <= 1e6:            # integer-valued sources stay integer after the sort (raw = L + S*y must be integral)
            for k in range(3):
                if not sp.isint[src[k]]: continue
                for i in range(3):
                    K = var(min(0.0, np.ceil(lo[src[k]])), max(0.0, np.floor(hi[src[k]])), True)
                    add([(y[i, k], Sx), (w[i, k], Lx), (K, -1.0)], 0.0, 0.0)
    n = n0 + len(extra)
    vlo = np.zeros(n); vhi = np.ones(n); integ = np.r_[np.ones(nP), np.zeros(n - nP)]
    for q, (l, u, it) in enumerate(extra):
        vlo[n0 + q], vhi[n0 + q] = l, u; integ[n0 + q] = 1 if it else 0
    A = coo_matrix((vals, (rows, cols)), shape=(r, n)).tocsr()
    if _debug is not None:
        _debug.update(A=A, lb=np.array(lb), ub=np.array(ub), vlo=vlo, vhi=vhi, integ=integ, preds=preds, leaf_terms=leaf_terms,
                      xi=xi, n0=n0, nP=nP, nL=nL, trips=trips, const=const, extra=extra, Lf=Lf, Sf=Sf)
    def solve(presolve, tl):
        return milp(np.zeros(n), integrality=integ, bounds=Bounds(vlo, vhi), constraints=LinearConstraint(A, lb, ub),
                    options={'time_limit': tl, 'disp': False, 'presolve': presolve})
    t0 = time.time(); res = solve(True, time_limit)
    if res.status == 2:   # confirm 'no evasion' without presolve before accepting it as a certificate
        res = solve(False, max(1.0, time_limit - (time.time() - t0)))
    if res.status == 2: return 'infeasible', None
    if res.x is None: return 'timeout', None
    x = base.copy()                                      # features without a variable keep their unmodified value
    for f in xf: x[f] = Lf[f] + Sf[f] * res.x[xi[f]]
    return 'feasible', x

def verify_attack(m, sp, x0, xa, lo, hi, eps, th):
    """Independent check of a MILP solution on the real model: undo the triplet sort (assign the sorted values
    to source features whose own interval contains them), round integer features, check every feature lies in
    the attacker's interval (the eps-box mapped to raw space), then apply the attacker's projection and score."""
    import itertools
    tol = lambda v: 1e-6 * max(1.0, abs(v))
    v = xa.copy()
    for a, b, c in sp.trip:
        if not sp.mask[[a, b, c]].all(): continue
        vals = xa[[a, b, c]]; src = (a, b, c); found = False
        for perm in itertools.permutations(range(3)):
            cand = vals.copy()
            for q in range(3):
                if sp.isint[src[perm[q]]]: cand[q] = np.round(cand[q])
            if all(lo[src[perm[q]]] - tol(lo[src[perm[q]]]) <= cand[q] <= hi[src[perm[q]]] + tol(hi[src[perm[q]]]) for q in range(3)):
                for q in range(3): v[src[perm[q]]] = cand[q]
                found = True; break
        if not found: return False
    v[sp.isint] = np.round(v[sp.isint])
    if not np.all((v >= lo - 1e-6 * np.maximum(1, np.abs(lo))) & (v <= hi + 1e-6 * np.maximum(1, np.abs(hi)))): return False
    return bool(m.score(sp.project(v[None].copy()))[0] < th)

def milp_main():
    t0 = time.time()
    log('MILP device', DEV, '| args', vars(args))
    df, feats = load()
    fams = [f for f in ['DoS', 'DDoS', 'PortScan', 'BruteForce', 'Botnet', 'WebAttack'] if (df.family == f).sum() >= 50]
    OUT = os.path.join(args.out, 'results_milp.csv'); rows = []
    def put(**r):
        rows.append(r); pd.DataFrame(rows).to_csv(OUT, index=False)
    for seed in [int(s) for s in args.seeds.split(',')]:
        rng = np.random.default_rng(seed)
        TR, TE = split_pools(df, seed)
        btr = sample(TR, 'BENIGN', args.benign_train, rng)
        bva, btr = btr.iloc[:len(btr) // 5], btr.iloc[len(btr) // 5:]
        atr = {f: sample(TR, f, args.fam_train, rng) for f in fams}
        ate = {f: sample(TE, f, args.fam_test, rng) for f in fams}
        X = lambda d: d[feats].to_numpy(np.float64)
        dtr = pd.concat([btr] + [atr[f] for f in fams]); sp = Space(X(dtr), feats)
        Xtr_, ytr_ = X(dtr), (dtr.family != 'BENIGN').to_numpy(int)
        sub = {f: rng.permutation(len(ate[f]))[:args.milp_n] for f in fams}
        for kind in [k for k in args.milp_models.split(',') if k]:
            m = Model(kind, sp, Xtr_, ytr_, seed); th = thr_at_fpr(m.score(X(bva)))
            trees, rule = tree_arrays(m)
            # leaf-sum must reproduce the model's own output (checked on 200 flows)
            Xchk = X(ate[fams[0]])[:200]
            own = np.array([tree_sum(trees, rule, x) for x in Xchk])
            ref = m.score(Xchk)
            if kind == 'xgb':
                margin = m.m.get_booster().predict(xgb.DMatrix(Xchk), output_margin=True)
                off = float(np.median(margin - own)); err = float(np.max(np.abs(margin - own - off)))
                target = (np.log(th / (1 - th)) - off) if 0 < th < 1 else (-np.inf if th <= 0 else np.inf)
            else:
                off, err, target = 0.0, float(np.max(np.abs(own - ref))), th
            log(f'seed {seed} {kind}: {len(trees)} trees, leaf-sum check max error {err:.2e}, threshold {th:.6f}')
            assert err < 1e-4, 'tree export does not reproduce the model output'
            for e in [float(v) for v in args.milp_eps.split(',')]:
                for f in fams:
                    Xs = X(ate[f])[sub[f]]
                    det0 = m.score(sp.project(Xs.copy())) >= th
                    cnt = {'certified': 0, 'evaded': 0, 'undecided': 0, 'missed_clean': int((~det0).sum())}
                    tsolve = []; st_log = {}
                    for i in np.flatnonzero(det0):
                        t1 = time.time()
                        lo, hi, base = attack_box(sp, Xs[i], e)
                        st, xa = milp_attack(trees, rule, lo, hi, target, args.milp_time, sp, base)
                        st_log[st] = st_log.get(st, 0) + 1
                        if st == 'infeasible': cnt['certified'] += 1
                        elif st == 'timeout': cnt['undecided'] += 1
                        else:
                            ok = verify_attack(m, sp, Xs[i], xa, lo, hi, e, th)
                            cnt['evaded' if ok else 'undecided'] += 1
                            if not ok: st_log['unverified'] = st_log.get('unverified', 0) + 1
                        tsolve.append(time.time() - t1)
                    nd = int(det0.sum())
                    put(seed=seed, model=kind, family=f, epsilon=e, n_eval=len(Xs), n_detected_clean=nd,
                        dr_clean=float(det0.mean()), certified=cnt['certified'], evaded=cnt['evaded'], undecided=cnt['undecided'],
                        dr_certified_lower=float(cnt['certified'] / len(Xs)),
                        dr_upper=float((nd - cnt['evaded']) / len(Xs)),
                        solve_s_median=float(np.median(tsolve)) if tsolve else np.nan,
                        solve_s_max=float(np.max(tsolve)) if tsolve else np.nan)
                    log(f'seed {seed} {kind} eps {e} {f}: {cnt} {st_log} | median solve {np.median(tsolve) if tsolve else 0:.2f}s')
        log(f'seed {seed} milp complete | {(time.time() - t0) / 60:.1f} min')
    log('MILP DONE rows', len(rows), 'md5', hashlib.md5(open(OUT, 'rb').read()).hexdigest())

# ------------------------------------------------------------------ open-set baselines (review round 4)
def openset_main():
    """LOAO with multi-class detectors (benign + each known family as its own class). Two alert scores, both
    thresholded at the same 1% FPR on benign validation flows:
      mc_notbenign : 1 - P(benign)
      mc_msp       : 1 if the predicted class is an attack, else 1 - max-class probability (low-confidence
                     'benign' predictions are treated as unknown: maximum-softmax-probability rejection)."""
    t0 = time.time()
    log('OPENSET device', DEV, '| args', vars(args))
    df, feats = load()
    fams = [f for f in ['DoS', 'DDoS', 'PortScan', 'BruteForce', 'Botnet', 'WebAttack'] if (df.family == f).sum() >= 50]
    OUT = os.path.join(args.out, 'results_openset.csv'); rows = []
    def put(**r):
        rows.append(r); pd.DataFrame(rows).to_csv(OUT, index=False)
    for seed in [int(s) for s in args.seeds.split(',')]:
        rng = np.random.default_rng(seed)
        TR, TE = split_pools(df, seed)
        btr = sample(TR, 'BENIGN', args.benign_train, rng)
        bva, btr = btr.iloc[:len(btr) // 5], btr.iloc[len(btr) // 5:]
        bte = sample(TE, 'BENIGN', args.benign_test, rng)
        atr = {f: sample(TR, f, args.fam_train, rng) for f in fams}
        ate = {f: sample(TE, f, args.fam_test, rng) for f in fams}
        X = lambda d: d[feats].to_numpy(np.float64)
        Xb, Xva = X(bte), X(bva)
        for held in fams:
            known = [f for f in fams if f != held]
            dtr = pd.concat([btr] + [atr[f] for f in known])
            cls = {'BENIGN': 0, **{f: i + 1 for i, f in enumerate(known)}}
            ytr = dtr.family.map(cls).to_numpy(int); Xtr = X(dtr)
            w = len(ytr) / (len(cls) * np.bincount(ytr, minlength=len(cls)))[ytr]
            for kind in ['rf', 'xgb']:
                t = time.time()
                if kind == 'rf':
                    m = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, class_weight='balanced', n_jobs=-1, random_state=seed).fit(Xtr, ytr)
                else:
                    m = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                                          tree_method='hist', device='cuda' if DEV == 'cuda' else 'cpu', objective='multi:softprob',
                                          random_state=seed, eval_metric='mlogloss').fit(Xtr, ytr, sample_weight=w)
                def scores(Xs):
                    P = m.predict_proba(Xs)
                    nb = 1 - P[:, 0]
                    msp = np.where(P.argmax(1) != 0, 1.0, 1 - P.max(1))
                    return {'mc_notbenign': nb, 'mc_msp': msp}
                sva, sb, sh = scores(Xva), scores(Xb), scores(X(ate[held]))
                for v in sva:
                    th = thr_at_fpr(sva[v])
                    put(seed=seed, model=kind, variant=v, family=held, n_eval=len(sh[v]),
                        detection_rate=float((sh[v] >= th).mean()), fpr=float((sb[v] >= th).mean()),
                        auroc=float(roc_auc_score(np.r_[np.zeros(len(sb[v])), np.ones(len(sh[v]))], np.r_[sb[v], sh[v]])))
                log(f'seed {seed} held {held} {kind} done in {time.time() - t:.0f}s', {v: round(rows[-2 + i]['detection_rate'], 3) for i, v in enumerate(sva)})
        log(f'seed {seed} openset complete | {(time.time() - t0) / 60:.1f} min')
    log('OPENSET DONE rows', len(rows), 'md5', hashlib.md5(open(OUT, 'rb').read()).hexdigest())

if __name__ == '__main__':
    openset_main() if args.openset else milp_main() if args.milp else (supp2_main() if args.supp2 else (supp_main() if args.supp else main()))
