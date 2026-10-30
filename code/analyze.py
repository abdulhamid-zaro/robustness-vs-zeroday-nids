#!/usr/bin/env python3
"""Aggregate results across seeds: mean, std, 95% t-CI; paired tests; figures; LaTeX tables."""
import sys, os, json
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = sys.argv[1] if len(sys.argv) > 1 else '/content/results/results.csv'
OUT = sys.argv[2] if len(sys.argv) > 2 else '/content/analysis'
os.makedirs(OUT, exist_ok=True)
r = pd.read_csv(RES)
T2 = pd.read_csv(sys.argv[3]) if len(sys.argv) > 3 and os.path.exists(sys.argv[3]) else None   # tree-aware attack results (supp2)
MODELS = ['rf', 'xgb', 'cnn', 'ft']
NAMES = {'rf': 'RF', 'xgb': 'XGBoost', 'cnn': '1D-CNN', 'ft': 'FT-Transformer'}
COLORS = {'rf': '#1f77b4', 'xgb': '#ff7f0e', 'cnn': '#2ca02c', 'ft': '#d62728'}
MARK = {'rf': 'o', 'xgb': 's', 'cnn': '^', 'ft': 'D'}

def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = len(x)
    if n == 0: return np.nan, np.nan, np.nan, 0
    m = x.mean(); s = x.std(ddof=1) if n > 1 else 0.0
    h = stats.t.ppf(0.975, n - 1) * s / np.sqrt(n) if n > 1 else np.nan
    return m, s, h, n

def agg(df, keys, val):
    rows = []
    for k, g in df.groupby(keys):
        m, s, h, n = ci(g[val]); k = k if isinstance(k, tuple) else (k,)
        rows.append(dict(zip(keys, k), mean=m, sd=s, ci95=h, n_seeds=n))
    return pd.DataFrame(rows)

summary = {}
# 1) in-distribution
ind = r[(r.scenario == 'in_dist') & (r.family == 'ALL')]
t_in = pd.concat([agg(ind, ['model'], v).assign(metric=v) for v in ['detection_rate', 'auroc', 'f1', 'fpr']])
t_in.to_csv(f'{OUT}/in_dist.csv', index=False)
# 2) perturbation scenarios (ALL families)
pert = r[(r.family == 'ALL') & r.scenario.isin(['noise', 'fgsm_transfer', 'pgd_transfer', 'pgd_whitebox', 'simba_blackbox'])].copy()
pert['drop'] = pert['dr_clean'] - pert['detection_rate']
t_pert = agg(pert, ['scenario', 'model', 'epsilon'], 'detection_rate')
t_drop = agg(pert, ['scenario', 'model', 'epsilon'], 'drop')
t_pert.to_csv(f'{OUT}/perturbation_dr.csv', index=False); t_drop.to_csv(f'{OUT}/perturbation_drop.csv', index=False)
# fraction of seed-level McNemar tests significant after Holm correction within each (scenario, model)
mc = r[(r.family == 'ALL') & r.mcnemar_p.notna() & r.scenario.isin(['noise', 'fgsm_transfer', 'pgd_transfer', 'pgd_whitebox', 'simba_blackbox'])].copy()
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m)
    run = 0
    for rank, i in enumerate(o):
        run = max(run, min(1.0, (m - rank) * p[i])); adj[i] = run
    return adj
mc['p_holm'] = holm(mc['mcnemar_p'].values)
mc.assign(sig=mc.p_holm < 0.05).groupby(['scenario', 'model', 'epsilon']).agg(sig_seeds=('sig', 'sum'), n=('sig', 'size')).reset_index().to_csv(f'{OUT}/mcnemar_holm.csv', index=False)
# 3) LOAO
lo = r[r.scenario == 'loao']
t_lo = pd.concat([agg(lo, ['family', 'model'], v).assign(metric=v) for v in ['detection_rate', 'auroc', 'fpr']])
t_lo.to_csv(f'{OUT}/loao.csv', index=False)
# 4) mixup
mx = r[r.scenario == 'mixup']
t_mx = pd.concat([agg(mx, ['model'], v).assign(metric=v) for v in ['detection_rate', 'dr_clean']])
t_mx.to_csv(f'{OUT}/mixup.csv', index=False)

# 5) key comparison: worst-case perturbation robustness (min DR over attacks at eps<=0.2) vs mean LOAO DR, paired by seed
rows = []
for (seed, model), g in r.groupby(['seed', 'model']):
    pg = g[(g.family == 'ALL') & g.scenario.isin(['fgsm_transfer', 'pgd_transfer', 'pgd_whitebox', 'simba_blackbox']) & (g.epsilon <= 0.2)]
    lg = g[g.scenario == 'loao']
    if len(pg) and len(lg):
        rmin = pg.detection_rate.min()
        if T2 is not None:
            q = T2[(T2.seed == seed) & (T2.model == model) & (T2.epsilon <= 0.2)]
            if len(q): rmin = min(rmin, q.groupby('epsilon').apply(lambda d: np.average(d.dr_tree, weights=d.n_eval)).min())
        rows.append({'seed': seed, 'model': model, 'robust_dr_min_eps02': rmin, 'loao_dr_mean': lg.detection_rate.mean(), 'loao_auroc_mean': lg.auroc.mean()})
gap = pd.DataFrame(rows); gap['gap'] = gap.robust_dr_min_eps02 - gap.loao_dr_mean
gap.to_csv(f'{OUT}/robustness_vs_zeroday_by_seed.csv', index=False)
tests = {}
for m in MODELS:
    g = gap[gap.model == m]
    if len(g) >= 2:
        w = stats.wilcoxon(g.robust_dr_min_eps02, g.loao_dr_mean) if len(g) >= 5 else None
        tt = stats.ttest_rel(g.robust_dr_min_eps02, g.loao_dr_mean)
        mm, s, h, n = ci(g.gap)
        tests[m] = {'gap_mean': mm, 'gap_ci95': h, 'n_seeds': n, 'paired_t_p': float(tt.pvalue),
                    'wilcoxon_p': (float(w.pvalue) if w is not None else None)}
# pooled over models x seeds (n = 4 x seeds)
if len(gap) >= 6:
    tests['pooled_wilcoxon_p'] = float(stats.wilcoxon(gap.robust_dr_min_eps02, gap.loao_dr_mean).pvalue)
    tests['pooled_n'] = int(len(gap))
rho = stats.spearmanr(gap.robust_dr_min_eps02, gap.loao_dr_mean)
tests['spearman_rho'] = float(rho.statistic); tests['spearman_p'] = float(rho.pvalue)
mm = gap.groupby('model')[['robust_dr_min_eps02', 'loao_dr_mean']].mean()
tests['model_rank_robust'] = list(mm.robust_dr_min_eps02.sort_values(ascending=False).index)
tests['model_rank_loao'] = list(mm.loao_dr_mean.sort_values(ascending=False).index)
summary['robustness_vs_zeroday'] = tests

# 6) AUROC vs operating point in LOAO (shows threshold matters)
summary['loao_auroc_vs_dr'] = lo.groupby('model')[['auroc', 'detection_rate']].mean().round(4).to_dict()
json.dump(summary, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)

# ---------------- figures
fams = ['DoS', 'DDoS', 'PortScan', 'BruteForce', 'Botnet', 'WebAttack']
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, sc, title in zip(axes, ['pgd_transfer', 'pgd_whitebox', 'simba_blackbox'], ['PGD (transfer from surrogate)', 'PGD (white-box, neural only)', 'SimBA (black-box, all models)']):
    for m in MODELS:
        d = t_pert[(t_pert.scenario == sc) & (t_pert.model == m)].sort_values('epsilon')
        if len(d):
            ax.errorbar(d.epsilon, d['mean'], yerr=d.ci95, marker=MARK[m], color=COLORS[m], capsize=3, label=NAMES[m])
    ax.set_title(title, fontsize=9); ax.set_xlabel(r'$\epsilon$ (std. log-space)'); ax.grid(alpha=.3)
axes[0].set_ylabel('Detection rate @ 1% FPR')
from matplotlib.lines import Line2D
fig.legend([Line2D([0], [0], color=COLORS[m], marker=MARK[m]) for m in MODELS], [NAMES[m] for m in MODELS], loc='lower center', ncol=4, fontsize=8, frameon=False)
plt.subplots_adjust(bottom=0.25)
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(f'{OUT}/fig_perturbation.pdf'); plt.savefig(f'{OUT}/fig_perturbation.png', dpi=200); plt.close()

piv = lo.groupby(['family', 'model']).detection_rate.mean().unstack()[MODELS].reindex(fams)
fig, ax = plt.subplots(figsize=(5.0, 3.8))
im = ax.imshow(piv.values, vmin=0, vmax=1, cmap='viridis')
ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([NAMES[m] for m in MODELS], fontsize=8, rotation=30, ha='right')
ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]; ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7, color='white' if v < .6 else 'black')
plt.colorbar(im, ax=ax, label='Detection rate @ 1% FPR'); ax.set_title('Held-out (unseen) attack family', fontsize=9)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_loao.pdf'); plt.savefig(f'{OUT}/fig_loao.png', dpi=200); plt.close()

# ---------------- LaTeX tables
def fmt(m, h): return '--' if np.isnan(m) else (f'{m:.3f}' + (f' $\\pm$ {min(h, 1.0):.3f}' if not np.isnan(h) else ''))
L = []
L.append('\\begin{table}[t]\\centering\\caption{Detection rate at 1\\% FPR on held-out (unseen) attack families, mean $\\pm$ 95\\% CI over seeds.}\\label{tab:loao}')
L.append('\\resizebox{\\columnwidth}{!}{\\begin{tabular}{l' + 'c' * len(MODELS) + '}\\toprule')
L.append('Held-out family & ' + ' & '.join(NAMES[m] for m in MODELS) + ' \\\\\\midrule')
d = t_lo[t_lo.metric == 'detection_rate']
for f in fams:
    L.append(f + ' & ' + ' & '.join(fmt(*d[(d.family == f) & (d.model == m)][['mean', 'ci95']].values[0]) if len(d[(d.family == f) & (d.model == m)]) else '--' for m in MODELS) + ' \\\\')
L.append('\\bottomrule\\end{tabular}}\\end{table}')
L.append('')
L.append('\\begin{table}[t]\\centering\\caption{Detection rate at 1\\% FPR under constrained perturbations ($\\epsilon$ in standardized log space), mean $\\pm$ 95\\% CI over seeds.}\\label{tab:pert}')
L.append('\\resizebox{\\columnwidth}{!}{\\begin{tabular}{ll' + 'c' * len(MODELS) + '}\\toprule')
L.append('Attack & $\\epsilon$ & ' + ' & '.join(NAMES[m] for m in MODELS) + ' \\\\\\midrule')
for sc in ['noise', 'pgd_transfer', 'pgd_whitebox', 'simba_blackbox']:
    for e in sorted(t_pert[t_pert.scenario == sc].epsilon.unique()):
        cells = []
        for m in MODELS:
            q = t_pert[(t_pert.scenario == sc) & (t_pert.model == m) & (t_pert.epsilon == e)]
            cells.append(fmt(q['mean'].values[0], q['ci95'].values[0]) if len(q) else '--')
        L.append(sc.replace('_', '\\_') + f' & {e:g} & ' + ' & '.join(cells) + ' \\\\')
L.append('\\bottomrule\\end{tabular}}\\end{table}')
open(f'{OUT}/tables.tex', 'w').write('\n'.join(L))
print(json.dumps(summary, indent=1, default=float))
print(open(f'{OUT}/tables.tex').read()[:3000])
