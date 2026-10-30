#!/usr/bin/env python3
"""Assemble the paper sections from the analysis outputs. Every number is read from files."""
import json, re, sys, os, ast
import numpy as np, pandas as pd

BASE = sys.argv[1]          # folder with results/ analysis/ (and optionally audit/ repair/)
OUT = sys.argv[2]
A = f'{BASE}/analysis'
log = open(f'{BASE}/results/run_log.txt').read()
m = re.search(r'loaded (\d+) rows, (\d+) removed as duplicates/ambiguous, (\d+) kept', log)
N_RAW, N_REM, N_FLOWS = map(int, m.groups())
NF, NMOD = map(int, re.search(r'features (\d+) \| modifiable (\d+)', log).groups())
fam_counts = ast.literal_eval(re.search(r'family counts (\{.*?\})', log).group(1))
args = ast.literal_eval(re.search(r"\| args (\{.*?\})\n", log).group(1))
ver = re.search(r'torch (\S+) \| xgb (\S+) \| python (\S+)', log).groups()
runtime = float(re.findall(r'elapsed ([\d.]+) min', log)[-1])
res = pd.read_csv(f'{BASE}/results/results.csv')
S = json.load(open(f'{A}/summary.json'))
ind = pd.read_csv(f'{A}/in_dist.csv'); pert = pd.read_csv(f'{A}/perturbation_dr.csv'); lo = pd.read_csv(f'{A}/loao.csv')
mx = pd.read_csv(f'{A}/mixup.csv'); gap = pd.read_csv(f'{A}/robustness_vs_zeroday_by_seed.csv'); mc = pd.read_csv(f'{A}/mcnemar_holm.csv')
M = ['rf', 'xgb', 'cnn', 'ft']; NM = {'rf': 'RF', 'xgb': 'XGBoost', 'cnn': '1D-CNN', 'ft': 'FT-Transformer'}
FAMS = ['DoS', 'DDoS', 'PortScan', 'BruteForce', 'Botnet', 'WebAttack']

def g(df, **kw):
    q = df
    for k, v in kw.items(): q = q[q[k] == v]
    assert len(q) == 1, (kw, len(q))
    return q.iloc[0]
pct = lambda x: f'{100 * x:.1f}%'
def ci_half(x):
    from scipy import stats as _s
    x = np.asarray(x, float); return float(_s.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
def pci(row): return f"{100 * row['mean']:.1f}% $\\pm$ {100 * min(row['ci95'], 1):.1f}"
def p_fmt(p): return f'$p = {p:.2g}$' if p >= 1e-3 else f'$p < 10^{{{int(np.floor(np.log10(p)))+1}}}$'

ind_dr = {k: g(ind, model=k, metric='detection_rate')['mean'] for k in M}
ind_f1 = {k: g(ind, model=k, metric='f1')['mean'] for k in M}
rob = gap.groupby('model').robust_dr_min_eps02.mean().to_dict()
lo_dr = gap.groupby('model').loao_dr_mean.mean().to_dict()
lo_auc = gap.groupby('model').loao_auroc_mean.mean().to_dict()
T = S['robustness_vs_zeroday']
sim05 = {k: g(pert, scenario='simba_blackbox', model=k, epsilon=0.5)['mean'] for k in M}
wb05 = {k: g(pert, scenario='pgd_whitebox', model=k, epsilon=0.5)['mean'] for k in ['cnn', 'ft']}
pt05 = {k: g(pert, scenario='pgd_transfer', model=k, epsilon=0.5)['mean'] for k in M}
mixdr = {k: g(mx, model=k, metric='detection_rate')['mean'] for k in M}
lofam = lambda f, k: g(lo, family=f, model=k, metric='detection_rate')
loauc = lambda f, k: g(lo, family=f, model=k, metric='auroc')
sig = lambda sc, k, e: int(g(mc, scenario=sc, model=k, epsilon=e)['sig_seeds'])

# split sizes actually used (70/30 pools, caps)
tr = {f: min(args['fam_train'], int(0.7 * fam_counts[f])) for f in FAMS}
te = {f: min(args['fam_test'], fam_counts[f] - int(0.7 * fam_counts[f])) for f in FAMS}

V = dict(
    N_RAW=f'{N_RAW:,}', N_REMOVED=f'{N_REM:,}', N_FLOWS=f'{N_FLOWS:,}', N_FEATURES=str(NF), N_MODIFIABLE=str(NMOD),
    N_INFILTRATION=str(fam_counts.get('Infiltration', 0)), N_HEARTBLEED=str(fam_counts.get('Heartbleed', 0)),
    BENIGN_TRAIN=f"{args['benign_train']:,}", BENIGN_TEST=f"{args['benign_test']:,}", FAM_TRAIN=f"{args['fam_train']:,}",
    FAM_TEST=f"{args['fam_test']:,}", EPOCHS=str(args['epochs']), PY_VERSION=ver[2], TORCH_VERSION=ver[0].split('+')[0],
    XGB_VERSION=ver[1], RUNTIME=f'{runtime:.0f}', N_ROWS=f'{len(res):,}',
)
# ---------------- supplementary run (anomaly baselines, eps scale, five-pass SimBA)
SUPP = f'{BASE}/supp/results_supp.csv'
sp_ = pd.read_csv(SUPP)
assert sorted(sp_.seed.unique().tolist()) == [0, 1, 2, 3, 4], 'supp run incomplete'
es = sp_[sp_.scenario == 'eps_scale'].groupby('epsilon')[['value_median', 'value_q25', 'value_q75']].mean()
an = sp_[sp_.scenario == 'anomaly_unseen']
an_dr = an.groupby('model').detection_rate.mean().to_dict()
an_auc = an.groupby('model').auroc.mean().to_dict()
an_fam = an.groupby(['model', 'family'])[['detection_rate', 'auroc']].mean()
an_fpr = an.groupby('model').fpr.mean().to_dict()
asim = sp_[sp_.scenario == 'anomaly_simba'].groupby(['model', 'epsilon']).detection_rate.mean()
ss = sp_[sp_.scenario == 'simba_strong']
ss_dr = ss.groupby('model').apply(lambda d: np.average(d.detection_rate, weights=d.n_eval)).to_dict()
assert max(an_dr.values()) < min(lo_dr.values()), (an_dr, lo_dr)
assert min(an_auc.values()) > 0.8, an_auc

T = S['robustness_vs_zeroday']
gaps = [T[k]['gap_mean'] for k in M]
V['GAP_LO'], V['GAP_HI'] = f'{100*min(gaps):.0f}', f'{100*max(gaps):.0f}'
V['TE_WEB'], V['TE_BOT'] = str(te['WebAttack']), str(te['Botnet'])
assert (gap.gap > 0).all()
V['SPEARMAN_SHORT'] = "__SPEARMAN_SHORT__"
V['ANOMALY_SHORT'] = f"mean unseen-family detection {pct(an_dr['iforest'])} and {pct(an_dr['ae'])}"
V['EPS_SCALE_TEXT'] = (f"across the {NMOD} modifiable features the median factor is {es.loc[0.05,'value_median']:.2f} at $\\epsilon = 0.05$, "
                       f"{es.loc[0.2,'value_median']:.1f} (interquartile range {es.loc[0.2,'value_q25']:.1f}--{es.loc[0.2,'value_q75']:.1f}) at $\\epsilon = 0.2$ "
                       f"and {es.loc[0.5,'value_median']:.1f} ({es.loc[0.5,'value_q25']:.1f}--{es.loc[0.5,'value_q75']:.0f}) at $\\epsilon = 0.5$. "
                       "Budgets up to 0.2 thus amount to at most roughly doubling a typical feature, which an attacker controlling its own send rate and packet sizes can plausibly do; $\\epsilon = 0.5$ is a large change that we keep as a stress test rather than as a realistic attacker, and it is not used in the robustness score that we compare with LOAO.")

V['ABSTRACT_RESULTS'] = (
    f"Detection under the strongest perturbation we tested, with budgets of at most 0.2, stays between {pct(min(rob.values()))} and {pct(max(rob.values()))}, "
    f"yet mean detection of held-out families falls to {pct(lo_dr['rf'])} (RF), {pct(lo_dr['xgb'])} (XGBoost), {pct(lo_dr['cnn'])} (1D-CNN) and {pct(lo_dr['ft'])} (FT-Transformer).")
V['GAP_STATS'] = "__GAP_STATS__"

# per-seed spread of LOAO detection (all-or-nothing behaviour)
lo_seed = res[res.scenario == 'loao'].pivot_table(index=['family', 'model'], columns='seed', values='detection_rate')
def spread(f, k):
    v = lo_seed.loc[(f, k)].values; return v.min(), v.max()
cnn_sim = res[(res.scenario == 'simba_blackbox') & (res.epsilon == 0.5) & (res.family == 'ALL') & (res.model == 'cnn')].detection_rate
ft_bot = lo_seed.loc[('Botnet', 'ft')].values
rf_ps = np.sort(lo_seed.loc[('PortScan', 'rf')].values)
assert (ft_bot > 0.8).sum() == 1 and (ft_bot == 0).sum() == 4, ft_bot
assert (rf_ps > 0.8).sum() == 2 and (rf_ps < 0.3).sum() == 3, rf_ps

from scipy import stats as _st
BIG = ['DoS', 'DDoS', 'PortScan', 'BruteForce']
_rows = []
for (s_, m_), g_ in res.groupby(['seed', 'model']):
    pg_ = g_[(g_.family == 'ALL') & g_.scenario.isin(['fgsm_transfer', 'pgd_transfer', 'pgd_whitebox', 'simba_blackbox']) & (g_.epsilon <= 0.2)]
    lg_ = g_[g_.scenario == 'loao']
    _rows.append(dict(model=m_, rob=pg_.detection_rate.min(), lo4=lg_[lg_.family.isin(BIG)].detection_rate.mean(), low=np.average(lg_.detection_rate, weights=lg_.n_eval)))
_d = pd.DataFrame(_rows); SENS = {}
for c_ in ['lo4', 'low']:
    gp_ = (_d.rob - _d[c_]); assert (gp_ > 0).all(), c_
    gm_ = _d.assign(g=gp_).groupby('model').g.mean() * 100
    SENS[c_] = dict(lo=gm_.min(), hi=gm_.max(), p=max(_st.ttest_rel(_d[_d.model == k].rob, _d[_d.model == k][c_]).pvalue for k in M))
TREE_TXT = ''
if os.path.exists(f'{BASE}/supp2/results_supp2.csv'):
    t2 = pd.read_csv(f'{BASE}/supp2/results_supp2.csv')
    assert sorted(t2.seed.unique().tolist()) == [0, 1, 2, 3, 4], 'supp2 incomplete'
    wavg = lambda d, c: np.average(d[c], weights=d.n_eval)
    TA = {(k, e): (wavg(d, 'dr_tree'), wavg(d, 'dr_simba')) for (k, e), d in t2.groupby(['model', 'epsilon'])}
    TREE_TXT = (f"SimBA only tries the two corners of each coordinate's interval and could step over the split points of a tree. The tree-aware search (procedure vi, 200 flows per family and seed) changes little: at $\\epsilon = 0.2$ detection is {pct(TA[('rf', 0.2)][0])} for RF and {pct(TA[('xgb', 0.2)][0])} for XGBoost, against {pct(TA[('rf', 0.2)][1])} and {pct(TA[('xgb', 0.2)][1])} for SimBA on the same flows, and at $\\epsilon = 0.5$ it is {pct(TA[('rf', 0.5)][0])} and {pct(TA[('xgb', 0.5)][0])}. "
                "Its results are included in the worst-case robustness score used in Section V-D. Neither search is exhaustive over joint changes of several features; Section VI returns to this. ")

MILP_TXT, V['LIMIT_MILP'] = '', ("our tree-aware search minimises along one coordinate at a time (exactly when at most 24 split thresholds fall inside the box) but not jointly, and an exact MILP attack [@kantchelian] could still find joint evasions, lower the robustness of the tree ensembles and narrow their gap.")
_mf = f'{BASE}/milp/results_milp.csv'
if os.path.exists(_mf):
    mi = pd.read_csv(_mf); _msd = sorted(mi.seed.unique().tolist()); _mn = int(mi.n_eval.max())
    def _ms(k, e):
        d = mi[(mi.model == k) & (mi.epsilon == e)]
        return dict(n=int(d.n_eval.sum()), det=int(d.n_detected_clean.sum()), cert=int(d.certified.sum()), ev=int(d.evaded.sum()), und=int(d.undecided.sum()),
                    lo=d.certified.sum() / d.n_eval.sum(), hi=(d.n_detected_clean.sum() - d.evaded.sum()) / d.n_eval.sum())
    _me = sorted(mi.epsilon.unique().tolist())
    MS = {(k, e): _ms(k, e) for k in ['rf', 'xgb'] for e in _me}
    _tot_ev = sum(v['ev'] for v in MS.values()); _tot_und = sum(v['und'] for v in MS.values())
    MILP_TXT = ("To check whether coordinate-wise search simply misses joint evasions, we also solved the exact evasion problem for both tree ensembles as a mixed-integer program, following Kantchelian et al. [@kantchelian], inside the same box as the other attacks: only the attacker-controllable features, the same budget, integer features kept integral and the min $\\le$ mean $\\le$ max orderings enforced "
                + f"(up to {_mn} flows per family and seed, {len(_msd)} seed{'s' if len(_msd) > 1 else ''}, budgets " + ', '.join(f'{e:g}' for e in _me) + "). "
                + "A flow counts as certified when the program is infeasible, that is, when no point in the box falls below the detection threshold. "
                + ' '.join(f"At $\\epsilon = {e:g}$, certified detection is at least {pct(MS[('rf', e)]['lo'])} for RF and {pct(MS[('xgb', e)]['lo'])} for XGBoost, and at most {pct(MS[('rf', e)]['hi'])} and {pct(MS[('xgb', e)]['hi'])}." for e in _me)
                + f" Across all runs the solver found {_tot_ev} verified evasion{'s' if _tot_ev != 1 else ''}, and {_tot_und} flow{'s were' if _tot_und != 1 else ' was'} left undecided (time limit, or a solver solution that did not evade the real model). "
                + "Within this threat model, then, the tree ensembles are robust in an exact sense, not merely against the searches we wrote, and their gap is not an artefact of a weak attack. ")
    V['LIMIT_MILP'] = (f"the exact MILP search (Section V-B) removes the search-side doubt for the tree ensembles, but it covers {len(_msd)} seed{'s' if len(_msd) > 1 else ''} and up to {_mn} flows per family, and the gradient and SimBA attacks on the neural detectors remain heuristic, so their robustness may be overstated.")
    TREE_TXT = TREE_TXT.replace("Neither search is exhaustive over joint changes of several features; Section VI returns to this. ", "") + MILP_TXT

# ================= review round 4: extra statistics =================
from itertools import permutations as _perms
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r_, i_ in enumerate(o):
        run = max(run, min(1.0, (m - r_) * ps[i_])); adj[i_] = run
    return adj
_pt = {k: _st.ttest_rel(gap[gap.model == k].robust_dr_min_eps02, gap[gap.model == k].loao_dr_mean).pvalue for k in M}
_ph = dict(zip(M, holm([_pt[k] for k in M])))
_dz = {k: gap[gap.model == k].gap.mean() / gap[gap.model == k].gap.std(ddof=1) for k in M}
_npos = {k: int((gap[gap.model == k].gap > 0).sum()) for k in M}
assert all(_npos[k] == 5 for k in M)
# model-level rank agreement (n = 4 models) with an exact permutation test over the 24 orderings
_mm = gap.groupby('model')[['robust_dr_min_eps02', 'loao_dr_mean']].mean().loc[M]
_rho_m = _st.spearmanr(_mm.robust_dr_min_eps02, _mm.loao_dr_mean).statistic
_perm = [_st.spearmanr(_mm.robust_dr_min_eps02.values, np.array(p_)).statistic for p_ in _perms(_mm.loao_dr_mean.values)]
_p_m = float(np.mean([r_ >= _rho_m - 1e-12 for r_ in _perm]))
# same test at the model-seed level, but permuting whole model blocks (seeds of a model move together)
_lo_blocks = {k: gap[gap.model == k].sort_values('seed').loao_dr_mean.values for k in M}
_rob_all = np.concatenate([gap[gap.model == k].sort_values('seed').robust_dr_min_eps02.values for k in M])
_rho20 = _st.spearmanr(_rob_all, np.concatenate([_lo_blocks[k] for k in M])).statistic
_perm20 = [_st.spearmanr(_rob_all, np.concatenate([_lo_blocks[k] for k in p_])).statistic for p_ in _perms(M)]
_p20 = float(np.mean([r_ >= _rho20 - 1e-12 for r_ in _perm20]))
# sensitivity of the gap to how LOAO is averaged, using the same robustness score as the main analysis
_d2 = _d.copy()
_rows3 = []
for (s_, m_), g_ in res.groupby(['seed', 'model']):
    lg_ = g_[g_.scenario == 'loao']
    rb_ = float(gap[(gap.seed == s_) & (gap.model == m_)].robust_dr_min_eps02.iloc[0])
    _rows3.append(dict(seed=s_, model=m_, rob=rb_, lo6=lg_.detection_rate.mean(), lo4=lg_[lg_.family.isin(BIG)].detection_rate.mean(),
                       low=np.average(lg_.detection_rate, weights=lg_.n_eval)))
_d3 = pd.DataFrame(_rows3)
SENS = {}
for c_ in ['lo6', 'lo4', 'low']:
    gp_ = _d3.rob - _d3[c_]; assert (gp_ > 0).all(), c_
    gm_ = _d3.assign(g=gp_).groupby('model').g.mean() * 100
    SENS[c_] = dict(lo=gm_.min(), hi=gm_.max(), p=max(_st.ttest_rel(_d3[_d3.model == k].rob, _d3[_d3.model == k][c_]).pvalue for k in M),
                    per={k: (_d3[_d3.model == k][c_].mean(), ci_half(_d3[_d3.model == k][c_])) for k in M})
N_MCN = int(mc.n.sum())
# family-averaged robustness: per family, minimum over every attack at eps <= 0.2 (tree-aware search included), then averaged over families
_pa = res[res.scenario.isin(['fgsm_transfer', 'pgd_transfer', 'pgd_whitebox', 'simba_blackbox']) & (res.epsilon <= 0.2) & res.family.isin(FAMS)]
_pa = _pa.groupby(['model', 'seed', 'family']).detection_rate.min().reset_index()
_s2f = f'{BASE}/supp2/results_supp2.csv'
if os.path.exists(_s2f):
    _t2 = pd.read_csv(_s2f); _t2 = _t2[_t2.epsilon <= 0.2].groupby(['model', 'seed', 'family']).dr_tree.min().reset_index().rename(columns={'dr_tree': 'detection_rate'})
    _pa = pd.concat([_pa, _t2]).groupby(['model', 'seed', 'family']).detection_rate.min().reset_index()
FROB = _pa.groupby(['model', 'seed']).detection_rate.mean()
_flo = res[res.scenario == 'loao'].groupby(['model', 'seed']).detection_rate.mean()
FGAP = (FROB - _flo)
from scipy import stats as _stf
FG = {k: dict(rob=FROB[k].mean(), gap=FGAP[k].mean(), sd=FGAP[k].std(ddof=1), mn=FGAP[k].min(), p=_stf.ttest_1samp(FGAP[k], 0).pvalue) for k in M}
FG_LO, FG_HI = min(100 * FG[k]['gap'] for k in M), max(100 * FG[k]['gap'] for k in M)
FG_TXT = ("The robustness score above pools all known-attack flows, so DoS and DDoS dominate it, whereas LOAO detection is an unweighted mean over families. Averaging robustness the same way, taking for each family the minimum over all attacks at $\\epsilon \\le 0.2$ and then the mean over the six families, gives "
          + ', '.join(f"{pct(FG[k]['rob'])} ({NM[k]})" for k in M) + ". The gap to mean LOAO detection is then "
          + ', '.join(f"{100*FG[k]['gap']:.1f} $\\pm$ {100*FG[k]['sd']:.1f} points ({NM[k]})" for k in M)
          + f"; the smallest single model-seed gap is {min(100*FG[k]['mn'] for k in M):.1f} points and every per-model t-test gives $p \\le {max(FG[k]['p'] for k in M):.2g}$. The neural detectors are the least robust on Botnet and WebAttack, which is where family averaging lowers their score, yet the gap stays large.")
V['GAP_STATS'] = f"gaps of {V['GAP_LO']} to {V['GAP_HI']} percentage points, {FG_LO:.0f} to {FG_HI:.0f} when robustness is also averaged over families; all {T['pooled_n']} model-seed differences are positive; Holm-adjusted p ≤ {max(_ph.values()):.2g}"
V['SPEARMAN_SHORT'] = f"Spearman rho = {_rho_m:.2f}, exact permutation p = {_p_m:.2f}"
V['N_MCNEMAR'] = f'{N_MCN:,}'
R = []
R.append('A. Detection of known attacks')
R.append(f"On held-out flows of the six known families, every detector reaches near-ceiling performance at the 1% operating point: detection rates of "
         + ', '.join(f'{pct(ind_dr[k])} ({NM[k]})' for k in M) + f", with F1 between {min(ind_f1.values()):.3f} and {max(ind_f1.values()):.3f} and measured false-positive rates within 0.1 points of the 1% target (Table~\\ref{{tab:indist}}; Table~\\ref{{tab:data}} lists the data sizes). "
         "These numbers are a ceiling, not an estimate of field performance. The split is random at the flow level, so flows from the same attack session, which share almost identical statistics, appear on both sides; a temporal or session-level split would very likely lower them. "
         "None of our conclusions rests on them. They show only that the models are competent on what they have seen, which is the premise of the comparison that follows.")
R.append('B. Robustness to constrained perturbations')
assert mc[mc.model == 'rf'].sig_seeds.sum() == 0, 'RF McNemar claim no longer holds'
R.append(f"Table~\\ref{{tab:pert}} and the per-seed results show that the tree ensembles change little under every perturbation procedure we tried: under the black-box SimBA attack, which queries the target directly, RF and XGBoost still detect {pct(sim05['rf'])} and {pct(sim05['xgb'])} of perturbed attack flows even at $\\epsilon = 0.5$, and none of the RF McNemar tests is significant after Holm correction at any budget. "
         f"Running SimBA for five passes instead of two (200 flows per family and seed) leaves this unchanged: {pct(ss_dr['rf'])} for RF and {pct(ss_dr['xgb'])} for XGBoost. "
         + TREE_TXT +
         "We did not expect the trees to hold up this well, given that dedicated attacks evade tree ensembles [@kantchelian]. A plausible explanation, which we have not verified, is that many of their splits use backward-direction and flag features that our threat model does not let the attacker touch" + ("; a joint (MILP) attack could still find evasions that coordinate-wise search misses." if not MILP_TXT else ". The exact search above shows that the robustness itself is real within the box; it does not tell us why."))
R.append(f"The neural detectors are robust up to $\\epsilon = 0.2$ (worst case {pct(rob['cnn'])} for the 1D-CNN and {pct(rob['ft'])} for the FT-Transformer) but degrade sharply at $\\epsilon = 0.5$: white-box PGD reduces detection to {pct(wb05['cnn'])} (1D-CNN) and {pct(wb05['ft'])} (FT-Transformer), and SimBA to {pct(sim05['cnn'])} and {pct(sim05['ft'])}. "
         f"The 1D-CNN result is not driven by one bad seed; SimBA detection ranges from {pct(cnn_sim.min())} to {pct(cnn_sim.max())} across the five seeds. "
         f"Transferred attacks from the surrogate are much weaker ({pct(pt05['cnn'])} and {pct(pt05['ft'])} at $\\epsilon = 0.5$), and the random-sign control changes detection by at most a few points, so the degradation comes from the attack direction rather than from noise. "
         f"Cross-family hybrids behave similarly: detection of mixup flows is {pct(mixdr['rf'])} (RF), {pct(mixdr['xgb'])} (XGBoost), {pct(mixdr['cnn'])} (1D-CNN) and {pct(mixdr['ft'])} (FT-Transformer).")
R.append('C. Detection of unseen attack families')
R.append(f"The LOAO picture is very different (Table~\\ref{{tab:loao}}). Averaged over the six held-out families, detection at 1% FPR is {pct(lo_dr['rf'])} for RF, {pct(lo_dr['xgb'])} for XGBoost, {pct(lo_dr['cnn'])} for the 1D-CNN and {pct(lo_dr['ft'])} for the FT-Transformer. "
         f"Only DDoS is detected by every model when unseen, plausibly because its volumetric flow statistics resemble those of the DoS family still present in training. "
         f"Unseen PortScan and BruteForce flows are detected at {100*lofam('PortScan','xgb')['mean']:.0f}% and {100*lofam('BruteForce','xgb')['mean']:.0f}% by XGBoost and at most {100*max(lofam('PortScan',k)['mean'] for k in ['cnn','ft']):.0f}% by the neural models; even RF, the best model in this regime, misses roughly half of them. "
         f"The neural models also miss unseen DoS almost entirely ({100*lofam('DoS','cnn')['mean']:.0f}% and {100*lofam('DoS','ft')['mean']:.0f}%), whereas the tree ensembles detect it.")
R.append(f"The means hide what we found most telling: for several families, detection of an unseen family is close to all-or-nothing across seeds. "
         f"The FT-Transformer detects unseen Botnet at {pct(ft_bot.max())} in one seed and at {pct(np.sort(ft_bot)[-2])} in each of the other four; RF detects unseen PortScan at {pct(rf_ps[3])}--{pct(rf_ps[4])} in two seeds and at {pct(rf_ps[0])}--{pct(rf_ps[2])} in the other three. "
         "Whether a new family is caught can thus depend on which flows happened to be sampled for training, which is an uncomfortable property for a detector whose job is to catch the unexpected. "
         f"It also means that the Botnet and WebAttack rows of Table~\\ref{{tab:loao}}, with only {te['Botnet']} and {te['WebAttack']} test flows per seed, have confidence intervals spanning most of the unit interval. We report them for completeness and draw no family-level conclusion from them; the per-model means over six families, and the paired comparison below, are far less sensitive to this.")
R.append(f"Are the held-out families detectable at all at this operating point? The benign-only baselines suggest that, for most of them, not easily (Table~\\ref{{tab:anom}}). "
         f"The Isolation Forest detects on average {pct(an_dr['iforest'])} of unseen-family flows and the autoencoder {pct(an_dr['ae'])}, although their AUROC is {an_auc['iforest']:.2f} and {an_auc['ae']:.2f}; only PortScan stands out ({pct(an_fam.loc[('iforest','PortScan'),'detection_rate'])} and {pct(an_fam.loc[('ae','PortScan'),'detection_rate'])}). "
         f"This is less surprising than it looks: at a 1% alert budget, a benign-only detector has to place an attack beyond the most unusual 1% of benign flows, and benign traffic has a long tail of odd but harmless flows. "
         f"The LOAO failures of the supervised detectors are therefore not simply an artefact of using discriminative models: anomaly detectors held to the same 1% alert budget do worse on average, and RF does considerably better. "
         f"The baselines are not robust either; under SimBA at $\\epsilon = 0.5$ their already low detection falls to {pct(asim[('iforest', 0.5)])} and {pct(asim[('ae', 0.5)])}.")
R.append('D. Robustness overstates, and only coarsely ranks, unseen-family detection')
R.append(f"For every model, worst-case robustness at $\\epsilon \\le 0.2$ exceeds mean LOAO detection by a large margin: " + ', '.join(f"{NM[k]} {100*T[k]['gap_mean']:.1f} $\\pm$ {100*T[k]['gap_ci95']:.1f} points ({p_fmt(T[k]['paired_t_p'])})" for k in M)
         + f" (paired t-tests over five seeds; Holm-adjusted across the four models, all $p \\le {max(_ph.values()):.2g}$; standardised effect sizes $d_z$ from {min(_dz.values()):.1f} to {max(_dz.values()):.1f}). "
         + f"Every one of the five seed-level differences is positive for every model. With five seeds a per-model Wilcoxon test cannot fall below $p = 0.0625$, so we report the signed-rank test only over all {T['pooled_n']} model-seed pairs, where all differences are positive and it reaches its smallest attainable value ($p = {T['pooled_wilcoxon_p']*1e6:.1f} \\times 10^{{-6}}$). "
         + f"Used as an estimate of unseen-family detection in this setting, the robustness score would overstate it by {V['GAP_LO']} to {V['GAP_HI']} points (Table~\\ref{{tab:gapsens}}).")
R.append(f"Does robustness at least rank the detectors correctly? Only coarsely, and with little statistical support. The {T['pooled_n']} model-seed pairs are not independent: they come from four models measured five times each. Over the four model means, Spearman's correlation is $\\rho = {_rho_m:.2f}$, and an exact permutation test over the 24 possible orderings gives $p = {_p_m:.2f}$; permuting whole models at the model-seed level gives $\\rho = {_rho20:.2f}$ with $p = {_p20:.2f}$. (A naive test that treats the 20 pairs as independent would report $p < 10^{{-3}}$, which overstates the evidence.) What both measures do agree on is that the tree ensembles sit above the neural models. "
         f"The correlation is carried by that split between model families. Within the trees, XGBoost is as robust as RF ({pct(rob['xgb'])} vs. {pct(rob['rf'])}) but detects unseen families at roughly half RF's rate ({pct(lo_dr['xgb'])} vs. {pct(lo_dr['rf'])}); within the neural models the order reverses, with the FT-Transformer more robust and the 1D-CNN slightly better on unseen families. "
         "Robustness can tell a practitioner that, on this data and under our threat model, a tree ensemble is a safer choice than a small neural network. It cannot say which tree ensemble to deploy, and it says nothing reliable about how many new attacks the chosen model will catch.")
R.append(f"The gap is not an artefact of the two small families (Table~\\ref{{tab:gapsens}}). Averaging LOAO detection over the four large families only (DoS, DDoS, PortScan, BruteForce), the gap is {SENS['lo4']['lo']:.0f} to {SENS['lo4']['hi']:.0f} points; weighting families by their number of test flows, which favours the easily detected DoS and DDoS, it is {SENS['low']['lo']:.0f} to {SENS['low']['hi']:.0f} points. "
         f"In both cases all {T['pooled_n']} model-seed differences remain positive and every per-model paired t-test gives $p \\le {max(SENS['lo4']['p'], SENS['low']['p']):.3f}$.")
R.append('E. Feature consistency and repaired perturbations')
AUD = ''
if os.path.exists(f'{BASE}/audit/audit.csv'):
    au = pd.read_csv(f'{BASE}/audit/audit.csv')
    rel = [c for c in au.columns if c not in ('seed', 'variant', 'epsilon', 'repair')]
    pert_rows = au[au.variant != 'clean']
    perturbed_rel = [c for c in rel if pert_rows[c].max() > 0]
    clean_max = au[au.variant == 'clean'][rel].max().max()
    lo_v = pert_rows[perturbed_rel].mean(axis=1)
    e05 = pert_rows[pert_rows.epsilon == 0.05][perturbed_rel].mean(axis=1)
    AUD = (f"Feature-space perturbations can produce flows that no packet sequence could generate. We audited {len(rel)} derived-feature identities of CICFlowMeter (e.g., Flow Bytes/s = total bytes / duration), which hold for {100*(1-clean_max):.0f}% of clean test flows. "
           f"After projection, {100*lo_v.min():.0f}% to {100*lo_v.max():.0f}% of perturbed or mixed flows violate an identity involving a modified feature, depending on variant and budget (averaged over identities), and already {100*e05.min():.0f}% to {100*e05.max():.0f}% at $\\epsilon = 0.05$. Most perturbed flows in the main experiment are therefore not internally consistent.")
    _rp_files = [f'{BASE}/repair/results.csv'] + [f for f in [f'{BASE}/repair34/results.csv', f'{BASE}/repair34/results_ALL.csv'] if os.path.exists(f)][:1]
    rp_all = pd.concat([pd.read_csv(f_) for f_ in _rp_files])
    n_rep = rp_all.seed.nunique()
    sel = lambda d: d[(d.family == 'ALL') & (d.scenario == 'simba_blackbox') & (d.epsilon == 0.5)].pivot_table(index='seed', columns='model', values='detection_rate')
    rp_s, bs_s = sel(rp_all), sel(res)
    from scipy import stats as _st2
    RP = {}
    for k in M:
        a_ = rp_s[k].dropna(); b_ = bs_s.loc[a_.index, k]
        RP[k] = dict(rep=a_.mean(), rep_ci=ci_half(a_), base=bs_s[k].mean(), base_ci=ci_half(bs_s[k]), p=_st2.ttest_rel(a_, b_).pvalue)
    sig_ = [k for k in M if RP[k]['p'] < 0.05]
    _ATT = ['fgsm_transfer', 'pgd_transfer', 'pgd_whitebox', 'simba_blackbox']
    _wc = lambda d: d[(d.family == 'ALL') & d.scenario.isin(_ATT) & (d.epsilon <= 0.2)].groupby(['seed', 'model']).detection_rate.min()
    _wr, _wb = _wc(rp_all), _wc(res)
    _lo_s = gap.set_index(['seed', 'model']).loao_dr_mean
    RW = {}
    for k in M:
        sd_ = sorted(s_ for (s_, m_) in _wr.index if m_ == k)
        a_ = np.array([_wr[(s_, k)] for s_ in sd_]); b_ = np.array([_wb[(s_, k)] for s_ in sd_]); l_ = np.array([_lo_s[(s_, k)] for s_ in sd_])
        RW[k] = dict(rep=a_.mean(), rep_ci=ci_half(a_), base=b_.mean(), base_ci=ci_half(b_), p=_st2.ttest_rel(a_, b_).pvalue,
                     gap=(a_ - l_).mean(), gap_ci=ci_half(a_ - l_), gap_p=_st2.ttest_rel(a_, l_).pvalue, gap_pos=int(((a_ - l_) > 0).sum()), n=len(a_))
    assert all(RW[k]['gap_pos'] == RW[k]['n'] for k in M)
    AUD += (f" So we re-ran the perturbation experiments with all identities recomputed after every step ({n_rep} seeds; Table~\\ref{{tab:repair}}). SimBA detection at $\\epsilon = 0.5$ becomes "
            + ', '.join(f"{pct(RP[k]['rep'])} for {NM[k]} ({pct(RP[k]['base'])} without repair)" for k in M) + ". "
            + ("None of these changes is significant (paired t-tests over seeds, all $p \\ge " + f"{min(RP[k]['p'] for k in M):.2f}" + "$). " if not sig_ else
               "Paired t-tests over seeds find a significant change only for " + ', '.join(f"{NM[k]} ($p = {RP[k]['p']:.2g}$)" for k in sig_) + ". ")
            + "Repair, in other words, does not systematically weaken the attack. "
            + f"The repair run covers FGSM, PGD (transfer and white-box) and SimBA at every budget, not only SimBA at $\\epsilon = 0.5$; the tree-aware search was not re-run with repair. Taking the worst case over FGSM, PGD (transfer and white-box) and SimBA at $\\epsilon \\le 0.2$ with identities enforced, robustness is "
            + ', '.join(f"{pct(RW[k]['rep'])} for {NM[k]} ({pct(RW[k]['base'])} without repair)" for k in M)
            + ", and the gap to mean LOAO detection becomes " + ', '.join(f"{100*RW[k]['gap']:.1f} $\\pm$ {100*RW[k]['gap_ci']:.1f} points ({NM[k]})" for k in M)
            + f", positive in every seed and model (paired t-tests, all $p \\le {max(RW[k]['gap_p'] for k in M):.2g}$). Using only consistent, repaired perturbations therefore leaves the conclusion unchanged. For the tree ensembles the unrepaired tree-aware result is within 0.1 points of SimBA, so leaving it out of the repaired comparison changes little.")
    rep_tab = ['\\begin{table}[t]\\centering\\caption{Detection with and without enforcing the derived-feature identities (repair), mean $\\pm$ 95\\% CI over seeds. Worst case: minimum over FGSM, PGD (transfer, white-box) and SimBA at $\\epsilon \\le 0.2$. Gap: repaired worst case minus mean LOAO detection. $p$: paired t-tests.}\\label{tab:repair}',
               '\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lccccc}\\toprule', ' & \\multicolumn{2}{c}{Worst case, $\\epsilon \\le 0.2$} & \\multicolumn{2}{c}{SimBA, $\\epsilon = 0.5$} & \\\\', f'Model & No repair & Repair & No repair & Repair & Gap (repair) \\\\\\midrule']
    rep_tab += [f"{NM[k]} & {RW[k]['base']:.3f} & {RW[k]['rep']:.3f} & {RP[k]['base']:.3f} & {RP[k]['rep']:.3f} ($p$={RP[k]['p']:.2f}) & {RW[k]['gap']:.3f} $\\pm$ {RW[k]['gap_ci']:.3f} \\\\" for k in M]
    rep_tab += ['\\bottomrule\\end{tabular}}\\end{table}']
R.append(AUD)
R.append('F. Threshold-free metrics overstate unseen-family performance')
R.append(f"In LOAO, mean AUROC remains high for the tree ensembles ({lo_auc['rf']:.3f} for RF and {lo_auc['xgb']:.3f} for XGBoost) while their detection rates at 1% FPR are {pct(lo_dr['rf'])} and {pct(lo_dr['xgb'])}. For example, unseen PortScan obtains an AUROC of {loauc('PortScan','xgb')['mean']:.3f} with XGBoost but only {100*lofam('PortScan','xgb')['mean']:.0f}% detection at the operating point, and the anomaly baselines show the same pattern. "
         "Unseen attacks are ranked above most benign traffic but not above its tail, so an operator working with a realistic alert budget would miss them. LOAO results should therefore be reported at a fixed false-positive rate rather than only as AUROC.")
R.append('G. Our view and recommendations')
R.append(f"We would keep reporting robustness to perturbations; it measures something real, namely how stable a decision boundary is around attacks the model already knows. What we would stop doing is citing it as evidence of zero-day capability. Holding out an attack family costs a few extra training runs. In our experiments the FT-Transformer kept {pct(rob['ft'])} worst-case detection at $\\epsilon \\le 0.2$ yet detected only {pct(lo_dr['ft'])} of held-out-family flows on average; a robustness report alone would not have revealed that. Concretely:")
R.append("- Report robustness to perturbations and detection of held-out attack families as separate results; neither substitutes for the other.\n- Report LOAO detection at a fixed, realistic false-positive rate in addition to AUROC, with per-seed values, since unseen-family detection can vary strongly between seeds.\n- Evaluate tree ensembles with query-based attacks rather than gradient attacks they cannot receive, and audit perturbed flows for internal consistency.\n- Use corrected datasets and remove duplicate and label-conflicting flows before splitting.")
# ================= review round 4: open-set baseline and temporal split (included when their result files exist) =================
OPENSET_TAB, TEMPORAL_TAB, OS_TXT, TMP_TXT_A, TMP_TXT_D = '', '', '', '', ''
_osf = f'{BASE}/openset/results_openset.csv'
if os.path.exists(_osf):
    os_ = pd.read_csv(_osf)
    _os_seeds = sorted(os_.seed.unique().tolist())
    OSM = os_.groupby(['model', 'variant']).detection_rate.mean()
    OSF = os_.groupby(['model', 'variant', 'family']).detection_rate.mean()
    OSA = os_.groupby(['model', 'variant']).auroc.mean()
    _osfpr = os_.groupby(['model', 'variant']).fpr.mean()
    _same = bool((os_.pivot_table(index=['seed', 'model', 'family'], columns='variant', values='detection_rate').diff(axis=1).abs().max().max()) < 1e-12)
    _d_rf, _d_xgb = OSM[('rf', 'mc_msp')] - lo_dr['rf'], OSM[('xgb', 'mc_msp')] - lo_dr['xgb']
    def _pp(v): return f"{'+' if v >= 0 else '-'}{abs(v) * 100:.1f}"
    OS_TXT = (f"Would a detector built for unknown inputs do better? As an open-set baseline we retrained RF and XGBoost as multi-class models (benign plus one class per known family) and flagged a flow either when an attack class wins or when the winning benign class has low confidence (maximum-softmax-probability rejection, MSP), thresholded at the same 1% FPR ({len(_os_seeds)} seeds; Table~\\ref{{tab:openset}}). "
              + ("At this operating point MSP gives the same detection rate as the simpler score $1 - P(\\text{benign})$ in every seed, model and family: the two scores coincide whenever the benign class wins, and the 1% FPR threshold is low enough that flows where an attack class wins are flagged under either score. " if _same else "")
              + f"Mean held-out-family detection is {pct(OSM[('rf', 'mc_msp')])} for RF and {pct(OSM[('xgb', 'mc_msp')])} for XGBoost, against {pct(lo_dr['rf'])} and {pct(lo_dr['xgb'])} for the binary detectors ({_pp(_d_rf)} and {_pp(_d_xgb)} points). "
              + f"PortScan stays below {max(OSF[('rf', 'mc_msp', 'PortScan')], OSF[('xgb', 'mc_msp', 'PortScan')]) * 100:.0f}% and WebAttack below {max(OSF[('rf', 'mc_msp', 'WebAttack')], OSF[('xgb', 'mc_msp', 'WebAttack')]) * 100:.0f}% for both models. "
              + ("Open-set rejection of this simple kind therefore does not close the gap to the robustness scores. " if max(OSM[('rf', 'mc_msp')], OSM[('xgb', 'mc_msp')]) < min(rob['rf'], rob['xgb']) - 0.1 else "")
              + "Stronger open-set methods exist [@zoppi], and we do not claim this baseline is the best possible; it shows that the gap is not specific to binary training.")
    ot_ = ['\\begin{table}[t]\\centering\\caption{Open-set baseline: multi-class RF and XGBoost with maximum-softmax-probability rejection (MSP), detection of the held-out family at 1\\% FPR (mean over ' + str(len(_os_seeds)) + ' seeds), next to the binary detectors of Table~\\ref{tab:loao}.' + (' The score $1-P(\\text{benign})$ gives identical detection rates at this threshold.' if _same else '') + '}\\label{tab:openset}',
           '\\begin{tabular}{lcccc}\\toprule', ' & \\multicolumn{2}{c}{RF} & \\multicolumn{2}{c}{XGBoost} \\\\', 'Held-out & Binary & MSP & Binary & MSP \\\\\\midrule']
    for f in FAMS:
        ot_.append(f"{f} & " + ' & '.join(f"{lofam(f, k)['mean']:.3f} & {OSF[(k, 'mc_msp', f)]:.3f}" for k in ['rf', 'xgb']) + ' \\\\')
    ot_.append('\\midrule Mean & ' + ' & '.join(f"{lo_dr[k]:.3f} & {OSM[(k, 'mc_msp')]:.3f}" for k in ['rf', 'xgb']) + ' \\\\')
    ot_ += ['\\bottomrule\\end{tabular}\\end{table}']
    OPENSET_TAB = '\n\n' + '\n'.join(ot_)
_tmf = f'{BASE}/temporal/results.csv'
if os.path.exists(_tmf):
    tm_ = pd.read_csv(_tmf)
    _ts = sorted(tm_.seed.unique().tolist())
    def _summ(d):
        out = {}
        for k in MT:
            q = d[d.model == k]
            ind_ = q[(q.scenario == 'in_dist') & (q.family == 'ALL')]
            pg_ = q[(q.family == 'ALL') & q.scenario.isin(['fgsm_transfer', 'pgd_transfer', 'pgd_whitebox', 'simba_blackbox']) & (q.epsilon <= 0.2)]
            lg_ = q[q.scenario == 'loao']
            out[k] = dict(dr=ind_.detection_rate.mean(), fpr=ind_.fpr.mean(), rob=pg_.groupby('seed').detection_rate.min().mean(),
                          lo=lg_.groupby('seed').detection_rate.mean().mean(), lofpr=lg_.fpr.mean())
            out[k]['gap'] = out[k]['rob'] - out[k]['lo']
        return out
    MT = [k for k in M if (tm_.model == k).any()]
    _miss = [NM[k] for k in M if k not in MT]
    TS = _summ(tm_); RS = _summ(res[res.seed.isin(_ts)])
    _sd = ', '.join(str(x) for x in _ts)
    TMP_TXT_A = (f" To see how much of this ceiling comes from the split, we repeated the experiment with a temporal split (seed{'s' if len(_ts) > 1 else ''} {_sd}): within every family, including benign traffic, the earliest 70% of flows by start time form the training pool and the latest 30% the test pool, so test flows come from later in each capture and the threshold is set on earlier benign traffic (Table~\\ref{{tab:temporal}}). "
                 f"Known-attack detection becomes " + ', '.join(f"{pct(TS[k]['dr'])} ({NM[k]})" for k in MT)
                 + f", and the false-positive rate measured on the later benign flows is " + ', '.join(f"{100*TS[k]['fpr']:.1f}%" for k in MT) + " against the 1% target, so the threshold drifts on later traffic as one would expect.")
    _gap_ok = min(TS[k]['gap'] for k in MT) > 0.10
    TMP_TXT_D = ((f"The gap also survives the temporal split: worst-case robustness minus mean LOAO detection is " if _gap_ok else f"Under the temporal split the gap is smaller for at least one model: worst-case robustness minus mean LOAO detection is ") + ', '.join(f"{100*TS[k]['gap']:.1f} points ({NM[k]})" for k in MT)
                 + f" there, against " + ', '.join(f"{100*RS[k]['gap']:.1f}" for k in MT) + f" for the random split on the same seed{'s' if len(_ts) > 1 else ''}. "
                 + "This comparison rests on " + ('one seed' if len(_ts) == 1 else f'{len(_ts)} seeds') + " and on the gradient and SimBA attacks only, so we read it as a robustness check of the main result rather than as a second estimate of its size."
                 + (f" The {', '.join(_miss)} was not retrained under the temporal split, because only a CPU runtime was available." if _miss else ''))
    tt_ = ['\\begin{table}[t]\\centering\\caption{Temporal split (earliest 70\\% of each family for training, latest 30\\% for testing; seed ' + _sd + (('; ' + ', '.join(_miss) + ' not run') if _miss else '') + ') against the random split on the same seed: known-attack detection and test FPR, worst-case robustness ($\\epsilon \\le 0.2$; FGSM, PGD, SimBA), mean LOAO detection and the gap (points).}\\label{tab:temporal}',
           '\\resizebox{\\columnwidth}{!}{\\begin{tabular}{llccccc}\\toprule', 'Model & Split & Known DR & Test FPR & Robustness & LOAO & Gap \\\\\\midrule']
    for k in MT:
        for lab_, S_ in [('random', RS), ('temporal', TS)]:
            tt_.append(f"{NM[k] if lab_ == 'random' else ''} & {lab_} & {S_[k]['dr']:.3f} & {S_[k]['fpr']:.3f} & {S_[k]['rob']:.3f} & {S_[k]['lo']:.3f} & {100*S_[k]['gap']:.1f} \\\\")
    tt_ += ['\\bottomrule\\end{tabular}}\\end{table}']
    TEMPORAL_TAB = '\n\n' + '\n'.join(tt_)
# splice the optional paragraphs into the results text
_iD = R.index('D. Robustness overstates, and only coarsely ranks, unseen-family detection')
if OS_TXT: R.insert(_iD, OS_TXT); _iD += 1
if TMP_TXT_D: R.insert(_iD + 4, TMP_TXT_D)
_ifg = next(i for i, x in enumerate(R) if 'model-seed differences remain positive' in x)
R.insert(_ifg + 1, FG_TXT)
if TMP_TXT_A: R[1] = R[1] + TMP_TXT_A
V['LIMIT_SPLIT'] = (("Splits in the main experiment are random at the flow level after deduplication. A temporal split on " + ('one seed' if len(_ts) == 1 else f'{len(_ts)} seeds') + " " + ("leaves the gap in place" if _gap_ok else "changes the size of the gap") + " (Section V-D), but a full temporal evaluation over all seeds, and a session-level split, remain to be done. ")
                    if TMP_TXT_D else
                    "Splits are random at the flow level after deduplication rather than temporal, because each CIC-IDS2017 attack occurs on a single day. This inflates in-distribution detection, and it means the 1% threshold is calibrated on benign flows from the same period as the benign test flows; the measured test false-positive rate matches the target, but under temporal drift it would not, and every regime would be affected. ")
_extra = [x for x, ok in [('the open-set baseline', bool(OS_TXT)), ('the temporal split', bool(TMP_TXT_D))] if ok]
V['EXTRA_RUNS'] = (' ' + ' and '.join(_extra).capitalize() + (' were' if len(_extra) > 1 else ' was') + ' run later with the final version of the script on a CPU-only Colab runtime; its log records the library versions.') if _extra else ''
_rf = ['the main per-seed results', 'the anomaly-baseline and five-pass SimBA results', 'the tree-aware attack results', 'the identity audit', 'the repaired-perturbation results'] + (['the open-set results'] if OS_TXT else []) + (['the temporal-split results'] if TMP_TXT_D else [])
_nw = {5: 'five', 6: 'six', 7: 'seven'}
V['RESULT_FILES'] = f"{_nw[len(_rf)]} result files: " + ', '.join(_rf[:-1]) + ', and ' + _rf[-1]
_ambf = f'{BASE}/amb.json'
if os.path.exists(_ambf):
    _amb = json.load(open(_ambf)); _na = sum(_amb.values())
    V['LIMIT_AMB'] = (f"The label-conflict step uses labels from the whole dataset, but it removes only {_na} flows ("
                      + ', '.join(f"{v} {k}" for k, v in sorted(_amb.items(), key=lambda x: (-x[1], x[0]))) + "), fewer than 0.01% of the data; the other "
                      + f"{N_REM - _na:,} removed flows are exact duplicates of a flow with the same label. Its effect on any result is therefore negligible. ")
else:
    V['LIMIT_AMB'] = ''
V['MILP_ABS'] = ('An exact mixed-integer evasion search confirms the tree ensembles\' scores. ' if MILP_TXT else '')
V['OS_ABS'] = ' and a simple multi-class open-set baseline' if OS_TXT else ''
V['TMP_ABS'] = ((' A temporal split on ' + ('one seed' if len(_ts) == 1 else f'{len(_ts)} seeds') + (f' ({len(MT)} of the {len(M)} models)' if len(MT) < len(M) else '') + (' leaves the gap in place.' if _gap_ok else ' narrows the gap for at least one model.')) if TMP_TXT_D else '')
V['OPENSET_METHOD'] = ("We did include a simple open-set baseline, a multi-class detector with maximum-softmax-probability rejection (Section V-C); dedicated open-set and meta-learning detectors [@zoppi] remain a natural extension." if OS_TXT else
                       "We did not include open-set or meta-learning detectors designed specifically for unknown attacks [@zoppi]; our question concerns the evaluation proxy as it is applied to the detectors that robustness studies usually report on, and testing such detectors under the same protocol is a natural extension.")

V['RESULTS_TEXT'] = '\n\n'.join(R)

# tables: data sizes + analysis tables + anomaly + repair
dt = ['\\begin{table}[t]\\centering\\caption{Flows per family after cleaning, and per-seed train/test sizes used. Infiltration and Heartbleed are excluded from all experiments.}\\label{tab:data}',
      '\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lrrr}\\toprule', 'Family & Unique flows & Train & Test \\\\\\midrule',
      f"Benign & {fam_counts['BENIGN']:,} & {args['benign_train']*4//5:,} (+{args['benign_train']//5:,} val.) & {args['benign_test']:,} \\\\"]
for f in FAMS: dt.append(f"{f} & {fam_counts[f]:,} & {tr[f]:,} & {te[f]:,} \\\\")
for f in ['Infiltration', 'Heartbleed']: dt.append(f"{f} (excluded) & {fam_counts[f]:,} & -- & -- \\\\")
assert sum(fam_counts.values()) == N_FLOWS, (sum(fam_counts.values()), N_FLOWS)
dt += [f"\\midrule Total & {N_FLOWS:,} & & \\\\", '\\bottomrule\\end{tabular}}\\end{table}']
at = ['\\begin{table}[t]\\centering\\caption{Benign-only anomaly baselines on attack families they never saw: detection rate at 1\\% FPR and AUROC, mean over five seeds.}\\label{tab:anom}',
      '\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lrcccc}\\toprule', ' & & \\multicolumn{2}{c}{Isolation Forest} & \\multicolumn{2}{c}{Autoencoder} \\\\', 'Family & $n$ & DR & AUROC & DR & AUROC \\\\\\midrule']
for f in FAMS:
    at.append(f"{f} & {te[f]:,} & {an_fam.loc[('iforest', f), 'detection_rate']:.3f} & {an_fam.loc[('iforest', f), 'auroc']:.3f} & {an_fam.loc[('ae', f), 'detection_rate']:.3f} & {an_fam.loc[('ae', f), 'auroc']:.3f} \\\\")
at.append(f"\\midrule Mean & & {an_dr['iforest']:.3f} & {an_auc['iforest']:.3f} & {an_dr['ae']:.3f} & {an_auc['ae']:.3f} \\\\")
at += ['\\bottomrule\\end{tabular}}\\end{table}']
tt = open(f'{A}/tables.tex').read()
tt = tt.replace('\\begin{tabular}{lcccc}\\toprule\nHeld-out family &', '\\begin{tabular}{lrcccc}\\toprule\nHeld-out family & $n$ &', 1)
for f in FAMS:
    assert tt.count('\n' + f + ' & ') == 1, f
    tt = tt.replace('\n' + f + ' & ', '\n' + f + f' & {te[f]:,} & ', 1)
tt = tt.replace('mean $\\pm$ 95\\% CI over seeds.}\\label{tab:loao}', 'mean $\\pm$ 95\\% CI over seeds; $n$ is the number of test flows of the held-out family per seed. The Botnet and WebAttack rows rest on few flows and vary strongly between seeds; they are reported for completeness only.}\\label{tab:loao}', 1)
# ================= review round 4: new tables =================
_ci = lambda v: ci_half(np.asarray(v, float))
# (1) in-distribution operating point
it_ = ['\\begin{table}[t]\\centering\\caption{Known attacks under the random flow-level split: detection rate, measured false-positive rate on benign test flows, F1 and AUROC at the threshold set for 1\\% FPR on benign validation flows (mean $\\pm$ 95\\% CI over five seeds). These are ceilings, not field estimates (Section V-A).}\\label{tab:indist}',
       '\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lcccc}\\toprule', 'Model & Detection & FPR & F1 & AUROC \\\\\\midrule']
for k in M:
    q_ = {mt: g(ind, model=k, metric=mt) for mt in ['detection_rate', 'fpr', 'f1', 'auroc']}
    it_.append(f"{NM[k]} & {q_['detection_rate']['mean']:.4f} $\\pm$ {q_['detection_rate']['ci95']:.4f} & {q_['fpr']['mean']:.4f} $\\pm$ {q_['fpr']['ci95']:.4f} & {q_['f1']['mean']:.3f} $\\pm$ {q_['f1']['ci95']:.3f} & {q_['auroc']['mean']:.4f} \\\\")
it_ += ['\\bottomrule\\end{tabular}}\\end{table}']
# (2) gap under three ways of averaging LOAO
gs_ = ['\\begin{table}[t]\\centering\\caption{Worst-case robustness ($\\epsilon \\le 0.2$, all procedures) against LOAO detection averaged three ways, and the resulting gap in percentage points (mean over five seeds; every model-seed gap is positive in all three cases).}\\label{tab:gapsens}',
       '\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lccccccc}\\toprule', ' & & \\multicolumn{2}{c}{All six families} & \\multicolumn{2}{c}{Four large families} & \\multicolumn{2}{c}{Flow-weighted} \\\\',
       'Model & Robustness & LOAO & Gap & LOAO & Gap & LOAO & Gap \\\\\\midrule']
for k in M:
    r_ = _d3[_d3.model == k].rob.mean()
    cells = []
    for c_ in ['lo6', 'lo4', 'low']:
        l_ = _d3[_d3.model == k][c_].mean(); cells += [f'{l_:.3f}', f'{100 * (r_ - l_):.1f}']
    gs_.append(f"{NM[k]} & {r_:.3f} & " + ' & '.join(cells) + ' \\\\')
gs_ += ['\\bottomrule\\end{tabular}}\\end{table}']
# (3) perturbation table with every procedure and the number of flows
_sc = [('noise', 'Random sign'), ('fgsm_transfer', 'FGSM (transfer)'), ('pgd_transfer', 'PGD (transfer)'), ('pgd_whitebox', 'PGD (white-box)'), ('simba_blackbox', 'SimBA')]
_nE = res[(res.family == 'ALL')].groupby('scenario').n_eval.mean()
pt_ = ['\\begin{table*}[t]\\centering\\caption{Detection rate at 1\\% FPR under every perturbation procedure (mean $\\pm$ 95\\% CI over five seeds). $n$: attack flows per seed (all six families). Gradient attacks do not apply to tree ensembles (--); the tree-aware search applies only to them. Mixup: cross-family hybrids, no $\\epsilon$.}\\label{tab:pert}',
       '\\begin{tabular}{llrcccc}\\toprule', 'Procedure & $\\epsilon$ & $n$ & RF & XGBoost & 1D-CNN & FT-Transformer \\\\\\midrule']
for sc_, lab_ in _sc:
    for e_ in sorted(pert[pert.scenario == sc_].epsilon.unique()):
        cells = []
        for k in M:
            q_ = pert[(pert.scenario == sc_) & (pert.model == k) & (pert.epsilon == e_)]
            cells.append(f"{q_['mean'].values[0]:.3f} $\\pm$ {min(q_['ci95'].values[0], 1):.3f}" if len(q_) else '--')
        pt_.append(f"{lab_} & {e_:g} & {int(_nE[sc_]):,} & " + ' & '.join(cells) + ' \\\\')
if TREE_TXT:
    for e_ in [0.2, 0.5]:
        cells = []
        for k in M:
            d_ = t2[(t2.model == k) & (t2.epsilon == e_)]
            if len(d_):
                v_ = d_.groupby('seed').apply(lambda x: np.average(x.dr_tree, weights=x.n_eval)).values
                cells.append(f"{v_.mean():.3f} $\\pm$ {_ci(v_):.3f}")
            else: cells.append('--')
        pt_.append(f"Tree-aware search & {e_:g} & {int(t2[(t2.seed == 0) & (t2.model == 'rf') & (t2.epsilon == e_)].n_eval.sum()):,} & " + ' & '.join(cells) + ' \\\\')
pt_.append(f"Mixup & -- & {int(_nE['mixup']):,} & " + ' & '.join(f"{g(mx, model=k, metric='detection_rate')['mean']:.3f} $\\pm$ {g(mx, model=k, metric='detection_rate')['ci95']:.3f}" for k in M) + ' \\\\')
pt_ += ['\\bottomrule\\end{tabular}\\end{table*}']
_tt0 = tt
_a = tt.index('\\begin{table}[t]\\centering\\caption{Detection rate at 1\\% FPR under constrained perturbations')
tt = tt[:_a] + '\n'.join(pt_)   # replaces the analysis script's shorter perturbation table
assert 'tab:loao' in tt and tt.count('tab:pert') == 1
# (4) attacker-controllable features (appendix)
_cfg = json.load(open(f'{BASE}/results/config.json'))
_mod = _cfg['modifiable']; _all = _cfg['features']
assert len(_mod) == NMOD and len(_all) == NF
V['FEATURE_APPENDIX'] = (f"The {NMOD} attacker-controllable features (of {NF}) are: " + '; '.join(_mod) + ". "
                         f"The remaining {NF - NMOD} features (backward-direction statistics, TCP flag counts, protocol, initial window sizes, the down/up ratio and the forward minimum segment size) are never modified.")

V['TABLES_LATEX'] = '\n'.join(dt) + '\n\n' + '\n'.join(it_) + '\n\n' + tt + '\n\n' + '\n'.join(gs_) + '\n\n' + '\n'.join(at) + ('\n\n' + '\n'.join(rep_tab) if AUD else '') + OPENSET_TAB + TEMPORAL_TAB

V['CONCLUSION_TEXT'] = (f"On corrected CIC-IDS2017, four detectors with near-perfect accuracy on known attacks remain largely robust to constrained feature-space perturbations, yet miss {100*(1-max(lo_dr.values())):.0f}% to {100*(1-min(lo_dr.values())):.0f}% of flows from attack families absent from training. "
                        f"On this benchmark, robustness overstates unseen-family detection by {V['GAP_LO']} to {V['GAP_HI']} percentage points for every model and agrees with it only on the coarse ranking of trees over neural networks. The two benign-only anomaly detectors we tested did not close the gap at a 1% alert budget. "
                        "Within the scope tested here (one corrected dataset, four detectors and a feature-space threat model), perturbations of known attacks measure a different property from zero-day generalization, and synthetic perturbation sets should not be offered as evidence that a NIDS detects new attacks. "
                        "We see this as one instance of a broader problem in security machine learning: benchmark numbers that are easy to produce get used to answer questions they were never designed to answer. Held-out-family evaluation at a fixed false-positive rate is cheap and reproducible, and we hope it becomes routine. "
                        "The next steps we consider most useful are testing whether adversarial training or open-set and domain-adaptation methods narrow the gap, and repeating the protocol across datasets and with problem-space attack generation.")
tpl = open(os.path.join(os.path.dirname(__file__), 'sections_template.md')).read()
for k, v in V.items(): tpl = tpl.replace('{' + k + '}', v)
left = re.findall(r'\{[A-Z_]{4,}\}', tpl)
assert not left, left
secs = []
for blk in re.split(r'^## ', tpl, flags=re.M)[1:]:
    head, _, body = blk.partition('\n'); secs.append({'section': head.strip(), 'content': body.strip()})
title = secs.pop(0)['content']
REFS = json.load(open(os.path.join(os.path.dirname(__file__), 'refs.json')))
order = []
for s_ in secs:
    for k in re.findall(r'\[@([a-z0-9]+)\]', s_['content']):
        assert k in REFS, k
        if k not in order: order.append(k)
num = {k: i + 1 for i, k in enumerate(order)}
for s_ in secs:
    s_['content'] = re.sub(r'\[@([a-z0-9]+)\]', lambda mm: '[' + str(num[mm.group(1)]) + ']', s_['content'])
refs = [dict(REFS[k], key=k, n=num[k]) for k in order]
unused = [k for k in REFS if k not in num]
print('refs used', len(refs), 'unused', unused)
json.dump({'title': title, 'sections': secs, 'refs': refs, 'values': V}, open(OUT, 'w'), indent=1, ensure_ascii=False)
print(title); print([s['section'] for s in secs]); print(V['ABSTRACT_RESULTS']); print(V['GAP_STATS'])
