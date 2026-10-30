#!/usr/bin/env python3
"""Counts the flows removed by the label-conflict step of experiment.py (same loading and cleaning code).

usage: python code/count_label_conflicts.py DATA_DIR OUT_JSON
"""
import sys, json
data, out = sys.argv[1], sys.argv[2]
src = open(__file__.replace('count_label_conflicts.py', 'experiment.py')).read()
src = src[:src.index('# attacker-controllable')]          # imports, arguments, family mapping and load()
sys.argv = ['experiment.py', '--data', data]
g = {'__name__': 'loader'}
exec(compile(src, 'experiment.py', 'exec'), g)
pd, np = g['pd'], g['np']
# re-run the cleaning steps of load() and keep the label-conflict rows
orig_dup = pd.DataFrame.duplicated
store = {}
def dup(self, subset=None, keep='first'):
    r = orig_dup(self, subset=subset, keep=keep)
    if keep is False: store['amb'] = self.loc[r, 'family'].value_counts().to_dict()
    return r
pd.DataFrame.duplicated = dup
df, feats = g['load']()
json.dump({k: int(v) for k, v in store['amb'].items()}, open(out, 'w'), indent=1)
print('label-conflict flows removed:', store['amb'], sum(store['amb'].values()))
