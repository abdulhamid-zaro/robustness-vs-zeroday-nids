#!/bin/sh
# Builds the folder layout that build_paper.py expects, as symbolic links into results/ and analysis/.
# usage: sh paper/make_layout.sh build && python paper/build_paper.py build paper.json
set -e
OUT=${1:-build}; ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
ln -sfn "$ROOT/results/main"            "$OUT/results"
ln -sfn "$ROOT/analysis"                "$OUT/analysis"
ln -sfn "$ROOT/results/supplementary"   "$OUT/supp"
ln -sfn "$ROOT/results/tree_attack"     "$OUT/supp2"
ln -sfn "$ROOT/results/audit"           "$OUT/audit"
ln -sfn "$ROOT/results/repair"          "$OUT/repair"
ln -sfn "$ROOT/results/repair_seeds34"  "$OUT/repair34"
[ -d "$ROOT/results/openset" ]  && ln -sfn "$ROOT/results/openset"  "$OUT/openset"
[ -d "$ROOT/results/temporal" ] && ln -sfn "$ROOT/results/temporal" "$OUT/temporal"
[ -d "$ROOT/results/milp" ] && ln -sfn "$ROOT/results/milp" "$OUT/milp"
[ -f "$ROOT/results/label_conflicts.json" ] && ln -sfn "$ROOT/results/label_conflicts.json" "$OUT/amb.json"
echo "layout ready in $OUT"
