# Adversarial Robustness Is Not Zero-Day Generalization

Code, configuration and per-seed results for the paper

> **Adversarial Robustness Is Not Zero-Day Generalization: A Controlled Evaluation of Flow-Based Intrusion Detectors on Corrected CIC-IDS2017**
> Abdulhamid Zaro, College of Professions and Applied Sciences, Hebron University (ORCID [0009-0003-6241-302X](https://orcid.org/0009-0003-6241-302X))

The paper compares two things on the same data, models and operating point (1% false-positive rate, five seeds):
- robustness of four flow-based detectors (Random Forest, XGBoost, 1D-CNN, FT-Transformer) to constrained feature-space perturbations of known attacks;
- their detection of attack families that are entirely absent from training (leave-one-attack-family-out, LOAO).

Every number in the paper is computed from the files in `results/` by `code/analyze.py` and `paper/build_paper.py`. No value was entered by hand.

## Repository layout

```
code/experiment.py     experiment script (all modes below)
code/analyze.py        aggregation, statistics, LaTeX tables, figures
results/main/          main run: per-seed results (results.csv, 2,800 rows), run log, config
results/supplementary/ anomaly baselines (Isolation Forest, autoencoder), five-pass SimBA, epsilon scale
results/tree_attack/   tree-aware coordinate search vs. SimBA on RF and XGBoost
results/audit/         derived-feature identity audit (CICFlowMeter relations)
results/repair/        perturbations with identities enforced, seeds 0-2 (full results)
results/repair_seeds34 same, seeds 3-4 (aggregate "ALL"-family rows only, which are the rows the paper uses)
results/openset/       multi-class open-set baseline (RF, XGBoost; MSP and 1-P(benign) scores), 5 seeds
results/milp/          exact MILP evasion search on RF and XGBoost (5 seeds, up to 200 flows per family, eps 0.2 and 0.5)
results/label_conflicts.json  flows removed by the label-conflict step (code/count_label_conflicts.py)
results/temporal/      temporal split, seed 0, RF/XGBoost/1D-CNN (FT-Transformer not run: CPU-only runtime)
paper/                 paper-build script: turns the result files into the paper text and tables
analysis/              outputs of analyze.py (summary.json, CSV tables, tables.tex, figures)
```

## Data

The dataset is the improved CIC-IDS2017 release of Liu et al. (IEEE CNS 2022), which builds on Engelen et al. (IEEE SPW 2021). It is not redistributed here. Download it from the authors:
https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip

Unzip the five daily CSV files into one folder, for example `data/`.

## Environment

The runs used Google Colab with an NVIDIA Tesla T4, Python 3.13.16, PyTorch 2.11.0 (+cu130), XGBoost 3.4.1, scikit-learn, pandas, NumPy, SciPy and matplotlib. The versions are recorded in every `run_log.txt`.

```
pip install -r requirements.txt
```

## Reproducing the results

```bash
# main experiment: in-distribution, perturbations, mixup, LOAO (about 76 min on a T4)
python code/experiment.py --data data --out results/main --seeds 0,1,2,3,4

# derived-feature identity audit
python code/experiment.py --data data --out results/audit --seeds 0,1,2 --models "" --skip_loao --audit_only

# perturbations with identities enforced (repair)
python code/experiment.py --data data --out results/repair --seeds 0,1,2,3,4 --skip_loao --repair

# anomaly baselines, five-pass SimBA, epsilon scale
python code/experiment.py --data data --out results/supplementary --seeds 0,1,2,3,4 --supp

# tree-aware coordinate search on RF and XGBoost
python code/experiment.py --data data --out results/tree_attack --seeds 0,1,2,3,4 --supp2

# multi-class open-set baseline (CPU is enough; about 50 min on 2 vCPUs)
python code/experiment.py --data data --out results/openset --seeds 0,1,2,3,4 --openset

# temporal split: earliest 70% of each family for training, latest 30% for testing
python code/experiment.py --data data --out results/temporal --seeds 0 --split temporal --models rf,xgb,cnn

# exact MILP evasion search on the tree ensembles (optional; needs scipy with HiGHS)
python code/experiment.py --data data --out results/milp --seeds 0,1,2,3,4 --milp --milp_n 200 --milp_eps 0.2,0.5 --milp_models xgb,rf

# flows removed by the label-conflict step
python code/count_label_conflicts.py data results/label_conflicts.json

# statistics, tables and figures
python code/analyze.py results/main/results.csv analysis results/tree_attack/results_supp2.csv

# paper text and tables from all result files
sh paper/make_layout.sh build && python paper/build_paper.py build paper.json
```

Notes:
- The experiments were run with successive revisions of `experiment.py`. Each revision added a mode (`--repair`/`--audit_only`, then `--supp`, then `--supp2`, then `--openset`, `--split temporal` and `--milp`). The last revision also maps CSE-CIC-IDS2018 labels to the same families and keeps flow timestamps for the temporal split; on CIC-IDS2017 the default (main) mode is functionally unchanged, so the main results are reproducible with the included final revision.
- The open-set and temporal runs used a CPU-only Colab runtime (2 vCPUs, Python 3.13.18, PyTorch 2.11.0+cpu); their run logs record the versions.
- The repair run was executed as seeds 0-2 and then seeds 3-4. For seeds 3-4 only the aggregate rows were kept.

## AI-use disclosure

An AI-based research assistant helped draft and edit parts of the paper, and wrote and ran the experiment code under the author's direction. The author designed the study, checked the code, data, analyses and conclusions, and takes full responsibility for the content.

## License

The code is released under the MIT License (see `LICENSE`). The result files are released under the same terms. The CIC-IDS2017 data remain under the terms of their original providers.

## Citation

A preprint will be linked here. Until then, please cite the repository:

```
A. Zaro, "Adversarial Robustness Is Not Zero-Day Generalization: A Controlled Evaluation of
Flow-Based Intrusion Detectors on Corrected CIC-IDS2017," code and data, 2026.
https://github.com/abdulhamid-zaro/robustness-vs-zeroday-nids
```
