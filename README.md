# Adversarial Robustness Is Not Zero-Day Generalization

[![arXiv](https://img.shields.io/badge/arXiv-2611.00713-b31b1b.svg)](https://arxiv.org/abs/2611.00713)

Code and per-seed results for the paper *Adversarial Robustness Is Not Zero-Day Generalization: A Controlled Evaluation of Flow-Based Intrusion Detectors on Corrected CIC-IDS2017* (Abdulhamid Zaro, Hebron University).

Preprint: [arXiv:2611.00713](https://arxiv.org/abs/2611.00713) (DOI [10.48550/arXiv.2611.00713](https://doi.org/10.48550/arXiv.2611.00713)).

## Data

Improved CIC-IDS2017 release (Liu et al., IEEE CNS 2022), not redistributed here:
https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip

Unzip the daily CSV files into `data/`.

## Run

```bash
pip install -r requirements.txt

# main experiment (robustness and leave-one-attack-family-out)
python code/experiment.py --data data --out results/main --seeds 0,1,2,3,4

# statistics, tables and figures
python code/analyze.py results/main/results.csv analysis results/tree_attack/results_supp2.csv
```

The other result folders come from the same script with `--supp`, `--supp2`, `--repair`, `--audit_only`, `--openset`, `--split temporal` and `--milp`.

## CSE-CIC-IDS2018

Improved CSE-CIC-IDS2018 release from the same page (https://intrusion-detection.distrinet-research.be/CNS2022/). It is too large to load whole, so we take a uniform per-family sample first:

```bash
python code/prep18.py CSECICIDS2018_improved.zip data18
python code/experiment.py --data data18 --out results/cic2018 --seeds 0,1,2 --models rf,xgb,cnn
```

## Citation

```bibtex
@misc{zaro2026robustness,
  title         = {Adversarial Robustness Is Not Zero-Day Generalization: A Controlled Evaluation of Flow-Based Intrusion Detectors on Corrected {CIC-IDS2017}},
  author        = {Zaro, Abdulhamid},
  year          = {2026},
  eprint        = {2611.00713},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  doi           = {10.48550/arXiv.2611.00713},
  url           = {https://arxiv.org/abs/2611.00713}
}
```

## License

MIT. The datasets remain under the terms of their original providers.
