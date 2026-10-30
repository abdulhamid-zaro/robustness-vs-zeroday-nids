# Adversarial Robustness Is Not Zero-Day Generalization

Code and per-seed results for the paper *Adversarial Robustness Is Not Zero-Day Generalization: A Controlled Evaluation of Flow-Based Intrusion Detectors on Corrected CIC-IDS2017* (Abdulhamid Zaro, Hebron University).

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

## License

MIT. The CIC-IDS2017 data remain under the terms of their original providers.
