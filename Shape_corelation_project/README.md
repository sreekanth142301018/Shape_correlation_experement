# SHAP Explanations under Feature Correlation: How Faithful are they?

Track 1 – Experimental investigation of an existing XAI approach
N. Venkata Nivas Yadav (142301029) · K. Sreekanth Naik (142301018)

## What this project does
- Builds synthetic data with a known rule `y = 3*x1 + 2*x2 + noise`, with x1–x2 correlation from 0.0 to 0.99 (plus two noise features).
- Trains a Random Forest (10 random seeds per level) and explains it with SHAP (tree-path and interventional) and permutation feature importance.
- Compares the explanations with the known truth, with each other (Spearman), and with a feature-removal test (shuffle vs retrain).
- Repeats the comparison on the scikit-learn Diabetes dataset, including a test where a fake copy of `bmi` is added.

## Folder contents
| Path | Contents |
|---|---|
| `SHAP_Correlation_Project.ipynb` | Main notebook, already executed (all outputs visible) |
| `shap_correlation_experiment.py` | Same code as a plain script |
| `results/` | All numbers as CSV files |
| `figures/` | All 12 figures as PNG |
| `report/` | Full project report (Word + PDF) explaining code, outputs and results |
| `requirements.txt` | Python libraries |

## How to run
**Google Colab:** upload the notebook, add a first cell `!pip install shap`, then Runtime → Run all (about 25–30 min on CPU).

**Your own computer:**
```bash
pip install -r requirements.txt
python shap_correlation_experiment.py      # or open the notebook in Jupyter
```
Set `QUICK_MODE = True` in Part 0 for a fast (<1 min) test run. Do not report quick-mode numbers.

All randomness uses fixed seeds, so a full run reproduces the numbers in the report (tiny differences are possible with other library versions; see `results/environment.json`).
