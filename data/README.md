# Dataset folder

Put the genuine CSV files here, unchanged and with these exact names. They are not
committed to the repository: `scripts/fetch_datasets.py` downloads them or prints the
exact steps for obtaining each one. The application validates every file against its
documented schema and refuses to train on a file that does not match. It never
substitutes another dataset.

| Dataset | File name | Source |
|---|---|---|
| Heart Failure Prediction (918 rows) | `heart.csv` | https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction |
| Pima Indians Diabetes (768 rows) | `diabetes.csv` (downloaded automatically) | Original UCI distribution via pinned mirror, SHA-256 verified by `scripts/fetch_datasets.py`; also on https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database |
| Framingham CHD (~4,240 rows) | `framingham.csv` | Framingham teaching extract on Kaggle, e.g. https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset |
| Stroke Prediction (5,110 rows) | `healthcare-dataset-stroke-data.csv` | https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset |
| Breast Cancer Wisconsin Diagnostic (569 rows) | *(none needed)* | bundled in scikit-learn (`load_breast_cancer`) |

Fetch and check the files with:

```powershell
python scripts\fetch_datasets.py   # downloads what it can, prints manual Kaggle steps otherwise
python scripts\check_datasets.py
```

Respect each dataset's licence and terms of use. These are public research
datasets, not data from the hospitals simulated in the application.
