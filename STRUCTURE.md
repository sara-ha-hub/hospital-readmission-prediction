# Project Structure

```
hospital-readmission-prediction/
├── data/
│   ├── raw/                    # diabetic_data.csv, IDS_mapping.csv (untouched)
│   └── processed/              # cleaned dataset after prep
├── notebooks/
│   └── 01_eda.ipynb            # exploration + modeling story
├── scripts/
│   ├── data_prep.py            # cleaning + define 30-day target
│   └── pipeline.py             # train/evaluate the 3 models
├── models/                     # saved trained model files
├── app/
│   └── app.py                  # Streamlit risk-score app
├── reports/
│   └── figures/                # ROC curve, confusion matrix, etc.
├── requirements.txt
└── README.md
```
