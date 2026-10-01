# Sydney Housing Price Prediction and Decision Support System
SIT307 Machine Learning — Task 8.1D (ML Mini Project)

This project predicts the sale price of residential properties in three Sydney suburbs with very
different markets: **Mount Druitt**, **Cabramatta** and **Vaucluse**. It is trained on 101 sold
listings collected manually from realestate.com.au and domain.com.au (May 2025 – Sep 2026), and
includes a Streamlit web app.

**Live app:** _add your Streamlit Community Cloud link here_

## Contents
| File / folder | What it is |
|---|---|
| `Sydney_Housing_Data.xlsx` | The collected dataset: 101 sold properties, 25 columns including agent descriptions and data-entry notes |
| `House listing.txt` | The raw text copied from the listing pages, used to build the spreadsheet |
| `Task 8.1D.ipynb` | The full analysis for Parts 1–5: data collection, EDA, feature engineering, models, error analysis and deployment |
| `housing_features.py` | The shared feature-engineering transformer, used by both the notebook **and** the app |
| `sydney_housing_engineered.csv` | The engineered features exported in Part 2 |
| `app/app.py` | The Streamlit web app |
| `app/model.joblib` | The trained random forest pipeline, saved by the notebook |
| `app/train_model.py` | Re-trains `model.joblib` from the spreadsheet if needed |
| `app/screenshots/` | Screenshots of the app used in the report |
| `requirements.txt` | Python packages needed by the app |

## Run the app locally
```bash
pip install -r requirements.txt
streamlit run app/app.py
```
The app opens at http://localhost:8501. To get a prediction, fill in the boxes and click **Estimate price**.

## Reproduce the analysis
```bash
pip install -r requirements.txt matplotlib seaborn jupyter
```
Open `Task 8.1D.ipynb` from this folder and choose **Kernel → Restart & Run All**. The nested
cross-validation in Part 3 takes about 3–5 minutes. Running the notebook re-creates
`sydney_housing_engineered.csv` and `app/model.joblib`.
