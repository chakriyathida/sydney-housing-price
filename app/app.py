"""Sydney Housing Price Estimator - Streamlit app (SIT307 Task 8.1D, Part 5).

Run from the project folder:
    pip install -r requirements.txt
    streamlit run app/app.py
"""
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from housing_features import PROPERTY_TYPES, RAW_COLUMNS, SUBURBS, HOUSE_TYPES  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
DATA_PATH = ROOT / "Sydney_Housing_Data.xlsx"

st.set_page_config(page_title="Sydney Housing Price Estimator", page_icon="🏠", layout="centered",
                   initial_sidebar_state="collapsed")

# Open Sans everywhere (loaded from Google Fonts)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
html, body, [class*="st-"], .stApp, .stMarkdown, button, input, textarea, select, label, p, h1, h2, h3,
div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
    font-family: 'Open Sans', sans-serif !important;
}
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:  # missing file or incompatible scikit-learn version -> retrain from the data
        from train_model import train
        return train(DATA_PATH, MODEL_PATH)


bundle = load_model()
model = bundle["pipeline"]


def predict(row):
    """Point prediction and 80% range (in dollars) for one property."""
    df = pd.DataFrame([row])
    log_pred = model.predict(df[RAW_COLUMNS])[0]
    lo, hi = bundle["interval"]["House" if row["Type"] in HOUSE_TYPES else "Strata"]
    return np.exp(log_pred), np.exp(log_pred + lo), np.exp(log_pred + hi)


st.title("Sydney Housing Price Estimator")

with st.form("property"):
    c1, c2 = st.columns(2)
    suburb = c1.selectbox("Suburb", SUBURBS, index=1)
    ptype = c2.selectbox("Property type", PROPERTY_TYPES, index=0)
    beds = c1.number_input("Bedrooms", 1, 12, 3)
    baths = c2.number_input("Bathrooms", 1, 6, 2)
    cars = c1.number_input("Car spaces", 0, 10, 1)
    land = c2.number_input("Land size (m²) — enter 0 for units or if unknown", 0, 3000, 550, step=10)
    sale_date = c1.date_input("Sale date", value=date.today())
    multi = c2.selectbox("Multi-dwelling / development site sale?", ["No", "Yes"])

    st.markdown("**Features**")
    f = st.columns(3)
    names = ["Pool", "Air Con", "Renovated", "Views", "Study", "Outdoor Space"]
    flags = {name: f[i % 3].checkbox(name) for i, name in enumerate(names)}

    desc = st.text_area("Agent description (optional)", height=110,
                        placeholder="Paste the listing description to improve the estimate")
    submitted = st.form_submit_button("Estimate price", type="primary")

if submitted:
    row = {"Suburb": suburb, "Type": ptype, "Bedrooms": beds, "Bathrooms": baths, "Cars": cars,
           "Land_m2": land if (land > 0 and ptype in HOUSE_TYPES) else np.nan,
           "SaleDate": pd.Timestamp(sale_date), **{k: "Yes" if v else "No" for k, v in flags.items()},
           "Description": desc, "MultiDwelling": int(multi == "Yes")}
    price, low, high = predict(row)
    st.metric("Estimated sale price", f"${price:,.0f}")
    st.caption(f"Likely range (80%): ${low:,.0f} – {high:,.0f}")
