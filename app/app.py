# Sydney Housing Price Estimator - Streamlit app (SIT307 Task 8.1D, Part 5)
# Run from the project folder:  streamlit run app/app.py

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Load the model that the notebook saved (Part 5.1)
saved = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
model = saved["model"]
LAND_MEDIANS = saved["land_medians"]
KEYWORDS = saved["keywords"]
YES_NO = ["Pool", "Air Con", "Renovated", "Views", "Study", "Outdoor Space"]


def make_features(data):
    # Same function as Part 2.6 of the notebook
    X = pd.DataFrame(index=data.index)
    X["Cabramatta"] = (data["Suburb"] == "Cabramatta").astype(int)
    X["Vaucluse"] = (data["Suburb"] == "Vaucluse").astype(int)
    X["IsHouse"] = data["Type"].isin(["House", "Semi/Duplex"]).astype(int)
    X["Bedrooms"] = data["Bedrooms"]
    X["Bathrooms"] = data["Bathrooms"]
    X["Cars"] = data["Cars"].fillna(1).clip(upper=6)

    land = data["Land"].fillna(data["Suburb"].map(LAND_MEDIANS))
    land = land.where(X["IsHouse"] == 1, 0)
    X["LogLand"] = np.log1p(land)

    X["Extras"] = (data[YES_NO] == "Yes").sum(axis=1)

    text = data["Description"].fillna("").str.lower()
    for name, words in KEYWORDS.items():
        X[name] = text.str.contains(words).astype(int)
    return X


st.title("Sydney Housing Price Estimator")
st.write("Estimate the sale price of a property in **Mount Druitt**, **Cabramatta** or **Vaucluse**, "
         "based on 101 recent sales.")

with st.form("property"):
    col1, col2 = st.columns(2)
    suburb = col1.selectbox("Suburb", ["Mount Druitt", "Cabramatta", "Vaucluse"])
    prop_type = col2.selectbox("Property type", ["House", "Unit/Apartment", "Townhouse", "Villa", "Semi/Duplex"])
    bedrooms = col1.number_input("Bedrooms", min_value=1, max_value=12, value=3)
    bathrooms = col2.number_input("Bathrooms", min_value=1, max_value=6, value=2)
    cars = col1.number_input("Car spaces", min_value=0, max_value=10, value=1)
    land = col2.number_input("Land size (m²), 0 for units or if unknown", min_value=0, max_value=3000, value=0)

    st.write("**Features**")
    c = st.columns(3)
    ticks = {}
    for i, feature in enumerate(YES_NO):
        ticks[feature] = c[i % 3].checkbox(feature)

    description = st.text_area("Agent description (optional)")
    submitted = st.form_submit_button("Estimate price")

if submitted:
    row = {"Suburb": suburb, "Type": prop_type, "Bedrooms": bedrooms, "Bathrooms": bathrooms,
           "Cars": cars, "Land": land if land > 0 else np.nan, "Description": description}
    for feature in YES_NO:
        row[feature] = "Yes" if ticks[feature] else "No"

    X = make_features(pd.DataFrame([row]))
    price = np.exp(model.predict(X[saved["features"]])[0])   # model predicts log(price)

    st.metric("Estimated sale price", "${:,.0f}".format(price))
    st.caption("For a typical property the model is within about ±{:.0f}% of the real sale price. "
               "Prestige homes (over $10M) and development sites are much harder to predict, "
               "so use this as a starting point, not a valuation.".format(saved["typical_error"]))
