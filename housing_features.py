"""Shared feature engineering for the Sydney housing price project (SIT307 Task 8.1D).

The same transformer is used in the notebook (training / cross-validation) and in the
Streamlit app (prediction), so a property is always encoded the same way.

Raw input columns expected (one row per property):
    Suburb, Type, Bedrooms, Bathrooms, Cars, Land_m2, SaleDate,
    Pool, Air Con, Renovated, Views, Study, Outdoor Space, Description, MultiDwelling
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SUBURBS = ["Mount Druitt", "Cabramatta", "Vaucluse"]          # Mount Druitt = reference level
PROPERTY_TYPES = ["House", "Semi/Duplex", "Townhouse", "Villa", "Unit/Apartment"]
HOUSE_TYPES = {"House", "Semi/Duplex"}                         # own their land; the rest are strata
YES_NO = ["Pool", "Air Con", "Renovated", "Views", "Study", "Outdoor Space"]
REF_DATE = pd.Timestamp("2025-05-01")                          # start of the collection window

KEYWORDS = {
    "kw_water":    r"harbour|ocean|beach|water view|bay\b|waterfront",
    "kw_luxury":   r"luxur|prestige|architect|designer|bespoke|resort|grand",
    "kw_devpot":   r"develop|granny|dual occ|subdivi|\br4\b|\br3\b|stca|zoning",
    "kw_investor": r"invest|rental|\brent\b|first home",
    "kw_reno":     r"renovat|refurbish|brand new|newly|updated",
}

RAW_COLUMNS = ["Suburb", "Type", "Bedrooms", "Bathrooms", "Cars", "Land_m2", "SaleDate",
               *YES_NO, "Description", "MultiDwelling"]


def _yes(v):
    """Turn Yes/No, True/False or 1/0 into 1/0."""
    if isinstance(v, str):
        return int(v.strip().lower() in {"yes", "y", "true", "1"})
    try:
        return int(bool(v)) if not pd.isna(v) else 0
    except (TypeError, ValueError):
        return 0


def keyword_flags(text):
    """Keyword flags for one agent description (used by the app to show what was detected)."""
    t = (text or "").lower()
    return {k: int(bool(pd.Series([t]).str.contains(p, regex=True).iloc[0])) for k, p in KEYWORDS.items()}


class HousingFeatures(BaseEstimator, TransformerMixin):
    """Raw property records -> numeric model features.

    fit() learns only the imputation values (median house land size per suburb and
    median car spaces per suburb x segment) from the training rows, so nothing leaks
    from the validation folds during cross-validation.
    """

    def __init__(self, interactions=True, max_cars=6, extra_keywords=None):
        self.interactions = interactions
        self.max_cars = max_cars
        self.extra_keywords = extra_keywords      # optional {name: regex} for experiments

    # ---------- helpers ----------
    @staticmethod
    def _base(X):
        X = X.copy()
        X["Suburb"] = X["Suburb"].astype(str)
        X["IsHouse"] = X["Type"].astype(str).isin(HOUSE_TYPES).astype(int)
        for c in ["Bedrooms", "Bathrooms", "Cars", "Land_m2"]:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        return X

    def fit(self, X, y=None):
        X = self._base(X)
        houses = X[X["IsHouse"] == 1]
        self.land_median_ = houses.groupby("Suburb")["Land_m2"].median().to_dict()
        self.land_median_all_ = float(houses["Land_m2"].median())
        self.cars_median_ = X.groupby(["Suburb", "IsHouse"])["Cars"].median().to_dict()
        self.cars_median_all_ = float(X["Cars"].median())
        self.feature_names_ = list(self.transform(X.head(1)).columns)
        return self

    def transform(self, X):
        X = self._base(X)
        out = pd.DataFrame(index=X.index)

        # Location (one-hot, Mount Druitt is the reference)
        for s in SUBURBS[1:]:
            out[f"sub_{s}"] = (X["Suburb"] == s).astype(int)

        # Property type and size
        out["IsHouse"] = X["IsHouse"]
        out["Bedrooms"] = X["Bedrooms"].fillna(2).clip(upper=8)
        out["Bathrooms"] = X["Bathrooms"].fillna(1).clip(upper=6)
        cars_fill = [self.cars_median_.get((s, h), self.cars_median_all_)
                     for s, h in zip(X["Suburb"], X["IsHouse"])]
        out["CarsCapped"] = X["Cars"].fillna(pd.Series(cars_fill, index=X.index)).clip(upper=self.max_cars)

        # Land: strata = 0 m2; missing house land = training median for that suburb (+ flag)
        land_fill = X["Suburb"].map(self.land_median_).fillna(self.land_median_all_)
        out["LandMissing"] = ((X["IsHouse"] == 1) & X["Land_m2"].isna()).astype(int)
        land = np.where(X["IsHouse"] == 1, X["Land_m2"].fillna(land_fill), 0.0)
        out["LogLand"] = np.log1p(land.astype(float))

        # Listed features
        flags = pd.DataFrame({c: X[c].map(_yes) if c in X else 0 for c in YES_NO}, index=X.index)
        out["AmenityScore"] = flags.sum(axis=1)
        out["Pool"] = flags["Pool"]
        out["Views"] = flags["Views"]

        # Time (years since the start of the collection window)
        dates = pd.to_datetime(X["SaleDate"], errors="coerce").fillna(pd.Timestamp.today().normalize())
        out["YearsSinceStart"] = (dates - REF_DATE).dt.days / 365.25

        # Unusual sale type
        out["MultiDwelling"] = X["MultiDwelling"].map(_yes) if "MultiDwelling" in X else 0

        # Text features from the agent description
        text = X["Description"].fillna("").astype(str).str.lower() if "Description" in X else pd.Series("", index=X.index)
        for k, pat in {**KEYWORDS, **(self.extra_keywords or {})}.items():
            out[k] = text.str.contains(pat, regex=True).astype(int)

        # Suburb x size/type interactions (lets a linear model give each suburb its own slopes)
        if self.interactions:
            for s in SUBURBS[1:]:
                out[f"sub_{s}_x_IsHouse"] = out[f"sub_{s}"] * out["IsHouse"]
                out[f"sub_{s}_x_Bedrooms"] = out[f"sub_{s}"] * out["Bedrooms"]
        return out.astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


def load_raw(path="Sydney_Housing_Data.xlsx"):
    """Read the collected spreadsheet and return (raw feature frame, price, full frame)."""
    df = pd.read_excel(path, sheet_name="Sold Properties")
    df = df.rename(columns={"Sale Price ($)": "Price", "Land Size (m²)": "Land_m2", "Car Spaces": "Cars",
                            "Property Type": "Type", "Sale Date": "SaleDate", "Agent Description": "Description"})
    df["MultiDwelling"] = df["Notes"].fillna("").str.contains("multi-lot|multi-dwelling", case=False).astype(int)
    return df[RAW_COLUMNS], df["Price"].astype(float), df
