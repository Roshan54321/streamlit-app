"""
PRT661 - Greater Darwin Property-Price Forecasting
Assessment 3 - Live Dashboard / Demo App

Run with:  streamlit run app.py
Expects to be run from the project's `code/` folder (same place as main.ipynb),
so that ../data/output/ contains the files saved by the notebook's Section 15.

Required files from ../data/output/:
  - model_final.joblib
  - model_final_metadata.json
  - model_evaluation_metrics_baseline.csv
  - model_evaluation_metrics_v2.csv
  - backtest_pooled_metrics.csv
  - predictions_split_A.csv
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("output")

st.set_page_config(
    page_title="Greater Darwin Property-Price Forecasting",
    page_icon="🏠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_bundle():
    path = OUTPUT_DIR / "model_final.joblib"
    if not path.exists():
        print(f"Warning: model bundle not found at {path}. Run the notebook first to generate it.")
        return None
    return joblib.load(path)


@st.cache_data
def load_metadata():
    path = OUTPUT_DIR / "model_final_metadata.json"
    if not path.exists():
        return {}
    return json.load(open(path))


@st.cache_data
def load_csv(name):
    path = OUTPUT_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


bundle = load_model_bundle()
metadata = load_metadata()
baseline_metrics = load_csv("model_evaluation_metrics_baseline.csv")
v2_metrics = load_csv("model_evaluation_metrics_v2.csv")
backtest_metrics = load_csv("backtest_pooled_metrics.csv")
predictions = load_csv("predictions_split_A.csv")

MISSING_FILES = bundle is None or any(
    df is None for df in [baseline_metrics, v2_metrics, backtest_metrics, predictions]
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏠 Greater Darwin Property-Price Forecasting")
st.caption("PRT661 - Data Science Practice | Group Dan1-Theme2 | Theme 2 - Predictive Analytics and Forecasting")

if MISSING_FILES:
    st.error(
        "Some expected files were not found in `../data/output/`. "
        "Run the full notebook first (through Section 15) so the model and "
        "metrics are saved, then restart this app."
    )
    st.stop()

tab_overview, tab_performance, tab_predict, tab_limitations = st.tabs(
    ["📋 Overview", "📊 Model Performance", "🔮 Try a Prediction", "⚠️ Limitations"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Overview
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Project Objective")
    st.write(
        "Predict the sale price of a Greater Darwin property (primary task - regression), "
        "as the foundation for a secondary affordability decision-support layer."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Model", metadata.get("model_name", "—"))
    col2.metric("Feature Set", metadata.get("feature_set", "—"))
    col3.metric("Pooled Backtest RMSE", f"${metadata.get('pooled_backtest_rmse', 0):,.0f}")
    col4.metric("Pooled Backtest MAPE", f"{metadata.get('pooled_backtest_mape', 0):.1f}%")

    st.info(
        "The RMSE/MAPE above come from a **12-fold rolling-origin backtest** "
        "(forecast one quarter ahead, using only earlier sales each time) - "
        "not a single lucky train/test split. This is the honest expected "
        "error for the deployed model."
    )

    st.subheader("Data Sources")
    st.markdown(
        """
        - **Property sales** - Homely, scraped, ~15,000+ Greater Darwin sales
        - **ABS Estimated Resident Population** - Greater Darwin, annual
        - **NT Annual Household Income** - NT Treasury accounts, annual
        - **RBA Cash Rate** - full change history, joined by sale date
        - **Darwin CPI** - quarterly, lagged one quarter (no look-ahead)
        """
    )

    st.subheader("Model Development Journey")
    st.markdown(
        """
        1. **Baseline** models (Linear Regression, Random Forest, XGBoost) on cleaned data
        2. **Diagnostics** found all three models under-predicting 2025 sales specifically
        3. **Improvement round** - added price momentum, interest-rate direction, recency-weighted training
        4. **Policy testing** - tested whether first-home-buyer policy changes had a measurable effect
          (bootstrap-tested, not just eyeballed)
        5. **Final selection** - chosen by rolling-origin backtest across 12 quarters, not one split
        """
    )

# ---------------------------------------------------------------------------
# TAB 2 — Model Performance
# ---------------------------------------------------------------------------
with tab_performance:
    st.subheader("Baseline vs Improved Model Comparison")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline (v1)**")
        st.dataframe(baseline_metrics.style.format(
            {"RMSE": "{:,.0f}", "MAE": "{:,.0f}", "R2": "{:.3f}", "MAPE": "{:.1f}"}
        ), use_container_width=True)
    with col2:
        st.markdown("**Improved (v2 - momentum + rate direction + recency weighting)**")
        st.dataframe(v2_metrics.style.format(
            {"RMSE": "{:,.0f}", "MAE": "{:,.0f}", "R2": "{:.3f}", "MAPE": "{:.1f}"}
        ), use_container_width=True)

    st.subheader("RMSE Comparison - Baseline vs Improved")
    cmp = baseline_metrics[["Model", "RMSE"]].merge(
        v2_metrics[["Model", "RMSE"]], on="Model", suffixes=("_baseline", "_improved")
    )
    fig_cmp = go.Figure()
    fig_cmp.add_bar(name="Baseline", x=cmp["Model"], y=cmp["RMSE_baseline"], marker_color="#94A3B8")
    fig_cmp.add_bar(name="Improved", x=cmp["Model"], y=cmp["RMSE_improved"], marker_color="#2E5090")
    fig_cmp.update_layout(barmode="group", yaxis_title="RMSE (AUD)")
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.subheader("Final Model - Pooled Rolling-Origin Backtest")
    st.caption("12 quarterly folds, forecasting one quarter ahead using only earlier sales each time")
    st.dataframe(backtest_metrics.style.format(
        {"RMSE": "{:,.0f}", "MAE": "{:,.0f}", "R2": "{:.3f}", "MAPE": "{:.1f}"}
    ), use_container_width=True)

    st.subheader("Predicted vs Actual (Final Model, Test Set)")
    final_key = metadata.get("model_name", "XGBoost").lower().replace(" ", "_")
    pred_col = f"pred_v2_{final_key}"
    if pred_col in predictions.columns:
        fig_scatter = px.scatter(
            predictions, x="price_numeric", y=pred_col,
            labels={"price_numeric": "Actual Price (AUD)", pred_col: "Predicted Price (AUD)"},
            opacity=0.5, color_discrete_sequence=["#2E5090"],
        )
        max_val = max(predictions["price_numeric"].max(), predictions[pred_col].max())
        fig_scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                               line=dict(color="red", dash="dash"))
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning(f"Column {pred_col} not found in predictions file.")

# ---------------------------------------------------------------------------
# TAB 3 — Try a Prediction (simplified live predictor)
# ---------------------------------------------------------------------------
with tab_predict:
    st.subheader("🔮 Estimate a Property Price")
    st.caption(
        "Adjust the property details below. Market-wide factors (interest rates, "
        "CPI, recent price momentum, population) are held at their most recent "
        "known values from the training data, since those change for the whole "
        "market, not for one property."
    )

    numeric_cols = metadata.get("numeric_cols", [])
    categorical_cols = metadata.get("categorical_cols", [])

    # Pull sensible defaults / ranges from the training predictions file where possible
    suburb_options = sorted(predictions["suburb"].dropna().unique().tolist()) if "suburb" in predictions.columns else []

    col1, col2, col3 = st.columns(3)
    with col1:
        beds = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
        baths = st.number_input("Bathrooms", min_value=0, max_value=6, value=2)
        cars = st.number_input("Car spaces", min_value=0, max_value=6, value=2)
    with col2:
        land_area_m2 = st.number_input("Land area (m²)", min_value=0, max_value=5000, value=700)
        distance_from_cbd_km = st.slider("Distance from Darwin CBD (km)", 0.0, 40.0, 10.0, 0.5)
    with col3:
        suburb = st.selectbox("Suburb", suburb_options) if suburb_options else st.text_input("Suburb")
        property_type = st.selectbox("Property type", ["house", "unit", "townhouse"])

    if st.button("Predict Price", type="primary"):
        # Build a single-row frame using the model's expected columns.
        # Simple property fields come from the form; everything else (macro
        # context, suburb target encoding, momentum) is filled from the most
        # recent row in the training predictions as a reasonable "today" proxy.
        row = {}
        reference_row = predictions.iloc[-1] if len(predictions) else None

        for col in numeric_cols:
            if col == "beds":
                row[col] = beds
            elif col == "baths":
                row[col] = baths
            elif col == "cars":
                row[col] = cars
            elif col == "land_area_m2":
                row[col] = land_area_m2
            elif col == "distance_from_cbd_km":
                row[col] = distance_from_cbd_km
            elif reference_row is not None and col in reference_row.index:
                row[col] = reference_row[col]
            else:
                row[col] = np.nan

        for col in categorical_cols:
            if col == "suburb":
                row[col] = suburb
            elif col == "property_type_from_address":
                row[col] = property_type
            else:
                row[col] = reference_row[col] if reference_row is not None and col in reference_row.index else None

        input_df = pd.DataFrame([row])[numeric_cols + categorical_cols]

        try:
            pred_log = bundle["pipeline"].predict(input_df)
            pred_price = pred_log[0]
            mape = metadata.get("pooled_backtest_mape", 12)
            low, high = pred_price * (1 - mape / 100), pred_price * (1 + mape / 100)

            st.success(f"### Estimated Price: ${pred_price:,.0f}")
            st.caption(
                f"Approximate range (±{mape:.0f}% MAPE from backtest): "
                f"${low:,.0f} - ${high:,.0f}"
            )
        except Exception as e:
            st.error(f"Could not generate a prediction with the current inputs: {e}")
            st.caption(
                "This usually means a feature the model expects (e.g. a macro "
                "column) could not be filled from the reference row. Check "
                "that predictions_split_A.csv has all the columns listed in "
                "model_final_metadata.json."
            )

# ---------------------------------------------------------------------------
# TAB 4 — Limitations
# ---------------------------------------------------------------------------
with tab_limitations:
    st.subheader("Known Limitations")
    st.markdown(
        """
        - **No new-vs-established flag** for properties
        - **NT-wide, not suburb-level** household income (annual accounts don't break down by suburb)
        - **Annual macro series joined on calendar year** - a mild look-ahead for the earliest sale year
        - **Darwin CPI (inflation) included**; **unemployment was considered but not included** -
          no suitable Greater Darwin / NT-level unemployment series was found at the right
          granularity and lag, so it was left out rather than forced in with a mismatched proxy
        - **Policy effects are confounded** with other 2025 market conditions - the model-implied
          effect is an association, not a proven causal estimate
        - **Sales after the end of the household-income series are excluded**
        - **No untouched holdout** after the final refit - the pooled backtest error is the
          honest expected-error estimate for the deployed model, not a fresh test score
        """
    )
    st.caption(
        "These limitations are logged throughout the notebook's audit trail "
        "(Section 16) as they were discovered, not written up after the fact."
    )
