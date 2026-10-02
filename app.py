"""
PRT661 - Greater Darwin Property-Price Forecasting
Assessment 3 - Live dashboard

Run with:  streamlit run app.py
Run from the project's `code/` folder (next to main.ipynb). It reads the files
the notebook saves in Sections 15-16 from ../data/output/.

Required files:
  model_final.joblib, model_final_metadata.json, app_reference.json,
  model_evaluation_metrics_baseline.csv, model_evaluation_metrics_v2.csv,
  backtest_pooled_metrics.csv, predictions_split_A.csv
Optional (extra charts appear if present):
  backtest_final_by_fold.csv, feature_importance_final.csv
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

OUTPUT_DIR = Path("output")
STRESS_LINE = 30.0        # common "30% of gross income" housing-stress benchmark
MIN_SUBURB_SALES = 10     # same rule the model uses before a suburb becomes "Other"
NAVY, GREY, RED, GREEN = "#1B2A4A", "#94A3B8", "#C0392B", "#2E7D5B"

st.set_page_config(page_title="Greater Darwin Property-Price Forecasting",
                   page_icon="🏠", layout="wide")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    path = OUTPUT_DIR / "model_final.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data
def load_json(name):
    path = OUTPUT_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_csv(name):
    path = OUTPUT_DIR / name
    return pd.read_csv(path) if path.exists() else None


bundle = load_bundle()
metadata = load_json("model_final_metadata.json")
reference = load_json("app_reference.json")
baseline = load_csv("model_evaluation_metrics_baseline.csv")
v2 = load_csv("model_evaluation_metrics_v2.csv")
pooled = load_csv("backtest_pooled_metrics.csv")
preds = load_csv("predictions_split_A.csv")
by_fold = load_csv("backtest_final_by_fold.csv")
importance = load_csv("feature_importance_final.csv")

required = [bundle, metadata, reference, baseline, v2, pooled, preds]
st.title("🏠 Greater Darwin Property-Price Forecasting")
st.caption("PRT661 Data Science Practice · Group Dan1-Theme2 · Theme 2: Predictive Analytics and Forecasting")
if any(x is None for x in required):
    st.error("Some files are missing from `../data/output/`. Run the notebook through Section 16, then restart.")
    st.stop()

FINAL_MODEL = metadata["model_name"]
FINAL_KEY = FINAL_MODEL.lower().replace(" ", "_")
final_row = pooled[(pooled["model"] == FINAL_MODEL) & (pooled["feature_set"] == metadata["feature_set"])].iloc[0]
MARKET = reference["market_context"]
SUBURBS = reference["suburbs"]
P10, P90 = reference["ratio_q"]["p10"], reference["ratio_q"]["p90"]


# ---------------------------------------------------------------------------
# Prediction and affordability helpers
# ---------------------------------------------------------------------------
def build_input(suburb_name, prop_type, beds, baths, cars, land):
    """One model-input row. Property fields come from the user and the suburb's
    median profile; market-wide fields are frozen at the latest training values."""
    s = SUBURBS[suburb_name]
    values = dict(MARKET)
    values.update({"beds": beds, "baths": baths, "cars": cars, "land_area_m2": land,
                   "distance_from_cbd_km": s["distance_from_cbd_km"],
                   "latitude": s["latitude"], "longitude": s["longitude"],
                   "suburb_target_enc": s["suburb_target_enc"],
                   "suburb_price_momentum_6m": s["suburb_price_momentum_6m"],
                   "suburb": s["model_suburb"], "property_type_from_address": prop_type})
    cols = metadata["numeric_cols"] + metadata["categorical_cols"]
    return {c: values.get(c, np.nan) for c in cols}


def predict_prices(rows):
    frame = pd.DataFrame(rows)[metadata["numeric_cols"] + metadata["categorical_cols"]]
    return bundle["pipeline"].predict(frame)


def monthly_repayment(loan, annual_rate_pct, years):
    r, n = annual_rate_pct / 100 / 12, years * 12
    return loan / n if r == 0 else loan * r / (1 - (1 + r) ** -n)


def rti(price, deposit_pct, rate_pct, years, income):
    """Repayment-to-income: yearly mortgage repayments as % of gross household income."""
    loan = price * (1 - deposit_pct / 100)
    return monthly_repayment(loan, rate_pct, years) * 12 / income * 100



def show_chart(fig):
    """Works on old and new Streamlit (use_container_width was replaced by width)."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_df(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


tab_over, tab_perf, tab_diag, tab_pred, tab_lim = st.tabs(
    ["📋 Overview", "📊 Model performance", "🔍 Diagnosis → Fix", "🔮 Predict + Affordability", "⚠️ Limitations"])

# ---------------------------------------------------------------------------
# 1. Overview
# ---------------------------------------------------------------------------
with tab_over:
    st.subheader("What this dashboard does")
    st.write("A buyer picks a property. The model estimates its sale price, shows an honest range, "
             "then turns that price into repayments and checks them against the buyer's own income.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final model", FINAL_MODEL)
    c2.metric("Typical error (RMSE)", f"${final_row['RMSE']:,.0f}")
    c3.metric("Typical % miss (MAPE)", f"{final_row['MAPE']:.1f}%")
    c4.metric("Share of price variation explained (R²)", f"{final_row['R2']:.2f}")
    st.info("These come from a **12-quarter rolling backtest**: the model forecasts one quarter ahead using only "
            "earlier sales, 12 times in a row, instead of one lucky train/test split. "
            "After the final refit on all sales there is no untouched test set, so this is the honest expected error.")
    with st.expander("What do RMSE, MAPE and R² mean in plain English?"):
        st.markdown(
            "- **RMSE** - a typical size of the miss in dollars. Big misses count extra.\n"
            "- **MAPE** - the average miss as a percentage of the real price. 11.8% means about $59k off a $500k house.\n"
            "- **R²** - 0 means no better than guessing the average price; 1 means perfect. 0.80 means most, not all, of the price differences are captured.")
    st.subheader("Data sources")
    st.markdown("- **Homely** sold listings (scraped) · **ABS** Estimated Resident Population (annual) · "
                "**NT** household income accounts (annual, NT-wide) · **RBA** cash rate (by sale date) · "
                "**Darwin CPI** (quarterly, lagged one quarter)")
    st.caption(f"Model trained on sales through {reference['trained_through']}. Market context in the predictor is frozen at that date.")

# ---------------------------------------------------------------------------
# 2. Model performance
# ---------------------------------------------------------------------------
fmt = {"RMSE": "{:,.0f}", "MAE": "{:,.0f}", "R2": "{:.3f}", "MAPE": "{:.1f}"}
with tab_perf:
    st.subheader("Baseline vs improved (single test window, Nov 2024 - Dec 2025)")
    a, b = st.columns(2)
    a.markdown("**Baseline (v1)**")
    with a:
        show_df(baseline.style.format(fmt))
    b.markdown("**Improved (v2): price momentum + rate direction + recency weighting**")
    with b:
        show_df(v2.style.format(fmt))
    cmp = baseline[["Model", "RMSE"]].merge(v2[["Model", "RMSE"]], on="Model", suffixes=("_v1", "_v2"))
    fig = go.Figure()
    fig.add_bar(name="Baseline", x=cmp["Model"], y=cmp["RMSE_v1"], marker_color=GREY)
    fig.add_bar(name="Improved", x=cmp["Model"], y=cmp["RMSE_v2"], marker_color=NAVY)
    fig.update_layout(barmode="group", yaxis_title="RMSE (AUD, lower is better)", height=340)
    show_chart(fig)

    st.subheader("Final selection: pooled rolling backtest (12 quarters, forecasting the next quarter each time)")
    show_df(pooled.style.format(fmt))
    st.caption("Backtest RMSE is lower than the single-window RMSE because the 12 folds include calmer quarters. "
               "The two are not like-for-like.")
    if by_fold is not None and "RMSE" in by_fold.columns:
        fig_f = px.bar(by_fold, x="fold", y="RMSE", color_discrete_sequence=[NAVY],
                       labels={"fold": "Backtest fold (quarter forecast)", "RMSE": "RMSE (AUD)"})
        fig_f.update_layout(height=320)
        show_chart(fig_f)
    col = f"pred_v2_{FINAL_KEY}"
    if col in preds.columns:
        st.markdown("**Predicted vs actual price (test window, before the final refit)**")
        fig_s = px.scatter(preds, x="price_numeric", y=col, opacity=0.45, color_discrete_sequence=[NAVY],
                           labels={"price_numeric": "Actual price (AUD)", col: "Predicted price (AUD)"})
        top = float(max(preds["price_numeric"].max(), preds[col].max()))
        fig_s.add_shape(type="line", x0=0, y0=0, x1=top, y1=top, line=dict(color=RED, dash="dash"))
        fig_s.update_layout(height=420)
        show_chart(fig_s)
        st.caption("Points on the red line are perfect. Points below the line are under-predictions.")
    if importance is not None:
        st.markdown("**What the final model leans on most**")
        top12 = importance.head(12).iloc[::-1]
        fig_i = px.bar(top12, x="importance", y="feature", orientation="h", color_discrete_sequence=[NAVY])
        fig_i.update_layout(height=380)
        show_chart(fig_i)
        st.caption("Importance shows what the model uses, not what causes prices to move.")

# ---------------------------------------------------------------------------
# 3. Diagnosis -> Fix
# ---------------------------------------------------------------------------
with tab_diag:
    st.subheader("The 2025 problem, and what closed the gap")
    st.write("At Assessment 2 every model guessed too low for 2025 sales. We added market-movement features "
             "(recent price trend, direction of interest rates, extra weight on recent sales). "
             "Below: average error as a percentage of the real price, by sale year. "
             "**Below zero means the model guessed too low.**")
    model_choice = st.selectbox("Model", ["Linear Regression", "Random Forest", "XGBoost"], index=2)
    key = model_choice.lower().replace(" ", "_")
    c1, c2 = f"pred_v1_{key}", f"pred_v2_{key}"
    if {c1, c2, "sold_date_iso"} <= set(preds.columns):
        d = preds.copy()
        d["year"] = pd.to_datetime(d["sold_date_iso"], utc=True).dt.year
        for c in (c1, c2):
            d[c + "_pct"] = (d[c] - d["price_numeric"]) / d["price_numeric"] * 100
        g = d.groupby("year")[[c1 + "_pct", c2 + "_pct"]].mean().reset_index()
        fig = go.Figure()
        fig.add_bar(name="Before (v1)", x=g["year"].astype(str), y=g[c1 + "_pct"], marker_color=GREY)
        fig.add_bar(name="After (v2)", x=g["year"].astype(str), y=g[c2 + "_pct"], marker_color=NAVY)
        fig.add_hline(y=0, line_color="black")
        fig.update_layout(barmode="group", yaxis_title="Average error (%)", height=380)
        show_chart(fig)
        st.caption("Test sales only (Nov 2024 onward). Random Forest does not improve like the other two; "
                   "that is a real result, not a plotting error.")
    else:
        st.warning("Prediction columns not found in predictions_split_A.csv.")

# ---------------------------------------------------------------------------
# 4. Predict + Affordability
# ---------------------------------------------------------------------------
with tab_pred:
    st.subheader("1 · Choose a property")
    names = sorted(SUBURBS)
    c1, c2, c3 = st.columns(3)
    with c1:
        suburb = st.selectbox("Suburb", names, index=names.index("Fannie Bay") if "Fannie Bay" in names else 0)
        ptype = st.selectbox("Property type", ["House", "Unit", "Townhouse"])
    with c2:
        beds = st.number_input("Bedrooms", 0, 10, 3)
        baths = st.number_input("Bathrooms", 0, 6, 2)
        cars = st.number_input("Car spaces", 0, 6, 2)
    with c3:
        land = st.number_input("Land area (m²)", 0, 5000, 700)
        st.caption(f"Distance to CBD is taken from the suburb's median "
                   f"({SUBURBS[suburb]['distance_from_cbd_km']:.1f} km), not typed in.")
    if SUBURBS[suburb]["n_sales"] < MIN_SUBURB_SALES:
        st.warning(f"{suburb} has only {SUBURBS[suburb]['n_sales']} recorded sales, so the model treats it as 'Other'. "
                   "Treat this estimate as rough.")

    price = float(predict_prices([build_input(suburb, ptype, beds, baths, cars, land)])[0])
    low, high = price * P10, price * P90

    st.subheader("2 · Estimated price and range")
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimate", f"${price:,.0f}")
    m2.metric("Low end (80% range)", f"${low:,.0f}")
    m3.metric("High end (80% range)", f"${high:,.0f}")
    st.caption(f"The range is measured, not guessed: in 12 quarters of real forecasts, about 80% of sales landed "
               f"between ×{P10:.2f} and ×{P90:.2f} of the model's estimate. Roughly 1 in 10 sold even higher, 1 in 10 lower.")

    st.subheader("3 · Can you afford it?")
    default_cash = float(MARKET.get("cash_rate_target", 3.6))
    f1, f2, f3, f4 = st.columns(4)
    income = f1.number_input("Gross household income (AUD / year)", 30_000, 1_000_000, 160_000, step=5_000)
    dep_pct = f2.slider("Deposit (%)", 5, 40, 20)
    margin = f3.slider("Lender margin over cash rate (pts)", 1.0, 4.0, 2.5, 0.1)
    years = f4.selectbox("Loan term (years)", [25, 30], index=1)
    cash = st.number_input("Cash rate used (%)", 0.0, 10.0, round(default_cash, 2), step=0.25,
                           help="Defaults to the RBA cash rate in force at the latest training sale.")
    mort_rate = cash + margin

    loan = price * (1 - dep_pct / 100)
    monthly = monthly_repayment(loan, mort_rate, years)
    ratio = rti(price, dep_pct, mort_rate, years, income)
    ratio_hi = rti(high, dep_pct, mort_rate, years, income)
    a1, a2, a3, a4, a5 = st.columns(5)
    a1.metric("Deposit", f"${price * dep_pct / 100:,.0f}")
    a2.metric("Loan", f"${loan:,.0f}")
    a3.metric("Monthly repayment", f"${monthly:,.0f}")
    a4.metric("Weekly repayment", f"${monthly * 12 / 52:,.0f}")
    a5.metric("Repayment-to-income", f"{ratio:.0f}%", delta=f"{ratio - STRESS_LINE:+.0f} pts vs {STRESS_LINE:.0f}% line",
              delta_color="inverse")
    if ratio > STRESS_LINE:
        st.error(f"Repayments take {ratio:.0f}% of your income: above the {STRESS_LINE:.0f}% housing-stress line.")
    else:
        st.success(f"Repayments take {ratio:.0f}% of your income: under the {STRESS_LINE:.0f}% line. "
                   f"If it sold at the top of the range it would be {ratio_hi:.0f}%.")

    st.markdown("**Interest-rate sensitivity** (same property, same deposit)")
    rates = np.arange(max(mort_rate - 2, 1.0), mort_rate + 3.01, 0.5)
    sens = pd.DataFrame({"Mortgage rate (%)": rates, "Repayment-to-income (%)": [rti(price, dep_pct, r, years, income) for r in rates]})
    fig = px.line(sens, x="Mortgage rate (%)", y="Repayment-to-income (%)", markers=True, color_discrete_sequence=[NAVY])
    fig.add_hline(y=STRESS_LINE, line_dash="dash", line_color=RED, annotation_text="30% stress line")
    fig.update_layout(height=320)
    show_chart(fig)

    st.subheader("4 · Compare suburbs (same property, same finances)")
    eligible = [s for s in names if SUBURBS[s]["n_sales"] >= MIN_SUBURB_SALES]
    est = predict_prices([build_input(s, ptype, beds, baths, cars, land) for s in eligible])
    comp = pd.DataFrame({"Suburb": eligible, "Estimated price": est})
    comp["Repayment-to-income (%)"] = [rti(p, dep_pct, mort_rate, years, income) for p in est]
    comp["Status"] = np.where(comp["Repayment-to-income (%)"] > STRESS_LINE, "Above 30% line", "Within 30% line")
    comp = comp.sort_values("Estimated price")
    fig = px.bar(comp, x="Estimated price", y="Suburb", orientation="h", color="Status",
                 color_discrete_map={"Above 30% line": RED, "Within 30% line": GREEN},
                 hover_data={"Repayment-to-income (%)": ":.0f"}, height=max(400, 18 * len(comp)))
    show_chart(fig)
    st.caption("Suburbs with fewer than 10 recorded sales are left out. All estimates use market conditions frozen at "
               f"{reference['trained_through']}. The estimate excludes stamp duty, mortgage insurance, fees and rates.")

# ---------------------------------------------------------------------------
# 5. Limitations
# ---------------------------------------------------------------------------
with tab_lim:
    st.subheader("Strongest part")
    st.markdown("A leak-aware, time-respecting pipeline: every learned step is fitted on past sales only, "
                "and the final model is chosen by a 12-quarter forward-looking backtest, not one split.")
    st.subheader("Other known limitations")
    st.markdown(
        "- **No untouched test set** after the final refit; the backtest error is the expected error.\n"
        "- **Market conditions are frozen** at the last training sale; the predictor does not know today's rate or CPI.\n"
        "- **Income is NT-wide**, not suburb-level, so it cannot explain differences between suburbs.\n"
        "- **Policy effects are unproven**: the three tested flags gave mixed results and are not in the final model.\n"
        "- **Unemployment is not included**: no suitable NT-level series was found at the right detail and lag.\n"
        "- **Property type** is guessed from the address and agrees with land-area presence only 58-70% of the time.\n"
        "- **One 80% range** is applied to every price; it is not yet calibrated by price band.\n"
        "- **Not benchmarked** against property.com.au, Domain or realestate.com.au.\n"
        "- **Affordability is simplified**: no stamp duty, mortgage insurance, fees, or running costs.")
    st.caption("These are also logged in the notebook's audit trail (Section 17) as they were found.")