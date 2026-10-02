"""
PRT661 - Greater Darwin Property-Price Forecasting - live dashboard

Run from the project's code/ folder:  streamlit run app.py
Reads ../data/output/. Needs the notebook's Section 15 outputs PLUS the Section 15.1
export cell (notebook_cell_15b_dashboard_export.py).
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

OUT = Path("output")
st.set_page_config(page_title="Greater Darwin Property-Price Forecasting", page_icon="🏠", layout="wide")


@st.cache_resource
def load_bundle():
    p = OUT / "model_final.joblib"
    return joblib.load(p) if p.exists() else None


@st.cache_data
def load_json(name):
    p = OUT / name
    return json.load(open(p)) if p.exists() else None


@st.cache_data
def load_csv(name):
    p = OUT / name
    return pd.read_csv(p) if p.exists() else None


bundle = load_bundle()
meta = load_json("model_final_metadata.json") or {}
ref = load_json("app_reference.json")
base_m, v2_m = load_csv("model_evaluation_metrics_baseline.csv"), load_csv("model_evaluation_metrics_v2.csv")
backtest = load_csv("backtest_pooled_metrics.csv")
by_fold = load_csv("backtest_final_by_fold.csv")
preds = load_csv("predictions_split_A.csv")
importance = load_csv("feature_importance_final.csv")
policy = load_csv("policy_verdict.csv")

required = {"model_final.joblib": bundle, "app_reference.json": ref, "baseline metrics": base_m,
            "v2 metrics": v2_m, "backtest_pooled_metrics.csv": backtest, "predictions_split_A.csv": preds}
missing = [k for k, v in required.items() if v is None]

st.title("🏠 Greater Darwin Property-Price Forecasting")
st.caption("PRT661 Data Science Practice | Group Dan1-Theme2 | Theme 2 - Predictive Analytics and Forecasting")
if missing:
    st.error(f"Missing in `../data/output/`: {', '.join(missing)}. Run the notebook through Section 15, "
             "then the Section 16 export cell, and restart the app.")
    st.stop()

MODEL = meta.get("model_name", "XGBoost")
KEY = MODEL.lower().replace(" ", "_")
tab_over, tab_perf, tab_story, tab_pred, tab_lim = st.tabs(
    ["📋 Overview", "📊 Performance", "🔍 Diagnosis → Fix", "🔮 Predict + Affordability", "⚠️ Limitations"])

# ---------------------------------------------------------------- Overview
with tab_over:
    c = st.columns(4)
    c[0].metric("Final model", MODEL)
    c[1].metric("Backtest RMSE", f"${meta.get('pooled_backtest_rmse', 0):,.0f}")
    c[2].metric("Backtest MAPE", f"{meta.get('pooled_backtest_mape', 0):.1f}%")
    c[3].metric("Trained through", str(meta.get("trained_through", "-")))
    st.info("Error figures come from a 12-fold rolling-origin backtest (each fold forecasts the next quarter "
            "from earlier sales only), not from one lucky split. No untouched holdout exists after the final refit.")
    st.markdown("""
**How the project got here - each step caused by the previous one**
1. **Baseline** (3 models, leak-free temporal split) → all three plateau at R² 0.42-0.56.
2. **Diagnosis** → every model under-predicts 2025 by roughly the same amount; not a tree-ceiling effect.
3. **Fix** → add market *movement* (price momentum, rate direction, recency weights) → RMSE falls for Linear Regression and XGBoost.
4. **Robustness** → repeat at 4 cutoffs, then a 12-fold backtest → XGBoost wins pooled.
5. **Policy hypothesis** → bootstrap-tested; mixed evidence, so policy flags are *not* in the deployed model.
""")

# ---------------------------------------------------------------- Performance
with tab_perf:
    fmt = {"RMSE": "{:,.0f}", "MAE": "{:,.0f}", "R2": "{:.3f}", "MAPE": "{:.1f}"}
    a, b = st.columns(2)
    a.markdown("**Baseline (v1) - single split**")
    a.dataframe(base_m.style.format(fmt), use_container_width=True)
    b.markdown("**Improved (v2) - momentum + rate direction + recency, re-tuned**")
    b.dataframe(v2_m.style.format(fmt), use_container_width=True)

    cmp = base_m[["Model", "RMSE"]].merge(v2_m[["Model", "RMSE"]], on="Model", suffixes=("_v1", "_v2"))
    fig = go.Figure([go.Bar(name="v1", x=cmp["Model"], y=cmp["RMSE_v1"], marker_color="#94A3B8"),
                     go.Bar(name="v2", x=cmp["Model"], y=cmp["RMSE_v2"], marker_color="#2E5090")])
    fig.update_layout(barmode="group", yaxis_title="Test RMSE (AUD)", title="Same test set: v1 vs v2")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Pooled rolling-origin backtest - basis for the final model choice**")
    st.dataframe(backtest.style.format(fmt), use_container_width=True)

    if by_fold is not None:
        fig = px.bar(by_fold, x="fold", y="RMSE", title=f"{MODEL}: RMSE by forecast quarter",
                     color_discrete_sequence=["#2E5090"])
        st.plotly_chart(fig, use_container_width=True)
    if importance is not None:
        top = importance.head(12).iloc[::-1]
        st.plotly_chart(px.bar(top, x="importance", y="feature", orientation="h",
                               title=f"What the deployed {MODEL} relies on", color_discrete_sequence=["#2E5090"]),
                        use_container_width=True)

# ---------------------------------------------------------------- Diagnosis -> Fix
with tab_story:
    st.subheader("Why the baseline was wrong - and what changed")
    p = preds.copy()
    p["year"] = pd.to_datetime(p["sold_date_iso"], utc=True).dt.year
    rows = []
    for v in ("v1", "v2"):
        col = f"pred_{v}_{KEY}"
        if col in p.columns:
            g = p.assign(err=(p["price_numeric"] - p[col]) / p["price_numeric"] * 100).groupby("year")["err"].mean()
            rows += [{"year": y, "version": v, "mean signed error %": e} for y, e in g.items()]
    if rows:
        st.plotly_chart(px.bar(pd.DataFrame(rows), x="year", y="mean signed error %", color="version", barmode="group",
                               title=f"{MODEL}: mean signed error by sale year (positive = model too low)",
                               color_discrete_map={"v1": "#94A3B8", "v2": "#2E5090"}), use_container_width=True)
    st.caption("2025 is where the baseline fell behind the market. v2 adds market-movement features to close that gap.")

    col = f"pred_v2_{KEY}"
    if col in p.columns:
        fig = px.scatter(p, x="price_numeric", y=col, opacity=0.4, color_discrete_sequence=["#2E5090"],
                         labels={"price_numeric": "Actual (AUD)", col: "Predicted (AUD)"},
                         title="Predicted vs actual - test set (v2)")
        m = float(max(p["price_numeric"].max(), p[col].max()))
        fig.add_shape(type="line", x0=0, y0=0, x1=m, y1=m, line=dict(color="red", dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

    if policy is not None:
        st.markdown("**Policy hypothesis (paired bootstrap, 95% CI on change in RMSE)**")
        st.dataframe(policy.style.format({"mean_dRMSE": "{:,.0f}", "CI95_low": "{:,.0f}", "CI95_high": "{:,.0f}",
                                          "avg_policy_uplift_%": "{:.1f}"}), use_container_width=True)
        st.caption("Mixed: each split has one model where policy hurts. Effects are associations confounded with "
                   "2025 rate cuts, so policy flags were left out of the final model.")

# ---------------------------------------------------------------- Predict + affordability
with tab_pred:
    st.subheader("Estimate a price, then test affordability")
    sub = ref["suburbs"]
    ctx = ref["market_context"]
    names = sorted(sub)
    c1, c2, c3 = st.columns(3)
    suburb = c1.selectbox("Suburb", names, index=names.index("Fannie Bay") if "Fannie Bay" in names else 0)
    s = sub[suburb]
    ptype = c1.selectbox("Property type", ["House", "Unit"])
    beds = c2.number_input("Bedrooms", 0, 10, s["beds"])
    baths = c2.number_input("Bathrooms", 0, 6, s["baths"])
    cars = c3.number_input("Car spaces", 0, 6, s["cars"])
    land = c3.number_input("Land area (m²)", 0, 5000, int(s["land_area_m2"] or 600))
    st.caption(f"{suburb}: {s['n_sales']} historical sales · ~{s['distance_from_cbd_km']:.1f} km from CBD "
               f"(from the suburb's median location). Market conditions are held at the latest training values "
               f"(through {ref['trained_through']}).")

    row = {**ctx, "beds": beds, "baths": baths, "cars": cars, "land_area_m2": land,
           "distance_from_cbd_km": s["distance_from_cbd_km"], "latitude": s["latitude"],
           "longitude": s["longitude"], "suburb_target_enc": s["suburb_target_enc"],
           "suburb_price_momentum_6m": s["suburb_price_momentum_6m"]}
    cats = {"suburb": s["model_suburb"], "property_type_from_address": ptype,
            "cash_rate_direction": ctx["cash_rate_direction"]}
    X = pd.DataFrame([{**row, **cats}])[meta["numeric_cols"] + meta["categorical_cols"]]
    price = float(bundle["pipeline"].predict(X)[0])      # pipeline already returns AUD (log target inverted)
    lo, hi = price * ref["ratio_q"]["p10"], price * ref["ratio_q"]["p90"]
    st.success(f"### Estimated price: ${price:,.0f}")
    st.caption(f"80% of backtest forecasts landed within ${lo:,.0f} - ${hi:,.0f} of this kind of estimate.")

    st.markdown("#### Affordability (secondary decision-support layer)")
    d1, d2, d3, d4 = st.columns(4)
    income = d1.number_input("Household income (AUD/yr)", 30_000, 500_000, 140_000, 5_000)
    dep = d2.slider("Deposit %", 5, 40, 20)
    margin = d3.slider("Lender margin over cash rate (pp)", 1.0, 4.0, 2.5, 0.1)
    years = d4.slider("Loan term (years)", 15, 30, 30)

    def repay(p, rate):
        r, n = rate / 100 / 12, years * 12
        loan = p * (1 - dep / 100)
        return loan * r / (1 - (1 + r) ** -n) if r else loan / n

    rate = ctx["cash_rate_target"] + margin
    monthly = repay(price, rate)
    share = monthly * 12 / income * 100
    k = st.columns(3)
    k[0].metric("Loan amount", f"${price * (1 - dep / 100):,.0f}")
    k[1].metric(f"Monthly repayment @ {rate:.2f}%", f"${monthly:,.0f}")
    k[2].metric("Repayment / income", f"{share:.0f}%", "above 30% = stress" if share > 30 else "within 30% guide",
                delta_color="inverse")
    sens = pd.DataFrame({"Rate": [f"{rate + d:.2f}%" for d in (-1, 0, 1, 2)],
                         "Monthly": [round(repay(price, rate + d)) for d in (-1, 0, 1, 2)]})
    st.dataframe(sens, use_container_width=True, hide_index=True)
    st.caption("Illustrative only: cash rate is the latest training value; margin, deposit and income are your inputs.")

# ---------------------------------------------------------------- Limitations
with tab_lim:
    st.markdown("""
- **No new-vs-established flag**; property type is parsed from the address and agrees with land-area presence only ~60-70% of the time.
- **Income is NT-wide**, not suburb-level; annual series joined on calendar year (mild look-ahead).
- **Unemployment not included** - no suitable NT-level series at the right granularity.
- **Geocode quality:** `distance_from_cbd_km` contains implausible values (maximum ~18,000 km), i.e. some listings have bad coordinates. The dashboard uses suburb *medians* to avoid them; validating coordinates in the pipeline is planned.
- **Policy effects are confounded** with 2025 rate cuts - associations, not causal estimates.
- **No untouched holdout** after the final refit; the pooled backtest error is the honest expected error.
- **Market context is frozen** at the latest training values, so estimates do not reflect conditions after the training data ends.
""")