import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from affordability import monthly_repayment, rti

# Folder holding the notebook outputs (copy data/output here, or point APP_OUTPUT_DIR at it).
OUTPUT_DIR = Path(os.environ.get("APP_OUTPUT_DIR", "output"))
STRESS_LINE = 30.0        # rule of thumb: repayments above 30% of gross income (the formal housing-stress measure targets lower-income households)
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
final_cfg = load_csv("predictions_final_config_splits_A_B.csv")
by_fold = load_csv("backtest_final_by_fold.csv")
importance = load_csv("feature_importance_final.csv")
policy_verdict = load_csv("policy_verdict.csv")
selvc = load_csv("selection_vs_confirmation.csv")

required = [bundle, metadata, reference, baseline, v2, pooled, preds]
st.title("🏠 Greater Darwin Property-Price Forecasting")
st.caption("PRT661 Data Science Practice · Group Dan1-Theme2 · Theme 2: Predictive Analytics and Forecasting")
if any(x is None for x in required):
    st.error(f"Some files are missing from `{OUTPUT_DIR}/`. Run the notebook through Section 16, copy `data/output` there "
             "(or set the APP_OUTPUT_DIR environment variable), then restart.")
    st.stop()

FINAL_MODEL = metadata["model_name"]
FINAL_KEY = FINAL_MODEL.lower().replace(" ", "_")
POLICY_BOOT_TXT = ""
if policy_verdict is not None and "verdict" in policy_verdict.columns:
    _pv = policy_verdict[policy_verdict["model"] == FINAL_MODEL]["verdict"].value_counts().to_dict()
    if _pv:
        POLICY_BOOT_TXT = ("The paired bootstrap calls the effect "
                           + ", ".join(f"{v} on {n} of {sum(_pv.values())} splits" for v, n in _pv.items()) + ". ")
final_row = pooled[(pooled["model"] == FINAL_MODEL) & (pooled["feature_set"] == metadata["feature_set"])].iloc[0]
# Policy-flag gain and model-choice robustness are computed from the notebook outputs, not typed in.
_base_row = pooled[(pooled["model"] == FINAL_MODEL) & (pooled["feature_set"] == "baseline")]
if metadata["feature_set"] == "+policy" and len(_base_row):
    POLICY_GAIN_TXT = f"about {(1 - final_row['RMSE'] / float(_base_row['RMSE'].iloc[0])) * 100:.1f}%"
else:
    POLICY_GAIN_TXT = "a small amount"
CHOICE_TXT = ""
if selvc is not None and {"model", "feature_set", "rank_on_selection", "RMSE_selection_folds"} <= set(selvc.columns):
    _w = selvc.loc[selvc["rank_on_selection"] == 1].iloc[0]
    if (_w["model"], _w["feature_set"]) == (FINAL_MODEL, metadata["feature_set"]):
        _s = sorted(selvc["RMSE_selection_folds"])
        CHOICE_TXT = (f"- **Model choice**: the chosen configuration also ranks first on the earlier folds alone, but only by {(_s[1] - _s[0]) / _s[1]:.1%} "
                      "over the next best, so read the ranking as a close call.\n")
    else:
        CHOICE_TXT = f"- **Model choice is close**: on the earlier folds alone {_w['model']} / {_w['feature_set']} ranked first, so the choice between them is not robust.\n"
MARKET = reference["market_context"]
SUBURBS = reference["suburbs"]
P10, P90 = reference["ratio_q"]["p10"], reference["ratio_q"]["p90"]
MIN_PLAUSIBLE_LAND = 50   # a suburb median below this is a data error, not a block size
_lands = [s["land_area_m2"] for s in SUBURBS.values()
          if s.get("land_area_m2") and s["land_area_m2"] >= MIN_PLAUSIBLE_LAND and s["n_sales"] >= MIN_SUBURB_SALES]
CITY_MEDIAN_LAND = float(np.median(_lands)) if _lands else 700.0
# Ranges the model was trained on (sales outside these were excluded). Older exports may not carry them.
BOUNDS = reference.get("training_bounds", {})
LAND_MAX = int(BOUNDS.get("land_max", 5000))
PRICE_MIN, PRICE_MAX = BOUNDS.get("price_min"), BOUNDS.get("price_max")
LATEST_RATE = reference.get("latest_cash_rate")
POLICY_IN_MODEL = [c for c in metadata["numeric_cols"] if c.startswith("policy_")]
TRAINED_THROUGH = pd.Timestamp(reference["trained_through"])
MONTHS_STALE = (pd.Timestamp.today().normalize() - TRAINED_THROUGH).days / 30.44


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


def interval_for(price):
    """80% range multipliers for this predicted price (per price band when the notebook exported them)."""
    if not reference.get("band_intervals_validated"):
        return P10, P90          # per-band ranges did not beat the pooled range out of sample
    for band in reference.get("ratio_q_by_band") or []:
        if band["upper_price"] is None or price <= band["upper_price"]:
            return band["p10"], band["p90"]
    return P10, P90


def predict_prices(rows):
    frame = pd.DataFrame(rows)[metadata["numeric_cols"] + metadata["categorical_cols"]]
    return bundle["pipeline"].predict(frame)


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
    st.info("These come from a **12-quarter rolling backtest**: the model is refitted on earlier sales only "
            "and forecasts the next quarter, 12 times in a row, instead of one lucky train/test split. "
            "Price-trend inputs may use earlier months of that quarter, as they would in real use. "
            "After the final refit on all sales there is no untouched test set, so this is the honest expected error. "
            "It assumes the model is refitted every quarter; a model left stale should be expected to do worse.")
    if MONTHS_STALE > 3:
        st.warning(f"This model was trained on sales through {reference['trained_through']}, about {MONTHS_STALE:.0f} months ago. "
                   "Market conditions and policy settings in the predictor are frozen at that date, so estimates may be out of date. Refit before relying on them.")
    with st.expander("What do RMSE, MAPE and R² mean in plain English?"):
        st.markdown(
            "- **RMSE** - a typical size of the miss in dollars. Big misses count extra.\n"
            f"- **MAPE** - the average miss as a percentage of the real price. {final_row['MAPE']:.1f}% means about ${final_row['MAPE'] / 100 * 500_000:,.0f} off a $500k house.\n"
            f"- **R²** - 0 means no better than guessing the average price; 1 means perfect. {final_row['R2']:.2f} means most, not all, of the price differences are captured.")
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
    best_v1, best_v2 = baseline.loc[baseline["RMSE"].idxmin()], v2.loc[v2["RMSE"].idxmin()]
    deltas = ", ".join(f"{r['Model']} {r['RMSE_v2'] - r['RMSE_v1']:+,.0f}" for _, r in cmp.iterrows())
    verdict = "better" if best_v2["RMSE"] < best_v1["RMSE"] else "WORSE"
    st.caption(f"Change in RMSE, baseline to improved (negative = better): {deltas}. "
               f"Best baseline model: {best_v1['Model']} (${best_v1['RMSE']:,.0f}); best improved model: {best_v2['Model']} (${best_v2['RMSE']:,.0f}), "
               f"so the best result on this single window is {verdict} than before. Gains are model-specific, and this is one window.")

    st.subheader("Final selection: pooled rolling backtest (12 quarters, forecasting the next quarter each time)")
    show_df(pooled.style.format(fmt))
    st.caption("Backtest and single-window RMSE are not like-for-like: they cover different periods, use different "
               "hyperparameters, and the backtest refits every quarter.")
    if by_fold is not None and "RMSE" in by_fold.columns:
        fig_f = px.bar(by_fold, x="fold", y="RMSE", color_discrete_sequence=[NAVY],
                       labels={"fold": "Backtest fold (quarter forecast)", "RMSE": "RMSE (AUD)"})
        fig_f.update_layout(height=320)
        show_chart(fig_f)
    naive = load_csv("backtest_benchmarks.csv")
    if naive is None:
        naive = load_csv("backtest_naive_baselines.csv")
    if naive is not None:
        st.markdown("**Does the model beat simple rules?** Same 12 folds, same forecast sales. "
                    "'Skill' is the model's RMSE improvement over that row; negative means the row did better.")
        show_df(naive)
        if "Ridge (baseline features)" not in set(naive["Model"]):
            st.caption("Ridge and hedonic-median benchmarks are produced by notebook Section 15.4; re-run it to add them. "
                       "Plain Linear Regression fails badly in one fold, so it flatters the model.")
    seg = load_csv("segment_errors_backtest.csv")
    seg_note = "12-quarter backtest"
    if seg is None:
        seg, seg_note = load_csv("segment_errors_splitA.csv"), "single split A only (Nov 2024 onward, model not refitted); re-run notebook Section 15.3 for the backtest version"
    if seg is not None:
        st.markdown("**Where is it accurate?** Error by suburb and by predicted-price band.")
        sub = seg[(seg["segment_type"] == "suburb") & seg["reliable"]].sort_values("MAPE")
        fig_sub = px.bar(sub, x="MAPE", y="segment", orientation="h", color_discrete_sequence=[NAVY],
                         labels={"MAPE": "Average % miss (MAPE)", "segment": ""}, hover_data={"n": True, "BiasPct": True})
        fig_sub.update_layout(height=max(300, 22 * len(sub)))
        show_chart(fig_sub)
        other = seg[seg["segment_type"] != "suburb"].drop(columns=["reliable"], errors="ignore")
        show_df(other)
        st.caption(f"Source: {seg_note}. Only suburbs with at least 30 forecast sales are charted. BiasPct is (actual - predicted) / actual, "
                   "so positive means the model was too low. Treat small differences between suburbs as noise.")
    ivt = load_csv("interval_validation_by_band.csv")
    iv_note = "12-quarter backtest, ranges set from earlier folds only"
    if ivt is None:
        ivt, iv_note = load_csv("interval_validation_splitA.csv"), "single split A: ranges set from the first 60% of that window and tested on the last 40%, with no refit"
    if ivt is not None:
        st.markdown("**Do the 80% ranges really cover 80%?** Out-of-sample coverage by predicted-price band.")
        show_df(ivt)
        st.caption(f"Source: {iv_note}. "
                   + ("Per-band ranges are used in the predictor." if reference.get("band_intervals_validated")
                      else "Per-band ranges did not come closer to 80% than the single pooled range, so the predictor uses the pooled range."))
    stress = load_csv("regime_and_shock_checks.csv")
    if stress is not None:
        st.markdown("**Downturn check (what is observable, plus a sensitivity).** Not a real downturn test: prices rose overall across the backtest, so these are local dips, not a broad fall.")
        show_df(stress.dropna(axis=1, how="all"))
    ablation = load_csv("ablation_feature_groups.csv")
    if ablation is not None:
        st.markdown("**Which feature groups matter?** Refit with one group removed at a time.")
        show_df(ablation)
        full_abl = float(ablation.loc[ablation["variant"] == "all features", "RMSE"].iloc[0])
        if abs(full_abl / final_row["RMSE"] - 1) > 1e-3:
            st.warning(f"This table's 'all features' RMSE (${full_abl:,.0f}) differs from the deployed model's backtest RMSE (${final_row['RMSE']:,.0f}): "
                       "it was run without the policy flags. Re-run the notebook to refresh it.")
        gains = ablation[(ablation["variant"] != "all features") & (ablation["change %"] < 0)]["variant"].tolist()
        st.caption("Differences under about 1% are within noise for a single run. "
                   + (f"Removing these groups did not hurt, so they are not clearly helping: {', '.join(gains)}." if gains else ""))
    scatter = None
    if final_cfg is not None:
        sub = final_cfg[(final_cfg["fold"].str.startswith("A")) & (final_cfg["model"] == FINAL_MODEL) &
                        (final_cfg["feature_set"] == metadata["feature_set"])]
        if len(sub):
            scatter = sub.rename(columns={"actual": "price_numeric", "pred": "pred_final"})
    col = "pred_final" if scatter is not None else f"pred_v2_{FINAL_KEY}"
    if scatter is None and col in preds.columns:
        scatter = preds
    if scatter is not None:
        split_rmse = float(np.sqrt(((scatter[col] - scatter["price_numeric"]) ** 2).mean()))
        st.markdown("**Predicted vs actual price (single split A, Nov 2024 onward, trained before it)**")
        fig_s = px.scatter(scatter, x="price_numeric", y=col, opacity=0.45, color_discrete_sequence=[NAVY],
                           labels={"price_numeric": "Actual price (AUD)", col: "Predicted price (AUD)"})
        top = float(max(scatter["price_numeric"].max(), scatter[col].max()))
        fig_s.add_shape(type="line", x0=0, y0=0, x1=top, y1=top, line=dict(color=RED, dash="dash"))
        fig_s.update_layout(height=420)
        show_chart(fig_s)
        st.caption(f"Points on the red line are perfect; points below it are under-predictions. RMSE on this single window is ${split_rmse:,.0f}, "
                   f"higher than the ${final_row['RMSE']:,.0f} backtest figure because the model is not refitted during this window.")
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
    st.subheader("The 2025 problem, and how far the fix went")
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
        last = g.iloc[-1]
        share_last = float((d["year"] == int(last["year"])).mean())
        before, after = last[c1 + "_pct"], last[c2 + "_pct"]
        row = cmp[cmp["Model"] == model_choice].iloc[0]
        st.caption(f"Test sales only (Nov 2024 onward; {share_last:.0%} of them are in {int(last['year'])}, so this window is mostly one year). Here error = predicted - actual, so below zero means too low (the notebook uses the opposite sign). "
                   f"For {model_choice} in {int(last['year'])}, average error moved from {before:+.1f}% to {after:+.1f}%, "
                   f"and RMSE changed by {row['RMSE_v2'] - row['RMSE_v1']:+,.0f} AUD. "
                   + ("The gap narrowed but did not close. " if abs(after) < abs(before) and after != 0 else "The gap did not narrow. ")
                   + "A smaller bias does not always mean a smaller RMSE; check the performance tab.")
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
        suburb = st.selectbox("Suburb", names, index=names.index("Fannie Bay") if "Fannie Bay" in names else 0,
                              format_func=lambda n: n if SUBURBS[n]["n_sales"] >= MIN_SUBURB_SALES else f"{n} (few sales)")
        ptype = st.selectbox("Property type", ["House", "Unit", "Townhouse"])
    with c2:
        sp = SUBURBS[suburb]
        beds = st.number_input("Bedrooms", 0, 10, int(sp.get("beds", 3)), key=f"beds_{suburb}")
        baths = st.number_input("Bathrooms", 0, 6, int(sp.get("baths", 2)), key=f"baths_{suburb}")
        cars = st.number_input("Car spaces", 0, 6, int(sp.get("cars", 2)), key=f"cars_{suburb}")
    with c3:
        raw_land = sp.get("land_area_m2")
        if not raw_land or raw_land < MIN_PLAUSIBLE_LAND:
            raw_land = CITY_MEDIAN_LAND   # missing or implausible suburb median: fall back to the city median
        land_default = int(min(raw_land, LAND_MAX))
        land = st.number_input("Land area (m²)", 0, LAND_MAX, land_default, key=f"land_{suburb}",
                               help=f"Defaults to the suburb's median block. The model was trained on blocks up to about {LAND_MAX:,} m²; larger blocks are outside its range. "
                                    "For a unit, enter a typical block for similar properties: the model filled missing land areas from similar properties in training.")
        st.caption(f"Distance to CBD is taken from the suburb's median "
                   f"({SUBURBS[suburb]['distance_from_cbd_km']:.1f} km), not typed in.")
    if SUBURBS[suburb]["n_sales"] < MIN_SUBURB_SALES:
        st.warning(f"{suburb} has only {SUBURBS[suburb]['n_sales']} recorded sales, so the model treats it as 'Other'. "
                   "Treat this estimate as rough.")

    price = float(predict_prices([build_input(suburb, ptype, beds, baths, cars, land)])[0])
    p10, p90 = interval_for(price)
    low, high = price * p10, price * p90
    if (PRICE_MAX is not None and price > PRICE_MAX) or (PRICE_MIN is not None and price < PRICE_MIN):
        st.warning("This estimate is outside the price range the model was trained on, so treat it as unreliable.")

    st.subheader("2 · Estimated price and range")
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimate", f"${price:,.0f}")
    m2.metric("Low end (80% range)", f"${low:,.0f}")
    m3.metric("High end (80% range)", f"${high:,.0f}")
    cov = reference.get("coverage_out_of_sample_pct")
    st.caption(f"The range is measured, not guessed" + (", set for this price band" if reference.get("band_intervals_validated") else "") + f": across the 12-quarter backtest, about 80% of sales "
               f"landed between ×{p10:.2f} and ×{p90:.2f} of the model's estimate."
               + (f" Set from earlier quarters only, it covered {cov:.0f}% of later sales" + (f" ({reference['coverage_n']:,} sales in the last {reference['coverage_folds']} quarters)." if reference.get("coverage_n") else ".") if cov else ""))

    st.subheader("3 · Can you afford it?")
    default_cash = float(LATEST_RATE["value"]) if LATEST_RATE else float(MARKET.get("cash_rate_target", 3.6))
    f1, f2, f3, f4 = st.columns(4)
    income = f1.number_input("Gross household income (AUD / year)", 30_000, 1_000_000, 160_000, step=5_000)
    dep_pct = f2.slider("Deposit (%)", 5, 40, 20)
    margin = f3.slider("Lender margin over cash rate (pts)", 1.0, 4.0, 2.5, 0.1)
    years = f4.selectbox("Loan term (years)", [25, 30], index=1)
    cash = st.number_input("Cash rate used (%)", 0.0, 10.0, round(default_cash, 2), step=0.25,
                           help=(f"Defaults to the latest RBA cash rate in the data ({LATEST_RATE['value']:.2f}% from {LATEST_RATE['date']}). "
                             "The price model itself still uses market conditions frozen at the last training sale.") if LATEST_RATE else
                         "Defaults to the RBA cash rate in force at the latest training sale.")
    mort_rate = cash + margin
    frozen_rate, frozen_dir = MARKET.get("cash_rate_target"), MARKET.get("cash_rate_direction")
    if frozen_rate is not None:
        st.caption(f"Two different rates are in play. The price estimate uses market conditions frozen at {reference['trained_through']} "
                   f"(cash rate {frozen_rate:.2f}%, {frozen_dir}). The repayments use the cash rate above ({cash:.2f}%) plus your margin = {mort_rate:.2f}%.")
        if abs(cash - frozen_rate) >= 0.25:
            st.info(f"The cash rate used for repayments ({cash:.2f}%) differs from the one the price model saw ({frozen_rate:.2f}%). "
                    "The model has not seen sales since then, so its price may not reflect the change.")

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
        st.error(f"Repayments take {ratio:.0f}% of your income: above the {STRESS_LINE:.0f}% rule-of-thumb line. "
                 "Lenders also test at a higher buffer rate, so approval may be tighter than this shows.")
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
    st.subheader("Weakest part")
    st.markdown("**The model has only been tested on 2023-2025**, when rates were hiked, held and cut but prices rose overall. "
                "It has not been tested through a broad market downturn, so the error and the 80% range may be too optimistic "
                "if conditions change.")
    if reference.get("coordinates_dropped") is not None:
        st.caption(f"Coordinates were validated before training: {reference['coordinates_repaired']} repaired, "
                   f"{reference['coordinates_dropped']} removed as outside Greater Darwin.")
    st.subheader("Other known limitations")
    st.markdown(
        "- **No untouched test set** after the final refit; the backtest error is the expected error.\n"
        "- **Market conditions are frozen** at the last training sale; the predictor does not know today's rate or CPI.\n"
        "- **Income is NT-wide**, not suburb-level, so it cannot explain differences between suburbs.\n"
        "- **Annual market series** (population, income) are lagged one year and constant within a year, so they act partly as a time trend.\n"
        "- **Very cheap and luxury sales** outside the training-period price range were excluded; estimates there are unreliable.\n"
        + (f"- **Policy flags are in the model** ({', '.join(POLICY_IN_MODEL)}). They are date-based, confounded with the 2025 rate cuts, "
           f"and improved the pooled backtest by only {POLICY_GAIN_TXT}. {POLICY_BOOT_TXT}Read them as association, not cause. "
           "The predictor assumes they stay in force as at the last training sale, including the HomeGrown flag, although the established-home part of that grant ended on 30 Sep 2025.\n" if POLICY_IN_MODEL else
           "- **Policy flags** were tested and left out of the model.\n") +
        ("- **Interval ranges** are set per predicted-price band and were checked out of sample; the pooled range is only a fallback.\n"
         if reference.get("band_intervals_validated") else
         "- **Interval ranges** are pooled over all backtest folds; the per-price-band ranges were not checked out of sample.\n") +
        CHOICE_TXT +
        "- **Unemployment is not included**: no suitable NT-level series was found at the right detail and lag.\n"
        "- **Property type** is guessed from the address and agrees with land-area presence on only 57-70% of rows.\n"
        "- **Affordability is simplified**: no stamp duty, mortgage insurance, fees, or running costs.")
    st.caption("These are also logged in the notebook's audit trail (Section 17) as they were found.")
