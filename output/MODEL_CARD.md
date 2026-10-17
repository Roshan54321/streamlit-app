# Model card: Greater Darwin property-price model

**Model:** XGBoost, feature set `+policy` (market-movement features, plus policy flags: policy_hlpe_stamp_duty, policy_homegrown_grant, policy_homegrown_months, policy_fed_5pct_no_caps). Refit on 8,209 sales through 2025-12-31.
**Intended use:** a decision-support estimate of the sale price of a Greater Darwin property, with an 80% range and a repayment check, for buyers comparing suburbs. Not for setting auction limits, valuations for lending, or any decision needing a precise price.

## Evidence
| Check | Result |
|---|---|
| Rolling backtest (12 quarterly folds, 3,685 forecast sales) | RMSE $79,123; MAPE 11.9%; R2 0.80 |
| Versus best simple rule (last-quarter or suburb median) | RMSE skill +50.5% |
| Versus Linear Regression on baseline features | RMSE skill +31.3% |
| Out-of-sample coverage of the nominal 80% range | 82.0% with the per-band ranges the dashboard uses (single pooled range: 83.4%); 2,940 sales in the last 9 folds, the first 3 folds only calibrate the range |
| Hyperparameters (backtest and deployed model) | tuned only on sales before 2023-01-01 |
| Choice stability (selection vs confirmation folds) | Winner on earlier folds alone: XGBoost / +policy (same as chosen); chosen config RMSE $76,928 on earlier folds, $82,235 on the latest 4 |

## Data
Homely sold listings (scraped, public), ABS Estimated Resident Population, NT annual household income accounts, RBA cash rate, Darwin CPI. 0 sales had a coordinate sign repaired and 209 were removed as outside Greater Darwin.

## Known limitations
- No untouched test set after the final refit; the backtest error is the expected error.
- Backtest covers 2023-2025: rates were hiked, held and cut and prices rose overall, but there was no broad market fall. The model has not been tested through a market downturn.
- Market context in the dashboard is frozen at the latest training sale, so the dashboard does not know about later rate, CPI or policy changes (for example the end of a grant).
- Accuracy depends on the segment. Across the 39 suburbs with at least 30 forecast sales, MAPE runs from 7% (Muirhead) to 22% (Bayview); by predicted price, from 14% (lowest third) to 10% (highest third). Treat small differences between suburbs as noise and use the range, not the point estimate.
- Against stronger yardsticks built from the same earlier sales, the model's RMSE skill is +25.6% versus Ridge and +38.3% versus a suburb x bedrooms x type median; the strongest benchmark is Ridge (baseline features).
- The 80% range was validated out of sample band by band: coverage 78% / 85% / 86% (low / middle / high predicted price) with the pooled range. With per-band ranges coverage is 80% / 81% / 84% (82.0% overall vs 83.4% pooled); these are closer to 80% and are used.
- Not a downturn test. 1988 forecast sales (54%) fell in hiking-rate periods (MAPE 11.4%); 1924 (52%) had falling 6-month price momentum (MAPE 11.1%), so the backtest does include rate hikes and local dips in price momentum, but prices rose overall and there was no broad market fall. Sensitivity: if prices came in 10% below the model's expectation, the 80% range would cover 68% of sales instead of 80%, so after a downturn the range would be too narrow and the model should be refitted.
- Income is NT-wide, not suburb-level. Unemployment is not included.
- Policy flags (policy_hlpe_stamp_duty, policy_homegrown_grant, policy_homegrown_months, policy_fed_5pct_no_caps) are in the model as date-based regime indicators. They are functions of the sale date, confounded with the 2025 rate cuts, so they show association, not cause. Pooled backtest gain versus the same model without them: 0.2% (small; the paired bootstrap, the one noise standard used in Section 14.7, gives: supported on 2 of 2 splits). policy_fed_5pct_no_caps started too late for any backtest fold to learn it, so it has no out-of-sample check.
- The Linear Regression benchmark fails badly in fold(s) ['2024-01-01'] (extrapolation); without them the skill versus it is +19.3%, not +31.3%. The simple rules ignore beds, baths and property type, so they are a weak yardstick.
- The backtest error assumes the model is refitted every quarter. Single-split errors without refitting, across four cutoffs (60% to 90% of the sale dates, Section 13.7; these use the earlier v2 model with recency weights and its original hyperparameters, not the deployed configuration), were $94,002-$104,313, so a model left stale for several months should be expected to do worse. Refit at least quarterly.
- Recency weighting (2-year half-life) raises the pooled backtest RMSE by 1.10% and helps in 2 of 12 folds; it does NOT clear the pre-set rule (more than 1% and at least 7 folds), so it is switched off: tuning, backtest and final refit run without weights. The hyperparameters were tuned without weights, which favours the unweighted run slightly.
- Property type is inferred from the address and agrees with land-area presence on 57-70% of rows (the notebook's own rule asks for 70%), yet it is one of the model's most important features. Treat property-type effects with caution.
- Annual population and income are lagged 1 year(s) (latest published value) and constant within a year, so they act partly as a time trend; their separate effects cannot be identified.
- Sales outside the price fences learned from the training years ($22,500 to $1,002,500) were excluded; the model is not intended for very cheap or luxury sales.
- The improvement diagnosis was made on the Section 8 test window, so that single-split gain is optimistic; the rolling backtest is the cleaner evidence.
- Affordability maths excludes stamp duty, mortgage insurance and fees.

## Ethics and privacy
Listings are public, but street addresses are personal data: they are used in memory to derive property type, are dropped from every CSV this notebook writes, and never reach the dashboard or model outputs. Check the source site's terms of use before redistributing the scraped data; the repository should hold derived tables and code, not the raw scrape.
