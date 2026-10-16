# OOP with Python: Assignment, Advent 2026 (MSCS & MSDS)

This repository holds the mini-projects completed so far: **Mini-Project 1** (district population forecaster), **Mini-Project 2** (solar micro-grid dispatch planner) and **Mini-Project 3** (fish stock and export risk model) and **Mini-Project 4** (rainfall and crop suitability analyser). Project 5 will be added to the same repository.

| Project | Notebook | Modules in `src/` |
|---|---|---|
| 1. UBOS District Population Forecaster | `project1_population.ipynb` | `district`, `forecasters`, `metrics`, `bootstrap`, `planning`, `plotting`, `data` |
| 2. Solar Micro-Grid Dispatch Planner | `project2_microgrid.ipynb` | `microgrid`, `demand`, `dispatch`, `microgrid_plotting` |
| 3. Lake Victoria Fish Stock & Export Risk Model | `project3_fish_stock.ipynb` | `fish`, `price`, `risk`, `fish_plotting` |
| 4. Rainfall Pattern & Crop Suitability Analyser | `project4_rainfall.ipynb` | `rainfall`, `similarity`, `seasons`, `nasa_power`, `rainfall_plotting` |

## Setup and run

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest                      # 102 unit tests
jupyter notebook            # open a notebook, then Kernel > Restart & Run All

# or headless:
jupyter nbconvert --to notebook --execute --inplace project1_population.ipynb
jupyter nbconvert --to notebook --execute --inplace project2_microgrid.ipynb
jupyter nbconvert --to notebook --execute --inplace project3_fish_stock.ipynb
jupyter nbconvert --to notebook --execute --inplace project4_rainfall.ipynb
```

Python 3.10+ is required. All random steps use `np.random.default_rng` with fixed seeds, so results are reproducible. Run Jupyter from inside this folder (the first cell of each notebook looks for `src/` next to it). The interactive `input()` demo in Project 2 uses scripted answers by default so the notebook runs unattended; set `INTERACTIVE = True` in its setup cell to type your own values.

## Repository layout

| Path | Purpose |
|---|---|
| `project1_population.ipynb`, `project2_microgrid.ipynb`, `project3_fish_stock.ipynb`, `project4_rainfall.ipynb` | The analyses, each ending with *Findings & Limitations* |
| `src/district.py` | P1: `DistrictPopulation` (validated arrays, `__repr__`, `__len__`, stats, growth, CAGR, split) |
| `src/forecasters.py` | P1: abstract `Forecaster` + linear, CAGR and Fibonacci-ratio subclasses |
| `src/metrics.py`, `src/bootstrap.py`, `src/planning.py`, `src/plotting.py`, `src/data.py` | P1: accuracy metrics and model selection, bootstrap intervals, classroom maths, figure, data |
| `src/microgrid.py` | P2: `MicroGrid` (determinant, condition number, rank, solve) and `HybridMicroGrid(MicroGrid)` |
| `src/demand.py` | P2: seeded demand generator, CSV loader with validation, robust `input()` prompt |
| `src/dispatch.py` | P2: infeasible-day repair (NNLS / clipping), statistics, cost model, Monte Carlo sensitivity |
| `src/microgrid_plotting.py` | P2: stacked-bar dispatch chart and sensitivity scatter |
| `src/fish.py` | P3: `FishStock` (logistic growth with harvesting, MSY theory, closed season) and the Fibonacci baseline |
| `src/price.py` | P3: `PriceModel`, a seeded random walk clipped to its bounds at every step |
| `src/risk.py` | P3: revenue functions, `describe_revenue` and `RiskAssessor` (CV rule, Value-at-Risk) |
| `src/fish_plotting.py` | P3: stock trajectories, revenue histogram, price paths, closed-season figure |
| `src/rainfall.py` | P4: `Region` and `CropRule` (thresholds derived from cited FAO seasonal water needs, planting windows) |
| `src/similarity.py` | P4: correct cosine similarity (dot product / norms), Pearson and Euclidean matrices |
| `src/seasons.py` | P4: circular `find_peaks` rainy-season detection with a prominence threshold |
| `src/nasa_power.py` | P4: NASA POWER fetch / aggregate / validate / load helpers (extension) |
| `src/rainfall_plotting.py` | P4: rainfall lines, crop-suitability heatmap, similarity matrices, box plots |
| `tests/` | pytest suite: 102 tests (normal cases and edge cases) |
| `data/` | P2: generated 30-day demand CSV; P4: crop classification CSV and (optional) `nasa_power_monthly_rainfall_2014_2023.csv` |
| `figures/` | Saved output figures |

---

## Mini-Project 1: UBOS District Population Forecaster

Forecasts 2025-2029 population for five Ugandan districts (Kampala, Wakiso, Gulu, Mbarara, Jinja), selects the best model by validation on a held-out period, and converts the forecast into primary-school classroom needs.

**Findings**
- **Best model:** CAGR in all five districts (hold-out MAPE 0.4-2.5 %), about 1.5-7x more accurate than a linear trend. The Fibonacci-ratio model is off by 36-40 % and is not usable.
- **Fastest growing:** Wakiso (CAGR 6.47 %). Kampala and Gulu tie exactly at 4.61 % because both grew by a factor of 1.5.
- **2029 forecast (thousands):** Kampala 2,255; Wakiso 2,285 (overtakes Kampala); Gulu 601; Mbarara 854; Jinja 769.
- **Classrooms needed by 2029** (18 % primary-age, 53 pupils per classroom): Wakiso 2,088; Kampala 1,544; Mbarara 539; Jinja 421; Gulu 412; about 5,000 in total.
- **Forecast variance** is 41-59 % of the historical variance only because the forecast is shorter and has no noise, not because the future is more certain.

**Assumptions:** populations are in thousands; Kampala, Wakiso and Gulu come from the brief, while **Mbarara and Jinja are illustrative (synthetic) series I added**. Train 2015-2021, test 2022-2024, selection by lowest MAPE. "Additional classrooms" = need(2029) - need(2024), each rounded up, assuming current classrooms exactly meet current need. The Fibonacci model applies each ratio to the same last value by default; a compounded variant is also implemented and discussed.

**Extensions:** residual-bootstrap 95 % prediction intervals (2,000 resamples); critical assessment of the Fibonacci-ratio model.

## Mini-Project 2: Solar Micro-Grid Dispatch Planner

Solves `3x + 2y = D1`, `4x + y = D2` for solar (x) and battery (y) each day for a Kasese health centre, for 30 days of demand, and analyses feasibility, cost and sensitivity.

**Findings**
- **Well-posed:** determinant -5 and condition number 5.83, so every demand pair has a unique solution and demand errors are magnified at most 5.8x.
- **Loop vs vectorised:** one 2 x 30 `scipy.linalg.solve` call is about 10-15x faster than a Python loop (the matrix is factorised once, not 30 times).
- **Infeasible days:** a day is feasible only if 0.75*D2 <= D1 <= 2*D2. Four quiet weekend days (6, 7, 21, 28) gave a negative battery value; non-negative least squares repaired them with a 0.9-2.6 kWh load mismatch a day (7.6 kWh in total, versus 17.0 kWh for clipping).
- **Volatility:** battery use is much more volatile than solar (st. dev. 7.3 vs 1.7 kWh; CV 56 % vs 12.5 %), because it absorbs the weekly swing in daytime load.
- **Cost:** UGX 235,353 per month (about 7,845 a day); the battery is 74 % of the cost for about 49 % of the energy.
- **Sensitivity:** a +/-5 % demand error moves solar by about 0.7 kWh and battery by 1.6 kWh on average; the amplification of relative error is 3.4x on average and at most 5.82, just under the condition number 5.83.

**Assumptions:** all demand data are synthetic. I chose the weekly pattern (busy weekdays, quiet weekends, steady critical load), so the *number* of infeasible days depends on that choice. Tariffs (UGX 150 and 450 per kWh) are the illustrative values from the brief. The hybrid grid's third equation is x + y + z = D3 (total generation), my own assumption.

**Extensions:** `HybridMicroGrid(MicroGrid)` with a diesel generator, including what happens when the new equation is linearly dependent (rank 2 < 3, infinitely many or no solutions, and the class refuses to solve); Monte Carlo sensitivity (1,000 draws x 30 days) related to the condition number.

## Mini-Project 3: Lake Victoria Fish Stock & Export Risk Model

Models a Jinja export cooperative's fish stock (discrete logistic growth with harvesting, r = 0.4, K = 10,000 t, N(0) = 4,000 t, 52 weeks) and its weekly revenue (bounded random-walk price starting at UGX 12,000/kg), then assesses sustainability and revenue risk.

**Findings**
- **Fibonacci baseline:** unbounded (about 1.618x per step, passing K at the 22nd term), with no density dependence or harvesting, so it is not a biological model.
- **Sustainability:** the stock settles at K(1 - h/r): 8,750, 7,500, 5,000 and 2,500 t for h = 0.05, 0.10, 0.20, 0.30. Yield peaks at h = r/2 = 0.20, where the weekly catch equals MSY = rK/4 = 1,000 t. h = 0.30 is overfishing (same long-run catch as h = 0.10, from a third of the stock); h >= 0.40 collapses the stock.
- **Why a variance threshold of 50,000 is meaningless:** variance is in UGX squared and changes with the unit (1.85e18 UGX^2 vs 1.85 UGX-billion^2 for the same data), whereas the CV (15.4 % weekly) does not.
- **Risk:** CV of annual revenue is about 11 % in every scenario, so all four are *Moderate*; sustainability status, not CV, separates them. At h = 0.10 the 5 % Value-at-Risk is UGX 78 bn, 17 % of the expected UGX 452 bn.
- **Closed season (8 weeks a year):** it reduces long-run revenue by 9-15 % when h <= 0.20 but raises it by 7.7 % at h = 0.30, through a boom-and-bust catch pulse after reopening; simply fishing at h = 0.20 is better still.

**Assumptions:** the weekly price volatility (UGX 400) and the risk thresholds (Low < 0.10 <= Moderate < 0.25 <= High, justified by the implied 16 % and 41 % bad-year losses) are my choices. Risk is measured on annual revenue over 10,000 simulated price paths. Value-at-Risk is reported as the 5th-percentile revenue and as the shortfall against mean revenue. The closed season is the first 8 weeks of each 52-week year.

**Extension:** the 8-week closed season over 5 years, for all four harvest rates, with common price paths.

## Mini-Project 4: Rainfall Pattern & Crop Suitability Analyser

Compares the monthly rainfall of Kampala, Gulu and Mbarara, classifies every month for maize, beans and coffee, corrects the flawed `math.cos` similarity method, detects rainy seasons automatically and ends with a farmer advisory for Mbarara.

**Findings**
- **Regimes:** annual rainfall is 1,600 mm (Kampala), 1,388 mm (Gulu) and 1,040 mm (Mbarara); Gulu is the most seasonal (CV 61 %) and Mbarara the most drought-prone (10 of 12 months below maize's 125 mm monthly band).
- **Cosine similarity:** `math.cos` takes one angle, not two vectors; the correct measure is `a.b / (|a||b|)`, verified against SciPy for all pairs. It rates regions as similar whatever their size (3x Kampala's rain scores 1.00), and is high (0.75-0.89) for any non-negative data, while Pearson correlation (-0.17 to 0.21) shows the seasonal timing differs.
- **Rainy seasons:** detection depends on the prominence threshold. At 30 % of the wettest month, Mbarara is bimodal and Gulu unimodal, as expected for south-western and northern Uganda. Kampala comes out unimodal, which does **not** match its known two seasons; this is an artefact of the supplied data (its second "peak" is a 10 mm wiggle), and the notebook says so.
- **Advisory (Mbarara):** grow beans, planted in March (about 350 mm over the cycle) and September (about 345 mm); no four-month maize window reaches the 500 mm FAO minimum, and the annual rainfall is below coffee's range.

**Assumptions:** the rainfall data are illustrative. Thresholds: maize 500-800 mm per cycle and beans 300-500 mm per cycle (FAO), coffee 1,200-1,800 mm per year (a general Robusta range, not Uganda-specific), each spread evenly over an assumed cycle of 4, 3 and 12 months. The prominence threshold (30 %) is my choice, justified by a sensitivity table.

**Extension (real data):** `src/nasa_power.py` aggregates NASA POWER daily rainfall (2014-2023) to monthly totals for box plots, a real-vs-illustrative climatology check and a planting-window reliability table. The notebook runs without internet: it uses `data/nasa_power_monthly_rainfall_2014_2023.csv` if present. To create it, set `FETCH_NASA = True` in the notebook's setup cell and run once with internet access, then commit the CSV.

## Limitations

See *Findings & Limitations* at the end of each notebook. In short: partly synthetic and smooth data, no real-data extra credit claimed, simple models (no battery state of charge in Project 2; one constant growth rate in Project 1; deterministic fish dynamics in Project 3, so its risk measures capture price risk only; screening-level crop rules in Project 4), and flat illustrative tariffs and classroom parameters.

## AI-use declaration

I used an AI assistant (Claude, by Anthropic) while working on these projects: to compare my draft notebooks, to restructure the draft code into the `src/` modules, to write the pytest tests, and to help draft the written analysis. I checked the results against independent calculations (each notebook has a "Sanity checks" section using a second method or a hand calculation) and I can explain any line of the code in a viva.
