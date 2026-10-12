# Kajola ESS Forecast Comparison — Reviewed Prototype

## Purpose
Compare the Version 3 LSTM and Differencing–ANN–Bi-LSTM forecasts and examine a transparent forecast-limited battery dispatch scenario using the same actual load and recorded solar-generation series.

## Included files
- `app.py` — Streamlit dashboard and ESS scenario model.
- `kajola_dataset_2024_2025.csv` — Kajola hourly dataset.
- `test_forecasts.csv` — corrected Version 3 held-out test forecasts. Its metrics reproduce the established Version 3 results.
- `requirements.txt` — Python dependencies.

## Run on Windows
1. Extract the ZIP.
2. Open PowerShell in the extracted folder.
3. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
4. Launch:
   ```powershell
   python -m streamlit run app.py
   ```

## Forecast test period
The test forecasts cover 2025-09-13 08:00 through 2025-12-31 23:00 (2,632 hourly rows). The forecast `Actual_Load_kW` values are checked against the original dataset at matching timestamps. The maximum observed mismatch should be zero.

## Dispatch policy
- Actual observed solar serves actual load first.
- Actual solar surplus can charge the battery, subject to charging power, capacity, SOC bounds, and charging efficiency.
- During a deficit, the selected forecast limits the planned discharge amount; actual load limits the amount that can actually serve the load. This means underforecasting can leave additional demand unmet. The rule is a transparent heuristic, not an optimized controller.
- The `Actual-load reference` is an idealized benchmark using actual demand to guide discharge. It is not a forecast controller and should not be represented as a deployable method.

## Default assumptions
- Battery capacity: 50 kWh
- Charge/discharge power: 15 kW each
- Initial SOC: 50%
- Minimum/maximum SOC: 10%/95%
- Charge/discharge efficiencies: 95%/95%
- Solar generation: recorded data aligned by timestamp

These are scenario assumptions, not confirmed physical specifications for a Kajola installation.

## Validation checks
The app checks:
- load energy balance;
- solar energy balance;
- SOC transition balance;
- minimum and maximum SOC constraints;
- charge/discharge power limits;
- non-negative SOC, unserved load, curtailed solar, charge and discharge.

Passing checks establishes internal accounting consistency only. It does not independently validate the model against physical battery measurements.

## Research integrity
The app reports forecast accuracy separately from ESS operational results. Operational outcomes are calculated against actual observed demand for both forecast controllers under the same conditions. Do not claim that one forecast improves real-world reliability unless the assumptions, dispatch rule, comparison baseline, and limitations are clearly documented.
