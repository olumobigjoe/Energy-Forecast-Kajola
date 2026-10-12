# Kajola ESS Forecast & Dispatch Lab — Reviewed UI

## Purpose
A light, high-contrast Streamlit dashboard for the Kajola research project. It compares the Version 3 LSTM and Differencing–ANN–Bi-LSTM load forecasts and evaluates a transparent battery dispatch heuristic using the same observed load and recorded solar-generation series.

## Main dashboard sections
- Overview — headline metrics and quick charts.
- Data Explorer — source dataset and held-out test observations.
- Model Comparison — forecast metrics and actual-versus-predicted plots.
- ESS Simulation — battery SOC, unserved energy, and capacity sensitivity.
- About & Validation — assumptions, limitations, internal checks, and downloads.

## Files included
- `app.py` — Streamlit dashboard and simulation logic.
- `kajola_dataset_2024_2025.csv` — Kajola hourly source dataset.
- `test_forecasts.csv` — corrected Version 3 held-out test forecasts.
- Validated forecast and ESS output CSVs — retained for traceability/reference.
- `requirements.txt` — Python dependencies.

## Run on Windows
1. Extract the ZIP folder.
2. Open PowerShell in the extracted folder.
3. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
4. Launch the app:
   ```powershell
   python -m streamlit run app.py
   ```

## Data period
The forecast test period covers 13 September 2025 08:00 through 31 December 2025 23:00 (2,632 hourly rows). The app compares the supplied actual-load series to the original dataset at matching timestamps and stops if the values do not match within tolerance.

## Dispatch policy
- Actual observed solar serves actual load first.
- Actual solar surplus may charge the battery within power and SOC limits.
- During an actual deficit, the forecast limits planned battery discharge; actual demand determines realized unmet load.
- The actual-load reference is an idealized benchmark only, not a deployable forecast controller.
- Solar generation is the recorded series; a future solar forecast is not included.

## Default assumptions
- Battery capacity: 50 kWh
- Maximum charge/discharge power: 15 kW each
- Initial SOC: 50%
- SOC range: 10%–95%
- Charge/discharge efficiency: 95% each

These are scenario assumptions, not measured specifications for a physical Kajola battery installation. Passing internal checks establishes accounting consistency under the implemented assumptions; it does not establish field performance. Obtain supervisor approval before formally changing the proposal's specified MATLAB/Simulink environment.
