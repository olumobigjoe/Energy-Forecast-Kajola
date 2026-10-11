# Kajola ESS Simulator (Python + Streamlit)

## Included
- `app.py`: interactive battery/solar/load scenario simulator
- `kajola_dataset_2024_2025.csv`: supplied Kajola hourly dataset
- `test_forecasts.csv`: Version 3 test forecasts (LSTM, hybrid, ANN and persistence)
- `requirements.txt`: Python packages

## Run on Windows
1. Extract this folder.
2. Open PowerShell in the extracted folder.
3. Create a virtual environment (recommended):
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
5. Launch:
   ```powershell
   streamlit run app.py
   ```

If PowerShell blocks activation, you can skip activation and run:
`python -m pip install -r requirements.txt`
then
`python -m streamlit run app.py`

## Model assumptions and limits
- One row represents one hour.
- Solar generation serves load first; only solar surplus can charge the battery.
- Battery discharge serves residual demand subject to power, SOC and efficiency limits.
- No grid charging, tariffs, degradation, inverter dynamics or forecast uncertainty are included yet.
- Default battery parameters are illustrative and must be justified/validated for the research.
- Forecast-based scenarios are limited to the Version 3 held-out test period and use recorded solar generation aligned by timestamp.
- Scenario outputs are simulated estimates, not measured real-world ESS performance.


## Validation additions
The dashboard includes automatic checks for:
- hourly load and solar energy-balance residuals;
- minimum and maximum battery state of charge;
- charging and discharging power limits;
- non-negative SOC, unserved load and curtailed solar.

A passing check confirms internal accounting consistency under the coded assumptions. It does not independently validate the model against a real battery or establish measured field performance.
