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


## Compare 25, 50, 75 and 100 kWh batteries
1. Start the app and leave the default comparison assumptions unchanged unless your study requires otherwise: 15 kW charge and discharge limits, 50% initial SOC, 10% minimum SOC, 95% maximum SOC, and 95% charging/discharging efficiencies.
2. In the sidebar, select **LSTM forecast (test period)**.
3. Keep the same simulation dates for every run. The supplied LSTM forecast covers **2025-09-13 08:00 to 2025-12-31 23:00** (2,632 hourly records). The date selector may remain at its full range; the forecast merge limits the actual LSTM scenario to its available timestamps.
4. Open the **Capacity comparison** tab. It runs 25, 50, 75 and 100 kWh capacities automatically, with both power limits fixed at 15 kW. Only capacity changes across those four scenarios.
5. Compare unserved energy, curtailed solar, estimated conversion losses, SOC minimum/maximum/final values, hours at SOC limits, and validation results. Use the CSV download to save the comparison.
6. If you change efficiency or SOC settings, change them only when intentionally running a separate experiment; keep them fixed within each capacity comparison.

The comparison table is computed by the app from the supplied LSTM forecast and recorded solar-generation data. It is not a claim of real-world battery performance. Do not compare it directly with the earlier full-dataset observed-load simulation as if both used the same time period: the LSTM scenario is only the held-out test period.
