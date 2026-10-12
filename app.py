import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Kajola ESS Forecast Comparison", page_icon="🔋", layout="wide")
DATA_DIR = Path(__file__).parent
DATA_FILE = DATA_DIR / "kajola_dataset_2024_2025.csv"
FORECAST_FILE = DATA_DIR / "test_forecasts.csv"
TOL = 1e-6

st.title("Kajola Energy Storage Simulator")
st.caption("Research prototype | One-hour-ahead forecast comparison | Kajola, Nigeria")
st.warning(
    "This is a simplified scenario simulation, not a validated battery engineering design. "
    "Forecasting metrics and ESS operational metrics are different evaluations. Review the assumptions below before using results in a thesis."
)

@st.cache_data
def load_inputs():
    if not DATA_FILE.exists() or not FORECAST_FILE.exists():
        raise FileNotFoundError("Both kajola_dataset_2024_2025.csv and test_forecasts.csv must be beside app.py.")
    d = pd.read_csv(DATA_FILE)
    f = pd.read_csv(FORECAST_FILE)
    d["Timestamp"] = pd.to_datetime(d["Timestamp"], errors="coerce")
    f["Timestamp"] = pd.to_datetime(f["Timestamp"], errors="coerce")
    for c in ["Load_kW", "Solar_Gen_kW"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    forecast_cols = ["Actual_Load_kW", "Naive_Persistence_kW", "ANN_kW", "LSTM_kW", "Diff-ANN-BiLSTM_kW"]
    for c in forecast_cols:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    d = d.dropna(subset=["Timestamp"]).sort_values("Timestamp").drop_duplicates("Timestamp").reset_index(drop=True)
    # Preserve chronology; forward fill only from prior observations, never backfill from future values.
    d[["Load_kW", "Solar_Gen_kW"]] = d[["Load_kW", "Solar_Gen_kW"]].ffill()
    d = d.dropna(subset=["Load_kW", "Solar_Gen_kW"]).reset_index(drop=True)
    f = f.dropna(subset=["Timestamp"] + forecast_cols).sort_values("Timestamp").drop_duplicates("Timestamp").reset_index(drop=True)
    merged = f.merge(d[["Timestamp", "Load_kW", "Solar_Gen_kW"]], on="Timestamp", how="left", validate="one_to_one")
    if merged["Solar_Gen_kW"].isna().any() or merged["Load_kW"].isna().any():
        raise ValueError("Some forecast timestamps could not be matched to observed load and solar data.")
    # Confirm supplied actual series matches the original data at each forecast timestamp.
    mismatch = (merged["Actual_Load_kW"] - merged["Load_kW"]).abs().max()
    return d, merged, float(mismatch)


def metric_rows(f):
    actual = f["Actual_Load_kW"].to_numpy(dtype=float)
    rows = []
    for label, col in [("Naïve persistence", "Naive_Persistence_kW"), ("ANN", "ANN_kW"),
                       ("LSTM", "LSTM_kW"), ("Differencing–ANN–Bi-LSTM", "Diff-ANN-BiLSTM_kW")]:
        pred = f[col].to_numpy(dtype=float)
        nz = np.abs(actual) > 1e-12
        rows.append({
            "Model": label,
            "MAE_kW": mean_absolute_error(actual, pred),
            "RMSE_kW": mean_squared_error(actual, pred) ** 0.5,
            "MAPE_percent_nonzero_actuals": float(np.mean(np.abs((actual[nz] - pred[nz]) / actual[nz])) * 100) if nz.any() else np.nan,
            "R2": r2_score(actual, pred),
        })
    return pd.DataFrame(rows)


def simulate_dispatch(frame, capacity_kwh, max_charge_kw, max_discharge_kw,
                      initial_soc_pct, min_soc_pct, max_soc_pct,
                      charge_efficiency, discharge_efficiency, forecast_col=None):
    """Simulate actual load/solar balance. Forecast limits planned discharge; actual load determines realized unmet demand.

    Policy: solar serves actual load first; actual solar surplus charges the battery. During a deficit,
    a forecast-informed controller may discharge up to its predicted net-load requirement, but never
    more than the actual residual load, power limit, or available battery energy. forecast_col=None
    is an actual-load-informed reference case, not a deployable forecast controller.
    """
    x = frame.reset_index(drop=True).copy()
    cap = float(capacity_kwh)
    soc_min = cap * float(min_soc_pct) / 100
    soc_max = cap * float(max_soc_pct) / 100
    soc = float(np.clip(cap * float(initial_soc_pct) / 100, soc_min, soc_max))
    rows = []
    for _, row in x.iterrows():
        actual = max(float(row["Actual_Load_kW"]), 0.0)
        solar = max(float(row["Solar_Gen_kW"]), 0.0)
        forecast = actual if forecast_col is None else max(float(row[forecast_col]), 0.0)
        direct = min(actual, solar)
        residual_actual = max(actual - direct, 0.0)
        solar_surplus = max(solar - actual, 0.0)
        charge = discharge = 0.0
        unserved = residual_actual
        curtailed = solar_surplus
        soc_before = soc

        # Charge from realized solar surplus; identical rule for all forecast models.
        if solar_surplus > 0 and soc < soc_max:
            max_input_from_soc = max((soc_max - soc) / max(charge_efficiency, 1e-12), 0.0)
            charge = min(solar_surplus, max_charge_kw, max_input_from_soc)
            soc += charge * charge_efficiency
            curtailed = solar_surplus - charge
        elif residual_actual > 0 and soc > soc_min:
            predicted_residual = max(forecast - solar, 0.0)
            planned_discharge = residual_actual if forecast_col is None else predicted_residual
            available_output = max((soc - soc_min) * discharge_efficiency, 0.0)
            discharge = min(residual_actual, planned_discharge, max_discharge_kw, available_output)
            soc -= discharge / max(discharge_efficiency, 1e-12)
            unserved = residual_actual - discharge

        rows.append({
            "Timestamp": row["Timestamp"], "Actual_Load_kW": actual,
            "Forecast_Load_kW": forecast, "Solar_Gen_kW": solar,
            "Direct_Solar_to_Load_kW": direct,
            "Residual_Load_before_Battery_kW": residual_actual,
            "Battery_Charge_Input_kW": charge, "Battery_Discharge_to_Load_kW": discharge,
            "Unserved_Load_kW": max(unserved, 0.0), "Curtailed_Solar_kW": max(curtailed, 0.0),
            "SOC_before_kWh": soc_before, "SOC_kWh": soc,
        })

    r = pd.DataFrame(rows)
    # Hourly time steps: kW values numerically equal kWh per interval.
    r["Load_Energy_kWh"] = r["Actual_Load_kW"]
    r["Solar_Energy_kWh"] = r["Solar_Gen_kW"]
    r["Unserved_Energy_kWh"] = r["Unserved_Load_kW"]
    r["Served_Load_kW"] = r["Actual_Load_kW"] - r["Unserved_Load_kW"]
    r["Load_Balance_Residual_kW"] = r["Actual_Load_kW"] - r["Direct_Solar_to_Load_kW"] - r["Battery_Discharge_to_Load_kW"] - r["Unserved_Load_kW"]
    r["Solar_Balance_Residual_kW"] = r["Solar_Gen_kW"] - r["Direct_Solar_to_Load_kW"] - r["Battery_Charge_Input_kW"] - r["Curtailed_Solar_kW"]
    r["SOC_Balance_Residual_kWh"] = r["SOC_kWh"] - (r["SOC_before_kWh"] + r["Battery_Charge_Input_kW"] * charge_efficiency - r["Battery_Discharge_to_Load_kW"] / max(discharge_efficiency, 1e-12))
    return r


def summarize(result, capacity_kwh, max_charge_kw, max_discharge_kw, min_soc_pct, max_soc_pct, charge_eff, discharge_eff):
    load_total = float(result["Load_Energy_kWh"].sum())
    solar_total = float(result["Solar_Energy_kWh"].sum())
    no_ess = float(np.maximum(result["Actual_Load_kW"] - result["Solar_Gen_kW"], 0).sum())
    unserved = float(result["Unserved_Energy_kWh"].sum())
    charge = float(result["Battery_Charge_Input_kW"].sum())
    discharge = float(result["Battery_Discharge_to_Load_kW"].sum())
    loss = charge * (1-charge_eff) + (discharge / max(discharge_eff, 1e-12) - discharge)
    soc_min = capacity_kwh * min_soc_pct / 100
    soc_max = capacity_kwh * max_soc_pct / 100
    checks = [
        ("Load balance", result["Load_Balance_Residual_kW"].abs().max() <= TOL),
        ("Solar balance", result["Solar_Balance_Residual_kW"].abs().max() <= TOL),
        ("SOC transition balance", result["SOC_Balance_Residual_kWh"].abs().max() <= TOL),
        ("Minimum SOC", result["SOC_kWh"].min() >= soc_min-TOL),
        ("Maximum SOC", result["SOC_kWh"].max() <= soc_max+TOL),
        ("Charge power limit", result["Battery_Charge_Input_kW"].max() <= max_charge_kw+TOL),
        ("Discharge power limit", result["Battery_Discharge_to_Load_kW"].max() <= max_discharge_kw+TOL),
        ("Non-negative values", result[["SOC_kWh", "Unserved_Load_kW", "Curtailed_Solar_kW", "Battery_Charge_Input_kW", "Battery_Discharge_to_Load_kW"]].min().min() >= -TOL),
    ]
    return {
        "Hours": len(result), "Load_kWh": load_total, "Solar_generation_kWh": solar_total,
        "Unserved_without_ESS_kWh": no_ess, "Unserved_with_ESS_kWh": unserved,
        "Unserved_reduction_vs_no_ESS_pct": (no_ess-unserved)/no_ess*100 if no_ess else 0,
        "Load_served_pct": (load_total-unserved)/load_total*100 if load_total else 0,
        "Battery_charge_input_kWh": charge, "Battery_delivered_kWh": discharge,
        "Curtailed_solar_kWh": float(result["Curtailed_Solar_kW"].sum()),
        "Estimated_conversion_losses_kWh": loss,
        "SOC_min_observed_kWh": float(result["SOC_kWh"].min()), "SOC_max_observed_kWh": float(result["SOC_kWh"].max()),
        "SOC_final_kWh": float(result["SOC_kWh"].iloc[-1]),
        "Hours_at_min_SOC": int(np.isclose(result["SOC_kWh"], soc_min, atol=1e-5).sum()),
        "Hours_at_max_SOC": int(np.isclose(result["SOC_kWh"], soc_max, atol=1e-5).sum()),
        "Max_load_balance_residual_kW": float(result["Load_Balance_Residual_kW"].abs().max()),
        "Max_solar_balance_residual_kW": float(result["Solar_Balance_Residual_kW"].abs().max()),
        "Max_SOC_balance_residual_kWh": float(result["SOC_Balance_Residual_kWh"].abs().max()),
        "Validation_checks_passed": sum(ok for _, ok in checks), "Validation_checks_total": len(checks),
        "Validation_details": checks,
    }

try:
    data, forecasts, actual_mismatch = load_inputs()
except Exception as e:
    st.error(f"Input-data problem: {e}")
    st.stop()

if actual_mismatch > 1e-6:
    st.error(f"Forecast actual-load column differs from source dataset (maximum difference {actual_mismatch:.6g} kW). Check the forecast file.")
    st.stop()

start_default = forecasts["Timestamp"].min().date()
end_default = forecasts["Timestamp"].max().date()
with st.sidebar:
    st.header("Simulation settings")
    st.caption("Forecast comparison uses the held-out test period by default.")
    start_date, end_date = st.date_input("Test-period dates", value=(start_default, end_default), min_value=start_default, max_value=end_default)
    capacity = st.slider("Battery capacity (kWh)", 5, 200, 50, 5)
    charge_power = st.slider("Maximum charge power (kW)", 1, 50, 15, 1)
    discharge_power = st.slider("Maximum discharge power (kW)", 1, 50, 15, 1)
    initial_soc = st.slider("Initial SOC (%)", 0, 100, 50, 5)
    min_soc = st.slider("Minimum SOC (%)", 0, 50, 10, 5)
    max_soc = st.slider("Maximum SOC (%)", 50, 100, 95, 5)
    charge_eff = st.slider("Charge efficiency (%)", 50, 100, 95, 1) / 100
    discharge_eff = st.slider("Discharge efficiency (%)", 50, 100, 95, 1) / 100

if start_date > end_date:
    st.error("Start date must be on or before end date.")
    st.stop()
if min_soc >= max_soc:
    st.error("Minimum SOC must be below maximum SOC.")
    st.stop()

frame = forecasts[(forecasts["Timestamp"].dt.date >= start_date) & (forecasts["Timestamp"].dt.date <= end_date)].copy().reset_index(drop=True)
if frame.empty:
    st.warning("No test-period rows for the selected dates.")
    st.stop()

metrics = metric_rows(frame)
st.subheader("Forecast accuracy on the held-out test period")
st.caption(f"{len(frame):,} hourly observations | {frame['Timestamp'].min()} to {frame['Timestamp'].max()} | MAE/RMSE/MAPE are based on actual load versus each forecast.")
st.dataframe(metrics.round(4), use_container_width=True, hide_index=True)

model_cols = {"LSTM": "LSTM_kW", "Hybrid (Differencing–ANN–Bi-LSTM)": "Diff-ANN-BiLSTM_kW"}
results = {}
summaries = []
for model_name, col in model_cols.items():
    results[model_name] = simulate_dispatch(frame, capacity, charge_power, discharge_power, initial_soc, min_soc, max_soc, charge_eff, discharge_eff, col)
    sm = summarize(results[model_name], capacity, charge_power, discharge_power, min_soc, max_soc, charge_eff, discharge_eff)
    sm["Controller"] = model_name
    summaries.append(sm)
# Actual-load-informed reference benchmark, included only as an upper-information reference.
results["Actual-load reference"] = simulate_dispatch(frame, capacity, charge_power, discharge_power, initial_soc, min_soc, max_soc, charge_eff, discharge_eff, None)
ref_summary = summarize(results["Actual-load reference"], capacity, charge_power, discharge_power, min_soc, max_soc, charge_eff, discharge_eff)
ref_summary["Controller"] = "Actual-load reference (benchmark only)"
summaries.append(ref_summary)
summary_df = pd.DataFrame([{k:v for k,v in s.items() if k != "Validation_details"} for s in summaries])

st.subheader("ESS outcomes evaluated against actual observed load")
st.caption("Both forecast controllers use the same actual demand and recorded solar series for realized energy accounting. The forecast limits planned battery discharge; actual demand determines how much load remains unserved. The actual-load reference is an idealized comparison, not a forecast controller.")
show_cols = ["Controller", "Unserved_without_ESS_kWh", "Unserved_with_ESS_kWh", "Unserved_reduction_vs_no_ESS_pct", "Load_served_pct", "Battery_charge_input_kWh", "Battery_delivered_kWh", "Curtailed_solar_kWh", "Estimated_conversion_losses_kWh", "SOC_final_kWh", "Validation_checks_passed", "Validation_checks_total"]
st.dataframe(summary_df[show_cols].round(3), use_container_width=True, hide_index=True)

for s in summaries:
    name = s["Controller"]
    if s["Validation_checks_passed"] != s["Validation_checks_total"]:
        st.error(f"Validation failed for {name}: {s['Validation_checks_passed']}/{s['Validation_checks_total']} checks passed.")

st.caption("Battery assumptions are scenario inputs, not measured Kajola battery specifications. The dispatch rule is intentionally simple and should be described as a forecast-limited heuristic, not an optimized energy-management controller.")

tab1, tab2, tab3, tab4 = st.tabs(["Forecast comparison", "ESS time series", "Battery capacity sensitivity", "Validation & downloads"])
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["Timestamp"], y=frame["Actual_Load_kW"], name="Actual load", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=frame["Timestamp"], y=frame["LSTM_kW"], name="LSTM forecast", line=dict(width=1.2)))
    fig.add_trace(go.Scatter(x=frame["Timestamp"], y=frame["Diff-ANN-BiLSTM_kW"], name="Hybrid forecast", line=dict(width=1.2)))
    fig.update_layout(title="Actual load versus forecasts", xaxis_title="Time", yaxis_title="Load (kW)")
    st.plotly_chart(fig, use_container_width=True)
    e = metrics[metrics["Model"].isin(["LSTM", "Differencing–ANN–Bi-LSTM"])][["Model", "MAE_kW", "RMSE_kW", "MAPE_percent_nonzero_actuals"]].melt(id_vars="Model", var_name="Metric", value_name="Error")
    st.plotly_chart(px.bar(e, x="Metric", y="Error", color="Model", barmode="group", title="LSTM versus hybrid forecasting errors"), use_container_width=True)

with tab2:
    fig_soc = go.Figure()
    for name in ["LSTM", "Hybrid (Differencing–ANN–Bi-LSTM)"]:
        r = results[name]
        fig_soc.add_trace(go.Scatter(x=r["Timestamp"], y=r["SOC_kWh"], name=f"{name} SOC"))
    fig_soc.add_hline(y=capacity*min_soc/100, line_dash="dash", annotation_text="Minimum SOC")
    fig_soc.add_hline(y=capacity*max_soc/100, line_dash="dash", annotation_text="Maximum SOC")
    fig_soc.update_layout(title="Battery state of charge by forecast controller", xaxis_title="Time", yaxis_title="Stored energy (kWh)")
    st.plotly_chart(fig_soc, use_container_width=True)
    u = pd.DataFrame([{"Controller": n, "Unserved energy (kWh)": float(results[n]["Unserved_Energy_kWh"].sum())} for n in results])
    st.plotly_chart(px.bar(u, x="Controller", y="Unserved energy (kWh)", title="Unserved energy by controller"), use_container_width=True)
    st.plotly_chart(px.line(results["LSTM"], x="Timestamp", y="Unserved_Load_kW", title="Hourly unserved load — LSTM controller"), use_container_width=True)
    st.plotly_chart(px.line(results["Hybrid (Differencing–ANN–Bi-LSTM)"], x="Timestamp", y="Unserved_Load_kW", title="Hourly unserved load — hybrid controller"), use_container_width=True)

with tab3:
    st.markdown("Battery capacity sensitivity: same test dates, observed demand, solar profile, power limits, SOC bounds, and efficiencies. Only capacity changes between scenarios.")
    cap_rows = []
    for cap in [25, 50, 75, 100]:
        for name, col in model_cols.items():
            r = simulate_dispatch(frame, cap, 15, 15, initial_soc, min_soc, max_soc, charge_eff, discharge_eff, col)
            sm = summarize(r, cap, 15, 15, min_soc, max_soc, charge_eff, discharge_eff)
            sm["Capacity_kWh"] = cap
            sm["Controller"] = name
            cap_rows.append({k:v for k,v in sm.items() if k != "Validation_details"})
    cap_df = pd.DataFrame(cap_rows)
    st.dataframe(cap_df.round(3), use_container_width=True, hide_index=True)
    st.plotly_chart(px.bar(cap_df, x="Capacity_kWh", y="Unserved_with_ESS_kWh", color="Controller", barmode="group", title="Unserved actual energy by battery capacity"), use_container_width=True)
    st.plotly_chart(px.bar(cap_df, x="Capacity_kWh", y="Curtailed_solar_kWh", color="Controller", barmode="group", title="Curtailed solar by battery capacity"), use_container_width=True)
    st.download_button("Download capacity sensitivity CSV", cap_df.to_csv(index=False).encode("utf-8"), "kajola_lstm_hybrid_capacity_sensitivity.csv", "text/csv")

with tab4:
    detail_rows = []
    for s in summaries:
        for label, ok in s["Validation_details"]:
            detail_rows.append({"Controller": s["Controller"], "Check": label, "Passed": bool(ok)})
    detail_df = pd.DataFrame(detail_rows)
    st.dataframe(detail_df, use_container_width=True, hide_index=True)
    st.write("Maximum absolute residuals:")
    st.dataframe(summary_df[["Controller", "Max_load_balance_residual_kW", "Max_solar_balance_residual_kW", "Max_SOC_balance_residual_kWh"]].round(10), use_container_width=True, hide_index=True)
    st.download_button("Download forecasting metrics CSV", metrics.to_csv(index=False).encode("utf-8"), "kajola_forecasting_metrics_v3.csv", "text/csv")
    st.download_button("Download ESS summary CSV", summary_df.to_csv(index=False).encode("utf-8"), "kajola_lstm_hybrid_ess_summary.csv", "text/csv")
    for name, r in results.items():
        safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("–", "-")
        st.download_button(f"Download hourly results — {name}", r.to_csv(index=False).encode("utf-8"), f"kajola_hourly_{safe}.csv", "text/csv", key=f"dl_{safe}")

with st.expander("Model assumptions and interpretation"):
    st.markdown("""
    - Each row represents a one-hour interval; hourly kW values are treated numerically as kWh per interval.
    - Solar serves the **actual observed load** first. Any actual solar surplus may charge the battery.
    - During an actual deficit, the LSTM or hybrid forecast limits planned battery discharge; the controller never discharges more than actual residual demand, the power limit, or available battery energy.
    - This is a transparent forecast-limited heuristic, not an optimized controller. Forecast errors can affect dispatch, but the result depends on the selected rule.
    - The actual-load reference uses actual demand to guide discharge and is only an idealized benchmark, not a deployable forecast-based controller.
    - Recorded solar generation is used as the same exogenous series for both controllers. Future solar forecasting is not included.
    - No grid charging, tariffs, degradation, temperature-dependent battery behaviour, inverter dynamics, or hardware validation is included.
    - Passing internal checks demonstrates accounting consistency under the implemented assumptions; it does not prove real-world battery performance.
    """)
st.caption("Research integrity: use these outputs as simulation estimates under stated assumptions. Do not describe them as measured field outcomes or claim the hybrid/LSTM is operationally superior without examining the results and limitations.")
