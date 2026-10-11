import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Kajola ESS Simulator", page_icon="🔋", layout="wide")

DATA_DIR = Path(__file__).parent
DATA_FILE = DATA_DIR / "kajola_dataset_2024_2025.csv"
FORECAST_FILE = DATA_DIR / "test_forecasts.csv"

st.title("🔋 Kajola Energy Storage Simulator")
st.caption("Research prototype • Kajola, Nigeria • Hourly data, 2024–2025")
st.info(
    "This dashboard simulates an assumed battery dispatch policy using the supplied load and solar-generation data. "
    "It is a research scenario model, not a validated engineering design. Review all assumptions before using results in a thesis."
)

@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing {DATA_FILE.name}. Put it beside app.py.")
    data = pd.read_csv(DATA_FILE)
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")
    for col in ["Load_kW", "Solar_Gen_kW"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["Timestamp", "Load_kW", "Solar_Gen_kW"]).sort_values("Timestamp")
    return data

@st.cache_data
def load_forecasts():
    if not FORECAST_FILE.exists():
        return None
    f = pd.read_csv(FORECAST_FILE)
    f["Timestamp"] = pd.to_datetime(f["Timestamp"], errors="coerce")
    return f.dropna(subset=["Timestamp"]).sort_values("Timestamp")

def simulate_ess(frame, capacity_kwh, max_charge_kw, max_discharge_kw,
                 initial_soc_pct, min_soc_pct, max_soc_pct,
                 charge_efficiency, discharge_efficiency, policy):
    """Hourly energy-balance model. Solar serves load first; surplus charges battery."""
    out = frame.copy().reset_index(drop=True)
    n = len(out)
    soc_min = capacity_kwh * min_soc_pct / 100.0
    soc_max = capacity_kwh * max_soc_pct / 100.0
    soc = min(max(capacity_kwh * initial_soc_pct / 100.0, soc_min), soc_max)

    rows = []
    for i, row in out.iterrows():
        load = max(float(row["Simulated_Load_kW"]), 0.0)
        solar = max(float(row["Solar_Gen_kW"]), 0.0)
        direct_solar = min(load, solar)
        remaining_load = load - direct_solar
        solar_surplus = max(solar - load, 0.0)

        charge_input = 0.0
        discharge_to_load = 0.0
        unserved = remaining_load
        curtailment = solar_surplus
        soc_before = soc

        # Charge only from surplus recorded solar generation.
        if solar_surplus > 0 and soc < soc_max:
            max_input_by_soc = max((soc_max - soc) / max(charge_efficiency, 1e-9), 0.0)
            charge_input = min(solar_surplus, max_charge_kw, max_input_by_soc)
            soc += charge_input * charge_efficiency
            curtailment = solar_surplus - charge_input

        # Discharge only to serve residual demand; no grid charging in this policy.
        elif remaining_load > 0 and soc > soc_min:
            available_output_by_soc = max((soc - soc_min) * discharge_efficiency, 0.0)
            discharge_to_load = min(remaining_load, max_discharge_kw, available_output_by_soc)
            soc -= discharge_to_load / max(discharge_efficiency, 1e-9)
            unserved = remaining_load - discharge_to_load

        rows.append({
            "Timestamp": row["Timestamp"], "Load_kW": load, "Solar_Gen_kW": solar,
            "Direct_Solar_to_Load_kW": direct_solar, "Residual_Load_before_Battery_kW": remaining_load,
            "Battery_Charge_Input_kW": charge_input, "Battery_Discharge_to_Load_kW": discharge_to_load,
            "Unserved_Load_kW": max(unserved, 0.0), "Curtailed_Solar_kW": max(curtailment, 0.0),
            "SOC_kWh": soc, "SOC_before_kWh": soc_before,
        })
    result = pd.DataFrame(rows)
    result["Load_Energy_kWh"] = result["Load_kW"] # each row represents one hour
    result["Unserved_Energy_kWh"] = result["Unserved_Load_kW"]
    result["Served_Load_kW"] = result["Load_kW"] - result["Unserved_Load_kW"]
    return result

data = load_data()
forecast = load_forecasts()

with st.sidebar:
    st.header("Simulation settings")
    date_min = data["Timestamp"].min().date()
    date_max = data["Timestamp"].max().date()
    start_date, end_date = st.date_input(
        "Simulation period", value=(date_min, date_max),
        min_value=date_min, max_value=date_max
    )
    st.caption("For faster initial testing, select a shorter period such as 7–30 days.")
    load_source = st.selectbox(
        "Load series",
        ["Observed load (dataset)", "LSTM forecast (test period)", "Hybrid forecast (test period)"],
        index=0
    )
    st.divider()
    st.subheader("Battery assumptions")
    capacity = st.slider("Usable battery capacity (kWh)", 5.0, 500.0, 50.0, 5.0)
    charge_power = st.slider("Maximum charging power (kW)", 1.0, 200.0, 15.0, 1.0)
    discharge_power = st.slider("Maximum discharging power (kW)", 1.0, 200.0, 15.0, 1.0)
    initial_soc = st.slider("Initial state of charge (%)", 0, 100, 50, 5)
    min_soc = st.slider("Minimum state of charge (%)", 0, 50, 10, 5)
    max_soc = st.slider("Maximum state of charge (%)", 50, 100, 95, 5)
    charge_eff = st.slider("Charging efficiency (%)", 50, 100, 95, 1) / 100.0
    discharge_eff = st.slider("Discharging efficiency (%)", 50, 100, 95, 1) / 100.0
    run_button = st.button("Run ESS simulation", type="primary", use_container_width=True)

if start_date > end_date:
    st.error("The start date must be on or before the end date.")
    st.stop()
if min_soc >= max_soc:
    st.error("Minimum state of charge must be below maximum state of charge.")
    st.stop()

if load_source != "Observed load (dataset)":
    if forecast is None:
        st.error("Forecast CSV is missing. Use observed load or add test_forecasts.csv beside app.py.")
        st.stop()
    pred_col = "LSTM_kW" if load_source.startswith("LSTM") else "Diff-ANN-BiLSTM_kW"
    load_frame = forecast[["Timestamp", "Actual_Load_kW", pred_col]].merge(
        data[["Timestamp", "Solar_Gen_kW"]], on="Timestamp", how="left"
    )
    load_frame = load_frame.rename(columns={pred_col: "Simulated_Load_kW"})
    load_frame = load_frame[["Timestamp", "Simulated_Load_kW", "Solar_Gen_kW"]]
else:
    load_frame = data[["Timestamp", "Load_kW", "Solar_Gen_kW"]].rename(columns={"Load_kW": "Simulated_Load_kW"})

load_frame = load_frame[
    (load_frame["Timestamp"].dt.date >= start_date) &
    (load_frame["Timestamp"].dt.date <= end_date)
].dropna(subset=["Simulated_Load_kW", "Solar_Gen_kW"]).sort_values("Timestamp").reset_index(drop=True)

if load_frame.empty:
    st.warning("No rows are available for the selected dates and load series. The LSTM/hybrid options only cover the held-out test period.")
    st.stop()

# Simulation runs on every rerun so changing a slider updates the outputs.
result = simulate_ess(
    load_frame, capacity, charge_power, discharge_power, initial_soc, min_soc, max_soc,
    charge_eff, discharge_eff, policy="solar_surplus_first"
)
total_load = float(result["Load_Energy_kWh"].sum())
unserved_no_ess = float(np.maximum(result["Load_kW"] - np.minimum(result["Load_kW"], result["Solar_Gen_kW"]), 0).sum())
unserved_with_ess = float(result["Unserved_Energy_kWh"].sum())
solar_energy = float(result["Solar_Gen_kW"].sum())
battery_charge = float(result["Battery_Charge_Input_kW"].sum())
battery_discharge = float(result["Battery_Discharge_to_Load_kW"].sum())
curtailed = float(result["Curtailed_Solar_kW"].sum())
load_served = max(total_load - unserved_with_ess, 0.0)
unmet_reduction = ((unserved_no_ess - unserved_with_ess) / unserved_no_ess * 100) if unserved_no_ess > 0 else 0.0
served_pct = (load_served / total_load * 100) if total_load > 0 else 0.0
# Estimated conversion loss inside battery based on hourly energy balance.
charge_stored = battery_charge * charge_eff
battery_internal_loss = battery_charge * (1-charge_eff) + (battery_discharge / max(discharge_eff, 1e-9) - battery_discharge)

st.subheader("Simulation summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total load", f"{total_load:,.1f} kWh")
c2.metric("Unserved energy with ESS", f"{unserved_with_ess:,.1f} kWh",
          delta=f"{unserved_no_ess-unserved_with_ess:,.1f} kWh vs no battery", delta_color="normal")
c3.metric("Load served", f"{served_pct:.2f}%")
c4.metric("Solar energy available", f"{solar_energy:,.1f} kWh")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Battery charged", f"{battery_charge:,.1f} kWh")
c6.metric("Battery delivered", f"{battery_discharge:,.1f} kWh")
c7.metric("Solar curtailed", f"{curtailed:,.1f} kWh")
c8.metric("Estimated battery conversion loss", f"{battery_internal_loss:,.1f} kWh")

st.caption(
    f"Assumptions: {capacity:.0f} kWh capacity; charge/discharge limits {charge_power:.0f}/{discharge_power:.0f} kW; "
    f"efficiencies {charge_eff:.0%}/{discharge_eff:.0%}; SOC range {min_soc}%–{max_soc}%. "
    "Each record is treated as one hour."
)

tab1, tab2, tab3, tab4 = st.tabs(["Energy balance", "Battery operation", "Forecast comparison", "Results table"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result["Timestamp"], y=result["Load_kW"], name="Load", line=dict(width=1.5)))
    fig.add_trace(go.Scatter(x=result["Timestamp"], y=result["Solar_Gen_kW"], name="Solar generation", line=dict(width=1.2)))
    fig.add_trace(go.Scatter(x=result["Timestamp"], y=result["Battery_Discharge_to_Load_kW"], name="Battery discharge", line=dict(width=1.2)))
    fig.add_trace(go.Scatter(x=result["Timestamp"], y=result["Unserved_Load_kW"], name="Unserved load", line=dict(width=1.2, dash="dot")))
    fig.update_layout(title="Hourly load, solar generation and ESS contribution", xaxis_title="Time", yaxis_title="Power (kW)", legend_title="Series")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**No-ESS versus ESS unserved energy**")
        compare = pd.DataFrame({
            "Scenario": ["Without ESS", "With ESS"],
            "Unserved energy (kWh)": [unserved_no_ess, unserved_with_ess]
        })
        fig2 = px.bar(compare, x="Scenario", y="Unserved energy (kWh)", text_auto=".2f")
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.markdown("**Hourly unserved load**")
        fig3 = px.line(result, x="Timestamp", y="Unserved_Load_kW", labels={"Unserved_Load_kW":"Unserved load (kW)", "Timestamp":"Time"})
        st.plotly_chart(fig3, use_container_width=True)

with tab2:
    fig_soc = go.Figure()
    fig_soc.add_trace(go.Scatter(x=result["Timestamp"], y=result["SOC_kWh"], name="Battery SOC (kWh)", fill="tozeroy"))
    fig_soc.add_hline(y=capacity*min_soc/100, line_dash="dash", annotation_text="Minimum SOC")
    fig_soc.add_hline(y=capacity*max_soc/100, line_dash="dash", annotation_text="Maximum SOC")
    fig_soc.update_layout(title="Battery state of charge", xaxis_title="Time", yaxis_title="Stored energy (kWh)")
    st.plotly_chart(fig_soc, use_container_width=True)

    fig_power = go.Figure()
    fig_power.add_trace(go.Bar(x=result["Timestamp"], y=result["Battery_Charge_Input_kW"], name="Charge input"))
    fig_power.add_trace(go.Bar(x=result["Timestamp"], y=-result["Battery_Discharge_to_Load_kW"], name="Discharge to load"))
    fig_power.update_layout(title="Battery charge (+) and discharge (−)", xaxis_title="Time", yaxis_title="Power (kW)", barmode="relative")
    st.plotly_chart(fig_power, use_container_width=True)

with tab3:
    st.markdown("The LSTM/hybrid forecast choices cover the held-out test period only. Forecast-based simulation is an illustrative operational scenario; solar generation is aligned from the recorded dataset.")
    if forecast is not None:
        merged = forecast[["Timestamp", "Actual_Load_kW", "LSTM_kW", "Diff-ANN-BiLSTM_kW"]].merge(
            data[["Timestamp", "Solar_Gen_kW"]], on="Timestamp", how="left"
        )
        merged = merged[(merged["Timestamp"].dt.date >= start_date) & (merged["Timestamp"].dt.date <= end_date)]
        if not merged.empty:
            figf = go.Figure()
            figf.add_trace(go.Scatter(x=merged["Timestamp"], y=merged["Actual_Load_kW"], name="Actual load"))
            figf.add_trace(go.Scatter(x=merged["Timestamp"], y=merged["LSTM_kW"], name="LSTM forecast"))
            figf.add_trace(go.Scatter(x=merged["Timestamp"], y=merged["Diff-ANN-BiLSTM_kW"], name="Hybrid forecast"))
            figf.update_layout(title="Actual load and model forecasts (test period)", xaxis_title="Time", yaxis_title="Load (kW)")
            st.plotly_chart(figf, use_container_width=True)
        else:
            st.warning("Select dates inside the test period to see forecast comparison.")
    else:
        st.warning("Forecast file not found.")

with tab4:
    st.dataframe(result, use_container_width=True, hide_index=True)
    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button("Download hourly ESS results (CSV)", csv, "kajola_ess_simulation_results.csv", "text/csv")
    summary = pd.DataFrame([{
        "start": str(result["Timestamp"].min()),
        "end": str(result["Timestamp"].max()),
        "hours_simulated": len(result),
        "load_energy_kWh": total_load,
        "solar_energy_kWh": solar_energy,
        "unserved_no_ESS_kWh": unserved_no_ess,
        "unserved_with_ESS_kWh": unserved_with_ess,
        "unserved_energy_reduction_percent": unmet_reduction,
        "load_served_percent": served_pct,
        "battery_charge_input_kWh": battery_charge,
        "battery_discharge_to_load_kWh": battery_discharge,
        "curtailed_solar_kWh": curtailed,
        "estimated_battery_conversion_loss_kWh": battery_internal_loss,
        "capacity_kWh": capacity,
        "charge_power_kW": charge_power,
        "discharge_power_kW": discharge_power,
        "initial_SOC_percent": initial_soc,
        "min_SOC_percent": min_soc,
        "max_SOC_percent": max_soc,
        "charge_efficiency": charge_eff,
        "discharge_efficiency": discharge_eff
    }])
    st.download_button("Download simulation summary (CSV)", summary.to_csv(index=False).encode("utf-8"),
                       "kajola_ess_summary.csv", "text/csv")

with st.expander("How to interpret the results"):
    st.markdown("""
    - **Unserved energy** is the estimated electricity demand not met by direct solar generation or battery discharge.
    - **Load served (%)** is the proportion of recorded/simulated load served under the stated assumptions.
    - **Battery charge/discharge** are hourly energy quantities because each input record represents one hour.
    - **Solar curtailed** is solar generation left unused after serving load and charging the battery.
    - **Battery conversion loss** estimates charge and discharge conversion losses from the selected efficiencies.
    - This simple dispatch rule charges only from solar surplus; it does not use grid charging, tariffs, degradation, temperature-dependent performance, or an inverter model.
    - The dataset's recorded solar generation and the assumed battery parameters should be validated before results are treated as engineering findings.
    """)
st.caption("Research integrity note: do not report the scenario outputs as measured field outcomes. Document assumptions and validate the dispatch model before final thesis submission.")
