import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Kajola ESS Forecast Lab", page_icon="🔋", layout="wide", initial_sidebar_state="expanded")
DATA_DIR = Path(__file__).parent
DATA_FILE = DATA_DIR / "kajola_dataset_2024_2025.csv"
FORECAST_FILE = DATA_DIR / "test_forecasts.csv"
TOL = 1e-6

# Clean, high-contrast light theme inspired by the supplied interface.
st.markdown(r"""
<style>
:root { color-scheme: light; }
.stApp { background: #F3F7FA; color: #1F2937; }
[data-testid="stHeader"] { background: rgba(243,247,250,0.95); }
[data-testid="stSidebar"] { background: #EAF1F6; border-right: 1px solid #D5E0E9; }
[data-testid="stSidebar"] * { color: #203247; }
.block-container { padding-top: 1.25rem; padding-bottom: 2.5rem; max-width: 1500px; }
.hero { background: linear-gradient(115deg,#FFFFFF 0%,#E8F5F4 48%,#EAF0FC 100%); border: 1px solid #D5E3EA; border-radius: 22px; padding: 28px 32px; margin-bottom: 18px; box-shadow: 0 5px 18px rgba(30,58,80,.06); }
.eyebrow { display: inline-block; background: #DDF4EF; color: #126B61; border: 1px solid #B6E4DA; border-radius: 20px; padding: 5px 11px; font-size: 12px; font-weight: 700; margin: 0 6px 8px 0; }
.hero h1 { color: #16324F; font-size: clamp(28px,3vw,40px); line-height: 1.15; margin: 8px 0 10px 0; }
.hero p { color: #43566B; font-size: 16px; margin: 0; max-width: 950px; }
.hero .meta { color: #617386; font-size: 12px; margin-top: 16px; }
.section-title { color: #173A5E; font-weight: 750; font-size: 22px; margin: 6px 0 4px 0; }
.section-sub { color: #617386; font-size: 13px; margin-bottom: 14px; }
.info-card { background: #FFFFFF; border: 1px solid #D8E3EB; border-radius: 15px; padding: 18px 20px; min-height: 125px; box-shadow: 0 2px 8px rgba(27,55,78,.035); }
.info-card h3 { color: #173A5E; font-size: 16px; margin: 0 0 8px 0; }
.info-card p, .info-card li { color: #43566B; font-size: 13px; line-height: 1.55; }
.metric-card { background: white; border: 1px solid #D8E3EB; border-radius: 13px; padding: 15px 17px; }
.metric-label { color: #607489; font-size: 12px; font-weight: 650; }
.metric-value { color: #173A5E; font-size: 26px; font-weight: 750; margin-top: 4px; }
.metric-help { color: #64778A; font-size: 11px; margin-top: 2px; }
div[data-testid="stTabs"] button { color: #35516B; font-weight: 650; border-radius: 8px 8px 0 0; }
div[data-testid="stTabs"] button[aria-selected="true"] { color: #126B61; border-bottom: 3px solid #20A894; background: #E5F6F2; }
.stButton>button, .stDownloadButton>button { border-radius: 9px; border: 1px solid #B8D2D8; background: #E4F4F2; color: #145E59; font-weight: 650; }
[data-testid="stAlert"] { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

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
    d[["Load_kW", "Solar_Gen_kW"]] = d[["Load_kW", "Solar_Gen_kW"]].ffill()
    d = d.dropna(subset=["Load_kW", "Solar_Gen_kW"]).reset_index(drop=True)
    f = f.dropna(subset=["Timestamp"] + forecast_cols).sort_values("Timestamp").drop_duplicates("Timestamp").reset_index(drop=True)
    merged = f.merge(d[["Timestamp", "Load_kW", "Solar_Gen_kW"]], on="Timestamp", how="left", validate="one_to_one")
    if merged["Solar_Gen_kW"].isna().any() or merged["Load_kW"].isna().any():
        raise ValueError("Some forecast timestamps could not be matched to observed load and solar data.")
    mismatch = (merged["Actual_Load_kW"] - merged["Load_kW"]).abs().max()
    return d, merged, float(mismatch)


def metric_rows(f):
    actual = f["Actual_Load_kW"].to_numpy(dtype=float)
    rows = []
    for label, col in [("Naïve persistence", "Naive_Persistence_kW"), ("ANN", "ANN_kW"), ("LSTM", "LSTM_kW"), ("Differencing–ANN–Bi-LSTM", "Diff-ANN-BiLSTM_kW")]:
        pred = f[col].to_numpy(dtype=float)
        nz = np.abs(actual) > 1e-12
        rows.append({"Model": label, "MAE_kW": mean_absolute_error(actual, pred), "RMSE_kW": mean_squared_error(actual, pred) ** 0.5,
                     "MAPE_percent_nonzero_actuals": float(np.mean(np.abs((actual[nz]-pred[nz])/actual[nz]))*100) if nz.any() else np.nan,
                     "R2": r2_score(actual, pred)})
    return pd.DataFrame(rows)


def simulate_dispatch(frame, capacity_kwh, max_charge_kw, max_discharge_kw, initial_soc_pct, min_soc_pct, max_soc_pct, charge_efficiency, discharge_efficiency, forecast_col=None):
    """Hourly forecast-limited heuristic; actual load is used for realized energy accounting."""
    x = frame.reset_index(drop=True).copy()
    cap = float(capacity_kwh)
    soc_min, soc_max = cap*float(min_soc_pct)/100, cap*float(max_soc_pct)/100
    soc = float(np.clip(cap*float(initial_soc_pct)/100, soc_min, soc_max))
    rows = []
    for _, row in x.iterrows():
        actual = max(float(row["Actual_Load_kW"]), 0.0)
        solar = max(float(row["Solar_Gen_kW"]), 0.0)
        forecast = actual if forecast_col is None else max(float(row[forecast_col]), 0.0)
        direct = min(actual, solar)
        residual = max(actual-direct, 0.0)
        surplus = max(solar-actual, 0.0)
        charge = discharge = 0.0
        unserved, curtailed, soc_before = residual, surplus, soc
        if surplus > 0 and soc < soc_max:
            input_limit = max((soc_max-soc)/max(charge_efficiency,1e-12),0.0)
            charge = min(surplus,max_charge_kw,input_limit)
            soc += charge*charge_efficiency
            curtailed = surplus-charge
        elif residual > 0 and soc > soc_min:
            predicted_residual = max(forecast-solar,0.0)
            planned = residual if forecast_col is None else predicted_residual
            available = max((soc-soc_min)*discharge_efficiency,0.0)
            discharge = min(residual,planned,max_discharge_kw,available)
            soc -= discharge/max(discharge_efficiency,1e-12)
            unserved = residual-discharge
        rows.append({"Timestamp":row["Timestamp"],"Actual_Load_kW":actual,"Forecast_Load_kW":forecast,"Solar_Gen_kW":solar,
                     "Direct_Solar_to_Load_kW":direct,"Residual_Load_before_Battery_kW":residual,"Battery_Charge_Input_kW":charge,
                     "Battery_Discharge_to_Load_kW":discharge,"Unserved_Load_kW":max(unserved,0.0),"Curtailed_Solar_kW":max(curtailed,0.0),
                     "SOC_before_kWh":soc_before,"SOC_kWh":soc})
    r = pd.DataFrame(rows)
    r["Load_Energy_kWh"] = r["Actual_Load_kW"]
    r["Solar_Energy_kWh"] = r["Solar_Gen_kW"]
    r["Unserved_Energy_kWh"] = r["Unserved_Load_kW"]
    r["Served_Load_kW"] = r["Actual_Load_kW"]-r["Unserved_Load_kW"]
    r["Load_Balance_Residual_kW"] = r["Actual_Load_kW"]-r["Direct_Solar_to_Load_kW"]-r["Battery_Discharge_to_Load_kW"]-r["Unserved_Load_kW"]
    r["Solar_Balance_Residual_kW"] = r["Solar_Gen_kW"]-r["Direct_Solar_to_Load_kW"]-r["Battery_Charge_Input_kW"]-r["Curtailed_Solar_kW"]
    r["SOC_Balance_Residual_kWh"] = r["SOC_kWh"]-(r["SOC_before_kWh"]+r["Battery_Charge_Input_kW"]*charge_efficiency-r["Battery_Discharge_to_Load_kW"]/max(discharge_efficiency,1e-12))
    return r


def summarize(r, capacity, charge_power, discharge_power, min_soc_pct, max_soc_pct, charge_eff, discharge_eff):
    load_total, solar_total = float(r["Load_Energy_kWh"].sum()), float(r["Solar_Energy_kWh"].sum())
    no_ess = float(np.maximum(r["Actual_Load_kW"]-r["Solar_Gen_kW"],0).sum())
    unserved = float(r["Unserved_Energy_kWh"].sum())
    charge, discharge = float(r["Battery_Charge_Input_kW"].sum()), float(r["Battery_Discharge_to_Load_kW"].sum())
    losses = charge*(1-charge_eff)+(discharge/max(discharge_eff,1e-12)-discharge)
    lo, hi = capacity*min_soc_pct/100, capacity*max_soc_pct/100
    checks = [("Load balance",r["Load_Balance_Residual_kW"].abs().max()<=TOL),("Solar balance",r["Solar_Balance_Residual_kW"].abs().max()<=TOL),
              ("SOC transition balance",r["SOC_Balance_Residual_kWh"].abs().max()<=TOL),("Minimum SOC",r["SOC_kWh"].min()>=lo-TOL),
              ("Maximum SOC",r["SOC_kWh"].max()<=hi+TOL),("Charge power limit",r["Battery_Charge_Input_kW"].max()<=charge_power+TOL),
              ("Discharge power limit",r["Battery_Discharge_to_Load_kW"].max()<=discharge_power+TOL),
              ("Non-negative values",r[["SOC_kWh","Unserved_Load_kW","Curtailed_Solar_kW","Battery_Charge_Input_kW","Battery_Discharge_to_Load_kW"]].min().min()>=-TOL)]
    return {"Hours":len(r),"Load_kWh":load_total,"Solar_generation_kWh":solar_total,"Unserved_without_ESS_kWh":no_ess,"Unserved_with_ESS_kWh":unserved,
            "Unserved_reduction_vs_no_ESS_pct":(no_ess-unserved)/no_ess*100 if no_ess else 0,"Load_served_pct":(load_total-unserved)/load_total*100 if load_total else 0,
            "Battery_charge_input_kWh":charge,"Battery_delivered_kWh":discharge,"Curtailed_solar_kWh":float(r["Curtailed_Solar_kW"].sum()),
            "Estimated_conversion_losses_kWh":losses,"SOC_min_observed_kWh":float(r["SOC_kWh"].min()),"SOC_max_observed_kWh":float(r["SOC_kWh"].max()),
            "SOC_final_kWh":float(r["SOC_kWh"].iloc[-1]),"Hours_at_min_SOC":int(np.isclose(r["SOC_kWh"],lo,atol=1e-5).sum()),
            "Hours_at_max_SOC":int(np.isclose(r["SOC_kWh"],hi,atol=1e-5).sum()),"Max_load_balance_residual_kW":float(r["Load_Balance_Residual_kW"].abs().max()),
            "Max_solar_balance_residual_kW":float(r["Solar_Balance_Residual_kW"].abs().max()),"Max_SOC_balance_residual_kWh":float(r["SOC_Balance_Residual_kWh"].abs().max()),
            "Validation_checks_passed":sum(ok for _,ok in checks),"Validation_checks_total":len(checks),"Validation_details":checks}

try:
    data, forecasts, actual_mismatch = load_inputs()
except Exception as exc:
    st.error(f"Input-data problem: {exc}")
    st.stop()
if actual_mismatch > 1e-6:
    st.error(f"Forecast actual-load column differs from source dataset by {actual_mismatch:.6g} kW. Check the forecast file.")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.markdown("## 🔋 Kajola ESS Lab")
    st.caption("Research prototype • rural solar microgrid")
    st.divider()
    st.markdown("### Battery system")
    capacity = st.slider("Capacity (kWh)", 5, 200, 50, 5)
    min_soc = st.slider("Minimum SOC / protection floor (%)", 0, 50, 10, 5)
    max_soc = st.slider("Maximum SOC (%)", 50, 100, 95, 5)
    initial_soc = st.slider("Initial SOC (%)", 0, 100, 50, 5)
    charge_power = st.slider("Maximum charge power (kW)", 1, 50, 15, 1)
    discharge_power = st.slider("Maximum discharge power (kW)", 1, 50, 15, 1)
    charge_eff = st.slider("Charge efficiency (%)", 50, 100, 95, 1)/100
    discharge_eff = st.slider("Discharge efficiency (%)", 50, 100, 95, 1)/100
    st.divider()
    st.markdown("### Test period")
    start_default, end_default = forecasts["Timestamp"].min().date(), forecasts["Timestamp"].max().date()
    dates = st.date_input("Choose test dates", value=(start_default,end_default), min_value=start_default,max_value=end_default)
    st.caption("Forecast metrics use the held-out test period by default.")

if len(dates) != 2:
    st.info("Select both a start and end date to continue.")
    st.stop()
start_date,end_date=dates
if start_date>end_date or min_soc>=max_soc:
    st.error("Check the date range and SOC limits. Minimum SOC must be below maximum SOC.")
    st.stop()
frame=forecasts[(forecasts["Timestamp"].dt.date>=start_date)&(forecasts["Timestamp"].dt.date<=end_date)].copy().reset_index(drop=True)
if frame.empty:
    st.warning("No forecast observations exist for those dates.")
    st.stop()
metrics=metric_rows(frame)
model_cols={"LSTM":"LSTM_kW","Hybrid (Differencing–ANN–Bi-LSTM)":"Diff-ANN-BiLSTM_kW"}
results={}; summaries=[]
for name,col in model_cols.items():
    results[name]=simulate_dispatch(frame,capacity,charge_power,discharge_power,initial_soc,min_soc,max_soc,charge_eff,discharge_eff,col)
    sm=summarize(results[name],capacity,charge_power,discharge_power,min_soc,max_soc,charge_eff,discharge_eff); sm["Controller"]=name; summaries.append(sm)
results["Actual-load reference"]=simulate_dispatch(frame,capacity,charge_power,discharge_power,initial_soc,min_soc,max_soc,charge_eff,discharge_eff,None)
ref=summarize(results["Actual-load reference"],capacity,charge_power,discharge_power,min_soc,max_soc,charge_eff,discharge_eff); ref["Controller"]="Actual-load reference (benchmark only)"; summaries.append(ref)
summary_df=pd.DataFrame([{k:v for k,v in x.items() if k!="Validation_details"} for x in summaries])

# Hero header
st.markdown(f"""
<div class="hero">
 <span class="eyebrow">⚡ Kajola, Nigeria</span><span class="eyebrow">🕒 {len(frame):,} hourly test samples</span><span class="eyebrow">🔋 {capacity} kWh battery scenario</span>
 <h1>🔋 Kajola Microgrid Forecast &amp; ESS Lab</h1>
 <p>Compare one-hour-ahead LSTM and hybrid load forecasts, then evaluate how each forecast affects a transparent solar-battery dispatch scenario.</p>
 <div class="meta">Research prototype • Data period: {frame['Timestamp'].min():%d %b %Y} – {frame['Timestamp'].max():%d %b %Y} • Prepared by Olumodeji I.A | Bigjoe Tech Company</div>
</div>
""",unsafe_allow_html=True)
st.warning("Research prototype only. Battery settings are scenario assumptions, not measured Kajola equipment specifications. ESS results are simulated estimates, not field measurements.")

tabs=st.tabs(["🏠 Overview","📊 Data Explorer","🧠 Model Comparison","🔋 ESS Simulation","ℹ️ About & Validation"])

with tabs[0]:
    st.markdown('<div class="section-title">Overview</div><div class="section-sub">A concise view of the dataset, forecast quality and current ESS scenario.</div>',unsafe_allow_html=True)
    lstm=metrics.loc[metrics.Model=="LSTM"].iloc[0]; hybrid=metrics.loc[metrics.Model=="Differencing–ANN–Bi-LSTM"].iloc[0]
    total_load=float(frame.Actual_Load_kW.sum()); total_solar=float(frame.Solar_Gen_kW.sum())
    cards=st.columns(4)
    for col,label,value,helptext in [
        (cards[0],"Hourly test records",f"{len(frame):,}","Held-out observations"),
        (cards[1],"LSTM MAE",f"{lstm.MAE_kW:.3f} kW","Lower is better"),
        (cards[2],"Hybrid MAE",f"{hybrid.MAE_kW:.3f} kW","Lower is better"),
        (cards[3],"Observed demand",f"{total_load:,.0f} kWh","Sum over selected hourly test period")]:
        with col: st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-help">{helptext}</div></div>',unsafe_allow_html=True)
    st.write("")
    left,right=st.columns([1.15,1])
    with left:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=frame.Timestamp,y=frame.Actual_Load_kW,name="Actual load",line=dict(color="#173A5E",width=2)))
        fig.add_trace(go.Scatter(x=frame.Timestamp,y=frame.LSTM_kW,name="LSTM forecast",line=dict(color="#159A8C",width=1.5)))
        fig.add_trace(go.Scatter(x=frame.Timestamp,y=frame["Diff-ANN-BiLSTM_kW"],name="Hybrid forecast",line=dict(color="#E69B3A",width=1.5)))
        fig.update_layout(title="Actual demand and model forecasts",xaxis_title="Date",yaxis_title="Load (kW)",template="plotly_white",legend_orientation="h",legend_y=-.25,margin=dict(l=10,r=10,t=50,b=45))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        ess=summary_df[summary_df.Controller.isin(["LSTM","Hybrid (Differencing–ANN–Bi-LSTM)"])].copy()
        ess["Controller"]=ess["Controller"].replace({"Hybrid (Differencing–ANN–Bi-LSTM)":"Hybrid"})
        fig=px.bar(ess,x="Controller",y="Unserved_with_ESS_kWh",color="Controller",text_auto=".0f",title="Unserved energy under each forecast controller",color_discrete_map={"LSTM":"#159A8C","Hybrid":"#E69B3A"})
        fig.update_layout(template="plotly_white",showlegend=False,xaxis_title="Controller",yaxis_title="Unserved energy (kWh)",margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig,use_container_width=True)
    st.markdown('<div class="info-card"><h3>How to read this dashboard</h3><p>Forecast metrics describe prediction error. ESS metrics describe the outcome of a particular dispatch rule and set of battery assumptions. A lower forecasting error does not automatically guarantee a large improvement in energy-storage operation.</p></div>',unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div class="section-title">Data Explorer</div><div class="section-sub">Inspect the original hourly series and the held-out forecast test records.</div>',unsafe_allow_html=True)
    d1,d2,d3=st.columns(3)
    d1.metric("Full dataset rows",f"{len(data):,}")
    d2.metric("Forecast test rows",f"{len(forecasts):,}")
    d3.metric("Actual-load mismatch",f"{actual_mismatch:.6f} kW")
    view=st.selectbox("Select data view",["Selected test-period data","Full Kajola dataset"])
    if view=="Selected test-period data":
        display=frame[["Timestamp","Actual_Load_kW","Solar_Gen_kW","LSTM_kW","Diff-ANN-BiLSTM_kW"]].copy()
    else:
        display=data.copy()
    st.dataframe(display.head(500),use_container_width=True,hide_index=True)
    st.caption("Showing at most the first 500 rows for readability. Download the CSV to inspect the complete selection.")
    st.download_button("Download displayed data as CSV",display.to_csv(index=False).encode("utf-8"),"kajola_data_explorer.csv","text/csv")
    fig=px.line(data,x="Timestamp",y="Load_kW",title="Kajola recorded hourly electricity demand",labels={"Load_kW":"Load (kW)","Timestamp":"Time"},color_discrete_sequence=["#173A5E"])
    fig.update_layout(template="plotly_white",margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig,use_container_width=True)

with tabs[2]:
    st.markdown('<div class="section-title">Model Comparison</div><div class="section-sub">Compare forecast accuracy using the selected held-out test dates.</div>',unsafe_allow_html=True)
    st.dataframe(metrics.round(4),use_container_width=True,hide_index=True)
    c1,c2=st.columns(2)
    with c1:
        error=metrics[metrics.Model.isin(["LSTM","Differencing–ANN–Bi-LSTM"])][["Model","MAE_kW","RMSE_kW","MAPE_percent_nonzero_actuals"]].melt(id_vars="Model",var_name="Metric",value_name="Error")
        fig=px.bar(error,x="Metric",y="Error",color="Model",barmode="group",title="LSTM versus hybrid error metrics",color_discrete_map={"LSTM":"#159A8C","Differencing–ANN–Bi-LSTM":"#E69B3A"})
        fig.update_layout(template="plotly_white",margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=frame.Actual_Load_kW,y=frame.LSTM_kW,mode="markers",name="LSTM",marker=dict(color="#159A8C",size=5,opacity=.6)))
        fig.add_trace(go.Scatter(x=frame.Actual_Load_kW,y=frame["Diff-ANN-BiLSTM_kW"],mode="markers",name="Hybrid",marker=dict(color="#E69B3A",size=5,opacity=.55)))
        low=min(frame.Actual_Load_kW.min(),frame.LSTM_kW.min(),frame["Diff-ANN-BiLSTM_kW"].min()); high=max(frame.Actual_Load_kW.max(),frame.LSTM_kW.max(),frame["Diff-ANN-BiLSTM_kW"].max())
        fig.add_trace(go.Scatter(x=[low,high],y=[low,high],mode="lines",name="Ideal prediction",line=dict(color="#64748B",dash="dash")))
        fig.update_layout(title="Predicted versus actual demand",xaxis_title="Actual load (kW)",yaxis_title="Predicted load (kW)",template="plotly_white",margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig,use_container_width=True)
    st.info("Current Version 3 results: LSTM has lower MAE, RMSE and MAPE than the hybrid model. Report this result transparently; do not describe the hybrid as the most accurate model in this test.")
    st.download_button("Download forecast metrics CSV",metrics.to_csv(index=False).encode("utf-8"),"kajola_forecasting_metrics_v3.csv","text/csv")

with tabs[3]:
    st.markdown('<div class="section-title">ESS Simulation</div><div class="section-sub">A transparent forecast-limited battery dispatch heuristic evaluated against actual observed load.</div>',unsafe_allow_html=True)
    show=summary_df[summary_df.Controller.isin(["LSTM","Hybrid (Differencing–ANN–Bi-LSTM)"])].copy()
    show["Controller"]=show["Controller"].replace({"Hybrid (Differencing–ANN–Bi-LSTM)":"Hybrid"})
    st.dataframe(show[["Controller","Unserved_without_ESS_kWh","Unserved_with_ESS_kWh","Unserved_reduction_vs_no_ESS_pct","Battery_charge_input_kWh","Battery_delivered_kWh","Curtailed_solar_kWh","Estimated_conversion_losses_kWh","SOC_final_kWh","Validation_checks_passed","Validation_checks_total"]].round(3),use_container_width=True,hide_index=True)
    left,right=st.columns(2)
    with left:
        fig=go.Figure()
        for name,color in [("LSTM","#159A8C"),("Hybrid (Differencing–ANN–Bi-LSTM)","#E69B3A")]:
            fig.add_trace(go.Scatter(x=results[name].Timestamp,y=results[name].SOC_kWh,name=name,line=dict(color=color,width=1.7)))
        fig.add_hline(y=capacity*min_soc/100,line_dash="dash",line_color="#D9534F",annotation_text="Minimum SOC")
        fig.add_hline(y=capacity*max_soc/100,line_dash="dash",line_color="#64748B",annotation_text="Maximum SOC")
        fig.update_layout(title="Battery state of charge",xaxis_title="Time",yaxis_title="Stored energy (kWh)",template="plotly_white",margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig,use_container_width=True)
    with right:
        u=pd.DataFrame([{"Controller":n,"Unserved energy (kWh)":float(results[n].Unserved_Energy_kWh.sum())} for n in ["LSTM","Hybrid (Differencing–ANN–Bi-LSTM)"]])
        u["Controller"]=u["Controller"].replace({"Hybrid (Differencing–ANN–Bi-LSTM)":"Hybrid"})
        fig=px.bar(u,x="Controller",y="Unserved energy (kWh)",color="Controller",text_auto=".1f",title="Unserved energy",color_discrete_map={"LSTM":"#159A8C","Hybrid":"#E69B3A"})
        fig.update_layout(template="plotly_white",showlegend=False,margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig,use_container_width=True)
    st.subheader("Capacity sensitivity")
    st.caption("Only capacity changes in this sensitivity check; power limits and other settings are held fixed at the values stated below.")
    cap_rows=[]
    for cap_s in [25,50,75,100]:
        for name,col in model_cols.items():
            rr=simulate_dispatch(frame,cap_s,15,15,initial_soc,min_soc,max_soc,charge_eff,discharge_eff,col)
            sm=summarize(rr,cap_s,15,15,min_soc,max_soc,charge_eff,discharge_eff)
            cap_rows.append({"Capacity_kWh":cap_s,"Controller":name,"Unserved_with_ESS_kWh":sm["Unserved_with_ESS_kWh"],"Unserved_reduction_vs_no_ESS_pct":sm["Unserved_reduction_vs_no_ESS_pct"],"Curtailed_solar_kWh":sm["Curtailed_solar_kWh"],"Validation_checks_passed":sm["Validation_checks_passed"],"Validation_checks_total":sm["Validation_checks_total"]})
    cap_df=pd.DataFrame(cap_rows)
    st.dataframe(cap_df.round(3),use_container_width=True,hide_index=True)
    fig=px.line(cap_df,x="Capacity_kWh",y="Unserved_with_ESS_kWh",color="Controller",markers=True,title="Unserved energy by battery capacity",color_discrete_map={"LSTM":"#159A8C","Hybrid (Differencing–ANN–Bi-LSTM)":"#E69B3A"})
    fig.update_layout(template="plotly_white",margin=dict(l=10,r=10,t=50,b=10))
    st.plotly_chart(fig,use_container_width=True)
    st.download_button("Download hourly LSTM results",results["LSTM"].to_csv(index=False).encode("utf-8"),"kajola_hourly_lstm_ess.csv","text/csv")
    st.download_button("Download hourly hybrid results",results["Hybrid (Differencing–ANN–Bi-LSTM)"].to_csv(index=False).encode("utf-8"),"kajola_hourly_hybrid_ess.csv","text/csv")
    st.download_button("Download capacity sensitivity CSV",cap_df.to_csv(index=False).encode("utf-8"),"kajola_capacity_sensitivity.csv","text/csv")

with tabs[4]:
    st.markdown('<div class="section-title">About, Method and Validation</div><div class="section-sub">Research context, assumptions and limits that should accompany any interpretation.</div>',unsafe_allow_html=True)
    left,right=st.columns(2)
    with left:
        st.markdown('<div class="info-card"><h3>🎯 Research aim</h3><p>Compare short-term load forecasting approaches and examine whether their forecasts change battery dispatch outcomes for the Kajola solar-energy scenario.</p><h3>🧪 Method</h3><ul><li>Differencing is used in the proposed hybrid forecasting workflow.</li><li>Bi-LSTM models temporal structure in the input sequence.</li><li>ANN maps inputs to a next-hour prediction.</li><li>Chronological test data are used for forecast and ESS comparison.</li></ul></div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="info-card"><h3>📁 Data</h3><p>The source file contains hourly Kajola observations from 2024–2025. The forecast comparison uses the held-out Version 3 test period.</p><h3>⚠️ Limitations</h3><ul><li>Battery capacity and efficiencies are scenario assumptions, not verified field specifications.</li><li>Recorded solar generation is used; future solar forecasting is not modelled.</li><li>The dispatch rule is a heuristic, not an optimized controller.</li><li>No tariffs, battery degradation, inverter dynamics or hardware validation are included.</li></ul></div>',unsafe_allow_html=True)
    st.write("")
    st.subheader("Internal validation checks")
    detail=[]
    for sm in summaries:
        for label,ok in sm["Validation_details"]: detail.append({"Controller":sm["Controller"],"Check":label,"Passed":bool(ok)})
    detail_df=pd.DataFrame(detail)
    st.dataframe(detail_df,use_container_width=True,hide_index=True)
    st.write("Maximum absolute energy-balance residuals")
    st.dataframe(summary_df[["Controller","Max_load_balance_residual_kW","Max_solar_balance_residual_kW","Max_SOC_balance_residual_kWh"]].round(10),use_container_width=True,hide_index=True)
    st.download_button("Download ESS summary CSV",summary_df.to_csv(index=False).encode("utf-8"),"kajola_lstm_hybrid_ess_summary.csv","text/csv")
    st.warning("Internal consistency checks do not validate the model against physical battery measurements. Treat all ESS results as simulation estimates under the stated assumptions. Supervisor approval is still required before formally changing the proposal's specified simulation environment.")

st.caption("Kajola ESS research prototype • Olumodeji I.A • Bigjoe Tech Company • Results are scenario-based and should be interpreted with the stated assumptions and limitations.")
