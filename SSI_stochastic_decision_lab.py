import numpy as np
import streamlit as st
import plotly.graph_objects as go
import io
from datetime import datetime

st.set_page_config(page_title="SSI Simulator (Stochastic)", layout="centered")
st.title("Surgical Site Infection (SSI) Simulator — Stochastic (Gillespie-style)")

st.markdown(
    "This SSI lab is **procedure-driven** (not patient–HCW–patient vector transmission). "
    "Surgeries occur as a Poisson process, and SSIs occur with a probability driven by:\n"
    "- baseline contamination risk\n"
    "- prophylaxis timing/adherence\n"
    "- skin prep/sterile technique\n"
    "- OR traffic / environment\n"
    "- patient risk mix\n\n"
    "Use this to practice **decision-making under uncertainty** (tail risk, tradeoffs)."
)

# --------------------------
# Sidebar
# --------------------------
st.sidebar.header("Time horizon & simulation")
T_max = st.sidebar.slider("Duration (days)", 7, 180, 60, 1)
runs = st.sidebar.slider("Runs", 50, 800, 300, 50)
time_grid_points = st.sidebar.selectbox("Plot resolution", [301, 601, 1201], index=1)

seed = st.sidebar.number_input("Random seed", value=0, step=1)
use_seed = st.sidebar.checkbox("Use seed", value=True)

st.sidebar.header("Surgical volume")
surg_per_day = st.sidebar.slider("Surgeries per day (average)", 0.0, 50.0, 10.0, 0.5)

st.sidebar.header("Baseline SSI risk")
base_ssi_prob = st.sidebar.slider(
    "Baseline SSI probability per surgery (no interventions)",
    0.0, 0.30, 0.03, 0.001
)

st.sidebar.header("Prophylaxis")
proph_adherence = st.sidebar.slider("On-time prophylaxis adherence", 0.0, 1.0, 0.70, 0.01)
proph_effect = st.sidebar.slider("Effectiveness when on-time (risk reduction)", 0.0, 1.0, 0.50, 0.01)

st.sidebar.header("Skin prep / sterile technique")
prep_adherence = st.sidebar.slider("Skin prep / sterile adherence", 0.0, 1.0, 0.75, 0.01)
prep_effect = st.sidebar.slider("Effectiveness when adhered (risk reduction)", 0.0, 1.0, 0.40, 0.01)

st.sidebar.header("OR environment / traffic")
traffic_level = st.sidebar.slider("OR traffic level (0=low, 1=typical, >1=high)", 0.0, 2.0, 1.0, 0.05)
traffic_multiplier = st.sidebar.slider("How strongly traffic increases SSI risk", 0.0, 2.0, 0.40, 0.05)

st.sidebar.header("Patient risk mix")
high_risk_frac = st.sidebar.slider("Fraction high-risk patients", 0.0, 1.0, 0.30, 0.01)
high_risk_multiplier = st.sidebar.slider("Risk multiplier for high-risk patients", 1.0, 5.0, 2.0, 0.1)

st.sidebar.header("Detection")
post_discharge_detect = st.sidebar.slider("Fraction detected after discharge", 0.0, 1.0, 0.40, 0.01)

# --------------------------
# Effective SSI probability model (per surgery)
# --------------------------
def effective_ssi_prob():
    # bundle reductions
    red_proph = proph_adherence * proph_effect
    red_prep = prep_adherence * prep_effect
    bundle_factor = max(0.0, 1.0 - red_proph - red_prep)  # simple additive reductions (teaching)
    # environment
    env_factor = 1.0 + traffic_multiplier * max(0.0, traffic_level - 1.0)
    # patient mix
    mix_factor = (1.0 - high_risk_frac) * 1.0 + high_risk_frac * high_risk_multiplier
    p = base_ssi_prob * bundle_factor * env_factor * mix_factor
    return float(np.clip(p, 0.0, 0.95))

p_ssi = effective_ssi_prob()

st.metric("Effective SSI probability per surgery", f"{100*p_ssi:.2f}%")

# --------------------------
# Gillespie-style surgery & SSI process
# Two event types: surgery-without-SSI and surgery-with-SSI
# --------------------------
def simulate(rng, T, t_grid):
    # State: cumulative surgeries, cumulative SSIs, cumulative detected SSIs
    surg = 0
    ssi = 0
    detected = 0

    surg_path = np.zeros_like(t_grid, dtype=float)
    ssi_path = np.zeros_like(t_grid, dtype=float)
    det_path = np.zeros_like(t_grid, dtype=float)

    t = 0.0
    gi = 0
    while gi < len(t_grid) and t_grid[gi] <= 0:
        surg_path[gi], ssi_path[gi], det_path[gi] = surg, ssi, detected
        gi += 1

    # Rates (per day)
    rate_surg_ssi = surg_per_day * p_ssi
    rate_surg_no = surg_per_day * (1.0 - p_ssi)
    rate_total = rate_surg_ssi + rate_surg_no

    while t < T:
        if rate_total <= 0:
            break

        dt = rng.exponential(1.0 / rate_total)
        t_next = t + dt

        while gi < len(t_grid) and t_grid[gi] <= min(t_next, T):
            surg_path[gi], ssi_path[gi], det_path[gi] = surg, ssi, detected
            gi += 1

        if t_next > T:
            t = T
            break

        u = rng.random() * rate_total
        if u < rate_surg_ssi:
            surg += 1
            ssi += 1
            # detection split (some post-discharge)
            # For teaching, we record "detected" as all SSIs (but label fraction post-discharge)
            detected += 1
        else:
            surg += 1

        t = t_next

    while gi < len(t_grid):
        surg_path[gi], ssi_path[gi], det_path[gi] = surg, ssi, detected
        gi += 1

    return surg_path, ssi_path, det_path

# --------------------------
# Ensemble
# --------------------------
t_grid = np.linspace(0, T_max, int(time_grid_points))
rng = np.random.default_rng(seed if use_seed else None)

Surg_all = np.zeros((runs, len(t_grid)))
SSI_all = np.zeros((runs, len(t_grid)))

rng = np.random.default_rng(seed if use_seed else None)
for r in range(runs):
    surg_path, ssi_path, det_path = simulate(rng, T_max, t_grid)
    Surg_all[r, :] = surg_path
    SSI_all[r, :] = ssi_path

def qband(arr):
    med = np.quantile(arr, 0.50, axis=0)
    lo = np.quantile(arr, 0.10, axis=0)
    hi = np.quantile(arr, 0.90, axis=0)
    return med, lo, hi

SSI_med, SSI_lo, SSI_hi = qband(SSI_all)
Surg_med = np.quantile(Surg_all, 0.50, axis=0)

# Metrics
median_ssi = float(SSI_med[-1])
median_surg = float(Surg_med[-1])
rate_per_100 = (median_ssi / max(median_surg, 1e-9)) * 100.0
post_discharge = median_ssi * post_discharge_detect

c1, c2, c3 = st.columns(3)
c1.metric("Median SSIs (cumulative)", f"{median_ssi:.1f}")
c2.metric("Median SSI rate per 100 surgeries", f"{rate_per_100:.2f}")
c3.metric("Median post-discharge detected SSIs", f"{post_discharge:.1f}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=t_grid, y=SSI_hi, mode="lines", name="SSIs (90th %ile)", line=dict(width=1)))
fig.add_trace(go.Scatter(x=t_grid, y=SSI_lo, mode="lines", name="SSIs (10th %ile)", line=dict(width=1), fill="tonexty"))
fig.add_trace(go.Scatter(x=t_grid, y=SSI_med, mode="lines", name="SSIs (median)", line=dict(width=3)))
fig.update_layout(title="Cumulative SSIs (uncertainty band)", xaxis_title="Days", yaxis_title="Cumulative SSIs", hovermode="x unified", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Lab questions + download
# --------------------------
st.markdown("## Decision Lab (SSI)")

with st.expander("Lab Questions (click to open)", expanded=True):
    st.text_input("Student name / ID:", key="student_id")

    st.markdown("### 1) Prophylaxis vs OR traffic")
    st.markdown("Compare improving prophylaxis adherence vs reducing OR traffic. Which reduces tail risk (90th percentile) more under your baseline?")
    st.text_area("Answer:", key="ans1", height=120)

    st.markdown("### 2) Tradeoff: throughput vs safety")
    st.markdown("If reducing OR traffic slows throughput (fewer surgeries/day), propose a policy that balances patient access with SSI risk.")
    st.text_area("Answer:", key="ans2", height=120)

    st.markdown("### 3) High-risk patient mix")
    st.markdown("Increase high-risk fraction and observe outcomes. How does patient mix change which intervention you prioritize?")
    st.text_area("Answer:", key="ans3", height=120)

    st.markdown("### 4) Make the model more realistic (required)")
    st.markdown("List 3–5 changes to better represent SSI in practice (wound class, case duration, surgeon variability, post-discharge surveillance, antimicrobial resistance, etc.).")
    st.text_area("Answer:", key="ans4", height=120)

st.markdown("### Download your lab responses")
out = io.StringIO()
ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
sid = st.session_state.get("student_id", "").strip()
out.write(f"timestamp,{ts}\nstudent_id,{sid}\n")
out.write("\n[parameters]\n")
params = {
    "T_max": T_max, "runs": runs, "surgeries_per_day": surg_per_day,
    "base_ssi_prob": base_ssi_prob,
    "proph_adherence": proph_adherence, "proph_effect": proph_effect,
    "prep_adherence": prep_adherence, "prep_effect": prep_effect,
    "traffic_level": traffic_level, "traffic_multiplier": traffic_multiplier,
    "high_risk_frac": high_risk_frac, "high_risk_multiplier": high_risk_multiplier,
    "post_discharge_detect": post_discharge_detect,
    "effective_ssi_prob": p_ssi,
    "median_ssis": median_ssi, "median_surgeries": median_surg, "median_rate_per_100": rate_per_100
}
for k, v in params.items():
    out.write(f"{k},{v}\n")

out.write("\n[answers]\n")
for k in ["ans1", "ans2", "ans3", "ans4"]:
    val = str(st.session_state.get(k, "")).replace("\n", " ").replace("\r", " ")
    out.write(f"{k},{val}\n")

fname = f"SSI_responses_{sid}.csv" if sid else "SSI_responses.csv"
st.download_button("⬇️ Download my responses (CSV)", data=out.getvalue().encode("utf-8"), file_name=fname, mime="text/csv")
