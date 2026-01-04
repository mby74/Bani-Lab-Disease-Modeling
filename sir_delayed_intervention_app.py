import io
from datetime import datetime
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

st.set_page_config(page_title="SIR Simulator: Delayed Intervention", layout="centered")

st.title("SIR Simulator (Local Community) — Delayed Intervention")
st.write(
    "This simulator uses an SIR model where **β changes at a chosen day** to represent a community intervention "
    "(e.g., behavior change, public messaging, policy changes that reduce effective infectious contacts)."
)

# -----------------------
# Sidebar: Inputs
# -----------------------
st.sidebar.header("Community & Initial Conditions")

N = st.sidebar.number_input("Total population (N)", min_value=1, value=1000, step=50)

S0 = st.sidebar.number_input("Initial susceptible (S0)", min_value=0, value=int(N - 1), step=1)
I0 = st.sidebar.number_input("Initial infected (I0)", min_value=0, value=1, step=1)
R0_init = st.sidebar.number_input("Initial recovered (R0)", min_value=0, value=0, step=1)

days = st.sidebar.slider("Simulation duration (days)", min_value=1, max_value=365, value=160, step=1)

st.sidebar.header("Disease Dynamics")

gamma = st.sidebar.slider(
    "Recovery rate γ (per day) — γ = 1/(avg infectious period)",
    min_value=0.0, max_value=2.0, value=0.10, step=0.01,
    help="Higher γ means people stop being infectious sooner."
)

st.sidebar.header("Transmission & Intervention (β changes over time)")

use_intervention = st.sidebar.checkbox(
    "Enable delayed intervention (β switches on a chosen day)",
    value=True
)

beta_before = st.sidebar.slider(
    "β before intervention (effective infectious contacts/day)",
    min_value=0.0, max_value=2.0, value=0.35, step=0.01
)

beta_after = st.sidebar.slider(
    "β after intervention (effective infectious contacts/day)",
    min_value=0.0, max_value=2.0, value=0.25, step=0.01,
    help="Lower β models fewer effective infectious contacts (e.g., distancing, masks)."
)

intervention_day = st.sidebar.slider(
    "Intervention start day",
    min_value=0, max_value=int(days), value=20, step=1,
    help="Day when β switches from 'before' to 'after'."
)

# Guardrail on initial conditions
if S0 + I0 + R0_init != N:
    st.error(f"S0 + I0 + R0 must equal N. Currently: {S0 + I0 + R0_init} (N={N}).")
    st.stop()

# -----------------------
# Helpers & clinical interpretation
# -----------------------
infectious_period_days = (1 / gamma) if gamma > 0 else np.inf
R0_before = (beta_before / gamma) if gamma > 0 else np.inf
R0_after = (beta_after / gamma) if gamma > 0 else np.inf

st.sidebar.subheader("Clinical interpretation")
st.sidebar.write("- **β**: effective infectious contacts per day in the community")
st.sidebar.write("- **γ**: recovery rate per day")
st.sidebar.write(
    f"- **Average infectious period** ≈ **1/γ = {infectious_period_days:.1f} days**"
    if np.isfinite(infectious_period_days) else "- **Average infectious period** ≈ **∞** (γ=0)"
)
st.sidebar.caption("Rule of thumb: γ=0.2 → ~5 days infectious; γ=0.1 → ~10 days infectious.")

# -----------------------
# Model: time-varying β(t)
# -----------------------
def beta_t(t: float) -> float:
    if use_intervention and t >= intervention_day:
        return beta_after
    return beta_before

def sir_rhs(t, y):
    S, I, R = y
    b = beta_t(float(t))
    dS = -b * S * I / N
    dI = b * S * I / N - gamma * I
    dR = gamma * I
    return [dS, dI, dR]

t_eval = np.linspace(0, days, days + 1)

sol = solve_ivp(
    sir_rhs,
    t_span=(0, days),
    y0=[S0, I0, R0_init],
    t_eval=t_eval,
    rtol=1e-6,
    atol=1e-9
)

S, I, R = sol.y

# -----------------------
# Summary stats
# -----------------------
peak_I = float(I.max())
day_peak = float(t_eval[int(I.argmax())])
final_R = float(R[-1])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("R₀ before", f"{R0_before:.2f}" if np.isfinite(R0_before) else "∞")
c2.metric("R₀ after", f"{R0_after:.2f}" if np.isfinite(R0_after) else "∞")
c3.metric("Peak infected", f"{peak_I:.0f}")
c4.metric("Day of peak", f"{day_peak:.0f}")
c5.metric("Final recovered", f"{final_R:.0f}")

# -----------------------
# Plot
# -----------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=t_eval, y=S, mode="lines", name="Susceptible (S)"))
fig.add_trace(go.Scatter(x=t_eval, y=I, mode="lines", name="Infected (I)"))
fig.add_trace(go.Scatter(x=t_eval, y=R, mode="lines", name="Recovered (R)"))

# Vertical line for intervention timing (if enabled)
if use_intervention:
    fig.add_vline(
        x=intervention_day,
        line_width=2,
        line_dash="dash",
        annotation_text="Intervention starts",
        annotation_position="top right"
    )

fig.update_layout(
    xaxis_title="Days",
    yaxis_title="People",
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Virtual lab questions
# -----------------------
st.markdown("## Virtual Lab Questions (Local Community)")

st.markdown(
    "Use the sliders to run experiments. The goal is to understand how **R₀**, intervention timing, "
    "and infectious period shape outbreaks at the community level."
)

with st.expander("Open lab questions", expanded=True):

    st.markdown("### 1) Outbreak threshold (R₀ near 1)")
    st.markdown("Adjust β and γ so that **R₀ before** is just below 1, then just above 1.")
    st.text_area(
        "What changes in the infected curve I(t) when you cross R₀ = 1?",
        key="q1", height=90
    )

    st.markdown("### 2) Timing matters: delayed intervention (interactive)")
    st.markdown(
        "Keep **β before** and **β after** fixed. Move **Intervention start day** earlier vs later. "
        "Observe peak infected, day of peak, and final recovered."
    )
    st.text_area(
        "What happened when the intervention started earlier vs later? Explain why that makes sense.",
        key="q2", height=90
    )

    st.markdown("### 3) Contact reduction vs faster isolation")
    st.markdown(
        "Compare two strategies: (A) lower β (fewer effective infectious contacts), (B) increase γ (shorter infectious period)."
    )
    st.text_area(
        "Which strategy reduced the outbreak more in your runs? Under what parameter values?",
        key="q3", height=90
    )

    st.markdown("### 4) Community planning target")
    st.markdown("Try to keep **peak infected below 5% of the population** at any one time.")
    st.text_input(
        "Record one set of values (β before, β after, intervention day, γ) that meets the goal:",
        key="q4"
    )

with st.expander("Show data table"):
    st.dataframe({"day": t_eval, "S": S, "I": I, "R": R})

st.caption("Note: This is a simplified deterministic SIR model intended for learning. Real outbreaks include many additional factors.")

st.markdown("### Download your lab responses")

# ---- Choose which keys to export (works even if some keys don't exist) ----
lab_keys = [
    # For your earlier lab version (checkbox + text areas)
    "q1a", "q1b", "ans1", "ans2", "ans3", "ans4", "ans5", "ans6", "ans_delay",
    # For the delayed-intervention fresh app
    "q1", "q2", "q3", "q4"
]

# ---- Collect parameters (adjust names depending on your app) ----
params = {}
# These exist in the delayed-intervention app:
for name in ["N", "S0", "I0", "R0_init", "days", "gamma", "beta_before", "beta_after", "intervention_day",
             "R0_before", "R0_after", "peak_I", "day_peak", "final_R"]:
    if name in globals():
        params[name] = globals()[name]

# If you have a single beta (non-intervention version), include it too:
for name in ["beta", "R0_basic"]:
    if name in globals():
        params[name] = globals()[name]

# ---- Build CSV in-memory ----
output = io.StringIO()
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
output.write("timestamp,{}\n".format(timestamp))

# Write parameters
output.write("\n[parameters]\n")
for k, v in params.items():
    output.write(f"{k},{v}\n")

# Write lab responses
output.write("\n[lab_responses]\n")
for k in lab_keys:
    if k in st.session_state:
        # Replace newlines to keep CSV readable
        val = str(st.session_state.get(k, "")).replace("\n", " ").replace("\r", " ")
        output.write(f"{k},{val}\n")

csv_bytes = output.getvalue().encode("utf-8")

st.download_button(
    label="⬇️ Download my lab responses (CSV)",
    data=csv_bytes,
    file_name="sir_lab_responses.csv",
    mime="text/csv"
)
