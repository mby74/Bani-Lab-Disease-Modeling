import io
from datetime import datetime
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

st.set_page_config(page_title="SIR Model Simulator", layout="centered")

st.title("SIR Disease Model Simulator")
st.write("Adjust the sliders to explore how an outbreak changes over time.")

# --- Sidebar controls ---
st.sidebar.header("Inputs")

N = st.sidebar.number_input("Total population (N)", min_value=1, value=1000, step=50)

S0 = st.sidebar.number_input("Initial susceptible (S0)", min_value=0, value=int(N - 1), step=1)
I0 = st.sidebar.number_input("Initial infected (I0)", min_value=0, value=1, step=1)
R0_init = st.sidebar.number_input("Initial recovered (R0)", min_value=0, value=0, step=1)

beta = st.sidebar.slider(
    "Transmission rate β (effective infectious contacts per person per day)",
    min_value=0.0, max_value=2.0, value=0.35, step=0.01,
    help="β controls how quickly infection spreads. Roughly: higher β → faster growth in cases."
)

gamma = st.sidebar.slider(
    "Recovery rate γ (per day)  —  γ = 1 / (average infectious period)",
    min_value=0.0, max_value=2.0, value=0.10, step=0.01,
    help="γ controls how quickly people stop being infectious. Higher γ → shorter infectious period."
)

# --- Clinical interpretation helpers ---
if gamma > 0:
    infectious_period_days = 1 / gamma
else:
    infectious_period_days = np.inf

st.sidebar.subheader("Clinical interpretation")
st.sidebar.write(f"- **R₀**:reproduction number")
st.sidebar.write(f"- **β**: effective infectious contacts per day")
st.sidebar.write(f"- **γ**: recovery rate per day")
st.sidebar.write(f"- **Average infectious period** ≈ **1/γ** = **{infectious_period_days:.1f} days**" if np.isfinite(infectious_period_days) else
                 "- **Average infectious period** ≈ **1/γ** = **∞** (γ=0)")

st.sidebar.caption("Rule of thumb: γ = 0.2 → ~5 days infectious; γ = 0.1 → ~10 days infectious.")

days = st.sidebar.slider("Simulation duration (days)", min_value=1, max_value=365, value=160, step=1)

# --- Guardrail: make sure initial conditions are valid ---
if S0 + I0 + R0_init != N:
    st.warning(f"S0 + I0 + R0 must equal N. Currently: {S0 + I0 + R0_init} (N={N}).")
    st.stop()

# --- SIR ODE system ---
def sir_rhs(t, y):
    S, I, R = y
    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
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

# --- Summary stats ---
R0_basic = beta / gamma if gamma > 0 else np.inf
peak_I = float(I.max())
day_peak = float(t_eval[I.argmax()])
final_R = float(R[-1])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("R₀ = β/γ", f"{R0_basic:.2f}" if np.isfinite(R0_basic) else "∞")
c2.metric("β (per day)", f"{beta:.2f}")
c3.metric("γ (per day)", f"{gamma:.2f}")
c4.metric("1/γ (days)", f"{infectious_period_days:.1f} d" if np.isfinite(infectious_period_days) else "∞")
c5.metric("Peak infected", f"{peak_I:.0f}")


# --- Plot ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=t_eval, y=S, mode="lines", name="Susceptible (S)"))
fig.add_trace(go.Scatter(x=t_eval, y=I, mode="lines", name="Infected (I)"))
fig.add_trace(go.Scatter(x=t_eval, y=R, mode="lines", name="Recovered (R)"))

fig.update_layout(
    xaxis_title="Days",
    yaxis_title="People",
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# Optional: show table
with st.expander("Show data table"):
    st.dataframe({"day": t_eval, "S": S, "I": I, "R": R})

st.caption("Tip: If R₀ > 1, outbreaks tend to grow initially; if R₀ < 1, they tend to fade.")
st.markdown("## Virtual Lab: Explore an SIR Outbreak in a Local Community")

st.markdown("""
Use the controls on the left to explore how an outbreak changes in a community.
Record your observations and the parameter values you used.
""")

with st.expander("Lab Questions (click to open)", expanded=True):

    st.markdown("### 1) Threshold behavior (R₀ near 1)")
    st.checkbox("I adjusted β and γ so R₀ was just below 1.", key="q1a")
    st.checkbox("I adjusted β and γ so R₀ was just above 1.", key="q1b")
    st.text_area("What changed in the infected curve I(t) when crossing R₀ = 1?",
                 key="ans1", height=80)

    st.markdown("### 2) Same R₀, different outbreak shape")
    st.markdown("Keep R₀ about the same (e.g., near 2), but change β and γ together (increase both or decrease both).")
    st.text_area("How did the timing and sharpness of the outbreak change, even when R₀ stayed similar?",
                 key="ans2", height=80)

    st.markdown("### 3) Contact reduction in the community (lower β)")
    st.markdown("Lower β by ~20–40% to represent fewer effective infectious contacts (e.g., distancing, masks, school closure).")
    st.text_area("What happened to (a) peak infected, (b) time to peak, and (c) total infected by the end?",
                 key="ans3", height=90)

    st.markdown("### 4) Faster isolation / shorter infectious period (higher γ)")
    st.markdown("Increase γ to represent earlier diagnosis + isolation (shorter infectious period).")
    st.text_area("Compare increasing γ vs decreasing β. Which seemed more effective in your runs, and why?",
                 key="ans4", height=90)

    st.markdown("### 5) Community planning target")
    st.markdown("Try to keep the peak infected below **5% of the population** at any one time.")
    st.text_input("Record one solution (β, γ, R₀) that meets the goal:", key="ans5")

    st.markdown("### 6) Final size (attack rate)")
    st.text_area("How many people were infected by the end (final recovered R)? How did lowering β change the final size?",
                 key="ans6", height=90)
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
