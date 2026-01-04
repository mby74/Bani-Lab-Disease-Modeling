import numpy as np
import streamlit as st
import plotly.graph_objects as go
import io
from datetime import datetime

st.set_page_config(page_title="Hospital Stochastic SIR (Binomial)", layout="centered")

st.title("Hospital Stochastic SIR — Binomial (Discrete-Time) Simulator")
st.write(
    "This is a **stochastic SIR** model for a hospital-like setting. "
    "We interpret **β** as the effective infectious contact rate in the facility. "
    "**Visitors increase β**; screening/PPE/cohorting reduce it; faster isolation increases **γ**.\n\n"
    "This is a **closed-population approximation** for learning (no admissions/discharges)."
)

# -----------------------
# Sidebar inputs
# -----------------------
st.sidebar.header("Hospital setting")
N = st.sidebar.number_input("People present in hospital environment (N)", min_value=10, value=200, step=10)

S0 = st.sidebar.number_input("Initial susceptible (S0)", min_value=0, value=int(N - 2), step=1)
I0 = st.sidebar.number_input("Initial infectious (I0)", min_value=0, value=2, step=1)
R0_init = st.sidebar.number_input("Initial recovered/immune (R0)", min_value=0, value=0, step=1)

days = st.sidebar.slider("Simulation duration (days)", min_value=1, max_value=120, value=60, step=1)

st.sidebar.header("Disease dynamics")
gamma = st.sidebar.slider(
    "Recovery/Isolation rate γ (per day)",
    min_value=0.0, max_value=2.0, value=0.2, step=0.01,
    help="Higher γ = shorter infectious period due to faster isolation/recovery."
)

st.sidebar.header("Transmission (β) + visitor policy")
beta_base = st.sidebar.slider(
    "Baseline β (no visitors) — effective infectious contacts/day",
    min_value=0.0, max_value=3.0, value=0.5, step=0.01,
    help="Represents within-hospital mixing from staff/patients when visitors are absent or minimal."
)

visitor_level_before = st.sidebar.slider(
    "Visitor level before policy (0 = none, 1 = normal)",
    min_value=0.0, max_value=1.5, value=1.0, step=0.05,
    help="Higher values represent more visitor traffic than typical."
)

visitor_level_after = st.sidebar.slider(
    "Visitor level after policy (0 = none, 1 = normal)",
    min_value=0.0, max_value=1.5, value=0.3, step=0.05
)

visitor_beta_multiplier = st.sidebar.slider(
    "How strongly visitors increase β",
    min_value=0.0, max_value=2.0, value=0.6, step=0.05,
    help="If 0.6 and visitor level is 1.0, β increases by 60% over baseline."
)

policy_day = st.sidebar.slider(
    "Day visitor policy changes",
    min_value=0, max_value=int(days), value=10, step=1
)

st.sidebar.header("Stochastic simulation")
dt = st.sidebar.selectbox("Time step (days)", [1.0, 0.5, 0.25], index=0)
runs = st.sidebar.slider("Number of stochastic runs", min_value=10, max_value=300, value=100, step=10)

seed = st.sidebar.number_input("Random seed (optional)", value=0, step=1)
use_seed = st.sidebar.checkbox("Use seed for reproducibility", value=True)

if S0 + I0 + R0_init != N:
    st.error(f"S0 + I0 + R0 must equal N. Currently: {S0 + I0 + R0_init} (N={N}).")
    st.stop()

if gamma < 0:
    st.error("γ must be non-negative.")
    st.stop()

# -----------------------
# Model helpers
# -----------------------
def beta_t(t):
    # Visitor-driven β change at policy day
    v = visitor_level_after if t >= policy_day else visitor_level_before
    return beta_base * (1.0 + visitor_beta_multiplier * v)

def simulate_one_path(rng):
    steps = int(np.round(days / dt))
    t = np.linspace(0, steps * dt, steps + 1)

    S = np.zeros(steps + 1, dtype=int)
    I = np.zeros(steps + 1, dtype=int)
    R = np.zeros(steps + 1, dtype=int)

    S[0], I[0], R[0] = int(S0), int(I0), int(R0_init)

    for k in range(steps):
        if I[k] == 0:
            S[k+1] = S[k]
            I[k+1] = I[k]
            R[k+1] = R[k]
            continue

        b = beta_t(t[k])
        # Discrete-time hazard approximations:
        # infection probability for a susceptible over dt: 1 - exp(-β * I/N * dt)
        p_inf = 1.0 - np.exp(-b * (I[k] / N) * dt)
        p_inf = np.clip(p_inf, 0.0, 1.0)

        # recovery probability over dt: 1 - exp(-γ dt)
        p_rec = 1.0 - np.exp(-gamma * dt)
        p_rec = np.clip(p_rec, 0.0, 1.0)

        new_inf = rng.binomial(S[k], p_inf)
        new_rec = rng.binomial(I[k], p_rec)

        # State update
        S[k+1] = S[k] - new_inf
        I[k+1] = I[k] + new_inf - new_rec
        R[k+1] = R[k] + new_rec

    return t, S, I, R

# -----------------------
# Run simulations
# -----------------------
rng = np.random.default_rng(seed if use_seed else None)

t, _, _, _ = simulate_one_path(rng)  # just to get time grid
steps = len(t)

S_all = np.zeros((runs, steps), dtype=float)
I_all = np.zeros((runs, steps), dtype=float)
R_all = np.zeros((runs, steps), dtype=float)

# reset RNG for consistent runs if seed enabled
rng = np.random.default_rng(seed if use_seed else None)
for r in range(runs):
    tr, S, I, R = simulate_one_path(rng)
    S_all[r, :] = S
    I_all[r, :] = I
    R_all[r, :] = R

# summary bands
def qband(arr):
    med = np.quantile(arr, 0.50, axis=0)
    lo = np.quantile(arr, 0.10, axis=0)
    hi = np.quantile(arr, 0.90, axis=0)
    return med, lo, hi

I_med, I_lo, I_hi = qband(I_all)
S_med, S_lo, S_hi = qband(S_all)
R_med, R_lo, R_hi = qband(R_all)

peak_I_med = float(np.quantile(I_all.max(axis=1), 0.50))
peak_I_90 = float(np.quantile(I_all.max(axis=1), 0.90))
final_infected_med = float(np.quantile(R_all[:, -1] - R0_init, 0.50))

# effective R0 before/after (using β/γ interpretation)
beta_before = beta_t(0)
beta_after = beta_t(days if policy_day == 0 else policy_day + 1e-6)
R0_before = beta_before / gamma if gamma > 0 else np.inf
R0_after = beta_after / gamma if gamma > 0 else np.inf
infectious_period = 1.0 / gamma if gamma > 0 else np.inf

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("R₀ before policy", f"{R0_before:.2f}" if np.isfinite(R0_before) else "∞")
c2.metric("R₀ after policy", f"{R0_after:.2f}" if np.isfinite(R0_after) else "∞")
c3.metric("Avg infectious period (1/γ)", f"{infectious_period:.1f} d" if np.isfinite(infectious_period) else "∞")
c4.metric("Median peak infectious", f"{peak_I_med:.0f}")
c5.metric("90th %ile peak infectious", f"{peak_I_90:.0f}")

# -----------------------
# Plot (median + 10–90% band for I)
# -----------------------
fig = go.Figure()

# Infected band
fig.add_trace(go.Scatter(x=t, y=I_hi, mode="lines", name="I (90th %ile)", line=dict(width=1), showlegend=True))
fig.add_trace(go.Scatter(x=t, y=I_lo, mode="lines", name="I (10th %ile)", line=dict(width=1), fill="tonexty", showlegend=True))
fig.add_trace(go.Scatter(x=t, y=I_med, mode="lines", name="I (median)", line=dict(width=3), showlegend=True))

# Policy day marker
fig.add_vline(x=policy_day, line_dash="dash", annotation_text="Visitor policy change", annotation_position="top right")

fig.update_layout(
    title="Infectious over time (stochastic uncertainty band)",
    xaxis_title="Days",
    yaxis_title="People infectious (I)",
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("Show S, I, R median curves"):
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=t, y=S_med, mode="lines", name="S (median)"))
    fig2.add_trace(go.Scatter(x=t, y=I_med, mode="lines", name="I (median)"))
    fig2.add_trace(go.Scatter(x=t, y=R_med, mode="lines", name="R (median)"))
    fig2.add_vline(x=policy_day, line_dash="dash")
    fig2.update_layout(xaxis_title="Days", yaxis_title="People", hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

# -----------------------
# Decision-making lab questions + downloadable answers
# -----------------------
st.markdown("## Decision Lab (Hospital visitors & patients)")

st.markdown(
    "Answer as if you are advising hospital leadership. Focus on **risk**, **uncertainty**, and **tradeoffs**. "
    "Use the stochastic band/percentiles to justify decisions."
)

with st.expander("Lab Questions (click to open)", expanded=True):
    st.text_input("Student name / ID (for your downloaded file):", key="student_id")

    st.markdown("### 1) Visitor policy decision under uncertainty")
    st.markdown(
        "Choose a visitor policy that starts on a given day (policy_day) by adjusting visitor level after policy. "
        "Target: keep the **90th percentile peak infectious** below a threshold you define (e.g., 10–20 infectious at once)."
    )
    st.text_area(
        "State your threshold and the policy you chose (visitor level after, policy day). "
        "Explain why you used the 90th percentile instead of the median.",
        key="ans1", height=110
    )

    st.markdown("### 2) Screening/PPE versus visitor restriction")
    st.markdown(
        "Compare two strategies:\n"
        "- **A)** stricter visitor restriction (lower visitor level after)\n"
        "- **B)** improve screening/PPE/cohorting (lower baseline β)\n\n"
        "Which gives a better reduction in **upper-tail risk** (90th percentile peak), and what are the operational tradeoffs?"
    )
    st.text_area("Your comparison and decision:", key="ans2", height=120)

    st.markdown("### 3) Faster isolation (γ) as a decision lever")
    st.markdown(
        "Assume you can increase γ by faster testing + isolation. "
        "How much γ improvement would you prioritize versus lowering β? "
        "Use results (median and 90th percentile outcomes) to justify."
    )
    st.text_area("Your recommendation:", key="ans3", height=120)

    st.markdown("### 4) Timing decision: act early or late?")
    st.markdown(
        "Keep the *same final visitor restriction* but move the policy day earlier vs later. "
        "What is the impact on peak risk and total infections? "
        "From a decision-making perspective, when is delaying the policy unacceptable?"
    )
    st.text_area("Your answer:", key="ans4", height=120)

    st.markdown("### 5) Make the model more realistic (reflection)")
    st.markdown(
        "List **at least 3** changes you would make to better represent a real hospital outbreak "
        "(think: admissions/discharges, staff, ward structure, asymptomatic infection, importation, testing delays, contact networks)."
    )
    st.text_area("Your proposed model improvements:", key="ans5", height=120)

st.markdown("### Download your lab responses")

# Build downloadable CSV
output = io.StringIO()
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
student_id = st.session_state.get("student_id", "").strip()

output.write(f"timestamp,{timestamp}\n")
output.write(f"student_id,{student_id}\n")

output.write("\n[parameters]\n")
param_items = {
    "N": N, "S0": S0, "I0": I0, "R0_init": R0_init, "days": days,
    "gamma": gamma, "beta_base": beta_base,
    "visitor_level_before": visitor_level_before, "visitor_level_after": visitor_level_after,
    "visitor_beta_multiplier": visitor_beta_multiplier, "policy_day": policy_day,
    "dt": dt, "runs": runs,
    "R0_before": R0_before, "R0_after": R0_after,
    "median_peak_I": peak_I_med, "p90_peak_I": peak_I_90,
    "median_total_new_infected": final_infected_med
}
for k, v in param_items.items():
    output.write(f"{k},{v}\n")

output.write("\n[answers]\n")
for k in ["ans1", "ans2", "ans3", "ans4", "ans5"]:
    val = str(st.session_state.get(k, "")).replace("\n", " ").replace("\r", " ")
    output.write(f"{k},{val}\n")

fname = f"sir_hospital_tau_responses_{student_id}.csv" if student_id else "sir_hospital_tau_responses.csv"
st.download_button(
    label="⬇️ Download my lab responses (CSV)",
    data=output.getvalue().encode("utf-8"),
    file_name=fname,
    mime="text/csv"
)

st.caption("Learning note: stochastic models show that policy decisions should consider uncertainty (not just average outcomes).")
