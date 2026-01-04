import numpy as np
import streamlit as st
import plotly.graph_objects as go
import io
from datetime import datetime

st.set_page_config(page_title="Hospital Stochastic SIR (Binomial + Importation)", layout="centered")

st.title("Hospital Stochastic SIR — Binomial (Discrete-Time) + Importation (κ)")
st.write(
    "This is a **stochastic SIR** model for a hospital-like setting.\n\n"
    "- **β** represents effective infectious contacts/day inside the facility.\n"
    "- **Visitors increase β**; screening/PPE/cohorting reduce it.\n"
    "- **γ** represents recovery/isolation rate (higher γ = shorter infectious period).\n"
    "- **κ (kappa)** represents **background importation** (admissions/staff introductions) that can create cases even if I=0.\n\n"
    "**Learning note:** This is a simplified closed-population approximation (no explicit admissions/discharges compartments)."
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
    min_value=0.0, max_value=2.0, value=0.20, step=0.01,
    help="Higher γ means shorter infectious period (faster isolation/recovery)."
)

st.sidebar.header("Transmission (β) + visitor policy")
beta_base = st.sidebar.slider(
    "Baseline β (no visitors) — effective infectious contacts/day",
    min_value=0.0, max_value=3.0, value=0.50, step=0.01
)

visitor_level_before = st.sidebar.slider(
    "Visitor level before policy (0 = none, 1 = normal)",
    min_value=0.0, max_value=1.5, value=1.0, step=0.05
)

visitor_level_after = st.sidebar.slider(
    "Visitor level after policy (0 = none, 1 = normal)",
    min_value=0.0, max_value=1.5, value=0.3, step=0.05
)

visitor_beta_multiplier = st.sidebar.slider(
    "How strongly visitors increase β",
    min_value=0.0, max_value=2.0, value=0.60, step=0.05,
    help="If 0.6 and visitor level is 1.0, β increases by 60% over baseline."
)

policy_day = st.sidebar.slider(
    "Day visitor policy changes",
    min_value=0, max_value=int(days), value=10, step=1
)

st.sidebar.header("Background importation (admissions/staff)")
kappa = st.sidebar.slider(
    "Importation pressure κ (expected new introductions/day when S≈N)",
    min_value=0.0, max_value=5.0, value=0.20, step=0.05,
    help="Models unavoidable introductions (admissions/staff). Even if I=0, new cases can appear."
)

st.sidebar.header("Stochastic simulation")
dt = st.sidebar.selectbox("Time step (days)", [1.0, 0.5, 0.25], index=0)
runs = st.sidebar.slider("Number of stochastic runs", min_value=10, max_value=300, value=100, step=10)

seed = st.sidebar.number_input("Random seed (optional)", value=0, step=1)
use_seed = st.sidebar.checkbox("Use seed for reproducibility", value=True)

# Guardrails
if S0 + I0 + R0_init != N:
    st.error(f"S0 + I0 + R0 must equal N. Currently: {S0 + I0 + R0_init} (N={N}).")
    st.stop()

# -----------------------
# Model helpers
# -----------------------
def beta_t(t):
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
        b = beta_t(t[k])

        # Infection probability from within-hospital transmission over dt
        p_inf = 1.0 - np.exp(-b * (I[k] / N) * dt)
        p_inf = np.clip(p_inf, 0.0, 1.0)

        # Recovery probability over dt
        p_rec = 1.0 - np.exp(-gamma * dt)
        p_rec = np.clip(p_rec, 0.0, 1.0)

        # Importation probability per susceptible over dt: rate_imp = (kappa/N) * S
        # Per susceptible hazard ≈ (kappa/N) over dt
        p_imp = 1.0 - np.exp(-(kappa / N) * dt)
        p_imp = np.clip(p_imp, 0.0, 1.0)

        new_inf = rng.binomial(S[k], p_inf) if S[k] > 0 else 0
        new_imp = rng.binomial(S[k], p_imp) if S[k] > 0 else 0
        new_rec = rng.binomial(I[k], p_rec) if I[k] > 0 else 0

        new_total_in = new_inf + new_imp
        new_total_in = min(new_total_in, S[k])

        S[k + 1] = S[k] - new_total_in
        I[k + 1] = I[k] + new_total_in - new_rec
        R[k + 1] = R[k] + new_rec

    return t, S, I, R

# -----------------------
# Run simulations
# -----------------------
rng = np.random.default_rng(seed if use_seed else None)
t, _, _, _ = simulate_one_path(rng)
steps = len(t)

S_all = np.zeros((runs, steps), dtype=float)
I_all = np.zeros((runs, steps), dtype=float)
R_all = np.zeros((runs, steps), dtype=float)

rng = np.random.default_rng(seed if use_seed else None)
for r in range(runs):
    tr, S, I, R = simulate_one_path(rng)
    S_all[r, :] = S
    I_all[r, :] = I
    R_all[r, :] = R

def qband(arr):
    med = np.quantile(arr, 0.50, axis=0)
    lo = np.quantile(arr, 0.10, axis=0)
    hi = np.quantile(arr, 0.90, axis=0)
    return med, lo, hi

I_med, I_lo, I_hi = qband(I_all)

peak_I_med = float(np.quantile(I_all.max(axis=1), 0.50))
peak_I_90 = float(np.quantile(I_all.max(axis=1), 0.90))
final_new_infected_med = float(np.quantile(R_all[:, -1] - R0_init, 0.50))

beta_before = beta_t(0.0)
beta_after = beta_t(float(policy_day) + 1e-6) if policy_day < days else beta_t(float(days))
R0_before = beta_before / gamma if gamma > 0 else np.inf
R0_after = beta_after / gamma if gamma > 0 else np.inf
infectious_period = 1.0 / gamma if gamma > 0 else np.inf

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("R₀ before policy", f"{R0_before:.2f}" if np.isfinite(R0_before) else "∞")
c2.metric("R₀ after policy", f"{R0_after:.2f}" if np.isfinite(R0_after) else "∞")
c3.metric("κ (imports/day)", f"{kappa:.2f}")
c4.metric("Median peak I", f"{peak_I_med:.0f}")
c5.metric("90th %ile peak I", f"{peak_I_90:.0f}")
c6.metric("Median total new infected", f"{final_new_infected_med:.0f}")

# -----------------------
# Plot: I uncertainty band
# -----------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=I_hi, mode="lines", name="I (90th %ile)", line=dict(width=1)))
fig.add_trace(go.Scatter(x=t, y=I_lo, mode="lines", name="I (10th %ile)", line=dict(width=1), fill="tonexty"))
fig.add_trace(go.Scatter(x=t, y=I_med, mode="lines", name="I (median)", line=dict(width=3)))
fig.add_vline(x=policy_day, line_dash="dash", annotation_text="Visitor policy change", annotation_position="top right")
fig.update_layout(
    title="Infectious over time (stochastic uncertainty band)",
    xaxis_title="Days",
    yaxis_title="People infectious (I)",
    hovermode="x unified",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Decision lab questions
# -----------------------
st.markdown("## Decision Lab (Visitors vs Unavoidable Importation)")
st.markdown(
    "Answer as if advising hospital leadership. Use **uncertainty** (e.g., 90th percentile peak) to justify decisions."
)

with st.expander("Lab Questions (click to open)", expanded=True):
    st.text_input("Student name / ID (for your downloaded file):", key="student_id")

    st.markdown("### 1) Visitor policy decision under uncertainty")
    st.markdown(
        "Choose a risk tolerance target (you define it), such as keeping the **90th percentile peak infectious** below a threshold. "
        "Adjust visitor level after policy and policy day to meet the target."
    )
    st.text_area(
        "State your threshold, your chosen policy, and why you used a tail-risk metric (e.g., 90th percentile) rather than the median.",
        key="ans1", height=120
    )

    st.markdown("### 2) Screening/PPE vs visitor restriction")
    st.markdown(
        "Compare two strategies:\n"
        "- **A)** stricter visitor restriction (lower visitor level after)\n"
        "- **B)** improve screening/PPE/cohorting (lower baseline β)\n\n"
        "Which better reduces **tail risk** (90th percentile peak), and what operational tradeoffs matter?"
    )
    st.text_area("Your comparison and decision:", key="ans2", height=120)

    st.markdown("### 3) Faster testing/isolation (γ) as a decision lever")
    st.markdown(
        "Assume you can improve testing turnaround and isolation practices (increase γ). "
        "How much would you prioritize improving γ vs reducing β? Justify with outcomes and feasibility."
    )
    st.text_area("Your recommendation:", key="ans3", height=120)

    st.markdown("### 4) The κ question: visitor policy cannot eliminate all risk")
    st.markdown(
        "With **κ > 0**, new infections can appear even if I hits 0 and visitors are restricted.\n\n"
        "What operational strategies specifically reduce κ (e.g., admission screening, staff testing, "
        "vaccination/boosters, sick leave policies, ventilation), and which would you prioritize first?"
    )
    st.text_area("Your κ-focused mitigation plan:", key="ans4", height=140)

    st.markdown("### 5) Make the model more realistic (reflection)")
    st.markdown(
        "List **at least 3** changes to better represent a real hospital outbreak (e.g., admissions/discharges flow, staff compartment, "
        "ward structure, asymptomatic infection, testing delay, contact networks, heterogeneous risk)."
    )
    st.text_area("Your proposed model improvements:", key="ans5", height=140)

# -----------------------
# Download responses
# -----------------------
st.markdown("### Download your lab responses")

output = io.StringIO()
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
student_id = st.session_state.get("student_id", "").strip()

output.write(f"timestamp,{timestamp}\n")
output.write(f"student_id,{student_id}\n")

output.write("\n[parameters]\n")
param_items = {
    "N": N, "S0": S0, "I0": I0, "R0_init": R0_init, "days": days,
    "gamma": gamma, "infectious_period_days_1_over_gamma": infectious_period,
    "beta_base": beta_base,
    "visitor_level_before": visitor_level_before, "visitor_level_after": visitor_level_after,
    "visitor_beta_multiplier": visitor_beta_multiplier,
    "policy_day": policy_day,
    "kappa_imports_per_day": kappa,
    "dt_days": dt, "runs": runs,
    "beta_before": beta_before, "beta_after": beta_after,
    "R0_before": R0_before, "R0_after": R0_after,
    "median_peak_I": peak_I_med, "p90_peak_I": peak_I_90,
    "median_total_new_infected": final_new_infected_med
}
for k, v in param_items.items():
    output.write(f"{k},{v}\n")

output.write("\n[answers]\n")
for k in ["ans1", "ans2", "ans3", "ans4", "ans5"]:
    val = str(st.session_state.get(k, "")).replace("\n", " ").replace("\r", " ")
    output.write(f"{k},{val}\n")

fname = f"sir_hospital_stochastic_tau_responses_{student_id}.csv" if student_id else "sir_hospital_stochastic_tau_responses.csv"
st.download_button(
    label="⬇️ Download my lab responses (CSV)",
    data=output.getvalue().encode("utf-8"),
    file_name=fname,
    mime="text/csv"
)

st.caption("Interpretation tip: κ represents baseline introductions you may not fully control; β/γ represent transmission/isolation you can influence operationally.")
