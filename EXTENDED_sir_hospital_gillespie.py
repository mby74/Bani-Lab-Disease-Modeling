import numpy as np
import streamlit as st
import plotly.graph_objects as go
import io
from datetime import datetime

st.set_page_config(page_title="Hospital Stochastic SIR (Gillespie + Importation)", layout="centered")

st.title("Hospital Stochastic SIR — Gillespie (Event-Driven) + Importation (κ)")
st.write(
    "This simulator uses the **Gillespie algorithm** (continuous-time Markov chain), a gold-standard stochastic method.\n\n"
    "- **β**: effective infectious contacts/day (visitors can increase this)\n"
    "- **γ**: recovery/isolation rate/day (faster testing/isolation increases this)\n"
    "- **κ (kappa)**: background importation (admissions/staff introductions), allowing cases to appear even if I=0\n\n"
    "**Learning note:** simplified closed-population learning model (no explicit admissions/discharges compartments)."
)

# -----------------------
# Sidebar inputs
# -----------------------
st.sidebar.header("Hospital setting")
N = st.sidebar.number_input("People present in hospital environment (N)", min_value=10, value=200, step=10)

S0 = st.sidebar.number_input("Initial susceptible (S0)", min_value=0, value=int(N - 2), step=1)
I0 = st.sidebar.number_input("Initial infectious (I0)", min_value=0, value=2, step=1)
R0_init = st.sidebar.number_input("Initial recovered/immune (R0)", min_value=0, value=0, step=1)

T_max = st.sidebar.slider("Simulation duration (days)", min_value=1, max_value=120, value=60, step=1)

st.sidebar.header("Disease dynamics")
gamma = st.sidebar.slider("Recovery/Isolation rate γ (per day)", min_value=0.0, max_value=2.0, value=0.20, step=0.01)

st.sidebar.header("Transmission (β) + visitor policy")
beta_base = st.sidebar.slider("Baseline β (no visitors)", min_value=0.0, max_value=3.0, value=0.50, step=0.01)

visitor_level_before = st.sidebar.slider("Visitor level before policy (0–1.5)", min_value=0.0, max_value=1.5, value=1.0, step=0.05)
visitor_level_after = st.sidebar.slider("Visitor level after policy (0–1.5)", min_value=0.0, max_value=1.5, value=0.3, step=0.05)

visitor_beta_multiplier = st.sidebar.slider("How strongly visitors increase β", min_value=0.0, max_value=2.0, value=0.60, step=0.05)
policy_day = st.sidebar.slider("Day visitor policy changes", min_value=0, max_value=int(T_max), value=10, step=1)

st.sidebar.header("Background importation (admissions/staff)")
kappa = st.sidebar.slider(
    "Importation pressure κ (expected new introductions/day when S≈N)",
    min_value=0.0, max_value=5.0, value=0.20, step=0.05,
    help="Models unavoidable introductions (admissions/staff). Even if I=0, new cases can appear."
)

st.sidebar.header("Stochastic simulation")
runs = st.sidebar.slider("Number of runs", min_value=10, max_value=300, value=100, step=10)
time_grid_points = st.sidebar.selectbox("Plot resolution (time points)", [301, 601, 1201], index=1)

seed = st.sidebar.number_input("Random seed (optional)", value=0, step=1)
use_seed = st.sidebar.checkbox("Use seed for reproducibility", value=True)

if S0 + I0 + R0_init != N:
    st.error(f"S0 + I0 + R0 must equal N. Currently: {S0 + I0 + R0_init} (N={N}).")
    st.stop()

# -----------------------
# Helpers
# -----------------------
def beta_t(t):
    v = visitor_level_after if t >= policy_day else visitor_level_before
    return beta_base * (1.0 + visitor_beta_multiplier * v)

def gillespie_path(rng, T, t_grid):
    S = int(S0); I = int(I0); R = int(R0_init)
    t = 0.0

    Sg = np.zeros_like(t_grid, dtype=float)
    Ig = np.zeros_like(t_grid, dtype=float)
    Rg = np.zeros_like(t_grid, dtype=float)

    gi = 0
    while gi < len(t_grid) and t_grid[gi] <= 0:
        Sg[gi], Ig[gi], Rg[gi] = S, I, R
        gi += 1

    # Continue even if I == 0 (because κ can re-introduce infections)
    while t < T and (S > 0 or I > 0):
        b = beta_t(t)

        rate_inf = b * S * I / N           # transmission: S -> I
        rate_rec = gamma * I               # recovery/isolation: I -> R
        rate_imp = (kappa / N) * S         # importation: S -> I

        rate_total = rate_inf + rate_rec + rate_imp
        if rate_total <= 0:
            break

        dt_event = rng.exponential(1.0 / rate_total)
        t_next = t + dt_event

        while gi < len(t_grid) and t_grid[gi] <= min(t_next, T):
            Sg[gi], Ig[gi], Rg[gi] = S, I, R
            gi += 1

        if t_next > T:
            t = T
            break

        u = rng.random()
        p_inf = rate_inf / rate_total
        p_rec = rate_rec / rate_total

        if u < p_inf:
            if S > 0:
                S -= 1
                I += 1
        elif u < p_inf + p_rec:
            if I > 0:
                I -= 1
                R += 1
        else:
            if S > 0:
                S -= 1
                I += 1

        t = t_next

    while gi < len(t_grid):
        Sg[gi], Ig[gi], Rg[gi] = S, I, R
        gi += 1

    return Sg, Ig, Rg

# -----------------------
# Run simulations
# -----------------------
t_grid = np.linspace(0, T_max, int(time_grid_points))

rng = np.random.default_rng(seed if use_seed else None)
S_all = np.zeros((runs, len(t_grid)))
I_all = np.zeros((runs, len(t_grid)))
R_all = np.zeros((runs, len(t_grid)))

rng = np.random.default_rng(seed if use_seed else None)
for r in range(runs):
    Sg, Ig, Rg = gillespie_path(rng, T_max, t_grid)
    S_all[r, :] = Sg
    I_all[r, :] = Ig
    R_all[r, :] = Rg

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
beta_after = beta_t(float(policy_day) + 1e-6) if policy_day < T_max else beta_t(float(T_max))
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
# Plot
# -----------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=t_grid, y=I_hi, mode="lines", name="I (90th %ile)", line=dict(width=1)))
fig.add_trace(go.Scatter(x=t_grid, y=I_lo, mode="lines", name="I (10th %ile)", line=dict(width=1), fill="tonexty"))
fig.add_trace(go.Scatter(x=t_grid, y=I_med, mode="lines", name="I (median)", line=dict(width=3)))
fig.add_vline(x=policy_day, line_dash="dash", annotation_text="Visitor policy change", annotation_position="top right")
fig.update_layout(
    title="Infectious over time (Gillespie uncertainty band)",
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
    "Treat this as a leadership decision exercise. Use uncertainty (e.g., 90th percentile peak) to justify choices."
)

with st.expander("Lab Questions (click to open)", expanded=True):
    st.text_input("Student name / ID (for your downloaded file):", key="student_id")

    st.markdown("### 1) Visitor policy decision under uncertainty")
    st.markdown(
        "Define a risk tolerance target (e.g., keep **90th percentile peak I** below a threshold you choose). "
        "Adjust visitor level after policy and policy day to meet it."
    )
    st.text_area(
        "State your threshold and policy choice. Why is using tail risk (90th percentile) appropriate in a hospital setting?",
        key="ans1", height=120
    )

    st.markdown("### 2) Screening/PPE vs visitor restriction")
    st.markdown(
        "Compare:\n"
        "- **A)** stricter visitor restriction (lower visitor level after)\n"
        "- **B)** improved screening/PPE/cohorting (lower baseline β)\n\n"
        "Which reduces **tail risk** more, and what are the operational tradeoffs?"
    )
    st.text_area("Your decision and reasoning:", key="ans2", height=120)

    st.markdown("### 3) Faster testing/isolation (γ)")
    st.markdown(
        "Assume you can increase γ by faster testing + isolation. "
        "How does improving γ compare with lowering β in terms of reducing peak risk and outbreak duration?"
    )
    st.text_area("Your recommendation:", key="ans3", height=120)

    st.markdown("### 4) The κ question: unavoidable introductions")
    st.markdown(
        "With **κ > 0**, outbreaks may occur even with strict visitor restrictions.\n\n"
        "Name specific interventions aimed at **reducing κ** (e.g., admission screening, staff testing, vaccination, "
        "sick leave policies, ventilation). Which would you prioritize first and why?"
    )
    st.text_area("Your κ-focused mitigation plan:", key="ans4", height=140)

    st.markdown("### 5) Make the model more realistic (reflection)")
    st.markdown(
        "List at least **3** changes to better represent a real hospital outbreak (e.g., admissions/discharges flow, staff compartment, "
        "ward structure, asymptomatic infection, testing delays, heterogeneous contact networks)."
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
    "N": N, "S0": S0, "I0": I0, "R0_init": R0_init, "T_max_days": T_max,
    "gamma": gamma, "infectious_period_days_1_over_gamma": infectious_period,
    "beta_base": beta_base,
    "visitor_level_before": visitor_level_before, "visitor_level_after": visitor_level_after,
    "visitor_beta_multiplier": visitor_beta_multiplier,
    "policy_day": policy_day,
    "kappa_imports_per_day": kappa,
    "runs": runs, "time_grid_points": time_grid_points,
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

fname = f"sir_hospital_gillespie_responses_{student_id}.csv" if student_id else "sir_hospital_gillespie_responses.csv"
st.download_button(
    label="⬇️ Download my lab responses (CSV)",
    data=output.getvalue().encode("utf-8"),
    file_name=fname,
    mime="text/csv"
)

st.caption("Interpretation tip: κ captures baseline introductions you may not fully control; policies targeting β/γ and κ act on different parts of the risk.")

