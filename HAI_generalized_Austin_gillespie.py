import numpy as np
import streamlit as st
import plotly.graph_objects as go
import io
from datetime import datetime

st.set_page_config(page_title="Generalized HAI Model (Austin core + infection module)", layout="centered")
st.title("Generalized AMR HAI Simulator — Austin-style colonization + infection module (Gillespie)")

st.markdown(
    "This app keeps Austin’s **patient–HCW–patient colonization engine** (small ICU populations → stochasticity matters) "
    "and adds a syndrome module for **infection given colonization/device exposure**.\n\n"
    "Use this for: **VAP/NV-HAP**, **CAUTI**, **CLABSI**, **MRSA bloodstream**, etc. (excluding SSI).\n\n"
    "Core mechanism is from Austin et al. (1999): patients colonized; HCWs transiently contaminated. :contentReference[oaicite:17]{index=17}"
)

# --------------------------
# Syndrome presets (teaching-friendly, not “validated”)
# --------------------------
SYNDROMES = {
    "VAP (vent-associated pneumonia)": {"device_name": "Ventilator", "u_default": 0.30, "haz_default": 0.030},
    "NV-HAP (non-vent hospital-acquired pneumonia)": {"device_name": "High-risk exposure", "u_default": 0.60, "haz_default": 0.010},
    "CAUTI (catheter-associated UTI)": {"device_name": "Urinary catheter", "u_default": 0.45, "haz_default": 0.020},
    "CLABSI (central line-associated BSI)": {"device_name": "Central line", "u_default": 0.35, "haz_default": 0.015},
    "MRSA bloodstream (proxy)": {"device_name": "Invasive exposure", "u_default": 0.25, "haz_default": 0.012},
    "Generic AMR HAI": {"device_name": "Risk exposure", "u_default": 0.40, "haz_default": 0.015},
}

# --------------------------
# Sidebar
# --------------------------
st.sidebar.header("Syndrome")
syndrome = st.sidebar.selectbox("Choose syndrome module", list(SYNDROMES.keys()), index=0)
device_name = SYNDROMES[syndrome]["device_name"]

st.sidebar.header("Time horizon & simulation")
T_max = st.sidebar.slider("Simulation duration (days)", 14, 180, 60, 1)
runs = st.sidebar.slider("Runs", 20, 400, 150, 10)
time_grid_points = st.sidebar.selectbox("Plot resolution", [301, 601, 1201], index=1)
seed = st.sidebar.number_input("Random seed", value=0, step=1)
use_seed = st.sidebar.checkbox("Use seed", value=True)

st.sidebar.header("Austin colonization engine (core)")
Np = st.sidebar.number_input("ICU beds (Np)", min_value=4, value=16, step=1)
Nh = st.sidebar.number_input("HCWs (Nh)", min_value=2, value=10, step=1)
n_prop = st.sidebar.slider("Proportion nursing staff (n)", 0.1, 1.0, 0.794, 0.01)

L = st.sidebar.slider("Admission rate L (patients/day)", 0.1, 5.0, 1.36, 0.01)
Dp_u = st.sidebar.slider("LOS uncolonized (days)", 1.0, 30.0, 9.8, 0.1)
d_los = st.sidebar.slider("Increase in LOS if colonized (d)", 0.0, 2.0, 0.551, 0.01)
Dp_c = Dp_u * (1 + d_los)
g_u = 1 / Dp_u
g_c = 1 / Dp_c

Dh = st.sidebar.slider("HCW contamination duration Dh (days)", 1/96, 1.0, 1/24, 1/96)
m = 1 / Dh

a = st.sidebar.slider("Contact rate a (per HCW per patient per day)", 0.1, 5.0, 1.38, 0.01)
bp = st.sidebar.slider("bp colonization probability per contact", 0.0, 0.5, 0.06, 0.005)
bh = st.sidebar.slider("bh contamination probability per contact", 0.0, 1.0, 0.40, 0.01)

p = st.sidebar.slider("p hand hygiene/barrier compliance", 0.0, 1.0, 0.50, 0.01)
q = st.sidebar.slider("q cohorting probability (nursing)", 0.0, 1.0, 0.80, 0.01)
f = st.sidebar.slider("f colonized on admission", 0.0, 0.5, 0.149, 0.001)

st.sidebar.header("Antibiotic pressure (optional)")
a_abx = st.sidebar.slider("a_abx fraction on antibiotics", 0.0, 1.0, 0.50, 0.01)
j = st.sidebar.slider("j relative risk during antibiotics", 1.0, 10.0, 3.0, 0.1)

# Infection module
st.sidebar.header(f"Infection module: {syndrome}")
u = st.sidebar.slider(f"{device_name} utilization (fraction of patients exposed)", 0.0, 1.0, SYNDROMES[syndrome]["u_default"], 0.01)

haz_inf = st.sidebar.slider(
    "Daily hazard of infection given colonization & exposure (per day)",
    0.0, 0.20, SYNDROMES[syndrome]["haz_default"], 0.001,
    help="Teaching parameter: infection risk per day among colonized patients with exposure."
)

bundle = st.sidebar.slider(
    "Bundle adherence (reduces infection risk given exposure)",
    0.0, 1.0, 0.60, 0.01,
    help="0.6 means 60% adherence to prevention bundle; higher reduces infection hazard."
)

bundle_effect = st.sidebar.slider(
    "Bundle effectiveness when adhered (0–1)",
    0.0, 1.0, 0.60, 0.01,
    help="If adhered, how strongly does it reduce infection risk?"
)

# --------------------------
# Helpers
# --------------------------
def cohort_factor(q_t):
    return max(0.0, 1.0 - q_t * n_prop)

def bp_effective(bp0):
    # 1 + a(j-1) factor :contentReference[oaicite:18]{index=18}
    return bp0 * (1.0 + a_abx * (j - 1.0))

def infection_hazard_effective():
    # risk reduction: adherence * effectiveness
    return haz_inf * (1.0 - bundle * bundle_effect)

# --------------------------
# Gillespie simulation
# --------------------------
def simulate(rng, T, t_grid):
    # Patient states:
    # Xp: uncolonized
    # Yp: colonized (not infected)
    # Ip: infected (subset, for outcome tracking; does not feed back into transmission here)
    # HCW: Xh, Yh
    Xp = Np - int(round(f * Np))
    Yp = int(round(f * Np))
    Ip = 0

    Xh = Nh
    Yh = 0

    # Tracks
    Yp_path = np.zeros_like(t_grid, dtype=float)
    Ip_path = np.zeros_like(t_grid, dtype=float)
    cum_inf_path = np.zeros_like(t_grid, dtype=float)

    cum_inf = 0.0
    t = 0.0
    gi = 0
    while gi < len(t_grid) and t_grid[gi] <= 0:
        Yp_path[gi], Ip_path[gi], cum_inf_path[gi] = Yp, Ip, cum_inf
        gi += 1

    while t < T:
        cf = cohort_factor(q)
        ic_factor = (1.0 - p)
        bp_t = bp_effective(bp)

        # Core Austin-like rates
        rate_dis_u = g_u * Xp
        rate_dis_c = g_c * (Yp + Ip)  # infected/colonized have longer LOS (simplification)
        rate_clear_h = (1.0 / Dh) * Yh
        rate_contam_h = (a * bh * Xh * (Yp + Ip)) * cf * ic_factor
        rate_col_p = (a * bp_t * Xp * Yh) * cf * ic_factor

        # Infection event among colonized patients with exposure
        haz_eff = infection_hazard_effective()
        # Expected exposed colonized ~ u * Yp
        rate_inf = haz_eff * (u * Yp)

        rate_total = rate_dis_u + rate_dis_c + rate_clear_h + rate_contam_h + rate_col_p + rate_inf
        if rate_total <= 0:
            break

        dt = rng.exponential(1.0 / rate_total)
        t_next = t + dt

        while gi < len(t_grid) and t_grid[gi] <= min(t_next, T):
            Yp_path[gi], Ip_path[gi], cum_inf_path[gi] = Yp, Ip, cum_inf
            gi += 1

        if t_next > T:
            t = T
            break

        u_draw = rng.random() * rate_total

        # Discharge uncolonized -> replace admission (colonized with prob f)
        if u_draw < rate_dis_u:
            if Xp > 0:
                Xp -= 1
                if rng.random() < f:
                    Yp += 1
                else:
                    Xp += 1

        # Discharge colonized/infected -> replace admission
        elif u_draw < rate_dis_u + rate_dis_c:
            # Prefer discharging infected if present (teaching simplification)
            if Ip > 0 and rng.random() < (Ip / max(Ip + Yp, 1e-6)):
                Ip -= 1
            elif Yp > 0:
                Yp -= 1
            else:
                # fallback
                pass

            if rng.random() < f:
                Yp += 1
            else:
                Xp += 1

        # HCW decontamination
        elif u_draw < rate_dis_u + rate_dis_c + rate_clear_h:
            if Yh > 0:
                Yh -= 1
                Xh += 1

        # HCW contamination
        elif u_draw < rate_dis_u + rate_dis_c + rate_clear_h + rate_contam_h:
            if Xh > 0:
                Xh -= 1
                Yh += 1

        # Patient colonization
        elif u_draw < rate_dis_u + rate_dis_c + rate_clear_h + rate_contam_h + rate_col_p:
            if Xp > 0:
                Xp -= 1
                Yp += 1

        # Infection
        else:
            if Yp > 0:
                Yp -= 1
                Ip += 1
                cum_inf += 1

        t = t_next

    while gi < len(t_grid):
        Yp_path[gi], Ip_path[gi], cum_inf_path[gi] = Yp, Ip, cum_inf
        gi += 1

    return Yp_path, Ip_path, cum_inf_path

# --------------------------
# Ensemble
# --------------------------
t_grid = np.linspace(0, T_max, int(time_grid_points))
rng = np.random.default_rng(seed if use_seed else None)

Y_all = np.zeros((runs, len(t_grid)))
I_all = np.zeros((runs, len(t_grid)))
CumInf_all = np.zeros((runs, len(t_grid)))

rng = np.random.default_rng(seed if use_seed else None)
for r in range(runs):
    Yp_path, Ip_path, cum_inf_path = simulate(rng, T_max, t_grid)
    Y_all[r, :] = Yp_path
    I_all[r, :] = Ip_path
    CumInf_all[r, :] = cum_inf_path

def qband(arr):
    med = np.quantile(arr, 0.50, axis=0)
    lo = np.quantile(arr, 0.10, axis=0)
    hi = np.quantile(arr, 0.90, axis=0)
    return med, lo, hi

I_med, I_lo, I_hi = qband(I_all)
Cum_med, Cum_lo, Cum_hi = qband(CumInf_all)

# Device-days (expected, for a teaching metric)
expected_device_days = u * Np * T_max
rate_per_1000 = (Cum_med[-1] / max(expected_device_days, 1e-9)) * 1000.0

c1, c2, c3 = st.columns(3)
c1.metric("Median cumulative infections", f"{Cum_med[-1]:.1f}")
c2.metric("Approx infections / 1000 device-days", f"{rate_per_1000:.1f}")
c3.metric("Infection hazard after bundle", f"{infection_hazard_effective():.4f}/day")

fig = go.Figure()
fig.add_trace(go.Scatter(x=t_grid, y=I_hi, mode="lines", name="Infected (90th %ile)", line=dict(width=1)))
fig.add_trace(go.Scatter(x=t_grid, y=I_lo, mode="lines", name="Infected (10th %ile)", line=dict(width=1), fill="tonexty"))
fig.add_trace(go.Scatter(x=t_grid, y=I_med, mode="lines", name="Infected (median)", line=dict(width=3)))
fig.update_layout(title=f"{syndrome}: infected patients over time", xaxis_title="Days", yaxis_title="Infected (count)", hovermode="x unified", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=t_grid, y=Cum_hi, mode="lines", name="Cumulative infections (90th %ile)", line=dict(width=1)))
fig2.add_trace(go.Scatter(x=t_grid, y=Cum_lo, mode="lines", name="Cumulative infections (10th %ile)", line=dict(width=1), fill="tonexty"))
fig2.add_trace(go.Scatter(x=t_grid, y=Cum_med, mode="lines", name="Cumulative infections (median)", line=dict(width=3)))
fig2.update_layout(title="Cumulative infections", xaxis_title="Days", yaxis_title="Cumulative infections", hovermode="x unified", template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

# --------------------------
# Lab questions + download
# --------------------------
st.markdown("## Decision Lab (General HAIs — transmission vs devices)")
with st.expander("Lab Questions (click to open)", expanded=True):
    st.text_input("Student name / ID:", key="student_id")

    st.markdown("### 1) Best lever for this syndrome")
    st.markdown("For your chosen syndrome, which matters more for reducing infections: lowering transmission (p/q/bp/bh) or reducing device risk (bundle, utilization u)? Explain using your simulations.")
    st.text_area("Answer:", key="ans1", height=120)

    st.markdown("### 2) Tail risk vs average")
    st.markdown("Find a setting where median infections are low but the 90th percentile is concerning. What would you recommend operationally, and why?")
    st.text_area("Answer:", key="ans2", height=120)

    st.markdown("### 3) Admission pressure (f) vs in-unit spread")
    st.markdown("Increase f and observe infections. At what point does admission pressure dominate outcomes? What policy targets f specifically?")
    st.text_area("Answer:", key="ans3", height=120)

    st.markdown("### 4) Model realism upgrade")
    st.markdown("How would you change this model to better match your syndrome (e.g., explicit device-days per patient, staff types, environmental reservoir, testing delay, heterogeneity)?")
    st.text_area("Answer:", key="ans4", height=120)

st.markdown("### Download your lab responses")
out = io.StringIO()
ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
sid = st.session_state.get("student_id", "").strip()
out.write(f"timestamp,{ts}\nstudent_id,{sid}\n")
out.write("\n[parameters]\n")
params = {
    "syndrome": syndrome, "T_max": T_max, "runs": runs, "Np": Np, "Nh": Nh, "n_prop": n_prop,
    "L": L, "Dp_u": Dp_u, "d_los": d_los, "Dh": Dh, "a": a, "bp": bp, "bh": bh, "p": p, "q": q, "f": f,
    "a_abx": a_abx, "j": j,
    "utilization_u": u, "haz_inf": haz_inf, "bundle": bundle, "bundle_effect": bundle_effect,
    "haz_after_bundle": infection_hazard_effective(),
    "median_cum_infections": float(Cum_med[-1]),
    "approx_infections_per_1000_device_days": float(rate_per_1000),
}
for k, v in params.items():
    out.write(f"{k},{v}\n")

out.write("\n[answers]\n")
for k in ["ans1", "ans2", "ans3", "ans4"]:
    val = str(st.session_state.get(k, "")).replace("\n", " ").replace("\r", " ")
    out.write(f"{k},{val}\n")

fname = f"HAI_generalized_responses_{sid}.csv" if sid else "HAI_generalized_responses.csv"
st.download_button("⬇️ Download my responses (CSV)", data=out.getvalue().encode("utf-8"), file_name=fname, mime="text/csv")
