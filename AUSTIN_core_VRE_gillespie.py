import numpy as np
import streamlit as st
import plotly.graph_objects as go
import io
from datetime import datetime

st.set_page_config(page_title="Austin VRE ICU Model (Gillespie)", layout="centered")
st.title("Austin et al. (1999) VRE ICU Model — Stochastic (Gillespie)")

st.markdown(
    "This app implements the **patient–HCW–patient** colonization model described by Austin et al. "
    "Patients are either **uncolonized (Xp)** or **colonized (Yp)**; HCWs are **clean (Xh)** or **contaminated (Yh)**. "
    "Colonization is assumed irreversible for a patient's ICU stay. :contentReference[oaicite:2]{index=2}  \n"
    "Admissions include a fraction **f** already colonized. :contentReference[oaicite:3]{index=3}  \n"
    "Hand hygiene/barrier precautions (p), cohorting (q), and antibiotic pressure (a_abx, j) modify transmission. "
    "The paper emphasizes stochasticity because ICU populations are small. :contentReference[oaicite:4]{index=4}"
)

# --------------------------
# Defaults from Table 1 (Cook County ICU)
# --------------------------
TABLE1_DEFAULTS = {
    "Np": 16,
    "Nh": 10,
    "n_prop": 0.794,          # proportion nursing staff (n)
    "L": 1.36,                # admissions/day
    "Dp_u": 9.8,              # LOS uncolonized (days)
    "d_los": 0.551,           # % increase in LOS for colonized
    "Dh": 1/24,               # duration of contamination (days)
    "a": 1.38,                # per-capita contact rate (per HCW per patient per day)
    "bp": 0.06,               # colonization probability per contact
    "bh": 0.40,               # contamination probability per contact
    "p": 0.501,               # compliance hand hygiene/barrier
    "q": 0.80,                # cohort probability (nursing)
    "f": 0.149,               # admission colonization prevalence
    # Antibiotic pressure example values from paper figure caption (a'=25%, j=3) and text
    "a_abx": 0.50,            # fraction of LOS on antibiotics (a)
    "j": 3.0                  # relative risk during antibiotics
}

# --------------------------
# Sidebar: choose baseline + controls
# --------------------------
st.sidebar.header("Baseline (Table 1)")
use_table1 = st.sidebar.checkbox("Use Cook County ICU (Table 1) defaults", value=True)

st.sidebar.header("Time horizon & simulation")
T_max = st.sidebar.slider("Simulation duration (days)", 30, 200, 133, 1)
runs = st.sidebar.slider("Number of stochastic runs", 20, 400, 150, 10)
time_grid_points = st.sidebar.selectbox("Plot resolution", [301, 601, 1201], index=1)

seed = st.sidebar.number_input("Random seed", value=0, step=1)
use_seed = st.sidebar.checkbox("Use seed (reproducible)", value=True)

st.sidebar.header("Core ICU structure")
Np = st.sidebar.number_input("Beds / patients in ICU (Np)", min_value=4, value=TABLE1_DEFAULTS["Np"] if use_table1 else 16, step=1)
Nh = st.sidebar.number_input("HCWs on shift (Nh)", min_value=2, value=TABLE1_DEFAULTS["Nh"] if use_table1 else 10, step=1)
n_prop = st.sidebar.slider("Proportion nursing staff (n)", 0.1, 1.0, TABLE1_DEFAULTS["n_prop"] if use_table1 else 0.8, 0.01)

st.sidebar.header("Flow (admissions/discharges)")
L = st.sidebar.slider("Admission rate L (patients/day)", 0.1, 5.0, TABLE1_DEFAULTS["L"] if use_table1 else 1.36, 0.01)
Dp_u = st.sidebar.slider("LOS uncolonized (days)", 1.0, 30.0, TABLE1_DEFAULTS["Dp_u"] if use_table1 else 9.8, 0.1)
d_los = st.sidebar.slider("Increase in LOS if colonized (d)", 0.0, 2.0, TABLE1_DEFAULTS["d_los"] if use_table1 else 0.55, 0.01)
Dp_c = Dp_u * (1 + d_los)

# Discharge hazards
g_u = 1.0 / Dp_u
g_c = 1.0 / Dp_c

st.sidebar.header("Contact & transmission")
a = st.sidebar.slider("Per-capita contact rate a (per HCW per patient per day)", 0.1, 5.0, TABLE1_DEFAULTS["a"] if use_table1 else 1.38, 0.01)
bp = st.sidebar.slider("bp: patient colonization prob per contact", 0.0, 0.5, TABLE1_DEFAULTS["bp"] if use_table1 else 0.06, 0.005)
bh = st.sidebar.slider("bh: HCW contamination prob per contact", 0.0, 1.0, TABLE1_DEFAULTS["bh"] if use_table1 else 0.40, 0.01)
Dh = st.sidebar.slider("Dh: duration of HCW contamination (days)", 1/96, 1.0, TABLE1_DEFAULTS["Dh"] if use_table1 else 1/24, 1/96)
m = 1.0 / Dh  # decontamination rate

st.sidebar.header("Control measures (Austin paper)")
p = st.sidebar.slider("p: hand hygiene/barrier compliance", 0.0, 1.0, TABLE1_DEFAULTS["p"] if use_table1 else 0.50, 0.01)
q = st.sidebar.slider("q: cohorting probability (nursing contacts)", 0.0, 1.0, TABLE1_DEFAULTS["q"] if use_table1 else 0.80, 0.01)
f = st.sidebar.slider("f: colonized on admission", 0.0, 0.5, TABLE1_DEFAULTS["f"] if use_table1 else 0.15, 0.001)

st.sidebar.header("Antibiotic pressure")
a_abx = st.sidebar.slider("a_abx: fraction of stay on antibiotics", 0.0, 1.0, TABLE1_DEFAULTS["a_abx"] if use_table1 else 0.5, 0.01)
j = st.sidebar.slider("j: relative risk during antibiotics", 1.0, 10.0, TABLE1_DEFAULTS["j"] if use_table1 else 3.0, 0.1)

st.sidebar.header("Paper-style staged intervention (Fig. 3b-like)")
replicate_fig3b = st.sidebar.checkbox(
    "Replicate staged interventions (30d p, 45d q, 60d abx↓, 90d isolate admissions)",
    value=True
)

# Staged intervention times and effects (from figure caption) :contentReference[oaicite:5]{index=5}
t_p = 30
t_q = 45
t_abx = 60
t_iso = 90

# In Fig 3b caption: strict IC p=50%; cohorting qn=64%; abx cut by 50% (a' = 25%, j=3) :contentReference[oaicite:6]{index=6}
p_stage = 0.50
qn_stage = 0.64
q_stage = min(1.0, qn_stage / max(n_prop, 1e-6))
a_abx_stage = 0.25
j_stage = 3.0

# Admissions isolation: "all further VRE-positive admissions are isolated" -> effectively f becomes 0 after t_iso :contentReference[oaicite:7]{index=7}
def controls_at_time(t):
    p_t, q_t, a_abx_t, j_t, f_t = p, q, a_abx, j, f
    if replicate_fig3b:
        # outbreak seeding: start with 1 colonized admission at day 0; keep f=0 unless isolation off? We'll handle seeding separately.
        # Staged controls:
        if t >= t_p:
            p_t = p_stage
        if t >= t_q:
            q_t = q_stage
        if t >= t_abx:
            a_abx_t = a_abx_stage
            j_t = j_stage
        if t >= t_iso:
            f_t = 0.0
    return p_t, q_t, a_abx_t, j_t, f_t

# Effective cohorting factor (Austin): R(q)=R0(1-q*n) :contentReference[oaicite:8]{index=8}
def cohort_factor(q_t):
    return max(0.0, 1.0 - q_t * n_prop)

def bp_effective(bp0, a_abx_t, j_t):
    # Paper: bp increased by factor 1 + a(j-1) :contentReference[oaicite:9]{index=9}
    return bp0 * (1.0 + a_abx_t * (j_t - 1.0))

# --------------------------
# Gillespie simulator (discrete counts + fixed beds by discharge->replacement)
# --------------------------
def simulate_gillespie(rng, T, t_grid):
    # State: Xp, Yp, Xh, Yh; with Xp+Yp = Np; Xh+Yh = Nh
    # Initial: use endemic-ish starting point or seed outbreak per paper fig3b
    if replicate_fig3b:
        # One colonized patient admitted on day 0 (caption) :contentReference[oaicite:10]{index=10}
        Yp = 1
        Xp = Np - Yp
    else:
        # start near admission prevalence
        Yp = int(round(f * Np))
        Xp = Np - Yp

    # HCW contamination starts low
    Yh = 0
    Xh = Nh

    Xp_path = np.zeros_like(t_grid, dtype=float)
    Yp_path = np.zeros_like(t_grid, dtype=float)
    Yh_path = np.zeros_like(t_grid, dtype=float)
    cum_acq_path = np.zeros_like(t_grid, dtype=float)  # cumulative acquisitions in ICU
    cum_acq = 0.0

    t = 0.0
    gi = 0
    while gi < len(t_grid) and t_grid[gi] <= 0:
        Xp_path[gi], Yp_path[gi], Yh_path[gi], cum_acq_path[gi] = Xp, Yp, Yh, cum_acq
        gi += 1

    while t < T:
        p_t, q_t, a_abx_t, j_t, f_t = controls_at_time(t)
        cf = cohort_factor(q_t)
        bp_t = bp_effective(bp, a_abx_t, j_t)

        # Apply infection control as multiplicative reduction in transmission
        # Barrier precautions reduce contamination and onward transmission :contentReference[oaicite:11]{index=11}
        ic_factor = (1.0 - p_t)

        # Event rates (Austin appendix structure) :contentReference[oaicite:12]{index=12}
        # 1) Patient discharge: uncolonized
        rate_dis_u = g_u * Xp
        # 2) Patient discharge: colonized
        rate_dis_c = g_c * Yp
        # 3) HCW decontamination
        rate_clear_h = m * Yh
        # 4) HCW contamination from colonized patient contact
        #    rate ~ a * bh * Xh * Yp  (scaled by cohorting + infection control)
        rate_contam_h = (a * bh * Xh * Yp) * cf * ic_factor
        # 5) Patient colonization from contaminated HCW contact
        #    rate ~ a * bp * Xp * Yh (scaled by cohorting + infection control + antibiotic factor)
        rate_col_p = (a * bp_t * Xp * Yh) * cf * ic_factor

        rate_total = rate_dis_u + rate_dis_c + rate_clear_h + rate_contam_h + rate_col_p
        if rate_total <= 0:
            break

        dt = rng.exponential(1.0 / rate_total)
        t_next = t + dt

        # record until t_next
        while gi < len(t_grid) and t_grid[gi] <= min(t_next, T):
            Xp_path[gi], Yp_path[gi], Yh_path[gi], cum_acq_path[gi] = Xp, Yp, Yh, cum_acq
            gi += 1

        if t_next > T:
            t = T
            break

        u = rng.random() * rate_total

        # Discharge uncolonized -> admit replacement (colonized with prob f_t)
        if u < rate_dis_u:
            if Xp > 0:
                Xp -= 1
                # immediate admission to keep beds filled
                if rng.random() < f_t:
                    Yp += 1
                else:
                    Xp += 1

        # Discharge colonized -> admit replacement (colonized with prob f_t)
        elif u < rate_dis_u + rate_dis_c:
            if Yp > 0:
                Yp -= 1
                # replacement admission
                if rng.random() < f_t:
                    Yp += 1
                else:
                    Xp += 1

        # HCW decontamination
        elif u < rate_dis_u + rate_dis_c + rate_clear_h:
            if Yh > 0:
                Yh -= 1
                Xh += 1

        # HCW contamination
        elif u < rate_dis_u + rate_dis_c + rate_clear_h + rate_contam_h:
            if Xh > 0:
                Xh -= 1
                Yh += 1

        # Patient colonization
        else:
            if Xp > 0:
                Xp -= 1
                Yp += 1
                cum_acq += 1

        t = t_next

    while gi < len(t_grid):
        Xp_path[gi], Yp_path[gi], Yh_path[gi], cum_acq_path[gi] = Xp, Yp, Yh, cum_acq
        gi += 1

    return Xp_path, Yp_path, Yh_path, cum_acq_path

# --------------------------
# Run ensemble
# --------------------------
t_grid = np.linspace(0, T_max, int(time_grid_points))
rng = np.random.default_rng(seed if use_seed else None)

Yp_all = np.zeros((runs, len(t_grid)))
Yh_all = np.zeros((runs, len(t_grid)))
Acq_all = np.zeros((runs, len(t_grid)))

rng = np.random.default_rng(seed if use_seed else None)
for r in range(runs):
    Xp_path, Yp_path, Yh_path, cum_acq_path = simulate_gillespie(rng, T_max, t_grid)
    Yp_all[r, :] = Yp_path
    Yh_all[r, :] = Yh_path
    Acq_all[r, :] = cum_acq_path

def qband(arr):
    med = np.quantile(arr, 0.50, axis=0)
    lo = np.quantile(arr, 0.025, axis=0)
    hi = np.quantile(arr, 0.975, axis=0)
    return med, lo, hi

Yp_med, Yp_lo, Yp_hi = qband(Yp_all)
Acq_med, Acq_lo, Acq_hi = qband(Acq_all)

yp_med_pct = 100 * (Yp_med / Np)
yp_hi_pct = 100 * (Yp_hi / Np)

# R0 formula (appendix) R0 = a^2 * bh*bp * Nh*Np*Dh*Dp :contentReference[oaicite:13]{index=13}
Dp_for_R0 = Dp_u  # typical LOS used in heuristic
R0_basic = (a**2) * bh * bp_effective(bp, a_abx, j) * Nh * Np * Dh * Dp_for_R0

st.metric("Approx. basic R₀ (heuristic)", f"{R0_basic:.2f}")

# --------------------------
# Plots
# --------------------------
fig = go.Figure()
fig.add_trace(go.Scatter(x=t_grid, y=yp_hi_pct, mode="lines", name="Patient colonized % (97.5%)", line=dict(width=1)))
fig.add_trace(go.Scatter(x=t_grid, y=100*(Yp_lo/Np), mode="lines", name="Patient colonized % (2.5%)", line=dict(width=1), fill="tonexty"))
fig.add_trace(go.Scatter(x=t_grid, y=yp_med_pct, mode="lines", name="Patient colonized % (median)", line=dict(width=3)))

if replicate_fig3b:
    for x, lab in [(t_p, "p"), (t_q, "q"), (t_abx, "abx↓"), (t_iso, "isolate admissions")]:
        fig.add_vline(x=x, line_dash="dash", annotation_text=lab, annotation_position="top right")

fig.update_layout(
    title="VRE colonization prevalence in patients (ensemble with 95% interval)",
    xaxis_title="Days",
    yaxis_title="% patients colonized",
    hovermode="x unified",
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=t_grid, y=Acq_hi, mode="lines", name="Cumulative acquisitions (97.5%)", line=dict(width=1)))
fig2.add_trace(go.Scatter(x=t_grid, y=Acq_lo, mode="lines", name="Cumulative acquisitions (2.5%)", line=dict(width=1), fill="tonexty"))
fig2.add_trace(go.Scatter(x=t_grid, y=Acq_med, mode="lines", name="Cumulative acquisitions (median)", line=dict(width=3)))
fig2.update_layout(
    title="Cumulative ICU acquisitions (stochastic ensemble)",
    xaxis_title="Days",
    yaxis_title="Cumulative acquisitions",
    hovermode="x unified",
    template="plotly_white"
)
st.plotly_chart(fig2, use_container_width=True)

# --------------------------
# Lab questions (paper-aligned) + download
# --------------------------
st.markdown("## Lab Questions (Austin paper — observe key outcomes)")
with st.expander("Lab Questions (click to open)", expanded=True):
    st.text_input("Student name / ID:", key="student_id")

    st.markdown("### 1) Stochasticity and small numbers")
    st.markdown("Under the **same parameters**, do you ever see very different outcomes across runs (e.g., near-elimination vs persistence)? Why is this expected in ICUs?")
    st.text_area("Answer:", key="ans1", height=120)

    st.markdown("### 2) Importation vs within-ICU transmission")
    st.markdown(
        "Vary **f** (colonized on admission). Identify a setting where within-ICU transmission appears controlled, "
        "but VRE persists because of colonized admissions (importation stabilizes persistence)."
    )
    st.text_area("Describe the settings you used and what you observed:", key="ans2", height=120)

    st.markdown("### 3) Hand hygiene (p) and cohorting (q) thresholds")
    st.markdown(
        "Find combinations of **p** and **q** where colonization prevalence declines substantially. "
        "How does increasing cohorting change the p level needed for control (qualitatively)?"
    )
    st.text_area("Answer:", key="ans3", height=120)

    st.markdown("### 4) Antibiotic restriction effect depends on transmission level")
    st.markdown(
        "Change antibiotic pressure (**a_abx, j**) and compare its impact when transmission is high vs already controlled. "
        "Does it match the paper’s message that antibiotic restriction helps most when transmission is low? :contentReference[oaicite:14]{index=14}"
    )
    st.text_area("Answer:", key="ans4", height=120)

    st.markdown("### 5) Model realism (reflection)")
    st.markdown(
        "Austin notes the framework can be modified for heterogeneity/compliance variation/environmental contamination. "
        "What changes would you add for a modern ICU? :contentReference[oaicite:15]{index=15}"
    )
    st.text_area("Your proposed improvements:", key="ans5", height=120)

st.markdown("### Download your lab responses")
out = io.StringIO()
ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
sid = st.session_state.get("student_id", "").strip()

out.write(f"timestamp,{ts}\nstudent_id,{sid}\n")
out.write("\n[parameters]\n")
params = {
    "T_max": T_max, "runs": runs, "Np": Np, "Nh": Nh, "n_prop": n_prop,
    "L": L, "Dp_u": Dp_u, "d_los": d_los, "Dp_c": Dp_c,
    "Dh": Dh, "a": a, "bp": bp, "bh": bh,
    "p": p, "q": q, "f": f, "a_abx": a_abx, "j": j,
    "replicate_fig3b": replicate_fig3b, "R0_basic_heuristic": R0_basic
}
for k, v in params.items():
    out.write(f"{k},{v}\n")

out.write("\n[answers]\n")
for k in ["ans1", "ans2", "ans3", "ans4", "ans5"]:
    val = str(st.session_state.get(k, "")).replace("\n", " ").replace("\r", " ")
    out.write(f"{k},{val}\n")

fname = f"AUSTIN_core_responses_{sid}.csv" if sid else "AUSTIN_core_responses.csv"
st.download_button("⬇️ Download my responses (CSV)", data=out.getvalue().encode("utf-8"), file_name=fname, mime="text/csv")
