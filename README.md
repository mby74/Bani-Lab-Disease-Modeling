[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
© 2026 Majid Bani, PhD — University of Missouri–Kansas City 

# Bani-Lab-Disease-Modeling

**Bani-Lab-Disease-Modeling** is a collection of interactive simulation labs designed to help **medical students and healthcare professionals** explore how clinical decisions and recommendations influence the spread of infectious diseases in communities and healthcare settings.

These labs are not predictive tools. They are **clinical reasoning aids** that allow learners to test *“what-if”* scenarios, examine uncertainty, and understand tradeoffs in infection prevention, hospital epidemiology, and antimicrobial stewardship.

This repository supports a simulation-based learning program at the **University of Missouri–Kansas City (UMKC) School of Medicine**.

---

##  Educational Goals

The simulations are designed to help learners:

- Understand how **physician actions and recommendations** (e.g., isolation, PPE use, antibiotic prescribing) alter infection dynamics
- Reason about **uncertainty and risk**, not just average outcomes
- Compare **early vs delayed interventions**
- Distinguish **controllable transmission** from **unavoidable importation risk**
- Translate modeling insights into **realistic clinical and operational decisions**

No prior background in modeling, programming, or epidemiology is required.

---

##  How the Labs Are Used

Each lab is integrated into a **3-step workflow**:

1. **Pre-Lab (10–20 min)**  
   Case-based reading introducing the clinical scenario and decision problem.

2. **In-Lab (30–60 min)**  
   Learners interact with the simulation, test interventions, and download results.

3. **Post-Lab (10 min)**  
   Short reflection: decision, justification, and model limitations.

The accompanying textbook (written in LaTeX) provides all clinical context and guidance.

---

##  Contents

### Module 1 — Community Transmission
- `sir_app.py`  
  Deterministic SIR model for basic intuition and thresholds.
- `sir_delayed_intervention_app.py`  
  Demonstrates the importance of timing in public health recommendations.

### Module 2 — Hospital Stochasticity & Uncertainty
- `sir_hospital_stochastic_tau.py`  
  Discrete-time stochastic hospital SIR model.
- `sir_hospital_gillespie.py`  
  Event-driven (Gillespie) stochastic hospital SIR model.

### Module 3 — Tradeoffs, Timing, and Importation Risk
- `EXTENDED_sir_hospital_stochastic_tau.py`  
- `EXTENDED_sir_hospital_gillespie.py`  

These models include **background importation (κ)** to represent admissions and staff exposures, highlighting why outbreaks can persist despite strong in-hospital controls.

### Module 4 — From Papers to Practice
- `AUSTIN_core_VRE_gillespie.py`  
  Implementation of the classic Austin et al. ICU VRE transmission model.
- `HAI_generalized_Austin_gillespie.py`  
  Extension to common HAIs (VAP, CAUTI, CLABSI, MRSA).
- `SSI_stochastic_decision_lab.py`  
  Separate stochastic model for **surgical site infections**, focused on procedures, prophylaxis, and OR practices.



---

## 🔐 Privacy & Data

These simulations are designed for safe educational use.

- No patient data is used or stored
- No login or authentication is required
- Student responses are downloaded locally by the user
- No data is retained on the server

These design choices support privacy, security, and responsible educational deployment.

---

## ⚠️ Important Disclaimer

These models are **educational tools only**.

They:
- Do **not** predict real outbreaks
- Do **not** replace infection prevention guidelines
- Do **not** represent validated operational forecasts

The models are intentionally simplified to support learning, reflection, and discussion about clinical decision-making under uncertainty.

---

## 📚 Key References

- Austin DJ, Bonten MJM, Weinstein RA, Slaughter S, Anderson RM.  
  *Transmission dynamics of vancomycin-resistant enterococci in intensive-care units.*  
  **PNAS**, 1999.

- Centers for Disease Control and Prevention (CDC).  
  *Core Infection Prevention and Control Practices for Safe Healthcare Delivery.*

- Centers for Disease Control and Prevention (CDC).  
  *Healthcare-Associated Infection (HAI) Prevention Guidelines*  
  (CAUTI, CLABSI, SSI).

- Society for Healthcare Epidemiology of America (SHEA) / Infectious Diseases Society of America (IDSA).  
  *Compendium of Strategies to Prevent Healthcare-Associated Infections.*

- Fineberg HV.  
  *Pandemic preparedness and response — lessons from COVID-19.*  
  **New England Journal of Medicine**.

---

##  Running the Apps Locally

### Requirements
- Python 3.9+
- Packages listed in `requirements.txt`

```bash
pip install -r requirements.txt
streamlit run sir_app.py










## 📄 License

This project is licensed under the  
**Creative Commons Attribution–NonCommercial 4.0 International License (CC BY-NC 4.0)**.

You are free to use, share, and adapt this work for **non-commercial educational
and academic purposes**, with appropriate attribution.

Commercial use requires explicit permission from the author.
