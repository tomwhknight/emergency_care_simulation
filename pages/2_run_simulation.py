import streamlit as st
import pandas as pd

from src.base_params import build_global_params, MASTER_SEED
from src.trial_streamlit import TrialStreamlit
from src.trial_alt_streamlit import AltTrialStreamlit
from src.trial_alt2_streamlit import AltTrial2Streamlit


# =====================================================
# Page setup
# =====================================================

st.set_page_config(layout="wide")

# =====================================================
# Session state
# =====================================================

if "simulation_complete" not in st.session_state:
    st.session_state.simulation_complete = False

if "trials" not in st.session_state:
    st.session_state.trials = None

if "all_patient_results" not in st.session_state:
    st.session_state.all_patient_results = None

if "all_summary_results" not in st.session_state:
    st.session_state.all_summary_results = None


# =====================================================
# App banner
# =====================================================

st.image("assets/uom.jpeg", width=120)

# =====================================================
# Fixed simulation settings
# =====================================================

burn_in_days = 0
burn_in_time = burn_in_days * 1440

simulation_days = 5
user_simulation_time = simulation_days * 1440
simulation_time = user_simulation_time + burn_in_time


# =====================================================
# Sidebar: run settings
# =====================================================

st.sidebar.header("Run settings")

st.sidebar.caption(
    "The model runs for a fixed 7-day simulation period after a 2-day burn-in."
)

total_runs = st.sidebar.slider(
    "Simulation runs",
    min_value=0,
    max_value=10,
    value=5,
    step=1,
    help="Number of repeated model runs. Baseline is 5 runs.",
)


# =====================================================
# Run simulation
# =====================================================

st.markdown("### Run simulation")
st.caption(
    "Run the baseline model and both alternative models using the selected demand "
    "assumptions below. Staffing and capacity are fixed at the calibrated baseline values."
)

run_button = st.button("▶️ Run simulation", use_container_width=True)


# =====================================================
# Demand controls
# =====================================================

st.markdown("### Demand assumptions")
st.caption(
    "Adjust demand around the baseline scenario. A value of 0% means no change "
    "from the calibrated baseline."
)

col_attendance, col_referrals = st.columns(2)

with col_attendance:
    st.markdown("##### ED attendance")
    attendance_change_pct = st.slider(
        "Change from baseline (%)",
        min_value=-50,
        max_value=50,
        value=0,
        step=10,
        help="Negative values reduce attendance; positive values increase attendance.",
    )

with col_referrals:
    st.markdown("##### Medical referrals")
    referral_change_pct = st.slider(
        "Change from baseline (%)",
        min_value=-50,
        max_value=50,
        value=0,
        step=10,
        help="Negative values reduce referrals; positive values increase referrals.",
    )


# =====================================================
# Build parameters from calibrated baseline
# =====================================================

attendance_multiplier = 1 + attendance_change_pct / 100
referral_odds_multiplier = 1 + referral_change_pct / 100

global_params = build_global_params()

global_params.burn_in_time = burn_in_time
global_params.simulation_time = simulation_time

global_params.attendance_multiplier = attendance_multiplier
global_params.referral_sensitivity_on = True
global_params.referral_odds_multiplier = referral_odds_multiplier


# =====================================================
# Run all three model scenarios
# =====================================================

if run_button:
    progress_bar = st.progress(0, text="Starting simulations...")

    baseline_trial = TrialStreamlit(global_params, MASTER_SEED)
    baseline_trial.run(total_runs, progress_bar=progress_bar)

    progress_bar.progress(0, text="Starting direct to medicine scenario...")

    alt_trial = AltTrialStreamlit(global_params, MASTER_SEED)
    alt_trial.run(total_runs, progress_bar=progress_bar)

    progress_bar.progress(0, text="Starting direct to consultant scenario...")

    alt2_trial = AltTrial2Streamlit(global_params, MASTER_SEED)
    alt2_trial.run(total_runs, progress_bar=progress_bar)

    progress_bar.progress(100, text="Combining results...")

    baseline_results = baseline_trial.agg_results_df.copy()
    alt_results = alt_trial.agg_results_df.copy()
    alt2_results = alt2_trial.agg_results_df.copy()

    baseline_results["Scenario"] = "Baseline"
    alt_results["Scenario"] = "Direct to Medicine"
    alt2_results["Scenario"] = "Direct to Consultant"

    all_patient_results = pd.concat(
        [baseline_results, alt_results, alt2_results],
        ignore_index=True,
    )

    baseline_summary = baseline_trial.agg_results_complete.copy()
    alt_summary = alt_trial.agg_results_complete.copy()
    alt2_summary = alt2_trial.agg_results_complete.copy()

    baseline_summary["Scenario"] = "Baseline"
    alt_summary["Scenario"] = "Direct to Medicine"
    alt2_summary["Scenario"] = "Direct to Consultant"

    all_summary_results = pd.concat(
        [baseline_summary, alt_summary, alt2_summary],
        ignore_index=True,
    )

    st.session_state.trials = {
        "Baseline": baseline_trial,
        "Direct to Medicine": alt_trial,
        "Direct to Consultant": alt2_trial,
    }

    st.session_state.all_patient_results = all_patient_results
    st.session_state.all_summary_results = all_summary_results
    st.session_state.simulation_complete = True

    progress_bar.empty()
    st.success("Simulation complete!")


# =====================================================
# Navigation to results
# =====================================================

if st.session_state.get("simulation_complete", False):
    st.markdown("---")

    if st.button("➡️ View results summary", use_container_width=True):
        st.switch_page("pages/3_results.py")