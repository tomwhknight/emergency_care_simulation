# app.py

import plotly.express as px
import streamlit as st
from src.global_parameters import GlobalParameters
from src.trial import Trial

if "trial" not in st.session_state:
    st.session_state.trial = None

st.set_page_config(layout="wide")

# 1. App Banner

# Two logos: UoM (left) and MFT (right)
col1, col2, col3 = st.columns([1.25, 2, 1.25])

with col1:
    st.image("/Users/thomasknight/Desktop/ACL/Side projects/sepsis/uom.jpeg")

with col3:
    st.image("/Users/thomasknight/Desktop/ACL/Side projects/sepsis/mft.png")

# App title

st.markdown(
    """
    <h1 style='text-align: center; font-size: 2.3em;'>
         Acute Care Pathway Simulation Dashboard
    </h1>
    <p style='text-align: center; font-size: 1.05em; color: gray;'>
        This dashboard allows users to explore outcomes and identify system bottlenecks in the emergency department, based on a simulated model of acute medical activity under different hypothetical scenarios.
    </p>
    """,
    unsafe_allow_html=True
)

# --- 1. Simulation parameters ---

st.sidebar.header("Simulation Settings")

user_simulation_days = st.slider(
    "Simulation Time (days)", 
    min_value=1, 
    max_value=7, 
    value=1, 
    step=1
)

user_simulation_time = user_simulation_days * 1440

total_runs = st.slider("Number of Simulation Runs", min_value=1, max_value=100, value=5)

burn_in_time = 1440 # burn in to prevent initiation bias
        
simulation_time =  user_simulation_time + burn_in_time    


# --- 2. Demand ---
st.sidebar.subheader("Demand")

ed_threshold = st.sidebar.slider(
    "ED referral threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.05
)
sdec_threshold = st.sidebar.slider(
    "SDEC appropriateness threshold", min_value=0.1, max_value=1.0, value=0.6, step=0.05
)

# --- 3. Capacity ---

st.sidebar.subheader("Capacity")

sdec_open_hour = st.sidebar.slider(
    "SDEC opening hours ", min_value = 0, max_value= 23, value= 7, step=1
)
sdec_close_hour = st.sidebar.slider(
    "SDEC closing hours ", min_value = 0, max_value= 23, value= 17, step=1 
)

# --- 4. Staffing resource ---

st.sidebar.subheader("Staffing")

walk_in_triage_nurse_capacity = st.sidebar.number_input(
    "Walk in triage nurse capacity", min_value=2, max_value= 5, value=2, step=1
)

ambulance_triage_nurse_capacity = st.sidebar.number_input(
    "Ambulance triage nurse capacity", min_value=1, max_value=5, value=1, step=1
)

ed_doctor_capacity = st.sidebar.number_input(
    "ED doctor capacity", min_value=10, max_value=50, value=20, step=1
)

medical_doctor_capacity = st.sidebar.number_input(
    "Medical doctor capacity", min_value=1, max_value=10, value=5, step=1
)

consultant_capacity = st.sidebar.number_input(
    "Consultant capacity", min_value=1, max_value=10, value=1, step=1
)

# --- 5. Fixed parameters ---

global_params = GlobalParameters(

    # Simulation 
    
    burn_in_time=burn_in_time,
    simulation_time=simulation_time,

    # Patient flow proportions

    ambulance_proportion = 20,
    walk_in_proportion = 80,
    proportion_direct_primary_care = 0.07,  
    
    # Fixed bed capacity

    weekday_sdec_base_capacity = 4,
    weekend_sdec_base_capacity = 4,   

    max_amu_available_beds = 20,
    max_sdec_capacity = 10,

    # Patient characterstics 
        
    ambulance_acuity_probabilities = {
    1: 0.02,    
    2: 0.40,  
    3: 0.50,     
    4: 0.05,
    5: 0.01,
    },  

    walk_in_acuity_probabilities = {
    1: 0.05,    
    2: 0.05,  
    3: 0.40,     
    4: 0.30,
    5: 0.20,
    },  

    # Thresholds
    sdec_threshold=sdec_threshold,
    ed_threshold=ed_threshold,

    # Staffing
    ambulance_triage_nurse_capacity=ambulance_triage_nurse_capacity,
    walk_in_triage_nurse_capacity=walk_in_triage_nurse_capacity,
    ed_doctor_capacity=ed_doctor_capacity,
    medical_doctor_capacity=medical_doctor_capacity,
    consultant_capacity=consultant_capacity,

    # SDEC capacity
    sdec_open_hour=sdec_open_hour,
    sdec_close_hour=sdec_close_hour,

    # Service times

    mean_triage_assessment_time = 5,
    stdev_triage_assessment_time = 2,
        
    mean_ed_assessment_time = 60,
    stdev_ed_assessment_time = 30,

    mu_ed_delay_time_discharge = 4.5,
    sigma_ed_delay_time_discharge = 1.0,

    mu_ed_delay_time_admission = 4.6,
    sigma_ed_delay_time_admission = 1.2,

    mean_initial_medical_assessment_time = 60,
    stdev_initial_medical_assessment_time = 30, 

    mean_consultant_assessment_time = 25,
    stdev_consultant_assessment_time = 10, 
       
     # Decision Probabilities

    initial_medicine_discharge_prob = 0.1,
    consultant_discharge_prob = 0.3)

    
# --- Run trial ---

if st.button("Run Simulation"):
    progress_bar = st.progress(0, text="Starting simulation...")    
    
    trial = Trial(global_params)
    trial.run(total_runs, progress_bar=progress_bar)  # handles everything internally
    
    # Create a placeholder for the progress bar
    progress_bar = st.progress(0, text="Running simulation...")
    st.session_state.trial = trial  
    st.session_state.simulation_complete = True

    progress_bar.empty()
    st.success("Simulation complete!")

if st.session_state.get("simulation_complete", False):
    st.markdown("---")
    if st.button("➡️ View Results Summary"):
        st.switch_page("pages/3_results.py")
     

