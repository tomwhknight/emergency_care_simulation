# run.py
# run.py


USE_ALT_MODEL = False  # Set to False to use the original model

if USE_ALT_MODEL:
    from src.trial_alt import AltTrial as Trial
    print("Running simulation with AltModel logic.")
else:
    from src.trial import Trial
    print("Running simulation with original model logic.")

# Import 

import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
from os.path import basename


# Import helpers

from src.global_parameters import GlobalParameters
from src.model import Model
from src.helper import rota_peak

# --- Calculate peak capacity first ---

shift_patterns_weekday = [
    # Tier 1 Resident
    {"role": "tier_1", "shift_name": "Early",     "start": "08:00", "end": "16:00", "count": 8, "breaks": 1},
    {"role": "tier_1", "shift_name": "Middle_1",  "start": "12:00", "end": "22:00", "count": 4, "breaks": 2},
    {"role": "tier_1", "shift_name": "Twilight",  "start": "16:00", "end": "00:00", "count": 5, "breaks": 1},
    {"role": "tier_1", "shift_name": "Night",     "start": "22:00", "end": "08:00", "count": 3, "breaks": 2},

    # Tier 2 Resident
    {"role": "tier_2", "shift_name": "Early",     "start": "08:00", "end": "16:00", "count": 4, "breaks": 1},
    {"role": "tier_2", "shift_name": "Middle",    "start": "12:00", "end": "20:00", "count": 5, "breaks": 1},
    {"role": "tier_2", "shift_name": "Twilight",  "start": "16:00", "end": "00:00", "count": 3, "breaks": 1},
    {"role": "tier_2", "shift_name": "Night",     "start": "22:00", "end": "08:00", "count": 3, "breaks": 2},

    # GP
    {"role": "GP",     "shift_name": "GP",        "start": "09:00", "end": "23:00", "count": 1, "breaks": 2},

    # ACP
    {"role": "ACP",    "shift_name": "Early",     "start": "09:00", "end": "17:00", "count": 2, "breaks": 1},
    {"role": "ACP",    "shift_name": "Late",      "start": "15:00", "end": "22:00", "count": 1, "breaks": 1},

]

# Weekend shift pattern — tuned to approximate your hourly means
# (Night ~7; 09–11 ~12–13; 12–15 ~16–18; 16–19 ~15–18; 21–23 ~13–15)

shift_patterns_weekend = [
    # --- Tier 1 Resident ---
    {"role": "tier_1", "shift_name": "Early",     "start": "08:00", "end": "16:15", "count": 7, "breaks": 1},
    {"role": "tier_1", "shift_name": "Middle_1",  "start": "12:00", "end": "22:00", "count": 5, "breaks": 2},
    {"role": "tier_1", "shift_name": "Twilight",  "start": "16:00", "end": "00:00", "count": 4, "breaks": 1},
    {"role": "tier_1", "shift_name": "Night",     "start": "22:00", "end": "08:00", "count": 4, "breaks": 2},

    # --- Tier 2 Resident ---
    {"role": "tier_2", "shift_name": "Early",     "start": "08:00", "end": "16:15", "count": 3, "breaks": 1},
    {"role": "tier_2", "shift_name": "Middle",    "start": "12:00", "end": "20:00", "count": 2, "breaks": 1},  # ends 20:00 (helps taper)
    {"role": "tier_2", "shift_name": "Twilight",  "start": "16:00", "end": "00:00", "count": 2, "breaks": 1},
    {"role": "tier_2", "shift_name": "Night",     "start": "22:00", "end": "08:00", "count": 3, "breaks": 2},

    # --- GP & ACP (keep 1 each as per your current rota) ---
    {"role": "GP",     "shift_name": "GP",        "start": "09:00", "end": "23:00", "count": 1, "breaks": 2},
    {"role": "ACP",    "shift_name": "Early",     "start": "09:00", "end": "17:00", "count": 1, "breaks": 1},
    {"role": "ACP",    "shift_name": "Late",      "start": "15:00", "end": "22:00", "count": 1, "breaks": 1},
]
global_params = GlobalParameters(

    ambulance_proportion = 0.2,
    walk_in_proportion = 0.8,

    # Source of referral
    proportion_direct_primary_care = 0.01,  
    
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

    # Staffing resource
    ambulance_triage_nurse_capacity = 1,
    walk_in_triage_nurse_capacity = 3,

    medical_doctor_capacity = 5,
    consultant_capacity = 1, 
    shift_patterns_weekday = shift_patterns_weekday,
    shift_patterns_weekend = shift_patterns_weekend,

    # SDEC capacity
    sdec_open_hour = 7, 
    sdec_close_hour = 18,

    weekday_sdec_base_capacity = 6,
    weekend_sdec_base_capacity = 5, 

    # AMU capacity
    max_amu_available_beds = 56,
    amu_queue_soft = 10,
    amu_queue_hard =  25,
    amu_surge_max_scale = 1.10,
    max_sdec_capacity = 5,

    # Service times
    mu_triage_assessment_time = 1.0,
    sigma_triage_assessment_time = 0.7,

    mu_ed_service_time = 3.95, 
    sigma_ed_service_time = 0.55, 

    mu_ed_decision_time = 4.20, 
    sigma_ed_decision_time = 0.95, 
    joint_gamma = 0.5,

    # Params to emulate the 240 min breach boudnary 

    decision_hazard_strength_240  = 0.175,
    adjustment_start = 240,
    adjustment_end = 300, 

    mu_medical_service_time = 4.35,
    sigma_medical_service_time = 0.55,

    mu_consultant_assessment_time = 2.95,
    sigma_consultant_assessment_time = 0.30, 

    # Routing logic

    sdec_prob_threshold = 0.10,
    paediatric_referral_rate = 0.10,
    prob_referral_to_medicine_adult = 0.525, 
    
    initial_medicine_discharge_prob = 0.00,
    consultant_discharge_prob = 0.20,
    
    mu_surgical_bed_delay = 5.20,
    sigma_surgical_bed_delay = 1.00,

    # Routing / scenario analysis
    direct_triage_threshold = None,  # None = no direct triage rule applied

    # Simulation

    simulation_time = 56150,
    cool_down_time = 2880,
    burn_in_time = 10080)  # burn in to prevent initiation bias 


# Scenario / sensitivity settings – set in run, not baked into GlobalParameters constructor
global_params.referral_sensitivity_on = False
global_params.referral_odds_multiplier = 1.0
global_params.scenario_name = None


global_params.max_ed_doctor_capacity = max(
    rota_peak(shift_patterns_weekday),
    rota_peak(shift_patterns_weekend), 
    ) + 2

MASTER_SEED = 20251001   # Master SEED for all process RNGs



def run_model(global_params, mu, sigma, total_runs=10):
    global_params.mu_ed_service_time = mu
    global_params.sigma_ed_service_time = sigma

    trial = Trial(global_params, master_seed=MASTER_SEED)
    trial.run(total_runs)

    df = trial.agg_results_df.copy()

    # Map columns to standard names
    rename_map = {
        "Time in System": "time_in_system",
        "Clock Hour of Arrival": "hour_of_day",
        "Admitted": "admitted",
        "Discharge Decision Point": "discharge_decision_point"
    }
    cols = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=cols)

    # Ensure columns exist, fill if not
    if "hour_of_day" not in df:
        df["hour_of_day"] = "00:00"    # dummy value if missing
    if "admitted" not in df:
        df["admitted"] = 0             # default to non-admitted

    patients_df = df.copy()
    queue_ed_df = trial.agg_ed_assessment_queue_monitoring_df.copy()

    return {
        "patients": df,
        "queue_ed": queue_ed_df
    }
