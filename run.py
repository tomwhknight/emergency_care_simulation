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
from src.helper import rota_peak, save_rota_check, save_params

# --- Calculate peak capacity first ---

shift_patterns_weekday = [
    # Tier 1 Resident
    {"role": "tier_1", "shift_name": "Early",     "start": "08:00", "end": "16:00", "count": 7, "breaks": 1},
    {"role": "tier_1", "shift_name": "Middle_1",  "start": "12:00", "end": "22:00", "count": 5, "breaks": 2},
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

    ambulance_proportion = 20,
    walk_in_proportion = 80,

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
    sdec_close_hour = 17,

    weekday_sdec_base_capacity = 6,
    weekend_sdec_base_capacity = 5, 

    # AMU capacity
    max_amu_available_beds = 2,
    max_sdec_capacity = 5,

    # Service times
    mu_triage_assessment_time = 1.85,
    sigma_triage_assessment_time = 0.4,

    mu_ed_service_time = 3.95, 
    sigma_ed_service_time = 0.40, 

    mu_ed_decision_time = 4.20, 
    sigma_ed_decision_time = 0.95, 

    mu_medical_service_time = 4.11,
    sigma_medical_service_time = 0.55,

    max_medical_service_time =  240,
    min_medical_service_time = 30, 

    mu_consultant_assessment_time = 3.2,
    sigma_consultant_assessment_time = 0.32, 

    # Routing logic

    sdec_prob_threshold = 0.10,
    paediatric_referral_rate = 0.10,

    initial_medicine_discharge_prob = 0.05,
    consultant_discharge_prob = 0.35,

    # Threshold for scerio analysis

    direct_triage_threshold = None, # Run None for no threshold applied

    # Simulation

    simulation_time = 44640,
    cool_down_time = 1440,
    burn_in_time = 2880)  # burn in to prevent initiation bias 

global_params.max_ed_doctor_capacity = max(
    rota_peak(shift_patterns_weekday),
    rota_peak(shift_patterns_weekend)
)

MASTER_SEED = 20251001   # Master SEED for all process RNGs

if __name__ == "__main__":
    trial = Trial(global_params, MASTER_SEED)
    
    t0 = time.perf_counter()
    info = trial.run(run_number=5)
    elapsed = time.perf_counter() - t0
    
    # Create folder for each batch           
    # --- Create folder for each batch and include ED service parameters ---
    mu  = global_params.mu_ed_service_time
    sigma = global_params.sigma_ed_service_time
    mu_dec = global_params.mu_ed_decision_time
    sigma_dec = global_params.sigma_ed_decision_time

    batch_label = f"batch_mu{mu:.2f}_sigma{sigma:.2f}_mu{mu_dec:.2f}_sigma{sigma_dec:.2f}"
    batch_dir = os.path.join(info["scenario_dir"], batch_label)
    os.makedirs(batch_dir, exist_ok=True)

    # --- Print a quick summary to console ---
    total_runs = info.get("total_runs", 1)   # fallback to what you passed in
    secs_per_run = elapsed / max(1, total_runs)
    print(
        f" Runtime: {elapsed:,.1f}s "
        f"({secs_per_run:.2f}s/run across {total_runs} runs)  "
        f"[seed={MASTER_SEED}]"
    )

    # --- Save params (JSON) ---

    params_filename = f"{basename(batch_dir)}.json"
    save_params(
        global_params,
        batch_dir,
        filename=params_filename,  # <- fixed typo
        extra={
            "master_seed": MASTER_SEED,
            "created": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 3),
            "total_runs": total_runs,
            "seconds_per_run": round(secs_per_run, 3),
            "ed_service_params": {
                "mu_ed_service_time": mu,
                "sigma_ed_service_time": sigma,
                "mu_ed_decision_time": mu_dec,
                "sigma_ed_decision_time": sigma_dec,
            },
            "timing_windows": {
                "burn_in_time": global_params.burn_in_time,
                "cool_down_time": global_params.cool_down_time,
                "simulation_time": global_params.simulation_time,
            },
        },
    )

    # --- Timings log (you were missing these two lines) ---
    timings_path = os.path.join(batch_dir, "timings.csv")
    header_needed = not os.path.exists(timings_path)


    with open(timings_path, "a", encoding="utf-8") as f:
        if header_needed:
            f.write("timestamp,master_seed,total_runs,elapsed_seconds,seconds_per_run,burn_in,cool_down,sim_time\n")
        f.write(
            f"{datetime.now().isoformat(timespec='seconds')},"
            f"{MASTER_SEED},"
            f"{total_runs},"
            f"{elapsed:.3f},"
            f"{secs_per_run:.3f},"
            f"{global_params.burn_in_time},"
            f"{global_params.cool_down_time},"
            f"{global_params.simulation_time}\n"
        )

    # Save rota checks into the *batch* folder
    save_rota_check(shift_patterns_weekday, batch_dir, filename="rota_weekday.csv")
    save_rota_check(shift_patterns_weekend, batch_dir, filename="rota_weekend.csv")

