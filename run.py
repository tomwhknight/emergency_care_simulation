# run.py


USE_ALT_MODEL = False  # Set to False to use the original model

if USE_ALT_MODEL:
    from src.trial_alt import AltTrial as Trial
    print("Running simulation with AltModel logic.")
else:
    from src.trial import Trial
    print("Running simulation with original model logic.")

from src.global_parameters import GlobalParameters
from src.model import Model
from src.helper import rota_peak, save_rota_check
import numpy as np
import pandas as pd


# --- Calculate peak capacity first ---

shift_patterns = [
        # Tier 1 Resident
        {"role": "tier_1", "shift_name": "Early",    "start": "08:00", "end": "15:45", "count": 7},
        {"role": "tier_1", "shift_name": "Early 2",  "start": "08:00", "end": "17:00", "count": 2},
        {"role": "tier_1", "shift_name": "Middle_1",   "start": "10:00", "end": "18:30", "count": 3},
        {"role": "tier_1", "shift_name": "Middle_2",   "start": "13:00", "end": "21:30", "count": 2},
        {"role": "tier_1", "shift_name": "Twilight",     "start": "16:00", "end": "23:30", "count": 5},
        {"role": "tier_1", "shift_name": "Night",   "start": "22:00", "end": "07:30", "count": 3},
    
        # Tier 2 Resident
        {"role": "tier_2", "shift_name": "Early",  "start": "08:00", "end": "15:45", "count": 3},
        {"role": "tier_2", "shift_name": "Middle", "start": "11:00", "end": "18:30", "count": 2},
        {"role": "tier_2", "shift_name": "Twilight","start": "16:00", "end": "23:30", "count": 5},
        {"role": "tier_2", "shift_name": "Night",    "start": "22:00", "end": "07:30", "count": 3},
        
        # GP
        {"role": "GP", "shift_name": "GP", "start": "09:00", "end": "22:30", "count": 1},

        # PA
        {"role": "PA", "shift_name": "Early", "start": "08:00", "end": "15:30", "count": 1},
        {"role": "PA", "shift_name": "Late", "start": "12:00", "end": "21:30", "count": 1},
    
        # ACP
        {"role": "ACP", "shift_name": "Early",       "start": "07:30", "end": "15:30", "count": 1},
        {"role": "ACP", "shift_name": "Late",        "start": "15:30", "end": "23:30", "count": 1},

        # ENP
        {"role": "ENP", "shift_name": "Early",     "start": "08:00", "end": "20:30", "count": 1},
        {"role": "ENP", "shift_name": "Late",        "start": "11:00", "end": "23:30", "count": 1},
    ]

# Path to your baseline output dir
output_dir = "/Users/thomasknight/Local files/DES/output/des_output/baseline"

rota_path = save_rota_check(shift_patterns, output_dir)
print(f"Rota check saved to {rota_path}")

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
    hca_capacity = 3, 

    medical_doctor_capacity = 5,
    consultant_capacity = 1, 
    shift_patterns = shift_patterns,

    # Bloods
    bloods_request_probability = 0.5,

    # SDEC capacity
    sdec_open_hour = 7, 
    sdec_close_hour = 18,

    weekday_sdec_base_capacity = 6,
    weekend_sdec_base_capacity = 5, 

    # AMU capacity
    max_amu_available_beds = 10,
    max_sdec_capacity = 5,

    # Service times
    mean_triage_assessment_time = 6,
    stdev_triage_assessment_time = 1,

    mean_blood_draw_time = 6, 
    stdev_blood_draw_time = 1,

    mu_blood_lab_time = 4.0,
    sigma_blood_lab_time = 0.5,

    mu_ed_service_time = 3.98, 
    sigma_ed_service_time = 0.4, 

    max_ed_service_time = 420,
    min_ed_service_time = 0,  

    mu_medical_service_time = 4.3,
    sigma_medical_service_time = 0.5,

    max_medical_service_time =  240,
    min_medical_service_time = 30, 

    mean_consultant_assessment_time = 30,
    stdev_consultant_assessment_time = 10, 

    # Routing logic

    medical_referral_rate = 0.50,
    paediatric_referral_rate = 0.1,

    initial_medicine_discharge_prob = 0.05,
    consultant_discharge_prob = 0.30,

    simulation_time = 11520,
    burn_in_time = 1440)  # burn in to prevent initiation bias 


global_params.max_ed_capacity = rota_peak(global_params.shift_patterns)

if __name__ == "__main__":
    trial = Trial(global_params)
    trial.run(run_number=50)

    # Example: save rota check
    output_dir = "/Users/thomasknight/Local files/DES/output/des_output/baseline"
    rota_path = save_rota_check(shift_patterns, output_dir)
    print(f"Rota check saved to {rota_path}")

