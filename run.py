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

if __name__ == "__main__":
    global_params = GlobalParameters(

        ambulance_proportion = 20,
        walk_in_proportion = 80,

        # Source of referral
        proportion_direct_primary_care = 0.03,  
        
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

        # Patient characteristic variables

        sdec_appropriate_rate = 0.10,

        # Staffing resource
        ambulance_triage_nurse_capacity = 1,
        walk_in_triage_nurse_capacity = 2,
        hca_capacity = 2, 

        ed_doctor_capacity = 26,
        medical_doctor_capacity = 5,
        consultant_capacity = 1, 

        # SDEC capacity
        bloods_request_probability = 0.3,

        # SDEC capacity
        sdec_open_hour = 7, 
        sdec_close_hour = 18,

        weekday_sdec_base_capacity = 4,
        weekend_sdec_base_capacity = 4, 

        # AMU capacity
        max_amu_available_beds = 10,
        max_sdec_capacity = 10,

        # Service times
        mean_triage_assessment_time = 5,
        stdev_triage_assessment_time = 2,

        mean_blood_draw_time = 5, 
        stdev_blood_draw_time = 2,

        mean_blood_lab_time = 90,
        stdev_blood_lab_time = 15,
    
        mu_ed_service_time = 4.37, 
        sigma_ed_service_time = 0.83, 

        max_ed_service_time = 240,
        min_ed_service_time = 20,  

        mu_medical_service_time = 4.3,
        sigma_medical_service_time = 0.5,

        max_medical_service_time =  240,
        min_medical_service_time = 30, 

        mean_consultant_assessment_time = 30,
        stdev_consultant_assessment_time = 10, 

        # Routing logic

        medical_referral_rate = 0.4,
        paediatric_referral_rate = 0.1,

        initial_medicine_discharge_prob = 0.10,
        consultant_discharge_prob = 0.4,

    
        
        simulation_time = 2880,
        burn_in_time = 1440)  # burn in to prevent initiation bias    
    
    trial = Trial(global_params)
    total_runs = 10
    trial.run(total_runs)

 
