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

def create_default_global_params():
    return GlobalParameters(
        ambulance_proportion = 20,
        walk_in_proportion = 80,
        proportion_direct_primary_care = 0.03,  
        ambulance_acuity_probabilities = {
            1: 0.02, 2: 0.40, 3: 0.50, 4: 0.05, 5: 0.01,
        },
        walk_in_acuity_probabilities = {
            1: 0.05, 2: 0.05, 3: 0.40, 4: 0.30, 5: 0.20,
        },
        sdec_appropriate_rate = 0.10,
        medical_referral_rate = 0.175,
        speciality_referral_rate = 0.150,
        ambulance_triage_nurse_capacity = 1,
        walk_in_triage_nurse_capacity = 2,
        ed_doctor_capacity = 26,
        medical_doctor_capacity = 5,
        consultant_capacity = 1,
        sdec_open_hour = 7,
        sdec_close_hour = 16,
        weekday_sdec_base_capacity = 4,
        weekend_sdec_base_capacity = 4,
        max_sdec_capacity = 10,
        max_amu_available_beds = 10,
        mean_triage_assessment_time = 5,
        stdev_triage_assessment_time = 2,
        mu_ed_assessment_discharge = 4.28,
        sigma_ed_assessment_discharge = 1.04,
        wb_shape_ed_assessment_admit = 1.6,
        wb_scale_ed_assessment_admit = 1 / 0.01,
        mu_ed_service_time = 4.2,
        sigma_ed_service_time = 0.5,
        max_ed_service_time = 180,
        min_ed_service_time = 25,
        mu_medical_service_time = 4.5,
        sigma_medical_service_time = 0.68,
        max_medical_service_time = 240,
        min_medical_service_time = 25,
        mean_consultant_assessment_time = 30,
        stdev_consultant_assessment_time = 10,
        initial_medicine_discharge_prob = 0.10,
        consultant_discharge_prob = 0.4,
        simulation_time = 11520,
        burn_in_time = 1440
    )

def run_model_with_params(param_dict):
    g = create_default_global_params()
    for key, value in param_dict.items():
        if hasattr(g, key):
            setattr(g, key, value)

    trial = Trial(g)
    trial.run(run_number=1)
    return trial.agg_results_df  # Or .run_results_df for individual patients

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

        # ED disposition
        medical_referral_rate = 0.175,
        speciality_referral_rate = 0.150,
        
        # Staffing resource
        ambulance_triage_nurse_capacity = 1,
        walk_in_triage_nurse_capacity = 2,

        ed_doctor_capacity = 26,
        medical_doctor_capacity = 5,
        consultant_capacity = 1, 

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
    
        mu_ed_assessment_discharge = 4.28, 
        sigma_ed_assessment_discharge = 1.04, 

        wb_shape_ed_assessment_admit = 1.55,
        wb_scale_ed_assessment_admit = 90.5,

        mu_ed_service_time = 4.35, 
        sigma_ed_service_time = 0.5,

        max_ed_service_time = 240,
        min_ed_service_time = 30,  

        mu_medical_service_time = 4.3,
        sigma_medical_service_time = 0.5,

        max_medical_service_time =  240,
        min_medical_service_time = 30, 

        mean_consultant_assessment_time = 30,
        stdev_consultant_assessment_time = 10, 
       
        initial_medicine_discharge_prob = 0.10,
        consultant_discharge_prob = 0.4,
        
        simulation_time = 11520,
        burn_in_time = 1440)  # burn in to prevent initiation bias    
    
    trial = Trial(global_params)
    total_runs = 10
    trial.run(total_runs)

 
