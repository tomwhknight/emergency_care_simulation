# run.py

USE_ALT_MODEL = True  # Set to False to use the original model

if USE_ALT_MODEL:
    from src.trial_alt import AltTrial as Trial
    print("Running simulation with AltModel logic.")
else:
    from src.trial import Trial
    print("Running simulation with original model logic.")

from src.global_parameters import GlobalParameters


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

        ed_doctor_capacity = 24,
        medical_doctor_capacity = 5,
        consultant_capacity = 1, 

        # SDEC capacity
        sdec_open_hour = 7, 
        sdec_close_hour = 16,

        weekday_sdec_base_capacity = 4,
        weekend_sdec_base_capacity = 4, 

        # AMU capacity
        max_amu_available_beds = 10,
        max_sdec_capacity = 10,

        # Service times
        mean_triage_assessment_time = 5,
        stdev_triage_assessment_time = 2,
    
        mu_ed_assessment_discharge = 4.1, 
        sigma_ed_assessment_discharge = 1.0, 

        wb_shape_ed_assessment_admit = 1.6,
        wb_scale_ed_assessment_admit = 1/0.01,

        mu_ed_service_time = 4.4, 
        sigma_ed_service_time = 0.5,
        max_ed_service_time = 120,
        min_ed_service_time = 30,  

        mu_medical_service_time = 4.5,
        sigma_medical_service_time = 0.68,

        max_medical_service_time = 240,
        min_medical_service_time = 30, 

        mean_consultant_assessment_time = 30,
        stdev_consultant_assessment_time = 10, 
       
        initial_medicine_discharge_prob = 0.10,
        consultant_discharge_prob = 0.4,
        
        simulation_time = 11520,
        burn_in_time = 1440) # burn in to prevent initiation bias
        
    trial = Trial(global_params)
    total_runs = 50
    trial.run(total_runs)