# run.py

from src.global_parameters import GlobalParameters
from src.trial import Trial

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

        sdec_appropriate_rate = 0.05,

        # ED disposition
        medical_referral_rate = 0.15,
        speciality_referral_rate = 0.05,
        
        # Staffing resource
        ambulance_triage_nurse_capacity = 1,
        walk_in_triage_nurse_capacity = 2,

        ed_doctor_capacity = 30,
        medical_doctor_capacity = 5,
        consultant_capacity = 1, 

        # SDEC capacity
        sdec_open_hour = 7, 
        sdec_close_hour = 16,

        weekday_sdec_base_capacity = 4,
        weekend_sdec_base_capacity = 4, 

        # AMU capacity
        max_amu_available_beds = 20,
        max_sdec_capacity = 10,

        # Service times
        mean_triage_assessment_time = 5,
        stdev_triage_assessment_time = 2,
    
        mu_ed_assessment_discharge = 4.2, 
        sigma_ed_assessment_discharge = 1.0, 

        wb_shape_ed_assessment_admit = 1.6,
        wb_scale_ed_assessment_admit = 1/0.01,

        mu_ed_service_time = 3.9, 
        sigma_ed_service_time = 0.5,
        max_ed_service_time = 240,
        min_ed_service_time = 30,  

        mu_medical_service_time = 4.5,
        sigma_medical_service_time = 0.68,

        max_medical_service_time = 240,
        min_medical_service_time = 30, 

        mean_consultant_assessment_time = 25,
        stdev_consultant_assessment_time = 10, 
       
        initial_medicine_discharge_prob = 0.1,
        consultant_discharge_prob = 0.3,
        
        simulation_time = 8880,
        burn_in_time = 1440) # burn in to prevent initiation bias
        
    trial = Trial(global_params)
    total_runs =5
    trial.run(total_runs)