# run.py

from src.global_parameters import GlobalParameters
from src.trial import Trial

if __name__ == "__main__":
    global_params = GlobalParameters(

        ambulance_proportion = 20,
        walk_in_proportion = 80,

        # Source of referral
        proportion_direct_primary_care = 0.07,  
        
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
        walk_in_triage_nurse_capacity = 2,

        ed_doctor_capacity = 20,
        medical_doctor_capacity = 5,
        consultant_capacity = 1, 

        # SDEC capacity
        sdec_open_hour = 7, 
        sdec_close_hour = 16,

        weekday_sdec_base_capacity = 10,
        weekend_sdec_base_capacity = 5, 

        # AMU capacity
        max_amu_available_beds = 20,
        max_sdec_capacity = 30,

        # Service times
        mean_triage_assessment_time = 5,
        stdev_triage_assessment_time = 2,
        
        mean_ed_assessment_time = 60,
        stdev_ed_assessment_time = 30,

        mu_ed_delay_time_discharge = 4.5,
        sigma_ed_delay_time_discharge = 1.0,

        mu_ed_delay_time_admission = 4.6,
        sigma_ed_delay_time_admission = 1.2,

        ed_discharge_prob = 0.75,
        ed_medicine_referral_prob = 0.15,
        ed_other_specialty_prob = 0.10,

        mean_initial_medical_assessment_time = 60,
        stdev_initial_medical_assessment_time = 30, 

        mean_consultant_assessment_time = 25,
        stdev_consultant_assessment_time = 10, 

        mean_sdec_assessment_time = 480,
        stdev_sdec_assessment_time = 60,
       
        initial_medicine_discharge_prob = 0.1,
        consultant_discharge_prob = 0.3,
          
        mean_sdec_capacity_release_interval = 30,
    
        simulation_time = 10000,
        burn_in_time = 1440) # burn in to prevent initiation bias
        
    trial = Trial(global_params)
    total_runs = 2
    trial.run(total_runs)