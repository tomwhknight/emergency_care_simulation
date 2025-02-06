# run.py

from src.global_parameters import GlobalParameters
from src.trial import Trial

if __name__ == "__main__":
    global_params = GlobalParameters(

        ambulance_proportion = 20,
        walk_in_proportion = 80,

        # Arrival rates

        ed_peak_mean_patient_arrival_time = 3.2,
        ed_off_peak_mean_patient_arrival_time = 6.4,
        
        # Patient characterstics 
        
        ambulance_acuity_probabilities = {
        "1": 0.02,    
        "2": 0.40,  
        "3": 0.50,     
        "4": 0.05,
        "5": 0.01,
        },  

        walk_in_acuity_probabilities = {
        "1": 0.05,    
        "2": 0.20,  
        "3": 0.40,     
        "4": 0.30,
        "5": 0.05,
        },  

        # Staffing resource

        ambulance_triage_nurse_capacity = 1,
        walk_in_triage_nurse_capacity = 2,
        ed_doctor_capacity = 37,
        medical_doctor_capacity = 5,
        consultant_capacity = 1, 

        # Bed resource

        num_ambulance_triage_bays = 5,
        num_triage_rooms = 2, 
        num_corridor_spaces = 15,
        num_utc_rooms = 12, 
        num_ed_majors_beds = 24, 

        # SDEC capacity

        sdec_open_hour = 8, 
        sdec_close_hour = 18,

        weekday_sdec_base_capacity = 7,
        weekend_sdec_base_capacity = 5, 

        max_amu_available_beds = 10,
        max_sdec_capacity = 30,

        mean_triage_assessment_time = 12.0,
        stdev_triage_assessment_time = 7,
        
        mean_ed_assessment_time= 30,
        stdev_ed_assessment_time = 10, 
        
        mean_referral_time = 60,
        stdev_referral_time = 20,

        mean_initial_medical_assessment_time = 60,
        
        mean_consultant_assessment_time = 25,
        stdev_consultant_assessment_time = 10, 

        mean_sdec_assessment_time = 480,
        stdev_sdec_assessment_time = 60,

        ed_discharge_rate = 0.70,
        medicine_discharge_rate = 0.10,
        utc_discharge_prob = 0.9, 

        mean_amu_bed_release_interval = 30,
        mean_sdec_capacity_release_interval = 30,
        
        simulation_time= 1440,
        burn_in_time = 0) # burn in to prevent initiation bias
        
    trial = Trial(global_params)
    total_runs = 2
    trial.run(total_runs)