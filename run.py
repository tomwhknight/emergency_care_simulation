# run.py

from src.global_parameters import GlobalParameters
from src.trial import Trial

if __name__ == "__main__":
    global_params = GlobalParameters(
        
        acuity_probabilities = {
        "low": 0.3,     # 30% chance
        "medium": 0.5,  # 50% chance
        "high": 0.2     # 20% chance
        },

        ed_peak_mean_patient_arrival_time = 3.2, 
        ed_off_peak_mean_patient_arrival_time= 9.6,
        
        triage_nurse_capacity = 2,
        ed_doctor_capacity = 37,
        medical_doctor_capacity = 5,
        consultant_capacity = 1, 

        sdec_open_hour = 8, 
        sdec_close_hour = 18,

        weekday_sdec_base_capacity = 7,
        weekend_sdec_base_capacity = 5, 

        max_amu_available_beds = 10,
        max_sdec_capacity = 30,

        mean_triage_assessment_time = 2.4,
        stdev_triage_assessment_time = 0.6,
        
        mean_ed_assessment_time= 30,
        stdev_ed_assessment_time = 10, 
        
        mean_referral_time = 60,
        stdev_referral_time = 20,

        mean_initial_medical_assessment_time = 60,
        
        mean_consultant_assessment_time = 25,
        stdev_consultant_assessment_time = 10, 

        mean_sdec_assessment_time = 480,
        stdev_sdec_assessment_time = 60,

        ed_discharge_rate = 0.05,
        medicine_discharge_rate = 0.5,

        mean_amu_bed_release_interval = 30,
        mean_sdec_capacity_release_interval = 30,
        
        simulation_time= 120,
        burn_in_time = 0) # burn in to prevent initiation bias
        
    trial = Trial(global_params)
    total_runs = 1
    trial.run(total_runs)