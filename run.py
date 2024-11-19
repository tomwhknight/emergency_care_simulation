# run.py

from src.global_parameters import GlobalParameters
from src.trial import Trial

if __name__ == "__main__":
    global_params = GlobalParameters(
        ed_peak_mean_patient_arrival_time = 3.2, 
        ed_off_peak_mean_patient_arrival_time= 9.6,
        triage_nurse_capacity = 3,
        ed_doctor_capacity = 10,
        medical_doctor_capacity = 4,
        consultant_capacity = 1, 

        mean_triage_assessment_time = 7,
        stdev_triage_assessment_time = 2,
        
        mean_ed_assessment_time= 45,
        stdev_ed_assessment_time = 10, 
        
        mean_referral_time = 60,
        stdev_referral_time = 20,

        mean_initial_medical_assessment_time = 60,
        
        mean_consultant_assessment_time = 15,
        stdev_consultant_assessment_time = 10, 

        unav_freq_consultant = 1440,
        unav_time_consultant = 660,

        admission_probability=0.20,
        mean_amu_bed_release_interval = 60,
        simulation_time= 2880,
        burn_in_time = 0) # burn in to prevent initiation bias
        
    trial = Trial(global_params)
    total_runs = 2
    trial.run(total_runs)