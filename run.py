# run.py

from src.global_parameters import GlobalParameters
from src.trial import Trial

if __name__ == "__main__":
    global_params = GlobalParameters(
        ed_peak_mean_patient_arrival_time = 3.2, 
        ed_off_peak_mean_patient_arrival_time= 9.6,
        triage_nurse_capacity = 2,
        ed_doctor_capacity = 10,
        medical_doctor_capacity = 1,
        consultant_capacity = 1, 

        mean_triage_assessment_time = 7,
        stdev_triage_assessment_time = 2,
        
        mean_ed_assessment_time= 30,
        stdev_ed_assessment_time = 10, 
        
        mean_referral_time = 60,
        stdev_referral_time = 20,

        mean_initial_medical_assessment_time = 60,
        
        mean_consultant_assessment_time = 25,
        stdev_consultant_assessment_time = 10, 

        ed_discharge_rate = 0.05,
        medicine_discharge_rate = 0.5,

        mean_amu_bed_release_interval = 30,
        simulation_time= 1440,
        burn_in_time = 0) # burn in to prevent initiation bias
        
    trial = Trial(global_params)
    total_runs = 1
    trial.run(total_runs)