# run.py

from src.global_parameters import GlobalParameters
from src.trial import Trial

if __name__ == "__main__":
    global_params = GlobalParameters(
        ed_peak_mean_patient_arrival_time = 9.6, 
        ed_off_peak_mean_patient_arrival_time= 3.2,
        triage_nurse_capacity = 2,
        ed_doctor_capacity = 5,
        medical_doctor_capacity = 4,
        consultant_capacity = 2, 

        mean_triage_assessment_time = 7,
        stdev_triage_assessment_time = 2,
        
        mean_ed_assessment_time= 60,
        stdev_ed_assessment_time = 25, 
        
        mean_medical_referral = 60,
        mean_initial_medical_assessment_time = 60,
        mean_consultant_assessment_time = 20, 
        admission_probability=0.25,
        
        mean_amu_bed_release_interval = 60,
        simulation_time= 2880,
        burn_in_time = 1440) # burn in to prevent initiation bias
        

    trial = Trial(global_params)
    total_runs = 2
    trial.run(total_runs)