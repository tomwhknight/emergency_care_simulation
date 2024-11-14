# run.py

from src.global_parameters import GlobalParameters
from src.trial import Trial

if __name__ == "__main__":
    global_params = GlobalParameters(
        mean_patient_arrival_time= 5,
        triage_nurse_capacity = 2,
        ed_doctor_capacity = 5,
        medical_doctor_capacity = 4,
        consultant_capacity = 2, 

        mean_triage_assessment_time = 15,
        stdev_triage_assessment_time = 7,
        
        mean_ed_assessment_time= 60,
        stdev_ed_assessment_time = 25, 
        
        mean_medical_referral = 60,
        mean_initial_medical_assessment_time = 60,
        mean_consultant_assessment_time = 20, 
        admission_probability=0.25,
        simulation_time= 1050,
        burn_in_time = 0, # burn in to prevent initiation bias
        amu_bed_rate=0.1)

    trial = Trial(global_params)
    total_runs = 2
    trial.run(total_runs)