# global_parameters.py

class GlobalParameters:
    """Contains constants and parameters used across the simulation (e.g., arrival rates, service times)."""
    def __init__(self, 
                 ed_peak_mean_patient_arrival_time, 
                 ed_off_peak_mean_patient_arrival_time, 
                 triage_nurse_capacity, 
                 ed_doctor_capacity, 
                 medical_doctor_capacity, 
                 consultant_capacity, 
                 mean_triage_assessment_time, 
                 stdev_triage_assessment_time,
                 mean_ed_assessment_time, 
                 stdev_ed_assessment_time,
                 mean_referral_time,
                 stdev_referral_time,
                 mean_initial_medical_assessment_time, 
                 mean_consultant_assessment_time,
                 stdev_consultant_assessment_time, 
                 unav_freq_consultant,
                 unav_time_consultant,
                 admission_probability,
                 mean_amu_bed_release_interval,    
                 burn_in_time,
                 simulation_time):
        
        # Define inter-arrival time

        self.ed_peak_mean_patient_arrival_time = ed_peak_mean_patient_arrival_time # 09:00-21:00
        self.ed_off_peak_mean_patient_arrival_time = ed_off_peak_mean_patient_arrival_time

         # Define the resources

        self.triage_nurse_capacity = triage_nurse_capacity
        self.ed_doctor_capacity = ed_doctor_capacity
        self.medical_doctor_capacity = medical_doctor_capacity
        self.consultant_capacity = consultant_capacity
    
        # assessment times
        
        self.mean_triage_assessment_time = mean_triage_assessment_time 
        self.stdev_triage_assessment_time = stdev_triage_assessment_time

        self.mean_ed_assessment_time = mean_ed_assessment_time
        self.stdev_ed_assessment_time = stdev_ed_assessment_time

        self.mean_referral_time = mean_referral_time
        self.stdev_referral_time = stdev_referral_time

        self.mean_initial_medical_assessment_time = mean_initial_medical_assessment_time
        self.mean_consultant_assessment_time = mean_consultant_assessment_time
        
        self.mean_consultant_assessment_time = mean_consultant_assessment_time
        self.stdev_consultant_assessment_time = stdev_consultant_assessment_time 

        self.unav_freq_consultant = unav_freq_consultant
        self.unav_time_consultant = unav_time_consultant


        # AMU bed release 
        self.mean_amu_bed_release_interval = mean_amu_bed_release_interval
        


        # Additional params
        self.admission_probability = admission_probability
        
        # sim duration

        self.burn_in_time = burn_in_time
        self.simulation_time = simulation_time