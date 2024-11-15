# global_parameters.py

class GlobalParameters:
    """Contains constants and parameters used across the simulation (e.g., arrival rates, service times)."""
    def __init__(self, mean_patient_arrival_time, 
                 triage_nurse_capacity, 
                 ed_doctor_capacity, 
                 medical_doctor_capacity, 
                 consultant_capacity, 
                 mean_triage_assessment_time, 
                 stdev_triage_assessment_time,
                 mean_ed_assessment_time, 
                 stdev_ed_assessment_time,
                 mean_medical_referral,
                 mean_initial_medical_assessment_time, 
                 mean_consultant_assessment_time, 
                 admission_probability,
                 amu_bed_capacity,
                 initial_amu_beds,
                 amu_bed_generation_rate,     
                 burn_in_time,
                 simulation_time):
        
        # Define inter-arrival time

        self.mean_patient_arrival_time = mean_patient_arrival_time
        
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

        self.mean_medical_referral = mean_medical_referral
        self.mean_initial_medical_assessment_time = mean_initial_medical_assessment_time
        self.mean_consultant_assessment_time = mean_consultant_assessment_time
        
        # amu capacity

        self.amu_bed_capacity = amu_bed_capacity  # Total AMU bed capacity
        self.initial_amu_beds = initial_amu_beds  # Initially available AMU beds
        self.amu_bed_generation_rate = amu_bed_generation_rate
        
        # Additional params
        self.admission_probability = admission_probability
        
        # sim duration

        self.burn_in_time = burn_in_time
        self.simulation_time = simulation_time