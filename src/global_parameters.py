# global_parameters.py

class GlobalParameters:
    """Contains constants and parameters used across the simulation (e.g., arrival rates, service times)."""
    def __init__(self, mean_patient_arrival_time, 
                 triage_nurse_capacity, 
                 ed_doctor_capacity, 
                 medical_doctor_capacity, 
                 consultant_capacity, 
                 mean_triage_assessment_time, 
                 mean_ed_assessment_time, 
                 mean_medical_referral, 
                 mean_initial_medical_assessment_time, 
                 mean_consultant_assessment_time, 
                 admission_probability, 
                 simulation_time, 
                 amu_bed_rate):
        
        # Define inter-arrival time

        self.mean_patient_arrival_time = mean_patient_arrival_time
        
         # Define the resources

        self.triage_nurse_capacity = triage_nurse_capacity
        self.ed_doctor_capacity = ed_doctor_capacity
        self.medical_doctor_capacity = medical_doctor_capacity
        self.consultant_capacity = consultant_capacity
    
        # mean assessment times
        
        self.mean_triage_assessment_time = mean_triage_assessment_time 
        self.ed_doctor_capacity = ed_doctor_capacity
        self.mean_ed_assessment_time = mean_ed_assessment_time
        self.mean_medical_referral = mean_medical_referral
        self.mean_initial_medical_assessment_time = mean_initial_medical_assessment_time
        self.mean_consultant_assessment_time = mean_consultant_assessment_time
        self.admission_probability = admission_probability
        self.simulation_time = simulation_time
        self.amu_bed_rate = amu_bed_rate