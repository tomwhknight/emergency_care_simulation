# global_parameters.py

class GlobalParameters:
    """Contains constants and parameters used across the simulation (e.g., arrival rates, service times)."""
    def __init__(self, mean_patient_arrival_time, mean_assessment_time, admission_probability, simulation_time, amu_bed_rate):
        self.mean_patient_arrival_time = mean_patient_arrival_time
        self.mean_assessment_time = mean_assessment_time
        self.admission_probability = admission_probability
        self.simulation_time = simulation_time
        self.amu_bed_rate = amu_bed_rate