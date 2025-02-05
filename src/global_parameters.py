# global_parameters.py

import os
class GlobalParameters:
    """Contains constants and parameters used across the simulation (e.g., arrival rates, service times)."""
    def __init__(self, 
                 ambulance_acuity_probabilities, 
                 walk_in_acuity_probabilities,
                 ambulance_peak_mean_patient_arrival_time, 
                 ambulance_off_peak_mean_patient_arrival_time,
                 walk_in_peak_mean_patient_arrival_time,
                 walk_in_off_peak_mean_patient_arrival_time, 
                 ambulance_triage_nurse_capacity,
                 walk_in_triage_nurse_capacity,  
                 ed_doctor_capacity, 
                 medical_doctor_capacity, 
                 consultant_capacity, 
                 num_ambulance_triage_bays,
                 num_triage_rooms, 
                 num_corridor_spaces,
                 num_utc_rooms, 
                 num_ed_majors_beds,
                 mean_sdec_capacity_release_interval, 
                 sdec_open_hour, 
                 sdec_close_hour,
                 weekday_sdec_base_capacity, 
                 weekend_sdec_base_capacity, 
                 max_sdec_capacity, 
                 max_amu_available_beds,
                 mean_triage_assessment_time, 
                 stdev_triage_assessment_time,
                 mean_ed_assessment_time, 
                 stdev_ed_assessment_time,
                 mean_referral_time,
                 stdev_referral_time,
                 mean_initial_medical_assessment_time, 
                 mean_consultant_assessment_time,
                 stdev_consultant_assessment_time, 
                 mean_sdec_assessment_time,
                 stdev_sdec_assessment_time,
                 ed_discharge_rate,
                 utc_discharge_prob, 
                 medicine_discharge_rate,
                 mean_amu_bed_release_interval,    
                 burn_in_time,
                 simulation_time):
        
        # Define the base project directory

         # Determine the project root directory (parent of src)
        self.project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # Define the file path relative to the project directory
        self.ed_staffing_file = os.path.join(self.project_dir, "data/staffing_resource/ed_staffing.csv")
        self.medicine_staffing_file = os.path.join(self.project_dir, "data/staffing_resource/medicine_staffing.csv")
        
        # Patient characterietics
        self.ambulance_acuity_probabilities = ambulance_acuity_probabilities
        self.walk_in_acuity_probabilities = walk_in_acuity_probabilities

        # Define inter-arrival time
        self.ambulance_peak_mean_patient_arrival_time = ambulance_peak_mean_patient_arrival_time 
        self.ambulance_off_peak_mean_patient_arrival_time = ambulance_off_peak_mean_patient_arrival_time

        self.walk_in_peak_mean_patient_arrival_time = walk_in_peak_mean_patient_arrival_time
        self.walk_in_off_peak_mean_patient_arrival_time = walk_in_off_peak_mean_patient_arrival_time

        # Define the resources

        self.ambulance_triage_nurse_capacity = ambulance_triage_nurse_capacity
        self.walk_in_triage_nurse_capacity = walk_in_triage_nurse_capacity
        self.ed_doctor_capacity = ed_doctor_capacity
        self.medical_doctor_capacity = medical_doctor_capacity
        self.consultant_capacity = consultant_capacity

        # Triage capacity settings
        self.num_ambulance_triage_bays = num_ambulance_triage_bays   # Number of bays for triage
        self.num_corridor_spaces = num_corridor_spaces # Max patients in the corridor
        self.num_triage_rooms = num_triage_rooms
        self.num_utc_rooms = num_utc_rooms
        self.num_ed_majors_beds = num_ed_majors_beds 

        self.utc_discharge_prob = utc_discharge_prob

        # Define store values 
        self.max_amu_available_beds = max_amu_available_beds # refers max available to transfer at any point (not total AMU capacity)
        self.max_sdec_capacity = max_sdec_capacity # refers max available to transfer at any point (not total SDEC capacity)
        self.mean_sdec_capacity_release_interval = mean_sdec_capacity_release_interval # refers to rate at which sdec slots open to ED referrals in the day

        self.sdec_open_hour = sdec_open_hour   # SDEC opens at 08:00
        self.sdec_close_hour = sdec_close_hour # SDEC stops accepting referrals at 18:00

        self.weekday_sdec_base_capacity = weekday_sdec_base_capacity
        self.weekend_sdec_base_capacity = weekend_sdec_base_capacity

        # Assessment times   
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

        self.mean_sdec_assessment_time = mean_sdec_assessment_time
        self.stdev_sdec_assessment_time = stdev_sdec_assessment_time

        # AMU bed release 
        self.mean_amu_bed_release_interval = mean_amu_bed_release_interval
        self.ed_discharge_rate = ed_discharge_rate
        self.medicine_discharge_rate = medicine_discharge_rate

        # sim duration
        self.burn_in_time = burn_in_time
        self.simulation_time = simulation_time