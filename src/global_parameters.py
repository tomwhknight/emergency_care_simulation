# global_parameters.py

import os
class GlobalParameters:
    """Contains constants and parameters used across the simulation (e.g., arrival rates, service times)."""
    def __init__(self, 
                 ambulance_proportion,
                 walk_in_proportion,
                 proportion_direct_primary_care,
                 ambulance_acuity_probabilities, 
                 walk_in_acuity_probabilities,
                 sdec_threshold,

                 ambulance_triage_nurse_capacity,
                 walk_in_triage_nurse_capacity,  
                 ed_doctor_capacity, 
                 medical_doctor_capacity, 
                 consultant_capacity, 
                 
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
                 
                 mean_initial_medical_assessment_time,
                 stdev_initial_medical_assessment_time, 

                 mu_ed_delay_time_discharge,
                 sigma_ed_delay_time_discharge,

                 mu_ed_delay_time_admission, 
                 sigma_ed_delay_time_admission, 

                 ed_discharge_prob,
                 ed_medicine_referral_prob,
                 ed_other_specialty_prob,

                 initial_medicine_discharge_prob,
                 consultant_discharge_prob,
          
                 mean_consultant_assessment_time,
                 stdev_consultant_assessment_time, 
                 
                 burn_in_time,
                 simulation_time):

         # Determine the project root directory (parent of src)
        self.project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
         # Define the file path for generators
        self.arrival_rate_file = os.path.join(self.project_dir, "data/generator_distributions/arrival_rate.csv")
        self.amu_bed_rate_file = os.path.join(self.project_dir, "data/generator_distributions/amu_bed_rate.csv")
        self.sdec_slot_rate_file = os.path.join(self.project_dir, "data/generator_distributions/sdec_slot_rate.csv")    
    
        # Define the file path for staffing resources
        self.ed_staffing_file = os.path.join(self.project_dir, "data/staffing_resource/ed_staffing.csv")
        self.medicine_staffing_file = os.path.join(self.project_dir, "data/staffing_resource/medicine_staffing.csv")
        
        # Define the file path for patient attributes
        self.news2_file = os.path.join(self.project_dir, "data/patient_attributes/news2_distribution.csv")
        self.admission_prob_file = os.path.join(self.project_dir, "data/patient_attributes/admission_prob_distribution.csv")

        # Mode of arrival
        self.ambulance_proportion = ambulance_proportion
        self.walk_in_proportion = walk_in_proportion
   
        # Patient characterietics
        self.ambulance_acuity_probabilities = ambulance_acuity_probabilities
        self.walk_in_acuity_probabilities = walk_in_acuity_probabilities
        
        # Source of referral 
        self.proportion_direct_primary_care = proportion_direct_primary_care

        self.sdec_threshold = sdec_threshold

        # ED disposition probabilities
        self.ed_discharge_prob = ed_discharge_prob
        self.ed_medicine_referral_prob = ed_medicine_referral_prob
        self.ed_other_specialty_prob = ed_other_specialty_prob

        # Medicine discharge probabilities

        self.initial_medicine_discharge_prob = initial_medicine_discharge_prob
        self.consultant_discharge_prob = consultant_discharge_prob
      
        # Define the resources

        self.ambulance_triage_nurse_capacity = ambulance_triage_nurse_capacity
        self.walk_in_triage_nurse_capacity = walk_in_triage_nurse_capacity
        self.ed_doctor_capacity = ed_doctor_capacity
        self.medical_doctor_capacity = medical_doctor_capacity
        self.consultant_capacity = consultant_capacity

        # Define store values 
        self.max_amu_available_beds = max_amu_available_beds # refers max available to transfer at any point (not total AMU capacity)
        self.max_sdec_capacity = max_sdec_capacity # refers max available to transfer at any point (not total SDEC capacity)
      
        self.sdec_open_hour = sdec_open_hour   # SDEC opens at 08:00
        self.sdec_close_hour = sdec_close_hour # SDEC stops accepting referrals at 18:00

        self.weekday_sdec_base_capacity = weekday_sdec_base_capacity
        self.weekend_sdec_base_capacity = weekend_sdec_base_capacity

        # Assessment times   
        self.mean_triage_assessment_time = mean_triage_assessment_time 
        self.stdev_triage_assessment_time = stdev_triage_assessment_time

        self.mean_ed_assessment_time = mean_ed_assessment_time
        self.stdev_ed_assessment_time = stdev_ed_assessment_time

        self.mu_ed_delay_time_discharge = mu_ed_delay_time_discharge
        self.sigma_ed_delay_time_discharge = sigma_ed_delay_time_discharge

        self.mu_ed_delay_time_admission = mu_ed_delay_time_admission
        self.sigma_ed_delay_time_admission = sigma_ed_delay_time_admission

        self.mean_initial_medical_assessment_time = mean_initial_medical_assessment_time
        self.stdev_initial_medical_assessment_time = stdev_initial_medical_assessment_time
        
        self.mean_consultant_assessment_time = mean_consultant_assessment_time
        self.stdev_consultant_assessment_time = stdev_consultant_assessment_time 

        # Sim duration
        self.burn_in_time = burn_in_time
        self.simulation_time = simulation_time