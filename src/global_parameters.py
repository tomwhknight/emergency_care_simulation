# global_parameters.py

import os
class GlobalParameters:
    """Contains constants and parameters used across the simulation (e.g., arrival rates, service times)."""
    def __init__(self, 
                             ambulance_proportion=20,
                 walk_in_proportion=80,
                 proportion_direct_primary_care=0.03,
                 ambulance_acuity_probabilities=None,
                 walk_in_acuity_probabilities=None,
                 medical_referral_rate=0.175,
                 speciality_referral_rate=0.150,
                 sdec_appropriate_rate=0.10,
                 ambulance_triage_nurse_capacity=1,
                 walk_in_triage_nurse_capacity=2,
                 ed_doctor_capacity=24,
                 medical_doctor_capacity=5,
                 consultant_capacity=1,
                 sdec_open_hour=7,
                 sdec_close_hour=16,
                 weekday_sdec_base_capacity=4,
                 weekend_sdec_base_capacity=4,
                 max_sdec_capacity=10,
                 max_amu_available_beds=10,
                 mean_triage_assessment_time=5,
                 stdev_triage_assessment_time=2,
                 mu_ed_assessment_discharge=4.15,
                 sigma_ed_assessment_discharge=1.0,
                 wb_shape_ed_assessment_admit=1.6,
                 wb_scale_ed_assessment_admit=1/0.01,
                 mu_ed_service_time=4.4,
                 sigma_ed_service_time=0.5,
                 max_ed_service_time=480,
                 min_ed_service_time=20,
                 mu_medical_service_time=4.5,
                 sigma_medical_service_time=0.68,
                 max_medical_service_time=240,
                 min_medical_service_time=30,
                 initial_medicine_discharge_prob=0.10,
                 consultant_discharge_prob=0.4,
                 mean_consultant_assessment_time=30,
                 stdev_consultant_assessment_time=10,
                 burn_in_time=1440,
                 simulation_time=11520):

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

        # Outcome thresholds

        self.sdec_appropriate_rate = sdec_appropriate_rate
        self.medical_referral_rate = medical_referral_rate
        self.speciality_referral_rate = speciality_referral_rate

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

        self.mu_ed_assessment_discharge = mu_ed_assessment_discharge
        self.sigma_ed_assessment_discharge = sigma_ed_assessment_discharge
        self.max_ed_service_time = max_ed_service_time
        self.min_ed_service_time = min_ed_service_time


        self.wb_shape_ed_assessment_admit = wb_shape_ed_assessment_admit
        self.wb_scale_ed_assessment_admit = wb_scale_ed_assessment_admit

        self.mu_ed_service_time = mu_ed_service_time
        self.sigma_ed_service_time = sigma_ed_service_time
     
        self.mu_medical_service_time =  mu_medical_service_time
        self.sigma_medical_service_time =  sigma_medical_service_time
        self.max_medical_service_time = max_medical_service_time
        self.min_medical_service_time = min_medical_service_time 

        self.mean_consultant_assessment_time = mean_consultant_assessment_time
        self.stdev_consultant_assessment_time = stdev_consultant_assessment_time 

        # Sim duration
        self.burn_in_time = burn_in_time
        self.simulation_time = simulation_time