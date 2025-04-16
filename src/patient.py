class Patient:
    """Class representing a patient in the system."""
    def __init__(self, patient_id, arrival_time, current_day, clock_hour, current_hour, source_of_referral, mode_arrival, age, news2, admission_prob, acuity, priority = 1):
        self.id = patient_id
        self.arrival_time = arrival_time
        self.current_day = current_day 
        self.clock_hour = clock_hour
        self.current_hour = current_hour
        
        # Patient characteristics 
        self.priority = priority # Used to block consultant activity
        self.mode_arrival = mode_arrival
        self.source_of_referral = source_of_referral
        self.age = age
        self.news2 = news2
        self.admission_prob = admission_prob
        self.acuity = acuity 
        
        # Triage attributes 
        self.wait_time_for_triage_nurse = 0.0
        self.triage_location = None 

        self.time_at_end_of_triage = 0.0  # Initialize end of triage time as 0
        self.triage_nurse_assessment_time = 0.0  # Initialize triage assessment time

        self.referral_to_medicine_time = 0.0
        self.ed_assessment_time = 0.0

        # Track outcome
        self.sdec_assessment_time = 0.0
        self.transferred_to_amu = False
 
        
        # Record disporition 
        self.ed_disposition = None  # Will be 'Admit - Medicine', 'Admit - Other' or 'Discharge'
        self.discharged = False
       

        # Initialize AMU-related attributes
        self.amu_admission_time = None  # Will be set when the patient is admitted to AMU
        self.joined_amu_queue_time = None  # Will be set when the patient joins the AMU queue