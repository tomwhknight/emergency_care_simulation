class Patient:
    """Class representing a patient in the system."""
    def __init__(self, patient_id, arrival_time, day_of_arrival, arrival_clock_time, current_hour, source_of_referral, mode_arrival, priority = 1):
        self.id = patient_id
        self.arrival_time = arrival_time
        self.arrival_clock_time = arrival_clock_time
        self.day_of_arrival = day_of_arrival
        self.current_hour = current_hour
        
        # Patient characteristics 
        self.priority = priority # Used to block consultant activity
        self.mode_arrival = mode_arrival
        self.source_of_referral = source_of_referral

       
        # Triage attributes 
        self.wait_time_for_triage_space = 0.0  # Initialize wait time as 0
        self.wait_time_for_triage_nurse = 0.0
        self.triage_location = None 

        self.time_at_end_of_triage = 0.0  # Initialize end of triage time as 0
        self.triage_nurse_assessment_time = 0.0  # Initialize triage assessment time
        self.triage_outcome = None

        self.referral_to_medicine_time = 0.0
        self.wait_time_for_utc_room = 0.0,
        self.ed_assessment_time = 0.0

        # Track outcome
        self.sdec_assessment_time = 0.0
        self.transferred_to_amu = False
        self.transferred_to_majors = False
        
        # Record disporition 
        self.ed_disposition = None  # Will be 'Admit - Medicine', 'Admit - Other' or 'Discharge'
        self.discharge = None

        # Binary metrics initialized to 0 (not exceeding thresholds yet)
        self.ed_4hrs_after_arrival = 0
        self.ed_4hrs_after_referral = 0
        self.ed_12hrs_after_arrival = 0
        self.ed_12hrs_after_referral = 0

        # Initialize AMU-related attributes
        self.amu_admission_time = None  # Will be set when the patient is admitted to AMU
        self.joined_amu_queue_time = None  # Will be set when the patient joins the AMU queue