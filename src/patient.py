class Patient:
    """Class representing a patient in the system."""
    def __init__(self, env, patient_id, arrival_time, current_day, clock_hour, current_hour,
                 source_of_referral, mode_arrival, age, adult, news2, referral_prob_cal, referral_score_raw, acuity, 
                 sdec_appropriate, ed_disposition,
                 priority=1):
        
        # Store environment for SimPy events
        self.env = env
        
        # Initialise standard attributes
        self.id = patient_id
        self.arrival_time = arrival_time
        self.current_day = current_day
        self.clock_hour = clock_hour
        self.current_hour = current_hour
        self.source_of_referral = source_of_referral
        self.mode_arrival = mode_arrival
        self.age = age
        self.adult = adult
        self.news2 = news2
        self.referral_prob_cal = referral_prob_cal
        self.referral_score_raw  = referral_score_raw
        self.acuity = acuity
        self.sdec_appropriate = sdec_appropriate
        self.ed_disposition = ed_disposition
        self.priority = priority
        self.discharge_event = env.event()

        self.dt_eligible = False         # Will be True if patient eligible for Direct Triage
        self.pathway_start = "ED"        # Either "ED" or "Direct-Medicine"

       
        # Initialize AMU-related attributes
        self.discharged = False
        self.amu_admission_time = None 
        self.joined_amu_queue_time = None  