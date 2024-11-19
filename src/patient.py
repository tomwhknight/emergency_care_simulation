class Patient:
    """Class representing a patient in the system."""
    def __init__(self, patient_id, arrival_time, day_of_arrival, arrival_clock_time, current_hour, priority = 1):
        self.id = patient_id
        self.arrival_time = arrival_time
        self.arrival_clock_time = arrival_clock_time
        self.day_of_arrival = day_of_arrival
        self.current_hour = current_hour
        self.priority = priority
        

        # Triage attributes 
        self.wait_time_for_triage = 0.0  # Initialize wait time as 0
        self.time_at_end_of_triage = 0.0  # Initialize end of triage time as 0
        self.triage_assessment_time = 0.0  # Initialize triage assessment time
        self.disposition = None  # Will be 'admit' or 'discharge'

        # Initialize AMU-related attributes
        self.amu_admission_time = None  # Will be set when the patient is admitted to AMU
        self.joined_amu_queue_time = None  # Will be set when the patient joins the AMU queue