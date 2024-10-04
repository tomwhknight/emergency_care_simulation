class Patient:
    """Represents a patient in the emergency care simulation."""
    
    id_counter = 0  # Class variable to track patient IDs
    
    def __init__(self, arrival_time):
        self.id = Patient.id_counter  # Unique identifier for each patient
        self.arrival_time = arrival_time  # Time of arrival
        self.triage_time = 0  # Time spent in triage
        self.pathway = None  # To store pathway: SDEC or Standard
        self.triage_completion_time = None  # Time when triage is completed
        Patient.id_counter += 1  # Increment the ID counter
