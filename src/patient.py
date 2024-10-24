class Patient:
    """Class representing a patient in the system."""
    def __init__(self, patient_id, arrival_time, hour_of_arrival, day_of_arrival):
        self.id = patient_id
        self.arrival_time = arrival_time
        self.disposition = None  # Will be 'admit' or 'discharge'
        self.hour_of_arrival = hour_of_arrival
        self.day_of_arrival = day_of_arrival