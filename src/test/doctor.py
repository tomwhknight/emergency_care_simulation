class Doctor:
    def __init__(self, doctor_id, shift_name, start_time, end_time):
        self.doctor_id = doctor_id
        self.shift_name = shift_name
        self.start_time = start_time
        self.end_time = end_time
        self.patients_seen = 0  # Track how many patients this doctor has assessed
    
    def assess_patient(self):
        # Logic to assess a patient
        self.patients_seen += 1