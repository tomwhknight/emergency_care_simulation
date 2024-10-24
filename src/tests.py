
from src.global_parameters import GlobalParameters
from src.patient import Patient

# Initialize global parameters
global_params = GlobalParameters(
    inter_arrival_time=10,
    triage_time=5,
    sdec_capacity=3,
    simulation_time=1440,  # 1 day in minutes
    triage_nurse_capacity=2
)

# Print global parameters to verify
print("Global Parameters:", vars(global_params))

# Create a few patients
patients = [Patient(patient_id=i) for i in range(1, 4)]

# Print patient details to verify
for patient in patients:
    print("Patient:", vars(patient))