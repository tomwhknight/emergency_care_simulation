import simpy
import random
from .patient import Patient  # Import Patient class

def get_random_inter_arrival_time(current_time):
    """Determine inter-arrival time based on the current simulation time."""
    if 0 <= current_time < 6:  # Early morning
        average_rate = 10  # Average 10 minutes between arrivals
    elif 6 <= current_time < 12:  # Morning
        average_rate = 5   # Average 5 minutes between arrivals
    elif 12 <= current_time < 18:  # Afternoon
        average_rate = 3   # Average 3 minutes between arrivals
    else:  # Evening
        average_rate = 7   # Average 7 minutes between arrivals

    return random.expovariate(1 / average_rate)

def get_triage_nurse_capacity(current_time):
    """Determine triage nurse capacity based on the current simulation time."""
    if 0 <= current_time < 6:  # Early morning
        return 1  # Fewer nurses available
    elif 6 <= current_time < 12:  # Morning
        return 3  # More nurses available
    elif 12 <= current_time < 18:  # Afternoon
        return 2  # Average number of nurses
    else:  # Evening
        return 1  # Fewer nurses available

class Model:
    """Represents the emergency care model with triage and pathways."""
    
    def __init__(self, env, global_params):
        self.env = env
        self.global_params = global_params
        self.patients = []  # List to hold all patients

    def patient_generator(self):
        """Generate patients at random intervals based on demand variation."""
        while True:
            current_simulation_time = self.env.now
            current_time = (current_simulation_time // 60) % 24  # Convert to 24-hour format
            
            # Determine inter-arrival time using stochastic demand
            inter_arrival_time = get_random_inter_arrival_time(current_time)

            yield self.env.timeout(inter_arrival_time)  # Wait for the next patient
            arrival_time = self.env.now
            
            patient = Patient(arrival_time)  # Create a new patient
            self.patients.append(patient)  # Store patient
            
            yield from self.triage(patient)

    def triage(self, patient):
        """Simulate the triage process."""
        current_time = (self.env.now // 60) % 24  # Current time in hours
        num_nurses = get_triage_nurse_capacity(current_time)  # Get current capacity
        triage_nurse_resource = simpy.Resource(self.env, capacity=num_nurses)  # Create resource
        
        with triage_nurse_resource.request() as request:  # Using triage nurse resource
            yield request  # Request the triage nurse resource
            
            # Generate a random triage time using an exponential distribution
            patient.triage_time = random.expovariate(1 / self.global_params.triage_time)  # Average triage time from parameters
            yield self.env.timeout(patient.triage_time)  # Simulate the triage time
            
            # Record the completion time
            patient.triage_completion_time = self.env.now  # Set the triage completion time

            # Calculate the time from arrival to triage completion
            time_to_triage_completion = patient.triage_completion_time - patient.arrival_time
            
            # Log the current time for debugging
            print(f'Patient {patient.id} arrived at {patient.arrival_time}. Triage completed at {patient.triage_completion_time}. Time to triage: {time_to_triage_completion:.2f} minutes.')

            # Determine if patient can go to SDEC based on current time
            if 8 <= current_time < 18:  # Only allow SDEC from 08:00 to 18:00
                if random.choice(['standard', 'sdec']) == 'sdec':
                    patient.pathway = 'SDEC'  # Assign to SDEC
                    yield from self.sdec(patient)
                else:
                    patient.pathway = 'Standard'  # Assign to Standard
                    yield from self.standard_pathway(patient)
            else:
                # If it's outside the SDEC hours, assign to standard pathway
                patient.pathway = 'Standard'
                yield from self.standard_pathway(patient)

    def sdec(self, patient):
        """Simulate the SDEC pathway."""
        sdec_time = random.expovariate(1 / self.global_params.triage_time)  # Example treatment time for SDEC
        yield self.env.timeout(sdec_time)  # Simulate the time taken in the SDEC pathway

    def standard_pathway(self, patient):
        """Simulate the standard pathway."""
        treatment_time = random.expovariate(1 / self.global_params.triage_time)  # Example treatment time for standard pathway
        yield self.env.timeout(treatment_time)  # Simulate the time taken in the standard pathway
