import simpy
import pandas as pd
import random  # Ensure random is imported
from .model import Model  # Import the Model class

class Trial:
    """Runs the simulation trial."""
    
    def __init__(self, global_params):
        self.global_params = global_params
        self.patient_data = []  # Initialize the list to store patient data

    def run(self, run_number):
        """Run the trial simulation."""
        print('Starting the simulation...')
        
        random.seed(self.global_params.random_seed)  # Optional: for reproducibility
        
        # Create the SimPy environment
        env = simpy.Environment()
        
        # Instantiate the model with the environment and global parameters
        model = Model(env, self.global_params)

        # Start the patient generation process
        env.process(model.patient_generator())
        
        # Run the simulation for the specified duration
        env.run(until=self.global_params.simulation_time)

        # Collect patient data after the simulation
        for patient in model.patients:
            arrival_hour = (patient.arrival_time // 60) % 24
            arrival_minute = patient.arrival_time % 60
            arrival_time_formatted = f"{int(arrival_hour):02d}:{int(arrival_minute):02d}"

            # Print statement for debugging
            if patient.triage_completion_time is not None:
                time_to_triage = patient.triage_completion_time - patient.arrival_time
                print(f'Patient {patient.id} arrived at {arrival_time_formatted}. Assigned pathway: {patient.pathway}. Time to triage: {time_to_triage:.2f} minutes.')
            else:
                print(f'Patient {patient.id} arrived at {arrival_time_formatted}. Assigned pathway: {patient.pathway}. Triage time not completed.')

            # Append patient data to the list, including time to triage completion
            self.patient_data.append({
                'ID': patient.id,
                'Arrival Time': arrival_time_formatted,
                'Triage Time': patient.triage_time,  # Time spent in triage
                'Pathway': patient.pathway,
                'Time to Triage Completion': patient.triage_completion_time - patient.arrival_time if patient.triage_completion_time is not None else None,
                'Run Number': run_number
            })

        # Create DataFrame from the collected patient data
        df = pd.DataFrame(self.patient_data)
        return df  # Return the DataFrame for this trial
