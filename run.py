from src.global_parameters import GlobalParameters  # Import GlobalParameters class
from src.trial import Trial  # Import Trial class
import pandas as pd
import os

if __name__ == "__main__":
    # Set global parameters
    global_params = GlobalParameters(
        inter_arrival_time=5,       # Average time between patient arrivals (not used in the updated model)
        triage_time=10,              # Average triage time
        sdec_capacity=2,            # Fixed capacity for SDEC pathway
        simulation_time=10080,      # Total simulation time in minutes (7 days)
        triage_nurse_capacity=2,    # Set the number of triage nurses (initial capacity)
        random_seed=42               # Set the random seed for reproducibility
    )

    # Number of runs for the simulation
    num_runs = 5  # Change this to the desired number of runs

    # Store results DataFrame
    all_patient_data = []

    for run_number in range(1, num_runs + 1):
        print(f"Starting simulation run {run_number} of {num_runs}...")
        
        # Run the trial and pass the run_number
        trial = Trial(global_params)
        trial_data = trial.run(run_number)  # Get the DataFrame from this trial
        all_patient_data.append(trial_data)  # Append the DataFrame

    # Concatenate all patient data into a single DataFrame
    final_df = pd.concat(all_patient_data, ignore_index=True)
    
    # Create the 'results' directory if it doesn't exist
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Output the DataFrame as a CSV file in the 'results' folder
    final_df.to_csv(os.path.join(results_dir, 'patient_data_simulation.csv'), index=False)  # Save to CSV without the index
