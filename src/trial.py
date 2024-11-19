import pandas as pd
import os
from src.model import Model

class Trial:
    def __init__(self, global_params):
        """Initialize the trial with global parameters."""
        self.global_params = global_params
        
        # Initalise empty DataFrames for aggregated results
        self.agg_results_df = pd.DataFrame() 
        self.agg_triage_queue_monitoring_df = pd.DataFrame(columns=["Simulation Time", "Hour of Day", "Queue Length"])
        self.agg_consultant_queue_monitoring_df = pd.DataFrame(columns=["Simulation Time", "Hour of Day", "Queue Length"])
        self.agg_amu_queue_df = pd.DataFrame()  # Initialize an empty DataFrame for AMU queue monitoring results

    def run(self, run_number):
        """Run the trial for the specified number of runs."""
        burn_in_time = self.global_params.burn_in_time

        for i in range(run_number):
            print(f"Starting simulation run {i + 1} with a burn-in period of {burn_in_time}")

            # Initialize the model for each run
            model = Model(self.global_params, burn_in_time, run_number=i+1)  # Create a new instance of the Model class with run_number
           
            # Run the model
            model.run()

            # Reset the index in model's run_results_df to make Patient ID a column
            model.run_results_df_reset = model.run_results_df.reset_index()

            # Add the 'Run Number' column to track which run the result came from
            model.run_results_df_reset["Run Number"] = i + 1

            # Concatenate the results of each run to the global results DataFrame
            self.agg_results_df = pd.concat([self.agg_results_df, model.run_results_df_reset], ignore_index=True)
    
            # Add the 'Run Number' column to the queue monitoring DataFrame
            model.triage_queue_monitoring_df["Run Number"] = i + 1

            # Concatenate queue monitoring data for this run to the aggregated DataFrame
            self.agg_triage_queue_monitoring_df = pd.concat([self.agg_triage_queue_monitoring_df, model.triage_queue_monitoring_df], ignore_index=True)


            # Add the 'Run Number' column to the consultant monitoring DataFrame
            model.consultant_queue_monitoring_df["Run Number"] = i + 1

            # Concatenate queue monitoring data for this run to the aggregated DataFrame
            self.agg_consultant_queue_monitoring_df = pd.concat([self.agg_consultant_queue_monitoring_df, model.consultant_queue_monitoring_df], ignore_index=True)

            # Concatenate the AMU queue results of each run to the global results DataFrame for AMU queue data
            self.agg_amu_queue_df = pd.concat([self.agg_amu_queue_df, model.amu_queue_df], ignore_index=True)

        # Move 'Run Number' to the first column for cleaner presentation
        cols = ["Run Number"] + [col for col in self.agg_results_df.columns if col != "Run Number"]
        self.agg_results_df = self.agg_results_df[cols]
        # Ensure the directory 'results' exists
        if not os.path.exists('results'):
            os.makedirs('results')

        # For queue monitoring results
        queue_cols = ["Run Number"] + [col for col in self.agg_triage_queue_monitoring_df.columns if col != "Run Number"]
        self.agg_triage_queue_monitoring_df = self.agg_triage_queue_monitoring_df[queue_cols]

        # Save patient-level results
        patient_result_path = os.path.join('data', 'results', 'results.csv')
        self.agg_results_df.to_csv(patient_result_path, index=False)
        print(f"Patient results saved to {patient_result_path}")

        # Save queue monitoring results
        triage_queue_result_path = os.path.join('data', 'results', 'triage_queue_monitoring_results.csv')
        self.agg_triage_queue_monitoring_df.to_csv(triage_queue_result_path, index=False)
        print(f"Queue monitoring results saved to {triage_queue_result_path}")

        # Save queue monitoring results
        consultant_queue_result_path = os.path.join('data', 'results', 'consultant_queue_monitoring_results.csv')
        self.agg_consultant_queue_monitoring_df.to_csv(consultant_queue_result_path, index=False)
        print(f"Queue monitoring results saved to {consultant_queue_result_path}")

         # Save queue monitoring results (AMU queue data)
        amu_queue_result_path = os.path.join('data', 'results', 'amu_queue_monitoring.csv')
        self.agg_amu_queue_df.to_csv(amu_queue_result_path, index=False)
        print(f"Queue monitoring results saved to {amu_queue_result_path}")

        
        return self.agg_results_df, self.agg_triage_queue_monitoring_df, self.agg_amu_queue_df