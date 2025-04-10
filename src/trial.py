import pandas as pd
import os
from src.model import Model

class Trial:
    def __init__(self, global_params):
        """Initialize the trial with global parameters."""
        self.global_params = global_params
        
        # Initalise empty DataFrame for aggregated patient level results
        self.agg_results_df = pd.DataFrame() 

        # Initalise empty DataFrame for aggregated summary level results
        self.agg_results_hourly = pd.DataFrame()
        self.agg_results_daily = pd.DataFrame() 
        self.agg_results_complete = pd.DataFrame() 

        # Initalise empty DataFrame for aggegated queue results

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
    
            # Call the outcome_measures() method to get aggregated results for this run
            hourly_data, daily_data, complete_data = model.outcome_measures()

            # Concatenate the results of this run to the global DataFrames
            self.agg_results_hourly = pd.concat([self.agg_results_hourly, hourly_data], ignore_index=True) # Hourly 
            self.agg_results_daily = pd.concat([self.agg_results_daily, daily_data], ignore_index=True) # Daily
            self.agg_results_complete = pd.concat([self.agg_results_complete, complete_data], ignore_index=True) # Complete

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


        # Save patient-level results
        patient_result_path = os.path.join('data', 'results', 'results.csv')
        self.agg_results_df.to_csv(patient_result_path, index=False)
        print(f"Patient results saved to {patient_result_path}")

       # Save hourly results
        hourly_result_path = os.path.join('data', 'results', 'summary_results_hour.csv')
        hourly_data.to_csv(hourly_result_path, index=False)
        print(f"Hourly results saved to {hourly_result_path}")

        # Save daily results
        daily_result_path = os.path.join('data', 'results', 'summary_results_day.csv')
        daily_data.to_csv(daily_result_path, index=False)
        print(f"Daily results saved to {daily_result_path}")

        # Save complete results
        complete_result_path = os.path.join('data', 'results', 'summary_results_complete.csv')
        complete_data.to_csv(complete_result_path, index=False)
        print(f"Complete results saved to {complete_result_path}")

        # Save queue monitoring results
        consultant_queue_result_path = os.path.join('data', 'results', 'consultant_queue_monitoring_results.csv')
        self.agg_consultant_queue_monitoring_df.to_csv(consultant_queue_result_path, index=False)
        print(f"Queue monitoring results saved to {consultant_queue_result_path}")

        # Save queue monitoring results (AMU queue data)
        amu_queue_result_path = os.path.join('data', 'results', 'amu_queue_monitoring.csv')
        self.agg_amu_queue_df.to_csv(amu_queue_result_path, index=False)
        print(f"Queue monitoring results saved to {amu_queue_result_path}")
    
        return self.agg_results_df, self.agg_amu_queue_df