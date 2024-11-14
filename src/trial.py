import pandas as pd
import os
from src.model import Model

class Trial:
    def __init__(self, global_params):
        """Initialize the trial with global parameters."""
        self.global_params = global_params
        self.agg_results_df = pd.DataFrame()  # Initialize an empty DataFrame for all run results

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
    
        # Move 'Run Number' to the first column for cleaner presentation
        cols = ["Run Number"] + [col for col in self.agg_results_df.columns if col != "Run Number"]
        self.agg_results_df = self.agg_results_df[cols]
        # Ensure the directory 'results' exists
        if not os.path.exists('results'):
            os.makedirs('results')

        # Save the results to a CSV file in the 'results' folder
        result_path = os.path.join('results', 'results.csv')
        self.agg_results_df.to_csv(result_path, index=False)
        print(f"Results saved to {result_path}")

        return self.agg_results_df