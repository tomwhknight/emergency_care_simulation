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
        for i in range(run_number):
            print(f"Starting simulation run {i + 1}...")
            model = Model(self.global_params, run_number=i+1)  # Create a new instance of the Model class with run_number
            model.run()  # Run the model
            
            # Concatenate the results of each run to the global results DataFrame
            self.agg_results_df = pd.concat([self.agg_results_df, model.run_results_df])
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