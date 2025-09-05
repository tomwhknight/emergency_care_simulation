import pandas as pd
import os
from src.model import Model

class Trial:
    def __init__(self, global_params):
        """Initialize the trial with global parameters."""
        self.global_params = global_params

        # Set output directory
        self.output_dir = "/Users/thomasknight/Local files/DES/output/des_output/baseline"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initalise empty DataFrame for aggregated patient level results
        self.agg_results_df = pd.DataFrame() 

         # Initalise empty DataFrame for aggregated event log
        self.agg_event_log = pd.DataFrame(columns=["run_number", "patient_id", "event", "timestamp"])

        # Initalise empty DataFrame for aggregated summary level results
        self.agg_results_hourly = pd.DataFrame()
        self.agg_results_daily = pd.DataFrame() 
        self.agg_results_complete = pd.DataFrame() 

        # Initalise empty DataFrame for aggegated queue results
        self.agg_ed_assessment_queue_monitoring_df = pd.DataFrame(columns=["Simulation Time", "Hour of Day", "Queue Length"])
        self.agg_consultant_queue_monitoring_df = pd.DataFrame(columns=["Simulation Time", "Hour of Day", "Queue Length"])
        self.agg_amu_queue_df = pd.DataFrame()  # Initialize an empty DataFrame for AMU queue monitoring results


    def run(self, run_number, progress_bar=None):
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
    
            # Concatenate the event log for this run to the global event log DataFrame
            self.agg_event_log = pd.concat([self.agg_event_log, model.event_log_df], ignore_index=True)
            
            # Call the outcome_measures() method to get aggregated results for this run
            hourly_data, daily_data, complete_data = model.outcome_measures()

            # Concatenate the results of this run to the global DataFrames
            self.agg_results_hourly = pd.concat([self.agg_results_hourly, hourly_data], ignore_index=True) # Hourly 
            self.agg_results_daily = pd.concat([self.agg_results_daily, daily_data], ignore_index=True) # Daily
            self.agg_results_complete = pd.concat([self.agg_results_complete, complete_data], ignore_index=True) # Complete
            
           
            # Create overall mean across runs
            self.overall_summary = (
                self.agg_results_complete.groupby("measure", as_index=False)
                .agg(mean_overall=("mean_value", "mean"))
            )
            # Define proportion-based metrics (already multiplied by 100 earlier in outcome_measures)
            proportion_metrics = [
                "SDEC Appropriate",
                "SDEC Accepted",
                "SDEC Accepted (of Appropriate)",
                ">4hr breach",
                ">12hr breach",
                "Proportion Referred - Medicine"
            ]

            # Round selectively
            self.overall_summary["mean_overall"] = self.overall_summary.apply(
                lambda row: round(row["mean_overall"] * 100, 2) if row["measure"] in proportion_metrics else round(row["mean_overall"], 1),
                axis=1
            )

            # Add the 'Run Number' column to the consultant monitoring DataFrame
            model.consultant_queue_monitoring_df["Run Number"] = i + 1

            # Add the 'Run Number' column to the consultant monitoring DataFrame
            model.ed_assessment_queue_monitoring_df["Run Number"] = i + 1

            # Concatenate queue monitoring data for this run to the aggregated DataFrame
            self.agg_consultant_queue_monitoring_df = pd.concat([self.agg_consultant_queue_monitoring_df, model.consultant_queue_monitoring_df], ignore_index=True)

            # Concatenate queue monitoring data for this run to the aggregated DataFrame
            self.agg_ed_assessment_queue_monitoring_df = pd.concat([self.agg_ed_assessment_queue_monitoring_df, model.ed_assessment_queue_monitoring_df], ignore_index=True)

            # Concatenate the AMU queue results of each run to the global results DataFrame for AMU queue data
            self.agg_amu_queue_df = pd.concat([self.agg_amu_queue_df, model.amu_queue_df], ignore_index=True)

            if progress_bar:
                percent = int((i + 1) / run_number * 100)
                progress_bar.progress(percent, text=f"Running simulation... {percent}%")

        # Move 'Run Number' to the first column for cleaner presentation
        
        # Only assign "Run Number" if it doesn't already exist (avoids overwriting in multi-run scenarios)
        if "Run Number" not in self.agg_results_df.columns:
            self.agg_results_df["Run Number"] = run_number

        # Move "Run Number" to the first column for presentation
        cols = ["Run Number"] + [col for col in self.agg_results_df.columns if col != "Run Number"]
        self.agg_results_df = self.agg_results_df[cols]

        # Ensure the directory 'results' exists
        if not os.path.exists('results'):
            os.makedirs('results')

        # Save patient-level results
        patient_result_path = os.path.join(self.output_dir, "baseline_results.csv")
        self.agg_results_df.to_csv(patient_result_path, index=False)
        print(f"Patient results saved to {patient_result_path}")

        # Save event log
        event_log_path = os.path.join(self.output_dir, "baseline_event_log.csv")
        self.agg_event_log.to_csv(event_log_path, index=False)
        print(f"Event log saved to {event_log_path}")

        # Save hourly results
        hourly_result_path = os.path.join(self.output_dir, "baseline_summary_results_hour.csv")
        self.agg_results_hourly.to_csv(hourly_result_path, index=False)
        print(f"Hourly results saved to {hourly_result_path}")

        # Save daily results
        daily_result_path = os.path.join(self.output_dir, "baseline_summary_results_day.csv")
        self.agg_results_daily.to_csv(daily_result_path, index=False)
        print(f"Daily results saved to {daily_result_path}")

        # Save aggregated complete results
        overall_summary_path = os.path.join(self.output_dir, "baseline_overall_summary.csv")
        self.overall_summary.to_csv(overall_summary_path, index=False)
        print(f"Aggregated summary results saved to {overall_summary_path}")

        # Save queue monitoring results
        consultant_queue_result_path = os.path.join(self.output_dir, "baseline_consultant_queue_monitoring_results.csv")
        self.agg_consultant_queue_monitoring_df.to_csv(consultant_queue_result_path, index=False)
        print(f"Queue monitoring results saved to {consultant_queue_result_path}")

        # Save queue monitoring results
        ed_assessment_queue_result_path = os.path.join(self.output_dir, "baseline_ed_assessment_queue_monitoring_results.csv")
        self.agg_ed_assessment_queue_monitoring_df.to_csv(ed_assessment_queue_result_path , index=False)
        print(f"Queue monitoring results saved to {ed_assessment_queue_result_path}")

        # Save AMU queue monitoring results
        amu_queue_result_path = os.path.join(self.output_dir, "baseline_amu_queue_monitoring.csv")
        self.agg_amu_queue_df.to_csv(amu_queue_result_path, index=False)
        print(f"Queue monitoring results saved to {amu_queue_result_path}")

        return self.agg_results_df, self.agg_amu_queue_df