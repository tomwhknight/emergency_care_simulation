import pandas as pd
import os
from src.model_alt import AltModel  # uses your modified model logic


class AltTrial:
    def __init__(self, global_params):
        self.global_params = global_params
        
        self.agg_results_df = pd.DataFrame()
        self.agg_event_log = pd.DataFrame(columns=["run_number", "patient_id", "event", "timestamp"])

        self.agg_results_hourly = pd.DataFrame()
        self.agg_results_daily = pd.DataFrame()
        self.agg_results_complete = pd.DataFrame()

        self.agg_consultant_queue_monitoring_df = pd.DataFrame(columns=["Simulation Time", "Hour of Day", "Queue Length"])
        self.agg_amu_queue_df = pd.DataFrame()

    def run(self, run_number, progress_bar=None):
        burn_in_time = self.global_params.burn_in_time

        for i in range(run_number):
            print(f"[ALT] Starting simulation run {i + 1} with a burn-in period of {burn_in_time}")
            model = AltModel(self.global_params, burn_in_time, run_number=i+1)
            model.run()

            model.run_results_df_reset = model.run_results_df.reset_index()
            model.run_results_df_reset["Run Number"] = i + 1
            
            self.agg_results_df = pd.concat([self.agg_results_df, model.run_results_df_reset], ignore_index=True)
            self.agg_event_log = pd.concat([self.agg_event_log, model.event_log_df], ignore_index=True)
            
            hourly_data, daily_data, complete_data = model.outcome_measures()
            self.agg_results_hourly = pd.concat([self.agg_results_hourly, hourly_data], ignore_index=True)
            self.agg_results_daily = pd.concat([self.agg_results_daily, daily_data], ignore_index=True)
            self.agg_results_complete = pd.concat([self.agg_results_complete, complete_data], ignore_index=True)

            self.overall_summary = (
                self.agg_results_complete.groupby("measure", as_index=False)
                .agg(mean_overall=("mean_value", "mean"))
            )
            proportion_metrics = [
                "SDEC Appropriate",
                "SDEC Accepted",
                "SDEC Accepted (of Appropriate)",
                ">4hr breach",
                ">12hr breach",
                "Proportion Referred - Medicine"
            ]
            self.overall_summary["mean_overall"] = self.overall_summary.apply(
                lambda row: round(row["mean_overall"] * 100, 2) if row["measure"] in proportion_metrics else round(row["mean_overall"], 1),
                axis=1
            )

            model.consultant_queue_monitoring_df["Run Number"] = i + 1
            self.agg_consultant_queue_monitoring_df = pd.concat([self.agg_consultant_queue_monitoring_df, model.consultant_queue_monitoring_df], ignore_index=True)
            self.agg_amu_queue_df = pd.concat([self.agg_amu_queue_df, model.amu_queue_df], ignore_index=True)

            if progress_bar:
                percent = int((i + 1) / run_number * 100)
                progress_bar.progress(percent, text=f"[ALT] Running simulation... {percent}%")

        cols = ["Run Number"] + [col for col in self.agg_results_df.columns if col != "Run Number"]
        self.agg_results_df = self.agg_results_df[cols]

        if not os.path.exists('results'):
            os.makedirs('results')

       # Save files under clearly labelled filenames
        self.agg_results_df.to_csv(os.path.join('data', 'results', 'results_from_alt_model.csv'), index=False)
        print("Saved: results_from_alt_model.csv")

        # Save event log
        event_log_path = os.path.join('data', 'results', 'event_log_from_alt_model.csv')
        self.agg_event_log.to_csv(event_log_path, index=False)
        print(f"Event log saved to {event_log_path}")

        self.agg_results_hourly.to_csv(os.path.join('data', 'results', 'summary_results_hour_from_alt_model.csv'), index=False)
        print("Saved: summary_results_hour_from_alt_model.csv")

        self.agg_results_daily.to_csv(os.path.join('data', 'results', 'summary_results_day_from_alt_model.csv'), index=False)
        print("Saved: summary_results_day_from_alt_model.csv")

        self.overall_summary.to_csv(os.path.join('data', 'results', 'overall_summary_from_alt_model.csv'), index=False)
        print("Saved: overall_summary_from_alt_model.csv")

        self.agg_consultant_queue_monitoring_df.to_csv(os.path.join('data', 'results', 'consultant_queue_monitoring_results_from_alt_model.csv'), index=False)
        print("Saved: consultant_queue_monitoring_results_from_alt_model.csv")

        self.agg_amu_queue_df.to_csv(os.path.join('data', 'results', 'amu_queue_monitoring_from_alt_model.csv'), index=False)
        print("Saved: amu_queue_monitoring_from_alt_model.csv")

        return self.agg_results_df, self.agg_amu_queue_df
