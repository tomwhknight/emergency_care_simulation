# src/trial.py

import os
from datetime import datetime
import numpy as np
import pandas as pd

from src.model import Model


class Trial:
    """
    Baseline trial runner: executes N stochastic runs, aggregates in-memory,
    and saves ONE set of CSVs (no per-run folders) into a timestamped batch.

    Output layout:
      <base_output_dir>/baseline/batch_YYYYmmdd_HHMMSS/
        - baseline_results.csv
        - baseline_event_log.csv
        - baseline_summary_hourly.csv
        - baseline_summary_daily.csv
        - baseline_summary_complete.csv
        - baseline_queue_ed_assessment.csv
        - baelines_initial_medicine.csv
        - baseline_queue_consultant.csv
        - baseline_queue_amu.csv
        - baseline_seed_manifest.csv
    """

    def __init__(self, global_params, master_seed, base_output_dir=None):
        self.global_params = global_params
        self.master_seed = master_seed

        # Default outside-project path unless caller overrides
        self.base_output_dir = os.path.abspath(
            base_output_dir or "/Users/thomasknight/Local files/DES/output/des_output"
        )
        os.makedirs(self.base_output_dir, exist_ok=True)

        # In-memory aggregates
        self.agg_results_df = pd.DataFrame()
        self.agg_event_log = pd.DataFrame(columns=["run_number", "patient_id", "event", "timestamp"])
        self.agg_results_hourly = pd.DataFrame()
        self.agg_results_daily = pd.DataFrame()
        self.agg_results_complete = pd.DataFrame()
        
        self.agg_ed_assessment_queue_monitoring_df = pd.DataFrame(
            columns=["Simulation Time", "Hour of Day", "Queue Length"]
        )
        
        self.agg_medical_queue_monitoring_df = pd.DataFrame(
            columns=["Simulation Time", "Hour of Day", "Queue Length"]
        )
        
        self.agg_consultant_queue_monitoring_df = pd.DataFrame(
            columns=["Simulation Time", "Hour of Day", "Queue Length"]
        )
        self.agg_amu_queue_df = pd.DataFrame()
        
        self.agg_ed_doctor_block_monitoring_df = pd.DataFrame(columns=[
            "Simulation Time", "Hour of Day", "Physical Capacity",
            "Rota Blockers", "Break Blockers", "Total Blockers",
            "Effective Capacity", "Active Patient Users", "Patient Queue Length",
            "Desired From Rota", "Run Number"
        ])

        self.agg_calibration_summary = pd.DataFrame(
            columns=["measure","mean_value","run_number","Scenario","DT Threshold"]
        )
        self.agg_calibration_deciles = pd.DataFrame(
            columns=["pcal_decile","mean_p_cal","obs_prop_medicine","n","run_number","Scenario","DT Threshold"]
        )
        
        # Record RNG seeds
        self.seed_manifest = []

        # Unique batch id so re-runs don’t overwrite
        self.batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _per_run_seeds(self, run_idx: int):
        """Derive 4 independent seeds from the master seed for this run and write them onto global_params."""
        ss = np.random.SeedSequence(self.master_seed, spawn_key=(run_idx,))
        s_arr, s_srv, s_pr, s_res = ss.spawn(4)

        self.global_params.seed_arrivals  = int(s_arr.generate_state(1)[0])
        self.global_params.seed_service   = int(s_srv.generate_state(1)[0])
        self.global_params.seed_probs     = int(s_pr.generate_state(1)[0])
        self.global_params.seed_resources = int(s_res.generate_state(1)[0])

        self.seed_manifest.append({
            "run_number": run_idx,
            "seed_arrivals":  self.global_params.seed_arrivals,
            "seed_service":   self.global_params.seed_service,
            "seed_probs":     self.global_params.seed_probs,
            "seed_resources": self.global_params.seed_resources,
        })

    def run(self, run_number: int, progress_bar=None):
        burn_in_time = self.global_params.burn_in_time

        # Batch root for this set of runs
        scenario_dir = os.path.join(self.base_output_dir, "baseline", f"batch_{self.batch_id}")
        os.makedirs(scenario_dir, exist_ok=True)

        buf_complete = []

        for i in range(run_number):
            run_idx = i + 1
            print(f"[BASELINE] Run {run_idx}/{run_number} | batch={self.batch_id}")

            # Per-run RNG seeds
            self._per_run_seeds(run_idx)

            # Build and run model
            model = Model(self.global_params, burn_in_time, run_number=run_idx)
            model.run()

            # Patient-level results (add Run Number, then aggregate)
            run_df = model.run_results_df.reset_index()
            run_df["Run Number"] = run_idx
            self.agg_results_df = pd.concat([self.agg_results_df, run_df], ignore_index=True)

            # Event log (already includes run_number from the model)
            self.agg_event_log = pd.concat([self.agg_event_log, model.event_log_df], ignore_index=True)

            # Summaries
            complete = model.outcome_measures()
            buf_complete.append(complete)

            # Queues (tag with run for traceability)
            ed_q = model.ed_assessment_queue_monitoring_df.copy()
            ed_q["Run Number"] = run_idx
            self.agg_ed_assessment_queue_monitoring_df = pd.concat(
                [self.agg_ed_assessment_queue_monitoring_df, ed_q], ignore_index=True
            )

            med_q = model.medical_queue_monitoring_df.copy()
            med_q["Run Number"] = run_idx
            self.agg_medical_queue_monitoring_df = pd.concat(
                [self.agg_medical_queue_monitoring_df, med_q], ignore_index=True
            )

            cons_q = model.consultant_queue_monitoring_df.copy()
            cons_q["Run Number"] = run_idx
            self.agg_consultant_queue_monitoring_df = pd.concat(
                [self.agg_consultant_queue_monitoring_df, cons_q], ignore_index=True
            )

            amu_q = model.amu_queue_df.copy()
            amu_q["Run Number"] = run_idx
            self.agg_amu_queue_df = pd.concat([self.agg_amu_queue_df, amu_q], ignore_index=True)

            ed_blocks = model.ed_doctor_block_monitoring_df.copy()
            ed_blocks["Run Number"] = run_idx
            self.agg_ed_doctor_block_monitoring_df = pd.concat(
                [self.agg_ed_doctor_block_monitoring_df, ed_blocks],
                ignore_index=True
            )

            # Calibration aggregates
            if hasattr(model, "calibration_summary"):
                self.agg_calibration_summary = pd.concat(
                    [self.agg_calibration_summary, model.calibration_summary],
                    ignore_index=True
                )
            if hasattr(model, "calibration_deciles"):
                self.agg_calibration_deciles = pd.concat(
                    [self.agg_calibration_deciles, model.calibration_deciles],
                    ignore_index=True
                )

            if progress_bar:
                pct = int(run_idx / run_number * 100)
                progress_bar.progress(pct, text=f"[BASELINE] Running simulation... {pct}%")


        # Build complete summary across runs in one shot
        self.agg_results_complete = (
            pd.concat(buf_complete, ignore_index=True) if buf_complete else pd.DataFrame()
        )


        tmp = self.agg_results_complete.copy()
        tmp["mean_value"] = pd.to_numeric(tmp["mean_value"], errors="coerce")
        self.agg_results_complete = (
            tmp.groupby(["measure"], dropna=False, as_index=False)["mean_value"]
            .mean()
            .rename(columns={"mean_value": "mean_across_runs"})
        )
        

        # --- Write aggregated outputs for the batch ---
        self.agg_results_df.to_csv(os.path.join(scenario_dir, "baseline_results.csv"), index=False)
        self.agg_event_log.to_csv(os.path.join(scenario_dir, "baseline_event_log.csv"), index=False)
        
        # Save per-run concatenated results for debugging
        self.agg_results_complete.to_csv(os.path.join(scenario_dir, "baseline_summary_complete_allruns.csv"), index=False)


        # Save per-run concatenated results related to queues

        self.agg_ed_assessment_queue_monitoring_df.to_csv(
            os.path.join(scenario_dir, "baseline_queue_ed_assessment.csv"), index=False
        )   
        self.agg_medical_queue_monitoring_df.to_csv(
            os.path.join(scenario_dir, "baseline_queue_medical.csv"), index=False
        )     
        self.agg_consultant_queue_monitoring_df.to_csv(
            os.path.join(scenario_dir, "baseline_queue_consultant.csv"), index=False
        )
        self.agg_amu_queue_df.to_csv(
            os.path.join(scenario_dir, "baseline_queue_amu.csv"), index=False
        )
        self.agg_ed_doctor_block_monitoring_df.to_csv(
            os.path.join(scenario_dir, "baseline_ed_doctor_blocks.csv"), index=False
        )

        self.agg_calibration_summary.to_csv(
            os.path.join(scenario_dir, "baseline_calibration_summary.csv"), index=False
        )
        self.agg_calibration_deciles.to_csv(
            os.path.join(scenario_dir, "baseline_calibration_deciles.csv"), index=False
        )
        pd.DataFrame(self.seed_manifest).to_csv(
            os.path.join(scenario_dir, "baseline_seed_manifest.csv"), index=False
        )

        # Return handles if you want to inspect programmatically
        return {
            "scenario_dir": scenario_dir,
            "patients": self.agg_results_df,
            "events": self.agg_event_log,
            "hourly": self.agg_results_hourly,
            "daily": self.agg_results_daily,
            "complete": self.agg_results_complete,
            "queue_ed": self.agg_ed_assessment_queue_monitoring_df,
            "queue_consultant": self.agg_consultant_queue_monitoring_df,
            "queue_amu": self.agg_amu_queue_df,
        }
