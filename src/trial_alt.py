# src/trial_alt.py

import os
import numpy as np
import pandas as pd

from datetime import datetime
from src.model_alt import AltModel  # direct-triage variant


class AltTrial:
    """
    Runs AltModel across N stochastic repetitions and saves ONE aggregated set of CSVs
    (no per-run folders) into a scenario/batch directory.

    Output layout (per batch):
      <base_output_dir>/<scenario>/batch_YYYYmmdd_HHMMSS/
        - alt_results.csv
        - alt_event_log.csv
        - alt_summary_hourly.csv
        - alt_summary_daily.csv
        - alt_summary_complete.csv
        - alt_queue_ed_assessment.csv
        - alt_queue_medical.csv
        - alt_queue_consultant.csv
        - alt_queue_amu.csv
        - alt_seed_manifest.csv
    """

    def __init__(self, global_params, master_seed, base_output_dir=None):
        self.global_params = global_params
        self.master_seed = master_seed

        # Default outside-project path unless overridden
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

    def run(self, run_number: int, dt_threshold: float | None = None, progress_bar=None):
        """
        Run N simulations.

        dt_threshold:
          If provided, sets global_params.direct_triage_threshold for this batch.
          If None, AltModel will use whatever is already on global_params.
        """
        if dt_threshold is not None:
            self.global_params.direct_triage_threshold = float(dt_threshold)

        burn_in_time = self.global_params.burn_in_time

        # Get scenario label exactly as AltModel will record it
        tmp_model = AltModel(self.global_params, burn_in_time, run_number=0)
        scenario_name = getattr(tmp_model, "_scenario_name", "alt")
        label         = getattr(tmp_model, "_policy_label", "unlabelled")

        # Batch root for this set of runs
        scenario_dir = os.path.join(
            self.base_output_dir,
            scenario_name,                                 
            f"batch_{self.batch_id}__{label}"              
        )
        os.makedirs(scenario_dir, exist_ok=True)

        for i in range(run_number):
            run_idx = i + 1
            print(f"[ALT] Run {run_idx}/{run_number} | scenario={scenario_name} | batch={self.batch_id}")

            # Per-run RNG seeds
            self._per_run_seeds(run_idx)

            # Build and run model
            model = AltModel(self.global_params, burn_in_time, run_number=run_idx)
            model.run()

            # Patient-level results (add Run Number, then aggregate)
            run_df = model.run_results_df.reset_index()
            run_df["Run Number"] = run_idx
            self.agg_results_df = pd.concat([self.agg_results_df, run_df], ignore_index=True)

            # Event log (already includes run_number)
            self.agg_event_log = pd.concat([self.agg_event_log, model.event_log_df], ignore_index=True)

            # Summaries
            hourly, daily, complete = model.outcome_measures()
            self.agg_results_hourly = pd.concat([self.agg_results_hourly, hourly], ignore_index=True)
            self.agg_results_daily = pd.concat([self.agg_results_daily, daily], ignore_index=True)
            self.agg_results_complete = pd.concat([self.agg_results_complete, complete], ignore_index=True)

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

            if progress_bar:
                pct = int(run_idx / run_number * 100)
                progress_bar.progress(pct, text=f"[ALT] Running simulation... {pct}%")

        # --- Write ONE set of aggregated outputs for the whole batch ---
        self.agg_results_df.to_csv(os.path.join(scenario_dir, "alt_results.csv"), index=False)
        self.agg_event_log.to_csv(os.path.join(scenario_dir, "alt_event_log.csv"), index=False)
        self.agg_results_hourly.to_csv(os.path.join(scenario_dir, "alt_summary_hourly.csv"), index=False)
        self.agg_results_daily.to_csv(os.path.join(scenario_dir, "alt_summary_daily.csv"), index=False)
        self.agg_results_complete.to_csv(os.path.join(scenario_dir, "alt_summary_complete.csv"), index=False)
        self.agg_ed_assessment_queue_monitoring_df.to_csv(
            os.path.join(scenario_dir, "alt_queue_ed_assessment.csv"), index=False
        )
        self.agg_consultant_queue_monitoring_df.to_csv(
            os.path.join(scenario_dir, "alt_queue_consultant.csv"), index=False
        )

        self.agg_medical_queue_monitoring_df.to_csv(
            os.path.join(scenario_dir, "alt_queue_medical.csv"), index=False
        )

        self.agg_amu_queue_df.to_csv(
            os.path.join(scenario_dir, "alt_queue_amu.csv"), index=False
        )
        pd.DataFrame(self.seed_manifest).to_csv(
            os.path.join(scenario_dir, "alt_seed_manifest.csv"), index=False
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
