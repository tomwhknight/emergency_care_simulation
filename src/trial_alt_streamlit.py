# src/trial_alt_streamlit.py

import numpy as np
import pandas as pd

from src.model_alt import AltModel


class AltTrialStreamlit:
    """
    Direct-to-medicine trial runner for Streamlit/hosted use.

    This version does not create folders and does not save any CSV files.
    """

    def __init__(self, global_params, master_seed):
        self.global_params = global_params
        self.master_seed = master_seed

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

        self.agg_resource_monitoring_df = pd.DataFrame(columns=[
            "Run Number", "Simulation Time", "Hour of Day",
            "Resource", "Physical Capacity", "Active Blockers",
            "Effective Capacity", "In Use (Patients)", "Queue (Patients)"
        ])

        self.seed_manifest = []

    def _per_run_seeds(self, run_idx: int):
        """Derive 4 independent seeds from the master seed for this run and write them onto global_params."""
        ss = np.random.SeedSequence(self.master_seed, spawn_key=(run_idx,))
        s_arr, s_srv, s_pr, s_res = ss.spawn(4)

        self.global_params.seed_arrivals = int(s_arr.generate_state(1)[0])
        self.global_params.seed_service = int(s_srv.generate_state(1)[0])
        self.global_params.seed_probs = int(s_pr.generate_state(1)[0])
        self.global_params.seed_resources = int(s_res.generate_state(1)[0])

        self.seed_manifest.append({
            "run_number": run_idx,
            "seed_arrivals": self.global_params.seed_arrivals,
            "seed_service": self.global_params.seed_service,
            "seed_probs": self.global_params.seed_probs,
            "seed_resources": self.global_params.seed_resources,
        })

    def run(self, run_number: int, dt_threshold: float | None = None, progress_bar=None):
        if dt_threshold is not None:
            self.global_params.direct_triage_threshold = float(dt_threshold)

        burn_in_time = self.global_params.burn_in_time
        buf_complete = []

        tmp_model = AltModel(self.global_params, burn_in_time, run_number=0)
        scenario_name = getattr(tmp_model, "_scenario_name", "alt")
        label = getattr(tmp_model, "_policy_label", "unlabelled")

        for i in range(run_number):
            run_idx = i + 1
            print(f"[ALT] Run {run_idx}/{run_number} | scenario={scenario_name}")

            self._per_run_seeds(run_idx)

            model = AltModel(self.global_params, burn_in_time, run_number=run_idx)
            model.run()

            run_df = model.run_results_df.reset_index()
            run_df["Run Number"] = run_idx
            self.agg_results_df = pd.concat([self.agg_results_df, run_df], ignore_index=True)

            self.agg_event_log = pd.concat([self.agg_event_log, model.event_log_df], ignore_index=True)

            complete = model.outcome_measures()
            if isinstance(complete, tuple):
                complete = complete[-1]
            buf_complete.append(complete)

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

            resmon = model.resource_monitoring_df.copy()
            resmon["Run Number"] = run_idx
            self.agg_resource_monitoring_df = pd.concat(
                [self.agg_resource_monitoring_df, resmon], ignore_index=True
            )

            amu_q = model.amu_queue_df.copy()
            amu_q["Run Number"] = run_idx
            self.agg_amu_queue_df = pd.concat([self.agg_amu_queue_df, amu_q], ignore_index=True)

            if progress_bar:
                pct = int(run_idx / run_number * 100)
                progress_bar.progress(pct, text=f"[ALT] Running simulation... {pct}%")

        self.agg_results_complete = (
            pd.concat(buf_complete, ignore_index=True) if buf_complete else pd.DataFrame()
        )

        if "mean_value" in self.agg_results_complete.columns:
            tmp = self.agg_results_complete.copy()
            tmp["mean_value"] = pd.to_numeric(tmp["mean_value"], errors="coerce")
            self.agg_results_complete = (
                tmp.groupby(["measure"], dropna=False, as_index=False)["mean_value"]
                .mean()
                .rename(columns={"mean_value": "mean_across_runs"})
            )

        return {
            "scenario_name": scenario_name,
            "policy_label": label,
            "patients": self.agg_results_df,
            "events": self.agg_event_log,
            "hourly": self.agg_results_hourly,
            "daily": self.agg_results_daily,
            "complete": self.agg_results_complete,
            "queue_ed": self.agg_ed_assessment_queue_monitoring_df,
            "queue_medical": self.agg_medical_queue_monitoring_df,
            "queue_consultant": self.agg_consultant_queue_monitoring_df,
            "queue_amu": self.agg_amu_queue_df,
            "resource_monitor": self.agg_resource_monitoring_df,
            "seed_manifest": pd.DataFrame(self.seed_manifest),
        }
