
"""
calibration.py — Full-grid calibration & diagnostics for DES vs Observed.

Outputs in src/results/:
  - calibration_grid_all.csv
  - time_in_system_summary_all.csv

"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# ------------------------- Import your runner -------------------------
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from run_calibration import run_model as _raw_run_model, global_params  

# ------------------------------ Config -------------------------------
OBS_TIS_PATH   = "data/calibration/observed_system_time.csv"
OBS_QUEUE_PATH = "data/calibration/observed_ed_assessment_queue.csv" 
SIM_QUEUE_COL = "Queue Length ED doctor"   # exact sim patient-level column name
SIM_HOUR_COL  = "Hour of Arrival"   

RESULTS_DIR = "src/results"

GRID_PATH = os.path.join(RESULTS_DIR, "calibration_grid_all.csv")
TIS_SUMMARY_PATH = os.path.join(RESULTS_DIR, "time_in_system_summary_all.csv")
QUEUE_SUMMARY_PATH = os.path.join(RESULTS_DIR, "queue_length_summary_all.csv")

# Parameter grid
mus    = np.array([3.85, 3.90, 3.95, 4.00, 4.05])
sigmas = np.array([0.45, 0.50, 0.55, 0.60, 0.65])

TOTAL_RUNS = 10
np.random.seed(42)

# ------------------------ Normalisation helpers ----------------------

def _normalise_sim_output(sim_patients: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(sim_patients, pd.DataFrame):
        raise TypeError("Expected sim_patients to be a pandas DataFrame.")
    if "time_in_system" not in sim_patients.columns:
        raise KeyError("Missing required column: 'time_in_system'")

    # ---- exclude booked appointments using discharge_decision_point ONLY ----
    if "discharge_decision_point" not in sim_patients.columns:
        raise KeyError("Missing required column: 'discharge_decision_point'")

    sim_patients = sim_patients.loc[
        sim_patients["discharge_decision_point"]
          .fillna("")
          .astype(str)
          .str.strip()
          .str.lower() != "booked_appointment"
    ].copy()
    # -----------------------------------------------------------------------

    out = sim_patients[["time_in_system"]].copy()
    out["time_in_system"] = pd.to_numeric(out["time_in_system"], errors="coerce")

    if SIM_QUEUE_COL in sim_patients.columns:
        out["queue_seen_at_arrival"] = pd.to_numeric(sim_patients[SIM_QUEUE_COL], errors="coerce")
    else:
        out["queue_seen_at_arrival"] = np.nan

    return out.dropna(subset=["time_in_system"])




def run_model_outputs(g, mu, sigma, total_runs=10) -> dict:
    """
    Expect run_calibration.run_model(...) to return:
      {"patients": <df>, "queue_ed": <df>}
    """
    raw = _raw_run_model(g, mu, sigma, total_runs=total_runs)

    if not isinstance(raw, dict):
        raise TypeError("run_model(...) must return a dict")
    
    if "patients" not in raw or "queue_ed" not in raw:
        raise KeyError("run_model(...) must return keys: 'patients' and 'queue_ed'")


    return raw

def run_patients_df(g, mu, sigma, total_runs=10) -> pd.DataFrame:
    raw = run_model_outputs(g, mu, sigma, total_runs=total_runs)
    return _normalise_sim_output(raw["patients"])

# ----------------------------- Metrics -------------------------------
def parse_hour_col(df: pd.DataFrame, col="hour_of_day") -> pd.Series:
    s = pd.to_datetime(df[col], format="%H:%M", errors="coerce")
    if s.isna().all():
        s = pd.to_datetime(df[col], errors="coerce")
    return s.dt.hour.astype("Int64")

def ensure_full_hours(tab: pd.DataFrame, on="hour") -> pd.DataFrame:
    hours = pd.DataFrame({on: range(24)})
    out = hours.merge(tab, on=on, how="left")
    for c in out.columns:
        if c != on and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].fillna(0)
    return out

def queue_profile_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean queue length (seen at arrival) by hour.
    Expects: 'hour_of_day' and 'queue_seen_at_arrival'
    """
    tmp = df.copy()
    tmp["hour"] = parse_hour_col(tmp, col="hour_of_day")
    tab = (tmp.dropna(subset=["hour", "queue_seen_at_arrival"])
             .groupby("hour", as_index=False)["queue_seen_at_arrival"]
             .mean()
             .rename(columns={"queue_seen_at_arrival": "mean_queue"}))
    return ensure_full_hours(tab, on="hour")

def queue_profile_table_sim(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["hour"] = pd.to_numeric(tmp[SIM_HOUR_COL], errors="coerce").astype("Int64")
    tmp["queue_seen_at_arrival"] = pd.to_numeric(tmp[SIM_QUEUE_COL], errors="coerce")
    tab = (tmp.dropna(subset=["hour", "queue_seen_at_arrival"])
             .groupby("hour", as_index=False)["queue_seen_at_arrival"]
             .mean()
             .rename(columns={"queue_seen_at_arrival": "mean_queue"}))
    return ensure_full_hours(tab, on="hour")


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


# ----------------------------- Output -------------------------------

def summarize_tis(vec_like) -> dict:
    v = pd.to_numeric(pd.Series(vec_like), errors="coerce").dropna().to_numpy()
    if v.size == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan,
                "p50": np.nan, "p75": np.nan, "p90": np.nan, "p95": np.nan, "p99": np.nan,
                "min": np.nan, "max": np.nan}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "p50": float(np.percentile(v, 50)),
        "p75": float(np.percentile(v, 75)),
        "p90": float(np.percentile(v, 90)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "min": float(v.min()),
        "max": float(v.max()),
    }

# --------------------------- Load observed ---------------------------


obs_tis_raw = pd.read_csv(OBS_TIS_PATH)

if "time_in_system" not in obs_tis_raw.columns:
    raise KeyError("Observed TIS file missing 'time_in_system' column.")

obs_tis_df = pd.DataFrame({
    "time_in_system": pd.to_numeric(obs_tis_raw["time_in_system"], errors="coerce")
})

obs_values = obs_tis_df["time_in_system"].dropna().to_numpy()
obs_tis_summary = summarize_tis(obs_values)

# --- Observed queue at arrival ---
obs_queue_raw = pd.read_csv(OBS_QUEUE_PATH)

required_q = ["queue_seen_at_arrival", "hour_of_day"]
missing_q = [c for c in required_q if c not in obs_queue_raw.columns]
if missing_q:
    raise KeyError(f"Observed queue file missing required columns: {missing_q}")

obs_queue_df = pd.DataFrame({
    "queue_seen_at_arrival": pd.to_numeric(obs_queue_raw["queue_seen_at_arrival"], errors="coerce"),
    "hour_of_day": obs_queue_raw["hour_of_day"],  # e.g. '21:00'
})

obs_queue_values  = obs_queue_df["queue_seen_at_arrival"].dropna().to_numpy()
obs_queue_summary = summarize_tis(obs_queue_values)
obs_queue_profile = queue_profile_table(obs_queue_df)


# ------------------------------ Run grid ----------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)

grid_rows = []
tis_rows = []
queue_rows = []


for mu in mus:
    for sigma in sigmas:
        print(f"Running mu={mu}, sigma={sigma}")
        raw = run_model_outputs(global_params, mu, sigma, total_runs=TOTAL_RUNS)

        # --- TIS normalised ---
        sim_df = _normalise_sim_output(raw["patients"])
        sim_values = sim_df["time_in_system"].to_numpy()

        # --- TIS metrics ---
        if sim_values.size >= 2 and obs_values.size >= 2:
            ks_stat, ks_p = ks_2samp(sim_values, obs_values)
            sim_mean = float(sim_values.mean())
            mean_diff = abs(sim_mean - obs_tis_summary["mean"])
        else:
            ks_stat = ks_p = sim_mean = mean_diff = np.nan

        # =======================
        # Queue length metrics
        # =======================
        sim_patients_raw = raw["patients"]

        # Distribution (KS + mean diff)
        sim_queue_values = pd.to_numeric(sim_patients_raw.get(SIM_QUEUE_COL), errors="coerce") \
                             .dropna().to_numpy()

        if sim_queue_values.size >= 2 and obs_queue_values.size >= 2:
            q_ks_stat, q_ks_p = ks_2samp(sim_queue_values, obs_queue_values)
            sim_q_mean = float(sim_queue_values.mean())
            q_mean_diff = abs(sim_q_mean - float(obs_queue_summary["mean"]))
        else:
            q_ks_stat = q_ks_p = sim_q_mean = q_mean_diff = np.nan

        # Hourly profile (RMSE)
        if SIM_HOUR_COL in sim_patients_raw.columns and SIM_QUEUE_COL in sim_patients_raw.columns:
            sim_queue_profile = queue_profile_table_sim(
                sim_patients_raw[[SIM_HOUR_COL, SIM_QUEUE_COL]].copy()
            )
            q_hourly_rmse = rmse(
                sim_queue_profile["mean_queue"].to_numpy(),
                obs_queue_profile["mean_queue"].to_numpy()
            )
        else:
            q_hourly_rmse = np.nan

        # --- Collect grid row ---
        grid_rows.append({
            "mu": mu, "sigma": sigma,

            # TIS
            "ks_stat": ks_stat, "ks_p": ks_p,
            "sim_mean": sim_mean, "obs_mean": obs_tis_summary["mean"],
            "mean_diff": mean_diff,
            "n_sim": int(sim_values.size),

            # Queue length
            "q_ks_stat": q_ks_stat, "q_ks_p": q_ks_p,
            "sim_q_mean": sim_q_mean, "obs_q_mean": float(obs_queue_summary["mean"]),
            "q_mean_diff": q_mean_diff,
            "q_hourly_rmse": q_hourly_rmse,
            "n_sim_queue": int(sim_queue_values.size),
        })

        # --- Summaries ---
        sim_tis_summary = summarize_tis(sim_values)
        tis_rows.append({"mu": mu, "sigma": sigma, "label": "Observed", **obs_tis_summary})
        tis_rows.append({"mu": mu, "sigma": sigma, "label": "DES", **sim_tis_summary})

        # --- Queue summaries ---
        sim_queue_summary = summarize_tis(sim_queue_values)
        queue_rows.append({"mu": mu, "sigma": sigma, "label": "Observed", **obs_queue_summary})
        queue_rows.append({"mu": mu, "sigma": sigma, "label": "DES", **sim_queue_summary})

    
# ------------------------------ Save all -----------------------------
pd.DataFrame(grid_rows).sort_values(
    by=["ks_stat", "mean_diff", "q_ks_stat", "q_hourly_rmse"],
    ascending=[True, True, True, True]
).to_csv(GRID_PATH, index=False)

pd.DataFrame(tis_rows).to_csv(TIS_SUMMARY_PATH, index=False)
pd.DataFrame(queue_rows).to_csv(QUEUE_SUMMARY_PATH, index=False)

print("\nSaved:")
print(f"  - {GRID_PATH}")
print(f"  - {TIS_SUMMARY_PATH}")
