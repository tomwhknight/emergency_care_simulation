
"""
calibration.py — Full-grid calibration & diagnostics for DES vs Observed.

Outputs in src/results/:
  - calibration_grid_all.csv
  - time_in_system_summary_all.csv
  - hourly_breach_profiles_all.csv
  - arrivals_per_hour_profiles_all.csv
  - admitted_mix_by_hour_profiles_all.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# ------------------------- Import your runner -------------------------
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from run_calibration import run_model as _raw_run_model, global_params  # noqa: E402

# ------------------------------ Config -------------------------------
OBS_PATH = "data/calibration/observed_system_time.csv"
RESULTS_DIR = "src/results"

GRID_PATH = os.path.join(RESULTS_DIR, "calibration_grid_all.csv")
TIS_SUMMARY_PATH = os.path.join(RESULTS_DIR, "time_in_system_summary_all.csv")
BREACH_PROFILES_PATH = os.path.join(RESULTS_DIR, "hourly_breach_profiles_all.csv")
ARRIVALS_PROFILES_PATH = os.path.join(RESULTS_DIR, "arrivals_per_hour_profiles_all.csv")
ADMITTED_MIX_PROFILES_PATH = os.path.join(RESULTS_DIR, "admitted_mix_by_hour_profiles_all.csv")

# Parameter grid
mus    = np.arange(4.06, 5.0, 0.02)
sigmas = np.arange(0.60, 0.65, 0.01)

TOTAL_RUNS = 10
THRESHOLD_MIN = 240
np.random.seed(42)

# ------------------------ Normalisation helpers ----------------------
POSSIBLE_TIS = [
    "time_in_system", "Time in System", "Time_in_System", "Time in ED",
    "time_in_ed", "Total Time in System", "total_time_in_system"
]
POSSIBLE_HOUR = [
    "hour_of_day", "Hour of Day", "clock_hour_of_arrival",
    "Clock Hour of Arrival", "clock_hour", "arrival_hour",
    "arrival_time", "Arrival Time", "sim_time_arrival", "simulation_arrival_time",
    "Clock Hour"
]
POSSIBLE_ADMIT = [
    "admitted", "Admitted", "ed_admitted", "ED Admitted",
    "ED Disposition", "ed_disposition", "Disposition", "ed_disposition_outcome"
]

def _first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _coerce_hour_series(s: pd.Series) -> pd.Series:
    """
    Accept 'HH:MM' strings, datetimes, or numeric minutes since an epoch.
    Return 'hour_of_day' as strings 'HH:MM' (top of hour) for robust parsing downstream.
    """
    # Try HH:MM strings
    dt = pd.to_datetime(s, format="%H:%M", errors="coerce")
    if dt.notna().any():
        return dt.dt.strftime("%H:00")

    # Try general datetime-like
    dt2 = pd.to_datetime(s, errors="coerce")
    if dt2.notna().any():
        return dt2.dt.strftime("%H:00")

    # Try numeric minutes since midnight
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().any():
        hr = np.floor((num.astype(float) % (24*60)) / 60.0).astype(int)
        return pd.Series([f"{int(h):02d}:00" if pd.notna(h) else np.nan for h in hr], index=s.index)

    # Give up
    return pd.Series([np.nan]*len(s), index=s.index)

def _map_admitted_from_dispo(v):
    if pd.isna(v):
        return np.nan
    t = str(v).strip().lower()
    # Non-admissions (aligns with your R logic)
    if ("discharge" in t or "pseudo_exit" in t or "paeds" in t or
        "other speciality" in t or "other specialty" in t or "sdec" in t):
        return 0
    # Admissions
    if ("refer - medicine" in t or "medicine" in t or "amu" in t or "admit" in t):
        return 1
    return np.nan

def _normalise_sim_output(sim_out) -> pd.DataFrame:
    """
    Try to coerce whatever run_model returns into a DF with:
    ['time_in_system','hour_of_day','admitted'].
    """
    if isinstance(sim_out, pd.DataFrame):
        df = sim_out.copy()

        # time_in_system
        tis_col = _first_existing(df, POSSIBLE_TIS)
        if tis_col is None:
            raise KeyError("Could not find a time-in-system column in sim output.")
        df["time_in_system"] = pd.to_numeric(df[tis_col], errors="coerce")

        # hour_of_day
        hour_col = _first_existing(df, POSSIBLE_HOUR)
        if hour_col is None:
            raise KeyError("Could not find an arrival hour/time column in sim output.")
        df["hour_of_day"] = _coerce_hour_series(df[hour_col])

        # admitted
        adm_col = _first_existing(df, POSSIBLE_ADMIT)
        if adm_col is None:
            # If truly missing, assume discharged (0) — but better to raise
            raise KeyError("Could not find an 'admitted' or disposition column in sim output.")
        if df[adm_col].dtype.kind in "biufc":
            df["admitted"] = pd.to_numeric(df[adm_col], errors="coerce").fillna(0).astype(int)
        else:
            df["admitted"] = df[adm_col].map(_map_admitted_from_dispo).fillna(0).astype(int)

        return df[["time_in_system", "hour_of_day", "admitted"]].dropna(subset=["time_in_system","hour_of_day"])

    # If you get here, your run_model isn’t returning a DataFrame (e.g., a vector).
    raise TypeError(
        "run_model(...) must return a pandas DataFrame. "
        "Minimal columns required: ['time_in_system','hour_of_day','admitted'].\n"
        "Tip: return your patient-level results DF with these fields."
    )

def run_model_df(g, mu, sigma, total_runs=10) -> pd.DataFrame:
    raw = _raw_run_model(g, mu, sigma, total_runs=total_runs)
    return _normalise_sim_output(raw)

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

def breach_table(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["hour"] = parse_hour_col(tmp, "hour_of_day")
    tis = pd.to_numeric(tmp["time_in_system"], errors="coerce")
    tmp["breach"] = (tis > THRESHOLD_MIN).astype("Int64")
    g = (tmp.dropna(subset=["hour"])
            .groupby("hour", dropna=False)["breach"]
            .agg(k=lambda x: int((x == 1).sum()),
                 n=lambda x: int(x.notna().sum()))
            .reset_index())
    g = ensure_full_hours(g, on="hour")
    g["p"] = (g["k"] + 0.5) / (g["n"] + 1.0)
    return g

def arrivals_table(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["hour"] = parse_hour_col(tmp, "hour_of_day")
    g = (tmp.dropna(subset=["hour"])
            .groupby("hour", dropna=False)
            .size()
            .reset_index(name="n"))
    g = ensure_full_hours(g, on="hour")
    total = g["n"].sum()
    g["prop"] = (g["n"] / total) if total > 0 else 0.0
    return g

def admitted_mix_table(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["hour"] = parse_hour_col(tmp, "hour_of_day")
    tmp["admitted"] = pd.to_numeric(tmp["admitted"], errors="coerce").fillna(0).astype(int)
    g = (tmp.dropna(subset=["hour"])
            .groupby("hour", dropna=False)["admitted"]
            .agg(admitted_n=lambda x: int((x == 1).sum()),
                 n=lambda x: int(x.notna().sum()))
            .reset_index())
    g = ensure_full_hours(g, on="hour")
    g["prop_admitted"] = (g["admitted_n"] + 0.5) / (g["n"] + 1.0)
    return g

def binomial_nll(obs_tab: pd.DataFrame, sim_tab: pd.DataFrame, eps=1e-9) -> float:
    m = pd.merge(obs_tab[["hour","k","n"]],
                 sim_tab[["hour","p"]],
                 on="hour", how="left")
    m["p"] = m["p"].fillna(0.5).clip(eps, 1 - eps)
    return float((-(m["k"] * np.log(m["p"]) + (m["n"] - m["k"]) * np.log(1 - m["p"]))).sum())

def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    if p.sum() > 0: p = p / p.sum()
    if q.sum() > 0: q = q / q.sum()
    return 0.5 * np.abs(p - q).sum()

def admitted_wmse(obs_tab: pd.DataFrame, sim_tab: pd.DataFrame) -> float:
    m = pd.merge(obs_tab[["hour", "n", "prop_admitted"]],
                 sim_tab[["hour", "prop_admitted"]],
                 on="hour", how="left",
                 suffixes=("_obs", "_sim"))
    m["prop_admitted_sim"] = m["prop_admitted_sim"].fillna(0.5)
    w = m["n"].astype(float)
    diff2 = (m["prop_admitted_sim"] - m["prop_admitted_obs"]) ** 2
    denom = w.sum()
    return float((w * diff2).sum() / denom) if denom > 0 else float(diff2.mean())

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
observed_raw = pd.read_csv(OBS_PATH)

# Normalise observed to required columns
obs_df = pd.DataFrame()
# time_in_system
obs_tis_col = _first_existing(observed_raw, POSSIBLE_TIS + ["time_in_system"])
if obs_tis_col is None:
    raise KeyError("Observed data missing time-in-system column.")
obs_df["time_in_system"] = pd.to_numeric(observed_raw[obs_tis_col], errors="coerce")

# hour_of_day
obs_hour_col = _first_existing(observed_raw, POSSIBLE_HOUR + ["hour_of_day"])
if obs_hour_col is None:
    raise KeyError("Observed data missing hour column (e.g., 'hour_of_day').")
obs_df["hour_of_day"] = _coerce_hour_series(observed_raw[obs_hour_col])

# admitted
obs_adm_col = _first_existing(observed_raw, POSSIBLE_ADMIT + ["admitted"])
if obs_adm_col is None:
    raise KeyError("Observed data missing admitted/Disposition column.")
if observed_raw[obs_adm_col].dtype.kind in "biufc":
    obs_df["admitted"] = pd.to_numeric(observed_raw[obs_adm_col], errors="coerce").fillna(0).astype(int)
else:
    obs_df["admitted"] = observed_raw[obs_adm_col].map(_map_admitted_from_dispo).fillna(0).astype(int)

# Distribution checks (KS, mean)
obs_values = obs_df["time_in_system"].dropna().to_numpy()
obs_tis_summary = summarize_tis(obs_values)

# Hourly profiles from observed
obs_breach_hour = breach_table(obs_df)
obs_arrivals_hour = arrivals_table(obs_df)
obs_admitted_hour = admitted_mix_table(obs_df)

# ------------------------------ Run grid ----------------------------
os.makedirs(RESULTS_DIR, exist_ok=True)

grid_rows = []
tis_rows = []
breach_profile_rows = []
arrivals_profile_rows = []
admitted_profile_rows = []

for mu in mus:
    for sigma in sigmas:
        sim_df = run_model_df(global_params, mu, sigma, total_runs=TOTAL_RUNS)

        # Profiles
        sim_breach_hour   = breach_table(sim_df)
        sim_arrivals_hour = arrivals_table(sim_df)
        sim_admitted_hour = admitted_mix_table(sim_df)

        # Metrics
        hourly_nll = binomial_nll(obs_breach_hour, sim_breach_hour)
        arrival_tv = total_variation(obs_arrivals_hour["prop"].values,
                                     sim_arrivals_hour["prop"].values)
        adm_wmse   = admitted_wmse(obs_admitted_hour, sim_admitted_hour)

        # KS & mean diff for total distribution
        sim_values = sim_df["time_in_system"].dropna().to_numpy()
        if sim_values.size >= 2 and obs_values.size >= 2:
            ks_stat, ks_p = ks_2samp(sim_values, obs_values)
            sim_mean = float(sim_values.mean())
            mean_diff = abs(sim_mean - obs_tis_summary["mean"])
        else:
            ks_stat = ks_p = sim_mean = mean_diff = np.nan

        # Grid metrics row
        grid_rows.append({
            "mu": mu, "sigma": sigma,
            "hourly_nll": hourly_nll,
            "arrival_tv": arrival_tv,
            "admitted_prop_wmse": adm_wmse,
            "ks_stat": ks_stat, "ks_p": ks_p,
            "sim_mean": sim_mean, "obs_mean": obs_tis_summary["mean"],
            "mean_diff": mean_diff,
            "n_sim": int(sim_values.size)
        })

        # TIS summaries (dup observed per param for easy joins)
        sim_tis_summary = summarize_tis(sim_values)
        tis_rows.append({"mu": mu, "sigma": sigma, "label": "Observed", **obs_tis_summary})
        tis_rows.append({"mu": mu, "sigma": sigma, "label": "DES", **sim_tis_summary})

        # Breach profiles
        tmp_obs_b = obs_breach_hour.copy()
        tmp_obs_b.insert(0, "sigma", sigma); tmp_obs_b.insert(0, "mu", mu)
        tmp_obs_b.insert(2, "label", "Observed")
        breach_profile_rows.append(tmp_obs_b)

        tmp_sim_b = sim_breach_hour.copy()
        tmp_sim_b.insert(0, "sigma", sigma); tmp_sim_b.insert(0, "mu", mu)
        tmp_sim_b.insert(2, "label", "DES")
        breach_profile_rows.append(tmp_sim_b)

        # Arrivals per hour
        tmp_obs_a = obs_arrivals_hour.copy()
        tmp_obs_a.insert(0, "sigma", sigma); tmp_obs_a.insert(0, "mu", mu)
        tmp_obs_a.insert(2, "label", "Observed")
        arrivals_profile_rows.append(tmp_obs_a)

        tmp_sim_a = sim_arrivals_hour.copy()
        tmp_sim_a.insert(0, "sigma", sigma); tmp_sim_a.insert(0, "mu", mu)
        tmp_sim_a.insert(2, "label", "DES")
        arrivals_profile_rows.append(tmp_sim_a)

        # Admitted mix by hour
        tmp_obs_adm = obs_admitted_hour.copy()
        tmp_obs_adm.insert(0, "sigma", sigma); tmp_obs_adm.insert(0, "mu", mu)
        tmp_obs_adm.insert(2, "label", "Observed")
        admitted_profile_rows.append(tmp_obs_adm)

        tmp_sim_adm = sim_admitted_hour.copy()
        tmp_sim_adm.insert(0, "sigma", sigma); tmp_sim_adm.insert(0, "mu", mu)
        tmp_sim_adm.insert(2, "label", "DES")
        admitted_profile_rows.append(tmp_sim_adm)

        print(f"mu={mu:.3f}, sigma={sigma:.3f}, n={sim_values.size}, "
              f"hourly_nll={hourly_nll:.3f}, arrival_tv={arrival_tv:.3f}, "
              f"admitted_wmse={adm_wmse:.5f}, KS={ks_stat}, mean diff={mean_diff}")

# ------------------------------ Save all -----------------------------
pd.DataFrame(grid_rows).sort_values(
    by=["hourly_nll", "arrival_tv", "admitted_prop_wmse", "ks_stat"],
    ascending=[True, True, True, True]
).to_csv(GRID_PATH, index=False)

pd.DataFrame(tis_rows).to_csv(TIS_SUMMARY_PATH, index=False)
pd.concat(breach_profile_rows, ignore_index=True).to_csv(BREACH_PROFILES_PATH, index=False)
pd.concat(arrivals_profile_rows, ignore_index=True).to_csv(ARRIVALS_PROFILES_PATH, index=False)
pd.concat(admitted_profile_rows, ignore_index=True).to_csv(ADMITTED_MIX_PROFILES_PATH, index=False)

print("\nSaved:")
print(f"  - {GRID_PATH}")
print(f"  - {TIS_SUMMARY_PATH}")
print(f"  - {BREACH_PROFILES_PATH}")
print(f"  - {ARRIVALS_PROFILES_PATH}")
print(f"  - {ADMITTED_MIX_PROFILES_PATH}")
