import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from run_calibration import run_model, global_params


# Load observed data (replace with your source)
observed = pd.read_csv("data/calibration/observed_system_time.csv")


# CLEAN the observed vector (do this right after loading)
obs_values = pd.to_numeric(observed["time_in_system"], errors="coerce").to_numpy()
obs_values = obs_values[np.isfinite(obs_values)]
obs_mean = float(obs_values.mean()) if len(obs_values) else np.nan

# Parameter grid
mus = np.arange(3.975, 3.985, 0.005)   
sigmas = np.arange(0.390, 0.400, 0.005) 

results = []

for mu in mus:
    for sigma in sigmas:
        sim_out = run_model(global_params, mu, sigma, total_runs=10)

        # Clean sim output
        sim_values = pd.to_numeric(pd.Series(sim_out), errors="coerce").to_numpy()
        sim_values = sim_values[np.isfinite(sim_values)]

        if len(sim_values) < 2 or len(obs_values) < 2:
            ks_stat = ks_p = sim_mean = mean_diff = np.nan
        else:
            ks_stat, ks_p = ks_2samp(sim_values, obs_values)
            sim_mean = float(sim_values.mean())
            mean_diff = abs(sim_mean - obs_mean)

        results.append({
            "mu": mu, "sigma": sigma,
            "ks_stat": ks_stat, "ks_p": ks_p,
            "sim_mean": sim_mean, "obs_mean": obs_mean,
            "mean_diff": mean_diff, "n_sim": len(sim_values)
        })

        print(f"mu={mu:.2f}, sigma={sigma:.2f}, n={len(sim_values)}, KS={ks_stat}, mean diff={mean_diff}")

# Save results
results_df = pd.DataFrame(results)
os.makedirs("src/results", exist_ok=True)
results_df.to_csv("src/results/calibration_results.csv", index=False)
print("Calibration results saved to src/results/calibration_results.csv")
