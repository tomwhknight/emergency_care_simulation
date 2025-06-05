import optuna
import numpy as np
import pandas as pd

# Allow Python to find src/ when running directly
import sys
import os
sys.path.append("/Users/thomasknight/Desktop/ACL/Projects/emergency_care_simulation")

from scipy.stats import ks_2samp
from scipy.stats import wasserstein_distance
from run import run_model_with_params  

# §1: Load or define your observed time-in-system data
# Replace this with actual data in your environment
# Example placeholder:
observed_df = pd.read_csv("data/results/observed_time_in_system.csv")
observed_time_in_system = observed_df["time_in_system"].dropna().to_numpy()
print(observed_df.columns)

# Pre-calculate observed stats
obs_mean = np.mean(observed_time_in_system)
obs_sd = np.std(observed_time_in_system)

# §2: Objective function with combined loss
def objective(trial):
    param_dict = {
        "mu_ed_assessment_discharge": trial.suggest_float("mu_ed_assessment_discharge", 3.5, 4.5), 
        "sigma_ed_assessment_discharge": trial.suggest_float("sigma_ed_assessment_discharge", 0.5, 1.5), 

        "wb_shape_ed_assessment_admit": trial.suggest_float("wb_shape_ed_assessment_admit", 1.2, 1.8),
        "wb_scale_ed_assessment_admit": trial.suggest_float("wb_scale_ed_assessment_admit", 50, 200),

        "mu_ed_service_time": trial.suggest_float("mu_ed_service_time", 4.0, 4.5),
        "sigma_ed_service_time": trial.suggest_float("sigma_ed_service_time", 0.75, 1.25),
        "min_ed_service_time": 30,
        "max_ed_service_time": 120
    }


    #  Run the model
    run_results = run_model_with_params(param_dict)
    sim_times = run_results["Time in System"].dropna()

    # Empty output
    if len(sim_times) == 0:
        return float("inf")

     # 1. Earth Mover's Distance (shape)
    emd = wasserstein_distance(sim_times, observed_time_in_system)

    # 2. Quantile error
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantile_error = np.mean([
        ((np.quantile(sim_times, q) - np.quantile(observed_time_in_system, q)) /
         np.quantile(observed_time_in_system, q)) ** 2
        for q in quantiles
    ])

    # 3. 4-hour breach error
    breach_threshold = 240  # minutes
    sim_breach = np.mean(sim_times > breach_threshold)
    obs_breach = np.mean(observed_time_in_system > breach_threshold)
    breach_error = ((sim_breach - obs_breach) / obs_breach) ** 2

    # §4: Weighted loss
    loss = (
        0.4 * emd +
        0.3 * quantile_error +
        0.3 * breach_error
    )

    return loss

# Run optimisation
def run_optimisation(n_trials=50):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("✅ Best parameters:")
    print(study.best_params)
    print("📉 Best loss:")
    print(study.best_value)

    # Step 3: Extract results as DataFrame
    optima_results = study.trials_dataframe(attrs=("number", "value", "params", "state"))

    # Step 4: Add metadata
    optima_results["source"] = "Optuna DES optimisation"
    optima_results["timestamp"] = pd.Timestamp.now()

    # Step 5: Sort by lowest loss
    optima_results = optima_results.sort_values("value").reset_index(drop=True)

    # Step 6: Save to file
    optima_results.to_csv("optima_results.csv", index=False)

    return optima_results

if __name__ == "__main__":
    run_optimisation(n_trials=500)
