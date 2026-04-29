import inspect
import os
import pandas as pd

from src.base_params import build_global_params, apply_alt2_defaults, MASTER_SEED
from src.trial_alt import AltTrial
from src.trial_alt2 import AltTrial2


# -----------------------------
# 1. Settings
# -----------------------------
TOTAL_RUNS = 20

MU_VALUES = [3.40, 3.55, 3.70]
SIGMA_VALUES = [0.30, 0.40, 0.50]

TRIAL_MAP = {
    "alt": AltTrial,
    "alt2": AltTrial2,
}

# Use the same base output root your Trial classes already use by default.
DES_OUTPUT_DIR = "/Users/thomasknight/Local files/DES/output/des_output"

SENSITIVITY_ROOT = os.path.join(
    DES_OUTPUT_DIR,
    "sensitivity",
    "consultant_assessment_time_grid",
)


# -----------------------------
# 2. Helpers
# -----------------------------
def scenario_label(mu_consultant_assessment_time: float, sigma_consultant_assessment_time: float) -> str:
    mu_label = str(mu_consultant_assessment_time).replace(".", "p")
    sigma_label = str(sigma_consultant_assessment_time).replace(".", "p")
    return f"consultant_mu_{mu_label}_sigma_{sigma_label}"


def run_trial(trial, total_runs: int):
    """
    Run either AltTrial or AltTrial2.
    Some trial classes accept dt_threshold, some may not.
    """
    signature = inspect.signature(trial.run)
    if "dt_threshold" in signature.parameters:
        return trial.run(total_runs, dt_threshold=None)
    return trial.run(total_runs)


def run_single_scenario(
    model_name: str,
    trial_class,
    mu_consultant_assessment_time: float,
    sigma_consultant_assessment_time: float,
    total_runs: int,
):
    g = build_global_params()

    if model_name == "alt2":
        g = apply_alt2_defaults(g)

    g.mu_consultant_assessment_time = mu_consultant_assessment_time
    g.sigma_consultant_assessment_time = sigma_consultant_assessment_time

    sensitivity_name = scenario_label(
        mu_consultant_assessment_time,
        sigma_consultant_assessment_time,
    )

    # Trial classes then append their own scenario folder + batch folder.
    scenario_output_root = os.path.join(SENSITIVITY_ROOT, sensitivity_name)
    os.makedirs(scenario_output_root, exist_ok=True)

    metadata = pd.DataFrame([
        {
            "model_name": model_name,
            "mu_consultant_assessment_time": mu_consultant_assessment_time,
            "sigma_consultant_assessment_time": sigma_consultant_assessment_time,
            "total_runs": total_runs,
            "master_seed": MASTER_SEED,
        }
    ])
    metadata.to_csv(
        os.path.join(scenario_output_root, f"{model_name}_scenario_parameters.csv"),
        index=False,
    )

    print(
        f"\nRunning {model_name} | "
        f"mu_consultant_assessment_time={mu_consultant_assessment_time} | "
        f"sigma_consultant_assessment_time={sigma_consultant_assessment_time}"
    )

    trial = trial_class(
        global_params=g,
        master_seed=MASTER_SEED,
        base_output_dir=scenario_output_root,
    )

    result = run_trial(trial, total_runs)

    return {
        "model_name": model_name,
        "mu_consultant_assessment_time": mu_consultant_assessment_time,
        "sigma_consultant_assessment_time": sigma_consultant_assessment_time,
        "scenario_dir": result["scenario_dir"],
    }


# -----------------------------
# 3. Main
# -----------------------------
def main():
    os.makedirs(SENSITIVITY_ROOT, exist_ok=True)

    manifest_rows = []

    for model_name, trial_class in TRIAL_MAP.items():
        for mu_consultant_assessment_time in MU_VALUES:
            for sigma_consultant_assessment_time in SIGMA_VALUES:
                manifest_row = run_single_scenario(
                    model_name=model_name,
                    trial_class=trial_class,
                    mu_consultant_assessment_time=mu_consultant_assessment_time,
                    sigma_consultant_assessment_time=sigma_consultant_assessment_time,
                    total_runs=TOTAL_RUNS,
                )
                manifest_rows.append(manifest_row)

    pd.DataFrame(manifest_rows).to_csv(
        os.path.join(SENSITIVITY_ROOT, "scenario_manifest.csv"),
        index=False,
    )

    print("\nConsultant assessment time grid sensitivity complete.")
    print(f"Outputs saved under: {SENSITIVITY_ROOT}")


if __name__ == "__main__":
    main()