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
CONSULTANT_DISCHARGE_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

TRIAL_MAP = {
    "alt": AltTrial,
    "alt2": AltTrial2,
}

# Use the same base output root your Trial classes already use by default.
# Change this one line only if your normal des_output folder lives somewhere else.
DES_OUTPUT_DIR = "/Users/thomasknight/Local files/DES/output/des_output"

SENSITIVITY_ROOT = os.path.join(
    DES_OUTPUT_DIR,
    "sensitivity",
    "consultant_discharge",
)


# -----------------------------
# 2. Helpers
# -----------------------------
def scenario_label_from_prob(discharge_prob: float) -> str:
    return f"consultant_discharge_{str(discharge_prob).replace('.', 'p')}"



def run_trial(trial, total_runs: int):
    """
    Run either AltTrial or AltTrial2.
    Some trial classes accept dt_threshold, some may not.
    """
    signature = inspect.signature(trial.run)
    if "dt_threshold" in signature.parameters:
        return trial.run(total_runs, dt_threshold=None)
    return trial.run(total_runs)



def run_single_scenario(model_name: str, trial_class, discharge_prob: float, total_runs: int):
    g = build_global_params()

    if model_name == "alt2":
        g = apply_alt2_defaults(g)

    g.consultant_discharge_prob = discharge_prob

    sensitivity_name = scenario_label_from_prob(discharge_prob)

    # This is the root that the trial class will write beneath.
    # Trial classes then append their own scenario folder + batch folder.
    scenario_output_root = os.path.join(SENSITIVITY_ROOT, sensitivity_name)
    os.makedirs(scenario_output_root, exist_ok=True)

    metadata = pd.DataFrame([
        {
            "model_name": model_name,
            "consultant_discharge_prob": discharge_prob,
            "total_runs": total_runs,
            "master_seed": MASTER_SEED,
        }
    ])
    metadata.to_csv(
        os.path.join(scenario_output_root, f"{model_name}_scenario_parameters.csv"),
        index=False,
    )

    print(f"\nRunning {model_name} | consultant_discharge_prob={discharge_prob}")

    trial = trial_class(
        global_params=g,
        master_seed=MASTER_SEED,
        base_output_dir=scenario_output_root,
    )

    result = run_trial(trial, total_runs)

    return {
        "model_name": model_name,
        "consultant_discharge_prob": discharge_prob,
        "scenario_dir": result["scenario_dir"],
    }


# -----------------------------
# 3. Main
# -----------------------------
def main():
    os.makedirs(SENSITIVITY_ROOT, exist_ok=True)

    manifest_rows = []

    for model_name, trial_class in TRIAL_MAP.items():
        for discharge_prob in CONSULTANT_DISCHARGE_VALUES:
            manifest_row = run_single_scenario(
                model_name=model_name,
                trial_class=trial_class,
                discharge_prob=discharge_prob,
                total_runs=TOTAL_RUNS,
            )
            manifest_rows.append(manifest_row)

    pd.DataFrame(manifest_rows).to_csv(
        os.path.join(SENSITIVITY_ROOT, "scenario_manifest.csv"),
        index=False,
    )

    print("\nConsultant discharge sensitivity complete.")
    print(f"Outputs saved under: {SENSITIVITY_ROOT}")


if __name__ == "__main__":
    main()
