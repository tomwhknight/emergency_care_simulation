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

# Proportion of ELIGIBLE patients who actually take the direct pathway.
# 1.0 = all eligible go direct; 0.0 = none go direct.
DIRECT_UPTAKE_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]

TRIAL_MAP = {
    "alt": AltTrial,    # direct-to-medicine (when DT-eligible)
    "alt2": AltTrial2,  # direct-to-consultant attempt (when Alt2-eligible)
}

# Use the same base output root your Trial classes already use by default.
# Change this one line only if your normal des_output folder lives somewhere else.
DES_OUTPUT_DIR = "/Users/thomasknight/Local files/DES/output/des_output"

SENSITIVITY_ROOT = os.path.join(
    DES_OUTPUT_DIR,
    "sensitivity",
    "direct_pathway_uptake",
)


# -----------------------------
# 2. Helpers
# -----------------------------
def scenario_label(uptake_prob: float) -> str:
    return f"direct_uptake_{str(uptake_prob).replace('.', 'p')}"


def run_trial(trial, total_runs: int):
    """
    Run either AltTrial or AltTrial2.
    Some trial classes accept dt_threshold, some may not.
    """
    signature = inspect.signature(trial.run)
    if "dt_threshold" in signature.parameters:
        return trial.run(total_runs, dt_threshold=None)
    return trial.run(total_runs)


def run_single_scenario(model_name: str, trial_class, uptake_prob: float, total_runs: int):
    # Build baseline global params (keeps referral volumes the same)
    g = build_global_params()

    if model_name == "alt2":
        g = apply_alt2_defaults(g)

    # --- NEW: direct pathway uptake controls ---
    # Alt (direct-to-medicine) reads this:
    g.direct_medicine_uptake_prob = float(uptake_prob)

    # Alt2 (direct-to-consultant attempt) reads this:
    g.alt2_direct_consultant_uptake_prob = float(uptake_prob)

    sensitivity_name = scenario_label(uptake_prob)

    # This is the root that the trial class will write beneath.
    # Trial classes then append their own scenario folder + batch folder.
    scenario_output_root = os.path.join(SENSITIVITY_ROOT, sensitivity_name)
    os.makedirs(scenario_output_root, exist_ok=True)

    metadata = pd.DataFrame([
        {
            "model_name": model_name,
            "direct_pathway_uptake_prob": uptake_prob,
            "total_runs": total_runs,
            "master_seed": MASTER_SEED,
        }
    ])
    metadata.to_csv(
        os.path.join(scenario_output_root, f"{model_name}_scenario_parameters.csv"),
        index=False,
    )

    print(f"\nRunning {model_name} | direct_pathway_uptake_prob={uptake_prob}")

    trial = trial_class(
        global_params=g,
        master_seed=MASTER_SEED,
        base_output_dir=scenario_output_root,
    )

    result = run_trial(trial, total_runs)

    return {
        "model_name": model_name,
        "direct_pathway_uptake_prob": uptake_prob,
        "scenario_dir": result["scenario_dir"],
    }


# -----------------------------
# 3. Main
# -----------------------------
def main():
    os.makedirs(SENSITIVITY_ROOT, exist_ok=True)

    manifest_rows = []

    for model_name, trial_class in TRIAL_MAP.items():
        for uptake_prob in DIRECT_UPTAKE_VALUES:
            manifest_row = run_single_scenario(
                model_name=model_name,
                trial_class=trial_class,
                uptake_prob=uptake_prob,
                total_runs=TOTAL_RUNS,
            )
            manifest_rows.append(manifest_row)

    pd.DataFrame(manifest_rows).to_csv(
        os.path.join(SENSITIVITY_ROOT, "scenario_manifest.csv"),
        index=False,
    )

    print("\nDirect pathway uptake sensitivity complete.")
    print(f"Outputs saved under: {SENSITIVITY_ROOT}")


if __name__ == "__main__":
    main()