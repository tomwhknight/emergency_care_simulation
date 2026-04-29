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

REFERRAL_ODDS_MULTIPLIERS = [
    1.125, 1.375, 1.625, 1.875]

TRIAL_MAP = {
    "alt": AltTrial,
    "alt2": AltTrial2,
}

DES_OUTPUT_DIR = "/Users/thomasknight/Local files/DES/output/des_output"

SENSITIVITY_ROOT = os.path.join(
    DES_OUTPUT_DIR,
    "sensitivity",
    "referral_odds_multiplier",
)


# -----------------------------
# 2. Helpers
# -----------------------------
def scenario_label_from_multiplier(referral_odds_multiplier: float) -> str:
    return f"referral_odds_multiplier_{str(referral_odds_multiplier).replace('.', 'p')}"


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
    referral_odds_multiplier: float,
    total_runs: int,
):
    g = build_global_params()

    if model_name == "alt2":
        g = apply_alt2_defaults(g)

    # Turn referral sensitivity on and apply multiplier
    g.referral_sensitivity_on = True
    g.referral_odds_multiplier = referral_odds_multiplier

    sensitivity_name = scenario_label_from_multiplier(referral_odds_multiplier)

    scenario_output_root = os.path.join(SENSITIVITY_ROOT, sensitivity_name)
    os.makedirs(scenario_output_root, exist_ok=True)

    metadata = pd.DataFrame([
        {
            "model_name": model_name,
            "referral_sensitivity_on": True,
            "referral_odds_multiplier": referral_odds_multiplier,
            "total_runs": total_runs,
            "master_seed": MASTER_SEED,
        }
    ])
    metadata.to_csv(
        os.path.join(scenario_output_root, f"{model_name}_scenario_parameters.csv"),
        index=False,
    )

    print(f"\nRunning {model_name} | referral_odds_multiplier={referral_odds_multiplier}")

    trial = trial_class(
        global_params=g,
        master_seed=MASTER_SEED,
        base_output_dir=scenario_output_root,
    )

    result = run_trial(trial, total_runs)

    return {
        "model_name": model_name,
        "referral_odds_multiplier": referral_odds_multiplier,
        "scenario_dir": result["scenario_dir"],
    }


# -----------------------------
# 3. Main
# -----------------------------
def main():
    os.makedirs(SENSITIVITY_ROOT, exist_ok=True)

    manifest_rows = []

    for model_name, trial_class in TRIAL_MAP.items():
        for referral_odds_multiplier in REFERRAL_ODDS_MULTIPLIERS:
            manifest_row = run_single_scenario(
                model_name=model_name,
                trial_class=trial_class,
                referral_odds_multiplier=referral_odds_multiplier,
                total_runs=TOTAL_RUNS,
            )
            manifest_rows.append(manifest_row)

    pd.DataFrame(manifest_rows).to_csv(
        os.path.join(SENSITIVITY_ROOT, "scenario_manifest.csv"),
        index=False,
    )

    print("\nReferral odds multiplier sensitivity complete.")
    print(f"Outputs saved under: {SENSITIVITY_ROOT}")


if __name__ == "__main__":
    main()