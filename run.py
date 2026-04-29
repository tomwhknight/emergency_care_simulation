# run.py

MODEL_VARIANT = "baseline"   # "baseline", "alt_1", "alt_2"

if MODEL_VARIANT == "baseline":
    from src.trial import Trial
    print("Running simulation with original model logic.")
elif MODEL_VARIANT == "alt_1":
    from src.trial_alt import AltTrial as Trial
    print("Running simulation with Alt-1 logic.")
elif MODEL_VARIANT == "alt_2":
    from src.trial_alt2 import AltTrial2 as Trial
    print("Running simulation with Alt-2 logic.")
else:
    raise ValueError(f"Unknown MODEL_VARIANT: {MODEL_VARIANT}")


# Imports
import os
import time
from datetime import datetime
from os.path import basename

# Import shared setup + helpers
from src.base_params import (
    build_global_params,
    apply_alt2_defaults,
    MASTER_SEED,
    shift_patterns_weekday,
    shift_patterns_weekend,
)
from src.helper import save_rota_check, save_params


# Build shared global parameters
global_params = build_global_params()

# Apply alt_2-specific staffing defaults when needed
if MODEL_VARIANT == "alt_2":
    global_params = apply_alt2_defaults(global_params)


if __name__ == "__main__":
    trial = Trial(global_params, MASTER_SEED)

    t0 = time.perf_counter()
    info = trial.run(run_number=50)
    elapsed = time.perf_counter() - t0

    # Create folder for each batch and include ED service parameters
    mu = global_params.mu_ed_service_time
    sigma = global_params.sigma_ed_service_time
    mu_dec = global_params.mu_ed_decision_time
    sigma_dec = global_params.sigma_ed_decision_time

    batch_label = f"batch_mu{mu:.2f}_sigma{sigma:.2f}_mu{mu_dec:.2f}_sigma{sigma_dec:.2f}"
    batch_dir = os.path.join(info["scenario_dir"], batch_label)
    os.makedirs(batch_dir, exist_ok=True)

    # Print a quick summary to console
    total_runs = info.get("total_runs", 50)
    secs_per_run = elapsed / max(1, total_runs)

    print(
        f" Runtime: {elapsed:,.1f}s "
        f"({secs_per_run:.2f}s/run across {total_runs} runs)  "
        f"[seed={MASTER_SEED}]"
    )

    # Save params (JSON)
    params_filename = f"{basename(batch_dir)}.json"
    save_params(
        global_params,
        batch_dir,
        filename=params_filename,
        extra={
            "master_seed": MASTER_SEED,
            "created": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": round(elapsed, 3),
            "total_runs": total_runs,
            "seconds_per_run": round(secs_per_run, 3),
            "ed_service_params": {
                "mu_ed_service_time": mu,
                "sigma_ed_service_time": sigma,
                "mu_ed_decision_time": mu_dec,
                "sigma_ed_decision_time": sigma_dec,
            },
            "timing_windows": {
                "burn_in_time": global_params.burn_in_time,
                "cool_down_time": global_params.cool_down_time,
                "simulation_time": global_params.simulation_time,
            },
            "model_variant": MODEL_VARIANT,
        },
    )

    # Timings log
    timings_path = os.path.join(batch_dir, "timings.csv")
    header_needed = not os.path.exists(timings_path)

    with open(timings_path, "a", encoding="utf-8") as f:
        if header_needed:
            f.write(
                "timestamp,model_variant,master_seed,total_runs,"
                "elapsed_seconds,seconds_per_run,burn_in,cool_down,sim_time\n"
            )
        f.write(
            f"{datetime.now().isoformat(timespec='seconds')},"
            f"{MODEL_VARIANT},"
            f"{MASTER_SEED},"
            f"{total_runs},"
            f"{elapsed:.3f},"
            f"{secs_per_run:.3f},"
            f"{global_params.burn_in_time},"
            f"{global_params.cool_down_time},"
            f"{global_params.simulation_time}\n"
        )

    # Save rota checks into the batch folder
    save_rota_check(shift_patterns_weekday, batch_dir, filename="rota_weekday.csv")
    save_rota_check(shift_patterns_weekend, batch_dir, filename="rota_weekend.csv")