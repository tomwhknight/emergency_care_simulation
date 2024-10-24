# run.py

from src.global_parameters import GlobalParameters
from src.trial import Trial

if __name__ == "__main__":
    global_params = GlobalParameters(
        mean_patient_arrival_time=5,
        mean_assessment_time=10,
        admission_probability=0.5,
        simulation_time=100,
        amu_bed_rate=15
    )

    trial = Trial(global_params)
    for run_number in range(1, 6):
        print(f"Starting simulation run {run_number}...")
        trial.run(run_number)