import pandas as pd
from src.global_parameters import GlobalParameters
from src.model import Model

# Create GlobalParameters instance
params = GlobalParameters(
    sim_duration = 960, 
    trigger_lambda=1/20,                     # one trigger every 20 min on avg
    news2_lambda=0.2,
    task_lambda = 1/20,
    initial_tasks_mean=10,
    initial_tasks_stv=25,
    a_side_junior_capacity=1,
    f_side_junior_capacity=1,
    registrar_capacity=1,
    mean_green_amber_task_assessment_time=15,
    stdev_green_amber_task_assessment_time=5,
    mean_red_task_assessment_time=45,
    stdev_red_task_assessment_time=10,
    mean_registrar_assessment_time=30,
    stdev_registrar_assessment_time=8
)

# Set number of runs
n_runs = 5

# Run the simulation
for run_number in range(n_runs):
    model = Model(params, run_number=run_number + 1)
    model.run()
