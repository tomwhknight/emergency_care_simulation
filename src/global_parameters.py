class GlobalParameters:
    """Holds global parameters for the simulation."""
    
    def __init__(self, inter_arrival_time, triage_time, sdec_capacity, simulation_time, triage_nurse_capacity, random_seed):
        self.inter_arrival_time = inter_arrival_time
        self.triage_time = triage_time
        self.sdec_capacity = sdec_capacity
        self.simulation_time = simulation_time
        self.triage_nurse_capacity = triage_nurse_capacity
        self.random_seed = random_seed  # For reproducibility
