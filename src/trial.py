# trial.py

from src.model import Model

class Trial:
    def __init__(self, global_params):
        self.global_params = global_params

    def run(self, run_number):
        """Run the model for one simulation run."""
        model = Model(self.global_params)
        model.run()
        return model.results_df