# model.py

import simpy
import random
import pandas as pd
from src.patient import Patient

class Model:
    def __init__(self, global_params):
        """1. Initialization: Set up environment, resources, and parameters."""
        self.global_params = global_params
        self.env = simpy.Environment()  # Create a SimPy environment within the model

        # AMU beds container
        self.amu_beds = simpy.Container(self.env, init=0)

        # ED doctor resource with FIFO queue
        self.ed_doctor = simpy.Resource(self.env, capacity=1)

        # Pandas DataFrame to store results
        self.results_df = pd.DataFrame(columns=["Patient ID", "Arrival Time", "Q Time Nurse", "Time with Nurse", "Assessment Time", "Disposition"])
        self.results_df.set_index("Patient ID", inplace=True)

    """2. Helper Functions, Generators, and Processes"""
    
    def generator_patient_arrivals(self):
        """Generate patient arrivals based on the inter-arrival time."""
        while True:
            patient_id = len(self.results_df) + 1
            patient = Patient(patient_id)
            patient.arrival_time = self.env.now

            # Start the ED assessment process for this patient
            self.env.process(self.ed_assessment(patient))

            # Wait for the next patient arrival
            inter_arrival_time = random.expovariate(1.0 / self.global_params.mean_patient_arrival_time)
            yield self.env.timeout(inter_arrival_time)

    def generator_amu_beds(self):
        """Generate AMU beds randomly based on the amu_bed_rate."""
        while True:
            bed_arrival_time = random.expovariate(1.0 / self.global_params.amu_bed_rate)
            yield self.env.timeout(bed_arrival_time)
            yield self.amu_beds.put(1)

    def ed_assessment(self, patient):
        """Simulate ED doctor assessment using FIFO."""
        with self.ed_doctor.request() as req:
            yield req

            assessment_time = random.expovariate(1.0 / self.global_params.mean_assessment_time)
            yield self.env.timeout(assessment_time)

            if random.random() < self.global_params.admission_probability:
                patient.disposition = 'admit'
                self.env.process(self.admission(patient))
            else:
                patient.disposition = 'discharge'

            # Store results in DataFrame
            self.results_df.loc[patient.id] = [patient.arrival_time, 0, 0, assessment_time, patient.disposition]

    def admission(self, patient):
        """Admit patient to AMU if bed available."""
        yield self.amu_beds.get(1)

    """3. Results Calculation Methods"""
    def calculate_run_results(self):
        """Calculates mean results for the run."""
        print(self.results_df)

    """4. Run Method"""
    def run(self):
        """Starts the simulation and runs until the end time."""
        self.env.process(self.generator_patient_arrivals())
        self.env.process(self.generator_amu_beds())
        self.env.run(until=self.global_params.simulation_time)
        self.calculate_run_results()