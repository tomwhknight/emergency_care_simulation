import simpy
import random
import pandas as pd
from src.patient import Patient
from src.helper import calculate_hour_of_day, calculate_day_of_week

class Model:
    def __init__(self, global_params, run_number):
        """Initialize the model with the given global parameters."""
        self.env = simpy.Environment()
        self.global_params = global_params
        self.run_number = run_number
        self.patient_counter = 0
        self.run_results_df = pd.DataFrame(columns=[
            "Patient ID", "Arrival Time", "Day of Arrival", "Time of Arrival", "ED Assessment Time", "Time at End of ED Assessment", "Referral Time", "Time End Referral", "Initial Medical Assessment Time",
            "Time End Medical Assessment", "Consultant Assessment Time", "Time End Consultant Assessment", "Disposition", "Time in System"
        ])

        self.run_results_df.set_index("Patient ID", inplace=True)

        # Create resources

        self.triage_nurse = simpy.Resource(self.env, capacity=self.global_params.triage_nurse_capacity)
        self.ed_doctor = simpy.Resource(self.env, capacity=self.global_params.ed_doctor_capacity)
        self.medical_doctor = simpy.Resource(self.env, capacity=self.global_params.medical_doctor_capacity)
        self.consultant = simpy.Resource(self.env, capacity=self.global_params.consultant_capacity)

    def generate_patient_arrivals(self):
        """Generate patient arrivals based on inter-arrival times."""
        while True:
            self.patient_counter += 1
            arrival_time = self.env.now
            hour_of_arrival = calculate_hour_of_day(arrival_time)
            day_of_arrival = calculate_day_of_week(arrival_time)

            # create instance of patient class using

            patient = Patient(self.patient_counter, arrival_time, hour_of_arrival, day_of_arrival)

            self.run_results_df.loc[patient.id] = [patient.arrival_time, patient.day_of_arrival, patient.hour_of_arrival, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, '', 0.0]
            print(f"Patient {patient.id} arrives at {patient.arrival_time}")
            self.env.process(self.ed_assessment(patient))
            mean_patient_arrival_time = random.expovariate(1.0 / self.global_params.mean_patient_arrival_time)  # Example mean inter-arrival time of 5
            yield self.env.timeout(mean_patient_arrival_time)

    def ed_assessment(self, patient):
        """Simulate ED assessment and referral combined process."""
        with self.ed_doctor.request() as req:
            yield req  # Wait until a doctor is available
            ed_assessment_time = random.expovariate(1.0 / self.global_params.mean_ed_assessment_time)
            yield self.env.timeout(ed_assessment_time)
            self.run_results_df.at[patient.id, "ED Assessment Time"] = ed_assessment_time
            patient.ed_assessment_time = ed_assessment_time

             # Record time from arrival to end of assessment
            total_time_at_end_of_assessment = self.env.now - patient.arrival_time
            self.run_results_df.at[patient.id, "Time at End of ED Assessment"] = total_time_at_end_of_assessment

            print(f"Patient {patient.id} completes ED assessment at {self.env.now}")
            self.env.process(self.refer_to_medicine(patient))

    def refer_to_medicine(self, patient):
        """Simulate the process of referring a patient to medicine."""
        # Simulate a referral time or any required delay
        referral_time = random.expovariate(1.0 / self.global_params.mean_medical_referral)
        yield self.env.timeout(referral_time)

        # Record the referral process in the results
        self.run_results_df.at[patient.id, "Referral Time"] = referral_time
        print(f"Patient {patient.id} referred to medicine at {self.env.now}")

        # Calculate total time from arrival to the end of medical assessment
        total_time_referral = self.env.now - patient.arrival_time
        patient.total_time_referral = total_time_referral
        self.run_results_df.at[patient.id, "Time End Referral"] = total_time_referral

        # After referral, proceed to medical assessment
        self.env.process(self.initial_medical_assessment(patient))

    def initial_medical_assessment(self, patient):
        """Simulate initial medical assessment."""
        with self.medical_doctor.request() as req:
            yield req  # Wait until medical staff is available
            med_assessment_time = random.expovariate(1.0 / self.global_params.mean_initial_medical_assessment_time)
            yield self.env.timeout(med_assessment_time)
            patient.initial_medical_assessment_time = med_assessment_time
            self.run_results_df.at[patient.id, "Initial Medical Assessment Time"] = med_assessment_time
            print(f"Patient {patient.id} completes initial medical assessment at {self.env.now}")
            
            # Calculate total time from arrival to the end of medical assessment
            total_time_medical = self.env.now - patient.arrival_time
            patient.total_time_medical = total_time_medical
            self.run_results_df.at[patient.id, "Time End Medical Assessment"] = total_time_medical
            
            # Pass to next process
            self.env.process(self.consultant_assessment(patient))

    def consultant_assessment(self, patient):
        """Simulate consultant assessment process."""
        with self.consultant.request() as req:
            yield req  # Wait until a consultant is available
            consultant_assessment_time = random.expovariate(1.0 / self.global_params.mean_consultant_assessment_time)
            yield self.env.timeout(consultant_assessment_time)
            patient.consultant_assessment_time = consultant_assessment_time
            self.run_results_df.at[patient.id, "Consultant Assessment Time"] = consultant_assessment_time

            # Calculate total time from arrival to the end of consultant assessment
            total_time_consultant = self.env.now - patient.arrival_time
            patient.total_time_consultant = total_time_consultant
            self.run_results_df.at[patient.id, "Time End Consultant Assessment"] = total_time_consultant

            # Example disposition (admit or discharge)
            if random.random() < 0.5:
                patient.disposition = 'admitted'
            else:
                patient.disposition = 'discharged'

            self.run_results_df.at[patient.id, "Disposition"] = patient.disposition
            self.run_results_df.at[patient.id, "Time in System"] = self.env.now - patient.arrival_time
            print(f"Patient {patient.id} {patient.disposition} at {self.env.now}")

    def run(self):
        """Run the simulation."""
        self.env.process(self.generate_patient_arrivals())
        self.env.run(until=self.global_params.simulation_time)
        self.run_results_df["Run Number"] = self.run_number
        self.run_results_df.reset_index(inplace=True)