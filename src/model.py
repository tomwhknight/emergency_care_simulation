import simpy
import random
import pandas as pd
from src.patient import Patient
from src.helper import calculate_hour_of_day, calculate_day_of_week
from src.helper import Lognormal

class Model:
    def __init__(self, global_params, burn_in_time, run_number):
        """Initialize the model with the given global parameters."""
        self.env = simpy.Environment()
        self.global_params = global_params
        self.run_number = run_number
        self.burn_in_time = burn_in_time
        self.patient_counter = 0

        # Instantiate the Lognormal distribution for triage assessment time
        self.triage_time_distribution = Lognormal(mean=self.global_params.mean_triage_assessment_time,
                                                  stdev=self.global_params.stdev_triage_assessment_time)
        
        self.ed_time_distribution = Lognormal(mean=self.global_params.mean_ed_assessment_time, 
                                                   stdev=self.global_params.stdev_ed_assessment_time)


        self.run_results_df = pd.DataFrame(columns=[ "Arrival Time", "Day of Arrival", "Hour of Arrival", 
        "Time Triage Starts", "Wait for Triage", "Triage Assessment Time", "Triage Complete", "ED Assessment Time", "Completed ED Assessment", 
        "Referral Time", "Arrival to Referral", "Initial Medical Assessment Time",  
        "Time Medical Assessment Complete", "Referral to Consultant", "Consultant Assessment Time", 
        "Arrival to Consultant Assessment", "Disposition", "Time in System", "Run Number"])
        self.run_results_df.index.name = "Patient ID"
        
        # Create resources

        # print triage nurse capacity to check its value
        print(f"Triage Nurse Capacity: {self.global_params.triage_nurse_capacity}")

        self.triage_nurse = simpy.Resource(self.env, capacity=self.global_params.triage_nurse_capacity)
        self.ed_doctor = simpy.Resource(self.env, capacity=self.global_params.ed_doctor_capacity)
        self.medical_doctor = simpy.Resource(self.env, capacity=self.global_params.medical_doctor_capacity)
        self.consultant = simpy.Resource(self.env, capacity=self.global_params.consultant_capacity)

    def record_result(self, patient_id, column, value):
        """Helper function to record results only if the burn-in period has passed."""
        if self.env.now > self.burn_in_time:
            self.run_results_df.at[patient_id, column] = value

    def generate_patient_arrivals(self):
        """Generate patient arrivals based on inter-arrival times."""
        while True:
            self.patient_counter += 1
            arrival_time = self.env.now
            hour_of_arrival = calculate_hour_of_day(arrival_time)
            day_of_arrival = calculate_day_of_week(arrival_time)

            # create instance of patient class using
            patient = Patient(self.patient_counter, arrival_time, hour_of_arrival, day_of_arrival)

            # Initialize a row for this patient in the DataFrame
            # Add a new row to the DataFrame for a patient using .loc[]
            self.run_results_df.loc[patient.id] = [arrival_time, day_of_arrival, hour_of_arrival, 
            0.0, 0.0, 0.0, 0.0, 0.0,  # Triage-related columns
            0.0, 0.0,            # ED Assessment-related columns
            0.0, 0.0,            # Referral-related columns
            0.0, 0.0,            # Medical Assessment-related columns
            0.0, 0.0, '', 0.0,        # Consultant-related columns, Disposition, and Time in System
            self.run_number]     # Run number
           
            # Record patient arrival
            self.record_result(patient.id, "Arrival Time", arrival_time)
            self.record_result(patient.id, "Day of Arrival", day_of_arrival)
            self.record_result(patient.id, "Hour of Arrival", hour_of_arrival)
        
            # start triage process
            self.env.process(self.triage(patient))
            print(f"Patient {patient.id} arrives at {arrival_time}")

            ED_inter_arrival_time = random.expovariate(1.0 / self.global_params.mean_patient_arrival_time) 
            yield self.env.timeout(ED_inter_arrival_time)

    def triage(self, patient):
        start_triage_q = self.env.now
        """Simulate triage."""
        # Print queue length before making the request
        print(f"Triage nurse queue length before request: {len(self.triage_nurse.queue)}")
        print(f"Triage nurse capacity before request: {self.triage_nurse.capacity}")
        
        with self.triage_nurse.request() as req:
            print(f"Patient {patient.id} is requesting a nurse at {self.env.now}")
            yield req  # Wait until a triage is available
            
            # Print the queue length and capacity after the request
            print(f"Triage nurse capacity after request: {self.triage_nurse.capacity}")
            print(f"Triage nurse queue length after request: {len(self.triage_nurse.queue)}")

            # Time triage assessment begins
            end_triage_q = self.env.now
            patient.wait_time_for_triage = end_triage_q - start_triage_q
            
            # Record the time spent waiting for triage and when triage starts
            self.record_result(patient.id, "Wait for Triage", patient.wait_time_for_triage)
            self.record_result(patient.id, "Time Triage Starts", end_triage_q)

            # Print the updated queue length after the request
            print(f"Triage nurse queue length after request: {len(self.triage_nurse.queue)}")

        # Simulate the actual triage assessment time using the lognormal distribution
        triage_assessment_time = self.triage_time_distribution.sample()
        patient.triage_assessment_time = triage_assessment_time
        
        yield self.env.timeout(triage_assessment_time)
 
        self.record_result(patient.id, "Triage Assessment Time", patient.triage_assessment_time)
        
        # Calculate and record the total time from arrival to the end of triage
        patient.time_at_end_of_triage = self.env.now - patient.arrival_time
        self.record_result(patient.id, "Triage Complete", patient.time_at_end_of_triage)

        print(f"Patient {patient.id} completed triage at {self.env.now}")

        # Additional print for queue state after triage completion
        print(f"Triage nurse queue length after triage: {len(self.triage_nurse.queue)}")

        # Proceed to ED assessment after triage
        self.env.process(self.ed_assessment(patient))

    def ed_assessment(self, patient):
        """Simulate ED assessment."""
        with self.ed_doctor.request() as req:
            yield req  # Wait until a doctor is available
            
            # Simulate the actual triage assessment time using the lognormal distribution
            ed_assessment_time = self.ed_time_distribution.sample()
            yield self.env.timeout(ed_assessment_time)
            
            # Record ed assessment time in the results # 
            self.record_result(patient.id, "ED Assessment Time", ed_assessment_time)
            patient.ed_assessment_time = ed_assessment_time

            # Calculate and record the total time from arrival to the end of ED assessment
            time_at_end_of_ed_assessment = self.env.now - patient.arrival_time
            self.record_result(patient.id, "Completed ED Assessment", time_at_end_of_ed_assessment)

            print(f"Patient {patient.id} completes ED assessment at {self.env.now}")

        # Proceed to referral   
        self.env.process(self.refer_to_medicine(patient))

    def refer_to_medicine(self, patient):
        """Simulate the process of referring a patient to medicine."""
        # Simulate a referral time or any required delay
        referral_time = random.expovariate(1.0 / self.global_params.mean_medical_referral)
        yield self.env.timeout(referral_time)

        # Record the referral time
        self.record_result(patient.id, "Referral Time", referral_time)
        print(f"Patient {patient.id} referred to medicine at {self.env.now}")

        # Calculate and record the total time from arrival to the end of the referral
        total_time_referral = self.env.now - patient.arrival_time
        self.record_result(patient.id, "Arrival to Referral", total_time_referral)

        # Capture the time when the referral is completed
        patient.referral_end_time = self.env.now  # Store this timestamp for later use

        # After referral, proceed to medical assessment
        self.env.process(self.initial_medical_assessment(patient))

    def initial_medical_assessment(self, patient):
        """Simulate initial medical assessment."""
        with self.medical_doctor.request() as req:
            yield req  # Wait until medical staff is available
            
            med_assessment_time = random.expovariate(1.0 / self.global_params.mean_initial_medical_assessment_time)
            yield self.env.timeout(med_assessment_time)
            
            # Record the initial medical assessment time
            self.record_result(patient.id, "Initial Medical Assessment Time", med_assessment_time)
            patient.initial_medical_assessment_time = med_assessment_time
            print(f"Patient {patient.id} completes initial medical assessment at {self.env.now}")
            
            # Calculate total time from arrival to the end of medical assessment
            total_time_medical = self.env.now - patient.arrival_time
            patient.total_time_medical = total_time_medical
            self.record_result(patient.id, "Time Medical Assessment Complete",  total_time_medical)
            
        # Pass to next process
        self.env.process(self.consultant_assessment(patient))

    def consultant_assessment(self, patient):
        """Simulate consultant assessment process."""
        with self.consultant.request() as req:
            yield req  # Wait until a consultant is available
            end_consultant_q = self.env.now

            # Calculate the waiting time from end of referral to start of consultant assessment
            wait_for_consultant = end_consultant_q - patient.referral_end_time
            self.record_result(patient.id, "Referral to Consultant", wait_for_consultant)

            consultant_assessment_time = random.expovariate(1.0 / self.global_params.mean_consultant_assessment_time)
            yield self.env.timeout(consultant_assessment_time)
            
            # Record the consultant assessment time
            self.record_result(patient.id, "Consultant Assessment Time", consultant_assessment_time)
            patient.consultant_assessment_time = consultant_assessment_time


            # Calculate and record the total time from arrival to the end of consultant assessment
            total_time_consultant = self.env.now - patient.arrival_time
            self.record_result(patient.id, "Arrival to Consultant Assessment", total_time_consultant)
            
            # Example disposition (admit or discharge)
            if random.random() < 0.5:
                patient.disposition = 'admitted'
            else:
                patient.disposition = 'discharged'

            # Record the patient's disposition and total time in the system
            self.record_result(patient.id, "Disposition", patient.disposition)
            time_in_system = self.env.now - patient.arrival_time
            self.record_result(patient.id, "Time in System", time_in_system)

        print(f"Patient {patient.id} {patient.disposition} at {self.env.now}")
   
    def run(self):
        """Run the simulation."""
        self.env.process(self.generate_patient_arrivals())
        self.env.run(until=self.global_params.simulation_time)
      