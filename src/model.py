import simpy
import random
import pandas as pd
from src.patient import Patient
from src.helper import calculate_hour_of_day, calculate_day_of_week, extract_hour
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
        "Referral Time", "Arrival to Referral", "Time Joined AMU Queue", "Time Admitted AMU", "Initial Medical Assessment Time",  
        "Time Medical Assessment Complete", "Referral to Consultant", "Consultant Assessment Time", 
        "Arrival to Consultant Assessment", "Disposition", "Time in System", "Run Number"])
        self.run_results_df.index.name = "Patient ID"

        # Initialize DataFrame to monitor triage nurse queue
        self.triage_queue_monitoring_df = pd.DataFrame(columns=['Time', 'Queue Length'])
    
        # Initialize the DataFrame for tracking the AMU bed queue times
        self.amu_queue_df = pd.DataFrame(columns=["Patient ID", "Time Joined AMU Queue", "Time Admitted to AMU"])

        # Create simpy resources for staffing levels
        self.triage_nurse = simpy.Resource(self.env, capacity=self.global_params.triage_nurse_capacity)
        self.ed_doctor = simpy.Resource(self.env, capacity=self.global_params.ed_doctor_capacity)
        self.medical_doctor = simpy.Resource(self.env, capacity=self.global_params.medical_doctor_capacity)
        self.consultant = simpy.PriorityResource(self.env, capacity=self.global_params.consultant_capacity)

        # Initialize the AMU bed container
        self.amu_beds = simpy.Container(self.env, capacity=self.global_params.amu_bed_capacity, init=self.global_params.initial_amu_beds)

    def record_result(self, patient_id, column, value):
        """Helper function to record results only if the burn-in period has passed."""
        if self.env.now > self.burn_in_time:
            self.run_results_df.at[patient_id, column] = value

    # --- Generator Methods ---

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
            
            # Prepare the data for the patient row
            patient_row = [
                arrival_time, day_of_arrival, hour_of_arrival, 
                0.0, 0.0, 0.0, 0.0, 0.0,  # Triage-related columns
                0.0, 0.0,                  # ED Assessment-related columns
                0.0, 0.0, 0.0, 0.0,                 # Referral-related columns
                0.0, 0.0,                  # Medical Assessment-related columns
                0.0, 0.0, '', 0.0,         # Consultant-related columns, Disposition, and Time in System
                self.run_number            # Run number
            ]

            # Print the length of the row and the DataFrame to check for mismatches
            print(f"Row length: {len(patient_row)}")  # Should print 21
            print(f"Number of columns in run_results_df: {len(self.run_results_df.columns)}")  # Should also print 21
            print(f"Patient row values: {patient_row}")  # Show the actual row being added

            # Record the patient data in the results DataFrame
            self.run_results_df.loc[patient.id] = patient_row
           
            # Record patient arrival
            self.record_result(patient.id, "Arrival Time", arrival_time)
            self.record_result(patient.id, "Day of Arrival", day_of_arrival)
            self.record_result(patient.id, "Hour of Arrival", hour_of_arrival)
        
            # start triage process
            self.env.process(self.triage(patient))
            print(f"Patient {patient.id} arrives at {arrival_time}")

            ED_inter_arrival_time = random.expovariate(1.0 / self.global_params.mean_patient_arrival_time) 
            yield self.env.timeout(ED_inter_arrival_time)

    def generate_amu_beds(self):
        """Simulate generating AMU beds at a variable rate."""
        while True:
        # Wait for the next bed generation event, based on the rate (in minutes)
            yield self.env.timeout(self.global_params.amu_bed_generation_rate)

            # Add 1 bed to the AMU container
            if self.amu_beds.level < self.global_params.amu_bed_capacity: 
                self.amu_beds.put(1)  # Add 1 bed
                print(f"{self.env.now}: Added 1 AMU bed, total: {self.amu_beds.level}")
            else:
                print(f"{self.env.now}: AMU bed capacity reached: {self.amu_beds.level}")

    # Method to monitor the triage queue
    def monitor_triage_queue_length(self, interval=10):
        """Monitor the triage nurse queue length at regular intervals."""
        while True:
            # Record the current time and queue length
            current_time = self.env.now
            queue_length = len(self.triage_nurse.queue)
        
            # Create a new DataFrame for the current row
            new_row = pd.DataFrame({
            'Time': [current_time],
            'Queue Length': [queue_length]
            })
        
            # Concatenate the new row with the existing DataFrame
            self.triage_queue_monitoring_df = pd.concat([self.triage_queue_monitoring_df, new_row], ignore_index=True)
        
            # Wait for the specified interval before checking again
            yield self.env.timeout(interval)


    # Method to track AMU queue
    def monitor_amu_queue(self, interval=10):
        """Monitor the AMU bed queue length at regular intervals."""
        while True:
            current_time = self.env.now
            queue_length = self.amu_beds.level  # Length of AMU bed queue

            # Create a new DataFrame row for the queue length
            new_row = pd.DataFrame({
            'Time': [current_time], 'Queue Length': 
            [queue_length]
            })
            
            # Concatenate the new row to the existing DataFrame
            self.amu_queue_df = pd.concat([self.amu_queue_df, new_row], ignore_index=True)

            # Wait before checking again
            yield self.env.timeout(interval)

    # Method to model restricted consultant working hours 
    def obstruct_consultant(self):
        """Block the consultant resource during off-hours (21:00 to 07:00)."""
        while True:
            current_hour = extract_hour(self.env.now)

        # Check if we are outside working hours (21:00 to 07:00)
            if current_hour >= 21 or current_hour < 7:
                print(f"{self.env.now:.2f}: Consultants are off-duty.")
            
            # Request consultant with a high priority to block it
                with self.consultant.request(priority=-1) as req:
                    yield req  # Request the resource
                    off_hours_duration = (7 - current_hour) % 24 * 60  # Time until 07:00
                    print(f"{self.env.now:.2f}: Consultants will be back at {self.env.now + off_hours_duration}")
                    yield self.env.timeout(off_hours_duration)  # Block the resource

            # Wait until the next hour to check availability again
            yield self.env.timeout(60)

    # --- Processes (Patient Pathways) ---

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

        # After referral, proceed to medical assessment and request AMU bed
        self.env.process(self.initial_medical_assessment(patient)) # Continue medical assessment process in EDso t
        self.env.process(self.refer_to_amu_bed(patient))  # Start the process for AMU bed request
 
    
    def refer_to_amu_bed(self, patient):
        """Request a bed for the patient if available, or wait for one."""
        print(f"Patient {patient.id} requesting AMU bed at {self.env.now}")

        # Record the time when the patient joins the AMU bed queue
        patient.joined_amu_queue_time = self.env.now
        self.record_result(patient.id, "Time Joined AMU Queue", patient.joined_amu_queue_time)

       # Create a DataFrame for the new row to be added
        new_row = pd.DataFrame([{
        "Patient ID": patient.id,
        "Time Joined AMU Queue": patient.joined_amu_queue_time,
        "Time Admitted to AMU": None  # To be filled when the patient is admitted
        }])

        # Concatenate the new row to the existing DataFrame
        self.amu_queue_df = pd.concat([self.amu_queue_df, new_row], ignore_index=True)

        # Request an AMU bed
        yield self.amu_beds.get(1)

        # When the bed is available, admit the patient
        patient.amu_admission_time = self.env.now
        print(f"Patient {patient.id} admitted to AMU at {patient.amu_admission_time}")

        # Update the admission time in the queue log dataframe
        self.amu_queue_df.loc[self.amu_queue_df['Patient ID'] == patient.id, "Time Admitted to AMU"] = patient.amu_admission_time

        # Calculate the time spent in the system
        time_admission_amu = patient.amu_admission_time - patient.arrival_time
        patient.time_admission_amu = time_admission_amu  # Store total time in system
        self.record_result(patient.id, "Time Admitted AMU", time_admission_amu)  # Log total time in system
        print(f"Patient {patient.id} admitted to AMU: {time_admission_amu} minutes")

        # Exit the process for the patient
        
        print(f"Patient {patient.id} exits the system after AMU admission.")
        return  # Patient leaves after admission and no further processes are triggered

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

            # Record the consultant assessment time
            consultant_assessment_time = random.expovariate(1.0 / self.global_params.mean_consultant_assessment_time)
            yield self.env.timeout(consultant_assessment_time)
            
            # Record the consultant assessment time
            self.record_result(patient.id, "Consultant Assessment Time", consultant_assessment_time)
            patient.consultant_assessment_time = consultant_assessment_time

            # Calculate and record the total time from arrival to the end of consultant assessment
            total_time_consultant = self.env.now - patient.arrival_time
            self.record_result(patient.id, "Arrival to Consultant Assessment", total_time_consultant)
            
            # Example disposition (admit or discharge)
            if random.random() < 0.85:
                patient.disposition = 'admitted'
            else:
                patient.disposition = 'discharged'

            # Record the patient's disposition and total time in the system
            self.record_result(patient.id, "Disposition", patient.disposition)
            time_in_system = self.env.now - patient.arrival_time
            self.record_result(patient.id, "Time in System", time_in_system)

        print(f"Patient {patient.id} {patient.disposition} at {self.env.now}")
   
    # --- Run Method ---

    def run(self):
        """Run the simulation."""
        
        # Start the patient arrival process
        self.env.process(self.generate_patient_arrivals())

        # Start the consultant obstruction process
        self.env.process(self.obstruct_consultant())

        # Start monitoring the triage nurse queue
        self.env.process(self.monitor_triage_queue_length())
   
        # Start the AMU bed generation process
        self.env.process(self.generate_amu_beds())  

        # Start monitoring the AMU bed queue
        self.env.process(self.monitor_amu_queue()) 
    
        # Run the simulation
        self.env.run(until=self.global_params.simulation_time)
      