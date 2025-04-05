import simpy
import random
import pandas as pd
import numpy as np
from src.patient import Patient
from src.helper import calculate_hour_of_day, extract_day_of_week, extract_hour
from src.helper import Lognormal

class Model:
    def __init__(self, global_params, burn_in_time, run_number):
        """Initialize the model with the given global parameters."""
        self.env = simpy.Environment()
        self.global_params = global_params
        self.run_number = run_number
        self.burn_in_time = burn_in_time
        self.patient_counter = 0
        self.total_beds_generated = 0 

        # Load CSV data
        self.ed_staffing_data = pd.read_csv(self.global_params.ed_staffing_file)
        self.medical_staffing_data = pd.read_csv(self.global_params.medicine_staffing_file)
        self.amu_bed_rate_data = pd.read_csv(self.global_params.amu_bed_rate_file)
        self.arrival_rate_data = pd.read_csv(self.global_params.arrival_rate_file)

        # Instantiate the Lognormal distribution for triage assessment time
        self.triage_time_distribution = Lognormal(mean=self.global_params.mean_triage_assessment_time,
                                                  stdev=self.global_params.stdev_triage_assessment_time)

        self.ed_assessment_time_distribution = Lognormal(mean=self.global_params.mean_ed_assessment_time,
                                                  stdev=self.global_params.stdev_ed_assessment_time)
        
        self.medical_assessment_time_distribution = Lognormal(mean=self.global_params.mean_initial_medical_assessment_time,
                                                  stdev=self.global_params.stdev_initial_medical_assessment_time)
        

        self.consultant_time_distribution = Lognormal(mean=self.global_params.mean_consultant_assessment_time, 
                                                   stdev=self.global_params.stdev_consultant_assessment_time)
        
        self.sdec_time_distribution = Lognormal(mean=self.global_params.mean_sdec_assessment_time, 
                                                   stdev=self.global_params.stdev_sdec_assessment_time)
        
        # Create results DF

         # Define standard columns, structured by category
        self.standard_cols = [
            # --- Arrival Information ---
            "Simulation Arrival Time",
            "Day of Arrival",
            "Clock Hour of Arrival",
            "Hour of Arrival",
            "Source of Referral"
            "Mode of Arrival",
            "Acuity",

            # --- Triage Information ---
            "Triage Location",
            "Queue Length Ambulance Triage Nurse",
            "Queue Length Walk in Triage Nurse",
            "Wait for Triage Nurse",
            "Triage Nurse Assessment Start Time",
            "Triage Nurse Assessment Service Time"
            "Triage Complete",
            "Triage Outcome",

            # --- SDEC Referral Information ---
            "SDEC Accepted",
            "SDEC Decision Reason",

            # --- ED assessment ---

            "Wait ED Assessment Time",
            "Queue Length ED doctor",
            "ED Assessment Start Time",
            "ED Assessment Time",
            "Completed ED Assessment",

            # --- Referral to Medicine ---
            "Simulation Referral Time",
            "Arrival to Referral",

            # --- AMU Admission ---
            "Time Joined AMU Queue",
            "Time Admitted to AMU",

            # --- Medical Assessment ---
            "Queue Length Medical Doctor",
            "Simulation Time Medical Assessment Starts",
            "Wait for Medical Assessment",
            "Initial Medical Assessment Time",
            "Arrival to Medical Assessment",

            # --- Consultant Review ---
            "Queue Length Consultant",
            "Simulation Time Added PTWR queue",
            "Simulation Time Consultant Assessment Starts",
            "Referral to Consultant Assessment",
            "Consultant Assessment Time",
            "Arrival to Consultant Assessment",

            # --- Discharge Information ---
            "Discharge Time",
            "Discharge Decision Point",
            "Time in System",

            # --- Run Information ---
            "Run Number"
            ]   

        # Create results DataFrame with structured standard columns
        self.run_results_df = pd.DataFrame(columns=self.standard_cols)
        self.run_results_df.index.name = "Patient ID"
            
        # Initialize DataFrame to monitor triage nurse queue
        self.triage_queue_monitoring_df = pd.DataFrame(columns=['Simulation Time', 'Hour of Day', 'Queue Length'])
    
        # Initialize DataFrame to monitor consultant queue
        self.consultant_queue_monitoring_df = pd.DataFrame(columns=['Simulation Time', 'Hour of Day', 'Queue Length'])
    
        # Initialize the DataFrame for tracking the AMU bed queue times
        self.amu_queue_df = pd.DataFrame(columns=["Patient ID", "Time Joined AMU Queue", "Time Admitted to AMU"])

        # Create simpy resources for staffing levels
        self.ambulance_triage_nurse = simpy.Resource(self.env, capacity=self.global_params.ambulance_triage_nurse_capacity)
        self.walk_in_triage_nurse = simpy.Resource(self.env, capacity = self.global_params.walk_in_triage_nurse_capacity)

        # Create simpy resources for ED clinical assessment
        self.ed_doctor = simpy.PriorityResource(self.env, capacity=self.global_params.ed_doctor_capacity)
        self.medical_doctor = simpy.PriorityResource(self.env, capacity=self.global_params.medical_doctor_capacity)
        self.consultant = simpy.PriorityResource(self.env, capacity=self.global_params.consultant_capacity)

        # Initialize the AMU bed container
        self.amu_beds = simpy.Store(self.env, capacity = self.global_params.max_amu_available_beds)

        # Initialize the SDEC capacity container
        self.sdec_capacity = simpy.Store(self.env, capacity = self.global_params.max_sdec_capacity)

    # Method to add results to the results dataframe
    def record_result(self, patient_id, column, value):

        """Helper function to record results only if the burn-in period has passed."""

        if self.env.now > self.burn_in_time:
            if column not in self.run_results_df.columns:
                print(f"Warning: Attempting to add a new column '{column}'. Ignoring update.")
                return  # Ignore the update if the column does not exist
            self.run_results_df.at[patient_id, column] = value
    
    # --- Generator Methods --
    def generate_arrivals(self):
        """Generate patient arrivals based on inter-arrival times."""
    
        while True:
            self.patient_counter += 1  # Shared counter across generators
            arrival_time = self.env.now
        
            # Add time variables
            arrival_clock_time = calculate_hour_of_day(arrival_time)
            day_of_arrival = extract_day_of_week(arrival_time)
            current_day = extract_day_of_week(self.env.now)
            current_hour = extract_hour(self.env.now)

            
            mode_of_arrival = random.choices(["Ambulance", "Walk-in"], 
                                         weights=[self.global_params.ambulance_proportion, 
                                                  self.global_params.walk_in_proportion])[0]

            # Explicitly assign source of referral

            
            # Assign mode of arrival and admission probability based on mode of arrival
            if mode_of_arrival == "Ambulance":
                acuity_levels = list(self.global_params.ambulance_acuity_probabilities.keys())
                acuity_weights = list(self.global_params.ambulance_acuity_probabilities.values())
            else:
                acuity_levels = list(self.global_params.walk_in_acuity_probabilities.keys())
                acuity_weights = list(self.global_params.walk_in_acuity_probabilities.values())

            acuity = random.choices(acuity_levels, weights=acuity_weights, k=1)[0]


            # Explicitly assign source of referral
            source_of_referral = random.choices(
                ["GP", "ED"],
            weights=[
                self.global_params.proportion_direct_primary_care,       # 0.07
                1 - self.global_params.proportion_direct_primary_care    # 0.93
                ]
            )[0]

            # Create instance of patient class
            patient = Patient(
            self.patient_counter,
            arrival_time,
            day_of_arrival,
            arrival_clock_time,
            current_hour,
            source_of_referral,
            mode_of_arrival, 
            acuity)

            # Initialise a dictionary of patient results 
            patient_results = {
            # --- Arrival Information ---
            "Patient ID": patient.id,
            "Simulation Arrival Time": arrival_time,
            "Day of Arrival": day_of_arrival,
            "Clock Hour of Arrival": arrival_clock_time,
            "Hour of Arrival": current_hour,
            "Mode of Arrival": mode_of_arrival,
            "Source of Referral": source_of_referral,
            "Acuity": acuity,
            

            # --- Triage-Related Metrics ---
            "Queue Lenght Walk-in Triage Nurse": 0.0,
            "Queue Lenght Ambulance Triage Nurse": 0.0,
            "Wait for Triage Nurse": 0.0,
            "Triage Nurse Assessment Start Time": 0.0,
            "Triage Nurse Assessment Service Time": 0.0,
            "Triage Complete": 0.0,
            "Triage Outcome": "",

            # --- SDEC Referral ---
            "SDEC Accepted": "",
            "SDEC Decision Reason": "",

            # --- ED Assessment Metrics ---
            "Queue Length ED doctor": 0.0,
            "Wait ED Assessment Time": 0.0,
            "ED Assessment Start Time": 0.0,
            "ED Assessment Service Time": 0.0,
            "Completed ED Assessment": 0.0,

            # --- Referral to Medicine ---
            "Simulation Referral Time": 0.0,
            "Arrival to Referral": 0.0,

            # --- AMU Process ---
            "Time Joined AMU Queue": 0.0,
            "Time Admitted to AMU": 0.0,

            # --- Medical Assessment Process ---
            "Queue Length Medical Doctor": 0.0,
            "Simulation Time Medical Assessment Starts": 0.0,
            "Wait for Medical Assessment": 0.0,
            "Initial Medical Assessment Time": 0.0,
            "Arrival to Medical Assessment": 0.0,

            # --- Consultant Review Process ---
            "Queue Length Consultant": 0.0,
            "Simulation Time Added PTWR queue": 0.0,
            "Simulation Time Consultant Assessment Starts": 0.0,
            "Referral to Consultant Assessment": 0.0,
            "Consultant Assessment Time": 0.0,
            "Arrival to Consultant Assessment": 0.0,

            # --- Discharge Information ---
            "Discharge Time": "",
            "Discharge Decision Point": "",
            "Time in System": 0.0,

            # --- Simulation Run Number ---
            "Run Number": self.run_number
            }

            # Ensure all columns from `self.standard_cols` exist
            for col in self.standard_cols:
                if col not in patient_results:
                    patient_results[col] = float('nan')  # Assign NaN if column is missing

            # Append the patient results as a row, keeping `patient.id` as the index
            self.run_results_df = pd.concat(
                [self.run_results_df, pd.DataFrame.from_records([patient_results]).set_index("Patient ID")],
                ignore_index=False
                )

            # Record patient arrival
            self.record_result(patient.id, "Simulation Arrival Time", patient.arrival_time)
            self.record_result(patient.id, "Day of Arrival", patient.day_of_arrival)
            self.record_result(patient.id, "Clock Hour of Arrival", patient.arrival_clock_time)
            self.record_result(patient.id, "Hour of Arrival", patient.current_hour)
            
            # Assign patient to correct triage process
            if mode_of_arrival == "Ambulance":
                print(f"Ambulance Patient {patient.id} arrives at {arrival_time}")
                self.env.process(self.ambulance_triage(patient))  # Send to ambulance triage
            else:
                print(f"Walk-in Patient {patient.id} arrives at {arrival_time}")
                self.env.process(self.walk_in_triage(patient))  # Send to walk-in triage

            # Get the mean arrival rate for day and hour
            mean_arrival_rate = self.arrival_rate_data.loc[
            (self.arrival_rate_data['hour'] == current_hour) & (self.arrival_rate_data ['day'] == current_day), 'mean_arrivals_per_min'
            ].values[0]

            # Sample time until next arrival
            arrival_interval = random.expovariate(mean_arrival_rate)
            yield self.env.timeout(arrival_interval)

    # Method to generate AMU beds
    def generate_amu_beds(self):

        """Periodically release beds based on a Poisson distribution."""
        while True:

        # Extract the current hour
            current_hour = extract_hour(self.env.now)
            current_day = extract_day_of_week(self.env.now)

            # Get the mean for day and hour
            mean_amu_bed_rate = self.amu_bed_rate_data.loc[
            (self.amu_bed_rate_data['hour'] == current_hour) & (self.amu_bed_rate_data ['day'] == current_day), 'mean_beds_available_per_min'
            ].values[0]

            # Sample time until next bed release using an exponential distribution
            amu_bed_release_interval = random.expovariate(mean_amu_bed_rate)
            yield self.env.timeout(amu_bed_release_interval)

            # Add a bed if there is space in the store
            if len(self.amu_beds.items) < self.amu_beds.capacity:
                yield self.amu_beds.put("Bed")
                self.total_beds_generated += 1  # Increment counter
                print(f"Bed added to AMU at {self.env.now}. Total beds available: {len(self.amu_beds.items)}")
            else: print(f"No space to add more beds at {self.env.now}.")
    
    # Method to generate SDEC capacity
    def generate_sdec_capacity(self):
        # Determine if it's a weekend
            current_day = extract_day_of_week(self.env.now)  # e.g., "Monday", "Saturday"
            is_weekend = current_day in ["Saturday", "Sunday"]

             # Fetch the base capacity based on the day type
            base_capacity = (
                self.global_params.weekday_sdec_base_capacity if not is_weekend 
                else self.global_params.weekend_sdec_base_capacity
        )

            # Add random noise to capacity
            random_variation = random.randint(-2, 2)  # Random noise of ±2 slots
            sdec_capacity = max(1, base_capacity + random_variation)  # Ensure capacity is non-negative

            # Clear existing SDEC capacity
            self.sdec_capacity.items = []  # Empty the store
            for _ in range(sdec_capacity):  # Refill with new capacity
                self.sdec_capacity.put("token")

            print(f"SDEC capacity reset to {sdec_capacity} slots on {current_day} at time {self.env.now:.2f}.")

            # Dynamically add slots throughout the day
            while extract_day_of_week(self.env.now) == current_day:  # Stay within the same day
                # Sample time until the next slot release
                sdec_capacity_release_interval = random.expovariate(1.0 / self.global_params.mean_sdec_capacity_release_interval)
                yield self.env.timeout(sdec_capacity_release_interval)

                # Add a slot if there's space in the store
                if len(self.sdec_capacity.items) < self.sdec_capacity.capacity:
                    self.sdec_capacity.put("token")
                    print(f"Slot added to SDEC at {self.env.now:.2f}. Total slots available: {len(self.sdec_capacity.items)}")
                else:
                    print(f"No space to add more slots to SDEC at {self.env.now:.2f}.")

                # Wait until the next day to reset the capacity
                next_day_start = (self.env.now // 1440 + 1) * 1440
                yield self.env.timeout(next_day_start - self.env.now)
    
    # Method to monitor the consultant queue
    def monitor_consultant_queue_length(self, interval=60):
        """Monitor consultant queue length at regular intervals."""
        while True:
        # Record the current time and queue length
            current_time = self.env.now
            queue_length = len(self.consultant.queue)
            hour_of_day = (current_time // 60) % 24
        
            # Print the current queue status for debugging
            print(f"[{self.env.now:.2f}] Monitoring Consultant Queue: {queue_length} patients waiting at hour {hour_of_day}")

        # Check if there are any patients currently in the queue
            if queue_length > 0:
                print(f"Consultant Queue Status: {queue_length} patients are still in the queue.")
                # Print patient IDs currently in the queue
                patient_ids = [req.priority for req in self.consultant.queue]
                print(f"Patients in Consultant Queue: {patient_ids}")
            else:
                print("Consultant Queue Status: Queue is empty.")

            # Create a new DataFrame for the current row
            new_row = pd.DataFrame({
                'Simulation Time': [current_time],
                'Hour of Day': [hour_of_day],
                'Queue Length': [queue_length]
            })

            # Concatenate the new row with the existing DataFrame
            self.consultant_queue_monitoring_df = pd.concat([self.consultant_queue_monitoring_df, new_row], ignore_index=True)
        
            # Wait for the specified interval before checking again
            yield self.env.timeout(interval)

    # Method to monitor AMU queue
    def monitor_amu_queue(self, interval=15):
        """Monitor the AMU bed queue length at regular intervals."""
        while True:
            current_time = self.env.now
            queue_length = len(self.amu_beds.items)  # Length of AMU bed queue

            # Create a new DataFrame row for the queue length
            new_row = pd.DataFrame({
            'Time': [current_time], 'Queue Length': 
            [queue_length]
            })
            
            # Concatenate the new row to the existing DataFrame
            self.amu_queue_df = pd.concat([self.amu_queue_df, new_row], ignore_index=True)

            # Wait before checking again
            yield self.env.timeout(interval)

    # --- Dynamic resource modelling ---

    # Method to model consultant working hours 
    def obstruct_consultant(self):
        """Simulate consultant unavailability between 21:00 and 07:00."""
        while True:
            # Extract the current hour
            current_hour = extract_hour(self.env.now)

            # Check if the current time is within the off-duty period (21:00–07:00)
            if current_hour >= 21 or current_hour < 7:
                print(f"{self.env.now:.2f}: Consultants are off-duty (21:00–07:00).")
                with self.consultant.request(priority=-1) as req:
                    yield req  # Block the resource
                    yield self.env.timeout(60)  # Hold the block for 1 hour
            else:
                print(f"{self.env.now:.2f}: Consultants are available.")

            # Wait until the next hour to check again
            yield self.env.timeout(60)
    
    # Method to model ED doctor working hours
    def obstruct_ed_doctor(self):
        """Simulate ED doctor unavailability based on shift patterns."""
        while True:
            # Extract the current hour
            current_hour = extract_hour(self.env.now)
        
            # Get the number of doctors available for the current hour
            available_doctors = self.ed_staffing_data.loc[
            self.ed_staffing_data['hour'] == current_hour, 'num_staff'
            ].values[0]
            print(f"Number of doctors available at hour {current_hour}: {available_doctors}")

            # Calculate the number of doctors to block
            ed_doctors_to_block = self.ed_doctor.capacity - available_doctors
           
            # Block excess doctors
            if ed_doctors_to_block > 0:
                print(f"{self.env.now:.2f}: Blocking {ed_doctors_to_block} ED doctors for hour {current_hour}.")
                for _ in range(ed_doctors_to_block):
                    self.env.process(self.block_doctor(60))  # Block each doctor for 1 hour
            else:
                print(f"{self.env.now:.2f}: No blocking required; all doctors available.")

            # Wait for the next hour to recheck staffing
            yield self.env.timeout(60)

    # Method to block ED doctor 
    def block_doctor(self, block_duration):
        """Simulate blocking a single doctor for a specific duration."""
        with self.ed_doctor.request(priority=-1) as req:
            yield req  # Acquire the resource to simulate it being blocked   
            yield self.env.timeout(block_duration)  # Simulate the blocking period

    # Method to replicate doctor breaks
    def doctor_break_cycle(self):
        """Randomly select a proportion of the currently available ED doctors for a 30-minute break every hour."""
        while True:
            # Step 1: Wait for the next break cycle (every 60 minutes)
            yield self.env.timeout(60)

            # Step 2: Get the current simulation hour
            current_hour = int(self.env.now // 60) % 24  # Convert simulation time (mins) to hour of the day

            # Step 3: Get the number of doctors currently available (shift-adjusted)
            available_doctors = self.ed_staffing_data.loc[
                self.ed_staffing_data['hour'] == current_hour, 'num_staff'
                ].values[0]  # Lookup the correct number of doctors on shift

            print(f"{self.env.now:.2f}: Doctors on shift at hour {current_hour}: {available_doctors}")

            # Step 4: Determine how many doctors to send on break (16% of available doctors)
            num_on_break = max(1, int(0.16 * available_doctors))  # Ensure at least 1 doctor takes a break

            # Step 5: Randomly select doctors to take a break
            for _ in range(num_on_break):
                with self.ed_doctor.request(priority=-1) as break_req:  # Priority to override patient requests
                    yield break_req  # Wait until doctor is free
                    print(f"{self.env.now:.2f}: A doctor goes on a 30-min break.")
                    yield self.env.timeout(30)  # Break duration
                    print(f"{self.env.now:.2f}: Doctor returns from break.")

    # Method to model medical doctor working hours
    def obstruct_medical_doctor(self):
        while True:
            # Extract the current hour
            current_hour = extract_hour(self.env.now)

            # Get the number of doctors available for the current hour
            available_medical_doctors = self.medical_staffing_data.loc[
            self.medical_staffing_data['hour'] == current_hour, 'num_staff'
            ].values[0]

            # Calculate the number of doctors to block
            medical_doctors_to_block = self.medical_doctor.capacity - available_medical_doctors

            # Block excess doctors
            if medical_doctors_to_block > 0:
                print(f"{self.env.now:.2f}: Blocking {medical_doctors_to_block} Medical doctors for hour {current_hour}.")
                for _ in range(medical_doctors_to_block):
                    self.env.process(self.block_medical_doctor(60))  # Block each doctor for 1 hour
            else:
                print(f"{self.env.now:.2f}: No blocking required; all medical doctors available.")

            # Wait for the next hour to recheck staffing
            yield self.env.timeout(60)
    
    # Method to block medical doctor
    def block_medical_doctor(self, block_duration):
        """Simulate blocking a medical doctor for a specific duration."""
        with self.medical_doctor.request(priority=-1) as req:
            yield req  # Acquire the resource to simulate it being blocked
            yield self.env.timeout(block_duration)  # Simulate the blocking period
    
    # Method to refer to sdec
    def refer_to_sdec(self, patient, fallback_process):
        """Simulate process of referral to SDEC"""

        # Check if SDEC is open
        current_hour = extract_hour(self.env.now)
        if current_hour < self.global_params.sdec_open_hour or current_hour >= self.global_params.sdec_close_hour:
            self.record_result(patient.id, "SDEC Accepted", False)
            self.record_result(patient.id, "SDEC Decision Reason", "Rejected: SDEC Closed")
            yield self.env.process(fallback_process(patient)) # Route to the fallback process
            return  # Route to the fallback process
        
        # Check if the patient is eligible for SDEC
        if patient.acuity in [1, 2, 3]:  # Example eligibility criterion
            self.record_result(patient.id, "SDEC Accepted", False)
            self.record_result(patient.id, "SDEC Decision Reason", "Rejected: High Acuity")
            yield self.env.process(fallback_process(patient))  # Route to the fallback process
            return

        # Check if SDEC has capacity
        if len(self.sdec_capacity.items) > 0:  # Capacity available
            try:
                sdec_capacity_token = yield self.sdec_capacity.get()  # Reserve one SDEC capacity token
                
                # Record acceptance in results
                self.record_result(patient.id, "SDEC Accepted", True)
                self.record_result(patient.id, "SDEC Decision Reason", "Accepted")
                print(f"Patient {patient.id} referred to SDEC at hour {current_hour}.")
        
                if hasattr(patient, "utc_room_req") and patient.utc_room_req is not None:
                    self.utc_rooms.release(patient.utc_room_req)
                    print(f"Patient {patient.id} released UTC room when moving to SDEC at {self.env.now}.")

                # Process the patient in SDEC
                yield self.env.process(self.sdec_process(patient, sdec_capacity_token))  # Process the patient in SDEC
            except simpy.Interrupt:
                print(f"Patient {patient.id}'s referral to SDEC was interrupted at hour {current_hour}.")
                yield self.env.process(fallback_process(patient)) # Route to fallback process
        else:
            print(f"Patient {patient.id} could not be referred to SDEC due to no capacity at hour {current_hour}.")
            yield self.env.process(fallback_process(patient))  # Route to fallback process

    # --- Processes (Patient Pathways) --- 

    # Simulate triage process
    def walk_in_triage(self, patient):
        """Simulate triage assessment for walk ins"""
        print(f"Walk-in Triage Queue at time of request: {len(self.walk_in_triage_nurse.queue)} patients at time {self.env.now}")
        with self.walk_in_triage_nurse.request() as req:
            yield req # Wait until a triage nurse is available
             # Record the queue length
            self.record_result(patient.id, "Queue Length Walk in Triage Nurse", len(self.walk_in_triage_nurse.queue))

            # Record the start time of ED assessment
            triage_nurse_assessment_start_time = self.env.now
            self.record_result(patient.id, "Triage Nurse Assessment Start Time", triage_nurse_assessment_start_time)
            print(f"Patient {patient.id} starts triage assessment at {triage_nurse_assessment_start_time}")

            # Sample from the triage nurse assessment distribution 
            triage_nurse_assessment_time = self.triage_time_distribution.sample()
            yield self.env.timeout(triage_nurse_assessment_time)

             # Record triage assessment time in the results # 
            self.record_result(patient.id, "Triage Nurse Assessment Service Time", triage_nurse_assessment_time)
            patient.triage_nurse_assessment_time = triage_nurse_assessment_time 

         # After triage, proceed to SDEC referral process (with ED assessment as fallback)
        yield self.env.process(self.refer_to_sdec(patient, fallback_process = self.ed_assessment))


    def ambulance_triage(self, patient):
        """Simulate triage assessment for Ambulance"""
        print(f"Ambulance Triage Queue at time of request: {len(self.ambulance_triage_nurse.queue)} patients at time {self.env.now}")
        
        with self.ambulance_triage_nurse.request() as req:
            yield req # Wait until a triage nurse is available
             # Record the queue length
            self.record_result(patient.id, "Queue Length Ambulance Triage Nurse", len(self.ambulance_triage_nurse.queue))

            # Record the start time of ED assessment
            triage_nurse_assessment_start_time = self.env.now
            self.record_result(patient.id, "Triage Nurse Assessment Start Time", triage_nurse_assessment_start_time)
            print(f"Patient {patient.id} starts triage assessment at {triage_nurse_assessment_start_time}")

            # Sample from the triage nurse assessment distribution 
            triage_nurse_assessment_time = self.triage_time_distribution.sample()
            yield self.env.timeout(triage_nurse_assessment_time)

             # Record triage assessment time in the results # 
            self.record_result(patient.id, "Triage Nurse Assessment Service Time", triage_nurse_assessment_time)
            patient.triage_nurse_assessment_time = triage_nurse_assessment_time 

         # After triage, proceed ED assessment
        yield self.env.process(self.ed_assessment(patient))


    # Simulate ED assessment process

    def ed_assessment(self, patient):
        """Simulate ED assessment."""
        print(f"ED Doctor Queue at time of request: {len(self.ed_doctor.queue)} patients at time {self.env.now}")
        self.record_result(patient.id, "Queue Length ED doctor", len(self.ed_doctor.queue))
        
        ed_doctor_request_time = self.env.now
        with self.ed_doctor.request() as req:
            yield req  # Wait until a doctor is available
            ed_assessment_start_time = self.env.now
            ed_doctor_wait_time = ed_assessment_start_time - ed_doctor_request_time 
            self.record_result(patient.id, "Wait ED Assessment Time", ed_doctor_wait_time)
            

            # Record the start time of ED assessment
            ed_assessment_start_time = self.env.now
            self.record_result(patient.id, "ED Assessment Start Time", ed_assessment_start_time)
            print(f"Patient {patient.id} starts ED assessment at {ed_assessment_start_time}")

            # Sample from the triage nurse assessment distribution 
            ed_assessment_time = self.ed_assessment_time_distribution.sample()
            yield self.env.timeout(ed_assessment_time)
            
            # Record ed assessment time in the results # 
            self.record_result(patient.id, "ED Assessment Service Time", ed_assessment_time)
            patient.ed_assessment_time = ed_assessment_time

            # Calculate and record the total time from arrival to the end of ED assessment
            time_at_end_of_ed_assessment = self.env.now - patient.arrival_time
            self.record_result(patient.id, "Completed ED Assessment", time_at_end_of_ed_assessment)
            print(f"Patient {patient.id} completes ED assessment at {self.env.now}")

            # Decide outcome of ED assessment
            ed_outcome = random.choices(
                population=["Discharge", "Medicine", "OtherSpecialty"],
                weights=[
                    self.global_params.ed_discharge_prob,
                    self.global_params.ed_medicine_referral_prob,
                    self.global_params.ed_other_specialty_prob
                ],
                k=1
            )[0]
            print(f"Patient {patient.id} ED outcome: {ed_outcome}")

            # Route accordingly
            if ed_outcome == "Discharge":
                decision_delay_discharge = np.random.lognormal(
                    mean = self.global_params.mu_ed_delay_time_discharge, 
                    sigma = self.global_params.sigma_ed_delay_time_discharge)
                yield self.env.timeout(decision_delay_discharge)       
                patient.discharge_time = self.env.now
                time_in_system = patient.discharge_time - patient.arrival_time
                self.record_result(patient.id, "Discharge Time", patient.discharge_time)
                self.record_result(patient.id, "Discharge Decision Point", "ed_discharge")
                self.record_result(patient.id, "Time in System", time_in_system)
                patient.ed_disposition = "Discharged ED"
                print(f"Patient {patient.id} discharged from ED at {patient.discharge_time}")
                return

            elif ed_outcome == "OtherSpecialty":
                decision_delay_admission = np.random.lognormal(
                    mean = self.global_params.mu_ed_delay_time_admission, 
                    sigma = self.global_params.sigma_ed_delay_time_admission
                    )
                yield self.env.timeout(decision_delay_admission)
                pseudo_departure_time = self.env.now
                time_in_system = pseudo_departure_time - patient.arrival_time
                self.record_result(patient.id, "Time in System", time_in_system)
                self.record_result(patient.id, "Discharge Decision Point", "ed_referred_other_specialty_pseudo_exit")
                self.record_result(patient.id, "Discharge Time", pseudo_departure_time)
                patient.ed_disposition = "Admit - Other Speciality"
                print(f"Patient {patient.id} Referred - Other Speciality {pseudo_departure_time}")
                return

            elif ed_outcome == "Medicine":
                decision_delay_admission = np.random.lognormal(
                    mean = self.global_params.mu_ed_delay_time_admission, 
                    sigma = self.global_params.sigma_ed_delay_time_admission
                    )
                yield self.env.timeout(decision_delay_admission)
                patient.referral_to_medicine_time = self.env.now
                self.record_result(patient.id, "Simulation Referral Time", patient.referral_to_medicine_time)
                patient.ed_disposition = "Referred - Medicine"
                print(f"Patient {patient.id} Referred - Medicine {patient.referral_to_medicine_time}")
                    
                    # Record total time from arrival to referral
                    
                total_time_referral = self.env.now - patient.arrival_time
                self.record_result(patient.id, "Arrival to Referral", total_time_referral)
                
                # Pass on to downstream Medicine pathway
                yield self.env.process(self.handle_ed_referral(patient))
            

    def handle_ed_referral(self, patient):
        """Handles referral after ED assessment when SDEC is rejected.
        Ensures patient is referred to AMU queue while also starting medical assessment."""
        
        self.env.process(self.refer_to_amu_bed(patient))          
        yield self.env.process(self.initial_medical_assessment(patient))  # Wait for this to finish

    # Simulate request for AMU bed

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

        yield self.amu_beds.get()  # Patient waits for a bed
        patient.amu_admission_time = self.env.now
        print(f"Patient {patient.id} admitted to AMU at {patient.amu_admission_time}")

        patient.transferred_to_amu = True  #

        # Record admission time
        self.record_result(patient.id, "Time Admitted to AMU", patient.amu_admission_time)
       
        # Update the DataFrame with admission time
        self.amu_queue_df.loc[self.amu_queue_df['Patient ID'] == patient.id, 'Time Admitted to AMU'] = patient.amu_admission_time
        self.record_result(patient.id, "Time Admitted to AMU", patient.amu_admission_time)
        return
    
        # Exit the process for the patient
   
    # Simulate initial medical assessment process

    def initial_medical_assessment(self, patient):
        """Simulate initial medical assessment and decide discharge or admission."""
        start_medical_queue_time = self.env.now
        print(f"Patient {patient.id} added to the medical take queue at {self.env.now}")
     
        # Queue length of take at the time patient referred 
        queue_length_medical_doctor = len(self.medical_doctor.queue)
        self.record_result(patient.id, "Queue Length Medical Doctor", queue_length_medical_doctor)
            
        with self.medical_doctor.request() as req:
            yield req  # Wait until medical staff is available
            
        # Check if the patient has already been admitted to AMU before the assessment starts
            if patient.amu_admission_time is not None and patient.amu_admission_time <= self.env.now:
                self.record_result(patient.id, "Discharge Decision Point", "admitted AMU pre-medical assessment")
                print(f"{self.env.now:.2f}: Patient {patient.id} admitted to AMU before initial medical assessment.")
                return  # Exit the process if the patient has already been admitted to AMU

             # Continue with medical assessment if not admitted
            end_medical_q = self.env.now
            self.record_result(patient.id, "Simulation Time Medical Assessment Starts", end_medical_q)
            print(f"{end_medical_q:.2f}: Medical doctor starts assessing Patient {patient.id}.")


            # Calculate the waiting time from queue entry to start of medical assessment
            wait_for_medical = end_medical_q - start_medical_queue_time
            self.record_result(patient.id, "Wait for Medical Assessment", wait_for_medical)

            # Sample the medical assessment time from a specified distribution
            med_assessment_time = random.expovariate(1.0 / self.global_params.mean_initial_medical_assessment_time)
            yield self.env.timeout(med_assessment_time)
            
            # Record the initial medical assessment time
            self.record_result(patient.id, "Initial Medical Assessment Time", med_assessment_time)
            patient.initial_medical_assessment_time = med_assessment_time
            print(f"Patient {patient.id} completes initial medical assessment at {self.env.now}")
            
            # Calculate total time from arrival to the end of medical assessment
            total_time_medical = self.env.now - patient.arrival_time
            patient.total_time_medical = total_time_medical
            self.record_result(patient.id, "Arrival to Medical Assessment",  total_time_medical)
            
            # Discharge decision with a low probability (e.g., 5%)
            if random.random() < self.global_params.medicine_discharge_prob:
                patient.discharged = True
                patient.discharge_time = self.env.now
                self.record_result(patient.id, "Discharge Time", patient.discharge_time)
                self.record_result(patient.id, "Discharge Decision Point", "after_initial_medical_assessment")
                print(f"Patient {patient.id} discharged at {patient.discharge_time} after initial medical assessment")
            
                # If the patient was in a UTC bed, release it now
                if hasattr(patient, "utc_room_req") and patient.utc_room_req is not None:
                    self.utc_rooms.release(patient.utc_room_req)
                    print(f"UTC bed released for Patient {patient.id} at {self.env.now}.")

            # Remove the patient from the AMU queue if they are still in it
            try:
                if patient in self.amu_beds.items:
                    self.amu_beds.items.remove(patient)
                    print(f"Patient {patient.id} removed from AMU queue due to discharge")
            except ValueError:
                pass  # Patient was not in the queue, nothing to remove

            if patient.discharged:    
                return  # End process here if discharged

            # If not discharged, proceed to consultant assessment
            self.env.process(self.consultant_assessment(patient))
        
    # Simulate consultant assessment process

    def consultant_assessment(self, patient):

        start_ptwr_queue_time = self.env.now
        self.record_result(patient.id, "Simulation Time Added PTWR queue", start_ptwr_queue_time)
        print(f"{start_ptwr_queue_time :.2f}: Patient {patient.id} added to ptwr queue.")

        # Queue length of take at the time patient referred 
        queue_length_consultant = len(self.consultant.queue)
        self.record_result(patient.id, "Queue Length Consultant", queue_length_consultant)
          
        # Log and save the time when the patient requests the consultant
        with self.consultant.request(priority = patient.priority) as req:
            print(f"[{self.env.now:.2f}] Consultant capacity: {self.consultant.capacity}")
            print(f"[{self.env.now:.2f}] Consultant queue length before request: {len(self.consultant.queue)}")
            yield req  # Wait until a consultant is available
            print(f"[{self.env.now:.2f}] Consultant queue length after assignment: {len(self.consultant.queue)}")
            
            end_consultant_q = self.env.now
            
            if patient.amu_admission_time:
                print(f"{self.env.now:.2f}: Patient {patient.id} admitted to AMU before consultant assessment.")
                return
            else:
                print(f"{self.env.now:.2f}: Patient {patient.id} remains in ED for consultant assessment.")

            self.record_result(patient.id, 'Simulation Time Consultant Assessment Starts', end_consultant_q)
            print(f"{end_consultant_q:.2f}: Consultant starts assessing Patient {patient.id}")

            # Log the current queue length right after a patient gets the consultant
            print(f"[{self.env.now:.2f}] Consultant queue length after assigning patient {patient.id}: {len(self.consultant.queue)}")

            # Calculate the waiting time from end of referral to start of consultant assessment
            wait_for_consultant = end_consultant_q - patient.referral_to_medicine_time
            self.record_result(patient.id, "Referral to Consultant Assessment", wait_for_consultant)

            # Simulate the actual triage assessment time using the lognormal distribution
            consultant_assessment_time = self.consultant_time_distribution.sample()
            patient.consultant_assessment_time = consultant_assessment_time 

            # Record the consultant assessment time
            self.record_result(patient.id, "Consultant Assessment Time", consultant_assessment_time)
            patient.consultant_assessment_time = consultant_assessment_time

            # Calculate and record the total time from arrival to the end of consultant assessment
            total_time_consultant = self.env.now - patient.arrival_time
            self.record_result(patient.id, "Arrival to Consultant Assessment", total_time_consultant)
            yield self.env.timeout(consultant_assessment_time)  # Simulate assessment duration
            print(f"{self.env.now:.2f}: Consultant finishes assessing Patient {patient.id}.")

        # Discharge decision with a 15% probability
        if random.random() < 0.15:  # 15% chance to discharge
            patient.discharged = True
            patient.discharge_time = self.env.now
            self.record_result(patient.id, "Discharge Time", patient.discharge_time)
            self.record_result(patient.id, "Discharge Decision Point", "after_consultant_assessment")
            print(f"Patient {patient.id} discharged at {patient.discharge_time} after consultant assessment")
        
            # If the patient was in a UTC bed, release it now
            if hasattr(patient, "utc_room_req") and patient.utc_room_req is not None:
                self.utc_rooms.release(patient.utc_room_req)
                print(f"UTC bed released for Patient {patient.id} at {self.env.now}.")

            # Remove the patient from the AMU queue if they are still in it
            try:
                if patient in self.amu_beds.items:
                    self.amu_beds.items.remove(patient)
                    print(f"Patient {patient.id} removed from AMU queue due to discharge")
            except ValueError:
                pass  # Patient was not in the queue, nothing to remove

            return  # End process here if discharged

        # If not discharged, patient continues to wait in the AMU queue for bed availability
        print(f"Patient {patient.id} remains in AMU queue awaiting bed after consultant assessment")

        time_in_system = self.env.now - patient.arrival_time
        self.record_result(patient.id, "Time in System", time_in_system)

    # SDEC process

    def sdec_process(self, patient, sdec_token):

        """Simulate the SDEC process for a patient
            Parameters:
            patient: The patient object being processed.
            sdec_token: The capacity unit reserved for the patient in SDEC.
        """

        # Record the start time of the SDEC process
        start_sdec_time = self.env.now
       
        # Simulate the actual triage assessment time using the lognormal distribution
        sdec_assessment_time = self.sdec_time_distribution.sample()
        patient.sdec_assessment_time = sdec_assessment_time 
        yield self.env.timeout(sdec_assessment_time)

        # Record the end time of the SDEC process
        end_sdec_time = self.env.now

        # Calculate and record the total time spent in SDEC
        patient.sdec_duration = end_sdec_time - start_sdec_time
        self.record_result(patient.id, "SDEC Duration", patient.sdec_duration)

        # Return the capacity token to the SDEC store
        self.sdec_capacity.put(sdec_token)
        
        # SDEC discharge decision 
        if random.random() < 0.95:  # 95% chance to discharge
            patient.discharged = True
            patient.discharge_time = self.env.now
            time_in_system = patient.discharge_time - patient.arrival_time
            self.record_result(patient.id, "Time in System", time_in_system)
            self.record_result(patient.id, "Discharge Time", patient.discharge_time)
            self.record_result(patient.id, "Discharge Decision Point", "after_sdec_assessment")
            print(f"Patient {patient.id} discharged at {patient.discharge_time} after sdec assessment")
        else:
            # Patient is not discharged, add to AMU bed queue
            print(f"Patient {patient.id} requires an AMU bed after SDEC assessment.")
            self.env.process(self.refer_to_amu_bed(patient))
      
    # --- Run Method ---

    def run(self):

        """Run the simulation."""
        
        # Start the patient arrival processes

        # Generate arrivals 
        self.env.process(self.generate_arrivals())
   
        # Start monitoring the triage nurse queue
        self.env.process(self.monitor_consultant_queue_length())

        # Start the AMU bed generation process
        self.env.process(self.generate_amu_beds())

        # Start the SDEC capacity generation process 
        self.env.process(self.generate_sdec_capacity()) 

        # Start monitoring the AMU bed queue
        self.env.process(self.monitor_amu_queue())

        # Start the ED doctor obstruction process for shifts
        self.env.process(self.obstruct_ed_doctor())

        # Start the ED doctor obstruction process for breaks
        self.env.process(self.doctor_break_cycle())

        # Start the ED doctor obstruction process
        self.env.process(self.obstruct_medical_doctor())

        # Start the consultant obstruction process
        self.env.process(self.obstruct_consultant())

        # Run the simulation
        self.env.run(until=self.global_params.simulation_time)

    
