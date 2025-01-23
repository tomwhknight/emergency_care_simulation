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
        self.total_beds_generated = 0 

        # Load CSV data
        self.ed_staffing_data = pd.read_csv(self.global_params.ed_staffing_file)
        self.medical_staffing_data = pd.read_csv(self.global_params.medicine_staffing_file)

        # Instantiate the Lognormal distribution for triage assessment time
        self.triage_time_distribution = Lognormal(mean=self.global_params.mean_triage_assessment_time,
                                                  stdev=self.global_params.stdev_triage_assessment_time)
        
        self.ed_time_distribution = Lognormal(mean=self.global_params.mean_ed_assessment_time, 
                                                   stdev=self.global_params.stdev_ed_assessment_time)

        self.referral_time_distribution = Lognormal(mean=self.global_params.mean_referral_time, 
                                                   stdev=self.global_params.stdev_referral_time)

        self.consultant_time_distribution = Lognormal(mean=self.global_params.mean_consultant_assessment_time, 
                                                   stdev=self.global_params.stdev_consultant_assessment_time)
        
        self.sdec_time_distribution = Lognormal(mean=self.global_params.mean_sdec_assessment_time, 
                                                   stdev=self.global_params.stdev_sdec_assessment_time)
        
        # Create results DF

        self.run_results_df = pd.DataFrame(columns=[
            "Simulation Arrival Time", 
            "Day of Arrival", 
            "Clock Hour of Arrival", 
            "Hour of Arrival",
            "Acuity",  
            "Wait for Triage Nurse", 
            "Triage Assessment Time", 
            "Triage Complete", 
            "SDEC Accepted",
            "SDEC Decision Reason", 
            "ED Assessment Start Time", 
            "ED Assessment Time", 
            "Completed ED Assessment",
            "Simulation Referral Time", 
            "Arrival to Referral", 
            "Time Joined AMU Queue", 
            "Time Admitted to AMU",
            "Simulation Time Medical Assessment Starts", 
            "Wait for Medical Assessment",
            "Initial Medical Assessment Time",  
            "Arrival to Medical Assessment",
            "Simulation Time Added PTWR queue",
            "Simulation Time Consultant Assessment Starts", 
            "Referral to Consultant Assessment", 
            "Consultant Assessment Time", 
            "Arrival to Consultant Assessment",
            "Discharge Time", 
            "Discharge Decision Point", 
            "Time in System", 
            "Run Number"
        ])
        self.run_results_df.index.name = "Patient ID"

        # Initialize DataFrame to monitor triage nurse queue
        self.triage_queue_monitoring_df = pd.DataFrame(columns=['Simulation Time', 'Hour of Day', 'Queue Length'])
    
        # Initialize DataFrame to monitor consultant queue
        self.consultant_queue_monitoring_df = pd.DataFrame(columns=['Simulation Time', 'Hour of Day', 'Queue Length'])
    
        # Initialize the DataFrame for tracking the AMU bed queue times
        self.amu_queue_df = pd.DataFrame(columns=["Patient ID", "Time Joined AMU Queue", "Time Admitted to AMU"])

        # Create simpy resources for staffing levels
        self.triage_nurse = simpy.Resource(self.env, capacity=self.global_params.triage_nurse_capacity)
        self.ed_doctor = simpy.PriorityResource(self.env, capacity=self.global_params.ed_doctor_capacity)
        self.medical_doctor = simpy.PriorityResource(self.env, capacity=self.global_params.medical_doctor_capacity)
        self.consultant = simpy.PriorityResource(self.env, capacity=self.global_params.consultant_capacity)

        # Initialize the AMU bed container
        self.amu_beds = simpy.Store(self.env, capacity = self.global_params.max_amu_available_beds)

        # Initialize the SDEC capacity container
        self.sdec_capacity = simpy.Store(self.env, capacity = self.global_params.max_sdec_capacity)

    def record_result(self, patient_id, column, value):

        """Helper function to record results only if the burn-in period has passed."""

        if self.env.now > self.burn_in_time:
            if column not in self.run_results_df.columns:
                print(f"Warning: Attempting to add a new column '{column}'. Ignoring update.")
                return  # Ignore the update if the column does not exist
            self.run_results_df.at[patient_id, column] = value
    
    # --- Generator Methods ---

    # Method to generate patient arrivals
    def generate_patient_arrivals(self):

        """Generate patient arrivals based on inter-arrival times."""
        
        while True:
            self.patient_counter += 1
            arrival_time = self.env.now
            
            # Add time variables
            
            arrival_clock_time = calculate_hour_of_day(arrival_time)
            day_of_arrival = calculate_day_of_week(arrival_time)
            current_hour = extract_hour(arrival_time)

            # Assign acuity level based on probabilities from GlobalParameters
            
            acuity_levels = list(self.global_params.acuity_probabilities.keys())
            acuity_weights = list(self.global_params.acuity_probabilities.values())
            acuity = random.choices(acuity_levels, weights=acuity_weights, k=1)[0]
            
             # Create instance of patient class using
            patient = Patient(self.patient_counter, arrival_time, day_of_arrival, arrival_clock_time, current_hour, acuity)


            # Initialize a row for this patient in the DataFrame
            patient_row = [
                arrival_time, 
                day_of_arrival,
                arrival_clock_time,
                current_hour,
                acuity,
                0.0, 0.0, 0.0,                 # Triage-related columns
                "", "",                            # Accepted for SDEC and Decision Reason
                0.0, 0.0, 0.0,                      # ED Assessment-related columns
                0.0, 0.0, 0.0, 0.0,                 # Referral-related columns
                0.0, 0.0, 0.0, 0.0,                          # Medical Assessment-related columns
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, '', 0.0,             # Consultant-related columns, Disposition, and Time in System
                self.run_number                     # Run number
            ]

            # Record the patient data in the results DataFrame
            self.run_results_df.loc[patient.id] = patient_row
           
            # Record patient arrival
            self.record_result(patient.id, "Simulation Arrival Time", patient.arrival_time)
            self.record_result(patient.id, "Day of Arrival", patient.day_of_arrival)
            self.record_result(patient.id, "Clock Hour of Arrival", patient.arrival_clock_time)
            self.record_result(patient.id, "Hour of Arrival", patient.current_hour)
        
            # Use the helper function to extract the current hour from simulation time
            
            # Determine arrival rate based on the current hour
            if 9 <= current_hour < 21:  # Peak hours (09:00 to 21:00)
                mean_interarrival_time = self.global_params.ed_peak_mean_patient_arrival_time
            else:  # Off-peak hours (21:00 to 09:00)
                mean_interarrival_time = self.global_params.ed_off_peak_mean_patient_arrival_time

            # start triage process
            self.env.process(self.triage(patient))
            print(f"Patient {patient.id} arrives at {patient.arrival_time}")

            # Convert mean inter-arrival time to a rate
            arrival_rate = 1.0 / mean_interarrival_time
            
            # Sample the inter-arrival time using an exponential distribution
            ED_inter_arrival_time = random.expovariate(arrival_rate) 
            yield self.env.timeout(ED_inter_arrival_time)

    # Method to generate AMU beds
    def generate_amu_beds(self):

        """Periodically release beds based on a Poisson distribution."""
        while True:
            # Sample time until next bed release using an exponential distribution
            amu_bed_release_interval = random.expovariate(1.0 / self.global_params.mean_amu_bed_release_interval)
            yield self.env.timeout(amu_bed_release_interval)

            # Add a bed if there is space in the store
            if len(self.amu_beds.items) < self.amu_beds.capacity:
                yield self.amu_beds.put("Bed")
                self.total_beds_generated += 1  # Increment counter
                print(f"Bed added to AMU at {self.env.now}. Total beds available: {len(self.amu_beds.items)}")
            else: print(f"No space to add more beds at {self.env.now}.")
    
    # Method to generate SDEC capacity
    def generate_sdec_capacity(self):
        while True:
        # Determine if it's a weekend
            current_day = calculate_day_of_week(self.env.now)  # e.g., "Monday", "Saturday"
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
            while calculate_day_of_week(self.env.now) == current_day:  # Stay within the same day
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
    
    # Method to monitor the triage queue
    def monitor_triage_queue_length(self, interval=60):
        """Monitor the triage nurse queue length at regular intervals."""
        while True:
            # Record the current time and queue length
            current_time = self.env.now
            queue_length = len(self.triage_nurse.queue)
            hour_of_day = (current_time // 60) % 24
            # Create a new DataFrame for the current row
            new_row = pd.DataFrame({
            'Simulation Time': [current_time],
            'Hour of Day': [hour_of_day],
            'Queue Length': [queue_length]
            })
        
            # Concatenate the new row with the existing DataFrame
            self.triage_queue_monitoring_df = pd.concat([self.triage_queue_monitoring_df, new_row], ignore_index=True)
        
            # Wait for the specified interval before checking again
            yield self.env.timeout(interval)

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

    # Method to track AMU queue
    def monitor_amu_queue(self, interval=10):
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

    # Method to model restricted consultant working hours 
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
    
    def obstruct_ed_doctor(self):
        """Simulate ED doctor unavailability based on shift patterns."""
        while True:
            # Extract the current hour
            current_hour = extract_hour(self.env.now)
        
            # Get the number of doctors available for the current hour
            available_doctors = self.ed_staffing_data.loc[
            self.ed_staffing_data['hour'] == current_hour, 'num_staff'
            ].values[0]

            # Calculate the number of doctors to block
            ed_doctors_to_block = self.ed_doctor.capacity - available_doctors

            # Block excess doctors
            if ed_doctors_to_block > 0:
                print(f"{self.env.now:.2f}: Blocking {ed_doctors_to_block} doctors for hour {current_hour}.")
                for _ in range(ed_doctors_to_block):
                    self.env.process(self.block_doctor(60))  # Block each doctor for 1 hour
            else:
                print(f"{self.env.now:.2f}: No blocking required; all doctors available.")

            # Wait for the next hour to recheck staffing
            yield self.env.timeout(60)

    def block_doctor(self, block_duration):
        """Simulate blocking a single doctor for a specific duration."""
        with self.ed_doctor.request(priority=-1) as req:
            yield req  # Acquire the resource to simulate it being blocked
            yield self.env.timeout(block_duration)  # Simulate the blocking period
    
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
                print(f"{self.env.now:.2f}: Blocking {medical_doctors_to_block} doctors for hour {current_hour}.")
                for _ in range(medical_doctors_to_block):
                    self.env.process(self.block_medical_doctor(60))  # Block each doctor for 1 hour
            else:
                print(f"{self.env.now:.2f}: No blocking required; all medical doctors available.")

            # Wait for the next hour to recheck staffing
            yield self.env.timeout(60)
    
    def block_medical_doctor(self, block_duration):
        """Simulate blocking a medical doctor for a specific duration."""
        with self.medical_doctor.request(priority=-1) as req:
            yield req  # Acquire the resource to simulate it being blocked
            yield self.env.timeout(block_duration)  # Simulate the blocking period
    

    # --- Processes (Patient Pathways) --- 

    # Simulate referral to SDEC

    def refer_to_sdec(self, patient, fallback_process):

        """Simulate process of referral to SDEC"""

        # Step 1: Check if SDEC is open
        current_hour = extract_hour(self.env.now)
        if current_hour < self.global_params.sdec_open_hour or current_hour >= self.global_params.sdec_close_hour:
            self.record_result(patient.id, "SDEC Accepted", False)
            self.record_result(patient.id, "SDEC Decision Reason", "Rejected: SDEC Closed")
            yield self.env.process(fallback_process(patient)) # Route to the fallback process
            return  # Route to the fallback process
        
        # Step 2: Check if the patient is eligible for SDEC
        if patient.acuity != "low":  # Example eligibility criterion
            self.record_result(patient.id, "SDEC Accepted", False)
            self.record_result(patient.id, "SDEC Decision Reason", "Rejected: High Acuity")
            yield self.env.process(fallback_process(patient))  # Route to the fallback process
            return

        # Step 3: Check if SDEC has capacity
        if len(self.sdec_capacity.items) > 0:  # Capacity available
            try:
                sdec_capacity_token = yield self.sdec_capacity.get()  # Reserve one SDEC capacity token
                
                # Record acceptance in results
                self.record_result(patient.id, "SDEC Accepted", True)
                self.record_result(patient.id, "SDEC Decision Reason", "Accepted")
                print(f"Patient {patient.id} referred to SDEC at hour {current_hour}.")
        
 
                # Process the patient in SDEC
                yield self.env.process(self.sdec_process(patient, sdec_capacity_token))  # Process the patient in SDEC
            except simpy.Interrupt:
                print(f"Patient {patient.id}'s referral to SDEC was interrupted at hour {current_hour}.")
                yield self.env.process(fallback_process(patient)) # Route to fallback process
        else:
            print(f"Patient {patient.id} could not be referred to SDEC due to no capacity at hour {current_hour}.")
            yield self.env.process(fallback_process(patient))  # Route to fallback process

    # Simulate triage process

    def triage(self, patient):
        
        """Simulate triage"""
        
        start_triage_q = self.env.now
        with self.triage_nurse.request() as req:
            yield req  # Wait until a triage is available
        
            # Time triage assessment begins
            start_triage_time = self.env.now
            patient.wait_time_for_triage_nurse = start_triage_time - start_triage_q
            
            # Record the time spent waiting for triage and when triage starts
            self.record_result(patient.id, "Wait for Triage Nurse", patient.wait_time_for_triage_nurse)

            # Simulate the actual triage assessment time using the lognormal distribution
            triage_assessment_time = self.triage_time_distribution.sample()
            patient.triage_assessment_time = triage_assessment_time
        
            # Record the time spent with the triage nurse 
            yield self.env.timeout(triage_assessment_time)
            self.record_result(patient.id, "Triage Assessment Time", patient.triage_assessment_time)
        
            # Calculate and record the total time from arrival to the end of triage
            patient.time_at_end_of_triage = self.env.now - patient.arrival_time
            self.record_result(patient.id, "Triage Complete", patient.time_at_end_of_triage)

        
        # Delegate to SDEC referral logic with ED assessment as fallback
        self.env.process(self.refer_to_sdec(patient, self.ed_assessment))

    # Simulate initial ED assessment

    def ed_assessment(self, patient):

        """Simulate ED assessment."""
        
        with self.ed_doctor.request() as req:
            yield req  # Wait until a doctor is available

            # Record the start time of ED assessment
            ed_assessment_start_time = self.env.now
            self.record_result(patient.id, "ED Assessment Start Time", ed_assessment_start_time)
            print(f"Patient {patient.id} starts ED assessment at {ed_assessment_start_time}")

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
   
        # Delegate to SDEC referral logic with medical assessment as fallback
        self.env.process(self.refer_to_sdec(patient, self.refer_to_medicine))

    # Simulate referral to medicine

    def refer_to_medicine(self, patient):
        
        """Simulate the process of referring a patient to medicine."""
        
        # Simulate the actual triage assessment time using the lognormal distribution
        referral_time = self.referral_time_distribution.sample()
        yield self.env.timeout(referral_time)
     
        # Decision: Discharge or proceed to further assessment
        if random.random() < self.global_params.ed_discharge_rate:
            patient.discharged = True
            patient.discharge_time = self.env.now
            self.record_result(patient.id, "Discharge Time", patient.discharge_time)
            self.record_result(patient.id, "Discharge Decision Point", "ed_discharge")
            print(f"Patient {patient.id} discharged at {patient.discharge_time} after referral to medicine")
            return  # End process here if discharged
         
            # After referral, proceed to medical assessment and request AMU bed

        patient.discharged = False
        patient.referral_to_medicine_time = self.env.now
        self.record_result(patient.id, "Simulation Referral Time", patient.referral_to_medicine_time)
        print(f"Patient {patient.id} referred to medicine at {self.env.now}")

        # Calculate and record the total time from arrival to the end of the referral
        total_time_referral = self.env.now - patient.arrival_time
        self.record_result(patient.id, "Arrival to Referral", total_time_referral)

        # Pass on to initial medical assessment and referral to amu bed

        self.env.process(self.initial_medical_assessment(patient)) # Continue medical assessment process in EDso t
        self.env.process(self.refer_to_amu_bed(patient))  # Start the process for AMU bed request
    
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

        # Record admission time
        self.record_result(patient.id, "Time Admitted to AMU", patient.amu_admission_time)
       
        # Update the DataFrame with admission time
        self.amu_queue_df.loc[self.amu_queue_df['Patient ID'] == patient.id, 'Time Admitted to AMU'] = patient.amu_admission_time
        self.record_result(patient.id, "Time Admitted to AMU", patient.amu_admission_time)

        # Patient exits the system after being admitted
        print(f"Patient {patient.id} exits the system after AMU admission")
        return
    
        # Exit the process for the patient
   
    # Simulate initial medical assessment process

    def initial_medical_assessment(self, patient):
        start_medical_queue_time = self.env.now
        print(f"{start_medical_queue_time:.2f}: Patient {patient.id} added to the medical queue.")
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
        if random.random() < self.global_params.medicine_discharge_rate:
            patient.discharged = True
            patient.discharge_time = self.env.now
            self.record_result(patient.id, "Discharge Time", patient.discharge_time)
            self.record_result(patient.id, "Discharge Decision Point", "after_initial_medical_assessment")
            print(f"Patient {patient.id} discharged at {patient.discharge_time} after initial medical assessment")
            
            # Remove the patient from the AMU queue if they are still in it
            try:
                if patient in self.amu_beds.items:
                    self.amu_beds.items.remove(patient)
                    print(f"Patient {patient.id} removed from AMU queue due to discharge")
            except ValueError:
                pass  # Patient was not in the queue, nothing to remove

            return  # End process here if discharged

    # If not discharged, proceed to consultant assessment
        patient.discharged = False
        self.env.process(self.consultant_assessment(patient))
        
    # Simulate consultant assessment process

    def consultant_assessment(self, patient):

        start_ptwr_queue_time = self.env.now
        self.record_result(patient.id, "Simulation Time Added PTWR queue", start_ptwr_queue_time)
        print(f"{start_ptwr_queue_time :.2f}: Patient {patient.id} added to ptwr queue.")

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
        
        # Start the patient arrival process
        self.env.process(self.generate_patient_arrivals())

        # Start monitoring the triage nurse queue
        self.env.process(self.monitor_triage_queue_length())
   
        # Start monitoring the triage nurse queue
        self.env.process(self.monitor_consultant_queue_length())

        # Start the AMU bed generation process
        self.env.process(self.generate_amu_beds())

        # Start the SDEC capacity generation process 
        self.env.process(self.generate_sdec_capacity()) 

        # Start monitoring the AMU bed queue
        self.env.process(self.monitor_amu_queue())

        # Start the consultant obstruction process
        self.env.process(self.obstruct_consultant())

        # Start the ED doctor obstruction process
        self.env.process(self.obstruct_ed_doctor())

         # Start the ED doctor obstruction process
        self.env.process(self.obstruct_medical_doctor())
    
        # Run the simulation
        self.env.run(until=self.global_params.simulation_time)

    

      