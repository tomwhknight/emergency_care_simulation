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
        "Time Triage Starts", "Wait for Triage", "Triage Assessment Time", "Triage Complete", "ED Assessment Start Time", "ED Assessment Time", "Completed ED Assessment", 
        "Referral Time", "Arrival to Referral", "Time Joined AMU Queue", "Time Admitted AMU", "Initial Medical Assessment Time",  
        "Time Medical Assessment Complete", "Referral to Consultant", "Consultant Assessment Time", 
        "Arrival to Consultant Assessment", "Discharge Time", "Discharge Decision Point", "Time in System", "Run Number"])
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
        self.amu_beds = simpy.Store(self.env)


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
                0.0, 0.0, 0.0, 0.0, 0.0,            # Triage-related columns
                0.0, 0.0, 0.0,                      # ED Assessment-related columns
                0.0, 0.0, 0.0, 0.0,                 # Referral-related columns
                0.0, 0.0,                           # Medical Assessment-related columns
                0.0, 0.0, '', 0.0, 0.0,             # Consultant-related columns, Disposition, and Time in System
                self.run_number                     # Run number
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

    # Method to generate AMU beds

    def generate_amu_beds(self):
        """Periodically release beds based on a Poisson distribution."""
        while True:
            # Sample time until next bed release using an exponential distribution
            amu_bed_release_interval = random.expovariate(1.0 / self.global_params.mean_amu_bed_release_interval)
            yield self.env.timeout(amu_bed_release_interval)

            # Check if there are patients waiting in the queue
            if len(self.amu_beds.items) > 0:
                # Admit the next patient from the queue
                admitted_patient = yield self.amu_beds.get()
                admitted_patient.amu_admission_time = self.env.now
                print(f"Patient {admitted_patient.id} admitted to AMU at {self.env.now}")

                # Update the queue DataFrame and record results
                amu_queue_df.loc[amu_queue_df['Patient ID'] == admitted_patient.id,
                             'Time Admitted to AMU'] = admitted_patient.amu_admission_time
                record_result(admitted_patient.id, "Time Admitted AMU", admitted_patient.amu_admission_time)
            else:
                print(f"No patients waiting for AMU bed at {self.env.now}. Bed remains available.")
    
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
   
        self.env.process(self.refer_to_medicine(patient))  # Proceed to referral  

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

        # Decision: Discharge or proceed to further assessment
        if random.random() < 0.75:  # Example: 50% chance to discharge
            patient.discharged = True
            patient.discharge_time = self.env.now
            self.record_result(patient.id, "Discharge Time", patient.discharge_time)
            self.record_result(patient.id, "Discharge Decision Point", "ed_discharge")
            print(f"Patient {patient.id} discharged at {patient.discharge_time} after referral to medicine")
            return  # End process here if discharged
         
        # After referral, proceed to medical assessment and request AMU bed

        patient.discharged = False
        print(f"Patient {patient.id} proceeding to further assessment")
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
        yield self.amu_beds.put(patient)
        print(f"Patient {patient.id} added to AMU bed queue at {self.env.now}")

        # Wait for a bed to become available and admit the patient
        admitted_patient = yield self.amu_beds.get()
        patient.amu_admission_time = self.env.now
        print(f"Patient {patient.id} admitted to AMU at {self.env.now}")

        # Update the DataFrame with admission time
        self.amu_queue_df.loc[self.amu_queue_df['Patient ID'] == patient.id, 'Time Admitted to AMU'] = patient.amu_admission_time
        self.record_result(patient.id, "Time Admitted AMU", patient.amu_admission_time)

        # Patient exits the system after being admitted
        print(f"Patient {patient.id} exits the system after AMU admission")
        return
    
        # Exit the process for the patient
   
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
            
    # Discharge decision with a low probability (e.g., 5%)
        if random.random() < 0.05:  # 5% chance to discharge
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
        print(f"Patient {patient.id} proceeding to consultant assessment")
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
      