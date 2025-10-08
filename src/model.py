import simpy
import pandas as pd
import numpy as np
from src.patient import Patient
from src.helper import calculate_hour_of_day, extract_day_of_week, extract_hour
from src.helper import exp_rv, wchoice, bern 

VERBOSE = False  # flip to True when debugging
def log(msg_or_fn):
    if VERBOSE:
        print(msg_or_fn() if callable(msg_or_fn) else msg_or_fn)

class Model:

    def __init__(self, global_params, burn_in_time, run_number):
        """Initialize the model with the given global parameters."""
        self.env = simpy.Environment()
        self.global_params = global_params
        self.run_number = run_number
        self.burn_in_time = burn_in_time
        self.patient_counter = 0
        self.amu_waiting_counter = 0
        self.total_beds_generated = 0 

        # Set individual seeds per process
        self.rng_arrivals  = np.random.Generator(np.random.PCG64(self.global_params.seed_arrivals))
        self.rng_service   = np.random.Generator(np.random.PCG64(self.global_params.seed_service))
        self.rng_probs     = np.random.Generator(np.random.PCG64(self.global_params.seed_probs))
        self.rng_resources = np.random.Generator(np.random.PCG64(self.global_params.seed_resources))


        # Load CSV data
        self.ed_staffing_data = pd.read_csv(self.global_params.ed_staffing_file)
        self.medical_staffing_data = pd.read_csv(self.global_params.medicine_staffing_file)
        self.amu_bed_rate_data = pd.read_csv(self.global_params.amu_bed_rate_file)
        self.sdec_slot_rate_data = pd.read_csv(self.global_params.sdec_slot_rate_file)
        self.arrival_rate_data = pd.read_csv(self.global_params.arrival_rate_file)
        self.booked_rate_data =  pd.read_csv(self.global_params.booked_rate_file)

        self.news_distribution_data = pd.read_csv(self.global_params.news2_file)
        self.admission_probability_distribution_data = pd.read_csv(self.global_params.admission_probability_file)
       
        # Create results DF

        # Define standard columns, structured by category
        self.standard_cols = [
            
            # --- Index ---
            "Patient ID",

            # --- Arrival Information ---
            "Simulation Arrival Time",
            "Day of Arrival",
            "Clock Hour of Arrival",
            "Hour of Arrival",
            "Patient Age",
            "Adult",
            "NEWS2",
            "Admission Probability",
            "Source of Referral",
            "Booked Appointment",
            "Mode of Arrival",
            "Acuity",
            "ED Disposition",
            

            # --- Triage Information ---
            "Queue Length Ambulance Triage Nurse",
            "Queue Length Walk in Triage Nurse",
            "Arrival to Triage Nurse Assessment",
            "Triage Nurse Assessment Service Time",

            # --- SDEC Referral Information ---

            "SDEC Appropriate",
            "SDEC Accepted",
            "SDEC Decision Reason",

            # --- Tests ---

            "Bloods Requested",
            "Queue Length Bloods",
            "Request to Bloods Obtained",
            "Arrival to Bloods Obtained",
            "Arrival to Bloods Reported",
            "Blood Draw Service Time",
            "Blood Lab Service Time",
            "Wait for Bloods After Assessment",

            # --- ED assessment ---

            "Queue Length ED doctor",
            "Arrival to ED Assessment",
            "ED Assessment Service Time",
            "ED Decision Time",
            "Post-Assessment Decision Delay",
            
    
            # --- Referral to Medicine ---
            "Arrival to Referral",

            # --- AMU Admission ---
            "Arrival to AMU Admission",
            "Referral to AMU Admission",

            # --- Medical Assessment ---
            "Queue Length Medical Doctor",
            "Arrival to Medical Assessment",
            "Referral to Medical Assessment",
            "Medical Assessment Service Time",
            "Arrival to End of Medical Assessment",

            # --- Consultant Review ---
            "Queue Length Consultant",
            "Referral to Consultant Assessment",
            "Consultant Assessment Service Time",
            "Arrival to Consultant Assessment",

            # --- Discharge Information ---
            "Discharge Decision Point",
            "Time in System",

            # --- Run Information ---
            "Run Number"
            ]   

        # Create results DataFrame with structured standard columns
        self.run_results_df = pd.DataFrame(columns=self.standard_cols)
        self.run_results_df = self.run_results_df.set_index("Patient ID")

        # --- Baseline scenario defaults so Trial/AltTrial can tag outputs ---
        self._scenario_name = "baseline"
        self._dt_threshold  = np.nan  # baseline has no direct-triage threshold

        # ⇩ Add scenario/triage experiment columns (no need to touch standard_cols)
        for c in ["Scenario", "DT Threshold", "DT Eligible", "Pathway Start"]:
            if c not in self.run_results_df.columns:
                self.run_results_df[c] = np.nan

       # --- Buffers (fast) ---
        self._event_log_buf = []
        self._ed_q_buf = []
        self._cons_q_buf = []
        self._amu_q_buf = []
        self._ed_block_buf = []

        # Placeholders so attributes exist before run()
        self.event_log_df = pd.DataFrame(columns=["run_number","patient_id","event","timestamp"])
        self.ed_assessment_queue_monitoring_df = pd.DataFrame(columns=["Simulation Time","Hour of Day","Queue Length"])
        self.consultant_queue_monitoring_df = pd.DataFrame(columns=["Simulation Time","Hour of Day","Queue Length"])
        self.amu_queue_df = pd.DataFrame(columns=["Time","Queue Length"])  # ← use Time (see note below)
        self.ed_doctor_block_monitoring_df = pd.DataFrame(columns=[
            "Simulation Time","Hour of Day","Physical Capacity","Rota Blockers","Break Blockers",
            "Total Blockers","Effective Capacity","Active Patient Users","Patient Queue Length","Desired From Rota"
        ])


        # Create simpy resources for staffing levels
        self.ambulance_triage_nurse = simpy.Resource(self.env, capacity=self.global_params.ambulance_triage_nurse_capacity)
        self.walk_in_triage_nurse = simpy.PriorityResource(self.env, capacity = self.global_params.walk_in_triage_nurse_capacity)
        self.hca = simpy.PriorityResource(self.env, capacity = self.global_params.hca_capacity)
        self.ed_doctor = simpy.PriorityResource(self.env, capacity=self.global_params.max_ed_doctor_capacity)
        self.medical_doctor = simpy.PriorityResource(self.env, capacity=self.global_params.medical_doctor_capacity)
        self.consultant = simpy.PriorityResource(self.env, capacity=self.global_params.consultant_capacity)

        
        # Create list to enforce rota blocks
        self._rota_blockers = []   # list of (process, stop_event)


        # Initialize the AMU bed container
        self.amu_beds = simpy.Store(self.env, capacity = self.global_params.max_amu_available_beds)

        # Initialize the SDEC capacity container
        self.sdec_capacity = simpy.Store(self.env, capacity = self.global_params.max_sdec_capacity)

    # Method to add results to the results dataframe
    def record_result(self, patient_id, column, value):
        """Record result only if after burn-in and patient is already in the results."""
        if self.env.now > self.burn_in_time:
            if column in self.run_results_df.columns and patient_id in self.run_results_df.index:
                self.run_results_df.at[patient_id, column] = value

    # Method to add results to an event log for BUPAR
    def record_event(self, patient, event_name):
        """Append an event to the in-memory buffer (fast). Build the DataFrame after run()."""
        # Skip recording events during burn-in
        if self.env.now <= self.burn_in_time:
            return

        self._event_log_buf.append({
            "run_number": self.run_number,
            "patient_id": patient.id,
            "event": event_name,
            "timestamp": self.env.now,
        })
    
    # --- Generator Methods ---
    def generate_arrivals(self):
        """Generate patient arrivals based on inter-arrival times."""
    
        while True:
            if self.env.now > self.burn_in_time:
                self.patient_counter += 1
                patient_id = self.patient_counter
            else:
                patient_id = np.nan

            arrival_time = self.env.now
        
            # Add time variables
            current_day = extract_day_of_week(arrival_time)
            current_hour = extract_hour(arrival_time)
            clock_hour = calculate_hour_of_day(arrival_time)
            
            mode_of_arrival = wchoice(
                ["Ambulance", "Walk-in"],
                [self.global_params.ambulance_proportion, self.global_params.walk_in_proportion],
                self.rng_probs
                )

            # Assign source of referral

            age_weights = {}

            for age in range(0, 5):
                age_weights[age] = 1.75       # Ages 0–4 → weight 1.75
            for age in range(5, 80):
                age_weights[age] = 1       # Ages 5–79 → weight 1
            for age in range(80, 101):
                age_weights[age] = 0.5     # Ages 80–100 → weight 0.5

            age_values = list(age_weights.keys())
            age_probs = list(age_weights.values())
            age = int(round(wchoice(age_values, age_probs, self.rng_probs)))

            # Assign mode of arrival and admission probability based on mode of arrival

            if mode_of_arrival == "Ambulance":
                acuity_levels = list(self.global_params.ambulance_acuity_probabilities.keys())
                acuity_weights = list(self.global_params.ambulance_acuity_probabilities.values())
            else:
                acuity_levels = list(self.global_params.walk_in_acuity_probabilities.keys())
                acuity_weights = list(self.global_params.walk_in_acuity_probabilities.values())

            acuity = wchoice(acuity_levels, acuity_weights, self.rng_probs)
            
            # Assign source of referral
            
            news2_values = self.news_distribution_data["news2"].tolist()
            news2_weights = self.news_distribution_data["news2_probs"].tolist()
            news2 = wchoice(news2_values, news2_weights, self.rng_probs)

            # Assign if patient is adult
            
            adult = age >= 16  
           
            # Assign source of referral

            source_of_referral = wchoice(
                ["GP", "ED"],
                [self.global_params.proportion_direct_primary_care,
                1 - self.global_params.proportion_direct_primary_care],
                self.rng_probs
            )

            # Admission probability 
            admission_probability = self.rng_probs.choice(
                self.admission_probability_distribution_data["p_cal"].values
            )

           # SDEC appropriate
            if adult:
                if (acuity not in [1, 2]) and (news2 < 4) and (admission_probability < 0.5):
                    sdec_appropriate = True
                else:
                    sdec_appropriate = False
            else:
                sdec_appropriate = False
                        
            # Assign priority level

            if acuity in [1, 2] or news2 > 4:
                priority = 0  # Higher priority
            else:
                priority = 1  # Lower priority

        
            # Create instance of patient class
            
            patient = Patient(
                env=self.env,
                patient_id=patient_id,
                arrival_time=arrival_time,
                current_day=current_day,
                clock_hour=clock_hour,
                current_hour=current_hour,
                source_of_referral=source_of_referral,
                mode_arrival= mode_of_arrival,
                age=age,
                adult=adult,
                news2=news2,
                admission_probability=admission_probability,
                acuity=acuity,
                sdec_appropriate=sdec_appropriate,
                ed_disposition=None,
                priority=priority
                )

            # Initialise a dictionary of patient results 

            patient_results = {
            # --- Arrival Information ---
            "Patient ID": patient.id,
            "Simulation Arrival Time": arrival_time,
            "Day of Arrival": current_day,
            "Clock Hour of Arrival": clock_hour,
            "Hour of Arrival": current_hour,
            "Mode of Arrival": mode_of_arrival,
            "Patient Age": age,
            "Adult": adult,
            "NEWS2": news2,
            "Source of Referral": source_of_referral,
            "Booked Appointment": "",
            "Acuity": acuity,
            "Admission Probability": admission_probability, 

            # --- Triage-Related Metrics ---
            "Queue Length Walk in Triage Nurse": np.nan,
            "Queue Length Ambulance Triage Nurse": np.nan,
            "Arrival to Triage Nurse Assessment": np.nan,
            "Triage Nurse Assessment Service Time": np.nan,

            # --- SDEC Referral ---
            "SDEC Appropriate": sdec_appropriate,
            "SDEC Accepted": np.nan,
            "SDEC Decision Reason": "",

            # --- Test related metrics ---

            "Bloods Requested": "",
            "Arrival to Bloods Reported": np.nan,
            "Blood Draw Service Time": np.nan,
            "Bloods Lab Service Time": np.nan, 
            "Wait for Bloods After Assessment": np.nan,

            # --- ED Assessment Metrics ---
            "Queue Length ED doctor": np.nan,
            "Arrival to ED Assessment": np.nan,
            "ED Assessment Service Time": np.nan,
            "ED Decision Time": np.nan,
            "Post-Assessment Decision Delay": np.nan,
            "ED Disposition": "",

            # --- Referral to Medicine ---
            "Arrival to Referral": np.nan,

            # --- AMU Process ---
            "Arrival to AMU Admission": np.nan,
            "Referral to AMU Admission": np.nan,

            # --- Medical Assessment Process ---
            "Queue Length Medical Doctor": np.nan,
            "Arrival to Medical Assessment": np.nan,
            "Referral to Medical Assessment": np.nan, 
            "Medical Assessment Service Time": np.nan,
      
            # --- Consultant Review Process ---
            "Queue Length Consultant": np.nan,
            "Referral to Consultant Assessment": np.nan,
            "Consultant Assessment Service Time": np.nan,
            "Arrival to Consultant Assessment": np.nan,

            # --- Discharge Information ---
            "Discharge Decision Point": "",
            "Time in System": np.nan,

            # --- Simulation Run Number ---
            "Run Number": self.run_number
            }

            # Ensure all columns from `self.standard_cols` exist
            for col in self.standard_cols:
                if col not in patient_results:
                    patient_results[col] = float('nan')  # Assign NaN if column is missing

    
            # Only add to run_results_df if simulation time > burn-in
            if self.env.now > self.burn_in_time:
            
                # Create the new row as a DataFrame with index set
                new_row = pd.DataFrame.from_records([patient_results]).set_index("Patient ID")
            
                # Append it using .loc 
                self.run_results_df.loc[patient_results["Patient ID"]] = new_row.iloc[0]

            # Record patient arrival
            self.record_result(patient.id, "Simulation Arrival Time", patient.arrival_time)
            self.record_result(patient.id, "Day of Arrival", patient.current_day)
            self.record_result(patient.id, "Clock Hour of Arrival", patient.clock_hour)
            self.record_result(patient.id, "Hour of Arrival", patient.current_hour)

            # Record arrival in event log
            self.record_event(patient, "arrival")

            if patient.ed_disposition == "Refer - Paeds":
                self.record_result(patient.id, "Discharge Decision Point", "ed_referred_paeds")
               
            row = self.booked_rate_data.loc[
                (self.booked_rate_data['hour'] == current_hour) &
                (self.booked_rate_data['day'] == current_day),
                'is_booked_prob'
            ]
            
            booked_appointment_prob = float(row.iloc[0]) if not row.empty else 0.0

            # Use your model RNG if you have one (preferred for reproducibility)
            is_booked = bern(booked_appointment_prob, self.rng_probs)
            self.record_result(patient.id, "Booked Appointment", bool(is_booked))

            if is_booked:
                log(lambda: f"[{self.env.now:.2f}] Patient {patient.id} has a booked appointment; no ED triage.")
    

            else:    
                if mode_of_arrival == "Ambulance":
                    log(lambda: f"Ambulance Patient {patient.id} arrives at {arrival_time}")
                    self.env.process(self.ambulance_triage(patient))
                else:
                    log(lambda: f"Walk-in Patient {patient.id} arrives at {arrival_time}")
                    self.env.process(self.walk_in_triage(patient))

            # --- Interpolated arrival rate instead of step function ---

            # How far into the current hour (0–1)
            frac = (self.env.now % 60) / 60.0
            next_hour = (current_hour + 1) % 24

            # Handle day rollover (e.g., Sunday 23:00 → Monday 00:00)
            days = list(self.arrival_rate_data['day'].unique())
            current_day_index = days.index(current_day)
            if next_hour == 0:  # crossed midnight
                next_day = days[(current_day_index + 1) % len(days)]
            else:
                next_day = current_day

            # Lookup rates for current and next hour
            r1 = self.arrival_rate_data.loc[
                (self.arrival_rate_data['hour'] == current_hour) &
                (self.arrival_rate_data['day'] == current_day),
                'mean_arrivals_per_min'
            ].values[0]

            r2 = self.arrival_rate_data.loc[
                (self.arrival_rate_data['hour'] == next_hour) &
                (self.arrival_rate_data['day'] == next_day),
                'mean_arrivals_per_min'
            ].values[0]

            # Linear interpolation between r1 and r2
            mean_arrival_rate = (1 - frac) * r1 + frac * r2

            # Sample next arrival
            arrival_interval = exp_rv(mean_arrival_rate, self.rng_arrivals)  # minutes
            yield self.env.timeout(arrival_interval)

    # Method to generate AMU beds
    def generate_amu_beds(self):

        """Periodically release beds based on a Poisson distribution."""
        while True:

            # Extract the current hour
            current_hour = extract_hour(self.env.now)
            current_day = extract_day_of_week(self.env.now)

            # Get the mean for day and hour
            mean_beds = self.amu_bed_rate_data.loc[
            (self.amu_bed_rate_data['hour'] == current_hour) & (self.amu_bed_rate_data ['day'] == current_day), 'mean_beds_available'
            ].values[0]

             # Sample number of beds to release this hour
            amu_beds_this_hour = self.rng_resources.poisson(mean_beds)
            
            for beds in range(amu_beds_this_hour):
                delay = self.rng_resources.uniform(0, 60)
                self.env.process(self.release_amu_bed_after_delay(delay))

            # Wait until the start of the next hour
            yield self.env.timeout(60)
    
    # Method to stagger bed release
    def release_amu_bed_after_delay(self, delay):
        yield self.env.timeout(delay)

        if len(self.amu_beds.items) < self.amu_beds.capacity:
            yield self.amu_beds.put("Bed")
            self.total_beds_generated += 1
            log(lambda: f"[{self.env.now:.1f}] Bed released | Total beds: {self.total_beds_generated}")
        else:
            log(lambda: f"[{self.env.now:.1f}] No space to add bed — store full.")

    # Method to generate SDEC capacity
    def generate_sdec_slots(self):
        """Periodically release SDEC slots using a time-varying Poisson process."""
        while True:
            
            # Identify the current day
            current_day = extract_day_of_week(self.env.now)
            current_hour = extract_hour(self.env.now)
            is_weekend = current_day in ["Sat", "Sun"]

            # Determine starting capacity based on day type
            base_capacity = (
                self.global_params.weekday_sdec_base_capacity
                if not is_weekend
                else self.global_params.weekend_sdec_base_capacity
            )

            # Reset SDEC capacity at start of the day
            self.sdec_capacity.items.clear()
            for _ in range(base_capacity):
                self.sdec_capacity.put("slot")
                log(lambda: f"[{self.env.now:.2f}] SDEC capacity reset to {base_capacity} on {current_day}")

            # Determine the end of the current day (midnight)
            next_day_time = ((self.env.now // 1440) + 1) * 1440

            # Release additional capacity during the day based on Poisson process
            while self.env.now < next_day_time:
                
                # Recalculate current day and hour inside loop
                current_day = extract_day_of_week(self.env.now)
                current_hour = extract_hour(self.env.now)
                
                # Get the mean for day and hour
                mean_sdec_slot_rate = self.sdec_slot_rate_data.loc[
                    (self.sdec_slot_rate_data['hour'] == current_hour) & (self.sdec_slot_rate_data['day'] == current_day), 
                    'mean_sdec_slots_available_per_min'
                ].values[0]

                if mean_sdec_slot_rate > 0:
                    sdec_slot_release_interval = exp_rv(mean_sdec_slot_rate, self.rng_arrivals)
                    yield self.env.timeout(sdec_slot_release_interval)

                    if len(self.sdec_capacity.items) < self.sdec_capacity.capacity:
                        self.sdec_capacity.put("slot")
                        log(lambda: f"[{self.env.now:.2f}] SDEC slot added. Total now: {len(self.sdec_capacity.items)}")
                    else:
                        log(lambda: f"[{self.env.now:.2f}] SDEC store full. No slot added.")
  
                else:
                    log(lambda: f"[{self.env.now:.2f}] No SDEC slots released this hour")
                    yield self.env.timeout(60) 

    # Method to monitor the consultant queue
    def monitor_ed_assessment_queue_length(self, interval=60):
        """Monitor ED assessment queue length (patients only) at regular intervals."""
        while True:
            current_time = self.env.now
            hour_of_day = int((current_time // 60) % 24)

            # Count ONLY patient requests (exclude any blocker requests)
            patient_queue = [r for r in self.ed_doctor.queue if not getattr(r, "is_block", False)]
            queue_length = len(patient_queue)


            # Buffer the row (FAST). Build the DataFrame once at the end of run()
            self._ed_q_buf.append({
                "Simulation Time": current_time,
                "Hour of Day": hour_of_day,
                "Queue Length": queue_length,
            })

            yield self.env.timeout(interval)

    # Method to monitor ED doctor blocking
    def monitor_ed_doctor_blocks(self, interval=15):
        """
        Monitor ED-doctor blockers and effective capacity.
        Uses ACTIVE blockers (in resource.users) for effective capacity.
        """
        last_eff = None
        while True:
            now = self.env.now
            hod = int((now // 60) % 24)
            phys_cap = int(self.ed_doctor.capacity)

            # Active blockers (holding capacity)
            rota_active  = self._count_active_blockers(self.ed_doctor, "rota")
            break_active = self._count_active_blockers(self.ed_doctor, "break")

            # Queued blockers (debug only)
            if VERBOSE:
                rota_queued  = sum(1 for r in self.ed_doctor.queue
                                if getattr(r, "is_block", False) and getattr(r, "block_type", None) == "rota")
                break_queued = sum(1 for r in self.ed_doctor.queue
                                if getattr(r, "is_block", False) and getattr(r, "block_type", None) == "break")
            else:
                rota_queued = break_queued = 0

            effective_capacity = max(0, phys_cap - (rota_active + break_active))

            try:
                desired_from_rota = int(self.get_available_doctors(now))
            except Exception:
                desired_from_rota = np.nan

            should_after_breaks = (np.nan if np.isnan(desired_from_rota)
                                else max(0, desired_from_rota - break_active))

            if last_eff is not None and effective_capacity > last_eff and effective_capacity != should_after_breaks:
                log(lambda: (f"[{now:6.1f}] MONITOR  ↑ eff {last_eff} → {effective_capacity} "
                            f"(desired={desired_from_rota}, after_breaks≈{should_after_breaks}) | "
                            f"rota_act={rota_active} rota_q={rota_queued} "
                            f"break_act={break_active} break_q={break_queued}"))

            last_eff = effective_capacity

            # ---- BUFFER (FAST) ----
            self._ed_block_buf.append({
                "Simulation Time":      now,
                "Hour of Day":          hod,
                "Physical Capacity":    phys_cap,
                "Rota Blockers":        rota_active,
                "Break Blockers":       break_active,
                "Total Blockers":       rota_active + break_active,
                "Effective Capacity":   effective_capacity,
                "Active Patient Users": sum(1 for r in self.ed_doctor.users if not getattr(r, "is_block", False)),
                "Patient Queue Length": sum(1 for r in self.ed_doctor.queue if not getattr(r, "is_block", False)),
                "Desired From Rota":    desired_from_rota,
            })

            yield self.env.timeout(interval)

    # Method to monitor the consultant queue
    def monitor_consultant_queue_length(self, interval=60):
        """Monitor consultant queue length at regular intervals (buffered)."""
        while True:
            current_time = self.env.now
            hour_of_day = int((current_time // 60) % 24)
            queue_length = len(self.consultant.queue)

            # Optional debug
            log(lambda: f"[{current_time:.2f}] Consultant queue={queue_length} @hour {hour_of_day}")
            if VERBOSE and queue_length > 0:
                patient_ids = [req.priority for req in self.consultant.queue]
                log(lambda: f"Patients in Consultant Queue: {patient_ids}")

            # Buffer the row (FAST). Build the DataFrame once at the end of run()
            self._cons_q_buf.append({
                "Simulation Time": current_time,
                "Hour of Day":     hour_of_day,
                "Queue Length":    queue_length,
            })

            yield self.env.timeout(interval)

    # Method to monitor AMU queue
    def monitor_amu_queue(self, interval=60):
        """Monitor the AMU bed queue length at regular intervals (buffered)."""
        while True:
            current_time = self.env.now
            queue_length = self.amu_waiting_counter  # current number waiting

            # Buffer a lightweight dict (fast)
            self._amu_q_buf.append({
                "Time": current_time,
                "Queue Length": queue_length,
            })

            yield self.env.timeout(interval)

    # --- Dynamic resource modelling ---

    # Method to block walk-in triage nurse 
    def obstruct_triage_nurse(self):
        """Simulate increased traige nurse capacity in peak hours 12:00 and 22:00."""
        while True:
            # Extract the current hour
            current_hour = extract_hour(self.env.now)

            # Check if the current time is within the off-duty period (21:00–07:00)
            if current_hour >= 22 or current_hour < 12:
                with self.walk_in_triage_nurse.request(priority=-1) as req:
                    yield req  # Block the resource
                    yield self.env.timeout(60)  # Hold the block for 1 hour
            else:
                log(lambda: f"{self.env.now:.2f}: Triage nurse capacity increased")

            # Wait until the next hour to check again
            yield self.env.timeout(60)

    def get_available_doctors(self, current_time_minutes, start_block=15, end_cutoff=45):
        """
        Calculate how many doctors are available to START a new patient based on
        current simulation time and shift patterns, with handover rules:
        - Block first `start_block` minutes after shift start.
        - Block last `end_cutoff` minutes before shift end.

        Args:
            current_time_minutes (float): Simulation time in minutes.

        Returns:
            int: Number of doctors available to start new patients now.
        """
        def to_minutes(t_str):
            h, m = map(int, t_str.split(":"))
            return 60 * h + m

        now = int(current_time_minutes % (24 * 60))
        available_count = 0

        patterns = (self.global_params.shift_patterns_weekend
            if extract_day_of_week(current_time_minutes) in ("Sat", "Sun")
            else self.global_params.shift_patterns_weekday)

        for shift in patterns:
            start = to_minutes(shift["start"])
            end   = to_minutes(shift["end"])
            count = shift["count"]

            if start < end:  # same-day shift (e.g., 08:00–16:00)
                active = (start <= now < end)
                if not active:
                    continue
                mins_since = now - start
                mins_to_end = end - now
            else:            # overnight shift (e.g., 22:00–07:30)
                active = (now >= start) or (now < end)
                if not active:
                    continue
                mins_since = (now - start) if now >= start else (1440 - start + now)
                mins_to_end = (end - now) if now < end else (1440 - now + end)

            # Apply handover rules
            if mins_since >= start_block and mins_to_end > end_cutoff:
                available_count += count

        return available_count

    def _count_blockers(self, resource, block_type=None):
        """
        Count blocker requests currently active or queued on a SimPy (Priority)Resource.
        If block_type is provided ('rota' or 'break'), count only that type.
        """
        def is_blocker(req):
            if not getattr(req, "is_block", False):
                return False
            return block_type is None or getattr(req, "block_type", None) == block_type

        active = sum(1 for r in resource.users if is_blocker(r))   # holding a slot
        queued = sum(1 for r in resource.queue if is_blocker(r))   # waiting to start
        return active + queued

    def _count_active_blockers(self, resource, block_type=None):
            def is_blocker(req):
                if not getattr(req, "is_block", False):
                    return False
                return block_type is None or getattr(req, "block_type", None) == block_type
            return sum(1 for r in resource.users if is_blocker(r))
    
    def _start_persistent_rota_blocker(self, stop_event):
        """Hold one ED-doctor slot until stop_event is triggered. Cancel if told to stop before acquiring."""
        def _runner():
            req = self.ed_doctor.request(priority=-1)
            req.is_block = True
            req.block_type = "rota"

            # Race: either we acquire, or we get told to stop first
            outcome = yield req | stop_event
            if stop_event in outcome:
                # Told to stop before acquiring → cancel request if still pending
                if not req.triggered:
                    req.cancel()
                return

            # Acquired: hold until told to stop, then release
            try:
                yield stop_event
            finally:
                self.ed_doctor.release(req)

        return self.env.process(_runner())

    def _sync_rota_blockers(self, target_count):
        """Ensure exactly target_count rota blockers exist (add/remove instantly)."""
        current = len(self._rota_blockers)

        # add missing blockers
        for _ in range(max(0, target_count - current)):
            stop_evt = self.env.event()
            proc = self._start_persistent_rota_blocker(stop_evt)
            self._rota_blockers.append((proc, stop_evt))

        # remove extras
        for _ in range(max(0, current - target_count)):
            proc, stop_evt = self._rota_blockers.pop()
            if not stop_evt.triggered:
                stop_evt.succeed()     # releases immediately

    def obstruct_ed_doctor(self, tick=15):

        """Rota controller: continuously match effective starters to rota (15/45 rules)."""
        while True:
            now = self.env.now
            desired = self.get_available_doctors(now)
            desired = max(0, min(self.global_params.max_ed_doctor_capacity, desired))
            phys_cap = self.global_params.max_ed_doctor_capacity
            target_rota_blockers = max(0, phys_cap - desired)

            # BEFORE
            eff_before = phys_cap - (
                self._count_active_blockers(self.ed_doctor, "rota") +
                self._count_active_blockers(self.ed_doctor, "break")
            )
            log(lambda: f"[{now:6.1f}] ROTA CTRL desired={desired:2d} phys={phys_cap:2d} "
                f"target_rota={target_rota_blockers:2d} | eff_before={eff_before:2d}")

            # Level to the target (add/remove persistent blockers)
            self._sync_rota_blockers(target_rota_blockers)

            # 🔸 Give SimPy a microstep so new blockers can run and acquire
            yield self.env.timeout(0)

            # AFTER
            active_rota  = self._count_active_blockers(self.ed_doctor, "rota")
            active_break = self._count_active_blockers(self.ed_doctor, "break")
            eff_after = phys_cap - (active_rota + active_break)
            # what we "should" see once breaks are considered
            should_starters = max(0, desired - active_break)
            log(lambda: f"[{self.env.now:6.1f}] ROTA CTRL eff_after={eff_after:2d} "
                f"(should≈{should_starters:2d}; desired={desired:2d}, breaks={active_break})")

            yield self.env.timeout(tick)


    def doctor_break_cycle(self, break_dur=30, jitter_max=30):
        """
        Schedule ED-doctor breaks day-by-day so weekend/weekday patterns are respected.
        - Uses shift_patterns_weekday vs shift_patterns_weekend each day.
        - Keeps each break fully inside its shift (handles overnight shifts).
        - Staggers starts ±jitter_max minutes across doctors.
        Call once in run(): self.doctor_break_cycle()
        """
        day = 24 * 60

        # ---------- helpers (local) ----------
        def to_min(t):
            h, m = map(int, t.split(":"))
            return 60*h + m

        def clamp_to_shift(st, en, raw_start, dur):
            """
            Return a minute-of-day start so the ENTIRE [start, start+break_dur] window sits inside the shift.
            Works for both same-day and overnight shifts via minute-of-day modular arithmetic.
            """
            rel = (raw_start - st) % day                 # raw start relative to shift start
            max_rel = max(0, dur - break_dur)            # last legal start that still fits fully
            rel = max(0, min(rel, max_rel))              # clamp inside [0, max_rel]
            return (st + rel) % day

        def schedule_day(day_index, patterns):
            """
            Create one-off break processes for a given day_index (0=day0).
            Skips breaks whose start time has already passed.
            """
            day_start = day_index * day
            now = self.env.now

            for s in patterns:
                count = int(s.get("count", 0))
                if count <= 0:
                    continue

                st = to_min(s["start"]); en = to_min(s["end"])
                dur = (en - st) % day
                if dur == 0:
                    continue

                # breaks: use rota value if present; else infer (<=8h -> 1, >8h -> 2)
                b = s.get("breaks")
                if b is None:
                    b = 2 if dur > 8*60 else 1
                if b <= 0:
                    continue

                # break "centres" within the shift
                centres = [ (st + dur//2) % day ] if b == 1 else [ (st + dur//3) % day, (st + 2*dur//3) % day ]

                # even offsets across doctors in [-jitter_max, +jitter_max]
                if count == 1:
                    offsets = [0]
                else:
                    offsets = [int(round((i - (count-1)/2) * (2*jitter_max/(count-1)))) for i in range(count)]

                for c in centres:
                    base_start = (c - break_dur//2) % day  # centre -> start of the break window
                    for off in offsets:
                        raw = (base_start + off) % day
                        start_min = clamp_to_shift(st, en, raw, dur)      # minute-of-day
                        abs_start = day_start + start_min                 # absolute sim time

                        if abs_start >= now:
                            # one-off break at abs_start for `break_dur`
                            self.env.process(self._one_off_break(abs_start, break_dur))

        def day_loop():
            """Schedule remainder of 'today', then repeat every midnight with the correct pattern."""
            # schedule remainder of the current day
            today_idx = int(self.env.now // day)
            today_name = extract_day_of_week(self.env.now)  # e.g., "Mon", "Sat"
            today_patterns = (self.global_params.shift_patterns_weekend
                            if today_name in ("Sat", "Sun")
                            else self.global_params.shift_patterns_weekday)
            schedule_day(today_idx, today_patterns)

            # then roll forward day by day
            while True:
                next_midnight = ((self.env.now // day) + 1) * day
                yield self.env.timeout(next_midnight - self.env.now)

                day_idx = int(self.env.now // day)
                dow = extract_day_of_week(self.env.now)
                pats = (self.global_params.shift_patterns_weekend
                        if dow in ("Sat", "Sun")
                        else self.global_params.shift_patterns_weekday)
                schedule_day(day_idx, pats)

        # launch the scheduler as a background process
        self.env.process(day_loop())

    def _one_off_break(self, abs_start, duration):
        """Block one ED-doctor slot for `duration` minutes starting at absolute sim time `abs_start`."""
        if self.env.now < abs_start:
            yield self.env.timeout(abs_start - self.env.now)
        # tag as a 'break' blocker so it doesn't count as a patient
        yield self.env.process(self.block_doctor(duration, block_type="break"))

    def block_doctor(self, block_duration, block_type):
        """Occupy one doctor slot for the block_duration."""
        with self.ed_doctor.request(priority=-1) as req:
            req.is_block = True
            req.block_type = block_type
            yield req
            acquired = self.env.now
            log(lambda: f"[{acquired:6.1f}] {block_type.upper():5} BLOCK ACQUIRED (dur={block_duration}) "
                f"| users={len(self.ed_doctor.users)} queue={len(self.ed_doctor.queue)}")
            yield self.env.timeout(block_duration)
        released = self.env.now
        log(lambda: f"[{released:6.1f}] {block_type.upper():5} BLOCK RELEASED                 "
            f"| users={len(self.ed_doctor.users)} queue={len(self.ed_doctor.queue)}")

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
                log(lambda: f"{self.env.now:.2f}: Blocking {medical_doctors_to_block} Medical doctors for hour {current_hour}.")
                for _ in range(medical_doctors_to_block):
                    self.env.process(self.block_medical_doctor(60))  # Block each doctor for 1 hour
            else:
                log(lambda: f"{self.env.now:.2f}: No blocking required; all medical doctors available.")

            # Wait for the next hour to recheck staffing
            yield self.env.timeout(60)
    
    # Method to block medical doctor
    def block_medical_doctor(self, block_duration):

        """Simulate blocking a medical doctor for a specific duration."""
        with self.medical_doctor.request(priority=-1) as req:
            yield req  # Acquire the resource to simulate it being blocked
            yield self.env.timeout(block_duration)  # Simulate the blocking period
    
    # Method to block consultants 
    def obstruct_consultant(self):
        """Simulate consultant unavailability between 21:00 and 07:00."""
        while True:
            # Extract the current hour
            current_hour = extract_hour(self.env.now)

            # Check if the current time is within the off-duty period (21:00–07:00)
            if current_hour >= 20 or current_hour < 7:
                log(lambda: f"{self.env.now:.2f}: Consultants are off-duty (21:00–07:00).")
                with self.consultant.request(priority=-1) as req:
                    yield req  # Block the resource
                    yield self.env.timeout(60)  # Hold the block for 1 hour
            else:
                log(lambda: f"{self.env.now:.2f}: Consultants are available.")

            # Wait until the next hour to check again
            yield self.env.timeout(60)
    
    # Method to refer to sdec
    def refer_to_sdec(self, patient, fallback_process):
        """Simulate process of referral to SDEC"""

        if not patient.adult:
            self.record_result(patient.id, "SDEC Accepted", False)
            self.record_result(patient.id, "SDEC Decision Reason", "Rejected: Paediatric")
            yield self.env.process(fallback_process(patient))
            return
        
        # Check Acuity
        if not patient.sdec_appropriate:
            self.record_result(patient.id, "SDEC Accepted", False)
            self.record_result(patient.id, "SDEC Decision Reason", "Rejected: Not Appropriate")
            yield self.env.process(fallback_process(patient))  # Route to fallback
            return
        
        # Check if SDEC is open
        current_hour = extract_hour(self.env.now)
        if current_hour < self.global_params.sdec_open_hour or current_hour >= self.global_params.sdec_close_hour:
            self.record_result(patient.id, "SDEC Accepted", False)
            self.record_result(patient.id, "SDEC Decision Reason", "Rejected: SDEC Closed")
            yield self.env.process(fallback_process(patient)) # Route to the fallback process
            return  # Route to the fallback process

        # Check capacity
        if len(self.sdec_capacity.items) > 0:
            yield self.sdec_capacity.get()  # Take slot immediately

            self.record_result(patient.id, "SDEC Accepted", True)
            self.record_result(patient.id, "SDEC Decision Reason", "Accepted")

            patient.ed_disposition = "SDEC Accepted"
            self.record_result(patient.id, "ED Disposition", "SDEC Accepted")


            self.record_result(patient.id, "Discharge Decision Point", "after_sdec_acceptance")
            self.record_event(patient, "sdec_acceptance")
            patient.discharged = True
            patient.discharge_time = self.env.now
            time_in_system = patient.discharge_time - patient.arrival_time
            self.record_result(patient.id, "Time in System", time_in_system)
            log(lambda: f"[{self.env.now:.2f}] Patient {patient.id} sent to SDEC immediately.")
            return
        
        else:
            self.record_result(patient.id, "SDEC Accepted", False)
            self.record_result(patient.id, "SDEC Decision Reason", "Rejected: No Capacity")
            yield self.env.process(fallback_process(patient))

    # --- Processes (Patient Pathways) --- 

    # Simulate triage process

    def walk_in_triage(self, patient):
        """Simulate triage assessment for walk ins"""
        log(lambda: f"Walk-in Triage Queue at time of request: {len(self.walk_in_triage_nurse.queue)} patients at time {self.env.now}")
        with self.walk_in_triage_nurse.request() as req:
            yield req # Wait until a triage nurse is available
             # Record the queue length
            self.record_result(patient.id, "Queue Length Walk in Triage Nurse", len(self.walk_in_triage_nurse.queue))

            # Record the start time of ED assessment
            triage_nurse_assessment_start_time = self.env.now
            self.record_result(patient.id, "Arrival to Triage Nurse Assessment", triage_nurse_assessment_start_time - patient.arrival_time)
            self.record_event(patient, "triage_start")
            log(lambda: f"Patient {patient.id} starts triage assessment at {triage_nurse_assessment_start_time}")

            # Sample from the triage nurse assessment distribution 
            triage_nurse_assessment_time = self.rng_service.lognormal(
            mean=self.global_params.mu_triage_assessment_time,
            sigma=self.global_params.sigma_triage_assessment_time
            )

            # Triage nurse assessment 
            yield self.env.timeout(triage_nurse_assessment_time)

             # Record triage assessment time in the results # 
            self.record_result(patient.id, "Triage Nurse Assessment Service Time", triage_nurse_assessment_time)
            patient.triage_nurse_assessment_time = triage_nurse_assessment_time 

            # Decide if blood tests are needed based on admission probability  
            if bern(self.global_params.bloods_request_probability, self.rng_probs):

                self.record_result(patient.id, "Bloods Requested", "Yes")
                yield self.env.process(self.tests_draw(patient))
                patient.bloods_requested = True
                patient.bloods_requested_at_triage = True
            else:
                self.record_result(patient.id, "Bloods Requested", "No")
                patient.bloods_requested = False
                patient.bloods_requested_at_triage = False


         # After triage, proceed to SDEC referral process (with ED assessment as fallback)
        yield self.env.process(self.refer_to_sdec(patient, fallback_process = self.ed_assessment))

    def ambulance_triage(self, patient):
        """Simulate triage assessment for Ambulance"""    
        with self.ambulance_triage_nurse.request() as req:
            yield req # Wait until a triage nurse is available
             # Record the queue length
            self.record_result(patient.id, "Queue Length Ambulance Triage Nurse", len(self.ambulance_triage_nurse.queue))

            # Record the start time of ED assessment
            triage_nurse_assessment_start_time = self.env.now
            self.record_result(patient.id, "Arrival to Triage Nurse Assessment", triage_nurse_assessment_start_time - patient.arrival_time)
            self.record_event(patient, "triage_start")
            log(lambda: f"Patient {patient.id} starts triage assessment at {triage_nurse_assessment_start_time}")

            # Sample from the triage nurse assessment distribution 
            triage_nurse_assessment_time = self.rng_service.lognormal(
            mean=self.global_params.mu_triage_assessment_time,
            sigma=self.global_params.sigma_triage_assessment_time
            )
            
            # Record triage assessment time in the results 
            self.record_result(patient.id, "Triage Nurse Assessment Service Time", triage_nurse_assessment_time)
            patient.triage_nurse_assessment_time = triage_nurse_assessment_time 

            # Decide if blood tests are needed based on admission probability   
            if bern(self.global_params.bloods_request_probability, self.rng_probs):

                self.record_result(patient.id, "Bloods Requested", "Yes")
                yield self.env.process(self.tests_draw(patient))
                patient.bloods_requested = True
                patient.bloods_requested_at_triage = True
            else:
                self.record_result(patient.id, "Bloods Requested", "No")
                patient.bloods_requested = False
                patient.bloods_requested_at_triage = False

        # After triage, proceed ED assessment
        yield self.env.process(self.refer_to_sdec(patient, fallback_process = self.ed_assessment))

    # Simulate bloods tests
    def tests_draw(self, patient):
        """Simulate an HCA drawing tests after triage."""
        with self.hca.request() as req:
            self.record_result(patient.id, "Queue Length Bloods", len(self.hca.queue))
            yield req
            
            # Sample test draw duration
            blood_draw_duration = self.rng_service.lognormal(
            mean=self.global_params.mu_blood_draw_time,
            sigma=self.global_params.sigma_blood_draw_time
            )
            
            time_bloods_obtained = self.env.now
            arrival_to_obtained = time_bloods_obtained - patient.arrival_time    

            self.record_result(patient.id, "Blood Draw Service Time", blood_draw_duration)
            self.record_result(patient.id, "Arrival to Bloods Obtained", arrival_to_obtained)
            
            log(lambda: f"[{self.env.now:.2f}] Patient {patient.id} Blood test draw complete.")

        # After draw, proceed to lab processing
        self.env.process(self.tests_lab(patient)) 

    def tests_lab(self, patient):
        """Simulate lab processing time for test results."""
        lab_duration = self.rng_service.lognormal(
                mean=self.global_params.mu_blood_lab_time,
                sigma=self.global_params.sigma_blood_lab_time
            )
        
        yield self.env.timeout(lab_duration)
        
        blood_complete_time = self.env.now
        patient.bloods_ready_time = blood_complete_time 

        total_test_time = blood_complete_time - patient.arrival_time
        self.record_result(patient.id, "Blood Lab Service Time", lab_duration)
        self.record_result(patient.id, "Arrival to Bloods Reported", total_test_time)
        log(lambda: f"[{self.env.now:.2f}] Patient {patient.id} lab results available.")

        if not patient.bloods_ready.triggered:
            patient.bloods_ready.succeed()
    
    # Simulate ED assessment process
    def ed_assessment(self, patient):
        """Simulate ED assessment, including possible wait for blood results and results review."""

        def finalise_disposition(patient, decision_point, event_name):
            patient.discharge_time = self.env.now
            time_in_system = patient.discharge_time - patient.arrival_time
            self.record_result(patient.id, "Discharge Decision Point", decision_point)
            self.record_result(patient.id, "Time in System", time_in_system)
            self.record_event(patient, event_name)

        # --- Initial assessment ---
        with self.ed_doctor.request(priority=patient.priority) as req:
            q_patients = sum(1 for r in self.ed_doctor.queue if not getattr(r, "is_block", False))
            log(lambda: f"Patient {patient.id} ED Doctor Queue at time of request: {q_patients} patients at time {self.env.now}")
            self.record_result(patient.id, "Queue Length ED doctor", q_patients)
            yield req

            ed_assessment_start_time = self.env.now
            wait_time = ed_assessment_start_time - patient.arrival_time
            self.record_result(patient.id, "Arrival to ED Assessment", wait_time)
            self.record_event(patient, "ed_assessment_start")

            # Sample service and decision times (resource held during service)
            service_time_sample = self.rng_service.lognormal(
                mean=self.global_params.mu_ed_service_time,
                sigma=self.global_params.sigma_ed_service_time
            )
            decision_time_sample = self.rng_service.lognormal(
                mean=self.global_params.mu_ed_decision_time,
                sigma=self.global_params.sigma_ed_decision_time
            )

            # Doctor works for the service time
            yield self.env.timeout(service_time_sample)

            # End of assessment
            ed_assessment_end_time = self.env.now
            service_time_actual = ed_assessment_end_time - ed_assessment_start_time
            self.record_result(patient.id, "ED Assessment Service Time", service_time_actual)

        # Compute any *extra* decision time beyond service
        excess_decision_after_assessment = max(0.0, decision_time_sample - service_time_actual)

        # Record the sampled decision time and the excess
        self.record_result(patient.id, "ED Decision Time", decision_time_sample)
        self.record_result(patient.id, "Post-Assessment Decision Delay", excess_decision_after_assessment)

        # If decision takes longer than service, wait the excess before moving on
        if excess_decision_after_assessment > 0:
            yield self.env.timeout(excess_decision_after_assessment)

        # --- Disposition logic ---
        if not patient.adult:
            admitted = bern(self.global_params.paediatric_referral_rate, self.rng_probs)
            if admitted:
                patient.ed_disposition = "Refer - Paeds"
            else:
                # Discharge → may need to wait for bloods
                if patient.bloods_ready_time > self.env.now:
                    wait_time = patient.bloods_ready_time - self.env.now
                    yield self.env.timeout(wait_time)
                    self.record_result(patient.id, "Wait for Bloods After Assessment", wait_time)
                    log(lambda: f"[{self.env.now:.2f}] Patient {patient.id} blood results available.")
                else:
                    self.record_result(patient.id, "Wait for Bloods After Assessment", 0)
                patient.ed_disposition = "Discharge"

        else:
            admitted = bern(patient.admission_probability, self.rng_probs)
            if not admitted:
                # Adult discharge → may need to wait for bloods
                if patient.bloods_ready_time > self.env.now:
                    wait_time = patient.bloods_ready_time - self.env.now
                    yield self.env.timeout(wait_time)
                    self.record_result(patient.id, "Wait for Bloods After Assessment", wait_time)
                    log(lambda: f"[{self.env.now:.2f}] Patient {patient.id} blood results available.")
                else:
                    self.record_result(patient.id, "Wait for Bloods After Assessment", 0)
                patient.ed_disposition = "Discharge"
            else:
                patient.ed_disposition = wchoice(
                ["Refer - Medicine", "Refer - Other Speciality"],
                [self.global_params.medical_referral_rate,
                1 - self.global_params.medical_referral_rate],
                self.rng_probs
            )

        self.record_result(patient.id, "ED Disposition", patient.ed_disposition)

        # --- Handle outcome ---
        outcome = patient.ed_disposition
        if outcome == "Discharge":
            finalise_disposition(patient, "ed_discharge", "discharge")
            return

        elif outcome == "Refer - Other Speciality":
            finalise_disposition(patient, "ed_referred_other_specialty_pseudo_exit", "referral_to_speciality")
            return

        elif outcome == "Refer - Paeds":
            finalise_disposition(patient, "ed_referred_paeds_pseudo_exit", "referral_to_paeds")
            return

        elif outcome == "Refer - Medicine":
            patient.referral_to_medicine_time = self.env.now
            total_time_referral = patient.referral_to_medicine_time - patient.arrival_time
            self.record_result(patient.id, "Arrival to Referral", total_time_referral)
            self.record_event(patient, "referral_to_medicine")
            yield self.env.process(self.handle_ed_referral(patient))
      
    def handle_ed_referral(self, patient):
        """Handles referral after ED assessment when SDEC is rejected.
        Ensures patient is referred to AMU queue while also starting medical assessment."""

        self.env.process(self.refer_to_amu_bed(patient))          
        yield self.env.process(self.initial_medical_assessment(patient))  # Wait for this to finish

    # Simulate request for AMU bed
    def refer_to_amu_bed(self, patient):
        """Request a bed for the patient if available, or exit early if discharged.
        cancel the pending Store.get() when discharge wins.
        """
        # Ensure the discharge event exists (consistent name!)
        if getattr(patient, "discharge_event", None) is None:
            patient.discharge_event = self.env.event()

        # Count this patient as waiting for an AMU bed
        self.amu_waiting_counter += 1
        log(lambda:f"Patient {patient.id} requesting AMU bed at {self.env.now}, queue size: {self.amu_waiting_counter}")

        # Create the bed request and wait for either: a bed OR a discharge
        bed_get = self.amu_beds.get()
        result = yield bed_get | patient.discharge_event

        # --- A) Discharged first -> cancel the pending get() to avoid a zombie consumer ---
        if (patient.discharge_event in result) and (bed_get not in result):
            try:
                bed_get.cancel()  # critical: remove orphaned get() from the Store's queue
            except Exception:
                pass
            self.amu_waiting_counter -= 1
            log(lambda: f"Patient {patient.id} discharged while waiting — leaving AMU queue at {self.env.now}.")
            return

        # --- B) Race: bed + discharge same instant -> return the bed immediately ---
        if (patient.discharge_event in result) and (bed_get in result):
            bed = result[bed_get]
            yield self.amu_beds.put(bed)
            self.amu_waiting_counter -= 1
            log(lambda: f"Patient {patient.id} discharged as bed arrived — bed returned at {self.env.now}.")
            return

        # --- C) Normal admission: bed arrived first ---
        bed = result[bed_get]
        self.amu_waiting_counter -= 1
        patient.amu_admission_time = self.env.now
        log(lambda: f"Patient {patient.id} admitted to AMU at {patient.amu_admission_time}")

        # Decision point labelling for your results
        if hasattr(patient, "consultant_assessment_time"):
            decision_point = "admitted_after_consultant"
        elif hasattr(patient, "initial_medical_assessment_time"):
            decision_point = "admitted_before_consultant"
        else:
            decision_point = "admitted_before_medical"

        self.record_result(patient.id, "Discharge Decision Point", decision_point)
        patient.arrival_to_amu_admission = patient.amu_admission_time - patient.arrival_time
        patient.referral_to_amu_admission = patient.amu_admission_time - patient.referral_to_medicine_time
        self.record_result(patient.id, "Arrival to AMU Admission", patient.arrival_to_amu_admission)
        self.record_result(patient.id, "Referral to AMU Admission", patient.referral_to_amu_admission)
        self.record_result(patient.id, "Time in System", patient.arrival_to_amu_admission)
        self.record_event(patient, "amu_admission")
        patient.transferred_to_amu = True
        return

    # Simulate initial medical assessment process
    def initial_medical_assessment(self, patient):
        """Simulate initial medical assessment and decide discharge or admission."""
        log(lambda: f"Patient {patient.id} added to the medical take queue at {self.env.now}")
        # Queue length of take at the time patient referred 
        queue_length_medical_doctor = len(self.medical_doctor.queue)
        self.record_result(patient.id, "Queue Length Medical Doctor", queue_length_medical_doctor)
            
        with self.medical_doctor.request() as req:
            yield req  # Wait until medical staff is available
            
        # Check if the patient has already been admitted to AMU before the assessment starts
            if patient.amu_admission_time is not None and patient.amu_admission_time <= self.env.now:
                log(lambda: f"{self.env.now:.2f}: Patient {patient.id} admitted to AMU before initial medical assessment.")
                return  # Exit the process if the patient has already been admitted to AMU

             # Continue with medical assessment if not admitted
            end_medical_queue_time = self.env.now
            arrival_to_medical = end_medical_queue_time - patient.arrival_time
            referral_to_medical = end_medical_queue_time -  patient.referral_to_medicine_time
            self.record_result(patient.id, "Arrival to Medical Assessment", arrival_to_medical)
            self.record_result(patient.id, "Referral to Medical Assessment", referral_to_medical)
            self.record_event(patient, "medical_assessment_start")
            log(lambda: f"{end_medical_queue_time:.2f}: Medical doctor starts assessing Patient {patient.id}.")

            # Sample the medical assessment time from log normal distribution
            raw_time = self.rng_service.lognormal(
            mean=self.global_params.mu_medical_service_time,
            sigma=self.global_params.sigma_medical_service_time
            )

            # Truncate to min and max values
            med_assessment_time = min(
                max(raw_time,self.global_params.min_medical_service_time),
            self.global_params.max_medical_service_time  
            )

            yield self.env.timeout(med_assessment_time)
            log(lambda: f"Patient {patient.id} spends {med_assessment_time} minutes with medical doctor")
            
            # Record the initial medical assessment time
            self.record_result(patient.id, "Medical Assessment Service Time", med_assessment_time)
            patient.initial_medical_assessment_time = med_assessment_time
            log(lambda: f"Patient {patient.id} completes initial medical assessment at {self.env.now}")
            
            # Calculate total time from arrival to the end of medical assessment
            total_time_medical = self.env.now - patient.arrival_time
            patient.total_time_medical = total_time_medical
            self.record_result(patient.id, "Arrival to End of Medical Assessment",  total_time_medical)
            self.record_event(patient, "initial_medical_assessment_end")
            

            # Discharge decision with a low probability (e.g., 5%)
            if bern(self.global_params.initial_medicine_discharge_prob, self.rng_probs):
                patient.discharged = True
                patient.discharge_time = self.env.now
                time_in_system = self.env.now - patient.arrival_time
                self.record_result(patient.id, "Discharge Decision Point", "after_initial_medical_assessment")
                self.record_result(patient.id, "Time in System", time_in_system)
                self.record_event(patient, "discharge")
                if getattr(patient, "discharge_event", None) and not patient.discharge_event.triggered:
                    patient.discharge_event.succeed()
                return
                        
            # If already admitted, skip consultant assessment
            if patient.amu_admission_time is not None and patient.amu_admission_time <= self.env.now:
                log(lambda: f"Patient {patient.id} already admitted to AMU, skipping consultant assessment.")
                return

            self.env.process(self.consultant_assessment(patient))
        
    # Simulate consultant assessment process
    def consultant_assessment(self, patient):
        """Simulate consultant review after initial medical assessment."""
    
        with self.consultant.request(priority = patient.priority) as req:
            self.record_result(patient.id, "Queue Length Consultant", len(self.consultant.queue))
            log(lambda: f"[{self.env.now:.2f}] Consultant capacity: {self.consultant.capacity}")
            log(lambda: f"[{self.env.now:.2f}] Consultant queue length before request: {len(self.consultant.queue)}")
            yield req 

             # If patient was already admitted to AMU, skip consultant assessment
            if patient.amu_admission_time is not None and patient.amu_admission_time <= self.env.now:
                log(lambda: f"{self.env.now:.2f}: Patient {patient.id} already admitted to AMU. Skipping consultant assessment.")
                return
            
            # Calculate the waiting time from end of referral to start of consultant assessment
            end_consultant_q = self.env.now
            log(lambda: f"{end_consultant_q:.2f}: Consultant starts assessing Patient {patient.id}")
            wait_for_consultant = end_consultant_q - patient.referral_to_medicine_time
            self.record_result(patient.id, "Referral to Consultant Assessment", wait_for_consultant)
            self.record_event(patient, "consultant_assessment_start")

            # Simulate consultant assessment time using the lognormal distribution
            
            # consultant_assessment_time = self.consultant_time_distribution.sample()
            consultant_assessment_time = self.rng_service.lognormal(
            mean=self.global_params.mu_consultant_assessment_time,
            sigma=self.global_params.sigma_consultant_assessment_time
            )

            patient.consultant_assessment_time = consultant_assessment_time
            yield self.env.timeout(consultant_assessment_time)  # Simulate assessment duration
            log(lambda: f"Patient {patient.id} spends {consultant_assessment_time} minutes with consultant")
            self.record_result(patient.id, "Consultant Assessment Service Time", consultant_assessment_time)

            # Calculate and record the total time from arrival to the end of consultant assessment
            total_time_consultant = self.env.now - patient.arrival_time
            self.record_result(patient.id, "Arrival to Consultant Assessment", total_time_consultant)
            self.record_event(patient, "consultant_assessment_end")
            
            # Discharge after consolutant assessment logic
            if bern(self.global_params.consultant_discharge_prob, self.rng_probs):
                patient.discharged = True
                patient.discharge_time = self.env.now
                time_in_system = self.env.now - patient.arrival_time
                self.record_result(patient.id, "Discharge Decision Point", "discharged_after_consultant_assessment")
                self.record_result(patient.id, "Time in System", time_in_system)
                self.record_event(patient, "discharge")
                if getattr(patient, "discharge_event", None) and not patient.discharge_event.triggered:
                    patient.discharge_event.succeed()
                return
      
    # --- Run Method ---

    def run(self):

        """Run the simulation."""
        
        # Start the patient arrival processes

        # Generate arrivals 
        self.env.process(self.generate_arrivals())
   
        # Start the AMU bed generation process
        self.env.process(self.generate_amu_beds())

        # Start the SDEC capacity generation process 
        self.env.process(self.generate_sdec_slots()) 

        # Start monitoring the AMU bed queue
        self.env.process(self.monitor_amu_queue())

        # Start monitoring the consultant queue
        self.env.process(self.monitor_ed_assessment_queue_length())

        # Start monitoring ED doctor blocking
        self.env.process(self.monitor_ed_doctor_blocks(interval=15))

        # Start monitoring the consultant queue
        self.env.process(self.monitor_consultant_queue_length())

        # Start the triage nurse obstruction process for shifts
        self.env.process(self.obstruct_triage_nurse())

        # Start the ED doctor obstruction process for shifts
        self.env.process(self.obstruct_ed_doctor())

        # Start the ED doctor obstruction process
        self.env.process(self.obstruct_medical_doctor())

        # Start the consultant obstruction process
        self.env.process(self.obstruct_consultant())

        # Start the ED doctor obstruction process for breaks
        self.doctor_break_cycle(break_dur=30, jitter_max=30)

        # Run the simulation
        self.env.run(until=self.global_params.simulation_time)

         # --- build DataFrames once from buffers (FAST) ---
        if getattr(self, "_event_log_buf", None):
            self.event_log_df = pd.DataFrame(self._event_log_buf)
        if getattr(self, "_ed_q_buf", None):
            self.ed_assessment_queue_monitoring_df = pd.DataFrame(self._ed_q_buf)
        if getattr(self, "_cons_q_buf", None):
            self.consultant_queue_monitoring_df = pd.DataFrame(self._cons_q_buf)
        if getattr(self, "_ed_block_buf", None):
            self.ed_doctor_block_monitoring_df = pd.DataFrame(self._ed_block_buf)
        if getattr(self, "_amu_q_buf", None):
            self.amu_queue_df = pd.DataFrame(self._amu_q_buf)

        # Add 4-hour breach column to individual results
        self.run_results_df['Breach 4hr'] = self.run_results_df['Time in System'].gt(240)

        # Add 12-hour breach column to individual results
        self.run_results_df['Breach 12hr'] = self.run_results_df['Time in System'].gt(720)

        # Tag for binding across scenarios
        self.run_results_df.loc[:, "Scenario"]      = self._scenario_name
        self.run_results_df.loc[:, "DT Threshold"]  = self._dt_threshold
        self.run_results_df["DT Eligible"]          = self.run_results_df["DT Eligible"].fillna(False)
        self.run_results_df["Pathway Start"]        = self.run_results_df["Pathway Start"].fillna("ED")

    def outcome_measures(self):

        # Remove cool down to prevent termination bias
        obs_start = self.global_params.burn_in_time
        obs_end   = self.global_params.simulation_time - self.global_params.cool_down_time

        copy = self.run_results_df.copy()
        copy = copy[(copy["Simulation Arrival Time"] >= obs_start) &
            (copy["Simulation Arrival Time"] <  obs_end)]

         # Window length (days) for rate normalisation
        window_minutes = max(obs_end - obs_start, 1)
        simulation_days = window_minutes / 1440.0
        total_attendances = len(copy)
        attendances_per_day = total_attendances / simulation_days

        #  Normalise DT fields for summaries ---
        # (baseline will have NaNs → treat as False / ED)
        copy["DT Eligible"] = copy.get("DT Eligible", False)
        copy["DT Eligible"] = copy["DT Eligible"].fillna(False)
        copy["Pathway Start"] = copy.get("Pathway Start", "ED").fillna("ED")

        # Aggregate by Hour
        hourly_data = copy.groupby(['Hour of Arrival']).agg({
            'Arrival to Triage Nurse Assessment': ['mean'],
            'Arrival to ED Assessment': ['mean'],
            'Arrival to Referral': ['mean'],
            'Arrival to Medical Assessment': ['mean'],
            'Arrival to Consultant Assessment': ['mean'], 
            'SDEC Appropriate': ['mean'],
            'SDEC Accepted': ['mean'], 
            'Time in System': ['mean'], 
            'Breach 4hr': ['mean'],
            'Breach 12hr': ['mean'],
            # Add additional measures as necessary
        }).reset_index()

        # Rename columns for clarity
        hourly_data.columns = ['hour_of_arrival', 'mean_arrival_triage', 'mean_arrival_ed',
                           'mean_arrival_referral', 'arrival_medical_assessment',
                           'mean_arrival_consultant_assessment', 'prop_sdec_appropriate', 'prop_sdec_accepted',  'time_in_ed',  'prop_>4hr_breach', 'prop_>12hr_breach']

        # Aggregate by Day
        daily_data = copy.groupby(['Day of Arrival']).agg({
            'Arrival to Triage Nurse Assessment': ['mean'],
            'Arrival to ED Assessment': ['mean'],
            'Arrival to Referral': ['mean'],
            'Arrival to Medical Assessment': ['mean'],
            'Arrival to Consultant Assessment': ['mean'],
            'SDEC Appropriate': ['mean'],
            'SDEC Accepted': ['mean'],  
            'Time in System': ['mean'],  
            'Breach 4hr': ['mean'],
            'Breach 12hr': ['mean'],
        }).reset_index()

        # Rename columns for clarity
        daily_data.columns = ['day_of_arrival', 'mean_arrival_triage', 'mean_arrival_ed',
                           'mean_arrival_referral', 'arrival_medical_assessment',
                           'mean_arrival_consultant_assessment', 'sdec_appopriate', 'prop_sdec_accepted', 'time_in_ed', 'prop_>4hr_breach', 'prop_>12hr_breach']

        # Now, aggregate across all runs
        complete_data = copy.agg({
            'Arrival to Triage Nurse Assessment': ['mean'],
            'Arrival to ED Assessment': ['mean'],
            'Arrival to Referral': ['mean'],
            'Arrival to Medical Assessment': ['mean'],
            'Arrival to Consultant Assessment': ['mean'],
            'SDEC Appropriate': ['mean'],
            'SDEC Accepted': ['mean'],
            'Time in System': ['mean'],  
            'Breach 4hr': ['mean'],
            'Breach 12hr': ['mean'],
        }).T.reset_index()

        complete_data.columns = ['measure', 'mean_value']

        # Append attendances per day as summary row
        attendance_row = pd.DataFrame([["Mean ED Attendances per Day", attendances_per_day]], columns=["measure", "mean_value"])
        complete_data = pd.concat([complete_data, attendance_row], ignore_index=True)

        # Add proportion referred to Medicine
        prop_medicine = copy['ED Disposition'].value_counts(normalize=True).get("Refer - Medicine", 0)
        complete_data.loc[len(complete_data.index)] = ['Proportion Referred - Medicine', prop_medicine]

         # Add SDEC appropriate
        sdec_appropriate = copy[copy["SDEC Appropriate"] == True]
        if len(sdec_appropriate) > 0:
                prop_sdec_accepted_among_appropriate = sdec_appropriate["SDEC Accepted"].mean()
        else:
            prop_sdec_accepted_among_appropriate = None  # or np.nan

        sdec_row = pd.DataFrame(
            [["SDEC Accepted (of Appropriate)", prop_sdec_accepted_among_appropriate]],
                columns=["measure", "mean_value"]
            )
        
        complete_data = pd.concat([complete_data, sdec_row], ignore_index=True)


        # --- Direct-triage summaries ---

        dt_eligible_share = copy["DT Eligible"].mean() if len(copy) else np.nan
        dt_used_share = (copy["Pathway Start"] == "Direct-Medicine").mean() if len(copy) else np.nan
        complete_data = pd.concat([
            complete_data,
            pd.DataFrame([
                ["Direct Triage Eligible (share)", dt_eligible_share],
                ["Direct Triage Used (share)", dt_used_share],
            ], columns=["measure", "mean_value"])
        ], ignore_index=True)

        # Store the aggregated results
        hourly_data["run_number"]  = self.run_number
        daily_data["run_number"]   = self.run_number
        complete_data["run_number"] = self.run_number

        scenario = getattr(self, "_scenario_name", "baseline")
        thresh   = getattr(self, "_dt_threshold", np.nan)
        for df in (hourly_data, daily_data, complete_data):
            df["Scenario"] = scenario
            df["DT Threshold"] = thresh

        self.results_hourly = hourly_data
        self.results_daily = daily_data
        self.results_complete = complete_data
    
        return hourly_data, daily_data, complete_data
    
