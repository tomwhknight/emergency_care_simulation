from src.model import Model
from src.helper import calculate_hour_of_day, extract_day_of_week, extract_hour
from src.helper import Lognormal
from src.patient import Patient
import random
import pandas as pd
import numpy as np


class AltModel(Model):

    
    # --- Generator Methods --
    def generate_arrivals(self):
        print("AltModel: generate_arrivals is running") 
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
            
            mode_of_arrival = random.choices(["Ambulance", "Walk-in"], 
                                         weights=[self.global_params.ambulance_proportion, 
                                                  self.global_params.walk_in_proportion])[0]

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
            age = int(round(random.choices(age_values, weights=age_probs, k=1)[0]))

            # Assign mode of arrival and admission probability based on mode of arrival

            if mode_of_arrival == "Ambulance":
                acuity_levels = list(self.global_params.ambulance_acuity_probabilities.keys())
                acuity_weights = list(self.global_params.ambulance_acuity_probabilities.values())
            else:
                acuity_levels = list(self.global_params.walk_in_acuity_probabilities.keys())
                acuity_weights = list(self.global_params.walk_in_acuity_probabilities.values())

            acuity = random.choices(acuity_levels, weights=acuity_weights, k=1)[0]
            
            # Assign source of referral
            
            news2_values = self.news_distribution_data["news2"].tolist()
            news2_weights = self.news_distribution_data["news2_probs"].tolist()
            news2 = random.choices(news2_values, weights=news2_weights, k=1)[0]

            # Determine if patient is adult
            adult = age >= 17  
           
            # Determine group label based on patient age and NEWS2
            if not adult:
                group = "under_17"
            elif age >= 75 and news2 > 4:
                group = "high_age_high_news"
            elif age >= 75 and news2 <= 4:
                group = "high_age_low_news"
            elif 17 <= age < 75 and news2 <= 4:
                group = "working_age_low_news"
            elif 17 <= age < 75 and news2 > 4:
                group = "working_age_high_news"
            else:
                group = "unknown"

            if adult:
                params = self.admission_prob_distribution_data[
                    self.admission_prob_distribution_data["group"] == group
                    ].iloc[0]

                # Sample from Beta distribution
                if params["dist"] == "beta":
                    a = params["shape1"]
                    b = params["shape2"]
                    admission_prob = random.betavariate(a, b) 

                elif params["dist"] == "norm":
                    mu = params["mean"]
                    sigma = params["sd"]
                    admission_prob = random.gauss(mu, sigma)

                # Clip probability to [0, 1]
                admission_prob = min(max(admission_prob, 0), 1)

            else:
                admission_prob = np.nan  # or np.nan if using pandas later
            
            # Assign source of referral

            source_of_referral = random.choices(
                ["GP", "ED"],
            weights=[
                self.global_params.proportion_direct_primary_care,       # 0.07
                1 - self.global_params.proportion_direct_primary_care    # 0.93
                ]
            )[0]

            # SDEC appropriate

            if adult:
                sdec_appropriate = random.random() < self.global_params.sdec_appropriate_rate
            else:
                sdec_appropriate = np.nan

            # Stochastic variation of ED disposition 
            
            if not adult:
                ed_disposition = random.choices(
                    ["Discharge", "Refer - Paeds"],
                    weights=[0.9, 0.1],
                    k=1
                )[0] 

            else:
                if random.random() < self.global_params.medical_referral_rate:
                    ed_disposition = "Refer - Medicine"
                else:
                    if random.random() < self.global_params.speciality_referral_rate:
                        ed_disposition = "Refer - Speciality"
                    else:
                        ed_disposition = "Discharge"
                        
          
            # --- Determine priority level ---
            if acuity in [1, 2] or news2 > 4:
                priority = 0  # Higher priority
            else:
                priority = 1  # Lower priority


            # Create instance of patient class
            
            patient = Patient(
            self.patient_counter,
            arrival_time,
            current_day,
            clock_hour,
            current_hour,
            source_of_referral,
            mode_of_arrival,
            age,
            adult,
            news2,
            admission_prob, 
            acuity,
            sdec_appropriate,
            ed_disposition,
            priority
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
            "Admission Probability": admission_prob ,
            "Source of Referral": source_of_referral,
            "Acuity": acuity,
            "ED Disposition": ed_disposition,


            # --- Triage-Related Metrics ---
            "Queue Length Walk in Triage Nurse": np.nan,
            "Queue Length Ambulance Triage Nurse": np.nan,
            "Arrival to Triage Nurse Assessment": np.nan,
            "Triage Nurse Assessment Service Time": np.nan,

            # --- SDEC Referral ---
            "SDEC Appropriate": sdec_appropriate,
            "SDEC Accepted": np.nan,
            "SDEC Decision Reason": "",

            # --- ED Assessment Metrics ---
            "Queue Length ED doctor": np.nan,
            "Arrival to ED Assessment": np.nan,
            "ED Assessment Time Total": np.nan,
            "ED Assessment Service Time": np.nan,
            "ED Assessment to Decision": np.nan,

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

            if patient.ed_disposition == "Refer - Paeds":
                self.record_result(patient.id, "Discharge Decision Point", "ed_referred_paeds")
            
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

    
    
    def walk_in_triage(self, patient):
        """Modified triage logic for walk-ins with conditional ED bypass"""
        print(f"Walk-in Triage Queue at time of request: {len(self.walk_in_triage_nurse.queue)} patients at time {self.env.now}")
        with self.walk_in_triage_nurse.request() as req:
            yield req
            self.record_result(patient.id, "Queue Length Walk in Triage Nurse", len(self.walk_in_triage_nurse.queue))

            triage_nurse_assessment_start_time = self.env.now
            self.record_result(patient.id, "Arrival to Triage Nurse Assessment", triage_nurse_assessment_start_time - patient.arrival_time)
            print(f"Patient {patient.id} starts triage assessment at {triage_nurse_assessment_start_time}")

            triage_nurse_assessment_time = self.triage_time_distribution.sample()
            yield self.env.timeout(triage_nurse_assessment_time)
            print(f"Patient {patient.id} spends {triage_nurse_assessment_time} minutes in triage")

            self.record_result(patient.id, "Triage Nurse Assessment Service Time", triage_nurse_assessment_time)
            patient.triage_nurse_assessment_time = triage_nurse_assessment_time

        # --- Determine priority level and conditional referral ---
        acuity = patient.acuity
        news2 = patient.news2
        if (acuity in [3, 4, 5] or news2 < 5) and patient.ed_disposition == "Refer - Medicine":
            print(f"Patient {patient.id} bypasses ED assessment")
            patient.referral_to_medicine_time = self.env.now
            yield self.env.process(self.handle_ed_referral(patient))
        else:
            yield self.env.process(self.refer_to_sdec(patient, fallback_process=self.ed_assessment))

    def ambulance_triage(self, patient):
        """Modified triage logic for ambulance arrivals with conditional ED bypass"""
        with self.ambulance_triage_nurse.request() as req:
            yield req
            self.record_result(patient.id, "Queue Length Ambulance Triage Nurse", len(self.ambulance_triage_nurse.queue))

            triage_nurse_assessment_start_time = self.env.now
            self.record_result(patient.id, "Arrival to Triage Nurse Assessment", triage_nurse_assessment_start_time - patient.arrival_time)
            print(f"Patient {patient.id} starts triage assessment at {triage_nurse_assessment_start_time}")

            triage_nurse_assessment_time = self.triage_time_distribution.sample()
            yield self.env.timeout(triage_nurse_assessment_time)
            print(f"Patient {patient.id} spends {triage_nurse_assessment_time} minutes in triage")

            self.record_result(patient.id, "Triage Nurse Assessment Service Time", triage_nurse_assessment_time)
            patient.triage_nurse_assessment_time = triage_nurse_assessment_time

        # --- Determine priority level and conditional referral ---
        acuity = patient.acuity
        news2 = patient.news2
        if (acuity in [3, 4, 5] or news2 < 5) and patient.ed_disposition == "Refer - Medicine":
            print(f"Patient {patient.id} bypasses ED assessment")
            patient.referral_to_medicine_time = self.env.now
            yield self.env.process(self.handle_ed_referral(patient))
        else:
            yield self.env.process(self.refer_to_sdec(patient, fallback_process=self.ed_assessment))
