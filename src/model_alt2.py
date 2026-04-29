# src/model_alt2.py
import numpy as np
from scipy.special import erfinv

from src.patient import Patient
from src.model import Model, log
from src.helper import bern, extract_hour


class AltModel2(Model):
    """
    Alt-2 scenario:
    - For ED patients referred to Medicine, if the ED decision is made between 09:00 and <20:00,
      bypass the post-assessment ED decision delay and attempt direct consultant assessment.
    - If consultant assessment has NOT STARTED by 21:00, hand over to the standard medical pathway.

    Key timing rules:
    - 20:00 (8pm): entry to direct-consultant route closes
    - 21:00 (9pm): spillover handover cut-off for those already in the direct-consultant queue
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scenario metadata for tagging outputs
        self._scenario_name = "alt2"
        self._policy_label = (
            "ed_to_consultant_0900_2000_skip_ed_decision_delay_"
            "handover_to_medicine_if_not_seen_by_2100"
        )
        self._dt_threshold = np.nan  # not used here; kept for compatibility

        # Scenario window definitions (minutes from midnight)
        self.alt2_start_minute = 9 * 60              # 09:00
        self.alt2_entry_close_minute = 20 * 60       # 20:00 (last entry <20:00)
        self.alt2_handover_cutoff_minute = 21 * 60   # 21:00 (spillover cut-off)


        # ---------------------------------------------
        # Alt-2 optional staffing redirection (medical -> ED)
        # ---------------------------------------------
        self.alt2_medical_redirection_on = bool(
            getattr(self.global_params, "alt2_medical_redirection_on", False)
        )

        # Number of medical staff redirected to ED in-hours
        self.alt2_redirection_n_medical_to_ed = int(
            getattr(self.global_params, "alt2_redirection_n_medical_to_ed", 3)
        )

        # Keep at least this many medical doctors available during redirection
        self.alt2_min_medical_capacity_floor = int(
            getattr(self.global_params, "alt2_min_medical_capacity_floor", 2)
        )

        # Redirection window (defaults aligned to Alt2 operational load incl spillover)
        self.alt2_redirection_start_hour = int(
            getattr(self.global_params, "alt2_redirection_start_hour", 9)
        )
        self.alt2_redirection_end_hour = int(
            getattr(self.global_params, "alt2_redirection_end_hour", 21)
        )



    # -------------------------------------------------
    # Alt-2 helper methods
    # -------------------------------------------------
    def _minute_of_day(self, sim_time):
        """Return minute-of-day (0..1439) from simulation time in minutes."""
        return float(sim_time % (24 * 60))

    def _alt2_direct_consultant_entry_open(self, sim_time):
        """
        Eligibility window checked at ED decision time:
        09:00 <= t < 20:00
        """
        t_mod = self._minute_of_day(sim_time)
        return self.alt2_start_minute <= t_mod < self.alt2_entry_close_minute

    def _time_until_same_day_handover_cutoff(self, sim_time):
        """
        Minutes until the same-day 21:00 handover cut-off.
        """
        day_start = sim_time - (sim_time % (24 * 60))
        cutoff_abs = day_start + self.alt2_handover_cutoff_minute
        return max(0.0, float(cutoff_abs - sim_time))

    
    def _alt2_staff_redirection_active(self, sim_time):
        """
        Alt-2 staffing redirection window:
        default 09:00 <= t < 21:00
        """
        if not self.alt2_medical_redirection_on:
            return False

        hour = int((sim_time // 60) % 24)
        start = self.alt2_redirection_start_hour
        end = self.alt2_redirection_end_hour

        # Support same-day and overnight windows
        if start < end:
            return start <= hour < end
        else:
            return (hour >= start) or (hour < end)

    # -------------------------------------------------
    # Alt-2 alteranative staffing model
    # -------------------------------------------------

    def get_available_doctors(self, current_time_minutes, start_block=5, end_cutoff=45):
        """
        Alt-2 override:
        - baseline ED rota availability from parent
        - optional +N medical staff redirected to ED in-hours
        """
        base_available = super().get_available_doctors(
            current_time_minutes=current_time_minutes,
            start_block=start_block,
            end_cutoff=end_cutoff,
        )

        if not self._alt2_staff_redirection_active(current_time_minutes):
            return base_available

        boosted_available = base_available + self.alt2_redirection_n_medical_to_ed

        # Respect physical ED cap (must be high enough in g/run if you want full +N realised)
        return int(min(self.global_params.max_ed_doctor_capacity, boosted_available))

    def obstruct_medical_doctor(self):
        """
        Alt-2 override of medical staffing:
        - uses baseline hour-level staffing from CSV
        - optionally redirects N medical staff to ED in-hours
        - preserves a minimum floor in medical
        """
        while True:
            current_hour = extract_hour(self.env.now)

            # Baseline available medical staff from your staffing CSV
            baseline_available = int(
                self.medical_staffing_data.loc[
                    self.medical_staffing_data["hour"] == current_hour, "num_staff"
                ].values[0]
            )

            available_medical_doctors = baseline_available

            # Apply Alt-2 staffing redirection (medical -> ED) during active window
            if self._alt2_staff_redirection_active(self.env.now):
                available_medical_doctors = max(
                    self.alt2_min_medical_capacity_floor,
                    baseline_available - self.alt2_redirection_n_medical_to_ed
                )

            # Clamp within physical bounds of the resource
            available_medical_doctors = max(
                0,
                min(int(self.medical_doctor.capacity), int(available_medical_doctors))
            )

            # Block down to target available capacity for this hour
            medical_doctors_to_block = int(self.medical_doctor.capacity) - available_medical_doctors

            if medical_doctors_to_block > 0:
                log(lambda: (
                    f"{self.env.now:.2f}: Alt2 medical blocking {medical_doctors_to_block} "
                    f"(baseline={baseline_available}, available={available_medical_doctors}, "
                    f"redir_active={self._alt2_staff_redirection_active(self.env.now)})"
                ))
                for _ in range(medical_doctors_to_block):
                    self.env.process(self.block_medical_doctor(60))
            else:
                log(lambda: (
                    f"{self.env.now:.2f}: Alt2 medical no blocking "
                    f"(baseline={baseline_available}, available={available_medical_doctors}, "
                    f"redir_active={self._alt2_staff_redirection_active(self.env.now)})"
                ))

            yield self.env.timeout(60)


    # -------------------------------------------------
    # Alt-2 referral handling (post-ED medicine referral)
    # -------------------------------------------------
    def handle_ed_referral_alt2(self, patient):
        """
        Alt-2 post-ED referral handler:
        - start AMU bed request in parallel (baseline behaviour)
        - attempt direct consultant until 21:00
        - if not started by 21:00, hand over to initial medical assessment
        """
        # Keep AMU bed request behaviour unchanged and parallel
        self.env.process(self.refer_to_amu_bed(patient))

        # Wait for direct consultant attempt / handover to complete
        yield self.env.process(self.consultant_assessment_direct_until_cutoff(patient))

    def consultant_assessment_direct_until_cutoff(self, patient):
        """
        Attempt direct consultant assessment (Alt-2).

        Rule:
        - If consultant starts before 21:00 -> proceed with consultant assessment
        - If consultant has NOT started by 21:00 -> hand over to standard medical pathway

        'Seen by consultant by 9pm' is implemented as consultant START before 21:00.
        If grant and timeout happen exactly together at 21:00, we treat that as handover.
        """
        # Safety: if already admitted to AMU before this starts, do nothing
        if getattr(patient, "amu_admission_time", None) is not None and patient.amu_admission_time <= self.env.now:
            return

        # If already at/past handover cutoff (e.g. due to delay), hand over immediately
        wait_to_cutoff = self._time_until_same_day_handover_cutoff(self.env.now)
        if wait_to_cutoff <= 0:
            if getattr(patient, "amu_admission_time", None) is not None and patient.amu_admission_time <= self.env.now:
                return
            self.record_event(patient, "alt2_handover_to_medicine")
            self.record_result(patient.id, "ED Pathway Subtype", "ED—Alt2 Handover to Medicine")
            yield self.env.process(self.initial_medical_assessment(patient))
            return

        # Record queue length and trace event
        self.record_result(patient.id, "Queue Length Consultant", len(self.consultant.queue))
        self.record_event(patient, "alt2_direct_consultant_attempt")

        # Manual request (not context manager) so we can cancel if cutoff happens first
        req = self.consultant.request(priority=patient.priority)
        cutoff_evt = self.env.timeout(wait_to_cutoff)

        outcome = yield req | cutoff_evt

        # -------------------------------------------------
        # A) Cutoff reached before consultant start -> handover
        # -------------------------------------------------
        if (cutoff_evt in outcome) and (req not in outcome):
            # If AMU admission already happened while waiting, no handover needed
            if getattr(patient, "amu_admission_time", None) is not None and patient.amu_admission_time <= self.env.now:
                try:
                    if not req.triggered:
                        req.cancel()
                except Exception:
                    pass
                return

            try:
                if not req.triggered:
                    req.cancel()
            except Exception:
                pass

            self.record_event(patient, "alt2_handover_to_medicine")
            self.record_result(patient.id, "ED Pathway Subtype", "ED—Alt2 Handover to Medicine")

            # Standard pathway from here
            yield self.env.process(self.initial_medical_assessment(patient))
            return

        # -------------------------------------------------
        # B) Tie at exactly cutoff (req + timeout same instant) -> handover
        # -------------------------------------------------
        if (cutoff_evt in outcome) and (req in outcome):
            # If AMU admission already happened, just release and stop
            if getattr(patient, "amu_admission_time", None) is not None and patient.amu_admission_time <= self.env.now:
                try:
                    self.consultant.release(req)
                except Exception:
                    pass
                return

            try:
                self.consultant.release(req)
            except Exception:
                pass

            self.record_event(patient, "alt2_handover_to_medicine")
            self.record_result(patient.id, "ED Pathway Subtype", "ED—Alt2 Handover to Medicine")
            yield self.env.process(self.initial_medical_assessment(patient))
            return

        # -------------------------------------------------
        # C) Consultant started before cutoff -> do consultant assessment
        # -------------------------------------------------
        try:
            # If AMU admission occurred while waiting, skip consultant assessment
            if getattr(patient, "amu_admission_time", None) is not None and patient.amu_admission_time <= self.env.now:
                log(lambda: f"{self.env.now:.2f}: Patient {patient.id} already admitted to AMU. Skipping consultant assessment (Alt-2 direct).")
                return

            end_consultant_q = self.env.now
            wait_for_consultant = end_consultant_q - patient.referral_to_medicine_time
            self.record_result(patient.id, "Referral to Consultant Assessment", wait_for_consultant)
            self.record_event(patient, "consultant_assessment_start")
            self.record_result(patient.id, "ED Pathway Subtype", "ED—Alt2 Direct Consultant")

            consultant_assessment_time = self.rng_service.lognormal(
                mean=self.global_params.mu_consultant_assessment_time,
                sigma=self.global_params.sigma_consultant_assessment_time,
            )

            patient.consultant_assessment_time = consultant_assessment_time
            yield self.env.timeout(consultant_assessment_time)
            self.record_result(patient.id, "Consultant Assessment Service Time", consultant_assessment_time)

            total_time_consultant = self.env.now - patient.arrival_time
            self.record_result(patient.id, "Arrival to Consultant Assessment", total_time_consultant)
            self.record_event(patient, "consultant_assessment_end")

            # Baseline discharge-after-consultant logic
            if (patient.amu_admission_time is None) and bern(self.global_params.consultant_discharge_prob, self.rng_probs):
                patient.discharged = True
                patient.discharge_time = self.env.now
                time_in_system = self.env.now - patient.arrival_time
                self.record_result(patient.id, "Discharge Decision Point", "discharged_after_consultant_assessment")
                self.record_result(patient.id, "Time in System", time_in_system)
                self.record_event(patient, "discharge")
                if getattr(patient, "discharge_event", None) and not patient.discharge_event.triggered:
                    patient.discharge_event.succeed()
                return

        finally:
            # Release consultant slot if acquired
            try:
                self.consultant.release(req)
            except Exception:
                pass

    # -------------------------------------------------
    # Override ED assessment (needed because ED decision delay is here)
    # -------------------------------------------------
    def ed_assessment(self, patient):
        """ED assessment with Alt-2 branch for eligible medicine referrals."""

        def finalise_disposition(patient, decision_point, event_name):
            patient.discharge_time = self.env.now
            time_in_system = patient.discharge_time - patient.arrival_time
            self.record_result(patient.id, "Discharge Decision Point", decision_point)
            self.record_result(patient.id, "Time in System", time_in_system)
            self.record_event(patient, event_name)

        def assign_ed_disposition():
            """
            Mirror baseline ED disposition logic, but earlier in the flow so Alt-2 can
            branch BEFORE the ED post-assessment decision delay is applied.
            """
            if not patient.adult:
                admitted = bern(self.global_params.paediatric_referral_rate, self.rng_probs)
                if admitted:
                    patient.ed_disposition = "Refer - Paeds"
                else:
                    patient.ed_disposition = "Discharge - Paeds"

            else:
                if patient.referral_intent:
                    # Use the split computed at arrival; if absent, compute once (back-compat)
                    if getattr(patient, "intend_medicine", None) is None:
                        patient.intend_medicine = (
                            self.rng_probs.random() < self.global_params.prob_referral_to_medicine_adult
                        )

                    if patient.intend_medicine:
                        patient.ed_disposition = "Refer - Medicine"
                        self.record_result(patient.id, "Referral Medicine", True)
                    else:
                        patient.ed_disposition = "Refer - Other Speciality"
                        self.record_result(patient.id, "Referral Medicine", False)
                else:
                    patient.ed_disposition = "Discharge"
                    self.record_result(patient.id, "Referral Medicine", False)

        # --- Initial ED doctor assessment (same structure as baseline) ---
        with self.ed_doctor.request(priority=patient.priority) as req:
            q_patients = sum(1 for r in self.ed_doctor.queue if not getattr(r, "is_block", False))
            log(lambda: f"Patient {patient.id} ED Doctor Queue at time of request: {q_patients} patients at time {self.env.now}")
            self.record_result(patient.id, "Queue Length ED doctor", q_patients)
            yield req

            ed_assessment_start_time = self.env.now
            wait_time = ed_assessment_start_time - patient.arrival_time
            self.record_result(patient.id, "Arrival to ED Assessment", wait_time)
            self.record_event(patient, "ed_assessment_start")

            # --- Joint sampling of service and decision times (same as baseline) ---
            eps = 1e-12
            gamma = self.global_params.joint_gamma

            if gamma is None:
                speed_factor_service = max(self.rng_service.random(), eps)
                speed_factor_decision = max(self.rng_service.random(), eps)
            else:
                speed_factor_service = max(self.rng_service.random(), eps)
                speed_factor_decision = speed_factor_service ** gamma

            z_service = np.sqrt(2) * erfinv(2 * speed_factor_service - 1)
            z_decision = np.sqrt(2) * erfinv(2 * speed_factor_decision - 1)

            service_time_sample = np.exp(
                self.global_params.mu_ed_service_time
                + self.global_params.sigma_ed_service_time * z_service
            )
            decision_time_sample = np.exp(
                self.global_params.mu_ed_decision_time
                + self.global_params.sigma_ed_decision_time * z_decision
            )

            # Doctor works for the service time
            yield self.env.timeout(service_time_sample)

            # End of assessment
            ed_assessment_end_time = self.env.now
            service_time_actual = ed_assessment_end_time - ed_assessment_start_time
            self.record_result(patient.id, "ED Assessment Service Time", service_time_actual)

        # --- Compute excess decision delay (same as baseline) ---
        excess_decision_after_assessment = max(0.0, decision_time_sample - service_time_actual)

        # Mirror 4-hour breach push (same as baseline)
        residual = max(0.0, decision_time_sample - service_time_actual)
        projected_total = (self.env.now - patient.arrival_time) + residual

        window_start = self.global_params.adjustment_start
        window_end = self.global_params.adjustment_end
        strength = self.global_params.decision_hazard_strength_240

        if (residual > 0.0) and (window_start <= projected_total < window_end):
            w = 1.0 - (projected_total - window_start) / (window_end - window_start)
            scale = 1.0 - strength * w
            residual *= scale
            excess_decision_after_assessment = residual
            decision_time_sample = service_time_actual + residual

        # -------------------------------------------------
        # Alt-2 key change:
        # Determine ED disposition BEFORE applying post-assessment decision delay
        # -------------------------------------------------
        assign_ed_disposition()
        outcome = patient.ed_disposition

        # Check Alt-2 eligibility at ED decision time (end of ED assessment)
        alt2_eligible = (
            outcome == "Refer - Medicine"
            and self._alt2_direct_consultant_entry_open(self.env.now)
        )

        p_uptake = float(getattr(self.global_params, "alt2_direct_consultant_uptake_prob", 1.0))
        p_uptake = max(0.0, min(1.0, p_uptake))

        alt2_direct_consultant = alt2_eligible and bern(p_uptake, self.rng_probs)

        # (optional but nice for auditing)
        self.record_result(patient.id, "Alt2 Eligible", bool(alt2_eligible))
        self.record_result(patient.id, "Alt2 Uptake Prob", p_uptake)
        self.record_result(patient.id, "Alt2 Used", bool(alt2_direct_consultant))

        # -------------------------------------------------
        # Record ED decision metrics / apply delay (modified for Alt-2)
        # -------------------------------------------------
        if alt2_direct_consultant:
            # In Alt-2 eligible cases the ED post-assessment decision delay is skipped
            self.record_result(patient.id, "ED Decision Time", service_time_actual)
            self.record_result(patient.id, "Post-Assessment Decision Delay", 0.0)
            self.record_result(patient.id, "Pathway Start", "ED")
            self.record_result(patient.id, "ED Pathway Subtype", "ED—Alt2 Direct Consultant Attempt")
            self.record_event(patient, "alt2_skip_ed_decision_delay")
        else:
            # Baseline behaviour
            self.record_result(patient.id, "ED Decision Time", decision_time_sample)
            self.record_result(patient.id, "Post-Assessment Decision Delay", excess_decision_after_assessment)

            if excess_decision_after_assessment > 0:
                yield self.env.timeout(excess_decision_after_assessment)

        # Record disposition field (same as baseline)
        self.record_result(patient.id, "ED Disposition", patient.ed_disposition)

        # -------------------------------------------------
        # Handle ED outcome (same branches as baseline, except Refer-Medicine alt-2 branch)
        # -------------------------------------------------
        if outcome == "Discharge":
            finalise_disposition(patient, "ed_discharge", "discharge")
            return

        elif outcome == "Refer - Other Speciality":
            patient.referral_to_surgical_time = self.env.now
            surgical_bed_delay = float(self.rng_service.lognormal(
                mean=self.global_params.mu_surgical_bed_delay,
                sigma=self.global_params.sigma_surgical_bed_delay
            ))
            self.record_result(patient.id, "Surgical Bed Delay", surgical_bed_delay)
            yield self.env.timeout(surgical_bed_delay)

            patient.surgical_departure_time = self.env.now
            self.record_result(
                patient.id,
                "Arrival to Other Admission",
                patient.surgical_departure_time - patient.arrival_time,
            )
            self.record_result(
                patient.id,
                "Referral to Other Admission",
                patient.surgical_departure_time - patient.referral_to_surgical_time,
            )

            finalise_disposition(patient, "ed_referred_other_specialty", "referral_to_speciality")
            return

        elif outcome == "Refer - Paeds":
            finalise_disposition(patient, "ed_referred_paeds", "referral_to_paeds")
            return

        elif outcome == "Discharge - Paeds":
            finalise_disposition(patient, "ed_discharge", "discharge_paeds")
            return

        elif outcome == "Refer - Medicine":
            patient.referral_to_medicine_time = self.env.now
            total_time_referral = patient.referral_to_medicine_time - patient.arrival_time
            self.record_result(patient.id, "Arrival to Referral", total_time_referral)
            self.record_event(patient, "referral_to_medicine")

            if getattr(patient, "discharge_event", None) is None:
                patient.discharge_event = self.env.event()

            if alt2_direct_consultant:
                yield self.env.process(self.handle_ed_referral_alt2(patient))
            else:
                yield self.env.process(self.handle_ed_referral(patient))


    