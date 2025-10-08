# src/model_alt.py
import numpy as np
from src.model import Model
from src.helper import extract_hour

class AltModel(Model):
    """
    SDEC behaviour is unchanged when accepted.
    If SDEC rejects / is closed / or has no capacity, apply this rule:
      - NEWS2 <= 4
      - acuity != 1
      - admission_probability >= direct_triage_threshold  (set in run.py)
    If the rule passes: refer directly to Medicine; otherwise: go to ED assessment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Pull the threshold placed on global_params by the trial/run
        th = getattr(self.global_params, "direct_triage_threshold", np.nan)
        try:
            th = float(th)
        except Exception:
            th = np.nan

        self._dt_threshold  = th
        self._scenario_name = f"alt_dt_{th:.2f}" if np.isfinite(th) else "alt_dt_unset"

    # ---------- Fallback after SDEC rejects: choose ED vs Direct-to-Medicine ----------
    def ed_or_direct(self, patient):
        # Prefer the value on global_params; fall back to the cached _dt_threshold
        threshold = getattr(self.global_params, "direct_triage_threshold", self._dt_threshold)
        if not (isinstance(threshold, (int, float)) and 0 <= threshold <= 1):
            raise ValueError("direct_triage_threshold must be in [0,1].")

        # RULE: NEWS2 ≤ 4, acuity != 1, prob ≥ threshold
        news_ok   = (patient.news2 <= 4)
        acuity_ok = (patient.acuity != 1)
        prob_ok   = (patient.admission_probability >= threshold)
        eligible  = news_ok and acuity_ok and prob_ok

        # Record for analysis
        self.record_result(patient.id, "DT Eligible", bool(eligible))
        self.record_result(patient.id, "Pathway Start", "Direct-Medicine" if eligible else "ED")

        if eligible:
            return self.env.process(self.direct_triage_to_medicine(patient))
        else:
            return self.env.process(self.ed_assessment(patient))

    # ---------- Perform the direct referral (reuses downstream logic) ----------
    def direct_triage_to_medicine(self, patient):
        """Bypass ED assessment—refer straight to Medicine (SDEC already rejected)."""
        # Keep disposition label identical so existing reports work
        patient.ed_disposition = "Refer - Medicine"
        self.record_result(patient.id, "ED Disposition", patient.ed_disposition)

        patient.referral_to_medicine_time = self.env.now
        self.record_result(
            patient.id, "Arrival to Referral",
            patient.referral_to_medicine_time - patient.arrival_time
        )
        self.record_event(patient, "direct_triage_referral")

        # Standard post-referral pipeline (unchanged)
        self.env.process(self.refer_to_amu_bed(patient))
        yield self.env.process(self.initial_medical_assessment(patient))

    # ---------- Only override: SDEC accepted path unchanged; fallback becomes ed_or_direct ----------
    def refer_to_sdec(self, patient, fallback_process):
        """SDEC unchanged for Accepted. If Rejected/Closed/No capacity -> ed_or_direct()."""

        # Paediatrics
        if not patient.adult:
            self.record_result(patient.id, "SDEC Accepted", False)
            self.record_result(patient.id, "SDEC Decision Reason", "Rejected: Paediatric")
            yield self.ed_or_direct(patient)
            return

        # Appropriateness
        if not patient.sdec_appropriate:
            self.record_result(patient.id, "SDEC Accepted", False)
            self.record_result(patient.id, "SDEC Decision Reason", "Rejected: Not Appropriate")
            yield self.ed_or_direct(patient)
            return

        # Opening hours
        current_hour = extract_hour(self.env.now)
        if (current_hour < self.global_params.sdec_open_hour) or (current_hour >= self.global_params.sdec_close_hour):
            self.record_result(patient.id, "SDEC Accepted", False)
            self.record_result(patient.id, "SDEC Decision Reason", "Rejected: SDEC Closed")
            yield self.ed_or_direct(patient)
            return

        # Capacity
        if len(self.sdec_capacity.items) > 0:
            # ACCEPT (unchanged)
            yield self.sdec_capacity.get()
            self.record_result(patient.id, "SDEC Accepted", True)
            self.record_result(patient.id, "SDEC Decision Reason", "Accepted")
            patient.ed_disposition = "SDEC Accepted"
            self.record_result(patient.id, "ED Disposition", "SDEC Accepted")
            self.record_result(patient.id, "Discharge Decision Point", "after_sdec_acceptance")
            self.record_event(patient, "sdec_acceptance")
            patient.discharged = True
            patient.discharge_time = self.env.now
            self.record_result(patient.id, "Time in System", patient.discharge_time - patient.arrival_time)
            return

        # No capacity -> new fallback
        self.record_result(patient.id, "SDEC Accepted", False)
        self.record_result(patient.id, "SDEC Decision Reason", "Rejected: No Capacity")
        yield self.ed_or_direct(patient)
        return
