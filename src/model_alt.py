# src/model_alt.py
import numpy as np
from src.model import Model
from src.helper import extract_hour
from src.helper import exp_rv, wchoice, bern 
from src.helper import dt_threshold_from_top_percent

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

        self._scenario_name = "alt"
        self._policy_label  = "clinical_only"
        self._dt_threshold  = np.nan

        # Treat direct_triage_threshold as a PERCENT (e.g., 10.0 = top 10%)
        pct = float(getattr(self.global_params, "direct_triage_threshold", 0) or 0)
        pct = max(0.0, min(100.0, pct))

        if pct > 0.0:
            th_val, label = dt_threshold_from_top_percent(self._p_raw, self._p_cal, pct)
            self._dt_threshold = float(th_val) 
            self._policy_label = label
        else:
            self._dt_threshold = np.nan
            self._policy_label = "clinical_only"     

    def ed_or_direct(self, patient):
        """
        After SDEC rejects/closed/no capacity:
        1) Decide Medicine intent with p_cal (Bernoulli) – once.
        2) If not Medicine → ED.
        3) If Medicine → clinical screen; if a cutoff is set, also require p_raw ≥ cutoff.
        """

        # 1) Medicine intent (rate control) — do once
        if getattr(patient, "medicine_intent", None) is None:
            patient.medicine_intent = bern(patient.referral_prob_cal, self.rng_probs)
            self.record_result(patient.id, "Referral Medicine", bool(patient.medicine_intent))

        if not patient.medicine_intent:
            # Not Medicine → always ED path
            self.record_result(patient.id, "DT Eligible", False)
            self.record_result(patient.id, "Pathway Start", "ED")
            self.record_result(patient.id, "DT Block Reason", "Not Medicine Intent")
            return self.env.process(self.ed_assessment(patient))

        # 2) Route policy for Medicine-intent patients
        th = self._dt_threshold  # ← already materialised in __init__ (raw cutoff or NaN)
        news_ok   = (patient.news2 <= 4)
        acuity_ok = (patient.acuity != 1)

        if np.isfinite(th):
            eligible = news_ok and acuity_ok and (patient.referral_score_raw >= th)
            block_reason = None if eligible else (
                "Clinical screen" if not (news_ok and acuity_ok) else "Score<threshold"
            )
        else:
            # clinical-only mode (no raw cutoff)
            eligible = news_ok and acuity_ok
            block_reason = None if eligible else "Clinical screen"

        self.record_result(patient.id, "DT Eligible", bool(eligible))

        if eligible:
            self.record_result(patient.id, "Pathway Start", "Direct-Medicine")
            self.record_result(patient.id, "ED Pathway Subtype", "Direct—Medicine")
            self.record_result(patient.id, "DT Block Reason", np.nan)
            return self.env.process(self.direct_triage_to_medicine(patient))
        else:
            self.record_result(patient.id, "Pathway Start", "ED")
            self.record_result(patient.id, "ED Pathway Subtype", "ED—Medicine")
            self.record_result(patient.id, "DT Block Reason", block_reason)
            return self.env.process(self.ed_assessment(patient))

        
    def direct_triage_to_medicine(self, patient):
        """Bypass ED assessment—refer straight to Medicine (SDEC already rejected)."""
        # Keep disposition label identical so existing reports work
        patient.ed_disposition = "Refer - Medicine"
        self.record_result(patient.id, "ED Disposition", patient.ed_disposition)

        patient.referral_to_medicine_time = self.env.now
        self.record_result(
            patient.id,
            "Arrival to Referral",
            patient.referral_to_medicine_time - patient.arrival_time,
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
