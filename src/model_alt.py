# src/model_alt.py
import numpy as np
import pandas as pd
from src.patient import Patient
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
        # If they’re not even an ANY-referral candidate, just do ED.
        if not getattr(patient, "referral_intent", False):
            self.record_result(patient.id, "DT Eligible", False)
            self.record_result(patient.id, "Pathway Start", "ED")
            self.record_result(patient.id, "DT Block Reason", "Not Any-Referral Intent")
            return self.env.process(self.ed_assessment(patient))

        # --- CHANGED: gate on arrival-time intent (medicine subset), plus clinical gates, plus optional threshold ---
        news_ok   = (patient.news2 <= 4)
        acuity_ok = (patient.acuity != 1)
        th        = self._dt_threshold

        intent_any = bool(getattr(patient, "referral_intent", False))
        intent_med = bool(getattr(patient, "intend_medicine", False))

        eligible = (
            intent_any
            and intent_med
            and news_ok
            and acuity_ok
            and (not np.isfinite(th) or patient.referral_score_raw >= th)
        )
        self.record_result(patient.id, "DT Eligible", bool(eligible))
        self.record_result(patient.id, "DT Threshold", self._dt_threshold)

        if eligible:
            p_uptake = float(getattr(self.global_params, "direct_medicine_uptake_prob", 1.0))
            p_uptake = max(0.0, min(1.0, p_uptake))

            take_direct = bern(p_uptake, self.rng_probs)

            if take_direct:
                self.record_result(patient.id, "Pathway Start", "Direct-Medicine")
                self.record_result(patient.id, "ED Pathway Subtype", "Direct—Medicine")
                self.record_result(patient.id, "DT Block Reason", np.nan)
                return self.env.process(self.direct_triage_to_medicine(patient))

            # Eligible, but not taken up -> send through ED (referral volumes stay the same)
            self.record_result(patient.id, "Pathway Start", "ED")
            self.record_result(patient.id, "ED Pathway Subtype", "ED—Medicine")
            self.record_result(patient.id, "DT Block Reason", "Uptake")
            self.record_event(patient, "direct_triage_uptake_blocked")
            return self.env.process(self.ed_assessment(patient))

        if not intent_med:
            reason = "Not Medicine subset"
            subtype = "ED—Other"
        elif not (news_ok and acuity_ok):
            reason = "Clinical screen"
            subtype = "ED—Medicine"
        elif np.isfinite(th) and not (patient.referral_score_raw >= th):
            reason = "Score<threshold"
            subtype = "ED—Medicine"
        else:
            reason = "Not eligible"
            subtype = "ED—Medicine"

        self.record_result(patient.id, "Pathway Start", "ED")
        self.record_result(patient.id, "ED Pathway Subtype", subtype)
        self.record_result(patient.id, "DT Block Reason", reason)
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
