from src.global_parameters import GlobalParameters
from src.helper import rota_peak

MASTER_SEED = 20251001


# --- Calculate peak capacity first ---
shift_patterns_weekday = [
    # Tier 1 Resident
    {"role": "tier_1", "shift_name": "Early",     "start": "08:00", "end": "16:00", "count": 8, "breaks": 1},
    {"role": "tier_1", "shift_name": "Middle_1",  "start": "12:00", "end": "22:00", "count": 4, "breaks": 2},
    {"role": "tier_1", "shift_name": "Twilight",  "start": "16:00", "end": "00:00", "count": 5, "breaks": 1},
    {"role": "tier_1", "shift_name": "Night",     "start": "22:00", "end": "08:00", "count": 3, "breaks": 2},

    # Tier 2 Resident
    {"role": "tier_2", "shift_name": "Early",     "start": "08:00", "end": "16:00", "count": 4, "breaks": 1},
    {"role": "tier_2", "shift_name": "Middle",    "start": "12:00", "end": "20:00", "count": 5, "breaks": 1},
    {"role": "tier_2", "shift_name": "Twilight",  "start": "16:00", "end": "00:00", "count": 3, "breaks": 1},
    {"role": "tier_2", "shift_name": "Night",     "start": "22:00", "end": "08:00", "count": 3, "breaks": 2},

    # GP
    {"role": "GP",     "shift_name": "GP",        "start": "09:00", "end": "23:00", "count": 1, "breaks": 2},

    # ACP
    {"role": "ACP",    "shift_name": "Early",     "start": "09:00", "end": "17:00", "count": 2, "breaks": 1},
    {"role": "ACP",    "shift_name": "Late",      "start": "15:00", "end": "22:00", "count": 1, "breaks": 1},
]

# Weekend shift pattern — tuned to approximate your hourly means
# (Night ~7; 09–11 ~12–13; 12–15 ~16–18; 16–19 ~15–18; 21–23 ~13–15)
shift_patterns_weekend = [
    # Tier 1 Resident
    {"role": "tier_1", "shift_name": "Early",     "start": "08:00", "end": "16:15", "count": 7, "breaks": 1},
    {"role": "tier_1", "shift_name": "Middle_1",  "start": "12:00", "end": "22:00", "count": 5, "breaks": 2},
    {"role": "tier_1", "shift_name": "Twilight",  "start": "16:00", "end": "00:00", "count": 4, "breaks": 1},
    {"role": "tier_1", "shift_name": "Night",     "start": "22:00", "end": "08:00", "count": 4, "breaks": 2},

    # Tier 2 Resident
    {"role": "tier_2", "shift_name": "Early",     "start": "08:00", "end": "16:15", "count": 3, "breaks": 1},
    {"role": "tier_2", "shift_name": "Middle",    "start": "12:00", "end": "20:00", "count": 2, "breaks": 1},
    {"role": "tier_2", "shift_name": "Twilight",  "start": "16:00", "end": "00:00", "count": 2, "breaks": 1},
    {"role": "tier_2", "shift_name": "Night",     "start": "22:00", "end": "08:00", "count": 3, "breaks": 2},

    # GP & ACP
    {"role": "GP",     "shift_name": "GP",        "start": "09:00", "end": "23:00", "count": 1, "breaks": 2},
    {"role": "ACP",    "shift_name": "Early",     "start": "09:00", "end": "17:00", "count": 1, "breaks": 1},
    {"role": "ACP",    "shift_name": "Late",      "start": "15:00", "end": "22:00", "count": 1, "breaks": 1},
]


def build_global_params():
    """Return a fresh GlobalParameters object for standard runs, calibration, or sensitivity work."""
    global_params = GlobalParameters(
        ambulance_proportion=0.2,
        walk_in_proportion=0.8,

        # Source of referral
        proportion_direct_primary_care=0.01,

        # Patient characteristics
        ambulance_acuity_probabilities={
            1: 0.02,
            2: 0.40,
            3: 0.50,
            4: 0.05,
            5: 0.01,
        },
        walk_in_acuity_probabilities={
            1: 0.05,
            2: 0.05,
            3: 0.40,
            4: 0.30,
            5: 0.20,
        },

        # Staffing resource
        ambulance_triage_nurse_capacity=1,
        walk_in_triage_nurse_capacity=3,
        medical_doctor_capacity=5,
        consultant_capacity=1,
        shift_patterns_weekday=shift_patterns_weekday,
        shift_patterns_weekend=shift_patterns_weekend,

        # SDEC capacity
        sdec_open_hour=7,
        sdec_close_hour=18,
        weekday_sdec_base_capacity=6,
        weekend_sdec_base_capacity=5,

        # AMU capacity
        max_amu_available_beds=56,
        amu_queue_soft=10,
        amu_queue_hard=25,
        amu_surge_max_scale=1.10,
        max_sdec_capacity=5,

        # Service times
        mu_triage_assessment_time=1.0,
        sigma_triage_assessment_time=0.7,
        mu_ed_service_time=4.00,
        sigma_ed_service_time=0.45,
        mu_ed_decision_time=4.20,
        sigma_ed_decision_time=0.95,
        joint_gamma=0.5,

        # 240-minute breach boundary
        decision_hazard_strength_240=0.175,
        adjustment_start=240,
        adjustment_end=300,

        mu_medical_service_time=4.35,
        sigma_medical_service_time=0.55,
        mu_consultant_assessment_time=2.95,
        sigma_consultant_assessment_time=0.30,

        # Routing logic
        sdec_prob_threshold=0.10,
        paediatric_referral_rate=0.10,
        prob_referral_to_medicine_adult=0.525,
        initial_medicine_discharge_prob=0.00,
        consultant_discharge_prob=0.20,
        mu_surgical_bed_delay=5.20,
        sigma_surgical_bed_delay=1.00,

        # Routing / scenario analysis
        direct_triage_threshold=None,

        # Simulation
        simulation_time=56160,
        cool_down_time=2880,
        burn_in_time=10080,
    )

    # Scenario / sensitivity defaults
    global_params.referral_sensitivity_on = False
    global_params.referral_odds_multiplier = 1.0
    global_params.scenario_name = None

    # Base ED physical capacity
    global_params.max_ed_doctor_capacity = max(
        rota_peak(shift_patterns_weekday),
        rota_peak(shift_patterns_weekend),
    ) + 2

    return global_params


def apply_alt2_defaults(global_params):
    """
    Apply the standard alt_2 staffing redirection settings.

    Use this in both:
    - run.py when MODEL_VARIANT == "alt_2"
    - sensitivity scripts when model_name == "alt2"
    """
    base_max_ed_doctor_capacity = max(
        rota_peak(shift_patterns_weekday),
        rota_peak(shift_patterns_weekend),
    ) + 2

    global_params.alt2_medical_redirection_on = False
    global_params.alt2_redirection_n_medical_to_ed = 3
    global_params.alt2_min_medical_capacity_floor = 2
    global_params.alt2_redirection_start_hour = 9
    global_params.alt2_redirection_end_hour = 21

    # Increase ED physical cap so redirected staff can exist numerically in-hours
    global_params.max_ed_doctor_capacity = (
        base_max_ed_doctor_capacity + global_params.alt2_redirection_n_medical_to_ed
    )

    return global_params