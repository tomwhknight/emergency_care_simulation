# helper.py

import pandas as pd
from collections import defaultdict
import os
import datetime
import json
import numpy as np
import math

def calculate_hour_of_day(simulation_time):
    """Converts the simulation time (in minutes) to a readable time of day (HH:MM)."""
    total_minutes = int(simulation_time % 1440)  # Get remainder of time in current day
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02}:{minutes:02}"  # Format time as HH:MM

def extract_day_of_week(simulation_time):
    """Calculates the day of the week based on simulation time (assuming day 0 is Monday)."""
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_of_week = int(simulation_time // 1440) % 7  # Divide by 1440 (minutes in a day)
    return days[day_of_week]

def extract_hour(simulation_time):
    """Extracts the hour of the day as an integer from the simulation time (in minutes)."""
    total_minutes = int(simulation_time % 1440)  # Get remainder of time in the current day
    return total_minutes // 60  # Return only the hour as an integer

class Lognormal:
    """
    Encapsulates a lognormal distirbution
    """
    def __init__(self, mean, stdev, random_seed=None):
        """
        Params:
        -------
        mean = mean of the lognormal distribution
        stdev = standard dev of the lognormal distribution
        """
        self.rand = np.random.default_rng(seed=random_seed)
        mu, sigma = self.normal_moments_from_lognormal(mean, stdev**2)
        self.mu = mu
        self.sigma = sigma

    def normal_moments_from_lognormal(self, m, v):
        '''
        Returns mu and sigma of normal distribution
        underlying a lognormal with mean m and variance v
        source: https://blogs.sas.com/content/iml/2014/06/04/simulate-lognormal
        -data-with-specified-mean-and-variance.html

        Params:
        -------
        m = mean of lognormal distribution
        v = variance of lognormal distribution

        Returns:
        -------
        (float, float)
        '''
        phi = math.sqrt(v + m**2)
        mu = math.log(m**2/phi)
        sigma = math.sqrt(math.log(phi**2/m**2))
        return mu, sigma

    def sample(self):
        """
        Sample from the normal distribution
        """
        return self.rand.lognormal(self.mu, self.sigma)


def exp_rv(rate: float, rng) -> float:
    """
    Exponential(rate) using a specific RNG stream.
    Returns np.inf if rate <= 0 so callers can yield timeout(np.inf) safely.
    """
    if rate <= 0:
        return float('inf')
    return -np.log1p(-rng.random()) / rate

def wchoice(items, weights, rng):
    """
    Weighted categorical choice using a specific RNG stream.
    items: sequence of labels
    weights: non-negative weights (need not be normalized)
    """
    items = np.asarray(list(items))
    p = np.asarray(list(weights), dtype=float)
    s = p.sum()
    if len(items) == 0:
        raise ValueError("wchoice called with empty items.")
    if s <= 0:
        # fallback: uniform over items
        idx = rng.integers(0, len(items))
        return items[idx]
    p = p / s
    idx = rng.choice(len(items), p=p)
    return items[idx]

def bern(p: float, rng) -> bool:
    """
    Bernoulli(p) using a specific RNG stream. Clamps p to [0,1].
    """
    if p <= 0:
        return False
    if p >= 1:
        return True
    return rng.random() < p





    # function to track resouce utilisation

    def audit_utilisation(self, activity_attribute, resource_attribute):
        activity_durations = [getattr(i, activity_attribute) for i in self.patient_objects]
        return sum(activity_durations) / (g.getattr(resource_attribute) * g.sim_duration)


def weighted_quantile(x, q, w=None):
    """
    Weighted quantile of x at quantile q in [0,1].
    If w is None, falls back to np.quantile(x, q).
    Notes:
      - Drops non-finite x/w.
      - Uses right-closed search so ties at the cutoff land in the upper group.
    """
    x = np.asarray(x, dtype=float)
    if w is None:
        return float(np.quantile(x, q))

    w = np.asarray(w, dtype=float)

    # drop non-finite
    m = np.isfinite(x) & np.isfinite(w)
    x = x[m]; w = w[m]

    # guard: empty after filtering
    if x.size == 0:
        return float("nan")

    order = np.argsort(x)
    x_ord = x[order]
    w_ord = w[order]

    cw = np.cumsum(w_ord)                  # cumulative weight
    cut = q * cw[-1]                       # target cumulative weight
    i = int(np.searchsorted(cw, cut, side="right"))
    i = min(max(i, 0), x_ord.size - 1)     # clamp into bounds
    return float(x_ord[i])


def dt_threshold_from_top_percent(p_raw, p_cal, top_percent):
    """
    Convert a single knob (top_percent, e.g. 10/25/50/75) into a p_raw cutoff.
    Uses weighted quantile with weights = p_cal so “top X%” is with respect to
    expected Medicine intent.

    Returns: (threshold_value, policy_label)
    """
    try:
        pct = float(top_percent)
    except Exception:
        pct = 0.0

    pct = max(0.0, min(100.0, pct))  # clamp

    if pct <= 0.0:
        return (float("nan"), "clinical_only")

    q = 1.0 - (pct / 100.0)  # top X% => quantile at 1 - X/100
    th = weighted_quantile(p_raw, q, w=p_cal)
    return (th, f"top{int(round(pct))}_raw_ge_{th:.3f}")

    # Rota math

    # --- Rota capacity helpers ---
    # --- Rota capacity helpers (copy/paste this whole block) ---

def time_to_minutes(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m

def _mins_to_end_at(mins_in_day: int, start_str: str, end_str: str):
    """
    Minutes-to-end for a shift at time mins_in_day.
    Returns None if shift is not active at that minute. Handles overnight.
    """
    day = 1440
    start = time_to_minutes(start_str)
    end   = time_to_minutes(end_str)
    # active?
    if start < end:
        if not (start <= mins_in_day < end):
            return None
        return end - mins_in_day
    else:
        # overnight (e.g., 22:00 -> 08:00)
        if not (mins_in_day >= start or mins_in_day < end):
            return None
        if mins_in_day < end:        # after midnight before end
            return end - mins_in_day
        else:                        # before midnight after start
            return day - mins_in_day + end

def rota_peak(shift_patterns, roles=None, end_cutoff=30) -> int:
    """
    Peak simultaneous headcount across the day **excluding shifts that are within
    the last `end_cutoff` minutes of finishing** at a given minute.
    Starters ARE included (no start-block here).
    - roles: optional set of roles to include (e.g., {"tier_1","tier_2"}).
    """
    if roles is not None:
        shift_patterns = [s for s in shift_patterns if s.get("role") in roles]

    peak = 0
    for m in range(1440):  # each minute of the day
        total = 0
        for sh in shift_patterns:
            mte = _mins_to_end_at(m, sh["start"], sh["end"])
            if mte is None:
                continue        # not active at this minute
            if mte <= end_cutoff:
                continue        # finishing soon → exclude from peak
            total += int(sh.get("count", 0))
        if total > peak:
            peak = total
    return peak

def staff_by_hour(shift_patterns, end_cutoff=30):
    """
    Counts staff at the **start of each hour** (H:00), excluding shifts that are
    within the last `end_cutoff` minutes of finishing at that minute.
    Returns (hours, totals, by_role) as before.
    """
    hours = list(range(24))
    totals = []
    by_role = defaultdict(list)
    roles = sorted(set(s["role"] for s in shift_patterns))

    for H in hours:
        t = H * 60  # start of the hour
        total_here = 0
        counts_here = {role: 0 for role in roles}
        for s in shift_patterns:
            mte = _mins_to_end_at(t, s["start"], s["end"])
            if mte is None:
                continue
            if mte <= end_cutoff:
                continue
            cnt = int(s.get("count", 0))
            total_here += cnt
            counts_here[s["role"]] += cnt
        totals.append(total_here)
        for role in roles:
            by_role[role].append(counts_here[role])

    return hours, totals, by_role

def rota_to_dataframe(shift_patterns, end_cutoff=30):
    hours, totals, by_role = staff_by_hour(shift_patterns, end_cutoff=end_cutoff)
    df = pd.DataFrame({"Hour": hours, "Total": totals})
    for role, counts in by_role.items():
        df[role] = counts
    return df

def save_rota_check(patterns, out_dir, filename="rota_check.csv",
                    start_block=0, end_cutoff=35, resolution_minutes=1):
    """
    Build and save a simple 24h availability table from `patterns` applying the same
    handover rules your sim uses. Does NOT rely on rota_peak(return_df=...).
    """
    os.makedirs(out_dir, exist_ok=True)

    def to_min(t):
        h, m = map(int, t.split(":"))
        return 60*h + m

    day = 24 * 60
    minutes = list(range(0, day, resolution_minutes))
    rows = []

    for now in minutes:
        available = 0
        for s in patterns:
            start = to_min(s["start"])
            end   = to_min(s["end"])
            count = int(s.get("count", 0))

            if start < end:
                active = (start <= now < end)
                if not active:
                    continue
                mins_since = now - start
                mins_to_end = end - now
            else:
                active = (now >= start) or (now < end)
                if not active:
                    continue
                mins_since = (now - start) if now >= start else (day - start + now)
                mins_to_end = (end - now) if now < end else (day - now + end)

            if mins_since >= start_block and mins_to_end > end_cutoff:
                available += count

        rows.append({"minute_of_day": now, "hour": now // 60, "effective_capacity": available})

    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    return path


def _to_jsonable(x):
    # numpy scalars/arrays
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    # sets
    if isinstance(x, set):
        return list(x)
    # datetime types
    if isinstance(x, (datetime.datetime, datetime.date, datetime.time)):
        return x.isoformat()
    return x

def _obj_to_dict(obj):
    if isinstance(obj, dict):
        return {k: _obj_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, np.ndarray)):
        return [_obj_to_dict(v) for v in list(obj)]
    # Generic object with attributes (e.g., GlobalParameters)
    try:
        return {
            k: _obj_to_dict(v)
            for k, v in obj.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }
    except AttributeError:
        return _to_jsonable(obj)

def save_params(params_obj, folder, filename="params.json", extra=None):
    """
    Save parameters (and optional metadata) as JSON in `folder/filename`.
    Returns the full path.
    """
    data = _obj_to_dict(params_obj)
    if extra:
        data.update(extra)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path
