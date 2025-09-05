# helper.py

import pandas as pd
from collections import defaultdict
import os
import datetime
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
    

    # function to track resouce utilisation

    def audit_utilisation(self, activity_attribute, resource_attribute):
        activity_durations = [getattr(i, activity_attribute) for i in self.patient_objects]
        return sum(activity_durations) / (g.getattr(resource_attribute) * g.sim_duration)
    

    # Rota math

    # --- Rota capacity helpers ---

def _hhmm_to_min(s: str) -> int:
    hh, mm = map(int, s.split(":"))
    return hh * 60 + mm

def _is_active(start_min: int, end_min: int, m: int) -> bool:
    """Supports overnight shifts (e.g., 22:00 → 08:00)."""
    if start_min < end_min:
        return start_min <= m < end_min
    else:
        return m >= start_min or m < end_min

def rota_peak(shift_patterns, roles=None) -> int:
    """
    Return peak simultaneous headcount across the day.
    - shift_patterns: list of dicts with keys: start, end, count, role
    - roles: optional set of roles to include (e.g., {"tier_1","tier_2"})
    """
    # optionally filter roles
    if roles is not None:
        shift_patterns = [s for s in shift_patterns if s.get("role") in roles]

    peak = 0
    for m in range(1440):  # every minute of the day
        total = 0
        for sh in shift_patterns:
            start = _hhmm_to_min(sh["start"])
            end   = _hhmm_to_min(sh["end"])
            cnt   = int(sh.get("count", 0))
            if _is_active(start, end, m):
                total += cnt
        peak = max(peak, total)
    return peak


def time_to_minutes(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m

def on_shift_at(mins_in_day: int, start_str: str, end_str: str) -> bool:
    start = time_to_minutes(start_str)
    end = time_to_minutes(end_str)
    if start < end:
        return start <= mins_in_day < end
    else:
        # Overnight shift (e.g. 22:00 → 07:30)
        return mins_in_day >= start or mins_in_day < end

def staff_by_hour(shift_patterns):
    hours = list(range(24))
    totals = []
    by_role = defaultdict(list)
    roles = sorted(set(s["role"] for s in shift_patterns))

    for H in hours:
        t = H * 60
        total_here = 0
        counts_here = {role: 0 for role in roles}
        for s in shift_patterns:
            if on_shift_at(t, s["start"], s["end"]):
                total_here += s["count"]
                counts_here[s["role"]] += s["count"]
        totals.append(total_here)
        for role in roles:
            by_role[role].append(counts_here[role])

    return hours, totals, by_role

def rota_to_dataframe(shift_patterns):
    hours, totals, by_role = staff_by_hour(shift_patterns)
    df = pd.DataFrame({"Hour": hours, "Total": totals})
    for role, counts in by_role.items():
        df[role] = counts
    return df

def save_rota_check(shift_patterns, output_dir, filename="rota_check.csv"):
    """Save rota sanity check table to a CSV in the given output directory."""
    df = rota_to_dataframe(shift_patterns)
    os.makedirs(output_dir, exist_ok=True)  # make sure dir exists
    filepath = os.path.join(output_dir, filename)
    df.to_csv(filepath, index=False)
    return filepath
