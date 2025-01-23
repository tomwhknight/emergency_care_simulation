# helper.py
import datetime
import numpy as np
import math

def calculate_hour_of_day(simulation_time):
    """Converts the simulation time (in minutes) to a readable time of day (HH:MM)."""
    total_minutes = int(simulation_time % 1440)  # Get remainder of time in current day
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02}:{minutes:02}"  # Format time as HH:MM

def calculate_day_of_week(simulation_time):
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