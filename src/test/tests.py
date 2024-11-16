import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters for mean inter-arrival times (in minutes)
mean_interarrival_peak = 3.2  # Mean inter-arrival time during peak hours (09:00-21:00)
mean_interarrival_off_peak = 9.6  # Mean inter-arrival time during off-peak hours (21:00-09:00)
simulation_time = 1440  # Total simulation time in minutes (24 hours)
num_runs = 365  # Number of 24-hour periods to simulate

def extract_hour(simulation_time):
    """Extracts the hour of the day as an integer from the simulation time (in minutes)."""
    return (simulation_time // 60) % 24  # Convert minutes to 24-hour format

def get_mean_interarrival_time(hour):
    """Determine the mean inter-arrival time based on the hour of the day."""
    if 9 <= hour < 21:  # Peak hours (09:00 to 21:00)
        return mean_interarrival_peak
    else:  # Off-peak hours (21:00 to 09:00)
        return mean_interarrival_off_peak

# Generate patient arrivals over multiple 24-hour periods
np.random.seed(42)  # Set seed for reproducibility
all_arrival_times = []

for _ in range(num_runs):
    current_time = 0
    arrival_times = []

    # Simulate patient arrivals for one 24-hour period
    while current_time < simulation_time:
        current_hour = extract_hour(current_time)
        mean_interarrival_time = get_mean_interarrival_time(current_hour)

        # Generate the inter-arrival time using an exponential distribution
        inter_arrival_time = np.random.exponential(mean_interarrival_time)
    
        # Increment the current time by the generated inter-arrival time
        current_time += inter_arrival_time

        # Stop if we exceed the simulation time
        if current_time >= simulation_time:
            break

        # Store the current time as the patient arrival time
        arrival_times.append(current_time)

    # Add the arrival times from this run to the overall list
    all_arrival_times.extend(arrival_times)

# Convert to a DataFrame for plotting
df = pd.DataFrame({'Arrival Time': all_arrival_times})
df['Hour'] = df['Arrival Time'] // 60  # Convert to hour of the day

# Plot the combined distribution of arrivals over all runs
plt.figure(figsize=(10, 6))
plt.hist(df['Hour'], bins=range(25), align='left', rwidth=0.8, color='skyblue', edgecolor='black')
plt.xticks(range(24))
plt.xlabel('Hour of the Day (24-hour clock)')
plt.ylabel('Number of Patient Arrivals')
plt.title(f'Distribution of Patient Arrivals Over {num_runs} Days')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()