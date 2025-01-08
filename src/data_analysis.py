import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the path to the CSV file
csv_path = '/Users/thomasknight/Desktop/ACL/Projects/emergency_care_simulation/data/results/results.csv'

def load_data(csv_path, burn_in_time=1440):
    """
    Load and filter the simulation results CSV.
    Applies a burn-in period filter.
    
    Parameters:
        csv_path (str): Path to the CSV file.
        burn_in_time (int): Burn-in time in minutes.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    # Check if the file exists
    if not os.path.exists(csv_path):
        print(f"Error: The file at {csv_path} was not found.")
        return None
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded results from {csv_path}")
    
    # Print the column names for verification
    print("Columns in the loaded data:", df.columns.tolist())
    
    # Check if required columns exist
    required_columns = ['Run Number', 'Day of Arrival', 'Hour of Arrival', 'Simulation Arrival Time']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing required columns in the CSV file.")
        return None

    # Convert 'Simulation Arrival Time' to numeric and filter by burn-in time
    df['Simulation Arrival Time'] = pd.to_numeric(df['Simulation Arrival Time'], errors='coerce')
    df = df[df['Simulation Arrival Time'] > burn_in_time]
    
    # Drop rows with missing values in the required columns
    df.dropna(subset=required_columns, inplace=True)
    print(f"Data after filtering: {len(df)} rows")

    return df

def plot_normalized_attendances_by_hour(df):
    """
    Plot the normalized number of ED attendances by hour of the day, averaged over runs.
    
    Parameters:
        df (pd.DataFrame): Filtered DataFrame containing simulation results.
    """
    # Ensure DataFrame is not empty
    if df is None or df.empty:
        print("Error: DataFrame is empty or None.")
        return

    # Count the number of runs for normalization
    num_runs = df['Run Number'].nunique()
    print(f"Number of simulation runs: {num_runs}")

    # Group by 'Run Number', 'Day of Arrival', and 'Hour of Arrival' to count patients
    attendances_by_hour = df.groupby(['Run Number', 'Day of Arrival', 'Hour of Arrival']).size().reset_index(name='Count')

    # Aggregate across all runs and days to get the total count for each hour
    total_attendances_by_hour = attendances_by_hour.groupby('Hour of Arrival')['Count'].sum()

    # Normalize by the number of runs
    average_attendances_by_hour = total_attendances_by_hour / num_runs

    # Create a bar plot
    plt.figure(figsize=(12, 6))
    average_attendances_by_hour.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xticks(range(24), rotation=0)
    plt.xlabel('Hour of the Day (24-hour clock)')
    plt.ylabel('Average Number of Arrivals per Run')
    plt.title('Average ED Arrivals by Hour of the Day (Excluding Burn-in)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Load the data
df = load_data(csv_path)


def plot_average_wait_for_triage_by_hour(df):
    """
    Plot the average wait time for a triage nurse by hour of the day.
    
    Parameters:
        df (pd.DataFrame): Filtered DataFrame containing simulation results.
    """
    # Ensure DataFrame is not empty
    if df is None or df.empty:
        print("Error: DataFrame is empty or None.")
        return

    # Group by 'Hour of Arrival' and calculate the average wait time
    avg_wait_by_hour = df.groupby('Hour of Arrival')['Wait for Triage Nurse'].mean()

    # Create a bar plot
    plt.figure(figsize=(12, 6))
    avg_wait_by_hour.plot(kind='bar', color='salmon', edgecolor='black')
    plt.xticks(range(24), rotation=0)
    plt.xlabel('Hour of the Day (24-hour clock)')
    plt.ylabel('Average Wait Time for Triage Nurse (minutes)')
    plt.ylim(0, 80)
    plt.title('Average Wait Time for Triage Nurse by Hour of the Day')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Path to the consultant queue monitoring results CSV
csv_path = '/Users/thomasknight/Desktop/ACL/Projects/emergency_care_simulation/data/results/consultant_queue_monitoring_results.csv'

# Load the consultant queue monitoring data
df = pd.read_csv(csv_path)
print(f"Loaded consultant queue monitoring results from {csv_path}")

# Ensure the necessary columns are present
if {'Simulation Time', 'Queue Length'}.issubset(df.columns):
    # Convert 'Simulation Time' to hours of the day
    df['Hour'] = (df['Simulation Time'] // 60) % 24

    # Group by hour of the day and calculate the average queue length
    avg_queue_per_hour = df.groupby('Hour')['Queue Length'].mean()

    # Plot the average queue length
    plt.figure(figsize=(10, 6))
    plt.plot(avg_queue_per_hour.index, avg_queue_per_hour.values, marker='o', linestyle='-')
    plt.xticks(range(24))
    plt.xlabel('Hour of the Day (24-hour clock)')
    plt.ylabel('Average Consultant Queue Length')
    plt.title('Average Consultant Queue Length by Hour (Aggregated Across Runs and Days)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
else:
    print("The required columns 'Simulation Time' and 'Queue Length' are missing from the data.")

# Load the data
df = load_data(csv_path)

# Plot the results using the loaded data
if df is not None:
    plot_normalized_attendances_by_hour(df)
    plot_average_wait_for_triage_by_hour(df)



def plot_average_consultant_queue_by_hour_and_day(df, burn_in_time):
    """
    Filters the consultant queue monitoring results based on burn-in time,
    aggregates queue length by hour and day, and plots the average queue length.

    Parameters:
        df (pd.DataFrame): DataFrame containing consultant queue monitoring results.
        burn_in_time (float): Time (in minutes) to exclude the burn-in period.
    """
    # Ensure necessary columns are numeric
    df['Simulation Time'] = pd.to_numeric(df['Simulation Time'], errors='coerce')
    df['Queue Length'] = pd.to_numeric(df['Queue Length'], errors='coerce')
    
    # Filter out rows within the burn-in time
    df = df[df['Simulation Time'] > burn_in_time]

    # Calculate the hour and day of the simulation
    df['Hour of Day'] = (df['Simulation Time'] // 60) % 24
    df['Day of Simulation'] = (df['Simulation Time'] // 1440).astype(int)  # 1440 minutes in a day

    # Group by 'Hour of Day' and 'Day of Simulation' and calculate average queue length
    hourly_summary = df.groupby(['Day of Simulation', 'Hour of Day'])['Queue Length'].mean().reset_index()

    # Create a pivot table for plotting (days as rows, hours as columns)
    pivot_table = hourly_summary.pivot(index='Day of Simulation', columns='Hour of Day', values='Queue Length')

    # Plot the heatmap-like bar chart
    plt.figure(figsize=(12, 8))
    plt.imshow(pivot_table, aspect='auto', cmap='coolwarm', origin='lower')
    plt.colorbar(label='Average Queue Length')
    plt.xticks(range(24), labels=[f"{hour}:00" for hour in range(24)], rotation=45)
    plt.yticks(range(len(pivot_table.index)), labels=[f"Day {day}" for day in pivot_table.index])
    plt.xlabel('Hour of the Day (24-hour clock)')
    plt.ylabel('Day of Simulation')
    plt.title('Average Consultant Queue Length by Hour and Day (Post Burn-In)')
    plt.tight_layout()
    plt.show()

# Example Usage
if __name__ == "__main__":
    # Load the consultant queue monitoring data
    csv_path = '/path_to/consultant_queue_monitoring_results.csv'
    df = pd.read_csv(csv_path)
    print(f"Loaded results from {csv_path}")

    # Define the burn-in period (e.g., 1440 minutes = 1 day)
    burn_in_time = 1440

    # Call the function to filter and plot the data
    plot_average_consultant_queue_by_hour_and_day(df, burn_in_time)



def plot_avg_time_to_consultant_from_csv(csv_path, burn_in_time):
    """
    Plot the average time from arrival to consultant assessment by hour of the day
    using the main results CSV file.
    
    Parameters:
    - csv_path: The path to the main results CSV file.
    - burn_in_time: The burn-in period in simulation time to filter out initial results.
    """
    # Load the CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded results from {csv_path}")

    # Filter out rows based on burn-in time
    df = df[pd.to_numeric(df['Arrival Time'], errors='coerce') > burn_in_time]

    # Calculate the hour of arrival
    df['Hour of Arrival'] = (df['Arrival Time'] // 60) % 24  # Convert simulation time to hour of the day

    # Calculate time from arrival to consultant assessment
    df['Time to Consultant'] = df['Arrival to Consultant Assessment']  # Ensure the column name matches

    # Group by hour of arrival and calculate the average time to consultant assessment
    avg_time_by_hour = df.groupby('Hour of Arrival')['Time to Consultant'].mean()

    # Plot the average time to consultant assessment
    plt.figure(figsize=(10, 6))
    avg_time_by_hour.plot(kind='bar', color='skyblue', edgecolor='black', alpha=0.8)
    plt.xlabel('Hour of Arrival (24-hour clock)')
    plt.ylabel('Average Time to Consultant Assessment (minutes)')
    plt.title('Average Time from Arrival to Consultant Assessment by Hour')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Example Usage
csv_path = '/Users/thomasknight/Desktop/ACL/Projects/emergency_care_simulation/data/results/results.csv'
# Replace with your actual CSV file path
burn_in_time = 1440  # Define your burn-in period
plot_avg_time_to_consultant_from_csv(csv_path, burn_in_time)