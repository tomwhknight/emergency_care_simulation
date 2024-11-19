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

def plot_average_time_to_consultant_assessment(df):
    """
    Plot the average time from patient arrival to consultant assessment by hour of arrival,
    excluding patients who were discharged before seeing a consultant.
    
    Parameters:
        df (pd.DataFrame): Filtered DataFrame containing simulation results.
    """
    # Ensure DataFrame is not empty
    if df is None or df.empty:
        print("Error: DataFrame is empty or None.")
        return

    # Check if the required column exists
    if 'Arrival to Consultant Assessment' not in df.columns:
        print("Error: 'Arrival to Consultant Assessment' column not found in the data.")
        return

    # Convert 'Arrival to Consultant Assessment' to numeric, forcing errors to NaN
    df['Arrival to Consultant Assessment'] = pd.to_numeric(df['Arrival to Consultant Assessment'], errors='coerce')

    # Debugging: Check the distribution of the column before filtering
    print("Distribution of 'Arrival to Consultant Assessment' before filtering:")
    print(df['Arrival to Consultant Assessment'].describe())
    
    # Remove rows with NaN or non-positive values
    filtered_df = df[(df['Arrival to Consultant Assessment'] > 0) & df['Arrival to Consultant Assessment'].notnull()]

    # Debugging: Check how many rows remain after filtering
    print(f"Rows after filtering: {len(filtered_df)}")
    print("Sample rows after filtering:")
    print(filtered_df[['Hour of Arrival', 'Arrival to Consultant Assessment']].head())

    # Check if there are any rows left after filtering
    if filtered_df.empty:
        print("No valid data available for plotting after filtering.")
        return

    # Group by 'Hour of Arrival' and calculate the average time to consultant assessment
    avg_time_to_consultant = filtered_df.groupby('Hour of Arrival')['Arrival to Consultant Assessment'].mean()

    # Check if the resulting Series is empty
# Load the data
df = load_data(csv_path)

# Plot the results using the loaded data
if df is not None:
    plot_normalized_attendances_by_hour(df)
    plot_average_wait_for_triage_by_hour(df)
    plot_average_time_to_consultant_assessment(df)

