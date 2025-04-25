import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import streamlit as st
from scipy.stats import mannwhitneyu

# --- Baseline file locations ---
BASELINE_DIR = "data/baseline"
BASELINE_SUMMARY_FILE = os.path.join(BASELINE_DIR, "baseline_overall_summary.csv")
BASELINE_PATIENT_FILE = os.path.join(BASELINE_DIR, "baseline_results.csv")

# Load data (lowercase vars)
baseline_summary = pd.read_csv(BASELINE_SUMMARY_FILE)
baseline_patient = pd.read_csv(BASELINE_PATIENT_FILE)

st.set_page_config(layout="wide")

# Two logos: UoM (left) and MFT (right)
col1, col2, col3 = st.columns([1.25, 2, 1.25])

with col1:
    st.image("/Users/thomasknight/Desktop/ACL/Side projects/sepsis/uom.jpeg")

with col3:
    st.image("/Users/thomasknight/Desktop/ACL/Side projects/sepsis/mft.png")

st.markdown(
    """
    <h1 style='text-align: center; font-size: 2.5em;'>
        Simulation Results
    </h1>
    """,
    unsafe_allow_html=True
)

# --- 2. Tabs for Further Exploration ---

tab_summary, tab_scenario_comparison, tab_run_variation, tab_time_trends, tab_resources, tab_patient_journey = st.tabs([
    "Summary",
    "Scenario Comparison",
    "Run-Level Variation",
    "Time-Based Analysis",
    "Resource Utilisation",
    "Patient Journey Explorer"
])

with tab_summary:

# Start by copying the summary

    if "trial" in st.session_state and st.session_state.trial is not None:
        trial = st.session_state.trial
        current_summary = trial.overall_summary.copy()
        
        # --- Helper functions ---
        def get_mean(summary_df, metric_name):
            """Fetch the mean value from a summary DataFrame given a metric name."""
            row = summary_df[summary_df["measure"] == metric_name]
            return row["mean_overall"].values[0] if not row.empty else None

        def format_delta(current, baseline, unit=""):
            """Format the delta with sign and unit."""
            if current is None or baseline is None:
                return None
            delta = current - baseline
            return f"{delta:+.1f}{unit}"
        

        st.header("Summary")


        st.markdown(
            "This page summarises the outputs from your simulation run, based on the input parameters you selected.\n\n"
            "Key metrics are shown alongside comparisons to a predefined baseline model "
            "*(see the Technical Details tab for how the baseline was defined and validated).*" "\n\n"
            "Use the tabs above to explore more detailed analysis of the simulation results"
        )
        st.markdown("### Demand")

        # --- Display metrics in 3 columns ---
        col1, col2, col3 = st.columns(3)

        with col1:
            current = get_mean(current_summary, "Mean ED Attendances per Day")
            baseline = get_mean(baseline_summary, "Mean ED Attendances per Day")
            value = f"{current:.1f}" if current is not None else "—"
            delta = format_delta(current, baseline, " per day")
            st.metric("Mean Attendances per Day", value, delta)
           
        with col2:
            current = get_mean(current_summary, "Proportion Referred - Medicine")
            baseline = get_mean(baseline_summary, "Proportion Referred - Medicine")
            value = f"{current:.1f} %" if current is not None else "—"
            delta = format_delta(current, baseline, " %")
            st.metric("Referred to Medicine", value, delta)

        with col3:
            current = get_mean(current_summary, "SDEC Accepted (of Appropriate)")
            baseline = get_mean(baseline_summary, "SDEC Accepted (of Appropriate)")
            value = f"{current:.1f} %" if current is not None else "—"
            delta = format_delta(current, baseline, " %")
            st.metric("Accepted to SDEC (of Appropriate)", value, delta)

        if st.session_state.trial is not None:
            trial = st.session_state.trial
            patient_level_results = trial.agg_results_df.copy()

        # Check that 'sim_time_arrival' exists
        if "Hour of Arrival" in patient_level_results.columns:
        
            # Group by hour to count number of arrivals
            arrival_counts = (
                patient_level_results.groupby(["Run Number", "Hour of Arrival"])
                .size()
                .reset_index(name="arrivals")
            )

            # Plot
            fig = px.line(
                arrival_counts,
                x="Hour of Arrival",
                y="arrivals",
                color="Run Number",
                title="Arrival Pattern by Run",
                labels={
                    "Hour of Arrival": "Hour of Arrival",
                    "arrivals": "Number of Arrivals per Hour"
                }
            )
            fig.update_layout(
                xaxis=dict(dtick=2),  # one tick per 12 hours
                yaxis_title="ED Attendance",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Arrival time column 'sim_time_arrival' not found in results.")

        st.markdown("### Performance")

        col4, col5, col6 = st.columns(3)

        with col4:
            current = get_mean(current_summary, "Time in System")
            baseline = get_mean(baseline_summary, "Time in System")
            value = f"{current:.1f} mins" if current is not None else "—"
            delta = format_delta(current, baseline, " mins")
            st.metric("Time in ED", value, delta)

        with col5:
            current = get_mean(current_summary, ">4hr breach")
            baseline = get_mean(baseline_summary, ">4hr breach")
            value = f"{current:.1f} %" if current is not None else "—"
            delta = format_delta(current, baseline, " %")
            st.metric(">4hr Breach", value, delta)

        with col6:
            current = get_mean(current_summary, ">12hr breach")
            baseline = get_mean(baseline_summary, ">12hr breach")
            value = f"{current:.1f} %" if current is not None else "—"
            delta = format_delta(current, baseline, " %")
            st.metric(">12hr Breach", value, delta)


    trial = st.session_state.get("trial", None)

    if trial is not None:
        run_results = trial.agg_results_df.copy()

        if "Time in System" in run_results.columns and "Run Number" in run_results.columns:
            
            run_results["Run Number"] = run_results["Run Number"].astype(str)
            
            fig = px.box(
                run_results,
                x="Run Number",
                y="Time in System",
                color="Run Number",
                points = "all",  
                title="Total Time in ED by Run",
                labels={
                    "Time in ED": "Time in ED (hrs)",
                    "Run Number": "Run"
                }
            )

            # Add 4-hour breach line
            fig.add_hline(
                y=240,
                line_dash="dash",
                line_color="black",
            )

            fig.update_layout(
                showlegend=False,
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                "<div style='text-align: center; font-size: 0.85rem; margin-top: -25px;'>"
                "Dashed line = 4-hour breach threshold (240 mins)"
                "</div>",
                unsafe_allow_html=True
            )

        else:
            st.info("Required columns not found in run results.")


with tab_scenario_comparison:
    st.header("Scenario Comparison")

    st.markdown(
    "This tab compares aggregated results from your current scenario against the predefined baseline model.\n\n"
    "Metrics are averaged across all simulation runs to account for stochastic variation.\n\n"
    "Use the dropdown to explore different outcome measures. "
    "*Baseline definitions are provided in the Technical Details tab.*"
    )

    st.markdown("### Performance")

    # Dropdown for metric selection
    available_metrics = [
        "Time in System",
        ">4hr breach",
        ">12hr breach",
        ]

    selected_metric = st.selectbox("Select a metric to compare:", available_metrics)
    
    
    # Prepare current data
    trial = st.session_state.get("trial", None)
    if trial is not None:
        scenario_df = trial.agg_results_df.copy()
        scenario_df["Scenario"] = "Current"

        # Load baseline mean
        baseline_df = baseline_patient.copy()
        baseline_df["Scenario"] = "Baseline"

        # Combine both datasets
        combined_df = pd.concat([scenario_df, baseline_df], ignore_index=True)
        combined_df = combined_df[[selected_metric, "Scenario"]].dropna()

        # Check if metric is binary (True/False or 0/1)
        unique_vals = combined_df[selected_metric].dropna().unique()
        is_binary = set(unique_vals).issubset({0, 1, True, False})

        if is_binary:
            combined_df[selected_metric] = combined_df[selected_metric].astype(int)
            proportions = combined_df.groupby("Scenario")[selected_metric].mean().reset_index()
            proportions[selected_metric] *= 100

            # Calculate group size (n) per scenario
            group_sizes = combined_df.groupby("Scenario")[selected_metric].count().reset_index()
            group_sizes.columns = ["Scenario", "N"]

            # Calculate proportion
            proportions = combined_df.groupby("Scenario")[selected_metric].mean().reset_index()
            proportions[selected_metric] *= 100  # Convert to percentage

            # Merge group sizes into proportions
            summary = pd.merge(proportions, group_sizes, on="Scenario")

            # Calculate standard error (SE) of the proportion
            p = summary[selected_metric].to_numpy() / 100
            n = summary["N"].to_numpy()
            summary["SE"] = 100 * np.sqrt((p * (1 - p)) / n)
            # Only keep rows with valid values


            fig = px.bar(
                summary,
                x="Scenario",
                y=selected_metric,
                error_y="SE",
                color="Scenario",
                title=f"Proportion of Patients with '{selected_metric}'",
                labels={selected_metric: "Percentage (%)"}
            )
          
            # Calculate max y value for dynamic scaling
            max_y = summary[selected_metric].max()  # Get max value for Y axis
            fixed_gap = max_y * 0.15  # Adjust the percentage to control the gap size

            for i, row in summary.iterrows():
                scenario = row["Scenario"]
                y = row[selected_metric] + row["SE"] + fixed_gap  # Fixed gap
                label = f"{row[selected_metric]:.1f}%"

                fig.add_trace(go.Scatter(
                    x=[scenario],
                    y=[y],
                    text=[label],
                    mode="text",
                    textfont=dict(size=24, color="black"),
                    showlegend=False
            ))

            fig.update_traces(
                error_y=dict(
                    thickness=2,  
                    width=32 
            ))

            fig.update_layout(
                title_x=0.01,
                height=400,
                margin=dict(t=50, b=40),
                uniformtext_minsize=12,
                uniformtext_mode='hide',
            )

            st.plotly_chart(fig, use_container_width=True)

        # Plot
        else:
            fig = px.box(
                combined_df,
                x="Scenario",
                y=selected_metric,
                color="Scenario",
                points="all",  # shows individual patient points
                title=f"{selected_metric} – Distribution by Scenario",
                labels={selected_metric: selected_metric}
            )
            fig.update_layout(title_x=0.01, height=450)
            
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("No simulation results available. Please run a scenario.")

with tab_time_trends:
    st.header("Time-Based Analysis")

    st.markdown(
    "This tab shows how performance and care processes vary by hour of patient arrival.\n\n"
    "Metrics are aggregated across all simulation runs to show mean trends with error bars representing statistical uncertainty.\n\n"
    "Use the dropdown to explore different process timings and outcomes across the 24-hour clock."
    )

    st.markdown("### Care processes")

    # Dropdown for metric selection
    arrival_metrics = {
        "Arrival → Triage Nurse":       "Arrival to Triage Nurse Assessment",
        "Arrival → ED Assessment":      "Arrival to ED Assessment",
        "Arrival → Referral":           "Arrival to Referral",
        "Arrival → Medical Assessment": "Arrival to Medical Assessment",
        "Arrival → Consultant Assessment": "Arrival to Consultant Assessment",
    }
    choice = st.selectbox("Choose a process interval:", list(arrival_metrics.keys()))
    col = arrival_metrics[choice]

    trial = st.session_state.get("trial", None)
    if not trial:
        st.info("Please run a simulation first.")
    else:
        # 2. Current data
        current_df = trial.agg_results_df.copy()
        if "Hour of Arrival" not in current_df.columns or col not in current_df.columns:
            st.error(f"No data available for {choice}")
        else:
            current_df = current_df[["Hour of Arrival", col]].dropna()
            current_df["Scenario"] = "Current"

            # 3. Baseline data
            try:
                baseline_df = baseline_patient[["Hour of Arrival", col]].dropna()
                baseline_df["Scenario"] = "Baseline"
            except Exception:
                st.warning("Baseline data missing or column not found.")
                baseline_df = pd.DataFrame(columns=["Hour of Arrival", col, "Scenario"])

            # 4. Combine and aggregate
            combined = pd.concat([current_df, baseline_df], ignore_index=True)

            grp = (
                combined
                .groupby(["Scenario", "Hour of Arrival"])[col]
                .agg(mean="mean", std="std", n="count")
                .reset_index()
            )
            grp["se"] = grp["std"] / np.sqrt(grp["n"])

            # 5. Plot
            fig = px.line(
                grp,
                x="Hour of Arrival",
                y="mean",
                color="Scenario",
                error_y="se",
                markers=True,
                title=f"{choice} by Hour of Arrival",
                labels={
                    "Hour of Arrival": "Hour of Arrival",
                    "mean": f"{choice} (mins)"
                }
            )
            fig.update_layout(
                title_x=0.01,
                height=450,
                margin=dict(t=50, b=40),
                xaxis=dict(dtick=1)
            )
            st.plotly_chart(fig, use_container_width=True)

            # 6. Optional: Show table
            if st.checkbox("Show hour-by-hour stats"):
                st.dataframe(
                    grp.rename(columns={
                        "Hour of Arrival": "Hour",
                        "mean": "Mean (mins)",
                        "se": "Std. Error",
                        "n": "Sample Size"
                    }),
                    use_container_width=True
                )
   
    st.markdown("### Queue")

    # Dropdown for metric selection
    queue_metrics = {
        "Triage Nurse": "Queue Length Walk in Triage Nurse",
        "ED Clinician": "Queue Length ED doctor",
        "Medical Clinician": "Queue Length Medical Doctor",
        "Consultant": "Queue Length Consultant",
    }
    queue_choice = st.selectbox("Choose a process interval:", list(queue_metrics.keys()))
    queue_col = queue_metrics[queue_choice]

    trial = st.session_state.get("trial", None)
    if not trial:
        st.info("Run a simulation first to see time-based trends.")
    else:
        current_data = trial.agg_results_df.copy()
        if "Hour of Arrival" not in current_data.columns or queue_col not in current_data.columns:
            st.error(f"Missing required columns: Hour of Arrival or {queue_col}")
        else:
            current_data = current_data[["Hour of Arrival", queue_col]].dropna()
            current_data["Scenario"] = "Current"

            # Load baseline patient-level results
            try:
                baseline_df = baseline_patient[["Hour of Arrival", queue_col]].dropna()
                baseline_df["Scenario"] = "Baseline"
            except Exception:
                st.warning("Baseline data could not be loaded or is missing the required column.")
                baseline_df = pd.DataFrame(columns=["Hour of Arrival", queue_col, "Scenario"])

            # Combine current and baseline
            combined = pd.concat([current_data, baseline_df], ignore_index=True)

            # Aggregate by hour and scenario
            grp = (
                combined
                .groupby(["Scenario", "Hour of Arrival"])[queue_col]
                .agg(mean="mean", std="std", n="count")
                .reset_index()
            )
            grp["se"] = grp["std"] / np.sqrt(grp["n"])

            # Plot
            fig = px.line(
                grp,
                x="Hour of Arrival",
                y="mean",
                color="Scenario",
                error_y="se",
                markers=True,
                title=f"{queue_choice} Queue Length by Hour of Arrival",
                labels={
                    "Hour of Arrival": "Hour of Arrival",
                    "mean": "Mean Queue Length"
                }
            )
            fig.update_layout(
                title_x=0.01,
                height=450,
                margin=dict(t=50, b=40),
                xaxis=dict(dtick=1),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Optional data table
            if st.checkbox("Show hourly queue data by scenario"):
                st.dataframe(
                    grp.rename(columns={
                        "Hour of Arrival": "Hour",
                        "mean": "Mean Queue Length",
                        "se": "Std. Error",
                        "n": "Sample Size"
                    }),
                    use_container_width=True
                )