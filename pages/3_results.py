import streamlit as st
import plotly.express as px

st.set_page_config(layout="wide")


# =====================================================
# Check results exist
# =====================================================

if "all_patient_results" not in st.session_state or st.session_state.all_patient_results is None:
    st.warning("No simulation results found. Please run the simulation first.")
    st.stop()

if "all_summary_results" not in st.session_state or st.session_state.all_summary_results is None:
    st.warning("No summary results found. Please run the simulation first.")
    st.stop()

patient_df = st.session_state.all_patient_results.copy()
summary_df = st.session_state.all_summary_results.copy()


# =====================================================
# Scenario names and colours
# =====================================================

scenario_name_map = {
    "Baseline": "Baseline",
    "Direct triage": "Direct to Medicine",
    "Direct to medicine": "Direct to Medicine",
    "Direct to Medicine": "Direct to Medicine",
    "Consultant streaming": "Direct to Consultant",
    "Direct consultant": "Direct to Consultant",
    "Direct to consultant": "Direct to Consultant",
    "Direct to Consultant": "Direct to Consultant",
}

scenario_order = [
    "Baseline",
    "Direct to Medicine",
    "Direct to Consultant",
]

scenario_colour_map = {
    "Baseline": "blue",
    "Direct to Medicine": "orange",
    "Direct to Consultant": "green",
}

if "Scenario" in patient_df.columns:
    patient_df["Scenario"] = patient_df["Scenario"].replace(scenario_name_map)

if "Scenario" in summary_df.columns:
    summary_df["Scenario"] = summary_df["Scenario"].replace(scenario_name_map)


# =====================================================
# Helper functions
# =====================================================

def first_existing_column(df, possible_columns):
    """Return the first column name that exists in df."""
    for column in possible_columns:
        if column in df.columns:
            return column
    return None


def build_run_level_means(patient_df, outcome_columns, run_column):
    """
    Calculate one mean value per scenario per run for each outcome.

    This prevents larger runs from dominating the headline comparison plots.
    """
    available_outcomes = [column for column in outcome_columns if column in patient_df.columns]

    if not available_outcomes or run_column is None:
        return None

    run_level_means = (
        patient_df
        .groupby(["Scenario", run_column], as_index=False)[available_outcomes]
        .mean()
    )

    return run_level_means


def build_average_across_runs(run_level_means, outcome_columns):
    """
    Average the run-level means across runs and calculate SD across runs.
    """
    available_outcomes = [column for column in outcome_columns if column in run_level_means.columns]

    average_across_runs = (
        run_level_means
        .groupby("Scenario", as_index=False)[available_outcomes]
        .agg(["mean", "std"])
    )

    average_across_runs.columns = [
        "Scenario" if column[0] == "Scenario" else f"{column[0]}_{column[1]}"
        for column in average_across_runs.columns
    ]

    return average_across_runs


def plot_mean_across_runs(average_across_runs, outcome_column, title):
    """Plot scenario-level mean of run means with SD error bars."""
    mean_column = f"{outcome_column}_mean"
    sd_column = f"{outcome_column}_std"

    if mean_column not in average_across_runs.columns:
        return

    fig = px.bar(
        average_across_runs,
        x="Scenario",
        y=mean_column,
        color="Scenario",
        color_discrete_map=scenario_colour_map,
        category_orders={"Scenario": scenario_order},
        error_y=sd_column if sd_column in average_across_runs.columns else None,
        title=title,
        labels={
            mean_column: "Mean time (minutes)",
            sd_column: "SD across runs",
        },
    )

    if outcome_column == "Time in System":
        fig.add_hline(y=240, line_dash="dash")

    st.plotly_chart(fig, use_container_width=True)


def plot_patient_distribution(patient_df, outcome_column, title):
    """Plot patient-level distribution by scenario."""
    if outcome_column not in patient_df.columns:
        return

    fig = px.box(
        patient_df,
        x="Scenario",
        y=outcome_column,
        color="Scenario",
        color_discrete_map=scenario_colour_map,
        category_orders={"Scenario": scenario_order},
        points=False,
        title=title,
        labels={outcome_column: "Time (minutes)"},
    )

    if outcome_column == "Time in System":
        fig.add_hline(y=240, line_dash="dash")

    st.plotly_chart(fig, use_container_width=True)


def plot_hourly_mean(patient_df, outcome_column, hour_column, run_column, title):
    """
    Plot average hourly outcome by scenario.

    First calculate mean per scenario/run/hour, then average those run-level
    hourly means across runs.
    """
    if outcome_column not in patient_df.columns or hour_column is None or run_column is None:
        return

    hourly_run_means = (
        patient_df
        .groupby(["Scenario", run_column, hour_column], as_index=False)[outcome_column]
        .mean()
    )

    hourly_mean_across_runs = (
        hourly_run_means
        .groupby(["Scenario", hour_column], as_index=False)[outcome_column]
        .mean()
    )

    fig = px.line(
        hourly_mean_across_runs,
        x=hour_column,
        y=outcome_column,
        color="Scenario",
        color_discrete_map=scenario_colour_map,
        category_orders={"Scenario": scenario_order},
        markers=True,
        title=title,
        labels={outcome_column: "Mean time (minutes)"},
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_three_metric_row(plot_func, df, title_prefix=""):
    """Plot the three main outcomes in a consistent three-column row."""
    cols = st.columns(3)

    metrics = [
        ("Time in System", "time in ED/system"),
        ("Arrival to ED Assessment", "time to ED assessment"),
        (consultant_column, "time to consultant assessment"),
    ]

    for col, (metric, label) in zip(cols, metrics):
        with col:
            if metric is not None:
                plot_func(
                    df,
                    metric,
                    f"{title_prefix}{label}",
                )
            else:
                st.info("Consultant assessment time column not found.")


def plot_three_hourly_metric_row(df, hour_column, run_column):
    """Plot the three hourly outcome trends in a consistent three-column row."""
    cols = st.columns(3)

    metrics = [
        ("Time in System", "Mean time in ED/system by hour of arrival"),
        ("Arrival to ED Assessment", "Mean time to ED assessment by hour of arrival"),
        (consultant_column, "Mean time to consultant assessment by hour of arrival"),
    ]

    for col, (metric, title) in zip(cols, metrics):
        with col:
            if metric is not None:
                plot_hourly_mean(
                    df,
                    metric,
                    hour_column,
                    run_column,
                    title,
                )
            else:
                st.info("Consultant assessment time column not found.")


# =====================================================
# Identify key columns
# =====================================================

run_column = first_existing_column(
    patient_df,
    ["run_number", "Run Number", "Run", "run"],
)

hour_column = first_existing_column(
    patient_df,
    ["Hour of Arrival", "hour_of_arrival", "Arrival Hour"],
)

consultant_column = first_existing_column(
    patient_df,
    [
        "Arrival to Consultant Assessment",
        "Time to Consultant Assessment",
        "Arrival to consultant assessment",
        "arrival_to_consultant_assessment",
    ],
)

outcome_columns = [
    "Time in System",
    "Arrival to ED Assessment",
]

if consultant_column is not None:
    outcome_columns.append(consultant_column)

run_level_means = build_run_level_means(patient_df, outcome_columns, run_column)

if run_level_means is not None:
    average_across_runs = build_average_across_runs(run_level_means, outcome_columns)
else:
    average_across_runs = None


# =====================================================
# Header
# =====================================================

col1, col2, col3 = st.columns([1.25, 2, 1.25])

with col1:
    st.image("assets/uom.jpeg")

with col2:
    st.markdown(
        "<h1 style='text-align: center;'>Simulation Results</h1>",
        unsafe_allow_html=True,
    )

with col3:
    st.image("assets/mft.png")

st.markdown("---")


# =====================================================
# Headline comparison plots: mean across runs
# =====================================================

st.subheader("Headline outcomes")

if average_across_runs is None:
    st.warning(
        "Could not calculate mean across runs because no run number column was found in the patient-level results."
    )
else:
    plot_three_metric_row(
        plot_mean_across_runs,
        average_across_runs,
        title_prefix="Mean ",
    )


# =====================================================
# Patient-level distributions
# =====================================================

st.subheader("Patient-level distributions")

plot_three_metric_row(
    plot_patient_distribution,
    patient_df,
    title_prefix="Patient-level ",
)


# =====================================================
# Time-of-day patterns
# =====================================================

st.subheader("Time-of-day patterns")

if hour_column is None:
    st.info("Hour of arrival column not found in the patient-level results.")
elif run_column is None:
    st.info("Run number column not found, so hourly averages across runs cannot be calculated.")
else:
    plot_three_hourly_metric_row(
        patient_df,
        hour_column,
        run_column,
    )