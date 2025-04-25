import plotly.express as px
import pandas as pd
import os
import streamlit as st

# Two logos: UoM (left) and MFT (right)
col1, col2, col3 = st.columns([1.25, 2, 1.25])

with col1:
    st.image("/Users/thomasknight/Desktop/ACL/Side projects/sepsis/uom.jpeg")

with col3:
    st.image("/Users/thomasknight/Desktop/ACL/Side projects/sepsis/mft.png")

st.markdown(
    """
    <h1 style='text-align: center; font-size: 2.5em;'>
        Technical details
    </h1>
    """,
    unsafe_allow_html=True
)

import streamlit as st

# --- DES Models ---
st.markdown("### Discrete Event Simulation (DES) Models")
st.markdown(""" Emergency care is a complex system, with unpredictable demand, variation in clinical needs, and limited resources.  
Understanding how this system behaves — and how it might respond to changes — requires tools that can capture both complexity and uncertainty.
This model uses a method called **Discrete Event Simulation (DES)** to replicate patient journeys through the front-door of the hospital.  
DES models are built from a small number of key components:

- **Entities**: in this case, individual patients, each with attributes like age, NEWS2 score, and probability of admission.
- **Resources**: such as triage nurses, ED doctors, and assessment rooms — which may be limited and vary by time of day.
- **Events**: things that happen at specific points in time, like a patient arriving, starting triage, or completing an assessment.
- **Queues**: where patients wait when resources are busy.
- **Processes**: sequences of events that define how entities move through the system.

The simulation is **probabilistic**, meaning that key elements — like when a patient arrives, how long they wait, or whether they're admitted — are drawn from distributions based on real data or expert knowledge.  
This allows the model to reflect the variability and unpredictability we observe in practice.

The model is implemented in **Python using SimPy**, an open-source simulation library that supports event-driven modelling.  
By running the model many times under different conditions, we can explore **“what-if” scenarios** — for example:
- What if we increase medical staffing after 6pm?
- What if we change referral criteria to SDEC?
- What happens if we reduce triage capacity overnight?

This approach allows teams to test operational changes in a simulated environment before introducing them in real life, supporting evidence-informed decision making.
""")

# --- Process Map ---
st.markdown("### Process Map")

st.image('/Users/thomasknight/Desktop/ACL/Projects/emergency_care_simulation/documentation/process_map.jpeg')

# --- Model Boundaries ---
st.markdown("### Model Boundaries")
st.markdown("""This simulation focuses on the care of acute medical patients within the Emergency Department (ED). It includes patients who are assessed and managed in the ED setting prior to discharge or referral.
Patients who are referred to acute medicine and subsequently admitted to Same Day Emergency Care (SDEC) or the Acute Medical Unit (AMU) are treated as having exited the model. In this simulation, SDEC and AMU function as sinks — once a patient is transferred to either area, no further events or delays are modelled.
When interpreting process outcomes — such as time to medical assessment or time to consultant assessment — it is important to note that these timings only apply to patients assessed in the ED. They exclude patients whose assessments occur after admission to SDEC or AMU, as those processes fall outside the scope of this model.
""")

# --- Attributes ---
st.markdown("### Patient Attributes")

st.markdown("#### Age")
st.markdown("Sampled using a weighted distribution to reflect real-world attendance patterns, with higher weights for children and older adults.")

st.markdown("#### NEWS2")
st.markdown("Sampled from a predefined probability distribution. It captures clinical acuity and influences referral and admission decisions.")

st.markdown("#### Admission Probability")
st.markdown("""
Assigned at patient generation using Beta or Normal distributions.  
Varies by patient group (e.g., older patients with high NEWS2 more likely to be admitted).  
Used in ED disposition decisions and SDEC eligibility.
""")

# --- Generators ---
st.markdown("### Patient Generators")
st.markdown("""
Patient arrivals are simulated using a **non-homogeneous Poisson process**.  
Arrival intensity varies by **hour of day and day of week**, matching empirical attendance data.  
This allows modelling of peak periods and diurnal variation in demand.
""")

# --- Resources and Rota Scheduling ---
st.markdown("### Resources and Rota Scheduling")

st.markdown("""
Time-varying capacity is defined through rota files for each resource type:  
**triage nurses, ED doctors, and medical staff**. Each has hourly resolution across days of the week.
""")

st.markdown("#### Capacity by Shift")
st.markdown("""
Resource capacity changes based on time of day. For example:  
higher staffing during the day, reduced levels overnight.  
These patterns are imported from CSVs and directly affect waiting times and queues.
""")

st.markdown("#### End-of-Shift Logic")
st.markdown("""
To mimic operational reality, staff are **not assigned new patients in the final 30 minutes of their shift**.  
This simulates natural tapering during handover and end-of-shift routines.
""")

st.markdown("#### Break Allocation")
st.markdown("""
Breaks are implemented using **blocking functions**.  
Resources are temporarily unavailable according to break timing rules, simulating real-world rest periods.
""")

# --- Service Time Assumptions ---
st.markdown("### Service Time Assumptions")

st.markdown("""
Process times (e.g., for triage, ED assessment, referral) are sampled from statistical distributions, such as lognormal or exponential.  
However, estimating true service times is difficult due to **overlapping tasks, shared queues, and parallel processes**.
""")