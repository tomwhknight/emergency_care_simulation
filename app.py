import streamlit as st


st.set_page_config(layout="wide")

# 1. App Banner

# Two logos: UoM (left) and MFT (right)
col1, col2, col3 = st.columns([1.25, 2, 1.25])

with col1:
    st.image("/Users/thomasknight/Desktop/ACL/Side projects/sepsis/uom.jpeg")

with col3:
    st.image("/Users/thomasknight/Desktop/ACL/Side projects/sepsis/mft.png")


# --- Title & Intro ---
st.markdown(
    """
    <h1 style='text-align: center; font-size: 2.5em;'>
    Acute Care Pathway Simulation Dashboard
    </h1>
    <p style='text-align: center; font-size: 1.05em; color: gray;'>
        An interactive dashboard for exploring acute medical service activity in the emergency department.
        Use the sidebar to run simulations, review outputs, or access technical details.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --- About the Dashboard ---
st.subheader("What is this dashboard for?")
st.markdown(
    """
    This tool helps clinicians, operational teams, and service designers understand how acute medical 
    admissions flow through the emergency department under different demand and capacity scenarios. 
    It is powered by a discrete event simulation (DES) model that tracks patients as they are assessed, 
    referred, and admitted or discharged.

    It can be used to:
    - Explore system behaviour under different attendance volumes or staffing configurations
    - Identify pressure points and delays in the ED-to-acute medicine pathway
    - Evaluate the potential impact of changes such as SDEC expansion, AMU capacity, or altered referral criteria
    """
)

# --- How to Use It ---
st.subheader("How to use the dashboard")
st.markdown(
    """
    1. **Go to _Run Simulation_** in the sidebar and configure the simulation parameters  
    2. **Click 'Run Simulation'** to generate patient-level and system-level results  
    3. **Review outputs** in the _Summary Results_ section, including breach rates, process times, and queue behaviour  
    4. Visit _Technical Notes_ to read about the underlying logic and data assumptions
    """
)

# --- Optional Footer ---
st.markdown("---")
st.caption("Developed by Tom Knight, Nicola Crompton, Tim Felton, Anthony part of an acute care modelling project. Last updated: April 2025.")