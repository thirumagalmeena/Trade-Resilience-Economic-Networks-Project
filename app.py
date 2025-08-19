import streamlit as st
from src import forecasting
from src import milestone1, milestone2, milestone3, milestone4, milestone5, milestone6,milestone8
from src import milestone11, milestone13, milestone7, milestone10
from src import visualization, policy_scenarios

def main():
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Forecasting", "Visualisations","Policies","Drought Shock Simulation", 
         "Trade Risk Analysis","China Export Drop Percentage", "Food Security Analysis",
         "Youth Unemployment Projection", "Export Sector Ageing Risk","Global Network Trade Graph",
         "Trade Relationship Mutual Benefit","Economic Prediction for 2030",
         "2030 Economic Scenario Projections","Trade Network Optimization"]
    )
    
    if page == "Home":
        st.subheader("Welcome to the Economic Data Analysis Dashboard")
        st.write("""
        This dashboard provides comprehensive analysis of economic data including:
        
        - **EDA**: Exploratory Data Analysis of economic indicators
        - **Time Series**: Historical trends and patterns analysis  
        - **Sentiment Analysis**: Economic sentiment from various sources
        - **Forecasting**: Economic projections and scenario modeling for 2030
        - **Trade Risk Analysis**: Trade dependency vulnerability assessment
        - **Food Security Analysis**: Agricultural import dependency and food security risk
        - **Export Sector Ageing Risk**: Impact of ageing demographics on export sector productivity
        
        Navigate using the sidebar to explore different sections.
        """)
        
        st.info("💡 Start with the **Forecasting** section to explore GDP growth, poverty rates, and trade resilience projections!")
        st.info("🌍 Check out **Trade Risk Analysis** to identify countries most vulnerable to trade partner collapse!")
        st.info("🥕 Explore **Food Security Analysis** to assess agricultural import risks!")
        st.info("👴 Dive into **Export Sector Ageing Risk** to evaluate labor shortage risks from ageing populations!")

    elif page == "Visualisations":
        visualization.main()

    elif page == "Forecasting":
        forecasting.main()
        
    elif page == "Policies":
        policy_scenarios.main()

    elif page == "Trade Risk Analysis":
        milestone1.main()

    elif page == "China Export Drop Percentage":
        milestone2.main()
    
    elif page == "Drought Shock Simulation":
        milestone3.main()
    
    elif page == "Food Security Analysis":
        milestone4.main()
    
    elif page == "Youth Unemployment Projection":
        milestone5.main()

    elif page == "Export Sector Ageing Risk":
        milestone6.main()

    elif page == "Global Network Trade Graph":
        milestone7.main()
    
    elif page == "Trade Relationship Mutual Benefit":
        milestone8.main()

    elif page == "Economic Prediction for 2030":
       milestone10.main()

    elif page == "2030 Economic Scenario Projections":
        milestone11.main()

    elif page == "Trade Network Optimization":
        milestone13.main()

if __name__ == "__main__":
    main()