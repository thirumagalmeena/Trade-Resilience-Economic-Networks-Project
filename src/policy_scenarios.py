import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import altair as alt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

DEFAULT_DATA_PATH = Path("D:/DPL 3/data/integrated_tren_dataset_with_indexes.csv")

POLICY_COLUMNS = {
    "country": ["Country"],
    "year": ["Year"],
    "gdp_growth": ["GDP growth (annual %)"],
    "gdp_per_capita": ["GDP per capita (current US$)"],
    "resilience_score": ["Resilience_Score"],
    "trade_dependency_index": ["Trade_Dependency_Index"],
    "shock_impact_score": ["Shock_Impact_Score"],
    "youth_unemployment": ["Unemployment, youth total (% of total labor force ages 15-24) (modeled ILO estimate)"],
    "advanced_education_unemployment": ["Unemployment with advanced education (% of total labor force with advanced education)"],
    "agr_yield_avg": ["agr_yield_avg"],
    "agr_production_avg": ["agr_production_avg"],
    "disaster_deaths": ["disaster_deaths"],
    "disaster_affected": ["disaster_affected"],
    "gdp_current": ["GDP (current US$)"],
    "total_unemployment": ["Unemployment, total (% of total labor force) (modeled ILO estimate)_y"],
    "poverty_headcount": ["Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)"],
}

@st.cache_data(show_spinner=False)
def load_policy_data() -> pd.DataFrame:
    try:
        if DEFAULT_DATA_PATH.exists():
            df = pd.read_csv(DEFAULT_DATA_PATH)
        else:
            st.error(f"Dataset not found at {DEFAULT_DATA_PATH}. Please provide the correct dataset.")
            return pd.DataFrame()
        
        df = df.drop_duplicates()
        
        if 'Country' in df.columns:
            df['Country'] = df['Country'].astype(str)
        if 'Year' in df.columns or 'year' in df.columns:
            year_col = 'Year' if 'Year' in df.columns else 'year'
            df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def resolve_policy_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping = {}
    for key, candidates in POLICY_COLUMNS.items():
        found = None
        for candidate in candidates:
            if candidate in df.columns:
                found = candidate
                break
        mapping[key] = found
    return mapping

def compute_policy_impact_index(df: pd.DataFrame, column_mapping: Dict[str, Optional[str]], 
                                latest_year_only: bool = True) -> pd.DataFrame:
    country_col = column_mapping['country']
    year_col = column_mapping['year']
    
    if not country_col or not year_col:
        st.error("Country or Year columns not found in data")
        return pd.DataFrame()
    
    if latest_year_only:
        df_latest = df[df[year_col] == df.groupby(country_col)[year_col].transform('max')]
    else:
        df_latest = df.copy()
    
    results = []
    
    for _, country_data in df_latest.iterrows():
        country = country_data[country_col]
        
        resilience = float(country_data.get(column_mapping['resilience_score'], 0.5))
        trade_dep = float(country_data.get(column_mapping['trade_dependency_index'], 0.5))
        shock_impact = float(country_data.get(column_mapping['shock_impact_score'], 0.5))
        youth_unemp = float(country_data.get(column_mapping['youth_unemployment'], 10.0))
        
        # Simple policy impact index as average vulnerability
        index = (trade_dep + shock_impact + youth_unemp / 100) / 3 * 100
        index = np.clip(index, 0, 100)
        
        results.append({
            'Country': country,
            'Resilience_Score': resilience,
            'Trade_Dependency': trade_dep,
            'Shock_Impact': shock_impact,
            'Youth_Unemployment': youth_unemp,
            'Policy_Impact_Index': index,
            'Risk_Category': 'High' if index > 75 else 'Medium' if index > 50 else 'Low'
        })
    
    return pd.DataFrame(results).sort_values('Policy_Impact_Index', ascending=False)

def simulate_policy_scenario(df: pd.DataFrame, pi_df: pd.DataFrame, 
                             column_mapping: Dict[str, Optional[str]], 
                             selected_countries: List[str], params: Dict) -> pd.DataFrame:
    results = []
    
    for country in selected_countries:
        if country not in pi_df['Country'].values:
            continue
        
        country_info = pi_df[pi_df['Country'] == country].iloc[0]
        country_data = df[df[column_mapping['country']] == country].iloc[-1]
        
        # Baseline values
        resilience = float(country_data.get(column_mapping['resilience_score'], 0.5))
        gdp_growth = float(country_data.get(column_mapping['gdp_growth'], 0))
        trade_dep = float(country_data.get(column_mapping['trade_dependency_index'], 0))
        shock_impact = float(country_data.get(column_mapping['shock_impact_score'], 0))
        youth_unemp = float(country_data.get(column_mapping['youth_unemployment'], 10))
        advanced_unemp = float(country_data.get(column_mapping['advanced_education_unemployment'], 10))
        agr_yield = float(country_data.get(column_mapping['agr_yield_avg'], 0))
        agr_prod = float(country_data.get(column_mapping['agr_production_avg'], 0))
        disaster_deaths = float(country_data.get(column_mapping['disaster_deaths'], 0))
        disaster_affected = float(country_data.get(column_mapping['disaster_affected'], 0))
        gdp_current = float(country_data.get(column_mapping['gdp_current'], 1e9))
        total_unemp = float(country_data.get(column_mapping['total_unemployment'], 10))
        poverty = float(country_data.get(column_mapping['poverty_headcount'], 20)) if column_mapping['poverty_headcount'] else 20
        
        # Apply scenarios based on params
        # Trade Diversification
        trade_red = params['trade_intensity'] if params['use_trade'] else 0
        trade_dep_new = trade_dep * (1 - trade_red)
        resilience += trade_red * 0.8
        
        # Youth Employment
        youth_red = params['youth_target'] if params['use_youth'] else 0
        youth_unemp_new = youth_unemp * (1 - youth_red)
        gdp_growth += youth_red * 0.6
        resilience += youth_red * 0.4
        
        # Agricultural Productivity
        agr_boost = params['agri_boost'] if params['use_agri'] else 0
        agr_yield_new = agr_yield * (1 + agr_boost)
        agr_prod_new = agr_prod * (1 + agr_boost)
        gdp_growth += agr_boost * 0.8
        resilience += agr_boost * 0.6
        poverty_new = poverty * (1 - agr_boost * 0.4)
        
        # Disaster Preparedness
        dis_invest = params['disaster_investment'] if params['use_dis'] else 0
        shock_impact_new = shock_impact * (1 - dis_invest * 10)
        resilience += dis_invest * 5
        disaster_deaths_new = disaster_deaths * (1 - dis_invest * 10)
        disaster_affected_new = disaster_affected * (1 - dis_invest * 10)
        
        # Education Investment
        edu_boost = params['edu_investment'] if params['use_edu'] else 0
        advanced_unemp_new = advanced_unemp * (1 - edu_boost)
        gdp_growth += edu_boost * 0.8
        
        resilience = np.minimum(1.0, resilience)
        shock_impact_new = np.maximum(0, shock_impact_new)
        
        # Recovery years estimation (simplified)
        recovery_years = max(1, min(5, int((trade_dep + shock_impact + youth_unemp / 10) / 20)))
        
        results.append({
            'Country': country,
            'Resilience_Delta': resilience - country_info['Resilience_Score'],
            'GDP_Growth_Delta': gdp_growth,
            'Trade_Dep_Delta': trade_dep_new - trade_dep,
            'Shock_Impact_Delta': shock_impact_new - shock_impact,
            'Youth_Unemp_Delta': youth_unemp_new - youth_unemp,
            'Policy_Impact_Index': country_info['Policy_Impact_Index'],
            'Recovery_Years': recovery_years
        })
    
    return pd.DataFrame(results)

def create_policy_visualizations(pi_df: pd.DataFrame, scenario_results: pd.DataFrame = None):
    st.subheader("Policy Impact Index Rankings")
    
    if not pi_df.empty:
        top_10 = pi_df.head(10)
        
        chart1 = alt.Chart(top_10).mark_bar().encode(
            x=alt.X('Policy_Impact_Index:Q', title='Policy Impact Index'),
            y=alt.Y('Country:N', sort='-x', title='Country'),
            color=alt.Color('Risk_Category:N',
                            scale=alt.Scale(domain=['Low', 'Medium', 'High'],
                                            range=['green', 'orange', 'red']),
                            title='Risk Level'),
            tooltip=['Country:N', 'Policy_Impact_Index:Q', 
                     'Trade_Dependency:Q', 'Shock_Impact:Q']
        ).properties(
            width=600,
            height=400,
            title="Top 10 Countries by Policy Impact Vulnerability"
        )
        
        st.altair_chart(chart1, use_container_width=True)
        
        st.subheader("Trade Dependency vs Resilience")
        
        scatter = alt.Chart(pi_df).mark_circle(size=100).encode(
            x=alt.X('Trade_Dependency:Q', title='Trade Dependency Index'),
            y=alt.Y('Resilience_Score:Q', title='Resilience Score'),
            color=alt.Color('Policy_Impact_Index:Q',
                            scale=alt.Scale(scheme='viridis'),
                            title='PII Score'),
            size=alt.Size('Shock_Impact:Q',
                          scale=alt.Scale(range=[50, 400]),
                          title='Shock Impact'),
            tooltip=['Country:N', 'Policy_Impact_Index:Q', 
                     'Trade_Dependency:Q', 'Resilience_Score:Q']
        ).properties(
            width=600,
            height=400,
            title="Policy Vulnerability Matrix"
        )
        
        st.altair_chart(scatter, use_container_width=True)
    
    if scenario_results is not None and not scenario_results.empty:
        st.subheader("Policy Scenario Simulation")
        
        scenario_chart = alt.Chart(scenario_results).mark_bar().encode(
            x=alt.X('Country:N', title='Country'),
            y=alt.Y('Resilience_Delta:Q', title='Resilience Delta'),
            color=alt.Color('Resilience_Delta:Q',
                            scale=alt.Scale(scheme='greens'),
                            title='Improvement'),
            tooltip=['Country:N', 'Resilience_Delta:Q', 'GDP_Growth_Delta:Q']
        ).properties(
            width=600,
            height=300,
            title="Estimated Resilience Improvement from Policies"
        )
        
        st.altair_chart(scenario_chart, use_container_width=True)

def main():
    st.markdown('<div class="main-header">Policy Scenario Explorer</div>', 
                unsafe_allow_html=True)
    st.caption("Exploring impacts of policy interventions on economic resilience")
 
    with st.spinner("Loading policy data..."):
        df = load_policy_data()
  
    if df.empty:
        st.error("Failed to load data. Please ensure the dataset is available and try again.")
        st.stop()
    
    st.sidebar.success(f"Loaded {len(df)} records for {df['Country'].nunique()} countries")

    column_mapping = resolve_policy_columns(df)

    st.sidebar.subheader("Analysis Parameters")
    latest_data_only = st.sidebar.checkbox("Use latest year data only", value=True)
    
    st.sidebar.subheader("Policy Toggles")
    use_trade = st.sidebar.checkbox("Trade Diversification", value=True)
    use_youth = st.sidebar.checkbox("Youth Employment", value=True)
    use_agri = st.sidebar.checkbox("Agricultural Productivity", value=True)
    use_dis = st.sidebar.checkbox("Disaster Preparedness", value=True)
    use_edu = st.sidebar.checkbox("Education Investment", value=True)
    
    st.sidebar.subheader("Policy Intensities")
    trade_intensity = st.sidebar.slider("Trade Diversification Intensity", 0.1, 0.4, 0.25, 0.05)
    youth_target = st.sidebar.slider("Youth Employment Reduction", 0.2, 0.8, 0.5, 0.1)
    agri_boost = st.sidebar.slider("Agricultural Yield Increase", 0.1, 0.4, 0.25, 0.05)
    disaster_investment = st.sidebar.slider("Disaster Investment (% GDP)", 0.01, 0.05, 0.02, 0.01)
    edu_investment = st.sidebar.slider("Education Investment Intensity", 0.2, 0.8, 0.5, 0.1)
    
    params = {
        'use_trade': use_trade,
        'use_youth': use_youth,
        'use_agri': use_agri,
        'use_dis': use_dis,
        'use_edu': use_edu,
        'trade_intensity': trade_intensity,
        'youth_target': youth_target,
        'agri_boost': agri_boost,
        'disaster_investment': disaster_investment,
        'edu_investment': edu_investment
    }
    
    st.header("Policy Impact Index (PII) Analysis")
    
    with st.spinner("Computing Policy Impact Index..."):
        pi_results = compute_policy_impact_index(df, column_mapping, latest_data_only)
    
    if pi_results.empty:
        st.error("Could not compute Policy Impact Index. Please check your data.")
        st.stop()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Countries Analyzed", len(pi_results))
    
    with col2:
        high_risk = len(pi_results[pi_results['Risk_Category'] == 'High'])
        st.metric("High Risk Countries", high_risk, delta=f"{high_risk/len(pi_results)*100:.1f}%")
    
    with col3:
        avg_pii = pi_results['Policy_Impact_Index'].mean()
        st.metric("Average PII Score", f"{avg_pii:.1f}")
    
    with col4:
        max_trade_dep = pi_results['Trade_Dependency'].max()
        st.metric("Max Trade Dependency", f"{max_trade_dep:.3f}")
    
    st.subheader("Top 3 Most Vulnerable Countries")
    
    top_3 = pi_results.head(3)
    
    for idx, (_, country_data) in enumerate(top_3.iterrows(), 1):
        with st.container(border=True):
            st.markdown(f"### #{idx} {country_data['Country']}")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("PII Score", f"{country_data['Policy_Impact_Index']:.1f}")
                st.metric("Risk Level", country_data['Risk_Category'])
            
            with c2:
                st.metric("Trade Dependency", f"{country_data['Trade_Dependency']:.3f}")
            
            with c3:
                st.metric("Resilience Score", f"{country_data['Resilience_Score']:.3f}")
    
    st.subheader("Complete Rankings")
    
    display_cols = ['Country', 'Policy_Impact_Index', 'Risk_Category',
                   'Trade_Dependency', 'Resilience_Score']
    
    formatted_results = pi_results[display_cols].copy()
    formatted_results['Trade_Dependency'] = formatted_results['Trade_Dependency'].apply(lambda x: f"{x:.3f}")
    formatted_results['Resilience_Score'] = formatted_results['Resilience_Score'].apply(lambda x: f"{x:.3f}")
    formatted_results['Policy_Impact_Index'] = formatted_results['Policy_Impact_Index'].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(
        formatted_results,
        column_config={
            "Country": "Country",
            "Policy_Impact_Index": "PII Score",
            "Risk_Category": "Risk Level",
            "Trade_Dependency": "Trade Dep Index",
            "Resilience_Score": "Resilience Score"
        },
        use_container_width=True
    )
    
    st.header("Policy Scenario Simulation")
    st.markdown("Simulate the impact of selected policy interventions")
    
    available_countries = pi_results['Country'].tolist()
    default_selection = top_3['Country'].tolist()
    
    selected_countries = st.multiselect(
        "Select countries to simulate policies:",
        available_countries,
        default=default_selection,
        help="Countries to include in the policy scenario simulation"
    )
    
    if selected_countries:
        with st.spinner("Running policy scenario simulation..."):
            scenario_results = simulate_policy_scenario(
                df, pi_results, column_mapping, selected_countries, params
            )
        
        if not scenario_results.empty:
            st.subheader("Policy Intervention Impacts")
            
            avg_res_delta = scenario_results['Resilience_Delta'].mean()
            max_res_delta = scenario_results['Resilience_Delta'].max()
            best_country = scenario_results.loc[scenario_results['Resilience_Delta'].idxmax(), 'Country']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Resilience Delta", f"{avg_res_delta:.3f}")
            with col2:
                st.metric("Max Resilience Delta", f"{max_res_delta:.3f}")
            with col3:
                st.metric("Most Improved", best_country)
            
            for _, result in scenario_results.iterrows():
                with st.container(border=True):
                    st.markdown(f"### {result['Country']}")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric(
                            "Resilience Delta",
                            f"{result['Resilience_Delta']:.3f}"
                        )
                    with c2:
                        st.metric("GDP Growth Delta", f"{result['GDP_Growth_Delta']:.2f}%")
                    with c3:
                        st.metric("Recovery Years", f"{result['Recovery_Years']:.0f}")
                    with c4:
                        st.metric("PII Score", f"{result['Policy_Impact_Index']:.1f}")
    
    create_policy_visualizations(pi_results, scenario_results if 'scenario_results' in locals() else None)
    
    with st.expander("Methodology & Assumptions", expanded=False):
        st.markdown("""
        ### Policy Impact Index (PII) Calculation
        
        **Components:**
        - Trade Dependency Index
        - Shock Impact Score
        - Youth Unemployment (normalized)
        - Averaged and scaled to 0-100
        
        **Risk Categories:**
        - **High Risk**: PII > 75
        - **Medium Risk**: PII 50-75
        - **Low Risk**: PII < 50
        
        ### Policy Scenario Simulation Assumptions
        
        1. **Trade Diversification:** Reduces dependency by intensity, boosts resilience.
        2. **Youth Employment:** Reduces unemployment by target, boosts GDP and resilience.
        3. **Agricultural Productivity:** Increases yield/production, boosts GDP/resilience, reduces poverty.
        4. **Disaster Preparedness:** Reduces shock impact/deaths/affected, boosts resilience.
        5. **Education Investment:** Reduces advanced unemployment, boosts GDP.
        6. **Recovery Years:** Estimated based on baseline vulnerabilities (1-5 years).
        
        ### Data Sources & Limitations
        
        - Uses aggregate economic metrics from dataset
        - Simplified linear assumptions for policy impacts
        - No dynamic interactions between policies modeled
        - Static model without temporal effects
        """)

if __name__ == "__main__":
    main()