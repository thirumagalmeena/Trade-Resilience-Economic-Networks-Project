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

TRADE_COLUMNS = {
    "country": ["Country"],
    "year": ["Year", "year"],
    "exports_gdp": ["Exports of goods and services (% of GDP)"],
    "imports_gdp": ["Imports of goods and services (% of GDP)"],
    "trade_gdp": ["Trade (% of GDP)"],
    "gdp_current": ["GDP (current US$)"],
    "trade_dependency_index": ["Trade_Dependency_Index"],
}

@st.cache_data(show_spinner=False)
def load_trade_data() -> pd.DataFrame:
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
        
        # Filter for recent years (2015-2023)
        df = df[df[year_col] >= 2015].copy()
        
        # Calculate additional trade metrics
        if 'Exports of goods and services (% of GDP)' in df.columns and 'Imports of goods and services (% of GDP)' in df.columns:
            df['Export_Import_Ratio'] = (
                df['Exports of goods and services (% of GDP)'] / 
                df['Imports of goods and services (% of GDP)']
            )
        
        # Fill missing values with group means
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df.groupby('Country')[numeric_cols].transform(
            lambda x: x.fillna(x.mean())
        )
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def resolve_trade_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    mapping = {}
    for key, candidates in TRADE_COLUMNS.items():
        found = None
        for candidate in candidates:
            if candidate in df.columns:
                found = candidate
                break
        mapping[key] = found
    return mapping

def compute_vulnerability_index(df: pd.DataFrame, column_mapping: Dict[str, Optional[str]], 
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
        
        trade_openness = float(country_data.get(column_mapping['trade_gdp'], 0))
        import_dependency = float(country_data.get(column_mapping['imports_gdp'], 0))
        export_reliance = float(country_data.get(column_mapping['exports_gdp'], 0))
        
        # Normalize vulnerability scores
        scaler = StandardScaler()
        features = pd.DataFrame({
            'Trade_Openness': [trade_openness],
            'Import_Dependency': [import_dependency],
            'Export_Reliance': [export_reliance]
        })
        vulnerability_score = scaler.fit_transform(features).mean(axis=1)[0] * 20 + 50
        vulnerability_score = np.clip(vulnerability_score, 0, 100)
        
        results.append({
            'Country': country,
            'Trade_Openness': trade_openness,
            'Import_Dependency': import_dependency,
            'Export_Reliance': export_reliance,
            'Vulnerability_Index': vulnerability_score,
            'Risk_Category': 'High' if vulnerability_score > 75 else 'Medium' if vulnerability_score > 50 else 'Low'
        })
    
    return pd.DataFrame(results).sort_values('Vulnerability_Index', ascending=False)

def simulate_china_export_shock(df: pd.DataFrame, vi_df: pd.DataFrame, 
                               column_mapping: Dict[str, Optional[str]], 
                               shock_percentage: float = 0.25) -> pd.DataFrame:
    results = []
    
    # Get China's trade data
    china_data = df[df[column_mapping['country']] == 'China'].iloc[-1] if not df[df[column_mapping['country']] == 'China'].empty else None
    if china_data is None:
        st.warning("China not found in dataset. Using synthetic data.")
        china_exports_gdp = 20.0  # Approximate China exports as % of GDP
        china_gdp = 17e12  # Approximate China GDP in USD
    else:
        china_exports_gdp = float(china_data.get(column_mapping['exports_gdp'], 20.0))
        china_gdp = float(china_data.get(column_mapping['gdp_current'], 17e12))
    
    china_export_drop = china_gdp * (china_exports_gdp / 100) * shock_percentage
    
    for _, country_info in vi_df.iterrows():
        country = country_info['Country']
        
        import_dependency = country_info['Import_Dependency']
        vulnerability_score = country_info['Vulnerability_Index']
        gdp_current = float(df[df[column_mapping['country']] == country][column_mapping['gdp_current']].iloc[-1]) if column_mapping['gdp_current'] else 0
        
        if country == 'China':
            gdp_loss_pct = shock_percentage * (china_exports_gdp / 100) * 0.8  # 80% pass-through
            multiplier = 1.0
        else:
            trade_dependency_factor = import_dependency / 100
            vulnerability_factor = vulnerability_score / 100
            china_trade_share = min(0.3, trade_dependency_factor * 0.5)  # Max 30% trade with China
            
            multiplier = 1.0
            if import_dependency > 50:
                multiplier = 2.0
            elif import_dependency > 30:
                multiplier = 1.5
            
            direct_impact = china_trade_share * shock_percentage * 0.6  # 60% pass-through
            indirect_impact = vulnerability_factor * 0.02 * shock_percentage
            gdp_loss_pct = (direct_impact + indirect_impact) * 100
        
        gdp_loss_usd = gdp_current * (gdp_loss_pct / 100)
        recovery_years = max(1, min(5, int(abs(gdp_loss_pct) / 5)))
        
        results.append({
            'Country': country,
            'GDP_Loss_Percentage': gdp_loss_pct,
            'GDP_Loss_USD': gdp_loss_usd,
            'Multiplier': multiplier,
            'Recovery_Years': recovery_years,
            'Vulnerability_Index': country_info['Vulnerability_Index']
        })
    
    return pd.DataFrame(results)

def create_shock_visualizations(vi_df: pd.DataFrame, shock_results: pd.DataFrame = None):
    st.subheader("Vulnerability Index Rankings")
    
    if not vi_df.empty:
        top_10 = vi_df.head(10)
        
        chart1 = alt.Chart(top_10).mark_bar().encode(
            x=alt.X('Vulnerability_Index:Q', title='Vulnerability Index'),
            y=alt.Y('Country:N', sort='-x', title='Country'),
            color=alt.Color('Risk_Category:N',
                            scale=alt.Scale(domain=['Low', 'Medium', 'High'],
                                          range=['green', 'orange', 'red']),
                            title='Risk Level'),
            tooltip=['Country:N', 'Vulnerability_Index:Q', 
                    'Import_Dependency:Q', 'Export_Reliance:Q']
        ).properties(
            width=600,
            height=400,
            title="Top 10 Countries by Trade Vulnerability"
        )
        
        st.altair_chart(chart1, use_container_width=True)
        
        st.subheader("Import Dependency vs Vulnerability")
        
        scatter = alt.Chart(vi_df).mark_circle(size=100).encode(
            x=alt.X('Import_Dependency:Q', title='Import Dependency (% of GDP)'),
            y=alt.Y('Vulnerability_Index:Q', title='Vulnerability Index'),
            color=alt.Color('Vulnerability_Index:Q',
                            scale=alt.Scale(scheme='viridis'),
                            title='VI Score'),
            size=alt.Size('Trade_Openness:Q',
                          scale=alt.Scale(range=[50, 400]),
                          title='Trade Openness'),
            tooltip=['Country:N', 'Vulnerability_Index:Q', 
                    'Import_Dependency:Q', 'Trade_Openness:Q']
        ).properties(
            width=600,
            height=400,
            title="Trade Vulnerability Matrix"
        )
        
        st.altair_chart(scatter, use_container_width=True)
    
    if shock_results is not None and not shock_results.empty:
        st.subheader("China Export Shock Simulation")
        
        shock_chart = alt.Chart(shock_results).mark_bar().encode(
            x=alt.X('Country:N', title='Country'),
            y=alt.Y('GDP_Loss_Percentage:Q', title='GDP Loss (%)'),
            color=alt.Color('GDP_Loss_Percentage:Q',
                            scale=alt.Scale(scheme='reds'),
                            title='Impact Severity'),
            tooltip=['Country:N', 'GDP_Loss_Percentage:Q', 'GDP_Loss_USD:Q']
        ).properties(
            width=600,
            height=300,
            title=f"Estimated GDP Impact from {shock_results.iloc[0]['GDP_Loss_Percentage']:.1f}% China Export Drop"
        )
        
        st.altair_chart(shock_chart, use_container_width=True)

def main():
    st.title('China Export Shock: Global Trade Impact Analysis')
    st.caption("Analyzing the global impact of a hypothetical drop in China's exports")
 
    with st.spinner("Loading trade data..."):
        df = load_trade_data()
  
    if df.empty:
        st.error("Failed to load data. Please ensure the dataset is available and try again.")
        st.stop()
    
    st.sidebar.success(f"Loaded {len(df)} records for {df['Country'].nunique()} countries")

    column_mapping = resolve_trade_columns(df)

    st.sidebar.subheader("Analysis Parameters")
    latest_data_only = st.sidebar.checkbox("Use latest year data only", value=True)
    shock_percentage = st.sidebar.slider(
        "China Export Drop (%)",
        min_value=10, max_value=50, value=25, step=5
    ) / 100
    
    st.header("Vulnerability Index (VI) Analysis")
    
    with st.spinner("Computing Vulnerability Index..."):
        vi_results = compute_vulnerability_index(df, column_mapping, latest_data_only)
    
    if vi_results.empty:
        st.error("Could not compute Vulnerability Index. Please check your data.")
        st.stop()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Countries Analyzed", len(vi_results))
    
    with col2:
        high_risk = len(vi_results[vi_results['Risk_Category'] == 'High'])
        st.metric("High Risk Countries", high_risk, delta=f"{high_risk/len(vi_results)*100:.1f}%")
    
    with col3:
        avg_vi = vi_results['Vulnerability_Index'].mean()
        st.metric("Average VI Score", f"{avg_vi:.1f}")
    
    with col4:
        max_import_dependency = vi_results['Import_Dependency'].max()
        st.metric("Max Import Dependency", f"{max_import_dependency:.1f}%")
    
    st.subheader("Top 3 Most Vulnerable Countries")
    
    top_3 = vi_results.head(3)
    
    for idx, (_, country_data) in enumerate(top_3.iterrows(), 1):
        with st.container(border=True):
            st.markdown(f"### #{idx} {country_data['Country']}")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("VI Score", f"{country_data['Vulnerability_Index']:.1f}")
                st.metric("Risk Level", country_data['Risk_Category'])
            
            with c2:
                st.metric("Import Dependency", f"{country_data['Import_Dependency']:.1f}%")
            
            with c3:
                st.metric("Trade Openness", f"{country_data['Trade_Openness']:.1f}%")
    
    st.subheader("Complete Rankings")
    
    display_cols = ['Country', 'Vulnerability_Index', 'Risk_Category',
                   'Import_Dependency', 'Trade_Openness']
    
    formatted_results = vi_results[display_cols].copy()
    formatted_results['Import_Dependency'] = formatted_results['Import_Dependency'].apply(lambda x: f"{x:.1f}%")
    formatted_results['Trade_Openness'] = formatted_results['Trade_Openness'].apply(lambda x: f"{x:.1f}%")
    formatted_results['Vulnerability_Index'] = formatted_results['Vulnerability_Index'].apply(lambda x: f"{x:.1f}")
    
    st.dataframe(
        formatted_results,
        column_config={
            "Country": "Country",
            "Vulnerability_Index": "VI Score",
            "Risk_Category": "Risk Level",
            "Import_Dependency": "Imports/GDP",
            "Trade_Openness": "Trade/GDP"
        },
        use_container_width=True
    )
    
    st.header("China Export Shock Simulation")
    st.markdown("Simulate the GDP impact of a China export drop")
    
    available_countries = vi_results['Country'].tolist()
    default_selection = top_3['Country'].tolist()
    
    selected_countries = st.multiselect(
        "Select countries to simulate export shock:",
        available_countries,
        default=default_selection,
        help="Countries to include in the China export shock simulation"
    )
    
    if selected_countries:
        with st.spinner("Running China export shock simulation..."):
            shock_results = simulate_china_export_shock(
                df, vi_results[vi_results['Country'].isin(selected_countries)], 
                column_mapping, shock_percentage
            )
        
        if not shock_results.empty:
            st.subheader(f"Impact of {shock_percentage*100:.0f}% China Export Drop")
            
            avg_impact = shock_results['GDP_Loss_Percentage'].mean()
            worst_impact = shock_results['GDP_Loss_Percentage'].max()
            worst_country = shock_results.loc[shock_results['GDP_Loss_Percentage'].idxmax(), 'Country']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average GDP Impact", f"{avg_impact:.2f}%")
            with col2:
                st.metric("Worst GDP Impact", f"{worst_impact:.2f}%")
            with col3:
                st.metric("Most Affected", worst_country)
            
            for _, result in shock_results.iterrows():
                with st.container(border=True):
                    st.markdown(f"### {result['Country']}")
                    
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        st.metric(
                            "GDP Loss",
                            f"{result['GDP_Loss_Percentage']:.2f}%",
                            f"${result['GDP_Loss_USD']/1e9:.2f}B"
                        )
                    with c2:
                        st.metric("Multiplier", f"{result['Multiplier']:.1f}")
                    with c3:
                        st.metric("Recovery Time", f"{result['Recovery_Years']:.0f} years")
                    with c4:
                        st.metric("VI Score", f"{result['Vulnerability_Index']:.1f}")
    
    create_shock_visualizations(vi_results, shock_results if 'shock_results' in locals() else None)
    
    # Download results
    if 'shock_results' in locals() and not shock_results.empty:
        st.subheader("Download Results")
        csv_data = shock_results.to_csv(index=False)
        st.download_button(
            label="Download Full Impact Analysis (CSV)",
            data=csv_data,
            file_name=f"china_export_shock_impact_{shock_percentage*100:.0f}percent.csv",
            mime="text/csv"
        )
    
    with st.expander("Methodology & Assumptions", expanded=False):
        st.markdown("""
        ### Vulnerability Index (VI) Calculation
        
        **Components:**
        - Trade Openness: Trade as % of GDP
        - Import Dependency: Imports as % of GDP
        - Export Reliance: Exports as % of GDP
        - Normalized to 0-100 scale using StandardScaler
        
        **Risk Categories:**
        - **High Risk**: VI > 75 (High trade dependency)
        - **Medium Risk**: VI 50-75 (Moderate trade dependency)
        - **Low Risk**: VI < 50 (Low trade dependency)
        
        ### China Export Shock Simulation Assumptions
        
        1. **Direct Impact on China:** GDP loss = Export drop × Export/GDP ratio × 0.8 (pass-through)
        2. **Indirect Impact on Others:**
           - Trade dependency factor based on import dependency (max 30% trade with China)
           - Direct impact: China trade share × shock × 0.6 (pass-through)
           - Indirect impact: Vulnerability score × 0.02 × shock
           - Multiplier: 1.0-2.0 based on import dependency
        3. **Recovery**: Estimated based on impact severity (1-5 years)
        
        ### Data Sources & Limitations
        
        - Uses aggregate trade metrics from dataset
        - Assumes max 30% trade share with China
        - No specific bilateral trade data
        - Simplified pass-through rates and static model
        """)

if __name__ == "__main__":
    main()