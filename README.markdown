Trade Resilience & Economic Networks (TREN) Dataset Project
Overview
The TREN Dataset Project analyzes economic, trade, demographic, agricultural, and welfare data (2000–2024) for 25 nations to predict 2030 economic outcomes under various crisis and policy scenarios. It provides data-driven strategies to enhance resilience against global recessions, trade disputes, climate disasters, pandemics, and political instability.
Countries
India, USA, Russia, France, Germany, Italy, China, Japan, Argentina, Portugal, Spain, Croatia, Belgium, Australia, Pakistan, Afghanistan, Israel, Iran, Iraq, Bangladesh, Sri Lanka, Canada, UK, Sweden, Saudi Arabia.
Dataset

Trade: Exports/imports (value, volume, partners), 2000–2024.
Economic: GDP, growth, inflation, trade/GDP, spending.
Agriculture: Food production, yields, livestock, trade.
Disasters: Type, severity, damages, lives lost, recovery.
Employment: Ratios, unemployment, labour participation.
Demographics: Population, urbanization, growth, age.
Resilience: Vulnerability, trade diversification, recovery scores.
Welfare: Poverty rates, life expectancy, Gini, HDI.

Tasks

Data Integration: Merge datasets into Country–Year framework, clean data.
Feature Engineering: Create Trade Dependency, Resilience, Spending Efficiency, Shock Impact indexes.
Modeling: Forecast GDP, poverty, trade resilience for 2030 under baseline, social spending, trade diversification, crisis scenarios.
Visualization: Heatmaps, trade networks, shock maps; identify vulnerabilities.
Policy: Recommend strategies, simulate "what-if" scenarios.

File Structure

dashboard/: summary_dashboard.png
data/
processed/: integrated_tren_dataset.csv
raw/: Original datasets


integrated_tren_dataset_with_indexes.csv: Engineered features
heatmaps/, shockmap/, trade_networks/, vulnerabilities/: Visualizations
src/: Scripts for preprocessing, engineering, modeling
app.py: Main application
requirements.txt: Dependencies

Installation

Clone: git clone https://github.com/your-username/tren-dataset-project.git
Navigate: cd tren-dataset-project
Install: pip install -r requirements.txt

Usage

Preprocess: python src/data_preprocessing.py
Engineer features: python src/feature_engineering.py
Run app: python app.py

License
MIT License. See LICENSE.
Contact
your-email@example.com