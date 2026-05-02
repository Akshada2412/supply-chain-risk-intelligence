import pandas as pd
import numpy as np
import os

np.random.seed(42)

countries = [
    ("United States", "North America", 2, 9),
    ("China", "Asia", 7, 8),
    ("Germany", "Europe", 2, 9),
    ("Japan", "Asia", 2, 8),
    ("United Kingdom", "Europe", 2, 9),
    ("France", "Europe", 2, 9),
    ("Netherlands", "Europe", 2, 10),
    ("South Korea", "Asia", 4, 8),
    ("India", "Asia", 4, 7),
    ("Canada", "North America", 2, 9),
    ("Italy", "Europe", 3, 8),
    ("Mexico", "North America", 5, 7),
    ("Singapore", "Asia", 2, 10),
    ("Australia", "Oceania", 2, 8),
    ("Spain", "Europe", 2, 8),
    ("Brazil", "South America", 5, 6),
    ("Poland", "Europe", 3, 8),
    ("Czechia", "Europe", 2, 8),
    ("Thailand", "Asia", 4, 7),
    ("Malaysia", "Asia", 3, 8),
    ("Vietnam", "Asia", 4, 7),
    ("Indonesia", "Asia", 4, 6),
    ("Turkey", "Europe/Asia", 6, 6),
    ("Saudi Arabia", "Middle East", 6, 5),
    ("UAE", "Middle East", 4, 8),
    ("South Africa", "Africa", 5, 6),
    ("Nigeria", "Africa", 7, 4),
    ("Egypt", "Africa", 6, 5),
    ("Argentina", "South America", 6, 5),
    ("Chile", "South America", 3, 7),
    ("Taiwan", "Asia", 3, 9),
    ("Hong Kong", "Asia", 4, 10),
    ("Israel", "Middle East", 5, 8),
    ("Sweden", "Europe", 2, 9),
    ("Norway", "Europe", 2, 9),
    ("Denmark", "Europe", 2, 9),
    ("Finland", "Europe", 2, 9),
    ("Austria", "Europe", 2, 9),
    ("Belgium", "Europe", 2, 9),
    ("Switzerland", "Europe", 2, 9),
    ("Ukraine", "Europe", 9, 4),
    ("Russia", "Europe/Asia", 9, 3),
    ("Iran", "Middle East", 9, 2),
    ("Iraq", "Middle East", 8, 3),
    ("Venezuela", "South America", 9, 2),
    ("Afghanistan", "Asia", 10, 1),
    ("Syria", "Middle East", 10, 1),
    ("Yemen", "Middle East", 10, 1),
    ("Philippines", "Asia", 4, 6),
    ("Pakistan", "Asia", 7, 4),
]

products = [
    ("Electronics", "8471", 0.35),
    ("Pharmaceuticals", "3004", 0.20),
    ("Auto Parts", "8708", 0.18),
    ("Semiconductors", "8542", 0.15),
    ("Raw Materials", "2601", 0.12),
]

years = [2018, 2019, 2020, 2021, 2022]

base_imports = {
    "United States": 380, "China": 270, "Germany": 150,
    "Japan": 125, "United Kingdom": 115, "France": 95,
    "Netherlands": 85, "South Korea": 80, "India": 68,
    "Canada": 65, "Italy": 57, "Mexico": 52,
    "Singapore": 47, "Australia": 41, "Spain": 36,
    "Brazil": 33, "Poland": 29, "Czechia": 26,
    "Thailand": 24, "Malaysia": 22, "Vietnam": 20,
    "Indonesia": 18, "Turkey": 17, "Saudi Arabia": 15,
    "UAE": 14, "South Africa": 12, "Taiwan": 19,
    "Hong Kong": 28, "Israel": 11, "Sweden": 13,
}

rows = []
for country, region, geo_risk, trade_openness in countries:
    base = base_imports.get(country, np.random.uniform(0.5, 8))
    for year in years:
        year_multiplier = {
            2018: 0.88,
            2019: 0.94,
            2020: 0.78,
            2021: 0.95,
            2022: 1.0
        }[year]
        for product, hs_code, product_share in products:
            noise = np.random.uniform(0.85, 1.15)
            import_val = base * 1e9 * product_share * year_multiplier * noise
            export_val = import_val * np.random.uniform(0.3, 1.8) * (trade_openness / 10)
            rows.append({
                "country": country,
                "region": region,
                "geo_risk_score": geo_risk,
                "trade_openness_score": trade_openness,
                "year": year,
                "product_category": product,
                "hs_code": hs_code,
                "import_value_usd": round(import_val, 2),
                "export_value_usd": round(export_val, 2),
            })

df = pd.DataFrame(rows)
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/trade_raw.csv", index=False)

print("Dataset generated!")
print("Shape:", df.shape)
print("Countries:", df["country"].nunique())
print("Products:", df["product_category"].nunique())
print("Years:", sorted(df["year"].unique()))