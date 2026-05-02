import pandas as pd
import sqlite3

# ---- SETUP: Load CSV into SQLite ----
df = pd.read_csv('data/raw/trade_raw.csv')
conn = sqlite3.connect('data/processed/supply_chain.db')
df.to_sql('trade_flows', conn, if_exists='replace', index=False)
print("✅ Database ready!")
print(f"   Total rows loaded: {len(df)}")
print("---")

# ============================================================
# QUERY 1 — Top 10 importing countries (2022, all products)
# LESSON: SELECT, WHERE, GROUP BY, ORDER BY, LIMIT
# ============================================================
q1 = """
SELECT 
    country,
    ROUND(SUM(import_value_usd) / 1000000000, 2) AS total_imports_billion_usd
FROM trade_flows
WHERE year = 2022
GROUP BY country
ORDER BY total_imports_billion_usd DESC
LIMIT 10
"""
print("🌍 TOP 10 IMPORTING COUNTRIES (2022):")
print(pd.read_sql_query(q1, conn).to_string(index=False))
print("---")

# ============================================================
# QUERY 2 — Which product category has highest global imports
# LESSON: GROUP BY on a different column
# ============================================================
q2 = """
SELECT 
    product_category,
    ROUND(SUM(import_value_usd) / 1000000000, 2) AS total_imports_billion_usd,
    COUNT(DISTINCT country) AS countries_importing
FROM trade_flows
WHERE year = 2022
GROUP BY product_category
ORDER BY total_imports_billion_usd DESC
"""
print("📦 IMPORTS BY PRODUCT CATEGORY (2022):")
print(pd.read_sql_query(q2, conn).to_string(index=False))
print("---")

# ============================================================
# QUERY 3 — COVID impact: global trade by year
# LESSON: Grouping by year to see trends
# ============================================================
q3 = """
SELECT 
    year,
    ROUND(SUM(import_value_usd) / 1000000000, 2) AS global_imports_billion_usd,
    ROUND(SUM(export_value_usd) / 1000000000, 2) AS global_exports_billion_usd
FROM trade_flows
GROUP BY year
ORDER BY year ASC
"""
print("📉 GLOBAL TRADE BY YEAR (COVID IMPACT):")
print(pd.read_sql_query(q3, conn).to_string(index=False))
print("---")

# ============================================================
# QUERY 4 — High risk countries with high import dependency
# LESSON: WHERE with multiple conditions using AND
# ============================================================
q4 = """
SELECT 
    country,
    region,
    geo_risk_score,
    ROUND(SUM(import_value_usd) / 1000000000, 2) AS total_imports_billion_usd
FROM trade_flows
WHERE year = 2022
  AND geo_risk_score >= 7
GROUP BY country, region, geo_risk_score
ORDER BY total_imports_billion_usd DESC
"""
print("⚠️  HIGH RISK COUNTRIES BY IMPORT VALUE:")
print(pd.read_sql_query(q4, conn).to_string(index=False))
print("---")

print("✅ All queries complete!")