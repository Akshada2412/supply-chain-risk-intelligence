import pandas as pd
import sqlite3

# Load data
conn = sqlite3.connect('data/processed/supply_chain.db')

# ============================================================
# HHI RISK SCORE — Herfindahl-Hirschman Index
#
# What is HHI?
# Used by US Dept of Justice to measure market concentration.
# We use it to measure how concentrated a country's imports
# are — i.e. how dependent they are on few suppliers.
#
# Formula: HHI = SUM of (each supplier's market share)^2
# Score 0-10000:
#   Below 1500 = low concentration (safe)
#   1500-2500  = moderate concentration (caution)
#   Above 2500 = high concentration (danger)
# ============================================================

print("=" * 55)
print("SUPPLY CHAIN RISK SCORING ENGINE")
print("Using Herfindahl-Hirschman Index (HHI) Methodology")
print("=" * 55)
print()

# STEP 1 — Calculate each country's share of global exports
# per product category (they are the "suppliers")
q_shares = """
WITH global_totals AS (
    SELECT
        product_category,
        SUM(export_value_usd) AS global_export_total
    FROM trade_flows
    WHERE year = 2022
    GROUP BY product_category
),
country_shares AS (
    SELECT
        t.country,
        t.product_category,
        t.geo_risk_score,
        SUM(t.export_value_usd) AS country_exports,
        g.global_export_total,
        ROUND(SUM(t.export_value_usd) * 100.0 
              / g.global_export_total, 4) AS market_share_pct
    FROM trade_flows t
    JOIN global_totals g 
        ON t.product_category = g.product_category
    WHERE t.year = 2022
    GROUP BY t.country, t.product_category, 
             t.geo_risk_score, g.global_export_total
)
SELECT * FROM country_shares
ORDER BY product_category, market_share_pct DESC
"""

df_shares = pd.read_sql_query(q_shares, conn)

print("STEP 1 — Top suppliers per product category:")
top = df_shares.groupby('product_category').head(3)
print(top[['product_category','country',
           'market_share_pct']].to_string(index=False))
print()

# STEP 2 — Calculate HHI per product category
print("STEP 2 — HHI Concentration Score per category:")

df_shares['share_squared'] = (df_shares['market_share_pct'] ** 2)

hhi = df_shares.groupby('product_category').agg(
    hhi_score=('share_squared', 'sum'),
    top_supplier=('country', 'first'),
    top_share_pct=('market_share_pct', 'first')
).reset_index()

hhi['hhi_score'] = hhi['hhi_score'].round(1)

def risk_label(score):
    if score < 1500:
        return 'LOW'
    elif score < 2500:
        return 'MODERATE'
    else:
        return 'HIGH'

hhi['concentration_risk'] = hhi['hhi_score'].apply(risk_label)
hhi = hhi.sort_values('hhi_score', ascending=False)

print(hhi.to_string(index=False))
print()

# STEP 3 — Combined Risk Score
# HHI + Geo Risk of top supplier = Final Risk Score
print("STEP 3 — Final Combined Risk Score:")

df_top_supplier = df_shares.loc[
    df_shares.groupby('product_category')['market_share_pct'].idxmax()
][['product_category', 'country', 'geo_risk_score', 'market_share_pct']]

hhi_final = hhi.merge(
    df_top_supplier[['product_category', 'geo_risk_score']],
    on='product_category'
)

# Normalize HHI to 0-10 scale and combine with geo risk
hhi_final['hhi_normalized'] = (
    hhi_final['hhi_score'] / 10000 * 10
).round(2)

hhi_final['combined_risk_score'] = (
    (hhi_final['hhi_normalized'] * 0.6) +
    (hhi_final['geo_risk_score'] * 0.4)
).round(2)

hhi_final = hhi_final.sort_values(
    'combined_risk_score', ascending=False
)

print(hhi_final[[
    'product_category',
    'top_supplier',
    'top_share_pct',
    'hhi_score',
    'geo_risk_score',
    'combined_risk_score',
    'concentration_risk'
]].to_string(index=False))

# Save results
hhi_final.to_csv(
    'data/processed/risk_scores.csv', index=False
)
print()
print("✅ Risk scores saved to data/processed/risk_scores.csv")