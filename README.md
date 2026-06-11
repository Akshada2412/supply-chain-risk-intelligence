# ⚠️ Supply Chain Risk Intelligence Platform

> An end-to-end analytics system that scores supplier 
> concentration risk across 50 countries and 5 product 
> categories using HHI methodology — built to solve the 
> $500B supply chain visibility problem exposed by COVID-19.

🔴 **[View Live Dashboard](https://supply-chain-risk-intelligence-gvc7bxaaez4m6tgv4duncf.streamlit.app/)**

---

## The Problem

After COVID-19, companies discovered they had dangerous 
dependency on single countries for critical goods. 
There was no accessible tool for operations teams to 
monitor supplier concentration risk in real time.

## The Solution

A risk intelligence platform that:
- Ingests trade flow data across 50 countries
- Computes **HHI concentration scores** per product category
- Weights scores against geopolitical risk indices
- Surfaces insights in an interactive executive dashboard

---

## Key Findings

| Product Category | HHI Score | Top Supplier | Weighted Geo Risk | Combined Risk |
|-----------------|-----------|--------------|-------------------|---------------|
| Raw Materials | 1249.6 | USA (29.3%) | 3.08/10 | 1.98 ⚠️ |
| Electronics | 650.5 | China (17.2%) | 3.54/10 | 1.81 |
| Auto Parts | 807.4 | USA (17.5%) | 3.18/10 | 1.76 |
| Pharmaceuticals | 758.4 | USA (14.1%) | 3.15/10 | 1.72 |
| Semiconductors | 923.6 | USA (22.4%) | 2.81/10 | 1.68 |

**Raw Materials is the highest combined risk category** — it has
the highest HHI concentration score (1,249.6) and a moderately
elevated share-weighted geo risk (3.08/10) across its supplier base.

**Electronics is the second highest risk** — China (geo risk 7/10)
holds 17.2% of global electronics exports, pulling the
share-weighted geo risk above other categories despite Electronics
having the most diversified supplier base by HHI.

**COVID-19 impact** — Global trade collapsed from $1.91T
(2019) to $1.58T (2020), a $328B single-year drop.

---

## Dashboard Features

- 📊 Top 15 importing countries by trade volume
- ⚠️ HHI risk score by product category  
- 📉 COVID-19 impact trend line (2018–2022)
- 🎯 Geo risk vs import dependency scatter plot
- 🚨 High risk supplier alert panel
- 🔧 Interactive filters — year, product, risk threshold

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data Processing | Python, Pandas, NumPy |
| Database | SQLite, SQL (CTEs, Window Functions) |
| Risk Scoring | HHI Methodology (DOJ Standard) |
| Dashboard | Streamlit, Plotly |
| Version Control | Git, GitHub |

---

## Project Structure

| supply-chain-risk-intelligence/                       |
| ├── data/                                             |
| │   ├── raw/              # Source trade flow data    |
| │   └── processed/        # Risk scores, SQLite DB    |
| ├── notebooks/                                        |
| │   ├── 01_generate_data.py    # Data generation      |
| │   ├── 02_sql_exploration.py  # SQL analysis         |
| │   └── 03_risk_scoring.py     # HHI engine           |
| ├── dashboard/                                        |
| │   └── app.py            # Streamlit application     |
| └── docs/                                             |
| └── executive_memo.md # PM deliverable                |

---

## Methodology

**Herfindahl-Hirschman Index (HHI)** measures market
concentration. Used by the US Department of Justice
for antitrust analysis. Applied here to measure export
concentration per product category — how dependent global
supply is on a small number of exporting countries.

- HHI < 1500 → Low concentration
- HHI 1500–2500 → Moderate concentration
- HHI > 2500 → High concentration (danger zone)

**Share-Weighted Geo Risk** = Σ(export_share_i × geo_risk_i) / 100

Each country's geopolitical risk score is weighted by its
export share — the same shares used in HHI. A country
supplying 25% of global exports contributes 25% to the
geo-risk component; a country supplying 0.5% contributes
0.5%. This ensures methodological consistency with HHI and
avoids distorting the score by treating all suppliers equally
regardless of their actual supply contribution.

**Combined Risk Score** = (HHI normalized × 0.6) +
(Share-weighted geo risk × 0.4)

**Data note:** The dataset is a synthetic model calibrated
against real-world trade volume estimates for 50 major
economies, with year multipliers that reproduce the
~17% COVID-19 trade collapse observed in 2020 WTO data.
The analytical methodology (HHI, share-weighted geo risk,
Holt-Winters forecasting, safety stock) is identical to
what would apply to real trade data from UN Comtrade
or WTO statistics.

---

## PM Governance Layer

This project includes full PM deliverables:
- Executive insight memo with recommendations
- Risk register for product categories
- Stakeholder-ready visualizations

---

## Author

**Akshada Karade**  
MS Engineering Management, UMass Amherst  
[LinkedIn](https://www.linkedin.com/in/akshadakarade2412/) | 
[Email](mailto:akshadakarade@gmail.com)