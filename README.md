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

| Product Category | HHI Score | Top Supplier | Combined Risk |
|-----------------|-----------|--------------|---------------|
| Electronics | 650.5 | China (17.2%) | 3.19 ⚠️ |
| Raw Materials | 1249.6 | USA (29.3%) | 1.55 |
| Semiconductors | 923.6 | USA (22.4%) | 1.35 |
| Auto Parts | 807.4 | USA (17.5%) | 1.29 |
| Pharmaceuticals | 758.4 | USA (14.1%) | 1.26 |

**Electronics is the highest risk category** — China supplies 
17.2% of global electronics exports with a geopolitical 
risk score of 7/10.

**COVID-19 impact** — Global trade collapsed from $1.9T 
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
for antitrust analysis. Applied here to supplier 
concentration per product category.

- HHI < 1500 → Low concentration
- HHI 1500–2500 → Moderate concentration  
- HHI > 2500 → High concentration (danger zone)

**Combined Risk Score** = (HHI normalized × 0.6) + 
(Geopolitical risk score × 0.4)

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


