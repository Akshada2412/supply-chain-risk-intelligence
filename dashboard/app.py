import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import os

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Supply Chain Risk Intelligence",
    page_icon="⚠️",
    layout="wide"
)

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    df = pd.read_csv('data/raw/trade_raw.csv')
    risk_df = pd.read_csv('data/processed/risk_scores.csv')
    return df, risk_df

def get_conn():
    df = pd.read_csv('data/raw/trade_raw.csv')
    conn = sqlite3.connect(':memory:')
    df.to_sql('trade_flows', conn,
              if_exists='replace', index=False)
    return conn

df, risk_df = load_data()
conn = get_conn()

# ---- HEADER ----
st.title("⚠️ Supply Chain Risk Intelligence Platform")
st.markdown(
    "Real-time supplier concentration risk scoring "
    "using **HHI Methodology** across 50 countries "
    "and 5 product categories (2018–2022)"
)
st.divider()

# ---- SIDEBAR FILTERS ----
st.sidebar.header("🔧 Filters")

selected_year = st.sidebar.selectbox(
    "Select Year",
    options=[2018, 2019, 2020, 2021, 2022],
    index=4
)

selected_products = st.sidebar.multiselect(
    "Product Categories",
    options=df['product_category'].unique().tolist(),
    default=df['product_category'].unique().tolist()
)

risk_threshold = st.sidebar.slider(
    "Min Geo Risk Score",
    min_value=1,
    max_value=10,
    value=1
)

# ---- FILTER DATA ----
filtered = df[
    (df['year'] == selected_year) &
    (df['product_category'].isin(selected_products)) &
    (df['geo_risk_score'] >= risk_threshold)
]

# ---- KPI METRICS ROW ----
col1, col2, col3, col4 = st.columns(4)

total_imports = filtered['import_value_usd'].sum()
total_exports = filtered['export_value_usd'].sum()
high_risk = filtered[
    filtered['geo_risk_score'] >= 7
]['country'].nunique()
avg_risk = filtered['geo_risk_score'].mean()

col1.metric(
    "Total Imports",
    f"${total_imports/1e12:.2f}T"
)
col2.metric(
    "Total Exports", 
    f"${total_exports/1e12:.2f}T"
)
col3.metric(
    "High Risk Countries",
    f"{high_risk}"
)
col4.metric(
    "Avg Geo Risk Score",
    f"{avg_risk:.1f}/10"
)

st.divider()

# ---- ROW 1: TOP IMPORTERS + RISK SCORES ----
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🌍 Top 15 Importing Countries")
    top_importers = (
        filtered.groupby('country')['import_value_usd']
        .sum()
        .reset_index()
        .sort_values('import_value_usd', ascending=False)
        .head(15)
    )
    top_importers['imports_billion'] = (
        top_importers['import_value_usd'] / 1e9
    ).round(1)

    fig1 = px.bar(
        top_importers,
        x='imports_billion',
        y='country',
        orientation='h',
        color='imports_billion',
        color_continuous_scale='Blues',
        labels={
            'imports_billion': 'Imports (USD Billion)',
            'country': 'Country'
        }
    )
    fig1.update_layout(
        height=450,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.subheader("⚠️ HHI Risk Score by Product Category")
    fig2 = px.bar(
        risk_df.sort_values('combined_risk_score', 
                            ascending=False),
        x='product_category',
        y='combined_risk_score',
        color='combined_risk_score',
        color_continuous_scale='Reds',
        text='combined_risk_score',
        labels={
            'combined_risk_score': 'Combined Risk Score',
            'product_category': 'Product Category'
        }
    )
    fig2.update_traces(textposition='outside')
    fig2.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ---- ROW 2: COVID TREND + GEO RISK SCATTER ----
col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("📉 Global Trade Trend (COVID Impact)")
    yearly = (
        df.groupby('year')
        .agg(
            imports=('import_value_usd', 'sum'),
            exports=('export_value_usd', 'sum')
        )
        .reset_index()
    )
    yearly['imports_t'] = yearly['imports'] / 1e12
    yearly['exports_t'] = yearly['exports'] / 1e12

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=yearly['year'],
        y=yearly['imports_t'],
        name='Imports',
        mode='lines+markers',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    fig3.add_trace(go.Scatter(
        x=yearly['year'],
        y=yearly['exports_t'],
        name='Exports',
        mode='lines+markers',
        line=dict(color='#ff7f0e', width=3),
        marker=dict(size=8)
    ))
    fig3.add_vrect(
        x0=2019.5, x1=2020.5,
        fillcolor='red',
        opacity=0.1,
        annotation_text="COVID-19",
        annotation_position="top left"
    )
    fig3.update_layout(
        height=400,
        xaxis_title='Year',
        yaxis_title='Trade Value (USD Trillion)',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02
        )
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_right2:
    st.subheader("🎯 Geo Risk vs Import Dependency")
    scatter_data = (
        filtered.groupby(
            ['country', 'geo_risk_score', 'region']
        )
        .agg(total_imports=('import_value_usd', 'sum'))
        .reset_index()
    )
    scatter_data['imports_b'] = (
        scatter_data['total_imports'] / 1e9
    ).round(1)

    fig4 = px.scatter(
        scatter_data,
        x='geo_risk_score',
        y='imports_b',
        color='region',
        size='imports_b',
        hover_name='country',
        labels={
            'geo_risk_score': 'Geopolitical Risk Score',
            'imports_b': 'Total Imports (USD Billion)',
            'region': 'Region'
        }
    )
    fig4.update_layout(height=400)
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ---- ROW 3: HIGH RISK ALERT TABLE ----
st.subheader("🚨 High Risk Supplier Alert Panel")
st.markdown(
    "Countries with **geo risk score ≥ 7** "
    "that represent significant import dependency"
)

alert_data = (
    filtered[filtered['geo_risk_score'] >= 7]
    .groupby(['country', 'region', 'geo_risk_score'])
    .agg(total_imports=('import_value_usd', 'sum'))
    .reset_index()
    .sort_values('total_imports', ascending=False)
)
alert_data['total_imports'] = (
    alert_data['total_imports'] / 1e9
).round(1)
alert_data.columns = [
    'Country', 'Region', 
    'Geo Risk Score', 'Total Imports ($B)'
]

def color_risk(val):
    if val >= 9:
        return 'background-color: #ffcccc'
    elif val >= 7:
        return 'background-color: #fff3cc'
    return ''

st.dataframe(
    alert_data.style.applymap(
        color_risk,
        subset=['Geo Risk Score']
    ),
    use_container_width=True,
    height=300
)

# ---- FOOTER ----
st.divider()
st.markdown(
    "**Built by Akshada Karade** | "
    "MS Engineering Management, UMass Amherst | "
    "Data: World Bank WITS & UN Comtrade | "
    "Methodology: HHI Concentration Scoring"
)