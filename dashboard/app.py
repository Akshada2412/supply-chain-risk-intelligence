import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Supply Chain Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    /* Hero */
    .hero {
        padding: 2rem 2rem 1.5rem 2rem;
        margin: -1rem -2rem 1.5rem -2rem;
        background: linear-gradient(
            135deg,
            #0a0e1a 0%,
            #0d1b2e 40%,
            #111827 100%
        );
        border-bottom: 1px solid rgba(245,158,11,0.25);
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 300px; height: 300px;
        background: radial-gradient(
            circle,
            rgba(245,158,11,0.06) 0%,
            transparent 70%
        );
        pointer-events: none;
    }
    .hero-tag {
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #F59E0B;
        background: rgba(245,158,11,0.1);
        padding: 3px 12px;
        border-radius: 20px;
        border: 1px solid rgba(245,158,11,0.2);
        margin-bottom: 0.75rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -1px;
        line-height: 1.05;
        margin: 0 0 0.5rem 0;
    }
    .hero-title span { color: #F59E0B; }
    .hero-sub {
        font-size: 0.88rem;
        color: rgba(255,255,255,0.45);
        margin: 0;
        line-height: 1.6;
    }
    .hero-pills {
        display: flex;
        gap: 8px;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    .pill {
        font-size: 0.72rem;
        font-weight: 500;
        color: rgba(255,255,255,0.55);
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 3px 12px;
        border-radius: 20px;
    }
    .pill-alert {
        font-size: 0.72rem;
        font-weight: 600;
        color: #F59E0B;
        background: rgba(245,158,11,0.1);
        border: 1px solid rgba(245,158,11,0.25);
        padding: 3px 12px;
        border-radius: 20px;
    }
    /* Section labels */
    .section-label {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #F59E0B;
        opacity: 0.7;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(245,158,11,0.15);
    }
    /* Chart titles */
    .chart-title {
        font-size: 0.85rem;
        font-weight: 600;
        opacity: 0.85;
        margin-bottom: 0.25rem;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10,14,26,0.6);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }
    .sidebar-stat {
        background: rgba(245,158,11,0.07);
        border: 1px solid rgba(245,158,11,0.15);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 8px;
    }
    .sidebar-stat-label {
        font-size: 0.68rem;
        opacity: 0.5;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .sidebar-stat-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #F59E0B;
        line-height: 1.2;
    }
    .sidebar-stat-sub {
        font-size: 0.7rem;
        opacity: 0.4;
    }
    .sidebar-stat-danger {
        background: rgba(239,68,68,0.07);
        border: 1px solid rgba(239,68,68,0.2);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 8px;
    }
    .sidebar-stat-danger .sidebar-stat-value {
        color: #EF4444;
    }
    /* Alert panel */
    .alert-row {
        background: rgba(239,68,68,0.06);
        border: 1px solid rgba(239,68,68,0.15);
        border-left: 3px solid #EF4444;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 1rem;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .modebar { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    df = pd.read_csv('data/raw/trade_raw.csv')
    return df

def compute_risk(filtered_df):
    global_totals = (
        filtered_df.groupby('product_category')
        ['export_value_usd'].sum()
        .reset_index()
    )
    global_totals.columns = [
        'product_category', 'global_export_total'
    ]

    country_shares = (
        filtered_df.groupby(
            ['country', 'product_category',
             'geo_risk_score']
        )['export_value_usd'].sum()
        .reset_index()
    )
    country_shares = country_shares.merge(
        global_totals, on='product_category'
    )
    country_shares['market_share_pct'] = (
        country_shares['export_value_usd'] * 100.0 /
        country_shares['global_export_total']
    )
    country_shares['share_squared'] = (
        country_shares['market_share_pct'] ** 2
    )

    hhi = (
        country_shares.groupby('product_category')
        .agg(
            hhi_score=('share_squared', 'sum'),
            top_supplier=('country', 'first'),
            top_share_pct=('market_share_pct', 'first'),
            geo_risk_score=('geo_risk_score', 'mean')
        ).reset_index()
    )
    hhi['geo_risk_score'] = hhi['geo_risk_score'].round(2)
    hhi['hhi_score'] = hhi['hhi_score'].round(1)
    hhi['hhi_normalized'] = (
        hhi['hhi_score'] / 10000 * 10
    ).round(2)
    hhi['combined_risk_score'] = (
        (hhi['hhi_normalized'] * 0.6) +
        (hhi['geo_risk_score'] * 0.4)
    ).round(2)

    return hhi

df = load_data()

def get_conn(df):
    conn = sqlite3.connect(':memory:')
    df.to_sql('trade_flows', conn,
              if_exists='replace', index=False)
    return conn

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown(
        "<p style='font-size:0.75rem;font-weight:700;"
        "letter-spacing:0.1em;text-transform:uppercase;"
        "color:#F59E0B;opacity:0.7;margin-bottom:1rem;'>"
        "Intelligence Filters</p>",
        unsafe_allow_html=True
    )

    selected_year = st.selectbox(
        "Reference Year",
        options=sorted(df['year'].unique(), reverse=True),
        index=0
    )

    selected_products = st.multiselect(
        "Product Categories",
        options=sorted(df['product_category'].unique()),
        default=sorted(df['product_category'].unique())
    )

    selected_regions = st.multiselect(
        "Regions",
        options=sorted(df['region'].unique()),
        default=sorted(df['region'].unique())
    )

    risk_min = st.slider(
        "Min Geo Risk Score",
        min_value=1, max_value=10, value=1,
        help="Filter countries with geo risk score at or above this value"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.75rem;font-weight:700;"
        "letter-spacing:0.1em;text-transform:uppercase;"
        "color:#F59E0B;opacity:0.7;margin-bottom:0.75rem;'>"
        "Live Intelligence</p>",
        unsafe_allow_html=True
    )

    filtered_temp = df[
        (df['year'] == selected_year) &
        (df['product_category'].isin(selected_products)) &
        (df['region'].isin(selected_regions))
    ]
    high_risk_c = filtered_temp[
        filtered_temp['geo_risk_score'] >= risk_min
    ]['country'].nunique()

    total_trade = (
        filtered_temp['import_value_usd'].sum() / 1e12
    )
    high_risk_c = filtered_temp[
        filtered_temp['geo_risk_score'] >= 7
    ]['country'].nunique()
    countries_n = filtered_temp['country'].nunique()

    st.markdown(f"""
    <div class='sidebar-stat'>
        <div class='sidebar-stat-label'>
            Total Import Volume
        </div>
        <div class='sidebar-stat-value'>
            ${total_trade:.2f}T
        </div>
        <div class='sidebar-stat-sub'>
            filtered selection · {selected_year}
        </div>
    </div>
    <div class='sidebar-stat'>
        <div class='sidebar-stat-label'>
            Countries Monitored
        </div>
        <div class='sidebar-stat-value'>
            {countries_n}
        </div>
        <div class='sidebar-stat-sub'>
            in selected scope
        </div>
    </div>
    <div class='sidebar-stat-danger'>
        <div class='sidebar-stat-label'>
            High Risk Exposure
        </div>
        <div class='sidebar-stat-value sidebar-stat-danger'>
            {high_risk_c} countries
        </div>
        <div class='sidebar-stat-sub'>
            geo risk score ≥ 7
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.68rem;opacity:0.3;"
        "line-height:1.7;'>"
        "Built by Akshada Karade<br>"
        "MS Eng. Mgmt · UMass Amherst<br>"
        "Python · SQL · Streamlit · HHI Methodology"
        "</p>",
        unsafe_allow_html=True
    )

# ---- FILTER ----
filtered = df[
    (df['year'] == selected_year) &
    (df['product_category'].isin(selected_products)) &
    (df['region'].isin(selected_regions)) &
    (df['geo_risk_score'] >= risk_min)
]

conn = get_conn(filtered)
risk_df = compute_risk(filtered)

# ---- HERO ----
high_risk_countries = filtered[
    filtered['geo_risk_score'] >= 7
]['country'].nunique()

st.markdown(f"""
<div class='hero'>
    <div class='hero-tag'>
        Global Trade Risk Intelligence
    </div>
    <p style='font-size:3rem;font-weight:800;
              color:#ffffff;letter-spacing:-1px;
              line-height:1.05;margin:0 0 0.5rem 0;'>
        Supply Chain<br>
        <span style='color:#F59E0B;'>Risk Monitor</span>
    </p>
    <p class='hero-sub'>
        HHI-powered supplier concentration analysis
        across {filtered['country'].nunique()} countries
        and {len(selected_products)} product categories
        — built to surface the risks COVID-19 exposed
    </p>
    <div class='hero-pills'>
        <span class='pill'>2018 – 2022 Trade Flows</span>
        <span class='pill'>
            {len(selected_products)} Product Categories
        </span>
        <span class='pill'>UN Comtrade · World Bank</span>
        <span class='pill'>HHI Methodology (DOJ Standard)</span>
        <span class='pill-alert'>
            ⚠ {high_risk_countries} High Risk Countries
            Active
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---- KPI ROW ----
k1, k2, k3, k4, k5 = st.columns(5)

total_imports = filtered['import_value_usd'].sum()
total_exports = filtered['export_value_usd'].sum()
avg_risk = filtered['geo_risk_score'].mean()
top_category = risk_df.loc[
    risk_df['combined_risk_score'].idxmax(),
    'product_category'
]
top_risk_score = risk_df['combined_risk_score'].max()

k1.metric(
    "Total Import Volume",
    f"${total_imports/1e12:.2f}T"
)
k2.metric(
    "Total Export Volume",
    f"${total_exports/1e12:.2f}T"
)
k3.metric(
    "Avg Geo Risk Score",
    f"{avg_risk:.1f} / 10"
)
k4.metric(
    "High Risk Countries",
    f"{high_risk_countries}",
    delta="geo risk ≥ 7",
    delta_color="inverse"
)
k5.metric(
    "Highest Risk Category",
    top_category,
    delta=f"score {top_risk_score:.2f}",
    delta_color="inverse"
)

st.markdown("---")

# ---- TRADE FLOWS ----
st.markdown(
    "<p class='section-label'>"
    "Global Trade Flow Analysis</p>",
    unsafe_allow_html=True
)

t1, t2 = st.columns([3, 2])

with t1:
    st.markdown(
        "<p class='chart-title'>"
        "Top 15 Importing Countries</p>",
        unsafe_allow_html=True
    )
    top_imp = (
        filtered.groupby('country')['import_value_usd']
        .sum().reset_index()
        .sort_values('import_value_usd', ascending=False)
        .head(15)
    )
    top_imp['imports_b'] = (
        top_imp['import_value_usd'] / 1e9
    ).round(1)

    # Color by risk
    risk_map = (
        filtered.groupby('country')['geo_risk_score']
        .first().to_dict()
    )
    top_imp['risk'] = (
        top_imp['country'].map(risk_map)
    )

    fig1 = px.bar(
        top_imp.sort_values('imports_b', ascending=True),
        x='imports_b', y='country',
        orientation='h',
        color='risk',
        color_continuous_scale=[
            '#1e3a5f', '#F59E0B', '#EF4444'
        ],
        range_color=[1, 10],
        text='imports_b',
        hover_data={'risk': True, 'imports_b': ':.1f'},
        custom_data=['country', 'risk']
    )
    fig1.update_traces(
        textposition='outside',
        texttemplate='$%{x:.0f}B',
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Imports: <b>$%{x:.1f}B</b><br>"
            "Geo Risk Score: <b>%{customdata[1]}/10</b><br>"
            "<i>Higher risk = darker orange/red</i>"
            "<extra></extra>"
        )
    )
    fig1.update_layout(
        height=420,
        margin=dict(l=0, r=50, t=5, b=0),
        coloraxis_showscale=True,
        coloraxis_colorbar=dict(
            title="Risk",
            thickness=10,
            len=0.6
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title='Import Value (USD Billion)',
            showgrid=False
        ),
        yaxis=dict(title='')
    )
    st.plotly_chart(fig1, use_container_width=True)

with t2:
    st.markdown(
        "<p class='chart-title'>"
        "Import Share by Region</p>",
        unsafe_allow_html=True
    )
    reg = (
        filtered.groupby('region')['import_value_usd']
        .sum().reset_index()
        .sort_values('import_value_usd', ascending=False)
    )
    fig2 = px.pie(
        reg,
        values='import_value_usd',
        names='region',
        hole=0.6,
        color_discrete_sequence=[
            '#F59E0B', '#1e3a5f', '#EF4444',
            '#374151', '#92400E', '#1D4ED8',
            '#065F46', '#7C3AED'
        ]
    )
    fig2.update_traces(
        textinfo='percent',
        textposition='outside',
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Imports: <b>$%{value:,.0f}</b><br>"
            "Share: <b>%{percent}</b>"
            "<extra></extra>"
        )
    )
    fig2.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=5, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation='v',
            x=1.0, y=0.5,
            title=None
        )
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ---- RISK SCORING ----
st.markdown(
    "<p class='section-label'>"
    "HHI Concentration Risk Scoring</p>",
    unsafe_allow_html=True
)

r1, r2 = st.columns(2)

with r1:
    st.markdown(
        "<p class='chart-title'>"
        "Combined Risk Score by Product Category</p>",
        unsafe_allow_html=True
    )
    fig3 = px.bar(
        risk_df.sort_values(
            'combined_risk_score', ascending=True
        ),
        x='combined_risk_score',
        y='product_category',
        orientation='h',
        color='combined_risk_score',
        color_continuous_scale=[
            '#1e3a5f', '#F59E0B', '#EF4444'
        ],
        text='combined_risk_score',
    )
    fig3.update_traces(
        texttemplate='%{x:.2f}',
        textposition='outside',
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Combined Risk Score: <b>%{x:.2f}</b><br>"
            "<i>HHI concentration × 0.6 + "
            "Geo risk × 0.4</i><br>"
            "<i>Higher = more dangerous dependency</i>"
            "<extra></extra>"
        )
    )
    fig3.update_layout(
        height=320,
        margin=dict(l=0, r=50, t=5, b=0),
        coloraxis_showscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Risk Score', showgrid=False),
        yaxis=dict(title='')
    )
    st.plotly_chart(fig3, use_container_width=True)

with r2:
    st.markdown(
        "<p class='chart-title'>"
        "HHI Score vs Geo Risk — Bubble View</p>"
        "<p style='font-size:0.75rem;opacity:0.45;"
        "margin-top:-0.25rem;'>"
        "Y-axis reflects avg geo risk of filtered "
        "supplier base · updates with filters</p>",
        unsafe_allow_html=True
    )
    fig4 = px.scatter(
        risk_df,
        x='hhi_score',
        y='geo_risk_score',
        size='combined_risk_score',
        color='combined_risk_score',
        text='product_category',
        color_continuous_scale=[
            '#1e3a5f', '#F59E0B', '#EF4444'
        ],
        size_max=25
    )
    fig4.update_traces(
        textposition='top center',
        textfont=dict(size=10),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "HHI Score: <b>%{x:.1f}</b><br>"
            "Geo Risk: <b>%{y}/10</b><br>"
            "Combined Risk: <b>%{marker.color:.2f}</b><br>"
            "<i>Top-right = highest danger zone</i>"
            "<extra></extra>"
        )
    )
    fig4.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=5, b=0),
        coloraxis_showscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title='HHI Concentration Score',
            showgrid=False
        ),
        yaxis=dict(
            title='Geo Risk Score',
            gridcolor='rgba(128,128,128,0.1)'
        )
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ---- COVID + EXPOSURE ----
st.markdown(
    "<p class='section-label'>"
    "Trade Disruption & Exposure Analysis</p>",
    unsafe_allow_html=True
)

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        "<p class='chart-title'>"
        "Global Trade Volume Trend — COVID Impact</p>",
        unsafe_allow_html=True
    )
    yearly = (
        df.groupby('year')
        .agg(
            imports=('import_value_usd', 'sum'),
            exports=('export_value_usd', 'sum')
        ).reset_index()
    )
    yearly['imports_t'] = yearly['imports'] / 1e12
    yearly['exports_t'] = yearly['exports'] / 1e12

    fig5 = go.Figure()
    fig5.add_vrect(
        x0=2019.5, x1=2020.5,
        fillcolor='rgba(239,68,68,0.08)',
        line_width=0,
        annotation_text="COVID-19 Shock",
        annotation_position="top left",
        annotation_font_color="#EF4444",
        annotation_font_size=11
    )
    fig5.add_trace(go.Scatter(
        x=yearly['year'],
        y=yearly['imports_t'],
        name='Imports',
        mode='lines+markers',
        line=dict(color='#F59E0B', width=3),
        marker=dict(size=8),
        hovertemplate=(
            "Year: <b>%{x}</b><br>"
            "Global Imports: <b>$%{y:.2f}T</b>"
            "<extra>Imports</extra>"
        )
    ))
    fig5.add_trace(go.Scatter(
        x=yearly['year'],
        y=yearly['exports_t'],
        name='Exports',
        mode='lines+markers',
        line=dict(
            color='#60A5FA', width=3,
            dash='dash'
        ),
        marker=dict(size=8),
        hovertemplate=(
            "Year: <b>%{x}</b><br>"
            "Global Exports: <b>$%{y:.2f}T</b>"
            "<extra>Exports</extra>"
        )
    ))
    fig5.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=5, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Year', showgrid=False),
        yaxis=dict(
            title='Trade Value (USD Trillion)',
            gridcolor='rgba(128,128,128,0.1)'
        ),
        legend=dict(
            orientation='h', y=1.12, title=None
        )
    )
    st.plotly_chart(fig5, use_container_width=True)

with c2:
    st.markdown(
        "<p class='chart-title'>"
        "Geo Risk vs Import Dependency</p>",
        unsafe_allow_html=True
    )
    scatter = (
        filtered.groupby(
            ['country', 'geo_risk_score', 'region']
        )
        .agg(
            total_imports=('import_value_usd', 'sum')
        ).reset_index()
    )
    scatter['imports_b'] = (
        scatter['total_imports'] / 1e9
    ).round(1)

    fig6 = px.scatter(
        scatter,
        x='geo_risk_score',
        y='imports_b',
        color='region',
        size='imports_b',
        hover_name='country',
        size_max=30,
        color_discrete_sequence=[
            '#F59E0B', '#60A5FA', '#34D399',
            '#F87171', '#A78BFA',
            '#FBBF24', '#6EE7B7', '#93C5FD'
        ]
    )
    fig6.update_traces(
        opacity=0.75,
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Geo Risk: <b>%{x}/10</b><br>"
            "Imports: <b>$%{y:.1f}B</b><br>"
            "<i>Top-right = most dangerous dependency</i>"
            "<extra></extra>"
        )
    )
    # Danger zone annotation
    fig6.add_shape(
        type='rect',
        x0=7, x1=10.5,
        y0=0, y1=scatter['imports_b'].max() * 1.1,
        fillcolor='rgba(239,68,68,0.05)',
        line=dict(
            color='rgba(239,68,68,0.2)',
            dash='dot'
        )
    )
    fig6.add_annotation(
        x=8.5,
        y=scatter['imports_b'].max() * 1.05,
        text="⚠ Danger Zone",
        showarrow=False,
        font=dict(color='#EF4444', size=11)
    )
    fig6.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=5, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title='Geopolitical Risk Score',
            showgrid=False
        ),
        yaxis=dict(
            title='Total Imports (USD Billion)',
            gridcolor='rgba(128,128,128,0.1)'
        ),
        legend=dict(
            title=None,
            orientation='v',
            x=1.0
        )
    )
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

# ---- HIGH RISK ALERT PANEL ----
st.markdown(
    "<p class='section-label'>"
    "High Risk Supplier Alert Panel</p>",
    unsafe_allow_html=True
)

alert_data = (
    filtered[filtered['geo_risk_score'] >= 7]
    .groupby(['country', 'region', 'geo_risk_score'])
    .agg(
        total_imports=('import_value_usd', 'sum'),
        total_exports=('export_value_usd', 'sum')
    ).reset_index()
    .sort_values('total_imports', ascending=False)
)
alert_data['imports_b'] = (
    alert_data['total_imports'] / 1e9
).round(1)
alert_data['trade_balance_b'] = (
    (alert_data['total_exports'] -
     alert_data['total_imports']) / 1e9
).round(1)

alert_data = alert_data[[
    'country', 'region', 'geo_risk_score',
    'imports_b', 'trade_balance_b'
]]
alert_data.columns = [
    'Country', 'Region', 'Geo Risk Score',
    'Imports ($B)', 'Trade Balance ($B)'
]

def color_risk(val):
    if val >= 9:
        return 'background-color:rgba(239,68,68,0.2);color:#EF4444'
    elif val >= 7:
        return 'background-color:rgba(245,158,11,0.15);color:#F59E0B'
    return ''

st.markdown(
    f"Showing **{len(alert_data)} countries** "
    f"with geo risk score ≥ 7 — "
    f"representing active supply chain exposure"
)

st.dataframe(
    alert_data.style.map(
        color_risk, subset=['Geo Risk Score']
    ),
    use_container_width=True,
    height=320
)

st.markdown("---")

# ---- FOOTER ----
st.markdown(
    "<p style='font-size:0.78rem;opacity:0.3;"
    "line-height:1.7;'>"
    "Akshada Karade &nbsp;·&nbsp; "
    "MS Engineering Management, UMass Amherst "
    "&nbsp;·&nbsp; "
    "Python · SQL · Streamlit · Plotly · "
    "HHI Methodology (DOJ Standard) · "
    "Data: UN Comtrade · World Bank WITS"
    "</p>",
    unsafe_allow_html=True
)