import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

def render_demand_forecasting_tab(df):
    """
    Demand Forecasting & Inventory Optimization tab.
    Accepts the existing trade_raw dataframe.
    """

    st.markdown("""
    <style>
    .df-metric{background:rgba(128,128,128,0.06);border:1px solid rgba(128,128,128,0.15);
               border-radius:8px;padding:1rem 1.2rem;text-align:center}
    .df-metric .val{font-size:1.7rem;font-weight:700}
    .df-metric .lbl{font-size:0.72rem;color:gray;text-transform:uppercase;
                    letter-spacing:0.06em;margin-top:3px}
    .df-metric .sub{font-size:0.72rem;color:gray;margin-top:2px}
    .rec-card{padding:0.85rem 1rem;border-left:3px solid;border-radius:0 6px 6px 0;
              margin-bottom:8px;background:rgba(128,128,128,0.04)}
    .rec-title{font-size:0.83rem;font-weight:600;margin-bottom:3px}
    .rec-sub{font-size:0.75rem;color:gray;line-height:1.55}
    .section-lbl{font-size:0.65rem;font-weight:700;letter-spacing:0.12em;
                 text-transform:uppercase;opacity:0.4;margin-bottom:1rem;
                 padding-bottom:0.5rem;border-bottom:1px solid rgba(128,128,128,0.15)}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("### Demand Forecasting & Inventory Optimization")
    st.markdown("Prophet-powered import demand forecasting · Inventory KPIs · Supplier performance · Reorder recommendations")
    st.markdown("---")

    # ── PREP DATA ──────────────────────────────────────────────────
    # Aggregate annual import demand by product category
    demand = (
        df.groupby(['year', 'product_category'])['import_value_usd']
        .sum().reset_index()
    )
    demand['import_bn'] = (demand['import_value_usd'] / 1e9).round(2)

    categories   = sorted(demand['product_category'].unique())
    all_countries = sorted(df['country'].unique())

    # ── CONTROLS ───────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        selected_cat = st.selectbox("Product Category", categories, index=0)
    with ctrl2:
        forecast_years = st.slider("Forecast Horizon (years)", 1, 5, 3)
    with ctrl3:
        lead_time_days = st.slider("Supplier Lead Time (days)", 7, 90, 30)

    cat_data = demand[demand['product_category'] == selected_cat].copy()
    cat_data = cat_data.sort_values('year')

    # ── SECTION 1: DEMAND FORECASTING ──────────────────────────────
    st.markdown("<p class='section-lbl'>Demand Forecasting — Prophet Model</p>",
                unsafe_allow_html=True)

    # Prepare Prophet input — use Jan 1 of each year as date
    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(cat_data['year'].astype(str) + '-01-01'),
        'y':  cat_data['import_bn'].values
    })

    # Fit Prophet
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.3,
        seasonality_mode='additive',
        interval_width=0.95
    )
    model.fit(prophet_df)

    # Forecast
    last_year   = int(cat_data['year'].max())
    future_yrs  = pd.date_range(
        start=f'{last_year+1}-01-01',
        periods=forecast_years, freq='YS'
    )
    future_df   = pd.DataFrame({'ds': future_yrs})
    all_future  = pd.concat([prophet_df[['ds']], future_df], ignore_index=True)
    forecast    = model.predict(all_future)

    hist_mask   = forecast['ds'] <= prophet_df['ds'].max()
    fore_mask   = forecast['ds'] >  prophet_df['ds'].max()

    # KPI metrics
    last_actual   = cat_data['import_bn'].iloc[-1]
    next_forecast = forecast[fore_mask]['yhat'].iloc[0]
    growth_pct    = ((next_forecast - last_actual) / last_actual * 100)
    avg_demand    = cat_data['import_bn'].mean()
    demand_vol    = cat_data['import_bn'].std() / avg_demand * 100

    m1,m2,m3,m4 = st.columns(4)
    for col, lbl, val, sub, clr in [
        (m1, "Latest Import Demand", f"${last_actual:.1f}B",
         f"{last_year} actual", "#4285F4"),
        (m2, "Next Year Forecast",   f"${next_forecast:.1f}B",
         f"{last_year+1} projected", "#34A853" if growth_pct>0 else "#EA4335"),
        (m3, "Projected Growth",     f"{growth_pct:+.1f}%",
         "vs prior year", "#34A853" if growth_pct>0 else "#EA4335"),
        (m4, "Demand Volatility",    f"{demand_vol:.1f}%",
         "coefficient of variation", "#FBBC04"),
    ]:
        col.markdown(f"""
        <div class='df-metric'>
          <div class='val' style='color:{clr}'>{val}</div>
          <div class='lbl'>{lbl}</div>
          <div class='sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Forecast chart
    fig_fc = go.Figure()

    # Confidence interval — forecast only
    fig_fc.add_trace(go.Scatter(
        x=pd.concat([forecast[fore_mask]['ds'],
                     forecast[fore_mask]['ds'][::-1]]),
        y=pd.concat([forecast[fore_mask]['yhat_upper'],
                     forecast[fore_mask]['yhat_lower'][::-1]]),
        fill='toself', fillcolor='rgba(66,133,244,0.12)',
        line=dict(color='rgba(0,0,0,0)'),
        name='95% Confidence Interval', showlegend=True
    ))

    # Historical fitted line
    fig_fc.add_trace(go.Scatter(
        x=forecast[hist_mask]['ds'],
        y=forecast[hist_mask]['yhat'],
        mode='lines',
        line=dict(color='rgba(66,133,244,0.4)', width=2, dash='dot'),
        name='Historical Trend (fitted)', showlegend=True
    ))

    # Actual historical points
    fig_fc.add_trace(go.Scatter(
        x=prophet_df['ds'], y=prophet_df['y'],
        mode='lines+markers',
        line=dict(color='#4285F4', width=2.5),
        marker=dict(size=8, color='#4285F4'),
        name='Actual Import Demand ($B)', showlegend=True
    ))

    # Forecast line
    fig_fc.add_trace(go.Scatter(
        x=forecast[fore_mask]['ds'],
        y=forecast[fore_mask]['yhat'],
        mode='lines+markers',
        line=dict(color='#34A853', width=2.5, dash='dash'),
        marker=dict(size=8, symbol='diamond', color='#34A853'),
        name='Forecasted Demand ($B)', showlegend=True
    ))

    # Divider line at last actual
    fig_fc.add_vline(
        x=prophet_df['ds'].max(),
        line_dash="dot", line_color="gray",
        annotation_text="Forecast starts",
        annotation_position="top right"
    )

    fig_fc.update_layout(
        height=360,
        margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Year', showgrid=False),
        yaxis=dict(title='Import Value ($B)',
                   gridcolor='rgba(128,128,128,0.12)'),
        legend=dict(orientation='h', y=1.12),
        hovermode='x unified'
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # ── SECTION 2: INVENTORY OPTIMIZATION ──────────────────────────
    st.markdown("---")
    st.markdown("<p class='section-lbl'>Inventory Optimization — Reorder Point & Safety Stock Analysis</p>",
                unsafe_allow_html=True)

    # Inventory calculations using demand data
    # Convert annual $B to monthly units proxy
    monthly_demand   = (last_actual * 1e9) / 12        # monthly $ demand
    daily_demand     = monthly_demand / 30             # daily demand
    demand_std       = (cat_data['import_bn'].std() * 1e9) / 12  # monthly std
    daily_demand_std = demand_std / 30

    # Safety stock (z=1.65 for 95% service level)
    z_score          = 1.65
    lead_time_demand = daily_demand * lead_time_days
    safety_stock     = z_score * daily_demand_std * np.sqrt(lead_time_days)
    reorder_point    = lead_time_demand + safety_stock

    # EOQ (Economic Order Quantity) — simplified
    holding_cost_pct = 0.25   # 25% of value per year
    ordering_cost    = 50000  # fixed cost per order
    annual_demand_v  = last_actual * 1e9
    eoq = np.sqrt((2 * annual_demand_v * ordering_cost) /
                  (holding_cost_pct * (annual_demand_v / 12)))

    # Stockout risk
    stockout_risk = (demand_vol / 100) * (lead_time_days / 30) * 100
    stockout_risk = min(stockout_risk, 95)

    inv1,inv2,inv3,inv4 = st.columns(4)
    for col, lbl, val, sub, clr in [
        (inv1, "Daily Demand",     f"${daily_demand/1e6:.1f}M",
         "average daily import value", "#4285F4"),
        (inv2, "Safety Stock",     f"${safety_stock/1e9:.2f}B",
         f"at 95% service level", "#34A853"),
        (inv3, "Reorder Point",    f"${reorder_point/1e9:.2f}B",
         f"at {lead_time_days}-day lead time", "#FBBC04"),
        (inv4, "Stockout Risk",    f"{stockout_risk:.1f}%",
         "probability without buffer", "#EA4335"),
    ]:
        col.markdown(f"""
        <div class='df-metric'>
          <div class='val' style='color:{clr}'>{val}</div>
          <div class='lbl'>{lbl}</div>
          <div class='sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    inv_col1, inv_col2 = st.columns(2)

    with inv_col1:
        # Inventory level simulation
        st.markdown("**Inventory Level Simulation — 12 Month Cycle**")
        months = np.arange(0, 13)
        eoq_units = eoq / (annual_demand_v / 12)

        inv_levels = []
        current    = eoq_units + safety_stock / (annual_demand_v / 12)
        for m in months:
            inv_levels.append(current)
            current -= 1
            if current <= (reorder_point / (annual_demand_v / 12)):
                current = eoq_units + safety_stock / (annual_demand_v / 12)

        fig_inv = go.Figure()
        fig_inv.add_trace(go.Scatter(
            x=months, y=inv_levels,
            fill='tozeroy', fillcolor='rgba(66,133,244,0.1)',
            line=dict(color='#4285F4', width=2),
            name='Inventory Level'
        ))
        fig_inv.add_hline(
            y=reorder_point / (annual_demand_v / 12),
            line_dash='dash', line_color='#FBBC04',
            annotation_text='Reorder Point',
            annotation_position='right'
        )
        fig_inv.add_hline(
            y=safety_stock / (annual_demand_v / 12),
            line_dash='dot', line_color='#EA4335',
            annotation_text='Safety Stock',
            annotation_position='right'
        )
        fig_inv.update_layout(
            height=280, margin=dict(l=0,r=80,t=10,b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Month', showgrid=False,
                       tickvals=list(range(13)),
                       ticktext=[f'M{i}' for i in range(13)]),
            yaxis=dict(title='Inventory Units (proxy)',
                       gridcolor='rgba(128,128,128,0.12)'),
            showlegend=False
        )
        st.plotly_chart(fig_inv, use_container_width=True)

    with inv_col2:
        # Lead time sensitivity analysis
        st.markdown("**Lead Time Sensitivity — Safety Stock vs Lead Time**")
        lt_range = np.arange(7, 91, 7)
        ss_range = [
            z_score * daily_demand_std * np.sqrt(lt) / 1e9
            for lt in lt_range
        ]
        rop_range = [
            (daily_demand * lt + z_score * daily_demand_std * np.sqrt(lt)) / 1e9
            for lt in lt_range
        ]

        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=lt_range, y=ss_range,
            mode='lines+markers',
            line=dict(color='#34A853', width=2),
            name='Safety Stock ($B)'
        ))
        fig_sens.add_trace(go.Scatter(
            x=lt_range, y=rop_range,
            mode='lines+markers',
            line=dict(color='#EA4335', width=2),
            name='Reorder Point ($B)'
        ))
        fig_sens.add_vline(
            x=lead_time_days, line_dash='dot',
            line_color='#FBBC04',
            annotation_text=f'Current: {lead_time_days}d',
            annotation_position='top'
        )
        fig_sens.update_layout(
            height=280, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Lead Time (days)', showgrid=False),
            yaxis=dict(title='Value ($B)',
                       gridcolor='rgba(128,128,128,0.12)'),
            legend=dict(orientation='h', y=1.1)
        )
        st.plotly_chart(fig_sens, use_container_width=True)

    # ── SECTION 3: SUPPLIER PERFORMANCE ────────────────────────────
    st.markdown("---")
    st.markdown("<p class='section-lbl'>Supplier Performance — Country-Level Analysis</p>",
                unsafe_allow_html=True)

    cat_country = (
        df[df['product_category'] == selected_cat]
        .groupby('country').agg(
            total_import=('import_value_usd', 'sum'),
            total_export=('export_value_usd', 'sum'),
            geo_risk=('geo_risk_score', 'mean'),
            trade_openness=('trade_openness_score', 'mean'),
            years=('year', 'nunique')
        ).reset_index()
    )
    cat_country['import_bn']     = (cat_country['total_import'] / 1e9).round(2)
    cat_country['export_bn']     = (cat_country['total_export'] / 1e9).round(2)
    cat_country['trade_balance'] = (cat_country['export_bn'] - cat_country['import_bn']).round(2)
    cat_country['import_share']  = (
        cat_country['import_bn'] / cat_country['import_bn'].sum() * 100
    ).round(1)

    # Supplier risk score: combination of geo_risk and import_share
    cat_country['supplier_risk_score'] = (
        (cat_country['geo_risk'] * 0.5) +
        (cat_country['import_share'] / 10 * 0.5)
    ).round(2)

    top10 = cat_country.nlargest(10, 'import_bn')

    sp1, sp2 = st.columns(2)

    with sp1:
        st.markdown("**Top 10 Import Sources — Share & Risk**")
        fig_sp = px.scatter(
            top10,
            x='import_share',
            y='geo_risk',
            size='import_bn',
            color='supplier_risk_score',
            text='country',
            color_continuous_scale=['#34A853','#FBBC04','#EA4335'],
            range_color=[0,10],
            size_max=40,
            hover_data={
                'import_bn': ':.1f',
                'import_share': ':.1f',
                'geo_risk': ':.1f',
                'trade_openness': ':.1f'
            }
        )
        fig_sp.update_traces(
            textposition='top center',
            textfont=dict(size=10),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Import Share: %{x:.1f}%<br>"
                "Geo Risk: %{y:.1f}/10<br>"
                "Import Value: $%{customdata[0]:.1f}B<br>"
                "<extra></extra>"
            )
        )
        fig_sp.add_vline(x=top10['import_share'].mean(),
                         line_dash='dot', line_color='gray',
                         annotation_text='Avg share',
                         annotation_position='bottom right')
        fig_sp.add_hline(y=5, line_dash='dot', line_color='#EA4335',
                         annotation_text='High risk threshold',
                         annotation_position='right')
        fig_sp.update_layout(
            height=320, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Import Market Share (%)', showgrid=False),
            yaxis=dict(title='Geopolitical Risk Score (1-10)',
                       gridcolor='rgba(128,128,128,0.12)'),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_sp, use_container_width=True)

    with sp2:
        st.markdown("**Supplier Performance Scorecard**")

        def risk_label(score):
            if score >= 6: return 'High Risk'
            elif score >= 3: return 'Medium Risk'
            return 'Low Risk'

        def risk_color(label):
            return {
                'High Risk':   'background-color:rgba(234,67,53,0.15);color:#EA4335',
                'Medium Risk': 'background-color:rgba(251,188,4,0.15);color:#B8860B',
                'Low Risk':    'background-color:rgba(52,168,83,0.15);color:#34A853',
            }.get(label, '')

        scorecard = top10[[
            'country','import_bn','import_share',
            'geo_risk','trade_openness','supplier_risk_score'
        ]].copy()
        scorecard['risk_label'] = scorecard['supplier_risk_score'].apply(risk_label)
        scorecard['import_bn']   = scorecard['import_bn'].apply(lambda x: f"${x:.1f}B")
        scorecard['import_share'] = scorecard['import_share'].apply(lambda x: f"{x:.1f}%")
        scorecard['geo_risk']    = scorecard['geo_risk'].apply(lambda x: f"{x:.1f}/10")
        scorecard['trade_openness'] = scorecard['trade_openness'].apply(lambda x: f"{x:.1f}/10")
        scorecard.columns = [
            'Country','Import Value','Share',
            'Geo Risk','Trade Openness','Risk Score','Risk Level'
        ]
        st.dataframe(
            scorecard.style.map(risk_color, subset=['Risk Level']),
            use_container_width=True,
            hide_index=True,
            height=320
        )

    # ── SECTION 4: EXECUTIVE RECOMMENDATIONS ───────────────────────
    st.markdown("---")
    st.markdown("<p class='section-lbl'>Executive Recommendations</p>",
                unsafe_allow_html=True)

    # Dynamic recommendations based on data
    high_risk_suppliers = top10[top10['supplier_risk_score'] >= 6]
    top_supplier        = top10.nlargest(1, 'import_share').iloc[0]
    conc_risk           = top_supplier['import_share'] > 25

    recs = [
        {
            "color": "#EA4335",
            "title": f"Demand Forecast Alert — {growth_pct:+.1f}% projected growth in {selected_cat}",
            "sub": (
                f"Prophet model projects ${next_forecast:.1f}B in {last_year+1} imports "
                f"({'increase' if growth_pct > 0 else 'decline'} from ${last_actual:.1f}B). "
                f"Adjust procurement volumes and safety stock buffers accordingly "
                f"before Q1 {last_year+1} contracting cycles."
            )
        },
        {
            "color": "#FBBC04",
            "title": f"Inventory Action — Reorder Point set at ${reorder_point/1e9:.2f}B",
            "sub": (
                f"With a {lead_time_days}-day supplier lead time and {demand_vol:.1f}% demand "
                f"volatility, maintain a safety stock of ${safety_stock/1e9:.2f}B "
                f"to achieve 95% service level. Current stockout risk without buffer: "
                f"{stockout_risk:.1f}%."
            )
        },
        {
            "color": "#EA4335" if conc_risk else "#34A853",
            "title": (
                f"{'High' if conc_risk else 'Moderate'} Concentration Risk — "
                f"{top_supplier['country']} holds {top_supplier['import_share']:.1f}% share"
            ),
            "sub": (
                f"{'Dangerous single-source dependency detected. ' if conc_risk else ''}"
                f"{top_supplier['country']} is the dominant import source for {selected_cat}. "
                f"{'Immediate diversification recommended — identify 2 alternative source countries.' if conc_risk else 'Monitor concentration levels and develop contingency sourcing plan.'}"
            )
        },
        {
            "color": "#EA4335" if len(high_risk_suppliers) > 0 else "#34A853",
            "title": (
                f"{len(high_risk_suppliers)} High-Risk Supplier "
                f"{'Nation' if len(high_risk_suppliers)==1 else 'Nations'} Flagged"
            ),
            "sub": (
                f"{'Countries flagged: ' + ', '.join(high_risk_suppliers['country'].tolist()) + '. ' if len(high_risk_suppliers)>0 else 'All top suppliers currently within acceptable risk parameters. '}"
                f"Recommend quarterly geo-risk reviews and dual-sourcing contracts "
                f"for any nation with geo risk > 5/10 and import share > 10%."
            )
        },
        {
            "color": "#4285F4",
            "title": f"Lead Time Optimization — Reduce to below 21 days where possible",
            "sub": (
                f"Sensitivity analysis shows safety stock requirements drop by "
                f"{((safety_stock - z_score*daily_demand_std*np.sqrt(21))/1e9):.2f}B "
                f"if lead time is reduced from {lead_time_days} to 21 days. "
                f"Prioritize near-shore suppliers for high-volatility categories."
            )
        },
    ]

    r1c, r2c = st.columns(2)
    for i, rec in enumerate(recs):
        col = r1c if i % 2 == 0 else r2c
        with col:
            st.markdown(f"""
            <div class='rec-card' style='border-color:{rec["color"]}'>
              <div class='rec-title' style='color:{rec["color"]}'>{rec["title"]}</div>
              <div class='rec-sub'>{rec["sub"]}</div>
            </div>""", unsafe_allow_html=True)

    # All categories trend
    st.markdown("---")
    st.markdown("<p class='section-lbl'>All Categories — Import Demand Trend Comparison</p>",
                unsafe_allow_html=True)

    all_trend = demand.copy()
    fig_all = px.line(
        all_trend, x='year', y='import_bn',
        color='product_category',
        markers=True,
        labels={'import_bn':'Import Value ($B)', 'year':'Year',
                'product_category':'Category'},
        color_discrete_sequence=[
            '#4285F4','#34A853','#FBBC04','#EA4335','#9C27B0','#00BCD4'
        ]
    )
    fig_all.update_layout(
        height=320, margin=dict(l=0,r=0,t=10,b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, title='Year'),
        yaxis=dict(gridcolor='rgba(128,128,128,0.12)',
                   title='Import Value ($B)'),
        legend=dict(orientation='h', y=1.12, title=None),
        hovermode='x unified'
    )
    st.plotly_chart(fig_all, use_container_width=True)

    st.markdown(
        "<p style='font-size:0.72rem;opacity:0.3;'>"
        "Demand forecasting powered by Meta Prophet · "
        "Inventory optimization using EOQ and safety stock models · "
        "Built by Akshada Karade · MS Engineering Management · UMass Amherst"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(page_title="Demand Forecasting", layout="wide")
    df = pd.read_csv("data/raw/trade_raw.csv")
    render_demand_forecasting_tab(df)
