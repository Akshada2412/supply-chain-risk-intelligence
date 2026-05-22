import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')


def render_demand_forecasting_tab(df):
    """
    Demand Forecasting & Inventory Optimization tab.
    Uses statsmodels Holt-Winters + SARIMAX for forecasting.
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
                 padding-bottom:0.5rem;
                 border-bottom:1px solid rgba(128,128,128,0.15)}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("### Demand Forecasting & Inventory Optimization")
    st.markdown(
        "Holt-Winters exponential smoothing forecast · "
        "Inventory KPIs · Supplier performance · Reorder recommendations"
    )
    st.markdown("---")

    # ── PREP DATA ──────────────────────────────────────────────────
    demand = (
        df.groupby(['year', 'product_category'])['import_value_usd']
        .sum().reset_index()
    )
    demand['import_bn'] = (demand['import_value_usd'] / 1e9).round(2)
    categories = sorted(demand['product_category'].unique())

    # ── CONTROLS ───────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    with ctrl1:
        selected_cat = st.selectbox("Product Category", categories, index=0)
    with ctrl2:
        forecast_years = st.slider("Forecast Horizon (years)", 1, 5, 3)
    with ctrl3:
        lead_time_days = st.slider("Supplier Lead Time (days)", 7, 90, 30)

    cat_data = (
        demand[demand['product_category'] == selected_cat]
        .sort_values('year')
        .reset_index(drop=True)
    )
    y = cat_data['import_bn'].values
    years = cat_data['year'].values

    # ── FORECAST — Holt-Winters ─────────────────────────────────────
    try:
        model = ExponentialSmoothing(
            y, trend='add', seasonal=None,
            initialization_method='estimated'
        )
        fit = model.fit(optimized=True, remove_bias=True)
        forecast_vals = fit.forecast(forecast_years)

        # Bootstrap confidence intervals
        residuals   = fit.resid
        resid_std   = residuals.std()
        ci_factor   = 1.96
        ci_upper    = forecast_vals + ci_factor * resid_std * np.sqrt(
            np.arange(1, forecast_years + 1))
        ci_lower    = forecast_vals - ci_factor * resid_std * np.sqrt(
            np.arange(1, forecast_years + 1))
        fitted_vals = fit.fittedvalues

    except Exception:
        # Fallback: linear trend
        z = np.polyfit(np.arange(len(y)), y, 1)
        p = np.poly1d(z)
        forecast_vals = p(np.arange(len(y), len(y) + forecast_years))
        fitted_vals   = p(np.arange(len(y)))
        resid_std     = np.std(y - fitted_vals)
        ci_upper = forecast_vals + 1.96 * resid_std
        ci_lower = forecast_vals - 1.96 * resid_std

    future_years  = np.arange(years[-1] + 1, years[-1] + 1 + forecast_years)
    last_actual   = float(y[-1])
    next_forecast = float(forecast_vals[0])
    growth_pct    = (next_forecast - last_actual) / last_actual * 100
    demand_vol    = (np.std(y) / np.mean(y)) * 100

    # ── SECTION 1: DEMAND FORECASTING ──────────────────────────────
    st.markdown(
        "<p class='section-lbl'>Demand Forecasting — "
        "Holt-Winters Exponential Smoothing Model</p>",
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4)
    for col, lbl, val, sub, clr in [
        (m1, "Latest Import Demand", f"${last_actual:.1f}B",
         f"{years[-1]} actual", "#4285F4"),
        (m2, "Next Year Forecast",   f"${next_forecast:.1f}B",
         f"{future_years[0]} projected",
         "#34A853" if growth_pct > 0 else "#EA4335"),
        (m3, "Projected Growth",     f"{growth_pct:+.1f}%",
         "vs prior year",
         "#34A853" if growth_pct > 0 else "#EA4335"),
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

    # CI band
    fig_fc.add_trace(go.Scatter(
        x=np.concatenate([future_years, future_years[::-1]]).tolist(),
        y=np.concatenate([ci_upper, ci_lower[::-1]]).tolist(),
        fill='toself', fillcolor='rgba(52,168,83,0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        name='95% Confidence Interval'
    ))
    # Fitted values
    fig_fc.add_trace(go.Scatter(
        x=years.tolist(), y=fitted_vals.tolist(),
        mode='lines',
        line=dict(color='rgba(66,133,244,0.4)', width=2, dash='dot'),
        name='Model Fit'
    ))
    # Actual
    fig_fc.add_trace(go.Scatter(
        x=years.tolist(), y=y.tolist(),
        mode='lines+markers',
        line=dict(color='#4285F4', width=2.5),
        marker=dict(size=8),
        name='Actual Import Demand ($B)'
    ))
    # Forecast
    fig_fc.add_trace(go.Scatter(
        x=future_years.tolist(), y=forecast_vals.tolist(),
        mode='lines+markers',
        line=dict(color='#34A853', width=2.5, dash='dash'),
        marker=dict(size=8, symbol='diamond', color='#34A853'),
        name='Forecasted Demand ($B)'
    ))
    fig_fc.add_vline(
        x=int(years[-1]),
        line_dash='dot', line_color='gray',
        annotation_text='Forecast starts',
        annotation_position='top right'
    )
    fig_fc.update_layout(
        height=360, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title='Year', showgrid=False,
                   tickmode='array',
                   tickvals=np.concatenate(
                       [years, future_years]).tolist()),
        yaxis=dict(title='Import Value ($B)',
                   gridcolor='rgba(128,128,128,0.12)'),
        legend=dict(orientation='h', y=1.12),
        hovermode='x unified'
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    # ── SECTION 2: INVENTORY OPTIMIZATION ──────────────────────────
    st.markdown("---")
    st.markdown(
        "<p class='section-lbl'>"
        "Inventory Optimization — Reorder Point & Safety Stock</p>",
        unsafe_allow_html=True
    )

    monthly_demand   = (last_actual * 1e9) / 12
    daily_demand     = monthly_demand / 30
    monthly_std      = (np.std(y) * 1e9) / 12
    daily_std        = monthly_std / 30
    z_score          = 1.65       # 95% service level
    safety_stock     = z_score * daily_std * np.sqrt(lead_time_days)
    reorder_point    = daily_demand * lead_time_days + safety_stock
    stockout_risk    = min((demand_vol / 100) * (lead_time_days / 30) * 100, 95)

    inv1, inv2, inv3, inv4 = st.columns(4)
    for col, lbl, val, sub, clr in [
        (inv1, "Daily Demand",
         f"${daily_demand/1e6:.1f}M", "avg daily import value", "#4285F4"),
        (inv2, "Safety Stock",
         f"${safety_stock/1e9:.2f}B", "at 95% service level", "#34A853"),
        (inv3, "Reorder Point",
         f"${reorder_point/1e9:.2f}B",
         f"at {lead_time_days}-day lead time", "#FBBC04"),
        (inv4, "Stockout Risk",
         f"{stockout_risk:.1f}%", "without buffer stock", "#EA4335"),
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
        st.markdown("**Inventory Cycle Simulation — 12 Months**")
        eoq_proxy    = 3.0   # months of stock per order
        current_inv  = eoq_proxy + safety_stock / monthly_demand
        rop_proxy    = reorder_point / monthly_demand
        ss_proxy     = safety_stock / monthly_demand
        inv_levels   = []
        cur          = current_inv
        for _ in range(13):
            inv_levels.append(max(cur, 0))
            cur -= 1
            if cur <= rop_proxy:
                cur = eoq_proxy + ss_proxy

        fig_inv = go.Figure()
        fig_inv.add_trace(go.Scatter(
            x=list(range(13)), y=inv_levels,
            fill='tozeroy', fillcolor='rgba(66,133,244,0.1)',
            line=dict(color='#4285F4', width=2), name='Inventory Level'
        ))
        fig_inv.add_hline(
            y=rop_proxy, line_dash='dash', line_color='#FBBC04',
            annotation_text='Reorder Point',
            annotation_position='right'
        )
        fig_inv.add_hline(
            y=ss_proxy, line_dash='dot', line_color='#EA4335',
            annotation_text='Safety Stock',
            annotation_position='right'
        )
        fig_inv.update_layout(
            height=280, margin=dict(l=0, r=80, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Month', showgrid=False,
                       tickvals=list(range(13)),
                       ticktext=[f'M{i}' for i in range(13)]),
            yaxis=dict(title='Months of Stock',
                       gridcolor='rgba(128,128,128,0.12)'),
            showlegend=False
        )
        st.plotly_chart(fig_inv, use_container_width=True)

    with inv_col2:
        st.markdown("**Lead Time Sensitivity — Safety Stock vs Lead Time**")
        lt_range = np.arange(7, 91, 7)
        ss_range = [z_score * daily_std * np.sqrt(lt) / 1e9 for lt in lt_range]
        rop_range = [
            (daily_demand * lt + z_score * daily_std * np.sqrt(lt)) / 1e9
            for lt in lt_range
        ]
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=lt_range.tolist(), y=ss_range,
            mode='lines+markers',
            line=dict(color='#34A853', width=2), name='Safety Stock ($B)'
        ))
        fig_sens.add_trace(go.Scatter(
            x=lt_range.tolist(), y=rop_range,
            mode='lines+markers',
            line=dict(color='#EA4335', width=2), name='Reorder Point ($B)'
        ))
        fig_sens.add_vline(
            x=lead_time_days, line_dash='dot', line_color='#FBBC04',
            annotation_text=f'Current: {lead_time_days}d',
            annotation_position='top'
        )
        fig_sens.update_layout(
            height=280, margin=dict(l=0, r=0, t=10, b=0),
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
    st.markdown(
        "<p class='section-lbl'>"
        "Supplier Performance — Country-Level Analysis</p>",
        unsafe_allow_html=True
    )

    cat_country = (
        df[df['product_category'] == selected_cat]
        .groupby('country').agg(
            total_import=('import_value_usd', 'sum'),
            total_export=('export_value_usd', 'sum'),
            geo_risk=('geo_risk_score', 'mean'),
            trade_openness=('trade_openness_score', 'mean'),
        ).reset_index()
    )
    cat_country['import_bn']    = (cat_country['total_import'] / 1e9).round(2)
    cat_country['import_share'] = (
        cat_country['import_bn'] / cat_country['import_bn'].sum() * 100
    ).round(1)
    cat_country['supplier_risk'] = (
        cat_country['geo_risk'] * 0.5 +
        (cat_country['import_share'] / 10) * 0.5
    ).round(2)

    top10 = cat_country.nlargest(10, 'import_bn').reset_index(drop=True)

    sp1, sp2 = st.columns(2)

    with sp1:
        st.markdown("**Top 10 Import Sources — Concentration vs Risk**")
        fig_sp = px.scatter(
            top10, x='import_share', y='geo_risk',
            size='import_bn', color='supplier_risk',
            text='country',
            color_continuous_scale=['#34A853', '#FBBC04', '#EA4335'],
            range_color=[0, 10], size_max=40,
        )
        fig_sp.update_traces(
            textposition='top center', textfont=dict(size=10),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Import Share: %{x:.1f}%<br>"
                "Geo Risk: %{y:.1f}/10<br>"
                "<extra></extra>"
            )
        )
        fig_sp.add_vline(
            x=float(top10['import_share'].mean()),
            line_dash='dot', line_color='gray',
            annotation_text='Avg share',
            annotation_position='bottom right'
        )
        fig_sp.add_hline(
            y=5, line_dash='dot', line_color='#EA4335',
            annotation_text='High risk threshold',
            annotation_position='right'
        )
        fig_sp.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title='Import Market Share (%)', showgrid=False),
            yaxis=dict(title='Geopolitical Risk (1-10)',
                       gridcolor='rgba(128,128,128,0.12)'),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_sp, use_container_width=True)

    with sp2:
        st.markdown("**Supplier Scorecard**")

        def risk_label(s):
            if s >= 6:   return 'High Risk'
            elif s >= 3: return 'Medium Risk'
            return 'Low Risk'

        def style_risk(val):
            colors = {
                'High Risk':   'background-color:rgba(234,67,53,0.15);color:#EA4335',
                'Medium Risk': 'background-color:rgba(251,188,4,0.15);color:#9a6c00',
                'Low Risk':    'background-color:rgba(52,168,83,0.15);color:#34A853',
            }
            return colors.get(val, '')

        sc = top10[[
            'country', 'import_bn', 'import_share',
            'geo_risk', 'trade_openness', 'supplier_risk'
        ]].copy()
        sc['risk_label']     = sc['supplier_risk'].apply(risk_label)
        sc['import_bn']      = sc['import_bn'].apply(lambda x: f"${x:.1f}B")
        sc['import_share']   = sc['import_share'].apply(lambda x: f"{x:.1f}%")
        sc['geo_risk']       = sc['geo_risk'].apply(lambda x: f"{x:.1f}/10")
        sc['trade_openness'] = sc['trade_openness'].apply(lambda x: f"{x:.1f}/10")
        sc.columns = [
            'Country', 'Import Value', 'Share',
            'Geo Risk', 'Trade Openness', 'Risk Score', 'Risk Level'
        ]
        st.dataframe(
            sc.style.map(style_risk, subset=['Risk Level']),
            use_container_width=True, hide_index=True, height=320
        )

    # ── SECTION 4: EXECUTIVE RECOMMENDATIONS ───────────────────────
    st.markdown("---")
    st.markdown(
        "<p class='section-lbl'>Executive Recommendations</p>",
        unsafe_allow_html=True
    )

    top_supplier = top10.nlargest(1, 'import_share').iloc[0]
    conc_risk    = top_supplier['import_share'] > 25
    high_risk_s  = top10[top10['supplier_risk'] >= 6]

    recs = [
        {
            "color": "#EA4335" if growth_pct > 10 else "#FBBC04",
            "title": (
                f"Demand Forecast Alert — {growth_pct:+.1f}% projected "
                f"growth in {selected_cat}"
            ),
            "sub": (
                f"Model projects ${next_forecast:.1f}B in "
                f"{future_years[0]} imports "
                f"({'increase' if growth_pct > 0 else 'decline'} from "
                f"${last_actual:.1f}B). Adjust procurement volumes and "
                f"safety stock buffers before Q1 {future_years[0]} "
                f"contracting cycles."
            )
        },
        {
            "color": "#FBBC04",
            "title": (
                f"Inventory Action — Reorder Point at "
                f"${reorder_point/1e9:.2f}B"
            ),
            "sub": (
                f"With {lead_time_days}-day lead time and {demand_vol:.1f}% "
                f"demand volatility, maintain safety stock of "
                f"${safety_stock/1e9:.2f}B for 95% service level. "
                f"Stockout risk without buffer: {stockout_risk:.1f}%."
            )
        },
        {
            "color": "#EA4335" if conc_risk else "#34A853",
            "title": (
                f"{'High' if conc_risk else 'Moderate'} Concentration Risk — "
                f"{top_supplier['country']} holds "
                f"{top_supplier['import_share']:.1f}% share"
            ),
            "sub": (
                f"{'Single-source dependency detected. ' if conc_risk else ''}"
                f"{top_supplier['country']} dominates {selected_cat} imports. "
                f"{'Immediate diversification recommended.' if conc_risk else 'Develop contingency sourcing plan.'}"
            )
        },
        {
            "color": "#EA4335" if len(high_risk_s) > 0 else "#34A853",
            "title": (
                f"{len(high_risk_s)} High-Risk Supplier "
                f"{'Nation' if len(high_risk_s) == 1 else 'Nations'} Flagged"
            ),
            "sub": (
                f"{'Nations: ' + ', '.join(high_risk_s['country'].tolist()) + '. ' if len(high_risk_s) > 0 else 'All top suppliers within acceptable parameters. '}"
                f"Recommend quarterly geo-risk reviews and dual-sourcing "
                f"contracts for nations with risk score above 6."
            )
        },
        {
            "color": "#4285F4",
            "title": "Lead Time Optimization — Reduce to 21 days where possible",
            "sub": (
                f"Safety stock requirements drop by "
                f"${(safety_stock - z_score*daily_std*np.sqrt(21))/1e9:.2f}B "
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

    # ── ALL CATEGORIES TREND ────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p class='section-lbl'>"
        "All Categories — Import Demand Trend Comparison</p>",
        unsafe_allow_html=True
    )

    fig_all = px.line(
        demand, x='year', y='import_bn',
        color='product_category', markers=True,
        labels={'import_bn': 'Import Value ($B)', 'year': 'Year',
                'product_category': 'Category'},
        color_discrete_sequence=[
            '#4285F4', '#34A853', '#FBBC04', '#EA4335', '#9C27B0', '#00BCD4'
        ]
    )
    fig_all.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
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
        "Demand forecasting powered by Holt-Winters Exponential Smoothing · "
        "Inventory optimization using safety stock and reorder point models · "
        "Built by Akshada Karade · MS Engineering Management · UMass Amherst"
        "</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(page_title="Demand Forecasting", layout="wide")
    df = pd.read_csv("data/raw/trade_raw.csv")
    render_demand_forecasting_tab(df)
