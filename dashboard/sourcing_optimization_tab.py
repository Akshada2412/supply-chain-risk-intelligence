"""
Sourcing Optimization tab — Supply Chain Risk Intelligence
-----------------------------------------------------------
Turns the dashboard's DESCRIPTIVE risk scoring (HHI + geo risk) into a
PRESCRIPTIVE decision: given each category's demand and each country's
export capacity + geopolitical risk, compute the sourcing allocation that
MINIMISES risk-weighted volume, subject to a diversification cap.

Model (per reference year):
    Sets      k = product category,  c = supplier country
    Decision  x[k,c] >= 0   volume of category k sourced from country c
              y[k,c] in {0,1} (optional MILP) 1 if supplier c is used for k
    Params    D[k]        = total category demand (= sum of exports)
              e[k,c]      = export_value_usd (current capacity)
              g[c]        = geo_risk_score (1-10)
              F           = max supplier scale-up factor (cap = e*F)
              alpha       = diversification cap (no supplier > alpha*D[k])
    Objective min  sum_{k,c} g[c] * x[k,c]      (risk-weighted volume)
    s.t.      sum_c x[k,c] = D[k]               (meet demand)
              x[k,c] <= min(e[k,c]*F, alpha*D[k])
              (MILP) x[k,c] <= cap*y[k,c] ; sum_c y[k,c] <= N

Solver: Gurobi (gurobipy) if available, else CBC via PuLP. Same result,
so the deployed app never breaks if the Gurobi license is unavailable.

Built by Akshada Karade · MS Engineering Management · UMass Amherst
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


# ============================================================
# SOLVER LAYER  —  Gurobi primary, CBC (PuLP) fallback
# ============================================================
def _build_params(supply):
    """supply: DataFrame[product_category, country, vol (export), geo]."""
    cats = list(supply["product_category"].unique())
    demand = supply.groupby("product_category")["vol"].sum().to_dict()
    rows = supply.to_dict("records")
    return cats, demand, rows


def _solve_gurobi(supply, alpha, F, max_suppliers):
    import gurobipy as gp
    from gurobipy import GRB

    cats, demand, rows = _build_params(supply)
    m = gp.Model("sourcing")
    m.Params.OutputFlag = 0

    x, y, cap = {}, {}, {}
    for r in rows:
        k, c = r["product_category"], r["country"]
        cap[(k, c)] = min(r["vol"] * F, alpha * demand[k])
        x[(k, c)] = m.addVar(lb=0, ub=cap[(k, c)], name=f"x_{k}_{c}")
        if max_suppliers:
            y[(k, c)] = m.addVar(vtype=GRB.BINARY, name=f"y_{k}_{c}")

    for k in cats:
        m.addConstr(gp.quicksum(x[(k, c)] for (kk, c) in x if kk == k) == demand[k])
        if max_suppliers:
            for (kk, c) in [p for p in x if p[0] == k]:
                m.addConstr(x[(k, c)] <= cap[(k, c)] * y[(k, c)])
            m.addConstr(gp.quicksum(y[(k, c)] for (kk, c) in y if kk == k) <= max_suppliers)

    geo = {(r["product_category"], r["country"]): r["geo"] for r in rows}
    m.setObjective(gp.quicksum(geo[(k, c)] * x[(k, c)] for (k, c) in x), GRB.MINIMIZE)
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        return None
    return {(k, c): x[(k, c)].X for (k, c) in x}


def _solve_pulp(supply, alpha, F, max_suppliers):
    import pulp

    cats, demand, rows = _build_params(supply)
    prob = pulp.LpProblem("sourcing", pulp.LpMinimize)

    x, y, cap = {}, {}, {}
    for r in rows:
        k, c = r["product_category"], r["country"]
        cap[(k, c)] = min(r["vol"] * F, alpha * demand[k])
        x[(k, c)] = pulp.LpVariable(f"x_{k}_{c}", lowBound=0, upBound=cap[(k, c)])
        if max_suppliers:
            y[(k, c)] = pulp.LpVariable(f"y_{k}_{c}", cat="Binary")
            prob += x[(k, c)] <= cap[(k, c)] * y[(k, c)]

    for k in cats:
        prob += pulp.lpSum(x[(k, c)] for (kk, c) in x if kk == k) == demand[k]
        if max_suppliers:
            prob += pulp.lpSum(y[(k, c)] for (kk, c) in y if kk == k) <= max_suppliers

    geo = {(r["product_category"], r["country"]): r["geo"] for r in rows}
    prob += pulp.lpSum(geo[(k, c)] * x[(k, c)] for (k, c) in x)
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[status] != "Optimal":
        return None
    return {(k, c): (x[(k, c)].value() or 0.0) for (k, c) in x}


@st.cache_data(show_spinner=False)
def solve_sourcing(supply, alpha, F, max_suppliers=None):
    """Returns (allocation_dict | None, solver_name)."""
    try:
        alloc = _solve_gurobi(supply, alpha, F, max_suppliers)
        if alloc is not None:
            return alloc, "Gurobi"
        # optimal-not-found under Gurobi -> report infeasible (same model)
        return None, "Gurobi"
    except Exception:
        alloc = _solve_pulp(supply, alpha, F, max_suppliers)
        return alloc, "CBC (PuLP)"


# ============================================================
# METRICS  —  match app.py conventions (share-weighted geo risk)
# ============================================================
def alloc_to_frame(supply, alloc):
    """Attach optimized volume to the supply frame; drop ~zero rows."""
    recs = []
    geo = {(r["product_category"], r["country"]): r["geo"]
           for r in supply.to_dict("records")}
    for (k, c), v in alloc.items():
        if v > 1e-3:
            recs.append(dict(product_category=k, country=c, vol=v, geo=geo[(k, c)]))
    return pd.DataFrame(recs)


def category_metrics(frame):
    """Per-category HHI, share-weighted geo risk, combined score."""
    out = []
    for k, g in frame.groupby("product_category"):
        tot = g["vol"].sum()
        share = g["vol"] / tot * 100.0
        hhi = float((share ** 2).sum())
        hhi_norm = hhi / 10000 * 10
        wgeo = float((share * g["geo"]).sum() / 100.0)   # = volume-weighted avg geo
        combined = hhi_norm * 0.6 + wgeo * 0.4
        top_i = g["vol"].idxmax()
        out.append(dict(
            product_category=k, hhi=round(hhi, 1), hhi_norm=round(hhi_norm, 3),
            geo=round(wgeo, 2), combined=round(combined, 2),
            top=g.loc[top_i, "country"], top_share=round(float(share.max()), 1),
            n=int((g["vol"] > 1e-3).sum()), demand=tot,
        ))
    return pd.DataFrame(out)


def portfolio_rollup(metrics):
    """Demand-weighted portfolio HHI / geo / combined."""
    w = metrics["demand"] / metrics["demand"].sum()
    return dict(
        hhi_norm=float((metrics["hhi_norm"] * w).sum()),
        geo=float((metrics["geo"] * w).sum()),
        combined=float((metrics["combined"] * w).sum()),
        top_share=float(metrics["top_share"].max()),
    )


# ============================================================
# FEASIBILITY DIAGNOSTIC  —  names the actually-binding constraint
# ============================================================
# The solver returns None on infeasibility but not *why*. Two constraints
# can independently make a category impossible to cover to 100%:
#   1. Supplier scale-up F  — each supplier caps at F x its real exports
#   2. Supplier-count N (MILP) — only the N largest suppliers may be used
# (the diversification cap alpha only binds when suppliers are very few).
# This pure-numpy check reconstructs the max achievable coverage per
# category and computes the exact threshold on each lever that fixes it.
def _coverage(e, D, alpha, F, N):
    caps = np.sort(np.minimum(e * F, alpha * D))[::-1]
    if N:
        caps = caps[:N]
    return caps.sum() / D


def _min_F(e, D, alpha, N, F_now):
    """Smallest scale-up factor (>= F_now) that reaches full coverage; None if F alone can't."""
    if _coverage(e, D, alpha, 1e9, N) < 1 - 1e-9:
        return None
    lo, hi = F_now, 1e6
    for _ in range(60):
        mid = (lo + hi) / 2
        if _coverage(e, D, alpha, mid, N) >= 1 - 1e-9:
            hi = mid
        else:
            lo = mid
    return hi


def _min_N(e, D, alpha, F_now):
    """Smallest supplier count (at current F) that reaches full coverage; None if impossible."""
    caps = np.sort(np.minimum(e * F_now, alpha * D))[::-1]
    idx = int(np.searchsorted(np.cumsum(caps), D - 1e-6))
    return idx + 1 if idx < len(caps) else None


def diagnose_infeasibility(supply, alpha, F, max_suppliers):
    """Returns (blocking_categories, global_F_fix, global_N_fix) with exact thresholds."""
    blocking = []
    for k, g in supply.groupby("product_category"):
        D = g["vol"].sum()
        e = g["vol"].values
        cov = _coverage(e, D, alpha, F, max_suppliers)
        if cov < 1 - 1e-9:
            blocking.append(dict(
                cat=k, coverage=cov,
                F_fix=_min_F(e, D, alpha, max_suppliers, F),
                N_fix=_min_N(e, D, alpha, F),
            ))
    blocking.sort(key=lambda r: r["coverage"])
    F_fixes = [b["F_fix"] for b in blocking if b["F_fix"] is not None]
    N_fixes = [b["N_fix"] for b in blocking if b["N_fix"] is not None]
    global_F = max(F_fixes) if len(F_fixes) == len(blocking) and F_fixes else None
    global_N = max(N_fixes) if len(N_fixes) == len(blocking) and N_fixes else None
    return blocking, global_F, global_N


# ============================================================
# MAIN TAB
# ============================================================
def render_sourcing_optimization_tab(df):
    st.markdown("""
    <style>
    .so-metric{background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.15);
               border-radius:8px;padding:0.9rem 1.1rem;text-align:center}
    .so-metric .val{font-size:1.6rem;font-weight:800;line-height:1.1}
    .so-metric .lbl{font-size:0.68rem;color:gray;text-transform:uppercase;
                    letter-spacing:0.06em;margin-top:4px}
    .so-metric .sub{font-size:0.72rem;margin-top:3px}
    .so-lbl{font-size:0.65rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
            color:#F59E0B;opacity:0.7;margin-bottom:1rem;padding-bottom:0.5rem;
            border-bottom:1px solid rgba(245,158,11,0.15)}
    .so-badge{display:inline-block;font-size:0.7rem;font-weight:700;color:#34D399;
              background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.25);
              padding:3px 12px;border-radius:20px}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("### Prescriptive Sourcing Optimization")
    st.markdown(
        "The risk monitor above *diagnoses* concentration. This engine *prescribes* "
        "the fix — a linear/mixed-integer program that reallocates sourcing to "
        "minimise geopolitical risk exposure while enforcing supplier diversification."
    )
    st.markdown("---")

    # ---- CONTROLS ----
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        year = st.selectbox("Reference Year",
                            sorted(df["year"].unique(), reverse=True), index=0,
                            key="so_year")
    with c2:
        alpha = st.slider("Diversification Cap  (max supplier share)",
                          0.08, 0.60, 0.20, 0.01, key="so_alpha",
                          help="No single country may supply more than this share of a category")
    with c3:
        F = st.slider("Supplier Scale-Up Factor",
                      1.0, 4.0, 1.5, 0.5, key="so_F",
                      help="How far above current export volume a supplier can flex")
    with c4:
        use_milp = st.toggle("MILP: cap supplier count", value=False, key="so_milp")
        max_sup = None
        if use_milp:
            max_sup = st.slider("Max suppliers / category", 4, 20, 10, key="so_n")

    # ---- BUILD SUPPLY TABLE ----
    d = df[df["year"] == year]
    supply = (d.groupby(["product_category", "country"])
                .agg(vol=("export_value_usd", "sum"),
                     geo=("geo_risk_score", "first"))
                .reset_index())

    # ---- SOLVE ----
    alloc, solver = solve_sourcing(supply, alpha, F, max_sup)

    if alloc is None:
        blocking, gF, gN = diagnose_infeasibility(supply, alpha, F, max_sup)

        cap_desc = (f"each capped at min({F:g}× exports, {alpha:.0%} of demand)"
                    if not max_sup else
                    f"limited to {max_sup} suppliers, each capped at "
                    f"min({F:g}× exports, {alpha:.0%} of demand)")
        lines = [
            f"**No feasible allocation** (solver: {solver}). With suppliers {cap_desc}, "
            f"{len(blocking)} categor{'y' if len(blocking)==1 else 'ies'} can't reach "
            f"100% of demand — the binding limit is supplier capacity, not the cap alone:"
        ]
        for b in blocking[:3]:
            lines.append(f"  •  **{b['cat']}** — max {b['coverage']*100:.0f}% coverage")

        fixes = []
        if gF is not None:
            snapped = np.ceil(gF / 0.5) * 0.5           # next reachable slider step
            fixes.append(f"raise **Supplier Scale-Up to {snapped:g}** (need ≥ {gF:.2f})")
        if max_sup and gN is not None:
            fixes.append(f"allow **≥ {gN} suppliers**")
        fixes.append("loosen the **diversification cap**")
        lines.append("Fix any one: " + "; ".join(fixes) + ".")

        st.error("\n\n".join(lines))
        return

    st.markdown(
        f"<span class='so-badge'>Solved with {solver} — optimal</span>",
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ---- METRICS: BEFORE vs AFTER ----
    sq_frame = supply.rename(columns={})            # status quo = current exports
    sq_metrics = category_metrics(sq_frame)
    opt_frame = alloc_to_frame(supply, alloc)
    opt_metrics = category_metrics(opt_frame)

    sq_p = portfolio_rollup(sq_metrics)
    opt_p = portfolio_rollup(opt_metrics)

    def delta_pct(before, after):
        if before == 0:
            return 0.0
        return (after - before) / before * 100.0

    cards = [
        ("Combined Risk Score", sq_p["combined"], opt_p["combined"], "lower is safer"),
        ("Avg Geo-Risk Exposure", sq_p["geo"], opt_p["geo"], "share-weighted, 0-10"),
        ("Concentration (HHI/10)", sq_p["hhi_norm"], opt_p["hhi_norm"], "lower = diversified"),
        ("Peak Supplier Share", sq_p["top_share"], opt_p["top_share"], "% of a category"),
    ]
    mcols = st.columns(4)
    for col, (lbl, before, after, sub) in zip(mcols, cards):
        dp = delta_pct(before, after)
        good = after <= before
        clr = "#34D399" if good else "#EF4444"
        arrow = "▼" if after < before else ("▲" if after > before else "—")
        suffix = "%" if "Share" in lbl else ""
        col.markdown(f"""
        <div class='so-metric'>
          <div class='val' style='color:{clr}'>{after:.2f}{suffix}</div>
          <div class='lbl'>{lbl}</div>
          <div class='sub' style='color:{clr}'>{arrow} {abs(dp):.0f}% vs status quo ({before:.2f}{suffix})</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- CHART 1: per-category risk, before vs after ----
    st.markdown("<p class='so-lbl'>Combined Risk Score — Status Quo vs Optimized</p>",
                unsafe_allow_html=True)
    merged = sq_metrics[["product_category", "combined"]].merge(
        opt_metrics[["product_category", "combined"]],
        on="product_category", suffixes=("_sq", "_opt"))
    merged = merged.sort_values("combined_sq", ascending=True)

    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        y=merged["product_category"], x=merged["combined_sq"], orientation="h",
        name="Status Quo", marker_color="#EF4444", opacity=0.55,
        text=merged["combined_sq"].round(2), textposition="outside"))
    fig1.add_trace(go.Bar(
        y=merged["product_category"], x=merged["combined_opt"], orientation="h",
        name="Optimized", marker_color="#34D399",
        text=merged["combined_opt"].round(2), textposition="outside"))
    fig1.update_layout(
        height=320, barmode="group", margin=dict(l=0, r=40, t=5, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Combined Risk Score (0.6·HHI + 0.4·geo)", showgrid=False),
        yaxis=dict(title=""),
        legend=dict(orientation="h", y=1.12, title=None))
    fig1.update_layout(modebar_remove=["*"])
    st.plotly_chart(fig1, use_container_width=True)

    # ---- CHART 2: EFFICIENT FRONTIER ----
    st.markdown("<p class='so-lbl'>Efficient Frontier — Diversification vs Risk</p>",
                unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.75rem;opacity:0.5;margin-top:-0.5rem;'>"
        "Each point sweeps the diversification cap. The status-quo point sits "
        "<b>above</b> the frontier — meaning it is dominated: the optimizer reaches "
        "lower risk at the same or lower concentration.</p>", unsafe_allow_html=True)

    @st.cache_data(show_spinner=False)
    def frontier(supply, F):
        pts = []
        for a in [0.60, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08]:
            al, _ = solve_sourcing(supply, a, F, None)
            if al is None:
                continue
            met = category_metrics(alloc_to_frame(supply, al))
            roll = portfolio_rollup(met)
            pts.append(dict(alpha=a, hhi=roll["hhi_norm"], geo=roll["geo"]))
        return pd.DataFrame(pts)

    fr = frontier(supply, F)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=fr["hhi"], y=fr["geo"], mode="lines+markers",
        line=dict(color="#F59E0B", width=2.5),
        marker=dict(size=9, color="#F59E0B"),
        text=[f"cap {a:.0%}" for a in fr["alpha"]],
        hovertemplate="Cap: %{text}<br>Concentration: %{x:.2f}<br>Geo-Risk: %{y:.2f}<extra></extra>",
        name="Efficient frontier"))
    # status quo point (dominated)
    fig2.add_trace(go.Scatter(
        x=[sq_p["hhi_norm"]], y=[sq_p["geo"]], mode="markers+text",
        marker=dict(size=16, color="#EF4444", symbol="x"),
        text=["Status quo"], textposition="top center",
        textfont=dict(color="#EF4444"),
        hovertemplate="STATUS QUO<br>Concentration: %{x:.2f}<br>Geo-Risk: %{y:.2f}<extra></extra>",
        name="Status quo"))
    # currently selected point
    fig2.add_trace(go.Scatter(
        x=[opt_p["hhi_norm"]], y=[opt_p["geo"]], mode="markers+text",
        marker=dict(size=15, color="#34D399", symbol="star"),
        text=[f"your cap ({alpha:.0%})"], textposition="bottom center",
        textfont=dict(color="#34D399"),
        hovertemplate="SELECTED<br>Concentration: %{x:.2f}<br>Geo-Risk: %{y:.2f}<extra></extra>",
        name="Selected"))
    fig2.update_layout(
        height=380, margin=dict(l=0, r=0, t=5, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Concentration  (HHI / 10 — lower = more diversified)",
                   gridcolor="rgba(128,128,128,0.1)"),
        yaxis=dict(title="Avg Geopolitical Risk (0-10)",
                   gridcolor="rgba(128,128,128,0.1)"),
        legend=dict(orientation="h", y=1.12, title=None))
    fig2.update_layout(modebar_remove=["*"])
    st.plotly_chart(fig2, use_container_width=True)

    # ---- CHART 3 / TABLE: REALLOCATION for a focus category ----
    st.markdown("<p class='so-lbl'>Reallocation Plan — the concrete sourcing shift</p>",
                unsafe_allow_html=True)
    focus = st.selectbox("Focus category",
                         sorted(supply["product_category"].unique()), key="so_focus")

    D = supply[supply["product_category"] == focus]["vol"].sum()
    sq_f = supply[supply["product_category"] == focus].copy()
    sq_f["sq_share"] = sq_f["vol"] / D * 100.0
    opt_f = opt_frame[opt_frame["product_category"] == focus].copy()
    opt_f["opt_share"] = opt_f["vol"] / opt_f["vol"].sum() * 100.0

    tbl = sq_f[["country", "geo", "sq_share"]].merge(
        opt_f[["country", "opt_share"]], on="country", how="outer").fillna(0.0)
    tbl["delta"] = tbl["opt_share"] - tbl["sq_share"]
    tbl = tbl[(tbl["sq_share"] > 0.5) | (tbl["opt_share"] > 0.5)]
    tbl = tbl.sort_values("delta")

    fcol1, fcol2 = st.columns([3, 2])
    with fcol1:
        top_moves = pd.concat([tbl.head(5), tbl.tail(5)]).drop_duplicates("country")
        top_moves = top_moves.sort_values("delta")
        colors = ["#EF4444" if v < 0 else "#34D399" for v in top_moves["delta"]]
        figm = go.Figure(go.Bar(
            y=top_moves["country"], x=top_moves["delta"], orientation="h",
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in top_moves["delta"]], textposition="outside"))
        figm.update_layout(
            height=340, margin=dict(l=0, r=40, t=5, b=0),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Change in sourcing share (pp)", showgrid=False,
                       zeroline=True, zerolinecolor="rgba(128,128,128,0.4)"),
            yaxis=dict(title=""))
        figm.update_layout(modebar_remove=["*"])
        st.plotly_chart(figm, use_container_width=True)
    with fcol2:
        show = tbl.copy()
        show["geo"] = show["geo"].apply(lambda v: f"{v:.0f}/10")
        show["sq_share"] = show["sq_share"].apply(lambda v: f"{v:.1f}%")
        show["opt_share"] = show["opt_share"].apply(lambda v: f"{v:.1f}%")
        show["delta"] = show["delta"].apply(lambda v: f"{v:+.1f}pp")
        show.columns = ["Country", "Geo Risk", "Current", "Optimized", "Δ"]
        st.dataframe(show, use_container_width=True, hide_index=True, height=340)

    st.markdown(
        "<p style='font-size:0.72rem;opacity:0.35;line-height:1.6;'>"
        "LP/MILP sourcing model · objective: minimise Σ geo-risk × volume · "
        "s.t. demand satisfaction, supplier capacity, and diversification cap · "
        "Solver: Gurobi (gurobipy) with CBC/PuLP fallback · "
        "Built by Akshada Karade · MS Engineering Management · UMass Amherst"
        "</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    st.set_page_config(page_title="Sourcing Optimization", layout="wide")
    _df = pd.read_csv("data/raw/trade_raw.csv")
    render_sourcing_optimization_tab(_df)