"""
risk_scoring.py — single source of truth for supply-chain risk scoring.

Import this everywhere (app.py, sourcing_optimization_tab.py, the scoring
engine that writes risk_scores.csv, and any notebook) so every surface of
the project reports the SAME numbers. Today the dashboard and the memo
disagree because two different geo-risk conventions are in use; this file
ends that.

CONVENTION (decided once, applied everywhere)
---------------------------------------------
  Concentration  = HHI on export shares per category (DOJ standard),
                   normalised to a 0-10 scale.
  Geopolitical   = SHARE-WEIGHTED geo risk = Σ(share_i · geo_i) / 100,
                   i.e. the volume-weighted average supplier risk.
                   NOT the single top supplier's raw score.
  Combined       = 0.6 · concentration + 0.4 · geopolitical.

WHY share-weighted, not top-supplier-raw:
  The top-supplier's raw score double-counts with HHI (both already reward
  concentration) and throws away information about the rest of the supplier
  base. Share-weighting uses the whole portfolio and is the more defensible
  choice in an interview.

IMPORTANT — report the two dimensions separately too:
  The single combined score can hide a real tension. A category can be #1
  on concentration but not on geopolitics (e.g. heavy but LOW-risk US
  dependence) or #1 on geopolitics but not concentration (e.g. China-heavy
  electronics). Always surface `hhi_normalized` and `weighted_geo_risk`
  alongside the combined ranking.
"""
import pandas as pd

W_CONCENTRATION = 0.6
W_GEOPOLITICAL = 0.4


def category_risk_table(df, year):
    """One risk row per product category for the given year."""
    d = df[df["year"] == year].copy()
    totals = d.groupby("product_category")["export_value_usd"].transform("sum")
    d["share"] = d["export_value_usd"] / totals * 100.0

    rows = []
    for cat, g in d.groupby("product_category"):
        share = g["share"]
        hhi = float((share ** 2).sum())
        hhi_norm = hhi / 10000 * 10
        weighted_geo = float((share * g["geo_risk_score"]).sum() / 100.0)
        combined = W_CONCENTRATION * hhi_norm + W_GEOPOLITICAL * weighted_geo

        top = g.loc[g["export_value_usd"].idxmax()]
        rows.append(dict(
            product_category=cat,
            hhi_score=round(hhi, 1),
            hhi_normalized=round(hhi_norm, 2),
            weighted_geo_risk=round(weighted_geo, 2),
            combined_risk_score=round(combined, 2),
            concentration_risk=("HIGH" if hhi >= 2500 else
                                "MODERATE" if hhi >= 1500 else "LOW"),
            top_supplier=top["country"],
            top_share_pct=round(float(share.loc[top.name]), 1),
        ))

    return (pd.DataFrame(rows)
            .sort_values("combined_risk_score", ascending=False)
            .reset_index(drop=True))


def worst_by_dimension(table):
    """Convenience: the category topping each individual risk dimension."""
    return dict(
        worst_combined=table.iloc[0]["product_category"],
        worst_concentration=table.loc[table["hhi_normalized"].idxmax(), "product_category"],
        worst_geopolitical=table.loc[table["weighted_geo_risk"].idxmax(), "product_category"],
    )


if __name__ == "__main__":
    df = pd.read_csv("data/raw/trade_raw.csv")
    tbl = category_risk_table(df, 2022)
    print(tbl.to_string(index=False))
    print()
    print(worst_by_dimension(tbl))
    tbl.to_csv("data/processed/risk_scores.csv", index=False)
    print("\nregenerated data/processed/risk_scores.csv")