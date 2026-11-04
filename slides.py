# /// script
# dependencies = [
#     "altair==6.3.0",
#     "marimo>=0.25.2",
#     "matplotlib==3.11.2",
#     "mendeleev==1.3.0",
#     "numpy==2.5.4",
#     "pandas==3.1.1",
#     "pyarrow==26.0.1",
#     "scikit-learn==1.9.1",
#     "wigglystuff==0.5.36",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", layout_file="layouts/slides.slides.json")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from notebook import app as analysis

    return analysis, mo


@app.cell(hide_code=True)
async def _(analysis):
    # Run notebook.py once and reuse its results, so the slides never repeat the analysis code.
    r = (await analysis.embed()).defs
    return (r,)


@app.cell(hide_code=True)
def _(mo, r):
    _adj = r["adjacency_test"].loc["all touching pairs"]
    mo.vstack(
        [
            mo.md(
                """
    # Elements, PCA and k-means
    ### Feature selection with PCA, and clustering compared with the periodic table

    **Q1** · Which of the 8 element features matter most?  
    **Q2** · Do k-means clusters match places in the periodic table?
    """
            ),
            mo.hstack(
                [
                    mo.stat(str(len(r["elements"])), label="Elements analysed", caption=f"of 118 ({len(r['dropped_symbols'])} superheavy removed)", bordered=True),
                    mo.stat(r["ranking"].index[0], label="Most important feature", caption="PCA method of Song et al. (2010)", bordered=True),
                    mo.stat(f"{_adj['observed same-cluster share']:.0%}", label="Table neighbours in the same cluster", caption=f"{_adj['null mean']:.0%} expected by chance", bordered=True),
                ],
                widths="equal",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, r):
    mo.md(
        f"""
    ## The data

    118 elements · 8 numeric features: atomic mass, electronegativity, atomic radius, ionization energy, electron affinity, melting point, boiling point, density.

    | Problem | What we do |
    |:---|:---|
    | `YearDiscovered` = "Ancient" | flag `DiscoveredAncient`; year left empty |
    | `StandardState` = "Expected to be a Solid" | keep "Solid"; flag `StatePredicted` |
    | Elements 100–118 have at most 4 of 8 features | remove them ({len(r['dropped_symbols'])} elements) |
    | No electron affinity for noble gases, group 2, Zn, Cd, Hg, N, Mn | set to 0 (the atom does not hold an extra electron) |
    | {int(r['imputed_mask'].to_numpy().sum())} other blanks | fill from the 5 most similar elements (KNN), checked against the **mendeleev** library |
    | Units differ by up to 10⁶ | rescale each feature to mean 0, spread 1 |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("## Q1 · The method of Song, Guo & Mei (2010)"),
            mo.mermaid(
                """
    graph LR
      A["8 features<br/>(rescaled)"] --> B["covariance<br/>matrix"] --> C["eigenvectors<br/>V1 … V8"] --> D["keep the<br/>top m"] --> E["score<br/>c_j = Σ |V_pj|"] --> F["sort →<br/>ranking"]
    """
            ),
            mo.md(
                r"""
    PCA finds **directions** in the data. Each direction is a mix of the 8 features.
    A feature with **small weights** in the important directions adds little, so it ranks low.

    $$c_j = \sum_{p=1}^{m} |V_{pj}| \qquad m = \text{number of directions with eigenvalue} > 1$$
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, r):
    mo.vstack(
        [
            mo.md(
                f"## Q1 · The ranking\nTop **{r['m_kaiser']}** directions = **{100 * r['explained'][: r['m_kaiser']].sum():.0f}%** of the variation. "
                "Each bar is a feature's score; the colours show what each direction adds."
            ),
            r["score_chart"],
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, r):
    _b = r["bootstrap_summary"]
    _order = list(r["ranking"].index)
    _top, _last = _order[0], _order[-1]
    mo.md(
        f"""
    ## Q1 · Answer

    **{" > ".join(_order[:3])}** > {" > ".join(_order[3:])}

    - **Stable ends.** Over 1000 bootstrap repeats, {_top} stays in the top 3 in {100 * _b.loc[_top, 'P(top 3)']:.0f}% of cases; {_last} stays in the bottom 3 in {100 * _b.loc[_last, 'P(bottom 3)']:.0f}%.
    - **Unstable middle.** The middle ranks change with $m$ and with small changes in the data.
    - **Rescale first.** Without rescaling, the kelvin temperatures take {100 * r['raw_explained'][0]:.0f}% of the variation and decide the ranking.
    - **Limit.** The method does not see repeated information. Melting and boiling point move together, so both rank high. The top 3 keep R² = {r['subset_r2'].loc[3, 'Song et al. top-n']:.2f} of all features; 3 random features keep {r['subset_r2'].loc[3, 'random subset (mean)']:.2f}.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, r):
    mo.vstack(
        [
            mo.md(f"## Q2 · k-means clusters (k = {r['n_clusters']}) on the periodic table\nk-means saw only the 8 features, **never** the positions. Hover an element for its data."),
            r["table_chart"],
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo, r):
    _a = r["adjacency_test"]
    _ari = r["cluster_agreement"]["ARI"]
    mo.vstack(
        [
            mo.md("## Q2 · Are table neighbours in the same cluster?\nNeighbours touch in the table: side by side, or one above the other. We shuffle the labels 5000 times to see what chance gives."),
            mo.hstack(
                [
                    mo.stat(f"{_a.loc['all touching pairs', 'observed same-cluster share']:.0%}", label="Neighbours in the same cluster", caption=f"{int(_a.loc['all touching pairs', 'n pairs'])} pairs", bordered=True),
                    mo.stat(f"{_a.loc['all touching pairs', 'null mean']:.0%}", label="Expected by chance", caption="5000 shuffles", bordered=True),
                    mo.stat(f"{_a.loc['all touching pairs', 'p-value (one-sided)']:.4f}", label="p-value", caption="no shuffle came close", bordered=True),
                ],
                widths="equal",
            ),
            mo.md(
                f"- Down a group: **{_a.loc['same group (up–down)', 'observed same-cluster share']:.0%}** vs {_a.loc['same group (up–down)', 'null mean']:.0%} by chance. Along a period: **{_a.loc['same period (left–right)', 'observed same-cluster share']:.0%}** vs {_a.loc['same period (left–right)', 'null mean']:.0%}.\n"
                f"- Match (ARI, 0 = chance, 1 = perfect): group {_ari['Group']:.2f}, family {_ari['GroupBlock']:.2f}, period {_ari['Period']:.2f}.\n"
                "- Same result with other ways to fill the blanks, and with no filled values at all."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Q2 · Answer: the clusters are areas of the table

    | Cluster | Area of the table | Typical features |
    |:---|:---|:---|
    | 1 | Noble gases + H, N, O, P | gases, highest ionization energy |
    | 2 | Light metals and semi-metals (Li–Mg, Sc–Zn, B–Sb) | middle values |
    | 3 | Centre of the heavy transition metals + C | highest melting point and density |
    | 4 | Halogens and chalcogens | highest electron affinity |
    | 5 | Groups 1–2 (Na–Fr, Ca–Ra) | lowest electronegativity, largest atoms |
    | 6 | Lanthanides, actinides + Hg, Tl, Pb, Bi | heavy, large |

    **Exceptions make chemical sense:** H is a gas, so it joins the noble gases. C has the highest melting point, so it joins the hard metals.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Summary

    **Q1.** PCA (Song et al.) ranks **melting point** first and **ionization energy** last. Both ends are stable.
    The method finds the features that drive the main patterns, but it keeps features that repeat each other.

    **Q2.** Yes: k-means groups follow the periodic table. Neighbours share a cluster 3–4× more often than chance.
    The clusters follow groups and blocks more than periods.

    *Full analysis, code and interactive charts: `notebook.py`.*
    """)
    return


if __name__ == "__main__":
    app.run()
