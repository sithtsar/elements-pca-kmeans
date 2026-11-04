# Elements, PCA and k-means

A course assignment about the 118 chemical elements and 8 of their properties
(atomic mass, electronegativity, atomic radius, ionization energy, electron affinity,
melting point, boiling point, density).

- **Q1.** Which properties matter most? We rank them with the PCA feature-selection
  method of Song, Guo & Mei (2010).
- **Q2.** Does k-means group the elements the same way as the periodic table?

**Short answers.** Q1: melting point matters most and ionization energy least.
Q2: yes. Elements that touch in the periodic table are in the same cluster 73% of
the time, against 20% by pure chance.

The full question is in [`PS.md`](PS.md).

---

## I only want to read it (no install)

| File | What it is |
|---|---|
| [`exports/report.pdf`](exports/report.pdf) | The full report: every step, chart and answer |
| [`exports/slides.pdf`](exports/slides.pdf) | 9 slides for a presentation |
| [`exports/slides.html`](exports/slides.html) | The same slides. Download and open in a browser; use the arrow keys |
| [`exports/notebook.html`](exports/notebook.html) | The full notebook. Download and open in a browser; hover the charts |
| [`exports/notebook.ipynb`](exports/notebook.ipynb) | Jupyter version with all outputs |

---

## I want to run it myself (3 steps)

**1. Install `uv`** (a small tool that installs Python and the packages for you).

Mac or Linux, in a terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows, in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close the terminal and open a new one.

**2. Download this project.**

```bash
git clone https://github.com/sithtsar/elements-pca-kmeans.git
cd elements-pca-kmeans
```

(No git? On GitHub, click **Code → Download ZIP**, unzip it, and open a terminal in that folder.)

**3. Open the notebook.**

```bash
uvx marimo edit --sandbox notebook.py
```

Your browser opens the notebook. The first start downloads the packages, which takes
1–2 minutes. Then press the **Start Tour** button at the top.

Things to try in the notebook:

- Drag the feature list in **"Make a guess"**, then compare your guess with PCA.
- Drag the blue number in **"Use the top 3 PCA directions"** or **"Split the elements into 6 clusters"**.
- Click an element in the periodic table to light up its cluster.

To show the slides instead:

```bash
uvx marimo run --sandbox slides.py
```

Use the arrow keys to move between slides.

---

## Rebuild the PDFs and other exports

```bash
./export.sh
```

This writes everything in `exports/` again. It only needs `uv`. The first run also
downloads a headless Chromium (about 100 MB), which draws the PDFs.
On Windows, run it in Git Bash or WSL.

---

## What is in this folder

| Path | What it is |
|---|---|
| `notebook.py` | The full analysis, as a [marimo](https://marimo.io) notebook |
| `slides.py` | The slide deck. It runs `notebook.py` and reuses its results |
| `data/PubChemElements_all.csv` | The data (PubChem) |
| `export.sh` | Rebuilds `exports/` |
| `layouts/` | Tells marimo to show `slides.py` as slides |

## Tools used

marimo (notebook and slides), pandas, NumPy, scikit-learn (PCA check, KNN imputation, k-means),
Altair and matplotlib (charts), wigglystuff (drag-to-rank list, inline sliders, guided tour,
parallel coordinates) and mendeleev (reference element data, used to check the cleaned data).

## Reference

F. Song, Z. Guo, D. Mei, "Feature selection using principal component analysis",
ICSEM 2010, pp. 27–30, doi:[10.1109/ICSEM.2010.14](https://doi.org/10.1109/ICSEM.2010.14).
The paper PDF is not in this repository because of copyright.
