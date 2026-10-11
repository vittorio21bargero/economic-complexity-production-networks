# Economic complexity with production networks and import dependencies

Extensions of the two standard economic complexity algorithms, the **Economic Complexity Index
(ECI)** and **Fitness–Complexity**, that use the **production network** between products and correct
countries' scores for the complexity of the **inputs they import**.

*Master's thesis, MSc in Physics of Complex Systems, University of Turin, carried out at the
[Complexity Science Hub](https://csh.ac.at) Vienna (2026). Supervisor: Prof. Marco Pangallo;
co-supervisors: Dr. Vito D. P. Servedio and Dr. Leonardo Ialongo (CSH).*

![Effect of the import filter on Extended Fitness](docs/figures/import_filter_2012.png)

*Extended Fitness of each country with and without the import filter (2012, HS4). Although the
filter is built only to remove the complexity of imported inputs, the countries it penalises most
are assembly economies (Bangladesh, Vietnam, Turkey, Pakistan, Indonesia, …), not simply the countries
most integrated in global value chains.*

## The idea

Classical complexity metrics are computed only from the bipartite network of which country exports
which product. This has two blind spots:

- **supply-chain blindness**: the complexity of a product is inferred only from who exports it,
  with no information on what it is made of;
- **assembly trap**: a country gets full credit for exporting a complex product even when its
  complex components are imported.

The extended algorithm keeps the country–product recursion and adds two blocks:

$$
\begin{aligned}
\bar{F}^{(n+1)} &= \phi_X\big(Q^{(n)}\big) - S^{(n)} && \text{country score, minus imported input complexity}\\
\bar{\Delta}^{(n)} &= \Theta_X\big(F^{(n)}\big) && \text{intrinsic complexity (as in ECI / Fitness)}\\
\bar{Q}^{(n+1)} &= Q^{(n)} \, \alpha C + \Delta^{(n)} && \text{product complexity = intrinsic + inherited from inputs}
\end{aligned}
$$

with each vector normalised by its mean at every step. $C_{p'p} = 1$ if product $p'$ is an input of
$p$ (from the AI-generated production network **AIPnet**, Fetzer et al. 2024), $\phi_X, \Theta_X$
are the maps of the classical ECI or Fitness algorithm, and $S$ subtracts the complexity of the
inputs that a country imports (built from the binary import matrix $M^I$). With $\alpha = 0$ and
$S = 0$ the classical algorithms are recovered exactly.

## Main results

- **Theory.** At the fixed point the product complexity is a *personalised Katz centrality* of the
  production network: each product accumulates the intrinsic complexity of all the supply chains
  that flow into it, damped at every step upstream. The dimensionless parameter $a = \alpha\,\rho(C)$
  makes the propagation strength comparable across networks; with $a = 1$ intrinsic and input
  complexity have comparable weight.
- **Numerics.** Extended Fitness converges in a few tens of iterations and keeps scores away from
  zero, unlike classical Fitness. The import penalty is evaluated with a Horner scheme, so the powers
  of $C$ are never built.
- **Growth regressions.** Among the metrics tested one at a time, Extended Fitness adds the most
  explanatory power to GDP per capita growth over 5 and 10 years. These are in-sample associations
  with wide uncertainty, not a proof of better forecasting.
- **A critical result: the number of inputs.** A null model that rewires the inputs of every
  product, keeping their number fixed, leaves the ranking of product complexity almost unchanged.
  Most of the network signal therefore comes from *how many* inputs a product has rather than
  *which* ones. This is not just an artefact (the mean number of inputs alone predicts growth
  almost as well), but it means that the current specification uses the supply-chain structure
  only in part. The thesis discusses which block of the algorithm to change to address it.

<p align="center">
  <img src="docs/figures/rewiring_null_model.png" width="45%" alt="Rewiring null model">
  <img src="docs/figures/growth_regressions_dR2.png" width="52%" alt="Growth regressions">
</p>

*Left: correlation (Pearson, solid; Spearman, dashed) between the scores on the original network
and on a network where a fraction η of the input links is rewired, preserving the number of inputs
of each product: product complexity Q, input complexity QC, intrinsic complexity Δ (HS4).
Right: increase in R² of growth regressions over a model with only initial GDP per capita, when
one complexity metric is added (mean N₁ = average number of inputs of the exported products).*

## Repository structure

```
src/                          algorithms and data pipeline
  algorithms_with_imports.py  Extended ECI / Fitness with import filter (core)
  algorithms.py               classical ECI, Fitness, extensions without filter
  preprocessing.py            from raw trade data to the input matrices
notebooks/
  data_generation/            run the algorithms, save results
  plotting/                   thesis figures
  econometrics/               growth regressions, external validation
  archive/                    early exploration
scripts/                      figures of this README
```

Notebooks are research logs, with comments in Italian.

## Reproducibility

- **Requirements:** Python 3 with NumPy, pandas, SciPy, Matplotlib, PyArrow (`pip install -r requirements.txt`).
- **Run:** the notebooks in `data_generation/`, then those in `plotting/` and `econometrics/`.
- **Data:** not included (size and licences). BACI trade flows (CEPII, HS 2002, 2002–2024), the
  AIPnet production network (Fetzer et al., 2024), Penn World Table 11.0, OECD TiVA.

## Author

Vittorio Bargero, MSc in Physics of Complex Systems, University of Turin.
