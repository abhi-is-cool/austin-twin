# Outstanding items after paper_v2.tex / paper_v3.tex

Results from working through the agreed next-step list. Everything already
folded into `paper_v3.tex` has been dropped; what follows is new, or now
inconsistent because of the v3 edits.

**Read §5.6 first — it is the settled end-to-end run and supersedes
the partial ones. §5.2 first if you want the reasoning behind it** — it is the one open item that
changes a headline number, and it corrects a claim I made earlier in this
file (§5.0). §4 also replaces an earlier claim of mine.

*All ceiling figures below are multi-day means over 49 cities unless a table
says otherwise. Where a superseded run is quoted for comparison it is
labelled as such.*

Artefacts: `outputs/ceiling_study.jsonl`, `ceiling_study_analysis.json`,
`decay_pareto.json`, `pm_variants.json`, `bootstrap_arco.json`,
`ceiling_definition_diag.json`, `advection_and_bias.json`,
`channel_prescreen.json`, `paired_era5_ci_arco.json`,
`ceiling_study_singletile.jsonl` and `ceiling_study_1day.jsonl`
(both superseded, kept for comparison).

---

## 1. The "ceiling" is not an upper bound as defined — now much more serious

The abstract defines the ceiling as Pearson *r* between MODIS LST and
**impervious fraction** and calls it "the maximum correlation any
land-cover-only model can achieve". The simulator consumes **three**
channels, and the multiple correlation of a three-predictor fit is ≥ the
univariate correlation of any one predictor by construction.

On the original four cities this looked like a technicality worth tidying.
Across 49 cities it is not (multi-day means):

| city | univariate *r*(LST, imp) | multiple *R* (3 channels) | difference |
|---|---|---|---|
| Boise | **−0.331** | **+0.462** | **0.79** |
| Albuquerque | −0.308 | +0.435 | 0.74 |
| Tucson | −0.254 | +0.345 | 0.60 |
| Denver | +0.023 | +0.467 | 0.44 |
| Phoenix | −0.095 | +0.223 | 0.32 |
| Pittsburgh | +0.414 | +0.417 | 0.00 |
| Raleigh | +0.527 | +0.529 | 0.00 |

In arid cities the univariate "ceiling" is not merely conservative, it is
**negative** — it reports that no land-cover model can work, while a plain
three-channel linear regression reaches +0.46. Any conclusion of the form
"the model achieves X% of its ceiling" is meaningless where the denominator
is negative.

It also fixes the Miami anomaly the paper currently hand-waves. §III-C
explains Miami's Scheme A `+0.832` exceeding ceiling `+0.827` as "within the
expected sampling variability of the bootstrap distribution". It is not
sampling noise, it is definitional — and against the correct multivariate
ceiling (`+0.8425`) Miami sits at **0.99×**, *at* the ceiling rather than
above it. Cleaner result, no excuse needed.

**Recommended fix.** Define the ceiling as the multiple correlation over the
channels the model consumes, and state it as a bound on any *linear* model
of those channels. Austin still exceeds it (1.08×) because the simulator is
nonlinear — worth saying plainly. Rankings are unchanged (Denver 0.03×,
Phoenix 0.20×, Miami 0.99×). `baseline_regression.py` already has the lstsq
machinery.

### 1.1 Single-day ceilings are unstable — and the published ones are optimistic

**This is the most consequential finding in this round.** The ceiling is now
computed over every sufficiently clear day in a 12-date pool (median 6 days
per city) rather than one hand-picked day. Against the published values:

| city | published (1 day) | multi-day mean | sd | observed range | n days |
|---|---|---|---|---|---|
| Austin | +0.671 | **+0.628** | 0.077 | [+0.539, +0.694] | 4 |
| Phoenix | +0.363 | **−0.095** | 0.137 | [−0.324, **+0.161**] | 11 |
| Denver | +0.131 | **+0.023** | 0.319 | [−0.323, +0.446] | 6 |

Two of the three do not survive. **Phoenix's published +0.363 lies above the
maximum of its own 11-day distribution** — I reproduce +0.363 exactly on
2024-07-22, so it is not a computation error; that day is simply an outlier.
Phoenix's typical ceiling is around −0.10, i.e. *negative*. Denver's daily
ceiling swings from −0.323 to +0.446, so its single-day value is close to
arbitrary. Only Austin is stable, and even there the published day sits near
the top of the range.

The consequence for §III-C is direct: "Phoenix achieves 23% of its ceiling
(+0.083 vs +0.363)" and "Denver achieves 3% (+0.004 vs +0.131)" are both
computed against unrepresentative denominators, and Phoenix's true
denominator is negative, making the percentage undefined (§1).

Across the whole sweep the median per-city sd of the daily ceiling is
**0.085**, with a maximum of 0.319 (Denver). A single-day ceiling therefore
carries roughly ±0.09 of noise — comparable to the spacing between many
cities in the study. **Ceilings should be quoted as multi-day means with a
spread**, never as one day.

---

## 2. Table II reconciled (resolved)

§II-C reports ARCO-recalibrated coefficients while Table II carried the
CDS-derived score and CI. The bootstrap has now been re-run under ARCO
forcing using the identical estimator (6×6 blocks, B = 2000, same seed):

| configuration | test *r* [95% CI] | test RMSE [95% CI] |
|---|---|---|
| paper_v2 Table II as printed | +0.758 [+0.722, +0.788] | 1.42 [1.36, 1.48] |
| CDS coefficients, ARCO forcing | +0.7579 [+0.7225, +0.7880] | 1.4210 [1.3550, 1.4750] |
| **ARCO coefficients, ARCO forcing** | **+0.754 [+0.718, +0.785]** | **1.40 [1.33, 1.46]** |

The middle row reproducing the published interval almost exactly is a good
check on the machinery. **Use the bottom row in Table II**, since §II-C
reports those coefficients.

### 2.1 The paired ERA5-contribution CI (the interval v3 left unstated)

v3 carries the ARCO point estimate (+0.007) and correctly declines to reuse
the CDS interval. Recomputed with the same paired estimator (6x6 blocks,
B = 2000, one shared tile-origin draw per replicate applied to both
configurations, valid masks intersected first):

| | delta r | 95% CI |
|---|---|---|
| v2: CDS coefficients + CDS forcing | +0.0105 | [+0.0051, +0.0160] |
| **ARCO coefficients + ARCO forcing** | **+0.0066** | **[+0.0039, +0.0091]** |

A = ship (ARCO coefficients, ARCO forcing); B = two-stage config under
synthetic diurnal forcing, i.e. the pre-ERA5 baseline.

**The interval excludes zero**, so v2's qualitative reading survives intact:
the ERA5 contribution is positive and statistically distinguishable from
zero, but small in absolute terms, with the principal benefit remaining
methodological. Two differences worth noting: the point estimate is ~40%
smaller than the CDS-era value, and the interval is *narrower* (width 0.005
vs 0.011). Pairing is what makes the test meaningful at all -- each
configuration's own CI is ~0.03 wide against a 0.007 difference, so unpaired
intervals overlap almost completely.

Script: `scripts/run_paired_era5_ci.py`; output `paired_era5_ci_arco.json`.

---

## 3. The PM mechanism split, measured (§III-D still wrong, but differently)

Each variant was *recalibrated* (Stage 2, D pinned, ARCO forcing), because
comparing ET formulations at coefficients fitted for another formulation
confounds the comparison.

| ET formulation | test *r* | vs linear ET |
|---|---|---|
| linear ET (reference) | **+0.7540** | — |
| PM, air VPD, k=3 (canonical) | +0.4110 | −0.3430 |
| PM, air VPD, k=1 | +0.4849 | −0.2690 |
| PM, air VPD, linear LAI | +0.4617 | −0.2922 |
| PM, **surface** VPD, k=3 | +0.4502 | −0.3038 |
| PM, surface VPD, k=1 | +0.5092 | −0.2448 |
| PM, surface VPD, linear LAI | +0.4886 | −0.2654 |
| **big-leaf bulk, k=3** | **+0.5936** | −0.1603 |
| big-leaf bulk, linear LAI | +0.5989 | −0.1551 |

Decomposition of the −0.343 regression:

- LAI fix alone: **+0.051 (15%)**
- VPD fix alone (surface VPD in the bounded PM form): **+0.039 (11%)**
- Both together: **+0.078 (23%)**

Three conclusions, and the first two contradict things the paper says:

1. **"Two mechanisms contribute roughly equally" is wrong, but so was my
   earlier correction.** I previously argued VPD decoupling was ~77% of the
   problem, inferring it as the residual after LAI. Measured directly, the
   two are comparable (15% and 11%) — but **together they account for only
   23%**. Roughly three-quarters of the regression is caused by neither
   named mechanism. The paper's mechanism isolation is incomplete, not
   merely mis-weighted.
2. **The proposed fix does not work.** Substituting T_i-dependent VPD into
   the canonical PM form — exactly what §III-D and §V recommend — recovers
   0.039 of 0.343 and leaves PM 0.30 below linear ET. It is not the path
   forward the paper claims.
3. **The formulation the paper explicitly rejected is the best one.**
   True big-leaf bulk PM reaches **+0.594**, recovering 0.183 — more than
   four times the recommended fix, and more than both named mechanisms
   combined. `penman_monteith.py`'s docstring rejects it for not being
   energy-bounded (it is capped at 1.5×Rn here). It is still 0.16 below
   linear ET, so PM remains a negative result, but the ordering in the paper
   is backwards.

### 3.1 CORRECTION: k_ET is inert under PM, not boundary-pinned

I previously wrote that `k_ET` "runs to 40–55 against a bound of 60, i.e.
the optimizer is at a boundary, so these variants are not cleanly
identified", and flagged loosening the bound as the obvious next experiment.
**That diagnosis was wrong, and the experiment would have been a null.**

`simulator.run` references `cfg.et_coeff` only in the *non*-PM branch. With
`use_pm_et=True` the latent flux comes entirely from
`latent_heat_flux_w_m2`, and `et_coeff` is never read. Measured directly:

| configuration | k_ET = 5 | k_ET = 60 | Δ |
|---|---|---|---|
| PM, surface VPD | +0.349753 | +0.349753 | **0.000e+00** |
| linear ET | +0.7525 | +0.7117 | −0.0409 |

Exactly zero to machine precision under PM; a clear effect under linear ET.
So k_ET was not pinned at a bound — it was an **unidentified** parameter
drifting freely, which also fits the reported values better (7.8, 19.9,
40.7, 40.9, 44.2, 47.0, 54.3, 54.9: scattered across the range, not piled
at 60).

**There was still a real confound underneath it**, just a different one:
differential evolution was searching a 4-dimensional space in which one
dimension did nothing, so the PM variants got roughly three-quarters of the
effective budget linear ET received on the parameters that matter. Refitting
with the free set reduced to the three absorptions is the fair comparison:

| ET formulation | test *r* (refit) | vs earlier |
|---|---|---|
| linear ET (reference) | +0.7540 | — |
| PM air VPD, k=3 (canonical) | +0.4160 | +0.005 |
| PM air VPD, linear LAI | +0.4603 | −0.001 |
| PM surface VPD, k=3 | +0.4483 | −0.002 |
| PM surface VPD, linear LAI | +0.4867 | −0.002 |
| big-leaf bulk, k=3 | +0.5947 | +0.001 |

Decomposition of the −0.338 regression, refit: LAI **13%**, VPD **10%**,
both together **21%**, big-leaf bulk **53%**.

**The split barely moves** (23% → 21%). So the wasted dimension was not
distorting the conclusion, and every §3 finding stands as written. Script:
`scripts/run_pm_identification.py`.

### 3.2 r_s absorbs 61% of the regression — but it is a sink, not an explanation

`r_s_leaf` (per-leaf-area stomatal resistance) has been fixed at the FAO-56
reference-grass value of 100 s/m throughout and never calibrated, making it
the other prime suspect for the ~77% neither named mechanism explains.
Swept with the three absorptions refit at each value:

| r_s_leaf (s/m) | test *r* | RMSE |
|---|---|---|
| 100 *(FAO-56 default)* | +0.4160 | 1.887 |
| 400 | +0.5518 | 1.726 |
| 800 | +0.6222 | 1.897 |
| 1600 | +0.6694 | 1.853 |
| **3200** | **+0.6855** | 1.896 |
| 6400 | +0.6563 | 1.986 |
| 1 000 000 | +0.5263 | 2.345 |

It recovers **+0.206 of the −0.338 regression (61%)** — far more than LAI
(13%) and VPD (10%) combined. But two controls stop that being a fix.

**Control 1: the curve turns over**, peaking at 3200 s/m and declining after.
So this is not simply "r_s → ∞ switches ET off". Worth knowing why the
limit is not the no-ET case: the sweep moves only the *vegetation* stomatal
path, while `r_s_water` (1 s/m) and `r_s_bare` (2000 s/m) stay fixed, so
water and bare cells keep evaporating however large `r_s_leaf` grows.

**Control 2, and it is decisive: no ET at all scores higher than any PM
variant.**

| configuration | test *r* | RMSE |
|---|---|---|
| **no ET at all** (`k_ET = 0`) | **+0.7580** | 1.409 |
| linear ET (k_ET = 8.68) | +0.7546 | — |
| best PM (r_s = 3200) | +0.6855 | 1.896 |
| canonical PM (r_s = 100) | +0.4160 | 1.887 |

Best-PM sits **0.073 below** simply deleting evapotranspiration. Tuning r_s
narrows the gap and never closes it.

So r_s is not the missing explanation — it is a **sink**. It can absorb most
of the regression, but only by taking a value ~32× reference grass, which is
indefensible as a stomatal resistance (stressed canopies reach a few hundred
s/m, not thousands). That it *wants* such a value is a symptom that
something else in the formulation is mis-specified — the LAI mapping, the
parallel bulk-resistance combination, or driving PM from spatially uniform
forcing — and r_s is soaking up the error.

**A separate result worth its own line:** removing ET entirely (+0.7580)
scores marginally *above* the fitted linear ET term (+0.7546), while RMSE is
essentially unchanged (1.409 vs 1.403). The ET term contributes nothing to
spatial pattern skill in this model and may cost a little. That reframes the
whole ET line of work: the question is not which ET formulation is best, but
whether the slab's ET term is doing useful work at all at this resolution.

*Correction to my own output:* the script first reported "PM at its best r_s
beats the no-ET control". It tested `abs(gap) < 0.02` and fell through to
the positive branch for a gap of **−0.073**, i.e. it declared a win when PM
was comfortably losing. Sign bug in my verdict logic, now fixed; the numbers
in the table were never affected.

Scripts: `run_pm_identification.py`, `run_rs_control.py`; outputs
`pm_identification.json`, `rs_control.json`.

## 3.3 Audit of paper_v3 (3), 2026-08-24

Checked every table against the artifacts. **Most of it verifies exactly**:
`tab:arid` (all five cities, ceiling/sd/gap), `tab:subset` (all four rows,
both targets), `tab:gap-verify`, `tab:decay`, the `tab:transfer` ceiling row,
the `tab:bootstrap_ci` ship row, and the abstract's headline figures
(R ≈ +0.46, ρ = +0.694 / +0.844, 27 of 49, Phoenix 4th, ρ = −0.55 / +0.50).
Four issues, in severity order.

### (1) HIGH — the transfer scores use the CDS coefficients, not the ARCO ones

§II-C reports the ARCO-recalibrated ship coefficients, and `tab:transfer`'s
ceiling row is ARCO-derived. But the transfer *scores* in that table, and
the three zero-shot CIs in `tab:bootstrap_ci`, come from
`outputs/transfer_*.json`, whose stored `austin_coefs` are the **CDS**
values:

| coefficient | CDS (used by transfer) | ARCO (reported in §II-C) | Δ |
|---|---|---|---|
| α_imp | 0.8032 | 0.8001 | −0.0031 |
| **α_veg** | **0.4343** | **0.4641** | **+0.0297** |
| α_wat | 0.1784 | 0.1756 | −0.0028 |
| **k_ET** | **10.975** | **8.677** | **−2.298** |

So the paper's central demonstration was not re-run with the coefficients
the paper states. α_veg moves ~7%, and vegetation absorption drives exactly
the vegetation/impervious contrast transfer depends on, so this is not
obviously negligible. I could find no disclosure of it in the text.

Re-running zero-shot transfer for the three cities under ARCO coefficients
and ARCO forcing would settle it; it needs ARCO forcing windows for Phoenix,
Miami and Denver (~45 s each) plus a handful of sims.

### (2) MEDIUM — `tab:pm` mixes refit and non-refit rows

The caption states every variant was recalibrated with k_ET excluded from
the search. Two rows predate that: **PM air VPD K=1 (+0.485)** and **PM
surface VPD K=1 (+0.509)** are from the original sweep, in which k_ET *was*
in the search. The other rows (canonical +0.416, big-leaf +0.594, r_s = 3200
+0.686) are refit values. Either re-run the two K=1 variants under the same
protocol or qualify the caption.

### (3) LOW — two values in `tab:predictors` come from the superseded artifact

The univariate column is drawn from `ceiling_study.jsonl` rather than the
settled `full_sweep.jsonl` for two covariates:

| covariate | paper | settled (`full_sweep`) | superseded (`ceiling_study`) |
|---|---|---|---|
| Water fraction | +0.521 | **+0.517** | +0.521 |
| LST spread | +0.362 | **+0.352** | +0.362 |

Both match the old artifact to three decimals, which confirms the
provenance. The other six covariates match the settled run exactly. Small,
but it is precisely the mixed-artifact drift the single-pass sweep was meant
to eliminate.

### (4) LOW — the abstract is stronger than its own body on stability

The abstract calls the 2–3× stability advantage "a positive finding for the
redefinition rather than a caveat on it", without the sample-wide context.
The body (lines ~347–357) states it properly: median per-city sd is 0.085
univariate versus 0.081 multivariate across 49 cities, "essentially a wash",
with the 2–3× tightening "confined to Phoenix and Denver specifically". The
body is exemplary here; a reader of the abstract alone would over-generalise.

Minor: the `tab:transfer` caption says *n* = 9–11 days, but Miami has 5. The
surrounding text does say so; the caption alone does not.

---

---

## 4. The diffusion trade-off is real — this REPLACES my earlier §4

**My previous claim was wrong.** I reported that lifting the D search floor
reaches the literature half-cooling target (197 m at D = 0.135) and might
dissolve two stated limitations. That rested on two errors:

- it used the **legacy, biased** estimator I was simultaneously arguing
  should not be trusted; and
- it varied D with coefficients **held fixed**, whereas the ship procedure
  refits them at each D. Refitting changes everything: the optimizer
  compensates for weak diffusion by driving `k_ET` from 5 up to 58 (near its
  bound), which suppresses the patch cooling the earlier sweep showed.

### 4.1 Two estimator problems, both fixed

- **Fit support depended on D.** The legacy exponential fit only used bins
  clearing a fixed 0.05 °C floor, and that set grows with D, so every
  apparent non-monotonicity coincided exactly with a change in `n_bins_fit`.
- **The quantity was not comparable to the literature.** The patch has a
  **1000 m radius** and cooling is roughly flat across its interior, so a
  centre-referenced half-cooling distance cannot fall below ~1000 m once the
  plateau resolves. Published green-space cooling distances are measured
  outward from the park boundary. `half_cooling_from_edge_m` now measures
  that, and it is the only version comparable to the 100–300 m band.

The edge-referenced estimator is **monotonic in D**; legacy still shows 3
reversals and centre-referenced 2.

### 4.2 The Pareto front (this delivers contribution #2)

Stage 2 refit at every D, ARCO forcing, held-out test days:

| D | test *r* | edge half-cooling |
|---|---|---|
| 0.050 | +0.6583 | 293 m |
| 0.199 | +0.6676 | 297 m |
| 0.501 | +0.7128 | 309 m |
| 0.794 | +0.7416 | 320 m |
| 1.259 | +0.7722 | 337 m |
| 1.996 | +0.7992 | 356 m |
| 3.165 | +0.8228 | 390 m |
| 5.018 | +0.8404 | 463 m |
| 12.615 | **+0.8526** | 628 m |
| 20.000 | +0.8523 | 727 m |

- Best *r* inside the 100–300 m band: **+0.6676** at D = 0.199.
- Best *r* overall: **+0.8526** at D = 12.6.
- **Cost of staying inside the band: −0.185 in *r***, not the 0.09 the paper
  reports. The trade-off is real and about twice as steep as stated.
- The published D = 1.0 gives edge half-cooling **~330 m** — just outside the
  band, and much closer to it than the reported 366 m suggests. **The 366 m
  figure should not be compared to the literature band at all**; it is a
  centre/fit-based number.

So §V's framing survives, and contribution #2 is now an actual curve rather
than two points. It is also cheap to regenerate: `scripts/run_decay_pareto.py`.

---

## 5. The thermal-gap rule at n = 49 (was n = 4)

paper_v2 §III-C states the built−vegetated gap "rank-orders the ceiling
perfectly across all tested cities". Run over **49 US cities** (55 attempted,
6 dropped for insufficient clear days), with each city's ceiling a multi-day
mean:

| covariate | Pearson | Spearman | p |
|---|---|---|---|
| **built−vegetated LST gap** | **+0.827** | **+0.844** | 2.7e-14 |
| water fraction mean | +0.383 | +0.521 | 0.00013 |
| mean LST | −0.613 | −0.465 | 0.00076 |
| impervious fraction mean | −0.238 | −0.377 | 0.0076 |
| vegetation fraction mean | +0.229 | +0.344 | 0.015 |
| LST spread (p98−p2) | +0.350 | +0.362 | 0.011 |
| impervious fraction σ | −0.495 | −0.323 | 0.024 |
| class entropy | +0.083 | −0.029 | 0.84 |

**The hypothesis holds strongly but the "perfectly" claim does not.**
Spearman ρ = +0.844 at p = 2.7e-14 — the gap is comfortably the best single
predictor, explaining R² = 0.684. But only **85% of city pairs (994/1176)
are concordant**, and rank ordering is not perfect. With n = 4 a perfect ordering
carries almost no evidence (1-in-24 by chance), so this is the claim being
properly tested rather than overturned. Recommend restating as a strong
monotone association with the measured ρ and p, and dropping "perfectly".

Largest violations: Los Angeles (gap +5.13, ceiling +0.518) ranks below
St. Louis (gap +1.22, ceiling +0.573) and Seattle (gap +1.23, +0.524).
Coastal Mediterranean cities are the systematic exception — large thermal
gap, mid-range ceiling.

A four-predictor fit reaches **R = 0.934 (R² = 0.873)**, up from 0.684 for
the gap alone — so the gap is the leading term but not the whole story.
Notably `impervious_std` enters with a **negative** coefficient: more
land-cover heterogeneity predicts a *lower* ceiling, reinforcing that the
constraint is thermal separability rather than spatial variety.

### 5.0 The cloud-dropout bias, tested and resolved

The four-date version dropped 18 of 55 cities for cloud cover, and dropout
is not independent of the variable under study: humid cities cloud out far
more than arid ones. That left a sample skewed toward the arid tail which
drives the association — a genuine threat to the headline result.

Fixed two ways. The candidate pool went from 4 dates to **12** across the
warm season, cutting dropout from 18 cities to **6** and lifting the sample
from 37 to **49**. Every one of the 12 recovered cities is humid or eastern —
Chicago, Milwaukee, Detroit, Pittsburgh, Columbus, St. Louis, Memphis,
Nashville, Atlanta, Birmingham, Charlotte, Raleigh — confirming the dropout
really was climate-correlated rather than random.

Then the association was refit on subsets, which tests the objection head-on
rather than merely reducing it:

| subset | n | Spearman ρ | p |
|---|---|---|---|
| all cities | 49 | +0.844 | 2.7e-14 |
| excluding negative-ceiling cities | 44 | +0.785 | 2.8e-10 |
| excluding the 5 most arid | 44 | +0.810 | 2.6e-11 |
| **humid half only** (below-median mean LST) | **24** | **+0.797** | **3.2e-06** |

Restricted to the humid half — no desert cities at all — ρ is still +0.797
at p = 3.2e-06, and removing the negative-ceiling cities leaves ρ = +0.785.

**Important qualification, added later: this holds against the *univariate*
ceiling only.** Against the multivariate ceiling the same humid-half subset
gives ρ = +0.361 at p = 0.083, which is not significant. See §5.2 — that
supersedes the stronger claim originally made here, and the two must not be
quoted independently.

### 5.1 The strongest new result: arid cities *invert*

Five cities have **negative** ceilings — impervious cover anti-correlates
with LST:

| city | ceiling (multi-day) | sd | gap | n days |
|---|---|---|---|---|
| Boise | −0.331 | 0.075 | −0.65 °C | 11 |
| Albuquerque | −0.308 | 0.256 | **−1.65 °C** | 10 |
| Tucson | −0.254 | 0.278 | −0.73 °C | 9 |
| Phoenix | −0.095 | 0.137 | −0.21 °C | 11 |
| Billings | −0.034 | 0.241 | −0.03 °C | 10 |

These are multi-day means, so the inversion is not a single-day accident.
It generalises the Phoenix narrative considerably: it is not that the
land-cover→temperature mapping *weakens* in arid cities — it **reverses**.
Note Phoenix itself is on this list once averaged, which the paper's
single-day +0.363 conceals entirely.
Natural desert surfaces classified as shrubland/vegetation run hotter than
built-up areas (which carry irrigation, shade and thermal mass). That is a
much stronger and more quotable statement than "class separability
collapses", and it is measured across five independent cities.

---

## 5.2 UNRESOLVED: the heuristic is validated against the metric the paper calls unsound

v3 argues the univariate ceiling is unsound (it goes negative in arid
cities), then validates the thermal-gap heuristic against the univariate
ceiling, justifying it as "the metric feasible to average over many dates at
this scale". **That justification is factually wrong** — `ceiling_multi_r`
is computed and averaged for all 49 cities and is already in
`ceiling_study.jsonl`. Swapping the target changes the result materially:

| target of the gap heuristic | Spearman ρ | 95% CI | p | concordant |
|---|---|---|---|---|
| univariate ceiling *r* (what v3 uses) | **+0.844** | [+0.694, +0.933] | 2.7e-14 | 85% |
| **multivariate ceiling *R*** (what v3 calls sound) | **+0.691** | [+0.487, +0.834] | 3.8e-08 | 76% |

The individual CIs overlap, so the difference has to be tested on shared
resamples rather than eyeballed. A **paired** bootstrap (B = 5000, same city
draws applied to both targets — their sampling noise is highly correlated,
being the same cities and the same imagery) gives

  Δρ = ρ(uni) − ρ(multi) = **+0.153, 95% CI [+0.034, +0.287]**

which **excludes zero**. The univariate target does give a significantly
stronger association; this is not an artefact of reading two overlapping
intervals.

And the robustness check that defends against the arid-tail objection does
**not** survive the swap:

| subset | n | ρ vs univariate | ρ vs multivariate |
|---|---|---|---|
| all cities | 49 | +0.844 (p 2.7e-14) | +0.691 (p 3.8e-08) |
| excluding negative univariate ceilings | 44 | +0.785 (p 2.8e-10) | +0.612 (p 1.0e-05) |
| excluding 5 most arid | 44 | +0.810 (p 2.6e-11) | +0.643 (p 2.5e-06) |
| **humid half only** | 24 | **+0.797 (p 3.2e-06)** | **+0.361 (p 0.083, n.s.)** |

The four-predictor fit also falls from R = 0.934 to **0.773** (R² 0.873 →
0.597); the gap alone falls from R = 0.827 to 0.663 (R² 0.684 → 0.439).

Two covariates behave differently across targets and are worth a line in the
paper: **water fraction** is the second-strongest predictor and *strengthens*
against the multivariate ceiling (ρ +0.521 → +0.586), as does **LST spread**
(+0.362 → +0.481). **Impervious σ** and **vegetation mean** lose significance
entirely (CIs cross zero). Class entropy is null against both.

Publication-ready outputs: `outputs/gap_dual_target.tex` (drop-in table with
bootstrap CIs and the paired Δρ row) and `gap_dual_target.json`.
Script: `scripts/run_dual_target_analysis.py`.

**This corrects something I asserted earlier.** I previously wrote that the
association "is not an artifact of the arid tail" on the strength of the
humid-half result. That holds against the univariate ceiling only. Against
the metric the paper itself argues for, the humid-subset association is
**not statistically significant**. Part of the univariate strength is
mechanical: both the gap and the univariate ceiling flip sign in arid
cities, so they co-move for a shared reason.

Related: **no city has a negative multivariate ceiling** (min +0.223 across
49). The "arid inversion" is a property of the built−vegetated gap and of
the *univariate* relationship — a three-channel fit absorbs the sign flip.
The physical statement (desert vegetation runs hotter than built surfaces)
stands on the gap, which is a direct measurement; but "five cities have
negative ceilings" is true only under the definition v3 rejects.

Three honest options, in order of preference:

1. **Report both targets side by side** and state plainly that the gap
   predicts the univariate ceiling strongly and the multivariate one
   moderately. Costs nothing, and pre-empts the obvious reviewer question.
2. **Reframe the heuristic** as a screen for *impervious-only* separability,
   which is what it actually tracks, and keep the univariate metric with
   that scope stated.
3. Keep as-is but delete the "feasible at scale" justification (false) and
   soften the humid-robustness claim, which does not hold for the sound
   metric.

Also fix: Denver's bullet lists gap +0.28 °C (single day); its multi-day
mean gap is **−0.22 °C**, i.e. Denver inverts too once averaged.

---

## 5.3 Review of paper_v3 (3): two stale figures

v3(3) handles the §5.2 tension well — both targets reported, the paired test
described correctly, and the humid-half claim narrowed rather than dropped.
Two numbers in it are still stale, and the first traces to a bug.

**`R > 0.6` (abstract line 39, §II line 215).** The claim is that the
univariate ceiling can be negative "even where a three-channel linear fit
reaches $R>0.6$". Checked against every run:

| run | max multivariate *R* among negative-univariate cities |
|---|---|
| multi-day, n = 49 (current) | **+0.462** (Boise) |
| single-day, multi-tile (superseded) | +0.580 (Tucson) |
| single-day, single-tile (superseded) | +0.633 (Tucson) |

Only the last exceeds 0.6, and that run predates the WorldCover tile fix
(§9.1) — i.e. the claim rests on a number produced by a bug. Defensible
wording is *R* ≈ **0.46**, which is still a striking contrast against a
negative univariate value.

**Denver's gap (§III-C bullet, line 368) reads +0.28 °C.** *Superseded by
§5.5 — and I overstated this.* I first reported the multi-day mean as
−0.22 °C (summer pool, n = 6) and said the sign flips. On the July pool
matching Denver's own transfer window it is **+0.17 ± 1.10 °C** (n = 9),
range [−1.16, +1.95]. Both estimates sit near zero with a spread far larger
than the mean, so the honest statement is that **Denver's gap is not
reliably signed**, not that it inverts. The published +0.28 is not wrong so
much as meaningless on its own.

---

## 5.4 RESOLVED: the multivariate ceiling is both sounder AND more stable

v3(3) flags an open caveat — the three-channel ceiling "has not been
independently re-verified against a multi-day pool", with "no mechanistic
reason to expect it is more stable". Now measured. Each focus city gets a
pool matching the season of its *own* transfer days (Austin Aug, Phoenix
Jul, Denver Jul, Miami Apr), because a ceiling used as a denominator has to
describe the same period as the numerator.

| city | n | uni published | uni multi-day | multi published | multi multi-day |
|---|---|---|---|---|---|
| Austin | 8 | +0.671 | +0.580 ± 0.125 | +0.704 | **+0.613 ± 0.114** |
| Phoenix | 11 | +0.363 | −0.130 ± 0.199 | +0.412 | **+0.271 ± 0.085** |
| Denver | 9 | +0.131 | +0.068 ± 0.258 | +0.395 | **+0.437 ± 0.085** |
| Miami | 5 | +0.827 | +0.831 ± 0.025 | +0.843 | **+0.841 ± 0.025** |

**The speculation was wrong, in the paper's favour.** The multivariate
ceiling is markedly *more* stable day to day, exactly where it matters:

| city | uni sd | multi sd | ratio |
|---|---|---|---|
| Denver | 0.258 | **0.085** | 0.33 |
| Phoenix | 0.199 | **0.085** | 0.43 |
| Austin | 0.125 | 0.114 | 0.91 |
| Miami | 0.025 | 0.025 | 1.02 |

For the two volatile cities the multivariate estimate is 2–3× tighter. That
makes sense: three channels can absorb day-to-day shifts in which surface
type dominates the LST field, whereas a single impervious-only correlation
swings with them. So the multivariate definition is not merely sounder in
principle — it is the better-behaved estimator in practice, and the caveat
can be replaced with this as a positive result.

**The percentages barely move**, which is the reassuring part:

| city | model | vs published single-day multi | vs multi-day multi |
|---|---|---|---|
| Phoenix | +0.083 | 20% | **31%** |
| Denver | +0.004 | 1% | **1%** |
| Miami | +0.832 | 99% | **99%** |

Miami's 99% survives on a tight estimate (sd 0.025), so "essentially at the
ceiling" is now properly supported rather than asserted. Denver is unchanged
at 1%. Phoenix rises from 20% to 31% but remains far below its ceiling, so
the qualitative reading holds.

One caveat worth carrying: **Austin's published +0.671 is near the top of
its own August range** (multi-day mean +0.580), so the source city's ceiling
was optimistic too — less dramatically than Phoenix, but in the same
direction. Every published single-day ceiling in this paper happened to land
on a favourable day.

Miami rests on only 5 clear April days; the other three have 8–11. Its sd is
the smallest of the four, so this is unlikely to matter, but the *n* should
be stated.

Script: `scripts/run_focus_city_ceilings.py`; output
`outputs/focus_city_ceilings.json` (includes every per-day value).

*Checked and cleared:* Austin's 2024-08-19 and 2024-08-22 both display as
+0.671, which looked like a cache collision. They are 0.6707031 and
0.6706773 — genuinely distinct rasters with distinct checksums, identical
only after rounding.

---

## 5.5 The NDVI result reverses under multi-day averaging

The paper's abstract states NDVI "raises Denver's ceiling by 47% while
helping Phoenix only marginally, inverting our original expectation."
Recomputed over each city's own seasonal pool, that reversal does not hold —
**the original expectation was right**.

| city | n | 3-channel | + NDVI | gain (multi-day) | sd | range | published (1 day) |
|---|---|---|---|---|---|---|---|
| **Phoenix** | 11 | +0.271 | +0.496 | **+0.2241** | 0.0785 | [+0.049, +0.309] | +0.0492 |
| Denver | 9 | +0.437 | +0.650 | +0.2129 | 0.1367 | [+0.000, +0.353] | +0.1863 |
| Austin | 8 | +0.613 | +0.651 | +0.0372 | 0.0094 | [+0.023, +0.047] | +0.0228 |
| Miami | 5 | +0.841 | +0.842 | +0.0012 | 0.0011 | [+0.000, +0.003] | +0.0030 |

Verdicts, using ±2 sd:

| city | gain ±2 sd | verdict |
|---|---|---|
| **Phoenix** | [+0.067, +0.381] | **worth integrating** — the only robustly positive case |
| Denver | [−0.061, +0.486] | not distinguishable from zero |
| Austin | [+0.018, +0.056] | positive but modest |
| Miami | [−0.001, +0.003] | not distinguishable from zero |

**Phoenix's published +0.0492 is the *minimum* of its own 11-day range.**
The single day the paper happens to use is Phoenix's least favourable, which
is what produced the "helps Denver, not Phoenix" reversal. Averaged, Phoenix
gains most (+0.224) *and* is the only city whose gain is clearly separated
from zero. Denver's mean gain is similar (+0.213) but swings from +0.000 to
+0.353 — too variable to claim.

So the physical story reverts to the intuitive one that §V originally
proposed: **an irrigation-sensitive channel does help the irrigated-desert
city.** That is a cleaner narrative than the current one, but the abstract,
§III-C and §V all assert the inverted version and need rewriting.

Note this is the *opposite* sampling direction from the ceilings (§5.4),
where every published day was favourable. Here the published day understates
the effect. Single-day estimates are not biased in a consistent direction —
they are simply noisy, which is why the fix has to be averaging rather than
picking a better day.

### 5.5.1 The gap, re-verified

| city | n | published (1 day) | multi-day | sd | range | sign flip |
|---|---|---|---|---|---|---|
| Austin | 8 | +2.11 | +2.21 | 0.40 | [+1.34, +2.67] | |
| **Phoenix** | 11 | +1.35 | **−0.36** | 0.71 | [−1.26, +1.35] | **YES** |
| Denver | 9 | +0.28 | +0.17 | 1.10 | [−1.16, +1.95] | |
| Miami | 5 | +8.37 | +8.75 | 0.68 | [+8.16, +9.78] | |

Austin and Miami are stable and their published values stand. **Phoenix's
gap changes sign**: published +1.35 is the *maximum* of its range, and the
multi-day mean is −0.36. That matters twice over — it puts Phoenix properly
among the inverting arid cities (§5.1, where it already appears on the
ceiling criterion), and it removes the apparent inconsistency of a city with
a positive gap having a negative ceiling.

Denver's gap is dominated by its own spread (±1.10 against a mean of +0.17)
and should be reported as indeterminate rather than given a sign.

Script: `scripts/run_gap_ndvi_multiday.py`; output `gap_ndvi_multiday.json`.

---

## 5.6 SETTLED: one end-to-end sweep, all numbers from one artifact

Everything below comes from a single pass (`run_full_sweep.py` ->
`full_sweep.jsonl` -> `analyze_full_sweep.py`) on the final pipeline:
multi-tile land cover, grid-keyed caches, debiased estimators, 12-date pool,
multi-day means. **49 cities, 6 skipped for cloud, 0 failures.** Median 6
clear days per city (range 3-12). This supersedes §5.2, §5.4 and §5.5 wherever
numbers differ; the earlier sections are kept for the reasoning.

**The gap heuristic replicates exactly.** Computed from one artifact rather
than three, the numbers are unchanged to within 0.003:

| target | ρ | 95% CI | p | concordant |
|---|---|---|---|---|
| univariate *r* | +0.844 | [+0.694, +0.934] | 2.6e-14 | 85% |
| multivariate *R* | **+0.694** | [+0.489, +0.837] | 3.3e-08 | 77% |

Paired difference +0.151 [+0.033, +0.287], excludes zero. Humid half against
the multivariate target: ρ = +0.361, p = 0.083, still not significant. So
every §5.2 conclusion stands.

### 5.6.1 Correction: the multivariate ceiling is NOT broadly more stable

§5.4 found it 2-3x more stable, and v3's abstract now says so. **That does
not generalise.** Across all 49 cities:

| | median per-city sd |
|---|---|
| univariate *r* | 0.085 |
| multivariate *R* | 0.081 |

Essentially a wash. The 2-3x advantage is real but confined to Phoenix and
Denver — two cities whose *univariate* estimate is unusually unstable, not a
general property of the multivariate one. The abstract's hedge ("for exactly
the cities where it matters") is defensible as written, but it should not be
read as a sample-wide claim, and §5.4's framing was too broad. The honest
version: the multivariate ceiling is no less stable in general, and markedly
more stable in the specific cities where the univariate one misbehaves.

### 5.6.2 The NDVI screen at 49 cities: a rule, not an anecdote

The four-city screen could only say "it helps Denver / it helps Phoenix."
Across 49 cities the gain is **predictable**, which is the claim the
actionability argument actually needs:

| predictor of NDVI gain | ρ | p |
|---|---|---|
| **multivariate ceiling** | **−0.546** | 4.9e-05 |
| **mean LST (aridity proxy)** | **+0.498** | 2.7e-04 |
| vegetation fraction | +0.277 | 0.054 |
| built−vegetated gap | −0.202 | 0.165 |

**The channel helps precisely where the ceiling is low** (ρ = −0.546) **and
where the city is hot and arid** (ρ = +0.498). That is a general screening
rule, far stronger than a per-city verdict, and it is exactly the form
"the diagnostic is actionable" should take.

Median gain +0.0642, range [+0.0027, +0.2974]. **27 of 49 cities are
robustly positive** (gain − 2sd > 0).

| rank | city | gain | sd | robust |
|---|---|---|---|---|
| 1 | Bakersfield | +0.2974 | 0.094 | yes |
| 2 | Spokane | +0.2603 | 0.082 | yes |
| 3 | Boise | +0.2384 | 0.030 | yes |
| **4** | **Phoenix** | **+0.2210** | 0.062 | yes |
| 5 | Salt Lake City | +0.2197 | 0.074 | yes |
| 6 | Denver | +0.1794 | 0.148 | **no** |
| … | Dallas / St Louis / Seattle / Buffalo | ~+0.003 | — | — |

**Correction to the abstract.** v3 says NDVI "helps Phoenix — the
irrigated-desert city — most and most reliably". Phoenix ranks **4th of
49**; Bakersfield, Spokane and Boise all gain more. The defensible claim is
that the gain concentrates in arid and semi-arid western cities — Phoenix
among them — and is predictable from aridity and from ceiling height. That
is a better result than the Phoenix-specific one, but it is a different
sentence.

Denver is confirmed **not** robustly positive (sd 0.148 against a mean of
+0.179), consistent with §5.5.

### 5.6.3 Covariates, settled

Against the multivariate target, only four covariates survive:

| covariate | ρ (multi) | p |
|---|---|---|
| built−vegetated gap | +0.694 | 3.3e-08 |
| water fraction | +0.583 | 1.1e-05 |
| LST spread | +0.475 | 5.6e-04 |
| mean LST | −0.418 | 0.0028 |
| impervious σ | −0.261 | 0.071 (n.s.) |
| impervious mean | −0.188 | 0.195 (n.s.) |
| vegetation mean | +0.136 | 0.352 (n.s.) |
| class entropy | +0.063 | 0.669 (n.s.) |

Note **impervious σ is no longer significant** against the sound target
(p = 0.071). §5's claim that "more land-cover heterogeneity predicts a lower
ceiling" holds only against the univariate target and should be dropped or
scoped.

Artifacts: `full_sweep.jsonl` (per-city, per-day, every quantity),
`full_sweep_analysis.json`, `full_sweep_tables.tex`.

---

## 5.7 GLOBAL EXTENSION: inversion is climatic — but the US cities are definition-fragile

28 cities on five continents, every one defined identically as a 25 km disc
(OSM administrative units are not comparable across countries: "Riyadh" is a
375,478 km2 province, "Perth" a 6 km2 CBD council, "Alice Springs" 1 km2 —
four grid cells). 10 US cities are included under the *same* definition as a
bridge. Artifact `global_extension.jsonl`.

### 5.7.1 The question is answered: climatic, not continental

**Nine cities outside the US invert**, on four continents — Amman (−0.429),
Riyadh (−0.342), Marrakesh (−0.156), Antofagasta (−0.124), Hermosillo
(−0.116), Madrid (−0.102), Alice Springs (−0.096), Cairo (−0.082), Mendoza
(−0.032).

| group | n | inverting | median ceiling | median gap |
|---|---|---|---|---|
| arid, non-US | 11 | **8** | −0.096 | −0.99 |
| arid, US | 8 | 2 | +0.049 | +0.55 |
| humid/temperate, non-US | 4 | 1 | +0.681 | +4.55 |
| humid/temperate, US | 5 | 0 | +0.615 | +2.47 |

Arid vs humid ceilings differ strongly (median −0.032 vs +0.615,
Mann-Whitney **p = 2.4e-04**). Arid US vs arid non-US do **not** differ
significantly (p = 0.075) — no continental effect detected. That is the test
the all-US sample could not run, and it comes out the right way. The
gap-ceiling association also replicates here (univariate ρ = +0.916,
p = 7.9e-12; multivariate ρ = +0.571, p = 0.0015), as does the aridity link
(mean LST vs ceiling ρ = −0.693, p = 4.3e-05).

*Note p = 0.075 is failure to reject, not proof of no difference, at n = 8
vs 11. The non-US arid group is genuinely more arid and inverts more often
(8/11 vs 2/8), which is what the aridity mechanism predicts.*

### 5.7.2 The awkward part: 4 of 5 published inverting cities flip

Running the same US cities under both definitions:

| city | OSM ceiling | disc ceiling | OSM gap | disc gap | flips |
|---|---|---|---|---|---|
| Phoenix | −0.095 | **+0.116** | −0.21 | +0.46 | **YES** |
| Tucson | −0.254 | **+0.084** | −0.73 | +0.64 | **YES** |
| Albuquerque | −0.308 | **+0.008** | −1.65 | +0.33 | **YES** |
| Boise | −0.331 | **+0.036** | −0.65 | +0.92 | **YES** |
| Billings | −0.034 | −0.101 | −0.03 | −0.61 | no |

**Four of the paper's five inverting cities stop inverting under a
reasonable alternative city definition.** Only Billings survives. Meanwhile
the genuinely hyper-arid non-US cities invert strongly and unambiguously
(Antofagasta gap −3.38 °C, Alice Springs −3.11 °C).

So the mechanism is real, and it is climatic — but **the specific US
examples the paper leads with sit near the sign boundary**, and their
inversion depends on drawing the city at its administrative limits rather
than as a fixed-radius disc. `tab:arid` as it stands is true only under the
OSM definition, and a reviewer who redefines the city gets a different table.

I do not have a confirmed mechanism for the flip. The obvious guess — that a
disc adds surrounding bare desert — does not obviously work, since
`build_worldcover_landuse` already folds class 60 (bare) into the vegetation
fraction at half weight, which should make vegetation *hotter* and the gap
more negative, not less. The ≥50% thresholds defining "built" and
"vegetated" cells change composition substantially between the two extents,
which is the likeliest route, but that is unverified.

### 5.7.3 What to do about it

The strongest honest version of the claim gets *better*, not worse:

1. **Lead with the global result.** Inversion demonstrated on genuinely arid
   cities across Arabia, the Levant, North Africa, the Atacama, Mexico and
   central Australia, under one uniform definition, is a far stronger claim
   than five American Southwest cities.
2. **Report the definition sensitivity explicitly** for the US cities rather
   than waiting for a reviewer to find it. It is a robustness caveat, not a
   refutation — Billings holds, and the non-US cities hold decisively.
3. **Keep Palm Springs as the counter-example**: no inversion (+0.255) and
   the largest positive gap in the sample (+6.34 °C), in the same desert as
   Yuma, because its landscape is irrigated. That is the mechanism working
   as predicted, and it is definition-robust.

### 5.7.4 Caveats

- 7 of 35 cities dropped for coverage: Dubai, Kuwait City, Jodhpur (Gulf and
  Thar dust haze trips MODIS QC — Dubai's best date reached 56.4 % against a
  60 % threshold), and Bangkok, Manila, Lagos, Sydney (tropical cloud). The
  threshold was **not** relaxed to rescue them; tuning an inclusion criterion
  to admit specific cities is how samples get biased.
- The humid non-US control group is consequently thin (n = 4), and several
  cities rest on few days: Kuala Lumpur 2, São Paulo 3, Houston 3, Austin 4.
- Madrid is grouped as temperate but is Köppen BSk (semi-arid); its
  inversion is consistent with the aridity mechanism, not a counter-example.
- Disc results are **not** interchangeable with the OSM-boundary 49-city
  numbers. Any table must say which definition it uses.

Scripts: `run_global_extension.py`, `disc_boundary.py`,
`preflight_boundaries.py`, `analyze_global_extension.py`.

---

## 6. Channel pre-screen: NDVI, and it points somewhere unexpected

§V proposes adding a channel distinguishing irrigated from natural
vegetation (NLCD irrigated crops, or a satellite ET product). Two problems
with those specific choices: NLCD is **US-only**, so it cannot support a
global study, and a satellite ET product is partly *derived from* MODIS LST,
so feeding it back in risks inflating the ceiling circularly.

MODIS NDVI (MOD13A1, 500 m, 16-day) avoids both — global, reflectance-based,
zero-auth via the same Planetary Computer STAC already in use. And because
the ceiling is a property of the data, a candidate channel can be evaluated
with **no model run at all**: recompute the ceiling with it included.

| city | 3-channel ceiling | + NDVI | gain | verdict |
|---|---|---|---|---|
| **Denver** | 0.3954 | **0.5818** | **+0.1863** | worth integrating |
| Phoenix | 0.4125 | 0.4617 | +0.0492 | marginal |
| Austin | 0.7035 | 0.7264 | +0.0228 | marginal |
| Miami | 0.8430 | 0.8461 | +0.0030 | will not help |

**This inverts the paper's expectation.** §V frames the missing channel as a
fix for Phoenix's irrigated desert scrub — but NDVI helps Phoenix only
marginally (+0.049) while lifting **Denver** by +0.186, from 0.395 to 0.582.
Denver moves from the weakest city in the study to mid-pack. Miami gains
nothing, as expected for a city already at its ceiling.

Two things follow. First, the actionable-diagnostic claim is now
demonstrated rather than asserted: the pre-screen costs one MODIS read per
city and gives a differentiated, falsifiable answer per city. Second,
Denver — which the paper omits entirely (§1.4 of the previous notes) — is
where the proposed remedy actually pays off.

*Denver's missing tile is now fixed* (§9.1). All Denver numbers above use
the mosaicked N39W105 + N39W108 pair, i.e. the whole city. That correction
matters here: it raised Denver's three-channel ceiling from 0.149 to 0.395,
which roughly halves the apparent NDVI gain (+0.353 → +0.186). The verdict
is unchanged — Denver is still the one city where the extra channel clearly
pays — but the effect is a 47% lift rather than a tripling.

---

## 7. Advection with ERA5 wind: strong negative result

| variant | test *r* | test RMSE |
|---|---|---|
| no advection, dt = 600 s | +0.7540 | 1.4031 |
| no advection, dt = 120 s | +0.7561 | 1.4030 |
| advection 25%, dt = 120 s | **−0.1057** | 2.1315 |
| advection 50%, dt = 120 s | **−0.2478** | 2.1296 |
| advection 100%, dt = 120 s | CFL violated | — |

**Cost first:** max (|u|+|v|) over the calibration window is 7.04 m/s, so the
advective condition |u|dt/dx ≤ 1 caps dt at **71 s** at 500 m — **2,432 steps
per 48 h instead of 288**, an 8.4× cost before any accuracy question.

**Then the result:** advection does not just fail to help, it destroys
skill — *r* goes negative even at 25% strength. The timestep is not the
cause (dt 600 → 120 with advection off moves *r* by +0.002).

The reason is physical, and it is a caveat the paper's proposal does not
address: this advects the **surface** field, and a slab's ground does not
move. A spatially uniform wind translates the whole UHI pattern downwind,
decorrelating it from the static land cover that generates it. What a wind
actually transports is *air*, which then exchanges heat with the surface —
a two-layer process. So "replace Fickian diffusion with advection-diffusion
driven by ERA5 wind" is under-specified as written; done literally on the
slab it is much worse than what exists. A coupled air layer would be a
substantially larger change than §V implies.

---

## 8. Does the absolute bias couple to skill? Yes, but weakly

§V lists the ~7 °C bias under honest limitations, and the anomaly-based
results invite the assumption that it cancels. It does not fully — the ET
term `k_ET · f_veg · max(T − T_air, 0)` is nonlinear with a clamp, so an
offset moves the model onto a different part of it. Sweeping `lw_coeff`:

| lw_coeff | mean bias | test *r* |
|---|---|---|
| 4.0 | −1.94 °C | +0.7577 |
| 8.0 (ship) | −2.84 °C | +0.7540 |
| 16.0 | −4.11 °C | +0.7455 |
| 24.0 | −4.90 °C | +0.7339 |

*r* does move with the bias, confirming the coupling is real — but the total
swing is **0.024** across a 3 °C change, which is *below* the bootstrap CI
half-width (~0.033). So it is a second-order effect: worth one honest
sentence rather than a rewrite, and worth noting the sign (smaller bias →
marginally higher *r*). The same sensitivity would apply to any
T_i-dependent PM formulation, so it interacts with §3.

---

## 9. New bug found: multi-city raster cache collisions (fixed)

Surfaced only at scale, and it could have corrupted results silently.

- **MODIS** cached by CRS + date. Austin, Dallas and Oklahoma City are all
  EPSG:32614, so they **overwrote each other's** warped rasters.
- **WorldCover** cached by tile (+ resolution after my earlier edit). Any two
  cities sharing a 3° tile collided.

With the original four cities every key happened to be unique, so this was
invisible. In the sweep it threw shape errors — but two cities with
coincidentally equal grid shapes would have silently swapped data. Both now
key on `grid.grid_key()` (CRS, resolution, shape, origin), covered by
`tests/test_cache_keys.py`.

Also fixed while here: a NaN-gap bug in my own half-cooling interpolation
that made four unrelated D values return an identical 875.0 m.

### 9.1 Second cache-class bug: cities straddling WorldCover tile edges

Chasing Denver's known missing tile turned up a much wider problem. Tiles
are 3 deg x 3 deg and the tile ID was derived from the city *centroid*, so
any city near a multiple of 3 silently lost whatever hung over the edge to
nodata. **11 of the 37 sweep cities were affected**, not just Denver:

| city | tile used | tiles actually needed |
|---|---|---|
| San Diego | N30W120 | N30W117 + N30W120 + N33W117 + N33W120 |
| Tulsa | N36W096 | N33W096 + N33W099 + N36W096 + N36W099 |
| **Tucson** | N30W111 | N30W111 + **N30W114** |
| Denver | N39W105 | N39W105 + **N39W108** |
| Dallas, Houston, Kansas City, Minneapolis, Omaha, Philadelphia, Reno | 1 tile | 2 tiles each |

Tucson matters most: it is the headline arid-inversion case and it was
missing a whole tile. `tiles_for_boundary()` now derives the full set from
the boundary's bounding box and `fetch_worldcover_for_grid` mosaicks them,
filling only nodata so overlaps are harmless. Covered by
`tests/test_worldcover_tiles.py` (including the corner case needing four
tiles and the off-by-one at exactly −105). All **36 tests pass**.

**What it changed.** The whole sweep was re-run. Five cities moved:

| city | single-tile ceiling | multi-tile | Δ |
|---|---|---|---|
| Denver | +0.318 | +0.446 | +0.127 |
| Kansas City | +0.560 | +0.610 | +0.050 |
| Tucson | −0.529 | −0.576 | −0.047 |
| Dallas | +0.685 | +0.704 | +0.020 |
| Reno | +0.083 | +0.083 | 0.000 |

Every headline conclusion **strengthened** at the time: the gap–ceiling
Spearman rose from +0.793 to +0.840 and the four-predictor fit from
R = 0.918 to 0.950. *(Both figures are from the 37-city single-day sweep and
are superseded by the 49-city multi-day run in §5: ρ = +0.844, R = 0.934.
They are kept only to show the direction of the tile fix.)* Tucson's
inversion deepened.

Two consequences for the paper. The published Denver *univariate* ceiling
(+0.131) is essentially unaffected (+0.133 recomputed, still within
tolerance) — impervious-only correlation barely notices the missing western
strip. But Denver's *three-channel* ceiling moves a lot, +0.149 → +0.395,
so the multivariate redefinition in §1 and this fix interact. Any figure
quoted for Denver should come from the mosaicked run.

The single-tile results are preserved in
`outputs/ceiling_study_singletile.jsonl` for comparison.

**Regression check.** After all of the above (cache keys, decay estimator,
PM variants, advection term, wind components on `Forcing`), the published
single-stage reproduction still holds exactly — `+0.8495 / 1.0893` across
NumPy, torch CPU, torch CUDA and torch CUDA+compile — and all **36 tests
pass**.

---

## 10. Performance claims in §I — mostly addressed in v3

v3 already carries the "single unbatched GPU run is 7x slower" caveat, so
this is largely resolved. One item may remain — check the current wording:

- **10.8 ms was measured on a desktop**, not a laptop — Core Ultra 7 270K.
  I have not benchmarked a laptop, and a thin-and-light would plausibly be
  2–3× slower. Either change the wording or drop the platform.
- **24,884 sims/sec requires `torch.compile` *and* batch ≈ 1024.** Eager
  PyTorch reaches only ~3,500/sec, and a single unbatched run on GPU is
  **7× slower than NumPy**. A reviewer who tries the obvious thing will miss
  by 7×, so the batching and compilation should be stated.

Supporting measurements (RTX 5060 Ti 16 GB, sm_120, torch 2.11.0+cu128):

| scale | configuration | result |
|---|---|---|
| 1 sim, 500 m | NumPy 10.8 ms vs CUDA 79.6 ms | GPU **7× slower** |
| B = 1024, 500 m | 41.2 ms → 24,884 sim/s | **268×** NumPy throughput |
| B = 256, 500 m | eager 79.3 ms vs fused 48.8 ms | fusion is ~4× |
| 30 m (810k cells), B = 32 | CPU 27.0 s vs CUDA 689 ms | 39× |
| peak VRAM, all benchmarks | 0.44 GB of 17.1 GB | memory not binding |

Two further facts worth having on hand: at 30 m with D = 1.0 the CFL forces
dt from 600 s to 180 s (960 steps per 48 h, a 3.3× cost *on top of* the 278×
cell count); and the torch port is autograd-clean, so `d(score)/d(coef)`
exists for all five coefficients — the enabler for fitting spatially-varying
coefficient *fields* instead of five global scalars, which is the most
direct attack on the boundary-pinned coefficients and the arid-city failure.

---

## 11. Reproducibility notes (not paper content, but costly to rediscover)

These are environment findings, kept because they will recur on any new
machine and are not recorded in the paper.

**TLS interception breaks four separate trust stores.** This machine runs
HTTPS-inspecting security software; each layer needs its own fix, on top of
the merged Windows-roots + certifi bundle built by `make_ca_bundle.ps1`:

| layer | symptom | fix |
|---|---|---|
| uv | `invalid peer certificate: UnknownIssuer` | `UV_NATIVE_TLS=1` |
| Python requests/ssl | `CERTIFICATE_VERIFY_FAILED` | `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE` |
| GDAL (schannel) | `certificate chain is incomplete` | **ignores `GDAL_HTTP_CAINFO`** — fetch over requests instead |
| gRPC (GCS) | handshake spam, silent HTTP fallback | `GRPC_DEFAULT_SSL_ROOTS_FILE_PATH` |

GDAL on Windows links curl against schannel, which reads the Windows store
directly and cannot be pointed at a CA file at all — hence `download_asset`
in `modis.py` rather than `GDAL_HTTP_UNSAFESSL=YES`.

**GDAL `/vsicurl` on large COGs is pathological here.** Reading the 123 MB
WorldCover tile that way stalled **44 minutes on 3.2 s of CPU** (one range
request per internal block, thousands of blocks, ~0.5 s round trip).
Downloading the whole tile takes **12.4 s at ~10 MB/s** — hence
`ensure_local_tile`, which also makes tiles reusable across cities sharing a
3° tile.

**Toolchain.** Python 3.14 does not work for this stack (3.12 does);
`torch.compile` on Windows needs `triton-windows` installed separately, and
fails only at compile time without it; `grid.py:90` still uses
`boundary.unary_union`, deprecated in geopandas 1.1.4 and removed in 2.x;
and the repo ships `data/` empty, so any reproduction needs a full refetch.

**Behaviour change to be aware of.** `patch_distance_decay` now defaults to
`estimator="robust"`, which changes `run_full_analysis` outputs relative to
the published 366 m. `estimator="legacy"` reproduces the old values, and
`run_decay_pareto.py` reports both. Also note the Pareto summary line
labelled "published ship D = 1.0" actually prints the nearest sweep point
(D = 0.794); true D = 1.0 interpolates to r ≈ 0.754 at ~330 m.

---

## 12. Suggested edits, in order

1. **Redefine the ceiling multivariately** (§1). Now essential rather than
   cosmetic: the univariate version is *negative* in five arid cities, which
   makes "% of ceiling" undefined there. Also fixes the Miami anomaly.
2. **Replace all three published ceilings with multi-day means** (§1.1).
   This is now the highest-priority item after §1: Phoenix's +0.363 is above
   the maximum of its own 11-day range and its typical ceiling is *negative*;
   Denver's daily value swings ±0.32. Both "% of ceiling" figures in §III-C
   rest on unrepresentative denominators.
3. **Use the ARCO bootstrap row in Table II**: `+0.754 [+0.718, +0.785]`,
   RMSE `1.40 [1.33, 1.46]` (§2).
4. **Rewrite §III-D** (§3). The split is 15% / 11%; both named mechanisms
   together explain only 23%; the recommended T_i-VPD fix recovers 0.039;
   and the big-leaf bulk form the module rejects is the best performer at
   +0.594.
5. **Replace the 366 m figure** with the edge-referenced ~330 m, and restate
   the trade-off cost as **−0.185** using the Pareto front (§4).
6. **Soften "perfectly"** to ρ = +0.844, p = 2.7e-14, n = 49 (§5), and add
   the **arid-inversion** result (§5.1) — but see §5.2 before choosing which
   ceiling definition to quote it against.
7. **Add the NDVI pre-screen** (§6): it demonstrates the diagnostic is
   actionable, and it identifies Denver rather than Phoenix as the city the
   extra channel rescues.
8. **Requalify the advection proposal** in §V (§7): as written it is
   destructive; it needs a coupled air layer, and costs 8.4× in timesteps.
9. **One sentence on the bias coupling** (§8) — real but second-order.
10. **Qualify the two performance claims in §I** (§10): desktop not laptop,
    and the GPU figure needs batching + compilation.

Two things to do before publishing the multi-city result:

- ~~Re-run the sweep with more candidate dates~~ — **done** (§5.0): n = 49,
  and the association survives on the humid half alone (ρ = +0.797).
- ~~Mosaic Denver's second WorldCover tile~~ — **done** (§9.1), and it
  turned out to affect 11 cities rather than one.

Open question worth pursuing: **~77% of the PM regression is explained by
neither named mechanism** (§3). Most likely candidates are the `r_s`
parameterization and the boundary-pinned `k_ET`, which runs to 40–55 against
a bound of 60 in nearly every PM variant.

---

## 13. What the renders exposed (2026-08-29)

Rendering the shipped pipeline surfaced three things worth keeping, none of
which change a published number.

**13.1 Table II's RMSE is an anomaly RMSE, and nothing says so.**
`bootstrap_uncertainty._r_and_rmse` mean-subtracts *both* fields before
scoring. On Austin 2024-08-21 the shipped configuration gives anomaly
RMSE 1.49 °C — consistent with the reported 1.40 — but a raw RMSE of
3.95 °C, because the simulation runs a mean 3.66 °C cold. Both numbers are
defensible; only one is in the paper, and the text does not state which.
A reader who recomputes RMSE the obvious way gets 2.8x the published value
and concludes the model is much worse than it is. One clause in §II fixes
this permanently: say the slab has no absolute-temperature anchor, so only
the spatial pattern is scored.

**13.2 The residual is spatially structured, not noise.**
On the anomaly panels the model reproduces the hot core but places it
slightly west of where MODIS puts it, leaving a coherent dipole in the
residual rather than salt-and-pepper error. That is a systematic
displacement, not random misfit, and it means the r = +0.75 is limited by a
*positional* error that more calibration of the existing coefficients cannot
remove. Plausible causes, in order of cheapness to test: the diffusion
trade-off already documented in §4 smearing the core; the absorption
composite being too coarse to place the core precisely; the peak-frame
selection picking a time offset from the Aqua overpass. Worth one diagnostic
before any further coefficient work — it may bound what calibration can
still buy.

**13.3 The counterfactual number depends on a definition that is easy to
misread.** `counterfactual.summarize` averages ΔT over every (cell,
timestep) pair. Under ARCO coefficients and ARCO forcing the +20% canopy
scenario gives −0.77 °C on that definition — matching the published
−0.74 °C — but −1.52 °C at the final frame, and per-frame values swing
between roughly −0.3 and −1.5 across the diurnal cycle. Any restatement of
the cooling number should say over what it is averaged. The spatial pattern
is also not uniform: cooling concentrates where there was impervious surface
to convert, and the water bodies show slight *warming* relative to baseline.

Renders: `outputs/figures/austin_surface_{2d,3d}.gif`,
`canopy_counterfactual_3d.gif`, `sim_vs_modis.png`, `arid_contrast.png`,
from `scripts/make_output_animations.py` and `scripts/make_extra_figures.py`.

---

## 14. The ceiling is mis-specified: it must match the model's receptive field (2026-09-21)

Work toward the readiness atlas, which required settling what the ceiling
actually bounds before mapping it at scale. It does not bound what the
paper says it bounds, and fixing the definition reverses the paper's
headline worked example.

**14.1 The shipped model exceeds its own city's ceiling.**
Austin ship configuration scores r = +0.754 on its two held-out days. The
multivariate ceiling on those same two days is +0.693; the 8-day mean is
+0.613. The model is therefore at **109%** of its test-day ceiling and
123% of the multi-day one. `ceiling.py`'s own docstring already said this
("so Austin exceeds even this"), but the manuscripts claim the ceiling is
"a per-city upper bound on the skill any land-cover-driven model can
achieve". It is not. Austin is also the one focus city with no
percent-of-ceiling figure in Table 2 -- including it would have shown
>100% and exposed the definition immediately.

**14.2 Nonlinearity is not the explanation.**
Gradient boosting under spatial block CV (contiguous 6x6 tiles, 5 folds)
on the same three per-cell channels gives Austin +0.681 on those days --
slightly *worse* than the linear fit, and still exceeded by the model
(111%). Across the focus cities the per-cell nonlinear gain is small and
signed inconsistently: austin -0.024, denver -0.014, miami +0.047,
phoenix +0.110. So the model is not winning by being nonlinear.

**14.3 It is winning by being spatial.** The ceiling regresses LST on each
cell's *own* land cover. The simulator diffuses heat, so a modelled cell
depends on its neighbours' land cover too. Adding mask-aware
neighbourhood-averaged copies of the same three channels at sigma = 1, 2,
4 cells (~0.5-2 km, bracketing the shipped diffusion length of ~590 m):

| city | linear | per-cell NL | +context | gain |
|---|---|---|---|---|
| austin | +0.613 | +0.589 | +0.789 | +0.200 |
| phoenix | +0.271 | +0.381 | **+0.689** | +0.307 |
| denver | +0.437 | +0.423 | **+0.756** | +0.333 |
| miami | +0.841 | +0.887 | +0.952 | +0.065 |

On Austin's test days the context ceiling is +0.883 and the model sits at
**85%** of it. The violation disappears. The ceiling was mis-specified,
not wrong in principle: it must be matched to the model's receptive field.

**14.4 This reverses the Denver/Phoenix contrast.** The papers argue that
Denver's higher ceiling leaves headroom calibration can exploit while
Phoenix's does not, and present that as the diagnostic's practical value.
Under receptive-field-matched ceilings both cities have large headroom
(Phoenix +0.689, Denver +0.756), and the achieved fractions become 11%
and 17% rather than 28% and 29%. Phoenix is no longer data-limited, so
"calibration cannot help there because the data will not support it" is
not supported. Whatever makes Phoenix degrade under re-optimisation
(+0.077 -> +0.033), it is not an absence of recoverable signal.

**14.5 The context gain is largest exactly where the arid inversion is.**
Phoenix +0.307 and Denver +0.333 against Miami +0.065. The arid
inversion appears to be a *per-cell* phenomenon that largely dissolves at
neighbourhood scale -- the urban core still separates from the desert
around it, even where individual built and vegetated cells do not
separate. This needs testing on the arid sample properly, but if it
holds, the inversion is a statement about pixel-scale land cover rather
than about whether arid cities are modellable.

**14.6 Robustness.** The context gain is not an artifact of smoothing
crossing fold boundaries. At block sides 6, 12 and 24 cells (3, 6 and
12 km, the last four times the smoothing reach) Austin gains +0.202,
+0.208, +0.170 and Phoenix +0.310, +0.252, +0.290. Note also that the
smoothing is applied to land cover, which is a known input everywhere, not
to LST, so no label information crosses folds.

**14.7 The mutual-information bound is not usable as written.** A KSG
k-NN estimate converted through the Gaussian relation
rho = sqrt(1 - e^{-2I}) fails its own bound property: Miami returns
+0.581 against an achievable +0.887. The conversion assumes joint
normality and Miami's LST is bimodal across its large water fraction.
Either drop the MI route or replace the conversion; do not report it as an
upper bound in its current form.

Artifacts: `outputs/nonlinear_ceiling_pilot.json`,
`outputs/receptive_field_test.json`. Code: `ceiling_nonlinear.py`,
`scripts/run_nonlinear_ceiling_pilot.py`,
`scripts/run_receptive_field_test.py`.

---

## 15. The arid inversion is real, but it is a pixel-scale effect (2026-09-21)

The receptive-field test of §14 extended to all 28 global-extension
cities (uniform 25 km disc, five continents). The focus-city pattern
holds and generalises, and it settles §14.5 in the direction that was
least convenient for the papers.

**15.1 Every city gains, and the gain is predicted by how low the
per-cell ceiling was.** Gains run from +0.019 (Chicago) to +0.593
(Boise). Against the per-cell ceiling, gain has Spearman
rho = **-0.869** (p = 2.1e-09). Against mean LST it is only +0.310
(p = 0.11), so aridity is not the direct driver -- it acts *through*
arid cities having low per-cell ceilings. Arid median gain +0.328 vs
non-arid +0.167 (Mann-Whitney p = 0.005).

**15.2 All eleven inverting cities recover completely.** Every city with
a negative univariate ceiling has a strongly positive context ceiling:

| city | univariate | per-cell NL | context |
|---|---|---|---|
| amman | -0.429 | +0.718 | +0.858 |
| riyadh | -0.342 | +0.409 | +0.612 |
| marrakesh | -0.156 | +0.347 | +0.743 |
| antofagasta | -0.124 | +0.217 | +0.687 |
| hermosillo | -0.116 | +0.407 | +0.686 |
| madrid | -0.102 | +0.259 | +0.468 |
| billings | -0.101 | +0.312 | +0.640 |
| alice springs | -0.096 | +0.113 | +0.559 |
| cairo | -0.082 | +0.840 | +0.912 |
| yuma | -0.067 | +0.721 | +0.866 |
| mendoza | -0.032 | +0.381 | +0.859 |

The inversion is therefore a statement about *pixel-scale* land cover,
not about whether the city is modellable. The built core still separates
thermally from the desert around it; individual built and vegetated cells
do not.

**15.3 No city in the sample is data-limited under a matched ceiling.**
The lowest context ceiling across all 28 is Windhoek at **+0.463**;
the median is **+0.779**. The between-city spread collapses from 0.858
(per-cell) to 0.526 (context). Much of the apparent variation in "data
readiness" is an artifact of measuring at pixel scale.

**15.4 The arid/non-arid ceiling difference loses significance.**
Per-cell: arid median +0.372 vs +0.632, Mann-Whitney **p = 0.018**.
Context: +0.727 vs +0.828, **p = 0.115**. Under a receptive-field-matched
ceiling, arid and non-arid cities do not have detectably different
ceilings. This directly qualifies the papers' central claim.

**15.5 The gap heuristic survives, weakened.** gap vs ceiling:
rho = +0.571 (linear), +0.540 (per-cell), +0.483 (context), the last
still significant at p = 0.009. So the heuristic predicts recoverable
signal at either scale, but less sharply once neighbourhoods are visible.

**15.6 What this does to the story.** The defensible claim is no longer
"arid cities have a data ceiling no model can exceed". It is: *pixel-scale
land cover fails in arid cities, and the failure is recoverable by
widening the receptive field rather than by adding channels.* That is more
useful to a practitioner and more falsifiable, but it is a materially
different paper from the one currently drafted. Sections 3.4, 3.5 and the
conclusions all need revisiting beyond the edits already made.

Caveat: "context" here is three Gaussian scales on three channels, not an
exhaustive search over receptive fields. The true matched ceiling is at
least this high, so the direction of every conclusion above is safe, but
the magnitudes are lower bounds.

Artifacts: `outputs/arid_receptive_field.jsonl`,
`outputs/arid_receptive_field_analysis.json`. Code:
`scripts/run_arid_receptive_field.py`,
`scripts/analyze_arid_receptive_field.py`.

---

## 16. The 49-city sweep at matched receptive field: the gap heuristic survives (2026-09-21)

Recomputed the US sweep's ceilings three ways on the cities' own OSM
boundaries and accepted dates, closing the scope gap left by §15.

**16.1 The headline association is unchanged.** gap vs ceiling:

| ceiling specification | rho | p |
|---|---|---|
| published multivariate | +0.694 | 3.3e-08 |
| linear, recomputed | +0.694 | 3.3e-08 |
| nonlinear per-cell | +0.718 | 6.2e-09 |
| nonlinear + context | **+0.716** | 7.3e-09 |

Paired bootstrap on the difference (context minus published), resampling
cities together: **+0.022, 95% CI [-0.085, +0.142]**. No significant
change. The thermal-gap heuristic predicts recoverable signal, not merely
per-pixel recoverable signal, so the published rho = +0.69 stands without
the scope caveat §15 required. The recomputed linear ceiling reproduces
the published one exactly, which also re-verifies the original pipeline.

**16.2 No US city is data-limited at matched receptive field.** Lowest
context ceiling is Spokane at +0.335, median +0.785, and zero cities have
a negative context ceiling. All five cities that invert under OSM limits
recover fully:

| city | univariate | per-cell | context |
|---|---|---|---|
| boise | -0.331 | +0.569 | +0.747 |
| albuquerque | -0.308 | +0.495 | +0.716 |
| tucson | -0.254 | +0.392 | +0.602 |
| phoenix | -0.095 | +0.348 | +0.662 |
| billings | -0.034 | +0.138 | +0.418 |

This matches the global sample exactly: inversion is pixel-scale on both
samples, under both city definitions.

**16.3 The gain is universal, and aridity is not what drives it.** Every
one of the 49 gains (min +0.026, median +0.221, max +0.371). Gain against
per-cell ceiling height rho = **-0.707** (p = 1.3e-08); against mean LST
only +0.208 (p = 0.15). Splitting at median LST, hot and cool halves gain
indistinguishably (+0.212 vs +0.225, p = 0.73).

This *corrects* the reading in §15.1 and the current draft. Globally the
arid group gained more (p = 0.005), which invited the phrase "arid cities
need a wider receptive field". In the US sample, where the aridity range
is narrower, there is no such split. The consistent statement across both
samples is the one the correlations support directly: **the gain is
predicted by how low the per-cell ceiling was, and aridity matters only
because arid cities are the ones with low per-cell ceilings.** Any
phrasing that makes aridity the cause rather than a correlate should be
corrected.

**16.4 Aridity still predicts the ceiling, more weakly.** mean LST vs
ceiling: rho = -0.418 (p = 0.003) published, -0.284 (p = 0.048) context.
Marginal at matched receptive field but not eliminated, unlike the
arid/non-arid group difference in the global sample.

**Process note.** During this run I overwrote the output file with
synthetic rows to smoke-test the analysis while the job was still
appending to it -- the exact clobber pattern §9 documents. No rows were
lost, verified by checking all 49 city names against `full_sweep.jsonl`
and confirming no duplicates, no synthetic leftovers and no gaps, but that
was luck. Smoke tests must use a scratch path.

Artifacts: `outputs/sweep_receptive_field.jsonl`,
`outputs/sweep_receptive_field_analysis.json`. Code:
`scripts/run_sweep_receptive_field.py`,
`scripts/analyze_sweep_receptive_field.py`.

---

## 17. CNN validation: the ceiling argument holds, including its predicted failure (2026-09-25)

The papers argue receptive field matters for convolutional models without
ever testing one. A fully-convolutional net was trained on the same three
land-cover channels under the same spatial block folds, at three receptive
fields: RF=1 (all 1x1, a per-cell MLP), RF=7 (three 3x3, ~3.5 km) and
RF=13 (six 3x3, ~6.5 km).

| city | per-cell | CNN RF=1 | context | CNN RF=7 | CNN RF=13 |
|---|---|---|---|---|---|
| austin | +0.589 | +0.605 | +0.789 | +0.785 | **+0.862** |
| phoenix | +0.381 | +0.407 | +0.689 | +0.608 | **+0.767** |
| denver | +0.423 | +0.492 | +0.756 | +0.693 | **+0.842** |
| miami | +0.887 | +0.876 | +0.952 | +0.949 | **+0.956** |

**17.1 The control works: architecture alone buys nothing.** RF=1 lands
within 0.07 of the per-cell GBM ceiling in every city, and below it in
Miami. A neural network with no spatial reach does not beat gradient
boosting on the same per-cell inputs. Whatever the gain is, it is not
"because it is a neural network".

**17.2 Spatial reach buys a great deal.** RF=7 exceeds the per-cell
ceiling by +0.196 (austin), +0.227 (phoenix), +0.270 (denver) and +0.062
(miami) -- the same ordering as the Gaussian-context gain, and for the
same reason.

**17.3 The matched ceiling actually bounds a matched model.** RF=7 sits
at or below the context ceiling in all four cities, and in two of them
almost exactly at it: austin +0.785 against +0.789, miami +0.949 against
+0.952. This is the first direct evidence that the receptive-field-matched
ceiling functions as a bound rather than merely as a larger number.

**17.4 And a wider model breaks it, exactly as the paper predicted.**
RF=13 exceeds the context ceiling in all four cities: +0.073, +0.078,
+0.086 and +0.004. The Limitations section already said a model class with
a wider receptive field "would need its own matched ceiling before a
percent-of-ceiling figure against it would mean anything". That is now
demonstrated rather than asserted, which converts the caveat from a hedge
into a property of the framework.

The cleanest statement of the whole result: a ceiling is not a property of
a city, it is a property of a city *and* a receptive field, and both must
be named before a percent-of-ceiling figure means anything.

Artifact: `outputs/cnn_validation.json`. Code:
`scripts/run_cnn_validation.py`. Ran on the RTX 5060 Ti; the GPU port
finally earns its keep.
