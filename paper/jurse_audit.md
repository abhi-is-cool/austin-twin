# Audit of `jurse.tex` (JURSE 2027 submission)

Checked every number against the artifacts in `austin-twin/outputs/`, plus
citations and length. **The condensation is accurate** — all five tables and
every statistic in the text verify, including the corrections from
`paper_v3_diff.md`, which have been applied. Four issues, none fatal.

## Verified correct — no change needed

| claim | value in paper | artifact |
|---|---|---|
| Ship config | +0.754 [+0.718, +0.785], RMSE 1.40 | `bootstrap_arco.json` |
| Transfer, zero-shot | +0.056 / +0.077 / +0.724 | `transfer_arco.json` |
| Transfer, Scheme A | +0.126 / +0.033 / +0.832 | `transfer_arco.json` |
| Ceilings | +0.437±0.085 / +0.271±0.085 / +0.841±0.025 | `focus_city_ceilings.json` |
| Gap heuristic | ρ = +0.69 (p 3e-8); univariate +0.84 | `full_sweep_analysis.json` |
| Paired difference | Δρ = +0.15 [+0.03, +0.29] | `full_sweep_analysis.json` |
| Arid five | all ceilings, sds, gaps, n | `full_sweep.jsonl` |
| NDVI screen | ρ = −0.55 (4.9e-5), +0.50 (2.7e-4), 27/49, median +0.06, Phoenix 4th (+0.22) | `full_sweep_analysis.json` |
| ET ablation | +0.758 / +0.755 / +0.686 / +0.416 | `rs_control.json`, `pm_identification.json` |
| PM decomposition | LAI+VPD 21%; r_s 61% at 3200 s/m (~32×) | `pm_identification.json` |
| ERA5 contribution | +0.007 [+0.004, +0.009] | `paired_era5_ci_arco.json` |
| Pareto | +0.67 at D=0.20; −0.19 vs +0.85 at D≈12.6 | `decay_pareto.json` |
| Advection | r = −0.11 at 25% strength | `advection_and_bias.json` |
| Bias coupling | Δr = 0.02 over a 3 °C sweep | `advection_and_bias.json` |
| Runtime | 10.8 ms, desktop CPU | benchmark |
| Ceiling contrast | multivariate exceeds R = +0.45 | max +0.462 |

**Citations: clean.** 20 cited, 20 defined, no orphans either way.

---

## Issue 1 — "9–12 dates per city" is wrong (medium)

Section II-C states multi-day ceilings use "9--12 dates per city,
cloud-free ... throughout". Neither sample matches:

| sample | actual n |
|---|---|
| focus cities | Miami **5**, Austin 8, Denver 9, Phoenix 11 |
| 49-city sweep | median **6**, range **3–12** |

The figure looks to have been read off `tab:arid`, whose five cities happen
to be 9–11. But Miami's ceiling (+0.841±0.025) — one of the three numbers
the transfer result rests on — comes from **5** days, and the 49-city
median is 6.

FIND:
```latex
report multi-day means (9--12 dates per city, cloud-free) throughout.
```
REPLACE:
```latex
report multi-day means throughout (5--12 cloud-free dates per focus city;
median 6 across the 49-city sample).
```

Worth fixing rather than fudging: a reviewer who divides 0.085 by sqrt(n)
will get a different answer than the text implies.

---

## Issue 2 — Scheme B is defined but never reported (medium)

Section II-C introduces "three schemes: zero-shot; Scheme~B (re-optimize
$k_{\mathrm{ET}}$ only); Scheme~A (re-optimize all four coefficients)", but
`tab:transfer` shows only zero-shot and Scheme A. Scheme B appears exactly
once in the paper, in that definition. A reader will look for it.

Two clean options.

**(a) Report it** — one row, values already computed
(`transfer_arco.json`):
```latex
    Scheme B, 3 days & $+0.063$ & $+0.071$ & $+0.776$ \\
```
It also strengthens the argument: Scheme B (k_ET only) barely moves any
city, so the Denver improvement is specifically the *absorption* refit, not
ET tuning.

**(b) Drop it from Methods** and describe two schemes. Cheaper on space, and
space is tight (Issue 4).

---

## Issue 3 — "humid-only quarter split" is a half (low)

The text says the heuristic generalizes "except a humid-only quarter split
($n{=}24$)". At n = 24 of 49 that is the **half** below median mean LST, not
a quarter.

FIND: `except a humid-only quarter split ($n{=}24$)`
REPLACE: `except the humid half ($n{=}24$)`

### 3a. A related overclaim introduced by compression

The same sentence calls the humid-subset failure "likely underpowered
rather than a genuine failure". `paper_v3 (3)` was more careful:

> We cannot distinguish, at $n=24$ and a weaker underlying signal, whether
> this is simple underpowering or whether the association genuinely
> concentrates in cities with pronounced aridity, and we flag it as open
> rather than claim it either way.

That is the defensible position, and the condensation has hardened it into
a claim the data does not support — this is the one subset where the
heuristic fails against the sound target (ρ = +0.361, p = 0.083), so it is
also the one a reviewer will press on. Suggested: "…except the humid half
($n{=}24$, $\rho{=}{+}0.36$, $p{=}0.08$), which we leave open: at this
sample size underpowering and a genuine concentration in arid cities are
not separable."

---

## Issue 4 — length is at the limit (check before submitting)

Rough estimate: **~4.0 pages** in IEEE two-column — 2,044 body words
excluding tables, 5 tables (1 full-width), 20 references. JURSE's limit is
typically 4 pages, so there is no margin, and my estimate is crude (easily
±0.5 page; table typesetting and float placement dominate).

I could not verify this: **no LaTeX toolchain is installed on this machine**,
so the file has never been compiled here. Before submitting, compile it and
check against the current CFP — I would not trust my estimate to decide
whether something has to be cut.

If it does overrun, the cheapest cuts are Issue 2 option (b), the Table I
positioning table (full-width, ~0.3 pg, and the least load-bearing), and the
counterfactual paragraph, which is the section least connected to the
paper's stated contributions.

---

## Note on framing

One thing the condensation does well and is worth preserving under edits:
the Denver/Phoenix contrast is now used *constructively* — the diagnostic
distinguishes fixable transfer failures from unfixable ones, rather than
just flagging low scores. That is a stronger claim than the earlier
"calibration cannot compensate" framing it replaced, and it survived the
transfer re-run that broke the older version.
