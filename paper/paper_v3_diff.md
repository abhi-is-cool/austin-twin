# Edit list for `paper_v3 (3).tex`

Every change below is a find/replace against the current file, in severity
order. Each states what is wrong, what to put instead, and the artifact the
replacement comes from, so nothing has to be taken on trust.

Verified-correct and needing **no** change: `tab:arid` (all five cities,
ceiling / sd / gap), `tab:subset` (all four rows, both targets),
`tab:gap-verify`, `tab:decay`, the `tab:transfer` ceiling row, the ship row
of `tab:bootstrap_ci`, and every headline figure in the abstract
(*R* ≈ +0.46, ρ = +0.694 / +0.844, 27 of 49, Phoenix 4th, ρ = −0.55 / +0.50).

---

## Edit 1 — the transfer matrix, re-run under the ARCO coefficients

**Severity: high. Changes numbers, one abstract figure, and one claim.**

The table's scores came from `transfer_*.json`, whose `austin_coefs` are the
CDS values (alpha_veg 0.4343), while Section II-C reports the ARCO ones
(alpha_veg 0.4641). Re-run with the identical protocol — same splits, same
five rows, same DE budget — using the ARCO config and ARCO forcing
(`scripts/run_transfer_arco.py`, output `transfer_arco.json`):

| Scheme | Denver | Phoenix | Miami |
|---|---|---|---|
| Zero-shot | **+0.056** *(was +0.004)* | +0.077 *(+0.083)* | **+0.724** *(+0.760)* |
| Scheme B, 1 day | **+0.063** *(+0.003)* | +0.071 *(+0.075)* | **+0.776** *(+0.787)* |
| Scheme B, 3 days | **+0.063** *(+0.003)* | +0.071 *(+0.074)* | **+0.776** *(+0.787)* |
| Scheme A, 1 day | **+0.126** *(+0.001)* | +0.033 *(+0.039)* | +0.832 *(+0.832)* |
| Scheme A, 3 days | **+0.126** *(+0.001)* | +0.033 *(+0.036)* | +0.832 *(+0.832)* |

Phoenix barely moves (all deltas <= 0.006). Miami's zero-shot drops 0.036.
**Denver moves a lot**, and in a way that changes the story.

FIND (the five score rows of `tab:transfer`) and REPLACE with:
```latex
    Zero-shot               & $+0.056$ & $+0.077$ & $+0.724$ \\
    Scheme B, 1 target day  & $+0.063$ & $+0.071$ & $+0.776$ \\
    Scheme B, 3 target days & $+0.063$ & $+0.071$ & $+0.776$ \\
    Scheme A, 1 target day  & $+0.126$ & $+0.033$ & $+0.832$ \\
    Scheme A, 3 target days & $+0.126$ & $+0.033$ & $+0.832$ \\
```
The Austin-ship reference row stays `+0.754`.

### 1a. The percent-of-ceiling figures change

Against the same multi-day multivariate ceilings already in the table:

| city | best score | ceiling | old % | **new %** |
|---|---|---|---|---|
| Denver | +0.126 (Scheme A) | +0.437 | 1% | **29%** |
| Phoenix | +0.077 (zero-shot) | +0.271 | 31% | **28%** |
| Miami | +0.832 (Scheme A) | +0.841 | 99% | **99%** |

ABSTRACT — FIND:
```latex
transfer experiment scored against multi-day ceiling means, Denver reaches
1\%, Phoenix 31\%, and Miami 99\% via light target-city re-optimization.
```
REPLACE:
```latex
transfer experiment scored against multi-day ceiling means, Denver and
Phoenix reach only 29\% and 28\% of their ceilings while Miami reaches
99\% via light target-city re-optimization.
```

Note the rhetorical loss: Denver was the dramatic "1%, essentially nothing"
case, and it is now statistically indistinguishable from Phoenix. The
contrast that survives is Miami (99%) against both arid cities (~28%),
which is still the point the diagnostic is making, just with two similar
cases instead of a spectrum.

### 1b. A claim that no longer holds

The text states re-optimization makes *both* Phoenix and Denver worse:

> both re-optimization schemes still make Phoenix and Denver \emph{worse},
> confirming target-city calibration cannot compensate for missing class
> separability regardless of exactly where the ceiling sits.

Under the ARCO coefficients that is **half true**. Phoenix still degrades
(+0.077 zero-shot to +0.033 under Scheme A). **Denver now improves**, and
substantially: +0.056 to +0.126, more than doubling. Suggested replacement:

```latex
Re-optimization still makes Phoenix \emph{worse} ($+0.077$ zero-shot to
$+0.033$ under Scheme~A), confirming that target-city calibration cannot
manufacture separability that the input channels do not carry. Denver
behaves differently: Scheme~A more than doubles its score ($+0.056$ to
$+0.126$), though it remains at 29\% of its ceiling. The distinction is
informative — Denver's ceiling ($+0.437$) is high enough that there is
headroom for calibration to exploit, whereas Phoenix's ($+0.271$) is not.
\end{quote}
```
(drop the stray `\end{quote}` — included only to mark the end of the
replacement block.)

**Caveat to state somewhere.** The re-run changes two things at once, the
coefficients *and* the forcing source, because the CDS NetCDFs for the
target cities were never on this machine and cannot be regenerated without
credentials. For Austin the forcing source alone was worth
Delta r = +0.0022 at fixed coefficients, so most of the Denver shift is
almost certainly the coefficients — but that is an inference, not a
measurement. Since the paper now uses ARCO throughout, the ARCO numbers are
the internally consistent ones either way.

Also update `tab:bootstrap_ci`, whose three zero-shot CIs are likewise
CDS-derived. The point estimates should become +0.056 / +0.077 / +0.724;
the intervals would need `bootstrap_uncertainty.py` re-run against the ARCO
transfer configs, which has not been done. Until then, either mark those
three rows as CDS-forced or drop the CI column for them.

---

## Edit 2 — `tab:pm`: the two K=1 rows now match the stated protocol

**Severity: medium. Numbers only.**

The caption says every variant was recalibrated with k_ET excluded from the
search; the two K=1 rows carried over from the earlier sweep, where k_ET was
still being searched. Refit under the same protocol as the rest of the table
(`scripts/run_pm_k1_refit.py`, output `pm_k1_refit.json`):

| row | published | refit | delta |
|---|---|---|---|
| PM, air VPD, K=1 | +0.485 | **+0.484** | −0.0009 |
| PM, surface VPD, K=1 | +0.509 | **+0.505** | −0.0042 |

FIND:
```latex
    PM, air VPD, $K=1$ & $+0.485$ \\
    PM, surface VPD, $K=1$ & $+0.509$ \\
```
REPLACE:
```latex
    PM, air VPD, $K=1$ & $+0.484$ \\
    PM, surface VPD, $K=1$ & $+0.505$ \\
```

Both shifts are small and neither changes the ordering, so this is about the
caption being true rather than about the result. Worth doing precisely
because the caption makes a specific methodological claim.

---

## Edit 3 — `tab:predictors`: two values from a superseded artifact

**Severity: low. Numbers only; no wording change.**

The univariate column takes two covariates from `ceiling_study.jsonl`
(the earlier 12-date run) instead of the settled `full_sweep.jsonl`. Both
match the old artifact to three decimals, which is how the provenance was
identified. The other six covariates already match the settled run exactly.

FIND (line ~500):
```latex
    Water fraction (mean) & $+0.521$ & $1.3\text{e-}4$ & $+0.583$ & $1.1\text{e-}05$ \\
```
REPLACE:
```latex
    Water fraction (mean) & $+0.517$ & $1.4\text{e-}4$ & $+0.583$ & $1.1\text{e-}05$ \\
```

FIND (line ~502):
```latex
    LST spread (p98$-$p2) & $+0.362$ & $0.011$ & $+0.475$ & $5.6\text{e-}04$ \\
```
REPLACE:
```latex
    LST spread (p98$-$p2) & $+0.352$ & $0.013$ & $+0.475$ & $5.6\text{e-}04$ \\
```

Source: `outputs/full_sweep_analysis.json`. Neither change affects any
conclusion — both covariates keep the same sign, rank and significance, and
the sentence noting that LST spread *strengthens* against the sound target
(+0.352 → +0.475) reads slightly better than before.

---

## Edit 4 — abstract overstates the stability result relative to the body

**Severity: low. Wording only.**

The body handles this correctly (lines ~347–357: median per-city sd 0.085
univariate vs 0.081 multivariate across 49 cities, "essentially a wash",
with the 2–3× tightening "confined to Phoenix and Denver specifically").
The abstract omits that context and calls the result "a positive finding for
the redefinition rather than a caveat on it", so a reader of the abstract
alone would over-generalise.

FIND:
```latex
while the multivariate ceiling used for the main
transfer experiment turns out to be $2$--$3\times$ more stable day-to-day
for exactly the cities where it matters, a positive finding for the
redefinition rather than a caveat on it.
```
REPLACE:
```latex
while the multivariate ceiling used for the main
transfer experiment proves $2$--$3\times$ more stable day-to-day in the two
cities whose univariate estimate is least reliable, and no less stable
across the sample as a whole.
```

This keeps the finding, drops the editorialising, and stops the sentence
claiming more than the body supports.

---

## Edit 5 — `tab:transfer` caption understates Miami's day count

**Severity: cosmetic.**

The caption says *n* = 9–11 days; Miami has 5. The surrounding text does say
so, but a table should stand alone.

FIND:
```latex
multi-day mean (95\% range via $\pm$sd, $n{=}9$--$11$ days)
```
REPLACE:
```latex
multi-day mean ($\pm$sd; $n{=}9$--$11$ days, except Miami $n{=}5$, limited
by cloud-free April coverage)
```

Also note the caption calls $\pm$sd a "95\% range". One standard deviation is
about 68\% under normality, not 95\%. Either drop "95\%" (recommended, since
these are 5–11 samples and normality is not established) or use $\pm2$sd.

---
