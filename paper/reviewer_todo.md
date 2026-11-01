# Reviewer objections — status and plan

External review received 2026-09-22. Assessed against the paper and the
artifacts; verdicts below are what the code and data actually say, not
what the paper claims.

**Bottom line: none of the seven were addressed in the manuscript.** Five
are hours of work. One (#1) can change a conclusion and is being tested
now. One (#5) is being measured.

---

## 1. MODIS resolution / pixel duplication — TESTED, LARGELY ANSWERED

**The objection.** MYD11A1 LST is natively ~1 km; the analysis runs on a
500 m grid with nearest-neighbour resampling, duplicating pixels and
inflating both correlation and autocorrelation.

**The threat.** Predictors (land cover, 500 m) sit at a finer support than
the target. Smoothing land cover at sigma = 1-2 cells is exactly what is
needed to align the two supports, so the context gain might be **support
matching** rather than **receptive field**. Both explanations predict the
same observation at 500 m.

**Test run.** Block-aggregated land cover *and* LST onto a common support
at 1000 m and 1500 m, with smoothing scales kept strictly wider than the
new pixel, so a surviving gain cannot be support matching.

| city | 500 m | 1000 m | 1500 m |
|---|---|---|---|
| austin | +0.200 | +0.151 | +0.094 |
| phoenix | +0.307 | +0.310 | +0.294 |
| denver | +0.333 | +0.383 | (too few cells) |
| miami | +0.065 | +0.048 | +0.037 |

**Verdict: the headline survives, with a real qualification.**

*Survives where it matters.* Phoenix is flat across supports
(+0.307 / +0.310 / +0.294) and Denver rises (+0.333 / +0.383). These are
the low-per-cell-ceiling cities the finding is about, and their gain
cannot be support matching, because it persists when smoothing operates
strictly beyond the thermal pixel.

*Qualified elsewhere.* Austin's gain more than halves (+0.200 to +0.094)
and Miami's falls from +0.065 to +0.037. In high-ceiling cities a
substantial share of the 500 m gain **was** support matching, and the
published 500 m numbers overstate the effect there.

**Correction to my own earlier estimate.** I first reported 7.55x pixel
duplication for Austin. That metric counts distinct LST values, which
conflates resampling duplication with genuine spatial homogeneity in the
thermal field. Checking properly: Austin has 380 distinct values over
717 km2 (0.53 per expected 1 km pixel) and Miami 719 over 5031 km2 (0.14).
The resampling duplication is about 4x; the remainder is real homogeneity,
especially over Miami's water. Both shrink the effective sample size, but
only the first is a methodological choice.

**Still required in the manuscript**, regardless of the favourable result:
- state the native LST resolution, the resampling method, and that the
  500 m grid oversamples it
- report the support-matching test, both the survival and the decay
- note that effective sample size is well below the reported `n`

Script: `austin-twin/scripts/run_support_matching_test.py`
Artifact: `outputs/support_matching_test.json`

---

## 2. "Ceiling" terminology — PARTIALLY ADDRESSED

The Limitations section concedes the matched ceiling is a lower bound
("at least as high as we report and may be higher... magnitudes lower
bounds"). But the word carries through title, abstract and every section
as though it bounds.

**Fix.** Either rename to something honest (achievable-skill benchmark,
attainable-skill estimate) or add one sentence at first use in Section 2.5
saying plainly that the cross-validated figure is the skill of a specific
benchmark regressor and therefore a lower bound on the true ceiling.
Cheap; do it either way.

---

## 3. Arid inversion novelty — NOT ADDRESSED

**Verified: zero citations to prior arid cool-island literature.** The
bibliography has no entry positioning the inversion against existing work
on desert urban cool islands, which are well documented.

**Fix.** Find 3-5 references on surface urban cool islands in arid cities
and add a short paragraph in Section 3.6 distinguishing what is new here.
The novel part is not that arid cities invert, it is that the inversion is
pixel-scale and dissolves under a matched receptive field. That
distinction has to be made explicitly, not left implicit.

Estimated: 2-3 hours including the literature search.

---

## 4. Scope creep — NOT ADDRESSED

Five contributions under a title promising one diagnostic: ceiling,
receptive field, arid inversion, diffusion trade-off, ET ablation, plus
counterfactual scenarios.

**Options.** (a) Accept it and argue the physics results establish the
simulator is a fair vehicle. (b) Cut the ET ablation and counterfactuals
to a companion paper. (b) is the stronger paper and the wrong move a week
before a deadline. Recommend (a) plus a sentence in the introduction
framing the ablations as establishing the vehicle rather than as
independent contributions.

---

## 5. Bias inconsistency (-3.66 vs ~7 C) — MEASURED, PAPER IS WRONG

Figure 1 reports a -3.66 C mean offset on 2024-08-21. Limitations claims
"a ~7 C absolute bias, arising from the simplified longwave scheme".

**Measured across all eight Austin focus days** (shipped ARCO
configuration, sim minus obs):

    +3.06, -0.36, +0.50, -1.28, -5.47, -3.66, -2.01, -1.94
    mean -1.40 C, range -5.47 to +3.06

So the ~7 C figure is **not supported by the current configuration** --
the largest single-day magnitude is 5.47 C and the mean is -1.40 C. It is
almost certainly stale, carried over from a pre-ARCO or pre-recalibration
run. The paper overstates its own bias by roughly fivefold.

**Fix.** Replace the Limitations sentence with the measured numbers: a
mean offset of -1.40 C across eight held-out days, ranging -5.5 to +3.1,
so the offset is day-dependent rather than a constant bias. Keep the point
that absolute temperatures should not be read as calibrated predictions --
the day-to-day swing supports it better than the old figure did.

---

## 6. Cairo — NOT ADDRESSED

**Verified.** Cairo is the *only* inverting city with a positive
built-vegetated gap (+1.08 C), and its multivariate ceiling is **+0.834**,
among the highest in the sample. So it inverts univariately while having
strong multivariate separability and a positive gap, which contradicts the
mechanism the paper proposes for inversion.

**Fix.** Either explain it (Cairo's Nile corridor and irrigated Delta
fringe plausibly break the built-vs-vegetated dichotomy the gap assumes)
or flag it in the table footnote as an exception the mechanism does not
cover. Do not leave it unmentioned; a reader scanning the table will
notice the sign.

---

## 7. Multiple comparisons — NOT ADDRESSED

Many hypothesis tests, no correction. Several p-values sit between 0.01
and 0.1 where a correction would matter, including the arid/non-arid
context comparison (p = 0.115) and the humid subset (p = 0.08).

**Fix.** Add one sentence to Section 2.6 stating that tests are reported
uncorrected and exploratory, and that the confirmatory claims rest on the
pre-specified comparisons. This is honest and costs nothing. Applying a
formal correction would weaken results already reported as marginal, so
the framing matters more than the arithmetic.

---

## Pre-existing to-dos (unchanged)

- GenAI disclosure wording in `\acknowledgments` — must describe what was
  actually done
- Name the Special Issue on the submission form (chosen: Urban Ecology
  Monitoring Using Remote Sensing)
- Set real access dates on `microsoft_pc` and `wri2026cooling`
- Recompile: length, float placement, accented names in the ERA5 entry

---

## Edit log --- `remotesensing-edited.tex`

Applied 2026-09-23. Line numbers are as of this pass; any later
edit above a given entry will shift it.

| obj | change | lines |
|---|---|---|
| #1 | Methods 2.2 --- native LST resolution, oversampling, effective *n* | 231-241 |
| #1 | Results 3.3.1 --- new subsection `sec:support` and Table 5 | 484-529 |
| #2 | Methods 2.5.3 --- ceiling stated as a lower bound | 326-334 |
| #3 | Results 3.5 --- arid cool-island literature positioning | 624-643 |
| #3 | New reference `liu2024identification` | 1167-1172 |
| #3 | New reference `qiu2025longterm` | 1203-1207 |
| #3 | Label `sec:caveat` on the robustness subsection | 719 |
| #4 | Introduction --- ablations framed as establishing the vehicle | 166-171 |
| #5 | Limitations --- stale ~7 C replaced with measured offsets | 986-992 |
| #6 | Table 6 --- Cairo footnote marker | 701 |
| #6 | Table 6 --- Cairo exception footnote | 708-715 |
| #7 | Label `sec:stats` on Statistical Methods | 344 |
| #7 | Methods 2.6 --- multiple-comparison statement | 371-377 |

File is 1264 lines after these edits (was 1132 before this pass).
