# jurse2027_draft.tex — edit log

Audit and fixes applied 2026-09-26. Line numbers are against the **draft as
received**; the file has since grown, so use the anchor text to find them.

Backup of the received version:
`scratchpad/jurse2027_draft.BEFORE.tex` (session-local, not in git).

## Verified correct before any edit

Every number in `tab:receptive` and `tab:cnn` matches
`outputs/receptive_field_test.json` and `outputs/cnn_validation.json`
exactly, as do Austin's `+0.754` / `+0.693` / 109% and the GBM's `+0.681`.
The three summary claims in §3.4 re-check clean:

| claim | holds |
|---|---|
| RF=1 within 0.07 of per-cell, all four cities | yes |
| RF=7 at or below context ceiling, all four | yes |
| RF=13 above context ceiling, all four | yes |

## Edits

| # | orig line | what | why |
|---|---|---|---|
| 1 | 217–220 | `\textbf{$x$}` → `{\boldmath$x$}` (4 cells) | `\textbf` does not bold math content, so `tab:cnn`'s caption promised bold that would not render. Same bug class as the `\boldmath` leak fixed in the MDPI tables. |
| 2 | 80 | new opening paragraph citing `oke1982energetic`, `voogt2003thermal`, `masson2000physically`, `kusaka2001simple`, `bueno2013urban`, `lyu2022integrated` | Six of eleven references were uncited. `cite.sty` will not warn on a manual `thebibliography`, but a reviewer notices. Cited rather than cut — they are the framing a JURSE reviewer expects. |
| 3 | 62 | abstract: "raises the ceiling above the model's score in every one of four test cities" → Austin-specific, gap general | **The substantive correction.** Only Austin had a violation to restore. Best transfer scores are Phoenix `+0.080`, Denver `+0.024`, Miami `+0.832` — all *below* even their per-cell ceilings (`+0.381`, `+0.423`, `+0.887`). The four-city claim was true but vacuous in three cities, and the transfer scores are not in the paper to support it. |
| 4 | 97 | intro: "restores the bound in every city we tested" → same scoping | Same overclaim. |
| 5 | 171 | §3.2: "In every city, the context ceiling exceeds both the simulator's score and the per-cell ceiling" → scoped | Same overclaim, third site. New text says Austin is the only city where a calibrated simulator is scored against its own ceiling, so the only one where a violation *could* arise, and the table's job is to show the gap is not peculiar to Austin. |
| 6 | 110 | §2.1: MYD11A1 1 km native resolution on a 500 m grid, 6.9 cells per distinct value in Austin | First reviewer's most serious objection, unaddressed here. Lands harder in this paper than in the journal versions because the whole claim is about spatial scale. |
| 7 | 199 | new §3.3 "The Gain Is Not a Resampling Artifact" (`sec:coarsen`) | Answers (6). Concession and defense together. **Corrected after review — see below.** |
| 8 | 150 | §2.3: both estimates are cross-validated lower bounds | The second reviewer's "why call it a ceiling if you estimate from below" question. One clause; keeps the direction of every comparison safe. |
| 9 | 240 | figure swapped: `fig1_ceiling_definition.pdf` → `fig_receptive_field.pdf`, new caption | **The other substantive problem.** The old figure is the univariate/multivariate scatter over 49 US cities — it is about *channel count*. The caption described per-cell versus context-matched ceilings, i.e. *receptive field*. Different axes; the figure argued a point the paper does not make. |
| 10 | 262 | Discussion: Bayes-error positioning, `cover2006elements` + `noshad2019learning` | An achievable-performance bound given a feature set is not a new object. Without this the novelty claim reads naive to a quantitative reviewer. New text puts the novelty on the *support* argument, which is where it actually is. |
| 11 | 259 | "reported in full in a companion journal manuscript currently in preparation" → "developed in a companion manuscript and are not claimed here" | Three journal versions exist. See the venue note below. |
| 12 | 322 | two new `\bibitem`s | Supports (10). Copied verbatim from `ieee.tex`, already verified. |
| 13 | 233 | `Table~\ref{tab:cnn} and Fig.~\ref{fig:ceiling} give the result.` | Both floats were uncalled. IEEE style wants every float referenced; an uncalled float reads as an oversight. |
| 14 | 13 | header comment updated for the figure swap | Was still naming `fig1_ceiling_definition.pdf`. |

Post-edit audit: 13/13 references cited, no dangling refs, no unused
labels, braces balanced, no `\textbf{$...$}` left, figure file present.

## New figure

`scripts/make_rf_figure.py` → `outputs/figures/fig_receptive_field.pdf`
(and `.png`, and `--dark` for slides). Four small multiples, one per city:
CNN correlation against receptive field (1, 7, 13 cells), with the
per-cell ceiling dashed and the context ceiling dotted, and the region
above the context ceiling shaded. Sized for IEEEtran single column
(3.5 in). Shares both axes so only outer labels are drawn, which is what
buys the room at that width.

Reads directly from `outputs/cnn_validation.json`; the three summary claims
are re-asserted on every run.

## Open — author's call

**Page budget.** Body text went from 1286 to 1646 words (+28%), on top of
three tables and a figure. This very likely overruns 4 pages. Nothing was
cut unilaterally. Section sizes now:

| words | section |
|---|---|
| 261 | Introduction |
| 134 | §2.1 Study Areas and Data |
| 73 | §2.2 Surface Energy-Balance Simulator |
| 217 | §2.3 The Land-Cover Data Ceiling |
| 111 | §3.1 Simulator Exceeds Its Own Per-Cell Ceiling |
| 125 | §3.2 Matching Receptive Field Restores the Bound |
| 97 | §3.3 The Gain Is Not a Resampling Artifact |
| 195 | §3.4 A Convolutional Network Confirms the Mechanism |
| 288 | Discussion |
| 97 | Conclusion |

Cut candidates, in the order I would take them:

1. **Discussion ¶1** (channel-count requirement, ~70 words). It concedes
   the channel point is well understood and then does not use it. The
   paper is about receptive field.
2. **§2.2** (~73 words) down to two sentences. The simulator only needs to
   be *a model with spatial reach*; its coefficients do not matter to the
   argument, and §3.4 carries the general claim without it.
3. **§3.2's closing paragraph** on percent-of-ceiling (~55 words). The
   Conclusion makes the same point.

That is ~200 words, which roughly cancels the growth.

**Prose voice.** Edits 2, 3, 5, 6, 7, 8 and 10 are new prose written here.
Rewrite them in your own voice before submitting.

**Venue.** JURSE's deadline (Nov 17, 2026) falls before any journal
decision would return, and this paper's core result — receptive field plus
the CNN — is also the journal versions' core result. Conference-then-journal
is normal and JURSE permits it, but the journal version must then cite the
JURSE paper and state what it adds. Decide the order before submitting
either.

**GenAI disclosure.** The Acknowledgment still carries generic wording.
Make it describe what you actually did.

## Correction, after author review

§3.3's first version gave three coarsening values for Phoenix but only two
for Austin, Denver and Miami, without saying which resolution the second
number was, and folded Denver in with Phoenix under "essentially
unchanged" when Denver's gain **grows**. Both were mine. The full table
from `outputs/support_matching_test.json`:

| city | 500 m | 1000 m | 1500 m |
|---|---|---|---|
| Austin | +0.200 | +0.151 | +0.094 |
| Phoenix | +0.307 | +0.310 | +0.294 |
| Denver | +0.333 | **+0.383** | — |
| Miami | +0.065 | +0.048 | +0.037 |

Austin and Miami *do* have 1500 m levels; the first version reported
endpoints and dropped the middle value. Denver's absence is real and now
stated in the text: it has 1584 valid cells at 500 m and 358 at 1000 m, and
at 1500 m falls under the `valid.sum() < 200` floor at
`scripts/run_support_matching_test.py:120`. It is the smallest of the four
study areas. `outputs/_support.log:134` records this as "too few cells
after aggregation".

Denver's gain growing under coarsening strengthens the section rather than
weakening it — support mismatch cannot explain a gain that gets *larger*
when support is matched — but it has to be reported as its own result, not
averaged into a word like "unchanged".

`scratchpad/fix_coarsen.py` asserts all eleven stated values against the
artifact before writing, and asserts that Denver has no 1500 m level and
that its 1000 m gain exceeds its 500 m gain.

Body text is now 1699 words (+32% on the received draft); §3.3 alone is
150. The page-budget note above applies with more force.

## Table II slimmed

`tab:cnn` repeated the per-cell and context ceiling columns that
`tab:receptive` already carries. Dropped both, taking a six-column table at
IEEEtran column width down to four:

| City | RF=1 | RF=7 | RF=13 |
|---|---|---|---|
| Austin | +0.605 | +0.785 | **+0.862** |
| Phoenix | +0.407 | +0.608 | **+0.767** |
| Denver | +0.492 | +0.693 | **+0.842** |
| Miami | +0.876 | +0.949 | **+0.956** |

That was the real duplication between the table and the new figure — not
table-versus-figure. The division is now clean: Table I has the exact
ceilings, Table II the exact CNN scores, the figure the comparison between
them. Nothing duplicated, nothing lost.

`scratchpad/slim_tabcnn.py` rebuilds every retained cell from
`outputs/cnn_validation.json` rather than copying it across, so the
slimming cannot introduce a transcription error, and asserts RF=13 exceeds
the context ceiling in each city before writing. `\usepackage{makecell}`
dropped — those stacked headers were its only use. §3.4's opening now
points at Table I, since the ceilings it compares against live there.

This saves table width, not words: body text is 1706 (+32.7% on the
received draft). The cut list above still stands.

## Conformed to the IEEE conference template

Against `IEEE-conference-template-062824` (the stock IEEE conference
template, which is what JURSE uses). Ten changes:

| what | why |
|---|---|
| preamble replaced with the template's, verbatim | Was `[10pt,conference]` with an ad-hoc package list. Now `[conference]` plus the template's eight packages, `\IEEEoverridecommandlockouts` and the `\BibTeX` definition. Added beyond the template: `booktabs` (three tables) and `url` (repository link). `array` dropped — loaded but never used. |
| **abstract: all math removed** | The template flags this CRITICAL: *"Do Not Use Symbols, Special Characters, Footnotes, or Math in Paper Title or Abstract."* Xplore extracts the abstract as plain text. `$r=+0.754$` → `r = +0.754`, and `$+0.693$`, `$+0.789$` likewise. The title was already clean. |
| author block reshaped | Template puts the organization in `\textit{}` on its own line, then city/country, then email. |
| floats `[!t]` → `[htbp]` | Template convention, 2 tables + 1 figure. |
| tables: `\centering` → `\begin{center}`, `\label` moved after the tabular | Template's table structure. |
| figure: `\centerline{\includegraphics{...}}` | Template's figure structure. |
| `\begin{thebibliography}{9}` → `{00}` | 13 entries; a single-digit width misaligns numbers 10–13. |
| `noshad2019learning`: added `[Online]. Available: ...` | Template's arXiv reference form. |

Checked and found already compliant: reference titles are sentence case;
`et al.` appears only in `zanaga2022esa` and `hersbach2020era5`, both far
past the six-author threshold where the template permits it; no manual
`\section*{References}` (IEEEtran emits that heading itself).

### Not compiled

There is no TeX toolchain on this machine, so nothing here has been built.
`scratchpad/lint_tex.py` stands in for a compile and passes: environment
nesting balanced, every loaded package used, no math in title or abstract,
float and bibliography conventions correct. It is comment-aware, because
the file's own header block documents these rules using the strings the
checks search for.

**Build it on Overleaf before trusting the page count.** Body text is
1706 words, up ~33% on the received draft, and the ranked cut list above
is still the plan if it overruns four pages.

### Submission bundle

`C:\dev\research\jurse2027_submission\` — ready to zip and upload:

- `jurse2027_draft.tex`
- `fig_receptive_field.pdf`
- `IEEEtran.cls` (from the template, in case Overleaf's differs)

## Page-4 orphan

Reference [13] (Hersbach) sat alone on page 4 of the first compile. Page
3's right column had to absorb two more lines. Two cuts, both from the
ranked list above, both removing text the paper makes elsewhere:

| cut | words | why it was safe |
|---|---|---|
| Discussion para 1 opening | 65 -> 38 | It conceded that the channel-count requirement is straightforward and then never used it. The sentence that earns its place is the contrast with receptive field, which now opens the paragraph. |
| III-B closing paragraph | 46 -> 27 | Two sentences making one point. The Conclusion and the abstract both already make it; the sharper half -- a figure can exceed 100% with nothing wrong -- is kept. |

46 words, roughly five column lines against the two needed. The margin is
deliberate: float reflow cannot be predicted without a compile, and there
is no TeX toolchain here.

**Recompile to confirm.** If [13] still spills, the next cut on the list
is II-B, which can go down to two sentences without touching the argument.

