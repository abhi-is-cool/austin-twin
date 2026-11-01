# Talk script — data-ceiling control for UHI model transfer

**Target: 12 minutes speaking + 3 minutes Q&A.** 11 slides. Cut markers `[CUT]`
show what to drop for a 10-minute slot.

Delivery notes: the whole talk hangs on one idea — *a low score can mean the
model failed or the data can't express the answer, and you can tell which.*
Say that idea three times: once in the setup, once with the numbers, once at
the end. Everything else is support.

Every number below is from `outputs/full_sweep.jsonl` and the artifacts
listed in `findings.md`. If asked for a figure you don't have, say so.

---

## Slide 1 — Title (0:00–0:30)

*On screen: title, your name, Westwood High School, Round Rock TX.*

> Good morning. I'm Abhiram Donepudi, and this is work on a problem that
> shows up whenever an urban heat island model gets applied to a new city.
>
> The question I want to answer is simple to state: when a model transfers
> badly, is that the model's fault, or is it the data's?

**Pause after that line.** It's the thesis.

---

## Slide 2 — The confounded evaluation (0:30–2:00)

*On screen: two panels, same low transfer score, two different causes.*

> Here's the situation. You calibrate a model in one city, apply it to
> another, and the score is poor. There are two completely different reasons
> that can happen.
>
> One: the model genuinely fails to capture the new city's thermal
> structure. Two: the input land-cover data doesn't separate thermal regimes
> in that city well enough for *any* land-cover-driven model to succeed.
>
> These call for opposite responses. The first says fix the model. The
> second says fix the data — no amount of model work will help. And from the
> score alone, they look identical.
>
> So low transfer scores get attributed to model deficiency almost by
> default, because there's no control that separates the two.

---

## Slide 3 — The control (2:00–3:15)

*On screen: the definition, one equation.*

> The control I'm proposing is deliberately simple. For a target city, take
> the observed satellite land surface temperature, and correlate it against
> the land-cover channels the model actually consumes. That correlation is
> an upper bound: it's the best any model driven by those channels could do
> there.
>
> Then you compare the model's transfer score against that bound. If the
> model is near it, the model is fine and the data is the limit. If it's far
> below, the model is the problem.
>
> Two things make this cheap. It needs no simulator, no calibration, no
> meteorology — just a boundary, a land-cover raster, and one satellite
> scene. And there's nothing to fit, so there's no train/test split to get
> wrong.

---

## Slide 4 — Getting the definition right (3:15–5:00)

*On screen:* `outputs/figures/fig1_ceiling_definition.pdf`
*(`_dark.pdf` for a dark deck). Every city sits above the dashed
y = x line; five sit left of zero, in the shaded band.*

> Now, the obvious way to compute that bound is a correlation with
> impervious fraction. That's what I did first, and it's wrong.
>
> The simulator consumes three channels — impervious, vegetation, water. The
> multiple correlation over three predictors is greater than or equal to the
> correlation with any one of them, by construction. So a three-channel
> model can exceed a one-channel "ceiling" without anything being wrong.
>
> On four cities that looked like a technicality. Across forty-nine cities
> it isn't. In five arid cities the univariate version goes *negative* —
> Boise at minus 0.33, Albuquerque minus 0.31, Tucson minus 0.25. It reports
> that no land-cover model can work. Meanwhile a plain three-channel linear
> fit in those same cities reaches R of 0.35 to 0.46.
>
> And a percentage of a negative ceiling is undefined. So the definition
> isn't cosmetic — it decides whether the diagnostic means anything at all.
>
> The correct statement is the multiple correlation over the channels the
> model uses. Across all forty-nine cities, that version is never negative.

**If short on time, cut the city-by-city numbers, keep "goes negative."**

---

## Slide 5 — What predicts a low ceiling (5:00–6:45)

*On screen: gap-vs-ceiling scatter, 49 cities; ρ and CI.*

> So the ceiling exists. The next question is whether you can predict it
> before you model anything.
>
> The best predictor I found is what I'll call the built–vegetated thermal
> gap: the mean temperature of cells that are mostly built up, minus the
> mean of cells that are mostly vegetated. One satellite day and a
> land-cover raster.
>
> Across forty-nine US cities, that gap tracks the ceiling with a Spearman
> rho of 0.69, p around 3 times 10 to the minus 8.
>
> I want to be careful here, because there's a trap I fell into. If you
> measure against the *univariate* ceiling instead, you get 0.84 — a much
> nicer number. But a paired bootstrap says that difference is real, not
> noise. The univariate target flatters the heuristic, partly because the
> gap and the univariate ceiling both flip sign in arid cities, so they move
> together for a shared reason.
>
> I report the honest one: 0.69 against the sound target.

**This slide is where you establish credibility. Don't rush it.**

---

## Slide 6 — Arid cities invert (6:45–8:00)

*On screen: the five negative-ceiling cities; a Tucson map if you have one.*

> The most interesting thing the sweep turned up is what happens in dry
> cities.
>
> The usual story is that land cover separates thermal regimes less well in
> arid climates. What actually happens is stronger than that. The
> relationship *reverses*. In Boise, Albuquerque, Tucson, Phoenix and
> Billings, impervious cover anti-correlates with temperature.
>
> The reason is that natural desert surfaces — which WorldCover labels
> shrubland or grassland, so the model reads them as vegetation — run hotter
> than built-up areas, which carry irrigation, shade and thermal mass.
>
> So it isn't that the signal is weak in these cities. The sign is wrong.
> And that's measured across five independent cities, on multi-day means,
> not a single scene.

---

## Slide 7 — Single days lie (8:00–9:15)

*On screen: Phoenix's 11-day ceiling distribution with the published day marked.*

> A caution that changed several of my own numbers.
>
> I originally computed each ceiling from one cloud-free day. That is not
> stable. Across the sweep, the day-to-day standard deviation of a city's
> ceiling has a median of about 0.085 — comparable to the spacing between
> many cities in the study.
>
> The clearest case is Phoenix. My published ceiling was plus 0.36. Over
> eleven clear days, Phoenix's ceiling ranges from minus 0.32 to plus 0.16,
> with a typical value of minus 0.10. The day I originally picked sits
> *above the maximum* of the city's own distribution.
>
> It wasn't a computation error — I reproduce that day exactly. It was an
> outlier. So every ceiling in this work is now a multi-day mean with a
> spread, and I'd recommend anyone using this diagnostic do the same.

**[CUT] if very short — but it's the most honest slide in the talk, and
audiences respond well to it.**

---

## Slide 8 — Making it actionable (9:15–10:30)

*On screen: NDVI gain vs ceiling scatter; ρ = −0.55.*

> A diagnostic that only tells you you're stuck isn't much use. So: can it
> tell you what would *un*-stick you?
>
> Because the ceiling is a property of the data, you can test a candidate
> input channel without running the model at all. Just recompute the ceiling
> with the channel included. If the ceiling doesn't move, no model using it
> can do better.
>
> I tried satellite NDVI — global, and retrievable anonymously, which
> matters for the zero-authentication design. Across forty-nine cities,
> twenty-seven show a robustly positive gain.
>
> And the gain is *predictable*. It correlates with the existing ceiling at
> rho minus 0.55, and with mean surface temperature at plus 0.50. In plain
> terms: **the channel helps precisely where the ceiling is low, and where
> the city is hot and dry.**
>
> That's a screening rule you can apply to a new city before committing to
> any modelling work.

---

## Slide 9 — The simulator, briefly (10:30–11:15)

*On screen: austin-twin schematic, the r and CI, the data sources.*

> Everything so far is independent of the model I used, which is the point —
> the control travels. But briefly, the vehicle.
>
> austin-twin is a physics-based slab energy-balance model of Austin on a
> 500-metre grid. Every input is retrievable by anonymous HTTP request:
> OpenStreetMap, ESA WorldCover, MODIS, and ERA5 through Google's public
> mirror. No API keys, no registration.
>
> It reaches r of 0.754, with a 95 percent interval of 0.718 to 0.785, on
> held-out satellite days, and it runs a 48-hour simulation in about eleven
> milliseconds on a desktop CPU.
>
> On transfer: Miami reaches 99 percent of its ceiling. Phoenix reaches 31
> percent, Denver about 1 percent — and the ceiling control is what tells
> you those last two are data problems, not model problems.

---

## Slide 10 — What didn't work (11:15–12:00)

*On screen: three negative results, one line each.*

> Three things I tried that failed, because I think they're worth more than
> the successes.
>
> Replacing the simple evapotranspiration term with canonical
> Penman-Monteith made things much worse — a drop of 0.34 in r. I'd
> attributed that to two mechanisms, leaf-area saturation and vapour-deficit
> decoupling. Measured directly, those two together explain only about a
> quarter of the loss. Three quarters is still unexplained. And the
> formulation I'd explicitly rejected as unbounded turns out to be the best
> performer of the ones I tested.
>
> Second, adding wind advection drove the score *negative*. The reason is
> physical: I was advecting the surface field, but ground doesn't move. What
> a wind transports is air, which then exchanges heat with the surface —
> that's a two-layer model, not a transport term.
>
> Third, the trade-off between matching satellite spatial structure and
> matching realistic cooling distances is real and about twice as steep as
> I'd reported.

**Deliver this briskly and without apology. It's a strength.**

---

## Slide 11 — Close (12:00–12:30)

*On screen: the three transferable contributions.*

> To close, three things that transfer beyond this particular simulator.
>
> A data-ceiling control, defined as the multiple correlation over the
> channels a model consumes — which separates model failure from data
> limits, and which you can compute for a new city in minutes.
>
> A screening rule: the built–vegetated thermal gap predicts that ceiling,
> and in arid cities the relationship inverts rather than weakens.
>
> And a way to evaluate a candidate input channel before building anything,
> which tells you the channel helps where the ceiling is low.
>
> Everything — code, calibration outputs, and the per-city data behind every
> number here — is public. Thank you.

---

# Rehearsal checklist

- Slides 4 and 5 carry the argument. If you're behind schedule, take time
  out of 9 and 10, not those.
- Say "multiple correlation over the channels the model consumes" slowly the
  first time. It's the one piece of jargon that must land.
- Practise the Phoenix line on slide 7 — "above the maximum of the city's
  own distribution" is the sentence people will remember.
- Have ready but not on a slide: n = 49, 6 dropped; median 6 clear days per
  city; NDVI 27 of 49 robustly positive; Austin r = 0.754.
- Figure 1 exists (`make_figure1.py`); slides 5, 7 and 8 still need theirs —
  the gap scatter, the Phoenix 11-day distribution, and NDVI gain vs ceiling.
- If a question exposes something you haven't checked, say "I haven't
  verified that" and move on. Several results in this work changed when I
  checked them properly, and that's worth saying out loud.
