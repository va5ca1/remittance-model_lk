# Sri Lanka Remittance Model

*A first attempt at econometrics by someone who just learned what a terminal is.*

---

![Final Dashboard](outputs/figures/final_dashboard.png)

---

## What is this?

I wanted to understand what actually drives the money Sri Lankan migrant 
workers send home from the Gulf. Remittances — worker money transfers — are 
one of Sri Lanka's biggest sources of foreign income, and I kept wondering: 
does it go up when oil prices rise? Does a weaker rupee make people send more? 
What did the 2022 economic collapse actually do to these flows?

So I built a model. With Python. Having never written a line of Python before.

This README is me walking through what I found, what confused me, and what 
I think it means. I did A-Level Statistics so I'm not coming in completely 
cold on the concepts — I know what a regression is, I know what R² means — 
but the econometrics side of this (time series, structural breaks, AR models) 
was genuinely new territory.

---

## The Data

Four sources, all public:

| What | Where from | How often | How far back |
|---|---|---|---|
| Remittances into Sri Lanka | Central Bank of Sri Lanka | Monthly | 2009 |
| LKR/USD exchange rate | Central Bank of Sri Lanka | Monthly | 1986 |
| Brent crude oil price | World Bank Pink Sheet | Monthly | 1960 |
| GCC countries' GDP | IMF World Economic Outlook | Annual | 2009 |

Getting these into one clean dataset was honestly the hardest part. The CBSL 
files use merged cells, the IMF export was a corrupted binary, and the World 
Bank dates are formatted like `1960M01`. The cleaning script 
(`src/02_data_cleaning.py`) handles all of that and spits out one tidy CSV: 
195 monthly rows from January 2009 to March 2025.

---

## What I tested

The core question is: **what predicts remittances?**

I used three candidate predictors:
- **LKR/USD exchange rate** — if the rupee weakens, each dollar sent home is 
  worth more LKR, which should theoretically encourage sending more
- **Brent oil price** — a proxy for how well the Gulf economy is doing; 
  Gulf countries run on oil revenue, which funds construction and services 
  jobs where most Sri Lankan migrants work
- **GCC GDP** — a more direct measure of Gulf economic output

I ran four models, each asking a slightly different version of the question.

---

## The Models — and what they actually told me

### Model 1 — The Basic One (Log-Log OLS, R² = 0.285)

This was my starting point. I took the log of everything and ran a standard 
regression. The idea with logs is nice: the coefficients become elasticities, 
so a coefficient of 0.5 means "a 1% increase in X leads to a 0.5% increase 
in remittances."

**What came out:** the signs were backwards. Higher oil prices apparently 
*reduce* remittances? A weaker rupee *reduces* remittances? That makes no 
sense economically.

**Why it happened:** this is called spurious regression, and it's a classic 
trap in time-series data. The Durbin-Watson statistic (a measure of whether 
your errors are random) came back at 0.40 — near zero is a red flag. What's 
happening is that all three variables trend upward over 2009–2025, and the 
regression is picking up those shared trends rather than real relationships. 
Oil was low in 2015–2016, remittances happened to be high then — so the model 
"concludes" oil hurts remittances. That's coincidence, not causation.

Finding this felt like a proper econometrics moment. The R² of 0.285 with a 
DW of 0.40 is basically the textbook Granger-Newbold (1974) spurious 
regression example. Spotting it is the point.

One coefficient did survive with a sensible sign: **GCC GDP (+1.43***)** — 
higher Gulf economic output is strongly associated with more remittances. 
That one holds up throughout every model.

---

### Model 2 — Adding the Crisis (R² = 0.427) ← *this is the main one*

The 2022 Sri Lanka crisis — sovereign default, rupee collapse from ~200 to 
~370 per dollar almost overnight — is the most dramatic event in the sample. 
So I added two dummy variables: one for the acute crisis months (2022), and 
one for the permanent post-crisis period after.

R² jumps from 0.285 to **0.427**. That 14 percentage point jump from just 
two variables is telling you: the crisis wasn't just a big swing, it was a 
*structural* event that changed the level and behaviour of remittances.

The two crisis coefficients:

**crisis_period = −0.52*****: during the acute 2022 crisis months, remittances 
were about 40% lower than the macro fundamentals would predict. Families 
needed money but the banking system was breaking down — informal channels, 
restrictions on transfers, and general chaos suppressed the flows.

**post_crisis = −0.20***: after the crisis stabilised, remittances settled at 
a level about 18% below the pre-crisis trend. A permanent downward shift. 
Some workers came home and didn't go back. Some Gulf employers stopped 
hiring. The relationship between fundamentals and remittances changed.

Something else interesting: once I added the crisis dummies, the exchange 
rate coefficient completely lost significance (p = 0.865). That tells you the 
"exchange rate effect" in Model 1 was mostly just the crisis in disguise — 
the rupee crashed during the crisis, so they were confounded until I 
controlled for the crisis directly.

---

### Model 3 — Growth Rates (R² = 0.059)

Instead of log levels, I used year-on-year growth rates. This strips out 
the trending behaviour and asks: do month-to-month swings in oil predict 
month-to-month swings in remittances?

Barely. R² = 0.059. Oil growth is significant but the relationship is weak 
once you remove the trend. The DW was 0.29 which is still alarming — even 
in growth rates there's autocorrelation. That means remittances have strong 
momentum; last month matters a lot.

---

### Model 4 — AR(1): adding last month as a predictor (R² = 0.799)

This is the one that actually fits well. I added last month's remittances 
as an extra predictor. The AR coefficient came out at **0.873*** — meaning 
87% of last month's level carries into this month. Remittances are sticky.

DW improves to 2.62 (that's clean). R² hits 0.799. The downside: once the 
lag is in, oil price and exchange rate lose significance — the lag absorbs 
their signal. This doesn't mean they're irrelevant; it means their effect 
plays out over several months rather than instantly.

I used this as the base for the forecasts.

---

## The Structural Break Test

![CUSUM Plot](outputs/figures/chow_cusum_plot.png)

Chow test at April 2022: **F = 2.23, p = 0.068** — significant at 10%, 
borderline at 5%.

On its own that sounds weak. But look at what happens when you split the 
sample and run separate regressions:

| Period | N | R² |
|---|---|---|
| Pre-crisis (Jan 2009 – Mar 2022) | 159 | 0.228 |
| Post-crisis (Apr 2022 – Mar 2025) | 36 | **0.798** |

The model explains 23% of variance pre-crisis and 80% post-crisis. That's a 
massive regime shift. My interpretation: before 2022, remittances were partly 
driven by seasonal cycles, informal hawala channels, and things not in this 
model. After 2022, workers started using formal banking channels to take 
advantage of the weaker rupee — making the flows much more responsive to 
macro variables.

The Brent oil interaction term changed significantly: oil price elasticity 
went from −0.45 pre-crisis to −1.28 post-crisis (interaction term significant 
at 5%). Gulf workers became more financially exposed to oil price swings after 
the crisis — possibly because remittance pressure from families increased.

The CUSUM test confirms: the model residuals exit the 5% confidence band 
around 2022, exactly where you'd expect.

The Chow test p-value being 0.068 rather than 0.05 is a legitimate limitation 
worth being upfront about — 36 post-crisis observations is a small sample, 
which limits the test's power.

---

## Scenario Forecasts (12 months from April 2025)

![Forecast Chart](outputs/figures/forecast_scenarios.png)

Using the AR(1) model, I simulated four possible futures:

**Baseline** — oil flat around $75, LKR stable, GCC modest growth.  
Remittances continue their post-crisis recovery trajectory.

**Gulf Boom** — oil rises ~27% over 12 months, strong GCC expansion.  
Best case for remittances; Gulf hiring picks up.

**GCC Recession** — oil falls ~35%, GCC GDP contracts.  
Significant downside; the oil-remittance link means a Gulf slowdown hits 
Sri Lankan households directly.

**LKR Crisis Repeat** — rupee depreciates ~35% again.  
Mixed: more LKR per dollar sent could encourage transfers, but a currency 
collapse also signals economic instability that historically suppresses flows.

These are scenario simulations, not point predictions. The point is to 
illustrate sensitivity to different macro environments.

---

## Summary of findings

1. **GCC economic output is the dominant driver.** The GCC GDP elasticity 
   (+1.14) is the most stable coefficient across every specification. The Gulf 
   economy matters more than the exchange rate for determining remittance flows.

2. **The 2022 crisis was structural, not cyclical.** Model fit triples 
   post-crisis. The crisis changed how remittances respond to fundamentals, 
   not just the level.

3. **The oil price elasticity doubled post-crisis.** Workers became more 
   exposed to Gulf commodity cycles after 2022.

4. **Remittances are persistent.** AR coefficient of 0.87 — shocks decay 
   over 5–6 months.

5. **Spurious regression is real and easy to stumble into.** The naive OLS 
   produced backwards signs. Recognising and diagnosing this was the most 
   important methodological step.

---

## How to run this yourself

```bash
git clone https://github.com/[yourusername]/sl-remittance-model
cd sl-remittance-model
pip install -r requirements.txt

python src/02_data_cleaning.py
python src/03_exploratory_analysis.py
python src/04_model_estimation.py
python src/05_chow_test.py
python src/06_forecast_simulation.py
python src/07_final_dashboard.py
```

---

## Project structure