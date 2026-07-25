# Methodology — [Project Name]

## 1. Objective
What this model/tool estimates or does, and the intended use case.

## 2. Data
- Source(s), sample period, population covered
- Note explicitly if data is synthetic, and why (e.g. real op-risk loss
  databases are proprietary — see limitations)

## 3. Methodology
- Approach taken (e.g. historical simulation, logistic regression + WoE/IV,
  Monte Carlo with correlated shocks)
- Key assumptions and why they were chosen
- Relevant regulatory reference points (e.g. EBA GL 2017/16, Basel
  stress-testing principles) where applicable — cite, don't just imply

## 4. Results
Summary of output, key metrics, plots (link to figures/ folder).

## 5. Limitations & Effective Challenge
This is the section a validator would actually write — be honest about it:
- **Data limitations** — sample size, representativeness, survivorship bias,
  synthetic vs. real-world data gaps
- **Model limitations** — simplifying assumptions that wouldn't hold in
  practice (e.g. i.i.d. returns, normal distribution tails, stationary
  correlations)
- **What a second-line reviewer would challenge** — pick 2-3 concrete
  weaknesses a validator would flag in a real review, and state how you'd
  address them given more time/data
- **Backtesting/monitoring caveats** — anything about the out-of-time window
  that limits how much weight to put on the monitoring results

## 6. References
List any regulatory guidance, papers, or datasets cited above in full.
