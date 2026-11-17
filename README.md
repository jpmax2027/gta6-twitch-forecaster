# gta6-twitch-forecaster

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/jpmax2027/gta6-twitch-forecaster/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/jpmax2027/gta6-twitch-forecaster/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day (2026-11-19, UTC), hours 2 to 24, as five quantiles per hour (5/25/50/75/95%), scored by mean pinball loss on the log scale.

## Approach

The program (`main.py`, Python 3.13, standard library only) learns from the past big launches bundled into every case as `comparables`. A console game that unlocks at midnight local time does not start on its UTC release day: New Zealand unlocks near 11:00 UTC the day before, so by hour 1 the category has already lived a whole first day — a ramp, an EU-evening peak, and an overnight decay into hour 1. What hour 1 cannot show is how much audience is still to come; the day before can. Launches whose day-before peak towers over their overnight trough tend to surge again on release day; launches that stayed near their peak overnight tend to run flat.

Concretely, for each hour h the model fits, over the comparables whose launch had already begun the day before their release day,

    ln(viewers_h / hour-1 print) = a_h + b_h * (f1 - mean f1) + noise,
    f1 = ln(day-before peak / hour-1 print),

with the slope b_h shrunk toward zero by n / (n + 6) and the case's own f1 read from its recorded history and clipped to the training range. Quantiles are `p1 * exp(mu + z * sigma)` with sigma the hour's residual spread (MAD-based, floored at 0.10, widened 1.1x). Where an hour has too few comparables to learn from, the forecast falls back to persistence — flat at hour 1's count with a spread of 0.15 * sqrt(h - 1), the house baseline.

## Validation

On the kit's four practice cases (past launches whose answers are the same SullyGnome hourly counts the cases were built from), this scores a mean pinball loss of about 0.235 against the house baseline's 0.387; leave-one-out over the in-regime comparables gives about 0.27 against the baseline's 0.42 on the same launches. The program is deterministic: two runs on the same case write byte-identical output, in well under a second per case.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input directory and writes one `<id>.out` per case: `{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}` for hours 2 to 24.

---

<sub>Entered on [make.fun](https://make.fun/?ref=zbqpv30a&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
