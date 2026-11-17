"""Forecast a launch day's Twitch viewers, hour by hour, from the day before.

The case gives one known number, hour 1's print p1, and a library of past big
launches (the comparables), each with hourly viewers from before its release
day to after it. A launch like this one -- a console game unlocking at
midnight local time, country by country -- does not start on its UTC release
day: New Zealand unlocks near 11:00 UTC the day before, so by hour 1 the
category has already lived a whole first day (a ramp, an EU-evening peak, an
overnight decay into hour 1). What hour 1 cannot show is how much audience is
still to come. The day before can: the deeper the overnight trough below the
previous day's peak, the bigger the release day's own surge tends to be.

So the forecast is a per-hour regression on the comparables whose launch had
already begun the day before their release day (their day-before peak at
least 1.2x their hour 1 and at least a fifth of their release day's peak):

    ln(viewers_h / p1) = a_h + b_h * (f1 - mean f1) + noise,

with f1 = ln(day-before peak / p1) of the case itself, read off its own
history; b_h shrunk toward 0 by n / (n + 6) so a handful of launches cannot
bend an hour far. The five quantiles are p1 * exp(mu + z * sigma), sigma the
residual spread at that hour (MAD-based, floored, widened 1.1x). With too few
comparables to learn from, the forecast falls back to persistence -- every
hour at hour 1's count, spread 0.15 * sqrt(h - 1) -- the house baseline.

Run as the grader runs it: python3 -P -s /bot/main.py, standard library
only, no network, deterministic. Reads every *.in in $MAKEFUN_INPUT_DIR
(else /input), writes <id>.out to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, in case a
# module shipped beside this file ever needs importing.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them,
# and the standard normal's quantile at each.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# The persistence fallback's spread, in log units per square-root hour.
BASELINE_SIGMA = 0.15
# A comparable counts as "launched the day before" when its day-before peak
# clears both of these against its own release day.
REGIME_PEAK_OVER_H1 = 1.2
REGIME_PEAK_OVER_DAY_PEAK = 0.2
# Below this many in-regime comparables, learn from all of them instead.
MIN_REGIME = 8
# Below this many points at an hour, that hour falls back to persistence.
MIN_HOUR_POINTS = 6
# The regression slope's shrinkage: b *= n / (n + SHRINK).
SHRINK = 6.0
# The residual spread's floor and its calibration widening.
SIGMA_FLOOR = 0.10
SIGMA_WIDEN = 1.1


def parse_ts(text):
    """An ISO 8601 UTC timestamp ('...Z') as an aware datetime, or None."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def median(xs):
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


def mad_sigma(xs, centre):
    """A robust spread: 1.4826 * median absolute deviation from centre."""
    return 1.4826 * median([abs(x - centre) for x in xs])


def round2(x):
    """x rounded to 2 decimals, half up; whole numbers stay whole in JSON."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def day_maps(hourly, day_start):
    """Split an hourly series around a release day.

    Returns (day0, day_before): hour-of-day 1..24 -> viewers for the release
    day itself and for the 24 hours before it. Hours the source never
    sampled are simply absent.
    """
    day0, day_before = {}, {}
    for point in hourly or []:
        at = parse_ts(point.get("from"))
        viewers = point.get("viewers")
        if at is None or viewers is None:
            continue
        offset = (at - day_start).total_seconds() / 3600.0
        if offset != int(offset):
            continue
        hour = int(offset) + 1
        if 1 <= hour <= 24:
            day0[hour] = viewers
        elif -23 <= hour <= 0:
            day_before[hour + 24] = viewers
    return day0, day_before


def comparable_rows(case):
    """The launches to learn from, as dicts of day curves and features.

    A row is kept when its release day is nearly whole (hour 1 and at least
    20 of its hours present, hour 1 above zero) and its day before has a
    peak, so its f1 -- ln(day-before peak / hour 1) -- is defined.
    """
    rows = []
    for comp in case.get("comparables") or []:
        start = parse_ts(comp.get("day_start"))
        if start is None:
            continue
        day0, day_before = day_maps(comp.get("hourly"), start)
        if day0.get(1, 0) <= 0 or len(day0) < 20 or not day_before:
            continue
        peak_before = max(day_before.values())
        if peak_before <= 0:
            continue
        rows.append({
            "day0": day0,
            "day_before": day_before,
            "f1": math.log(peak_before / day0[1]),
            "in_regime": (peak_before >= REGIME_PEAK_OVER_H1 * day0[1]
                          and peak_before >= REGIME_PEAK_OVER_DAY_PEAK * max(day0.values())),
        })
    return rows


def training_rows(case):
    """The in-regime comparables, or all of them when too few qualify."""
    rows = comparable_rows(case)
    regime = [r for r in rows if r["in_regime"]]
    return regime if len(regime) >= MIN_REGIME else rows


def case_f1(case, day_start, p1, mean_f1):
    """This case's own f1 from its history: ln(day-before peak / hour 1).

    The day before the release day is read from history_hourly; its hours
    are whole-hour means ending at the checkpoint. With too little of that
    day recorded, fall back to the training mean (no adjustment).
    """
    if day_start is None or p1 <= 0:
        return mean_f1
    before = []
    for point in case.get("history_hourly") or []:
        at = parse_ts(point.get("from"))
        viewers = point.get("viewers")
        if at is None or viewers is None:
            continue
        if day_start - timedelta(hours=24) <= at < day_start and viewers > 0:
            before.append(viewers)
    if len(before) < 6:
        return mean_f1
    return math.log(max(before) / p1)


def fit_hour(rows, hour, f1_target):
    """(mu, sigma) of ln(viewers_hour / p1) for a case with the given f1.

    Least squares over the rows that sampled this hour, the slope shrunk by
    n / (n + SHRINK), the target's f1 clipped to the training range so an
    unusual case cannot extrapolate. sigma is the residuals' MAD spread,
    floored and widened; mu carries the residuals' median, so the middle
    quantile is the predictive median. None when too few rows sampled it.
    """
    pts = [(r["f1"], math.log(r["day0"][hour] / r["day0"][1]))
           for r in rows if r["day0"].get(hour, 0) > 0]
    if len(pts) < MIN_HOUR_POINTS:
        return None
    n = len(pts)
    mean_f = sum(p[0] for p in pts) / n
    mean_y = sum(p[1] for p in pts) / n
    var_f = sum((p[0] - mean_f) ** 2 for p in pts)
    cov = sum((p[0] - mean_f) * (p[1] - mean_y) for p in pts)
    slope = cov / var_f * (n / (n + SHRINK)) if var_f > 0 else 0.0
    f_lo = min(p[0] for p in pts)
    f_hi = max(p[0] for p in pts)
    f_clipped = min(max(f1_target, f_lo), f_hi)
    resid = [y - (mean_y + slope * (f - mean_f)) for f, y in pts]
    centre = median(resid)
    mu = mean_y + slope * (f_clipped - mean_f) + centre
    sigma = max(mad_sigma(resid, centre), SIGMA_FLOOR) * SIGMA_WIDEN
    return mu, sigma


def hour_quantiles(p1, hour, fit):
    """One hour's five quantiles: the model's, or persistence without one."""
    if fit is None:
        mu, sigma = 0.0, BASELINE_SIGMA * math.sqrt(hour - 1)
    else:
        mu, sigma = fit
    out = []
    for z in Z:
        q = p1 * math.exp(mu + z * sigma)
        q = min(max(q, 0.01), 1_000_000_000)
        if out and q < out[-1]:
            q = out[-1]
        out.append(round2(q))
    return out


def forecast_case(case):
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is not None and first.get("viewers"):
        p1 = max(1, first["viewers"])
    else:
        # No hour-1 print: anchor on the last recorded point rather than fail.
        seen = [p["viewers"] for p in case.get("history") or [] if p.get("viewers")]
        p1 = max(1, seen[-1]) if seen else 1
    rows = training_rows(case)
    mean_f1 = sum(r["f1"] for r in rows) / len(rows) if rows else 0.0
    f1_target = case_f1(case, parse_ts(case.get("day_start")), p1, mean_f1)
    hours = []
    for row in case.get("forecast") or []:
        hour = row["hour"]
        fit = fit_hour(rows, hour, f1_target) if rows else None
        hours.append({"hour": hour, "q": hour_quantiles(p1, hour, fit)})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
