from __future__ import annotations

import math


def wilson_lower_bound(
    successes: int,
    trials: int,
    *,
    z: float = 1.96,
) -> float:
    """Return the lower Wilson score bound for a binomial proportion.

    Neutral observations must be removed by the caller before supplying
    ``trials``.  With no decisive observation the conservative lower bound is
    zero.  The result is clamped only to protect against floating-point drift.
    """
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise TypeError("successes must be an integer")
    if isinstance(trials, bool) or not isinstance(trials, int):
        raise TypeError("trials must be an integer")
    if trials < 0:
        raise ValueError("trials must be non-negative")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials")
    if not isinstance(z, (int, float)) or isinstance(z, bool):
        raise TypeError("z must be a finite positive number")
    z_value = float(z)
    if not math.isfinite(z_value) or z_value <= 0.0:
        raise ValueError("z must be a finite positive number")
    if trials == 0:
        return 0.0

    proportion = successes / trials
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / trials
    centre = proportion + z_squared / (2.0 * trials)
    margin = z_value * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z_squared / (4.0 * trials * trials)
    )
    result = (centre - margin) / denominator
    return max(0.0, min(1.0, result))
