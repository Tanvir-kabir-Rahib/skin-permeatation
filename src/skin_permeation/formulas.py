from __future__ import annotations

POTTS_GUY_INTERCEPT = -2.72
POTTS_GUY_LOGP_COEFFICIENT = 0.71
POTTS_GUY_MW_COEFFICIENT = -0.0061


def calculate_formula_logkp(log_p: float, molecular_weight: float) -> float:
    """Return the Potts-Guy logKp estimate used as the formula-based reference."""

    return (
        POTTS_GUY_INTERCEPT
        + (POTTS_GUY_LOGP_COEFFICIENT * float(log_p))
        + (POTTS_GUY_MW_COEFFICIENT * float(molecular_weight))
    )
