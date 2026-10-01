from __future__ import annotations


def risk_band(probability: float) -> tuple[str, float]:
    p = max(0.0, min(1.0, float(probability)))
    if p < 0.25:
        return "Low", p * 100.0
    if p < 0.50:
        return "Medium", p * 100.0
    return "High", p * 100.0
