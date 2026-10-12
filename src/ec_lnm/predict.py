"""Full-precision inference from clinical variables and supplied patient signatures."""
import json
import math
from importlib.resources import files
from pathlib import Path

BINARY = {"HE4", "MR_reported_DMI", "MR_reported_LNM"}

def load_specifications(path=None):
    source = Path(path) if path else files("ec_lnm").joinpath("assets/model_specifications.json")
    return json.loads(source.read_text(encoding="utf-8"))

def expit(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)

def predict(row, spec):
    if len(spec["input_features"]) != len(spec["coefficients"]):
        raise ValueError("Coefficient and feature dimensions differ")
    values = []
    for name in spec["input_features"]:
        value = float(row[name])
        if not math.isfinite(value):
            raise ValueError(f"{name}: missing or non-finite input")
        if name in BINARY and value not in (0, 1):
            raise ValueError(f"{name} must be 0 or 1")
        if name == "MR_reported_diameter" and value <= 0:
            raise ValueError("MR_reported_diameter must be positive, in cm")
        values.append(value)
    eta = spec["intercept"] + sum(b * x for b, x in zip(spec["coefficients"], values))
    return expit(eta)

def nomogram_points(row, spec=None):
    spec = spec or load_specifications()["Nomogram"]
    lower = [0, 0, 0, 0, -3, -4]
    upper = [1, 1, 10, 1, 4, 4]
    beta = spec["coefficients"]
    unit = max(b * (hi - lo) for b, lo, hi in zip(beta, lower, upper)) / 100
    points = [b * (float(row[k]) - lo) / unit for k, b, lo in zip(spec["input_features"], beta, lower)]
    total = sum(points)
    baseline = spec["intercept"] + sum(b * lo for b, lo in zip(beta, lower))
    probability = predict(row, spec)
    if abs(probability - expit(baseline + total * unit)) > 1e-12:
        raise ArithmeticError("Point-to-probability inconsistency")
    return {"component_points": dict(zip(spec["input_features"], points)), "total_points": total, "probability": probability, "logit_per_point": unit, "zero_point_eta": baseline}
