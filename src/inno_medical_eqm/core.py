"""Predictive maintenance for medical equipment: failure-risk model, health index and constrained maintenance plan.

Lab prototype on SYNTHETIC data only (TRL 4).
"""
import numpy as np

TYPES = {"ventilator": (3, 1.0), "infusion_pump": (1, 0.6), "ecg": (1, 0.7), "ultrasound": (2, 0.8), "autoclave": (2, 0.9)}  # (repair hours, criticality)
FEATURES = ["age_years", "days_since_service", "vibration_rms", "temp_excess_c", "errors_30d", "load_pct"]

CRITERIA = [
    ("auc", ">=", 0.80, "held-out ROC AUC for failure in next 30 days"),
    ("calibration_gap", "<=", 0.04, "|mean predicted - observed failure rate| on held-out devices"),
    ("brier_gain", ">=", 0.05, "Brier score improvement over constant-rate baseline"),
]


def simulate(seed=0, n=1200):
    rng = np.random.default_rng(seed)
    t = rng.choice(list(TYPES), n)
    X = np.column_stack([rng.gamma(3, 2.2, n), rng.uniform(0, 400, n), rng.gamma(2, 0.6, n), rng.gamma(1.5, 2.0, n), rng.poisson(1.2, n), rng.uniform(20, 95, n)])
    w = np.array([0.20, 0.006, 0.55, 0.12, 0.25, 0.015])
    z = (X - X.mean(0)) / X.std(0)
    logit = -3.0 + 1.6 * (z * (w * X.std(0))).sum(1) + rng.normal(0, 0.3, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return t, X, y


def fit(X, y, l2=1e-2, iters=600, lr=0.3):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd]); b = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ b)); g = Z.T @ (p - y) / len(y) + l2 * np.r_[0, b[1:]]
        b -= lr * g
    return {"mu": mu, "sd": sd, "b": b}


def predict(model, X):
    Z = np.column_stack([np.ones(len(X)), (np.asarray(X) - model["mu"]) / model["sd"]])
    return 1 / (1 + np.exp(-Z @ model["b"]))


def health_index(p):
    return 100.0 * (1 - np.asarray(p))


def plan(devices, hours_available):
    """devices: list of dict(id,type,p). Greedy by risk*criticality per repair hour, never exceeding hours_available."""
    cand = sorted(devices, key=lambda d: -(d["p"] * TYPES[d["type"]][1]) / TYPES[d["type"]][0])
    left, out = hours_available, []
    for d in cand:
        h = TYPES[d["type"]][0]
        if h <= left:
            out.append({"id": d["id"], "type": d["type"], "risk": round(float(d["p"]), 3), "hours": h}); left -= h
    return out


def expected_failures_by_type(devices):
    agg = {}
    for d in devices:
        agg[d["type"]] = agg.get(d["type"], 0.0) + d["p"]
    return agg


def auc(y, s):
    order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(seed=0):
    t, X, y = simulate(seed)
    idx = np.random.default_rng(seed + 5).permutation(len(y)); cut = int(0.7 * len(y)); tr, te = idx[:cut], idx[cut:]
    m = fit(X[tr], y[tr]); p = predict(m, X[te])
    base = y[tr].mean()
    brier = np.mean((p - y[te]) ** 2); brier0 = np.mean((base - y[te]) ** 2)
    return {"auc": auc(y[te], p), "calibration_gap": float(abs(p.mean() - y[te].mean())), "brier_gain": float(1 - brier / brier0), "base_rate": float(y.mean())}


def generate(seed=0):
    t, X, y = simulate(seed, 600)
    return {"devices": [dict(device="D%04d" % i, type=t[i], **{f: round(float(X[i, j]), 2) for j, f in enumerate(FEATURES)}, failed_30d=int(y[i])) for i in range(len(y))]}


def demo(seed=0):
    t, X, y = simulate(seed)
    m = fit(X[:900], y[:900]); p = predict(m, X[900:])
    devs = [{"id": "D%d" % i, "type": t[900 + i], "p": p[i]} for i in range(len(p))]
    return "Weekly plan (40 technician-hours): " + str(plan(devs, 40)[:5])
