"""Prediction helpers."""


def predict_regime(model, features):
    """Return class predictions from a fitted model."""
    return model.predict(features)


def predict_regime_probabilities(model, features):
    """Return class probabilities from a fitted model."""
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model does not expose predict_proba")
    return model.predict_proba(features)
