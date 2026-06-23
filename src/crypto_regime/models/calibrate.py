"""Probability calibration helpers."""


def calibrate_classifier(model, features, target, method: str = "isotonic"):
    """Fit a calibrated wrapper around an existing classifier."""
    from sklearn.calibration import CalibratedClassifierCV

    calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    return calibrated.fit(features, target)
