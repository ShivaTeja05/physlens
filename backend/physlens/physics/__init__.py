"""Deterministic physics: calibration, segmentation, model fitting and prediction.

Nothing in here calls an AI model. Every number PhysLens shows comes from this package.
"""
from .calibration import Calibration, calibrate_known_object, calibrate_ruler
from .models import FitResult, akaike_weights, consistent_flights, fit_all, fit_free_fall, fit_horizontal
from .predict import (BounceLaw, BouncePrediction, ContactReading, Estimate, contact_samples, measure_flight,
                      predict_next_flight, restitution_samples)
from .segment import Flight, find_bounces, split_flights

STANDARD_GRAVITY = 9.81
