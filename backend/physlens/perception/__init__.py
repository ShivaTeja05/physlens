from .base import Detection, Tracker, TrackResult
from .motion_tracker import MotionBlobTracker
from .opencv_tracker import ColorBlobTracker


def default_tracker(color: str | None = None) -> Tracker:
    """SAM 3.1 on the Meta Model API when a key is configured, otherwise the OpenCV motion tracker (colour is
    only a hint). Set PHYSLENS_TRACKER=motion to force the offline tracker (no API calls, no cost)."""
    import os
    if os.environ.get("PHYSLENS_TRACKER", "").lower() != "motion":
        try:
            from .sam_tracker import SamTracker
            return SamTracker(color=color)
        except RuntimeError:
            pass
    return MotionBlobTracker(color=color)
