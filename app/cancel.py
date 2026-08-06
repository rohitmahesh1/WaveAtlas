from __future__ import annotations


class CancellationRequested(Exception):
    """Raised to stop a running job after the user requests cancellation."""

