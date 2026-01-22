"""
One Euro Filter implementation for signal smoothing.

Extracted to its own module to avoid circular imports.
"""

import math
from typing import Optional


class OneEuroFilter:
    """
    One Euro Filter - adaptive low-pass filter for noisy input.

    Adapts smoothing based on signal speed:
    - Slow movement = heavy smoothing (reduces jitter)
    - Fast movement = light smoothing (reduces latency)

    Reference: Casiez et al. "1€ Filter: A Simple Speed-based Low-pass
    Filter for Noisy Input in Interactive Systems" (CHI 2012)
    """

    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0
    ):
        """
        Initialize One Euro Filter.

        Args:
            min_cutoff: Minimum cutoff frequency (Hz). Lower = smoother but more lag.
                        Good starting value: 1.0
            beta: Speed coefficient. Higher = more responsive to fast movements.
                  Good starting value: 0.007
            d_cutoff: Derivative cutoff frequency for velocity smoothing.
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    def _smoothing_factor(self, te: float, cutoff: float) -> float:
        """Calculate smoothing factor alpha from cutoff frequency."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x: float, t: Optional[float] = None) -> float:
        """
        Apply One Euro Filter to a single value.

        Args:
            x: Input value.
            t: Timestamp in seconds. If None, uses current time.

        Returns:
            Filtered value.
        """
        import time
        if t is None:
            t = time.perf_counter()

        if self._x_prev is None:
            self._x_prev = x
            self._t_prev = t
            return x

        te = t - self._t_prev
        if te <= 0:
            return self._x_prev

        # Estimate velocity (derivative)
        dx = (x - self._x_prev) / te

        # Smooth the derivative
        a_d = self._smoothing_factor(te, self.d_cutoff)
        dx_smooth = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Adaptive cutoff based on velocity
        cutoff = self.min_cutoff + self.beta * abs(dx_smooth)

        # Filter the signal
        a = self._smoothing_factor(te, cutoff)
        x_filtered = a * x + (1.0 - a) * self._x_prev

        # Update state
        self._x_prev = x_filtered
        self._dx_prev = dx_smooth
        self._t_prev = t

        return x_filtered

    def reset(self) -> None:
        """Reset filter state."""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None
