"""Shared pieces for the visibility stand-ins the tests build."""

# Third-party
import numpy as np


class BestRollFromVisibility:
    """Gives a visibility double the ``get_best_roll`` the roll sweep calls.

    Answers roll 0 and marks visible whatever the double's own
    ``get_visibility`` says, so a test that scripts visibility by the
    minute keeps driving the sweep the same way.
    """

    def get_best_roll(self, coord, times, roll_step=None,
                      min_power_frac=None, weights=None):
        visible = np.atleast_1d(
            np.asarray(self.get_visibility(coord, times), dtype=bool)
        )
        return {
            "roll_deg": 0.0,
            "n_visible": int(visible.sum()),
            "visible": visible,
            "boresight_visible": visible,
            "n_st_pass": visible.astype(int),
            "solar_power_frac": np.where(visible, 1.0, np.nan),
        }
