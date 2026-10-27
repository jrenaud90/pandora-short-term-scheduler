"""Transit timing for priority-2 observations.

The long-term calendar places a planet's priority-2 observations around one
of its transits: baseline before, observations across the transit, baseline
after. The short-term scheduler must not move or shrink them so far that the
transit itself goes unobserved, so it needs each transit's predicted time.

Ephemerides come from PandoraTargetList's target definition files (T0 in
BJD_TDB, period, total duration T14), read directly as JSON: the package
itself is not imported, since its dependencies pin an older Python. The
transit timing follows ``pandoraobservations.transits.transit_contacts``.
"""

# Standard library
import json
import re
from pathlib import Path
from typing import Dict, Tuple

# Third-party
import numpy as np
from astropy import units as u
from astropy.constants import c as SPEED_OF_LIGHT
from astropy.coordinates import get_body_barycentric
from astropy.time import Time, TimeDelta

# Definition file keys an ephemeris needs
DEFINITION_KEYS = {
    "planet_name": "Planet Name",
    "period_days": "Period (days)",
    "epoch_bjd_tdb": "Transit Epoch (BJD_TDB)",
    "duration_hr": "Transit Duration (hrs)",
    "ra_deg": "RA",
    "dec_deg": "DEC",
}


def planet_key(name: str) -> str:
    """Name with case and every non-alphanumeric character dropped.

    The calendar and the definition files spell names differently
    (``"HAT-P-11b"``, ``"HAT-P-11 b"``, ``"V1298_Tauc"``); this key is what
    they are matched on.
    """
    return re.sub(r"[^0-9a-z]", "", str(name).lower())


def load_ephemerides(target_list_dir) -> Dict[str, dict]:
    """Transit ephemeris of every planet in a PandoraTargetList tree.

    Parameters
    ----------
    target_list_dir : str or Path
        A PandoraTargetList checkout, or its ``target_definition_files``
        directory.

    Returns
    -------
    dict
        ``planet_key`` of the planet name to ``planet_name``,
        ``period_days``, ``epoch_bjd_tdb``, ``duration_hr``, ``ra_deg``,
        ``dec_deg`` and ``definition_file``. A planet listed in several
        categories takes its first definition in path order; one missing
        any of those values is left out.

    Raises
    ------
    FileNotFoundError
        If the directory holds no ``target_definition_files`` tree.
    """
    tree = Path(target_list_dir)
    if (tree / "target_definition_files").is_dir():
        tree = tree / "target_definition_files"
    if tree.name != "target_definition_files" or not tree.is_dir():
        raise FileNotFoundError(
            f"{target_list_dir} is not a PandoraTargetList checkout or its "
            "target_definition_files directory."
        )

    ephemerides: Dict[str, dict] = {}
    for path in sorted(tree.rglob("*_target_definition.json")):
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        values = {key: info.get(name) for key, name in DEFINITION_KEYS.items()}
        if any(value is None for value in values.values()):
            continue
        values["definition_file"] = str(path.relative_to(tree))
        ephemerides.setdefault(planet_key(values["planet_name"]), values)
    return ephemerides


def transit_epoch(ephemeris: dict, near: Time) -> int:
    """Orbits since the ephemeris epoch of the transit nearest ``near``.

    The barycentric correction (minutes) is left out, so a time almost
    exactly half a period from two transits may pick either.
    """
    return int(
        np.rint(
            (near.tdb.jd - ephemeris["epoch_bjd_tdb"])
            / ephemeris["period_days"]
        )
    )


def transit_core(
    ephemeris: dict, epoch: int, middle_frac: float
) -> Tuple[Time, Time]:
    """The middle of one transit, in geocentric UTC.

    The predicted mid-transit (BJD_TDB) is moved from the solar system
    barycenter to the geocenter by removing the light travel time toward
    the star along the Earth's barycentric position (up to 8.3 min; no IERS
    table needed). Pandora's offset from the geocenter (under 25 ms) is
    ignored.

    Parameters
    ----------
    ephemeris : dict
        One entry of ``load_ephemerides``.
    epoch : int
        Orbits since the ephemeris epoch (``transit_epoch``).
    middle_frac : float
        Share of the total duration T14, centered on mid-transit, that the
        core spans.

    Returns
    -------
    core_start, core_stop : astropy.time.Time
        The core, in UTC.
    """
    mid_bjd = Time(
        ephemeris["epoch_bjd_tdb"] + epoch * ephemeris["period_days"],
        format="jd",
        scale="tdb",
    )

    ra = np.radians(ephemeris["ra_deg"])
    dec = np.radians(ephemeris["dec_deg"])
    toward_star = np.array(
        [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
    )
    earth = get_body_barycentric("earth", mid_bjd, ephemeris="builtin")
    delay = toward_star @ earth.xyz.to_value(u.m) / SPEED_OF_LIGHT.value
    mid = (mid_bjd - TimeDelta(delay, format="sec")).utc

    half_core = 0.5 * middle_frac * ephemeris["duration_hr"] * u.hour
    return mid - half_core, mid + half_core
