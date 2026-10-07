"""Tests for add_pri12_in_freetime.

After every timing pass and the merge, idle time is offered to the
window's priority-1/2 targets: the most recently observed visible target
wins, the addition joins the visit of the targets around it, flies that
visit's roll, and is judged under its own keepout model. The pattern
double and helpers come from the movement-limit tests.
"""

# Third-party
import numpy as np
from astropy import units as u

# First-party/Local
from shortschedule.models import ScienceCalendar, Visit
from shortschedule.scheduler import ScheduleProcessor
from tests.test_movement_limit import T0, _make_seq, _PatternVis, _processor
from tests.test_priority_0_keepout import NOMINAL, TLE1, TLE2


def _seq(sid, target, start_min, duration_min, priority=1, roll=10.0):
    seq = _make_seq(
        sid, target, start_min=start_min, duration_min=duration_min
    )
    seq.priority, seq.roll = priority, roll
    return seq


def _proc(pattern=None, window_minutes=120, min_num=1, min_duration=12):
    if pattern is None:
        pattern = np.ones(window_minutes, dtype=bool)
    proc = _processor(_PatternVis(pattern))
    proc.window_start = T0
    proc.window_end = T0 + window_minutes * u.min
    proc.pri12_freetime_visibility = None
    proc.pri12_freetime_min_duration = min_duration
    proc.pri12_freetime_fill_min_num = min_num
    # Integration counts are covered elsewhere and need real payloads.
    proc._update_payload_parameters = lambda calendar: calendar
    return proc


def _minute(time):
    return int(np.rint((time - T0).sec / 60.0))


def _spans(visit):
    return [
        (
            seq.target,
            seq.priority,
            _minute(seq.start_time),
            _minute(seq.stop_time),
        )
        for seq in visit.sequences
    ]


def test_most_recent_target_fills_and_merges_into_its_observation(capsys):
    """B was observed last before the idle stretch, so it gets it, and the
    addition joins B's own observation in the same visit."""
    proc = _proc(window_minutes=60)
    calendar = ScienceCalendar(
        metadata={},
        visits=[
            Visit(
                id="v1",
                sequences=[
                    _seq("s1", "A", 0, 10),
                    _seq("s2", "B", 10, 10),
                    _seq("s3", "P", 50, 10, priority=0),
                ],
            )
        ],
    )

    proc._fill_free_time(calendar)

    assert _spans(calendar.visits[0]) == [
        ("A", 1, 0, 10),
        ("B", 1, 10, 50),
        ("P", 0, 50, 60),
    ]
    summary = proc.gap_report["processing_summary"]
    assert summary["free_time_observations_added"] == 1
    assert summary["free_time_minutes_added"] == 30
    assert "FREE TIME: added 30 min of priority-1 B" in capsys.readouterr().out


def test_boundary_stretch_joins_the_visit_holding_the_target():
    """Between a priority-0 visit and B's visit, the addition of B goes to
    B's visit and flies its roll there."""
    proc = _proc(window_minutes=60)
    calendar = ScienceCalendar(
        metadata={},
        visits=[
            Visit(id="v0", sequences=[_seq("s1", "B", 0, 10, roll=30.0)]),
            Visit(id="v1", sequences=[_seq("s1", "P", 10, 10, priority=0)]),
            Visit(id="v2", sequences=[_seq("s1", "B", 40, 20, roll=70.0)]),
        ],
    )

    proc._fill_free_time(calendar)

    assert _spans(calendar.visits[1]) == [("P", 0, 10, 20)]
    assert _spans(calendar.visits[2]) == [("B", 1, 20, 60)]
    assert calendar.visits[2].sequences[0].roll == 70.0


def test_target_new_to_a_visit_flies_its_nearest_visit_roll():
    proc = _proc(window_minutes=70)
    calendar = ScienceCalendar(
        metadata={},
        visits=[
            Visit(id="v1", sequences=[_seq("s1", "A", 0, 10, roll=25.0)]),
            Visit(id="v2", sequences=[_seq("s1", "P", 10, 10, priority=0)]),
            Visit(id="v3", sequences=[_seq("s1", "Q", 50, 20, priority=0)]),
        ],
    )

    proc._fill_free_time(calendar)

    added = [seq for seq in calendar.visits[1].sequences if seq.target == "A"]
    assert [(_minute(s.start_time), _minute(s.stop_time)) for s in added] == [
        (20, 50)
    ]
    assert added[0].roll == 25.0 and added[0].priority == 1


def test_judged_under_its_own_model():
    """The nominal model sees the whole stretch; the stricter one only its
    last 15 minutes, which is all that is added."""
    strict = np.ones(60, dtype=bool)
    strict[20:35] = False
    proc = _proc(window_minutes=60)
    proc.pri12_freetime_visibility = _PatternVis(strict)
    calendar = ScienceCalendar(
        metadata={},
        visits=[
            Visit(
                id="v1",
                sequences=[
                    _seq("s1", "A", 0, 20),
                    _seq("s2", "P", 50, 10, priority=0),
                ],
            )
        ],
    )

    proc._fill_free_time(calendar)

    assert _spans(calendar.visits[0]) == [
        ("A", 1, 0, 20),
        ("A", 1, 35, 50),
        ("P", 0, 50, 60),
    ]


def test_stretches_shorter_than_the_minimum_are_left_idle():
    proc = _proc(window_minutes=40, min_duration=12)
    calendar = ScienceCalendar(
        metadata={},
        visits=[
            Visit(
                id="v1",
                sequences=[
                    _seq("s1", "A", 0, 20),
                    _seq("s2", "P", 30, 10, priority=0),
                ],
            )
        ],
    )

    proc._fill_free_time(calendar)

    assert len(calendar.visits[0].sequences) == 2
    assert (
        proc.gap_report["processing_summary"]["free_time_observations_added"]
        == 0
    )


def test_a_visit_below_the_minimum_count_gets_nothing(capsys):
    """Two stretches fit in the visit, three are required."""
    proc = _proc(window_minutes=100, min_num=3)
    calendar = ScienceCalendar(
        metadata={},
        visits=[
            Visit(
                id="v1",
                sequences=[
                    _seq("s1", "A", 0, 10),
                    _seq("s2", "P", 30, 10, priority=0),
                    _seq("s3", "Q", 60, 40, priority=0),
                ],
            )
        ],
    )

    proc._fill_free_time(calendar)

    assert len(calendar.visits[0].sequences) == 3
    assert "2 observation(s) fit its idle time" in capsys.readouterr().out


def test_mixed_priorities_never_merge():
    """The addition is priority 1, its neighbor priority 2: kept apart."""
    proc = _proc(window_minutes=60)
    calendar = ScienceCalendar(
        metadata={},
        visits=[
            Visit(
                id="v1",
                sequences=[
                    _seq("s1", "A", 0, 20, priority=2),
                    _seq("s2", "P", 50, 10, priority=0),
                ],
            )
        ],
    )

    proc._fill_free_time(calendar)

    assert _spans(calendar.visits[0]) == [
        ("A", 2, 0, 20),
        ("A", 1, 20, 50),
        ("P", 0, 50, 60),
    ]


class TestConfiguration:
    def test_off_by_default_and_no_model_built(self):
        scheduler = ScheduleProcessor(TLE1, TLE2, **NOMINAL)

        assert scheduler.add_pri12_in_freetime is False
        assert scheduler.pri12_freetime_visibility is None
        settings = scheduler._settings_for_header()
        assert settings["Add_Pri12_In_Freetime"] == "False"
        assert "Pri12_Freetime_Fill_Earthlimb_Min_Deg" not in settings

    def test_its_limb_is_flat_and_written_to_the_header(self):
        scheduler = ScheduleProcessor(
            TLE1,
            TLE2,
            add_pri12_in_freetime=True,
            pri12_freetime_fill_earthlimb_min=59,
            **NOMINAL,
        )
        model = scheduler.pri12_freetime_visibility

        assert model.use_dynamic_earthlimb is False
        assert (
            float(getattr(model.earthlimb_min, "value", model.earthlimb_min))
            == 59
        )
        settings = scheduler._settings_for_header()
        assert settings["Add_Pri12_In_Freetime"] == "True"
        assert settings["Pri12_Freetime_Fill_Earthlimb_Min_Deg"] == "59"
        assert settings["Pri12_Freetime_Min_Duration_Min"] == "12"
        assert settings["Pri12_Freetime_Fill_Min_Num"] == "3"

    def test_none_judges_additions_under_the_nominal_model(self):
        scheduler = ScheduleProcessor(
            TLE1,
            TLE2,
            add_pri12_in_freetime=True,
            pri12_freetime_fill_earthlimb_min=None,
            **NOMINAL,
        )

        assert scheduler.pri12_freetime_visibility is None
        assert (
            "Pri12_Freetime_Fill_Earthlimb_Min_Deg"
            not in scheduler._settings_for_header()
        )
