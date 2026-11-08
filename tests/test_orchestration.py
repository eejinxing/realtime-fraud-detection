import pytest

from frauddetect.config import DAY
from frauddetect.monitor import retrain_trigger
from frauddetect.registry import read_pointer

dagster = pytest.importorskip("dagster")


class FakeConnection:
    """Just enough of a database connection for the monitoring queries."""

    def __init__(self, newest_ts, cohort_rows):
        self.newest_ts = newest_ts
        self.cohort_rows = cohort_rows

    def cursor(self):
        return FakeCursor(self)


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.sql = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.sql = sql

    def fetchone(self):
        return (self.connection.newest_ts,)

    def fetchall(self):
        return self.connection.cohort_rows


def test_trigger_fires_when_confirmed_fraud_is_being_missed():
    # (day, model_version, decisions, confirmed_fraud, caught). Today is day 111.
    rows = [(100, "v1", 3000, 25, 12), (104, "v1", 3000, 20, 9)]
    trigger = retrain_trigger(FakeConnection(111.5 * DAY, rows))
    assert trigger["as_of_day"] == 111
    assert trigger["alert"] and trigger["recall"] < 0.85


def test_trigger_ignores_days_whose_chargebacks_have_not_arrived_yet():
    """Yesterday's fraud has barely been confirmed, so yesterday proves nothing."""
    recent_only = [(109, "v1", 3000, 40, 0), (110, "v1", 3000, 40, 0)]
    assert retrain_trigger(FakeConnection(111.5 * DAY, recent_only)) is None


def test_trigger_stays_quiet_when_fraud_is_caught_or_data_is_thin():
    caught = [(100, "v1", 3000, 25, 24), (104, "v1", 3000, 20, 19)]
    assert retrain_trigger(FakeConnection(111.5 * DAY, caught)) is None
    thin = [(100, "v1", 3000, 3, 0)]
    assert retrain_trigger(FakeConnection(111.5 * DAY, thin)) is None
    assert retrain_trigger(FakeConnection(None, [])) is None


def test_retrain_job_runs_end_to_end_and_the_live_model_follows_the_gate(scratch_run):
    from frauddetect.orchestration import retrain_job

    paths, _ = scratch_run
    config = {
        "as_of_day": 110.0,
        "n_trials": 0,
        "check_database": False,
        "artifact_dir": str(paths.artifact_dir),
        "data_dir": str(paths.data_dir),
    }
    result = retrain_job.execute_in_process(
        run_config={"ops": {"plan_retrain": {"config": config}}}
    )
    assert result.success
    outcome = result.output_for_node("promote_step")
    # Whatever the gate decided, the live model must be the one it said.
    assert outcome["live"] == read_pointer(paths)
    assert outcome["live"] == ("v2" if outcome["promoted"] else "v1")
    assert outcome["reason"]


def test_definitions_load_with_a_stopped_sensor():
    from frauddetect.orchestration import defs

    assert defs.get_job_def("retrain_job")
    sensor = defs.get_sensor_def("missed_fraud_sensor")
    assert sensor.default_status == dagster.DefaultSensorStatus.STOPPED
