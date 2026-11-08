"""Dagster definitions: the retraining job and the sensor that can start it.

    dagster dev -m frauddetect.orchestration        (the web page is http://localhost:3000)

The job checks for missed fraud, trains a challenger, compares it with the live
model and promotes it only if it passes the gate in mlops.py. All the logic
lives in plain functions elsewhere. This file only decides the order and gives
each step a place to show up in the interface.
"""

import os
from dataclasses import replace
from pathlib import Path

import psycopg
from dagster import (
    Config,
    DefaultSensorStatus,
    Definitions,
    OpExecutionContext,
    RunRequest,
    SkipReason,
    job,
    op,
    sensor,
)

from frauddetect.config import Paths, PipelineConfig, RetrainConfig
from frauddetect.mlops import compare, train_challenger
from frauddetect.monitor import build_report, retrain_trigger
from frauddetect.registry import promote


class PlanConfig(Config):
    """What to retrain on. `as_of_day` is the simulated day whose labels are used."""

    as_of_day: float
    n_trials: int = 10
    check_database: bool = True
    # Empty means the standard locations. Tests point these somewhere private.
    artifact_dir: str = ""
    data_dir: str = ""


def pipeline_config(state: dict) -> PipelineConfig:
    paths = Paths()
    if state["artifact_dir"]:
        paths = replace(paths, artifact_dir=Path(state["artifact_dir"]))
    if state["data_dir"]:
        paths = replace(paths, data_dir=Path(state["data_dir"]))
    return PipelineConfig(paths=paths, n_trials=state["n_trials"])


@op
def plan_retrain(config: PlanConfig) -> dict:
    return {
        "as_of_day": config.as_of_day,
        "n_trials": config.n_trials,
        "check_database": config.check_database,
        "artifact_dir": config.artifact_dir,
        "data_dir": config.data_dir,
    }


@op
def check_missed_fraud(context: OpExecutionContext, state: dict) -> dict:
    """Report on missed fraud and input drift. It informs the run but never stops it."""
    report = {}
    if state["check_database"]:
        day = int(state["as_of_day"])
        with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
            report = build_report(connection, pipeline_config(state).paths, day - 14, day + 1)
        recall = report["recall"]
        context.log.info(
            "Confirmed fraud %s, caught %s, alert=%s", recall["confirmed_fraud"], recall["caught"], recall["alert"]
        )
    return {**state, "report": report}


@op
def train_challenger_step(context: OpExecutionContext, state: dict) -> dict:
    trained = train_challenger(state["as_of_day"], pipeline_config(state))
    context.log.info("Trained %s on %s frauds", trained["version"], trained["train_frauds"])
    return {**state, "trained": trained}


@op
def compare_step(context: OpExecutionContext, state: dict) -> dict:
    result = compare(
        state["trained"]["version"], state["as_of_day"], pipeline_config(state), RetrainConfig()
    )
    context.log.info("Gate: promote=%s. %s", result["promote"], result["reason"])
    return {**state, "comparison": result}


@op
def promote_step(context: OpExecutionContext, state: dict) -> dict:
    comparison = state["comparison"]
    if comparison["promote"]:
        promote(pipeline_config(state).paths, comparison["challenger"])
        context.log.warning("%s is now live", comparison["challenger"])
    else:
        context.log.info("Keeping %s. %s", comparison["champion"], comparison["reason"])
    return {
        "live": comparison["challenger"] if comparison["promote"] else comparison["champion"],
        "promoted": comparison["promote"],
        "reason": comparison["reason"],
    }


@job
def retrain_job():
    promote_step(compare_step(train_challenger_step(check_missed_fraud(plan_retrain()))))


@sensor(
    job=retrain_job,
    minimum_interval_seconds=300,
    # Starting a retrain on its own is a decision for whoever runs the system.
    default_status=DefaultSensorStatus.STOPPED,
)
def missed_fraud_sensor():
    url = os.environ.get("DATABASE_URL")
    if not url:
        return SkipReason("DATABASE_URL is not set")
    with psycopg.connect(url) as connection:
        trigger = retrain_trigger(connection)
    if trigger is None:
        return SkipReason("confirmed fraud is being caught at an acceptable rate")
    return RunRequest(
        run_key=f"day-{trigger['as_of_day']}",
        run_config={"ops": {"plan_retrain": {"config": {"as_of_day": float(trigger["as_of_day"])}}}},
    )


defs = Definitions(jobs=[retrain_job], sensors=[missed_fraud_sensor])
