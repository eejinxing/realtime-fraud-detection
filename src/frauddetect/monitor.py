"""Watches the live system for two different kinds of trouble.

    python -m frauddetect.monitor report --since-day 100

* Input drift (PSI): did the data the model sees change? It needs no labels, so
  it answers at once. It cannot see fraud that looks like normal spending.
* Missed fraud: of the fraud that chargebacks have since confirmed, how much
  did the model let through? This is the real measure, but it only becomes
  visible days after the fact.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import psycopg

from frauddetect.config import DAY, Paths
from frauddetect.drift import feature_psi
from frauddetect.registry import load_meta, read_pointer

RECALL_ALERT_BELOW = 0.85
MIN_CONFIRMED_FRAUD = 30
PSI_WATCH = 0.1
PSI_ALERT = 0.25

COHORT_SQL = """
    SELECT day, model_version, decisions, confirmed_fraud, caught
    FROM cohort_recall WHERE day >= %s AND day < %s ORDER BY day, model_version
"""
LIVE_SQL = """
    SELECT features, score FROM decisions
    WHERE event_ts >= %s AND event_ts < %s AND model_version = %s AND features IS NOT NULL
"""


def cohort_recall(connection, since_day: int, until_day: int) -> pd.DataFrame:
    with connection.cursor() as cursor:
        cursor.execute(COHORT_SQL, (since_day, until_day))
        columns = ["day", "model_version", "decisions", "confirmed_fraud", "caught"]
        return pd.DataFrame(cursor.fetchall(), columns=columns)


def recall_alert(
    cohorts: pd.DataFrame,
    threshold: float = RECALL_ALERT_BELOW,
    min_confirmed: int = MIN_CONFIRMED_FRAUD,
) -> dict:
    """Alert when the confirmed fraud in these cohorts was caught too rarely.

    With too few confirmed frauds the number means little, so no alert is
    raised until there are enough of them.
    """
    confirmed = int(cohorts["confirmed_fraud"].sum())
    caught = int(cohorts["caught"].sum())
    recall = caught / confirmed if confirmed else None
    enough = confirmed >= min_confirmed
    return {
        "confirmed_fraud": confirmed,
        "caught": caught,
        "recall": recall,
        "enough_data": enough,
        "alert": bool(enough and recall is not None and recall < threshold),
    }


def retrain_trigger(
    connection, lookback_days: int = 21, maturity_days: int = 7
) -> dict | None:
    """Should a retrain start? Returns what to retrain on, or None if all is well.

    The clock is the newest decision in the database. The simulation runs far
    ahead of real time, so the wall clock would say nothing useful.

    Only days at least `maturity_days` old are judged. A chargeback takes about
    that long to arrive, so the last few days hold almost no confirmed fraud
    yet, and a verdict on them would rest on too little to mean anything.
    """
    with connection.cursor() as cursor:
        cursor.execute("SELECT max(event_ts) FROM decisions")
        newest = cursor.fetchone()[0]
    if newest is None:
        return None
    today = int(newest // DAY)
    cohorts = cohort_recall(connection, today - lookback_days, today - maturity_days + 1)
    alert = recall_alert(cohorts)
    return {"as_of_day": today, **alert} if alert["alert"] else None


def live_psi(
    connection, reference: dict, from_day: int, to_day: int, version: str
) -> dict[str, float]:
    with connection.cursor() as cursor:
        cursor.execute(LIVE_SQL, (from_day * DAY, to_day * DAY, version))
        rows = cursor.fetchall()
    if not rows:
        return {}
    X = np.asarray([row[0] for row in rows], dtype=np.float64)
    scores = np.asarray([row[1] for row in rows], dtype=np.float64)
    return feature_psi(reference, X, scores)


def psi_status(value: float) -> str:
    if value >= PSI_ALERT:
        return "ALERT"
    return "watch" if value >= PSI_WATCH else "stable"


def build_report(connection, paths: Paths, since_day: int, until_day: int) -> dict:
    version = read_pointer(paths)
    reference = load_meta(paths, version)["reference"]
    cohorts = cohort_recall(connection, since_day, until_day)
    psi_values = live_psi(connection, reference, since_day, until_day, version)
    return {
        "live_model": version,
        "recall": recall_alert(cohorts),
        "psi": {name: {"psi": value, "status": psi_status(value)} for name, value in psi_values.items()},
        "cohorts": cohorts.to_dict("records"),
    }


def format_report(report: dict) -> str:
    recall = report["recall"]
    rate = f"{recall['recall']:.1%}" if recall["recall"] is not None else "n/a"
    if recall["alert"]:
        verdict = "ALERT: too much confirmed fraud was let through"
    else:
        verdict = "ok" if recall["enough_data"] else "not enough confirmed fraud yet to judge"
    lines = [
        f"Live model: {report['live_model']}",
        "",
        "Missed fraud, as confirmed by chargebacks",
        f"  confirmed frauds {recall['confirmed_fraud']}, caught {recall['caught']}, recall {rate}",
        f"  {verdict}",
        "",
        "Input drift (PSI, needs no labels)",
    ]
    lines += [
        f"  {name:26s} {item['psi']:.3f}  {item['status']}" for name, item in report["psi"].items()
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    report = commands.add_parser("report", help="print drift and missed-fraud status")
    report.add_argument("--since-day", type=int, required=True)
    report.add_argument("--until-day", type=int, default=10_000, help="exclusive end day")
    report.add_argument("--text", action="store_true", help="a short summary instead of JSON")
    args = parser.parse_args()

    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        result = build_report(connection, Paths(), args.since_day, args.until_day)
    print(format_report(result) if args.text else json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
