"""Checks Prometheus and Grafana against a running stack. Run it from the project root.

It asks Prometheus which targets are up, then sends every query on the Grafana
dashboard through Grafana itself, so a typo in any panel shows up here instead
of as an empty graph later.
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROMETHEUS = "http://localhost:9090"
GRAFANA = "http://localhost:3001"
DASHBOARD = Path("monitoring/grafana/dashboards/fraud.json")


def get(url: str):
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read())


def post(url: str, body: dict):
    request = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        return {"http_error": error.code, "body": error.read().decode()[:300]}


def prometheus_value(expression: str):
    result = get(f"{PROMETHEUS}/api/v1/query?query={urllib.parse.quote(expression)}")
    series = result["data"]["result"]
    return [(s["metric"], s["value"][1]) for s in series]


def main() -> int:
    failures = 0

    targets = get(f"{PROMETHEUS}/api/v1/targets")["data"]["activeTargets"]
    up = {}
    for target in targets:
        up.setdefault(target["labels"]["job"], []).append(target["health"])
    print("Prometheus targets:", {job: f"{health.count('up')}/{len(health)} up" for job, health in up.items()})
    if up.get("api", []).count("up") != 1 or up.get("worker", []).count("up") < 1:
        print("  PROBLEM: expected the api and at least one worker to be up")
        failures += 1

    for label, expression in {
        "decisions made": "sum(fraud_decisions_total)",
        "by source": "sum by (source) (fraud_decisions_total)",
        "live model": "max(fraud_model_version)",
        "worker p99 seconds": 'histogram_quantile(0.99, sum by (le) (rate(fraud_scoring_seconds_bucket{source="worker"}[1m])))',
    }.items():
        print(f"  {label}: {prometheus_value(expression)}")

    health = get(f"{GRAFANA}/api/health")
    print("Grafana:", health.get("database"), health.get("version"))
    dashboards = get(f"{GRAFANA}/api/search?query=Fraud")
    print("Dashboards provisioned:", [d["title"] for d in dashboards])
    if not dashboards:
        failures += 1

    for datasource in get(f"{GRAFANA}/api/datasources"):
        try:
            result = get(f"{GRAFANA}/api/datasources/uid/{datasource['uid']}/health")
        except urllib.error.HTTPError as error:
            result = {"status": f"HTTP {error.code}", "message": error.read().decode()[:70]}
        print(f"Data source {datasource['name']}: {result.get('status')} {result.get('message', '')[:70]}")
        if result.get("status") != "OK":
            failures += 1

    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    print("Every panel query, sent through Grafana:")
    for panel in dashboard["panels"]:
        queries = []
        for target in panel["targets"]:
            queries.append(
                {**target, "datasource": panel["datasource"], "intervalMs": 15000, "maxDataPoints": 500}
            )
        answer = post(f"{GRAFANA}/api/ds/query", {"queries": queries, "from": "now-15m", "to": "now"})
        if "http_error" in answer:
            print(f"  FAIL  {panel['title']}: {answer}")
            failures += 1
            continue
        problems, rows = [], 0
        for result in answer["results"].values():
            if result.get("error"):
                problems.append(result["error"][:150])
            for frame in result.get("frames", []):
                values = frame["data"]["values"]
                rows += len(values[0]) if values else 0
        status = "FAIL" if problems else "ok  "
        failures += bool(problems)
        print(f"  {status}  {panel['title']}: {rows} rows {problems or ''}")

    print("RESULT:", "all checks passed" if not failures else f"{failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    import urllib.parse

    sys.exit(main())
