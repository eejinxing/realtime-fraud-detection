# Real-time card fraud detection

This project scores card transactions and returns one of three decisions: approve, review, or decline. I used a simulator to test what happens when a new type of fraud appears and the first chargebacks take days to arrive.

Transactions can come in through FastAPI or Kafka. Both paths use the same card features and ONNX model. Redis holds card history across processes, and a Kafka consumer writes decisions and chargebacks to PostgreSQL. Everything runs locally with Docker Compose.

## Run it locally

Use Docker with Compose v2. Run these commands from the repository root. The shell scripts use Bash, so on Windows run them in WSL.

```bash
docker compose --profile ops run --rm --build ops
docker compose up --build --wait api worker sink
curl --fail http://localhost:8000/health
```

The first command generates the dataset, trains XGBoost and exports the baseline as `v1`. It resets the local model registry, so use the retraining command below if you want to keep previous versions.

Open [the API docs](http://localhost:8000/docs) to try `POST /v1/score`, or publish the first 3,000 simulated transactions through Kafka.

```bash
docker compose run --rm -T producer python -m frauddetect.stream produce --n 3000 --rate 200
docker compose exec -T postgres psql -U fraud -d fraud -c 'SELECT decision, count(*) FROM decisions GROUP BY decision;'
```

Use a fresh stack for `bash scripts/smoke.sh` after training `v1`. The script removes this Compose project's volumes, replays 3,000 transactions, and checks both the PostgreSQL row count and the scores against the offline baseline.

## Chargebacks and retraining

`scripts/stage3_demo.sh` trains a baseline, replays traffic, releases chargebacks, and demonstrates promotion and rollback. It resets this Compose project's volumes and model registry. On an already-running stack with traffic replayed through day 111, use the commands below to run the chargeback and retraining steps on their own.

```bash
docker compose run --rm -T producer python -m frauddetect.stream labels --as-of-day 112
docker compose --profile ops run --rm -T ops python -m frauddetect.monitor report --since-day 100 --until-day 112 --text
docker compose --profile ops run --rm -T ops python -m frauddetect.mlops retrain --as-of-day 112 --text
```

The sink consumes chargebacks asynchronously. Wait for the labels to reach PostgreSQL before reading the report. Day numbers start at the beginning of the simulation, and the report's end day is exclusive.

The promotion gate compares the challenger and the live model on a week with mostly settled chargebacks. It requires at least 30 confirmed frauds, a cost reduction of at least 5%, a lower cost in at least 90% of resamples, and a PR-AUC drop no greater than 0.02. Running scorers pick up promotions and rollbacks from `artifacts/current.json`.

### Optional services

| Service | Start command | Local address |
| --- | --- | --- |
| Grafana and Prometheus | `docker compose --profile monitoring up -d grafana prometheus` | [Grafana](http://localhost:3001), [Prometheus](http://localhost:9090) |
| Dagster | `docker compose --profile orchestration up -d dagster` | [Dagster](http://localhost:3000) |
| Flink | `docker compose --profile flink up -d flink-jobmanager flink-taskmanager` | [Flink](http://localhost:8081) |

Once the Flink cluster has an available task slot, submit the SQL job with `docker compose --profile flink run --rm flink-submit`. It writes decision counts in 10-second processing-time windows to PostgreSQL. Dagster runs the retraining workflow and leaves the automatic retraining sensor stopped by default.

## What the simulation showed

| Baseline v1, default-seed Docker run | Result |
| --- | ---: |
| PR-AUC, chronological split | 0.756 |
| PR-AUC, random split | 0.966 |
| Recall, review and decline combined | 74.9% |
| Precision, review and decline combined | 33.1% |
| Precision of the review queue | 0.9% |
| Recall for the new `low_and_slow` pattern | 0% |

The random split mixed transactions from before and after the new fraud pattern. The chronological split gives a better picture of what happened at deployment. In the day-112 replay, 50 of 67 confirmed frauds were flagged. The eight monitored distributions stayed below the PSI warning threshold, and the promotion gate kept `v1` because the challenger cost more on the holdout.

The simulator makes account takeover and card testing easier to catch than they would be with real data. The new pattern is still a blind spot. Moving the holdout back far enough for labels to settle also moves training further into the past. The [audit](AUDIT.md) has the counts, the seed comparisons, and the limits of both findings.

## What is still open

- API tail latency varied at 200 requests per second in earlier WSL runs. The current delivery code needs a fresh benchmark before using the old streaming figures as a reference.
- Card features assume transactions arrive in timestamp order for each card. Redis protects concurrent writes, and the cached features for a transaction ID expire after one day.
- Kafka workers wait for output acknowledgements before recording offsets. Direct API requests queue their Kafka event without waiting for the broker. PostgreSQL keeps the first decision for a transaction ID.
- The stack is a local simulation. Dagster runs in development mode and the labels come from generated chargebacks.
- Kubernetes packaging and an analyst-facing workflow are still open.

## Tests

Use Python 3.12 in a virtual environment and run the following from the repository root.

```bash
python -m pip install -e ".[train,ops,dev]"
python -m ruff check .
python -m pytest
```

Stop the stack with `docker compose --profile '*' down`. This keeps the named volumes. Add `-v` only when you want to reset them.
