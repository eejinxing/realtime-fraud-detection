"""Kafka side of the system.

    python -m frauddetect.stream bootstrap   load card history into Redis, once
    python -m frauddetect.stream produce     publish transactions to Kafka
    python -m frauddetect.stream labels      publish chargebacks that have arrived
    python -m frauddetect.stream worker      transactions -> score -> decisions
    python -m frauddetect.stream sink        decisions and labels -> PostgreSQL

Transactions are keyed by card, so one card's events stay in order on a single
partition. Offsets are committed only after the work is done, which gives
at-least-once delivery. The Redis store and the PostgreSQL primary key make a
redelivered event harmless.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import psycopg
import redis
from confluent_kafka import Consumer, KafkaError, KafkaException, Producer
from confluent_kafka.admin import AdminClient, NewTopic
from psycopg.types.json import Jsonb
from pydantic import Field, FiniteFloat, ValidationError

from frauddetect import metrics
from frauddetect.config import DAY, Paths, SplitConfig
from frauddetect.features import TXN_FIELDS
from frauddetect.merchant_risk import load_snapshots
from frauddetect.scoring import Decision, Scorer, Transaction
from frauddetect.store import RedisFeatureStore, load_engine

TRANSACTIONS = "transactions"
DECISIONS = "decisions"
LABELS = "labels"
DEAD_LETTER = "transactions.dlq"
# Workers can only run in parallel up to this number.
N_PARTITIONS = 4
BATCH_SIZE = 500
POLL_SECONDS = 0.02
DELIVERY_WAIT_SECONDS = 10

INSERT_DECISION = """
    INSERT INTO decisions
        (transaction_id, card_id, amount, event_ts, decision, score, reasons, pipeline_ms,
         model_version, features)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (transaction_id) DO NOTHING
"""
INSERT_LABEL = """
    INSERT INTO labels (transaction_id, label_ts)
    VALUES (%s, %s)
    ON CONFLICT (transaction_id) DO NOTHING
"""

log = logging.getLogger("frauddetect.stream")


def decision_event(txn: Transaction, result: Decision, produced_at: float | None) -> dict:
    return {
        **result.model_dump(),
        "card_id": txn.card_id,
        "amount": txn.amount,
        "ts": txn.ts,
        "pipeline_ms": (time.time() - produced_at) * 1000 if produced_at else None,
        "features": result.features,
    }


class StreamTransaction(Transaction):
    produced_at: FiniteFloat | None = Field(default=None, exclude=True)


def handle_transaction(scorer: Scorer, raw: bytes | None) -> tuple[str, str | None, bytes]:
    """Return the topic, key and value to publish for one incoming message.

    A malformed message goes to a dead-letter topic. It must not stop the
    worker, and it must not be dropped silently either.
    """
    try:
        txn = StreamTransaction.model_validate_json(raw)
    except ValidationError as error:
        metrics.DEAD_LETTERS.inc()
        body = {
            "error": str(error)[:500],
            "raw": raw.decode("utf-8", "replace") if raw is not None else None,
        }
        return DEAD_LETTER, None, json.dumps(body).encode()

    started = time.perf_counter()
    result = scorer.score(txn)
    metrics.record_decision(result, "worker", time.perf_counter() - started)
    produced_at = txn.produced_at
    if produced_at:
        metrics.PIPELINE_SECONDS.observe(max(time.time() - produced_at, 0.0))
    event = decision_event(txn, result, produced_at)
    return DECISIONS, txn.card_id, json.dumps(event).encode()


def decision_row(raw: bytes) -> tuple:
    event = json.loads(raw)
    return (
        event["transaction_id"],
        event["card_id"],
        event["amount"],
        event["ts"],
        event["decision"],
        event["score"],
        Jsonb(event["reasons"]),
        event["pipeline_ms"],
        event["model_version"],
        Jsonb(event["features"]),
    )


def label_row(raw: bytes) -> tuple:
    event = json.loads(raw)
    return event["transaction_id"], event["label_ts"]


def _bootstrap_servers() -> str:
    return os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")


def make_producer() -> Producer:
    return Producer({"bootstrap.servers": _bootstrap_servers(), "linger.ms": 5})


def _make_consumer(group: str, topics: list[str]) -> Consumer:
    consumer = Consumer(
        {
            "bootstrap.servers": _bootstrap_servers(),
            "group.id": group,
            # Offsets are stored by hand once the work is really done, and a
            # background thread commits them, so the main loop never blocks.
            # Waiting on commit here stalled every worker for over a second.
            "enable.auto.commit": True,
            "auto.commit.interval.ms": 1000,
            "enable.auto.offset.store": False,
            # How long a dead worker's partitions wait before another takes over.
            "session.timeout.ms": 6000,
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe(topics)
    return consumer


def _good_messages(consumer: Consumer) -> list:
    messages = consumer.consume(BATCH_SIZE, timeout=POLL_SECONDS)
    for message in messages:
        if message.error():
            log.warning("Kafka error: %s", message.error())
    return [message for message in messages if not message.error()]


def publish(
    producer: Producer, topic: str, key: str | None, value: bytes, on_delivery=None
) -> None:
    while True:
        try:
            producer.produce(topic, key=key, value=value, on_delivery=on_delivery)
            producer.poll(0)
            return
        except BufferError:
            producer.poll(0.1)


class DeliveryAcks:
    def __init__(self) -> None:
        self.error = None

    def record(self, error, _message) -> None:
        if error is not None and self.error is None:
            self.error = error

    def wait(self, producer: Producer) -> None:
        pending = producer.flush(DELIVERY_WAIT_SECONDS)
        if self.error is not None:
            raise RuntimeError(f"Kafka delivery failed: {self.error}")
        if pending:
            raise RuntimeError(f"Timed out waiting for delivery of {pending} Kafka records")


def process_transactions(scorer: Scorer, consumer: Consumer, producer: Producer, messages) -> None:
    acks = DeliveryAcks()
    last_messages = {}
    for message in messages:
        result = handle_transaction(scorer, message.value())
        publish(producer, *result, on_delivery=acks.record)
        last_messages[message.topic(), message.partition()] = message

    # Decisions and dead letters go to different topics. A later record can
    # be acknowledged first, so commit progress only after the entire batch.
    acks.wait(producer)
    for message in last_messages.values():
        consumer.store_offsets(message=message)


def run_worker() -> None:
    artifact_dir = Path(os.environ.get("FRAUD_ARTIFACT_DIR", Paths().artifact_dir))
    scorer = Scorer(artifact_dir)
    metrics.serve(int(os.environ.get("METRICS_PORT", "9100")))
    consumer = _make_consumer("decision-worker", [TRANSACTIONS])
    producer = make_producer()
    log.info("Worker started")
    try:
        while True:
            messages = _good_messages(consumer)
            if messages:
                process_transactions(scorer, consumer, producer, messages)
    finally:
        consumer.close()


def write_batch(connection, messages: list) -> None:
    """Store one batch of decisions and labels. Repeats are ignored by the primary keys."""
    decisions = [decision_row(m.value()) for m in messages if m.topic() == DECISIONS]
    labels = [label_row(m.value()) for m in messages if m.topic() == LABELS]
    with connection.cursor() as cursor:
        if decisions:
            cursor.executemany(INSERT_DECISION, decisions)
        if labels:
            cursor.executemany(INSERT_LABEL, labels)
    connection.commit()


def run_sink() -> None:
    consumer = _make_consumer("decision-sink", [DECISIONS, LABELS])
    log.info("Sink started")
    try:
        with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
            while True:
                messages = _good_messages(consumer)
                if not messages:
                    continue
                write_batch(connection, messages)
                for message in messages:
                    consumer.store_offsets(message=message)
    finally:
        consumer.close()


def run_labels(as_of_day: float) -> None:
    """Publish the chargebacks that had arrived by `as_of_day`.

    In production these come from the card network. The simulator knows when
    each fraud's chargeback arrives, so it can release them on the same schedule.
    """
    df = pd.read_parquet(Paths().transactions)
    deploy_ts = SplitConfig().deploy_day * DAY
    arrived = df[
        (df["is_fraud"] == 1)
        & (df["ts"] >= deploy_ts)
        & (df["label_available_ts"] <= as_of_day * DAY)
    ]
    producer = make_producer()
    acks = DeliveryAcks()
    for transaction_id, label_ts in zip(arrived["transaction_id"], arrived["label_available_ts"]):
        value = json.dumps({"transaction_id": transaction_id, "label_ts": label_ts}).encode()
        publish(producer, LABELS, transaction_id, value, on_delivery=acks.record)
    acks.wait(producer)
    log.info("Published %s chargebacks that arrived by day %s", f"{len(arrived):,}", as_of_day)


def run_producer(n: int, rate: float, skip: int, from_day: float, to_day: float | None) -> None:
    df = pd.read_parquet(Paths().transactions)
    keep = df["ts"] >= from_day * DAY
    if to_day is not None:
        keep &= df["ts"] < to_day * DAY
    df = df[keep].iloc[skip : skip + n]
    producer = make_producer()
    acks = DeliveryAcks()
    started = time.perf_counter()
    for i, txn in enumerate(df[["transaction_id", *TXN_FIELDS]].to_dict("records")):
        if rate > 0:
            wait = started + i / rate - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        value = json.dumps({**txn, "produced_at": time.time()}).encode()
        publish(producer, TRANSACTIONS, txn["card_id"], value, on_delivery=acks.record)
    acks.wait(producer)
    elapsed = time.perf_counter() - started
    log.info("Published %s transactions in %.1fs", f"{len(df):,}", elapsed)


def ensure_topics() -> None:
    """Create topics up front, so consumers never start before their topic exists."""
    admin = AdminClient({"bootstrap.servers": _bootstrap_servers()})
    topics = [
        NewTopic(topic, num_partitions=N_PARTITIONS, replication_factor=1)
        for topic in (TRANSACTIONS, DECISIONS, LABELS, DEAD_LETTER)
    ]
    for future in admin.create_topics(topics).values():
        try:
            future.result()
        except KafkaException as error:
            if error.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                raise


def run_bootstrap() -> None:
    ensure_topics()
    paths = Paths()
    store = RedisFeatureStore(redis.Redis.from_url(os.environ["REDIS_URL"]))
    engine = load_engine(paths.feature_state)
    snapshots = load_snapshots(paths.merchant_risk)
    loaded = store.bootstrap(engine, snapshots, from_day=int(SplitConfig().deploy_day))
    log.info("Loaded %s cards" if loaded else "Already loaded (%s cards)", len(engine.cards))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["bootstrap", "produce", "labels", "worker", "sink"])
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--rate", type=float, default=0, help="events per second, 0 = no limit")
    parser.add_argument("--skip", type=int, default=0, help="rows to skip after the first day")
    parser.add_argument("--from-day", type=float, default=SplitConfig().deploy_day)
    parser.add_argument("--to-day", type=float, default=None)
    parser.add_argument("--as-of-day", type=float, help="labels: release chargebacks up to this day")
    args = parser.parse_args()

    if args.command == "bootstrap":
        run_bootstrap()
    elif args.command == "produce":
        run_producer(args.n, args.rate, args.skip, args.from_day, args.to_day)
    elif args.command == "labels":
        if args.as_of_day is None:
            parser.error("labels requires --as-of-day")
        run_labels(args.as_of_day)
    elif args.command == "worker":
        run_worker()
    else:
        run_sink()


if __name__ == "__main__":
    main()
