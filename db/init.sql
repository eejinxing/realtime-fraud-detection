CREATE TABLE decisions (
    transaction_id TEXT PRIMARY KEY,
    card_id        TEXT NOT NULL,
    amount         DOUBLE PRECISION NOT NULL,
    event_ts       DOUBLE PRECISION NOT NULL,
    decision       TEXT NOT NULL CHECK (decision IN ('approve', 'review', 'decline')),
    score          DOUBLE PRECISION NOT NULL,
    reasons        JSONB NOT NULL,
    pipeline_ms    DOUBLE PRECISION,
    model_version  TEXT NOT NULL DEFAULT 'unknown',
    features       JSONB,
    stored_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX decisions_card_idx ON decisions (card_id, event_ts);
CREATE INDEX decisions_event_ts_idx ON decisions (event_ts);

-- Chargebacks. They arrive days after the purchase, and only for fraud.
-- A transaction with no row here is treated as legitimate once it is old enough.
CREATE TABLE labels (
    transaction_id TEXT PRIMARY KEY,
    label_ts       DOUBLE PRECISION NOT NULL,
    received_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Written by the Flink job: decisions counted in 10-second windows.
CREATE TABLE decision_rate (
    window_start  TIMESTAMP NOT NULL,
    window_end    TIMESTAMP NOT NULL,
    decision      TEXT NOT NULL,
    model_version TEXT NOT NULL,
    n             BIGINT NOT NULL,
    amount_sum    DOUBLE PRECISION,
    avg_score     DOUBLE PRECISION,
    PRIMARY KEY (window_start, decision, model_version)
);

-- What a fraud analyst works through, highest risk first.
CREATE VIEW review_queue AS
SELECT transaction_id, card_id, amount, score, reasons, stored_at
FROM decisions
WHERE decision = 'review'
ORDER BY score DESC;

-- How much of the confirmed fraud each day's decisions let through.
-- Recall is measured among confirmed fraud only, so late chargebacks do not
-- bias it: a fraud that is confirmed later is still caught or missed the same way.
CREATE VIEW cohort_recall AS
SELECT
    floor(d.event_ts / 86400)::int AS day,
    d.model_version,
    count(*) AS decisions,
    count(l.transaction_id) AS confirmed_fraud,
    count(l.transaction_id) FILTER (WHERE d.decision <> 'approve') AS caught,
    round(
        100.0 * count(l.transaction_id) FILTER (WHERE d.decision <> 'approve')
        / NULLIF(count(l.transaction_id), 0), 1
    ) AS recall_pct
FROM decisions d
LEFT JOIN labels l USING (transaction_id)
GROUP BY 1, 2
ORDER BY 1, 2;
