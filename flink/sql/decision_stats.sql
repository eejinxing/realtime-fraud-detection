-- Counts decisions in 10-second windows, straight from the decisions topic.
-- Grafana reads the result from PostgreSQL.
--
-- The windows use the time a decision arrives, not the transaction time inside
-- it, because the simulated transaction times run far ahead of the clock.

SET 'pipeline.name' = 'decision-stats';
SET 'sql-client.execution.result-mode' = 'tableau';

CREATE TABLE decisions (
  transaction_id STRING,
  decision STRING,
  score DOUBLE,
  model_version STRING,
  card_id STRING,
  amount DOUBLE,
  proc_time AS PROCTIME()
) WITH (
  'connector' = 'kafka',
  'topic' = 'decisions',
  'properties.bootstrap.servers' = 'kafka:29092',
  'properties.group.id' = 'flink-decision-stats',
  'scan.startup.mode' = 'latest-offset',
  'format' = 'json',
  'json.ignore-parse-errors' = 'true'
);

CREATE TABLE decision_rate (
  window_start TIMESTAMP(3),
  window_end TIMESTAMP(3),
  decision STRING,
  model_version STRING,
  n BIGINT,
  amount_sum DOUBLE,
  avg_score DOUBLE,
  PRIMARY KEY (window_start, decision, model_version) NOT ENFORCED
) WITH (
  'connector' = 'jdbc',
  'url' = 'jdbc:postgresql://postgres:5432/fraud',
  'table-name' = 'decision_rate',
  'username' = 'fraud',
  'password' = 'fraud'
);

INSERT INTO decision_rate
SELECT
  window_start,
  window_end,
  decision,
  model_version,
  COUNT(*),
  SUM(amount),
  AVG(score)
FROM TABLE(TUMBLE(TABLE decisions, DESCRIPTOR(proc_time), INTERVAL '10' SECOND))
GROUP BY window_start, window_end, decision, model_version;
