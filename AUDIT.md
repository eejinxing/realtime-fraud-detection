# Model and pipeline audit

These results are from synthetic transactions. The default seed was used while developing the simulator, features, and policy. Seeds 11–14 and 21–24 were used for separate checks.

## Leakage checks

- Changing later transactions left earlier features unchanged. Changing chargebacks that had not arrived left the merchant snapshots unchanged.
- The strongest individual feature on the test period was `is_new_merchant`, with ROC-AUC of 0.912. The simulator assigns random merchants to some fraud patterns, which makes this feature unusually informative.
- Three runs with shuffled training labels scored 0.0034, 0.0182, and 0.0121 average precision. The test fraud rate was 0.0036, and the trained model scored 0.756. The higher shuffled scores were observed in a small positive class.
- The guards caught two deliberate changes to the merchant history calculation. One ignored arrival dates and the other included chargebacks from the transaction's day. Fraud labels and pattern names are excluded from model features and API input fields.

The same validation period was used for early stopping, tuning and threshold selection. Test results informed project development, including the choice to add merchant history and the original 30-fraud promotion minimum. Those decisions limit how far the default-seed results generalize.

## Fit and decision quality

| Window | Rows | Frauds | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Training | 212,231 | 997 | 0.998 | 1.000 |
| Validation | 45,521 | 204 | 0.956 | 1.000 |
| Test | 60,479 | 219 | 0.756 | 0.953 |
| Test, excluding `low_and_slow` | 60,424 | 164 | 0.997 | |

The test gap almost disappears when the new pattern is removed. On this dataset, a depth-2 XGBoost model scored 0.934 on validation and 0.761 on test. The tuned model scored 0.956 and 0.756, respectively. Logistic regression scored 0.848 and 0.645.

| Decision | Total | Fraud | Legitimate |
| --- | ---: | ---: | ---: |
| Decline | 175 | 161 | 14 |
| Review | 320 | 3 | 317 |
| Approve | 59,984 | 55 | 59,929 |

The review tier saved $459 over a decline-only policy on validation and about $27 on test. All 55 missed frauds were `low_and_slow`. Their total amount was $2,518 out of $27,332 in simulated fraud. Raising recall to 90% using the scores reduced the best available precision to about 2%.

The same setup produced 0.751 test PR-AUC and 27.8% precision on Windows, and 0.756 test PR-AUC and 33.1% precision in Docker. Both runs had 74.9% recall. The cause of the platform difference was not established.

## Results across seeds

| Seed | Chronological PR-AUC | Random PR-AUC | Recall | Precision | New pattern recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11 | 0.717 | 0.957 | 69.7% | 25.5% | 1.5% |
| 12 | 0.775 | 0.975 | 76.4% | 26.0% | 2.1% |
| 13 | 0.678 | 0.959 | 66.9% | 22.1% | 1.6% |
| 14 | 0.787 | 0.971 | 78.8% | 22.6% | 1.9% |

The original retraining gate evaluated the most recent seven days, when many chargebacks were still pending. Only 8–13 frauds in each holdout were confirmed, below the gate's 30-case minimum. The gate retained the incumbent four times. Two challengers were cheaper in the following days by 8–12%, and two were more expensive by 13–16%.

## Chargeback maturity and promotion

The revised gate estimates the 90th-percentile chargeback delay from older transactions and moves both holdout and validation windows back. Each window spans one week. The previous design used five validation days and seven holdout days.

| Seed | Confirmed fraud, old holdout | Confirmed fraud, revised holdout | Revised gate | Later v1 cost | Later old challenger cost | Later revised challenger cost |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 21 | 15 | 109 | Keep v1 | $1,500 | $1,456 | $1,721 |
| 22 | 14 | 103 | Keep v1 | $1,510 | $1,499 | $1,602 |
| 23 | 11 | 104 | Keep v1 | $1,254 | $1,351 | $1,378 |
| 24 | 16 | 111 | Promote | $1,822 | $1,557 | $1,817 |

The revised gate reached a comparison in all four runs. Its choices agreed with the following eight days of simulated cost four times. Keeping v1 would have agreed three times. The one promotion saved $5 over that future period, so this experiment provides limited evidence for promoting a challenger. It used four seeds with one comparison per seed.

The revised challengers trained on older transactions and cost more than the old challengers on all four seeds. Refitting with more recent data after the gate would put a different model into service and needs a separate evaluation.

## Earlier performance measurements

These results came from a single laptop running WSL before the worker's delivery acknowledgement fix. The benchmark warmed Kafka with 20,000 events before collecting timings.

| Path | Observed result |
| --- | --- |
| Kafka replay, about 56,000 transactions | All expected decisions reached PostgreSQL, with a maximum offline score gap of 2.38e-07 |
| One worker killed during replay | All expected decisions arrived. The maximum affected-message latency was 8.1 seconds |
| Kafka at 500 events/s | P50 about 50 ms and P99 500–790 ms across two runs |
| Kafka at 1,000 events/s | Backlog grew and P50 reached about 2 seconds across two runs |
| API at 100 requests/s | P99 19 ms with Kafka publishing in one run |
| API at 200 requests/s | P99 varied from 28 to 500 ms |

The slow API requests spent time on different Redis round trips. Switching to uvloop reduced median latency, while the cause of the tail latency remained unresolved.

## Reproduce the checks

After installing `.[train,ops,dev]`, train the baseline with `python -m frauddetect.train` or the Compose command in the [README](README.md). Run these commands from the repository root.

```bash
python -m pytest tests/test_leakage.py
python -m frauddetect.audit leakage
python -m frauddetect.audit fit
python -m frauddetect.audit errors
python -m frauddetect.audit seeds --seeds 21 22 23 24
```

Use new seeds for any follow-up comparison that changes the training windows or promotion criteria.
