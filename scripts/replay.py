"""Send post-deployment transactions to the running service and measure it.

Start the service on a fresh process first, because replaying changes its state:
    uvicorn frauddetect.service:app --port 8000
    python scripts/replay.py --n 5000
"""

import argparse
import time

import httpx
import numpy as np
import pandas as pd

from frauddetect.config import DAY, Paths, SplitConfig
from frauddetect.features import TXN_FIELDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/score")
    args = parser.parse_args()

    paths = Paths()
    df = pd.read_parquet(paths.transactions)
    df = df[df["ts"] >= SplitConfig().deploy_day * DAY].head(args.n)
    expected = pd.read_parquet(paths.test_scores).set_index("transaction_id")["score"]

    latencies, gaps, decisions = [], [], []
    columns = ["transaction_id", *TXN_FIELDS]
    with httpx.Client(timeout=5) as client:
        for txn in df[columns].to_dict("records"):
            started = time.perf_counter()
            response = client.post(args.url, json=txn)
            latencies.append((time.perf_counter() - started) * 1000)
            response.raise_for_status()
            body = response.json()
            decisions.append(body["decision"])
            gaps.append(abs(body["score"] - expected[txn["transaction_id"]]))

    latencies = np.array(latencies)
    print(f"requests: {len(latencies):,}")
    print(f"latency p50: {np.percentile(latencies, 50):.2f} ms")
    print(f"latency p99: {np.percentile(latencies, 99):.2f} ms")
    print(f"throughput (one client): {1000 / latencies.mean():.0f} req/s")
    print(f"max score gap vs offline pipeline: {max(gaps):.2e}")
    print(pd.Series(decisions).value_counts().to_string())


if __name__ == "__main__":
    main()
