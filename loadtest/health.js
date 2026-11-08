// Baseline: the same arrival rate against an endpoint that does no work.
// If this shows the same tail latency, the cause is the environment, not the scoring code.
import http from 'k6/http';

const URL = __ENV.API_URL || 'http://localhost:8000/health';
const RATE = Number(__ENV.RATE || 200);

export const options = {
  scenarios: {
    steady: {
      executor: 'constant-arrival-rate',
      rate: RATE,
      timeUnit: '1s',
      duration: __ENV.DURATION || '20s',
      preAllocatedVUs: 50,
      maxVUs: 200,
    },
  },
  summaryTrendStats: ['avg', 'p(50)', 'p(95)', 'p(99)', 'max'],
};

export default function () {
  http.get(URL);
}
