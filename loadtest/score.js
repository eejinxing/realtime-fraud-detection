// Load test for the synchronous API, at a fixed arrival rate so that a slow
// server cannot hide its latency by receiving fewer requests.
import http from 'k6/http';
import { check } from 'k6';

const URL = __ENV.API_URL || 'http://localhost:8000/v1/score';
const RATE = Number(__ENV.RATE || 200);
const N_CARDS = 3000;
const AFTER_SIMULATION = 121 * 86400;
const RUN_ID = Date.now();

export const options = {
  scenarios: {
    steady: {
      executor: 'constant-arrival-rate',
      rate: RATE,
      timeUnit: '1s',
      duration: __ENV.DURATION || '30s',
      preAllocatedVUs: 50,
      maxVUs: 200,
    },
  },
  summaryTrendStats: ['avg', 'p(50)', 'p(95)', 'p(99)', 'max'],
  thresholds: {
    http_req_duration: ['p(99)<50'],
    http_req_failed: ['rate<0.001'],
  },
};

export default function () {
  const card = Math.floor(Math.random() * N_CARDS);
  const body = JSON.stringify({
    transaction_id: `load-${RUN_ID}-${__VU}-${__ITER}`,
    ts: AFTER_SIMULATION + (Date.now() - RUN_ID) / 1000,
    card_id: `c${card}`,
    merchant_id: `m${Math.floor(Math.random() * 2000)}`,
    merchant_category: Math.floor(Math.random() * 10),
    amount: Math.round(Math.random() * 20000) / 100 + 1,
    lat: 40 + Math.random(),
    lon: -75 + Math.random(),
    device_id: `d${card}`,
    channel: Math.random() < 0.35 ? 'online' : 'pos',
  });
  const response = http.post(URL, body, { headers: { 'Content-Type': 'application/json' } });
  check(response, { 'status is 200': (r) => r.status === 200 });
}
