import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

/**
 * k6 Load & Concurrency Benchmark for What The Docs RAG Backend
 *
 * Requirements:
 *   1. Docker containers running: `docker compose up -d`
 *   2. Server running: `uv run uvicorn app.main:app --port 8000`
 *   3. Run test: `k6 run benchmarks/k6_query_benchmark.js`
 */

// Custom metrics to distinguish cache hits vs misses
const cacheHitDuration = new Trend('cache_hit_duration', true);
const cacheMissDuration = new Trend('cache_miss_duration', true);
const cacheHitRate = new Rate('cache_hit_rate');

export const options = {
  stages: [
    { duration: '10s', target: 20 }, // Warm-up & ramp to 20 Virtual Users (VUs)
    { duration: '30s', target: 50 }, // Sustained peak load at 50 VUs
    { duration: '5s', target: 0 },   // Ramp-down
  ],
  thresholds: {
    // Assert that 95% of cache hits return in under 10ms, and 99% in under 20ms
    'cache_hit_duration': ['p(95)<10', 'p(99)<20'],
    // System error rate must be below 1%
    'http_req_failed': ['rate<0.01'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

const QUERIES = [
  'What is PostgreSQL pgvector HNSW index?',
  'How does Reciprocal Rank Fusion combine vector and lexical search?',
  'What are the advantages of token sliding window chunking?',
  'Explain the Redis query cache invalidation strategy.',
];

export default function () {
  // Select a query from the pool
  const queryText = QUERIES[Math.floor(Math.random() * QUERIES.length)];

  const payload = JSON.stringify({
    query: queryText,
    top_k: 5,
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
    tags: { name: 'PostQuery' },
  };

  const res = http.post(`${BASE_URL}/api/query`, payload, params);

  const ok = check(res, {
    'status is 200': (r) => r.status === 200,
    'has answer': (r) => {
      try {
        const body = JSON.parse(r.body);
        return body.answer && body.answer.length > 0;
      } catch (e) {
        return false;
      }
    },
  });

  if (ok && res.status === 200) {
    const body = JSON.parse(res.body);
    const isCached = body.cached === true;

    cacheHitRate.add(isCached);

    if (isCached) {
      cacheHitDuration.add(res.timings.duration);
    } else {
      cacheMissDuration.add(res.timings.duration);
    }
  }

  // Small pacing between requests per VU
  sleep(0.05);
}
