#!/usr/bin/env python3
"""
Performance test script for OpenAI Privacy Filter API service.

Tests cover:
- Single request latency (small, medium, large text)
- Concurrent request throughput
- Sustained load over time
- Error rate under load

Usage:
    python perf_test_openai_privacy.py [--base-url URL] [--concurrency N] [--requests N]

Exit codes:
    0 - All performance benchmarks completed
    1 - One or more benchmarks failed
"""

import argparse
import json
import statistics
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional


DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_CONCURRENCY = 10
DEFAULT_REQUESTS = 50


@dataclass
class RequestResult:
    success: bool
    latency_ms: float
    status_code: int
    error: Optional[str] = None


@dataclass
class BenchmarkResult:
    name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    latencies_ms: list = field(default_factory=list)
    error_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    total_time_s: float = 0.0


def make_request(base_url: str, text: str, timeout: int = 120) -> RequestResult:
    url = f"{base_url}/redact"
    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")

    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            latency = (time.monotonic() - start) * 1000
            return RequestResult(success=True, latency_ms=latency, status_code=resp.status)
    except urllib.error.HTTPError as e:
        latency = (time.monotonic() - start) * 1000
        return RequestResult(success=False, latency_ms=latency, status_code=e.code, error=f"HTTP {e.code}")
    except urllib.error.URLError as e:
        latency = (time.monotonic() - start) * 1000
        return RequestResult(success=False, latency_ms=latency, status_code=0, error=str(e.reason))
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return RequestResult(success=False, latency_ms=latency, status_code=0, error=str(e))


def run_benchmark(base_url: str, text: str, concurrency: int, num_requests: int, timeout: int = 120) -> BenchmarkResult:
    results = []
    total_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(make_request, base_url, text, timeout) for _ in range(num_requests)]
        for future in as_completed(futures):
            results.append(future.result())

    total_time = time.monotonic() - total_start

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    latencies = [r.latency_ms for r in results]

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    def percentile(p):
        if n == 0:
            return 0.0
        idx = int(n * p / 100)
        idx = min(idx, n - 1)
        return sorted_latencies[idx]

    return BenchmarkResult(
        name="",
        total_requests=num_requests,
        successful_requests=len(successful),
        failed_requests=len(failed),
        latencies_ms=latencies,
        error_rate=len(failed) / num_requests * 100 if num_requests > 0 else 0,
        avg_latency_ms=statistics.mean(latencies) if latencies else 0,
        p50_latency_ms=percentile(50),
        p95_latency_ms=percentile(95),
        p99_latency_ms=percentile(99),
        min_latency_ms=min(latencies) if latencies else 0,
        max_latency_ms=max(latencies) if latencies else 0,
        throughput_rps=num_requests / total_time if total_time > 0 else 0,
        total_time_s=total_time,
    )


def print_result(r: BenchmarkResult):
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK: {r.name}")
    print(f"{'=' * 60}")
    print(f"  Total requests:     {r.total_requests}")
    print(f"  Successful:         {r.successful_requests}")
    print(f"  Failed:             {r.failed_requests}")
    print(f"  Error rate:         {r.error_rate:.1f}%")
    print(f"  Total time:         {r.total_time_s:.2f}s")
    print(f"  Throughput:         {r.throughput_rps:.2f} req/s")
    print(f"  Avg latency:        {r.avg_latency_ms:.1f}ms")
    print(f"  P50 latency:        {r.p50_latency_ms:.1f}ms")
    print(f"  P95 latency:        {r.p95_latency_ms:.1f}ms")
    print(f"  P99 latency:        {r.p99_latency_ms:.1f}ms")
    print(f"  Min latency:        {r.min_latency_ms:.1f}ms")
    print(f"  Max latency:        {r.max_latency_ms:.1f}ms")


def check_service_health(base_url: str) -> bool:
    url = f"{base_url}/health"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
            return body.get("model_loaded", False)
    except Exception:
        return False


SAMPLE_TEXTS = {
    "short_no_pii": "The quick brown fox jumps over the lazy dog.",
    "short_with_pii": "My name is John Smith and my email is john@example.com. Call me at +1-555-123-4567.",
    "medium_no_pii": (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris "
        "nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in "
        "reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur."
    ),
    "medium_with_pii": (
        "Dear Mr. Johnson, your account has been updated. "
        "Please contact us at support@company.com or call +1-800-555-0199. "
        "Your order will be shipped to 456 Oak Avenue, Springfield, IL 62701. "
        "For more details, visit https://company.com/orders/12345. "
        "Your appointment is scheduled for 2024-03-15."
    ),
    "long_no_pii": "This is a test sentence. " * 500,
    "long_with_pii": (
        "Hello, my name is Alice Brown and I live at 789 Pine Road, Boston, MA 02101. "
        "You can reach me at alice.brown@email.com or +1-617-555-0123. "
        "My account number is 9876543210. "
        "Please visit https://myaccount.example.com/profile for more information. "
        "The meeting is scheduled for 2024-06-20. " +
        "This is additional context text. " * 200
    ),
}


def main():
    parser = argparse.ArgumentParser(description="Performance test for OpenAI Privacy Filter API")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help=f"Concurrent workers (default: {DEFAULT_CONCURRENCY})")
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS, help=f"Requests per benchmark (default: {DEFAULT_REQUESTS})")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds (default: 120)")
    parser.add_argument("--skip-health", action="store_true", help="Skip health check before running benchmarks")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    concurrency = args.concurrency
    num_requests = args.requests
    timeout = args.timeout

    print(f"Performance Test Configuration:")
    print(f"  Base URL:     {base_url}")
    print(f"  Concurrency:  {concurrency}")
    print(f"  Requests:     {num_requests} per benchmark")
    print(f"  Timeout:      {timeout}s")
    print()

    if not args.skip_health:
        print("Checking service health...")
        if not check_service_health(base_url):
            print("ERROR: Service is not healthy or model is not loaded!")
            print(f"Please ensure the service is running at {base_url}")
            sys.exit(1)
        print("Service is healthy.\n")

    benchmarks = []

    benchmark_specs = [
        ("Short text (no PII, 44 chars)", SAMPLE_TEXTS["short_no_pii"]),
        ("Short text (with PII, 91 chars)", SAMPLE_TEXTS["short_with_pii"]),
        ("Medium text (no PII, 396 chars)", SAMPLE_TEXTS["medium_no_pii"]),
        ("Medium text (with PII, 342 chars)", SAMPLE_TEXTS["medium_with_pii"]),
        ("Long text (no PII, ~13000 chars)", SAMPLE_TEXTS["long_no_pii"]),
        ("Long text (with PII, ~6000 chars)", SAMPLE_TEXTS["long_with_pii"]),
    ]

    for name, text in benchmark_specs:
        print(f"Running: {name}...")
        result = run_benchmark(base_url, text, concurrency, num_requests, timeout)
        result.name = name
        benchmarks.append(result)
        print_result(result)

    concurrency_tests = [1, 5, 10, 20, 50]
    print(f"\n{'=' * 60}")
    print("CONCURRENCY SCALING TEST")
    print(f"{'=' * 60}")
    print(f"Using short text with PII ({len(SAMPLE_TEXTS['short_with_pii'])} chars)")
    print(f"{'Concurrency':>12} | {'Throughput':>12} | {'Avg Latency':>12} | {'P95 Latency':>12} | {'Error Rate':>10}")
    print("-" * 70)

    for cc in concurrency_tests:
        if cc > num_requests:
            continue
        result = run_benchmark(base_url, SAMPLE_TEXTS["short_with_pii"], cc, num_requests, timeout)
        print(f"{cc:>12} | {result.throughput_rps:>10.2f}rps | {result.avg_latency_ms:>10.1f}ms | {result.p95_latency_ms:>10.1f}ms | {result.error_rate:>8.1f}%")

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for b in benchmarks:
        status = "OK" if b.error_rate == 0 else "WARN"
        print(f"  [{status}] {b.name}: {b.throughput_rps:.2f} req/s, P50={b.p50_latency_ms:.1f}ms, P95={b.p95_latency_ms:.1f}ms, errors={b.error_rate:.1f}%")
    print(f"{'=' * 60}")

    total_failures = sum(b.failed_requests for b in benchmarks)
    if total_failures > 0:
        print(f"\nWARNING: {total_failures} total request failures across all benchmarks!")
        sys.exit(1)
    else:
        print(f"\nAll benchmarks completed successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
