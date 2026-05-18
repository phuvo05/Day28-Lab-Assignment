# scripts/production_readiness_check.py
import subprocess

import redis
import requests

results = {}


def check(name, fn):
    try:
        fn()
        results[name] = "PASS"
        print(f"  [PASS] {name}")
    except Exception as exc:
        results[name] = f"FAIL: {exc}"
        print(f"  [FAIL] {name}: {exc}")


def assert_status(url, expected=200, **kwargs):
    response = requests.get(url, timeout=10, **kwargs)
    response.raise_for_status()
    if response.status_code != expected:
        raise AssertionError(f"expected {expected}, got {response.status_code}")


def check_unauthorized_rejected():
    response = requests.get("http://localhost:8000/admin", timeout=10)
    if response.status_code not in [401, 403, 404]:
        raise AssertionError(f"expected 401/403/404, got {response.status_code}")


def check_qdrant_collection():
    response = requests.get("http://localhost:6333/collections/documents", timeout=10)
    response.raise_for_status()


def check_kafka_topics():
    container_result = subprocess.run(
        [
            "docker", "compose", "ps", "-q", "kafka",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if container_result.returncode != 0:
        raise AssertionError(container_result.stderr.strip())

    container_id = container_result.stdout.strip()
    if not container_id:
        raise AssertionError("kafka container not found")

    result = subprocess.run(
        [
            "docker", "exec", container_id, "kafka-topics", "--list",
            "--bootstrap-server", "localhost:9092",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr.strip())
    if "data.raw" not in result.stdout:
        raise AssertionError("data.raw topic not found")


print("\n=== RELIABILITY ===")
check("Health check endpoint", lambda: assert_status("http://localhost:8000/health"))
check("API Gateway responds", lambda: assert_status("http://localhost:8000/docs"))

print("\n=== OBSERVABILITY ===")
check("Prometheus up", lambda: assert_status("http://localhost:9090/-/healthy"))
check("Grafana up", lambda: assert_status("http://localhost:3000/api/health", auth=("admin", "admin")))
check("Metrics endpoint exposed", lambda: assert_status("http://localhost:8000/metrics"))

print("\n=== SECURITY ===")
check("Unauthorized request rejected", check_unauthorized_rejected)

print("\n=== VECTOR STORE ===")
check("Qdrant healthy", lambda: assert_status("http://localhost:6333/healthz"))
check("Collection exists", check_qdrant_collection)

print("\n=== FEATURE STORE ===")
check("Redis reachable", lambda: redis.Redis(host="localhost", port=6379).ping())

print("\n=== KAFKA ===")
check("Kafka topics exist", check_kafka_topics)

passed = sum(1 for value in results.values() if value == "PASS")
total = len(results)
score = (passed / total) * 100
print(f"\n{'=' * 40}")
print(f"Production Readiness Score: {passed}/{total} = {score:.0f}%")
print(f"Target: >80% - Status: {'READY' if score >= 80 else 'NOT READY'}")
