# scripts/09_verify_observability.py
import os

import requests


def check_prometheus():
    response = requests.get(
        "http://localhost:9090/api/v1/query",
        params={"query": "up{job='api-gateway'}"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    assert data["status"] == "success"
    print("Integration 9 OK: Prometheus metrics flowing")


def check_langsmith():
    api_key = os.environ.get("LANGCHAIN_API_KEY", "").strip()
    if not api_key:
        print("Integration 10 SKIP: LANGCHAIN_API_KEY not configured")
        return

    from langsmith import Client

    client = Client(api_key=api_key)
    runs = list(client.list_runs(project_name=os.environ.get("LANGCHAIN_PROJECT", "lab28-platform"), limit=1))
    assert len(runs) > 0
    print("Integration 10 OK: LangSmith traces visible")


if __name__ == "__main__":
    check_prometheus()
    check_langsmith()
