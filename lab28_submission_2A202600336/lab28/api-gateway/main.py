# api-gateway/main.py
from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator
import httpx
import os
import time

app = FastAPI(title="AI Platform API Gateway")
Instrumentator().instrument(app).expose(app)

VLLM_URL = os.environ.get("VLLM_NGROK_URL", "").strip()
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"


def search_context(embedding: list) -> list:
    try:
        response = httpx.post(
            f"{QDRANT_URL}/collections/documents/points/search",
            json={"vector": embedding, "limit": 3, "with_payload": True},
            timeout=5,
        )
        response.raise_for_status()
        return response.json().get("result", [])
    except httpx.HTTPError:
        return []


def local_demo_response(prompt: str) -> dict:
    query = prompt.split("Query: ", 1)[-1]
    return {
        "answer": f"Local demo answer for: {query}",
        "model": "local-demo",
    }


def call_llm(prompt: str) -> dict:
    if not VLLM_URL:
        return local_demo_response(prompt)

    try:
        response = httpx.post(
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        return {
            "answer": result["choices"][0]["message"]["content"],
            "model": result.get("model", MODEL_NAME),
        }
    except (httpx.HTTPError, KeyError, IndexError):
        return local_demo_response(prompt)


@app.post("/api/v1/chat")
async def chat(request: Request):
    body = await request.json()
    query = body.get("query")
    if not query:
        raise HTTPException(status_code=422, detail="query is required")

    embedding = body.get("embedding", [0.0] * 384)
    if not isinstance(embedding, list) or len(embedding) != 384:
        raise HTTPException(status_code=422, detail="embedding must contain 384 numbers")

    start = time.time()
    context = search_context(embedding)
    prompt = f"Context: {context}\n\nQuery: {query}"
    llm_result = call_llm(prompt)
    latency = (time.time() - start) * 1000

    return {
        "answer": llm_result["answer"],
        "latency_ms": round(latency, 2),
        "model": llm_result["model"],
        "context_count": len(context),
    }


@app.get("/health")
def health():
    return {"status": "ok"}
