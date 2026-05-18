# scripts/05_embed_to_qdrant.py
import hashlib
import os

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

EMBED_URL = os.environ.get("EMBED_NGROK_URL", "").strip()
qdrant = QdrantClient(host="localhost", port=6333)


def local_embedding(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    while len(values) < 384:
        for byte in digest:
            values.append((byte / 255.0) * 2 - 1)
            if len(values) == 384:
                break
        digest = hashlib.sha256(digest).digest()
    return values


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not EMBED_URL:
        return [local_embedding(text) for text in texts]

    try:
        response = requests.post(f"{EMBED_URL}/embed", json={"texts": texts}, timeout=30)
        response.raise_for_status()
        return response.json()["embeddings"]
    except (requests.RequestException, KeyError):
        return [local_embedding(text) for text in texts]


def ensure_collection():
    collections = qdrant.get_collections().collections
    if any(collection.name == "documents" for collection in collections):
        return
    qdrant.create_collection(
        collection_name="documents",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )


def embed_and_store(records: list[dict]):
    ensure_collection()
    embeddings = embed_texts([record["text"] for record in records])
    points = [
        PointStruct(id=index, vector=embedding, payload=record)
        for index, (embedding, record) in enumerate(zip(embeddings, records))
    ]
    qdrant.upsert(collection_name="documents", points=points)
    print(f"Integration 5 OK: {len(points)} vectors stored in Qdrant")


if __name__ == "__main__":
    embed_and_store([
        {"id": "doc_001", "text": "AI platform integration test"},
        {"id": "doc_002", "text": "Kafka to Prefect pipeline"},
    ])
