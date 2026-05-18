# scripts/01_ingest_to_kafka.py
from kafka import KafkaProducer
import json, time

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode()
)

def ingest_data(records: list[dict]):
    for record in records:
        producer.send("data.raw", value=record)
        print(f"Sent: {record['id']}")
    producer.flush()

# Test
sample_data = [
    {"id": "doc_001", "text": "AI platform integration test", "timestamp": time.time()},
    {"id": "doc_002", "text": "Kafka to Airflow pipeline", "timestamp": time.time()},
]
ingest_data(sample_data)
print("Integration 1 OK: Data -> Kafka")
