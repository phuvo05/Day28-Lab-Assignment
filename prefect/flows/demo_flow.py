from prefect import flow, task


@task
def extract_records():
    print("Kafka -> Delta demo extract")
    return ["doc_001", "doc_002", "doc_003"]


@task
def save_records(records):
    print(f"Saved {len(records)} records to Delta Lake")
    return len(records)


@flow(name="Lab 28 Kafka to Delta Demo")
def lab28_demo_flow():
    records = extract_records()
    count = save_records(records)
    print(f"Lab 28 flow completed with {count} records")
    return count


if __name__ == "__main__":
    lab28_demo_flow()
