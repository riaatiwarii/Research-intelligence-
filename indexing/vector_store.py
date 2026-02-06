from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

COLLECTION = "papers"

client = QdrantClient(host="localhost", port=6333)

def init_collection(dim):
    if COLLECTION not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE
            )
        )

def upsert(vector, payload):
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )
        ]
    )

def search(vector, limit=5):
    return client.search(
        collection_name=COLLECTION,
        query_vector=vector,
        limit=limit
    )
