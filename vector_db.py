from qdrant_client import QdrantClient, models
import os
import uuid

os.environ["ORT_LOGGING_LEVEL"] = "3"

_collection_name = "callcenter_bge"
_model_name = "BAAI/bge-base-en-v1.5"
_cuda_options = {"cuda": True, "device": 1}
_client = None
_similarity_threshold = 0.90  # cosine similarity threshold to consider duplicate

def _init_client():
    global _client
    if _client is None:
        _client = QdrantClient(url="http://localhost:6333")
        if not _client.collection_exists(_collection_name):
            _client.create_collection(
                collection_name=_collection_name,
                vectors_config=models.VectorParams(
                    size=768,
                    distance=models.Distance.COSINE
                )
            )
    return _client

_client = _init_client()

def update_database(new_payload):
    docs_to_add = []
    ids_to_add = []

    for data in new_payload:
        doc_vector = models.Document(
            text=data["document"],
            model=_model_name,
            options=_cuda_options
        )

        hits = _client.query_points(
            collection_name=_collection_name,
            query=doc_vector,
            limit=1
        ).points

        if not hits or hits[0].score < _similarity_threshold:
            docs_to_add.append(doc_vector)
            ids_to_add.append(data.get("id") or str(uuid.uuid4()))

    if docs_to_add:
        _client.upload_collection(
            collection_name=_collection_name,
            vectors=docs_to_add,
            ids=ids_to_add,
            payload=[data for data in new_payload if data["document"] in [d.text for d in docs_to_add]]
        )

def search(query_text, k=3):
    query_doc = models.Document(
        text=query_text,
        model=_model_name,
        options=_cuda_options
    )
    hits = _client.query_points(
        collection_name=_collection_name,
        query=query_doc,
        limit=k
    ).points
    return [{"id": hit.id, "document": hit.payload["document"], "score": hit.score} for hit in hits]

def get_all_documents(limit=1000):
    points, _next_page = _client.scroll(
        collection_name=_collection_name,
        limit=limit
    )
    return [{"id": p.id, "document": p.payload["document"]} for p in points]

from qdrant_client.http.models import PointIdsList

def delete_document_by_id(doc_id):
    _client.delete(
        collection_name=_collection_name,
        points_selector=PointIdsList(points=[doc_id])
    )

def delete_search_results(query_text, k=10):
    results = search(query_text, k)
    ids_to_delete = [r["id"] for r in results]
    if ids_to_delete:
        _client.delete(
            collection_name=_collection_name,
            points_selector=PointIdsList(points=ids_to_delete)
        )
    return ids_to_delete

