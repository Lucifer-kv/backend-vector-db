from fastapi.testclient import TestClient
from src.api import app
from src.api.dependencies import get_db
from src.services.vector_db import VectorDB
import pytest
from src.models.indexing import SearchResult, HNSWIndex
from datetime import datetime

client = TestClient(app)

@pytest.fixture(autouse=True)
def reset_db():
    db = get_db()
    with db.lock.for_writing():
        db.libraries.clear()
        db.indexes.clear()
    yield

def test_create_library():
    response = client.post("/libraries/", json={
        "name": "test-lib",
        "metadata": {
            "name": "Test Library",
            "createdAt": "2025-04-30T00:00:00.000000"
        }
    })
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["name"] == "test-lib"
    assert data["metadata"]["name"] == "Test Library"
    assert isinstance(data["created_at"], str)

def test_get_library():
    create_response = client.post("/libraries/", json={
        "name": "test-lib",
        "metadata": {
            "name": "Test Library",
            "createdAt": "2025-04-30T00:00:00.000000"
        }
    })
    assert create_response.status_code == 200
    library_id = create_response.json()["id"]
    
    response = client.get(f"/libraries/{library_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == library_id
    assert data["name"] == "test-lib"
    assert data["metadata"]["name"] == "Test Library"

def test_document_crud():
    create_response = client.post("/libraries/", json={
        "name": "test-lib",
        "metadata": {
            "name": "Test Library",
            "createdAt": "2025-04-30T00:00:00.000000"
        }
    })
    assert create_response.status_code == 200
    library_id = create_response.json()["id"]
    
    response = client.post(f"/documents/?library_id={library_id}", json={
        "metadata": {
            "name": "Test Document",
            "createdAt": "2025-04-30T00:00:00.000000"
        }
    })
    assert response.status_code == 200
    data = response.json()
    document_id = data["id"]
    assert data["chunks"] == []
    assert data["metadata"]["name"] == "Test Document"
    
    response = client.get(f"/documents/{document_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == document_id
    
    response = client.patch(f"/documents/{document_id}", json={
        "metadata": {
            "name": "Updated Document",
            "createdAt": "2025-04-30T00:00:00.000000"
        }
    })
    assert response.status_code == 200
    assert response.json() == {"message": "Document updated"}
    
    response = client.delete(f"/documents/{document_id}")
    assert response.status_code == 200
    assert response.json() == {"message": "Document deleted"}
    
    response = client.get(f"/documents/{document_id}")
    assert response.status_code == 404

def test_chunk_crud():
    create_response = client.post("/libraries/", json={
        "name": "test-lib",
        "metadata": {
            "name": "Test Library",
            "createdAt": "2025-04-30T00:00:00.000000"
        }
    })
    assert create_response.status_code == 200
    library_id = create_response.json()["id"]
    
    response = client.post(f"/documents/?library_id={library_id}", json={
        "metadata": {
            "name": "Test Document",
            "createdAt": "2025-04-30T00:00:00.000000"
        }
    })
    assert response.status_code == 200
    document_id = response.json()["id"]
    
    chunk1 = {
        "text": "Hello world",
        "metadata": {
            "name": "Chunk 1",
            "createdAt": "2025-04-30T00:00:00.000000"
        }
    }
    response = client.post(f"/chunks/?library_id={library_id}&document_id={document_id}", json=chunk1)
    assert response.status_code == 200
    assert "chunk_id" in response.json()
    chunk_id = response.json()["chunk_id"]
    assert isinstance(chunk_id, str)
    
    response = client.get(f"/documents/{document_id}")
    assert response.status_code == 200
    chunks = response.json()["chunks"]
    assert len(chunks) == 1
    assert chunks[0]["id"] == chunk_id
    
    response = client.get(f"/chunks/{chunk_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Hello world"
    assert data["metadata"]["name"] == "Chunk 1"
    assert isinstance(data["metadata"]["createdAt"], str)
    
    update_data = {
        "text": "Updated text",
        "metadata": {
            "name": "Chunk 1 Updated",
            "createdAt": data["metadata"]["createdAt"]
        }
    }
    response = client.patch(f"/chunks/{chunk_id}", json=update_data)
    assert response.status_code == 200
    assert response.json() == {"message": "Chunk updated"}
    
    response = client.get(f"/chunks/{chunk_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Updated text"
    assert data["metadata"]["name"] == "Chunk 1 Updated"
    
    invalid_chunk = {
        "text": "Invalid chunk",
        "metadata": {
            "name": "Invalid Chunk"
        }
    }
    response = client.post(f"/chunks/?library_id={library_id}&document_id={document_id}", json=invalid_chunk)
    assert response.status_code == 422
    
    response = client.delete(f"/chunks/{chunk_id}")
    assert response.status_code == 200
    assert response.json() == {"message": "Chunk deleted"}
    
    response = client.get(f"/chunks/{chunk_id}")
    assert response.status_code == 404

def test_search():
    create_response = client.post("/libraries/", json={
        "name": "test-lib",
        "metadata": {
            "name": "Test Library",
            "createdAt": "2025-04-30T00:00:00.000000"
        }
    })
    assert create_response.status_code == 200
    library_id = create_response.json()["id"]
    
    response = client.post(f"/documents/?library_id={library_id}", json={
        "metadata": {
            "name": "Test Document",
            "createdAt": "2025-04-30T00:00:00.000000"
        }
    })
    assert response.status_code == 200
    document_id = response.json()["id"]
    
    chunk1 = {
        "text": "Hello world",
        "metadata": {
            "name": "Chunk 1",
            "createdAt": "2025-04-30T00:00:00.000000"
        }
    }
    response = client.post(f"/chunks/?library_id={library_id}&document_id={document_id}", json=chunk1)
    assert response.status_code == 200
    assert "chunk_id" in response.json()
    
    chunk2 = {
        "text": "This is a test",
        "metadata": {
            "name": "Chunk 2",
            "createdAt": "2025-04-30T00:00:00.000000"
        }
    }
    response = client.post(f"/chunks/?library_id={library_id}&document_id={document_id}", json=chunk2)
    assert response.status_code == 200
    assert "chunk_id" in response.json()
    
    response = client.post(
        f"/libraries/{library_id}/search",
        json={"query_text": "Hello world"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "similarity" in data
    assert data["text"] == "Hello world"
    assert 0 <= data["similarity"] <= 1.0

def test_search_with_metadata_filters():
    create_response = client.post("/libraries/", json={
        "name": "test-lib",
        "metadata": {
            "name": "Test Library",
            "createdAt": "2025-04-30T00:00:00.000000"
        }
    })
    assert create_response.status_code == 200
    library_id = create_response.json()["id"]
    
    response = client.post(f"/documents/?library_id={library_id}", json={
        "metadata": {
            "name": "Test Document",
            "createdAt": "2025-04-30T00:00:00.000000"
        }
    })
    assert response.status_code == 200
    document_id = response.json()["id"]
    
    chunk1 = {
        "text": "Hello world",
        "metadata": {
            "name": "Chunk One",
            "createdAt": "2025-04-29T10:00:00.000000"
        }
    }
    response = client.post(f"/chunks/?library_id={library_id}&document_id={document_id}", json=chunk1)
    assert response.status_code == 200
    assert "chunk_id" in response.json()
    
    chunk2 = {
        "text": "This is a test",
        "metadata": {
            "name": "Chunk Two",
            "createdAt": "2025-04-30T12:00:00.000000"
        }
    }
    response = client.post(f"/chunks/?library_id={library_id}&document_id={document_id}", json=chunk2)
    assert response.status_code == 200
    assert "chunk_id" in response.json()
    
    response = client.post(
        f"/libraries/{library_id}/search",
        json={
            "query_text": "test",
            "metadata_filters": {
                "name": "two",
                "createdAt": "2025-04-30T00:00:00.000000"
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "This is a test"
    assert 0 <= data["similarity"] <= 1.0

    response = client.post(
        f"/libraries/{library_id}/search",
        json={
            "query_text": "hello",
            "metadata_filters": {
                "name": "one",
                "createdAt": "2025-04-30T00:00:00.000000"
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == ""
    assert data["similarity"] == 0.0

def test_search_all_libraries():
    create_response1 = client.post("/libraries/", json={
        "name": "lib1",
        "metadata": {
            "name": "Library One",
            "createdAt": "2025-04-30T00:00:00.000000"
        }
    })
    assert create_response1.status_code == 200
    library_id1 = create_response1.json()["id"]
    
    response = client.post(f"/documents/?library_id={library_id1}", json={
        "metadata": {
            "name": "Document One",
            "createdAt": "2025-04-30T00:00:00.000000"
        }
    })
    assert response.status_code == 200
    document_id1 = response.json()["id"]
    
    chunk1 = {
        "text": "Hello world",
        "metadata": {
            "name": "Chunk One",
            "createdAt": "2025-04-29T10:00:00.000000"
        }
    }
    response = client.post(f"/chunks/?library_id={library_id1}&document_id={document_id1}", json=chunk1)
    assert response.status_code == 200
    assert "chunk_id" in response.json()
    
    create_response2 = client.post("/libraries/", json={
        "name": "lib2",
        "metadata": {
            "name": "Library Two",
            "createdAt": "2025-04-30T00:00:00.000000"
        }
    })
    assert create_response2.status_code == 200
    library_id2 = create_response2.json()["id"]
    
    response = client.post(f"/documents/?library_id={library_id2}", json={
        "metadata": {
            "name": "Document Two",
            "createdAt": "2025-04-30T00:00:00.000000"
        }
    })
    assert response.status_code == 200
    document_id2 = response.json()["id"]
    
    chunk2 = {
        "text": "This is a test",
        "metadata": {
            "name": "Chunk Two",
            "createdAt": "2025-04-30T12:00:00.000000"
        }
    }
    response = client.post(f"/chunks/?library_id={library_id2}&document_id={document_id2}", json=chunk2)
    assert response.status_code == 200
    assert "chunk_id" in response.json()
    
    response = client.post(
        "/search",
        json={
            "query_text": "test",
            "metadata_filters": {
                "name": "two",
                "createdAt": "2025-04-30T00:00:00.000000"
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["library_id"] == library_id2
    assert data["text"] == "This is a test"
    assert 0 <= data["similarity"] <= 1.0

def test_search_matching_screenshot():
    db = get_db()
    with db.lock.for_writing():
        db.libraries.clear()
        db.indexes.clear()
    
    create_response1 = client.post("/libraries/", json={
        "name": "lib1",
        "metadata": {
            "name": "Library One",
            "createdAt": "2025-04-30T00:00:00.000000"
        }
    })
    assert create_response1.status_code == 200
    library_id1 = create_response1.json()["id"]
    
    response = client.post(f"/documents/?library_id={library_id1}", json={
        "metadata": {
            "name": "Document One",
            "createdAt": "2025-04-30T00:00:00.000000"
        }
    })
    assert response.status_code == 200
    document_id = response.json()["id"]
    
    chunk1 = {
        "text": "hello",
        "metadata": {
            "name": "Kevin's Chunk",
            "createdAt": "2025-04-30T22:00:00.000000"
        }
    }
    response = client.post(f"/chunks/?library_id={library_id1}&document_id={document_id}", json=chunk1)
    assert response.status_code == 200
    assert "chunk_id" in response.json()
    
    response = client.post(
        "/search",
        json={
            "query_text": "hi",
            "metadata_filters": {
                "name": "kevin",
                "createdAt": "2025-04-29T00:00:00.000000"
            }
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["library_id"] == library_id1
    assert data["text"] == "hello"
    assert 0 <= data["similarity"] <= 1.0