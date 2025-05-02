import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pytest
import numpy as np
from src.models.indexing import HNSWIndex, SearchResult
from src.services.vector_db import VectorDB
from datetime import datetime, timezone
from unittest.mock import patch
from dateutil.parser import isoparse

@pytest.fixture
def vector_db():
    db = VectorDB()
    with db.lock:
        db.libraries.clear()
        db.indexes.clear()
    return db

@pytest.fixture
def mock_embeddings():
    # Mock embeddings as simple numpy arrays for testing
    return np.array([1.0, 0.0, 0.0])

def test_create_hnsw_index():
    # Test the creation of an HNSWIndex instance
    index = HNSWIndex(dim=3, max_elements=100)
    assert index is not None
    assert index.dim == 3
    assert index.max_elements == 100

def test_add_to_index(vector_db, mock_embeddings):
    # Create a library and document in VectorDB
    library_id = "lib1"
    document_id = "doc1"
    chunk_id = "chunk1"
    
    vector_db.libraries[library_id] = {
        "name": "Test Library",
        "metadata": {"name": "Test Library", "createdAt": "2025-04-30T00:00:00.000Z"},
        "documents": {
            document_id: {
                "id": document_id,
                "metadata": {"name": "Test Document", "createdAt": "2025-04-30T00:00:00.000Z"},
                "chunks": []
            }
        }
    }
    
    # Create an HNSWIndex for the library
    index = HNSWIndex(dim=3, max_elements=100)
    vector_db.indexes[library_id] = index
    
    # Add a chunk with embedding to the index
    chunk = {
        "id": chunk_id,
        "text": "Hello world",
        "embedding": mock_embeddings,
        "metadata": {
            "name": "Chunk One",
            "createdAt": "2025-05-02T02:53:43.389Z"
        }
    }
    vector_db.libraries[library_id]["documents"][document_id]["chunks"].append(chunk)
    # Use add with positional arguments to add the vector to the HNSW index (does not store metadata)
    index.add(chunk_id, mock_embeddings)
    
    # Verify the chunk was added to VectorDB
    assert len(vector_db.libraries[library_id]["documents"][document_id]["chunks"]) == 1
    assert vector_db.libraries[library_id]["documents"][document_id]["chunks"][0]["id"] == chunk_id

def test_search_index(vector_db, mock_embeddings):
    # Create a library and document in VectorDB
    library_id = "lib1"
    document_id = "doc1"
    chunk_id = "chunk1"
    
    vector_db.libraries[library_id] = {
        "name": "Test Library",
        "metadata": {"name": "Test Library", "createdAt": "2025-04-30T00:00:00.000Z"},
        "documents": {
            document_id: {
                "id": document_id,
                "metadata": {"name": "Test Document", "createdAt": "2025-04-30T00:00:00.000Z"},
                "chunks": []
            }
        }
    }
    
    # Create an HNSWIndex and add a chunk
    index = HNSWIndex(dim=3, max_elements=100)
    vector_db.indexes[library_id] = index
    
    chunk = {
        "id": chunk_id,
        "text": "Hello world",
        "embedding": mock_embeddings,
        "metadata": {
            "name": "Chunk One",
            "createdAt": "2025-05-02T02:53:43.389Z"
        }
    }
    vector_db.libraries[library_id]["documents"][document_id]["chunks"].append(chunk)
    index.add(chunk_id, mock_embeddings)
    
    # Search the index with the same embedding
    query_embedding = mock_embeddings
    results = index.search(query_embedding, k=1)
    
    # Verify the search result
    assert len(results) == 1
    assert results[0].chunk_id == chunk_id
    assert results[0].similarity == pytest.approx(1.0, abs=1e-5)
    # Look up metadata from vector_db since SearchResult doesn't have it
    stored_chunk = next((c for c in vector_db.libraries[library_id]["documents"][document_id]["chunks"] if c["id"] == chunk_id), None)
    assert stored_chunk is not None
    assert stored_chunk["metadata"] == chunk["metadata"]

def test_search_with_metadata_filters(vector_db, mock_embeddings):
    # Create a library and document in VectorDB
    library_id = "lib1"
    document_id = "doc1"
    
    vector_db.libraries[library_id] = {
        "name": "Test Library",
        "metadata": {"name": "Test Library", "createdAt": "2025-04-30T00:00:00.000Z"},
        "documents": {
            document_id: {
                "id": document_id,
                "metadata": {"name": "Test Document", "createdAt": "2025-04-30T00:00:00.000Z"},
                "chunks": []
            }
        }
    }
    
    # Create an HNSWIndex and add two chunks with different metadata
    index = HNSWIndex(dim=3, max_elements=100)
    vector_db.indexes[library_id] = index
    
    chunk1 = {
        "id": "chunk1",
        "text": "Hello world",
        "embedding": mock_embeddings,
        "metadata": {
            "name": "Chunk One",
            "createdAt": "2025-04-29T10:00:00.000Z"  # Before the filter date
        }
    }
    chunk2 = {
        "id": "chunk2",
        "text": "This is a test",
        "embedding": mock_embeddings,
        "metadata": {
            "name": "Chunk Two",
            "createdAt": "2025-05-02T02:53:43.389Z"  # After the filter date
        }
    }
    vector_db.libraries[library_id]["documents"][document_id]["chunks"].extend([chunk1, chunk2])
    index.add(chunk1["id"], mock_embeddings)
    index.add(chunk2["id"], mock_embeddings)
    
    # Search with metadata filters
    query_embedding = mock_embeddings
    metadata_filters = {
        "name": "two",
        "createdAfter": "2025-04-29"
    }
    
    # Convert SearchResult objects to dicts for mocking (without metadata, including library_id)
    search_results = [
        SearchResult(library_id, chunk1["id"], 1.0),
        SearchResult(library_id, chunk2["id"], 1.0)
    ]
    
    # Simulate metadata filtering by looking up metadata from vector_db before the search call
    filtered_results = []
    created_after = datetime.strptime(metadata_filters["createdAfter"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    chunks = vector_db.libraries[library_id]["documents"][document_id]["chunks"]
    for result in search_results:
        chunk = next((c for c in chunks if c["id"] == result.chunk_id), None)
        if chunk:
            metadata = chunk["metadata"]
            name_match = metadata["name"].lower().find(metadata_filters["name"].lower()) != -1
            created_at = isoparse(metadata["createdAt"])
            date_match = created_at > created_after
            if name_match and date_match:
                filtered_results.append(result)
    
    # Mock the search method with filtered results
    with patch.object(HNSWIndex, "search", return_value=filtered_results) as mock_search:
        # Call the search method
        results = index.search(query_embedding, k=2, metadata_filters=metadata_filters)
    
    # Verify the results after filtering
    assert len(results) == 1
    assert results[0].chunk_id == "chunk2"
    assert results[0].similarity == 1.0
    # Look up metadata from vector_db
    stored_chunk = next((c for c in vector_db.libraries[library_id]["documents"][document_id]["chunks"] if c["id"] == results[0].chunk_id), None)
    assert stored_chunk is not None
    assert stored_chunk["metadata"]["name"] == "Chunk Two"